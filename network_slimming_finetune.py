import argparse
import copy
import logging
import mmcv
import os
import time
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from os import path as osp
import json

from mmdet3d import __version__
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed, train_detector
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)

from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument('--slimming_json', default="", help='the dir to save logs and models')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # add a logging filter
    logging_filter = logging.Filter('mmdet')
    logging_filter.filter = lambda record: record.find('mmdet') != -1

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # cfg._cfg_dict.model.backbone.out_channels
    # cfg._cfg_dict.model.backbone.in_channels
    # cfg._cfg_dict.model.backbone.layer_nums

    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed


    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    channels_list = cfg._cfg_dict.model.neck.out_channels

    model = MMDataParallel(
        model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    # model.module.backbone.block
    # model.module.neck.deblocks
    rpn_dict = {}
    rpn_list = []
    rpn_key_list = []
    for i in range(3):
        count = 0
        for m in model.module.backbone.blocks[i]:
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                rpn_list.append(size)
                bn[index:(index + size)] = m.weight.data.abs().clone()
                key = "blocks_{}_{}".format(i, count)
                rpn_key_list.append(key)
                rpn_dict[key] = list(range(index, index+size))
                index += size
                count += 1
    for i in range(3):
        count = 0
        for m in model.module.neck.deblocks[i]:
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                rpn_list.append(size)
                key = "deblocks_{}_{}".format(i, count)
                rpn_key_list.append(key)
                rpn_dict[key] = list(range(index, index + size))
                bn[index:(index + size)] = m.weight.data.abs().clone()
                count += 1
                index += size
    bn = bn[:index]
    y, ind = torch.sort(bn)
    thre_index = int(index * 0.5)
    thre = y[thre_index]
    ind_pick = torch.where(y > thre)
    ind_end = ind[ind_pick]
    ind_end = [int(v) for v in list(ind_end.numpy())]
    for key in rpn_dict.keys():
        value_list = rpn_dict[key]
        tmp = []
        for var in ind_end:
            if var in value_list:
                tmp.append(var)
        tmp.sort()
        rpn_dict[key] = tmp
    with open("depth_description_0.5.json", "w") as f:
        json.dump(rpn_dict, f)
    if len(args.slimming_json) != 0:
        with open(args.slimming_json, "r") as f:
            rpn_dict_ = json.load(f)
        cfg._cfg_dict.model.backbone.type = 'SECONDSlim'
        cfg._cfg_dict.model.backbone.out_channels = []
        cfg._cfg_dict.model.neck.in_channels = []
        cfg._cfg_dict.model.neck.out_channels = []
        layer_num_plus = [k + 1 for k in cfg._cfg_dict.model.backbone.layer_nums]
        for i in range(len(cfg._cfg_dict.model.backbone.layer_nums)):
            for j in range(layer_num_plus[i]):
                block_count = "blocks_{}_{}".format(i, j)
                cfg._cfg_dict.model.backbone.out_channels.append(
                    len(rpn_dict_[block_count]) if len(rpn_dict_[block_count]) != 0 else 2)
                if j == layer_num_plus[i] - 1:
                    cfg._cfg_dict.model.neck.in_channels.append(
                        len(rpn_dict_[block_count]) if len(rpn_dict_[block_count]) != 0 else 2)
        for i in range(len(cfg._cfg_dict.model.backbone.layer_nums)):
            deblocks_count = "deblocks_{}_0".format(i)
            cfg._cfg_dict.model.neck.out_channels.append(
                len(rpn_dict_[deblocks_count]) if len(rpn_dict_[deblocks_count]) != 0 else 2)
        cfg._cfg_dict.model.bbox_head.in_channels = sum(cfg._cfg_dict.model.neck.out_channels)
        cfg._cfg_dict.model.bbox_head.feat_channels = sum(cfg._cfg_dict.model.neck.out_channels)
        slim_model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    optimizer = build_optimizer(slim_model, cfg.optimizer)
    runner = EpochBasedRunner(
        slim_model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)

    signal_num_plus = [k + 1 for k in cfg._cfg_dict.model.backbone.layer_nums]
    signal_list = [int(sum(signal_num_plus[:int(i+1)])-1) for i in range(len(signal_num_plus))]
    signal = 0
    deconv_cfg_list = []
    for i in range(3):
        for m in model.module.backbone.blocks[i]:
            if isinstance(m,  nn.Conv2d):
                if signal == 0:
                    key = rpn_key_list[signal]
                    mask_cfg = rpn_dict[key]
                    if signal == 0:
                        mask_total = 0
                    else:
                        mask_total = sum(rpn_list[:int(signal)])
                    if len(mask_cfg) == 0:
                        mask_cfg.append(mask_total)
                        mask_cfg.append(int(mask_total + 1))
                        rpn_dict[key] = mask_cfg
                    mask_cfg = torch.tensor(mask_cfg) - mask_total
                    rpn_conv_length = rpn_list[signal]

                    mask = torch.zeros(rpn_conv_length)
                    mask_cfg = torch.tensor(mask_cfg)
                    mask[mask_cfg] = 1
                    m.weight.data = m.weight.data[mask_cfg, :, :, :]
                    print("signal: {} conv shape: {}".format(signal, m.weight.data.shape))
                    if m.bias:
                        m.bias.data = m.bias.data[mask_cfg, :, :, :]
                        # m.bias.data = m.bias.data[:, mask_cfg_after, :, :]
                    signal = signal + 1
                else:
                    m.weight.data = m.weight.data[:, mask_cfg, :, :]
                    # after
                    key = rpn_key_list[signal]
                    mask_cfg = rpn_dict[key]
                    if signal == 0:
                        mask_total = 0
                    else:
                        mask_total = sum(rpn_list[:int(signal)])
                    if len(mask_cfg) == 0:
                        mask_cfg.append(mask_total)
                        mask_cfg.append(int(mask_total + 1))
                        rpn_dict[key] = mask_cfg
                    mask_cfg = torch.tensor(mask_cfg) - mask_total
                    rpn_conv_length = rpn_list[signal]
                    if signal in signal_list:
                        deconv_cfg_list.append(mask_cfg)
                    mask = torch.zeros(rpn_conv_length)
                    mask_cfg = torch.tensor(mask_cfg)
                    mask[mask_cfg] = 1

                    m.weight.data = m.weight.data[mask_cfg, :, :, :]
                    print("signal: {} conv shape: {}".format(signal, m.weight.data.shape))
                    if m.bias:
                        m.bias.data = m.bias.data[mask_cfg, :, :, :]
                        # m.bias.data = m.bias.data[:, mask_cfg_after, :, :]
                    signal = signal + 1

            if isinstance(m, nn.BatchNorm2d):
                mask_cfg_ = list(mask_cfg.numpy())
                m.weight.data = m.weight.data[mask_cfg_]
                m.bias.data = m.bias.data[mask_cfg_]
                m.running_mean = m.running_mean[mask_cfg_]
                m.running_var = m.running_var[mask_cfg_]
                print("signal: {} bn shape: {}".format(signal, m.weight.data.shape[0]))
    neck_list = []
    for i in range(3):
        for m in model.module.neck.deblocks[i]:
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = m.weight.data[deconv_cfg_list[i], :, :, :]
                key = rpn_key_list[signal]
                mask_cfg = rpn_dict[key]
                if signal == 0:
                    mask_total = 0
                else:
                    mask_total = sum(rpn_list[:int(signal)])
                if len(mask_cfg) == 0:
                    mask_cfg.append(mask_total)
                    mask_cfg.append(int(mask_total + 1))
                    rpn_dict[key] = mask_cfg
                mask_cfg = torch.tensor(mask_cfg) - mask_total
                neck_list.append(mask_cfg)
                rpn_conv_length = rpn_list[signal]

                mask = torch.zeros(rpn_conv_length)
                mask_cfg = torch.tensor(mask_cfg)
                mask[mask_cfg] = 1
                m.weight.data = m.weight.data[:, mask_cfg, :, :]
                print("signal: {} conv shape: {}".format(signal, m.weight.data.shape))
                if m.bias:
                    m.bias.data = m.bias.data[:, mask_cfg, :, :]
                signal = signal + 1

            if isinstance(m, nn.BatchNorm2d):
                mask_cfg = list(mask_cfg.numpy())
                m.weight.data = m.weight.data[mask_cfg]
                m.bias.data = m.bias.data[mask_cfg]
                m.running_mean = m.running_mean[mask_cfg]
                m.running_var = m.running_var[mask_cfg]
                print("signal: {} bn shape: {}".format(signal, m.weight.data.shape[0]))
    # cfg._cfg_dict.model.neck.out_channels

    channels_mile = [int(sum(channels_list[:int(i + 1)])) for i in range(len(channels_list))]
    for m in model.module.bbox_head.modules():
        tensor_list = []

        if isinstance(m, nn.Conv2d):
            for i in range(len(channels_mile)):
                if i == 0:
                    part = m.weight.data[:, 0:channels_mile[i], :, :]
                else:
                    part = m.weight.data[:, channels_mile[int(i-1)]:channels_mile[i], :, :]
                tensor_list.append(part[:, neck_list[i], :, :])
            tensor_concat = torch.cat(tensor_list, dim=1)
            m.weight.data = tensor_concat
            print(m.weight.data.shape)

    runner.save_checkpoint("./", "slimmed_model.pth")


if __name__ == "__main__":
    main()
