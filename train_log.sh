python tools/train.py /home/mk/mmdetection3d-master/configs/pointpillars/pointpillars_kitti_3class_mb2.py  --work-dir benchmark_model/pp_light_mb2

python tools/train.py /home/mk/mmdetection3d-master/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py --work-dir benchmark_model/pp_origin

python tools/train.py /home/mk/mmdetection3d-master/configs/pointpillars/pointpillars_kitti_3class_invertmb2.py --work-dir benchmark_model/pp_invert_mb2

python tools/train.py /home/mk/mmdetection3d-master/configs/pointpillars/pointpillars_kitti_3class_mb2_res.py --work-dir benchmark_model/pp_invert_mb2_res

CUDA_VISIBLE_DEVICES=1 python tools/train.py /home/mk/mmdetection3d-master/configs/pointpillars/pointpillars_kitti_3class_mb2_compress.py --work-dir benchmark_model/pp_compress

CUDA_VISIBLE_DEVICES=0 python tools/train.py /home/mk/mmdetection3d-master/configs/pointpillars/pointpillars_kitti_class_invertmb2_compress.py --work-dir benchmark_model/pp_invertmb2_compress