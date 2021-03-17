_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
# dataset settings
data_root = '/home/mk/mmdetection3d-master/data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
# PointPillars adopted a different sampling strategies among classes\
model = dict(
    backbone=dict(
        _delete_=True,
        type='SECOND',
        in_channels=64,
        layer_nums=[2, 3, 3],
        layer_strides=[2, 2, 2],
        out_channels=[16, 32, 64]),
    neck=dict(
        _delete_=True,
        type='SECONDFPN',
        in_channels=[16, 32, 64],
        upsample_strides=[1, 2, 4],
        out_channels=[64, 64, 64]),
    bbox_head=dict(
            _delete_=True,
            type='Anchor3DHead',
            num_classes=3,
            in_channels=192,
            feat_channels=192,
            use_direction_classifier=True,
            anchor_generator=dict(
                type='Anchor3DRangeGenerator',
                ranges=[
                    [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                    [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                    [0, -39.68, -1.78, 70.4, 39.68, -1.78],
                ],
                sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
                rotations=[0, 1.57],
                reshape_out=False),
            diff_rad_by_sin=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
            loss_dir=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)))
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10))

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(type='LoadPointsFromFile', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=5)
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
total_epochs = 150












