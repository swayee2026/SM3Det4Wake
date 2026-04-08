# Copyright (c) OpenMMLab. All rights reserved.
"""
ShipWake Detection - Debug Configuration for SWIM Dataset

Minimal configuration for pipeline validation:
- Small dataset (subset of SWIM)
- Batch size = 1
- Single epoch
- Minimal model parameters
"""

# Debug settings
angle_version = 'le90'
debug_mode = True

# Model configuration (minimal)
model = dict(
    type='ShipWakeDualDetector',
    backbone=dict(
        type='ConvNeXt_moe_wake',
        arch='tiny',
        in_channels=3,
        stem_patch_size=4,
        drop_path_rate=0.0,  # Disable for debug
        layer_scale_init_value=0.0,  # Disable for debug
        out_indices=[0, 1, 2, 3],
        MoE_Block_inds=[[], [], [0], [0]],  # Minimal MoE
        noisy_gating=True,
        num_experts=4,
        top_k=2,
        gate='cosine',
        # Wake-specific modules
        use_geometric_mamg=True,
        use_wake_residual=True,
        mamg_alpha=0.2,
        mamg_beta=0.5,
        residual_lambda=0.1,
        init_cfg=None  # No pretrained for debug
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=128,  # Reduced for debug
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    # Wake detection head (OBB)
    wake_roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='WakeOBBHead',
            num_classes=1,
            in_channels=256,
            feat_channels=128,  # Reduced for debug
            stacked_convs=2,    # Reduced for debug
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # Ship detection head (Point regression)
    ship_roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='ShipPointHead',
            in_channels=256,
            feat_channels=128,  # Reduced for debug
            stacked_convs=2,    # Reduced for debug
            num_anchors=1,
            loss_center=dict(
                type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            loss_direction=dict(
                type='CosineSimilarityLoss', loss_weight=0.5),
            loss_conf=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0))),
    # Training configuration
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                gpu_assign_thr=200,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,  # Reduced for debug
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=1000,  # Reduced for debug
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                gpu_assign_thr=200,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=256,  # Reduced for debug
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=500)))

# SWIM Dataset configuration (debug)
dataset_type = 'SWIMDataset'
data_root = 'data/SWIM_debug/'  # Use small subset for debug

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSWIMAnnotations', with_wake_bbox=True, with_ship_point=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.0, version=angle_version),  # No augmentation for debug
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(800, 800)),
    dict(type='SWIMFormatBundle'),
    dict(type='CollectSWIM', keys=['img', 'gt_wake_bboxes', 'gt_wake_labels',
                                    'gt_ship_points', 'gt_ship_directions', 
                                    'gt_ship_labels']),
]

# Testing pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSWIMAnnotations', with_wake_bbox=True, with_ship_point=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='SWIMFormatBundle'),
            dict(type='CollectSWIM', keys=['img']),
        ])
]

# Data configuration (minimal)
data = dict(
    samples_per_gpu=1,  # Single sample per GPU
    workers_per_gpu=0,  # No multiprocessing for debug
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/train.txt',
        img_prefix=data_root,
        wake_ann_dir='Annotations',
        ship_ann_dir='Landmarks',
        img_dir='PNGImages',
        pipeline=train_pipeline,
        version=angle_version),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        wake_ann_dir='Annotations',
        ship_ann_dir='Landmarks',
        img_dir='PNGImages',
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root,
        wake_ann_dir='Annotations',
        ship_ann_dir='Landmarks',
        img_dir='PNGImages',
        pipeline=test_pipeline,
        version=angle_version))

# Optimizer (minimal)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'neck': dict(lr_mult=1.0),
            'rpn_head': dict(lr_mult=1.0),
            'ship_roi_head': dict(lr_mult=1.0),
            'wake_roi_head': dict(lr_mult=1.0),
        }))

# Training schedule (minimal)
lr_config = dict(
    policy='fixed',  # No LR schedule for debug
    warmup=None)

# Runner (single epoch for debug)
runner = dict(type='EpochBasedRunner', max_epochs=1)

# Checkpoint and logging
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,  # Log every iteration
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# Evaluation
evaluation = dict(interval=1, metric='mAP')

# Workflow
workflow = [('train', 1)]

# Distortion
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
work_dir = './work_dirs/shipwake_debug'

# Visualization settings
vis_config = dict(
    enabled=True,
    save_dir='./vis_debug',
    save_interval=1,  # Save every iteration
    save_feature_maps=True,
    save_masks=True,
    save_moe_gates=True)
