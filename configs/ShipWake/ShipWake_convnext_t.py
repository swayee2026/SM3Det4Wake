# Copyright (c) OpenMMLab. All rights reserved.
"""
ShipWake Detection - Full Training Configuration for SWIM Dataset

This configuration implements the full ShipWake detection model with:
- ConvNeXt MoE backbone with WakeResidual and GeometricMAMG
- Dual detection heads: WakeOBBHead + ShipPointHead (single-stage)
- DSO (Dynamic Submodule Optimization) for differentiated learning rates
- SWIM Dataset with dual annotations
"""

# Model configuration
angle_version = 'le90'
num_classes = 2  # wake, ship

model = dict(
    type='ShipWakeDualDetector',
    backbone=dict(
        type='ConvNeXt_moe_wake',
        arch='tiny',
        in_channels=3,
        stem_patch_size=4,
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
        # MoE configuration
        MoE_Block_inds=[[], [], [0, 2, 4], [0, 2]],
        noisy_gating=True,
        num_experts=4,  # wake, ship, background, mixed
        top_k=2,
        gate='cosine',
        # Wake-specific enhancements
        use_geometric_mamg=True,
        use_wake_residual=True,
        mamg_alpha=0.2,
        mamg_beta=0.5,
        residual_lambda=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://convnext/tiny')),
    
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    
    # Dual detection head
    bbox_head=dict(
        type='ShipWakeDualHead',
        in_channels=256,
        wake_head_cfg=dict(
            type='WakeOBBHead',
            num_classes=1,
            in_channels=256,
            feat_channels=256,
            stacked_convs=4,
            strides=[8, 16, 32, 64, 128],
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ship_head_cfg=dict(
            type='ShipPointHead',
            in_channels=256,
            feat_channels=256,
            stacked_convs=4,
            num_anchors=1,
            strides=[8, 16, 32, 64, 128],
            center_sampling_radius=1.5,
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
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            gpu_assign_thr=600,
            iou_calculator=dict(type='RBboxOverlaps2D'),
            ignore_iof_thr=-1),
        sampler=dict(
            type='RRandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False,
        # Test config
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000),
    
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

# Optimizer with DSO support
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            # Backbone and shared components
            'backbone': dict(lr_mult=0.1),
            'neck': dict(lr_mult=1.0),
            # Detection heads (DSO will adjust these)
            'bbox_head.wake_head': dict(lr_mult=1.0),
            'bbox_head.ship_head': dict(lr_mult=1.0),
        }))

# SWIM Dataset configuration
# TODO edit dataset dir on auto-DL
dataset_type = 'SWIMDataset'
data_root = '/root/autodl-tmp/swim/SWIM_Dataset_1.0.0'

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# Training pipeline for SWIM dataset
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSWIMAnnotations', with_wake_bbox=True, with_ship_point=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        version=angle_version),
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

# Data configuration
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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

# Learning rate schedule with DSO
lr_config = dict(
    policy='dynamic',  # Use DSO hook
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11],
    # DSO specific arguments
    extra_args={
        'T': 3,                    # Temperature for softmax
        'b': 0.4,                  # Bias for KL divergence
        'ema': 0.001,              # EMA decay rate
        'backbone_policy': 'sigmoid_kl',  # KL divergence based backbone LR
        'head_policy': 'normal'    # Normal head policy
    },
    # Map loss names to modules for DSO
    reweight_losses={
        # Wake detection losses
        'wake_loss_cls': 'bbox_head.wake_head',
        'wake_loss_bbox': 'bbox_head.wake_head',
        # Ship detection losses
        'ship_loss_center': 'bbox_head.ship_head',
        'ship_loss_direction': 'bbox_head.ship_head',
        'ship_loss_conf': 'bbox_head.ship_head'
    })

# Training schedule
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', out_dir='/root/tf-logs/', interval=10)
    ])

# Evaluation
evaluation = dict(interval=1, metric='mAP')

# Misc
dist_params = dict(backend='nccl')
log_level = 'INFO' # 'WARNING' if needed?
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/shipwake_convnext_t'

# Visualization settings (optional, for debugging)
vis_config = dict(
    enabled=False,  # Enable during debugging
    save_dir='./vis_results',
    save_interval=100,
    save_feature_maps=True,
    save_masks=True,
    save_moe_gates=True)
