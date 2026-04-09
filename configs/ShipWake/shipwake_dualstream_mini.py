"""
ShipWake DualStream - Mini Configuration for Data Pipeline Validation

This is a minimal configuration for:
- Quick data pipeline testing
- Sanity check on small dataset (10-50 images)
- GPU memory testing
- Overfitting test on single batch

Characteristics:
- Tiny backbone (atto - smallest variant)
- Minimal channels and depths
- Single GPU, small batch size
- Minimal training iterations
- No fancy augmentations
"""

# Inherit from base configs
_base_ = [
    '../_base_/datasets/SwimShip.py',
    '../_base_/default_runtime.py'
]

# =============================================================================
# Model Configuration (MINIMAL)
# =============================================================================
angle_version = 'le90'
num_classes = 2  # ship, wake (background is implicit)

model = dict(
    type='ShipWakeDualDetector',
    
    # Backbone: Smallest ConvNeXt variant
    backbone=dict(
        type='ConvNeXt_DualStream',
        arch='atto',  # Smallest: depths=[2,2,6,2], channels=[40,80,160,320]
        in_channels=3,
        # Minimal MoE config
        MoE_Block_inds=[[], [], [0], [0]],  # Only 2 MoE layers
        num_experts=2,  # Minimal experts
        top_k=1,
        gate='cosine',
        # Enable dual stream features
        use_residual_transform=True,
        use_mask_guidance=True,
        mask_stages=[0, 1],  # Only first 2 stages generate masks
        init_lambda=0.1,
        init_alpha=0.2,
        init_beta=0.2,
        # No pretrained weights for quick test
        init_cfg=None,
    ),
    
    # Necks: Lightweight FPN
    ship_neck=dict(
        type='FPN',
        in_channels=[40, 80, 160, 320],  # Match atto channels
        out_channels=128,  # Reduced from 256
        num_outs=4,  # Reduced from 5
        add_extra_convs='on_output',
    ),
    wake_neck=dict(
        type='FPN',
        in_channels=[40, 80, 160, 320],
        out_channels=128,
        num_outs=4,
        add_extra_convs='on_output',
    ),
    
    # Detection Heads: Lightweight RetinaNet style
    ship_bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=1,  # Only ship
        in_channels=128,
        stacked_convs=2,  # Minimal
        feat_channels=128,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,  # Minimal
            ratios=[1.0],
            strides=[4, 8, 16, 32],
        ),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    wake_bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=1,  # Only wake
        in_channels=128,
        stacked_convs=2,
        feat_channels=128,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0, 2.0, 4.0],  # Wakes are often elongated
            strides=[4, 8, 16, 32],
        ),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    
    # Training configs
    ship_train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D'),
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    ship_test_cfg=dict(
        nms_pre=100,  # Minimal
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.5),
        max_per_img=50,  # Minimal
    ),
    wake_train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D'),
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    wake_test_cfg=dict(
        nms_pre=100,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.5),
        max_per_img=50,
    ),
    
    # Multi-task reweighting (disabled for mini config)
    multi_tasks_reweight=None,
    reweight_losses=dict(
        ship_loss_cls='ship_bbox_head',
        ship_loss_bbox='ship_bbox_head',
        wake_loss_cls='wake_bbox_head',
        wake_loss_bbox='wake_bbox_head',
    ),
)

# =============================================================================
# Training Schedule (MINIMAL)
# =============================================================================

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Slow backbone updates
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,  # Very short warmup
    warmup_ratio=1.0 / 3,
    step=[5, 8],  # Minimal schedule
)

# Runner: Minimal iterations for sanity check
runner = dict(type='IterBasedRunner', max_iters=100)  # Just 100 iterations

checkpoint_config = dict(interval=50)  # Save every 50 iterations

# =============================================================================
# Data Configuration (MINIMAL)
# =============================================================================

# Override dataset config for mini testing
data = dict(
    samples_per_gpu=1,  # Single sample per GPU
    workers_per_gpu=0,  # No multiprocessing for debugging
    persistent_workers=False,
    
    # For overfitting test, use same data for train/val
    train=dict(
        _delete_=True,  # Override base config
        type='DOTADataset',
        ann_file='data/SwimShip/mini/annfiles/',  # Point to mini dataset
        img_prefix='data/SwimShip/mini/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(512, 512)),  # Smaller image
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(512, 512)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ],
        version=angle_version,
    ),
    val=dict(
        _delete_=True,
        type='DOTADataset',
        ann_file='data/SwimShip/mini/annfiles/',
        img_prefix='data/SwimShip/mini/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img']),
                ]
            )
        ],
        version=angle_version,
    ),
    test=dict(
        _delete_=True,
        type='DOTADataset',
        ann_file='data/SwimShip/mini/annfiles/',
        img_prefix='data/SwimShip/mini/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img']),
                ]
            )
        ],
        version=angle_version,
    ),
)

# =============================================================================
# Runtime Configuration
# =============================================================================

# Logging
log_config = dict(
    interval=10,  # Log every 10 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', out_dir='/root/tf-logs/', interval=10)  # Enable tensorboard for AutoDL
    ]
)

# Evaluation
evaluation = dict(
    interval=50,  # Evaluate every 50 iterations
    metric='mAP',
    save_best='mAP',
)

# No distributed training for mini config
dist_params = dict(backend='nccl')
log_level = 'INFO'

# Disable cudnn benchmark for reproducibility
cudnn_benchmark = False

# Seed for reproducibility
seed = 0

# Debug mode
debug = True
