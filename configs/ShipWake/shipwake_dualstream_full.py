"""
ShipWake DualStream - Full Training Configuration

This is the full-scale configuration for:
- Complete SwimShip dataset training
- Multi-GPU distributed training
- Full data augmentation
- State-of-the-art hyperparameters

Characteristics:
- ConvNeXt-Tiny backbone with pretrained weights
- 4 experts in MoE, top-1 routing
- Dual FPN necks for ship and wake
- Separate detection heads with ODMRefine
- DSO (Dynamic Submodule Optimization)
- Full augmentation pipeline
"""

# Inherit from base configs
_base_ = [
    '../_base_/datasets/SwimShip.py',
    '../_base_/default_runtime.py'
]

# =============================================================================
# Model Configuration (FULL)
# =============================================================================

angle_version = 'le90'
num_classes = 2  # ship, wake

# Model architecture
model = dict(
    type='ShipWakeDualDetector',
    
    # Backbone: ConvNeXt-Tiny with DualStream
    backbone=dict(
        type='ConvNeXt_DualStream',
        arch='tiny',  # depths=[3,3,9,3], channels=[96,192,384,768]
        in_channels=3,
        drop_path_rate=0.2,  # Stochastic depth for regularization
        layer_scale_init_value=1e-6,
        
        # MoE Configuration
        MoE_Block_inds=[[], [], [0, 2, 4, 6, 8], [0, 2]],  # Key layers
        num_experts=4,  # ship expert, wake expert, background x2
        top_k=1,  # Sparse activation
        gate='cosine',
        noisy_gating=True,
        
        # Dual Stream Configuration
        use_residual_transform=True,
        use_mask_guidance=True,
        mask_stages=[0, 1, 2],  # Generate masks at stages 1-3
        init_lambda=0.1,  # Residual fusion weight
        init_alpha=0.2,   # Ship guidance strength
        init_beta=0.2,    # Wake guidance strength
        
        # Pretrained weights
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://convnext/tiny_1k',  # ImageNet pretrained
            prefix='backbone',
        ),
    ),
    
    # Necks: Feature Pyramid Networks for multi-scale
    ship_neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],  # Match ConvNeXt-Tiny
        out_channels=256,
        num_outs=5,  # P2, P3, P4, P5, P6
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
    wake_neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
    ),
    
    # Detection Heads: ODMRefineHead for oriented detection
    ship_bbox_head=dict(
        type='ODMRefineHead',
        num_classes=1,  # Only ship
        in_channels=256,
        stacked_convs=4,  # Deeper head for ship (complex texture)
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],  # Ships vary in aspect ratio
            strides=[4, 8, 16, 32, 64],
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
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        train_cfg=dict(
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
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000,
        ),
    ),
    wake_bbox_head=dict(
        type='ODMRefineHead',
        num_classes=1,  # Only wake
        in_channels=256,
        stacked_convs=3,  # Slightly shallower for wake
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8, 16],  # Wakes are larger
            ratios=[1.0, 2.0, 4.0, 8.0],  # Wakes are very elongated
            strides=[4, 8, 16, 32, 64],
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
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        train_cfg=dict(
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
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000,
        ),
    ),
    
    # DSO (Dynamic Submodule Optimization) configuration
    multi_tasks_reweight='dso',
    reweight_losses=dict(
        ship_loss_cls='ship_bbox_head',
        ship_loss_bbox='ship_bbox_head',
        wake_loss_cls='wake_bbox_head',
        wake_loss_bbox='wake_bbox_head',
    ),
)

# =============================================================================
# Training Schedule (FULL)
# =============================================================================

# Number of GPUs and batch size
gpu_number = 8
samples_per_gpu = 2  # Effective batch size = 8 * 2 = 16

# Optimizer: AdamW with weight decay
optimizer = dict(
    type='AdamW',
    lr=0.0001,  # Base learning rate
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),  # Slower backbone
            'ship_neck': dict(lr_mult=1.0),
            'wake_neck': dict(lr_mult=1.0),
            'ship_bbox_head': dict(lr_mult=1.0),
            'wake_bbox_head': dict(lr_mult=1.0),
            # MoE experts get normal LR
            'backbone.stages.2': dict(lr_mult=0.5),  # Stage 3 with MoE
            'backbone.stages.3': dict(lr_mult=0.5),  # Stage 4 with MoE
        }
    )
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    fp16=dict(loss_scale='dynamic'),  # Mixed precision training
)

# Learning rate schedule with DSO
lr_config = dict(
    policy='dynamic',  # Use DSO policy
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11],  # Epochs to decay LR (for 12 epoch training)
    
    # DSO-specific parameters
    extra_args=dict(
        T=3,  # Temperature for softmax
        b=0.4,  # Bias for sigmoid KL
        ema=0.001,  # EMA decay for loss history
        backbone_policy='sigmoid_kl',  # Backbone LR adjustment policy
        head_policy='normal',  # Head LR adjustment policy
    ),
    reweight_losses=dict(
        ship_loss_cls='ship_bbox_head',
        ship_loss_bbox='ship_bbox_head',
        wake_loss_cls='wake_bbox_head',
        wake_loss_bbox='wake_bbox_head',
    ),
)

# Training epochs and iterations
# Assuming ~1000 images in train set, batch_size=16
# 1 epoch = 1000 / 16 = 62.5 iterations
# 12 epochs = ~750 iterations
runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(interval=1)  # Save every epoch

# =============================================================================
# Data Configuration (FULL)
# =============================================================================

# Override base dataset config with full settings
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,  # More workers for faster data loading
    persistent_workers=True,  # Keep workers alive between epochs
    
    # Training with full augmentation
    train=dict(
        type='DOTADataset',
        ann_file='data/SwimShip/train/annfiles/',
        img_prefix='data/SwimShip/train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(800, 800)),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version=angle_version,
            ),
            dict(
                type='PolyRandomRotate',
                rotate_ratio=0.5,
                angles_range=180,
                auto_bound=False,
                version=angle_version,
            ),
            dict(
                type='RandomAffine',
                max_rotate_degree=0,
                max_translate_ratio=0.1,
                scaling_ratio_range=(0.9, 1.1),
                max_shear_degree=0,
                border_val=(114, 114, 114),
            ),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
            ),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(800, 800)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ],
        version=angle_version,
    ),
    
    # Validation
    val=dict(
        type='DOTADataset',
        ann_file='data/SwimShip/val/annfiles/',
        img_prefix='data/SwimShip/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
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
    
    # Testing with multi-scale TTA (Test Time Augmentation)
    test=dict(
        type='DOTADataset',
        ann_file='data/SwimShip/test/annfiles/',
        img_prefix='data/SwimShip/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(800, 800), (1024, 1024)],  # Multi-scale testing
                flip=True,
                flip_direction=['horizontal', 'vertical'],
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
    interval=50,  # Log every 50 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', out_dir='/root/tf-logs/', interval=10),  # Enable tensorboard for AutoDL
    ]
)

# Evaluation
evaluation = dict(
    interval=1,  # Evaluate every epoch
    metric='mAP',
    save_best='mAP',
    rule='greater',
)

# Distributed training
dist_params = dict(backend='nccl')
log_level = 'INFO'

# cudnn benchmark for faster training
cudnn_benchmark = True

# Find unused parameters for DSO
find_unused_parameters = True

# Random seed for reproducibility
seed = 42

# Work directory
work_dir = './work_dirs/shipwake_dualstream_full'

# Load from (for resuming)
load_from = None

# Resume from (for resuming)
resume_from = None
