"""
ShipWake DualStream - Standard Configuration (Single/2-GPU)

This is a balanced configuration for:
- Single GPU or 2-GPU training (e.g., RTX 3090/4090)
- Standard SwimShip dataset training
- Good performance with reasonable training time

Characteristics:
- ConvNeXt-Tiny backbone
- Batch size 4 (2 samples x 2 GPUs, or 4 samples x 1 GPU)
- Standard data augmentation
- DSO enabled
- ~6 hours training on RTX 3090
"""

_base_ = [
    '../_base_/datasets/SwimShip.py',
    '../_base_/default_runtime.py'
]

# =============================================================================
# Model Configuration
# =============================================================================

angle_version = 'le90'
num_classes = 2

model = dict(
    type='ShipWakeDualDetector',
    
    backbone=dict(
        type='ConvNeXt_DualStream',
        arch='tiny',
        in_channels=3,
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        
        # MoE
        MoE_Block_inds=[[], [], [0, 2, 4, 6, 8], [0, 2]],
        num_experts=4,
        top_k=1,
        gate='cosine',
        noisy_gating=True,
        
        # Dual Stream
        use_residual_transform=True,
        use_mask_guidance=True,
        mask_stages=[0, 1, 2],
        init_lambda=0.1,
        init_alpha=0.2,
        init_beta=0.2,
        
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://convnext/tiny_1k',
            prefix='backbone',
        ),
    ),
    
    ship_neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
    ),
    wake_neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
    ),
    
    # Use RotatedRetinaHead for efficiency
    ship_bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
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
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
    ),
    wake_bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 2.0, 4.0],
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
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
    ),
    
    ship_train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D'),
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    ship_test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=1000,
    ),
    wake_train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D'),
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    wake_test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=1000,
    ),
    
    multi_tasks_reweight='dso',
    reweight_losses=dict(
        ship_loss_cls='ship_bbox_head',
        ship_loss_bbox='ship_bbox_head',
        wake_loss_cls='wake_bbox_head',
        wake_loss_bbox='wake_bbox_head',
    ),
)

# =============================================================================
# Training Schedule (Standard)
# =============================================================================

# For single GPU or 2 GPUs
gpu_number = 1  # Change to 2 if using 2 GPUs
samples_per_gpu = 4  # Total batch size = 4 or 8

optimizer = dict(
    type='AdamW',
    lr=0.0001 / 8 * (gpu_number * samples_per_gpu),  # Linear scaling
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='dynamic',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11],
    extra_args=dict(
        T=3,
        b=0.4,
        ema=0.001,
        backbone_policy='sigmoid_kl',
        head_policy='normal',
    ),
    reweight_losses=dict(
        ship_loss_cls='ship_bbox_head',
        ship_loss_bbox='ship_bbox_head',
        wake_loss_cls='wake_bbox_head',
        wake_loss_bbox='wake_bbox_head',
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)

# =============================================================================
# Data Configuration
# =============================================================================

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    persistent_workers=True,
    
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(800, 800)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ],
        version=angle_version,
    ),
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
    test=dict(
        type='DOTADataset',
        ann_file='data/SwimShip/test/annfiles/',
        img_prefix='data/SwimShip/test/images/',
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
)

# =============================================================================
# Runtime
# =============================================================================

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

evaluation = dict(interval=1, metric='mAP', save_best='mAP')
dist_params = dict(backend='nccl')
log_level = 'INFO'
cudnn_benchmark = True
find_unused_parameters = True
seed = 42

work_dir = './work_dirs/shipwake_dualstream_standard'
