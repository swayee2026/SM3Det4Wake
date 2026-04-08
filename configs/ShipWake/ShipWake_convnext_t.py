_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model configuration for ship-wake detection
angle_version = 'le90'
num_classes = 2  # ship and wake

model = dict(
    type='ShipWakeDualDetector',
    backbone=dict(
        type='ConvNeXt_moe_wake',
        arch='tiny',
        drop_path_rate=0.1,
        # MoE configuration
        MoE_Block_inds=[[], [], [0, 2, 4], [0, 2]],
        num_experts=4,  # ship, wake, background, mixed
        top_k=2,
        gate='cosine',
        # Geometric MAMG
        use_geometric_mamg=True,
        mamg_alpha=0.2,
        mamg_beta=0.5,
        # Wake Residual
        use_wake_residual=True,
        residual_lambda=0.1,
        # Pretrained weights
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/pretrained/convnext-tiny.pth'
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
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
    # Ship detection head
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
            type='ShipWakeHead',
            num_classes=1,  # Single class for ship
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            use_direction=True,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # Wake detection head
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
            type='ShipWakeHead',
            num_classes=1,  # Single class for wake
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            use_direction=True,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # Training config
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                gpu_assign_thr=600,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
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
            debug=False)),
    # Testing config
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000))
)

# Optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=1.0),
            'neck': dict(lr_mult=1.0),
            'rpn_head': dict(lr_mult=1.0),
            'ship_roi_head': dict(lr_mult=1.0),
            'wake_roi_head': dict(lr_mult=1.0),
        })
)

# Learning rate schedule with DSO
lr_config = dict(
    policy='dynamic',
    warmup='linear',
    extra_args={
        'T': 3,
        'b': 0.4,
        'ema': 0.001,
        'backbone_policy': 'sigmoid_kl',
        'head_policy': 'normal'
    },
    reweight_losses={
        # RPN losses
        'rpn_loss_cls': 'rpn_head',
        'rpn_loss_bbox': 'rpn_head',
        # Ship ROI losses
        'ship_loss_cls': 'ship_roi_head',
        'ship_loss_bbox': 'ship_roi_head',
        'ship_acc': 'ship_roi_head',
        # Wake ROI losses
        'wake_loss_cls': 'wake_roi_head',
        'wake_loss_bbox': 'wake_roi_head',
        'wake_acc': 'wake_roi_head',
    },
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11]
)

# Runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# Dataset configuration (placeholder - needs to be adapted for SwimShip dataset)
dataset_type = 'ShipWakeDataset'  # Custom dataset needed
data_root = 'data/SwimShip/'

# ... (rest of data configuration similar to SOI_Det but for single dataset)
