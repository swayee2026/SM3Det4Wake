# Copyright (c) OpenMMLab. All rights reserved.
"""
ShipWake Detection - Full Training Configuration for SWIM Dataset

This configuration implements the full ShipWake detection model with:
- ConvNeXt MoE backbone with WakeResidual and GeometricMAMG (3 modules activated)
- Dual detection heads: WakeOBBHead + ShipPointHead (single-stage)
- DSO (Dynamic Submodule Optimization) for differentiated learning rates
- SWIM Dataset with dual annotations

Usage:
    python tools/train.py configs/ShipWake/ShipWake_convnext_t.py
"""

# =============================================================================
# GLOBAL CONFIGURATION VARIABLES - Modify these for different experiments
# =============================================================================

# --- Dataset Paths ---
dataset_type = "SWIMDataset"
data_root = "/root/autodl-tmp/swim/SWIM_Dataset_1.0.0/"  # Full dataset path
angle_version = "le90"  # Rotation angle encoding: 'le90' or 'oc'

# --- Image Configuration ---
img_scale = 600  # Image patch size (both height and width)
# Recommended range: 400-1024. Larger sizes improve accuracy but require more memory.
# For SWIM dataset: 600 provides good balance between detail and efficiency.

# --- Training Hyperparameters ---
samples_per_gpu = 2  # Batch size per GPU
# Recommended range: 1-4 for single RTX 4090 (24GB). Reduce if OOM occurs.
workers_per_gpu = 4  # Data loading workers per GPU
# Recommended range: 2-8. Higher speeds up loading but uses more CPU memory.

base_lr = 1e-4  # Base learning rate
# Recommended range: 1e-5 to 1e-3. Lower for fine-tuning, higher from scratch.
weight_decay = 0.05  # L2 regularization coefficient
# Recommended range: 0.01-0.0001. Higher prevents overfitting on small datasets.

max_epochs = 12  # Total training epochs
# Recommended range: 12-24 for full training. Can reduce to 6-8 for quick experiments.

# --- Model Architecture Parameters ---
backbone_arch = "tiny"  # ConvNeXt variant: 'tiny', 'small', 'base'
# 'tiny' for fast training, 'small'/'base' for better accuracy (requires more GPU memory)

# Module Activation Flags (3 proposed modules)
use_moe = True  # Mixture of Experts module
use_geometric_mamg = True  # Geometric Multi-Attention Mask Gate
use_wake_residual = True  # Wake-specific Residual Connections

# MoE Configuration
moe_num_experts = 4  # Number of experts: wake, ship, background, mixed
# Recommended range: 2-8. More experts increase capacity but risk expert collapse.
moe_top_k = 2  # Number of experts activated per token
# Recommended range: 1-3. Higher k increases computation but may improve routing.
moe_gate = "cosine"  # Gating mechanism: 'cosine' or 'softmax'
moe_block_indices = [[], [], [0, 2, 4], [0, 2]]  # MoE layer placement in each stage

# Geometric MAMG Configuration
mamg_alpha = 0.2  # Geometric mask guidance strength
# Recommended range: 0.1-0.5. Higher values emphasize geometric priors more.
mamg_beta = 0.5  # Feature fusion balance parameter
# Recommended range: 0.3-0.7. Controls balance between semantic and geometric features.

# Wake Residual Configuration
residual_lambda = 0.1  # Residual connection weight
# Recommended range: 0.05-0.3. Higher values preserve more low-level features.

# Detection Head Configuration
wake_feat_channels = 256  # Feature channels for wake detection head
# Recommended range: 128-512. Higher for complex wake patterns.
ship_feat_channels = 256  # Feature channels for ship detection head
wake_stacked_convs = 4  # Number of conv layers in wake head
ship_stacked_convs = 4  # Number of conv layers in ship head
# Recommended range: 2-6. Deeper heads for complex patterns but more parameters.

# --- Loss Weights ---
wake_cls_weight = 1.0  # Wake classification loss weight
wake_bbox_weight = 1.0  # Wake bounding box regression loss weight
ship_center_weight = 1.0  # Ship center point regression loss weight
ship_direction_weight = 0.5  # Ship direction regression loss weight
ship_conf_weight = 1.0  # Ship confidence loss weight

# --- Data Augmentation ---
flip_ratio_horizontal = 0.25  # Horizontal flip probability
flip_ratio_vertical = 0.25  # Vertical flip probability
flip_ratio_diagonal = 0.25  # Diagonal flip probability
# Recommended range: 0.0-0.5. Higher improves generalization but may distort patterns.

rotate_ratio = 0.5  # Random rotation probability
# Recommended range: 0.3-0.8. Essential for rotation-invariant wake detection.
rotate_angle_range = 180  # Maximum rotation angle
# Recommended range: 90-180. SWIM uses le90 encoding.

center_sampling_radius = 1.5  # Radius for center sampling (in strides)
# Recommended range: 1.0-3.0. Larger values assign more positives per GT.

# --- Optimizer Configuration ---
backbone_lr_mult = 0.1  # Backbone learning rate multiplier
# Recommended range: 0.01-0.5. Lower for fine-tuning pretrained backbones.
neck_lr_mult = 1.0  # FPN neck learning rate multiplier
head_lr_mult = 1.0  # Detection heads learning rate multiplier

# --- Learning Rate Schedule (DSO) ---
warmup_iters = 500  # Warmup iterations
# Recommended range: 200-1000. Stabilizes early training.
warmup_ratio = 0.333  # Initial LR = base_lr * warmup_ratio during warmup
lr_step_epochs = [8, 11]  # Epochs to decay learning rate
# Adjust based on max_epochs. Typically decay at 2/3 and 5/6 of total epochs.

# DSO-specific parameters
dso_temperature = 3.0  # Temperature for softmax in DSO
# Recommended range: 1.0-5.0. Higher makes task weights more uniform.
dso_bias = 0.4  # Bias for KL divergence in DSO
# Recommended range: 0.1-0.5. Controls task weight imbalance tolerance.
dso_ema_decay = 0.001  # EMA decay rate for DSO statistics
# Recommended range: 0.0001-0.01. Lower for smoother task weight updates.

# --- Testing/Evaluation Configuration ---
nms_pre = 2000  # Maximum detections before NMS
# Recommended range: 1000-5000. Higher retains more candidates.
max_per_img = 2000  # Maximum detections per image
# Recommended range: 500-3000. Depends on expected object density.
score_thr = 0.05  # Confidence threshold for detections
# Recommended range: 0.01-0.1. Lower for higher recall.
nms_iou_thr = 0.1  # NMS IoU threshold
# Recommended range: 0.05-0.3. Lower suppresses more overlapping detections.
min_bbox_size = 0  # Minimum bounding box size

# --- Logging and Checkpointing ---
log_interval = 50  # Log every N iterations
checkpoint_interval = 1  # Save checkpoint every N epochs
eval_interval = 1  # Evaluate every N epochs
work_dir = "./work_dirs/shipwake_convnext_t"  # Output directory
tensorboard_dir = "/root/tf-logs/"  # TensorBoard log directory

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

num_classes = 2  # wake, ship

model = dict(
    type="ShipWakeDualDetector",
    backbone=dict(
        type="ConvNeXt_moe_wake",
        arch=backbone_arch,
        in_channels=3,
        stem_patch_size=4,
        drop_path_rate=0.1,  # Stochastic depth rate for regularization
        layer_scale_init_value=1e-6,  # Layer scale initialization
        out_indices=[0, 1, 2, 3],
        # MoE configuration
        MoE_Block_inds=moe_block_indices if use_moe else [[], [], [], []],
        noisy_gating=True,
        num_experts=moe_num_experts,
        top_k=moe_top_k,
        gate=moe_gate,
        # Wake-specific enhancements (3 modules)
        use_geometric_mamg=use_geometric_mamg,
        use_wake_residual=use_wake_residual,
        mamg_alpha=mamg_alpha,
        mamg_beta=mamg_beta,
        residual_lambda=residual_lambda,
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://convnext/tiny"),
    ),
    neck=dict(
        type="FPN", in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5
    ),
    # Single-stage dual detection head
    bbox_head=dict(
        type="ShipWakeDualHead",
        in_channels=256,
        wake_head_cfg=dict(
            type="WakeOBBHead",
            num_classes=1,
            in_channels=256,
            feat_channels=wake_feat_channels,
            stacked_convs=wake_stacked_convs,
            strides=[8, 16, 32, 64, 128],
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=wake_cls_weight,
            ),
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=wake_bbox_weight),
        ),
        ship_head_cfg=dict(
            type="ShipPointHead",
            in_channels=256,
            feat_channels=ship_feat_channels,
            stacked_convs=ship_stacked_convs,
            num_anchors=1,
            strides=[8, 16, 32, 64, 128],
            center_sampling_radius=center_sampling_radius,
            loss_center=dict(
                type="SmoothL1Loss", beta=1.0, loss_weight=ship_center_weight
            ),
            loss_direction=dict(
                type="CosineSimilarityLoss", loss_weight=ship_direction_weight
            ),
            loss_conf=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=ship_conf_weight,
            ),
        ),
    ),
    # Training configuration
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            gpu_assign_thr=600,
            iou_calculator=dict(type="RBboxOverlaps2D"),
            ignore_iof_thr=-1,
        ),
        sampler=dict(
            type="RRandomSampler",
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True,
        ),
        pos_weight=-1,
        allowed_border=-1,  # Required for RotatedAnchorHead
        debug=False,
        # Test config (used during validation)
        nms_pre=nms_pre,
        min_bbox_size=min_bbox_size,
        score_thr=score_thr,
        nms=dict(iou_thr=nms_iou_thr),
        max_per_img=max_per_img,
    ),
    test_cfg=dict(
        nms_pre=nms_pre,
        min_bbox_size=min_bbox_size,
        score_thr=score_thr,
        nms=dict(iou_thr=nms_iou_thr),
        max_per_img=max_per_img,
    ),
)

# =============================================================================
# DATA PIPELINE
# =============================================================================

# Image normalization (ImageNet statistics)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# Training pipeline
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadSWIMAnnotations", with_wake_bbox=True, with_ship_point=True),
    dict(type="RResize", img_scale=(img_scale, img_scale)),
    dict(
        type="RRandomFlip",
        flip_ratio=[flip_ratio_horizontal, flip_ratio_vertical, flip_ratio_diagonal],
        direction=["horizontal", "vertical", "diagonal"],
        version=angle_version,
    ),
    dict(
        type="PolyRandomRotate",
        rotate_ratio=rotate_ratio,
        angles_range=rotate_angle_range,
        auto_bound=False,
        version=angle_version,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=(img_scale, img_scale)),
    dict(type="SWIMFormatBundle"),
    dict(
        type="CollectSWIM",
        keys=[
            "img",
            "gt_wake_bboxes",
            "gt_wake_labels",
            "gt_ship_points",
            "gt_ship_directions",
            "gt_ship_labels",
        ],
    ),
]

# Testing/Validation pipeline
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadSWIMAnnotations", with_wake_bbox=True, with_ship_point=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(img_scale, img_scale),
        flip=False,
        transforms=[
            dict(type="RResize"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="SWIMFormatBundle"),
            dict(type="CollectSWIM", keys=["img"]),
        ],
    ),
]

# Data configuration
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "ImageSets/Main/train.txt",
        img_prefix=data_root,  # Root directory (aligned with debug config)
        wake_ann_dir="Annotations",
        ship_ann_dir="Landmarks",
        img_dir="PNGImages",
        pipeline=train_pipeline,
        version=angle_version,
        filter_empty_gt=False,
    ),  # Keep images without annotations for training
    val=dict(
        type=dataset_type,
        ann_file=data_root + "ImageSets/Main/val.txt",
        img_prefix=data_root,
        wake_ann_dir="Annotations",
        ship_ann_dir="Landmarks",
        img_dir="PNGImages",
        pipeline=test_pipeline,
        version=angle_version,
        filter_empty_gt=False,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "ImageSets/Main/test.txt",
        img_prefix=data_root,
        wake_ann_dir="Annotations",
        ship_ann_dir="Landmarks",
        img_dir="PNGImages",
        pipeline=test_pipeline,
        version=angle_version,
        filter_empty_gt=False,
    ),
)

# =============================================================================
# OPTIMIZER CONFIGURATION
# =============================================================================

optimizer = dict(
    type="AdamW",
    lr=base_lr,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=backbone_lr_mult),
            "neck": dict(lr_mult=neck_lr_mult),
            "bbox_head.wake_head": dict(lr_mult=head_lr_mult),
            "bbox_head.ship_head": dict(lr_mult=head_lr_mult),
        }
    ),
)

# =============================================================================
# LEARNING RATE SCHEDULE (with DSO)
# =============================================================================

lr_config = dict(
    policy="dynamic",  # Use DSO hook
    warmup="linear",
    warmup_iters=warmup_iters,
    warmup_ratio=warmup_ratio,
    step=lr_step_epochs,
    # DSO specific arguments
    extra_args={
        "T": dso_temperature,
        "b": dso_bias,
        "ema": dso_ema_decay,
        "backbone_policy": "sigmoid_kl",
        "head_policy": "normal",
    },
    # Map loss names to modules for DSO
    reweight_losses={
        "wake_loss_cls": "bbox_head.wake_head",
        "wake_loss_bbox": "bbox_head.wake_head",
        "ship_loss_center": "bbox_head.ship_head",
        "ship_loss_direction": "bbox_head.ship_head",
        "ship_loss_conf": "bbox_head.ship_head",
    },
)

# =============================================================================
# RUNNER AND HOOKS
# =============================================================================

runner = dict(type="EpochBasedRunner", max_epochs=max_epochs)
checkpoint_config = dict(interval=checkpoint_interval)

log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook", out_dir=tensorboard_dir, interval=10),
    ],
)

# =============================================================================
# EVALUATION AND MISC
# =============================================================================

evaluation = dict(interval=eval_interval, metric="mAP")

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None  # Path to pretrained checkpoint if resuming
resume_from = None  # Path to checkpoint for resuming training
workflow = [("train", 1)]

# Visualization settings (optional, for debugging)
vis_config = dict(
    enabled=False,  # Enable during debugging to visualize features
    save_dir="./vis_results",
    save_interval=100,
    save_feature_maps=True,
    save_masks=True,
    save_moe_gates=True,
)
