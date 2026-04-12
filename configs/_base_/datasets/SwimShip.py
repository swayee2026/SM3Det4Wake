# Copyright (c) OpenMMLab. All rights reserved.
# SWIM Dataset Configuration for Ship-Wake Detection
# 
# This configuration uses the SWIMDataset class which supports:
# - Dual annotation sources: Annotations/ (wake OBB) and Landmarks/ (ship point+direction)
# - SWIM dataset format: ImageSets/Main/train.txt for train/val/test splits
#
# Dataset structure:
#   SWIM_Dataset_1.0.0/
#   ├── Annotations/        # Wake targets (OBB format, XML files with robndbox)
#   ├── Landmarks/          # Ship targets (point + direction, XML files with pointtheta)
#   ├── PNGImages/          # Image files
#   ├── Negative/           # Negative samples
#   └── ImageSets/Main/     # Train/val/test split files (train.txt, val.txt, test.txt)

# Dataset type - using SWIMDataset for dual annotation format
dataset_type = 'SWIMDataset'
#TODO dataset dir
#data_root = '/root/autodl-tmp/swim/SWIM_Dataset_1.0.0/'
data_root = '/root/autodl-tmp/swim/tiny_swim/'

# Image normalization (ImageNet stats)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# Image size
img_size = 800  # Can be adjusted based on GPU memory

# Angle version for rotated boxes
angle_version = 'le90'  # [0, 90) degree range

# Training pipeline for ship-wake detection
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadSWIMAnnotations', 
        with_wake_bbox=True,      # Load wake OBB annotations from Annotations/
        with_ship_point=True,     # Load ship point+direction from Landmarks/
        with_ship_direction=True
    ),
    dict(type='RResize', img_scale=(img_size, img_size)),
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
    dict(type='Pad', size=(img_size, img_size)),
    dict(type='SWIMFormatBundle'),  # Custom format bundle for dual annotations
    dict(
        type='CollectSWIM', 
        keys=[
            'img', 
            'gt_wake_bboxes',      # Wake OBB: (x, y, w, h, angle)
            'gt_wake_labels',       # Wake labels
            'gt_ship_points',       # Ship points: (px, py)
            'gt_ship_directions',   # Ship directions: (cosθ, sinθ)
            'gt_ship_labels'        # Ship labels
        ]
    ),
]

# Testing/Validation pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadSWIMAnnotations',
        with_wake_bbox=True,
        with_ship_point=True,
        with_ship_direction=True
    ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='SWIMFormatBundle'),
            dict(type='CollectSWIM', keys=['img']),
        ])
]

# Dataset configurations
# Note: Using ImageSets/Main/*.txt for train/val/test splits
train_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'ImageSets/Main/train.txt',  # Image ID list file
    img_prefix=data_root,
    wake_ann_dir='Annotations',      # Subdirectory for wake XML annotations
    ship_ann_dir='Landmarks',        # Subdirectory for ship landmark XML
    img_dir='PNGImages',             # Subdirectory for images
    pipeline=train_pipeline,
    version=angle_version
)

val_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'ImageSets/Main/val.txt',
    img_prefix=data_root,
    wake_ann_dir='Annotations',
    ship_ann_dir='Landmarks',
    img_dir='PNGImages',
    pipeline=test_pipeline,
    version=angle_version
)

test_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'ImageSets/Main/test.txt',
    img_prefix=data_root,
    wake_ann_dir='Annotations',
    ship_ann_dir='Landmarks',
    img_dir='PNGImages',
    pipeline=test_pipeline,
    version=angle_version
)

# Data loader configuration
data = dict(
    samples_per_gpu=2,      # Batch size per GPU
    workers_per_gpu=2,      # Data loading workers per GPU
    train=train_dataset,
    val=val_dataset,
    test=test_dataset,
    # For debugging with small dataset
    persistent_workers=False,  # Set True for full training
)

# -----------------------------------------------------------------------------
# Legacy Configuration (for reference only, DO NOT USE)
# -----------------------------------------------------------------------------
# The following configuration was used for the old DOTA-format dataset:
#
#   train_dataset_legacy = dict(
#       type='DOTADataset',
#       ann_file=data_root + 'train/annfiles/',  # TXT format annotations
#       img_prefix=data_root + 'train/images/',
#       pipeline=train_pipeline,
#       version=angle_version
#   )
#
# NEW: Use SWIMDataset with XML annotations from Annotations/ and Landmarks/
# -----------------------------------------------------------------------------
