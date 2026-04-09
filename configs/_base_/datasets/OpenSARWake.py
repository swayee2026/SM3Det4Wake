# Copyright (c) OpenMMLab. All rights reserved.
# OpenSARWake Dataset Configuration
#
# This configuration uses the OpenSARWakeDataset class which supports:
# - Single annotation source: samples.json (wake polygons converted to OBB)
# - No ship annotations available
#
# Dataset structure:
#   OpenSARWake/
#   ├── data/               # Image files (PNG format)
#   ├── metadata.json       # Dataset metadata (FiftyOne format)
#   └── samples.json        # Annotations (FiftyOne Polylines format)

# Dataset type - using OpenSARWakeDataset
# Note: ann_file points directly to samples.json, not ImageSets
dataset_type = 'OpenSARWakeDataset'
data_root = 'data/OpenSARWake/'

# Image normalization (SAR images may need different normalization)
# Using ImageNet stats as default, can be adjusted for SAR
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# Image size (OpenSARWake images are 1024x1024)
img_size = 1024

# Angle version for rotated boxes
angle_version = 'le90'  # [-90, 0) degree range

# Training pipeline for OpenSARWake dataset
# Only wake annotations are loaded (no ship annotations)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadOpenSARWakeAnnotations', 
        with_wake_bbox=True      # Load wake OBB annotations only
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
    dict(type='OpenSARWakeFormatBundle'),  # Custom format bundle for wake-only annotations
    dict(
        type='CollectOpenSARWake', 
        keys=[
            'img', 
            'gt_wake_bboxes',      # Wake OBB: (x, y, w, h, angle)
            'gt_wake_labels',       # Wake labels
        ]
    ),
]

# Testing/Validation pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadOpenSARWakeAnnotations',
        with_wake_bbox=True
    ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='OpenSARWakeFormatBundle'),
            dict(type='CollectOpenSARWake', keys=['img']),
        ])
]

# Dataset configurations
# Note: ann_file points to samples.json directly
train_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'samples.json',  # Path to samples.json
    img_prefix=data_root,
    img_dir='data',              # Subdirectory for images
    pipeline=train_pipeline,
    version=angle_version,
    filter_empty_gt=True         # Filter images without wake annotations
)

val_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'samples.json',
    img_prefix=data_root,
    img_dir='data',
    pipeline=test_pipeline,
    version=angle_version,
    filter_empty_gt=False        # Keep all images for validation
)

test_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'samples.json',
    img_prefix=data_root,
    img_dir='data',
    pipeline=test_pipeline,
    version=angle_version,
    filter_empty_gt=False
)

# Data loader configuration
data = dict(
    samples_per_gpu=2,      # Batch size per GPU
    workers_per_gpu=2,      # Data loading workers per GPU
    train=train_dataset,
    val=val_dataset,
    test=test_dataset,
    persistent_workers=False,  # Set True for full training
)
