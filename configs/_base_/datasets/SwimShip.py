# Copyright (c) OpenMMLab. All rights reserved.
# SwimShip Dataset Configuration for Ship-Wake Detection

# Dataset type - using standard DOTA format for oriented boxes
dataset_type = 'DOTADataset'  # Can be changed to custom dataset if needed
data_root = 'data/SwimShip/'

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
    dict(type='LoadAnnotations', with_bbox=True),
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
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# Testing/Validation pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

# Dataset configurations
train_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'train/annfiles/',
    img_prefix=data_root + 'train/images/',
    pipeline=train_pipeline,
    version=angle_version
)

val_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'val/annfiles/',
    img_prefix=data_root + 'val/images/',
    pipeline=test_pipeline,
    version=angle_version
)

test_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'test/annfiles/',
    img_prefix=data_root + 'test/images/',
    pipeline=test_pipeline,
    version=angle_version
)

# Data loader configuration
data = dict(
    samples_per_gpu=2,  # Batch size per GPU
    workers_per_gpu=2,  # Data loading workers per GPU
    train=train_dataset,
    val=val_dataset,
    test=test_dataset,
    # For debugging with small dataset
    persistent_workers=False,  # Set True for full training
)
