# SM3Det4Wake Project Guide for AI Agents

## Project Overview

SM3Det4Wake is a deep learning framework for ship wake detection in optical remote sensing images. This project is based on SM3Det (Single Model for Multi-Modal datasets and Multi-Task object Detection) architecture, customized for single-modality (optical), dual-target (ship point + wake bounding box) scenarios.

The project extends the OpenMMLab MMRotate framework to support multi-modal object detection, specifically designed for simultaneous detection of ships (point regression + direction) and wakes (oriented bounding boxes).

### Core Features

- Grid-level MoE: Sparse mixture of experts for multi-scale feature learning
- Asynchronous Learning Rate Optimization: DSO (Dynamic Submodule Optimization) based differentiated training
- Dual-task Detection: Simultaneous ship (point) and wake (rotated box) detection
- Residual Connection Enhancement: Interleaved StripConv and LowFreq residual modules
- Geometric Attention Guidance: Direction-aware cross-attention between ships and wakes

---

## Technology Stack

| Component | Version | Description |
|-----------|---------|-------------|
| Python | 3.10 | Programming language |
| PyTorch | 1.12.0 | Deep learning framework |
| MMCV-full | 1.5.3 - 1.8.0 | OpenMMLab computer vision library |
| MMDetection | 2.25.1 - 3.0.0 | Object detection framework |
| CUDA | 11.3 | GPU acceleration |

Key dependencies (see requirements/runtime.txt):
- mmcv-full: Core computer vision operations
- mmdet: Detection model implementations
- e2cnn: Equivariant CNN library (from git)
- numpy, matplotlib, scipy: Scientific computing
- pycocotools: COCO dataset utilities

---

## Project Structure

```
SM3Det4Wake/
├── mmrotate/                    # Main package (v0.3.4)
│   ├── apis/                    # Training and inference APIs
│   ├── core/                    # Core utilities
│   │   ├── anchor/              # Anchor generation
│   │   ├── bbox/                # Rotated bbox operations
│   │   ├── hook/                # Training hooks
│   │   │   └── dynamic_lr.py    # DSO dynamic LR hook
│   │   └── visualization/       # Visualization
│   ├── datasets/                # Dataset implementations
│   │   ├── pipelines/           # Data loading pipelines
│   │   │   └── swim_loading.py  # SWIM dataset loading
│   │   └── swim.py              # SWIM dataset class
│   ├── models/                  # Model implementations
│   │   ├── backbones/           # Backbone networks
│   │   │   ├── convnext_moe_wake.py    # Main backbone (MoE + GeometricMAMG)
│   │   │   ├── geometric_mamg.py       # Geometric attention module
│   │   │   └── wake_residual_transform.py  # Residual transform module
│   │   ├── dense_heads/         
│   │   │   └── ship_wake_head.py       # Dual heads (WakeOBB + ShipPoint)
│   │   └── detectors/
│   │       └── shipwake_dual_detector.py  # Dual detector
│   └── utils/                   # Utility functions
├── configs/                     # Configuration files
│   ├── _base_/                  # Base configurations
│   │   ├── datasets/            # Dataset configs
│   │   └── schedules/           # LR schedules
│   └── ShipWake/                # ShipWake-specific configs
│       ├── ShipWake_convnext_t.py       # Main training config
│       └── shipwake_convnext_t_debug.py # Debug config
├── tools/                       # Scripts and utilities
│   ├── train.py                 # Training script
│   ├── test.py                  # Testing script
│   ├── dist_train.sh            # Distributed training
│   ├── dist_test.sh             # Distributed testing
│   ├── validate_pipeline.py     # Pipeline validation
│   └── visualize_intermediate.py # Feature visualization
├── docker/                      # Docker configuration
├── docs/                        # Documentation
└── requirements/                # Dependency files
```

---

## Installation and Setup

### Prerequisites

- Python 3.10
- CUDA 11.3 capable GPU
- Linux/Windows environment

### Installation Steps

```bash
# 1. Create conda environment
conda create -n SM3Det4Wake python=3.10
conda activate SM3Det4Wake

# 2. Install PyTorch with CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install MMCV
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install project (editable mode)
pip install -e .
```

### Dataset Preparation

Download SWIM dataset:
```bash
curl -L -o ./swimship-wake-imagery-mass.zip https://www.kaggle.com/api/v1/datasets/download/lilitopia/swimship-wake-imagery-mass
```

Expected directory structure:
```
data/SWIM_Dataset_1.0.0/
├── Annotations/        # Wake OBB annotations (XML with robndbox)
├── Landmarks/          # Ship point+direction annotations (XML with pointtheta)
├── PNGImages/          # Image files
└── ImageSets/Main/     # Train/val/test split files
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

## Build and Run Commands

### Training

```bash
# Single GPU training
python tools/train.py configs/ShipWake/ShipWake_convnext_t.py

# Multi-GPU training (8 GPUs)
./tools/dist_train.sh configs/ShipWake/ShipWake_convnext_t.py 8

# Resume training
python tools/train.py configs/ShipWake/ShipWake_convnext_t.py --resume-from work_dirs/latest.pth

# Debug mode (small dataset)
python tools/train.py configs/ShipWake/shipwake_convnext_t_debug.py
```

### Testing and Evaluation

```bash
# Test with evaluation
python tools/test.py configs/ShipWake/ShipWake_convnext_t.py checkpoint.pth --eval mAP

# Distributed testing
./tools/dist_test.sh configs/ShipWake/ShipWake_convnext_t.py checkpoint.pth 8

# Visualize results
python tools/test.py configs/ShipWake/ShipWake_convnext_t.py checkpoint.pth --show-dir vis_results
```

### Validation and Debugging

```bash
# Validate data pipeline
python tools/validate_pipeline.py configs/ShipWake/shipwake_convnext_t_debug.py --work-dir ./work_dirs/validate

# Visualize intermediate features
python tools/visualize_intermediate.py configs/ShipWake/ShipWake_convnext_t.py checkpoint.pth --img data/SWIM_Dataset_1.0.0/PNGImages/00001.png --save-dir ./vis_results
```

---

## Code Organization and Key Modules

### Model Registration System

The project uses MMEngine/MMCV's registry pattern for modularity. From mmrotate/models/builder.py:
- ROTATED_BACKBONES = MODELS
- ROTATED_DETECTORS = MODELS
- ROTATED_HEADS = MODELS
- ROTATED_NECKS = MODELS

Models are registered using decorators:
```python
@ROTATED_DETECTORS.register_module()
class ShipWakeDualDetector(RotatedBaseDetector):
    pass
```

### Key Model Components

1. **Backbones** (mmrotate/models/backbones/)
   - convnext_moe_wake.py: Main backbone with MoE + GeometricMAMG + WakeResidual
   - geometric_mamg.py: Geometric attention module for ship-wake cross-guidance
   - wake_residual_transform.py: Interleaved residual transforms

2. **Detectors** (mmrotate/models/detectors/)
   - shipwake_dual_detector.py: Dual-task detector for ship and wake

3. **Detection Heads** (mmrotate/models/dense_heads/)
   - ship_wake_head.py: Dual heads - WakeOBBHead (rotated bbox) + ShipPointHead (point+direction)

4. **Dynamic LR Hook** (mmrotate/core/hook/dynamic_lr.py)
   - Implements DSO (Dynamic Submodule Optimization)
   - KL-divergence and min-max based adjustment policies

### Configuration System

Uses Python-based configs with inheritance:

```python
# Inherit from base configs
_base_ = [
    '../_base_/datasets/SwimShip.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Override specific settings
model = dict(...)
optimizer = dict(...)
```

---

## Code Style Guidelines

### Linting and Formatting

- **Pylint**: Configuration in `.pylintrc`
  - Ignores: CVS, configs directories
  - Fail threshold: 10.0
  
- **Formatting**: Follows OpenMMLab style (via yapf)
  - Config markers in files: `# yapf:disable` / `# yapf:enable`
  - PEP8 based style with modifications

- **Import order**: Standard library -> Third-party -> Local
  - Configured in `setup.cfg` with isort settings

### File Organization

- Each module has `__init__.py` exporting public APIs
- Builder functions in `builder.py` for each module
- Copyright header on all source files: `# Copyright (c) OpenMMLab. All rights reserved.`

### Naming Conventions

- Classes: PascalCase (e.g., ConvNeXtMoE, DynamicLrUpdaterHook)
- Functions/Variables: snake_case (e.g., build_detector, angle_version)
- Constants: UPPER_CASE (e.g., ROTATED_BACKBONES)
- Private: _leading_underscore

---

## Testing Instructions

### Running Tests

```bash
# Install test dependencies
pip install -r requirements/tests.txt

# Run pytest
pytest tests/

# Run with coverage
coverage run -m pytest tests/
coverage report
```

### Test Dependencies

From requirements/tests.txt:
- pytest: Testing framework
- coverage: Code coverage
- flake8: Linting
- yapf: Code formatting
- asynctest: Async testing

### Validation

```bash
# Validate config
python tools/misc/print_config.py configs/ShipWake/ShipWake_convnext_t.py

# Browse dataset
python tools/misc/browse_dataset.py configs/ShipWake/ShipWake_convnext_t.py
```

---

## Security Considerations

### License

- Code: Apache License 2.0 (from OpenMMLab)
- Commercial use should verify dataset and model licenses

### Dependencies

- External git dependencies in requirements/runtime.txt (e2cnn)
- Verify checksums when downloading pretrained models

### Data Handling

- Dataset paths are configurable; avoid hardcoding absolute paths
- Cache files are stored in cache/ directory (created automatically)

---

## Key Configuration Parameters

### ShipWake Model Config

Key parameters in configs (e.g., configs/ShipWake/ShipWake_convnext_t.py):

```python
# MoE Configuration
num_experts = 4          # Number of experts in MoE layers (wake, ship, background, mixed)
top_k = 2                # Top-k experts to activate
MoE_Block_inds = [[], [], [0,2,4], [0,2]]  # Which blocks use MoE

# Wake-specific enhancements
use_geometric_mamg = True   # Enable geometric attention
use_wake_residual = True    # Enable residual connections
mamg_alpha = 0.2           # Cross-guidance strength
mamg_beta = 0.5            # Direction alignment weight
residual_lambda = 0.1      # Residual fusion weight

# Dual detection heads
bbox_head = dict(
    type='ShipWakeDualHead',
    wake_head_cfg=dict(...),    # WakeOBBHead config
    ship_head_cfg=dict(...)     # ShipPointHead config
)

# Dynamic LR Adjustment (DSO)
lr_config = dict(
    policy='dynamic',
    extra_args={'T': 3, 'b': 0.4, 'ema': 0.001, 
                'backbone_policy': 'sigmoid_kl', 
                'head_policy': 'normal'},
    reweight_losses={...}  # Map loss names to modules
)
```

### Target Types

| Target Type | Annotation Format | Detection Head | Output Dimension |
|-------------|-------------------|----------------|------------------|
| Ship | Point + Direction | ShipPointHead | (x, y, cos, sin) |
| Wake | Rotated Bounding Box | WakeOBBHead | (x, y, w, h, angle) |

---

## Troubleshooting

### Common Issues

1. **MMCV Version Mismatch**
   ```
   AssertionError: MMCV==x.x.x is used but incompatible.
   ```
   Solution: Install correct MMCV version for your PyTorch/CUDA

2. **CUDA Out of Memory**
   - Reduce samples_per_gpu in config
   - Enable mixed precision: fp16 = dict(loss_scale='dynamic')

3. **Dataset Not Found**
   - Ensure data_root in config points to correct location
   - Check ImageSets/Main/*.txt files exist

---

## References

- MMRotate: https://github.com/open-mmlab/mmrotate
- SM3Det: https://github.com/zcablii/SM3Det
- SWIM Dataset: https://www.kaggle.com/datasets/lilitopia/swimship-wake-imagery-mass
- OpenMMLab: https://openmmlab.com/
