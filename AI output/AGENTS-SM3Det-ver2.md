# SM3Det4Wake Project Guide for AI Agents

## Project Overview

SM3Det4Wake is a research project for **Multi-Modal Remote Sensing Object Detection**. It implements SM3Det (Single Model for Multi-Modal datasets and Multi-Task object Detection), which is designed to detect objects from various remote sensing sensor modalities (SAR, RGB, Infrared) using a single unified model.

Key innovations:
- **Grid-level sparse Mixture of Experts (MoE)** backbone for joint knowledge learning while preserving modality-specific features
- **Dynamic Learning Rate Adjustment (DLA)** for handling varying learning complexities across tasks and modalities
- Support for both horizontal and oriented bounding box detection

The project is based on [OpenMMLab](https://github.com/open-mmlab)'s MMRotate framework and extends it for multi-modal scenarios.

---

## Technology Stack

| Component | Version | Description |
|-----------|---------|-------------|
| Python | 3.10 | Programming language |
| PyTorch | 1.12.0 | Deep learning framework |
| MMCV | 1.5.3 - 1.8.0 | OpenMMLab computer vision library |
| MMDetection | 2.25.1 - 3.0.0 | Object detection framework |
| CUDA | 11.3 | GPU acceleration |

Key dependencies:
- `mmcv-full`: Core computer vision operations
- `mmdet`: Detection model implementations
- `timm`: Pretrained models and layers
- `scipy`, `scikit-learn`: Scientific computing

---

## Project Structure

```
SM3Det4Wake/
├── mmrotate/                 # Main package (v0.3.4)
│   ├── apis/                 # Training and inference APIs
│   ├── core/                 # Core utilities (bbox, anchor, evaluation, hooks)
│   │   ├── anchor/           # Anchor generation utilities
│   │   ├── bbox/             # Rotated bbox operations (assigners, samplers, coders)
│   │   ├── evaluation/       # Evaluation metrics
│   │   └── hook/             # Training hooks (dynamic_lr.py for DLA)
│   ├── datasets/             # Dataset implementations and pipelines
│   │   ├── pipelines/        # Data augmentation and transforms
│   │   └── samplers/         # Multi-source dataset samplers
│   ├── models/               # Model implementations
│   │   ├── backbones/        # Backbone networks (ConvNeXt MoE, LSK, VAN, etc.)
│   │   ├── dense_heads/      # Detection heads (RPN, Retina, FCOS, etc.)
│   │   ├── detectors/        # Detector implementations
│   │   ├── losses/           # Loss functions
│   │   ├── necks/            # FPN variants
│   │   └── roi_heads/        # RoI-based detection heads
│   └── utils/                # Utility functions
├── configs/                  # Configuration files
│   ├── _base_/               # Base configurations
│   │   ├── datasets/         # Dataset configs (SOI_Det, DOTA, HRSC, etc.)
│   │   └── schedules/        # Learning rate schedules
│   ├── SM3Det/               # SM3Det model configs
│   └── ShipWake/             # ShipWake-specific configs
├── tools/                    # Scripts and utilities
│   ├── train.py              # Training script
│   ├── test.py               # Testing/evaluation script
│   ├── dist_train.sh         # Distributed training launcher
│   ├── dist_test.sh          # Distributed testing launcher
│   ├── analysis_tools/       # Log analysis, benchmarking
│   └── cam/                  # Class Activation Mapping tools
├── mmcv/                     # Custom MMCV library (embedded)
├── mmcv_custom/              # MMCV extensions (optimizers, checkpoint loaders)
├── SWIM-dataset/             # Ship Wake dataset samples
└── docs/                     # Documentation images
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
conda create -n SM3Det python==3.10
conda activate SM3Det

# 2. Install PyTorch with CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install MMCV
pip install mmcv-full==1.6.1 -f \
    https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# 4. Install requirements
pip install -r requirements.txt

# 5. Install custom mmcv
cd mmcv
python setup.py install
cd ..

# 6. Install mmrotate
pip install -e .
```

---

## Build and Run Commands

### Training

```bash
# Single GPU training
python tools/train.py configs/SM3Det/SM3Det_convnext_t.py

# Distributed training (8 GPUs)
sh ./tools/dist_train.sh configs/SM3Det/SM3Det_convnext_t.py 8

# Resume training
python tools/train.py configs/SM3Det/SM3Det_convnext_t.py --resume-from work_dirs/latest.pth

# Auto-resume from latest checkpoint
python tools/train.py configs/SM3Det/SM3Det_convnext_t.py --auto-resume
```

### Testing and Evaluation

```bash
# Test with evaluation
python tools/test.py configs/SM3Det/SM3Det_convnext_t.py checkpoint.pth --eval bbox

# Distributed testing
sh ./tools/dist_test.sh configs/SM3Det/SM3Det_convnext_t.py checkpoint.pth 8

# Format results for submission (without evaluation)
python tools/test.py configs/SM3Det/SM3Det_convnext_t.py checkpoint.pth \
    --format-only --eval-options submission_dir=./results

# Visualize results
python tools/test.py configs/SM3Det/SM3Det_convnext_t.py checkpoint.pth \
    --show-dir vis_results
```

### Dataset Preparation

Dataset configurations are in `configs/_base_/datasets/`. Supported datasets:
- **SOI-Det**: Multi-modal dataset (SAR, RGB, Infrared)
- **SARDet-50K**: SAR object detection dataset
- **DOTA**: Aerial image dataset
- **DroneVehicle**: RGB-Infrared vehicle dataset
- **HRSC2016**: Ship detection dataset
- **SSDD**: SAR ship detection dataset

Data should be organized under `data/` directory as specified in config files.

---

## Code Organization and Key Modules

### Model Registration System

The project uses MMEngine/MMCV's registry pattern for modularity:

```python
# From mmrotate/models/builder.py
ROTATED_BACKBONES = MODELS
ROTATED_DETECTORS = MODELS
ROTATED_HEADS = MODELS
ROTATED_NECKS = MODELS
```

Models are registered using decorators:
```python
@ROTATED_DETECTORS.register_module()
class TriSourceOneOneDetector(RotatedBaseDetector):
    ...
```

### Key Model Components

1. **Backbones** (`mmrotate/models/backbones/`)
   - `convnext_moe.py`: ConvNeXt with sparse MoE
   - `lsk_moe.py`, `van_moe.py`: Other backbone variants with MoE
   - `convnext_moe_wake.py`: ShipWake-specific variant

2. **Detectors** (`mmrotate/models/detectors/`)
   - `trisource_H1stage_R1stage_detector.py`: Tri-source detector (1-stage + 1-stage)
   - `trisource_H1stage_R2stage_detector.py`: Tri-source detector (1-stage + 2-stage)
   - `trisource_H2stage_R1stage_detector.py`: Tri-source detector (2-stage + 1-stage)
   - `trisource_H2stage_R2stage_detector.py`: Tri-source detector (2-stage + 2-stage)

3. **Dynamic Learning Rate Hook** (`mmrotate/core/hook/dynamic_lr.py`)
   - Implements DLA (Dynamic Learning Rate Adjustment)
   - Supports KL-divergence based and min-max based adjustment policies

### Configuration System

Uses Python-based configs with inheritance:

```python
_base_ = [
    '../_base_/datasets/SOI_Det.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Override specific settings
model = dict(...)
optimizer = dict(...)
```

---

## Development Conventions

### Code Style

- **Linting**: Pylint configuration in `.pylintrc`
  - Ignores: `CVS`, `configs`
  - Fail threshold: 10.0
  
- **Formatting**: Follows OpenMMLab style (via `yapf`)
  - Config markers in files: `# yapf:disable` / `# yapf:enable`

- **Import order**: Standard library → Third-party → Local
  ```python
  import torch
  import torch.nn as nn
  from mmcv.runner import BaseModule
  from ..builder import ROTATED_BACKBONES
  ```

### File Organization

- Each module has `__init__.py` exporting public APIs
- Builder functions in `builder.py` for each module
- Copyright header on all source files: `# Copyright (c) OpenMMLab. All rights reserved.`

### Naming Conventions

- Classes: `PascalCase` (e.g., `ConvNeXtMoE`, `DynamicLrUpdaterHook`)
- Functions/Variables: `snake_case` (e.g., `build_detector`, `angle_version`)
- Constants: `UPPER_CASE` (e.g., `ROTATED_BACKBONES`)
- Private: `_leading_underscore`

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

### Test Structure

Test-related files:
- `requirements/tests.txt`: Test dependencies
- `tools/analysis_tools/`: Analysis and benchmarking utilities
  - `analyze_logs.py`: Training log analysis
  - `benchmark.py`: Inference speed benchmarking
  - `get_flops.py`: FLOPs calculation

### Validation

```bash
# Validate config
python tools/misc/print_config.py configs/SM3Det/SM3Det_convnext_t.py

# Browse dataset
python tools/misc/browse_dataset.py configs/SM3Det/SM3Det_convnext_t.py
```

---

## Security Considerations

1. **License**: 
   - Code: Apache License 2.0 (from OpenMMLab)
   - Model weights/usage: Creative Commons Attribution-NonCommercial 4.0 International
   - **Commercial use requires formal permission**

2. **Dependencies**:
   - External git dependencies in `requirements/runtime.txt` (e2cnn)
   - Verify checksums when downloading pretrained models

3. **Data Handling**:
   - Dataset paths are configurable; avoid hardcoding absolute paths
   - Cache files are stored in `cache/` directory (created automatically)

4. **Model Serving**:
   - Deployment tools in `tools/deployment/` for TorchServe integration
   - Custom handler provided (`mmrotate_handler.py`)

---

## Key Configuration Parameters

### SM3Det Model Config

Key parameters in configs (e.g., `configs/SM3Det/SM3Det_convnext_t.py`):

```python
# MoE Configuration
num_experts = 8          # Number of experts in MoE layers
top_k = 3                # Top-k experts to activate
MoE_Block_inds = [[],[],[i*2 for i in range(5)],[0,2]]  # Which blocks use MoE

# Multi-modal branches
branch_field = ['sar', 'rgb', 'ifr']  # SAR, RGB, Infrared
source_ratio = [2,1,1]   # Sampling ratio for each modality

# Training
total_images = 46260+25028+17990  # Total training samples
gpus = 8
batch_size = sum(source_ratio)

# Dynamic LR Adjustment
lr_config = dict(
    policy='dynamic',
    extra_args={'T':3, 'b':0.4, 'ema': 0.001, 
                'backbone_policy':'sigmoid_kl', 
                'head_policy':'normal'},
    reweight_losses={...}  # Map loss names to modules
)
```

---

## Troubleshooting

### Common Issues

1. **MMCV Version Mismatch**
   ```
   AssertionError: MMCV==x.x.x is used but incompatible.
   ```
   Solution: Install correct MMCV version for your PyTorch/CUDA

2. **CUDA Out of Memory**
   - Reduce `samples_per_gpu` in config
   - Enable mixed precision: `fp16 = dict(loss_scale='dynamic')`

3. **Dataset Not Found**
   - Ensure `data_root` in config points to correct location
   - Check cache directory permissions

---

## References

- Paper: [SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](http://arxiv.org/abs/2412.20665)
- Official Implementation: https://github.com/zcablii/SM3Det
- Dataset: [SOI-Det on Kaggle](https://www.kaggle.com/datasets/greatbird/soi-det)
- MMRotate Docs: https://mmrotate.readthedocs.io/
