# SM3Det Project - AI Agent Guide

## Project Overview

SM3Det is a unified model for Multi-Modal Remote Sensing Object Detection. It is built on top of the MMRotate framework (OpenMMLab's rotation detection toolbox) and extends it with multi-modal learning capabilities.

The project contains two main components:

1. **SM3Det**: A unified model for Multi-Modal datasets and Multi-Task object Detection (M2Det) that handles:
   - Multi-modal inputs: SAR (Synthetic Aperture Radar), RGB optical imagery, and Infrared (IFR) imagery
   - Multiple task types: Horizontal bounding box detection (HBB) and Oriented bounding box detection (OBB)
   - Uses a grid-level sparse Mixture of Experts (MoE) backbone to capture shared knowledge while preserving modality-specific representations

2. **BabelRS**: A language-pivoted pretraining framework for heterogeneous multi-modal remote sensing detection that:
   - Uses Concept-Shared Instruction Aligning (CSIA) to align sensor modalities to linguistic concepts
   - Implements Layerwise Visual-Semantic Annealing (LVSA) for fine-grained semantic guidance
   - Based on InternVL architecture with ViT backbone

## Technology Stack

- **Framework**: PyTorch 1.12.0+, MMRotate 0.3.4, MMCV 1.5.3-1.8.0, MMDetection 2.25.1-3.0.0
- **Language**: Python 3.10+
- **Key Dependencies**:
  - PyTorch ecosystem: torch, torchvision, torchaudio
  - OpenMMLab: mmcv-full, mmdet
  - Scientific computing: numpy, scipy, scikit-learn
  - Vision: pycocotools, matplotlib, PIL
  - Special: e2cnn (for rotation equivariant CNNs)

## Project Structure

```
SM3Det4Wake/
├── mmrotate/                    # Core library code
│   ├── apis/                    # Training and inference APIs
│   ├── core/                    # Core components
│   │   ├── anchor/             # Anchor generation for rotated objects
│   │   ├── bbox/               # Bounding box operations (assigners, coders, samplers)
│   │   ├── evaluation/         # Evaluation metrics
│   │   ├── hook/               # Training hooks (includes DynamicLrUpdaterHook)
│   │   ├── patch/              # Image patching utilities
│   │   ├── post_processing/    # NMS and post-processing
│   │   └── visualization/      # Visualization utilities
│   ├── datasets/                # Dataset implementations
│   │   ├── pipelines/          # Data preprocessing transforms
│   │   └── samplers/           # Multi-source data samplers
│   ├── models/                  # Model architectures
│   │   ├── backbones/          # Backbone networks (ConvNeXt MoE, LSK, VAN, InternViT)
│   │   ├── dense_heads/        # Detection heads (RPN, RetinaNet, FCOS, etc.)
│   │   ├── detectors/          # Detector implementations
│   │   ├── losses/             # Loss functions
│   │   ├── necks/              # FPN variants
│   │   ├── roi_heads/          # RoI heads for two-stage detectors
│   │   └── utils/              # Model utilities
│   └── utils/                   # Utility functions
├── configs/                     # Model configurations
│   ├── _base_/                 # Base configurations
│   │   ├── datasets/           # Dataset configs (DOTA, SARDet, DroneVehicle, SOI_Det)
│   │   ├── schedules/          # Training schedules
│   │   └── default_runtime.py  # Default runtime settings
│   └── SM3Det/                 # SM3Det model configs
├── local_configs/               # Local/experimental configurations
├── BabelRS_configs/             # BabelRS-specific configurations
├── BabelRS_pretrain/            # BabelRS pretraining code
│   ├── internvl/               # InternVL model implementation
│   ├── eval/                   # Evaluation scripts
│   └── tools/                  # Training/pretraining scripts
├── tools/                       # Training and testing scripts
│   ├── train.py                # Main training script
│   ├── test.py                 # Main testing script
│   ├── dist_train.sh           # Distributed training launcher
│   ├── dist_test.sh            # Distributed testing launcher
│   └── analysis_tools/         # Analysis utilities
├── mmcv/                        # Custom MMCV modifications
└── mmcv_custom/                 # Additional MMCV customizations
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.3+
- GCC 5+

### Step-by-step Installation

```bash
# 1. Clone repository
git clone https://github.com/zcablii/SM3Det.git
cd SM3Det

# 2. Create conda environment
conda create -n SM3Det python==3.10
conda activate SM3Det

# 3. Install PyTorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html

# 4. Install MMCV
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# 5. Install dependencies
pip install -r requirements.txt

# 6. Install custom mmcv
cd mmcv
python setup.py install
cd ..

# 7. Install mmrotate
pip install -e .
```

## Build and Test Commands

### Training

```bash
# Single GPU training
python tools/train.py ${CONFIG_FILE} [optional arguments]

# Multi-GPU training
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]

# Example: Train SM3Det with 8 GPUs
./tools/dist_train.sh configs/SM3Det/SM3Det_convnext_t.py 8

# Example: Train BabelRS
./tools/dist_train.sh BabelRS_configs/BabelRS_20kstep.py 8
```

### Testing/Evaluation

```bash
# Single GPU testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# Multi-GPU testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]

# Evaluate with mAP metric
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mAP

# Format results for submission (DOTA format)
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --format-only --eval-options submission_dir=work_dirs/results
```

### Common Optional Arguments

- `--work-dir ${WORK_DIR}`: Override working directory
- `--resume-from ${CHECKPOINT_FILE}`: Resume training from checkpoint
- `--auto-resume`: Auto-resume from latest checkpoint
- `--no-validate`: Disable validation during training
- `--gpu-ids ${GPU_IDS}`: Specify GPU IDs
- `--cfg-options`: Override config settings (key=value format)

## Code Style Guidelines

The project follows OpenMMLab coding standards:

### Import Ordering (isort)
- Line length: 79 characters
- Order: standard library → third-party → first-party (mmrotate)
- See `setup.cfg` for isort configuration

### Code Formatting (yapf)
- Based on PEP8 style
- Blank lines before nested classes/functions
- Split expressions after opening parentheses

### Running Linters

```bash
# Code formatting
yapf -i mmrotate/models/your_file.py

# Import sorting
isort mmrotate/models/your_file.py

# Linting
flake8 mmrotate/
```

## Key Architecture Components

### 1. Multi-Modal Detector (TriSourceDetector)

Located in `mmrotate/models/detectors/trisource_H1stage_R2stage_detector.py`:

- Handles three modalities: SAR, RGB, IFR
- Separate heads for each modality:
  - SAR: Single-stage detector (GFLHead)
  - RGB: Two-stage detector (RPN + RoI Head)
  - IFR: Two-stage detector (RPN + RoI Head)
- Shared backbone with MoE for cross-modal learning
- Dynamic learning rate adjustment based on loss history

### 2. MoE Backbone (ConvNeXt_moe_MultiInput)

Located in `mmrotate/models/backbones/convnext_moe.py`:

- Grid-level sparse Mixture of Experts
- Dynamic routing for modality-specific features
- Configurable expert number and top-k routing
- Supports multiple ConvNeXt variants (tiny, small, base)

### 3. Dynamic Learning Rate Hook

Located in `mmrotate/core/hook/dynamic_lr.py`:

- `DynamicLrUpdaterHook`: Adjusts learning rates based on training dynamics
- Supports multiple policies: min, avg, max, kl, sigmoid_kl
- EMA-based loss tracking for stable updates

### 4. Multi-Task Data Loading

Located in `mmrotate/datasets/samplers/multi_source_sampler.py`:

- Balanced sampling from multiple datasets
- Configurable source ratios

## Configuration System

Configurations use Python files with inheritance:

```python
_base_ = [
    '../_base_/datasets/SOI_Det.py',      # Dataset config
    '../_base_/schedules/schedule_1x.py',  # Training schedule
    '../_base_/default_runtime.py'         # Runtime settings
]

# Override specific settings
model = dict(
    backbone=dict(...),
    neck=dict(...),
    ...
)
```

### Key Configuration Parameters

- `source_ratio`: Sampling ratio for multi-dataset training (e.g., [2,1,1] for SAR:RGB:IFR)
- `MoE_Block_inds`: Which blocks to apply MoE (list of layer indices)
- `num_experts`/`top_k`: MoE routing parameters
- `lr_config`: Learning rate policy with dynamic adjustment parameters
- `reweight_losses`: Mapping of loss names to network components for dynamic LR

## Dataset Preparation

### SOI-Det Dataset

The main multi-modal dataset combining:
- **SARDet-50K**: SAR imagery with horizontal bounding boxes
- **DOTA**: RGB optical imagery with oriented bounding boxes
- **DroneVehicle**: Infrared imagery with oriented bounding boxes

Download from: https://www.kaggle.com/datasets/greatbird/soi-det

### Dataset Structure

```
data/SOI_Det/
├── SARDet_50K/
│   ├── Annotations/
│   └── JPEGImages/
├── DOTA_800pix/
│   ├── train/
│   └── val/
└── DroneVehicle/
    └── dota_train/
```

### Pre-trained Models

- BabelRS ViT-Large: Available on HuggingFace
- SM3Det checkpoints: Available on Kaggle

## Testing Instructions

### Unit Tests

```bash
# Run pytest
pytest tests/

# Run specific test
pytest tests/test_specific_module.py
```

### Model Evaluation

```bash
# Evaluate on validation set
python tools/test.py configs/SM3Det/SM3Det_convnext_t.py checkpoints/model.pth --eval mAP

# Generate submission for DOTA online evaluation
python tools/test.py configs/SM3Det/SM3Det_convnext_t.py checkpoints/model.pth --format-only --eval-options submission_dir=./results
```

## Common Development Tasks

### Adding a New Backbone

1. Create file in `mmrotate/models/backbones/`
2. Register with `@ROTATED_BACKBONES.register_module()`
3. Implement `forward()` method
4. Add to `mmrotate/models/backbones/__init__.py`
5. Create config file in `configs/`

### Adding a New Dataset

1. Create dataset class in `mmrotate/datasets/`
2. Register with `@DATASETS.register_module()`
3. Add data pipeline in `configs/_base_/datasets/`
4. Update `mmrotate/datasets/__init__.py`

### Adding a New Loss

1. Create loss class in `mmrotate/models/losses/`
2. Register with `@LOSSES.register_module()`
3. Implement `forward()` method
4. Add to `mmrotate/models/losses/__init__.py`

## Security Considerations

- Model checkpoint files are loaded with `torch.load()` - ensure checkpoints come from trusted sources
- Dataset paths in configs use relative paths - verify data integrity
- The project uses pickle for caching - don't load caches from untrusted sources
- No network operations in core training code except for distributed training initialization

## License

This project is licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) for non-commercial use only. Commercial use requires formal permission.

## Important Notes for AI Agents

1. **Config Inheritance**: Always understand the `_base_` inheritance chain when modifying configs
2. **Multi-Modal Handling**: The code uses `gather_dict_values()` to handle multi-modal batching - ensure proper key naming (sar/rgb/ifr)
3. **Dynamic LR**: When modifying training schedules, be aware of the DynamicLrUpdaterHook and its interaction with loss computation
4. **MoE Components**: Expert routing depends on dataset tags - ensure `datasets` parameter is correctly passed through the pipeline
5. **Version Compatibility**: Strict version requirements for MMCV (1.5.3-1.8.0) and MMDetection (2.25.1-3.0.0) - check compatibility before upgrading
