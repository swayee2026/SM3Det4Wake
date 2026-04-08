## This repository is the official implementation of ArXiv "[SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](http://arxiv.org/abs/2412.20665 )"



## [SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](http://arxiv.org/abs/2412.20665 )

With the rapid advancement of remote sensing technology, high-resolution multi-modal imagery is now more widely accessible. Conventional Object detection models are trained on a single dataset, often restricted to a specific imaging modality and annotation format. However, such an approach overlooks the valuable shared knowledge across multi-modalities and limits the model's applicability in more versatile scenarios. This paper introduces a new task called Multi-Modal Datasets and Multi-Task Object Detection (M2Det) for remote sensing, designed to accurately detect horizontal or oriented objects from any sensor modality. This task poses challenges due to 1) the trade-offs involved in managing multi-modal modelling and 2) the complexities of multi-task optimization. To address these, we establish a benchmark dataset and propose a unified model, SM3Det (Single Model for Multi-Modal datasets and Multi-Task object Detection). SM3Det leverages a grid-level sparse MoE backbone to enable joint knowledge learning while preserving distinct feature representations for different modalities. Furthermore, it integrates a consistency and synchronization optimization strategy using dynamic learning rate adjustment, allowing it to effectively handle varying levels of learning difficulty across modalities and tasks. Extensive experiments demonstrate SM3Det's effectiveness and generalizability, consistently outperforming specialized models on individual datasets.

![net_arch](docs/SM3Det.png)

## SM3Det Model 

**Model Architecture:**

- We propose integrating a plug-and-play grid-level sparse Mixture of Experts (MoE) architecture into backbone networks, enabling the model to capture both shared knowledge and modality-specific representations. Through dynamic routing, the experts operate on local spatial features, allowing the model to adaptively process information at a grid level, which is crucial for object detection tasks. 

**Model Optimization:**

- We propose a novel Dynamic Learning Rate Adjustment (DLA) method that adaptively adjusts the learning rates of different network components with tailored policies. DLA accommodates the varying learning complexities across different tasks and modalities by balancing the relative convergence rate and guaranteeing optimization direction consistency. 
Unlike traditional techniques that primarily modify loss weights or gradients—often lacking precise manipulation over specific network submodules or suffering from inefficiencies—our DLA provides fine-grained control while maintaining optimization efficiency.

Main configuration files are put under configs/SM3Det/


**SOI-Det DATASET DOWNLOAD at:** 

* [Dataset](https://www.kaggle.com/datasets/greatbird/soi-det) 



## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

# Usage
- Clone this repository:
```
git clone https://github.com/zcablii/SM3Det.git
```
## Object Detection
### Installation

- Create a conda environment:
```
cd SM3Det
conda create -n SM3Det python==3.10
conda activate SM3Det
```

- Install the required packages:
```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install -r requirements.txt
```

- Install mmcv:
```
cd ../mmcv
python setup.py install
cd ../mmrotate
```
- Install mmrotate:
```
pip install -e .
```

### Train
```
sh ./tools/dist_train.sh BabelRS_configs/BabelRS_20kstep.py 8 
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
