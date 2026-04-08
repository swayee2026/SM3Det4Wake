# SM3Det4Wake: Ship Wake Detection in Optical Remote Sensing

基于 SM3Det 的光学遥感图像船只尾迹目标检测框架

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.12.0-orange.svg)](https://pytorch.org/)

## 目录

- [项目概述](#项目概述)
- [使用场景与实验目标](#使用场景与实验目标)
- [与 SM3Det 的异同](#与-sm3det-的异同)
- [关键模块介绍](#关键模块介绍)
- [数据通道概览](#数据通道概览)
- [安装与使用](#安装与使用)
- [项目结构](#项目结构)
- [引用](#引用)

## 项目概述

SM3Det4Wake 是一个针对**光学遥感图像中船只尾迹检测**任务而设计的深度学习框架。本项目基于 SM3Det (Single Model for Multi-Modal datasets and Multi-Task object Detection) 的多模态检测架构，针对**单模态（光学）、双目标（船只点+尾迹框）**的特殊场景进行了定制化改进。

### 核心特性

- 🚢 **双任务检测**: 同时检测船只（点回归）和尾迹（旋转框检测）
- 🔄 **残差连接增强**: 交错的 StripConv 和 LowFreq 残差模块
- 🧭 **几何注意力引导**: 船只与尾迹互相关的方向感知注意力机制
- ⚡ **异步学习率优化**: 基于 DSO (Dynamic Submodule Optimization) 的差异化训练策略
- 🎯 **Grid-level MoE**: 稀疏专家混合网络用于多尺度特征学习

## 使用场景与实验目标

### 应用场景

本项目主要针对以下场景：

1. **港口监控**: 检测进出港船只及其尾迹轨迹
2. **航道管理**: 监测船只航行方向和尾迹扩散
3. **海洋环境监测**: 通过尾迹分析船只速度和载重
4. **海上搜救**: 通过尾迹追踪失踪船只

### 实验目标

| 目标类型 | 标注格式 | 检测头 | 输出维度 |
|---------|---------|--------|---------|
| 船只 (Ship) | 点标注 + 方向 | ShipPointHead | `(x, y, cosθ, sinθ)` |
| 尾迹 (Wake) | 旋转框 | WakeOBBHead | `(x, y, w, h, θ)` |

**数据集**: SWIM-Ship Wake Imagery Mass (光学遥感图像，768×768 分辨率)

## 与 SM3Det 的异同

### 沿用模块

| 模块 | 说明 | 文件位置 |
|------|------|----------|
| **Grid-level MoE** | 稀疏专家混合网络，8 experts / top-2 | `convnext_moe.py` |
| **ConvNeXt Backbone** | 分层特征提取，4-stage 架构 | `convnext_moe_wake.py` |
| **DSO 学习率调整** | 基于损失差异的动态学习率 | `dynamic_lr.py` |
| **Router 机制** | Cosine-gating 专家选择 | `MoE_layer` class |

### 本项目创新改进

#### 1. 残差连接模块 (Residual Block)

**改进点**: 在 backbone 的每一层新增与 MoE 并行的残差连接

**交错配置**:
| Stage | 分辨率 | 残差类型 | 设计目标 |
|-------|--------|---------|---------|
| 1 | 200×200 | LowFreqResidual (5×5) | 保留船只高频纹理 |
| 2 | 100×100 | StripConvResidual (1×7, 7×1) | 提取尾迹线性结构 |
| 3 | 50×50 | LowFreqResidual (7×7) | 强化深层语义纹理 |
| 4 | 25×25 | StripConvResidual (1×11, 11×1) | 强化大尺寸尾迹结构 |

**文件**: `mmrotate/models/backbones/wake_residual_transform.py`

#### 2. 双流几何注意力引导 (GeometricMAMG)

**改进点**: 增强方向一致性计算

```python
# 原 SM3Det
F_guided = F_input * (1 + α * Mask_conf)

# 本改进 (位置+方向)
directional_attention = Mask_ship_conf * (1 + cos(θ_ship - θ_wake))
F_wake_guided = F_input * (1 + α * directional_attention)
```

**输出通道**:
- `Mask_ship = [Confidence_map, Direction_map (cosθ, sinθ)]`
- `Mask_wake = [Confidence_map, Direction_map (cosθ, sinθ)]`

**文件**: `mmrotate/models/backbones/geometric_mamg.py`

#### 3. 差异化检测头 (Dual Detection Heads)

**改进点**: 针对不同的目标类型设计专门的检测头

**WakeOBBHead**:
- 任务: 尾迹旋转框检测
- 输出: 分类分数 + 5维框回归 `(x, y, w, h, θ)`
- 损失: Focal Loss + Smooth L1

**ShipPointHead**:
- 任务: 船只点位置 + 方向回归
- 输出: 位置偏移 `(dx, dy)` + 方向 `(cosθ, sinθ)` + 置信度
- 损失: Smooth L1 (位置) + Cosine Similarity (方向) + Focal Loss (置信度)
- Target Assignment: Center Sampling (radius=1.5 × stride)

**文件**: `mmrotate/models/dense_heads/ship_wake_head.py`

## 关键模块介绍

### 1. ConvNeXt_moe_wake (Backbone)

```python
ConvNeXt_moe_wake(
    arch='tiny',
    MoE_Block_inds=[[], [], [0,2,4], [0,2]],  # Stage 3-4 使用 MoE
    num_experts=4,
    top_k=2,
    use_geometric_mamg=True,    # 启用几何注意力
    use_wake_residual=True,     # 启用残差连接
)
```

### 2. ShipWakeDualHead (Detection Head)

```python
ShipWakeDualHead(
    in_channels=256,
    wake_head_cfg={
        'type': 'WakeOBBHead',
        'num_classes': 1,
        'loss_cls': FocalLoss,
        'loss_bbox': SmoothL1Loss,
    },
    ship_head_cfg={
        'type': 'ShipPointHead',
        'center_sampling_radius': 1.5,
        'loss_center': SmoothL1Loss,
        'loss_direction': CosineSimilarityLoss,
        'loss_conf': FocalLoss,
    }
)
```

### 3. DynamicLrUpdaterHook (DSO)

```python
DynamicLrUpdaterHook(
    policy='dynamic',
    extra_args={
        'T': 3,
        'b': 0.4,
        'ema': 0.001,
        'backbone_policy': 'sigmoid_kl',
        'head_policy': 'normal'
    },
    reweight_losses={
        'wake_loss_cls': 'bbox_head.wake_head',
        'ship_loss_center': 'bbox_head.ship_head',
        # ...
    }
)
```

### 4. SWIMDataset (Data Loader)

```python
SWIMDataset(
    ann_file='ImageSets/Main/train.txt',
    img_prefix='SWIM_Dataset_1.0.0/',
    wake_ann_dir='Annotations',      # XML with <robndbox>
    ship_ann_dir='Landmarks',        # XML with <pointtheta>
    img_dir='PNGImages',
)
```

## 数据通道概览

```
Input Image (B, 3, 800, 800)
    |
    v
[Backbone: ConvNeXt_moe_wake]
    ├── Stage 1: WakeResidual(LowFreq) + GeometricMAMG → (B, 96, 200, 200)
    ├── Stage 2: WakeResidual(StripConv) + GeometricMAMG → (B, 192, 100, 100)
    ├── Stage 3: MoE + WakeResidual(LowFreq) + GeometricMAMG → (B, 384, 50, 50)
    └── Stage 4: MoE + WakeResidual(StripConv) + GeometricMAMG → (B, 768, 25, 25)
    |
    v
[Neck: FPN] → Features P2-P6
    |
    v
[BBox Head: ShipWakeDualHead]
    |
    ├── WakeOBBHead
    │   ├── cls_score: (B, num_anchors, H, W)
    │   └── bbox_pred: (B, num_anchors×5, H, W)
    │
    └── ShipPointHead
        ├── center_pred: (B, num_anchors×2, H, W)
        ├── direction_pred: (B, num_anchors×2, H, W)
        └── conf_pred: (B, num_anchors, H, W)
    |
    v
Loss Computation
    ├── Wake: loss_cls + loss_bbox
    ├── Ship: loss_center + loss_direction + loss_conf
    └── MoE: gate_loss (load balancing)
```

### 数据格式说明

#### Wake 标注 (Annotations/00001.xml)
```xml
<object>
  <type>robndbox</type>
  <name>wake</name>
  <robndbox>
    <cx>602.8032</cx>    <!-- 中心 x -->
    <cy>53.0397</cy>    <!-- 中心 y -->
    <w>44.4618</w>      <!-- 宽度 -->
    <h>96.8959</h>      <!-- 高度 -->
    <angle>0.53</angle> <!-- 角度 (弧度) -->
  </robndbox>
</object>
```

#### Ship 标注 (Landmarks/00001.xml)
```xml
<object>
  <type>pointtheta</type>
  <name>wake</name>
  <pointtheta>
    <px>581.688</px>              <!-- 点 x -->
    <py>83.013</py>               <!-- 点 y -->
    <theta1>-1.230</theta1>      <!-- 尾迹方向1 -->
    <theta2>-0.749</theta2>      <!-- 尾迹方向2 -->
  </pointtheta>
</object>
<!-- Ship方向 = opposite_of(mean(theta1, theta2)) -->
```

## 安装与使用

### 环境要求

- Python 3.10
- PyTorch 1.12.0
- CUDA 11.3
- MMCV-full 1.6.1
- MMDetection 2.25.1+

### 安装步骤

```bash
# 1. 创建环境
conda create -n SM3Det4Wake python=3.10
conda activate SM3Det4Wake

# 2. 安装 PyTorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 3. 安装 MMCV
pip install mmcv-full==1.6.1 -f \
    https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装本项目
pip install -e .
```

### 数据准备

```bash
# 数据集目录结构
data/
└── SWIM_Dataset_1.0.0/
    ├── Annotations/        # Wake XML annotations
    ├── Landmarks/          # Ship point+direction XML annotations
    ├── PNGImages/          # Image files
    └── ImageSets/Main/     # train.txt, val.txt, test.txt
```

### 训练

```bash
# 单 GPU 训练
python tools/train.py configs/ShipWake/shipwake_convnext_t.py

# 多 GPU 训练 (8 GPUs)
./tools/dist_train.sh configs/ShipWake/shipwake_convnext_t.py 8

# 恢复训练
python tools/train.py configs/ShipWake/shipwake_convnext_t.py \
    --resume-from work_dirs/latest.pth

# 调试模式 (小数据集验证)
python tools/train.py configs/ShipWake/shipwake_convnext_t_debug.py
```

### 验证

```bash
# 运行验证脚本
python tools/validate_pipeline.py \
    configs/ShipWake/shipwake_convnext_t_debug.py \
    --work-dir ./work_dirs/validate
```

### 测试与推理

```bash
# 测试
python tools/test.py configs/ShipWake/shipwake_convnext_t.py \
    checkpoints/model.pth --eval bbox

# 可视化中间结果
python tools/visualize_intermediate.py \
    configs/ShipWake/shipwake_convnext_t.py \
    checkpoints/model.pth \
    --img data/SWIM_Dataset_1.0.0/PNGImages/00001.png \
    --save-dir ./vis_results
```

### 自定义配置

关键配置参数说明：

```python
# 在配置文件中修改
model = dict(
    backbone=dict(
        MoE_Block_inds=[[], [], [0,2,4], [0,2]],  # MoE 层位置
        num_experts=4,                              # 专家数量
        mamg_alpha=0.2,                            # 注意力强度
        mamg_beta=0.5,                             # 方向对齐权重
        residual_lambda=0.1,                       # 残差连接权重
    ),
    bbox_head=dict(
        ship_head_cfg=dict(
            center_sampling_radius=1.5,  # 正样本采样半径
        )
    )
)

# DSO 学习率配置
lr_config = dict(
    extra_args=dict(
        T=3,          # 温度系数
        b=0.4,        # KL 散度偏置
        ema=0.001,    # EMA 衰减率
    )
)
```

## 项目结构

```
SM3Det4Wake/
├── mmrotate/
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── convnext_moe_wake.py      # 主 backbone (含 MoE + GeometricMAMG + WakeResidual)
│   │   │   ├── geometric_mamg.py         # 几何注意力模块
│   │   │   └── wake_residual_transform.py # 残差变换模块
│   │   ├── dense_heads/
│   │   │   └── ship_wake_head.py         # 双检测头 (WakeOBBHead + ShipPointHead)
│   │   └── detectors/
│   │       └── shipwake_dual_detector.py # 主检测器
│   ├── datasets/
│   │   ├── swim.py                       # SWIM 数据集
│   │   └── pipelines/
│   │       └── swim_loading.py           # 自定义数据加载
│   └── core/
│       └── hook/
│           └── dynamic_lr.py             # DSO 学习率调整
├── configs/
│   └── ShipWake/
│       ├── shipwake_convnext_t.py        # 完整训练配置
│       └── shipwake_convnext_t_debug.py  # 调试配置
├── tools/
│   ├── train.py                          # 训练脚本
│   ├── test.py                           # 测试脚本
│   ├── validate_pipeline.py              # 数据通道验证
│   └── visualize_intermediate.py         # 可视化工具
└── README-SM3Det4Wake.md                 # 本文件
```

## 引用

如果您使用了本项目，请引用：

```bibtex
@article{li2024sm3det,
  title={SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection},
  author={Li, Y.
  journal={arXiv preprint arXiv:2412.20665},
  year={2024}
}
```

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议。

## 致谢

- 基于 [OpenMMLab MMRotate](https://github.com/open-mmlab/mmrotate) 框架
- SM3Det 论文作者的开源实现
- SWIM 数据集提供者

---

**联系**: 如有问题或建议，欢迎提交 Issue 或 PR。
