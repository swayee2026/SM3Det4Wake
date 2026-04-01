# ShipWake DualStream 模型配置完成指南

## 📁 已创建文件清单

### 1. 核心代码模块

```
mmrotate/models/backbones/
├── wake_residual_transform.py      # 交错残差变换模块
├── mutual_attention_mask.py        # 互引导注意力蒙版模块
├── convnext_dualstream.py          # DualStream Backbone
└── __init__.py                     # 模块注册

mmrotate/models/detectors/
├── shipwake_dual_detector.py       # 双检测器
└── __init__.py                     # 检测器注册
```

### 2. 配置文件

```
configs/
├── _base_/
│   └── datasets/
│       └── SwimShip.py             # SwimShip数据集基础配置
└── ShipWake/
    ├── shipwake_dualstream_mini.py      # 最小化验证配置
    ├── shipwake_dualstream_standard.py  # 标准训练配置（1-2 GPU）
    ├── shipwake_dualstream_full.py      # 全量训练配置（8 GPU）
    └── README.md                        # 配置使用说明
```

### 3. 测试工具

```
tools/
└── quick_test.py                   # 快速测试脚本
```

---

## 🚀 快速开始

### 步骤1: 准备数据

创建数据集目录结构：

```bash
mkdir -p data/SwimShip/{train,val,test}/{images,annfiles}

# 放置图像和标注文件
cp your_images/*.png data/SwimShip/train/images/
cp your_annotations/*.txt data/SwimShip/train/annfiles/
```

标注格式（DOTA format）：
```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

类别定义：
- `ship`: 船只 (label 0)
- `wake`: 尾迹 (label 1)

---

### 步骤2: 验证安装

```bash
# 测试数据管线和模型
python tools/quick_test.py \
    --config configs/ShipWake/shipwake_dualstream_mini.py \
    --data-root data/SwimShip/mini \
    --test-model \
    --vis
```

预期输出：
```
============================================================
ShipWake DualStream Quick Test
============================================================
✓ Dataset loaded: 10 samples
✓ Model built: ShipWakeDualDetector
✓ Backbone forward passed
✓ Training forward passed
✓ Inference passed
============================================================
✓ All tests passed!
============================================================
```

---

### 步骤3: 训练模型

#### 选项A: 最小化验证（推荐首次运行）

```bash
# 单GPU，100次迭代，验证数据管线
python tools/train.py \
    configs/ShipWake/shipwake_dualstream_mini.py \
    --work-dir work_dirs/mini_test \
    --gpu-ids 0
```

#### 选项B: 标准训练（单GPU/双GPU）

```bash
# 单GPU
python tools/train.py \
    configs/ShipWake/shipwake_dualstream_standard.py \
    --work-dir work_dirs/standard_1gpu \
    --gpu-ids 0

# 双GPU
./tools/dist_train.sh \
    configs/ShipWake/shipwake_dualstream_standard.py 2 \
    --work-dir work_dirs/standard_2gpu
```

#### 选项C: 全量训练（8 GPU集群）

```bash
./tools/dist_train.sh \
    configs/ShipWake/shipwake_dualstream_full.py 8 \
    --work-dir work_dirs/full_8gpu
```

---

### 步骤4: 测试模型

```bash
# 单GPU测试
python tools/test.py \
    configs/ShipWake/shipwake_dualstream_standard.py \
    work_dirs/standard_1gpu/latest.pth \
    --eval mAP

# 多GPU测试
./tools/dist_test.sh \
    configs/ShipWake/shipwake_dualstream_standard.py \
    work_dirs/standard_1gpu/latest.pth 2 \
    --eval mAP
```

---

## 📊 配置对比

| 特性 | Mini | Standard | Full |
|-----|------|----------|------|
| **Backbone** | ConvNeXt-Atto | ConvNeXt-Tiny | ConvNeXt-Tiny |
| **Channels** | [40,80,160,320] | [96,192,384,768] | [96,192,384,768] |
| **Batch Size** | 1 | 4 (x1-2 GPU) | 2 (x8 GPU) |
| **Epochs** | 100 iters | 12 epochs | 12 epochs |
| **Image Size** | 512x512 | 800x800 | 800x800 |
| **Augmentation** | 无 | 标准 | 完整 |
| **DSO** | 关闭 | 开启 | 开启 |
| **预训练** | 无 | ImageNet | ImageNet |
| **显存需求** | ~4GB | ~12GB | ~20GB/GPU |
| **训练时间** | ~5分钟 | ~4-6小时 | ~1-2小时 |

---

## 🔧 核心模块说明

### 1. Wake Residual Transform

```python
# Stage-wise alternating transforms
Stage 0: LowFreqFilter (5x5)  -> 船只密集纹理
Stage 1: StripConv (1x7,7x1)  -> 尾迹线性结构
Stage 2: LowFreqFilter (7x7)  -> 深层船只语义
Stage 3: StripConv (1x11,11x1) -> 全局尾迹结构
```

### 2. Mutual Attention Mask Guidance (MAMG)

```python
# 对称蒙版生成
Mask_ship = Conv3x3(Feature) -> Sigmoid
Mask_wake = Conv5x5(Feature) -> Sigmoid

# 双向引导
Ship_pathway = Feature * (1 + α * Mask_wake)
Wake_pathway = Feature * (1 + β * Mask_ship)
```

### 3. Grid-Level MoE

```python
# 4专家配置
Expert 0: 船只特征专家
Expert 1: 尾迹特征专家
Expert 2: 背景专家
Expert 3: 通用特征专家

# Top-1稀疏路由
Gating = CosineSimilarity(Feature, Expert_embeddings)
Selected = Top1(Gating)
```

### 4. Dynamic Submodule Optimization (DSO)

```python
# 头学习率调整（基于收敛速度）
LR_head_i ∝ 1 / convergence_speed_i

# Backbone学习率调整（基于一致性）
Consistency = 1 - KL(Current_losses || Historical_losses)
LR_backbone ∝ Sigmoid((Consistency - b) * T)
```

---

## 🐛 故障排除

### 问题1: CUDA Out of Memory

```bash
# 解决: 使用更小的配置
python tools/train.py configs/ShipWake/shipwake_dualstream_mini.py

# 或手动调整
python tools/train.py configs/ShipWake/shipwake_dualstream_standard.py \
    --cfg-options data.samples_per_gpu=1
```

### 问题2: 数据加载失败

```bash
# 检查数据格式
python tools/quick_test.py \
    --config configs/ShipWake/shipwake_dualstream_mini.py \
    --data-root data/SwimShip/train \
    --test-data
```

### 问题3: 模型导入错误

```python
# 检查模块是否正确注册
from mmrotate.models import build_detector

# 应该能够成功导入
cfg = dict(type='ShipWakeDualDetector', ...)
model = build_detector(cfg)
```

### 问题4: DSO Hook未生效

```python
# 确保配置中包含
lr_config = dict(
    policy='dynamic',
    extra_args={'T': 3, 'b': 0.4, ...},
    reweight_losses={...}
)
```

---

## 📈 监控训练

### TensorBoard

```bash
# 在训练目录中启动
tensorboard --logdir work_dirs/standard_1gpu

# 访问 http://localhost:6006
```

### 关键指标

| 指标 | 说明 | 目标值 |
|-----|------|--------|
| `ship_loss_cls` | 船只分类损失 | < 0.5 |
| `ship_loss_bbox` | 船只回归损失 | < 1.0 |
| `wake_loss_cls` | 尾迹分类损失 | < 0.5 |
| `wake_loss_bbox` | 尾迹回归损失 | < 1.0 |
| `gate_loss` | MoE负载均衡损失 | < 0.1 |
| `mAP` | 平均精度 | > 0.6 |

---

## 🔬 进阶使用

### 可视化中间特征

```python
# 在训练代码中添加
model.backbone.forward_with_visualization(
    img,
    save_dir='work_dirs/visualizations'
)
```

### 自定义锚框

```python
# 针对尾迹的长宽比优化
wake_bbox_head=dict(
    anchor_generator=dict(
        ratios=[1.0, 2.0, 4.0, 8.0],  # 更长的尾迹
    )
)
```

### 调整DSO参数

```python
lr_config = dict(
    extra_args=dict(
        T=5,      # 更平缓的学习率调整
        b=0.3,    # 更低的一致性阈值
        ema=0.01, # 更快的历史衰减
    )
)
```

---

## 📚 相关文档

- [SM3Det原始论文](https://arxiv.org/abs/2412.20665)
- [MMRotate文档](https://mmrotate.readthedocs.io/)
- [配置文件详细说明](configs/ShipWake/README.md)

---

## 💡 开发建议

### 阶段性开发流程

```
Phase 1: 数据验证
  └── 使用mini配置验证10张图片

Phase 2: 单GPU训练
  └── 使用standard配置训练 baseline

Phase 3: 消融实验
  └── 分别测试各模块贡献
  └── 移除MoE / 移除Residual / 移除MAMG

Phase 4: 全量训练
  └── 使用full配置多GPU训练

Phase 5: 测试评估
  └── 生成可视化结果
  └── 计算mAP指标
```

### 论文实验建议

1. **Baseline**: 单流单头检测器
2. **Ours**: 完整DualStream模型
3. **Ablation**:
   - w/o MoE
   - w/o Residual
   - w/o MAMG
   - w/o DSO

---

## 📞 获取帮助

如有问题，请检查：
1. 数据格式是否符合DOTA标准
2. GPU显存是否足够（至少8GB）
3. 依赖包是否正确安装（mmcv-full, mmrotate）
4. 配置文件路径是否正确

---

**祝训练顺利！** 🚢🌊
