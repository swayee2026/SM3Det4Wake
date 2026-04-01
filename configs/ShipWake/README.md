# ShipWake DualStream 配置文件说明

## 配置文件列表

| 配置文件 | 用途 | GPU需求 | 训练时间* | 推荐场景 |
|---------|------|---------|----------|---------|
| `shipwake_dualstream_mini.py` | 最小化验证 | 1x 8GB | 5分钟 | 数据管线验证、Debug |
| `shipwake_dualstream_standard.py` | 标准训练 | 1-2x 24GB | 4-6小时 | 单GPU/双GPU训练 |
| `shipwake_dualstream_full.py` | 全量训练 | 8x 24GB | 1-2小时 | 多GPU分布式训练 |

*训练时间基于SwimShip数据集(约1000张训练图像)估算

---

## 快速开始

### 1. 数据管线验证 (Mini Config)

用于验证数据加载、模型前向/反向传播是否正常：

```bash
# 单GPU验证
python tools/train.py configs/ShipWake/shipwake_dualstream_mini.py \
    --work-dir work_dirs/mini_test \
    --gpu-ids 0

# 或单批次过拟合测试
python tools/train.py configs/ShipWake/shipwake_dualstream_mini.py \
    --work-dir work_dirs/overfit_test \
    --gpu-ids 0 \
    --cfg-options data.samples_per_gpu=1 runner.max_iters=100
```

**关键参数：**
- `arch='atto'` - 最小backbone (channels=[40,80,160,320])
- `samples_per_gpu=1` - 单样本批次
- `max_iters=100` - 仅100次迭代
- `img_scale=(512, 512)` - 小图像尺寸

---

### 2. 标准训练 (Standard Config)

推荐用于大多数场景，单GPU或双GPU训练：

```bash
# 单GPU训练
python tools/train.py configs/ShipWake/shipwake_dualstream_standard.py \
    --work-dir work_dirs/standard_1gpu \
    --gpu-ids 0

# 双GPU训练
./tools/dist_train.sh configs/ShipWake/shipwake_dualstream_standard.py 2 \
    --work-dir work_dirs/standard_2gpu
```

**关键参数：**
- `arch='tiny'` - ConvNeXt-Tiny backbone
- `samples_per_gpu=4` - 平衡显存和训练速度
- `max_epochs=12` - 标准训练周期
- DSO动态学习率调整启用

---

### 3. 全量训练 (Full Config)

用于最终模型训练，需要多GPU：

```bash
# 8 GPU分布式训练
./tools/dist_train.sh configs/ShipWake/shipwake_dualstream_full.py 8 \
    --work-dir work_dirs/full_8gpu
```

**关键参数：**
- 8 GPU并行
- 完整数据增强
- Multi-scale testing
- Tensorboard日志

---

## 数据集准备

### 目录结构

```
data/SwimShip/
├── train/
│   ├── images/
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ...
│   └── annfiles/
│       ├── 000001.txt
│       ├── 000002.txt
│       └── ...
├── val/
│   ├── images/
│   └── annfiles/
└── test/
    ├── images/
    └── annfiles/
```

### 标注格式

DOTA格式（Oriented Bounding Box）：
```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

类别标签：
- `0`: ship (船只)
- `1`: wake (尾迹)

---

## 可视化调试

### 中间特征可视化

```python
# 在训练代码中添加
model.backbone.forward_with_visualization(
    img, 
    save_dir='work_dirs/visualizations'
)
```

### 注意力蒙版可视化

自动在 `work_dirs/{exp_name}/visualizations/` 中保存：
- `batchX_sampleY_masks.png` - 各stage的船只/尾迹蒙版热力图

---

## 关键超参数说明

### DualStream 模块参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `num_experts` | MoE专家数量 | 4 (2目标+2背景) |
| `top_k` | 激活专家数 | 1 (稀疏激活) |
| `mask_stages` | 生成mask的stage | [0,1,2] |
| `init_lambda` | 残差融合权重初始值 | 0.1 |
| `init_alpha` | 船只引导强度 | 0.2 |
| `init_beta` | 尾迹引导强度 | 0.2 |

### DSO参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `T` | Softmax温度 | 3 |
| `b` | Sigmoid偏置 | 0.4 |
| `ema` | 损失历史EMA衰减 | 0.001 |
| `backbone_policy` | Backbone LR策略 | 'sigmoid_kl' |
| `head_policy` | Head LR策略 | 'normal' |

---

## 训练技巧

### 1. 显存不足时的调整

```python
# 减小batch size
data = dict(samples_per_gpu=2)

# 使用更小的backbone
backbone = dict(arch='atto')  # 或 'femto', 'pico'

# 减少FPN输出
ship_neck = dict(num_outs=4)  # 从5减到4

# 启用梯度检查点
backbone = dict(with_cp=True)
```

### 2. 过拟合处理

```python
# 增加Drop Path率
backbone = dict(drop_path_rate=0.3)

# 增加Weight Decay
optimizer = dict(weight_decay=0.1)

# 更强的数据增强
train_pipeline = [
    dict(type='RandomAffine', scaling_ratio_range=(0.8, 1.2)),
    dict(type='PhotoMetricDistortion', ...),
]
```

### 3. 收敛速度优化

```python
# 更大的初始学习率
optimizer = dict(lr=0.0002)

# 更短的warmup
lr_config = dict(warmup_iters=200)

# 使用预训练权重
backbone = dict(
    init_cfg=dict(
        checkpoint='path/to/pretrained.pth'
    )
)
```

---

## 测试与评估

### 单图像测试

```python
from mmrotate.apis import init_detector, inference_detector

# 初始化模型
model = init_detector(
    'configs/ShipWake/shipwake_dualstream_standard.py',
    'work_dirs/standard_1gpu/latest.pth',
    device='cuda:0'
)

# 推理
result = inference_detector(model, 'test_image.jpg')
# result[0]: ship detections
# result[1]: wake detections
```

### 批量评估

```bash
# 单GPU评估
python tools/test.py configs/ShipWake/shipwake_dualstream_standard.py \
    work_dirs/standard_1gpu/latest.pth \
    --eval mAP

# 多GPU评估
./tools/dist_test.sh configs/ShipWake/shipwake_dualstream_standard.py \
    work_dirs/standard_1gpu/latest.pth 2 \
    --eval mAP
```

---

## 常见问题

### Q: 训练时出现 "RuntimeError: CUDA out of memory"

A: 使用mini配置逐步排查，或调整：
```python
data = dict(samples_per_gpu=1)  # 减小batch size
backbone = dict(arch='atto')     # 使用更小backbone
```

### Q: 船只和尾迹的检测性能差异大

A: 调整DSO参数或独立优化：
```python
# 为尾迹设置更大的学习率
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'wake_bbox_head': dict(lr_mult=1.5),
        }
    )
)
```

### Q: Mask可视化没有输出

A: 确保启用了mask生成：
```python
backbone = dict(
    use_mask_guidance=True,
    mask_stages=[0, 1, 2],  # 至少包含一些stage
)
```

---

## 参考

- SM3Det Paper: [AAAI 2025]
- SwimShip Dataset: [Add dataset link]
- MMRotate Docs: https://mmrotate.readthedocs.io/
