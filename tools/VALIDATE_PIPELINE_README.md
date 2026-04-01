# Pipeline Validation Script 使用说明

## 概述

`validate_pipeline.py` 是一个完整的数据通路验证脚本，用于验证 ShipWake DualStream 模型的所有组件是否正常工作，包括：
- 数据加载和预处理
- 模型构建
- 前向传播（包含中间特征可视化）
- 注意力蒙版生成
- 训练步骤执行

## 使用方法

### 基本用法

```bash
python tools/validate_pipeline.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/validation_output \
    --config configs/ShipWake/shipwake_dualstream_mini.py \
    --num-samples 10
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|-----|------|--------|------|
| `--data-root` | ✓ | - | 数据集目录路径（需包含 `images/` 和 `annfiles/`） |
| `--save-path` | ✓ | - | 输出保存路径（可视化、日志、检查点） |
| `--config` | ✗ | `shipwake_dualstream_mini.py` | 配置文件路径 |
| `--num-samples` | ✗ | 10 | 验证的样本数量 |
| `--device` | ✗ | `cuda:0` | 使用的设备 |
| `--num-iterations` | ✗ | 5 | 训练迭代次数 |

## 输出目录结构

```
work_dirs/validation_output/
├── visualizations/          # 可视化输出
│   ├── sample_000_input.png      # 输入样本可视化
│   ├── sample_001_input.png
│   ├── sample_002_input.png
│   ├── sample_003_input.png
│   ├── sample_004_input.png
│   └── backbone/                 # 中间特征可视化
│       ├── batch0_sample0_masks.png
│       └── ...
├── logs/                    # 日志文件
│   └── validation_20250115_143052.log
└── checkpoints/             # 检查点
    └── validation_checkpoint.pth
```

## 验证步骤

脚本按以下6个步骤执行验证：

### [1/6] 配置加载
- 加载配置文件
- 覆盖数据路径
- 设置验证参数

### [2/6] 数据加载验证
- 构建数据集
- 加载指定数量的样本
- 验证数据格式（图像、标注）
- 可视化前5个样本

### [3/6] 模型构建验证
- 构建模型
- 统计参数量
- 输出模型结构

### [4/6] 前向传播验证
- **Backbone前向**：提取多尺度特征，可视化注意力蒙版
- **训练模式前向**：计算损失
- **推理模式前向**：生成检测结果

### [5/6] 训练步骤验证
- 创建优化器
- 执行训练迭代
- 验证梯度回传
- 保存检查点

### [6/6] 生成总结
- 统计输出文件
- 报告验证结果

## 典型使用场景

### 场景1: 首次验证（推荐）

准备10张图片的mini数据集：

```bash
# 创建mini数据集目录
mkdir -p data/SwimShip/mini/{images,annfiles}

# 复制10张图片和对应标注
cp your_images/{0001..0010}.png data/SwimShip/mini/images/
cp your_annotations/{0001..0010}.txt data/SwimShip/mini/annfiles/

# 运行验证
python tools/validate_pipeline.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/first_validation \
    --num-samples 10
```

### 场景2: CPU环境验证

```bash
python tools/validate_pipeline.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/cpu_validation \
    --device cpu \
    --num-samples 3
```

### 场景3: 验证特定配置

```bash
# 使用standard配置验证
python tools/validate_pipeline.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/standard_validation \
    --config configs/ShipWake/shipwake_dualstream_standard.py \
    --num-samples 5
```

### 场景4: 快速冒烟测试

```bash
python tools/validate_pipeline.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/smoke_test \
    --num-samples 2 \
    --num-iterations 1
```

## 预期输出示例

### 控制台输出

```
[2025-01-15 14:30:52] [INFO] ============================================================
[2025-01-15 14:30:52] [INFO] ShipWake DualStream Pipeline Validation
[2025-01-15 14:30:52] [INFO] ============================================================
[2025-01-15 14:30:52] [INFO] Config: configs/ShipWake/shipwake_dualstream_mini.py
[2025-01-15 14:30:52] [INFO] Data Root: data/SwimShip/mini
[2025-01-15 14:30:52] [INFO] Save Path: work_dirs/validation_output
[2025-01-15 14:30:52] [INFO] Num Samples: 10
[2025-01-15 14:30:52] [INFO] Device: cuda:0
[2025-01-15 14:30:52] [INFO] ============================================================

[2025-01-15 14:30:52] [INFO] 
[1/6] Loading Configuration...
[2025-01-15 14:30:52] [INFO] ✓ Configuration loaded successfully
[2025-01-15 14:30:52] [INFO]   - Train data: data/SwimShip/mini/annfiles/
[2025-01-15 14:30:52] [INFO]   - Batch size: 1
[2025-01-15 14:30:52] [INFO]   - Model type: ShipWakeDualDetector

[2025-01-15 14:30:53] [INFO] 
[2/6] Validating Data Loading...
[2025-01-15 14:30:53] [INFO] ✓ Dataset built: 10 samples
[2025-01-15 14:30:53] [INFO]   Sample 0:
[2025-01-15 14:30:53] [INFO]     - Image shape: (3, 512, 512)
[2025-01-15 14:30:53] [INFO]     - GT boxes: 3
[2025-01-15 14:30:53] [INFO]     - GT labels: [0, 0, 1]
[2025-01-15 14:30:53] [INFO]     - Saved visualization: work_dirs/validation_output/visualizations/sample_000_input.png
...
[2025-01-15 14:30:58] [INFO] ✓ Successfully loaded 10 samples

[2025-01-15 14:30:58] [INFO] 
[3/6] Validating Model Building...
[2025-01-15 14:31:00] [INFO] ✓ Model built: ShipWakeDualDetector
[2025-01-15 14:31:00] [INFO]   - Device: cuda:0
[2025-01-15 14:31:00] [INFO]   - Total parameters: 15,234,567
[2025-01-15 14:31:00] [INFO]   - Trainable parameters: 15,234,567
...

[2025-01-15 14:31:02] [INFO] 
[4/6] Validating Forward Pass...
[2025-01-15 14:31:02] [INFO]   Testing backbone forward...
[2025-01-15 14:31:03] [INFO]   ✓ Backbone forward with visualization passed
[2025-01-15 14:31:03] [INFO]     Output features: 4 levels
[2025-01-15 14:31:03] [INFO]       Level 0: torch.Size([1, 96, 128, 128])
[2025-01-15 14:31:03] [INFO]       Level 1: torch.Size([1, 192, 64, 64])
[2025-01-15 14:31:03] [INFO]       Level 2: torch.Size([1, 384, 32, 32])
[2025-01-15 14:31:03] [INFO]       Level 3: torch.Size([1, 768, 16, 16])
[2025-01-15 14:31:03] [INFO]     Gate loss: 0.023456
...
[2025-01-15 14:31:05] [INFO]   ✓ Training forward passed
[2025-01-15 14:31:05] [INFO]   Losses:
[2025-01-15 14:31:05] [INFO]     ship_loss_cls: 0.654321
[2025-01-15 14:31:05] [INFO]     ship_loss_bbox: 1.234567
[2025-01-15 14:31:05] [INFO]     wake_loss_cls: 0.543210
[2025-01-15 14:31:05] [INFO]     wake_loss_bbox: 0.987654
[2025-01-15 14:31:05] [INFO]     gate_loss: 0.023456

[2025-01-15 14:31:06] [INFO] 
[5/6] Validating Training Step...
[2025-01-15 14:31:06] [INFO] ✓ Optimizer created
[2025-01-15 14:31:06] [INFO]   - Type: AdamW
[2025-01-15 14:31:06] [INFO]   - LR: 0.0001
[2025-01-15 14:31:06] [INFO]   - Weight decay: 0.05
[2025-01-15 14:31:06] [INFO] 
[2025-01-15 14:31:06] [INFO]   Running 5 training iterations...
[2025-01-15 14:31:07] [INFO]     Iter 0: loss=3.356789, max_grad=0.123456
[2025-01-15 14:31:07] [INFO]       Detailed losses:
[2025-01-15 14:31:07] [INFO]         ship_loss_cls: 0.654321
[2025-01-15 14:31:07] [INFO]         ship_loss_bbox: 1.234567
[2025-01-15 14:31:07] [INFO]         wake_loss_cls: 0.543210
[2025-01-15 14:31:07] [INFO]         wake_loss_bbox: 0.987654
[2025-01-15 14:31:08] [INFO]     Iter 1: loss=3.123456, max_grad=0.098765
...
[2025-01-15 14:31:12] [INFO] ✓ Training step validation passed
[2025-01-15 14:31:12] [INFO] ✓ Checkpoint saved: work_dirs/validation_output/checkpoints/validation_checkpoint.pth

[2025-01-15 14:31:12] [INFO] 
[6/6] Generating Summary...
[2025-01-15 14:31:12] [INFO] 
Output Files:
[2025-01-15 14:31:12] [INFO]   Visualizations: 15 files
[2025-01-15 14:31:12] [INFO]     - sample_000_input.png
[2025-01-15 14:31:12] [INFO]     - sample_001_input.png
...
[2025-01-15 14:31:12] [INFO]   Logs: 1 files
[2025-01-15 14:31:12] [INFO]     - validation_20250115_143052.log
[2025-01-15 14:31:12] [INFO]   Checkpoints: 1 files
[2025-01-15 14:31:12] [INFO]     - validation_checkpoint.pth

[2025-01-15 14:31:12] [INFO] 
[2025-01-15 14:31:12] [INFO] ============================================================
[2025-01-15 14:31:12] [INFO] ✓ VALIDATION COMPLETED SUCCESSFULLY
[2025-01-15 14:31:12] [INFO] ============================================================
[2025-01-15 14:31:12] [INFO] All outputs saved to: work_dirs/validation_output
[2025-01-15 14:31:12] [INFO] ============================================================
```

## 故障排除

### 问题1: "CUDA out of memory"

**解决**: 使用更小的配置或减少样本数
```bash
python tools/validate_pipeline.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/validation_output \
    --config configs/ShipWake/shipwake_dualstream_mini.py \
    --num-samples 2 \
    --device cpu  # 或切换到CPU
```

### 问题2: "Dataset is empty"

**解决**: 检查数据路径和格式
```bash
# 检查目录结构
ls data/SwimShip/mini/images/
ls data/SwimShip/mini/annfiles/

# 确保标注文件是DOTA格式
head data/SwimShip/mini/annfiles/0001.txt
```

### 问题3: "Module not found"

**解决**: 确保在正确的conda环境中
```bash
conda activate sm3det
python tools/validate_pipeline.py ...
```

### 问题4: 可视化失败

**解决**: 安装matplotlib
```bash
pip install matplotlib
```

## 验证检查清单

- [x] 数据加载成功（10张图片）
- [x] 模型构建成功（无错误）
- [x] Backbone前向传播成功（输出4层特征）
- [x] 注意力蒙版生成成功（可视化文件存在）
- [x] 训练前向传播成功（计算所有损失）
- [x] 推理前向传播成功（生成检测结果）
- [x] 梯度回传成功（梯度范数正常）
- [x] 优化器步骤成功（参数更新）
- [x] 检查点保存成功（.pth文件存在）

## 下一步

验证成功后，你可以：

1. **查看可视化结果**
   ```bash
   ls work_dirs/validation_output/visualizations/
   ```

2. **检查详细日志**
   ```bash
   cat work_dirs/validation_output/logs/validation_*.log
   ```

3. **开始正式训练**
   ```bash
   python tools/train.py \
       configs/ShipWake/shipwake_dualstream_standard.py \
       --work-dir work_dirs/standard_training
   ```

## 注意事项

1. **mini配置**: 默认使用mini配置（ConvNeXt-Atto），验证通过后可改用standard配置
2. **样本数量**: 建议首次运行使用10张图片，确认无误后可减少
3. **设备选择**: CPU模式较慢但无需GPU，适合环境验证
4. **保存路径**: 每次运行使用不同的保存路径，避免覆盖之前的结果
