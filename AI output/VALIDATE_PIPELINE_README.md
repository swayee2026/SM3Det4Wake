# ShipWake4Det Pipeline Validation Script

## 概述

`validate_pipeline.py` 是专为 ShipWake4Det 项目设计的管线验证脚本，重点验证三个创新模块的正确性：

1. **GeometricMAMG** - 几何感知互注意力蒙版引导模块
2. **WakeResidualTransform** - 交错残差变换模块
3. **ShipWakeDualHead** - 双检测头模块

脚本执行轻量级验证（10张图片），不训练模型，仅验证：
- 模块是否正确构建
- 输入输出维度是否对齐
- 几何蒙版是否正常生成和传递
- 梯度是否正常流动

## 使用方法

### 基本用法

```bash
python tools/validate_pipeline.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/validation_output \
    --config configs/ShipWake/ShipWake_convnext_t.py \
    --num-samples 10
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|-----|------|--------|------|
| `--data-root` | ✓ | - | 数据集目录路径（需包含 `images/` 和 `annfiles/`） |
| `--save-path` | ✓ | - | 输出保存路径（可视化、日志） |
| `--config` | ✗ | `ShipWake_convnext_t.py` | 配置文件路径 |
| `--num-samples` | ✗ | 10 | 验证的样本数量 |
| `--device` | ✗ | `cuda:0` | 使用的设备 |

## 输出目录结构

```
work_dirs/validation_output/
├── visualizations/              # 可视化输出
│   ├── sample_input.png         # 输入样本（如实现）
│   └── geometric_masks/         # 几何蒙版可视化
│       ├── stage_0_geo_mask.png    # Stage 0 的 6通道几何蒙版
│       ├── stage_1_geo_mask.png
│       ├── stage_2_geo_mask.png
│       └── stage_3_geo_mask.png
└── logs/                        # 日志文件
    └── validation_YYYYMMDD_HHMMSS.log
```

### 几何蒙版可视化内容

每个 `stage_X_geo_mask.png` 包含 6 个子图：
- **Ship Confidence**: 船只目标的置信度热力图
- **Wake Confidence**: 尾迹目标的置信度热力图
- **Direction Alignment**: 方向一致性分数（红=不一致，绿=一致）
- **Ship Direction**: 船只方向向量场
- **Wake Direction**: 尾迹方向向量场
- **Combined Overlay**: 船只（红）和尾迹（蓝）叠加效果

## 验证步骤

脚本按以下5个步骤执行验证：

### [1/5] 配置加载
- 加载配置文件
- 覆盖数据路径为 mini 数据集
- 设置 batch_size=1 便于调试

### [2/5] 数据加载验证
- 构建数据集
- 加载指定数量的样本（默认10张）
- 验证每张图片的 shape 和标注格式
- 统计船只/尾迹数量

### [3/5] 模型构建与模块检查
- 构建 ShipWakeDualDetector
- **检查 GeometricMAMG**:
  - 是否启用 `use_geometric_mamg`
  - 每个 stage 是否有 mask_generator
  - 每个 stage 是否有 propagator
  - 每个 stage 是否有 fusion
- **检查 WakeResidualTransform**:
  - 是否启用 `use_wake_residual`
  - 每个 stage 的变换类型（StripConvResidual/LowFreqResidual）
- **检查 MoE Backbone**:
  - 专家数量
  - top-k 设置
- **检查 Dual Heads**:
  - ship_roi_head 和 wake_roi_head 是否存在

### [4/5] 前向传播验证（核心）

#### Test 1: Backbone with Intermediates
```python
output, intermediates = backbone.forward_with_intermediates(img)
```

验证内容：
- ✓ `forward_with_intermediates` 可正常调用
- ✓ 返回 `intermediates` 列表（每个 stage 一个 dict）
- ✓ **geo_mask 维度**: `(B, 6, H, W)`
  - Channel 0: ship_confidence [0, 1]
  - Channel 1-2: ship_direction (cosθ, sinθ)
  - Channel 3: wake_confidence [0, 1]
  - Channel 4-5: wake_direction (cosθ, sinθ)
- ✓ **方向归一化**: 方向向量模长 ≈ 1
- ✓ **dir_alignment**: 方向一致性分数 [0, 1]
- ✓ **guidance weights**: 交叉引导权重正确计算

#### Test 2: Training Forward
```python
losses = model.forward_train(img, img_metas, gt_bboxes, gt_labels)
```

验证内容：
- ✓ 损失计算正常（ship_loss_cls, ship_loss_bbox, wake_loss_cls, wake_loss_bbox, gate_loss）
- ✓ 所有损失为有效数值（非 NaN/Inf）

#### Test 3: Gradient Flow
```python
total_loss.backward()
```

验证关键模块的梯度：
- ✓ `backbone.mamg_modules` 有梯度
- ✓ `backbone.residual_stages` 有梯度
- ✓ `ship_roi_head` 有梯度
- ✓ `wake_roi_head` 有梯度

#### Test 4: Inference Forward
```python
results = model.simple_test(img, img_metas)
```

验证内容：
- ✓ 推理模式正常运行
- ✓ 返回检测结果（bboxes, labels）

### [5/5] 几何蒙版传播验证

验证跨 stage 的蒙版传递：
- ✓ **空间缩放**: 每个 stage 的空间分辨率减半（800→400→200→100）
- ✓ **蒙版演化**: 相邻 stage 的蒙版有显著差异（非简单复制）
- ✓ **方向一致性**: 方向向量保持归一化

## 典型使用场景

### 场景1: 首次验证（推荐）

准备10张图片的 mini 数据集：

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

### 场景3: 快速冒烟测试

```bash
python tools/validate_pipeline.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/smoke_test \
    --num-samples 2
```

### 场景4: 调试特定模块

如果只想验证 GeometricMAMG：

```python
# 在脚本中临时禁用其他模块
model.backbone.use_wake_residual = False
```

## 预期输出示例

### 控制台输出

```
[2025-01-15 14:30:52] [INFO] ============================================================
[2025-01-15 14:30:52] [INFO] ShipWake4Det Pipeline Validation
[2025-01-15 14:30:52] [INFO] ============================================================
...
[2025-01-15 14:30:58] [INFO] 
[3/5] Validating Model Building...
[2025-01-15 14:31:00] [INFO] ✓ Model built: ShipWakeDualDetector
[2025-01-15 14:31:00] [INFO] 
  Checking Innovative Modules:
[2025-01-15 14:31:00] [INFO]   ✓ GeometricMAMG: ENABLED
[2025-01-15 14:31:00] [INFO]     - MAMG modules: 4 (one per stage)
[2025-01-15 14:31:00] [INFO]     - Stage 0 mask_generator: ✓
[2025-01-15 14:31:00] [INFO]     - Stage 0 propagator: ✓
[2025-01-15 14:31:00] [INFO]     - Stage 0 fusion: ✓
[2025-01-15 14:31:00] [INFO]   ✓ WakeResidualTransform: ENABLED
[2025-01-15 14:31:00] [INFO]     - Residual stages: 4
[2025-01-15 14:31:00] [INFO]     - Stage 0 transform: LowFreqResidual
[2025-01-15 14:31:00] [INFO]     - Stage 1 transform: StripConvResidual
...
[2025-01-15 14:31:02] [INFO] 
[4/5] Validating Forward Pass...
[2025-01-15 14:31:02] [INFO] 
  Test 1: Backbone forward with intermediate outputs...
[2025-01-15 14:31:03] [INFO]   ✓ Backbone forward_with_intermediates passed
[2025-01-15 14:31:03] [INFO]     - Intermediates: 4 stages
[2025-01-15 14:31:03] [INFO] 
    Stage 0 intermediate outputs:
[2025-01-15 14:31:03] [INFO]       - geo_mask shape: torch.Size([1, 6, 200, 200])
[2025-01-15 14:31:03] [INFO]         ship_conf range: [0.023, 0.987]
[2025-01-15 14:31:03] [INFO]         wake_conf range: [0.015, 0.954]
[2025-01-15 14:31:03] [INFO]         ship_dir norm (should be ~1): 1.000
[2025-01-15 14:31:03] [INFO]       - dir_alignment shape: torch.Size([1, 1, 200, 200])
[2025-01-15 14:31:03] [INFO]         range: [0.234, 0.987]
[2025-01-15 14:31:03] [INFO]     Saved: work_dirs/validation_output/visualizations/geometric_masks/stage_0_geo_mask.png
...
[2025-01-15 14:31:05] [INFO] 
  Test 3: Gradient flow validation...
[2025-01-15 14:31:06] [INFO]   ✓ backbone.mamg_modules: avg_grad=0.045623, max_grad=0.234567
[2025-01-15 14:31:06] [INFO]   ✓ backbone.residual_stages: avg_grad=0.034512, max_grad=0.198765
[2025-01-15 14:31:06] [INFO]   ✓ ship_roi_head: avg_grad=0.123456, max_grad=0.456789
[2025-01-15 14:31:06] [INFO]   ✓ wake_roi_head: avg_grad=0.098765, max_grad=0.345678
[2025-01-15 14:31:06] [INFO]   ✓ Gradient flow validation passed
...
[2025-01-15 14:31:12] [INFO] 
[5/5] Generating Summary...
[2025-01-15 14:31:12] [INFO] 
Validation Checklist:
[2025-01-15 14:31:12] [INFO]   [✓] Data loading
[2025-01-15 14:31:12] [INFO]   [✓] Model building
[2025-01-15 14:31:12] [INFO]   [✓] GeometricMAMG module
[2025-01-15 14:31:12] [INFO]   [✓] WakeResidual module
[2025-01-15 14:31:12] [INFO]   [✓] MoE backbone
[2025-01-15 14:31:12] [INFO]   [✓] Dual detection heads
[2025-01-15 14:31:12] [INFO]   [✓] Forward pass (train)
[2025-01-15 14:31:12] [INFO]   [✓] Forward pass (eval)
[2025-01-15 14:31:12] [INFO]   [✓] Gradient flow
[2025-01-15 14:31:12] [INFO]   [✓] Mask propagation
[2025-01-15 14:31:12] [INFO] 
[2025-01-15 14:31:12] [INFO] ============================================================
[2025-01-15 14:31:12] [INFO] ✓ VALIDATION COMPLETED SUCCESSFULLY
[2025-01-15 14:31:12] [INFO] ============================================================
```

## 故障排除

### 问题1: "geo_mask shape mismatch"

**现象**: 期望 `(B, 6, H, W)` 但得到其他形状

**解决**: 检查 `GeometricMaskGenerator` 的输出通道数
```python
# 在 geometric_mamg.py 中确认
self.conv = nn.Sequential(
    ..., 
    nn.Conv2d(hidden_dim, 6, 1)  # 必须是6通道
)
```

### 问题2: "Direction vectors not normalized"

**现象**: 方向向量模长偏离 1.0

**解决**: 检查 `normalize_direction` 函数是否在 forward 中被调用

### 问题3: "No gradients for mamg_modules"

**现象**: 梯度流验证时 mamg_modules 没有梯度

**可能原因**:
- 模块未加入模型参数（检查 `__init__` 中是否用 `self.` 注册）
- 前向传播时未使用这些模块

### 问题4: "CUDA out of memory"

**解决**: 
```bash
python tools/validate_pipeline.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/validation_output \
    --device cpu \
    --num-samples 2
```

### 问题5: 可视化失败

**现象**: `visualize_single_stage` 报错

**解决**:
```bash
pip install matplotlib
```

或禁用可视化：
```python
# 在脚本中设置
VISUALIZATION_AVAILABLE = False
```

## 与原版脚本的区别

| 特性 | 原版脚本 | 本脚本 |
|-----|---------|--------|
| **目标模型** | DualStream | ShipWake4Det |
| **核心验证** | 通用数据流 | 三个创新模块 |
| **可视化** | 通用特征图 | 几何蒙版专用可视化 |
| **梯度检查** | 整体梯度 | 分模块梯度统计 |
| **训练步骤** | 保存检查点 | 仅验证梯度流 |
| **蒙版验证** | 无 | 跨层传播验证 |

## 验证通过标准

以下检查项必须全部通过：

- [x] 10张图片加载成功
- [x] GeometricMAMG 模块存在且可前向传播
- [x] WakeResidual 模块存在且可前向传播
- [x] MoE 模块正常工作
- [x] Dual Heads 正常工作
- [x] geo_mask 维度为 (B, 6, H, W)
- [x] 方向向量已归一化
- [x] 训练损失为有限数值
- [x] 所有关键模块有梯度
- [x] 蒙版跨层传播正常

## 下一步

验证成功后：

1. **查看几何蒙版可视化**
   ```bash
   ls work_dirs/validation_output/visualizations/geometric_masks/
   ```

2. **检查详细日志**
   ```bash
   cat work_dirs/validation_output/logs/validation_*.log
   ```

3. **开始正式训练**
   ```bash
   python tools/train.py \
       configs/ShipWake/ShipWake_convnext_t.py \
       --work-dir work_dirs/standard_training
   ```

## 注意事项

1. **纯验证性质**: 本脚本不保存检查点，不优化模型
2. **轻量级**: 仅需10张图片，快速验证
3. **聚焦创新点**: 重点验证 GeometricMAMG、WakeResidual、DualHeads
4. **可视化依赖**: 需要 matplotlib 生成几何蒙版图
