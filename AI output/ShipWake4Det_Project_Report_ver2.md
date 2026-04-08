# ShipWake4Det: 光学遥感图像船只-尾迹协同检测

## 项目概述

**基于SM3Det改进的单模态双目标协同检测框架**

---

## 一、实验目标与研究背景

### 1.1 研究动机

在光学遥感图像中，**船只目标检测**是海洋监测、航道管理、海上搜救等应用的核心任务。然而，船只与其尾迹（wake）存在紧密的几何关联，尾迹检测对于：
- **航向估计**：尾迹方向反映船只航行方向
- **隐身目标发现**：尾迹可能比船体本身更易被检测
- **运动状态分析**：尾迹长度、形态反映船只速度

### 1.2 核心观察

| 特征维度 | 船只 (Ship) | 尾迹 (Wake) |
|---------|------------|-------------|
| **空间尺寸** | 小（通常<50×50像素） | 大（可达数百像素） |
| **纹理特征** | 稠密、边缘清晰、高频丰富 | 稀疏、线性延伸、低频结构 |
| **几何形态** | 紧凑的矩形/多边形 | 细长的线状/带状 |
| **下采样特性** | 浅层即可捕获 | 需要深层全局感知 |
| **方向信息** | 船体朝向 | 航行方向（与船体一致） |

**关键洞察**：船只与尾迹存在**几何位置强关联**——尾迹起点位于船尾，沿航行方向延伸。这一先验知识可用于双向特征引导。

### 1.3 与SM3Det的核心差异

| 对比维度 | SM3Det | ShipWake4Det (本工作) |
|---------|--------|---------------------|
| **任务场景** | 多模态（SAR/光学/红外）+ 多任务（HBB/OBB） | 单模态（光学RGB）+ 单任务（OBB） |
| **目标设置** | 多类目标（SAR 6类、DOTA 15类等） | 2类目标（船只、尾迹） |
| **MoE设计** | 8专家，跨**模态**路由 | 4专家，跨**目标类型**路由 |
| **空间引导** | 无显式几何建模 | **几何感知MAMG**（方向+位置） |
| **特征提取** | 标准残差连接 | **交错残差**（条带卷积/低频滤波） |
| **检测头** | 每模态一个头 | 每目标类型一个头 + DSO |

**核心区别**：从"多模态联合学习"降级为"单模态多目标协同学习”，保留并改造了MoE的动态路由思想，新增了几何感知的空间引导机制。

---

## 二、模型概览与数据流

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ShipWake4Det 整体架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Image (B, 3, 800, 800)                                               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ConvNeXt_moe_wake Backbone                        │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │   │
│  │  │ Stage 1 │─▶│ Stage 2 │─▶│ Stage 3 │─▶│ Stage 4 │                │   │
│  │  │         │  │         │  │(MoE)    │  │(MoE)    │                │   │
│  │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │                │   │
│  │  │ │MAMG │ │  │ │MAMG │ │  │ │MAMG │ │  │ │MAMG │ │                │   │
│  │  │ │+Res │ │  │ │+Res │ │  │ │+Res │ │  │ │+Res │ │                │   │
│  │  │ └──┬──┘ │  │ └──┬──┘ │  │ └──┬──┘ │  │ └──┬──┘ │                │   │
│  │  └────┼────┘  └────┼────┘  └────┼────┘  └────┼────┘                │   │
│  │       │            │            │            │                       │   │
│  │       ▼            ▼            ▼            ▼                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  Intermediates: geo_masks, dir_alignment (for visualization) │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  FPN Neck (P2, P3, P4, P5, P6)                                             │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Dual Detection Heads                         │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                         │   │
│  │  │   Ship Head     │    │   Wake Head     │                         │   │
│  │  │  (OBB + Dir)    │    │  (OBB + Dir)    │                         │   │
│  │  └────────┬────────┘    └────────┬────────┘                         │   │
│  │           │                      │                                   │   │
│  │           ▼                      ▼                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │            DSO (Dynamic Submodule Optimization)              │    │   │
│  │  │   Adaptive LR: ship_loss ──▶ ship_head LR                   │    │   │
│  │  │                wake_loss ──▶ wake_head LR                   │    │   │
│  │  │                consistency ──▶ backbone LR                  │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  Output: Ship BBoxes (x, y, w, h, θ) + Wake BBoxes (x, y, w, h, θ)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 详细数据流Pipeline

#### Stage 0: 输入处理
```python
# 输入: x (B, 3, 800, 800)
x = self.downsample_layers[0](x)  # (B, 96, 200, 200)
```

#### Stage 1-4: 核心处理流程（以Stage i为例）

```
输入特征: x (B, C_i, H_i, W_i)
    │
    ├──▶ [1] Downsample (if i > 0)
    │       x = downsample_layers[i](x)
    │
    ├──▶ [2] Apply Previous Geometric Mask (if available)
    │       x = x * (1 + 0.1 * (prev_geo_mask_ship + prev_geo_mask_wake))
    │
    ├──▶ [3] ConvNeXt Blocks (Main Branch)
    │       for block in stage:
    │           x, gate_loss = block(x)  # MoE in specific layers
    │
    ├──▶ [4] Geometric MAMG
    │       x, geo_mask, debug_info = mamg_modules[i](x, prev_geo_mask)
    │       # geo_mask: (B, 6, H, W) [ship_conf, ship_dx, ship_dy, wake_conf, wake_dx, wake_dy]
    │       # debug_info: direction alignment, guidance weights
    │
    ├──▶ [5] Wake Residual Fusion
    │       residual_feat = residual_stages[i].transform(residual_input)
    │       x = residual_stages[i].fusion(x, residual_feat)
    │
    └──▶ [6] Output for Next Stage
            prev_geo_mask = geo_mask
            if i in out_indices:
                outs.append(norm(x))
```

#### Neck & Detection Heads

```
FPN Features: [P2, P3, P4, P5, P6]
    │
    ├──▶ RPN Head ──▶ Proposals (N, 5)
    │
    ├──▶ Ship ROI Head ──▶ Ship Detections (x, y, w, h, θ) + Direction (cosθ, sinθ)
    │
    └──▶ Wake ROI Head ──▶ Wake Detections (x, y, w, h, θ) + Direction (cosθ, sinθ)
```

### 2.3 关键函数调用链

```
ShipWakeDualDetector.forward_train()
    ├── extract_feat()
    │   └── ConvNeXt_moe_wake.forward_with_intermediates()
    │       ├── downsample_layers[i]()
    │       ├── ConvNeXtBlock.forward() [with MoE]
    │       ├── GeometricMAMG.forward()
    │       │   ├── GeometricMaskGenerator.forward()
    │       │   ├── GeometricPropagator.forward() (if prev_mask exists)
    │       │   ├── CrossGuidedFusion.forward()
    │       │   │   └── compute_direction_alignment()
    │       │   └── visualize_masks() (optional)
    │       └── WakeResidualStage.forward()
    │           ├── StripConvResidual/LowFreqResidual.forward()
    │           └── ResidualFusion.forward()
    │
    ├── rpn_head.forward_train()
    ├── ship_roi_head.forward_train()
    │   └── ShipWakeHead.forward()
    │       ├── ORConv2d.forward()
    │       └── RotationInvariantPooling.forward()
    ├── wake_roi_head.forward_train()
    │
    └── DynamicLrUpdaterHook.get_dynamic_lr()
        ├── compute task-specific LR weights
        └── compute backbone LR weight (sigmoid_kl)
```

---

## 三、创新模块详细介绍

### 模块1: GeometricMAMG (几何感知互注意力蒙版引导)

**目标**: 利用船只与尾迹的方向一致性和空间关联，实现双向特征引导。

**输入输出**:
```
输入:
    - feat: (B, C, H, W) 当前层特征
    - prev_geo_mask: (B, 6, H', W') 上一层几何蒙版 (可选)

输出:
    - guided_feat: (B, C, H, W) 引导后的特征
    - geo_mask: (B, 6, H, W) 当前层几何蒙版
    - debug_info: 可视化信息
```

**核心组件**:

#### 1.1 GeometricMaskGenerator
生成6通道几何蒙版：
```python
# 通道分配:
# 0: ship_confidence [0, 1]
# 1: ship_direction_x (cos θ)
# 2: ship_direction_y (sin θ)
# 3: wake_confidence [0, 1]
# 4: wake_direction_x (cos θ)
# 5: wake_direction_y (sin θ)

geo_mask = mask_conv(feat)  # (B, 6, H, W)
geo_mask[:, [0, 3]] = sigmoid(geo_mask[:, [0, 3]])  # 置信度
geo_mask[:, [1,2,4,5]] = normalize_direction(geo_mask[:, [1,2,4,5]])  # 方向
```

#### 1.2 GeometricPropagator
跨层传播几何蒙版，保持时空一致性：
```python
def forward(self, curr_mask, prev_mask):
    # 上采样到当前分辨率
    prev_up = F.interpolate(prev_mask, size=(H, W), mode='bilinear')
    
    # 重归一化方向向量（插值后可能不再是单位向量）
    prev_up = normalize_direction(prev_up)
    
    # 置信度加权融合
    curr_conf, prev_conf = curr_mask[:, 0], prev_up[:, 0]
    w_curr = curr_conf / (curr_conf + prev_conf + eps)
    w_prev = prev_conf / (curr_conf + prev_conf + eps)
    
    fused_conf = torch.max(curr_conf, prev_conf * 0.8)  # 时间衰减
    fused_dir = w_curr * curr_dir + w_prev * prev_dir
    
    return fused_mask
```

#### 1.3 CrossGuidedFusion
基于方向一致性的交叉引导：
```python
def forward(self, feat, ship_mask, wake_mask):
    # 计算方向一致性: cos(Δθ) = (v_ship · v_wake)
    dir_alignment = compute_direction_alignment(
        ship_mask['direction'], wake_mask['direction']
    )  # (B, 1, H, W), range [0, 1]
    
    # 尾迹特征受船只引导: F_wake = F * (1 + α·ship_conf·alignment)
    wake_guidance = ship_mask['conf'] * (1 + beta * dir_alignment)
    wake_feat = feat * (1 + alpha * wake_guidance)
    
    # 船只特征受尾迹引导: F_ship = F * (1 + α·wake_conf·alignment)
    ship_guidance = wake_mask['conf'] * (1 + beta * dir_alignment)
    ship_feat = feat * (1 + alpha * ship_guidance)
    
    # 融合两个分支
    fused = torch.cat([ship_feat, wake_feat], dim=1)
    guided_feat = fusion_conv(fused)
    
    return guided_feat
```

**可视化输出**:
- 船只/尾迹置信度热力图
- 方向向量场（quiver plot）
- 方向一致性分数
- 叠加效果

---

### 模块2: WakeResidualTransform (交错残差变换)

**目标**: 针对船只（稠密纹理）和尾迹（稀疏线性）的不同特征分布，设计交替的特征提取策略。

**核心洞察**: 
- 下采样过程中，**船只信息**在浅层即可充分捕获（高频纹理）
- **尾迹信息**需要深层全局感知（低频结构）

**交错配置**:

| Stage | 下采样倍率 | 分辨率(800×800输入) | 变换类型 | 设计目标 |
|-------|-----------|-------------------|---------|---------|
| 1 | 4× | 200×200 | LowFreqResidual (5×5) | 保留船只高频纹理，平滑背景噪声 |
| 2 | 8× | 100×100 | StripConvResidual (1×7, 7×1) | 初步提取尾迹线性结构 |
| 3 | 16× | 50×50 | LowFreqResidual (7×7) | 强化深层船只语义纹理 |
| 4 | 32× | 25×25 | StripConvResidual (1×11, 11×1) | 强化大尺寸尾迹全局结构 |

**核心组件**:

#### 2.1 StripConvResidual (条带卷积残差)
```python
class StripConvResidual(nn.Module):
    """不对称卷积捕获线性结构"""
    def __init__(self, channels, kernel_size=7):
        # 水平条带: 1×k
        self.h_conv = nn.Conv2d(channels, channels//2, (1, k), groups=channels//2)
        # 垂直条带: k×1
        self.v_conv = nn.Conv2d(channels, channels//2, (k, 1), groups=channels//2)
        
    def forward(self, x):
        h_feat = self.h_conv(x)  # 捕获水平线
        v_feat = self.v_conv(x)  # 捕获垂直线
        concat = torch.cat([h_feat, v_feat], dim=1)
        return fusion_conv(concat)
```

#### 2.2 LowFreqResidual (低频滤波残差)
```python
class LowFreqResidual(nn.Module):
    """高斯模糊保留低频结构"""
    def __init__(self, channels, kernel_size=5):
        # 创建可学习的高斯核
        self.gaussian_kernel = self._create_gaussian(kernel_size, sigma)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        smoothed = F.conv2d(x, self.gaussian_kernel, padding=pad, groups=C)
        return smoothed + self.residual_weight * x
```

#### 2.3 ResidualFusion (可学习融合)
```python
class ResidualFusion(nn.Module):
    """主分支与残差分支的通道注意力融合"""
    def __init__(self, channels):
        self.lambda_residual = nn.Parameter(torch.tensor(0.1))
        self.channel_att = nn.Sequential(
            nn.Linear(channels*2, channels//4),
            nn.ReLU(),
            nn.Linear(channels//4, channels*2),
            nn.Sigmoid()
        )
        
    def forward(self, main_feat, residual_feat):
        # 加权残差
        weighted_res = self.lambda_residual * residual_feat
        
        # 通道注意力
        concat = torch.cat([main_feat, weighted_res], dim=1)
        att = self.channel_att(global_avg_pool(concat))
        
        # 融合
        fused = fusion_conv(concat * att)
        return main_feat + fused  # 残差连接
```

---

### 模块3: ShipWakeDualHead (双检测头与DSO)

**目标**: 分别检测船只和尾迹，通过DSO动态平衡两类目标的学习难度。

**架构设计**:

```
Shared FPN Features
        │
        ├──▶ RPN ──▶ Proposals
        │
        ├──▶ Ship ROI Head
        │       ├── RoI Align
        │       ├── ORConv (Orientation-Sensitive)
        │       ├── Rotation-Invariant Pooling
        │       └── ShipWakeHead
        │           ├── Classification: Ship/Background
        │           ├── Regression: (x, y, w, h, θ)
        │           └── Direction: (cos θ, sin θ) for geometric mask
        │
        └──▶ Wake ROI Head
                ├── RoI Align
                ├── ORConv
                ├── Rotation-Invariant Pooling
                └── ShipWakeHead
                    ├── Classification: Wake/Background
                    ├── Regression: (x, y, w, h, θ)
                    └── Direction: (cos θ, sin θ)
```

**核心组件**:

#### 3.1 ShipWakeHead
```python
class ShipWakeHead(ODMRefineHead):
    """单类OBB检测头，输出方向信息"""
    def __init__(self, num_classes=1, use_direction=True):
        self.use_direction = use_direction
        
        # 标准OBB头组件
        self.or_conv = ORConv2d(in_c, out_c, kernel=3, arf_config=(1, 8))
        self.or_pool = RotationInvariantPooling(256, 8)
        self.odm_cls = nn.Conv2d(feat_c, num_anchors * num_classes, 3)
        self.odm_reg = nn.Conv2d(feat_c, num_anchors * 5, 3)  # [x,y,w,h,θ]
        
        # 新增：方向输出用于几何蒙版
        if use_direction:
            self.odm_dir = nn.Conv2d(feat_c, num_anchors * 2, 3)  # [cosθ, sinθ]
    
    def forward_single(self, x):
        or_feat = self.or_conv(x)
        cls_feat = self.or_pool(or_feat)
        reg_feat = or_feat
        
        cls_score = self.odm_cls(cls_feat)
        bbox_pred = self.odm_reg(reg_feat)
        dir_pred = self.odm_dir(reg_feat)
        
        # 归一化方向向量
        dir_pred = normalize_direction(dir_pred)
        
        return cls_score, bbox_pred, dir_pred
```

#### 3.2 DSO (Dynamic Submodule Optimization)
```python
# lr_config配置
lr_config = dict(
    policy='dynamic',
    extra_args={
        'T': 3,           # Softmax温度
        'b': 0.4,         # KL散度偏置
        'ema': 0.001,     # 历史loss平滑系数
        'backbone_policy': 'sigmoid_kl',
        'head_policy': 'normal'
    },
    reweight_losses={
        # Ship头相关loss
        'ship_loss_cls': 'ship_roi_head',
        'ship_loss_bbox': 'ship_roi_head',
        # Wake头相关loss
        'wake_loss_cls': 'wake_roi_head',
        'wake_loss_bbox': 'wake_roi_head',
    }
)

# DSO动态学习率计算
class DynamicLrUpdaterHook:
    def get_dynamic_lr(self, runner):
        # 1. 获取当前loss
        ship_loss = runner.outputs['ship_loss_cls'] + runner.outputs['ship_loss_bbox']
        wake_loss = runner.outputs['wake_loss_cls'] + runner.outputs['wake_loss_bbox']
        
        # 2. 计算相对收敛速度
        ship_speed = ship_loss / ema_ship_loss
        wake_speed = wake_loss / ema_wake_loss
        
        # 3. Head学习率：慢的任务获得更高LR
        ship_lr_weight = 2 * softmax(ship_speed / T)
        wake_lr_weight = 2 * softmax(wake_speed / T)
        
        # 4. Backbone学习率：基于loss一致性
        ship_prob = softmax(ship_loss)
        wake_prob = softmax(wake_loss)
        kl_div = KL(ship_prob, wake_prob)
        consistency = 1 - kl_div
        backbone_lr_weight = 2 * sigmoid((consistency - b) * T)
        
        return [base_lr * weight for weight in lr_weights]
```

---

## 四、可视化能力

### 4.1 训练过程中间结果
```python
# 在每个stage后自动保存
intermediates = backbone.forward_with_intermediates(x)

for i, inter in enumerate(intermediates):
    visualize_single_stage(inter, save_path=f'stage_{i}.png')
    # 输出: 置信度热力图、方向向量场、一致性分数
```

### 4.2 推理结果可视化
```python
visualize_detection_results(
    img, 
    bboxes=[ship_bboxes, wake_bboxes],
    labels=[0]*len(ships) + [1]*len(wakes),
    scores=scores,
    save_path='result.png'
)
```

### 4.3 训练曲线
```python
visualize_training_progress(
    loss_history={
        'ship_loss_cls': [...],
        'ship_loss_bbox': [...],
        'wake_loss_cls': [...],
        'wake_loss_bbox': [...]
    },
    save_path='training_curves.png'
)
```

---

## 五、实施状态

### 已完成
- ✅ GeometricMAMG 模块 (含方向一致性计算)
- ✅ WakeResidualTransform 模块 (条带卷积/低频滤波)
- ✅ ShipWakeDualHead 双检测头
- ✅ ShipWakeDualDetector 检测器
- ✅ ConvNeXt_moe_wake 集成backbone
- ✅ 可视化工具函数
- ✅ 配置文件

### 待完成
- ⏳ SwimShip 数据集适配
- ⏳ 数据增强策略
- ⏳ 端到端训练验证

---

## 六、关键代码文件

| 文件路径 | 说明 |
|---------|------|
| `mmrotate/models/backbones/geometric_mamg.py` | 几何感知MAMG实现 |
| `mmrotate/models/backbones/wake_residual_transform.py` | 交错残差变换 |
| `mmrotate/models/backbones/convnext_moe_wake.py` | 集成backbone |
| `mmrotate/models/dense_heads/ship_wake_head.py` | 双检测头 |
| `mmrotate/models/detectors/shipwake_dual_detector.py` | 检测器 |
| `mmrotate/utils/visualization.py` | 可视化工具 |
| `configs/ShipWake/ShipWake_convnext_t.py` | 配置文件 |

---

*报告生成时间: 2026年4月*
*项目: ShipWake4Det - 基于SM3Det的光学遥感船只-尾迹协同检测*
