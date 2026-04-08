# ShipWake DualStream 数据流通道技术报告

## 组会汇报：光学遥感图像船只-尾迹协同检测模型

---

## 1. 模型整体架构概览

### 1.1 系统架构图

```
输入图像 [B, 3, H, W]
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DualStream Backbone                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ConvNeXt_DualStream.forward()                               │   │
│  │  ├── Stem: dataset_stems['single']() -> downsample_layers[0] │   │
│  │  │                                                           │   │
│  │  ├── Stage 0 (4x) : 200x200                                   │   │
│  │  │   └── ConvNeXtBlock_DualStream.forward()                  │   │
│  │  │       ├── depthwise_conv()                                │   │
│  │  │       ├── MoE_layer.forward()  <- 4 Experts, top-1         │   │
│  │  │       ├── WakeResidualBlock.forward()  <- LowFreq(5x5)     │   │
│  │  │       ├── ResidualFusion.forward()                        │   │
│  │  │       └── MutualMaskGenerator.generate_masks()            │   │
│  │  │           -> mask_ship [B,1,200,200]                       │   │
│  │  │           -> mask_wake [B,1,200,200]                       │   │
│  │  │                                                           │   │
│  │  ├── Stage 1 (8x) : 100x100  <- mask from Stage 0            │   │
│  │  │   └── MaskGuidedFusion.forward()                         │   │
│  │  │       └── F_guided = F x (1 + alpha x mask_ship/wake)     │   │
│  │  │                                                           │   │
│  │  ├── Stage 2 (16x) : 50x50                                   │   │
│  │  │                                                           │   │
│  │  └── Stage 3 (32x) : 25x25                                   │   │
│  │                                                               │   │
│  └── Output: feats [P2, P3, P4, P5] + gate_loss                 │   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├──> ship_neck (FPN) ---> ship_bbox_head (ODMRefineHead)
    │                          └── Ship detection results
    │
    └──> wake_neck (FPN) ---> wake_bbox_head (ODMRefineHead)
                               └── Wake detection results
```

### 1.2 训练阶段完整数据流

```python
# 主入口: ShipWakeDualDetector.forward_train()
def forward_train(img, img_metas, gt_bboxes, gt_labels):
    
    # Step 1: 特征提取
    ship_feats, wake_feats, gate_loss = self.extract_feat(img)
    #    └── ConvNeXt_DualStream.forward(img) -> feats, gate_loss
    
    # Step 2: GT分离（按类别）
    ship_gt_bboxes, ship_gt_labels, wake_gt_bboxes, wake_gt_labels = \
        self._split_ship_wake_gt(gt_bboxes, gt_labels)
    
    # Step 3: 船只检测头前向
    ship_losses = self.ship_bbox_head.forward_train(
        ship_feats, img_metas, ship_gt_bboxes, ship_gt_labels
    )
    
    # Step 4: 尾迹检测头前向
    wake_losses = self.wake_bbox_head.forward_train(
        wake_feats, img_metas, wake_gt_bboxes, wake_gt_labels
    )
    
    # Step 5: 合并损失
    losses = {}
    losses.update({'ship_' + k: v for k, v in ship_losses.items()})
    losses.update({'wake_' + k: v for k, v in wake_losses.items()})
    losses['gate_loss'] = gate_loss
    
    # Step 6: DSO动态加权（通过hook自动处理）
    return losses
```

---

## 2. 核心模块详细数据流

### 2.1 模块一：Grid-Level MoE（改进版）

**模块定位**: `mmrotate/models/backbones/convnext_moe.py`

#### 功能说明
将原始SM3Det的多模态MoE改造为单模态双目标检测。门控网络学习区分：
- 船只区域（小尺寸、高密度纹理）
- 尾迹区域（大尺寸、线性结构）
- 背景区域

#### 数据流详情

```python
# 输入特征（来自Depthwise Conv）
x: Tensor[B, C, H, W]  # e.g., [2, 192, 100, 100]

# 步骤1: 特征reshape为序列
x_seq = x.reshape(-1, C)  # [B*H*W, C] = [20000, 192]

# 步骤2: 门控网络计算相似度
# CosineTopKGate.forward()
logits = torch.matmul(
    F.normalize(cosine_projector(x_seq), dim=1),  # [20000, 128]
    F.normalize(sim_matrix, dim=0)                 # [128, 4]
)  # -> [20000, 4]

# 步骤3: Top-K选择（K=1）
top_logits, top_indices = logits.topk(k=1, dim=-1)  # [20000, 1]
gates = softmax(top_logits)  # [20000, 1]

# 步骤4: 专家路由（SparseDispatcher）
# 每个空间位置激活1个专家
dispatcher = SparseDispatcher(num_experts=4, gates=gates)
expert_inputs = dispatcher.dispatch(x_seq)  # List[Tensor]

# 步骤5: 专家计算（并行的1x1卷积）
expert_outputs = []
for i in range(4):
    out = self.experts[i](expert_inputs[i])  # FFN: Linear->GELU->Linear
    expert_outputs.append(out)

# 步骤6: 结果合并
y = dispatcher.combine(expert_outputs)  # [20000, C]
y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

# 输出
return y, gate_loss
```

#### 输入输出规格

| 变量 | 形状 | 说明 |
|-----|------|------|
| Input x | [B, C, H, W] | 深度卷积后的特征 |
| Output y | [B, C, H, W] | MoE融合后的特征 |
| Output gate_loss | Scalar | 负载均衡损失（CV^2） |
| gates | [B*H*W, K] | 每个位置的门控权重 |
| load | [num_experts] | 每个专家的负载统计 |

---

### 2.2 模块二：交错残差变换（Wake Residual Transform）

**模块定位**: `mmrotate/models/backbones/wake_residual_transform.py`

#### 功能说明
与MoE并行，通过交错设置的变换提取互补特征：
- **低频滤波（Stage 0, 2）**: 高斯模糊平滑背景，保留船只纹理
- **条带卷积（Stage 1, 3）**: 1xk和kx1卷积提取尾迹线性结构

#### 数据流详情

```python
# 类: WakeResidualBlock.forward()
def forward(self, x, stage_idx):
    """
    Args:
        x: [B, C, H, W] - 来自depthwise conv的特征
        stage_idx: 0, 1, 2, 3
    """
    
    # 根据stage选择变换类型
    if stage_idx in [0, 2]:  # 低频滤波 - 适合船只
        # LowFreqFilterBlock.forward()
        # 步骤1: 构建高斯核（预计算，buffer）
        kernel = self.gaussian_kernel  # [1, 1, K, K]
        
        # 步骤2: 深度可分离高斯模糊
        kernel_expanded = kernel.expand(C, 1, -1, -1)  # [C, 1, K, K]
        blurred = F.conv2d(
            x, kernel_expanded, 
            padding=self.kernel_size//2, 
            groups=C
        )  # [B, C, H, W]
        
        # 步骤3: 可学习通道缩放
        out = blurred * self.channel_scale.view(1, C, 1, 1)
        
    else:  # stage_idx in [1, 3] - 条带卷积 - 适合尾迹
        # StripConvBlock.forward()
        # 步骤1: 水平条带卷积
        h_feat = self.h_conv(x)  # Conv(1, k), [B, C/2, H, W]
        
        # 步骤2: 垂直条带卷积
        v_feat = self.v_conv(x)  # Conv(k, 1), [B, C/2, H, W]
        
        # 步骤3: 拼接并融合
        concat_feat = torch.cat([h_feat, v_feat], dim=1)  # [B, C, H, W]
        out = self.fusion_conv(concat_feat)  # Conv(1x1) + BN + ReLU
    
    return out


# 类: ResidualFusion.forward()
def forward(self, f_moe, f_residual):
    """
    融合MoE输出和残差变换输出
    """
    # 步骤1: 对齐残差特征
    f_residual_aligned = self.align_conv(f_residual)  # Conv(1x1) + BN + ReLU
    
    # 步骤2: 可学习加权融合
    lambda_val = torch.sigmoid(self.lambda_residual)  # 限制在(0,1)
    f_fused = f_moe + lambda_val * f_residual_aligned
    
    return f_fused
```

#### 各Stage配置

| Stage | 下采样率 | 特征尺寸 | 变换类型 | 核大小 | 适用目标 |
|-------|---------|---------|---------|--------|---------|
| 0 | 4x | 200x200 | LowFreqFilter | 5x5 | 船只纹理（高频） |
| 1 | 8x | 100x100 | StripConv | 1x7, 7x1 | 尾迹线（中频） |
| 2 | 16x | 50x50 | LowFreqFilter | 7x7 | 船只语义（低频） |
| 3 | 32x | 25x25 | StripConv | 1x11, 11x1 | 尾迹全局（超低频） |

#### 输入输出规格

| 变量 | 形状 | 说明 |
|-----|------|------|
| Input x | [B, C, H, W] | 深度卷积输出 |
| Output (residual) | [B, C, H, W] | 变换后的残差特征 |
| f_moe | [B, C, H, W] | MoE分支输出 |
| f_residual | [B, C, H, W] | 残差分支输出 |
| Output (fused) | [B, C, H, W] | 融合后特征 |
| lambda_residual | Scalar (learnable) | 融合权重初始值0.1 |

---

### 2.3 模块三：互引导注意力蒙版（MAMG）

**模块定位**: `mmrotate/models/backbones/mutual_attention_mask.py`

#### 功能说明
利用船只-尾迹几何关联，对称生成注意力蒙版：
- **船只蒙版**：尾迹蒙版作为先验，增强船只尾部区域
- **尾迹蒙版**：船只蒙版作为先验，引导尾迹方向延伸

#### 数据流详情

```python
# 类: MutualMaskGenerator.forward()
def forward(self, x):
    """
    从当前层特征生成对称蒙版
    Args:
        x: [B, C, H, W] - 当前层融合特征
    """
    # 船只蒙版分支（小核，关注细节）
    ship_logit = self.ship_mask_conv(x)  # Conv3x3->BN->ReLU->Conv1x1
    #     [B, C, H, W] -> [B, 1, H, W]
    mask_ship = torch.sigmoid(ship_logit)  # [B, 1, H, W], values in [0,1]
    
    # 尾迹蒙版分支（大核，关注上下文）
    wake_logit = self.wake_mask_conv(x)  # Conv5x5->BN->ReLU->Conv1x1
    #     [B, C, H, W] -> [B, 1, H, W]
    mask_wake = torch.sigmoid(wake_logit)  # [B, 1, H, W], values in [0,1]
    
    return {
        'mask_ship': mask_ship,  # [B, 1, H, W]
        'mask_wake': mask_wake,  # [B, 1, H, W]
    }


# 类: MaskGuidedFusion.forward()
def forward(self, features, masks):
    """
    使用上一层蒙版引导当前层特征
    Args:
        features: [B, C, H, W] - 下一层输入特征
        masks: dict with 'mask_ship', 'mask_wake' from previous layer
    """
    mask_ship = masks['mask_ship']  # [B, 1, H, W]
    mask_wake = masks['mask_wake']  # [B, 1, H, W]
    
    # 约束引导强度
    alpha = torch.sigmoid(self.alpha)  # 初始0.2
    beta = torch.sigmoid(self.beta)    # 初始0.2
    
    # 船只通路: 尾迹蒙版引导（尾迹起点有船）
    # F_ship = F * (1 + alpha * Mask_wake)
    ship_guided = features * (1 + alpha * mask_wake)  # [B, C, H, W]
    
    # 尾迹通路: 船只蒙版引导（尾迹从船延伸）
    # F_wake = F * (1 + beta * Mask_ship)
    wake_guided = features * (1 + beta * mask_ship)  # [B, C, H, W]
    
    # 拼接并融合
    concat = torch.cat([ship_guided, wake_guided], dim=1)  # [B, 2C, H, W]
    fused = self.fusion_conv(concat)  # Conv1x1->BN->ReLU, [B, C, H, W]
    
    return fused
```

#### 跨层蒙版传播

```
Stage 0 Output ---> generate_masks() ---> mask_ship_s0 [B,1,200,200]
                                      ---> mask_wake_s0 [B,1,200,200]
                                              |
                                              v
Stage 1 Input <--- MaskGuidedFusion() <-------+
    |
    +---> generate_masks() ---> mask_ship_s1 [B,1,100,100]
                           ---> mask_wake_s1 [B,1,100,100]
```

#### 输入输出规格

| 变量 | 形状 | 说明 | 值域 |
|-----|------|------|------|
| Input x (current) | [B, C, H, W] | 当前层特征 | - |
| Output mask_ship | [B, 1, H, W] | 船只注意力蒙版 | [0, 1] |
| Output mask_wake | [B, 1, H, W] | 尾迹注意力蒙版 | [0, 1] |
| Input features (next) | [B, C, H', W'] | 下一层特征 | - |
| Input masks (prev) | Dict[str, Tensor] | 上一层蒙版 | [0, 1] |
| Output (guided) | [B, C, H', W'] | 引导后特征 | - |
| alpha | Scalar | 船只引导强度（可学习） | init=0.2 |
| beta | Scalar | 尾迹引导强度（可学习） | init=0.2 |

---

## 3. 完整前向传播数据流（详细版）

### 3.1 ConvNeXtBlock_DualStream 详细流程

```python
class ConvNeXtBlock_DualStream.forward(x, prev_masks=None, return_masks=True):
    """
    单个DualStream Block的完整数据流
    """
    # ==================== 输入 ====================
    x: [B, C, H, W]           # 输入特征
    prev_masks: Dict or None  # 上一层传来的蒙版
    
    # ==================== Step 1: 应用上层蒙版 ====================
    if prev_masks is not None:
        x = mask_fusion.forward(x, prev_masks)  # [B, C, H, W]
        # 内部: x * (1 + alpha*mask_wake) + x * (1 + beta*mask_ship) -> fusion
    
    # ==================== Step 2: 标准ConvNeXt处理 ====================
    shortcut = x                              # 残差连接保存
    x = depthwise_conv(x)                     # [B, C, H, W], 7x7 depthwise
    x = x.permute(0, 2, 3, 1)                 # [B, H, W, C], NHWC格式
    x = layer_norm(x)                         # [B, H, W, C]
    
    # ==================== Step 3: MoE处理 ====================
    x_moe, loss_moe = moe_layer(x)            # [B, H, W, C], scalar
    # 内部: 
    #   - reshape to [B*H*W, C]
    #   - top-1 routing to 4 experts
    #   - each expert: Linear(C, 4C) -> GELU -> Linear(4C, C)
    #   - combine and reshape back
    
    # ==================== Step 4: 残差变换（并行） ====================
    x_for_residual = x_moe.permute(0, 3, 1, 2)  # [B, C, H, W], NCHW
    
    # Stage-specific transform
    if stage_idx == 0:    # LowFreq, k=5
        x_residual = gaussian_blur(x_for_residual)
    elif stage_idx == 1:  # Strip, k=7
        x_residual = strip_conv_h(x_for_residual) + strip_conv_v(x_for_residual)
    elif stage_idx == 2:  # LowFreq, k=7
        x_residual = gaussian_blur(x_for_residual)
    else:                 # Strip, k=11
        x_residual = strip_conv_h(x_for_residual) + strip_conv_v(x_for_residual)
    
    # 融合
    x_fused = residual_fusion(x_moe_nchw, x_residual)  # [B, C, H, W]
    # 内部: x_moe + sigmoid(lambda) * x_residual
    
    # ==================== Step 5: 生成当前层蒙版 ====================
    if return_masks:
        masks = mask_generator(x_fused)       # Dict{'mask_ship': [B,1,H,W], ...}
        # 内部: Conv->Sigmoid for each mask
    
    # ==================== Step 6: Layer Scale + 残差连接 ====================
    x_fused = x_fused * gamma.view(1, -1, 1, 1)  # [B, C, H, W], element-wise
    x_out = shortcut + drop_path(x_fused)        # [B, C, H, W]
    
    # ==================== 输出 ====================
    return x_out, loss_moe, masks
```

### 3.2 特征维度变化追踪

以输入图像 `[2, 3, 800, 800]` 为例：

| Stage | 操作 | 输出尺寸 | 主要模块 |
|-------|------|---------|---------|
| Input | - | [2, 3, 800, 800] | - |
| Stem | Conv(4x4, s=4) | [2, 96, 200, 200] | dataset_stems['single'] |
| Stage 0 | Blockx3 | [2, 96, 200, 200] | MoE(0) + LowFreq(5x5) + MaskGen |
| Downsample | Conv(2x2, s=2) | [2, 192, 100, 100] | downsample_layers[1] |
| Stage 1 | Blockx3 | [2, 192, 100, 100] | MoE(0,2) + Strip(7x7) + MaskFusion |
| Downsample | Conv(2x2, s=2) | [2, 384, 50, 50] | downsample_layers[2] |
| Stage 2 | Blockx9 | [2, 384, 50, 50] | MoE(0,2,4,6,8) + LowFreq(7x7) + MaskFusion |
| Downsample | Conv(2x2, s=2) | [2, 768, 25, 25] | downsample_layers[3] |
| Stage 3 | Blockx3 | [2, 768, 25, 25] | MoE(0,2) + Strip(11x11) + MaskGen |

**输出金字塔**: 
- P2: [2, 96, 200, 200] 
- P3: [2, 192, 100, 100]
- P4: [2, 384, 50, 50]
- P5: [2, 768, 25, 25]

---

## 4. 检测头数据流

### 4.1 双检测头并行处理

```
Backbone Output
    ├──> ship_neck (FPN)
    │       ├── P2: [B, 256, 200, 200]
    │       ├── P3: [B, 256, 100, 100]
    │       ├── P4: [B, 256, 50, 50]
    │       ├── P5: [B, 256, 25, 25]
    │       └── P6: [B, 256, 13, 13] (extra)
    │
    │       └──> ship_bbox_head (RotatedRetinaHead)
    │               └── Loss: ship_loss_cls, ship_loss_bbox
    │
    └──> wake_neck (FPN)
            ├── P2: [B, 256, 200, 200]
            ├── P3: [B, 256, 100, 100]
            ├── P4: [B, 256, 50, 50]
            ├── P5: [B, 256, 25, 25]
            └── P6: [B, 256, 13, 13] (extra)
            
            └──> wake_bbox_head (RotatedRetinaHead)
                    └── Loss: wake_loss_cls, wake_loss_bbox
```

### 4.2 检测头损失计算

```python
# RotatedRetinaHead.forward_train()
def forward_train(feats, img_metas, gt_bboxes, gt_labels):
    
    # Step 1: 生成anchor
    anchors = self.anchor_generator(feats)  # List[Tensor], each [N, 5]
    
    # Step 2: 特征提取
    cls_scores, bbox_preds = self.forward(feats)  # List[Tensor]
    # cls_scores: each level [B, num_anchors*1, H, W]
    # bbox_preds: each level [B, num_anchors*5, H, W]
    
    # Step 3: 计算损失
    losses = self.loss(
        cls_scores, bbox_preds,
        anchors, gt_bboxes, gt_labels
    )
    
    return {
        'loss_cls': losses['loss_cls'],    # FocalLoss
        'loss_bbox': losses['loss_bbox'],  # SmoothL1
    }
```

---

## 5. DSO（动态子模块优化）数据流

### 5.1 Hook机制

```python
# DynamicLrUpdaterHook (mmrotate/core/hook/dynamic_lr.py)
class DynamicLrUpdaterHook:
    
    def after_train_iter(self, runner):
        # 获取当前损失
        losses = runner.outputs['log_vars']
        
        # 提取任务相关损失
        task_losses = []
        for name in self.reweight_losses.keys():
            if name in losses:
                task_losses.append(losses[name])
        
        # 计算各任务的收敛速度
        # w_i = history_loss_i / current_loss_i
        
        # 调整各head的学习率
        for i, loss in enumerate(task_losses):
            lr_weight = softmax(w_i / T)
            # 应用到对应参数的optimizer
        
        # 计算backbone的一致性分数
        C = 1 - KL(P(current_losses) || P(history_losses))
        backbone_lr_weight = sigmoid((C - b) * T) * 2
        
        # 调整backbone学习率
        self._set_lr(runner, new_lrs)
```

### 5.2 学习率调整示意

```
Iteration t:
    Losses: ship_loss=0.5, wake_loss=0.8
    
    Head LR Adjustment:
        ship_head:  LR x 1.0  (收敛快，保持)
        wake_head:  LR x 1.2  (收敛慢，加速)
    
    Backbone LR Adjustment:
        Consistency C = 0.85  (损失稳定)
        backbone_LR = base_LR x 1.1  (可以加快)
```

---

## 6. 总结：关键创新点数据流

### 创新点1：Grid-Level MoE（单模态适配）
- **输入**: 局部特征 [BxHxW, C]
- **处理**: 每个空间位置独立路由到专家
- **输出**: 融合特征 [BxHxW, C] + 负载损失

### 创新点2：交错残差变换
- **输入**: 深度卷积特征 [B, C, H, W]
- **处理**: Stage-wise交替（低频<->条带）
- **融合**: 可学习加权 F_moe + lambda*F_residual
- **输出**: 增强特征 [B, C, H, W]

### 创新点3：互引导注意力蒙版
- **输入**: 当前层特征 -> 生成蒙版
- **传播**: 蒙版Resize后传递给下一层
- **融合**: F * (1 + alpha*mask)
- **输出**: 引导特征 [B, C, H, W]

---

## 汇报要点提示

1. **重点展示**: 3个创新模块的输入输出对比图
2. **可视化**: 蒙版热力图（红色=船只，蓝色=尾迹）
3. **数据**: 训练曲线展示DSO的动态调整过程
4. **对比**: 与单流 baseline 的特征图差异
