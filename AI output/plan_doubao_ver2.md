# ShipWake4Det (SWIM适配版) - Vibe Coding 指导手册
## 文档说明
本手册用于指导基于SM3Det基座模型、适配SWIM-Ship Wake Imagery Mass数据集的光学遥感图像船只-尾迹协同检测模型开发，覆盖**核心代码修改、数据集适配、训练配置、验证可视化**全流程，同时对齐论文撰写的创新点与实验支撑需求。

## 一、项目核心定位
### 1.1 任务边界
| 维度         | 具体说明                                                                 |
|--------------|--------------------------------------------------------------------------|
| 模态         | 单模态（光学RGB遥感图像）                                                |
| 目标类型     | 2类（船只Ship、尾迹Wake）                                                |
| 数据集       | SWIM-Ship Wake Imagery Mass（VOC格式、768×768像素、船只点标注+尾迹OBB标注） |
| 基座模型     | SM3Det（复用ConvNeXt backbone、grid-level MoE、DSO动态LR模块）            |
| 核心创新     | 交错残差变换模块、方向+位置双流注意力蒙版引导                            |
| 适配修改     | 检测头（适配船只点标注/尾迹OBB标注）、数据集加载管道                     |

### 1.2 关键约束
- 船只标注仅为**点+方向**（3元组：x,y,θ），学习速度快于尾迹（OBB标注），需通过DSO差异化调节LR；
- SWIM图像尺寸为768×768，需适配SM3Det的800×800输入（或修改backbone输入逻辑）；
- 尾迹标注含OBB+类别（是否为尾迹），船只仅有点位置+方向，无类别标注（默认正样本）。

## 二、核心修改模块实现指南
### 模块1：交错残差变换（核心创新点1）
#### 目标
针对船只（稠密高频）和尾迹（稀疏低频线性）的特征差异，在backbone的4个stage中交错部署残差变换，与grid-level MoE并行引入全局信息。

#### 实现细节
| 关键项         | 具体要求                                                                 |
|----------------|--------------------------------------------------------------------------|
| 代码路径       | `mmrotate/models/backbones/wake_residual_transform.py`                   |
| 核心组件       | 1. `LowFreqResidual`：高斯滤波保留低频，强化船只语义纹理<br>2. `StripConvResidual`：不对称条带卷积捕获尾迹线性结构 |
| Stage配置      | 严格对齐如下表格（基于SWIM 768×768输入，下采样后分辨率需重新计算）：<br>| Stage | 下采样倍率 | 输入768×768后分辨率 | 变换类型 | 核尺寸 |
|-------|-----------|-------------------|---------|---------|
| 1 | 4× | 192×192 | LowFreqResidual | 5×5 |
| 2 | 8× | 96×96 | StripConvResidual | 1×7/7×1 |
| 3 | 16× | 48×48 | LowFreqResidual | 7×7 |
| 4 | 32× | 24×24 | StripConvResidual | 1×11/11×1 |
| 融合逻辑       | 残差分支与MoE主分支通过`ResidualFusion`（通道注意力）融合，代码需新增：<br>```python<br>class ResidualFusion(nn.Module):<br>    def __init__(self, channels):<br>        super().__init__()<br>        self.lambda_res = nn.Parameter(torch.tensor(0.1))<br>        self.channel_att = nn.Sequential(<br>            nn.AdaptiveAvgPool2d(1),<br>            nn.Flatten(),<br>            nn.Linear(channels*2, channels//4),<br>            nn.ReLU(),<br>            nn.Linear(channels//4, channels*2),<br>            nn.Sigmoid()<br>        )<br>    def forward(self, main_feat, res_feat):<br>        # 加权残差<br>        res_weighted = self.lambda_res * res_feat<br>        # 通道注意力融合<br>        concat = torch.cat([main_feat, res_weighted], dim=1)<br>        att = self.channel_att(concat).unsqueeze(-1).unsqueeze(-1)<br>        fused = concat * att<br>        # 残差连接<br>        return main_feat + fused[:, :main_feat.shape[1], :, :]<br>``` |
| 集成到Backbone | 修改`mmrotate/models/backbones/convnext_moe_wake.py`的`ConvNeXt_moe_wake`类，在每个stage的MoE模块后插入残差变换：<br>```python<br>class ConvNeXt_moe_wake(nn.Module):<br>    def __init__(self, ...):<br>        # 初始化残差模块<br>        self.residual_stages = nn.ModuleList([<br>            LowFreqResidual(channels=96, kernel_size=5),  # Stage1<br>            StripConvResidual(channels=192, kernel_size=7), # Stage2<br>            LowFreqResidual(channels=384, kernel_size=7), # Stage3<br>            StripConvResidual(channels=768, kernel_size=11) # Stage4<br>        ])<br>        self.res_fusion = ResidualFusion(channels=...)<br>    def forward(self, x):<br>        for i, stage in enumerate(self.stages):<br>            x = stage(x)  # MoE主分支<br>            res_feat = self.residual_stages[i](x)<br>            x = self.res_fusion(x, res_feat)  # 融合残差<br>        return x<br>``` |
| 验证要点       | 打印各stage输出特征尺寸，确保下采样后分辨率与上表一致；可视化残差分支融合前后的特征热力图。 |

### 模块2：方向+位置双流注意力蒙版（核心创新点2）
#### 目标
利用船只-尾迹的几何关联（尾迹沿船只航向延伸），生成含**置信度、方向、距离**的注意力蒙版，实现双向特征引导。

#### 实现细节
| 关键项         | 具体要求                                                                 |
|----------------|--------------------------------------------------------------------------|
| 代码路径       | `mmrotate/models/backbones/geometric_mamg.py`                            |
| 蒙版通道设计   | 替换原6通道蒙版，改为6通道（船只3通道+尾迹3通道）：<br>| 目标 | 通道 | 维度 | 说明 |
|------|------|------|------|
| 船只 | 0 | (H,W) | 置信度[0,1] |
| 船只 | 1-2 | (H,W,2) | 方向向量(cosθ, sinθ) |
| 船只 | 3 | (H,W) | 到船只中心的距离场 |
| 尾迹 | 4 | (H,W) | 置信度[0,1] |
| 尾迹 | 5-6 | (H,W,2) | 方向向量(cosθ, sinθ) |
| 尾迹 | 7 | (H,W) | 到尾迹中心的距离场 |
| （注：实际实现为(B,8,H,W)，后拆分Ship_mask=[0-3], Wake_mask=[4-7]） |
| 方向注意力计算 | 重写`compute_direction_alignment`函数，融合方向差+置信度+距离：<br>```python<br>def compute_direction_alignment(ship_dir, wake_dir, ship_conf, wake_conf, ship_dist, wake_dist):<br>    # 计算方向余弦相似度：cos(θ_ship - θ_wake)<br>    dir_dot = (ship_dir[:, 0:1] * wake_dir[:, 0:1]) + (ship_dir[:, 1:2] * wake_dir[:, 1:2])<br>    # 距离权重：距离目标越近，权重越高<br>    dist_weight = 1 / (1 + torch.exp(ship_dist + wake_dist))<br>    # 方向注意力 = 置信度 × (1 + 方向相似度) × 距离权重<br>    directional_att = ship_conf * wake_conf * (1 + dir_dot) * dist_weight<br>    return directional_att<br>``` |
| 特征引导公式   | 替换原融合公式，实现位置+方向的双引导：<br>```python<br># 尾迹特征受船只引导<br>F_wake_guided = F_input * (1 + α * directional_att)<br># 船只特征受尾迹引导<br>F_ship_guided = F_input * (1 + β * directional_att)<br>``` |
| 跨层传播       | 修改`GeometricPropagator`，新增距离场的插值归一化（避免下采样后距离值失真）：<br>```python<br>def forward(self, curr_mask, prev_mask):<br>    prev_up = F.interpolate(prev_mask, size=curr_mask.shape[2:], mode='bilinear')<br>    # 方向归一化<br>    prev_up[:, 1:3] = normalize_direction(prev_up[:, 1:3])<br>    prev_up[:, 5:7] = normalize_direction(prev_up[:, 5:7])<br>    # 距离场归一化（0-1）<br>    prev_up[:, 3] = prev_up[:, 3] / (prev_up[:, 3].max() + 1e-6)<br>    prev_up[:, 7] = prev_up[:, 7] / (prev_up[:, 7].max() + 1e-6)<br>    # 置信度加权融合<br>    ...<br>    return fused_mask<br>``` |
| 验证要点       | 可视化各stage的蒙版（置信度热力图、方向向量场、距离场），验证跨层传播后的蒙版一致性。 |

### 模块3：检测头适配（数据集适配核心）
#### 目标
适配SWIM数据集的标注格式：船只仅有点+方向（x,y,θ），尾迹为OBB（x,y,w,h,θ）+类别（是否为尾迹），对接DSO动态LR模块。

#### 实现细节
| 关键项         | 具体要求                                                                 |
|----------------|--------------------------------------------------------------------------|
| 代码路径       | `mmrotate/models/dense_heads/ship_wake_head.py`                          |
| 检测头拆分     | 1. **ShipHead**：预测3元组（x,y,θ），无分类分支（船只标注无负样本）<br>2. **WakeHead**：预测5元组OBB（x,y,w,h,θ）+1维分类（尾迹/背景） |
| ShipHead实现   | ```python<br>class ShipHead(ODMRefineHead):<br>    def __init__(self, ...):<br>        super().__init__(num_classes=1)<br>        # 移除分类分支，仅保留回归（x,y,θ）<br>        self.odm_reg = nn.Conv2d(feat_c, num_anchors * 3, 3)  # 替换原5维为3维<br>        self.odm_dir = None  # 复用θ作为方向，无需额外输出<br>    def forward_single(self, x):<br>        or_feat = self.or_conv(x)<br>        reg_feat = or_feat<br>        bbox_pred = self.odm_reg(reg_feat)  # (x,y,θ)<br>        # θ归一化到[-π/2, π/2]<br>        bbox_pred[:, 2:3] = torch.atan(bbox_pred[:, 2:3])<br>        return None, bbox_pred, None  # 无分类、无额外方向<br>``` |
| WakeHead实现   | 复用原OBB检测头，保留分类+回归：<br>```python<br>class WakeHead(ODMRefineHead):<br>    def __init__(self, ...):<br>        super().__init__(num_classes=1)<br>        self.odm_cls = nn.Conv2d(feat_c, num_anchors * 1, 3)  # 尾迹/背景分类<br>        self.odm_reg = nn.Conv2d(feat_c, num_anchors * 5, 3)  # OBB (x,y,w,h,θ)<br>    def forward_single(self, x):<br>        or_feat = self.or_conv(x)<br>        cls_feat = self.or_pool(or_feat)<br>        reg_feat = or_feat<br>        cls_score = self.odm_cls(cls_feat)<br>        bbox_pred = self.odm_reg(reg_feat)<br>        # θ归一化<br>        bbox_pred[:, 4:5] = torch.atan(bbox_pred[:, 4:5])<br>        return cls_score, bbox_pred, None<br>``` |
| DSO对接       | 修改`mmrotate/core/hook/dynamic_lr.py`的`DynamicLrUpdaterHook`，适配新的loss映射：<br>```python<br>reweight_losses={<br>    # Ship头（仅回归loss）<br>    'ship_loss_reg': 'ship_roi_head',<br>    # Wake头（分类+回归loss）<br>    'wake_loss_cls': 'wake_roi_head',<br>    'wake_loss_bbox': 'wake_roi_head',<br>}<br>``` |
| 损失函数适配   | ShipHead仅用L1损失（点+方向回归），WakeHead用FocalLoss（分类）+ SmoothL1Loss（OBB回归）：<br>在`ship_wake_head.py`的`loss`函数中：<br>```python<br>def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels):<br>    if self.is_ship_head:<br>        # 船只仅回归损失<br>        loss_reg = self.reg_loss(bbox_preds, gt_bboxes)<br>        return dict(ship_loss_reg=loss_reg)<br>    else:<br>        # 尾迹分类+回归损失<br>        loss_cls = self.cls_loss(cls_scores, gt_labels)<br>        loss_bbox = self.reg_loss(bbox_preds, gt_bboxes)<br>        return dict(wake_loss_cls=loss_cls, wake_loss_bbox=loss_bbox)<br>``` |
| 验证要点       | 打印检测头输出维度（ShipHead: (B,3,H,W)，WakeHead: (B,1,H,W)+(B,5,H,W)）；验证DSO是否为ShipHead分配更高LR（因学习速度快）。 |

## 三、SWIM数据集适配（重点）
### 3.1 数据集基础信息
- 格式：PASCAL VOC（XML标注文件）；
- 图像尺寸：768×768像素；
- 标注：
  - 船只：点坐标（x,y）+航向θ（XML中`<direction>`字段）；
  - 尾迹：OBB（x1,y1,x2,y2,x3,y3,x4,y4）+类别（`<name>wake</name>`）；
- 数据量：11600正样本（含尾迹）+3010负样本。

### 3.2 数据集加载实现
| 关键项         | 具体要求                                                                 |
|----------------|--------------------------------------------------------------------------|
| 代码路径       | `mmrotate/datasets/swim_shipwake.py`（新增文件）                          |
| 数据集类定义   | ```python<br>@DATASETS.register_module()<br>class SWIMShipWakeDataset(CustomDataset):<br>    CLASSES = ('ship', 'wake')<br>    def __init__(self, ann_file, img_prefix, pipeline, **kwargs):<br>        super().__init__(ann_file, img_prefix, pipeline, **kwargs)<br>    def load_annotations(self, ann_file):<br>        # 解析VOC格式XML<br>        data_infos = []<br>        with open(ann_file) as f:<br>            for line in f:<br>                img_id = line.strip()<br>                xml_path = os.path.join(self.img_prefix, 'Annotations', f'{img_id}.xml')<br>                tree = ET.parse(xml_path)<br>                root = tree.getroot()<br>                # 图像尺寸<br>                width = int(root.find('size/width').text)<br>                height = int(root.find('size/height').text)<br>                # 解析标注<br>                anns = []<br>                for obj in root.findall('object'):<br>                    label = obj.find('name').text<br>                    if label == 'ship':<br>                        # 船只：x,y,θ<br>                        x = float(obj.find('point/x').text)<br>                        y = float(obj.find('point/y').text)<br>                        theta = float(obj.find('direction').text)<br>                        anns.append(dict(<br>                            type='ship',<br>                            bbox=[x, y, theta],<br>                            label=0  # ship对应label 0<br>                        ))<br>                    elif label == 'wake':<br>                        # 尾迹：OBB（转x,y,w,h,θ）<br>                        poly = [float(p) for p in obj.find('polygon').text.split(',')]<br>                        # 多边形转OBB（使用mmrotate的poly2obb函数）<br>                        x, y, w, h, theta = poly2obb(poly, 'le90')<br>                        anns.append(dict(<br>                            type='wake',<br>                            bbox=[x, y, w, h, theta],<br>                            label=1  # wake对应label 1<br>                        ))<br>                data_infos.append(dict(<br>                    filename=f'{img_id}.jpg',<br>                    width=width,<br>                    height=height,<br>                    ann=dict(bboxes=anns)<br>                ))<br>        return data_infos<br>``` |
| 数据预处理管道 | 修改`configs/_base_/datasets/swim_shipwake.py`，适配768→800尺寸：<br>```python<br>train_pipeline = [<br>    dict(type='LoadImageFromFile'),<br>    dict(type='LoadAnnotations', with_bbox=True, with_theta=True),  # 新增theta加载<br>    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),  # 768→800<br>    dict(type='RandomFlip', flip_ratio=0.5),<br>    dict(type='Normalize', **img_norm_cfg),<br>    dict(type='Pad', size_divisor=32),<br>    dict(type='DefaultFormatBundle'),<br>    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),<br>]<br>``` |
| 配置文件       | 在`configs/ShipWake/ShipWake_convnext_t.py`中指定数据集：<br>```python<br>dataset_type = 'SWIMShipWakeDataset'<br>data_root = 'data/SWIM/'<br>data = dict(<br>    train=dict(<br>        type=dataset_type,<br>        ann_file=data_root + 'ImageSets/Main/train.txt',<br>        img_prefix=data_root + 'JPEGImages/',<br>        pipeline=train_pipeline<br>    ),<br>    val=dict(...),<br>    test=dict(...)<br>)<br>``` |
| 验证要点       | 运行`tools/analysis_tools/browse_dataset.py`，可视化加载后的图像+标注（船只点、尾迹OBB），确认标注转换正确。 |

## 四、训练配置与运行
### 4.1 核心配置修改
修改`configs/ShipWake/ShipWake_convnext_t.py`，适配SWIM数据集和模型修改：
```python
# 模型配置
model = dict(
    backbone=dict(
        type='ConvNeXt_moe_wake',
        arch='tiny',
        residual_stages=True,  # 启用残差变换
        moe_block_inds=[2,3],  # Stage3/4启用MoE
        num_experts=4,  # 跨目标类型路由
        top_k=2
    ),
    roi_head=dict(
        ship_roi_head=dict(
            type='ShipHead',
            num_classes=1,
            reg_dim=3  # 船只3维回归
        ),
        wake_roi_head=dict(
            type='WakeHead',
            num_classes=1,
            reg_dim=5  # 尾迹5维回归
        )
    ),
    # DSO动态LR配置
    lr_config=dict(
        policy='dynamic',
        extra_args={
            'T': 3,
            'b': 0.4,
            'ema': 0.001,
            'backbone_policy': 'sigmoid_kl',
            'head_policy': 'normal'
        },
        reweight_losses={
            'ship_loss_reg': 'ship_roi_head',
            'wake_loss_cls': 'wake_roi_head',
            'wake_loss_bbox': 'wake_roi_head'
        }
    )
)
# 数据集配置
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='SWIMShipWakeDataset',
        ann_file='data/SWIM/ImageSets/Main/train.txt',
        img_prefix='data/SWIM/JPEGImages/'
    ),
    val=dict(
        type='SWIMShipWakeDataset',
        ann_file='data/SWIM/ImageSets/Main/val.txt',
        img_prefix='data/SWIM/JPEGImages/'
    )
)
# 输入尺寸适配
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)
```

### 4.2 训练/测试命令
```bash
# 单GPU训练
python tools/train.py configs/ShipWake/ShipWake_convnext_t.py --work-dir work_dirs/shipwake_swim

# 多GPU训练
./tools/dist_train.sh configs/ShipWake/ShipWake_convnext_t.py 8 --work-dir work_dirs/shipwake_swim

# 测试（评估mAP）
python tools/test.py configs/ShipWake/ShipWake_convnext_t.py work_dirs/shipwake_swim/latest.pth --eval mAP

# 可视化推理结果
python tools/analysis_tools/visualize_results.py configs/ShipWake/ShipWake_convnext_t.py work_dirs/shipwake_swim/latest.pth --show-dir work_dirs/vis
```

## 五、验证与可视化（论文支撑）
### 5.1 核心验证项
| 验证内容               | 工具/代码                                                                 |
|------------------------|--------------------------------------------------------------------------|
| 残差变换特征有效性     | 对比有无残差变换的特征可视化（`mmrotate/utils/visualization.py`），打印船只/尾迹的特征响应值 |
| 注意力蒙版引导效果     | 可视化方向注意力热力图，计算船只-尾迹的方向对齐分数（`compute_direction_alignment`输出） |
| DSO动态LR效果          | 绘制训练过程中ShipHead/WakeHead/Backbone的LR曲线（`tools/analysis_tools/plot_lr.py`） |
| 检测精度               | 计算船只点回归的MAE（平均绝对误差）、尾迹OBB的mAP@0.5                    |

### 5.2 论文撰写支撑
1. **创新点1（交错残差变换）**：
   - 对比实验：关闭残差变换/仅用LowFreq/仅用StripConv的精度差异；
   - 可视化：各stage残差分支的特征响应（船只稠密区域vs尾迹线性区域）。
2. **创新点2（双流注意力蒙版）**：
   - 消融实验：仅位置引导vs位置+方向引导的精度差异；
   - 可视化：注意力蒙版叠加在原图上的效果（置信度+方向向量）。
3. **数据集适配**：
   - 统计SWIM数据集的船只/尾迹分布、标注质量；
   - 对比SM3Det原模型（多模态）与本模型（单模态+双目标）的精度/效率。

## 六、常见问题排查
1. **标注解析错误**：检查XML中船只点/尾迹多边形的字段名，确保`poly2obb`函数的参数（如角度范围`le90`）正确；
2. **特征尺寸不匹配**：确认各stage的下采样倍率与残差模块的通道数对齐；
3. **DSO LR无变化**：检查`DynamicLrUpdaterHook`是否注册，loss名称与`reweight_losses`映射是否一致；
4. **显存溢出**：降低`samples_per_gpu`（如从8改为4），或减小backbone的arch（如tiny→nano）。

## 七、交付物清单
1. 核心代码文件：`geometric_mamg.py`/`wake_residual_transform.py`/`ship_wake_head.py`/`swim_shipwake.py`；
2. 配置文件：`ShipWake_convnext_t.py`；
3. 训练日志+精度曲线；
4. 可视化结果（特征/蒙版/检测结果）；
5. 消融实验表格（支撑论文创新点）。