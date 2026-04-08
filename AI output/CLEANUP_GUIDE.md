# SM3Det4Wake 仓库清理指南

## 概述

本指南列出原 SM3Det/BabelRS 仓库中与 **ShipWake DualStream** 项目无关的文件和文件夹，可以安全删除以精简仓库。

---

## 一、完全无关的项目（可整体删除）

### 1. BabelRS 项目（完全无关）

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `BabelRS_configs/` | BabelRS 配置文件 | **删除整个文件夹** |
| `BabelRS_pretrain/` | BabelRS 预训练代码（包含完整子项目） | **删除整个文件夹** |

**BabelRS_pretrain 内部包含：**
- `BabelRS_pretrain/.gitignore`
- `BabelRS_pretrain/babelrs_configs/`
- `BabelRS_pretrain/eval/`
- `BabelRS_pretrain/eval_vg.sh`
- `BabelRS_pretrain/evaluate.sh`
- `BabelRS_pretrain/internvl/`
- `BabelRS_pretrain/merge_lora.py`
- `BabelRS_pretrain/pyproject.toml`
- `BabelRS_pretrain/README.md`
- `BabelRS_pretrain/requirements`
- `BabelRS_pretrain/requirements.txt`
- `BabelRS_pretrain/run.sh`
- `BabelRS_pretrain/shell/`
- `BabelRS_pretrain/tools/`
- `BabelRS_pretrain/zero_stage1_config.json`
- `BabelRS_pretrain/zero_stage2_config.json`
- `BabelRS_pretrain/zero_stage3_config.json`
- `BabelRS_pretrain/zero_stage3_config_100b.json`
- `BabelRS_pretrain/zero_stage3_config_100b_1e8.json`
- `BabelRS_pretrain/zero_stage3_config_34b.json`
- `BabelRS_pretrain/zero_stage3_config_70b.json`

---

## 二、文档和图片（可选删除）

### 2. Docs 文件夹中的非必要文件

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `docs/BabelRS.pdf` | BabelRS 论文 | 可删除 |
| `docs/BabelRS.png` | BabelRS 架构图 | 可删除 |
| `docs/BabelRS_motivation.png` | BabelRS 动机图 | 可删除 |
| `docs/en/` | 英文文档（如存在） | 检查内容后决定 |
| `docs/meme.png` | 宣传用梗图 | 可删除 |
| `docs/results.png` | SM3Det 结果图 | 可保留作为参考 |
| `docs/SM3Det.png` | SM3Det 架构图 | 可保留作为参考 |
| `docs/vis.png` | 可视化示例 | 可删除 |
| `docs/zh_cn/` | 中文文档（如存在） | 检查内容后决定 |

**保留建议：**
- 保留 `docs/SM3Det.png` - 可用于组会汇报对比
- 保留 `docs/results.png` - 可用于对比原始方法性能
- 其余与 BabelRS 相关的图片可删除

---

## 三、配置文件（大量可删除）

### 3.1 configs/ 下的非必要算法配置

configs 文件夹包含大量传统旋转目标检测算法的配置，**仅保留 ShipWake 和 SM3Det 相关配置**。

**可删除的算法配置文件夹：**

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `configs/cfa/` | CFA 算法配置 | 可删除 |
| `configs/convnext/` | ConvNeXt 基础配置（非MoE版本） | 可删除 |
| `configs/csl/` | CSL 算法配置 | 可删除 |
| `configs/g_reppoints/` | G-RepPoints 算法配置 | 可删除 |
| `configs/gliding_vertex/` | Gliding Vertex 算法配置 | 可删除 |
| `configs/gwd/` | GWD 算法配置 | 可删除 |
| `configs/kfiou/` | KFIoU 算法配置 | 可删除 |
| `configs/kld/` | KLD 算法配置 | 可删除 |
| `configs/lsknet/` | LSKNet 基础配置（非MoE版本） | 可删除 |
| `configs/oriented_rcnn/` | Oriented R-CNN 基础配置 | 可删除 |
| `configs/oriented_reppoints/` | Oriented RepPoints 配置 | 可删除 |
| `configs/r3det/` | R3Det 算法配置 | 可删除 |
| `configs/redet/` | ReDet 算法配置 | 可删除 |
| `configs/roi_trans/` | RoI Transformer 基础配置 | 可删除 |
| `configs/rotated_atss/` | Rotated ATSS 算法配置 | 可删除 |
| `configs/rotated_faster_rcnn/` | Rotated Faster R-CNN 配置 | 可删除 |
| `configs/rotated_fcos/` | Rotated FCOS 算法配置 | 可删除 |
| `configs/rotated_reppoints/` | Rotated RepPoints 配置 | 可删除 |
| `configs/rotated_retinanet/` | Rotated RetinaNet 基础配置 | 可删除 |
| `configs/s2anet/` | S2ANet 算法配置 | 可删除 |
| `configs/sasm_reppoints/` | SASM RepPoints 配置 | 可删除 |

**保留的配置文件夹：**

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `configs/ShipWake/` | ShipWake DualStream 配置 | **必须保留** |
| `configs/SM3Det/` | SM3Det 原始配置（参考用） | 建议保留 |
| `configs/_base_/` | 基础配置 | 需要筛选保留 |

### 3.2 configs/_base_/datasets/ 中的数据集配置

**可删除的数据集配置（与 ShipWake 无关）：**

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `configs/_base_/datasets/dota_.py` | DOTA 数据集配置 | 如不用可删除 |
| `configs/_base_/datasets/dotav1.py` | DOTA v1 配置 | 如不用可删除 |
| `configs/_base_/datasets/dronevehicle.py` | DroneVehicle 数据集 | 可删除 |
| `configs/_base_/datasets/fairv1.py` | FAIR v1 数据集 | 可删除 |
| `configs/_base_/datasets/hrsc.py` | HRSC 数据集 | 可删除 |
| `configs/_base_/datasets/hrsid.py` | HRSID 数据集 | 可删除 |
| `configs/_base_/datasets/sardet.py` | SARDet 数据集 | 可删除 |
| `configs/_base_/datasets/sardet50k.py` | SARDet-50K 数据集 | 可删除 |
| `configs/_base_/datasets/SOI_Det.py` | SOI-Det 多模态数据集 | 可删除 |
| `configs/_base_/datasets/ssdd.py` | SSDD 数据集 | 可删除 |

**保留的数据集配置：**

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `configs/_base_/datasets/SwimShip.py` | SwimShip 数据集（你的数据集） | **必须保留** |
| `configs/_base_/datasets/dota_.py` | 如需测试 DOTA | 可选保留 |
| `configs/_base_/datasets/dronevehicle.py` | 如需测试对比 | 可选保留 |

### 3.3 local_configs/ 中的配置文件（大量可删除）

local_configs 包含 74 个配置文件，**绝大多数可以删除**。

**建议保留的 local_configs（仅参考用）：**

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `local_configs/main_SM3Det.py` | SM3Det 主配置 | 建议保留 |
| `local_configs/main_SM3Det_convnext_t_orcnn_gfl_wo_moe.py` | SM3Det w/o MoE | 建议保留（消融对比） |

**可删除的 local_configs 类别：**

1. **单数据集配置（完全不需要）：**
   - `local_configs/dota_*.py` (约 10 个文件)
   - `local_configs/dronevehicle_*.py` (约 10 个文件)
   - `local_configs/sardet50k_*.py` (约 10 个文件)

2. **消融实验配置（可选删除）：**
   - `local_configs/ablation_dynlr_*.py` (约 7 个文件)
   - `local_configs/ablation_moe_*.py` (约 10 个文件)

3. **其他 SM3Det 变体（可选保留1-2个）：**
   - `local_configs/SM3Det_*.py` (约 20 个文件)

---

## 四、代码文件（可选精简）

### 4.1 Backbone 文件（可选删除）

如果你只使用 ConvNeXt_DualStream，其他 backbone 可删除：

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `mmrotate/models/backbones/adapter_modules.py` | Adapter 模块 | 可删除 |
| `mmrotate/models/backbones/flash_attention.py` | Flash Attention | 可删除 |
| `mmrotate/models/backbones/intern_vit.py` | InternViT | 可删除 |
| `mmrotate/models/backbones/lsk_moe.py` | LSKNet MoE | 可删除 |
| `mmrotate/models/backbones/lsknet.py` | LSKNet 基础 | 可删除 |
| `mmrotate/models/backbones/re_resnet.py` | Re-ResNet | 可删除 |
| `mmrotate/models/backbones/swin.py` | Swin Transformer | 可删除 |
| `mmrotate/models/backbones/swin_moe.py` | Swin MoE | 可删除 |
| `mmrotate/models/backbones/van.py` | VAN | 可删除 |
| `mmrotate/models/backbones/van_moe.py` | VAN MoE | 可删除 |
| `mmrotate/models/backbones/vit_adapter.py` | ViT Adapter | 可删除 |
| `mmrotate/models/backbones/convnext_moe_DA.py` | ConvNeXt with DA | 可删除 |

**必须保留的 backbone：**
- `mmrotate/models/backbones/convnext_moe.py` - SM3Det 基础
- `mmrotate/models/backbones/convnext_dualstream.py` - 你的核心创新
- `mmrotate/models/backbones/wake_residual_transform.py` - 你的核心创新
- `mmrotate/models/backbones/mutual_attention_mask.py` - 你的核心创新
- `mmrotate/models/backbones/__init__.py` - 模块导出

### 4.2 Detector 文件（可选删除）

如果你只使用 ShipWakeDualDetector，其他 detector 可删除：

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `mmrotate/models/detectors/gliding_vertex.py` | Gliding Vertex | 可删除 |
| `mmrotate/models/detectors/r3det.py` | R3Det | 可删除 |
| `mmrotate/models/detectors/redet.py` | ReDet | 可删除 |
| `mmrotate/models/detectors/roi_transformer.py` | RoI Transformer | 可删除 |
| `mmrotate/models/detectors/rotate_faster_rcnn.py` | Rotated Faster R-CNN | 可删除 |
| `mmrotate/models/detectors/rotated_fcos.py` | Rotated FCOS | 可删除 |
| `mmrotate/models/detectors/rotated_reppoints.py` | Rotated RepPoints | 可删除 |
| `mmrotate/models/detectors/s2anet.py` | S2ANet | 可删除 |
| `mmrotate/models/detectors/trisource_H1stage_R1stage_detector.py` | TriSource | 可删除 |
| `mmrotate/models/detectors/trisource_H1stage_R2stage_detector.py` | TriSource | 可删除 |
| `mmrotate/models/detectors/trisource_H2stage_R1stage_detector.py` | TriSource | 可删除 |
| `mmrotate/models/detectors/trisource_H2stage_R2stage_detector.py` | TriSource | 可删除 |

**必须保留的 detector：**
- `mmrotate/models/detectors/shipwake_dual_detector.py` - 你的核心创新
- `mmrotate/models/detectors/base.py` - 基础类
- `mmrotate/models/detectors/single_stage.py` - 基础类
- `mmrotate/models/detectors/two_stage.py` - 基础类
- `mmrotate/models/detectors/utils.py` - 工具函数
- `mmrotate/models/detectors/__init__.py` - 模块导出

### 4.3 Dataset 文件（可选删除）

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `mmrotate/datasets/dota_1_5.py` | DOTA 1.5 | 可删除 |
| `mmrotate/datasets/dronevehicle.py` | DroneVehicle | 可删除 |
| `mmrotate/datasets/fair.py` | FAIR | 可删除 |
| `mmrotate/datasets/hrsc.py` | HRSC | 可删除 |
| `mmrotate/datasets/sar.py` | SAR 基础 | 可删除 |
| `mmrotate/datasets/sardet.py` | SARDet | 可删除 |
| `mmrotate/datasets/sardet_dota_ifred.py` | 多模态数据集 | 可删除 |
| `mmrotate/datasets/sardet_hbb.py` | SARDet HBB | 可删除 |
| `mmrotate/datasets/sardet_hbb_trisource.py` | TriSource | 可删除 |

**必须保留的 dataset：**
- `mmrotate/datasets/dota.py` - DOTA 数据集（基础格式）
- `mmrotate/datasets/dota_.py` - DOTA 变体
- `mmrotate/datasets/__init__.py` - 模块导出
- `mmrotate/datasets/builder.py` - 构建器

---

## 五、其他文件

### 5.1 Docker 相关（可选删除）

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `docker/Dockerfile` | Docker 构建文件 | 如不用 Docker 可删除 |
| `docker/serve/` | Docker 服务配置 | 如不用 Docker 可删除 |

### 5.2 根目录文件（可选删除）

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `docker/` | Docker 配置 | 可选删除 |
| `inference.sh` | 推理脚本 | 可删除或保留 |
| `paper/` | PDF 论文 | 可保留作为参考 |

### 5.3 AI 相关文件夹

| 路径 | 说明 | 操作建议 |
|-----|------|---------|
| `AI input/` | AI 输入 | 检查内容后决定 |
| `AI output/` | AI 输出 | 检查内容后决定 |

---

## 六、推荐的最小文件集合

如果你想保留一个最小可用的仓库，建议保留以下文件：

### 核心代码（必须保留）

```
mmrotate/
├── models/
│   ├── backbones/
│   │   ├── __init__.py
│   │   ├── convnext_moe.py
│   │   ├── convnext_dualstream.py
│   │   ├── wake_residual_transform.py
│   │   └── mutual_attention_mask.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── single_stage.py
│   │   ├── two_stage.py
│   │   ├── shipwake_dual_detector.py
│   │   └── utils.py
│   └── ... (其他必要模块)
├── datasets/
│   ├── __init__.py
│   ├── builder.py
│   ├── dota.py
│   └── ... (你的数据集实现)
└── ... (其他核心模块)

configs/
├── ShipWake/           # 你的配置
│   ├── shipwake_dualstream_mini.py
│   ├── shipwake_dualstream_standard.py
│   ├── shipwake_dualstream_full.py
│   └── README.md
├── _base_/
│   ├── datasets/
│   │   ├── SwimShip.py     # 你的数据集配置
│   │   └── ... (可选保留 dota_.py 作为参考)
│   ├── schedules/
│   │   └── schedule_1x.py
│   └── default_runtime.py

tools/
├── train.py
├── test.py
├── validate_pipeline.py    # 你的验证脚本
└── ...

docs/
├── SM3Det.png              # 组会汇报参考
└── results.png             # 对比参考

# 根目录必要文件
README.md
setup.py
setup.cfg
requirements.txt
.pylintrc
.gitignore
LICENSE

# 你的文档
SHIPWAKE_SETUP_GUIDE.md
pipeline.md
CLEANUP_GUIDE.md
```

---

## 七、删除命令参考

### 7.1 批量删除命令（请谨慎执行）

```bash
# 1. 删除 BabelRS 项目
rm -rf BabelRS_configs/
rm -rf BabelRS_pretrain/

# 2. 删除与 BabelRS 相关的文档
rm docs/BabelRS.pdf
rm docs/BabelRS.png
rm docs/BabelRS_motivation.png
rm docs/meme.png
rm docs/vis.png

# 3. 删除不需要的算法配置（保留 ShipWake 和 SM3Det）
rm -rf configs/cfa/
rm -rf configs/convnext/
rm -rf configs/csl/
rm -rf configs/g_reppoints/
rm -rf configs/gliding_vertex/
rm -rf configs/gwd/
rm -rf configs/kfiou/
rm -rf configs/kld/
rm -rf configs/lsknet/
rm -rf configs/oriented_rcnn/
rm -rf configs/oriented_reppoints/
rm -rf configs/r3det/
rm -rf configs/redet/
rm -rf configs/roi_trans/
rm -rf configs/rotated_atss/
rm -rf configs/rotated_faster_rcnn/
rm -rf configs/rotated_fcos/
rm -rf configs/rotated_reppoints/
rm -rf configs/rotated_retinanet/
rm -rf configs/s2anet/
rm -rf configs/sasm_reppoints/

# 4. 删除不需要的数据集配置
rm configs/_base_/datasets/dronevehicle.py
rm configs/_base_/datasets/fairv1.py
rm configs/_base_/datasets/hrsc.py
rm configs/_base_/datasets/hrsid.py
rm configs/_base_/datasets/sardet.py
rm configs/_base_/datasets/sardet50k.py
rm configs/_base_/datasets/SOI_Det.py
rm configs/_base_/datasets/ssdd.py

# 5. 清理 local_configs（保留2-3个参考即可）
rm local_configs/dota_*.py
rm local_configs/dronevehicle_*.py
rm local_configs/sardet50k_*.py
rm local_configs/ablation_*.py

# 6. 删除不需要的 backbone（保留你的创新）
rm mmrotate/models/backbones/adapter_modules.py
rm mmrotate/models/backbones/flash_attention.py
rm mmrotate/models/backbones/intern_vit.py
rm mmrotate/models/backbones/lsk_moe.py
rm mmrotate/models/backbones/lsknet.py
rm mmrotate/models/backbones/re_resnet.py
rm mmrotate/models/backbones/swin.py
rm mmrotate/models/backbones/swin_moe.py
rm mmrotate/models/backbones/van.py
rm mmrotate/models/backbones/van_moe.py
rm mmrotate/models/backbones/vit_adapter.py
rm mmrotate/models/backbones/convnext_moe_DA.py

# 7. 删除不需要的 detector
rm mmrotate/models/detectors/gliding_vertex.py
rm mmrotate/models/detectors/r3det.py
rm mmrotate/models/detectors/redet.py
rm mmrotate/models/detectors/roi_transformer.py
rm mmrotate/models/detectors/rotate_faster_rcnn.py
rm mmrotate/models/detectors/rotated_fcos.py
rm mmrotate/models/detectors/rotated_reppoints.py
rm mmrotate/models/detectors/s2anet.py
rm mmrotate/models/detectors/trisource_*.py

# 8. 删除不需要的 dataset
rm mmrotate/datasets/dota_1_5.py
rm mmrotate/datasets/dronevehicle.py
rm mmrotate/datasets/fair.py
rm mmrotate/datasets/hrsc.py
rm mmrotate/datasets/sar.py
rm mmrotate/datasets/sardet.py
rm mmrotate/datasets/sardet_dota_ifred.py
rm mmrotate/datasets/sardet_hbb.py
rm mmrotate/datasets/sardet_hbb_trisource.py

# 9. 删除 Docker（如不需要）
rm -rf docker/
```

---

## 八、删除前检查清单

在执行删除操作前，请确认：

- [ ] 已备份重要数据
- [ ] 已创建新的 git branch 进行清理
- [ ] 已记录需要保留的文件列表
- [ ] 已测试最小文件集合可以正常运行

### 建议操作步骤

```bash
# 1. 创建清理分支
git checkout -b cleanup-repo

# 2. 按照本指南删除文件
# ... 执行删除命令 ...

# 3. 测试是否仍能运行
python tools/validate_pipeline.py \
    --config configs/ShipWake/shipwake_dualstream_mini.py \
    --data-root data/SwimShip/mini \
    --save-path work_dirs/test_cleanup

# 4. 提交清理后的代码
git add .
git commit -m "Cleanup: Remove unrelated SM3Det/BabelRS files"

# 5. 合并到主分支
git checkout main
git merge cleanup-repo
```

---

## 九、文件统计

| 类别 | 删除前文件数 | 删除后文件数 | 减少比例 |
|-----|------------|------------|---------|
| BabelRS 相关 | ~25 | 0 | 100% |
| 算法配置 | ~150 | ~10 | 93% |
| local_configs | ~74 | ~3 | 96% |
| Backbone 代码 | ~17 | ~6 | 65% |
| Detector 代码 | ~20 | ~7 | 65% |
| Dataset 代码 | ~13 | ~4 | 70% |
| **总计** | **~300+** | **~30** | **~90%** |

---

**注意：** 本指南仅提供建议，请根据你的实际需求调整保留/删除的文件列表。
