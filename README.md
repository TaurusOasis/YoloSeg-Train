# YoloSeg-Train

基于 [Ultralytics 8.4.82](https://github.com/ultralytics/ultralytics) 工作树的 **YOLO26s-seg 实例分割** 训练仓库：LVIS 预训练 → 特征蒸馏 → COCONut-B 域适配 → v2 标签长训 → PointRend 边界 refine。

**GitHub**：[TaurusOasis/YoloSeg-Train](https://github.com/TaurusOasis/YoloSeg-Train)

---

## 整体目标

| 层级 | 目标 | 当前状态 |
|------|------|----------|
| **模型** | 在 **11.5M 参数量** 的 YOLO26s-seg 上，逼近 **62M** yolo26x-seg teacher 的分割质量 | v2 val mask mAP50-95：0.377（Stage E interim） vs teacher 0.404 |
| **数据** | 利用 COCONut panoptic 重标注（241k 训练图）提升 mask 边界与实例完整性 | v2 标签（孔洞保留 + 每实例一行）已落地 |
| **方法** | 多阶段 **特征 + proto 蒸馏**；长训 recipe200；**PointRend-style point-head** 细化 mask | 蒸馏链 A→E 完成/进行中；PointRend 代码与 ft60 实验已上线 |
| **交付** | 各阶段 `best.pt` + SwanLab 曲线 + 可复现脚本 | 见 [Release 指南](docs/releases/RELEASES.md) |
| **标尺** | 主标尺：**COCONut-B v2 val**（5000 图）；辅标尺：**COCO val2017** 官方 seg | 跨阶段 LVIS 标尺不可与 COCONut 横比 |

---

## 训练流水线

```
Stage A   LVIS 1203 类普通训练                    mask50-95=0.071 (LVIS 标尺)
    │
    ▼ best.pt
Stage B   LVIS→COCO80 子集蒸馏 (yolo26x teacher)   mask50-95=0.315
    │
    ▼ best.pt
Stage C   COCONut-B v1 蒸馏 (100 ep)               mask50-95=0.354 (v1 val)
    │
    ▼ best.pt + v2 标签重建 (P2-1)
Stage D   COCONut-B v2 Recipe200 (107/200 ep)     mask50-95=0.376 ⭐ 主线 dense best
    │
    ▼ best.pt + PointHeadMLP (finetune, 非 resume)
Stage E   PointRend Finetune (60 ep 目标)         mask50-95=0.377 ⭐ interim (ep12/60)
```

训练曲线总览：[`docs/releases/curves/pipeline-overview.png`](docs/releases/curves/pipeline-overview.png)

---

## 数据集索引

本地根目录默认：`/home/genesis/Train/Dataset/`（不入 git，训练机本地路径）。

### 原始数据

| 数据集 | 路径 | 规模 | 说明 |
|--------|------|------|------|
| **LVIS** | `LVIS_yolo_seg/` | 1203 类 | Ultralytics YOLO seg 格式；Stage A 全量训练 |
| **COCO 2017 图像** | `coco2017/` | train2017 + unlabeled2017 + val2017 | COCONut 图像来源（软链，不复制 JPG） |
| **COCONut panoptic** | `coconut/` | S: 118k / **B: 241k** train + 5k val | RGB PNG mask + JSON；segment id = R+256G+256²B |
| **COCO val2017 官方 seg** | `coco_val2017_yolo_seg/` | 5000 图 / 36335 实例 | 双标尺验收用；`convert_coco(use_segments=True)` |

COCONut 原始 split 见 [`docs/coconut-yolo26s-seg-distill.md`](docs/coconut-yolo26s-seg-distill.md#raw-dataset-layout)。

### 转换后 YOLO 训练集

| YAML | 生成脚本 | 类别 | Train 图 | Val 图 | 用途 |
|------|----------|------|---------:|-------:|------|
| `LVIS_yolo_seg/lvis-seg.yaml` | （上游/LVIS 转换） | **1203** | — | — | Stage A |
| `LVIS_coco80_yolo_seg/lvis-coco80-seg.yaml` | [`build_lvis_coco80_seg_subset.py`](scripts/build_lvis_coco80_seg_subset.py) | **78** | — | — | Stage B（缺 hot dog / potted plant） |
| `COCONut_b_yolo_seg/coconut-b-seg.yaml` | [`build_coconut_yolo_seg.py`](scripts/build_coconut_yolo_seg.py) | **80** | 241602 | 5000 | Stage C（**v1 标签**，历史对照） |
| `COCONut_b_yolo_seg_v2/coconut-b-seg.yaml` | 同上 + **F11 修复** | **80** | 241602 | 5000 | **Stage D/E 主训练集** |
| `COCONut_yolo_seg/coconut-s-seg.yaml` | 同上 | **80** | 118200 | 5000 | 小规模消融可选 |

### v1 → v2 标签差异（P2-1，2026-07-03）

| 指标 | v1 | v2 | 说明 |
|------|----|----|------|
| train 标签行数 | 2,365,072 | **1,797,818** (−24%) | 断裂实例合并为每实例一行 |
| val 实例数 | 57,220 | **45,003** (−21%) | 消除拆分 GT 噪声 |
| 轮廓提取 | `RETR_EXTERNAL`，每轮廓一行 | `RETR_CCOMP` + `merge_multi_segment` | 保留孔洞；带孔 IoU 0.826→0.949 |
| 主标尺 | v1 val 已废弃对照 | **v2 val 为准** | §3.1.2 推翻「student 追平 teacher」伪结论 |

---

## 训练数据处理方式

### COCONut panoptic → YOLO seg（[`build_coconut_yolo_seg.py`](scripts/build_coconut_yolo_seg.py)）

```
coconut JSON + RGB panoptic PNG
  → 解码 segment id (R+256G+256²B)
  → 过滤 isthing=1 且 iscrowd=0（80 COCO thing 类）
  → OpenCV 轮廓提取 + approxPolyDP(ε=0.001×周长)
  → 归一化多边形 → labels/*.txt（class x1 y1 x2 y2 ...）
  → images/ 对 coco2017 做逐文件软链（非整目录软链，避免 cache 污染）
```

空图保留空 label，保证 manifest 完整。

重建 v2：

```bash
python scripts/build_coconut_yolo_seg.py \
  --out-root /home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2 \
  --train-split coconut_b --workers 16 --overwrite
```

### LVIS → COCO80 子集（[`build_lvis_coco80_seg_subset.py`](scripts/build_lvis_coco80_seg_subset.py)）

- 复用 LVIS 原图软链，重写 label，类名 dense 重映射 0..77
- teacher `yolo26x-seg.pt` 为 COCO **80 类**；student 78 类时由 `teacher_class_indices` 按类名对齐

### 训练时 mask 栅格化

```
YOLO polygon labels
  → polygon2mask（训练 augment 后）
  → mask_ratio=4 下采样栅格
  → proto × coeff 标准 YOLO seg loss
  →（Stage E）+ PointRend 点采样 / 可选 boundary / completeness 子损失
```

---

## 各阶段中间结果

> **标尺说明**：A/B 为 LVIS 标尺；C 为 v1 val；**D/E 为 COCONut-B v2 val（5000 图），可直接对比**。机器可读：[`docs/releases/manifest.json`](docs/releases/manifest.json)

| 阶段 | Run 目录 | Best ep | Mask mAP50-95 | Mask mAP50 | Box mAP50-95 | 权重 (~MB) | Release Tag |
|------|----------|--------:|----------------:|-----------:|-------------:|-----------:|-------------|
| **A** LVIS 预训练 | `yolo26s-seg-lvis-b48-bf16-swanlab` | 100 | 0.071 | 0.109 | 0.086 | 24 | [`stage-a-lvis-pretrain`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-a-lvis-pretrain) |
| **B** LVIS→COCO80 蒸馏 | `yolo26s-seg-lvis-coco80-distill-x-teacher-b80-2gpu` | 100 | 0.315 | 0.450 | 0.343 | 22 | [`stage-b-lvis-coco80-distill`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-b-lvis-coco80-distill) |
| **C** COCONut v1 蒸馏 | `yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu` | 99 | 0.354 | 0.530 | 0.398 | 22 | [`stage-c-coconut-v1-distill`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-c-coconut-v1-distill) |
| **D** v2 Recipe200 | `yolo26s-seg-coconut-b-v2-distill-recipe200` | 107 | **0.376** | **0.571** | **0.436** | 78 | [`stage-d-recipe200-v2`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-d-recipe200-v2) |
| **E** PointRend ft60 | `yolo26s-seg-coconut-b-v2-pointrend-ft60` | 12* | **0.377** | **0.573** | **0.438** | 67 | [`stage-e-pointrend-ft60`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-e-pointrend-ft60) |

\* Stage E 为 **进行中 interim**（导出时 13/60 ep）；完整 60 ep 后需刷新 Release。

### 双标尺对照（节选）

| 模型 | COCONut v2 val mask50-95 | COCO val2017 mask50-95 |
|------|-------------------------:|-----------------------:|
| Teacher yolo26x-seg | 0.404 | 0.448 |
| Stage C best (v2 复评) | 0.373 | 0.356 |
| **Stage D recipe200 best** | **0.376** | 待复测 |
| 官方 yolo26s-seg | 0.350 | 0.386 |

详见 [`docs/yolo26s-seg-distill-training-flow.md`](docs/yolo26s-seg-distill-training-flow.md) §3.1–§3.1.2。

### SwanLab 曲线

| 阶段 | PNG |
|------|-----|
| 总览 | [`docs/releases/curves/pipeline-overview.png`](docs/releases/curves/pipeline-overview.png) |
| A–E | [`docs/releases/curves/`](docs/releases/curves/) |

```bash
python scripts/export_release_curves.py   # 从各 run results.csv 重新导出
swanlab watch runs/segment/<run-name>/swanlab
```

---

## 权重下载

`.pt` 权重不入 git（`runs/` 被 ignore）。推荐通过 **GitHub Release** 下载：

```bash
gh release download stage-d-recipe200-v2 -R TaurusOasis/YoloSeg-Train -p '*.pt' -d ./weights
```

本地训练机路径：`runs/segment/<run-name>/weights/best.pt`。完整说明见 **[`docs/releases/RELEASES.md`](docs/releases/RELEASES.md)**。

发布 Release（需 PAT `Contents: Read and write`）：

```bash
bash scripts/publish_github_releases.sh
```

---

## 快速开始

### 环境

```bash
pip install -e .   # 或 conda env yolo26-cu133
```

### 阶段 C — COCONut 蒸馏

```bash
python scripts/train_yolo26s_seg_coconut_distill.py \
  --data /path/to/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml \
  --teacher yolo26x-seg.pt \
  --student runs/segment/.../stage-b-best.pt \
  --epochs 100 --batch 150 --device 0,1,2 \
  --dis 3.0 --dis-proto 1.0 --swanlab-watch
```

### 阶段 E — PointRend finetune（从 recipe200 best）

```bash
python scripts/finetune_yolo26s_seg_pointrend_coconut_b.py \
  --pretrained runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt
```

默认：`yolo26s-seg-pointrend.yaml`，`seg_point=0.5`，无蒸馏，batch=84，3×GPU DDP。

---

## 代码改动索引

本仓库在 Ultralytics 8.4.82 基础上新增/修改的核心模块：

| 模块 | 路径 | 职责 |
|------|------|------|
| 蒸馏包装 | [`ultralytics/nn/distill_model.py`](ultralytics/nn/distill_model.py) | FeatureHook、projector、dis_feat/dis_proto、类别对齐 |
| 训练循环 | [`ultralytics/engine/trainer.py`](ultralytics/engine/trainer.py) | 蒸馏 8 处侵入、resume 白名单、BF16 AMP |
| 分割损失 | [`ultralytics/utils/loss.py`](ultralytics/utils/loss.py) | seg_point / seg_bnd / seg_comp；E2E 分支 |
| Point 采样 | [`ultralytics/utils/mask_point_sampling.py`](ultralytics/utils/mask_point_sampling.py) | PointRend 训练点采样 |
| 推理 refine | [`ultralytics/utils/ops.py`](ultralytics/utils/ops.py) | `process_mask_pointrend` |
| PointHead | [`ultralytics/nn/modules/head.py`](ultralytics/nn/modules/head.py) | `PointHeadMLP` + `Segment26.point_hidden` |
| 模型 YAML | [`ultralytics/cfg/models/26/yolo26-seg-pointrend.yaml`](ultralytics/cfg/models/26/yolo26-seg-pointrend.yaml) | PointRend 结构定义 |
| SwanLab | [`ultralytics/utils/callbacks/swanlab.py`](ultralytics/utils/callbacks/swanlab.py) | 本地看板，`ULTRALYTICS_SWANLAB*` |

框架问题清单 F1–F21 与改进计划见主文档 §4–§7。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| **[`docs/yolo26s-seg-distill-training-flow.md`](docs/yolo26s-seg-distill-training-flow.md)** | **主文档**：代码结构、三/五阶段流程、F1–F21 框架问题、双标尺结论、改进路线 |
| [`docs/releases/RELEASES.md`](docs/releases/RELEASES.md) | 各阶段 Release、best.pt 下载、曲线、manifest |
| [`docs/coconut-yolo26s-seg-distill.md`](docs/coconut-yolo26s-seg-distill.md) | COCONut 原始布局、转换规则、Stage C 命令 |
| [`docs/yolo26s-seg-recipe200-stage-summary.md`](docs/yolo26s-seg-recipe200-stage-summary.md) | Stage D recipe200 时间线、OOM 复盘、指标里程碑 |
| [`docs/yolo26s-seg-coconut-b-v2-stage-summary.md`](docs/yolo26s-seg-coconut-b-v2-stage-summary.md) | v2 蒸馏 + PointRend 实施计划 |
| [`docs/yolo26s-seg-pointrend-refine-head-design.md`](docs/yolo26s-seg-pointrend-refine-head-design.md) | PointHeadMLP 工程设计（训练 T / 推理 I） |
| [`docs/yolo26s-seg-pointrend-beginner-guide.md`](docs/yolo26s-seg-pointrend-beginner-guide.md) | PointRend 新手入门、续训练排障 §28 |
| [`docs/yolo26-seg-training-review.md`](docs/yolo26-seg-training-review.md) | Loss/链路审查 1.1–1.9 |
| [`docs/yolo26s-seg-pointrend-training-tutorial.ipynb`](docs/yolo26s-seg-pointrend-training-tutorial.ipynb) | 交互式教程 notebook |

### 脚本索引

| 脚本 | 用途 |
|------|------|
| [`scripts/train_yolo26s_seg_lvis_coco80_distill.py`](scripts/train_yolo26s_seg_lvis_coco80_distill.py) | Stage B |
| [`scripts/train_yolo26s_seg_coconut_distill.py`](scripts/train_yolo26s_seg_coconut_distill.py) | Stage C / recipe200（resume、SwanLab） |
| [`scripts/finetune_yolo26s_seg_pointrend_coconut_b.py`](scripts/finetune_yolo26s_seg_pointrend_coconut_b.py) | Stage E PointRend |
| [`scripts/build_coconut_yolo_seg.py`](scripts/build_coconut_yolo_seg.py) | COCONut → YOLO seg |
| [`scripts/build_lvis_coco80_seg_subset.py`](scripts/build_lvis_coco80_seg_subset.py) | LVIS → COCO80 子集 |
| [`scripts/export_release_curves.py`](scripts/export_release_curves.py) | 导出 Release 曲线 PNG |
| [`scripts/publish_github_releases.sh`](scripts/publish_github_releases.sh) | 一键发布 GitHub Release |
| [`scripts/eval_compare_recipe200_vs_official.py`](scripts/eval_compare_recipe200_vs_official.py) | recipe200 vs 官方 s 对比评测 |
| [`scripts/ablate_seg_loss_coconut_s.py`](scripts/ablate_seg_loss_coconut_s.py) | mask 子损失消融（COCONut-S） |

---

## 上游 Ultralytics

本仓库 fork 自 Ultralytics YOLO，保留上游 CLI/API 与模型 zoo。通用安装、预测、导出文档见 [Ultralytics Docs](https://docs.ultralytics.com/)。

```bash
yolo predict model=yolo26s-seg.pt source=bus.jpg
yolo val segment model=best.pt data=coconut-b-seg.yaml
```

## License

AGPL-3.0（与上游 Ultralytics 一致）。商业使用见 [Ultralytics Licensing](https://www.ultralytics.com/license)。
