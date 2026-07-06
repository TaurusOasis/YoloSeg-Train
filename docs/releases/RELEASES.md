# YOLO26s-seg 训练阶段 Release 与权重下载

> 仓库：[TaurusOasis/YoloSeg-Train](https://github.com/TaurusOasis/YoloSeg-Train)  
> 曲线数据来源：各 run 的 `results.csv`（与 SwanLab 记录的 val 指标列一致；recipe200 未保留 swanlab/ 目录，但指标同源）。

## 训练链路

```
A  LVIS 普通训练 (1203类)
    ↓ best.pt
B  LVIS→COCO80 蒸馏 (yolo26x teacher)
    ↓ best.pt
C  COCONut-B v1 蒸馏 (100 epoch)
    ↓ best.pt
D  COCONut-B v2 Recipe200 (107/200 ep, 中断)
    ↓ best.pt
E  PointRend Finetune (60 ep 目标, 进行中)
```

## Release 一览

| GitHub Release                                                                                                         | 阶段                   | Best epoch | Mask mAP50-95 | 权重文件                                   |
| ---------------------------------------------------------------------------------------------------------------------- | ---------------------- | ---------- | ------------- | ------------------------------------------ |
| [`stage-a-lvis-pretrain`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-a-lvis-pretrain)             | A · LVIS 预训练        | 100        | 0.071         | `yolo26s-seg-lvis-b48-best.pt`             |
| [`stage-b-lvis-coco80-distill`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-b-lvis-coco80-distill) | B · LVIS→COCO80 蒸馏   | 100        | 0.315         | `yolo26s-seg-lvis-coco80-distill-best.pt`  |
| [`stage-c-coconut-v1-distill`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-c-coconut-v1-distill)   | C · COCONut-B v1 蒸馏  | 99         | 0.354         | `yolo26s-seg-coconut-v1-distill-best.pt`   |
| [`stage-d-recipe200-v2`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-d-recipe200-v2)               | D · v2 Recipe200       | 107        | **0.376**     | `yolo26s-seg-coconut-v2-recipe200-best.pt` |
| [`stage-e-pointrend-ft60`](https://github.com/TaurusOasis/YoloSeg-Train/releases/tag/stage-e-pointrend-ft60)           | E · PointRend finetune | 12\*       | **0.377**     | `yolo26s-seg-pointrend-ft60-best.pt`       |

\* Stage E 为 **进行中 run 的 interim release**（ep12/60 时导出）；完整 60 epoch 结束后请用新 tag 覆盖或追加 `stage-e-pointrend-ft60-v2`。

> **COCONut-B v2 val 标尺**（5000 图）从 Stage D 起可直接对比；Stage A/B 为 LVIS 标尺，数值不可与 C/D/E 横比。

## 下载方式

### 1. GitHub Release（推荐）

```bash
# 示例：下载 Stage D recipe200 best
gh release download stage-d-recipe200-v2 \
  --repo TaurusOasis/YoloSeg-Train \
  --pattern '*.pt' \
  --dir ./weights

# 或浏览器打开 Release 页面 → Assets → 点击 .pt 下载
```

### 2. 本地 runs 目录（训练机）

权重默认在 `runs/segment/<run-name>/weights/best.pt`（已被 `.gitignore` 排除，不入 git）：

| 阶段 | 本地路径                                                                                      |
| ---- | --------------------------------------------------------------------------------------------- |
| A    | `runs/segment/yolo26s-seg-lvis-b48-bf16-swanlab/weights/best.pt`                              |
| B    | `runs/segment/yolo26s-seg-lvis-coco80-distill-x-teacher-b80-2gpu/weights/best.pt`             |
| C    | `runs/segment/yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu/weights/best.pt` |
| D    | `runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt`                     |
| E    | `runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft60/weights/best.pt`                        |

### 3. 加载权重继续训练 / finetune

```python
from ultralytics import YOLO

# Stage D → PointRend finetune（Stage E 脚本同款）
model = YOLO("yolo26s-seg-pointrend.yaml")
model.train(
    pretrained="path/to/yolo26s-seg-coconut-v2-recipe200-best.pt",
    data="/path/to/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml",
    resume=False,  # 结构变化时必须 finetune，不能 resume
    ...
)
```

或使用已发布 Release 权重：

```bash
gh release download stage-d-recipe200-v2 -R TaurusOasis/YoloSeg-Train -p '*.pt'
python scripts/finetune_yolo26s_seg_pointrend_coconut_b.py \
  --pretrained ./yolo26s-seg-coconut-v2-recipe200-best.pt
```

## SwanLab 训练曲线

各阶段 val / train 曲线 PNG 见 [`curves/`](curves/)：

| 曲线图                                                                      | 说明                            |
| --------------------------------------------------------------------------- | ------------------------------- |
| [`pipeline-overview.png`](curves/pipeline-overview.png)                     | 五阶段 best Mask mAP50-95 对比  |
| [`stage-a-lvis-pretrain.png`](curves/stage-a-lvis-pretrain.png)             | Stage A（SwanLab run）          |
| [`stage-b-lvis-coco80-distill.png`](curves/stage-b-lvis-coco80-distill.png) | Stage B（SwanLab run）          |
| [`stage-c-coconut-v1-distill.png`](curves/stage-c-coconut-v1-distill.png)   | Stage C（SwanLab run）          |
| [`stage-d-recipe200-v2.png`](curves/stage-d-recipe200-v2.png)               | Stage D（results.csv；107 ep）  |
| [`stage-e-pointrend-ft60.png`](curves/stage-e-pointrend-ft60.png)           | Stage E（SwanLab run；interim） |

本地 SwanLab Dashboard（Stage E 示例）：

```bash
swanlab watch runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft60/swanlab
```

重新导出曲线：

```bash
python scripts/export_release_curves.py
```

### 发布 GitHub Release（上传 best.pt）

文档与曲线已入 git；**权重 `.pt` 需通过 Release 附件分发**（不入 git）。

```bash
# 需 gh token 具备 repo / releases 写权限
bash scripts/publish_github_releases.sh

# 预览命令不实际上传
bash scripts/publish_github_releases.sh --dry-run
```

若 `gh release create` 报 `HTTP 403`，请在 GitHub → Settings → Developer settings → PAT 中勾选 **Contents: Read and write** 后重新 `gh auth login`。

## 各阶段 Release 详情

### Stage A — `stage-a-lvis-pretrain`

- **Run**: `yolo26s-seg-lvis-b48-bf16-swanlab`
- **数据**: LVIS 1203 类，BF16 AMP
- **Best**: ep100 · mask mAP50-95 **0.071**（LVIS 标尺）
- **下游**: 初始化 Stage B

### Stage B — `stage-b-lvis-coco80-distill`

- **Run**: `yolo26s-seg-lvis-coco80-distill-x-teacher-b80-2gpu`
- **Teacher**: `yolo26x-seg.pt`
- **Best**: ep100 · mask mAP50-95 **0.315**
- **下游**: 初始化 Stage C

### Stage C — `stage-c-coconut-v1-distill`

- **Run**: `yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu`
- **数据**: COCONut-B **v1** 标签，100 epoch
- **Best**: ep99 · mask mAP50-95 **0.354**（v1 val）；v2 复评约 **0.373**
- **下游**: 初始化 Stage D recipe200

### Stage D — `stage-d-recipe200-v2` ⭐ 当前主线 best（dense）

- **Run**: `yolo26s-seg-coconut-b-v2-distill-recipe200`
- **数据**: COCONut-B **v2** 标签，`dis=3.0`, `multi_scale=0.25`, batch=90
- **Best**: ep107 · mask mAP50-95 **0.376** / mask50 **0.571** / box50-95 **0.436**
- **状态**: 107/200 中断（ep108 OOM）；详见 [`yolo26s-seg-recipe200-stage-summary.md`](../yolo26s-seg-recipe200-stage-summary.md)
- **下游**: 初始化 Stage E PointRend finetune

### Stage E — `stage-e-pointrend-ft60` ⭐ PointRend interim

- **Run**: `yolo26s-seg-coconut-b-v2-pointrend-ft60`
- **模型**: `yolo26s-seg-pointrend.yaml`（+ PointHeadMLP）
- **配置**: `seg_point=0.5`, `seg_point_refine=True`, `seg_point_boundary=True`, 无蒸馏
- **Best (interim)**: ep12 · mask mAP50-95 **0.377**（略超 Stage D +0.001）
- **状态**: 60 epoch 训练进行中；本 Release 为 **快照**，非最终版
- **脚本**: `scripts/finetune_yolo26s_seg_pointrend_coconut_b.py`

## 机器可读 manifest

[`manifest.json`](manifest.json) 含各阶段 best 指标、曲线路径、本地 `best.pt` 相对路径，供 CI / 脚本使用。
