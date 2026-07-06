# YOLO26s-seg PointRend 续训练 — 实验结果总结

> **日期**：2026-07-07  
> **范围**：recipe200 蒸馏基线 → PointRend + boundary refine 续训练（ft60 / ft60-nobnd）  
> **关联文档**：[设计说明](yolo26s-seg-pointrend-refine-head-design.md) · [代码走读](yolo26s-seg-pointrend-beginner-guide.md) · [代码全局梳理](yolo26s-seg-pointrend-code-overview.md)

---

## 1. 实验链路

```text
Stage B  LVIS·COCO80 蒸馏 (yolo26x teacher)
    └─► recipe200  COCONut-B 蒸馏 (107/200 ep, OOM 停, mask mAP50-95=0.376)
            ├─► pointrend-ft60      PointRend + seg_point_boundary=True   (60/60 ✅)
            └─► pointrend-ft60-nobnd  PointRend + seg_point_boundary=False  (60/60 ✅)
```

| 运行名        | 目录                                                         | 初始化              | 唯一差异                   |
| ------------- | ------------------------------------------------------------ | ------------------- | -------------------------- |
| `recipe200`   | `runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200`    | Stage B best        | 蒸馏，无 `point_head`      |
| `ft60` (+bnd) | `runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft60`       | recipe200 `best.pt` | `seg_point_boundary=True`  |
| `ft60-nobnd`  | `runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft60-nobnd` | recipe200 `best.pt` | `seg_point_boundary=False` |

**共享配方**（两路续训练）：`yolo26s-seg-pointrend.yaml`，`pretrained=recipe200 best.pt`（非 resume），60 epoch，batch=84，MuSGD lr0=0.003，cos_lr，multi_scale=0.15，无蒸馏，`seg_point=0.5`，`seg_point_refine=True`，`seg_point_num=64`，`seg_point_o2o=0`，`overlap_mask=True`，seed=0。

入口脚本：`scripts/finetune_yolo26s_seg_pointrend_coconut_b.py`（`--no-boundary` 切换 nobnd）。

---

## 2. 训练期验证指标（Ultralytics 内置 val）

协议：`overlap_mask=True`，与训练同管线；**勿与独立 eval（§3）数值直接对比**。

| 运行           | peak ep | mask mAP50-95 | mask mAP50 | box mAP50-95 | val/seg @peak | fitness     |
| -------------- | ------- | ------------- | ---------- | ------------ | ------------- | ----------- |
| recipe200      | 107     | 0.3759        | 0.5713     | 0.4362       | 2.247         | —           |
| ft60 (+bnd)    | 52      | 0.3958        | 0.5993     | 0.4607       | 2.655         | 0.85668     |
| **ft60-nobnd** | **51**  | **0.3976**    | **0.6016** | **0.4611**   | 3.118         | **0.85872** |

**相对 recipe200**：两路续训练 mask mAP50-95 均 **+2.0pt** 左右，box 同步 **+2.5pt**。

### 2.1 Boundary 消融（同 epoch，训练 val）

| Epoch | +bnd   | nobnd  | Δ (+bnd − nobnd) |
| ----- | ------ | ------ | ---------------- |
| 10    | 0.3766 | 0.3784 | −0.0019          |
| 25    | 0.3811 | 0.3844 | −0.0032          |
| 40    | 0.3868 | 0.3897 | −0.0029          |
| 52    | 0.3958 | 0.3972 | −0.0014          |
| 60    | 0.3942 | 0.3947 | −0.0005          |

**结论**：GT-Sobel boundary 加权采样在训练 val 上**未带来收益**，全程略低于 uniform-in-bbox（nobnd），差约 0.1–0.3pt。

---

## 3. 独立双标尺 eval（standalone `YOLO.val`，batch=16）

数据来源：`runs/segment/eval_compare_recipe200_vs_official/summary_*.json`（2026-07-03 官方/recipe200；2026-07-05 ft60）。

| 标尺                             | recipe200 | ft60 (+bnd) | 官方 yolo26s-seg | ft60 vs 官方  |
| -------------------------------- | --------- | ----------- | ---------------- | ------------- |
| **COCONut v2 val** mask mAP50-95 | 0.3421    | **0.3810**  | 0.3495           | **+3.2pt** ✅ |
| **COCO val2017** mask mAP50-95   | 0.3290    | **0.3588**  | 0.3859           | **−2.7pt** ❌ |

- COCONut：ft60 大幅超过官方 yolo26s-seg。
- COCO：仍低于官方，但较 recipe200 缺口从 **−5.7pt 收窄到 −2.7pt**。
- **ft60-nobnd 尚未跑独立双标尺**（待补 `eval_compare_recipe200_vs_official.py --model ...`）。

### 3.1 指标口径说明

| 协议                 | ft60 (+bnd) mask mAP50-95 | 说明                |
| -------------------- | ------------------------- | ------------------- |
| 训练 val peak (ep52) | 0.3958                    | 内置 validator      |
| 独立 eval            | 0.3810                    | `eval_compare` 脚本 |
| val_twice indirect   | 0.3810                    | 与独立 eval 一致    |

训练 peak 与独立 eval 差约 **1.5pt**，跨实验对比须固定协议。

---

## 4. val_twice：推理侧 PointRend 细分（ft60 best.pt）

脚本：`scripts/val_twice_pointrend.py`  
数据：COCONut v2 val，3 seeds，`seg_point_subdiv_k=3`  
结果：`runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft60/val_twice_best.json`

| 路径                                                   | mask mAP50-95   | mask mAP75 |
| ------------------------------------------------------ | --------------- | ---------- |
| **indirect**（`seg_point_refine_infer=False`，部署用） | **0.3810**      | 0.3972     |
| direct（`seg_point_refine_infer=True`）                | 0.3573 ± 0.0000 | 0.3634     |
| **Δ (direct − indirect)**                              | **−2.4pt**      | −3.4pt     |

box mAP50-95 两路径均为 0.4634（不变量 sanity 通过）。

**结论**：PointHeadMLP 的 +2pt 收益来自**训练侧正则化**；当前推理细分路径为**净负收益**。根因：训练（proto 160×160 损失空间 + GT-Sobel 偏置采样）与推理（letterbox 全分辨率 + pred-uncertainty ROI）点分布不一致。

**部署与批量测试必须**：`seg_point_refine_infer=False`（checkpoint 内默认已是 False）。

---

## 5. 当前最优模型（批量测试）

按**训练期 fitness + mask mAP50-95**：

```text
/home/genesis/Train/Code/ultralytics/runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft60-nobnd/weights/best.pt
```

| 属性                 | 值                                                      |
| -------------------- | ------------------------------------------------------- |
| peak                 | ep51，mask mAP50-95=**0.3976**，fitness=**0.85872**     |
| 结构                 | `yolo26s-seg-pointrend.yaml` + PointHeadMLP (hidden=64) |
| `seg_point_boundary` | **False**                                               |

**备选**：

```text
# +bnd 版（训练 val 低 0.2pt，双标尺已验收）
.../yolo26s-seg-coconut-b-v2-pointrend-ft60/weights/best.pt

# 蒸馏基线（无 point head）
.../yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt

# COCO 官方标尺
/home/genesis/Train/Code/ultralytics/yolo26s-seg.pt
```

### 5.1 批量验证命令

```bash
cd /home/genesis/Train/Code/ultralytics
CKPT=runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft60-nobnd/weights/best.pt

# 双标尺
python scripts/eval_compare_recipe200_vs_official.py \
  --device 0 --batch 16 --suffix _nobnd \
  --model pointrend-nobnd=$CKPT

# COCONut val
conda run -n yolo26-cu133 python -c "
from ultralytics import YOLO
YOLO('$CKPT').val(
    data='/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml',
    imgsz=640, batch=16, device=0, seg_point_refine_infer=False)
"

# val_twice（测推理细分，预期为负）
python scripts/val_twice_pointrend.py --ckpt $CKPT \
  --data /home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml \
  --device 0 --seeds 0 1 2
```

---

## 6. 实验组织评估

### 6.1 已完成

- recipe200 → PointRend 续训练（两路配对 ablation：boundary on/off）
- ft60 双标尺 + val_twice 验收
- 训练侧 bug 修复（蒸馏 ckpt 解包、point OOM、MuSGD Conv1d）+ 单测

### 6.2 缺口

| 优先级 | 缺口                              | 影响                                        |
| ------ | --------------------------------- | ------------------------------------------- |
| P1     | nobnd 未跑独立双标尺 / val_twice  | 最优 ckpt 缺独立 eval 支撑                  |
| P2     | G0–G6 消融未跑（COCONut-S，30ep） | +2pt 无法归因到 point/MLP/boundary/comp/bnd |
| P3     | 无统一实验 manifest               | run→ckpt→eval 靠目录名追溯                  |
| P4     | 推理细分 (I) 未对齐训练           | val_twice −2.4pt，当前废弃                  |
| P5     | COCO 仍低于官方                   | 需 response KD 或 COCO 混训（P3-3）         |

---

## 7. 结论与下一步

1. **PointRend 续训练有效**：相对 recipe200，mask **+2.0pt**（训练 val）；COCONut 独立 eval **+3.9pt** vs recipe200、**+3.2pt** vs 官方。
2. **boundary 采样无益**：nobnd 略优于 +bnd；建议默认 `seg_point_boundary=False`。
3. **当前推荐 ckpt**：`ft60-nobnd/weights/best.pt`（待 nobnd 双标尺确认）。
4. **推理**：禁用 `seg_point_refine_infer`；point head 仅作训练正则。
5. **下一步**：nobnd 双标尺验收 → G0/G4m 消融（COCONut-S）→ COCO 缺口（response KD）。

---

_维护：新 run 完成后更新 §2–§5 表格，并将 eval JSON 路径记入本节或 `runs/segment/eval_compare_recipe200_vs_official/`。_
