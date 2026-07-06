# YOLO26s-seg COCONut-B v2 蒸馏阶段性总结

> 日期：2026-07-04  
> 目标：归档 `yolo26s-seg-coconut-b-v2-distill-recipe200` 当前训练进展、稳定性问题、精度位置，并给出下一阶段加入 PointRend-style point loss / boundary mask loss 的实施计划。

---

## 1. 当前结论

当前 recipe200 训练已经从早期平台区继续提升到新的最好点：

| 指标          | 当前 best epoch 107 | 旧阶段 C v2 复评 | teacher yolo26x-seg v2 |
| ------------- | ------------------: | ---------------: | ---------------------: |
| Box mAP50-95  |         **0.43624** |            0.432 |                  0.513 |
| Mask mAP50-95 |         **0.37592** |            0.373 |                  0.404 |
| Box mAP50     |         **0.58887** |                - |                      - |
| Mask mAP50    |         **0.57125** |                - |                      - |

阶段性判断：

- recipe200 已小幅超过旧阶段 C 在 COCONut-B v2 val 上的结果，说明修正标签 + 更长训练 + 增强配方是正收益。
- 当前仍明显落后 teacher，尤其 box mAP50-95 仍差约 `0.077`，mask mAP50-95 差约 `0.028`。
- 曲线到 epoch 107 仍在上升，没有自然收敛迹象。
- 当前 run 在 epoch 108 中途停止，原因不是正常结束，而是 non-finite recovery 后 OOM。
- 当前 best 还没有重新跑 COCO official val2017 双标尺评估，不能直接推断 COCO 标尺是否同步提升。

---

## 2. 训练配置与数据

Run:

```text
runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200
```

数据：

```text
/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml
train: 241602 images
val: 5000 images / 45003 instances
classes: COCO 80
```

核心配置来自 `args.yaml` / checkpoint：

| 参数                  |              当前值 |
| --------------------- | ------------------: |
| epochs                |                 200 |
| completed epochs      |                 107 |
| batch                 |                  90 |
| device                |               0,1,2 |
| imgsz                 |                 640 |
| multi_scale           |                0.25 |
| mosaic                |                 1.0 |
| close_mosaic          |                  20 |
| copy_paste            |                 0.4 |
| mixup                 |                 0.1 |
| optimizer             |               MuSGD |
| lr0 / lrf             |         0.01 / 0.01 |
| cos_lr                |                True |
| amp                   | True, BF16 autocast |
| mask_ratio            |                   4 |
| teacher               |    `yolo26x-seg.pt` |
| dis / dis_proto       |           3.0 / 1.0 |
| distill_warmup_epochs |                 3.0 |
| distill_loss_clip     |                10.0 |

当前训练仍是标准 YOLO segment 监督：

```text
COCONut panoptic RGB PNG + JSON
  -> build_coconut_yolo_seg.py
  -> YOLO .txt polygon labels
  -> polygon2mask
  -> mask_ratio=4 raster mask
  -> proto/coeff mask loss
```

即使 COCONut 源数据是像素级 panoptic mask，当前 loss 实际看到的是多边形重栅格化后的低分辨率监督。对于 640 输入，`mask_ratio=4` 通常是 160x160；开启 `multi_scale=0.25` 后，训练时约在 120x120 到 200x200 之间浮动。

---

## 3. 训练进展曲线

关键 epoch：

| epoch | Box mAP50-95 | Mask mAP50-95 |   Box mAP50 |  Mask mAP50 |  Box recall | Mask recall |    dis_feat |   dis_proto |      lr/pg0 |
| ----: | -----------: | ------------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
|    13 |      0.40790 |       0.35382 |     0.55528 |     0.53909 |     0.51560 |     0.50390 |     0.73780 |     0.44071 |     0.02974 |
|    16 |      0.40723 |       0.35238 |     0.55442 |     0.53814 |     0.51397 |     0.49726 |     0.73409 |     0.43623 |     0.02959 |
|    50 |      0.41335 |       0.35639 |     0.56142 |     0.54416 |     0.52059 |     0.50645 |     0.71189 |     0.42116 |     0.02581 |
|    75 |      0.42218 |       0.36423 |     0.57301 |     0.55463 |     0.52727 |     0.51797 |     0.68840 |     0.40291 |     0.02105 |
|   100 |      0.43217 |       0.37195 |     0.58449 |     0.56574 |     0.53645 |     0.52385 |     0.66087 |     0.38271 |     0.01538 |
|   107 |  **0.43624** |   **0.37592** | **0.58887** | **0.57125** | **0.54063** | **0.53200** | **0.65143** | **0.37632** | **0.01375** |

解读：

- epoch 17 到 107 期间，Box mAP50-95 从约 0.408 提升到 0.436，增幅约 `+0.028`。
- Mask mAP50-95 从约 0.352 提升到 0.376，增幅约 `+0.024`。
- mAP50 与 mAP50-95 同步上升，说明不是单纯阈值排序改善，高 IoU 区间也在提升。
- recall 明显上升，特别是 mask recall 从约 0.50 到 0.532，说明更长训练和增强对漏检/漏分有帮助。
- `dis_feat` / `dis_proto` 持续下降，teacher feature/proto 对齐仍在推进。
- 学习率仍不低，曲线未表现出平台期；继续训练仍有潜在收益。

---

## 4. 停止原因与稳定性分析

当前训练不是正常跑满 200 epoch，而是在 epoch 108 中段停止。

最后状态：

```text
last complete epoch in results.csv: 107
last.pt: 2026-07-04 11:43
best.pt: 2026-07-04 11:43
checkpoint epoch field: 106
next resume display epoch: 108/200
```

失败日志：

```text
Non-finite loss at epoch 108, batch 1423/2685
Recovering from last.pt
CUDA out of memory. Tried to allocate 248 MiB.
GPU0 free only 84 MiB.
```

本次 non-finite 日志里 rank0 打印的 `loss_items` 本身是有限值：

```text
[1.1886, 2.0008, 1.4261, 0.00678, 0.2057, 0.6093, 0.4350]
```

因此更可能是 DDP 其它 rank 出现非有限值，经过 all-reduce 后触发全局恢复。随后恢复路径在 student proto forward 阶段 OOM。

对应样本原始标签检查：

| sample                       | label rows | coord range            | format |
| ---------------------------- | ---------: | ---------------------- | ------ |
| `train2017/000000383462`     |          3 | `[0.051562, 0.954688]` | OK     |
| `train2017/000000562232`     |         13 | `[0.000000, 0.998437]` | OK     |
| `unlabeled2017/000000100365` |          3 | `[0.093750, 0.998437]` | OK     |

判断：

- 不是明显的原始标签坏行。
- b90 相比 b100 已稳定很多，成功从 epoch 17 跑到 107。
- 但 `batch=90 + multi_scale=0.25 + yolo26x teacher + proto distill` 仍然在高分辨率 batch / recovery 路径上贴近 32GB 显存上限。
- DDP 下当前 OOM 自动降 batch 逻辑不会生效；一旦恢复路径 OOM，整次 torchrun 直接失败。

下一次若继续 recipe200 基线，建议不要原样 b90：

```text
优先：batch=80 或 84，保持 multi_scale=0.25
备选：batch=90，multi_scale 降到 0.125~0.20
更稳：修复 non-finite recovery，在 DDP 下先清显存/重建 dataloader 或跳过异常 batch
```

---

## 5. 与既有基线的关系

已知同标尺结果：

| 模型/阶段                | Val 标尺         | Box mAP50-95 | Mask mAP50-95 |
| ------------------------ | ---------------- | -----------: | ------------: |
| teacher `yolo26x-seg.pt` | COCONut-B v2 val |        0.513 |         0.404 |
| 旧 student C best        | COCONut-B v2 val |        0.432 |         0.373 |
| 当前 recipe200 best      | COCONut-B v2 val |    **0.436** |     **0.376** |

COCO official val2017 已知旧结论：

| 模型/阶段                 | COCO official val2017 Box | COCO official val2017 Mask |
| ------------------------- | ------------------------: | -------------------------: |
| teacher `yolo26x-seg.pt`  |                     0.558 |                      0.448 |
| official `yolo26s-seg.pt` |                     0.468 |                      0.386 |
| 旧 student C best         |                     0.415 |                      0.356 |

当前 recipe200 best 尚未重新跑 COCO official val2017。必须补这一步才能判断 recipe200 是否只提升 COCONut v2 域内，还是也改善 COCO 官方标尺。

建议下一步先做两组评估：

```text
1. COCONut-B v2 val: 当前 best.pt vs teacher vs old C
2. COCO official val2017 YOLO-seg: 当前 best.pt vs official yolo26s-seg vs teacher
```

---

## 6. 当前蒸馏形态的局限

当前 recipe200 使用的是现有 KD：

```text
3 层 neck feature L2, teacher score weighted
proto MSE, one2many proto, teacher P3 score weighted
```

它没有显式包含：

- response-level cls logits KD；
- box DFL / decoded box KD；
- mask coefficient KD；
- final mask-logit KD；
- boundary-aware KD。

这解释了当前现象：

- COCONut v2 val 有持续收益；
- mask mAP 提升慢于 teacher gap；
- box gap 仍大，说明 teacher 的分类/框响应知识没有充分转移。

因此“加 boundary/PointRend loss”主要是补 mask 质量，尤其高 IoU 和边界；它不能替代 response KD 对 box/recall 的补充。后续若目标是同时追 COCO 官方标尺，仍建议另起 P3-3 response KD。

---

## 7. Boundary / PointRend 下一阶段计划

### 7.1 目标

当前标签链路会把 COCONut 源 panoptic mask 压成 YOLO poly，再在训练中按 `mask_ratio=4` 栅格化。下一阶段的目标不是立刻改数据格式，而是在现有 YOLO-seg 框架内增加更关注边界和难点像素的 mask loss：

```text
dense BCE              负责整体区域
Focal-Tversky          补充完整性 / 减少漏分
Sobel boundary L2      补充边界位置和边缘形状
PointRend-style points 补充不确定点 / 边界难点
```

### 7.2 已有本地 WIP

当前工作区已经存在一组未提交的本地改动，应视为 WIP，尚未等同稳定上线：

| 文件                                          | 作用                                                           | 状态         |
| --------------------------------------------- | -------------------------------------------------------------- | ------------ |
| `ultralytics/utils/mask_point_sampling.py`    | PointRend-style `point_sample`、不确定点采样、point focal/dice | 已有         |
| `ultralytics/utils/mask_boundary_loss.py`     | Sobel magnitude + per-instance boundary L2                     | 已有         |
| `ultralytics/utils/mask_completeness_loss.py` | Focal-Tversky completeness loss                                | 已有         |
| `ultralytics/utils/loss.py`                   | `single_mask_loss()` 已接 `seg_comp/seg_bnd/seg_point`         | 已有本地修改 |
| `ultralytics/cfg/default.yaml`                | 新增 `seg_comp/seg_bnd/seg_point*`，默认 0                     | 已有本地修改 |
| `tests/test_mask_boundary_loss.py`            | Sobel/boundary helper 单测                                     | 已有         |
| `tests/test_mask_completeness_loss.py`        | Tversky helper 单测                                            | 已有         |
| `scripts/ablate_seg_loss_coconut_s.py`        | COCONut-S 30 epoch 消融脚本                                    | 已有         |

当前 recipe200 训练没有启用这些 loss。`args.yaml` 中没有 `seg_comp/seg_bnd/seg_point`，且默认值为 0；因此当前 best 可以作为“无 boundary/point loss”的基线。

### 7.3 设计边界

PointRend 分两种：

1. **loss-only PointRend-style sampling**  
   不改模型结构，不加 point head，不改推理。只在训练时对不确定点采样，并在这些点上加 focal + dice。

2. **完整 PointRend head**  
   需要 point head / fine-grained features / inference subdivision，改模型结构和导出路径，复杂度高。

当前建议先做第 1 种。原因：

- 风险小，能直接在当前 proto/coeff mask 上工作；
- 显存增量小于 `mask_ratio=2`；
- 可以快速消融是否提升 high-IoU mask；
- 不影响现有 checkpoint 推理结构。

### 7.4 建议实现策略

当前 WIP 的方向总体合理，但正式训练前建议补齐 4 个点：

1. **分项日志**

    当前 `seg_comp/seg_bnd/seg_point` 加进 `seg_loss` 后，`results.csv` 只能看到总 `seg_loss`。建议至少在调参阶段增加内部 debug 或临时日志，记录：

    ```text
    train/seg_bce
    train/seg_comp
    train/seg_bnd
    train/seg_point
    ```

    否则很难判断是 boundary 在起作用，还是把主 BCE 压坏。

2. **默认关闭，显式开启**

    `default.yaml` 中保持：

    ```yaml
    seg_comp: 0.0
    seg_bnd: 0.0
    seg_point: 0.0
    ```

    这能保证现有 checkpoint resume 行为不变。

3. **低权重起步**

    不建议第一次 COCONut-B 大训练就用 `1.0/1.0/1.0` 全开。建议先从小权重开始：

    ```text
    seg_comp=0.2~0.5
    seg_bnd=0.05~0.25
    seg_point=0.2~0.5
    seg_point_num=112 或 196
    ```

    原因：COCONut v2 当前训练 mask 仍是 poly -> raster -> downsample，边界本身有台阶/锯齿；过强 boundary loss 会放大标签 aliasing。

4. **避免与 mask_ratio=2 同时首测**

    `mask_ratio=2` 是另一个强变量，会明显增加显存和 mask 监督分辨率。第一轮不要同时改：

    ```text
    实验 1：mask_ratio=4 + boundary/point loss
    实验 2：mask_ratio=2 + baseline loss
    实验 3：mask_ratio=2 + 最优 boundary/point 组合
    ```

### 7.5 推荐消融顺序

先用 COCONut-S 或 COCONut-B 小轮数跑消融，不直接开 200 epoch：

| 实验            | seg_comp | seg_bnd | seg_point | 目的                                  |
| --------------- | -------: | ------: | --------: | ------------------------------------- |
| G0 baseline     |        0 |       0 |         0 | 复现当前 loss                         |
| G1 completeness |      0.3 |       0 |         0 | 看是否减少漏分/孔洞                   |
| G2 boundary     |        0 |     0.1 |         0 | 看边界项是否提升 mask AP75/AP95       |
| G3 point        |        0 |       0 |       0.3 | 看 uncertain points 是否改善 high-IoU |
| G4 comp+point   |      0.3 |       0 |       0.3 | 区域完整性 + 难点采样                 |
| G5 bnd+point    |        0 |     0.1 |       0.3 | 边界 + 难点采样                       |
| G6 all-low      |      0.3 |     0.1 |       0.3 | 低权重组合                            |

评估指标必须包括：

```text
Box mAP50-95
Mask mAP50-95
Mask AP50 / AP75
mask AP75 - AP50 trend
small / medium / large
per-class top/bottom delta
COCO official val2017
COCONut-B v2 val
```

成功标准：

- mask mAP50-95 提升，且 box mAP 不显著下降；
- mask AP75/AP95 或 mAP50-95 提升幅度大于 AP50，说明边界质量是真的改善；
- COCO official 不下降，至少不能只在 COCONut v2 上涨；
- per-class 底部类别没有大面积退化。

### 7.6 与继续 recipe200 的关系

当前 recipe200 baseline 仍未跑满，且 epoch 107 还在涨。建议分两条线：

**线 A：补完当前 baseline**

```text
resume current best/last
batch=80~84
保持 multi_scale=0.25
继续到 200 或至少 150
```

目的：得到没有 boundary/PointRend loss 的强基线。

**线 B：小规模 boundary/PointRend 消融**

```text
从当前 best.pt 或同一初始权重出发
epochs=30~50
mask_ratio=4
batch 按显存重新估计
只改 seg_comp/seg_bnd/seg_point
```

目的：找 loss 组合，不消耗多日 3GPU 直接大跑。

如果线 B 有稳定收益，再把最佳组合合入线 A 的后半程 fine-tune：

```text
best recipe200 checkpoint
lower lr / cos tail
seg loss gains low weight
30~50 epoch fine-tune
```

---

## 8. 建议的下一步操作清单

1. **先保全并评估当前 best**
    - 当前 best: `runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt`
    - 先跑 COCONut-B v2 val 复评；
    - 再跑 COCO official val2017；
    - 输出 per-class AP top/bottom。

2. **继续 baseline**

    不建议原样 b90 继续。建议：

    ```bash
    /home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python scripts/train_yolo26s_seg_coconut_distill.py \
      --resume runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/last.pt \
      --data /home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml \
      --batch 80 \
      --device 0,1,2 \
      --workers 8
    ```

3. **稳定 WIP loss 代码**
    - 跑相关单测；
    - 增加 loss 分项日志或至少临时 debug 输出；
    - 确认默认 `seg_* = 0` 时与 legacy loss bit-level / 数值近似一致；
    - 确认 DDP + BF16 下无 NaN。

4. **跑 COCONut-S / 短周期消融**

    优先小权重组合：

    ```text
    seg_comp=0.3
    seg_bnd=0.1
    seg_point=0.3
    seg_point_num=112
    ```

5. **决定是否升级监督分辨率**

    若 boundary/point loss 对 AP75/AP95 有收益，再考虑：

    ```text
    mask_ratio=2
    或更密 poly / approx_epsilon 更小
    ```

    但这会显著增加显存，应和 batch/multi_scale 分开消融。

---

## 9. 风险与注意事项

- Boundary loss 会放大标签边界噪声；当前 YOLO poly + `mask_ratio=4` 不是 COCONut 源 mask 的无损监督。
- PointRend-style point loss 当前只是 loss，不是完整 PointRend head；推理质量提升取决于 proto/coeff 是否能表达边界，不会带来推理时超分细化。
- Tversky completeness 对小物体可能更敏感，权重过高会造成 mask 外扩。
- 当前最大稳定性问题仍是 recovery path OOM。继续蒸馏大训练前，建议降低 batch 或改 DDP recovery。
- Response KD 仍缺失。Boundary/PointRend 主要补 mask，不会根治 box gap 和 COCO recall gap。

---

## 10. 阶段结论

recipe200 已经证明：

```text
COCONut-B v2 修正标签 + 200 epoch 配方 + copy_paste/mixup/multi_scale
```

比旧阶段 C 更好，但当前训练还没有跑满，也没有完成 COCO official 双标尺复评。下一阶段应同时推进两件事：

1. 用更稳的 batch 继续当前 baseline，拿到完整 150/200 epoch 曲线；
2. 用短周期消融验证 PointRend-style point loss、Sobel boundary loss、Focal-Tversky completeness loss 是否真的提升 mask high-IoU，而不是只增加训练噪声。

在确认收益前，不建议直接把 boundary/PointRend 全权重大规模加入 200 epoch 主线训练。
