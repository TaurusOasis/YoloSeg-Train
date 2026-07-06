# YOLO26s-seg COCONut 蒸馏 · recipe200 阶段性总结

> **文档日期**：2026-07-05  
> **Run**：`yolo26s-seg-coconut-b-v2-distill-recipe200`  
> **状态**：**已中断**（107/200 epoch 完成，ep108 @52% 崩溃）  
> **配套**：[`yolo26s-seg-distill-training-flow.md`](yolo26s-seg-distill-training-flow.md)（全链路）、[`yolo26-seg-training-review.md`](yolo26-seg-training-review.md)（loss 审查）

---

## 1. 执行摘要

| 维度               | 结论                                                                         |
| ------------------ | ---------------------------------------------------------------------------- |
| **主线目标**       | P2-2：v2 标签 + 升级配方 + 全权重蒸馏，200 epoch 长训                        |
| **实际进度**       | **107/200（53.5%）**；batch=90 段自 ep17 连续稳定 ~90 epoch                  |
| **最优权重**       | **ep107 `best.pt`**，v2 val mask mAP50-95 = **0.376**                        |
| **vs 阶段 C best** | mask +0.003、box +0.004 → **v2 标尺已超越 C best**                           |
| **vs Teacher**     | mask 仍低 **0.028**（0.376 vs 0.404）                                        |
| **停止原因**       | ep108 高分辨率 batch non-finite → 恢复 last.pt → **recovery OOM**            |
| **mask 子损失**    | PointRend / boundary / completeness **代码就绪，recipe200 未启用**（gain=0） |

---

## 2. 训练链路总览

```
A. LVIS 普通训练 (1203类)                    ✅ 完成
B. LVIS·COCO80 蒸馏 → best.pt               ✅ 完成
C. COCONut-B 蒸馏 v1 标签 (100ep, e98)     ✅ 完成  mask50-95=0.341 (v1 val)
P2-1 v2 标签 (F11 孔洞/断裂修复)            ✅ 完成  COCONut_b_yolo_seg_v2/
P2-2 recipe200 (v2 + 升级配方, 200ep)       ⏸ 107/200 中断
P3   mask 子损失消融 (seg_comp/bnd/point)    📋 代码就绪，待 GPU + 计划
```

### 2.1 recipe200 配方（`args.yaml`）

| 项          | 值                                                                  |
| ----------- | ------------------------------------------------------------------- |
| 数据        | `COCONut_b_yolo_seg_v2/coconut-b-seg.yaml`                          |
| 初始化      | 阶段 C `best.pt`                                                    |
| Teacher     | `yolo26x-seg.pt`，`dis=3.0`, `dis_proto=1.0`                        |
| epochs      | 200，`cos_lr=True`, `close_mosaic=20`                               |
| 增强        | `mosaic=1.0`, `copy_paste=0.4`, `mixup=0.1`, **`multi_scale=0.25`** |
| batch       | 全局 **90**（3×5090D，每卡 30）                                     |
| mask 子损失 | **`seg_comp=0`, `seg_bnd=0`, `seg_point=0`**                        |

---

## 3. recipe200 训练时间线

| 时间         | 事件                                   |
| ------------ | -------------------------------------- |
| 07-02 17:03  | 启动 batch=96，C best 初始化           |
| 07-02 20:58  | SIGINT 中断（~ep11）                   |
| 07-02 22:30  | resume batch=150 → **OOM**             |
| 07-03 00:28  | resume batch=100 → ep17 **OOM + inf**  |
| 07-03 00:53  | **resume batch=90** → 稳定段开始       |
| 07-03 ~ep48  | 走出 ep18 平台，指标单调上行           |
| 07-04 ~ep104 | **追平 C best**（mask 50-95 ≈ 0.374）  |
| 07-04 11:43  | **ep107 刷新 best**（mask 0.376）      |
| 07-04 11:55  | **ep108 batch1423/2685 崩溃**（见 §5） |

---

## 4. 验证指标

### 4.1 标尺对照（mask mAP50-95）

| 模型                      |    v2 val | COCO val2017 | 说明                                   |
| ------------------------- | --------: | -----------: | -------------------------------------- |
| Teacher yolo26x           |     0.404 |            — | `v2val-teacher-x/`                     |
| 阶段 C best (e98)         | **0.373** |        0.356 | 100ep v1 训练 + v2 复评                |
| **recipe200 best (e107)** | **0.376** |       待复测 | 当前交付候选                           |
| 官方 yolo26s-seg          |     0.350 |    **0.386** | ep91 时评测                            |
| recipe200 ep91 快照       |     0.369 |        0.329 | `eval_compare.../summary_plots_*.json` |

### 4.2 里程碑曲线（mask mAP50-95）

|   Epoch |        值 | 阶段        |
| ------: | --------: | ----------- |
|       1 |     0.344 | 起跳        |
|      18 |     0.354 | 早期平台峰  |
|      48 |     0.356 | 平台突破    |
|      76 |     0.364 | 稳定爬升    |
|      91 |     0.369 | 逼近 C best |
|     104 |     0.374 | 追平 C best |
| **107** | **0.376** | **best**    |

ep107 完整指标（v2 val，45003 实例）：

|          |           Box |          Mask |
| -------- | ------------: | ------------: |
| mAP50    |         0.589 |         0.571 |
| mAP50-95 |     **0.436** |     **0.376** |
| P / R    | 0.673 / 0.541 | 0.669 / 0.532 |

### 4.3 P2-2 验收清单

| 项                   | 目标         |         ep107 | 状态      |
| -------------------- | ------------ | ------------: | --------- |
| v2 val 超 C best     | mask > 0.373 |         0.376 | ✅        |
| v2 超官方预训练      | mask > 0.350 |         0.376 | ✅        |
| COCO val2017 硬约束  | mask ≥ 0.386 | 未测（ep107） | ❓        |
| 200ep + close_mosaic | 充分收敛     |       107/200 | ❌ 未完成 |

---

## 5. 停止原因（ep108 详解）

### 5.1 事件链

```
ep108, batch 1423/2685
  Size=800 (multi_scale 峰值), Instances=536, GPU_mem≈22.4G
    ↓
某 DDP rank loss/gradient 非有限
  (rank0 loss_items 可能仍有限；all_reduce(MAX) 全局判 bad)
    ↓
trainer: "Non-finite ... recovering from last.pt" (attempt 1/3)
    ↓
epoch 108 从 batch 0 重来 → student forward → proto BN
    ↓
CUDA OOM: 需 248 MiB，GPU0 仅 84 MiB 空闲（已占 31.29 GiB）
    ↓
DDP exitcode=1 → SIGTERM 清理其它 rank
```

### 5.2 机制（代码）

- `trainer.py`：`loss_bad` 经 **`dist.all_reduce(ReduceOp.MAX)`**，任一 rank bad → 全局恢复。
- 恢复路径 `_handle_nan_recovery`：reload EMA → **不释放显存** → 同一 epoch 立即 forward → 在满显存下 OOM。
- 已查 ep108 问题 batch 的 label：**坐标/格式正常**，更像 **高分辨率 + 大实例数 + 蒸馏双 forward** 的数值/显存边界，而非标签损坏。

### 5.3 根因归纳

| 层级 | 原因                                                         |
| ---- | ------------------------------------------------------------ |
| 直接 | recovery forward OOM                                         |
| 触发 | 800px 峰值 batch 上 non-finite（某 rank）                    |
| 结构 | `batch=90` + `multi_scale=0.25` + teacher 蒸馏 ≈ 显存上限    |
| 历史 | batch=100/150 曾 OOM；batch=90 稳定 90 epoch 后在 ep108 触顶 |

---

## 6. 产物与路径

### 6.1 权重

```
runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/
├── best.pt      ← ep107，mask50-95=0.376（推荐推理/续训基线）
├── last.pt      ← ep107（与 best 同 epoch）
├── epoch100.pt / epoch105.pt / …  (save_period=5)
```

### 6.2 日志与监控

| 资源     | 路径                                                                                            |
| -------- | ----------------------------------------------------------------------------------------------- |
| 训练 log | `runs/segment/train_logs/yolo26s-seg-coconut-b-v2-distill-recipe200.resume-b90.log`             |
| 指标 CSV | `runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/results.csv`                           |
| SwanLab  | `runs/segment/swanlab/yolo26s-seg-coconut-b-v2-distill-recipe200/run-20260703_005356-vtl1c8cp/` |
| 对比评测 | `runs/segment/eval_compare_recipe200_vs_official/`（ep18–91 快照，ep107 待补）                  |

### 6.3 可视化评测（ep91 快照）

```
eval_compare_recipe200_vs_official/
├── recipe200-best__coconut-v2-val_plots/val_batch*_pred.jpg
├── yolo26s-seg-official__coconut-v2-val_plots/...
└── summary_plots_20260703_025037.json
```

---

## 7. Mask 子损失：已实现 vs 未启用

**策略①（strategy-1）**：在现有 `single_mask_loss` 内叠加子项，**默认 gain=0**，recipe200 **零增量**。

### 7.1 配置键（`default.yaml`）

```yaml
seg_comp: 0.0 # Focal-Tversky 完整性（α=0.3 β=0.7 γ=0.75）
seg_bnd: 0.0 # Sobel 边界 L2（GT 边界带加权）
seg_point: 0.0 # PointRend Lite：uncertainty 采样 + focal + dice
seg_point_num: 112
seg_point_oversample: 3
seg_point_importance: 0.75
```

### 7.2 公式（`loss.py::single_mask_loss`，gain>0 时）

```
total = bce_term                           # 原 YOLO mask BCE（crop + area norm）
      + comp_w  × Σ FocalTversky_i         # FN 加权，促 mask 完整
      + bnd_w   × Σ SobelL2_i              # pred/GT 边界梯度对齐
      + point_w × Σ (focal_i + dice_i)     # 112 点，full-grid uncertainty 采样
```

- **PointRend**：`mask_point_sampling.py`（D2/SAM3 fork）；点损失 = **sigmoid focal + dice**，非 BCE。
- **Boundary**：`ops.sobel_magnitude()` + 边界带权重。
- **Short-circuit**：三者均为 0 → 与原版 BCE ** bitwise 等价**（单测已覆盖）。

### 7.3 相关文件（**未提交**，工作区）

| 文件                                            | 状态                       |
| ----------------------------------------------- | -------------------------- |
| `ultralytics/utils/mask_point_sampling.py`      | ✅ 新增                    |
| `ultralytics/utils/loss.py`                     | ✅ 扩展 `single_mask_loss` |
| `ultralytics/utils/ops.py`                      | ✅ `sobel_magnitude`       |
| `ultralytics/cfg/default.yaml`                  | ✅ 6 个新键                |
| `scripts/ablate_seg_loss_coconut_s.py`          | ✅ G0–G6 消融脚本          |
| `scripts/eval_compare_recipe200_vs_official.py` | ✅ 双标尺评测              |
| `ultralytics/data/dali_seg.py`                  | 🚧 实验性，未接入训练      |
| `tests/test_engine.py`                          | ✅ plumbing + 等价性       |

### 7.4 与 segPipeline eval 的关系

| 组件                    | 训练       | 评测                                 |
| ----------------------- | ---------- | ------------------------------------ |
| `seg_bnd` (Sobel)       | train loss | —                                    |
| `seg_point` (PointRend) | train loss | —                                    |
| `boundary_f_score`      | —          | `segPipeline/.../boundary.py` 后处理 |

训练 loss 与 eval 边界 F-score **不同链路**；需 `eval_mask_boundary.py`（待写）做 checkpoint 级对照。

---

## 8. 数据与标签（P2-1 回顾）

- **v2 数据集**：`Dataset/COCONut_b_yolo_seg_v2/`（F11：孔洞保留 + 每实例一行）
- train 标签行 −24%；val 实例 57220→45003
- v1「student 追平 teacher」为标签噪声伪象；v2 复评 teacher 仍领先 mask +0.031

**当前 GT 瓶颈仍大于 loss 选型**：poly → `approxPolyDP` → `mask_ratio=4`（640→160）对边界/孔洞的影响，大于 BCE vs PointRend 之差。

---

## 9. 下一步计划

### Phase A：恢复 recipe200 长训（优先）

**不要**原样 `batch=90 + multi_scale=0.25` 硬 resume。

| 方案           |  batch | multi_scale | 说明                                |
| -------------- | -----: | ----------: | ----------------------------------- |
| **A1（推荐）** | **84** |        0.25 | 降 ~7% 显存，保留增强               |
| A2             |     90 |    **0.15** | 上限 736px，避 ep108 类 800px batch |
| A3（最稳）     | **80** |    **0.15** | 跑满 ep180 close_mosaic             |

```bash
cd /home/genesis/Train/Code/ultralytics
conda activate yolo26-cu133
python scripts/train_yolo26s_seg_coconut_distill.py \
  --resume runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/last.pt \
  --data /home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml \
  --batch 84 --device 0,1,2 --workers 8
# 可选：multi_scale=0.15
```

续训目标：ep107→200 + **ep180 close_mosaic**；COCO val2017 复测 ep107/最终 best。

**可选工程**：recovery 前 `torch.cuda.empty_cache()`；或 recovery 时跳过当前 multi_scale batch。

### Phase B：Mask 子损失消融（GPU 空闲或 recipe200 完成后）

**原则**：先 **G0/G2/G4**（单因子），再组合；**不在 recipe200 主线上直接开大 gain**。

| 组  | seg_comp | seg_bnd | seg_point | 目的               |
| --- | -------: | ------: | --------: | ------------------ |
| G0  |        0 |       0 |         0 | baseline           |
| G2  |        0 |       1 |         0 | boundary only      |
| G4  |        0 |       0 |         1 | **PointRend only** |
| G5  |        0 |       1 |         1 | boundary + point   |
| G6  |        1 |       1 |         1 | all                |

```bash
# COCONut-S 30ep，无蒸馏，脚本已就绪
python scripts/ablate_seg_loss_coconut_s.py --group G4 --device 0
```

**推荐初始 gain（消融起点，非最终 recipe）**：

```yaml
seg_point: 0.5 # 或 1.0，与 seg_bnd 不要同时过大
seg_bnd: 0.0 # G4 单测 point 时为 0
seg_comp: 0.0 # 完整性靠 G1 单独测
```

验收：v2 val mask 50-95 + segPipeline `boundary_f_score`（待 eval 脚本）。

### Phase C：评测与文档收口

1. **ep107 双标尺复测**（带图）：`eval_compare_recipe200_vs_official.py --plots --suffix _ep107`
2. **`scripts/eval_mask_boundary.py`**：checkpoint × v2 val × boundary F-score / HFR
3. **提交** mask loss 代码 + 消融结果；recipe200 与 loss 实验 **分支/提交隔离**
4. 更新 [`yolo26s-seg-distill-training-flow.md`](yolo26s-seg-distill-training-flow.md) §P2-2 状态

### Phase D：低优先级

- DALI 训练预处理（`dali_seg.py`，需 fix label sync + batch 显存评估）
- Response KD（P3-3，COCO 泛化 gap 大时上调优先级）
- v2 标签二次修订（当前不优先）

---

## 10. 风险与决策点

| 风险                      | 缓解                                               |
| ------------------------- | -------------------------------------------------- |
| 续训再次 non-finite/OOM   | batch 84 + 可选 multi_scale 0.15                   |
| mask loss 与蒸馏抢显存    | 消融用 COCONut-S、无 teacher；主训 gain 小步加     |
| point+bnd 同时过大        | G5 仅作组合探路；生产 recipe 二选一为主            |
| best 已够交付但长训未完成 | ep107 best 可冻结；续训为追 teacher + close_mosaic |
| 未提交代码漂移            | 优先 commit strategy-1 + 单测                      |

---

## 11. 关键结论（给后续 Agent）

1. **recipe200 有价值**：ep107 已超 C best，蒸馏 + v2 标签 + 升级配方有效。
2. **停止非标签问题**：ep108 800px batch + DDP non-finite + recovery OOM；batch=90 触顶。
3. **mask 子损失已 code-ready**，recipe200 全程 gain=0；下一步是 **续训完成长训 + 独立消融引入 PointRend/boundary**。
4. **当前 best**：`weights/best.pt` @ ep107，mask mAP50-95 **0.376**。

---

_文档随训练/消融进展更新；下次里程碑：续训启动、ep107 复测、G0/G2/G4 消融结果。_
