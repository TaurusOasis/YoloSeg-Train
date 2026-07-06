# YOLO26s-seg PointRend Loss 新手入门教程

> **文档日期**：2026-07-07（§14–§29 代码层级增补）  
> **读者**：第一次接触本仓库 mask 子损失 / PointRend 的开发者  
> **配套**：[实验结果总结](yolo26s-seg-pointrend-experiment-results.md) · [设计文档（工程师向）](yolo26s-seg-pointrend-refine-head-design.md) · [recipe200 阶段总结](yolo26s-seg-recipe200-stage-summary.md) · [蒸馏主文档](yolo26s-seg-distill-training-flow.md)

---

## 1. 先建立正确预期（30 秒版）

在本仓库里，「PointRend」**不是** Detectron2 论文里那套「推理时迭代细分 mask」的完整系统，而是拆成两块：

| 部件           | 名字                             | 训练时    | 推理时    |
| -------------- | -------------------------------- | --------- | --------- |
| **(T) 训练侧** | Point loss + 可选 `PointHeadMLP` | ✅ 已接入 | ❌ 不参与 |
| **(I) 推理侧** | 迭代 subdivision                 | ❌ 未做   | ❌ 未做   |

**训练时发生的事：** 在每个正样本实例的 mask 上，随机采一批「不确定点」，用 focal + dice 监督这些点的 logit；若模型带了 MLP，则用 neck 细特征 + 粗 logit refine 后再算 loss。

**推理时发生的事：** 仍走老路径 `coeff @ proto → bilinear 上采样 → 阈值`，**边界不会立刻变锐**。验收应看 **mask mAP75 / 高 IoU 档**，不是肉眼看边界或 AP50。

---

## 2. 与本仓库其它 mask 子项的关系

PointRend 是 **strategy-1** 三子项之一，都挂在同一个函数里，默认 **gain=0**（不影响旧训练）：

| cfg 键      | 作用                            | 实现文件                          |
| ----------- | ------------------------------- | --------------------------------- |
| `seg_comp`  | 完整性（Focal-Tversky，偏 FN）  | `utils/mask_completeness_loss.py` |
| `seg_bnd`   | 边界对齐（Sobel L2）            | `utils/mask_boundary_loss.py`     |
| `seg_point` | PointRend 点 loss（focal+dice） | `utils/mask_point_sampling.py`    |

三者 **互不替代**，可单独或组合开启；PointRend 另有 **MLP 模式**（见 §5）。

---

## 3. 架构鸟瞰

```text
                    ┌─────────────────────────────────────┐
                    │  YOLO.train(..., seg_point=1.0)     │
                    └─────────────────┬───────────────────┘
                                      ▼
┌─────────────── Segment26.forward ───────────────────────────────┐
│  neck feats → preds["feats"]                                     │
│  proto, mask_coefficient                                         │
│  point_head.zero_loss() → preds["..."]["point_refine_dummy"]    │  ← DDP 占位
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
                    E2ELoss (one2many + one2one)
                                ▼
                    v8SegmentationLoss.loss()
                      ├─ TAL 分配 → fg_mask, target_gt_idx
                      └─ calculate_segmentation_loss()
                            └─ single_mask_loss()  ← 所有 mask 子项在这里
                                  ├─ BCE（必有，除非全子项 gain=0 短路）
                                  ├─ seg_comp × Tversky
                                  ├─ seg_bnd × Sobel L2
                                  └─ seg_point × (focal + dice) [+ MLP refine]
```

**记住一个入口：** 改 mask 监督，几乎总是看 `single_mask_loss`（`utils/loss.py`）。

---

## 4. 代码地图（按阅读顺序）

| 顺序 | 文件                                      | 你要看什么                                            |
| ---- | ----------------------------------------- | ----------------------------------------------------- |
| 1    | `cfg/default.yaml`                        | `seg_point*` 等默认超参                               |
| 2    | `cfg/models/26/yolo26-seg-pointrend.yaml` | 带 `point_hidden` 的模型定义                          |
| 3    | `nn/tasks.py` (~1937)                     | `parse_model` 如何解析 Segment26 第 4 参              |
| 4    | `nn/modules/head.py`                      | `PointHeadMLP`、`Segment26.point_head`、forward dummy |
| 5    | `utils/loss.py`                           | `v8SegmentationLoss`、`single_mask_loss` point 分支   |
| 6    | `utils/mask_point_sampling.py`            | 采样 + focal/dice                                     |
| 7    | `scripts/ablate_seg_loss_coconut_s.py`    | G0–G6 消融入口                                        |
| 8    | `scripts/smoke_point_head_ddp.py`         | 2-GPU DDP 梯度 smoke                                  |
| 9    | `tests/test_engine.py`                    | 单测（搜 `point` / `mask_point`）                     |

设计细节与评审历史见 [yolo26s-seg-pointrend-refine-head-design.md](yolo26s-seg-pointrend-refine-head-design.md)。

---

## 5. 两种 Point 模式（必懂）

### 5.1 构建级：`point_hidden`（YAML）

```yaml
# yolo26-seg-pointrend.yaml 最后一层
- [[16, 19, 22], 1, Segment26, [nc, 32, 256, 128]] # 128 = point_hidden
```

- `point_hidden > 0` → 构建 `PointHeadMLP`（参数量很小）
- 无第 4 参（普通 `yolo26-seg.yaml`）→ 无 MLP，只能走 **Lite**

### 5.2 运行时：`seg_point_refine` + `seg_point`

| point_hidden | seg_point | seg_point_refine | 行为                                                |
| :----------: | :-------: | :--------------: | --------------------------------------------------- |
|      0       |    >0     |        —         | **Lite**：`pl = coarse`（粗 einsum 采样）           |
|      >0      |    >0     |      False       | **Lite**（head 存在但 loss 不用 MLP；dummy 保 DDP） |
|      >0      |    >0     |       True       | **MLP**：`pl = point_head(feat, coarse)`            |
|     任意     |     0     |       任意       | point 分支 **关闭**（等价 legacy BCE-only 短路）    |

**Lite 与 MLP 在 zero-init 时等价：** 新训练的 MLP 初始输出等于 coarse，不会破坏已有曲线起点。

---

## 6. 逐步跟读：一次 forward + loss（新手版）

### Step 1 — 模型输出里有什么

`Detect.forward_head` 把 neck 特征放进 preds：

```149:158:ultralytics/ultralytics/nn/modules/head.py
    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        ...
        return dict(boxes=boxes, scores=scores, feats=x)
```

`Segment26.forward` 在训练态还会 stash **DDP dummy**（值为 0，连接 point_head 参数）：

```475:482:ultralytics/ultralytics/nn/modules/head.py
                if self.training and getattr(self, "point_head", None) is not None:
                    dummy = self.point_head.zero_loss()
                    preds["one2many"]["point_refine_dummy"] = dummy
                    preds["one2one"]["point_refine_dummy"] = dummy
```

`getattr` 是为 **旧 teacher ckpt**（无 `point_head` 键）蒸馏时防崩溃，不是可有可无的细节。

### Step 2 — loss 里如何取特征

```509:514:ultralytics/ultralytics/utils/loss.py
        point_refine_dummy = preds.get("point_refine_dummy")
        point_feats = preds["feats"][0] if self.hyp_get("seg_point_refine", False) and self.point_head is not None else None
```

- 细特征 = **P3**（`feats[0]`，约 80×80），**不新增** `point_feats` 键
- end2end 下 one2one 分支的 feats **已 detach**（`head.py` 里 `x_detach`），勿手动改

### Step 3 — `single_mask_loss` 里 point 分支

逻辑概要（见 `loss.py` ~655–700）：

1. `pred_mask = einsum(coeff, proto)` → 粗 logits `(n, H, W)`
2. `no_grad`：按 ROI 或全图做 **不确定性采样** → `coords (n, P, 2)`
3. `coarse = point_sample(pm4, coords)`；`pg = point_sample(gt, coords)`
4. 若 MLP：`pl = point_head(point_sample(feats), coarse)`；否则 `pl = coarse`
5. `total += seg_point * (focal(pl, pg) + dice(pl, pg)).sum()`

### Step 4 — gain 链（容易算错）

```text
L_mask = BCE + seg_comp·Tversky + seg_bnd·Boundary + seg_point·(focal+dice)
L_seg  = mean_batch(L_mask) + point_refine_dummy(=0)
L_seg  *= hyp.box          # 默认 7.5 — 整列 seg_loss 都乘
L_total = E2E 加权(one2many, one2one)
```

因此 **`seg_point=1.0` 的有效权重 ≈ 1.0 × 7.5 × o2m/o2o**，不是字面 1.0。

---

## 7. 核心模块说明

### 7.1 `PointHeadMLP`

```268:311:ultralytics/ultralytics/nn/modules/head.py
class PointHeadMLP(nn.Module):
    ...
    def forward(self, point_feats, coarse_logits):
        ...
        delta = self.mlp(torch.cat((point_feats, coarse_logits), dim=1)).squeeze(1)
        return coarse_logits.squeeze(1) + delta

    def zero_loss(self):
        return sum((p.sum() * 0.0 for p in self.parameters()), ...)
```

- **Conv1d** 在点维做 1×1 MLP，输入 `(N, C, P)`
- 末层 **zero-init** → 初始 `delta=0` → refined == coarse
- `zero_loss()`：DDP 用，不污染 loss 数值

### 7.2 点采样

- **全图 legacy**：`get_uncertain_point_coords_with_randomness`（`seg_point_roi < 0` 且未开 boundary）
- **ROI（默认）**：`get_uncertain_point_coords_in_roi`（`seg_point_roi >= 0`）
- **边界加权（可选）**：`seg_point_boundary=True`，在 bbox 内按 GT Sobel 幅度 multinomial 采点

不确定性定义：`calculate_uncertainty = -|logit|`（class-agnostic，与 Mask2Former 一致）。

### 7.3 点 loss

- `point_sigmoid_focal_loss_per_instance`：α=0.25, γ=2，逐实例对 P 个点 mean
- `point_dice_loss_per_instance`：逐实例 dice

---

## 8. 配置速查

| 参数                   | 默认  | 说明                               |
| ---------------------- | ----- | ---------------------------------- |
| `seg_point`            | 0.0   | 点 loss 子 gain；0=关闭 point 分支 |
| `seg_point_refine`     | False | True 且模型有 MLP 时走 refine      |
| `seg_point_num`        | 112   | 每实例采样点数 P                   |
| `seg_point_oversample` | 3     | 不确定性过采样倍率                 |
| `seg_point_importance` | 0.75  | 不确定点占比（其余随机）           |
| `seg_point_roi`        | 0.0   | ≥0 bbox+margin ROI；<0 legacy 全图 |
| `seg_point_boundary`   | False | GT Sobel 边界加权采样              |
| `box`                  | 7.5   | **整列 seg_loss 乘数**             |

---

## 9. 怎么跑（复制即用）

环境示例：`conda activate yolo26-cu133`，工作目录 `ultralytics/`。

### 9.1 单测（不占用 GPU 或极轻量）

```bash
pytest tests/test_engine.py -k "point or mask_point or single_mask" -q
```

### 9.2 Lite：只有 point loss，无 MLP

```python
from ultralytics import YOLO

YOLO("yolo26s-seg.yaml").train(
    data="/path/to/coconut-s-seg.yaml",
    epochs=30,
    seg_point=1.0,
    seg_point_roi=0.0,
    project="runs/segment",
    name="point-lite-smoke",
)
```

### 9.3 MLP：PointRend 训练侧完整模式

```python
YOLO("yolo26s-seg-pointrend.yaml").train(
    data="/path/to/coconut-s-seg.yaml",
    epochs=30,
    seg_point=1.0,
    seg_point_refine=True,
    seg_point_roi=0.0,
    batch=32,  # 显存紧张时降低
    seg_point_num=64,  # 可先 64 再升到 112
    project="runs/segment",
    name="point-mlp-smoke",
)
```

### 9.4 消融脚本（G0–G6）

```bash
python scripts/ablate_seg_loss_coconut_s.py --group G4m --device 0,1,2
# G4  = Lite；G4m = MLP（pointrend yaml + seg_point_refine=True）
```

### 9.5 从旧 ckpt 加 point head（finetune，不是 resume）

```python
YOLO("yolo26s-seg-pointrend.yaml").train(
    data="COCONut_b_yolo_seg_v2/coconut-b-seg.yaml",
    weights="runs/segment/.../weights/best.pt",  # 旧 seg 权重
    resume=False,                                 # 必须新 run
    seg_point=0.5,
    seg_point_refine=True,
    ...
)
```

**禁止：** 旧 seg ckpt + 新 pointrend 结构 + `resume=True`（optimizer/state 形状不匹配）。

### 9.6 DDP smoke

```bash
python -m torch.distributed.run --nproc_per_node=2 \
  scripts/smoke_point_head_ddp.py
```

---

## 10. 代码评审（2026-07-05）

### 10.1 做得好的地方

| 项                | 说明                                                                      |
| ----------------- | ------------------------------------------------------------------------- |
| **单一入口**      | 所有 mask 子项集中在 `single_mask_loss`，与 legacy BCE 共存清晰           |
| **默认零增量**    | `seg_*=0` 短路回 G3-Eq1，recipe200 类在途 run 不受影响                    |
| **DDP 混合解法**  | forward dummy + criterion 真 MLP，比「只放 loss 里」更 compile/静态图友好 |
| **zero-init MLP** | 训练起点稳定，Lite↔MLP 可对比                                            |
| **ROI 默认**      | 修复 Lite 时代全图背景采样浪费                                            |
| **蒸馏守卫**      | `getattr(..., point_head)` 避免旧 teacher forward 崩溃                    |
| **测试**          | point/ROI/legacy 等价性/ yaml 构建均有单测；DDP 有 smoke 脚本             |

### 10.2 风险与限制（读代码时要心里有数）

| 级别 | 问题                         | 影响                                               | 建议                                       |
| ---- | ---------------------------- | -------------------------------------------------- | ------------------------------------------ |
| 🔴   | **推理不用 point head**      | 训练 MLP 只间接改善 proto/coeff；可视化边界不变    | 验收 AP75+；若要锐边界需做 (I) subdivision |
| 🔴   | **GT 160×160 poly 栅格**     | point 监督上限受 polygon 简化 + aliasing 约束      | 结合 boundary F-score；别期待 miracle      |
| 🟡   | **细特征仅 P3**              | 非 Proto26 内部 fused feat，可能损失多尺度边界信息 | 后续 Proto26 暴露 fused feat 对比实验      |
| 🟡   | **point loss 折进 seg_loss** | 日志看不到 point 子项量级                          | 临时 hook 或 val 时看 AP75                 |
| 🟡   | **显存随 n×P 涨**            | `expand(n,C,H,W)` + 多实例图 OOM                   | 降 batch / `seg_point_num`                 |
| 🟡   | **TAL 错配**                 | point 监督绑定 TAL 正样本，分配错则 mask 错        | 与 BCE 同源问题                            |
| 🟢   | **compile 路径**             | 设计称 dummy 安全，但缺专项 1-epoch smoke          | 开 compile 前跑一轮                        |
| 🟢   | **实验债务**                 | G0–G6 消融尚无 run 落盘                            | COCONut-S 30ep 先 G4 vs G4m                |

### 10.3 与设计文档 / 旧 memory 的一致性

- 权威设计：[yolo26s-seg-pointrend-refine-head-design.md](yolo26s-seg-pointrend-refine-head-design.md)（训练侧 T 已落地）
- 早期 strategy-1 memory 写「无 MLP」→ **已被 pointrend 设计推翻**；以设计 doc + 本教程为准
- (I) 推理细分、Proto26 fused feat、G0–G6 真跑 → **仍待办**

---

## 11. 常见问题 FAQ

**Q：开了 `seg_point=1` 为什么 val 边界还是糊？**  
A：推理不走 point head。看 mask mAP50-95 / AP75，或等 (I) 推理细分。

**Q：`yolo26s-seg.yaml` 和 `yolo26s-seg-pointrend.yaml` 区别？**  
A：后者 Segment26 多 `point_hidden`，可构建 MLP；仍需 `seg_point_refine=True` 才在 loss 里用 MLP。

**Q：`seg_point_roi=-1` 干什么？**  
A：退回全图 [0,1]² 采样，用于和旧 Lite 行为消融对照。

**Q：蒸馏还能开吗？**  
A：可以。teacher 无 point_head；`getattr` 守卫已修 forward。但双 forward 占显存，消融建议 **关 distill**。

**Q：怎么知道 MLP 有没有梯度？**  
A：跑 `smoke_point_head_ddp.py` 或训练一步后 `param.grad` 非 None；`seg_point_refine=False` 时 grad≈0（仅 dummy）。

**Q：resume 老 run 加了 point head 行吗？**  
A：不行。结构变了 → **finetune from checkpoint**（新 run、`resume=False`、optimizer 重 init）。

---

## 12. 建议学习路径（7 天）

| 天  | 任务                                                        |
| --- | ----------------------------------------------------------- |
| D1  | 读本文 §1–§6；跑 `pytest -k point`                          |
| D2  | 读 §14–§17 + `single_mask_loss`；画数据流图                 |
| D3  | 读 §18–§20 + `PointHeadMLP`；理解 dummy / E2E 梯度          |
| D4  | COCONut-S 1 epoch Lite（G4） smoke                          |
| D5  | COCONut-S 1 epoch MLP（G4m） smoke；对比 loss 有限性        |
| D6  | 读设计 doc §2.5/§2.6/§6.1；理解 finetune vs resume          |
| D7  | 规划 recipe200 ep107 → pointrend finetune 或 G0–G6 全量消融 |

---

## 13. 进阶阅读

| 文档                                                                                       | 内容                                                                    |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| 本文 §14–§25                                                                               | **代码层级走读**：调用栈、dict 契约、形状表、API 索引、决策树、单测矩阵 |
| [yolo26s-seg-pointrend-refine-head-design.md](yolo26s-seg-pointrend-refine-head-design.md) | DDP、ROI、checkpoint、导出判定                                          |
| [yolo26s-seg-recipe200-stage-summary.md](yolo26s-seg-recipe200-stage-summary.md)           | 主线训练状态、OOM、Phase B 消融计划                                     |
| [yolo26s-seg-distill-training-flow.md](yolo26s-seg-distill-training-flow.md)               | 蒸馏框架 F1–F21                                                         |
| Detectron2 PointRend                                                                       | 论文/原 repo — 对比完整 (T)+(I)                                         |

---

## 14. 全链路调用栈（从 `train()` 到 `backward`）

以 `YOLO(...).train()`、end2end seg、3-GPU DDP 为例：

```text
models/yolo/model.py          YOLO.train() → SegmentationTrainer.train()
models/yolo/segment/train.py  SegmentationTrainer.get_model() → SegmentationModel
engine/trainer.py             _setup_train() → DDP wrap → 训练循环
  ├─ preprocess_batch()       /255；DALI 仅 decode/resize
  ├─ model(batch["img"])      BaseModel.forward → Sequential backbone+head
  │     nn/modules/head.py    Detect.forward → Segment26.forward
  └─ model.loss(batch, preds) 或 unwrap(model).loss(...)  [compile 路径]
        nn/tasks.py           SegmentationModel.init_criterion → E2ELoss
        utils/loss.py         E2ELoss.__call__
          ├─ v8SegmentationLoss.loss(one2many_preds)
          └─ v8SegmentationLoss.loss(one2one_preds)
                calculate_segmentation_loss → single_mask_loss (逐图)
engine/trainer.py             loss.backward() → clip_grad → optimizer.step()
```

**改 PointRend 时最常打开的 4 个断点：**

1. `Segment26.forward` — dummy 是否 stash
2. `v8SegmentationLoss.loss` — `point_feats` / `point_refine_dummy`
3. `single_mask_loss` — ROI 采样 + MLP
4. `PointHeadMLP.forward` — refine 输出

---

## 15. `preds` / `batch` 字典契约

### 15.1 训练态 `preds`（end2end，one2many 分支示例）

进入 `v8SegmentationLoss.loss(preds, batch)` 的 `preds` 已是 **单分支 dict**（E2ELoss 拆开后再传入）：

| 键                   | 形状（典型）                              | 用途                          |
| -------------------- | ----------------------------------------- | ----------------------------- |
| `boxes`              | `(B, 4×reg_max, A)`                       | 检测 DFL 输入                 |
| `scores`             | `(B, nc, A)`                              | 分类                          |
| `feats`              | `[P3,P4,P5]`                              | neck 特征；**point 用 `[0]`** |
| `mask_coefficient`   | `(B, nm, A)`                              | mask 系数                     |
| `proto`              | `(B, nm, Hm, Wm)` 或 `(p, semantic)` 元组 | proto + 可选语义分割          |
| `point_refine_dummy` | 标量                                      | DDP 占位（可选）              |

one2one 分支结构相同；`feats` 已在 `Detect.forward` 里 **detach**。

### 15.2 `batch`（segment 相关）

| 键                           | 说明                                                  |
| ---------------------------- | ----------------------------------------------------- |
| `img`                        | `(B,3,H,W)` 归一化后 float                            |
| `masks`                      | overlap=True → `(B,Hm,Wm)` id map；False → 按实例堆叠 |
| `sem_masks`                  | Proto26 语义分割 GT（若 proto 返回 tuple）            |
| `bboxes`, `cls`, `batch_idx` | 检测标签                                              |

### 15.3 criterion 唯一入参不变量

设计约束：**除 `batch` 外，loss 所需数据必须来自 head 训练态 forward 的 `preds`**（见设计 doc §8）。PointRend 没有新增持久 dict 键（除 dummy 标量）；细特征复用 `feats[0]`。

---

## 16. 张量形状对照表（`imgsz=640`，单图有 n 个正实例）

| 阶段           | 张量                  | 形状                              | stride / 备注                    |
| -------------- | --------------------- | --------------------------------- | -------------------------------- |
| 输入           | `batch["img"]`        | `(B,3,640,640)`                   | —                                |
| P3             | `preds["feats"][0]`   | `(B, C, 80, 80)`                  | stride 8                         |
| proto          | `proto`               | `(B, 32, 160, 160)`               | mask_ratio=4                     |
| GT mask        | `batch["masks"]`      | `(B, 160, 160)` 或 overlap id map | polygon2mask                     |
| 粗 mask        | `pred_mask` = einsum  | `(n, 160, 160)`                   | 单图内 n 个实例                  |
| 采样坐标       | `coords`              | `(n, P, 2)`                       | P=`seg_point_num`，归一化 [0,1]² |
| 细特征（扩维） | `point_feats_i`       | `(n, C, 80, 80)`                  | `expand`，不复制存储             |
| 点特征         | `point_sample(feats)` | `(n, C, P)`                       | grid_sample                      |
| 粗 logit 点    | `coarse`              | `(n, P)`                          | 有 grad                          |
| refine         | `pl`                  | `(n, P)`                          | MLP 或 =coarse                   |

**proto 与 GT 尺寸不一致时：** `loss()` 会把 **proto 上采样** 到 `masks` 分辨率（`loss.py:527-529`），不是下采样 GT。

---

## 17. 函数级 API 索引

### 17.1 模型构建

| 符号                    | 文件                    | 作用                                                    |
| ----------------------- | ----------------------- | ------------------------------------------------------- |
| `SegmentationModel`     | `nn/tasks.py:581`       | seg 任务入口                                            |
| `init_criterion`        | `nn/tasks.py:592`       | `E2ELoss(..., v8SegmentationLoss)`                      |
| `parse_model` Segment26 | `nn/tasks.py:1937-1945` | 剥第 4 参 → extend → append scaled `point_hidden`       |
| `Segment26.__init__`    | `head.py:439`           | `point_head = PointHeadMLP(ch[0], point_hidden) if ...` |

**YAML → 实参顺序（Segment26）：**  
`[nc, nm, npr, reg_max, end2end, ch_tuple, point_hidden]`

### 17.2 Loss

| 符号                          | 文件           | 作用                                                             |
| ----------------------------- | -------------- | ---------------------------------------------------------------- |
| `v8SegmentationLoss.__init__` | `loss.py:496`  | `self.point_head = getattr(model.model[-1], "point_head", None)` |
| `v8SegmentationLoss.loss`     | `loss.py:509`  | 取 feats/dummy；调 `calculate_segmentation_loss`                 |
| `calculate_segmentation_loss` | `loss.py:704`  | 逐图循环；`expand` point_feats；`hyp_get` 注入 kwargs            |
| `single_mask_loss`            | `loss.py:575`  | **静态方法**；BCE + comp + bnd + point                           |
| `E2ELoss.__call__`            | `loss.py:1335` | o2m/o2o 加权；`parse_output` 拆 preds                            |

### 17.3 采样与点 loss

| 符号                                         | 文件                        | 作用                                |
| -------------------------------------------- | --------------------------- | ----------------------------------- | ----- | ----------------- |
| `point_sample`                               | `mask_point_sampling.py:22` | bilinear grid_sample，coords∈[0,1]² |
| `calculate_uncertainty`                      | `:43`                       | `-                                  | logit | `，要求 `(N,1,P)` |
| `get_uncertain_point_coords_with_randomness` | `:49`                       | 全图 legacy                         |
| `get_uncertain_point_coords_in_roi`          | `:186`                      | bbox+margin；可选 `weight_map`      |
| `point_sigmoid_focal_loss_per_instance`      | `:242`                      | 输出 `(N,)`                         |
| `point_dice_loss_per_instance`               | `:272`                      | 输出 `(N,)`                         |

### 17.4 cfg 注册

| 键                                       | 类型集合            | 文件                      |
| ---------------------------------------- | ------------------- | ------------------------- |
| `seg_point`, `seg_point_roi`             | `CFG_FLOAT_KEYS`    | `cfg/__init__.py:203-204` |
| `seg_point_importance`                   | `CFG_FRACTION_KEYS` | `:237`                    |
| `seg_point_num`, `seg_point_oversample`  | `CFG_INT_KEYS`      | `:248-249`                |
| `seg_point_refine`, `seg_point_boundary` | `CFG_BOOL_KEYS`     | `:291-292`                |

---

## 18. `parse_model` 与 YAML 第 4 参（逐步）

YAML 行：

```yaml
- [[16, 19, 22], 1, Segment26, [nc, 32, 256, 128]]
#                                      ↑nm ↑npr ↑point_hidden
```

`parse_model` 处理逻辑：

1. `len(args)>3` → `point_hidden=args[3]`，`args=args[:3]`（只剩 nc,nm,npr）
2. `args.extend([reg_max, end2end, ch_list])`
3. `args[2]`（npr）按 width 缩放
4. 若曾取出 point_hidden → `args.append(make_divisible(min(ph,max_ch)*width, 8))`

**旧 YAML 仅 3 参：** 不剥、不 append → `Segment26(point_hidden=0)` → 无 MLP。  
**误读防护：** `reg_max` 由 extend 注入，不会被当成 point_hidden。

---

## 19. `single_mask_loss` 决策树

```text
single_mask_loss(...)
│
├─ comp_w==0 && bnd_w==0 && point_w==0 ?
│     YES → 仅 BCE crop+area，return（G3-Eq1 legacy 等价）
│     NO  → total = BCE
│
├─ comp_w>0 → + comp_w * tversky_loss_per_instance(...)
├─ bnd_w>0  → + bnd_w * boundary_l2_loss_per_instance(...)
│
└─ point_w>0 && n>0 && num_points>0 ?
      YES →
        no_grad:
          use_roi = (roi_margin>=0) or boundary_w
          if use_roi:
            boxes_norm = xyxy / [W,H,W,H]
            weight_map = sobel(gt) if boundary_w else None
            coords = get_uncertain_point_coords_in_roi(...)
          else:
            coords = get_uncertain_point_coords_with_randomness(...)
          pg = point_sample(gm4, coords)
        coarse = point_sample(pm4, coords)   # 有 grad
        if point_head and point_feats:
            pl = point_head(point_sample(point_feats, coords), coarse)
        else:
            pl = coarse
        total += point_w * (focal(pl,pg) + dice(pl,pg)).sum()
      NO  → （跳过 point 分支）
│
return total
```

**ROI 调度真值表：**

| `seg_point_roi` | `seg_point_boundary` | 采样器                             |
| --------------- | -------------------- | ---------------------------------- |
| ≥ 0             | False                | ROI，margin=roi                    |
| ≥ 0             | True                 | ROI，margin=max(roi,0)，Sobel 加权 |
| < 0             | False                | legacy 全图                        |
| < 0             | True                 | **仍 ROI**（boundary 强制 ROI）    |

---

## 20. End2End 双分支：梯度落在哪

| 模块 / 张量                    | one2many             | one2one              |
| ------------------------------ | -------------------- | -------------------- |
| backbone（经 P3 feats）        | ✅ point 路径        | ❌ feats detached    |
| proto                          | ✅                   | ❌ detached          |
| mask coeff (cv4 / one2one_cv4) | ✅                   | ✅                   |
| `PointHeadMLP` 参数            | ✅ criterion + dummy | ✅ criterion + dummy |
| `point_refine_dummy`           | 加在 loss[1]         | 加在 loss[1]         |

E2E 权重：`o2m` 从 0.8 衰减到 0.1，`o2o` 互补（`E2ELoss.decay`）。

**MLP 真梯度来源：** criterion 里 `point_head(feats, coarse)`；dummy 只提供 0 梯度连接 forward 图。

---

## 21. `overlap_mask` 与 GT 提取

默认 `overlap_mask: True`（`default.yaml:44`）。在 `calculate_segmentation_loss` 内：

```753:757:ultralytics/ultralytics/utils/loss.py
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
```

Point loss 的 `pg` 来自上述 `gt_mask` 的双线性采样——与 BCE 同一 GT 源。  
消融脚本常用 `overlap_mask=False`（COCONut 实例分离标签）；**对比实验须统一此开关**。

---

## 22. 蒸馏路径中的 PointRend

`DistillationModel.loss`（`nn/distill_model.py`）：

1. teacher forward（hooks 抓 neck feat）
2. student forward → `student_model.loss(batch, preds)`
3. 额外 `dis_feat` / `dis_proto`，**不读 point_head**

要点：

- 学生 `v8SegmentationLoss` 正常算 point loss（若 cfg 开启）
- Teacher **无** `point_head`；`Segment26.forward` 用 `getattr(..., "point_head", None)` 跳过 dummy
- 双 forward + 点采样 → 显存压力大；**消融建议关 distill**

---

## 23. 训练 vs 推理：代码分叉

| 阶段              | 入口                                                 | point head                         |
| ----------------- | ---------------------------------------------------- | ---------------------------------- |
| **训练**          | `Segment26.forward` → `E2ELoss` → `single_mask_loss` | MLP 在 criterion；dummy 在 forward |
| **val（训练中）** | 同训练 forward；validator 算 mAP                     | 同左                               |
| **predict**       | `segment/predict.py` → `process_mask`                | **不参与**                         |
| **export**        | `AutoBackend` 仅 det+proto                           | **无 feats**                       |

推理插入点（(I) 未做）：`utils/ops.py` `process_mask` 在 matmul 与 crop 之间；需先从 predict 路径传入 neck feats（设计 doc §3.1）。

---

## 24. 单测矩阵（`tests/test_engine.py`）

| 测试                                                      | 验证内容                            |
| --------------------------------------------------------- | ----------------------------------- |
| `test_mask_point_coords_full_grid`                        | legacy 采样 ∈ [0,1]²                |
| `test_mask_point_coords_in_roi`                           | ROI 内 + 退化 bbox 回退             |
| `test_mask_point_coords_weighted_in_roi`                  | Sobel 加权 + 退化                   |
| `test_point_focal_dice_per_instance`                      | 形状、可反传                        |
| `test_point_head_mlp_zero_init_is_coarse_residual`        | init 时 refined==coarse             |
| `test_single_mask_loss_all_gains_disabled_matches_legacy` | G3-Eq1；Lite≈MLP@init；num_points=0 |
| `test_segmentation_loss_optional_hyp_injection_is_finite` | `getattr` 防御 + hyp 注入           |
| `test_segmentation_loss_boundary_roi_path_is_finite`      | boundary 强制 ROI                   |
| `test_segmentation_loss_cfg_overrides_are_accepted`       | cfg 键类型                          |
| `test_yolo26_pointrend_yaml_builds_optional_point_head`   | pointrend vs legacy yaml            |

**推荐命令：**

```bash
pytest tests/test_engine.py -k "point or mask_point or single_mask or boundary_roi or pointrend" -q
python -m torch.distributed.run --nproc_per_node=2 scripts/smoke_point_head_ddp.py
```

---

## 25. 调试清单（按症状）

| 症状                         | 先查                           | 常见原因                                |
| ---------------------------- | ------------------------------ | --------------------------------------- |
| DDP `unused parameters`      | forward 是否 stash dummy       | `point_head` 存在但 seg_point=0 / 无 fg |
| `AttributeError: point_head` | teacher 是否旧 ckpt            | 缺 getattr 守卫（已修）                 |
| loss NaN                     | bf16 下 Sobel/point_sample     | 应用 fp32-internal（MLP 已 `.float()`） |
| OOM                          | n×P×C×H×W                      | 降 batch / seg_point_num                |
| MLP 无 grad                  | seg_point_refine / point_feats | False 或 point_hidden=0                 |
| 开了 point loss 边界仍糊     | predict 路径                   | 正常；看 AP75                           |
| resume 失败                  | 结构是否变                     | 加 point_head → finetune                |
| Lite vs MLP 无差别           | 是否 zero-init 初期            | 需训若干 epoch 或查 AP75                |

**临时看 point/bce 量级比：** 参考 `tmp/smoke_seg_losses.py` 的 ratio 诊断（不依赖改 loss_names）。

---

## 26. PointRend + Boundary + Refine 训练：源码问题清单（2026-07-05）

本节针对 **完整训练配置**：

```python
YOLO("yolo26s-seg-pointrend.yaml").train(
    seg_point=1.0,
    seg_point_refine=True,  # MLP refine
    seg_point_boundary=True,  # GT Sobel 边界加权采样
    seg_point_roi=0.0,  # bbox ROI（boundary 会强制 ROI≥0）
    # 可选叠加：seg_bnd=1.0（dense 边界 L2，与 boundary 采样不同）
)
```

先澄清三个易混开关（**源码里是三套独立机制**）：

| 开关                 | 作用层                               | 源码位置                                                          |
| -------------------- | ------------------------------------ | ----------------------------------------------------------------- |
| `seg_point_refine`   | MLP refine 点 logit                  | `loss.py:514` 取 `feats[0]` + `point_head`                        |
| `seg_point_boundary` | **仅** 改变点坐标采样分布            | `loss.py:672-682` `sobel_magnitude(gt)` → `_weighted_rand_in_roi` |
| `seg_bnd`            | 全图 dense Sobel L2（与 point 无关） | `loss.py:647-653` `boundary_l2_loss_per_instance`                 |

---

### 26.1 训练能跑通，但「验收指标」与训练目标错位 🔴

**现象：** 开了 MLP + boundary 训练，val 可视化边界仍糊，AP50 可能动、AP75/AP95 不一定反映 boundary 采样收益。

**源码根因：**

- 训练：`single_mask_loss` 里 `point_head(pf, coarse)` 参与反传（`loss.py:693-695`）。
- 验证/推理：`Segment26.forward` 在 `not training` 时返回 `(det, proto)`（`head.py:483-485`）；`process_mask` 仅 `coeff @ proto` + crop（`ops.py:507-522`），**无 feats、无 point_head、无 subdivision**。

因此 **MLP 权重只通过训练 loss 间接塑造 proto/coeff**，不会在 val mask 上直接生效。应用 mask mAP 高 IoU 档验收，不能靠肉眼看 predict 边界。

---

### 26.2 细特征分辨率：P3 80×80 vs mask/proto 160×160 🟡

**数据流：**

```655:695:ultralytics/ultralytics/utils/loss.py
            pm4 = pred_mask.float().unsqueeze(1)   # (n,1,160,160) 典型
            ...
                coarse = point_sample(pm4, coords, ...)   # 在 mask 网格采样
            if point_head is not None and point_feats is not None:
                pf = point_sample(point_feats, coords, ...)  # 在 P3 网格采样
                pl = point_head(pf, coarse)
```

`coords` 是 [0,1]² 归一化坐标，几何上对齐同一图像位置，但 **P3 有效空间分辨率是 mask 的 1/2**。boundary 采样在 160×160 像素格上用 Sobel 选点，MLP 却在更粗的 P3 上取特征 — boundary refine 的上限被 neck 最浅层限制。设计 doc 中的 Proto26 fused feat（task #9）尚未接入。

---

### 26.3 E2E one2one 分支：MLP 的 fine-feature 梯度被截断 🟡

```165:168:ultralytics/ultralytics/nn/modules/head.py
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
```

`v8SegmentationLoss.loss` 对 **one2many 与 one2one 各跑一遍**（`E2ELoss.__call__`）。one2one 的 `preds["feats"][0]` 已 detach → `point_feats` 无梯度 → MLP 在该分支 **只从 coarse logit 路径** 获得对 mask coeff/proto 的梯度，**不能**通过 one2one 更新 P3 backbone。

E2E 加权（o2m 0.8→0.1）意味着训练后期 one2one 权重上升，**point refine 对 backbone 的监督随 epoch 减弱**。这是架构选择，不是 boundary 特有 bug，但 boundary+refine 对边界特征敏感，影响更大。

---

### 26.4 Boundary 采样 vs 不确定性采样：目标冲突 🟡

`get_uncertain_point_coords_in_roi` 流程（`mask_point_sampling.py:186-239`）：

1. **oversample 池**：若 `weight_map` 非空，用 GT Sobel multinomial 在 bbox 内抽 `P×oversample` 个候选点。
2. **importance 子集**：在这批候选上算 **pred** 的 `-|logit|` 不确定性，top-k 保留（默认 75%）。
3. **random 余量**：再抽 25%，同样 boundary 加权。

**问题：** 若 pred 在 bbox **内部**（远离 GT 边界）错误但 logits 不确定，这些点 **很难进入** boundary 加权的 oversample 池；只有 25% random 余量可能碰到。boundary 模式强化「在真边界附近学 pred」，但 **弱化了对内部误检区域的 point 监督**。与全图 legacy 采样或纯 ROI uniform 的行为不同，消融时必须固定 `seg_point_boundary` 再对比。

---

### 26.5 GT 与 ROI 的上限（非代码 bug，但训练天花板）🔴

| 环节      | 源码                                                 | 影响                                                     |
| --------- | ---------------------------------------------------- | -------------------------------------------------------- |
| mask 栅格 | `Format.mask_ratio=4` → 640→160（`default.yaml:45`） | boundary Sobel 在 160 格上，1 像素 ≈ 4 原图像素          |
| polygon   | `polygon2mask` / `approxPolyDP`（`data/utils.py`）   | 真边界被简化、锯齿                                       |
| `pg` 目标 | `point_sample(gm4, coords)` 双线性（`loss.py:691`）  | 软标签，非硬 0/1                                         |
| ROI bbox  | TAL 分配的 **GT bbox**（`tal.py:270-271` → `mxyxy`） | 框大则 bbox 内背景 Sobel≈0，靠 weight_empty 回退 uniform |

boundary 加权在 **GT 边界 aliasing** 上建权重图，MLP 学的是「在栅格边界附近拟合栅格 GT」，不是亚像素真边界。

---

### 26.6 显存与算力：boundary + refine 叠加成本 🟡

`calculate_segmentation_loss` 对每个 fg anchor 调用 `single_mask_loss`（`loss.py:749-778`）：

- **同一 GT 多 anchor 正样本** → 同一实例 mask 重复算 point loss（不同 pred coeff），n 随 TAL 正样本数涨。
- `point_feats_i = point_feats[i:i+1].expand(n, ...)`（`:759-760`）不复制显存，但 **forward 仍按 n 实例** 做 `point_sample` + Conv1d MLP。
- boundary 路径：每实例 `_weighted_rand_in_roi` 建 `(N,H,W)` bbox_mask + `multinomial(H×W)`（`mask_point_sampling.py:160-178`），H=W=160 时单实例尚可，**密集小目标图** n 大时显著慢于 Lite。

recipe200 ep108 OOM 发生在 **seg_point=0** 的 distill 配置；若叠加 point+boundary+refine+distill，OOM 风险更高。

---

### 26.7 实验与工具链缺口 🟡

| 缺口           | 证据                                                                                                                         |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 无 preset 组   | `ablate_seg_loss_coconut_s.py` 有 G4m，**无** `seg_point_boundary=True` 的 G4mb/G6m                                          |
| 单测未覆盖组合 | `test_segmentation_loss_boundary_roi_path_is_finite` 未传 `point_feats`/`seg_point_refine`；MLP+boundary 联调 **无专门测试** |
| DDP smoke      | `smoke_point_head_ddp.py` 未开 `seg_point_boundary`                                                                          |
| 无生产 run     | `runs/segment/` 无 ablate 落盘                                                                                               |
| 日志不可观测   | `loss_names` 仅 `seg_loss`（`segment/train.py:66`），point/boundary 子项不可分                                               |

---

### 26.8 蒸馏与 finetune 路径 🟡

- **Teacher** 无 `point_head`；student forward 靠 `getattr` 跳过 dummy（`head.py:475`）— 不 crash。
- **DistillationModel.loss** 双 forward + student `loss()` 内 point 采样（`distill_model.py:316-321`），**不蒸馏 point_head**。
- 从 recipe200 ep107 `best.pt` finetune：须 `yolo26s-seg-pointrend.yaml` + `resume=False`；optimizer 重 init，**MLP 从 zero-init 开始**，前几 epoch point loss 等价 Lite。

---

### 26.9 与 `seg_bnd` 同时开启时的重复监督 🟢

- `seg_bnd`：对 **pred 全图** Sobel 与 GT Sobel 做 L2（`mask_boundary_loss.py:55-58`）。
- `seg_point_boundary`：只影响 **112 个采样点** 的位置。

可同时开，但二者都依赖 GT Sobel，**边界信号相关**；G5/G6 含 `seg_bnd+seg_point` 但 G4m **不含 boundary 采样**。应用「G4m + boundary」单独消融，避免与 `seg_bnd` 混淆。

---

### 26.10 推荐最小 smoke 命令（跑通再长训）

```python
from ultralytics import YOLO

YOLO("yolo26s-seg-pointrend.yaml").train(
    data="/path/to/coconut-s-seg.yaml",
    epochs=1,
    batch=8,
    imgsz=640,
    seg_point=1.0,
    seg_point_refine=True,
    seg_point_boundary=True,
    seg_point_roi=0.0,
    seg_point_num=64,  # 先降 P，稳定后再 112
    overlap_mask=False,  # 与 COCONut 消融脚本习惯一致时可显式指定
    project="runs/segment",
    name="pointrend-boundary-refine-smoke",
)
```

**通过标准：** loss 有限、无 DDP unused、1 epoch 结束有 ckpt；对比 `seg_point_boundary=False` 同 seed 的 loss 曲线（非 AP）。

---

### 26.11 问题优先级汇总

| 级别 | 问题                      | 是否阻塞启动训练 | 缓解                              |
| ---- | ------------------------- | ---------------- | --------------------------------- |
| 🔴   | 推理不用 MLP              | 否（训练可跑）   | AP75 验收；未来做 (I)             |
| 🔴   | GT 160 栅格上限           | 否               | 接受上限；或改 mask_ratio（大改） |
| 🟡   | P3 非 fused feat          | 否               | task #9                           |
| 🟡   | o2one feats detach        | 否               | 架构既定                          |
| 🟡   | boundary×uncertainty 冲突 | 否               | 调 importance_ratio / 消融        |
| 🟡   | 显存/算力                 | **可能**         | 降 batch/P；关 distill            |
| 🟡   | 工具链/消融缺口           | 否               | 补 G4mb preset、联调单测          |
| 🟢   | seg_bnd 重复              | 否               | 分 ablation 组                    |

**结论：** 源码路径 **已闭合**，`seg_point_refine=True` + `seg_point_boundary=True` 可以启动训练；主要风险是 **验收方式、GT/分辨率上限、显存、以及缺少 boundary+MLP 的实证消融**，而非明显的 forward/loss 逻辑错误。

---

**结论：** 源码路径 **已闭合**，`seg_point_refine=True` + `seg_point_boundary=True` 可以启动训练；主要风险是 **验收方式、GT/分辨率上限、显存、以及缺少 boundary+MLP 的实证消融**，而非明显的 forward/loss 逻辑错误。

---

## 27. 训练流程全图（2026-07-05 重梳）

按 **时间顺序** 走一遍当前 seg 训练代码，并在每一层标注与 PointRend / boundary / refine 相关的问题。

### 27.1 总览

```text
YOLO.train
  → SegmentationTrainer._do_train
       _setup_train: model / DistillationModel / DDP / EMA / dataloader
       每 batch: preprocess_batch → forward(img) → loss(batch,preds) → backward
       每 epoch: validator(EMA.eval) → mAP + val/seg_loss(日志)
```

### 27.2 阶段 A — 模型构建与 criterion

| 步骤          | 文件                   | 要点                                  |
| ------------- | ---------------------- | ------------------------------------- |
| `get_model`   | `segment/train.py:58`  | `SegmentationModel(yaml)`             |
| YAML→Module   | `tasks.py:parse_model` | Segment26 第 4 参 → `point_hidden`    |
| Head          | `head.py:462`          | 条件构建 `PointHeadMLP`               |
| Criterion     | `tasks.py:592`         | `E2ELoss(self, v8SegmentationLoss)`   |
| 蒸馏包裹      | `trainer.py:370`       | `DistillationModel`；**compile 禁用** |
| hyp 注入      | `detect/train.py:158`  | `model.args = self.args`              |
| Criterion hyp | `loss.py:352`          | `self.hyp = model.args`               |

**问题 A1–A4：** finetune≠resume（MLP 新键）；criterion 懒创建；蒸馏双 forward 增显存且不蒸馏 point_head；改 cfg 须重建 criterion（resume 时 `trainer.py:1118-1122` 已处理 E2E）。

### 27.3 阶段 B — 数据 batch

| 步骤                     | 产出                                                  |
| ------------------------ | ----------------------------------------------------- |
| `Format`（`augment.py`） | `masks` @ **H/4×W/4**；`overlap_mask` id map 或 stack |
| `preprocess_batch`       | img `/255`；`multi_scale` 只缩放 img                  |

**问题 B1–B4：** 160 栅格 GT 上限；默认 `overlap_mask=True` 与 COCONut 消融习惯可能不一致；DALI 不 offload polygon2mask。

### 27.4 阶段 C — forward（训练）

```text
Detect.forward:
  one2many ← forward_head(P3,P4,P5)     # feats 有 grad
  one2one  ← forward_head(P3.detach…)  # feats 无 grad
Segment26:
  proto → 写入 o2m / o2o(detach)
  training & point_head → point_refine_dummy (=0)
```

**问题 C1–C4：** DDP dummy；one2one detach；P3 80×80 非 fused feat；val/eval 不 stash dummy（无 backward，一般安全）。

### 27.5 阶段 D — loss（训练核心）

```text
E2ELoss: o2m.loss + o2o.loss（加权 0.8→0.1 / 0.2→0.9）
v8SegmentationLoss.loss:
  TAL → fg_mask, target_gt_idx, target_bboxes(GT box)
  proto 上采样至 mask 尺寸（若不一致）
  calculate_segmentation_loss → single_mask_loss（逐 fg anchor）
    BCE + [comp][bnd] + [point: ROI/boundary采样 → MLP? → focal+dice]
  loss[1] += dummy; loss[1] *= hyp.box(7.5)
```

**问题 D1–D6：** gain 链；同一 GT 多 anchor 重复 point 计算；boundary×uncertainty 冲突；seg_loss 不可分；无 fg 占位；NaN/OOM 恢复（recipe200 ep108）。

### 27.6 阶段 E — backward / EMA

`trainer.py:484-516`：DDP loss 缩放 → AMP backward → grad clip → EMA；每 epoch `E2ELoss.update()`。

**问题 E1–E3：** point_head 在 o2o 分支无 P3 梯度；compile+DDP 靠 dummy，boundary+MLP+compile 缺专项验证。

### 27.7 阶段 F — 训练内 validation（关键分叉）

```text
validator.py:161-167  → EMA.eval()
preds = model(img)    → predict 路径（非 training dict）
val/seg_loss          → criterion 仍算 point（detach，无 backward）
postprocess           → process_mask（无 point_head）→ mask mAP → fitness → best.pt
```

**问题 F1–F4（最重要）：** **fitness/checkpoint 由 inference mask mAP 驱动，与训练 MLP 路径脱节**；高 IoU 档才可能有信号；predict 边界不变是预期行为。

### 27.8 阶段 G — checkpoint

| 场景                    | 注意                                   |
| ----------------------- | -------------------------------------- |
| 旧 seg → pointrend yaml | `resume=False` finetune；MLP zero-init |
| 同结构 resume           | E2E criterion 重建                     |
| Distill resume          | teacher 重建；projector 从 ckpt 恢复   |

### 27.9 问题矩阵（按流程阶段）

| 阶段       | 代码级问题                                     |    阻塞开训？    |
| ---------- | ---------------------------------------------- | :--------------: |
| A 构建     | finetune/resume 混用；distill+point OOM        |       可能       |
| B 数据     | 160 GT；overlap 默认值                         |        否        |
| C forward  | o2one detach；P3 分辨率                        |        否        |
| D loss     | gain 链；per-anchor 重复；boundary×uncertainty |        否        |
| E backward | o2o 无 backbone grad；compile                  |        否        |
| F val      | **mAP 不用 MLP**；fitness 错位                 | 否（但误导验收） |
| G ckpt     | 结构变更 migration                             |    配置错误时    |

### 27.10 推荐开训配置（PointRend+boundary+refine）

```python
# 注意：加载旧权重的正确参数是 pretrained=<ckpt>（trainer.setup_model 的 pretrained 分支），
# 不存在 weights= 这个 train kwarg。
YOLO("yolo26s-seg-pointrend.yaml").train(
    pretrained=".../best.pt",
    resume=False,
    seg_point=0.5,
    seg_point_refine=True,
    seg_point_boundary=True,
    seg_point_roi=0.0,
    seg_point_num=64,
    distill_model=None,
    compile=False,
    name="pointrend-boundary-refine-finetune",
)
```

验收：同 seed 对比 `seg_point_boundary=False`；看 **mask mAP75-95** 与 `val/seg_loss`，不看 predict 边界。

---

## 28. 续训练落地记录（2026-07-05）：recipe200 → pointrend-ft60

**入口**：`scripts/finetune_yolo26s_seg_pointrend_coconut_b.py` → run `yolo26s-seg-coconut-b-v2-pointrend-ft60`。

**初始化链路**：`YOLO("yolo26s-seg-pointrend.yaml").train(pretrained=recipe200 best.pt)`。ckpt 是 DistillationModel 包裹（ep107，mask mAP50-95=0.376），按名/形状迁移 **844/850** 参数；6 个 `point_head` 新参数 zero-init，step-0 行为与 recipe200 best 完全一致。**不能 resume**：结构变更后 optimizer state 不匹配。

**配置要点**：无蒸馏、60 epoch、batch 84、MuSGD lr0=0.003 + cos_lr、multi_scale 0.15（recipe200 在 batch 90 / ms 0.25 时 Size=800 OOM）、`seg_point=0.5` + `seg_point_refine` + `seg_point_boundary`、`seg_point_num=64`、`seg_point_o2o=0`（one2one feats 本就 detach）、`overlap_mask=True`（延续 recipe200）。

**上线前连环排障（三个真实 bug，均已修复 + 单测）**：

| #   | 崩溃点                                         | 根因                                                                                                                                      | 修复                                                                           | 单测                                                                    |
| --- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| 1   | `DistillationModel.__init__: teacher.to(None)` | ckpt 保存时剥离 teacher；finetune 未配 `distill_model`，setup_model 仍按蒸馏重建                                                          | `setup_model` 无 `distill_model` 时解包 `student_model`                        | `test_setup_model_unwraps_distill_ckpt_without_distill_model`           |
| 2   | point loss OOM（31.3 GB）                      | `point_feats` 按实例 `expand(n,C,H,W)` 后过 grid_sample：forward 零拷贝，**backward 实体化整块连续梯度**（600 实例 ×128×80×80 ≈ 2 GB/图） | 传共享 `(1,C,H,W)` + 坐标合并为 `(1, N*P, 2)` 一次采样再 reshape，两侧数值等价 | `test_single_mask_loss_gain_interaction`（shared vs expanded allclose） |
| 3   | `muon.py: assert len(G.shape)==2`              | `muon_update` 只 reshape 4D 卷积；`PointHeadMLP` 的 Conv1d 权重是 **3D**，进 muon 组直接断言失败                                          | ndim>2 统一展平 `(out, -1)`；3D 的 scale 用展平维度（4D 保持旧口径）           | `test_musgd_muon_update_handles_conv1d_weights`                         |

**经验**：给模型加任何新模块时，除了 loss/DDP，还要过一遍 **优化器参数分组**（MuSGD 按 ndim 路由）和 **ckpt 加载路径**（蒸馏包裹、结构差异）这两个隐蔽维度。

---

## 29. ft60 验收结果（2026-07-05/07）：双标尺 + val_twice + nobnd 对照

> **完整表格与 ckpt 路径**见 [实验结果总结](yolo26s-seg-pointrend-experiment-results.md)。

### 29.1 训练结果

- `pointrend-ft60`（+bnd）：60/60，peak **mask mAP50-95=0.3958 @ep52**
- `pointrend-ft60-nobnd`：60/60，peak **mask mAP50-95=0.3976 @ep51**（训练 val 略优）
- 相对 recipe200 0.376：**+2.0pt**；box 同步 +2.5pt

### 29.2 双标尺验收（仅 +bnd 已跑，eval batch=16）

| 标尺 | ft60 (+bnd) | 官方 yolo26s-seg |
|------|-------------|------------------|
| COCONut v2 val | **0.381** ✅ | 0.3495 |
| COCO val2017 | 0.3588 ❌ | 0.3859（缺口较 recipe200 减半） |

### 29.3 val_twice：推理细分负收益

indirect **0.3810** vs direct 0.3573 → **Δ −2.4pt**。部署保持 `seg_point_refine_infer=False`。

### 29.4 boundary 对照结论

nobnd 全程略优于 +bnd（peak 0.3976 vs 0.3958）；**建议默认 `seg_point_boundary=False`**。nobnd 独立双标尺待补。

---

*维护：PointRend 代码或 cfg 变更时，同步更新 §5–§29 各表与问题清单；指标以 `yolo26s-seg-pointrend-experiment-results.md` 为准。*
