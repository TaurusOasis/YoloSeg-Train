# PointRend Point-Refine Head 设计文档（YOLO26s-seg）

> 日期：2026-07-04（代码走读见 [新手入门教程](yolo26s-seg-pointrend-beginner-guide.md) §14–§25）
> 目标：归档"在 YOLO26s-seg 上增加 PointRend-style point refine head"的代码梳理与实现设计——区分训练侧 point-head MLP 与推理侧迭代细分，定位承载改动点、导出可行性、蒸馏/loss_names 契约，并给出最小风险的落地顺序。
> 状态：**训练侧 (T) + 推理/验证侧 (I) 细分均已落地实现**（见 §3、§10）；2026-07-05 追加 refine/check：E2E one2one 点损失/MLP 可独立控，PointRend zero-init 已对齐为 `process_mask` no-op，`seg_point_refine_infer=True` 时 predict 与标准 val 均可显式走 MLP 细分，recipe200 baseline 仍停 epoch 107 且 ckpt 不含 `point_head`（见 §11）。剩余待办为消融与实验（见 §10.2）。本文推翻上一轮"strategy-1 lite：无 MLP / 无细分"的约束（见 `~/.claude/.../memory/seg-loss-subgains-strategy1.md`）。
> 实现差异（vs 原设计，详见 §2.5/§2.6/§2.7/§2.9/§10）：DDP 用"forward `zero_loss()` 占位 + criterion 逐实例 MLP"混合解法（compile-safe，方案 A1 已不需要）；运行时 `seg_point_refine` cfg 已启用（由 dummy 解耦 desync）；`PointHeadMLP` 用 Conv1d + 恒等残差 zero-init；细特征用 `preds["feats"][0]` 不新增键；**§2.6 ROI 采样已落地**（`get_uncertain_point_coords_in_roi` + `seg_point_roi` cfg，默认 bbox-restricted）；**§2.6 进阶 GT Sobel boundary band 加权采样已落地**（`_weighted_rand_in_roi` + `seg_point_boundary` cfg，默认关闭、强制走 ROI）；`head.py` Segment26.forward 的 `point_head` 访问已加 `getattr` 守卫以兼容 pre-PointRend teacher ckpt（修复 recipe200 dis=6.0 蒸馏崩溃）。
> Changelog：2026-07-04 经五轮评审纠正 P1–P6（YAML/resume/ROI/staticmethod/feats 键/DDP unused/模式开关/ckpt migration/显存/GT 上限），见各节正文；§2.6 ROI 采样已实现（新增 ROI 采样器 + cfg + 单测）；§2.6 进阶 GT Sobel boundary band 加权采样已实现（`_weighted_rand_in_roi` + `seg_point_boundary` cfg + 单测，代码追平前向书写的设计）；同日修 `head.py` Segment26.forward 对 pre-PointRend teacher ckpt 的 `point_head` 缺失崩溃（`getattr` 守卫，见 §10.1）；**§2.6 进阶混合候选池修复**（oversample 50/50 边界加权 + bbox 均匀、余量改 bbox 均匀、top-k 在合并池上做，消解边界加权 × pred-uncertainty top-k 的监督目标冲突，见 §2.6）；**(I) 推理 subdivision 已落地**（`process_mask_pointrend` + SegmentationPredictor 接线 + `seg_point_refine_infer`/`seg_point_subdiv_k` cfg + 单测，PyTorch-only、导出自动禁用，见 §3）；2026-07-05 refine/check：`process_mask_pointrend` 改为 crop-at-proto → full-res ROI delta scatter，zero-init 与标准 `process_mask` bitwise 等价；新增 `seg_point_o2o` / `seg_point_refine_o2o` / `e2e_final_o2m`，补 one2one detach 下的边界监督强度控制；同日补 `SegmentationValidator` PointRend 分叉，显式 `seg_point_refine_infer=True` 时标准 val 输出 full-res refined masks 并同步 GT mask 分辨率。

---

## 0. 背景：现有 seg point 子项（PointRend-Lite）

`v8SegmentationLoss.single_mask_loss`（`ultralytics/utils/loss.py`，`@staticmethod` 约 `:574` 起）已含 `seg_point` 子项，是 **PointRend 的训练侧骨架，默认 Lite（无 MLP）、无细分推理**：

- 不确定性点采样：`get_uncertain_point_coords_with_randomness`（`utils/mask_point_sampling.py`，full-grid [0,1]²），`no_grad` 包裹坐标采样 + GT 重采样。
- `pl = point_sample(pm4, coords, ...)` —— Lite 模式下**预测直接来自粗 einsum logits**，没有 MLP refine；MLP 模式（`point_hidden>0` 且运行时 `seg_point_refine=True`）下 `pl = point_head(细特征, 粗 logit)`（§2.1/§2.7，已落地）。
- 损失：focal（α=0.25,γ=2）+ dice，逐实例，折进 `seg_loss` 档。
- 默认 gain 0 → 短路到与 legacy BCE 位等（G3-Eq1），不影响在途训练。

PointRend 全量 = 这套骨架 **+ 两个部件**：(T) point-head MLP 替换 `pl` 来源；(I) 推理侧迭代细分。

---

## 1. PointRend 的两个独立部件

| 部件                      | 作用                                                     | 现状                                                                                     | 本文决策                                                                                                                                                                                |
| ------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(T) Point-head MLP**    | 训练时在采样点上把"细特征(+粗logit)" refine 成逐点 logit | Lite 默认无 MLP（`pl` 取粗 logits）；`point_hidden>0` + `seg_point_refine=True` 时走 MLP | **本轮加并已落地**；MLP 在 `single_mask_loss` 内逐实例跑（§2.1）；DDP 用 forward `zero_loss()` 占位 + criterion 真梯度混合解法、compile-safe（§2.5）                                    |
| **(I) 迭代细分推理/验证** | 粗掩码→找不确定点→MLP refine→散点修正，循环 K 次         | 已有标准 `process_mask`；PointRend 为可选 PyTorch-only 后处理                            | **已落地但默认关闭**（见 §3、§10）：`seg_point_refine_infer=True` + PyTorch backend 时启用；predict 还要求 `retina_masks=False`；导出和 native JSON/TXT 路径保持标准/native mask 后处理 |

二者可独立上线。推荐先只做 (T)，(I) 收益边际且仅 PyTorch 推理可用、对导出用户无增益。

---

## 2. 训练路径：Point-head MLP + point-refine loss

### 2.1 基线方案 B：MLP 在 `single_mask_loss` 内逐实例运行

PointRend 训练本质是"逐实例"在不确定点上 refine。现有 `seg_point` 已在 `single_mask_loss` 的 point 分支（约 `loss.py:655-700`）内对**每个正实例**做不确定性采样 + focal+dice，只是 Lite 模式下 `pl` 取自粗 einsum logits（`loss.py:692` 的 `coarse`、`:697` 的 `pl=coarse`）。加 MLP 的承载改动 = 把 `pl` 来源从"粗 logits 现采"换成 `point_head(细特征, 粗 logit)`，**仍在 `single_mask_loss` 内、仍逐实例、仍 `no_grad` 包裹坐标采样**。这条路径与现有 `seg_point` 结构完全一致、改动最小、保留逐正样本不确定性采样。

```python
# single_mask_loss 内 point_w>0 分支。pred_mask=粗 einsum (n,H,W)；
# point_feats_i 已由 calculate_segmentation_loss 从 preds["feats"][0] 取该图并 expand 到 (n,Cf,Hf,Wf)
pm4 = pred_mask.float().unsqueeze(1)
gm4 = gt_mask.float().unsqueeze(1)
with torch.no_grad():
    coords = get_uncertain_point_coords_with_randomness(
        pm4.detach(), calculate_uncertainty, num_points, oversample_ratio, importance_ratio
    )  # (n,P,2)
    pg = point_sample(gm4, coords, align_corners=False).squeeze(1)
coarse_p = point_sample(pm4, coords, align_corners=False).squeeze(1)  # (n,P) 粗 logit 辅助输入
if point_head is not None and point_feats_i is not None:
    feat_p = point_sample(point_feats_i, coords, align_corners=False)  # (n,Cf,P)
    pl = point_head(feat_p, coarse_p)  # (n,P) refined，带 grad
else:
    pl = coarse_p
total = total + point_w * (point_sigmoid_focal_loss_per_instance(pl, pg) + point_dice_loss_per_instance(pl, pg)).sum()
```

`pred_mask`（粗 einsum）继续用于不确定性采样与现有 BCE 项；`pg`/focal/dice/`total` 形式不变。**不确定性采样应限制到 ROI（见 §2.6），否则继承 seg_point 的全图背景采样弱点。**

### 2.2 细特征来源：直接用 `preds["feats"][0]`，无需新增键

`Detect.forward_head` 已把各层 neck 特征图 `feats=x` 塞进 preds dict（`head.py:158`），loss 里已用 `preds["feats"][0]` 算 imgsz（`loss.py:531`）。**直接复用 `preds["feats"][0]` 即可，无需新增 `point_feats` 键**，MLP 不在 forward 跑、forward 不需任何改动。

**⚠ end2end one2one 的 detach 是现有契约，但对 boundary/refine 是监督强度风险**：`Detect.forward` 对 one2one 分支用 `x_detach = [xi.detach() for xi in x]`（`head.py:166`），`Segment26.forward` 也把 one2one `proto` detach（`head.py:472-474`）。直接用 `preds["feats"][0]`（按 branch 取）即继承该语义：one2many 的 point/refine loss 可反传到 P3/backbone/proto，one2one 的 point/refine loss 只稳定更新 one2one mask coeff 与 `point_head`，不会经 P3/proto 更新 backbone。这与 E2E 原设计一致，但在 boundary/refine finetune 中会让后期特征塑形变弱（默认 o2m 0.8 → 0.1、o2o 0.2 → 0.9）。本轮补齐三个控制项：

- `seg_point_o2o`：one2one 分支的 `seg_point` 乘子；设 `0` 可做 one2many-only point supervision。
- `seg_point_refine_o2o`：one2one 是否跑 MLP；设 `False` 时 one2one 点损失退回 Lite（`pl=coarse`），避免用 detach P3 训练 refine MLP。
- `e2e_final_o2m`：E2E schedule 的最终 one2many 权重；边界/refine finetune 可从默认 `0.1` 提到 `0.3` 左右，保留更多带 backbone 梯度的 one2many 监督。

不建议手动把 raw `x[0]` 挂给 one2one：那会绕开现有 detach 设计，让 one2one 点 head 反传到 backbone（与 proto/box 的 one2one 语义不符）。若要真正研究 one2one backbone gradient，应作为单独 E2E 策略开关，而不是在 PointRend loss 里偷偷绕过。

**P3 是合理的第一版 fine feature，但不是 Segment26 proto 的唯一输入**：

- **旧 `Segment`**：`self.proto(x[0])`（`head.py:370`）→ Proto 吃 raw `x[0]` = P3 (80×80, stride 8)，此时 `feats[0]` 确实就是 Proto 的输入。
- **`Segment26`**：`self.proto(x)` 把整个 [P3,P4,P5] 喂 `Proto26`（`block.py:1980` 类、`:1999` `feat = x[0]`），内部 `feat = x[0] + up(x[1]) + up(x[2])` 多尺度融合。故 P3(`feats[0]`) 是**最高分辨率基础特征、也是 Proto26 融合的 base feature**——作第一版 fine feature 合理；但**不是** Proto26 实际用的融合特征。
- **后续可对比**：proto 特征（Proto26 内部 fused `feat`，需 Proto26 暴露、且要正确处理 one2one detach）或 fused proto 前特征，看是否比 raw P3 更涨点。第一版先用 `feats[0]` 起步。

criterion 在 `loss.py:513` 一并取出 `preds["feats"][0]`（`point_feats = preds["feats"][0] if hyp_get("seg_point_refine",...) and self.point_head is not None else None`），沿 `calculate_segmentation_loss → single_mask_loss` 透传（per-image 取出，即下文的 `point_feats_i`）。

### 2.3 `single_mask_loss` 是 staticmethod —— 接线方式（P2）

`single_mask_loss` 现为 `@staticmethod`（当前约 `loss.py:574`），**无法 `self.point_head`**。两条接法：

- **(推荐) 显式传参**：`calculate_segmentation_loss`（已是实例方法）持有 `self.point_head` 引用（在 `v8SegmentationLoss.__init__` 设 `self.point_head = getattr(model.model[-1], "point_head", None)`），把 `point_head` 与 per-image `point_feats_i` 作为新形参传入 `single_mask_loss`。保持 staticmethod 不动，改动最小。
- (备选) 去 staticmethod 改实例方法：则 `self.point_head` 可用，但 `single_mask_loss` 调用点（`loss.py:762-778`）需改 `self.single_mask_loss(...)`，且 `v8SegmentationLoss` 需持有 point_head 引用。

**采用显式传参**。注意 `v8SegmentationLoss.__init__`（`loss.py:495`）现只从 `model` 取 `stride/nc/hyp`，新增 `self.point_head = getattr(model.model[-1], "point_head", None)`（`:502`，仅当 head 含 point_head 时非 None）。

### 2.4 逐实例 feature batch 维扩展（显存/正确性，关键细节）

`point_feats_i` 是 per-image 张量 `(1,Cf,Hf,Wf)` 或 `(Cf,Hf,Wf)`，**共享于该图所有实例**；而 `coords` 是逐实例 `(n,P,2)`。`point_sample` 要求 input 与 coords 的 batch 维对齐 → **必须把细特征扩成 `(n,Cf,Hf,Wf)`**：

- 用 `expand`（广播 view，**不复制显存**），**不要** `repeat`/`tile`（会实打实占 `n×Cf×Hf×Wf`）。
- 显存仍按 `n×Cf×Hf×Wf` 计入 autograd 中间量（`point_sample` 输出 `(n,Cf,P)`，再喂 MLP）。`n`=正实例数（几十），`Cf`=融合 feat 通道，`Hf=Wf=80` → 单图量级可控，但 **N 大/多目标时需核算峰值**，必要时对 `point_feats_i` 做 stride-4 下采样再采样（PointRend 原版亦在较低分辨率 feat 上采点）。
- 正确性陷阱：若忘扩维，`point_sample` 会广播错位或静默产出错误逐点特征。

### 2.5 DDP / torch.compile（已落地：forward 占位 + criterion 逐实例）

实际实现采用**混合解法**（比原"方案 B vs A1"二选一更优雅，且 compile-safe）：

- **MLP 真正跑在 criterion**（方案 B，逐实例、保留逐正样本不确定性采样）——产生真实梯度。
- **同时在 `Segment26.forward` 内跑 `point_head.zero_loss()` 占位**（`head.py`）：训练态 `if self.training and getattr(self, "point_head", None) is not None` 时算 `dummy = self.point_head.zero_loss()`，stash 进 `preds["one2many"/"one2one"]["point_refine_dummy"]`（或非 end2end 的 `preds["point_refine_dummy"]`）；loss 侧 `loss[1] = loss[1] + point_refine_dummy`（`loss.py`）。用 `getattr(..., None)` 而非裸 `self.point_head` 是为 **pre-PointRend teacher ckpt** 兜底：`torch.load` 经 `__setstate__` 重建 `nn.Module`、**不重跑 `__init__`**，故旧 `yolo26x-seg.pt` teacher 的 Segment26 `__dict__` 里**没有 `point_head` 键**；而 `_forward_teacher_for_distillation`（`distill_model.py`）会把 teacher head `training=True` 后跑 forward → 裸 `self.point_head` 会 `AttributeError` 崩溃（recipe200 dis=6.0 即此配置）。`getattr` 守卫让无 `point_head` 的 teacher 安全跳过 dummy（值同 None 分支），对新建/有 `point_head` 的模型零行为变化。
- `zero_loss()`（`head.py` PointHeadMLP）= `sum((p.sum()*0.0 for p in self.parameters()), ...)`——**连接到 point_head 所有参数、梯度为 0**，不污染 loss，但使参数**在 DDP forward 图内可达** → DDP 不报 unused、`find_unused_parameters=False`/`static_graph=True`（compile）路径也安全。

机制：dummy 使参数"在 forward 被使用"满足 DDP/static_graph 一致性检查；真实梯度来自 criterion 的逐实例 MLP 调用（criterion 跑在 unwrapped 模型，但参数是同一对象，autograd hook 触发 DDP allreduce）。最终参数梯度 = 真实梯度（criterion）+ 0（dummy）= 真实梯度。

**这同时解决了 P1**：point_head 已建但 `seg_point=0` / 无 fg / `num_points=0` / `seg_point_refine=False` 任一导致 criterion 不调 MLP 时，dummy 仍在 forward 跑 → 参数仍可达 → DDP 安全（该 step 参数获 0 梯度，等价冻结，但无 unused 报错）。条件性构建（无第 4 参→`point_head=None`）仍保留，用于"根本不需要 head"的 Lite 模型。

**仍必做**：~~实现后 2-GPU 梯度同步核验~~ **已完成**（`scripts/smoke_point_head_ddp.py`，§10.2）；compile/静态图路径若启用可再补一轮。

**方案 A1（前移 forward）已不需要**——dummy 已覆盖 compile/静态图安全性，且保留了 B 的逐正样本采样优势。仅作未来若需把点 refine 进导出图时的备选。

### 2.6 不确定性采样限制到 ROI（P1，**已落地：bbox+margin ROI 采样器**）

> **实现状态（2026-07-04）**：已落地。新增 `get_uncertain_point_coords_in_roi`（`utils/mask_point_sampling.py`），把不确定性采样 + 随机余项都限制到**逐实例 bbox（按 `margin` 扩张、clamp 到 [0,1]）**内，不再全图采。由 cfg `seg_point_roi`（FLOAT，默认 `0.0`）控制：`>=0` → ROI 采样（`0.0`=精确 bbox，`>0`=扩张 margin）；`<0` → 退回 legacy `get_uncertain_point_coords_with_randomness` full-grid [0,1]² 采样（供 G0–G6 消融与旧 seg_point 对照）。`single_mask_loss` point 分支按 `roi_margin`/`boundary_w` 调度，`calculate_segmentation_loss` 透传 `roi_margin=float(self.hyp_get("seg_point_roi", 0.0))` 与 `boundary_w=bool(self.hyp_get("seg_point_boundary", False))`。
>
> **默认行为变更**：`seg_point=0` 时 point 分支整体短路 → 对在途 baseline 零影响；一旦开启 `seg_point>0`，默认即走 ROI（`seg_point_roi=0.0`），不再继承 full-grid 背景采样弱点。退化 bbox（x2<=x1 或 y2<=y1，margin clamp 后）逐实例回退 full-grid，避免空域采样。

`single_mask_loss` 已有 `xyxy`（mask 像素坐标），归一化为 `boxes_norm = xyxy / [W,H,W,H]`（`W,H = pred_mask.shape[-2:]`）后喂 ROI 采样器。

- **第一版（已落地）**：bbox+margin 限制（`_rand_in_roi` 在 `[x1,x2]×[y1,y2]` 内均匀采，不确定性 top-k + 随机余项均限 ROI）。
- **进阶（已落地，默认关闭）**：GT Sobel 边界 band 加权采样（复用 `mask_boundary_loss.py` 的 `sobel_magnitude`）由 `seg_point_boundary=True` 开启；它会强制走 ROI，并把 GT Sobel magnitude 作为边界加权的依据。
- **监督目标冲突与修复（2026-07-04 评审）**：初版把 oversample 候选池整池按 Sobel multinomial 抽，导致 pred-uncertainty top-k（默认 75%）只能在「边界附近候选」里选——bbox 内部错但不确定的点（远离 GT 边界的 FP/FN 区域）几乎进不了候选池（Sobel≈0 → multinomial 权重≈0），只能靠 25% 随机余量覆盖，且余量本身也按边界加权，比 legacy 全图 / 纯 ROI-uniform 的内部覆盖更弱。即「强化真边界附近学 pred」与「弱化内部误检区域 point 监督」冲突。修复为**混合候选池**：oversample = 50% `_weighted_rand_in_roi`（Sobel 边界加权）+ 50% `_rand_in_roi`（bbox 内均匀）合并后做 pred-uncertainty top-k，25% 随机余量改为**始终** bbox 内均匀（不再边界加权）。边界仍被过采样（约 5× 其像素占比）以聚焦边界 point 监督，但内部不确定点重新可被 top-k 选到——消解冲突。无新增 cfg（固定 50/50 混合）。
- 注意：限制 ROI 后 `num_points` 的有效点更稀疏，需相应调大 `oversample_ratio`。**消融必须固定 `seg_point_boundary`**（它与 ROI/uniform 的 point 监督语义不同，不可自由组合）。

### 2.7 PointHeadMLP 结构（已落地：Conv1d + 恒等残差 zero-init）

实际实现（`head.py` PointHeadMLP）：

- **Conv1d 而非 Linear**：`nn.Conv1d(Cf+1, hidden, 1) → ReLU → Conv1d(hidden, hidden, 1) → ReLU → Conv1d(hidden, 1, 1)`。输入 `(N, Cf, P)`——在 P（点）维上做 1×1 卷积，等价逐点 MLP，但天然适配 `(N,C,P)` 张量、无需展平。
- **恒等残差 zero-init**：末层 `Conv1d` 的 weight/bias 全 `zero_init` → 初始 `delta=0` → `refined = coarse + delta == coarse`。训练初始即"不改粗 logit"，再逐步学 refine（对应 §6.1 的 zero-init 建议）。
- `forward(point_feats (N,Cf,P), coarse (N,P) or (N,1,P))`：cat 沿通道维 → MLP → `delta` → `return coarse + delta`。`point_feats.float()`/`coarse.float()` 强制 fp32（与 §9 P5 的 fp32-internal 一致）。
- `zero_loss()`：`sum((p.sum()*0.0 for p in self.parameters()), start=...)`——DDP 占位用（§2.5）。
- `in_channels = ch[0]`（P3 通道），`hidden_channels = point_hidden`（YAML 第 4 参，经 `make_divisible` 缩放，见 §5）。

### 2.8 验收口径：拆开 train / val / predict 的 Point-head 参与方式

当前代码不是"所有路径都跑 MLP"，而是三条路径各有边界：

- **train**：`v8SegmentationLoss.single_mask_loss` 在 point 分支里跑 `point_head(pf, coarse)`（当 `seg_point>0`、`seg_point_refine=True`、模型有 `point_head`，且当前 branch 未被 `seg_point_o2o`/`seg_point_refine_o2o` 关掉）。这是训练侧辅助监督。
- **val**：默认仍走标准 mask 组装（`coeff @ proto → crop/upsample → threshold`），此时 val/best.pt fitness 衡量的是训练侧 point/refine 对 backbone/proto/coeff 的**间接收益**，可能低估 MLP 后处理的直接收益。显式传 `seg_point_refine_infer=True`、PyTorch 模型有 `point_head` 且不是 `save_json/save_txt` native 路径时，`SegmentationValidator` 会走 `process_mask_pointrend`，预测 mask 输出 full-res，并把 GT mask 保持在同一 full-res 尺寸做 IoU。
- **predict**：默认仍标准 `process_mask`；只有 PyTorch backend、`seg_point_refine_infer=True`、`retina_masks=False`、且模型有 `point_head` 时，`SegmentationPredictor` 才走 `process_mask_pointrend`。`retina_masks=True` 与导出后端自动回退标准路径。

因此验收现在有两个分叉：

1. **标准 fitness 分叉**：`seg_point_refine_infer=False`。训练消融的主指标仍看 **mask AP75/AP95**（高 IoU 阈值档）和 Mask mAP50-95，排除推理后处理变量，衡量 point/refine 对标准 `coeff @ proto` mask 的间接收益。`Metric.map75` 已存在；AP95 可从 `metrics.seg.all_ap[:, 9].mean()` 读。
2. **PointRend 直接验收分叉**：`seg_point_refine_infer=True`。用于回答"MLP 参与 val/predict 后边界是否更稳"。这条分叉会改变 val mask 分辨率和后处理 latency，不应和默认 fitness 直接混算；建议另起 `model.val(..., seg_point_refine_infer=True, save_json=False, save_txt=False)` 与 predict 可视化/latency 一起报告。

**为什么 best.pt fitness 必须留在标准分叉（`seg_point_refine_infer=False`）**：`process_mask_pointrend` 的细分采点走 `get_uncertain_point_coords_in_roi`，内部有 `torch.rand` —— **随机**。每次 val 的掩码会随采点不同而抖动 → fitness 抖动 → best.pt 选择 / early-stop 被细分随机性污染、不可复现。标准 `process_mask` 是确定性的。因此 best.pt/checkpoint 选择必须站在确定性度量上；MLP 直接收益应当用**事后、固定种子、可重复**的度量验收，而不是混进训练 fitness。注意 `segment/train.py` 的 `get_validator` 用 `args=copy(self.args)` 把 cfg 透传给 validator，故 `seg_point_refine_infer` 是**单一开关**统一控制 train-val / standalone-val / predict 三条路径——训练内 val 同样会响应它；推荐训练期保持 `False`，事后单独打开。

**推荐"val twice"验收协议（解验收分叉）**：

1. 训练内 val：`seg_point_refine_infer=False`（ft01 现状，保持）→ best.pt/fitness 走确定性 `process_mask`，checkpoint 选择不受细分随机性影响，且与训练日志里的 mask mAP 可对齐校验。
2. ft01 结束后，对 `best.pt`（与 `last.pt`）各 val 两次：`seg_point_refine_infer=False`（间接基线，应≈训练日志）vs `=True`（直接收益，K 次细分，MLP 真正参与），比 `metrics.seg.map50-95` / `all_ap[:,9].mean()`(AP95) / AP75。**Δ = MLP 的直接收益**，可归因（不会和"选了不同 checkpoint"混淆）。
3. 直接那次为压住细分随机性，固定 `torch.manual_seed` 或跑 3 次取均值再比；同时报告 predict 可视化边界锐度与 latency（细分 K 次的代价）。

若两条分叉的 `AP75/AP95` 都不涨而仅 `AP50` 涨，说明 point/refine 主要改善了粗 mask 或检出，不一定改善边界，应回到 §2.6 的 ROI/boundary band 采样、§2.2 的 P3 分辨率、§4 的 one2one detach 权重检查。

### 2.9 Lite vs MLP 模式切换 + 与 seg_bnd/seg_comp 的关系（P2/P3）

**模式开关（已落地：构建级 + 运行时 `seg_point_refine`，二者由 dummy 解耦）**：

- **构建级**（YAML 第 4 参 `point_hidden`）：存在 → 构建 `PointHeadMLP`；不存在 → `self.point_head=None`（Lite，`pl` 取粗 logits）。`Segment26.__init__`：`self.point_head = PointHeadMLP(ch[0], point_hidden) if point_hidden > 0 else None`。
- **运行时**（cfg `seg_point_refine: bool`）：loss 里 `point_feats = preds["feats"][0] if self.hyp_get("seg_point_refine", False) and self.point_head is not None else None`；`single_mask_loss` 内 `if point_head is not None and point_feats is not None: pl = point_head(pf, coarse) else: pl = coarse`。
    - **四种组合**：head 未建 → 必 Lite；head 已建 + `seg_point_refine=True` → MLP；head 已建 + `seg_point_refine=False` → **Lite（`pl=coarse`）但 head 仍在 forward 跑 dummy**（参数获 0 梯度、等价冻结、DDP 安全）；`seg_point=0` → 点 loss 整体关闭（dummy 仍保 DDP 安全）。
    - **desync 安全**：原担心"head 已建但走 Lite 会触发 P1 unused"——由 §2.5 的 forward `zero_loss()` dummy 解决（dummy 与 `seg_point_refine` 无关、只要 head 存在就跑）。故运行时 `seg_point_refine` 可安全切换 MLP/Lite 做消融，不必动 YAML/重训。
- 两种模式**共用 `seg_point` gain** 作点 loss 权重。

**与 seg_bnd / seg_comp 的关系（P3，不叠两套 point loss）**：

- **同一 `single_mask_loss` 内只会有一套点 loss**：Lite 或 MLP 二选一（由 `point_head is not None and point_feats is not None` 决定），不叠两套。`seg_point` gain 共用。
- **seg_bnd（Sobel 边界 L2）/ seg_comp（Focal-Tversky 完整性）可与 MLP 点 loss 并行**：三者作用维度不同——seg_bnd 管整掩码边界梯度对齐，seg_comp 管 FN-asymmetric 完整性，point head 管逐点 refine。它们都加在 `total` 上、各自独立 gain（`seg_bnd`/`seg_comp`/`seg_point`），可同时开启、互不冲突。
- 消融矩阵三维正交：`seg_comp ∈ {0,>0}` × `seg_bnd ∈ {0,>0}` × 点模式 ∈ {Lite, MLP, 0}（点模式由 `point_hidden`+`seg_point_refine`+`seg_point` 共同决定）。

---

## 3. 推理路径：迭代细分（PyTorch-only）

> **实现状态（2026-07-05 refine 后）**：已落地。新增 `process_mask_pointrend`（`utils/ops.py`），当前语义先严格对齐标准 `process_mask`：粗 logits = `(masks_in @ protos).view(n,mh,mw)` → 在 proto 分辨率按 bbox crop → 直接上采样到目标 `shape`；随后做 K 次 full-resolution refine pass：在预测 bbox ROI 内采不确定点（`get_uncertain_point_coords_in_roi`，推理无 GT，故不使用 Sobel `weight_map`）、`point_sample(feats, coords)` 取 P3 细特征、`point_head(feat, coarse_at)` 得 refined logit，再用 `_scatter_refine_delta` 只写回 `refined - coarse_at`。zero-init `PointHeadMLP` 时 delta=0，输出与标准 `process_mask(..., upsample=True)` bitwise 等价，避免未训练 MLP 改坏推理。
>
> **关键经验**：`AutoBackend` 对 `format=="pt"` 透传模型完整返回 → `preds[1]` 即 head dict、含 `feats`（end2end 嵌在 `preds[1]["one2one"]` 下）——**无需改 head eval 返回**（§3.1 原设想的"让 head 把 feats 带到 preds[0] 元组"不需要）。导出后端返回 `[det, proto]`（无 feats）→ `_extract_pointrend` 返回 None → 细分自动禁用，导出路径维持现状（与 §3.3 一致）。`SegmentationValidator` 已复用同一抽取逻辑，`seg_point_refine_infer=True` 时标准 val 可直接验收 MLP 后处理；`save_json/save_txt` native 路径仍回退 native mask。
>
> **注意**：训练侧 point head 可用 GT bbox/Sobel boundary 候选池（§2.6）；推理/验证细分没有 GT boundary weight，当前只用预测 bbox ROI，不用 boundary `weight_map`。`seg_point_roi` 在推理/验证中只作为 bbox margin（负数会 clamp 到 0）。predict 的 `retina_masks=True` 路径暂不跑 PointRend，避免 letterbox/original 坐标系混用。

### 3.1 hook 位置（原设计与实际落地差异）

`process_mask`（`utils/ops.py:507-531`）的清晰插入缝在 **`ops.py:522`（粗 logits matmul）与 `ops.py:528`（crop/upsample）之间**：

```python
masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # ops.py:522 粗 logits (N,160,160)
# ← 细分插这里
masks = crop_mask(masks, bboxes * ratios)  # ops.py:528
masks = F.interpolate(masks[None], shape, mode="bilinear")[0]  # ops.py:530
return masks.gt_(0.0).byte()  # ops.py:531
```

实际落地时发现 PyTorch `AutoBackend` 会保留完整返回：`preds[1]` 即 head dict，里面已有 `feats`；end2end 时 feats 嵌在 `preds[1]["one2one"]`。因此不需要改 `Segment26.forward` eval 返回，也不需要走 `_feats` hook；`SegmentationPredictor.postprocess` 直接抽 `(point_head, feats[0])` 并透传到 `construct_result`。导出后端仍只返回 `[det_tensor, proto]`、没有 neck feature，所以 `_extract_pointrend` 返回 None，细分自动禁用。

因此细分推理**仅 PyTorch 后端可用**，与 §3.3 导出判定一致，定位为 host 侧 PyTorch 后处理。`retina_masks=True` 暂不启用 PointRend，保持 native mask 的 original-resolution 坐标链不混入 letterbox ROI。

### 3.2 细分循环

```text
logits = (masks_in @ protos).view(N, mh, mw)
logits = crop_mask(logits, bboxes scaled to proto grid)
logits = interpolate(logits, target_shape)             # 与 process_mask(..., upsample=True) 对齐
boxes_norm = bboxes / [w, h, w, h]
for _ in range(max(1, K)):
    coords = get_uncertain_point_coords_in_roi(logits.detach(), calc_unc, npt, ..., boxes_norm, margin)
    feat = point_sample(point_feats.expand(N, -1, -1, -1), coords)
    coarse_at = point_sample(logits, coords)
    refined = point_head(feat, coarse_at)
    logits = scatter_add_delta(logits, coords, refined - coarse_at)
final = logits.gt_(0).byte()
```

### 3.3 导出可行性判定（关键，决定 (I) 不进导出图）

- **mask 组装从来不在导出图里**：ONNX/TRT 仅输出 `(detections+coeffs, proto)`（`exporter.py:926,930-932`），`process_mask` 永远是 host 侧 PyTorch 后处理（`predict.py:86-111`）。
- **固定 K + 固定采样数 + top-k 不确定性（非阈值 mask）** → 可导出：源里手写 `for _ in range(K)`（K 字面常量），trace 展开成 K 个静态子图——同 `NMSModel` batch 循环（`exporter.py:1573`）套路，已被 codebase 接受。
- **数据依赖动态循环**（阈值 `uncertainty>τ`、变长 gather、跑到收敛）→ **不可导出**：`jit.trace` 抓不住数据依赖控制流；ONNX `Loop` op 在 TRT/OpenVINO/NCNN 普遍失败。

**结论**：细分应作为 `SegmentationPredictor.construct_result` 的 PyTorch 后处理（与 `process_mask` 同级），不进导出图。导出路径维持现状。给导出模型加细分需固定 K 展开版、图膨胀 + TRT 兼容性风险，**默认不做**。

---

## 4. 蒸馏 / E2ELoss 契约

- 点 refine loss 仍留在 `v8SegmentationLoss.loss` 内，但 **E2ELoss 已补 branch-aware 控制**：`E2ELoss.__init__` 给两支 criterion 标记 `loss_branch="one2many"/"one2one"`；`v8SegmentationLoss._point_loss_gain()` 在 one2one 上乘 `seg_point_o2o`；`_point_refine_enabled()` 在 one2one 上再受 `seg_point_refine_o2o` 门控。这样可显式处理 §2.2 的 one2one detach 监督弱化问题。
- `e2e_final_o2m` 已接入 E2E decay 的最终 one2many 权重（默认仍 `0.1`，保持旧行为）。边界/refine finetune 可升到 `0.3` 左右，让后期仍保留更多 one2many P3/backbone/proto 梯度；若只想要 one2many point 监督，可设 `seg_point_o2o=0.0`。
- **教师不需要 point head**：`loss_proto` 只读 `proto`（`distill_model.py:403-432`），`loss_feat` 只读 neck 特征 `feats_idx[:-1]`（`:336`），都不碰点 head 输出。点 head 参数仅由学生正则 loss 训练——PointRend refine 本就是学生侧精修，可接受。
- 将来若做"点 logit 蒸馏"，照 `loss_proto` 套路加 `loss_point` 标量 + `"dis_point"` 名字（`engine/trainer.py:388-389`）。本轮不做。
- `decouple_outputs`/`FeatureHook` 不受影响：本设计**不新增 feats 键**——细特征直接复用 `preds["feats"][0]`（§2.2）；新增的只有 `preds[...]["point_refine_dummy"]`（DDP 占位标量，§2.5），蒸馏 `loss_proto`/`loss_feat` 不读它，透传不消费、无害。
- `get_distill_layers`（`distill_model.py:248-258`）返回 `[*head.f, head.i]`：加子模块改参数量但不改 neck 通道，projector 维度探测不受影响。

---

## 5. parse_model / YAML 契约（已落地）

参考 `Segment.__init__` 建 `self.proto`/`self.cv4` 的子模块模式（`head.py:300-303`）。实际实现：

- `Segment26.__init__`（`head.py`）签名：`(nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=(), point_hidden=0)`——`point_hidden` 放 **`ch` 之后、带默认 0**（P1 兼容性要求）。`self.point_head = PointHeadMLP(ch[0], point_hidden) if point_hidden > 0 else None`（条件性构建，`point_hidden=0` → Lite）。
- **parse_model 处理**（`tasks.py`，Segment26 分支）：先 `point_hidden = args[3] if m is Segment26 and len(args) > 3 else None`，若取出则 `args = args[:3]`（剥掉第 4 参，避免被后面的 `args.extend([reg_max, end2end, ch])` 错位）；然后 `args.extend([reg_max, end2end, [ch[x] for x in f]])`；npr 仍按 `args[2] = make_divisible(min(args[2], max_channels)*width, 8)` 缩放；最后 `if point_hidden is not None: args.append(make_divisible(min(point_hidden, max_channels)*width, 8))`（缩放后 append 在 `ch` 之后）。最终传给 `__init__` 的实参 = `[nc, nm, npr, reg_max, end2end, ch, point_hidden]`，与签名顺序一致。
- **兼容性**：旧 YAML `[nc, 32, 256]`（`len(args)==3`）→ `point_hidden=None` → 不剥、不 append → `point_hidden=0` 默认 → Lite、无新参数。新 YAML `[nc, 32, 256, 128]` → MLP。`reg_max` 永远是 `args.extend` 注入、不会被误读为 `point_hidden`。
- 新模型 YAML：`ultralytics/cfg/models/26/yolo26-seg-pointrend.yaml`，末行 `- [[16, 19, 22], 1, Segment26, [nc, 32, 256, 128]]`（`point_hidden=128`）；scale 别名 n/s/m/l/x 与 `yolo26-seg.yaml` 一致。
- `cfg2task`（`tasks.py:2025-2039`）靠模块名小写含 `"segment"` 识别任务：继续用 `Segment26` 子类无需新分支；若改名须保 `"segment"` 子串。

---

## 6. loss_names 决策

| 方案                                  | 改动                                                                                     | resume 影响                                                  |
| ------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **方案 1（折进 seg_loss，推荐起步）** | 点 refine loss 归 `loss[1]`，`loss_names` 5 元组不动                                     | loss 向量长度不变 → **同架构 run 间 resume 安全**（见 §6.1） |
| 方案 2（独立列 pt_loss）              | `loss` 向量 5→6（`loss.py:506`），`seg/train.py:66` 加 `"pt_loss"`，E2ELoss 两支自动对齐 | loss_items 形状变 → **破坏 checkpoint resume**               |

`label_loss_items`/`progress_string`（`detect/train.py:216-245`）按 `len(loss_names)` 自适应；方案 2 注意 `:227-229` val 前缀过滤分支。**本轮用方案 1**。

### 6.1 resume vs finetune-from-checkpoint（P1，关键）

"loss 向量长度不变 → resume 安全"**只对同一 point-head 架构的 run 之间 resume 成立**。**从当前 recipe200 `last.pt`/`best.pt` 直接 `resume=True` 并新增 point head 是错误的**——模型结构（多了 `point_head` 子模块）和 optimizer state（多了 point_head 参数组）都变了，不应按 resume 走。正确方式（finetune from checkpoint）：

1. 用旧 `best.pt` 作为 **pretrained / weights** 初始化 backbone+neck+旧 head（`model = YOLO("best.pt")` 后 `.load()`，或 `weights=` 参数）。
2. **新建一个 run**（新 run 目录，`resume=False`）。
3. **optimizer 重新初始化**（不加载旧 optimizer state）。
4. **`point_head` 随机/zero-init**：随机初始化（默认）或最后一层 zero-init 使初始 refine 为恒等（输出 0 → 粗 logit 不变），训练更稳。
5. 学习率按 finetune 调（通常比 from-scratch 小）。

即"**finetune from checkpoint**，不是 resume"。同理，从无 point head 的旧 ckpt 加载到含 point head 的新模型时，`point_head` 的键在 state_dict 中不存在 → 需 `strict=False` 或显式跳过。

**P4｜checkpoint migration 步骤（双向）**：

- **旧 ckpt → 新模型（含 point_head）**：`point_head` 键不在旧 state_dict → 加载用 `strict=False`（Ultralytics `YOLO.load`/`attempt_load` 默认容忍 missing）；`point_head` 随机 init（或末层 zero-init 使初始 refine≈恒等）。**不要** `resume=True`（optimizer state 形状不匹配）。
- **新 ckpt → 旧代码（无 point_head）**：多出的 `point_head.*` 键 → `attempt_load` 用 `strict=False` 自动 ignore 多余键；旧代码无该子模块，不消费、无害。
- **同架构新 run 间 resume**（都含 point_head、同 `point_hidden`）：state_dict 完全匹配 → `resume=True` 安全（含 optimizer state、EMA、epoch）。
- **不同 `point_hidden` 的两 run 间**：point_head 形状不同 → 不能 resume，按 finetune 处理。
- 一句话：**结构变（增/删/改 point_head）= finetune（`strict=False` + 新 optimizer）；结构不变 = resume**。

---

## 7. 推荐落地顺序（最小风险）

1. **训练侧 (T) 主体已落地**（§10.1）：方案 B MLP + §2.6 bbox ROI 采样 + §2.5 DDP 混合解法；**2-GPU allreduce 已核验**（`scripts/smoke_point_head_ddp.py`）。从 recipe200 旧 ckpt 加 point head 须 **finetune-from-checkpoint**（§6.1），非 resume。
2. **E2E refine 权重先定住**（§2.2/§4）：默认 one2one late-heavy 会削弱 P3/backbone/proto 的 refine 梯度；边界/refine finetune 建议先试 `seg_point_o2o=0.0` 或 `seg_point_refine_o2o=False`，并把 `e2e_final_o2m` 提到约 `0.3` 做稳健起点。
3. **推理/验证侧 (I) 已落地但默认关**（§3）：`process_mask_pointrend` + `SegmentationPredictor`/`SegmentationValidator` 接线（feats 经 PyTorch eval 返回透传、无需改 head eval）+ cfg `seg_point_refine_infer`/`seg_point_subdiv_k`，PyTorch-only，导出、predict retina、val native JSON/TXT 路径自动回退。训练消融先固定 `seg_point_refine_infer=False`，PointRend 直接验收再显式打开。
4. **消融**：G0(无) vs +seg_point-lite(现,loss侧无参) vs +point-MLP(T,criterion侧带参) vs +MLP+subdiv(T+I)。**验收看 `mask AP75/AP95` 涨幅**（§2.8）：训练侧 (T) 是辅助监督；(I) subdivision 上线后推理边界可直接变锐。

---

## 8. 承载改动点一览（file:line）

| 关注点                             | 锚点                                                                                                                                                  | 改动                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YAML head args                     | `cfg/models/26/yolo26-seg.yaml:52`；解析 `nn/tasks.py:1924-1945`（Detect 族 append `[reg_max,end2end,ch]` 在 `:1941`）                                | `point_hidden` 放 `ch` 之后（形参最后、带默认）；`parse_model` 专门处理 Segment26 可选第 4 个 YAML 参数（`len(args)>=4` 才启用）；旧 YAML 无第 4 参仍兼容                                                                                                                                                                                                                                                                                                                                         |
| head 子模块构造                    | `nn/modules/head.py:335-352`(Seg), `:439-462`(Segment26)                                                                                              | `self.point_head = PointHeadMLP(...)`；optimizer 已过滤 frozen 参数，新 head 自动进参数组                                                                                                                                                                                                                                                                                                                                                                                                         |
| head forward                       | `head.py:158`(feats=x), `:166`(one2one x_detach), `:464-482`(Segment26)                                                                               | **复用 `preds["feats"][0]`（无需新增 feats 键，one2one 已自动 detach）**；MLP 真跑在 criterion，**但 forward 内新增 `point_head.zero_loss()` 占位 stash 到 `preds[...]["point_refine_dummy"]`**（§2.5 混合解法）                                                                                                                                                                                                                                                                                  |
| 细特征来源                         | `head.py:158`,`:370`(Seg); `block.py:1999-2008`(Proto26)                                                                                              | 第一版用 `feats[0]`=P3（最高分辨率 base，也是 Proto26 融合基底）；后续可对比 Proto26 内部 fused `feat`（需 Proto26 暴露 + 正确处理 one2one detach）                                                                                                                                                                                                                                                                                                                                               |
| loss: MLP 运行处（方案 B）         | `utils/loss.py:509-572`(`loss()` 取 feats[0]/dummy), `:575-702`(`single_mask_loss` point 分支), `:704-782`(`calculate_segmentation_loss` 透传)        | `single_mask_loss` 内逐实例跑 MLP；`point_head`+`point_feats_i`(=`preds["feats"][0]` per-image) 显式传参（§2.3）；per-image 用 `point_feats[i:i+1].expand(n,-1,-1,-1)` 广播对齐 coords（§2.4）；`loss[1] += point_refine_dummy` 后 `loss[1] *= self.hyp.box`                                                                                                                                                                                                                                      |
| DDP / compile                      | `engine/trainer.py:375-380`(wrap), `:303-309`(distill)                                                                                                | **已落地混合解法** + **2-GPU 核验通过**（`scripts/smoke_point_head_ddp.py`）                                                                                                                                                                                                                                                                                                                                                                                                                      |
| DDP unused（P1）                   | `head.py` PointHeadMLP.zero_loss + forward stash；`utils/loss.py` `loss[1] += point_refine_dummy`                                                     | point_head 已建但 `seg_point=0`/无 fg/`num_points=0`/`seg_point_refine=False` 任一导致 criterion 不调 MLP 时，dummy 仍在 forward 跑 → 参数可达、DDP 安全；条件性构建保留（无第 4 参→`point_head=None`）                                                                                                                                                                                                                                                                                           |
| E2ELoss                            | `utils/loss.py:1343-1381`                                                                                                                             | 已加 branch-aware 控制：`loss_branch` 标记 one2many/one2one；one2one point gain 乘 `seg_point_o2o`；one2one MLP 受 `seg_point_refine_o2o` 控制；`e2e_final_o2m` 接入 decay 终值，边界/refine finetune 可保留更多 one2many 梯度                                                                                                                                                                                                                                                                    |
| loss_names / 日志                  | `models/yolo/segment/train.py:66`                                                                                                                     | 方案 1 无改；方案 2 加 `"pt_loss"` + 向量 5→6                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| resume / finetune                  | `engine/trainer.py` resume 路径                                                                                                                       | 同架构 run 间 resume 安全；从 recipe200 旧 ckpt 加 point head 须 **finetune-from-checkpoint**（pretrained+新 run+optimizer 重 init+point_head zero/随机 init），非 resume（§6.1）；migration 双向 `strict=False`（§6.1 P4）                                                                                                                                                                                                                                                                       |
| 蒸馏                               | `nn/distill_model.py:296-357`                                                                                                                         | 无（学生 criterion 承载；教师无需 point head）                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 推理/验证 feats plumbing（已落地） | `models/yolo/segment/predict.py`；`models/yolo/segment/val.py`；`nn/autobackend.py`；`utils/ops.py`(`process_mask_pointrend`/`_scatter_refine_delta`) | **无需改 head eval**：PyTorch eval 透传 → `preds[1]` head dict 含 `feats`（end2end 嵌 `one2one`）→ `_extract_pointrend` 抓 `(point_head, feats[0])` → predict/val 后处理 → `process_mask_pointrend`；导出后端返回 `[det,proto]` 无 feats → 细分自动禁用；predict `retina_masks=True` 与 val `save_json/save_txt` native path 回退标准/native path（§3）                                                                                                                                           |
| cfg 键                             | `cfg/default.yaml` + `cfg/__init__.py`                                                                                                                | 已加 `seg_point`(FLOAT,共用 gain)/`seg_point_importance`(FRACTION)/`seg_point_num`+`seg_point_oversample`(INT)/`seg_point_refine`+`seg_point_boundary`(BOOL)/`dali`(BOOL)；**模式开关 = 构建级 YAML 第 4 参 `point_hidden`（决定 head 存在与否）+ 运行时 `seg_point_refine`（决定 head 是否参与点 loss）**（§2.9）；E2E 控制 `seg_point_o2o`(FLOAT)/`seg_point_refine_o2o`(BOOL)/`e2e_final_o2m`(FLOAT)；推理侧 `seg_point_refine_infer`(BOOL,默认关)+`seg_point_subdiv_k`(INT,默认 3) 已加（§3） |

**承载不变量**：loss 所需一切必须能从 head 训练态 forward 返回的 `preds` dict 取到——那是 criterion 除 `batch` 外的唯一入参（`nn/tasks.py:347` `return self.criterion(preds, batch)`）。**带参数子模块（point_head）走混合解法**（§2.5），2-GPU allreduce 已核验（`scripts/smoke_point_head_ddp.py`）。

---

## 9. 阻塞与前置

- **recipe200 仍停在 epoch 107/200**（non-finite recovery 后 OOM，非正常结束）：继续训练前需重新实时核验 GPU 进程/显存状态，并决定 resume baseline 还是另起 Point-head finetune run。
- 训练侧 (T) 的实现不依赖 GPU 即可写代码 + CPU smoke，但消融真跑需 3-GPU 空闲，受 [[recipe200-gpu-sharing-oom]] 约束。
- 本设计与现有 `seg_point` 子项默认 gain 0 一致：未开 `seg_point` 时 `single_mask_loss` 仍短路到 legacy BCE，对在途 baseline 零影响。
- **P5｜显存**：MLP 本身参数量极小，增量主要在 `point_sample(fine_feat)` 对 fg instances 的逐实例采样——per-image 细特征经 `expand` 广播成 `(n,Cf,Hf,Wf)` 后，`point_sample` 输出 `(n,Cf,P)`（§2.4）。recipe200 已在 epoch 108 OOM 边缘（见 stage summary），加 point head 后应**下调 batch（如 84→更低）或减小 `seg_point_num`**（如 112→64），先在单卡 smoke 测峰值再上 3-GPU。bf16 autocast 下 `point_sample`/MLP 应走 fp32-internal（参考 F21 `sobel_magnitude` 的 `autocast(enabled=False)` 做法）避免精度/显存抖动。
- **P6｜GT 仍是 poly 栅格，head 突破不了 160×160 监督上限**：GT mask 由 `polygon2mask` 在 mask_ratio=4（640→160）栅格化、且 polygon 经 `approxPolyDP` 简化（见 `seg-loss-subgains-strategy1`）。point head 的逐点监督**同样受限于 GT 质量**：边界 aliasing、小目标 poly 简化后失真、v2 标签的合法 holes——这些上限不会因加 MLP 消失。boundary 放大标签 aliasing 风险在 stage summary 已提过，point head 同样受约束。若 GT 边界本身不准，point head 学到的是"对 aliasing GT 的过拟合"而非真实边界 refine——验收时需结合 AP75/AP95 与 GT 质量一并判断（§2.8）。

---

## 10. 实现状态（已落地 vs 待办）

> 本节对照实际代码（`git diff HEAD` + 未跟踪新文件）记录训练侧 (T) 与推理/验证侧 (I) 的落地情况。剩余待办为消融与实验（§10.2）。

### 10.1 已落地（训练侧 T + 推理侧 I）

**head（`nn/modules/head.py`）**

- `PointHeadMLP`：`Conv1d(Cf+1→hidden→hidden→1)` 逐点 MLP，末层 `zero_init` → 初始 `refined == coarse`（恒等残差）；`forward(point_feats (N,Cf,P), coarse (N,P)|(N,1,P))` 强制 fp32；`zero_loss()` 返回连接所有参数的标量 0（DDP 占位）。已 export 进 `nn/modules/__init__.__all__`。
- `Segment26.__init__(..., point_hidden=0)`：`self.point_head = PointHeadMLP(ch[0], point_hidden) if point_hidden > 0 else None`（条件性构建）。
- `Segment26.forward`：训练态 `if self.training and self.point_head is not None` 时算 `dummy = self.point_head.zero_loss()`，end2end 下 stash 到 `preds["one2many"]["point_refine_dummy"]` 与 `preds["one2one"]["point_refine_dummy"]`（同一 dummy 对象），非 end2end stash 到 `preds["point_refine_dummy"]`。

**parse_model / YAML（`nn/tasks.py` + `cfg/models/26/yolo26-seg-pointrend.yaml`）**

- `parse_model` Segment26 分支：`len(args)>3` 时取 `point_hidden=args[3]` 并 `args=args[:3]`（剥掉，避免被 `args.extend([reg_max,end2end,ch])` 错位）→ extend → npr 缩放 → `args.append(make_divisible(min(point_hidden,max_channels)*width,8))`（缩放后置末位）。旧 YAML `[nc,32,256]`（无第 4 参）→ `point_hidden=None` → Lite、无新参数，新旧共存。
- 新 YAML `yolo26-seg-pointrend.yaml`：末行 `[[16,19,22], 1, Segment26, [nc, 32, 256, 128]]`；scale 别名 n/s/m/l/x 与 `yolo26-seg.yaml` 一致；`end2end: True`、`reg_max: 1`。

**loss（`utils/loss.py` + `utils/mask_point_sampling.py` + `mask_boundary_loss.py` + `mask_completeness_loss.py`）**

- `v8SegmentationLoss.__init__`：`self.point_head = getattr(model.model[-1], "point_head", None)`。
- `loss()`：`point_refine_dummy = preds.get("point_refine_dummy")`；`point_w = self._point_loss_gain()` 按 branch 预计算；仅当 `point_w>0`、`_point_refine_enabled()`、且模型有 `point_head` 时取 `point_feats = preds["feats"][0]`；透传给 `calculate_segmentation_loss(point_w=..., point_feats=...)`；末尾 `if point_refine_dummy is not None: loss[1] += point_refine_dummy` 后 `loss[1] *= self.hyp.box`。
- `calculate_segmentation_loss`：per-image `point_feats_i = point_feats[i:i+1].expand(mask_idx.shape[0], -1, -1, -1)`（expand 广播、不复制显存），连同 `point_head=getattr(self, "point_head", None)`、`roi_margin`、`boundary_w` 与各 gain 透传给 `single_mask_loss`。**用 `getattr` 而非 `self.point_head`**：该方法可被单元测试用 `object.__new__(v8SegmentationLoss)` 跳过 `__init__` 单独调用（不设 `point_head` 属性），裸 `self.point_head` 会 `AttributeError`；`getattr(..., None)` 退化为 Lite（`pl=coarse`）且与 `__init__` 的 `getattr(model.model[-1], "point_head", None)` 风格一致。正常训练路径 `__init__` 已设 `self.point_head`，行为不变。
- `single_mask_loss`（仍 `@staticmethod`，显式传参）：新增 kwargs `comp_w/bnd_w/point_w/num_points/oversample_ratio/importance_ratio/point_head/point_feats/roi_margin/boundary_w`；`comp_w==bnd_w==point_w==0` 时短路返回 legacy BCE（G3-Eq1，对在途 baseline 零影响）；point 分支：`no_grad` 包裹 coords 采样 + GT 重采，`roi_margin>=0 or boundary_w` 时 `boxes_norm=xyxy/[W,H,W,H]` 走 `get_uncertain_point_coords_in_roi`（§2.6 ROI），否则 legacy full-grid；`boundary_w=True` 时用 `sobel_magnitude(gt_mask)` 作为 ROI 候选采样权重；`coarse=point_sample(pm4,coords)`，`point_head is not None and point_feats is not None` 时 `pl=point_head(point_sample(point_feats,coords), coarse)` 否则 `pl=coarse`（Lite），focal+dice 逐实例求和加到 `total`。
- DDP dummy 一致性：end2end 下同一 dummy 对象被 one2many/one2one 两支各 `loss[1] += dummy` 一次 → 参数在两支 backward 图都可达，dummy 值恒 0 不污染 loss。
- `E2ELoss`：`one2many.loss_branch="one2many"`、`one2one.loss_branch="one2one"`；one2one `seg_point` 乘 `seg_point_o2o`，one2one MLP 受 `seg_point_refine_o2o` 门控；`e2e_final_o2m` 替代硬编码 final o2m=0.1（默认仍 0.1）。

**cfg（`cfg/default.yaml` + `cfg/__init__.py`）**

- 新增 FLOAT：`seg_comp`/`seg_bnd`/`seg_point`/`seg_point_roi`/`seg_point_o2o`/`e2e_final_o2m`；FRACTION：`seg_point_importance`；INT：`seg_point_num`/`seg_point_oversample`/`seg_point_subdiv_k`；BOOL：`seg_point_refine`/`seg_point_boundary`/`seg_point_refine_o2o`/`seg_point_refine_infer`/`dali`。均带内联 `# (type) 描述` 注释，经 `get_cfg` 自动透传。`seg_point_roi` 默认 `0.0`（ROI bbox-restricted）；`<0` 且 `seg_point_boundary=False` 时训练退回 legacy full-grid（消融对照），推理侧会 clamp 到 0 以保持 bbox ROI。

**ops / 采样器（`utils/ops.py` + `utils/mask_point_sampling.py`）**

- `sobel_magnitude`：3×3 Sobel |grad|，`torch.autocast(device_type=..., enabled=False)` 强制 fp32（F21），供 `mask_boundary_loss.boundary_l2_loss_per_instance` 复用。
- `get_uncertain_point_coords_in_roi` + `_rand_in_roi` + `_weighted_rand_in_roi`（§2.6 已落地）：逐实例 bbox+margin 限制的不确定性采样器；退化 bbox 回退 full-grid；可选 `weight_map`（GT Sobel 边界 band）时走**混合候选池**——oversample 50% 边界加权 + 50% bbox 均匀合并后做 pred-uncertainty top-k，25% 随机余量始终 bbox 均匀（消解边界加权 × top-k 的内部 FP/FN 监督冲突，§2.6）。`single_mask_loss` point 分支按 `roi_margin>=0 or boundary_w` 调度它，否则走 legacy `get_uncertain_point_coords_with_randomness`。

**DALI（`data/dali_seg.py` + `models/yolo/detect/train.py`）**

- `dali=True` + segment 时走 `YOLOSegDALILoader`（GPU JPEG decode/resize，CPU label/mask 格式化）；`preprocess_batch` 弹出 `batch["dali"]` 标记、归一化保持一致。**与本设计正交**——是独立的训练加速实验，非 point head 依赖。

**测试（`yolo26-cu133` conda env targeted 通过）**

- 已跑定向 pytest（边界/completeness/point/refine/E2E cfg/predict/val/load/resume/forward/backward smoke）：此前 `24 passed in 1.65s`；本轮补 val 分叉后追加 `15 passed in 1.81s`。覆盖 `test_segmentation_loss_e2e_point_branch_controls`、`test_segment26_forward_point_refine_dummy_and_detach_contract`、`test_e2e_point_refine_backward_branch_gradient_routes`、`test_process_mask_pointrend_basic`、`test_pointrend_infer_subdivision_smoke`、`test_segmentation_validator_pointrend_postprocess`、`test_segmentation_loss_cfg_overrides_are_accepted`、`test_model_load_unwraps_distillation_student_checkpoint`、`test_resume_point_refine_overrides_are_whitelisted` 等关键路径。
- 已跑 2-GPU DDP smoke：`scripts/smoke_point_head_ddp.py`（`torchrun --nproc_per_node=2`，`find_unused_parameters=False`；MLP 路径 point_head grads synced across 2 ranks；`seg_point_refine=False` dummy 路径 zero point_head grads OK）。
- `test_engine.py` 新增覆盖包括：`test_mask_point_coords_full_grid`、`test_mask_point_coords_in_roi`（§2.6 ROI 采样器：点落在 bbox+margin 内、退化 bbox 回退 full-grid）、`test_point_focal_dice_per_instance`、`test_point_head_mlp_zero_init_is_coarse_residual`（断言 init 时 `refined==coarse`）、`test_sobel_magnitude_constant_is_near_zero`、`test_single_mask_loss_all_gains_disabled_matches_legacy`（`seg_point=0`==legacy BCE；point_lite==point_refine@init；`num_points=0`==legacy）、`test_segmentation_loss_optional_hyp_injection_is_finite`（用 `object.__new__` 跳过 `__init__` 隔离测 `calculate_segmentation_loss`，依赖上面的 `getattr` 防御）、`test_segmentation_loss_cfg_overrides_are_accepted`（含 `seg_point_roi`/`seg_point_boundary` 类型校验）、`test_yolo26_pointrend_yaml_builds_optional_point_head`（断言新 YAML 建 `point_head`、旧 YAML `point_head is None`）。§2.6 进阶新增 `test_mask_point_coords_weighted_in_roi`（断言混合候选池下边界 band 仍被过采样（vs 均匀 ~0.1）但内部点保持可达、退化 bbox 回退 full-grid）与 `test_segmentation_loss_boundary_roi_path_is_finite`（`seg_point_boundary=True` 且 `seg_point_roi<0` 时强制走 ROI、端到端有限且可反传）。
- `test_mask_boundary_loss.py`、`test_mask_completeness_loss.py`：子 loss helper 单测。
- 注：`test_train_reuses_loaded_checkpoint_model[kwargs*]` 失败与本设计**无关**——它实跑一次训练并要求落盘 `best.pt`/`last.pt`，因 `weights/path with spaces/...` 无 checkpoint 产出而 `FileNotFoundError`，属测试基础设施/环境问题，不触碰 seg point loss 路径。

**推理/验证侧 (I)（`utils/ops.py` + `models/yolo/segment/predict.py` + `models/yolo/segment/val.py` + `cfg/`）**

- `process_mask_pointrend`（`utils/ops.py`）：粗 logits `(masks_in @ protos).view(n,mh,mw)` → 按标准 `process_mask` 先在 proto grid crop → 直接上采样到 target shape → K 次 full-res ROI uncertain-point refine → `point_head(feat, coarse_at)` → `_scatter_refine_delta(logits, coords, refined - coarse_at)` → `.gt_(0).byte()`；空输入回 `(0,h,w)`。zero-init head 下与标准 `process_mask(..., upsample=True)` bitwise 等价。
- `_scatter_refine_delta`：按 `coords` 的最近像素把 refined-minus-coarse delta 加回 full-res logits（与 `point_sample(align_corners=False)` 坐标约定一致：`rows=(coords[...,1]*h-0.5).round().long().clamp(0,h-1)`），非采样点保持原双线性值。
- `SegmentationPredictor`：`_point_head()`（仅 `format=="pt"`，`getattr(self.model.model.model[-1], "point_head", None)`，try/except 容错）、`_extract_pointrend(preds)`（`seg_point_refine_infer` 开 + `preds[1]` 是 dict + point_head 存在 → 返回 `(point_head, feats[0])`；end2end 嵌 `preds[1]["one2one"]`；否则 None）、`postprocess` 抽 pointrend 透传、`construct_results` 按图切 `(point_head, batch_feats[i:i+1])`、`construct_result` 仅在 `pointrend is not None and not retina_masks` 时调 `process_mask_pointrend`；`retina_masks=True` 保持 native path。
- `SegmentationValidator`：`init_metrics()` 解析 raw/AutoBackend 模型的 `point_head`；`_extract_pointrend()` 复用 `preds[1]` head dict 的 `feats[0]`；`postprocess()` 在 `seg_point_refine_infer=True` 且非 native JSON/TXT 时调 `process_mask_pointrend`，输出 full-res masks；`_prepare_batch()` 同步把 GT masks 保持 full-res，避免 IoU 维度错配。默认关闭时仍是 legacy proto-resolution `process_mask`，所以默认 train val/best.pt fitness 不变。
- cfg：`seg_point_refine_infer`（BOOL，默认关）、`seg_point_subdiv_k`（INT，默认 3，0/1=单次全分辨率 refine）；均经 `cfg/__init__.py` 的 `CFG_BOOL_KEYS`/`CFG_INT_KEYS` 注册。
- 测试：`test_process_mask_pointrend_basic`（K∈{1,3} × zero-init/random；形状/dtype/binary/局部化；空→(0,H,W)）、`test_pointrend_infer_subdivision_smoke`（真 predict、imgsz=96、conf=0.0、subdiv ON→(N,96,96)、默认关也通过）、`test_segmentation_validator_pointrend_postprocess`（默认 val 输出 24×24；开 `seg_point_refine_infer=True` 输出 96×96，并验证 GT mask 同步 full-res）、cfg-override 扩展含 `seg_point_refine_infer`+`seg_point_subdiv_k`。

### 10.2 待办

- **§2.6 进阶：GT Sobel boundary band 加权采样**：已落地（含混合候选池修复）但默认关闭（`seg_point_boundary=False`）；下一步是做消融，判断边界加权采样是否提升 AP75/AP95 和 boundary F-score。**消融必须固定 `seg_point_boundary`**（其混合候选池改变了 point 监督语义，不可与 ROI/uniform 自由组合）。
- **(I) 细分验收**：代码已落地（`process_mask_pointrend` + SegmentationPredictor/SegmentationValidator 接线 + `seg_point_refine_infer`/`seg_point_subdiv_k` cfg + 单测，见 §3）。待办转为：真实数据端到端对比默认 val fitness vs `seg_point_refine_infer=True` val 的 AP75/AP95、predict 可视化边界锐度和 latency；确认导出路径自动禁用（exported backend 无 feats）。
- **Proto26 fused-feat 对比实验**：当前 fine feature=`feats[0]`=P3；后续对比 Proto26 内部 fused `feat`（需 Proto26 暴露 + 正确处理 one2one detach），看是否更涨点。
- **recipe200 决策**：停在 epoch 107/200；继续前需实时核验 GPU 进程/显存状态，并决定 resume baseline 还是另起 Point-head finetune run（见 §9）。2026-07-05 复核：recipe200 进程已停，GPU 显存约 2 MiB，`nvidia-smi pmon` 未见训练/计算进程；`utilization.gpu` 偶发异常读数需启动前再查。
- **G0–G6 消融真跑**：需 3-GPU 空闲 + 用户配置，受 [[recipe200-gpu-sharing-oom]] 约束。现可对照 `seg_point_roi>=0`（ROI）vs `<0`（legacy full-grid）两种点采样。

---

## 11. 2026-07-05 refine/check 记录与下一步计划

### 11.1 当前 recipe200 baseline 状态

- Run：`runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200`。
- `results.csv` 最新仍为 epoch 107/200（未新增 epoch）。epoch 107 指标：Box mAP50 `0.58887`、Box mAP50-95 `0.43624`、Mask mAP50 `0.57125`、Mask mAP50-95 `0.37592`；`best.pt` 与 `last.pt` 时间戳均为 2026-07-04 11:43。
- ckpt 顶层为 `DistillationModel`，inner `student_model` 为 `SegmentationModel`，head 为 `Segment26`，**无 `point_head` attr**；`seg_point` / `seg_point_refine` / `seg_point_boundary` / `seg_point_refine_infer` 均未写入旧 `train_args`。结论：这是干净 baseline，新增 Point-head 只能 finetune-from-checkpoint，不能 `resume=True`。
- GPU 核验：3 张 RTX 5090 D 当前显存占用约 2 MiB，`nvidia-smi pmon` 未见训练/计算进程；`utilization.gpu` 查询偶见 GPU1/2 100% 但无进程占用，按显存与 pmon 判断没有当前训练任务。

### 11.2 本轮代码 refine 要点

- **E2E one2one detach 风险收敛**：新增 `seg_point_o2o`、`seg_point_refine_o2o`、`e2e_final_o2m`。默认保持旧行为；边界/refine finetune 可显式降低/关闭 one2one point refine，保留 one2many backbone 梯度。
- **蒸馏 ckpt finetune 入口修复**：`BaseModel.load()` 遇到 `DistillationModel` wrapper 会自动 unwrap `student_model`，从 recipe200 `best.pt` 初始化新 `yolo26s-seg-pointrend.yaml` 已实测成功（`Transferred 844/850 items`，新 `point_head` 保留初始化）。
- **resume point/refine override 放行**：`check_resume()` 白名单已加入 `seg_point*` / `seg_comp` / `seg_bnd` / `e2e_final_o2m`，同架构 resume 时可调 point 数量、ROI/boundary、one2one 门控和 E2E 终值；resume 后 E2E criterion 会重建并恢复 schedule。
- **推理 PointRend no-op parity**：`process_mask_pointrend` 改成标准 `process_mask` crop/upsample 后，在 full-res ROI 内 scatter `refined - coarse` delta。zero-init head 下输出与标准 `process_mask` 完全一致。
- **predict 路径防错**：PointRend 推理只在 PyTorch + `seg_point_refine_infer=True` + `retina_masks=False` 时启用；`retina_masks=True` 和导出后端回退标准路径。
- **val 验收分叉补齐**：`SegmentationValidator` 默认仍走 legacy `process_mask`（best.pt fitness 可比）；显式 `seg_point_refine_infer=True` 时走 `process_mask_pointrend` full-res masks，并同步 GT mask full-res，避免 PointRend val IoU 维度错配。

### 11.3 本轮验证

```bash
/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python -m pytest \
  tests/test_mask_boundary_loss.py tests/test_mask_completeness_loss.py \
  tests/test_engine.py::test_mask_point_coords_full_grid \
  tests/test_engine.py::test_mask_point_coords_in_roi \
  tests/test_engine.py::test_mask_point_coords_weighted_in_roi \
  tests/test_engine.py::test_point_focal_dice_per_instance \
  tests/test_engine.py::test_point_head_mlp_zero_init_is_coarse_residual \
  tests/test_engine.py::test_single_mask_loss_all_gains_disabled_matches_legacy \
  tests/test_engine.py::test_segmentation_loss_optional_hyp_injection_is_finite \
  tests/test_engine.py::test_segmentation_loss_boundary_roi_path_is_finite \
  tests/test_engine.py::test_segmentation_loss_e2e_point_branch_controls \
  tests/test_engine.py::test_segment26_forward_point_refine_dummy_and_detach_contract \
  tests/test_engine.py::test_e2e_point_refine_backward_branch_gradient_routes \
  tests/test_engine.py::test_process_mask_pointrend_basic \
  tests/test_engine.py::test_pointrend_infer_subdivision_smoke \
  tests/test_engine.py::test_segmentation_loss_cfg_overrides_are_accepted \
  tests/test_engine.py::test_yolo26_pointrend_yaml_builds_optional_point_head \
  tests/test_engine.py::test_model_load_unwraps_distillation_student_checkpoint \
  tests/test_engine.py::test_resume_point_refine_overrides_are_whitelisted -q
# 此前核心集合：24 passed

/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python -m pytest \
  tests/test_engine.py::test_resume_point_refine_overrides_are_whitelisted \
  tests/test_engine.py::test_mask_point_coords_full_grid \
  tests/test_engine.py::test_mask_point_coords_in_roi \
  tests/test_engine.py::test_mask_point_coords_weighted_in_roi \
  tests/test_engine.py::test_point_focal_dice_per_instance \
  tests/test_engine.py::test_point_head_mlp_zero_init_is_coarse_residual \
  tests/test_engine.py::test_segmentation_loss_e2e_point_branch_controls \
  tests/test_engine.py::test_segment26_forward_point_refine_dummy_and_detach_contract \
  tests/test_engine.py::test_e2e_point_refine_backward_branch_gradient_routes \
  tests/test_engine.py::test_process_mask_pointrend_basic \
  tests/test_engine.py::test_pointrend_infer_subdivision_smoke \
  tests/test_engine.py::test_segmentation_validator_pointrend_postprocess \
  tests/test_engine.py::test_segmentation_loss_cfg_overrides_are_accepted \
  tests/test_engine.py::test_yolo26_pointrend_yaml_builds_optional_point_head \
  tests/test_engine.py::test_model_load_unwraps_distillation_student_checkpoint -q
# 本轮追加 val 分叉后核心回归：15 passed in 1.81s

/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29531 scripts/smoke_point_head_ddp.py
# OK: 2-GPU point_head DDP smoke
```

补充：一次宽泛 `-k "point ..."` pytest 会误匹配 `test_train_reuses_loaded_checkpoint_model[...]`，该组因本地 `weights/path with spaces/nonexistent-best.pt` 未产出 `best.pt/last.pt` 抛 `FileNotFoundError`，属于测试基础设施/环境问题，不触碰 PointRend loss/predict 路径。

### 11.4 下一步 finetune 建议

建议先开一个小而稳的 Point-head finetune run，目标是验证训练侧边界监督是否提升 AP75/AP95，而不是一开始追求肉眼 predict 锐化。下面是**带蒸馏的 Python API 示例**；当前真实脚本入口默认不蒸馏、默认参数另见 §15.7，下一阶段更建议先跑 §15.7 的 no-distill 保守命令：

```python
from ultralytics import YOLO

model = YOLO("yolo26s-seg-pointrend.yaml").load(
    "runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt"
)
model.train(
    data="/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml",
    epochs=40,
    imgsz=640,
    batch=72,  # recipe200 batch=90 曾在 epoch 108 OOM，先降一档
    device="0,1,2",
    resume=False,
    project="runs/segment",
    name="yolo26s-seg-coconut-b-v2-pointrend-ft01",
    optimizer="MuSGD",
    cos_lr=True,
    close_mosaic=10,
    mask_ratio=4,
    overlap_mask=True,
    distill_model="/home/genesis/Train/Code/ultralytics/yolo26x-seg.pt",
    dis=3.0,
    dis_proto=1.0,
    distill_warmup_epochs=3.0,
    seg_point=0.2,
    seg_point_refine=True,
    seg_point_num=64,
    seg_point_oversample=3,
    seg_point_importance=0.75,
    seg_point_roi=0.0,
    seg_point_boundary=True,
    seg_point_o2o=0.0,
    seg_point_refine_o2o=False,
    e2e_final_o2m=0.3,
    seg_point_refine_infer=False,
)
```

第一轮训练建议仍保持 `seg_point_refine_infer=False`：这样 train-time val / best.pt fitness 继续走标准 `process_mask`，可和 recipe200 baseline 直接比较，避免把训练收益与推理后处理收益混在一起。但这也意味着 best.pt fitness **可能低估** PointRend MLP 后处理的直接收益。

训练完成后补跑第二条显式验收分叉：

```python
model = YOLO("runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft01/weights/best.pt")
metrics_refine = model.val(
    data="/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml",
    imgsz=640,
    batch=72,
    device="0,1,2",
    seg_point_refine_infer=True,
    save_json=False,
    save_txt=False,
)
```

两条分叉都看 `metrics.seg.map75`、`metrics.seg.all_ap[:, 9].mean()`（AP95）和 Mask mAP50-95；PointRend 分叉再看 predict 可视化/latency。若 AP75/AP95 不动，优先排查 P3 80×80 细特征上限与 Proto26 fused feature 暴露实验，而不是继续加大 point loss。

---

## 12. 入口与构建链路核验（A）

### 12.1 实际入口顺序

```text
YOLO.train(...)
  → SegmentationTrainer(overrides)
  → setup_model()
      → SegmentationModel(yaml)
      → parse_model: Segment26 YAML 第 4 参 point_hidden
      → model.load(weights)  # 旧 ckpt/蒸馏 ckpt 只迁移交集权重
  → set_model_attributes(): model.args = trainer.args
  → [可选] DistillationModel(student_model, teacher_model)
  → DDP / ModelEMA
  → 首次 student_model.loss()
      → init_criterion()
      → E2ELoss(v8SegmentationLoss) 或 v8SegmentationLoss
```

### 12.2 A1：旧 seg ckpt 到 PointRend YAML

旧 `yolo26s-seg` / recipe200 ckpt 没有 `point_head.*`，所以新增 PointRend head 必须按 **finetune-from-checkpoint** 走：新 YAML 建模，`load(best.pt)` 迁移交集权重，新 optimizer，`resume=False`。已修复蒸馏 wrapper ckpt 加载：`BaseModel.load()` 会 unwrap `DistillationModel.student_model`，避免 `model.0.conv.weight` key mismatch。实测 recipe200 `best.pt` → `yolo26s-seg-pointrend.yaml`：`Transferred 844/850 items`，新 `point_head` 存在且保持 zero-init。

### 12.3 A2/A3：criterion 懒创建与 cfg 变更

`BaseModel.loss()` 第一次被调用时才创建 criterion；`v8SegmentationLoss.__init__` 会一次性绑定当时的 `model.model[-1].point_head`，`self.hyp = model.args`。正常训练流里 `set_model_attributes()` 先把 `trainer.args` 挂到 model，再进入首次 loss，所以 point/refine cfg 会被 criterion 看到。

约束：

- 如果替换 `model.args` 对象、切换 head、或手动改结构，需 `model.criterion = None` 让下次 loss 重建 criterion。
- 如果只是原地改 `model.args.seg_point` 这类字段，普通 point gain 可动态读到；但 `E2ELoss.final_o2m` 在 `__init__` 固化，所以改 `e2e_final_o2m` 后仍需重建 criterion。
- resume 路径对 E2E 已显式重建 criterion 并按 `start_epoch` 恢复 o2m/o2o schedule；本轮已把 `seg_point*` / `e2e_final_o2m` 加入 resume override 白名单。

### 12.4 A4：蒸馏路径的边界

蒸馏训练会强制关闭 compile：`distill_model is not None and compile=True` 时 trainer 改回 eager。`DistillationModel.loss()` 每步先 teacher forward、再 student forward，然后 student regular seg loss 才跑 point/refine；distill 只做 `loss_feat` / `loss_proto`，不蒸馏 `point_head` logits。代价是显存峰值更高：teacher+student 双 forward、proto distill、point sampling/MLP 同时存在。Point-head finetune 建议先降 batch 或 `seg_point_num`，并固定 `seg_point_refine_infer=False` 做训练指标验证。

---

## 13. Forward 训练 batch 链路核验（C）

### 13.1 实际 forward 张量语义

`Detect.forward` 先用原始 neck features 跑 one2many，再对 one2one 使用 `x_detach = [xi.detach() for xi in x]`：

```text
one2many:
  feats = x                         # P3/P4/P5 有 grad
  mask_coefficient from cv4         # 有 grad
  proto = Proto26(x)                # 有 grad（可能为 tuple: proto, semantic）

one2one:
  feats = [xi.detach() for xi in x] # P3/P4/P5 无 grad
  mask_coefficient from one2one cv4 # one2one head 参数仍有 grad
  proto = Proto26(x).detach()       # proto tuple 时逐项 detach
```

`Segment26.forward` 仅在 `self.training and point_head is not None` 时写入 `point_refine_dummy`。end2end 下同一个 dummy 标量同时挂到 `preds["one2many"]` 和 `preds["one2one"]`；eval/val 不写 dummy，因为没有 backward/unused-param 检查需求。

### 13.2 C1：DDP 混合解法

真实 MLP 仍在 criterion 的 point 分支运行，保留逐实例不确定点采样；forward dummy 只负责让 `point_head` 参数在 DDP forward 图内可达。2-GPU smoke 覆盖两条路径：

- `seg_point_refine=True`：criterion 真 MLP 产生非零梯度，point_head grads 跨 rank 同步。
- `seg_point_refine=False`：criterion 不跑 MLP，但 forward dummy 给 point_head 产生零梯度，DDP 不报 unused。

### 13.3 C2：one2one detach 的训练含义

one2one 的 P3/proto 无梯度不是实现疏漏，而是 E2E 原契约。对 boundary/refine 来说，它意味着 one2one point/refine loss 不塑形 backbone/P3/proto，只能更新 one2one coeff head 与 `point_head`。随着默认 E2E schedule 后期 one2one 权重升高，refine 对 backbone 的监督会变弱。本轮的控制项就是为这个风险准备：

- `seg_point_o2o=0.0`：one2one 不算 point loss。
- `seg_point_refine_o2o=False`：one2one 点损失退回 Lite，不用 detach P3 训练 MLP。
- `e2e_final_o2m≈0.3`：保留更多 one2many 权重，让后期仍有 P3/backbone/proto 梯度。

### 13.4 C3：细特征仍是 P3，不是 Proto26 fused feat

训练、val(PointRend 分叉) 和 predict 当前都用 `preds["feats"][0]` / `feats[0]` 作为 Point-head fine feature。640 输入下 P3 为 80×80，而 mask/proto supervision grid 是 160×160（`mask_ratio=4`）。坐标用 `[0,1]^2` 几何对齐，但 fine feature 分辨率上限仍卡在 P3；Proto26 内部 fused feature 尚未暴露给 point head。若 AP75/AP95 不涨，下一优先级是做 Proto26 fused feature 暴露/对比，而不是盲目加大 point loss。

### 13.5 C4：eval/val 无 dummy

eval/val 不 stash `point_refine_dummy` 是预期行为：无 backward 时 DDP unused-param 不是问题。val 默认不走 `process_mask_pointrend`，所以默认验证指标仍衡量训练侧 point/refine 对标准 `coeff @ proto` mask 的间接收益；显式 `seg_point_refine_infer=True` 时才走 PointRend full-res 后处理。dummy 契约已由 `test_segment26_forward_point_refine_dummy_and_detach_contract` 覆盖：training forward 有 dummy、one2many feats/proto 有 grad、one2one feats/proto detach；eval forward 无 dummy。val 分叉由 `test_segmentation_validator_pointrend_postprocess` 覆盖。

---

## 14. Backward / EMA 链路核验（E）

### 14.1 实际训练循环顺序

```text
forward under autocast
  → loss, loss_items
  → DDP rank: loss *= world_size
  → scaler.scale(loss).backward()
  → optimizer_step()
      → scaler.unscale_(optimizer)
      → clip_grad_norm_(model.parameters(), max_norm=10.0)
      → scaler.step(optimizer)
      → scaler.update()
      → optimizer.zero_grad()
      → EMA.update(model)
epoch end:
  → criterion.update()  # E2ELoss o2m/o2o schedule
  → EMA.update_attr(...)
```

`E2ELoss.update()` 在每个 epoch 的 batch loop 完整结束、validation 前执行；resume 时会重建 criterion，并把 `updates=start_epoch-1` 后调用一次 `update()` 恢复 o2m/o2o 权重。

### 14.2 E1：branch backward 梯度路由

用 `test_e2e_point_refine_backward_branch_gradient_routes` 单独反传 one2many 和 one2one，得到当前真实梯度边界：

- **one2many**：backbone/P3、Proto26、one2many mask coeff head、point_head 都有梯度；这是边界/refine 真正塑形 backbone/proto 的路径。
- **one2one**：backbone/P3 与 Proto26 无梯度（features/proto 都 detach）；one2one mask coeff head 与 point_head 有梯度。这里的 point_head 输入特征是 detached P3 常量，coarse logit 经 detached proto + one2one coeff 得到，所以能训练 MLP/coeff，但不塑形 P3/proto。

因此先前“one2one 仅 coeff/proto + MLP”要精确改成：**one2one 仅 one2one coeff head + point_head 有梯度，proto/P3/backbone 无梯度**。这正是 `seg_point_o2o`、`seg_point_refine_o2o`、`e2e_final_o2m` 要控制的风险。

### 14.3 E2：compile + DDP + boundary/MLP smoke

`scripts/smoke_point_head_ddp.py` 已支持：

```bash
python -m torch.distributed.run --nproc_per_node=2 scripts/smoke_point_head_ddp.py
python -m torch.distributed.run --nproc_per_node=2 scripts/smoke_point_head_ddp.py --compile --boundary
```

默认路径保持严格 `find_unused_parameters=False`，验证 MLP 真梯度 allreduce 与 dummy-only 零梯度路径。`--compile --boundary` 按 trainer 语义尝试 `attempt_compile()` 后用 DDP `static_graph=True`，同时打开 `seg_bnd=0.1` 与 `seg_point_boundary=True`。

本机 `yolo26-cu133` 环境没有 Triton，CUDA `torch.compile` 会在 first forward 抛 `TritonMissing`。已修 `attempt_compile()`：CUDA 下无 Triton 时明确 warning 并回退 uncompiled，避免训练 first forward 崩。当前专项 smoke 结果：

```text
compile: Triton unavailable for CUDA torch.compile, continuing uncompiled
[mlp] point_head grads synced across 2 ranks
[dummy] seg_point_refine=False: zero point_head grads OK
OK: 2-GPU point_head DDP smoke
```

这说明当前环境尚未真正覆盖 Inductor compiled graph；它覆盖的是 **compile=True 配置下的安全回退 + DDP static_graph + boundary + MLP + dummy**。若之后安装 working Triton，需要重新跑同一命令，确认真实 compiled graph 也通过。

---

## 15. 整体代码流程与当前进展总览（2026-07-05）

### 15.1 当前结论

当前代码已经从"只在训练侧有 Lite point loss"推进到 **训练侧 Point-head MLP + 推理/验证侧 PointRend 后处理均可用** 的状态，但默认仍保持 legacy 行为以保证可比性：

- **训练侧已落地**：`Segment26` 可由 pointrend YAML 构建 `PointHeadMLP`；`v8SegmentationLoss.single_mask_loss` 在 point 分支内逐实例采样不确定点，并在 `seg_point_refine=True` 时用 `point_head(point_feats, coarse)` 替代 Lite 的 `coarse`。
- **边界采样已落地**：point coords 默认可限制在 bbox ROI；`seg_point_boundary=True` 时用 GT Sobel boundary band 加权候选池，但仍保留 bbox 均匀候选，避免只盯边界而漏掉内部 FP/FN。
- **E2E 风险已可控**：one2one 的 P3/proto detach 是原始契约；新增 `seg_point_o2o`、`seg_point_refine_o2o`、`e2e_final_o2m` 用于把边界/refine 监督更多留在 one2many 梯度路径。
- **predict 已接线**：PyTorch backend + `seg_point_refine_infer=True` + `retina_masks=False` 时走 `process_mask_pointrend`；导出后端、无 feats、`retina_masks=True` 自动回退。
- **val 已接线但默认关**：默认 train-val/best.pt fitness 仍走确定性 `process_mask`；显式 `seg_point_refine_infer=True` 时 `SegmentationValidator` 走 PointRend full-res masks，并同步 GT mask full-res。
- **recipe200 baseline 未继续训练**：当前 baseline 仍停 epoch 107/200；旧 ckpt 没有 `point_head`，新增 PointRend 必须 finetune-from-checkpoint，不能 resume 改结构。

### 15.2 端到端训练流程

```text
YOLO.train(...)
  → SegmentationTrainer
  → SegmentationModel(yaml)
  → parse_model
      Segment26 第 4 参 point_hidden > 0 → build PointHeadMLP
  → init_criterion
      end2end → E2ELoss(one2many + one2one v8SegmentationLoss)
      normal  → v8SegmentationLoss
  → [可选] DistillationModel(student + teacher)
  → DDP / AMP / EMA
```

训练 batch 内的真实数据流：

```text
backbone/neck feats = [P3, P4, P5]
  → Detect.forward
      one2many: feats/proto 保持梯度
      one2one : feats detach；Segment26 里 proto 也 detach
  → Segment26.proto(x)
  → training + point_head 存在:
      point_refine_dummy = point_head.zero_loss()
      挂到 one2many/one2one，保证 DDP find_unused_parameters=False 安全
  → criterion
      point_w = seg_point * (seg_point_o2o if one2one else 1)
      point_feats = preds["feats"][0] only if point_w>0 && seg_point_refine branch enabled
      single_mask_loss:
        coeff @ proto → coarse logits
        bbox/GT boundary ROI 采点
        point_sample(P3, coords) + coarse_at
        PointHeadMLP → refined point logits
        focal + dice 加回 seg_loss 档
  → AMP backward → grad clip(10) → optimizer → EMA
  → epoch end: E2ELoss.update() 调 o2m/o2o schedule
```

梯度边界当前是清楚的：

- **one2many**：backbone/P3、Proto26、one2many mask coeff head、point_head 都有梯度，是边界/refine 真正塑形特征的主路径。
- **one2one**：backbone/P3/proto 无梯度；只有 one2one mask coeff head 与 point_head 有梯度。因此 boundary/refine finetune 不建议让后期完全 one2one-heavy。

### 15.3 val / predict 验收流程

默认分叉保持可比：

```text
seg_point_refine_infer=False
  train-time val / standalone val:
    coeff @ proto → process_mask → proto-resolution masks
    GT masks 下采样到 imgsz//4
    best.pt fitness 确定、可与 baseline 比较
  predict:
    标准 process_mask，可视化不直接体现 MLP 后处理
```

PointRend 直接验收分叉：

```text
seg_point_refine_infer=True
  PyTorch model eval returns ((outputs[0], proto), preds_dict)
  preds_dict["feats"][0] 提供 P3 fine feature
  SegmentationValidator / SegmentationPredictor 抽 (point_head, feats[0])
  process_mask_pointrend:
    coeff @ proto → proto crop → upsample to target shape
    ROI uncertain points → point_head refine → scatter delta
  val:
    pred masks full-res；GT masks 同步 full-res
  predict:
    retina_masks=False 时输出 refined full-res letterboxed masks
```

这条分叉使用随机采点，不能直接混进训练期 best.pt 选择。推荐 **val twice**：训练时固定 `seg_point_refine_infer=False` 选 checkpoint；训练结束后对同一个 `best.pt/last.pt` 再跑 `seg_point_refine_infer=False/True`，比较 AP75/AP95、Mask mAP50-95 和 latency。

### 15.4 当前代码变更状态

| 模块               | 当前进展                                                          | 备注                                        |
| ------------------ | ----------------------------------------------------------------- | ------------------------------------------- |
| YAML / parse_model | 已支持 `Segment26(..., point_hidden)`                             | 旧 YAML 不传第 4 参仍无 point head          |
| `PointHeadMLP`     | 已实现 Conv1d + residual zero-init                                | init 时 `refined == coarse`，冷启动 no-op   |
| training loss      | 已接入 ROI/boundary point loss + MLP refine                       | loss 向量长度不变，resume 同架构安全        |
| DDP dummy          | 已接入 forward `zero_loss()`                                      | criterion 跑真 MLP，dummy 只保参数可达      |
| E2E controls       | 已接入 `seg_point_o2o` / `seg_point_refine_o2o` / `e2e_final_o2m` | 处理 one2one detach 下监督变弱问题          |
| ckpt migration     | 已修 `DistillationModel.student_model` unwrap                     | recipe200 best.pt 可迁移到新 YAML           |
| predict            | 已接入 `process_mask_pointrend`                                   | 默认关；PyTorch-only；retina/export 回退    |
| val                | 已接入 PointRend 验收分叉                                         | 默认关；开关开时 full-res pred/GT 对齐      |
| compile            | 已加无 Triton fallback                                            | 当前环境只覆盖安全回退，未覆盖真实 Inductor |

### 15.5 当前训练进展

- `runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/results.csv` 最新仍为 epoch 107/200。
- epoch 107 指标：Box mAP50 `0.58887`、Box mAP50-95 `0.43624`、Mask mAP50 `0.57125`、Mask mAP50-95 `0.37592`。
- `best.pt` 与 `last.pt` 时间戳仍为 2026-07-04 11:43。
- 该 ckpt 顶层是 `DistillationModel`，student 为旧 `Segment26`，没有 `point_head`；必须新建 `yolo26s-seg-pointrend.yaml` 后 `load(best.pt)` 做 finetune。
- 2026-07-05 复核 GPU：3 张 RTX 5090 D 显存占用约 2 MiB，`nvidia-smi pmon` 未见训练/计算进程；`utilization.gpu` 查询偶见 GPU1/2 100% 但无进程占用，按显存与 pmon 判断没有当前训练任务。

### 15.6 下一步执行顺序

1. **开 Point-head finetune ft01**：从 recipe200 `best.pt` finetune 到 `yolo26s-seg-pointrend.yaml`，`resume=False`，训练期保持 `seg_point_refine_infer=False`。
2. **先保守显存**：recipe200 曾在后段 OOM，ft01 建议降 batch 或先用 `seg_point_num=64`；蒸馏 + teacher forward + point sampling 会抬高峰值。
3. **建议覆盖起点**：脚本默认偏激进（见 §15.7），第一轮建议显式覆盖为 `seg_point=0.2`、`seg_point_refine=True`、`seg_point_boundary=True`、`seg_point_o2o=0.0`、`seg_point_refine_o2o=False`、`e2e_final_o2m=0.3`。
4. **验收两条线**：训练日志看标准 fitness；结束后对相同 ckpt 跑 `seg_point_refine_infer=False/True` 双 val，重点看 Mask mAP50-95、AP75、AP95。
5. **若 AP75/AP95 不涨**：先查 P3 80×80 fine feature 上限与 GT mask 栅格/aliasing，再做 Proto26 fused feature 暴露实验，不优先盲目加大 point loss。

### 15.7 下一阶段运行代码完备性核验

下一阶段真正会运行的入口是：

```bash
cd /home/genesis/Train/Code/ultralytics
conda run -n yolo26-cu133 python scripts/finetune_yolo26s_seg_pointrend_coconut_b.py
```

已完成的无修改 smoke / preflight：

- `scripts/finetune_yolo26s_seg_pointrend_coconut_b.py` 存在，`ast.parse` 通过，`--help` 能正常打印参数。
- `runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt` 存在；`/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml` 存在。
- 物理 YAML 文件是 `ultralytics/cfg/models/26/yolo26-seg-pointrend.yaml`；`YOLO("yolo26s-seg-pointrend.yaml")` 的 scale alias 能正常解析到它。
- s 规模构建结果：`SegmentationModel` + `Segment26`，`point_head is not None`，`point_hidden=64`（YAML 128 经 s 宽度缩放后得到）。
- recipe200 `best.pt` 加载进新 PointRend YAML：`Transferred 844/850 items from pretrained weights`；新 `point_head` 仍存在，末层 weight abs sum 为 `0.0`，保持 zero-init no-op。
- 脚本使用 `YOLO("yolo26s-seg-pointrend.yaml").train(pretrained=best.pt, resume=False, ...)` 是正确路径：`Trainer.setup_model()` 会从 `args.pretrained` 加载 ckpt；若 ckpt 是无 teacher 的 `DistillationModel`，会 unwrap `student_model`；随后 `SegmentationTrainer.get_model(..., weights=...)` 调 `model.load(weights)`，不需要脚本里显式 `.load(best.pt)`。

脚本默认值与“保守起跑建议”有差异，需要运行前明确选择：

| 项                     | 脚本当前默认 | 更保守建议 | 说明                                                                             |
| ---------------------- | -----------: | ---------: | -------------------------------------------------------------------------------- |
| `epochs`               |           60 |      40-60 | ft 验证优先，不必一开始拉太长                                                    |
| `batch`                |           84 |   72 或 84 | recipe200 曾在 batch 90/multi_scale 0.25 OOM；84 默认不蒸馏，若开蒸馏建议降到 72 |
| `multi_scale`          |         0.15 |       0.15 | 已比 recipe200 0.25 保守                                                         |
| `seg_point`            |          0.5 |   0.2 起步 | 0.5 更激进；先看 AP75/AP95 是否稳涨                                              |
| `seg_point_num`        |           64 |         64 | 合理，控制显存                                                                   |
| `seg_point_boundary`   |         True |       True | 默认边界加权采样开启；如需纯 ROI 消融用 `--no-boundary`                          |
| `seg_point_o2o`        |          0.0 |        0.0 | 脚本硬编码 one2many-only point supervision，符合 detach 风险控制                 |
| `seg_point_refine_o2o` |        False |      False | 脚本硬编码，one2one 退回 Lite                                                    |
| `e2e_final_o2m`        |          0.1 |        0.3 | 若重点做边界/refine，建议显式 `--e2e-final-o2m 0.3` 保留 one2many 梯度           |
| `distill`              |        False | False 起步 | 不蒸馏省 teacher VRAM；若要开蒸馏需显式 `--distill` 并降 batch                   |

建议下一阶段第一条命令用保守覆盖，避免把“point loss 太强 / 后期 one2many 梯度太弱 / 显存”三件事混在一起：

```bash
conda run -n yolo26-cu133 python scripts/finetune_yolo26s_seg_pointrend_coconut_b.py \
  --epochs 60 \
  --batch 72 \
  --seg-point 0.2 \
  --seg-point-num 64 \
  --e2e-final-o2m 0.3 \
  --name yolo26s-seg-coconut-b-v2-pointrend-ft01
```

训练期脚本没有设置 `seg_point_refine_infer`，因此保持 default `False`：train-time val / best.pt fitness 走标准 `process_mask`，可和 recipe200 直接比较。训练结束后再单独跑 PointRend 直接验收：

```python
from ultralytics import YOLO

model = YOLO("runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft01/weights/best.pt")
metrics_std = model.val(
    data="/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml",
    imgsz=640,
    batch=72,
    device="0,1,2",
    seg_point_refine_infer=False,
)
metrics_refine = model.val(
    data="/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml",
    imgsz=640,
    batch=72,
    device="0,1,2",
    seg_point_refine_infer=True,
    save_json=False,
    save_txt=False,
)
```

运行前仍需人工确认的风险：

- **GPU util 偶发异常读数**：显存和 pmon 为空，但 `utilization.gpu` 对 GPU1/2 偶见 100%；启动长训前再跑一次 `nvidia-smi pmon -c 1`，以 pmon/显存为准。
- **真实 Inductor compile 未覆盖**：当前环境无 Triton，`compile=True` 会安全回退；ft01 建议保持默认 `compile=False`。
- **蒸馏路径显存**：脚本默认不蒸馏；如加 `--distill`，teacher+student 双 forward 会显著抬高峰值，建议先降 batch 并做短 smoke。
- **P3 分辨率上限**：当前 point head 细特征仍是 P3 80×80；若 AP75/AP95 不涨，下一步优先做 Proto26 fused feature 暴露/对比。
