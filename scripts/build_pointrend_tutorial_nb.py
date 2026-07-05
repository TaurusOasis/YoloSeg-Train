#!/usr/bin/env python3
"""Build the PointRend boundary-refine training tutorial notebook.

Emits ``docs/yolo26s-seg-pointrend-training-tutorial.ipynb``. The notebook follows the
**actual ft60 training code path** step-by-step (YAML build -> finetune load -> Segment26
forward + dummy -> E2ELoss -> single_mask_loss boundary ROI blended pool -> backward/DDP
mixed -> inference subdivision) and wires debug/comparison hooks
(``PointHeadHook``, ``PointSamplerHook``, boundary on/off, val-twice reference).

Run:
  conda run -n yolo26-cu133 python scripts/build_pointrend_tutorial_nb.py
"""

from __future__ import annotations

import json

NB_PATH = "docs/yolo26s-seg-pointrend-training-tutorial.ipynb"


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


cells = []

# ---------------------------------------------------------------------------
cells.append(
    md(
        """# YOLO26s-seg PointRend Boundary-Refine 训练流程 Tutorial

> 本 notebook 按 **正在跑的 `yolo26s-seg-coconut-b-v2-pointrend-ft60`** 的真实代码路径，逐段拆解
> PointRend boundary-refine 的训练逻辑，并挂调试/对比 hook，方便观察中间量、做消融对比。
>
> **两部分**：(T) 训练侧 point-head MLP + §2.6 GT Sobel boundary-band **混合候选池** ROI 采样；
> (I) 推理侧迭代细分（ft60 默认关 `seg_point_refine_infer=False`，本 notebook 末尾单独演示）。
>
> **运行环境**：`yolo26-cu133` conda env + `ultralytics` editable install。默认 **CPU** 跑（用 `yolo26n` 小模型，
> coco8-seg 1 epoch），**不占用 ft60 的 3 张 GPU**。ft60 跑完后再切 `device="cuda"` 复跑大模型。
>
> **代码引用**：每步标注 `file:line`，可点开对照。设计全貌见
> `docs/yolo26s-seg-pointrend-refine-head-design.md`。
>
> **ft60 实际 cfg**（与本 notebook 教学用的小 cfg 同结构、不同数值）：
> `epochs=60, batch=84, lr0=0.003, seg_point=0.5, seg_point_refine=True, seg_point_boundary=True,
> seg_point_o2o=0.0, seg_point_refine_o2o=False, e2e_final_o2m=0.1, seg_point_num=64,
> seg_point_roi=0.0, seg_comp=0, seg_bnd=0, seg_point_refine_infer=False, 无蒸馏, copy_paste=0.4+mixup=0.1+multi_scale=0.15`。"""
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## 环境与工具

导入 + 设备选择 + 一个把张量统计打印成一行的小工具。**默认 CPU**，避免和 ft60 抢 GPU。"""
    )
)

cells.append(
    code(
        '''import os, sys, copy, importlib
from pathlib import Path
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path.cwd()))  # editable ultralytics

# 默认 CPU：不干扰正在跑的 ft60（3-GPU）。ft60 结束后可改成 "cuda:0"。
DEVICE = "cpu"
print(f"torch={torch.__version__}  device={DEVICE}")

from ultralytics import YOLO
from ultralytics.nn.modules.head import PointHeadMLP, Segment26
import ultralytics.utils.loss as L  # 点采样器导入到此 namespace，hook 在此 monkey-patch
import ultralytics.utils.ops as ops
from ultralytics.utils.ops import process_mask, process_mask_pointrend, sobel_magnitude
from ultralytics.utils.mask_point_sampling import (
    get_uncertain_point_coords_in_roi, _weighted_rand_in_roi, _rand_in_roi, point_sample,
)

def stat(t, name=""):
    """打印张量形状 + min/mean/max，用于一眼看清中间量。"""
    if t is None:
        print(f"{name:18s} None"); return
    t = t.detach().float()
    print(f"{name:18s} shape={tuple(t.shape)}  min={t.min():.4f}  mean={t.mean():.4f}  max={t.max():.4f}")

# ft60 用的是 yolo26s；本 notebook 教学用 yolo26n（CPU 上 1 epoch 几十秒）。
TUT_YAML = "yolo26n-seg-pointrend.yaml"
FT60_PRETRAINED = "runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt"
print("TUT_YAML      =", TUT_YAML)
print("FT60_PRETRAINED exists:", Path(FT60_PRETRAINED).exists())'''
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 1 — 从 YAML 构建模型：`point_hidden` 第 4 参 → `PointHeadMLP`

`yolo26-seg-pointrend.yaml` 末行 `Segment26 [nc, 32, 256, 128]`：前 3 参走标准 Segment 路径，
**第 4 参 `128` = `point_hidden`**。`parse_model` 专门剥出它（`nn/tasks.py:1938-1946`）：

```python
point_hidden = None
if m is Segment26 and len(args) > 3:
    point_hidden = args[3]; args = args[:3]      # 剥出第 4 参
args.extend([reg_max, end2end, [ch[x] for x in f]])
...
if point_hidden is not None:
    args.append(make_divisible(min(point_hidden, max_channels) * width, 8))  # 缩放后追加
```

`Segment26.__init__`（`head.py:462`）据此建头：

```python
self.point_head = PointHeadMLP(ch[0], point_hidden) if point_hidden > 0 else None
```

`PointHeadMLP`（`head.py:268-311`）是 Conv1d 小 MLP，**末层 zero-init → 恒等残差**：`refined = coarse + delta`，初始 `delta=0`。这是"加 head 不改 step-0 行为"的关键。"""
    )
)

cells.append(
    code(
        """# 构建模型：YOLO() 解析 YAML → model.model 是 SegmentationModel，其 model[-1] 是 Segment26 head
yolo = YOLO(TUT_YAML)
seg_model = yolo.model                       # SegmentationModel
head = seg_model.model[-1]                    # Segment26
ph = head.point_head
print("head class:", type(head).__name__)
print("point_head :", ph)
assert isinstance(ph, PointHeadMLP)
print("in_channels =", ph.in_channels, " hidden_channels =", ph.hidden_channels)

# zero-init 验证：末层 Conv1d 的 weight/bias 全 0 → delta 恒 0 → refined == coarse
w, b = ph.mlp[-1].weight, ph.mlp[-1].bias
print(f"mlp[-1].weight all_zero={torch.equal(w, torch.zeros_like(w))}  "
      f"bias all_zero={torch.equal(b, torch.zeros_like(b))}")

# 直接验证恒等残差：随便喂一对 (point_feats, coarse) → refined 必须等于 coarse
n, c, p = 3, ph.in_channels, 64
feat = torch.randn(n, c, p)
coarse = torch.randn(n, p)
refined = ph(feat, coarse)
print("zero-init refined == coarse :", torch.allclose(refined, coarse, atol=1e-6))
print("=> 加 point_head 不改 step-0 输出，finetune 起点 == recipe200 best（no-op）")"""
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 2 — Finetune 加载 recipe200 best（ft60 的 step-0）

recipe200 ckpt 顶层是 `DistillationModel`，inner `student_model` 是 `SegmentationModel`，**head 无 `point_head`**。
`BaseModel.load()` 自动 unwrap `DistillationModel`，把 backbone+neck+旧 head 的权重搬到新 pointrend YAML
（实测 `Transferred 844/850`，6 个新参数 = point_head，**保留 zero-init**）。结构变了 → 必须 **finetune（`resume=False`）**，不能 resume。

> 本 cell 用 `yolo26s-seg-pointrend.yaml` 匹配 recipe200 架构；仅做加载+验证，**不训练**，CPU 上几秒。
> 若 `FT60_PRETRAINED` 不存在（ft60 还没产出 best 之外的 ckpt），cell 会跳过。"""
    )
)

cells.append(
    code(
        """pre = Path(FT60_PRETRAINED)
if not pre.exists():
    print(f"[skip] {pre} 不存在（ft60 未跑到产出 best.pt，或路径不同）。可改成 last.pt 重试。")
else:
    # 用 26s pointrend YAML 匹配 recipe200 架构，load 旧 best（finetune-from-checkpoint）
    yolo_s = YOLO("yolo26s-seg-pointrend.yaml")
    yolo_s.load(str(pre))                       # 打印 Transferred N/850；自动 unwrap DistillationModel
    ph_s = yolo_s.model.model[-1].point_head
    w = ph_s.mlp[-1].weight
    print("loaded point_head zero-init preserved :", torch.equal(w, torch.zeros_like(w)))
    print("=> step-0 与 recipe200 best 行为一致；point_head 从 0 开始学边界 refine")"""
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 3 — 前向路径（训练态）：`Segment26.forward` + dummy stash + `feats[0]=P3`

`Segment26.forward`（`head.py:464-485`）训练态关键动作：

```python
outputs = Detect.forward(self, x)              # end2end → preds = {"one2many":..., "one2one":...}
proto = self.proto(x)                           # Proto26 融合 P3/P4/P5
preds["one2many"]["proto"] = proto
preds["one2one"]["proto"]   = proto.detach()    # one2one 的 proto 被 detach（head.py:472-474）
if self.training and point_head is not None:
    dummy = self.point_head.zero_loss()         # 标量 0，但连着所有 point_head 参数
    preds["one2many"]["point_refine_dummy"] = dummy   # 同一 dummy 对象塞两支
    preds["one2one"]["point_refine_dummy"]   = dummy
return preds                                    # 训练态直接返回 preds dict
```

**dummy 的作用**：`zero_loss()`（`head.py:309-311`）= `sum(p.sum()*0 for p in params)`，值恒 0 但**梯度图连到 point_head 全部参数** → DDP `find_unused_parameters=False` / `static_graph=True` 时 point_head 参数在 forward 图里可达，不报 unused。真梯度来自 criterion 里的 MLP（Step 5）。这就是"混合 DDP 解法"。

`feats[0]` = neck 最高分辨率特征 P3（也是 Proto26 融合基底）—— loss 侧用它作 fine feature。"""
    )
)

cells.append(
    code(
        '''# 训练态前向，看 preds 结构。先注册 PointHeadHook 抓 MLP 的输入输出（Step 5 详用）
class PointHeadHook:
    """forward_hook on point_head：抓 (feat_p, coarse_p) -> refined，记录 |delta| 统计。"""
    def __init__(self):
        self.records = []
        self._h = None
    def attach(self, ph):
        self._h = ph.register_forward_hook(self._fn)
    def _fn(self, module, inp, out):
        feat_p, coarse = inp[0], inp[1]
        refined = out
        delta = (refined - coarse.reshape_as(refined)).detach()
        self.records.append(dict(
            n=feat_p.shape[0], p=feat_p.shape[-1],
            delta_abs_mean=delta.abs().mean().item(),
            delta_max=delta.abs().max().item(),
            coarse_mean=coarse.float().mean().item(),
        ))
    def remove(self):
        if self._h: self._h.remove()
    def summary(self):
        import numpy as np
        if not self.records: print("(no point_head calls)"); return
        d = np.array([r["delta_abs_mean"] for r in self.records])
        print(f"point_head called {len(self.records)}x  delta_abs_mean: "
              f"min={d.min():.5f}  mean={d.mean():.5f}  max={d.max():.5f}")

phook = PointHeadHook(); phook.attach(head.point_head)

seg_model.train()
x = torch.randn(1, 3, 96, 96)
preds = seg_model(x)                            # 训练态返回 dict
print("preds type:", type(preds).__name__, "keys:", list(preds.keys()))
if "one2many" in preds:
    o2m = preds["one2many"]
    print("one2many keys:", list(o2m.keys()))
    print("dummy is scalar 0 :", torch.equal(o2m["point_refine_dummy"], torch.zeros((), device=o2m["point_refine_dummy"].device)))
    print("feats[0] (P3) shape:", tuple(o2m["feats"][0].shape))
    print("proto shape:", tuple(o2m["proto"].shape))
# 训练态还没跑 loss → point_head 不会被调（MLP 只在 criterion 里跑），hook 此刻为空：
phook.summary()
phook.remove()'''
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 4 — 损失分发：`E2ELoss` 加权 one2many / one2one

`SegmentationModel.init_criterion()`（`nn/tasks.py:593`）end2end 下返回 `E2ELoss(self, v8SegmentationLoss)`：
两个 `v8SegmentationLoss` 实例，`loss_branch` 分别标 `"one2many"` / `"one2one"`（`loss.py:1375-1376`）。

`E2ELoss.__call__`（`loss.py:1378-1385`）：
```python
loss = loss_one2many[0]*self.o2m + loss_one2one[0]*self.o2o
```
权重 schedule（`loss.py:1365-1395`）：`o2m` 从 **0.8 线性衰减到 `e2e_final_o2m`（ft60=0.1）**，`o2o=1-o2m`。
→ **后期 one2many 权重低 → 带 backbone 梯度的 boundary 监督变弱**（设计文档 C2 / §2.2）；ft60 保持 0.1（未提），是验收时要重点看的怀疑点。

`v8SegmentationLoss.__init__`（`loss.py:503`）持有 point_head 引用：
```python
self.point_head = getattr(model.model[-1], "point_head", None)
```
分支级开关（`loss.py:510-522`）：`_point_loss_gain` 给 one2one 乘 `seg_point_o2o`（ft60=0 → one2one 无点损失）；
`_point_refine_enabled` 给 one2one 再 AND `seg_point_refine_o2o`（ft60=False → one2one 不跑 MLP）。**ft60 点监督只在 one2many**。"""
    )
)

cells.append(
    code(
        """# 构建 criterion 看 schedule。需要给 model 挂上 args（hyp），用 get_cfg 注入 ft60 同款 seg cfg。
from ultralytics.cfg import get_cfg, DEFAULT_CFG
cfg = get_cfg(DEFAULT_CFG, overrides=dict(
    seg_point=0.5, seg_point_refine=True, seg_point_boundary=True,
    seg_point_o2o=0.0, seg_point_refine_o2o=False, e2e_final_o2m=0.1,
    seg_point_num=64, seg_point_roi=0.0, seg_point_importance=0.75, seg_point_oversample=3,
    seg_comp=0.0, seg_bnd=0.0, epochs=60, overlap_mask=True,
))
seg_model.args = cfg              # criterion.hyp = model.args

crit = seg_model.init_criterion()           # E2ELoss(self, v8SegmentationLoss)
print("criterion:", type(crit).__name__)
print("o2m_init=", round(crit.o2m,3), " o2o_init=", round(crit.o2o,3),
      " final_o2m=", crit.final_o2m, " epochs=", crit.one2one.hyp.epochs)
print("one2many.loss_branch=", crit.one2many.loss_branch, " one2one.loss_branch=", crit.one2one.loss_branch)
print("one2many point_head is not None:", crit.one2many.point_head is not None)

# 看 schedule 衰减曲线
import numpy as np
xs = np.arange(0, 60, 5)
o2ms = [crit.decay(int(x)) for x in xs]
print("epoch -> o2m :", {int(x): round(float(v),3) for x,v in zip(xs,o2ms)})
print("=> 后期 o2m=0.1：one2many（带 backbone 梯度的点监督）权重低；这是 C2 怀疑点")"""
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 5 — `single_mask_loss` 点分支：**boundary ROI 混合候选池**（ft60 的核心）

`v8SegmentationLoss.loss`（`loss.py:524-594`）→ `calculate_segmentation_loss`（`loss.py:735-822`）逐图循环 →
`single_mask_loss`（`loss.py:597-733`）。点分支（`loss.py:677-731`，ft60 走这段）：

```python
pm4 = pred_mask.float().unsqueeze(1); gm4 = gt_mask.float().unsqueeze(1)
with torch.no_grad():
    use_roi = roi_margin >= 0.0 or boundary_w          # ft60: roi=0.0, boundary=True -> use_roi=True
    if use_roi:
        eff_margin = max(roi_margin, 0.0) if boundary_w else roi_margin   # =0.0
        boxes_norm = xyxy / [w_m,h_m,w_m,h_m]
        weight_map = sobel_magnitude(gt_mask) if boundary_w else None      # GT 边界 band
        coords = get_uncertain_point_coords_in_roi(
            pm4.detach(), calculate_uncertainty, num_points,
            oversample_ratio, importance_ratio, boxes_norm, margin=eff_margin,
            weight_map=weight_map)                  # 混合候选池在此
    pg = point_sample(gm4, coords, ...).squeeze(1)
coarse = point_sample(pm4, coords, ...).squeeze(1)
if point_head is not None and point_feats is not None:
    pf = point_sample(point_feats, coords, ...)      # 采 P3 细特征
    pl = point_head(pf, coarse)                       # MLP refine（真梯度在此）
else:
    pl = coarse
total += point_w * (focal_i + dice_i).sum()           # 折进 seg_loss
```

**混合候选池**（`mask_point_sampling.py:218-249`，本 session 修复的 §2.6 进阶冲突）：
oversample = **50% `_weighted_rand_in_roi`（Sobel 边界加权）+ 50% `_rand_in_roi`（bbox 均匀）**合并后做 pred-uncertainty **top-k**；
25% 随机余量**始终 bbox 均匀**。→ 边界过采样约 5× 像素占比，同时**内部错但不确定的 FP/FN 区域仍能被 top-k 选到**（不被边界加权半池饿死）。

`loss[1] += point_refine_dummy`（`loss.py:590-591`）把 forward 的 dummy 加进来（值 0，连参数）；
`loss[1] *= self.hyp.box`（`loss.py:593`）seg 乘 box gain（7.5）；`return loss * batch_size`（:594）。

下面挂 `PointSamplerHook`（monkey-patch 采样器）抓 coords + weight_map，再跑一个真训练 epoch 看 hook 输出。"""
    )
)

cells.append(
    code(
        """# PointSamplerHook：monkey-patch loss.py namespace 里的采样器，抓 weight_map + 最终 coords，
# 并按 weight_map 给最终点打“边界 vs 内部”标签。
class PointSamplerHook:
    def __init__(self):
        self.records = []
        self._orig = None
    def attach(self):
        self._orig = L.get_uncertain_point_coords_in_roi
        def wrapped(logits, uf, num_points, osr, isr, boxes_norm, margin=0.0, weight_map=None):
            coords = self._orig(logits, uf, num_points, osr, isr, boxes_norm, margin=margin, weight_map=weight_map)
            # 用 weight_map 给最终点打标签（无 weight_map=boundary off → 全 -1 占位）
            if weight_map is not None:
                wm = weight_map.unsqueeze(1)                 # (N,1,H,W)
                w_at = point_sample(wm, coords, align_corners=False).squeeze(1)  # (N,P)
                bnd_frac = (w_at > 0).float().mean(dim=1)     # 每实例边界点占比
            else:
                bnd_frac = torch.full((coords.shape[0],), -1.0)
            self.records.append(dict(
                n=coords.shape[0], p=coords.shape[1],
                weight_map_is_None=(weight_map is None),
                bnd_frac_mean=bnd_frac.mean().item(),
                margin=margin,
            ))
            return coords
        L.get_uncertain_point_coords_in_roi = wrapped
    def remove(self):
        if self._orig is not None:
            L.get_uncertain_point_coords_in_roi = self._orig
    def summary(self, label=""):
        import numpy as np
        if not self.records: print(f"{label}(no sampler calls)"); return
        bf = np.array([r["bnd_frac_mean"] for r in self.records if r["bnd_frac_mean"] >= 0])
        wm_none = sum(r["weight_map_is_None"] for r in self.records)
        print(f"{label}calls={len(self.records)}  weight_map_None_calls={wm_none}")
        if bf.size:
            print(f"{label}  最终点中边界点占比: min={bf.min():.3f} mean={bf.mean():.3f} max={bf.max():.3f}")

print("PointHeadHook + PointSamplerHook 已定义。下一步训练时挂上。")"""
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 6 — 跑一个真训练 epoch（coco8-seg，CPU，ft60 同款 seg cfg），收集 hook 输出

用 `YOLO.train(...)` 跑真实训练循环（与 ft60 同一代码路径），hook 在循环中自动触发。
小 cfg：`yolo26n` + imgsz 96 + batch 2 + 1 epoch + coco8-seg（8 图）。CPU 上约 30–90 秒。

> `optimizer="SGD"`（教程用；ft60 用 `MuSGD`）。`close_mosaic=0`、关 copy_paste/mixup/multi_scale 让 CPU 快。"""
    )
)

cells.append(
    code(
        """import shutil, tempfile
phook = PointHeadHook(); phook.attach(yolo.model.model[-1].point_head)
shook = PointSamplerHook(); shook.attach()

# 先在 cpu 上跑一个 epoch，seg cfg 用 ft60 同款
res = yolo.train(
    data="coco8-seg.yaml",
    epochs=1, imgsz=96, batch=2, device=DEVICE, workers=0,
    optimizer="SGD", lr0=0.01, cos_lr=True, close_mosaic=0,
    project="runs/segment", name="pointrend-tutorial-on", exist_ok=True,
    save=False, plots=False, verbose=False,
    seg_point=0.5, seg_point_refine=True, seg_point_boundary=True,
    seg_point_o2o=0.0, seg_point_refine_o2o=False, e2e_final_o2m=0.1,
    seg_point_num=64, seg_point_roi=0.0, seg_point_importance=0.75, seg_point_oversample=3,
    seg_comp=0.0, seg_bnd=0.0,
)

shook.summary("[boundary=True] ")
phook.summary()
print("=> 看 boundary=True 时最终点边界占比（应明显高于 uniform 的 ~0.1）")"""
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 7 — 反向 / DDP 混合解法：dummy 连参数，真梯度来自 criterion

`zero_loss()` 让 point_head 参数进 forward 图（DDP unused 安全）；**真梯度**来自 criterion 里 `point_head(pf, coarse)` 的 focal+dice。
训练一个 step 后，point_head 末层 weight 应有非零 grad。单卡 CPU 无 DDP allreduce，但 grad 路径与 3-GPU 一致。

> ft60 是 3-GPU DDP 实跑——它正常收敛且无 `find_unused_parameters` 报错，**本身即 3-GPU 上对混合 DDP 解法的实证**（比 #10 计划的 2-GPU 验证更强）。"""
    )
)

cells.append(
    code(
        """# 上一个 cell 的 train() 已做过 backward；检查 point_head 真有梯度
ph = yolo.model.model[-1].point_head
g = ph.mlp[-1].weight.grad
if g is None:
    print("grad is None —— 可能该 epoch 没有正样本或点分支未触发；重跑 Step 6 一次。")
else:
    print(f"mlp[-1].weight.grad: shape={tuple(g.shape)}  abs_mean={g.abs().mean():.6f}  "
          f"nonzero={int((g.abs() > 0).sum().item())}/{g.numel()}")
    print("=> grad 非零 = MLP 真梯度经 criterion 流回（dummy 只是 DDP 占位，不提供学习信号）")"""
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 8 — 对比消融：`seg_point_boundary=True` vs `False`

同一 cfg 跑两次，对比采样器的**边界点占比** + MLP **delta 强度**。
- `boundary=True`：混合候选池（50% Sobel 边界 + 50% 均匀）→ 边界点占比高。
- `boundary=False`：纯 ROI 均匀采样（`weight_map=None`）→ 边界点占比≈自然比例（~0.1）。

> 消融必须固定 `seg_point_boundary`（其混合池改变了 point 监督语义，不可与 ROI/uniform 自由组合）——见设计文档 §10.2。"""
    )
)

cells.append(
    code(
        """def run_one(boundary, name):
    y = YOLO(TUT_YAML)
    ph_ = PointHeadHook(); ph_.attach(y.model.model[-1].point_head)
    sh_ = PointSamplerHook(); sh_.attach()
    y.train(
        data="coco8-seg.yaml", epochs=1, imgsz=96, batch=2, device=DEVICE, workers=0,
        optimizer="SGD", lr0=0.01, cos_lr=True, close_mosaic=0,
        project="runs/segment", name=name, exist_ok=True, save=False, plots=False, verbose=False,
        seg_point=0.5, seg_point_refine=True, seg_point_boundary=boundary,
        seg_point_o2o=0.0, seg_point_refine_o2o=False, e2e_final_o2m=0.1,
        seg_point_num=64, seg_point_roi=0.0, seg_point_importance=0.75, seg_point_oversample=3,
        seg_comp=0.0, seg_bnd=0.0,
    )
    sh_.summary(f"[boundary={boundary}] ")
    import numpy as np
    if ph_.records:
        d = np.array([r["delta_abs_mean"] for r in ph_.records])
        print(f"[boundary={boundary}] delta_abs_mean: min={d.min():.5f} mean={d.mean():.5f} max={d.max():.5f}")
    ph_.remove(); sh_.remove()
    return y

y_bnd  = run_one(True,  "pointrend-tutorial-bnd")
y_uni  = run_one(False, "pointrend-tutorial-uni")
print("\\n=> 对比两次的“最终点边界占比”与 delta_abs_mean。boundary=True 应边界点更密、delta（学习信号）更集中在边界")"""
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 9 — 推理侧迭代细分（I）：ft60 默认关，本节单独演示

`process_mask_pointrend`（`ops.py:584-648`）——粗 logits crop→上采样到 full-res → K 次循环：
全图 `get_uncertain_point_coords_in_roi` 采点（**推理无 GT，故不用 Sobel weight_map**）→
`point_head(feat, coarse_at)` refine → `_scatter_refine_delta` 只写回 `refined - coarse_at` delta。

**zero-init 时 delta=0 → 与标准 `process_mask` bitwise 等价**（`test_process_mask_pointrend_basic` 已锁）。
ft60 `seg_point_refine_infer=False` → 训练内 val 走 `process_mask`（确定性），best.pt fitness 不受细分随机性污染。
**MLP 直接收益**要等 ft60 跑完，对 best.pt 做"val twice"（下一节）。

> 推理采点用 `torch.rand` → **随机** → 不能入 best.pt fitness。这是训练内 val 保持 `process_mask` 的理由（设计文档 §2.8）。"""
    )
)

cells.append(
    code(
        """# eval 前向：Segment26.eval() 返回 ((det, proto), feats_dict)
yolo.model.eval()
with torch.no_grad():
    raw = yolo.model(torch.randn(1, 3, 96, 96))
print("eval raw[0] is tuple:", isinstance(raw[0], tuple), " raw[1] is dict:", isinstance(raw[1], dict))
det, proto = raw[0]
feats_dict = raw[1]
feats = feats_dict.get("one2one", feats_dict).get("feats") if isinstance(feats_dict, dict) else None
print("proto shape:", tuple(proto.shape), " feats[0] (P3) shape:", tuple(feats[0].shape) if feats else None)

# 跑标准 process_mask 与 pointrend 细分（zero-init head），验证 bitwise 等价
from ultralytics.utils.ops import process_mask, process_mask_pointrend
# 用一个有检测的合成系数（NMS 后给个 bbox）
import torch as T
n = 2
masks_in = T.randn(n, proto.shape[0])
bboxes = T.tensor([[8., 8., 80., 80.], [10., 10., 70., 70.]])
shape = (96, 96)
base = process_mask(proto, masks_in, bboxes, shape, upsample=True)
refined_masks = process_mask_pointrend(
    proto, masks_in, bboxes, shape, head.point_head, feats[0][:1],
    num_points=64, oversample_ratio=3, importance_ratio=0.75, subdivisions=3, roi_margin=0.0)
print("zero-init pointrend == process_mask :", torch.equal(refined_masks, base))
print("=> 未训练 MLP 推理细分是 no-op；训练后 delta 会改边界像素（且只在 bbox 内）")

# 演示训练后（非零 delta）的局部化：手动给末层加小扰动
with torch.no_grad():
    head.point_head.mlp[-1].weight.normal_(0, 0.3)
    head.point_head.mlp[-1].bias.normal_(0, 0.3)
trained = process_mask_pointrend(
    proto, masks_in, bboxes, shape, head.point_head, feats[0][:1],
    num_points=64, oversample_ratio=3, importance_ratio=0.75, subdivisions=3, roi_margin=0.0)
changed = (trained != base)
for i,(x1,y1,x2,y2) in enumerate(bboxes.int().tolist()):
    outside = changed[i,:y1,:].sum()+changed[i,y2:,:].sum()+changed[i,:,:x1].sum()+changed[i,:,x2:].sum()
    print(f"inst {i}: bbox外改变像素数={int(outside)} （应≈0，delta scatter 只在 bbox ROI 内）")
# 复位 head，避免影响后续
head.point_head.mlp[-1].weight.zero_(); head.point_head.mlp[-1].bias.zero_()"""
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Step 10 — 映射到真实 ft60 + "val twice" 验收 hook

ft60 训练内 val 用 `seg_point_refine_infer=False` → results.csv 的 Mask mAP 是**间接收益**
（`process_mask`，MLP 不参与推理）。**MLP 直接收益**测不到。`SegmentationValidator` 已接 pointrend 路径
（`val.py:_resolve_point_head` + `postprocess` 在 `_uses_pointrend()`=True 且非 native 时走 `process_mask_pointrend`）。

**验收协议**（设计文档 §2.8 "val twice"）：ft60 跑完后对 `best.pt`/`last.pt` 各 val 两次：

| val | `seg_point_refine_infer` | 含义 |
|---|---|---|
| 1 | `False` | 间接基线（=训练日志，确定性） |
| 2 | `True` | 直接收益（K=3 细分，MLP 参与，**随机**→固定种子/3 次均值） |

比 `metrics.seg.map50-95` / `all_ap[:,9].mean()`(AP95) / AP75。**Δ = MLP 直接收益**，可归因。

下面是可直接用于 ft60 best.pt 的 val-twice 脚本（GPU，ft60 跑完后执行）。"""
    )
)

cells.append(
    code(
        '''# === ft60 跑完后执行（GPU）=== val twice 验收：间接基线 vs 直接收益
def val_twice(ckpt, data, imgsz=640, device="cuda:0", seed=0):
    """对同一 ckpt val 两次，返回 (indirect, direct) 两份 metrics。"""
    import copy
    torch.manual_seed(seed)
    # 间接基线
    m_ind = YOLO(ckpt).val(data=data, imgsz=imgsz, device=device,
                          seg_point_refine_infer=False, save_json=False, save_txt=False, verbose=False)
    # 直接收益（细分；固定种子压随机性）
    torch.manual_seed(seed)
    m_dir = YOLO(ckpt).val(data=data, imgsz=imgsz, device=device,
                          seg_point_refine_infer=True, seg_point_subdiv_k=3,
                          save_json=False, save_txt=False, verbose=False)
    seg = lambda m: (m.seg.map, m.seg.map50, m.seg.map75, m.seg.all_ap[:, 9].mean() if hasattr(m.seg,"all_ap") else None)
    print(f"ckpt={ckpt}")
    print(f"  indirect(seg_point_refine_infer=False): "
          f"map50-95={m_ind.seg.map:.5f} map50={m_ind.seg.map50:.5f} map75={m_ind.seg.map75:.5f}")
    print(f"  direct  (seg_point_refine_infer=True ): "
          f"map50-95={m_dir.seg.map:.5f} map50={m_dir.seg.map50:.5f} map75={m_dir.seg.map75:.5f}")
    print(f"  Δ direct-indirect: map50-95={m_dir.seg.map-m_ind.seg.map:+.5f} "
          f"map75={m_dir.seg.map75-m_ind.seg.map75:+.5f}")
    return m_ind, m_dir

# 用法（取消注释执行；需 ft60 产出 best.pt 且 GPU 空闲）：
# FT60_DATA = "/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml"
# FT60_BEST = "runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft60/weights/best.pt"
# val_twice(FT60_BEST, FT60_DATA, imgsz=640, device="cuda:0", seed=0)
# # 重复 3 个 seed 取均值更稳（细分采点随机）：
# # for s in range(3): val_twice(FT60_BEST, FT60_DATA, device="cuda:0", seed=s)
print("val_twice() 已定义。ft60 跑完、GPU 空闲后取消上面注释执行。")'''
    )
)

# ---------------------------------------------------------------------------
cells.append(
    md(
        """## Hook 速查（复制即用）

| Hook | 挂在哪 | 抓什么 | 用途 |
|---|---|---|---|
| `PointHeadHook` | `head.point_head.register_forward_hook` | `(feat_p, coarse) -> refined`，`delta=refined-coarse` | 看 MLP 学习信号强度（zero-init 时 delta≈0） |
| `PointSamplerHook` | monkey-patch `ultralytics.utils.loss.get_uncertain_point_coords_in_roi` | `weight_map`、最终 `coords`、边界点占比 | 看混合候选池边界过采样 vs 均匀 |
| forward hook on `head.proto` | `head.proto.register_forward_hook` | proto | 看 Proto26 输出 |
| `val_twice` | — | indirect vs direct Mask mAP | MLP 直接收益量化验收 |

**挂到真实 ft60 best.pt 上观察**（不训练，只推理 + hook）：
```python
y = YOLO("yolo26s-seg-pointrend.yaml").load(FT60_BEST)
ph = PointHeadHook(); ph.attach(y.model.model[-1].point_head)
y.predict(source, seg_point_refine_infer=True, retina_masks=False, imgsz=640)
ph.summary()  # 看推理细分时 MLP 的 delta（训练后应非零且集中在边界）
```

---
**对应 ft60 代码路径速查**：
1. YAML 第 4 参 → `parse_model` 剥 `point_hidden`（`tasks.py:1938-1946`）→ `Segment26.__init__` 建 `PointHeadMLP` zero-init（`head.py:462`）
2. `Segment26.forward` 训练态 stash `zero_loss()` dummy 到 one2many/one2one（`head.py:475-478`）
3. `E2ELoss` 加权 o2m(0.8→0.1)/o2o（`loss.py:1357-1395`）；分支级 `seg_point_o2o=0`/`seg_point_refine_o2o=False` → 点监督只在 one2many（`loss.py:510-522`）
4. `single_mask_loss` 点分支：`sobel_magnitude(gt)` → 混合候选池（50/50 边界+均匀）→ top-k → `point_head(pf, coarse)` → focal+dice（`loss.py:677-731`，`mask_point_sampling.py:218-249`）
5. `loss[1] += point_refine_dummy`；`loss[1] *= hyp.box`（7.5）；`*batch_size`（`loss.py:590-594`）
6. 反向：dummy 连参数（DDP safe）+ criterion 真梯度（`head.py:309-311`）
7. 推理细分 `process_mask_pointrend`（`ops.py:584-648`），ft60 默认关；验收走 `val_twice`"""
    )
)


# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (yolo26-cu133)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "title": "YOLO26s-seg PointRend Boundary-Refine 训练流程 Tutorial",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {NB_PATH}  ({len(cells)} cells)")
