# YOLO26s-seg PointRend Boundary-Refine — 代码全局梳理

> 状态快照（2026-07-07）：PointRend 实现已提交 `9b4e3834d`；ft60 / ft60-nobnd 续训练均完成 60/60。**实验结果见 [yolo26s-seg-pointrend-experiment-results.md](yolo26s-seg-pointrend-experiment-results.md)**。设计决策与待办见 `yolo26s-seg-pointrend-refine-head-design.md`。

## 1. 提交与文件布局

| 范围                                                                                               | 状态                 |
| -------------------------------------------------------------------------------------------------- | -------------------- |
| PointRend 实现（28 文件，+6105/−28）                                                               | 已提交 `9b4e3834d`   |
| Tutorial hook 修复（`scripts/build_pointrend_tutorial_nb.py` + `docs/...training-tutorial.ipynb`） | 未提交（本 session） |

实现触及的核心文件：

- `ultralytics/nn/modules/head.py` — `PointHeadMLP` + `Segment26`（point_head 子模块 + DDP dummy）
- `ultralytics/nn/tasks.py` — `parse_model` 对 `Segment26` 可选第 4 参 `point_hidden` 的处理；`SegmentationModel.init_criterion` 走 `E2ELoss(self, v8SegmentationLoss)`
- `ultralytics/utils/loss.py` — `v8SegmentationLoss` 点分支 + `single_mask_loss` + `calculate_segmentation_loss` + `E2ELoss` schedule
- `ultralytics/utils/mask_point_sampling.py` — ROI / 混合候选池采样器 + 逐实例 focal/dice
- `ultralytics/utils/ops.py` — `sobel_magnitude` / `_scatter_refine_delta` / `process_mask_pointrend`
- `ultralytics/models/yolo/segment/predict.py` — `_extract_pointrend` + `construct_result` 推理细分
- `ultralytics/models/yolo/segment/val.py` — validator 侧 pointrend 路径（验收分叉）
- `ultralytics/cfg/default.yaml` + `cfg/__init__.py` — 13 个 `seg_point*` / `seg_comp` / `seg_bnd` / `e2e_final_o2m` / `dali` 设置 + 类型注册
- `ultralytics/cfg/models/26/yolo26-seg-pointrend.yaml` — head 末行 `Segment26, [nc, 32, 256, 128]`（128=point_hidden），`end2end: True`
- `ultralytics/engine/trainer.py` — distill-checkpoint unwrap（finetune-from-recipe200）+ resume args 白名单
- `ultralytics/optim/muon.py` — MuSGD 对 `PointHeadMLP` Conv1d 3D 权重的正交化缩放修正
- `ultralytics/models/yolo/detect/train.py` + `ultralytics/data/dali_seg.py` — 实验性 DALI seg 数据加载
- `ultralytics/utils/torch_utils.py` — CUDA compile 缺 Triton 的降级 warning

## 2. 整体数据流

```
训练: img → backbone → neck → Segment26.forward
  ├─ Detect.forward → feats/det  (end2end: one2many 带梯度 / one2one detach)
  ├─ Proto26 → proto (o2m 带 grad, o2o detach)
  ├─ point_head.zero_loss() → point_refine_dummy  stash 进两支 preds   ← DDP 占位
  └─ 训练态返回 preds dict
   ↓ E2ELoss → v8SegmentationLoss(one2many) + v8SegmentationLoss(one2one)
   ↓ loss.loss → calculate_segmentation_loss → 单图循环 single_mask_loss
   ↓ point 分支: get_uncertain_point_coords_in_roi(boundary 混合池) → point_sample
   ↓ point_head(pf, coarse) → refined   ← 真梯度来源 (focal+dice)
   ↓ loss[1] += point_refine_dummy  (值恒 0, 只连参数)

推理: AutoBackend(pt) 透传 → preds[1] head dict (含 feats)
  ├─ predict.py: _extract_pointrend → process_mask_pointrend (K 次细分)
  └─ val.py:     _extract_pointrend → process_mask_pointrend (seg_point_refine_infer=True 时)
```

## 3. 五大实现支柱

### A. PointHeadMLP + Segment26（`nn/modules/head.py`）

- **PointHeadMLP**（head.py:268）：Conv1d 链 `(C+1 → H → H → 1)`，输入 `cat(point_feats, coarse_logits)`，**末层 zero-init 恒等残差** `refined = coarse + delta`（init delta=0 → 加 head 是 step-0 no-op，finetune 起点 == recipe200 best）。`forward` 全程 `.float()`（bf16 下 fp32-internal）。`zero_loss()` = `sum(p.sum()*0)` 连接所有参数供 DDP 记账。
- **Segment26**（head.py:388）：第 6 形参 `point_hidden`（>0 才建 point_head）；`forward` 在 end2end 训练态把 `zero_loss()` dummy stash 到 `preds["one2many"]["point_refine_dummy"]` 与 `preds["one2one"][...]`，并用 `getattr(self,"point_head",None)` 守卫（pre-PointRend teacher 经 `torch.load` 重建不重跑 `__init__`、`__dict__` 无该键，蒸馏 forward 会崩 — 已修）。

### B. 损失分发（`utils/loss.py`）

- **v8SegmentationLoss**（loss.py:494）：`__init__` 抓 `self.point_head = getattr(model.model[-1],"point_head",None)`；`hyp_get` 兼容 namespace/dict。
- **分支门控**：`_point_loss_gain`（one2one 乘 `seg_point_o2o`，ft60=0 → 点监督只在 one2many）；`_point_refine_enabled`（one2one 还需 `seg_point_refine_o2o`，ft60=False）。
- **loss()**（loss.py:524）：`point_feats = preds["feats"][0] if point_w>0 and refine_enabled and point_head is not None else None`；`loss[1] += point_refine_dummy`（DDP 占位）；末尾 `loss[1] *= self.hyp.box`。
- **single_mask_loss**（staticmethod, loss.py:595）：点分支 `use_roi = roi_margin>=0 or boundary_w`；`weight_map = sobel_magnitude(gt_mask) if boundary_w`；`coords = get_uncertain_point_coords_in_roi(...)`；`pl = point_head(pf, coarse) if point_head and point_feats else coarse`；`total += point_w*(focal+dice)`。**显存优化**：per-image `point_feats[i:i+1]` 保持 (1,C,H,W)，sample 时 merge coords 单次 grid_sample（expand 的 backward 会物化 (N,C,H,W) 连续梯度 → recipe200 batch84 OOM）。
- **E2ELoss**（loss.py:1357）：`o2m 0.8→e2e_final_o2m(0.1)` 线性衰减、`o2o=1-o2m`；seg 经 `init_criterion`（tasks.py:593）用 `E2ELoss(self, v8SegmentationLoss)`。

### C. 点采样（`utils/mask_point_sampling.py`）

- `_rand_in_roi`（:82）：bbox+margin 均匀采点，退化 bbox 回退 full-grid。
- `_weighted_rand_in_roi`（:112）：bbox 内按 `weight_map`（Sobel magnitude）`multinomial` 加权采点；两级回退（退化 bbox→full-grid、bbox 有效但权重全 0→bbox 内均匀）。
- `get_uncertain_point_coords_in_roi`（:186）：**混合候选池** — oversample = 50% Sobel 加权 + 50% 均匀合并后做 pred-uncertainty top-k；25% 随机余量**始终** bbox 均匀。边界过采样 ~5×、内部 FP/FN 仍可达 top-k（解了"整池 Sobel 加权饿死内部不确定点"的冲突）。`weight_map=None` 时退化为纯 ROI 均匀。

### D. 推理细分（`utils/ops.py` + predict/val）

- **sobel_magnitude**（ops.py:493）：fp32-internal（`torch.autocast(enabled=False)`）。
- **`_scatter_refine_delta`**（ops.py:526）：把 `refined-coarse` delta 按 nearest 像素回写到 logit map（非采样像素保留双线性值，碰撞 last-write）。
- **`process_mask_pointrend`**（ops.py:545）：粗 logits crop→双线性上采样到 full-res → K 次循环（全图 `get_uncertain_point_coords_in_roi` 采点、**推理无 GT 故无 weight_map** → `point_head(feat,coarse_at)` → scatter delta）→ `.gt_(0)`。zero-init 下与 `process_mask` bitwise 等价（单测锁）。
- **predict.py**（:79）+ **val.py**（:115）：`_extract_pointrend` 都从 `preds[1]` head dict 抓 `(point_head, feats[0])`，end2end 嵌 `preds[1]["one2one"]`；导出后端无 feats → 自动禁用。

### E. 配置 + parse_model + 训练管线

- **cfg/default.yaml**（:111-124）：13 个新设置 + `e2e_final_o2m` + `dali`，`cfg/__init__.py` 注册到 `CFG_FLOAT/INT/BOOL_KEYS` → 经 `get_cfg` 自动传播。
- **parse_model**（tasks.py:1938）：`if m is Segment26 and len(args)>3: point_hidden=args[3]; args=args[:3]`，最后 `args.append(make_divisible(min(point_hidden,max_channels)*width,8))`。旧 YAML 无第 4 参 = Lite，新旧共存。
- **yolo26-seg-pointrend.yaml**：head 末行 `[[16,19,22], 1, Segment26, [nc, 32, 256, 128]]`，`end2end: True`。
- **trainer.py**：(1) `setup_model` 加 distill-checkpoint unwrap（finetune-from-recipe200 自动剥 teacher）；(2) resume args 白名单加全部 `seg_point*` / `e2e_final_o2m`。
- **optim/muon.py**：MuSGD 对 `PointHeadMLP` 的 **Conv1d 3D 权重**修正（`ndim>2` flatten 正交化，按 2D dims 缩放，避免无意义的 in/k 尺度）。
- **detect/train.py** + **dali_seg.py**：`dali=True` 且 task=segment 时走 GPU JPEG decode/resize（实验性）。
- **torch_utils.py**：CUDA compile 时 Triton 缺失的降级 warning。

## 4. DDP 混合解法（关键设计）

MLP 真跑在 criterion（逐实例、真梯度）+ `Segment26.forward` 跑 `zero_loss()` dummy 连参数进 forward 图 → DDP `find_unused_parameters=False` / `static_graph=True`（compile）安全。同一 dummy 对象被 o2m/o2o 各加一次（值恒 0 不污染）。**ft60 是 3-GPU DDP 实跑、正常收敛无 unused 报错 → 本身即 3-GPU 实证**。2-GPU allreduce 同步另有 `scripts/smoke_point_head_ddp.py`（CPU/gloo 或 GPU/nccl 均可）。

## 5. 验收分叉（`seg_point_refine_infer`）

单一开关控制 train-val / standalone-val / predict（经 `args=copy(self.args)`）：

- **ft60 训练内 val = False** → 走 `process_mask`（确定性），best.pt fitness **不受细分随机性污染**（`process_mask_pointrend` 推理采点用 `torch.rand` → 随机，不能入 fitness）。
- **MLP 直接收益**要 ft60 跑完后对 best.pt 做 "val twice"：`False`（间接基线）vs `True`（直接，固定种子/3 次均值），比 Mask mAP50-95 / AP75 / AP95，Δ = MLP 直接收益。validator pointrend 路径已接线 + 单测绿（`test_segmentation_validator_pointrend_postprocess`，52 passed）。脚本：`scripts/val_twice_pointrend.py`。

## 6. 验收口径提醒

- 本轮训练侧 (T) 已全量落地；推理收益只在 `seg_point_refine_infer=True` 时体现。
- 看 **Mask AP75/AP95** 是否涨（不是 AP50）；不要期待可视化边界立刻锐化 —— 那要等 (I) subdivision 上线并训练充分。
- GT 仍是 poly 栅格（mask_ratio=4→160，approxPolyDP 简化）；point head 突破不了 160×160 监督上限、受 aliasing / 小目标失真 / v2 holes 约束。
- 消融必须固定 `seg_point_boundary`（其混合候选池改变了 point 监督语义，不可与 ROI/uniform 自由组合）。
- 从 recipe200 旧 ckpt 加 point head 须 **finetune-from-checkpoint**（`strict=False` + zero init），**不 resume**。
