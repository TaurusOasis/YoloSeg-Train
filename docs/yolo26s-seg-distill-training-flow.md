# YOLO26s-seg 蒸馏训练：代码结构梳理 · 框架问题分析 · 改进方向

> 用途：完整记录 yolo26s-seg 三阶段实例分割蒸馏训练链路的代码结构、执行流程，深入分析当前训练框架的问题并给出改进建议，以及数据集扩展 / 训练 trick 的切入角度。
>
> - 仓库：ultralytics（工作树，基于 `8.4.82`，上游 commit `10d17c168`）
> - 梳理日期：2026-07-03（07-01 首版；07-02 补代码结构详梳与问题深析；07-03 阶段 C 完成，补 teacher 同标尺基线与最终评估、改进计划细化）
> - 配套文档：`docs/yolo26-seg-training-review.md`（loss/链路审查，编号 1.1–1.9 沿用）、`docs/coconut-yolo26s-seg-distill.md`（COCONut 数据+命令）

---

## 1. 总览：三阶段训练链路

三条 yolo26s-seg 实例分割训练线均落在 `runs/segment/`，关系为 **B → C 继承 + A 独立基线**：

```
A. yolo26s-seg 在完整 LVIS 上普通训练             (1203 类, optimizer=auto)        独立基线
B. yolo26s-seg 在 LVIS·COCO80 子集上蒸馏 yolo26x-seg (78 类, MuSGD, 从 yolo26s-seg.yaml 起) ─┐
                                                                                          │ best.pt 作为初始化
C. yolo26s-seg 在 COCONut-B 上蒸馏 yolo26x-seg      (80 类, MuSGD) ◀───────────────────────┘
```

| 阶段 | run 目录 | 数据 YAML | 类别数 | student 起始 | teacher | 优化器 | batch/device | dis / dis_proto |
|---|---|---|---|---|---|---|---|---|
| **A 普通LVIS** | `yolo26s-seg-lvis-b48-bf16-swanlab` | `LVIS_yolo_seg/lvis-seg.yaml` | **1203** | yolo26s-seg | — | `auto` | 48 / 0,1,2 | — |
| **B LVIS蒸馏** | `yolo26s-seg-lvis-coco80-distill-x-teacher-b80-2gpu` | `LVIS_coco80_yolo_seg/lvis-coco80-seg.yaml` | **78** | `yolo26s-seg.yaml`(从零) | `yolo26x-seg.pt` | `MuSGD` | 150 / 0,1,2（run 名与实际不符，见 §4.6） | dis=3.0 / 0 |
| **C COCONut蒸馏** | `yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu` | `COCONut_b_yolo_seg/coconut-b-seg.yaml` | **80** | **B 的 best.pt** | `yolo26x-seg.pt` | `MuSGD` | 150 / 0,1,2 | dis=3.0 / **dis_proto=1.0** |

> ⚠️ **类别数澄清**：`yolo26x-seg.pt` 是标准 COCO **80 类** teacher。阶段 B 用的是 LVIS 与 COCO80 的重叠子集，因 `hot dog`、`potted plant` 在 LVIS v1 无对应类别被剔除，实际为 **78 类**。阶段 C 的 COCONut-B 才是完整 **80 类**。teacher 与 student 类别数不一致由 `teacher_class_indices` 按名字匹配处理（§2.4）。

三阶段共有超参：`epochs=100, patience=100`（关闭早停）、`imgsz=640, close_mosaic=10, warmup_epochs=3, lr0=0.01, lrf=0.01, amp=True(bf16), seed=0`。增强均为默认：`mosaic=1.0, fliplr=0.5, scale=0.5, translate=0.1, hsv 默认`；**`copy_paste=0, mixup=0, cutmix=0, multi_scale=0, cos_lr=False`**（这是后续 trick 的主要空间，见 §7）。

---

## 2. 代码结构详细梳理

### 2.1 模块地图

| 文件 | 行数 | 性质 | 职责 |
|---|---:|---|---|
| `ultralytics/nn/distill_model.py` | 439 | **新增** | 蒸馏核心：`DistillationModel`（teacher-student 包装）、`FeatureHook`、特征/proto 蒸馏损失、类别对齐、序列化 |
| `ultralytics/engine/trainer.py` | 1264 | **修改** | 通用训练循环；内嵌 7 处蒸馏特判（§2.5）、BF16 AMP 分支、OOM 自动降 batch |
| `ultralytics/utils/loss.py` | 1394 | **修改** | `v8SegmentationLoss` / `E2ELoss`；P0 零面积 NaN clamp（:616）、E2E 日志加权修复（:1196） |
| `ultralytics/utils/torch_utils.py` | ~1050 | **修改** | `ModelEMA` 剥离 teacher（:688-691）、checkpoint 只存 student（:764-769）、`autocast_dtype`（BF16） |
| `ultralytics/utils/callbacks/swanlab.py` | 183 | **新增** | SwanLab 本地看板回调，环境变量驱动，NaN→None 防序列化崩溃 |
| `ultralytics/utils/callbacks/base.py` | — | **修改** | 注册 swanlab 回调 |
| `ultralytics/cfg/default.yaml` | — | **修改** | 新增 `dis_proto / distill_warmup_epochs / distill_loss_clip` 三项 |
| `scripts/train_yolo26s_seg_coconut_distill.py` | 382 | **新增** | 阶段 C 入口：数据自动构建、SwanLab 配置/tmux 看板、resume save_dir 推断、CLI 透传 |
| `scripts/train_yolo26s_seg_lvis_coco80_distill.py` | 119 | **新增** | 阶段 B 入口（功能子集，无 SwanLab/resume 逻辑） |
| `scripts/build_coconut_yolo_seg.py` | 384 | **新增** | COCONut panoptic RGB mask → YOLO seg 多边形标签（多进程） |
| `scripts/build_lvis_coco80_seg_subset.py` | — | **新增** | LVIS → COCO80 交集子集（78 类重映射，images 软链） |
| `scripts/batch_lvis_eval_checkpoints.py` 等 4 个 | — | **新增** | LVIS 官方 API 逐 checkpoint 评估、预测导出/重映射、标签诊断 |
| `tests/test_engine.py::test_distill_resume` | — | **修改** | 蒸馏 resume 唯一测试（detect 任务，非 seg） |

### 2.2 蒸馏训练执行时序（脚本 → loss）

```
scripts/train_yolo26s_seg_coconut_distill.py
 ├─ parse_args()                    # CLI + --key=value 透传 Ultralytics overrides
 ├─ configure_swanlab()             # 写 ULTRALYTICS_SWANLAB* 环境变量（DDP 子进程继承）
 ├─ prepare_dataset()               # 缺 YAML 时调 build_coconut_yolo_seg.py 子进程
 └─ YOLO(model_source).train(**train_args)
     └─ SegmentationTrainer(BaseTrainer)._do_train()
         ├─ _setup_train()
         │   ├─ setup_model()                          # 加载 student（或 resume 的 DistillationModel）
         │   ├─ distill 时禁用 compile                  # trainer.py:304-306
         │   ├─ freeze "teacher_model." 前缀参数         # trainer.py:319-320
         │   ├─ AMP: bf16 可用则跳过 FP16 check、关 GradScaler  # trainer.py:341-356
         │   ├─ 包装 DistillationModel(student, teacher) # trainer.py:370-371
         │   ├─ DDP(find_unused_parameters=True)        # trainer.py:375-380（distill 下 compile=False）
        │   ├─ build_optimizer()                       # MuSGD 4 param groups + head/semseg lr×3；跳过 requires_grad=False（teacher 不进优化器，07-03 修复 F4）
        │   ├─ loss_names += ("dis_feat","dis_proto")  # trainer.py:388-389（07-03 拆列，修复 F14）
        │   └─ ModelEMA(model)                         # EMA 内 teacher 置 None
         └─ 每 batch：
             ├─ set_distill_warmup_factor(ni/warmup_iters)   # trainer.py:472-476
             ├─ loss, loss_items = model(batch)               # → DistillationModel.loss()
             │   ├─ _forward_teacher_for_distillation(img)    # no_grad + head.training hack，hook 抓 teacher 特征
             │   ├─ student(img)                              # hook 抓 student 特征
             │   ├─ student.loss(batch, preds)                # E2ELoss(box/seg/cls/dfl/sem)
             │   ├─ Σ loss_sl2(projector(s_feat), t_feat)     # 3 层 neck，teacher score 空间加权 L2，× dis
             │   ├─ dis_proto>0: loss_proto(s_proto, t_proto) # 归一化 MSE + P3 前景加权，× dis_proto
             │   └─ sanitize(× warmup_factor) × batch_size    # nan_to_num + clamp[0, clip]
             └─ backward → optimizer_step → ema.update
```

### 2.3 `DistillationModel` 内部结构（`nn/distill_model.py`）

- **构造**（:63-114）：加载 teacher 并冻结 → `get_distill_layers` 取 Detect 头输入层 + 头本身（YOLO26 → `[16,19,22,23]`）→ 对 teacher/student 同层注册 `FeatureHook`（共享 dict 存输出）→ dummy forward 探测通道维 → 为每对 neck 特征建 `projector`（Conv1×1-ReLU-Conv1×1，student→teacher 通道）。
- **类别对齐**（:144-160）：`_resolve_teacher_class_indices` 将双方 `names` 小写化后按名字取交集；teacher 全覆盖时返回 `None`（C：80=80 走全量），部分覆盖时 `index_select` 只保留同名通道（B：80→78）。精确字符串匹配；07-03 起未匹配的 student 类别会显式 WARNING 列出（修复 F10 的静默丢类）；`names` 为 property，setter 写入 student 并重算 indices，保证 trainer `set_model_attributes` 在 resume 场景下也能穿透 wrapper 同步（修复 F18）。
- **teacher 前向 hack**（:247-260）：临时置 `teacher_head.training=True`（子模块保持 eval，BN 不更新），让 Detect/Segment 返回训练格式 raw preds/proto、跳过推理后处理——规避此前 teacher 蒸馏时的 CUDA launch timeout。
- **损失**（:268-319）：
  - `loss_sl2`（:321-341）：teacher 头 scores（one2many+one2one 平均、sigmoid、逐类 max）作空间注意力权重的逐元素 MSE，按 `score.sum()×C` 归一。
  - `loss_proto`（:361-390）：one2many 分支 proto 逐样本/通道标准化后 MSE，用 teacher P3 前景分数 `1+score` 加权；尺寸不符时双线性插值 student proto，**通道不符时警告一次并置 0（静默 no-op）**。
  - 鲁棒性：`sanitize_distill_loss`（nan_to_num + clamp[0, distill_loss_clip]）、`distill_warmup_factor` 线性 ramp。
  - 07-03 起特征/proto 蒸馏损失拆为两个分量返回（`dis_feat`/`dis_proto`），进度条与 `results.csv` 分列监控；验证模式返回零占位保持张量形状，但 `label_loss_items(prefix="val")` 会剔除这两列，`results.csv` 不再出现误导性的 `val/dis_*=0`（修复 F14）。
- **序列化**（:162-187）：`__getstate__` 就地清空共享特征 dict（hook 持有同一对象，替换属性无效）、丢弃 hook 句柄；`__setstate__` 重注册 hook、补默认属性。

### 2.4 checkpoint / EMA / resume 路径

- `ModelEMA.__init__`（`torch_utils.py:688-691`）：deepcopy 后把 `teacher_model` 置 None——EMA 只平滑 student + projector。
- 保存 checkpoint（`trainer.py:684-703`）：存 EMA（无 teacher）；最终 `strip_optimizer`（`torch_utils.py:764-769`）进一步只留 `student_model`，所以 `best.pt` 是纯 student，可直接推理/作下一阶段初始化。
- resume（`trainer.py:814-830`）：checkpoint 里是无 teacher 的 `DistillationModel` → 重建 student（07-03 起回填 checkpoint names，修复 F18）→ 从 `args.distill_model` 路径重建 teacher → 恢复 projector 权重。
- resume 参数白名单（`trainer.py:944-952`）：仅 `device/batch/epochs/.../distill_model/dis_proto/distill_warmup_epochs/distill_loss_clip` 允许覆盖，其余一律沿用 checkpoint 的 `train_args`。

### 2.5 trainer 中的蒸馏侵入点清单（耦合度评估用）

| # | 位置 | 内容 |
|---|---|---|
| 1 | `trainer.py:304-306` | distill 时强制 `compile=False` |
| 2 | `trainer.py:319-320` | freeze 列表追加 `teacher_model.` 前缀 |
| 3 | `trainer.py:370-371` | 包装 `DistillationModel` |
| 4 | `trainer.py:388-389` | `loss_names += ("dis_feat", "dis_proto")` |
| 5 | `trainer.py:472-476` | 每 batch `hasattr` 探测并注入 warmup factor |
| 6 | `trainer.py:782-793` | resume 重建 DistillationModel/teacher/projector |
| 7 | `trainer.py:944-952` | resume 白名单扩充蒸馏参数 |
| 8 | `trainer.py:1014-1018` | resume 加载 EMA 时 student/projector 分开 load |

（另有 `torch_utils.py` 2 处、`default.yaml` 3 项、`test_engine.py` 1 处。）

### 2.6 数据准备链路

```
原始 LVIS_yolo_seg ──build_lvis_coco80_seg_subset──▶ LVIS_coco80_yolo_seg (78类) ── 阶段B ──▶ best.pt ─┐
                                                                                                     │ student init
coconut(panoptic) + coco2017 ──build_coconut_yolo_seg──▶ COCONut_b_yolo_seg (80类) ── 阶段C ──────────┘──▶ 最终模型
```

`build_coconut_yolo_seg.py` 核心转换：panoptic RGB mask 按 `R+256G+256²B` 解码 segment id → 只保留 `isthing=1 且 iscrowd=0` → 每个 segment 二值化后提取轮廓 → `approxPolyDP(ε=0.001×周长)` 简化 → 写 YOLO 标签。COCONut-B 训练集 241602 图（train2017+unlabeled2017），val 为 relabeled_coco_val 5000 张。
**07-03 修复（F11）**：改用 `RETR_CCOMP` 保留孔洞边界，同一实例的全部轮廓（断裂碎片 + 孔洞）经 `merge_multi_segment` 细桥合并为**每实例一行标签**——原 `RETR_EXTERNAL` + 每轮廓一行的做法会把孔洞填实、把遮挡断裂实例拆成多个实例（val 集抽样 2554 实例中 398 个受影响，15.6%）。修复后带孔实例还原 IoU 从 0.826 提升到 0.949；细桥在多碎片实例上引入轻微填充伪影（整体 mean IoU 0.938 与旧法持平），实例数语义正确性收益远大于此。**标签已重建（07-03）**：`Dataset/COCONut_b_yolo_seg_v2/`，train 标签行 2365072→1797818（−24%），val 57220→45003，thing_segments 数不变（见 §5 P2-1）。

---

## 3. 各阶段实际结果摘要（2026-07-03 终版）

> 详细逐阶段曲线分析见 §8 附录。关键前提：**三阶段验证集不同（LVIS val / LVIS-coco80 val / COCONut relabeled val），跨阶段 mAP 不构成同一标尺**。

| 阶段 | 验证集 | mAP50-95(B) | mAP50-95(M) | 状态 | 收敛 |
|---|---|---:|---:|---|---|
| A | LVIS val（LVIS API） | box_AP_all=0.198 @e85 | mask_AP_all=0.167 | 完成 | 否（仍升） |
| B | LVIS-coco80 val（78 类） | 0.343 @e100 | 0.315 | 完成 | 否（仍升） |
| C | COCONut relabeled val（80 类） | **0.402**（best.pt=e98 终评） | **0.341** | **完成 100/100（07-02 08:25）** | 缓升未平台（末 10 epoch +0.0025/0.0034） |

### 3.1 ⚠️ 核心发现：在 COCONut val（v1 标签）标尺上，student 已追平 teacher（2026-07-03 补测；**该结论已被 §3.1.2 v2 复评推翻，保留供追溯**）

teacher 基线此前一直缺失（原 F13）。07-03 用 `yolo26x-seg.pt` 在同一 COCONut relabeled val、同一验证配置下补测（`runs/segment/yolo26x-seg-teacher-coconut-val/`）：

| 模型 | 参数量 | Box P / R / mAP50 / mAP50-95 | Mask P / R / mAP50 / mAP50-95 |
|---|---|---|---|
| **teacher** yolo26x-seg（官方 COCO 训练） | ~62M | 0.677 / 0.508 / 0.546 / **0.400** | 0.681 / 0.486 / 0.522 / **0.327** |
| **student** 阶段 C best.pt（e98） | 11.5M | 0.657 / 0.503 / 0.546 / **0.402** | 0.649 / 0.486 / 0.523 / **0.341** |

**student 以 1/5.4 的参数量在 box 上追平（0.402 vs 0.400）、在 mask 上反超 teacher（0.341 vs 0.327）。** 解读要点：

1. 这不说明 student 绝对能力超过 teacher——teacher 是官方 COCO 标注训练的，在 COCONut 重标注 mask 分布上被系统性低估（mask 0.327 远低于其 COCO val 的 0.470）；student 直接拟合了 COCONut 标注分布，占了域内优势。
2. ~~但它明确说明：**在当前训练分布上，teacher 的软监督已无增量信息可挖**~~——**该推论已被 §3.1.2 推翻**：v1 标签把断裂实例拆成多行 GT，系统性压低 teacher 的 recall/mAP，"追平"是标签噪声伪象。v2 标签复评显示 teacher 仍领先 box +0.081 / mask +0.031。
3. teacher 与 student 在 COCO val 官方标尺上的对比见 §3.1.1（07-03 已补测）：**student 在 COCO 标尺上明显低于官方 yolo26s-seg**，"追平 teacher"是 COCONut 标尺特有现象。

### 3.1.1 COCO val2017 官方标尺对照（P1-1，2026-07-03 补测）

官方 `instances_val2017.json` 经 `convert_coco(use_segments=True)` 转 YOLO seg（5000 图 / 36335 实例，数据集在 `Dataset/coco_val2017_yolo_seg/`），三模型同一验证配置（batch=48，Ultralytics 内部评测，非 pycocotools，绝对值略低于官方发布值但同标尺内可比）：

| 模型 | Box P / R / mAP50 / mAP50-95 | Mask P / R / mAP50 / mAP50-95 | run 目录 |
|---|---|---|---|
| **teacher** yolo26x-seg | 0.758 / 0.661 / 0.728 / **0.558** | 0.747 / 0.639 / 0.694 / **0.448** | `cocoval-teacher-x/` |
| **官方 yolo26s-seg** | 0.703 / 0.582 / 0.637 / **0.468** | 0.690 / 0.559 / 0.607 / **0.386** | `cocoval-official-s/` |
| **student** 阶段 C best.pt | 0.687 / 0.530 / 0.581 / **0.415** | 0.693 / 0.523 / 0.567 / **0.356** | `cocoval-student-c/` |

（B best.pt 为 78 类模型，类别索引与 COCO80 标签错位，需先做索引重映射才能进此表，暂缺。）

**双标尺合并结论（F13 完全闭合）**：

| 标尺 | student C vs teacher | student C vs 官方 s |
|---|---|---|
| COCONut relabeled val（v1 标签） | box **+0.002** / mask **+0.014**（追平/反超，**后证为伪象，见 §3.1.2**） | — |
| COCONut relabeled val（v2 标签） | box **−0.081** / mask **−0.031**（落后） | — |
| COCO val2017 官方 | box **−0.143** / mask **−0.092**（大幅落后） | box **−0.053** / mask **−0.030**（落后） |

1. **student C 确实过拟合了 COCONut 标注风格**：在 COCO 官方标尺上比现成的官方 yolo26s-seg 还低 5.3 box / 3.0 mask 点。若下游评测/部署以 COCO 风格为准，当前 C 模型不是最优选择。
2. recall 是主要损失来源（0.530 vs 官方 s 的 0.582）：COCONut 重标注删改了部分 COCO 框（含 crowd 处理差异、F12 的 iscrowd 丢弃），student 学到了"更保守"的检出行为。
3. **改进不能只盯 COCONut val 单标尺，P2-2 起所有实验双标尺验收（COCONut val + COCO val2017），COCO 标尺不得低于官方 yolo26s-seg 成为硬约束之一**（若两标尺冲突，按下游用途取舍并明确记录）。

### 3.1.2 ⚠️ v2 标签标尺复评：推翻"student 追平 teacher"（2026-07-03）

F11 修复后的 v2 标签（每实例一行、孔洞保留，`COCONut_b_yolo_seg_v2/`）重转了同一 COCONut relabeled val（实例数 57220→45003，−21% 为拆分实例合并）。在 v2 val 上复评：

| 模型 | Box mAP50 / mAP50-95 | Mask mAP50 / mAP50-95 | run 目录 |
|---|---|---|---|
| **teacher** yolo26x-seg | 0.681 / **0.513** | 0.640 / **0.404** | `v2val-teacher-x/` |
| **student** C best.pt | 0.582 / **0.432** | 0.575 / **0.373** | `v2val-student-c/` |

**teacher 大幅领先（box +0.081 / mask +0.031）——§3.1 的"追平/反超"是 v1 标签噪声的伪象**：v1 把遮挡断裂实例拆成多行 GT，teacher 对这类实例只出一个完整检测，被迫与多个碎片 GT 匹配而 recall 受罚；student 拟合了同样拆分风格的训练标签，在 v1 标尺占了"分布内"便宜。v2 消除该噪声后真实差距显现，且与 COCO 官方标尺（§3.1.1 gap −0.143）方向一致。

**对计划的修正**：teacher 软监督仍有充足增量信息，蒸馏主轴恢复有效——P2-2 的 `dis` **不再减半**（维持 3.0），P3-3 response KD 价值进一步上调；三标尺中以 v2 val + COCO val2017 为准，v1 val 结果仅作历史对照。

### 3.2 训练过程结论

- 三阶段全程数值稳定：`dis_loss` 平稳下降（B 2.42→0.58；C warmup 后 1.17→0.79），无 NaN/OOM/CUDA timeout，鲁棒性三件套（warmup / sanitize / teacher 前向 hack）实测有效。
- C 起跳高（epoch1 box 0.306）来自 B 的 best.pt 初始化，符合"LVIS 预训练 + COCONut 蒸馏"设计；全程增益 +0.096 box / +0.070 mask。
- 收敛速度：mAP50-95(B) 每 10 epoch 增量从中期 +0.008~0.009 衰减到末期 +0.0025，**放缓但未到平台**；lr 末期 0.0006 已近终点，直接续训收益有限，拉长总 epoch 重训或 cos_lr 更合理。
- close_mosaic（e91 起）效应明确：`sem_loss` 0.150→0.070 腰斩、`cls_loss` 1.44→1.39，最后 10 epoch 贡献了约 +0.003 box。
- best.pt = e98（fitness=box+mask 加权和），非末轮，`save_period=5` 的 checkpoint 序列完整可回溯。
- 逐类短板（best.pt 终评，box mAP50-95）：`hair drier` 0.054、`tie` 0.073、`backpack` 0.147、`potted plant` 0.155、`handbag` 0.162、`toothbrush` 0.167、`spoon` 0.171、`bench` 0.191——集中在**小 / 细长 / 密集遮挡**物体；person（19576 实例）box 0.431 / mask 0.351 属正常水平。mask-box gap 整体 -0.061（teacher -0.073），student 的 mask 相对质量略优，proto 蒸馏 + COCONut mask 标注质量可能均有贡献。
- 官方参考（COCO val 标尺，不可直接对比）：yolo26s-seg = 47.3 box / 40.0 mask；yolo26x-seg = 56.5 / 47.0。

---

## 4. 训练框架问题深入分析

按层次组织：架构耦合 → 蒸馏算法 → 数据标签 → 监控 → 工程 → 实验管理。编号 F1–F17（框架层），沿用 review 文档的 1.x 编号处不重复展开。

### 4.1 架构与模块化

**F1（中）蒸馏逻辑横切 trainer，未做功能隔离。**
§2.5 列出的 8 处侵入点散布在 `trainer.py`（1264 行）的模型构建、freeze、DDP、日志、训练循环、resume 各环节，靠 `isinstance` / `hasattr` / `args.distill_model is not None` 特判。后果：① 每次上游同步（rebase 8.4.x）都要人工合并这 8 处；② 蒸馏行为无法单独测试（唯一测试 `test_distill_resume` 走的是完整 trainer）；③ 新增蒸馏形式（logits KD、mask KD）还要继续加特判。
改进方向：把蒸馏收敛为独立组件——warmup 注入改为 `on_train_batch_start` 回调（框架已有完整 callback 体系，SwanLab 就是这么接的）；freeze/EMA/resume 的 teacher 处理收进 `DistillationModel` 自身的接口（如 `get_trainable_modules()` / `state_for_checkpoint()`），trainer 只面向该接口。这与"接口与实现分层、独立功能暴露接口"的工程约定一致。

**F2（中）warmup factor 每 batch 由 trainer 计算注入（`trainer.py:472-476`）。**
`DistillationModel` 明明持有 `student_model.args`（`distill_warmup_epochs` 就在其中），却要 trainer 每步 `hasattr` 探测再 set。职责放错位置，且 `nb`（每 epoch 步数）变化时（OOM 降 batch 重建 dataloader）warmup_iters 会跳变。改法：把 `ni/nb` 通过一次性接口传入，或模型内部自增 step 计数。

**F3（低）teacher 前向 hack 脆弱。**
`_forward_teacher_for_distillation` 靠临时翻转 `head.training` 标志换取训练格式输出，依赖 Detect/Segment `forward` 内部对 `self.training` 的分支实现；上游重构头部（如引入 `export`/`fuse` 分支变化）会静默改变行为。已用注释锚定原因（CUDA launch timeout），但更稳的做法是给 Detect/Segment 增加显式的 `forward_raw()`/`return_raw=True` 接口，蒸馏调用方不再碰 training 标志。

**F4（低-中）`build_optimizer` 未过滤 `requires_grad=False` 参数（`trainer.py:1087-1099`）。**
参数收集遍历整个 `DistillationModel.named_modules()`，冻结的 teacher 参数（yolo26x-seg，~60M）全部进入 MuSGD param groups。torch 优化器对 `grad=None` 的参数跳过 step，所以不影响正确性，但：① 日志里的参数组统计虚高、误导；② optimizer 遍历开销白付；③ 若未来换成对 `requires_grad=False` 不容忍的优化器实现会直接踩雷。一行过滤 `if not param.requires_grad: continue` 即可。

**F18（中→✅ 已修 07-03）resume 时 student names 丢失 + teacher 类别对齐静默失效。**
问题链路：resume 路径 `setup_model()`（`trainer.py:814-822`）用 `get_model(cfg, weights=ckpt.student_model)` 重建 student——只 load 权重，names 落回 `tasks.py` 的数字默认 `{0:'0',1:'1',...}`；`DistillationModel` 随即在构造时用数字 names 计算 `teacher_class_indices`（与 teacher 真名交集为空 → 静默回退"全量 teacher 类"）；之后 `set_model_attributes()` 把数据集真名写到 `self.model.names` 时 **wrapper 已存在，只写到外层**，student 不同步、indices 不重算。后果：① 经历过 resume 的 run，checkpoint（strip 后的纯 student）类名全是数字——**B/C 的 best.pt 实测均如此**（评测显示类名 "79" 的根因）；② 阶段 B（78↔80）resume 后的训练段蒸馏 score 加权通道错位（C 是 80↔80 全匹配，回退全量恰好无实害）。
修复（07-03）：`DistillationModel` 增加 `names` property——getter 直读 `student_model.names`，setter 写入 student 并重算 `teacher_class_indices`，trainer 的 `set_model_attributes` 无需改动即自动同步；resume 重建 student 时先回填 checkpoint 里的 names（`trainer.py:818-824`），构造期对齐即正确。历史 checkpoint（B/C 的 best/last）已按各自数据 YAML 就地补回真名。冒烟：resume 后 strip 出的 student names 为真名、无误匹配 WARNING；`test_distill_resume` 通过。

**F5（低）DDP 全程 `find_unused_parameters=True`（`trainer.py:375-380`）。**
蒸馏下 compile 被禁 → `find_unused_parameters=not compile = True`。该选项每次 backward 多一遍图遍历，3 GPU × 241k 图 × 100 epoch 的训练里是持续开销。teacher 已冻结（不在 DDP 梯度桶里）、student+projector 每步全参与，理论上可以 `False`；若历史上因 E2E 双分支存在 unused 参数，应查明后针对性处理而非全局兜底。

### 4.2 蒸馏算法层

**F6（中）蒸馏信号单一：只有中间特征 L2 + proto MSE，无 response 级蒸馏。**
当前只对 3 层 neck 特征（projector 后 score 加权 L2）和 proto map（仅 C 启用）做拟合；teacher 的 **分类 logits、box 回归分布、mask 系数** 均未蒸馏。检测蒸馏文献里 response KD（温度化 KL）与特征 KD 通常互补，且 response KD 对小 student 的 cls 分支收益明显。head 层特征（`feats_idx[-1]`）已经被 hook 抓到了，只用来算 score 权重，**顺手就能加 logits KD**（注意用 `teacher_class_indices` 对齐通道）。

**F7（中）proto 蒸馏只对 one2many 分支、与 GT 无关联。**
`loss_proto` 拟合的是 teacher 的 proto 基底（经归一化），但 mask = coeff × proto，**基底相似不代表最终 mask 相似**（student 的 coeff 空间可以任意旋转）。更直接的目标是蒸馏 teacher 的最终 mask 预测（对 assigned 正样本，teacher soft mask 作为额外监督），或至少同时约束 coeff。当前设计作为正则是安全的（C 阶段实测无害），但预期收益有限，值得消融验证（B 无 proto、C 有，两者数据不同无法直接归因）。

**F8（低-中）`dis` 权重全程恒定，无衰减调度。**
`distill_warmup_epochs` 只解决冷启动，训练后期 teacher 的软监督会与 GT 硬监督竞争（尤其 teacher 在 COCONut 重标注 mask 上并非最优——teacher 是官方 COCO 训练的）。常见做法是蒸馏权重随训练余弦衰减到 0.3–0.5×，让 student 末期主要拟合 GT。当前框架加这个调度很容易（warmup factor 机制直接复用）。

**F9（低）teacher 在强增强图上前向，蒸馏信号带噪。**
teacher 与 student 吃同一张 mosaic/仿射增强图。teacher 自身训练也用过 mosaic，所以不算错误，但 mosaic 拼缝、大尺度抖动区域 teacher 分数不可靠，score 加权会把噪声区域降权（设计上已缓解）。若追求更干净的信号，可选：close_mosaic 后再启用 proto 蒸馏、或 teacher 用 clean 图前向（代价是几何对齐复杂化，需权衡）。

**F10（低）类别对齐是精确字符串匹配（`_resolve_teacher_class_indices`）。**
`hot dog` vs `hotdog`、`airplane` vs `aeroplane` 这类差异会静默丢类，仅日志一行提示数量。本链路 78/80 已人工核对过没问题，但作为通用机制应支持显式 class-map 配置，匹配数量异常时（如 <90%）应 warning 甚至 fail-fast。

### 4.3 数据与标签质量

**F11（中）panoptic → 多边形转换的两类系统性损耗（`build_coconut_yolo_seg.py:149-166`）。**
① `RETR_EXTERNAL` 只取外轮廓——**带孔洞的 mask（甜甜圈、杯柄）孔洞被填实**，mask 面积系统性偏大；② 一个被遮挡断裂成多个连通域的实例，**每条轮廓单独写一行标签 = 一个实例被拆成多个实例**，实例数虚增、与 val（同样方式转换）形成一致性偏差但与 COCO 官方标注形成分布差异。改法：孔洞用 `RETR_CCOMP` + 内外轮廓桥接（Ultralytics 官方 COCO 转换的 merge multi-segment 做法）；断裂部件同实例的多条轮廓可用细连接线合并为单条多边形。
③ 次要：`min_area=4` 像素的碎片会产生大量近退化多边形（P0 clamp 已防 NaN，但仍是噪声正样本）；`approx_epsilon=0.001` 对小目标可能过度简化。

**F12（低）`iscrowd` 区域直接丢弃。**
COCO 官方评测把 crowd 区域作为 ignore 处理；当前转换直接丢标签且训练时无 ignore 机制，crowd 区域的预测会被当负样本压制。对 person/car 等密集类有轻微伤害。

**F13（流程，中→✅ 已闭合 07-03）验证标尺不统一，蒸馏收益无法量化。**
三阶段各用各的 val；teacher 在 LVIS-coco80 val 上的基线目录（`yolo26x-seg-coco80-on-lvis-val/`）为空。当前没有任何一张表能回答"蒸馏比不蒸馏好多少""student 距 teacher 还差多少"。必须在统一标尺（建议标准 COCO val2017 + COCONut relabeled val 双尺）上补齐：teacher / B / C / 官方 yolo26s-seg / 无蒸馏对照。
**07-03 更新**：双标尺均已补齐——COCONut val（§3.1：student 追平/反超 teacher）+ COCO val2017（§3.1.1：student 低于官方 yolo26s-seg，过拟合 COCONut 标注风格）。仅剩无蒸馏对照组（P3-4）。

### 4.4 监控与可观测性

**F14（低）`dis_loss` 是单一混合标量。**
特征蒸馏（3 层求和）与 proto 蒸馏折叠成一列，无法分辨谁在下降、谁停滞；`val/dis_loss` 恒 0 又占一列（`results.csv` 可见 `0,0`），易误读。改法：`loss_names` 拆 `dis_feat/dis_proto` 两列，val 阶段跳过或标 N/A。
**F15（低）resume 每次新建 SwanLab run**，曲线分片（C 已有 3 段），对比曲线要手工拼接。SwanLab 支持 resume 到既有 run（`id`/`resume` 参数），回调里从 checkpoint 目录持久化 run id 即可续接。

### 4.5 工程与可维护性

**F16（中）测试覆盖不足。**
蒸馏仅 `test_distill_resume`（detect 任务）。未覆盖：seg 任务 proto 蒸馏、teacher/student 类别失配（78↔80 路径）、warmup factor 调度、sanitize/clip 行为、`__getstate__/__setstate__`（DDP pickle 路径）。这些恰是本链路的核心改动，回归全靠跑真实训练发现，代价极高。建议按 review 文档 §7 的模式补最小单测（coco8-seg + yolo26n-seg 蒸馏 yolo26s-seg 冒烟 2 epoch 可全覆盖）。

### 4.5b 07-03 下午追加：resume 事故链暴露的三个问题（F19–F21，均已修）

> 背景：recipe200 训练在 epoch 12 被 SIGINT 中断后，第一次 resume 在 epoch 12 起步即 CUDA OOM 整组崩溃。复盘暴露出一条完整的问题链。

**F19（中→✅ 已修 07-03）resume 时脚本 argparse 默认值静默覆盖 checkpoint 超参。**
问题链路：训练脚本 `build_train_args` 的 resume 分支把 `--batch/--patience/--workers` 等**默认值**一律塞进 overrides；trainer `check_resume` 的白名单机制只判断 `k in overrides`，无法区分"用户显式指定"与"脚本默认"。实测后果：原始 run `--patience 200` 启动，resume 未传 patience → 被脚本默认 100 静默覆盖（args.yaml 可查）；batch 同理维持了 96→150 类误覆盖的可能（F17 的 B 阶段命名漂移即同源）。
修复：① 脚本侧 `cli_provided()` 扫描 `sys.argv`，resume 分支**只传显式输入的参数**，并打印继承/覆盖清单；② trainer `check_resume` 打印实际生效的 overrides，对白名单外且值不同的 overrides 显式 WARNING（原先静默忽略，`data` 就是典型：resume 传 `--data` 实际不生效）；③ 白名单补上 `dis`（原先只有 `dis_proto`，蒸馏权重中途不可调）。

**F20（中，缓解）DDP 训练中途 OOM 无兜底，单点 OOM 整组崩溃。**
`trainer.py` 的 OOM 自动降 batch 仅在**首 epoch + 单 GPU**生效（`epoch > start_epoch or RANK != -1` 直接 raise）；多卡训练任何 rank 中途 OOM → 全组退出，只能人工 resume。本次事故正是 rank1 在 `loss_sl2` 处分配 676MB 失败。彻底方案（skip-batch 恢复：catch OOM → 全 rank all_reduce 标志 → 同步跳过该 batch + empty_cache）改动训练主循环风险较高，暂不做；当前缓解：F21 降低峰值 + 脚本默认注入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（multi_scale 每 batch 变尺寸导致的分配器碎片是 OOM 的放大器）+ batch 留足余量（multi_scale=0.25 时峰值显存 ≈ 基准 × 1.56）。

**F21（中→✅ 已修 07-03）`loss_sl2` 一次性物化两个 (N,C,HW) 大张量，multi_scale 峰值下 OOM。**
原实现 `F.mse_loss(reduction="none")` 物化完整 mse 张量，再 `mse * score` 又一个同尺寸临时张量，二者同时存活；800px 峰值时单层 neck 特征的临时分配即数百 MB × 3 层。修复：按 batch 维 chunk(4) 分块累加 `((s-t)²·w).sum()`，峰值临时内存约降一半（saved-for-backward 的 diff 不变，但双全尺寸临时消除）；数值与梯度已验证逐位等价（absdiff=0）。

### 4.6 实验管理

**F17（低但易踩坑）run 命名与实际配置漂移 + 中途改超参。**
阶段 B run 名为 `...b80-2gpu`，`args.yaml` 实际 `batch=150, device=0,1,2`——resume 白名单允许改 batch/device，中途从 b80/2GPU 切到了 b150/3GPU。这改变了 `nbs/accumulate` 与 weight_decay 缩放（`trainer.py:283-284`），等效学习率动态发生变化，results.csv 曲线前后不可严格比较，且目录名误导。约定：resume 改资源参数时在 run 目录写 `NOTES.md` 记录切换点；新实验命名不要把 batch/GPU 写进名字（易过期），改进 args.yaml/SwanLab config 里查。
另有废弃 run `yolo26s-seg-coconut-b-distill-x-teacher-b150-3gpu`（无 lvispretrain，空权重，疑似 CUDA timeout 修复前的版本）应清理或归档，避免混淆。

### 4.7 问题优先级汇总

| 编号 | 问题 | 严重度 | 改动量 | 建议时机 |
|---|---|---|---|---|
| F13 | 统一验证标尺 + teacher 基线缺失 | **高**（影响所有结论） | 运行侧 | **✅ 双标尺已闭合（07-03，§3.1 + §3.1.1）** |
| F6 | 无 response 级蒸馏 | 中 | distill_model.py | 短期消融（P1-1 后优先级上调，见 P3-3） |
| F1/F2 | 蒸馏与 trainer 解耦 | 中 | 重构 | 中期（rebase 前做） |
| F16 | 蒸馏单测缺失 | 中 | tests | 短期 |
| F4 | optimizer 收纳冻结 teacher 参数 | 低-中 | 1 行 | **✅ 已修（07-03）** |
| F7/F8 | proto 蒸馏目标 / dis 衰减调度 | 低-中 | 小 | 消融驱动 |
| F14/F15 | dis_loss 拆列 / SwanLab 续接 | 低 | 小 | **✅ 已修（07-03）** |
| F5/F3/F12/F17 | DDP 开销/前向 hack/iscrowd/命名 | 低 | 小 | 按需 |
| F10 | 类别匹配静默丢类 | 低 | 小 | **✅ 已修（07-03，未匹配类显式 WARNING）** |
| F18 | resume 丢 student names / 类别对齐失效 | 中 | 小 | **✅ 已修（07-03，names property 同步 + resume 回填）** |
| F19 | resume 脚本默认值静默覆盖超参 | 中 | 小 | **✅ 已修（07-03，只传显式参数 + 双侧日志 + 白名单补 dis）** |
| F20 | DDP 中途 OOM 无兜底 | 中 | 主循环 | 缓解（expandable_segments + F21 降峰值）；skip-batch 恢复待 P4-3 一并做 |
| F21 | loss_sl2 显存峰值（双全尺寸临时张量） | 中 | 小 | **✅ 已修（07-03，分块累加，数值逐位等价）** |
| F11 | 转换标签孔洞/断裂损耗 | 中 | 脚本 | **✅ 代码已修 + 标签已重建到 v2（07-03）** |

**07-03 修复明细**（均已通过冒烟验证：coco8-seg 蒸馏 2 epoch + detect 蒸馏 resume 测试 + 转换脚本 2554 实例 IoU 对比）：
- F4：`build_optimizer` 跳过 `requires_grad=False` 参数——优化器张量数从 1088（含 teacher）降到 454（=可训练参数数）。
- F14：`DistillationModel.loss` 返回 `dis_feat`/`dis_proto` 两个分量，进度条与 `results.csv` 分列；`label_loss_items(prefix="val")` 剔除蒸馏列，不再输出误导性的 `val/dis_*=0`。
- F15：SwanLab 回调把 run id 持久化到 `<log_dir>/.swanlab_run_id`，resume 时以 `id + resume="allow"` 续接（local 模式仍生成新 run 目录但共享 id；offline/online 模式真正续接）。
- F10：`_resolve_teacher_class_indices` 对无 teacher 匹配的 student 类别输出 WARNING 并列出类名。
- F11：转换脚本改 `RETR_CCOMP` + `merge_multi_segment`，每实例一行、孔洞保留（带孔实例 IoU 0.826→0.949，受影响实例占 15.6%）。
- F18：`DistillationModel.names` property 同步 student names 并重算 `teacher_class_indices`；resume 重建 student 时回填 checkpoint names；B/C 历史 checkpoint 的数字类名已就地修复（详见 §4.1 F18）。
- F19：脚本 resume 只透传显式 CLI 参数（`cli_provided` 扫 `sys.argv`）；trainer `check_resume` 输出生效 overrides 与被忽略项 WARNING；白名单补 `dis`。
- F21：`loss_sl2` 改分块累加，消除双 (N,C,HW) 临时张量；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 进脚本默认（F20 缓解）。验证：数值/梯度逐位等价 + coco8-seg 蒸馏冒烟 + `test_distill_resume` 通过。

review 文档遗留项：1.2（独立 seg/sem gain）、1.4（o2m 可配）仍未落地，与 F 系列并行推进；1.1（NaN clamp）、1.3（E2E 日志）已应用。
⚠️ 兼容性说明：旧 run 的 `results.csv` 是 `train/dis_loss` 单列，新 run 为 `train/dis_feat`+`train/dis_proto` 两列，跨代对比时 dis_loss≈dis_feat+dis_proto。

---

## 5. 下一步改进详细计划（2026-07-03 修订）

> 前提变化（07-03 两次修订）：阶段 C 已完成；§3.1 一度得出"teacher 已被追平"，但 §3.1.2 v2 标签复评证明这是 v1 标签噪声伪象——**teacher 在干净标尺上仍领先 box +0.081 / mask +0.031（v2 val）、+0.143 / +0.092（COCO val2017），蒸馏与数据质量并列为主增长轴**。计划结构不变：评测收口（P1）→ 数据 + 配方 + 蒸馏主线（P2）→ 蒸馏消融（P3）→ 工程还债（P4）。

### P1 评测收口（不动训练代码，1 天内）

| # | 任务 | 做法 | 产出/验收 |
|---|---|---|---|
| P1-1 | **COCO val2017 官方标尺对照**（F13 剩余项） | **✅ 已完成（07-03）**，结果与结论见 §3.1.1：C 在 COCO 标尺低于官方 yolo26s-seg（box −0.053 / mask −0.030），确认标注风格过拟合；B best（78 类）需索引重映射暂缺 | 双标尺结论已固化：P2 起全部实验双标尺验收 |
| P1-2 | 逐类诊断固化 | 从 C best 终评提取 per-class AP，标记 mAP50-95(B)<0.2 的弱类清单（当前：hair drier/tie/backpack/potted plant/handbag/toothbrush/spoon/bench） | 弱类清单进文档，作为 P2-3 采样加权与后续对照的基线 |
| P1-3 | 实验区清理 | 归档废弃 run（无 lvispretrain 空目录）；B 目录补 `NOTES.md` 记录 b80/2GPU→b150/3GPU 切换点（F17） | runs/segment 只留有效 run，每个 run 有可追溯说明 |

### P2 主增长轴：数据 + 训练配方（1–2 周，GPU 主要投入）

| # | 任务 | 做法 | 预期/验收 |
|---|---|---|---|
| P2-1 | **标签质量修复**（F11/F12） | **✅ 标签已重建（07-03）**：新数据集 `Dataset/COCONut_b_yolo_seg_v2/`（旧版保留对照）。统计对比符合预期：train 标签行 2365072 → **1797818（−24%，断裂实例合并生效）**，thing_segments 1833988 不变，即旧法每实例平均写 1.29 行；val 57220 → 45003。v2 val 复评已完成（§3.1.2），v2 val 取代 v1 val 成为主标尺。iscrowd ignore 机制暂缓（需框架级支持） | ✅ 完成；后续训练一律指向 `COCONut_b_yolo_seg_v2/coconut-b-seg.yaml` |
| P2-2 | **配方升级重训**（替代原"续训"方案：lr 已衰减到 0.0006，resume 收益有限） | 从 C best.pt 起，**用 v2 标签**：`epochs=200, cos_lr=True, close_mosaic=20, copy_paste=0.4, mixup=0.1, multi_scale=0.25`；蒸馏权重**维持 `dis=3.0, dis_proto=1.0`**（原"减半"依据的 §3.1 追平结论已被 §3.1.2 推翻，teacher 仍有充足增量信息）。**🚀 已启动（07-03）**：run `yolo26s-seg-coconut-b-v2-distill-recipe200`，batch=96（multi_scale 上限 800px 需留显存，实测 24.1G/32G）、3×5090D，约 21 min/epoch ≈ 3 天 | 双标尺验收：v2 val 上超过 C best 复评基线（box 0.432 / mask 0.373，§3.1.2）**且** COCO val2017 不低于官方 yolo26s-seg（box ≥ 0.468 / mask ≥ 0.386，§3.1.1 硬约束）。曲线出现平台才算训练充分 |
| P2-3 | 弱类补强 | 对 P1-2 弱类清单：核查训练集实例数与标签质量（碎片/退化多边形占比），copy_paste 天然缓解；必要时用 `set_class_weights` 钩子加权 | 弱类 box mAP50-95 平均 +0.03 且整体不降 |
| P2-4 | 数据规模化（可选，视 P2-2 收益） | COCONut-L（~358k 图，`--train-split` 已参数化）；或 teacher/SAM3 伪标签扩充（见 §6） | 同标尺对照，增益/成本比达标再纳入主线 |

### P3 蒸馏侧消融（每项 ≤ 短程对照成本，结论驱动取舍）

| # | 任务 | 做法 | 判据 |
|---|---|---|---|
| P3-1 | proto 蒸馏 on/off（F7） | P2-2 配方上 `dis_proto=0` 对照组 | mask AP 差 <0.002 则默认关闭，简化链路 |
| P3-2 | `dis` 余弦衰减（F8） | warmup factor 机制上加衰减分支（`distill_decay=cosine`，末期降到 0.3×） | 优于恒定 dis 则并入默认 |
| P3-3 | response KD（F6） | head cls logits 温度化 KL（`teacher_class_indices` 对齐通道），`dis_cls=1.0` 起步 | **P1-1 已显示 student 的 COCO 泛化明显弱于 teacher（box −0.143，§3.1.1）→ 优先级上调**：teacher 的 cls 软标签可能是把 COCO 风格知识传回 student 的低成本通道，值得在 P2-2 之后做一轮 |
| P3-4 | 无蒸馏对照 | P2-2 同配方 `distill_model=None` | 量化蒸馏的真实增益（v2 标尺下 teacher 仍领先，预期蒸馏为正收益，需给出数值） |

### P4 工程还债（穿插进行，不占 GPU）

| # | 任务 | 内容 |
|---|---|---|
| P4-1 | 小修（半天） | **✅ 已完成（07-03）**：F4 / F14 / F15 / F10，明细见 §4.7 |
| P4-2 | 蒸馏单测（F16） | coco8-seg + yolo26n-seg 蒸馏 yolo26s-seg 冒烟：proto 蒸馏、类别失配（人工构造 names 差异）、warmup/sanitize、`__getstate__` pickle 往返 |
| P4-3 | 解耦重构（F1/F2/F3/F20，安排在下次 rebase 上游前） | 接口设计：`DistillationModel` 暴露 `get_trainable_modules()`（trainer freeze/optimizer 只面向它）、`state_for_checkpoint()`/`load_from_checkpoint()`（EMA 剥离与 resume 重建收进模型自身）、`on_train_batch_start(ni, nb)`（warmup/衰减调度自管理，经 callback 注入替代 trainer 内 hasattr 特判）；Detect/Segment 加 `forward(raw=True)` 显式接口替代 training-flag hack；DDP skip-batch OOM 恢复（catch → all_reduce 标志 → 全 rank 同步跳批 + empty_cache）与主循环改动一并做 |
| P4-4 | review 遗留 | 1.2 独立 `seg`/`sem` gain（默认回退 box 保兼容）、1.4 o2m 可配 |

### 执行顺序与依赖

```
P1-1 COCO标尺对照 ──┬─▶ 决定 P3-3 是否做
P1-2 弱类清单 ─────┼─▶ P2-3
P2-1 标签修复 ─────┴─▶ P2-2 配方重训（主线，200 epoch ≈ 2×当前时长）
                        ├─▶ P3-1/P3-2/P3-4 消融（与主线并行，用 30–50 epoch 短程）
P4-1/P4-2 随时插空；P4-3 在 rebase 前完成
```

纪律：每次只改一个变量；短程（30 epoch，COCONut-S 或 fraction=0.3）先筛，胜出者上全量；所有对照都在 COCONut val + COCO val 双标尺报数。

---

## 6. 数据集扩展的切入角度

按"性价比 = 预期增益 / 工程成本"排序：

1. **先榨干现有数据的质量**（成本最低）：F11/F12 的标签修复直接作用于全部 241k 训练图；用工作区已有的 **fastdup**（`Code/fastdup`）对 COCONut/LVIS 训练集做近重复检测、异常图（过暗/损坏/极端长宽比）与标签离群（超大/超碎实例）清洗——重复图会放大过拟合且浪费 epoch。
2. **换更大的同源数据**：COCONut 家族还有 **COCONut-L**（COCONut-B + LVIS 图，~358k）；转换脚本已参数化（`--train-split`），扩展成本主要是磁盘和一次转换。这是当前链路最顺的规模化路径。
3. **teacher 伪标签半监督**（蒸馏框架的自然延伸）：用 yolo26x-seg（或工作区的 **SAM3**）给无标注源打伪标签——Objects365 / OpenImages 图像量大且域接近 COCO；高置信度伪标签当 GT，低置信度区域只做特征/logits 蒸馏（无标签蒸馏分支，当前 loss 结构加一个 `batch["unlabeled"]` 路径即可）。注意伪标签阈值消融，避免 confirmation bias。
4. **直接引入带 mask 的公开集**：OpenImages instance masks（~2.7M masks，需类别映射到 80 类子集）、LVIS 全量（已在本地，可作 80 类子集补充源，注意与 COCO val 图像去重）。
5. **针对性补弱类**：先看 per-class AP（validator 已输出），对落后类用 copy_paste 源库或补采样；LVIS 的 repeat-factor 采样思路可迁移（框架有 `set_class_weights` 钩子）。
6. **域特化**（若下游是人像/抠图）：工作区已有 P3M-10K、HIM2K、VideoMatting240K 等 matting 数据，可转成 person 类实例分割监督，强化边缘质量。

共性注意：多源混合时 ① 类别名对齐要过 F10 的映射机制；② 各源标注风格（amodal vs modal、边缘松紧）不一致会互相拉扯，建议按源分组消融而非一次全混。

## 7. 训练 trick 的切入角度

当前配置基本是"默认增强 + 恒定蒸馏权重"，可挖空间从大到小：

1. **分割专用增强（预期收益最大）**：`copy_paste=0.3~0.5`（当前 0；对实例分割普遍 +1~2 mask AP，且天然缓解类不平衡）、`mixup=0.1`、`multi_scale=0.25~0.5`（蒸馏路径已支持动态尺寸，见 `test_distill_resume` 的 `multi_scale=0.5`）。注意 copy_paste 会造出新的遮挡/断裂 mask，需先落地 F11 的标签修复与 P0 clamp（已应用）。
2. **训练时长与调度**：100 epoch 明显未收敛（三阶段 best 均在末轮），优先直接拉长到 200~300 epoch + `cos_lr=True`；`close_mosaic` 相应放大到 20~30。这比任何精细 trick 都实在。
3. **蒸馏调度类**：`dis` 余弦衰减（F8）、proto 蒸馏延迟启用（close_mosaic 后）、response KD（F6）、warmup 已有。进阶：teacher 换 clean 图前向（F9）、多 teacher（26x + 26l 平均）——按消融结果决定，不建议叠加超过 2 个未验证的蒸馏 trick。
4. **优化器与 batch**：MuSGD 的 head/semseg lr×3 分组已内置；显存允许时增大 batch 并按 `nbs=64` 的线性缩放检查等效 lr；`weight_decay` 会随 batch/accumulate 自动缩放（`trainer.py:284`），改 batch 时留意。
5. **监督结构**：落地 review 1.2（独立 `seg`/`sem` gain）后单独调 mask 权重；o2m 起点/终点（1.4）对 e2e 头收敛速度有影响，官方 recipe 用 0.705。
6. **评测与选择**：best.pt 由 fitness（默认偏 mAP50-95）选择，长训时建议 `save_period=5` 保留多 checkpoint 事后按目标指标挑；最终交付前用 EMA 权重（已默认）+ 更大 imgsz 验证一次。

执行纪律：每次只改一个变量，用固定 seed + 同一 val 标尺跑对照；短程代理实验（COCONut-S / 30 epoch）先筛，胜出者再上 COCONut-B 全程。

---

## 8. 附录：逐阶段详细分析（2026-07-01 快照）

> 数据来源：`runs/segment/<run>/results.csv`、`args.yaml`、`lvis_batch_eval_summary.csv`、swanlab 本地日志、`train_logs/`。

### 8.A 阶段 A — 完整 LVIS 普通训练

已完成 100 epoch。`results.csv` 内嵌的 COCO 式 mAP 对 1203 类长尾严重失真（末轮 0.086/0.071，无参考意义）；LVIS 官方 API 逐 checkpoint 评估（`lvis_batch_eval_summary.csv`）显示 box AP_all 0.084@e5 → 0.198@e85、mask AP_all → 0.167，单调上升未见平台。稀有类 AP_r 仅 0.028——s 规模在 1203 类长尾上对稀有类基本没学到（任务难点，非 bug）。**A 的权重未被 B/C 使用**，定位是独立基线。

### 8.B 阶段 B — LVIS·COCO80 子集蒸馏

已完成 100 epoch。student 从 `yolo26s-seg.yaml` 零起，teacher yolo26x-seg，dis=3.0 无 proto。`mAP50-95(B)` 0.002→0.343、(M)→0.315，best 在末轮仍升；`dis_loss` 2.42→0.58 平稳，无 NaN/尖峰。类别失配 80→78 经 `teacher_class_indices` 处理无报错。遗留：验证集非标准 COCO val 无法对齐官方数字；teacher 同标尺基线未测（F13）；run 名与实际 batch/device 不符（F17）。

### 8.C 阶段 C — COCONut-B 蒸馏（LVIS 预训练初始化）【已完成】

**07-02 08:25 跑满 100 epoch。** student 用 B 的 best.pt 初始化，dis=3.0、dis_proto=1.0、distill_warmup_epochs=3。关键曲线（训练中验证值）：

| epoch | mAP50(B) | mAP50-95(B) | mAP50(M) | mAP50-95(M) | train/dis_loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.433 | 0.306 | 0.420 | 0.271 | 0.42（warmup 压低） |
| 30 | 0.491 | 0.357 | 0.480 | 0.318 | 1.08 |
| 60 | 0.524 | 0.383 | 0.510 | 0.340 | 0.99 |
| 90 | 0.541 | 0.395 | 0.526 | 0.350 | 0.86 |
| 100 | 0.546 | 0.398 | 0.530 | 0.353 | 0.79 |

- best.pt = **e98**（fitness=box+mask 加权和）；best.pt 终评：box 0.657P/0.503R/0.546/0.402，mask 0.649/0.486/0.523/0.341。
- 收敛速率逐段衰减（+0.009→+0.0025/10epoch），末期 lr≈0.0006，缓升未平台——续 resume 收益有限，重训拉长 epoch 更合理（P2-2）。
- close_mosaic（e91）后 `sem_loss` 0.150→0.070、`cls_loss` 1.44→1.39，末 10 epoch 贡献约 +0.003 box。
- 全程无 NaN/尖峰；resume 造成 swanlab 曲线 3 段分片（F15）。
- **teacher 同标尺基线（07-03 补测）与 student 对比见 §3.1**：box 追平（0.402 vs 0.400）、mask 反超（0.341 vs 0.327）。

### 8.D 可比性说明

三阶段 val 集互不相同，0.343(B) 与 0.402(C) **不可直接比较**——C 的优势部分来自 B 初始化与更"顺"的重标注 val。同标尺对比见 §3.1（COCONut val 已闭合），COCO val 官方标尺对照见 P1-1（待做）。
