# YoloSeg-Train — GitHub Copilot Instructions

This repository is a **training fork** of [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) for YOLO26s-seg instance segmentation on LVIS / COCONut. It is **not** the upstream library release pipeline.

For full architecture, CLI, and engine conventions, read [`CLAUDE.md`](../CLAUDE.md) at the repo root.

## Project goal

Train and finetune **YOLO26s-seg** with optional **teacher distillation** (yolo26x-seg) and **PointRend boundary refine** on COCONut-B v2 labels. Stages A–E are documented in `README.md` and `docs/releases/`.

## Custom modules — read before editing

| Area                 | Path                                                   | Purpose                                      |
| -------------------- | ------------------------------------------------------ | -------------------------------------------- |
| Distillation wrapper | `ultralytics/nn/distill_model.py`                      | Teacher–student feature + proto MSE          |
| Seg / point losses   | `ultralytics/utils/loss.py`                            | `seg_point`, `seg_bnd`, E2E one2many/one2one |
| PointRend head       | `ultralytics/nn/modules/head.py`                       | `PointHeadMLP`, zero-init on finetune        |
| PointRend YAML       | `ultralytics/cfg/models/26/yolo26s-seg-pointrend.yaml` | Adds point_head to recipe200 backbone        |
| Boundary utils       | `ultralytics/utils/mask_boundary_loss.py`              | Sobel boundary-weighted point sampling       |
| Trainer hooks        | `ultralytics/engine/trainer.py`                        | Resume, DDP, distill rebuild, seg_point cfg  |

## Training scripts (`scripts/`)

- `train_yolo26s_seg_lvis_coco80_distill.py` — Stage B (LVIS·COCO80 distill)
- `train_yolo26s_seg_coconut_distill.py` — Stage C (COCONut-B distill)
- `finetune_yolo26s_seg_pointrend_coconut_b.py` — Stage E (PointRend finetune from recipe200 best.pt)

**Conventions:** argparse at top, `DEFAULT_*` path constants, `parse_known_args()` for Ultralytics overrides. When model structure changes (e.g. adding `point_head`), use `pretrained=<ckpt>` + `resume=False` — never `resume=True`.

## Key docs

- `docs/yolo26s-seg-distill-training-flow.md` — distillation issues F1–F17
- `docs/yolo26-seg-training-review.md` — loss review 1.1–1.9
- `docs/yolo26s-seg-pointrend-refine-head-design.md` — PointRend design
- `docs/coconut-yolo26s-seg-distill.md` — COCONut data + commands

Dataset paths (local, not in repo): `/home/genesis/Train/Dataset/`.

## Coding conventions

- Line length **120**; `ruff format` / `ruff check`
- New training settings → `ultralytics/cfg/default.yaml` with typed inline comment
- Task behavior via `task_map()` subclasses, not task branches inside `engine/`
- Doctests in docstrings are executed by pytest (`--doctest-modules`)

## Safe tasks for Copilot Coding Agent

- Unit tests under `tests/` (e.g. point_head init, mask boundary loss)
- Docstring / README / doc fixes
- Single-file refactors with clear acceptance criteria
- Small bug fixes isolated to one module

## Forbidden — do NOT

- Delete or revert files under `scripts/`, `docs/yolo26*`, `docs/releases/`, `ultralytics/utils/mask_*`, or PointRend YAML/tests
- Bulk-merge upstream `ultralytics/ultralytics` main without explicit review
- Change `cfg/default.yaml` training defaults without documenting in `docs/`
- Auto-update `pyproject.toml` export optional-deps in ways that touch non-export files
- Set `resume=True` when loading a checkpoint whose architecture differs (e.g. adding point_head)

## PR review checklist

Before merging any bot or agent PR:

```bash
git fetch origin pull/ID/head:pr-test
git diff main..pr-test --stat
```

Reject if the diff touches `scripts/finetune_*`, `docs/yolo26*`, or deletes >100 lines outside `pyproject.toml` / `.github/`.

## OOM / training notes (from prior runs)

- recipe200 OOM at batch 90 + multi_scale 0.25 on imgsz 640, 3×GPU — use batch ≤84, multi_scale ≤0.15
- Point supervision on **one2many** only (`seg_point_o2o=0.0`); one2one feats are detached
- Distillation off by default for PointRend finetune (marginal gain, saves teacher VRAM)
