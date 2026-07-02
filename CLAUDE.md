# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Ultralytics YOLO — the library behind the `yolo` / `ultralytics` CLI and the `from ultralytics import YOLO` Python API. Supports detection, instance/semantic segmentation, classification, pose, oriented-bbox (OBB), tracking, and export to many inference formats. AGPL-3.0 licensed. Python >=3.8, PyTorch >=1.8.

## Common commands

```bash
pip install -e .            # editable install; required so `yolo` CLI and `ultralytics.*` imports resolve

# Tests (pytest, with --doctest-modules enabled by default in pyproject — docstring >>> examples are executed)
pytest                              # full suite (downloads weights/datasets on first run)
pytest tests/test_python.py -k "test_predict"   # single test / keyword filter
pytest --slow                      # include tests marked slow (skipped by default)
pytest -n auto                     # parallel via pytest-xdist
pytest tests/test_exports.py --export-env <id>   # run only export tests for a given export environment

# Lint / format (line-length = 120 everywhere)
ruff format ultralytics            # formatter of record (yapf/isort/docformatter configs also present)
ruff check ultralytics
codespell ultralytics              # uses the ignore-words-list / skip config in pyproject.toml

# CLI smoke check
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` and `ultralytics` console scripts both resolve to `ultralytics.cfg:entrypoint`.

## Architecture (the big picture)

### Model → task dispatch (the central mechanism)
`engine/model.py` defines `Model` (the class behind `YOLO()`). A model's *task* (detect/segment/classify/pose/obb/semantic) is inferred at load time by `nn/tasks.guess_model_task` from the architecture/weights, and determines which trainer/validator/predictor classes are used. Dispatch happens through `Model._smart_load(key)`, which looks up `key` in the model class's `task_map()` dict:

```
task_map() -> { "detect": { "model": DetectionModel, "trainer": DetectionTrainer,
                            "validator": DetectionValidator, "predictor": DetectionPredictor }, ... }
```

The per-task `task_map()` implementations live in `models/yoo/{task}/...` (each task subpackage has `train.py`, `val.py`, `predict.py` exporting the task-specific Trainer/Validator/Predictor). Adding a new task = new subpackage under `models/yolo/` + a `task_map` entry + a `nn/tasks.py` model class + a `cfg/models` YAML family.

`MODES = {train, val, predict, export, track, benchmark}` and `TASKS = {detect, segment, classify, pose, obb, semantic}` are the taxonomy defined in `cfg/__init__.py`; the CLI `entrypoint` parses `yolo TASK MODE args...` against these.

### Engine (`ultralytics/engine/`)
The reusable training/inference framework, task-agnostic:
- `trainer.py` — `BaseTrainer.train()` → `_do_train()` loop. Subclasses override hooks: `get_model`, `get_dataloader`, `preprocess_batch`, `get_validator`, `criterion`, `optimizer_step`. Handles DDP, AMP, EMA, checkpointing, callbacks.
- `validator.py` — `BaseValidator`; runs evaluation, builds metrics, plots.
- `predictor.py` — `BasePredictor`; streaming inference pipeline.
- `exporter.py` — ONNX/TensorRT/CoreML/OpenVINO/TFLite/etc. export.
- `results.py` — `Results`/`Boxes`/`Masks`/`Keypoints`/`Probs` output objects.
- `tuner.py` — hyperparameter search.

### Neural net (`ultralytics/nn/`)
- `tasks.py` — `BaseModel` and all task-specific model classes (`DetectionModel`, `SegmentationModel`, `PoseModel`, `ClassificationModel`, `OBBModel`, `SemanticSegmentationModel`, `RTDETRDetectionModel`, `WorldModel`, `YOLOEModel`, …). `parse_model(d, ch)` turns a YAML architecture dict into a `nn.Module` graph; `yaml_model_load` reads a `cfg/models/*.yaml`. This is where the layer-by-layer build happens.
- `modules/` — `conv.py`, `block.py`, `head.py`, `transformer.py`, `activation.py` building blocks referenced by name from YAML.
- `autobackend.py` — runtime inference backend selection (PyTorch/ONNX/OpenVINO/TRT/CoreML/…).
- `distill_model.py` — `DistillationModel` + `FeatureHook`, feature/logit knowledge-distillation wrapper (teacher + student). Used by the distillation training scripts under `scripts/`; the `yolo26x-seg.pt` teacher at repo root is for that work. Distillation losses live in `utils/loss.py`.

### Config (`ultralytics/cfg/`)
- `default.yaml` — the single source of truth for *all* training/val/predict/export settings and hyperparameters. Parsed by `cfg.get_cfg` into an `IterableSimpleNamespace`. CLI args and `model.train(...)` kwargs merge on top of this. When adding a setting, add it here with an inline `# (type) description` comment.
- `datasets/*.yaml` — dataset definitions (paths, class counts, names, download URLs).
- `models/{v3,v5,v6,v8,v9,v10,v11,v12,26,...}/*.yaml` — model architectures. `yolo26*-seg.yaml` etc. define the current default family.

### Data (`ultralytics/data/`)
`build.py` builds dataloaders, `dataset.py`/`base.py` define `YOLODataset`/`BaseDataset`, `augment.py` holds the mosaic/mixup/perspective transforms, `loaders.py` stream sources (images/videos/streams/tensors), `converter.py` converts COCO/YOLO/DOTA/etc. annotation formats.

### Utils (`ultralytics/utils/`)
`loss.py` (task loss functions + distillation losses), `tal.py` (TaskAlignedAssigner), `ops.py` (NMS, box conversions), `metrics.py` (mAP/AP), `torch_utils.py` (model init/EMA/fuse/paraphrase), `callbacks/` (per-experiment-tracker hooks: tensorboard, wb, mlflow, comet, clearml, neptune, dvc, raytune, swanlab, hub, platform, base). Callbacks register on trainer/validator/predictor events.

### Other top-level packages
`trackers/` (ByteTrack/BotSORT/OC-SORT/etc.), `solutions/` (analytics/heatmap/counter/gym/speed/parking — dispatched via `cfg.SOLUTION_MAP`), `hub/` (Ultralytics HUB auth/session), `optim/` (Muon optimizer).

## Conventions

- **Line length 120**; Ruff is the formatter (`ruff format`), with Google-style pydocstrings configured. `yapf`, `isort`, `docformatter`, and `codespell` configs in `pyproject.toml` are also used in CI — keep to their settings.
- **Docstrings are executable tests**: `pytest` runs with `--doctest-modules`, so `>>>` examples in docstrings must stay valid (run `pytest ultralytics/path/to/module.py` to check one file's doctests).
- New settings go in `cfg/default.yaml` with a typed inline comment; they propagate automatically through `get_cfg`.
- New model architectures go in `cfg/models/<family>/<name>.yaml` and are built by `nn/tasks.parse_model` from the block names registered in `nn/modules/`.
- Task-specific behavior is added by subclassing the `engine/` base classes and wiring them in via the model's `task_map()`, not by branching on task inside the engine.