# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests import MODEL, SOURCE, TASK_MODEL_DATA
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo import classify, detect, obb, pose, segment, semantic
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import ASSETS, DEFAULT_CFG, IS_RASPBERRYPI, WEIGHTS_DIR
from ultralytics.utils.torch_utils import unwrap_model


def test_func(*args, **kwargs):
    """Test function used as a callback stub to verify callback registration."""
    print("callback test passed")


def test_export(monkeypatch, tmp_path):
    """Test model exporting functionality by adding a callback and verifying its execution."""
    monkeypatch.chdir(tmp_path)
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "on_export_start callback not registered"
    f = exporter(model=YOLO("yolo26n.yaml").model)
    YOLO(f)(SOURCE)  # exported model inference


@pytest.mark.parametrize(
    "trainer_cls,validator_cls,predictor_cls,data,model,weights",
    [
        (
            detect.DetectionTrainer,
            detect.DetectionValidator,
            detect.DetectionPredictor,
            "coco8.yaml",
            "yolo26n.yaml",
            MODEL,
        ),
        (
            segment.SegmentationTrainer,
            segment.SegmentationValidator,
            segment.SegmentationPredictor,
            "coco8-seg.yaml",
            "yolo26n-seg.yaml",
            WEIGHTS_DIR / "yolo26n-seg.pt",
        ),
        (
            classify.ClassificationTrainer,
            classify.ClassificationValidator,
            classify.ClassificationPredictor,
            "imagenet10",
            "yolo26n-cls.yaml",
            None,
        ),
        (obb.OBBTrainer, obb.OBBValidator, obb.OBBPredictor, "dota8.yaml", "yolo26n-obb.yaml", None),
        (pose.PoseTrainer, pose.PoseValidator, pose.PosePredictor, "coco8-pose.yaml", "yolo26n-pose.yaml", None),
        (
            semantic.SemanticSegmentationTrainer,
            semantic.SemanticSegmentationValidator,
            semantic.SemanticSegmentationPredictor,
            "cityscapes8.yaml",
            "yolo26n-sem.yaml",
            None,
        ),
    ],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_task(trainer_cls, validator_cls, predictor_cls, data, model, weights):
    """Test YOLO training, validation, and prediction for various tasks."""
    overrides = {
        "data": data,
        "model": model,
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "mask_ratio": 1,
        "overlap_mask": False,
    }

    # Trainer
    trainer = trainer_cls(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "on_train_start callback not registered"
    trainer.train()

    # Validator
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data
    cfg.imgsz = 32
    val = validator_cls(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "on_val_start callback not registered"
    val(model=trainer.best)

    # Predictor
    pred = predictor_cls(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "on_predict_start callback not registered"

    # Determine model path for prediction
    model_path = weights if weights else trainer.best
    if model == "yolo26n.yaml":  # only for detection
        # Confirm there is no issue with sys.argv being empty
        with mock.patch.object(sys, "argv", []):
            result = pred(source=ASSETS, model=model_path)
            assert len(result) > 0, f"Predictor returned no results for {model}"
    else:
        result = pred(source=ASSETS, model=model_path)
        assert len(result) > 0, f"Predictor returned no results for {model}"

    # Test resume functionality
    with pytest.raises(AssertionError):
        trainer_cls(overrides={**overrides, "resume": trainer.last}).train()


@pytest.mark.parametrize("task,weight,data", TASK_MODEL_DATA)
def test_resume_incomplete(task, weight, data, tmp_path):
    """Test training resumes from an incomplete checkpoint."""
    train_args = {
        "data": data,
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": task,
        "imgsz": 32,
        "exist_ok": True,
    }

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    def disable_final_eval(trainer):
        trainer.final_eval = lambda: None

    model = YOLO(weight)
    model.add_callback("on_train_start", disable_final_eval)
    model.add_callback("on_train_epoch_end", stop_after_first_epoch)
    model.train(**train_args)
    last_path = model.trainer.last
    _, ckpt = load_checkpoint(last_path)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"

    # Resume training using the checkpoint
    resume_model = YOLO(last_path)
    resume_model.train(resume=True, **train_args)
    assert resume_model.trainer.start_epoch == resume_model.trainer.epoch == 1, "resume test failed"


def test_resume_epochs_override(tmp_path: Path):
    """Test resume can extend the total epoch budget via the epochs override (whitelisted in check_resume)."""
    train_args = {
        "data": "coco8.yaml",
        "model": "yolo26n.yaml",
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": "resume_epochs",
        "imgsz": 32,
        "exist_ok": True,
    }

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    trainer = detect.DetectionTrainer(overrides=train_args)
    trainer.final_eval = lambda: None
    trainer.add_callback("on_train_epoch_end", stop_after_first_epoch)
    trainer.train()
    _, ckpt = load_checkpoint(trainer.last)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"

    # Resume with a larger epoch budget: the override must extend total epochs instead of being ignored.
    resume = detect.DetectionTrainer(overrides={**train_args, "resume": trainer.last, "epochs": 4})
    resume.final_eval = lambda: None
    resume.train()
    assert resume.epochs == 4, "epochs override should extend the resumed run"
    assert resume.start_epoch == 1 and resume.epoch == 3, "resumed run should train the extended epochs to completion"


def test_distill_resume(tmp_path: Path):
    """Test knowledge distillation resumes from an incomplete checkpoint."""
    overrides = {
        "data": "coco8.yaml",
        "model": "yolo26n.yaml",
        "distill_model": WEIGHTS_DIR / "yolo26s.pt",
        "imgsz": 32,
        "multi_scale": 0.5,  # vary per-batch image size to exercise dynamic distillation score splitting
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": "distill",
        "exist_ok": True,
    }

    # Train for one epoch then interrupt to produce a resumable checkpoint
    trainer = detect.DetectionTrainer(overrides=overrides)

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    trainer.final_eval = lambda: None
    trainer.add_callback("on_train_epoch_end", stop_after_first_epoch)
    trainer.train()
    _, ckpt = load_checkpoint(trainer.last)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"
    assert isinstance(ckpt["ema"], DistillationModel), "distillation EMA wraps the student model"
    assert ckpt["ema"].teacher_model is None, "teacher should be stripped from the EMA/checkpoint"
    assert ckpt["ema"].projector is not None, "the distillation projector should be persisted in the EMA checkpoint"

    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.final_eval = lambda: None
    trainer.train()
    model = unwrap_model(trainer.model)
    assert isinstance(model, DistillationModel), "resume should rebuild the DistillationModel"
    assert model.teacher_model is not None, "resume should rebuild the teacher from the distill_model path"
    assert model.student_model.names == trainer.data["names"], "resume should sync dataset names to the student"
    assert model.names == trainer.data["names"], "distillation wrapper names should proxy student names"
    assert trainer.start_epoch == trainer.epoch == 1, "resume test failed"


# Segmentation distillation teacher: prefer a local checkpoint, else let Ultralytics resolve/download the name.
SEG_TEACHER = next(
    (p for p in (WEIGHTS_DIR / "yolo26s-seg.pt", WEIGHTS_DIR.parent / "yolo26s-seg.pt") if p.exists()),
    "yolo26s-seg.pt",
)


def _build_seg_distill_model(dis_proto: float = 1.0, imgsz: int = 32) -> DistillationModel:
    """Build a standalone segmentation DistillationModel outside the trainer for fast unit tests.

    The student is created from the config YAML, so it carries cfg-default numeric class names ({0: '0', ...})
    that do not match the teacher, which is exactly the fixture needed to exercise class-alignment code paths.
    """
    student = YOLO("yolo26n-seg.yaml").model
    args = get_cfg(DEFAULT_CFG)
    args.imgsz = imgsz
    args.dis = 3.0
    args.dis_proto = dis_proto
    args.distill_loss_clip = 10.0
    student.args = args
    student.eval()
    return DistillationModel(teacher_model=SEG_TEACHER, student_model=student)


def test_distill_seg_proto(tmp_path: Path):
    """Test segmentation proto distillation trains and logs dis_feat/dis_proto as separate finite columns."""
    overrides = {
        "data": "coco8-seg.yaml",
        "model": "yolo26n-seg.yaml",
        "distill_model": SEG_TEACHER,
        "dis_proto": 1.0,  # enable the segmentation-specific proto distillation branch
        "distill_warmup_epochs": 0.0,  # no ramp so proto loss is non-zero within a single-batch epoch
        "imgsz": 32,
        "multi_scale": 0.5,  # vary per-batch size to exercise dynamic score splitting and chunked loss_sl2
        "epochs": 1,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": "distill_seg",
        "exist_ok": True,
    }
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.final_eval = lambda: None
    trainer.train()
    assert "dis_feat" in trainer.loss_names and "dis_proto" in trainer.loss_names, "seg distillation splits dis columns"
    items = trainer.label_loss_items(trainer.tloss)
    dis = {k.split("/")[-1]: v for k, v in items.items() if "dis_" in k}
    assert {"dis_feat", "dis_proto"} <= set(dis), "both distillation components must be logged"
    assert all(v == v and v >= 0 for v in dis.values()), f"distillation losses must be finite and non-negative: {dis}"
    assert dis["dis_proto"] > 0, "proto distillation should contribute against a segmentation teacher"
    # validation drops the distillation columns (no misleading val/dis_*=0)
    assert not any("dis_" in k for k in trainer.label_loss_items(None, prefix="val")), "val must omit dis columns"


def test_distill_class_alignment():
    """Test teacher class-index resolution across full-match, subset, and synonym-mismatch student names."""
    dm = _build_seg_distill_model()
    teacher_names = {int(k): v for k, v in dm.teacher_model.names.items()}

    # Exact full overlap collapses to None (use every teacher class, no index_select).
    dm.names = dict(teacher_names)
    assert dm.teacher_class_indices is None, "full name overlap should use all teacher classes"

    # Dropping two student classes must select exactly the remaining teacher channels.
    subset = dict(enumerate(list(teacher_names.values())[:-2]))
    dm.names = subset
    idx = dm.teacher_class_indices
    assert idx is not None and idx.numel() == len(teacher_names) - 2, "subset must map to matching teacher indices"

    # A synonym/formatting difference drops only that one class (exact lowercase matching).
    renamed = dict(teacher_names)
    renamed[min(renamed)] = f"{renamed[min(renamed)]}_synonym_xyz"
    dm.names = renamed
    idx2 = dm.teacher_class_indices
    assert idx2 is not None and idx2.numel() == len(teacher_names) - 1, "renamed class should silently drop"


def test_distill_sanitize_and_warmup():
    """Test distillation loss sanitize (NaN/Inf/clip/negative) and warmup-factor clamping."""
    dm = _build_seg_distill_model()
    assert dm.sanitize_distill_loss(torch.tensor([float("nan")])).item() == 0.0, "NaN -> 0"
    assert dm.sanitize_distill_loss(torch.tensor([float("inf")])).item() == 10.0, "Inf -> clip"
    assert dm.sanitize_distill_loss(torch.tensor([999.0])).item() == 10.0, "spike clamped to clip"
    assert dm.sanitize_distill_loss(torch.tensor([-5.0])).item() == 0.0, "negative -> 0"

    dm.student_model.args.distill_loss_clip = 0.0  # 0 disables the finite upper clamp
    assert dm.sanitize_distill_loss(torch.tensor([999.0])).item() == 999.0, "clip=0 leaves large values"
    assert dm.sanitize_distill_loss(torch.tensor([float("inf")])).item() == 0.0, "clip=0 still maps Inf to 0"

    for factor, expected in [(2.0, 1.0), (-1.0, 0.0), (0.5, 0.5)]:
        dm.set_distill_warmup_factor(factor)
        assert dm.distill_warmup_factor == expected, f"warmup factor {factor} should clamp to {expected}"


def test_distill_pickle_roundtrip():
    """Test __getstate__/__setstate__ clear captured features and re-register hooks on deepcopy."""
    import copy

    dm = _build_seg_distill_model()
    # Batch size 2 mirrors the constructor's probe forward and avoids BatchNorm's N=1 train-mode error.
    imgs = torch.zeros(2, 3, dm.student_model.args.imgsz, dm.student_model.args.imgsz)
    dm._forward_teacher_for_distillation(imgs)
    dm.student_model(imgs)
    assert dm._teacher_feats and dm._student_feats, "forward passes should populate the shared feature dicts"

    clone = copy.deepcopy(dm)
    assert clone._teacher_feats == {} and clone._student_feats == {}, "pickling must not carry captured grad tensors"
    assert len(clone._student_hooks) == len(clone.feats_idx), "student hooks must be re-registered after unpickling"
    assert len(clone._teacher_hooks) == len(clone.feats_idx), "teacher hooks must be re-registered after unpickling"
    assert clone.names == dm.names, "wrapper names still proxy the student after a round-trip"

    # Re-registered hooks must actually capture on a fresh forward.
    clone._forward_teacher_for_distillation(imgs)
    clone.student_model(imgs)
    assert clone._student_feats and clone._teacher_feats, "re-registered hooks should capture features again"


@pytest.mark.parametrize(
    "ckpt",
    [
        {"model": OrderedDict([("a", torch.zeros(1))])},  # state_dict saved under the "model" key
        {"model": {"a": torch.zeros(1)}},  # plain-dict "model" value
        OrderedDict([("a", torch.zeros(1))]),  # bare state_dict, no "model" key
    ],
)
def test_load_checkpoint_state_dict_rejected(ckpt, tmp_path):
    """Test a state_dict checkpoint raises a clear TypeError instead of a cryptic AttributeError/KeyError."""
    weight = tmp_path / "bad.pt"
    torch.save(ckpt, weight)
    with pytest.raises(TypeError, match="supported Ultralytics checkpoint format"):
        load_checkpoint(weight)


def test_nan_recovery():
    """Test NaN loss detection and recovery during training."""
    nan_injected = [False]

    def inject_nan(trainer):
        """Inject NaN into loss during batch processing to test recovery mechanism."""
        if trainer.epoch == 1 and trainer.tloss is not None and not nan_injected[0]:
            trainer.tloss *= torch.tensor(float("nan"))
            nan_injected[0] = True

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 3}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", inject_nan)
    trainer.train()
    assert nan_injected[0], "NaN injection failed"


def test_checkpoint_fp16_overflow():
    """Test a finite model whose weights overflow fp16 is still checkpointed (clamped) instead of skipped."""

    def inflate_ema(trainer):
        """Push an EMA weight above the fp16 max (65504) so its fp16 snapshot would otherwise become Inf."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = 1.0e5

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 2}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", inflate_ema)
    trainer.train()
    assert trainer.last.exists(), "checkpoint not saved for a finite model with fp16-overflowing weights"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )
    # Validation must leave the live EMA fp32 and unchanged; checkpoint serialization may clamp its fp16 copy.
    ema_param = next(iter(trainer.ema.ema.parameters()))
    assert ema_param.dtype == torch.float32 and torch.isfinite(ema_param).all() and ema_param.flatten()[0] == 1.0e5, (
        "validation corrupted the live EMA"
    )


@pytest.mark.parametrize(
    "kwargs,uses_weights",
    [({}, True), ({"pretrained": True}, True), ({"pretrained": False}, False), ({"pretrained": MODEL}, True)],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_reuses_loaded_checkpoint_model(monkeypatch, kwargs, uses_weights):
    """Test training reuses loaded checkpoint config while respecting the pretrained argument."""
    model = YOLO("yolo26n.yaml")
    model.ckpt = {"checkpoint": True}
    model.ckpt_path = "/tmp/fake.pt"
    model.overrides["model"] = "ul://glenn-jocher/m2/exp-14"
    model.overrides["pretrained"] = False
    original_model = model.model
    captured = {}

    class FakeTrainer:
        def __init__(self, overrides=None, _callbacks=None):
            self.overrides = overrides
            self.callbacks = _callbacks
            self.model = None
            self.validator = SimpleNamespace(metrics=None)
            self.best = MODEL.parent / "nonexistent-best.pt"
            self.last = MODEL
            captured["trainer"] = self

        def get_model(self, cfg=None, weights=None, verbose=True):
            captured["cfg"] = cfg
            captured["weights"] = weights
            return original_model

        def train(self):
            return None

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    monkeypatch.setattr(model, "_smart_load", lambda key: FakeTrainer)
    monkeypatch.setattr(
        "ultralytics.engine.model.load_checkpoint",
        lambda path: (original_model, {"checkpoint": True}),
    )

    model.train(data="coco8.yaml", epochs=1, **kwargs)

    assert captured["trainer"].model is original_model, "Trainer model does not match original"
    assert captured["cfg"] == original_model.yaml, f"Config mismatch: {captured['cfg']} != {original_model.yaml}"
    assert captured["weights"] is (original_model if uses_weights else None), "Unexpected weights loaded"


@pytest.mark.parametrize("pretrained,uses_weights", [(True, True), (False, False), (MODEL, True)])
def test_setup_model_respects_pretrained_arg_for_pt_models(monkeypatch, pretrained, uses_weights):
    """Test .pt models use checkpoint config while respecting the pretrained argument."""
    captured = {}
    checkpoint_model = SimpleNamespace(yaml={"nc": 80})
    trainer = object.__new__(BaseTrainer)
    trainer.model = "yolo26n.pt"
    trainer.args = SimpleNamespace(pretrained=pretrained)
    trainer.resume = False

    def fake_get_model(cfg=None, weights=None, verbose=True):
        captured["cfg"] = cfg
        captured["weights"] = weights
        return SimpleNamespace()

    trainer.get_model = fake_get_model
    monkeypatch.setattr(
        "ultralytics.engine.trainer.load_checkpoint", lambda path: (checkpoint_model, {"checkpoint": True})
    )

    trainer.setup_model()

    assert captured["cfg"] == checkpoint_model.yaml, "Checkpoint config was not used"
    assert captured["weights"] is (checkpoint_model if uses_weights else None), "Unexpected weights loaded"
