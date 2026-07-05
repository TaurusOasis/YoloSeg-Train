# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

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


def test_resume_point_refine_overrides_are_whitelisted(monkeypatch, tmp_path: Path):
    """Resume should allow same-architecture point/refine loss overrides to reach model.args."""
    import ultralytics.engine.trainer as trainer_module

    last = tmp_path / "last.pt"
    data = tmp_path / "data.yaml"
    last.touch()
    data.write_text("path: .\ntrain: images\nval: images\nnames: {0: obj}\n", encoding="utf-8")
    ckpt_args = vars(get_cfg(DEFAULT_CFG)).copy()
    ckpt_args.update(
        {
            "data": str(data),
            "model": "yolo26n-seg-pointrend.yaml",
            "task": "segment",
            "mode": "train",
            "resume": False,
            "seg_point": 0.0,
            "seg_point_refine": False,
            "seg_point_num": 112,
            "seg_point_o2o": 1.0,
            "seg_point_refine_o2o": True,
            "e2e_final_o2m": 0.1,
        }
    )
    monkeypatch.setattr(trainer_module, "check_file", lambda p: str(p))
    monkeypatch.setattr(trainer_module, "load_checkpoint", lambda _: (SimpleNamespace(args=ckpt_args), {"epoch": 0}))
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(resume=str(last), data=str(data))

    BaseTrainer.check_resume(
        trainer,
        {
            "resume": str(last),
            "data": str(data),
            "seg_point": 0.2,
            "seg_point_refine": True,
            "seg_point_num": 64,
            "seg_point_o2o": 0.0,
            "seg_point_refine_o2o": False,
            "e2e_final_o2m": 0.3,
        },
    )

    assert trainer.resume is True
    assert trainer.args.seg_point == 0.2
    assert trainer.args.seg_point_refine is True
    assert trainer.args.seg_point_num == 64
    assert trainer.args.seg_point_o2o == 0.0
    assert trainer.args.seg_point_refine_o2o is False
    assert trainer.args.e2e_final_o2m == 0.3


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


def test_musgd_muon_update_handles_conv1d_weights():
    """MuSGD's muon path must accept 3D Conv1d weights (PointHeadMLP) without asserting."""
    from ultralytics.optim.muon import MuSGD, muon_update

    head = torch.nn.Conv1d(8, 4, kernel_size=1)
    # Mirror build_optimizer routing: only ndim >= 2 params enter the muon group, biases go to SGD.
    opt = MuSGD([
        {"params": [head.weight], "lr": 0.01, "use_muon": True,
         "momentum": 0.9, "nesterov": True, "weight_decay": 0.0},
        {"params": [head.bias], "lr": 0.01, "use_muon": False,
         "momentum": 0.9, "nesterov": True, "weight_decay": 0.0},
    ])
    out = head(torch.randn(2, 8, 16))
    out.sum().backward()
    opt.step()
    assert torch.isfinite(head.weight).all()

    grad = torch.randn(4, 8, 1)
    update = muon_update(grad, torch.zeros_like(grad))
    assert update.shape == (4, 8)
    assert torch.isfinite(update).all()


def test_setup_model_unwraps_distill_ckpt_without_distill_model(monkeypatch):
    """Finetuning from a teacher-stripped distillation ckpt without distill_model uses the bare student."""
    from ultralytics.engine import trainer as trainer_module
    from ultralytics.nn.tasks import SegmentationModel

    student = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80, verbose=False)
    wrapped = object.__new__(trainer_module.DistillationModel)
    torch.nn.Module.__init__(wrapped)
    wrapped.student_model = student
    wrapped.teacher_model = None

    captured = {}
    trainer = object.__new__(BaseTrainer)
    trainer.model = "yolo26n-seg-pointrend.yaml"
    trainer.args = SimpleNamespace(pretrained="distill_ckpt.pt", distill_model=None)
    trainer.resume = False

    def fake_get_model(cfg=None, weights=None, verbose=True):
        captured["cfg"] = cfg
        captured["weights"] = weights
        return SimpleNamespace()

    trainer.get_model = fake_get_model
    monkeypatch.setattr(trainer_module, "load_checkpoint", lambda path: (wrapped, {"checkpoint": True}))

    trainer.setup_model()

    assert captured["weights"] is student, "Student was not unwrapped from the distillation checkpoint"
    assert captured["cfg"] == "yolo26n-seg-pointrend.yaml"


def test_mask_point_coords_full_grid():
    """Uncertainty sampling returns coords in [0, 1]^2 on the full mask grid."""
    from ultralytics.utils.mask_point_sampling import get_uncertain_point_coords_with_randomness

    logits = torch.randn(2, 1, 32, 32)
    coords = get_uncertain_point_coords_with_randomness(
        logits, lambda x: -x.abs(), num_points=16, oversample_ratio=3, importance_sample_ratio=0.75
    )
    assert coords.shape == (2, 16, 2)
    assert (coords >= 0).all() and (coords <= 1).all()


def test_mask_point_coords_in_roi():
    """ROI uncertainty sampling confines points to each instance bbox (+margin), degenerate -> full grid."""
    from ultralytics.utils.mask_point_sampling import get_uncertain_point_coords_in_roi

    torch.manual_seed(0)
    n, num_points = 3, 24
    logits = torch.randn(n, 1, 32, 32)
    # Instance 0: tight box [0.2,0.2]-[0.5,0.5]; 1: wide box; 2: degenerate (x2<=x1) -> full-grid fallback.
    boxes = torch.tensor(
        [[0.20, 0.20, 0.50, 0.50], [0.10, 0.05, 0.90, 0.95], [0.50, 0.50, 0.50, 0.50]]
    )
    margin = 0.05
    coords = get_uncertain_point_coords_in_roi(
        logits, lambda x: -x.abs(), num_points, oversample_ratio=3, importance_sample_ratio=0.75,
        boxes_norm=boxes, margin=margin,
    )
    assert coords.shape == (n, num_points, 2)
    assert (coords >= 0).all() and (coords <= 1).all()
    # Expanded bbox bounds (clamped to [0,1]); all sampled points must lie inside them.
    x1 = (boxes[:, 0] - margin).clamp(0.0, 1.0)
    y1 = (boxes[:, 1] - margin).clamp(0.0, 1.0)
    x2 = (boxes[:, 2] + margin).clamp(0.0, 1.0)
    y2 = (boxes[:, 3] + margin).clamp(0.0, 1.0)
    for i in range(n):
        if x2[i] > x1[i] and y2[i] > y1[i]:  # non-degenerate: points must be inside expanded bbox
            assert (coords[i, :, 0] >= x1[i] - 1e-5).all() and (coords[i, :, 0] <= x2[i] + 1e-5).all()
            assert (coords[i, :, 1] >= y1[i] - 1e-5).all() and (coords[i, :, 1] <= y2[i] + 1e-5).all()


def test_mask_point_coords_weighted_in_roi():
    """Blended boundary-weighted ROI sampling over-represents the GT Sobel band but keeps interior
    points reachable; degenerate bbox -> full-grid fallback."""
    from ultralytics.utils.mask_point_sampling import get_uncertain_point_coords_in_roi

    torch.manual_seed(0)
    n, num_points, H, W = 2, 256, 32, 32
    logits = torch.randn(n, 1, H, W)
    # Weight map: a thin vertical boundary band at x ~= 0.5 (high weight in column band, ~0 elsewhere).
    weight = torch.zeros(n, H, W)
    band = (torch.arange(W).float() / W)
    in_band = (band >= 0.45) & (band <= 0.55)  # central ~10% columns
    weight[:] = in_band[None, None].float() * 10.0  # high weight only in the band columns
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])  # row 1 degenerate -> full-grid
    coords = get_uncertain_point_coords_in_roi(
        logits, lambda x: -x.abs(), num_points, oversample_ratio=1, importance_sample_ratio=0.5,
        boxes_norm=boxes, margin=0.0, weight_map=weight,
    )
    assert coords.shape == (n, num_points, 2)
    assert (coords >= 0).all() and (coords <= 1).all()
    # Non-degenerate row 0: oversample is a 50/50 blend (boundary-weighted + uniform-in-bbox) and the
    # remainder is uniform, so the band is over-represented versus the uniform baseline (~0.1) but
    # the interior is NOT crowded out — interior FP/FN regions stay reachable by the top-k selection.
    frac_in_band = ((coords[0, :, 0] >= 0.45) & (coords[0, :, 0] <= 0.55)).float().mean()
    assert frac_in_band > 0.2, f"boundary half should over-represent the band vs uniform ~0.1, got {frac_in_band}"
    # Interior coverage: the uniform half + uniform remainder must place some points OUTSIDE the band.
    assert ((coords[0, :, 0] < 0.45) | (coords[0, :, 0] > 0.55)).any(), "interior points must stay reachable"
    # Degenerate row 1: uniform full-grid fallback -> not confined to the (point) bbox, still in [0,1].
    assert (coords[1] >= 0).all() and (coords[1] <= 1).all()


def test_point_focal_dice_per_instance():
    """Per-instance focal and dice helpers return one scalar per instance."""
    from ultralytics.utils.mask_point_sampling import (
        point_dice_loss_per_instance,
        point_sigmoid_focal_loss_per_instance,
    )

    logits = torch.randn(3, 8, requires_grad=True)
    targets = torch.rand(3, 8)
    focal = point_sigmoid_focal_loss_per_instance(logits, targets)
    dice = point_dice_loss_per_instance(logits, targets)
    assert focal.shape == (3,)
    assert dice.shape == (3,)
    assert torch.isfinite(focal).all() and torch.isfinite(dice).all()
    focal.sum().backward()
    assert logits.grad is not None


def test_point_head_mlp_zero_init_is_coarse_residual():
    """Point-head MLP starts as an identity residual over coarse logits."""
    from ultralytics.nn.modules import PointHeadMLP

    point_head = PointHeadMLP(in_channels=4, hidden_channels=8)
    point_feats = torch.randn(2, 4, 5)
    coarse = torch.randn(2, 5)

    refined = point_head(point_feats, coarse)

    assert refined.shape == coarse.shape
    assert torch.allclose(refined, coarse)


def test_sobel_magnitude_constant_is_near_zero():
    """Constant masks have near-zero Sobel magnitude."""
    from ultralytics.utils.ops import sobel_magnitude

    x = torch.ones(2, 16, 16)
    mag = sobel_magnitude(x)
    assert mag.shape == (2, 16, 16)
    assert mag.max() < 1.1e-3


def test_single_mask_loss_all_gains_disabled_matches_legacy():
    """seg_point=0 must match the original dense BCE-only mask loss."""
    from ultralytics.utils.loss import v8SegmentationLoss

    torch.manual_seed(0)
    n, h, w, c = 4, 40, 40, 32
    gt = (torch.rand(n, h, w) > 0.5).float()
    pred = torch.randn(n, c, requires_grad=True)
    proto = torch.randn(c, h, w)
    xyxy = torch.tensor(
        [[5.0, 5.0, 30.0, 30.0], [2.0, 2.0, 20.0, 25.0], [10.0, 10.0, 35.0, 38.0], [0.0, 0.0, 15.0, 15.0]]
    )
    area = torch.tensor([0.25, 0.15, 0.2, 0.1])

    pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
    loss_map = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, gt, reduction="none")
    from ultralytics.utils.ops import crop_mask

    legacy = (crop_mask(loss_map, xyxy).mean(dim=(1, 2)) / area).sum()
    current = v8SegmentationLoss.single_mask_loss(
        gt, pred, proto, xyxy, area, comp_w=0.0, bnd_w=0.0, point_w=0.0
    )
    assert torch.allclose(legacy, current)
    with_comp = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, comp_w=1.0)
    with_bnd = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, bnd_w=1.0)
    with_point = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, point_w=1.0)
    from ultralytics.nn.modules import PointHeadMLP

    point_head = PointHeadMLP(in_channels=c, hidden_channels=16)
    point_feats = torch.randn(n, c, h // 2, w // 2)
    torch.manual_seed(1)
    point_lite = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, point_w=1.0)
    torch.manual_seed(1)
    point_refine = v8SegmentationLoss.single_mask_loss(
        gt,
        pred,
        proto,
        xyxy,
        area,
        point_w=1.0,
        point_head=point_head,
        point_feats=point_feats,
    )
    assert torch.allclose(point_lite, point_refine)
    # Shared per-image (1, C, H, W) feats must match the per-instance expanded path (merged-coords sampler).
    shared_feats = torch.randn(1, c, h // 2, w // 2)
    torch.manual_seed(1)
    point_shared = v8SegmentationLoss.single_mask_loss(
        gt, pred, proto, xyxy, area, point_w=1.0, point_head=point_head, point_feats=shared_feats
    )
    torch.manual_seed(1)
    point_expanded = v8SegmentationLoss.single_mask_loss(
        gt, pred, proto, xyxy, area, point_w=1.0, point_head=point_head,
        point_feats=shared_feats.expand(n, -1, -1, -1),
    )
    assert torch.allclose(point_shared, point_expanded, atol=1e-6)
    for v in (with_comp, with_bnd, with_point):
        assert torch.isfinite(v)
        assert v >= legacy
    point_zero = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, point_w=1.0, num_points=0)
    assert torch.allclose(point_zero, legacy)
    (with_comp + with_bnd + with_point + point_refine).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert point_head.mlp[-1].weight.grad is not None


def test_segmentation_loss_optional_hyp_injection_is_finite():
    """calculate_segmentation_loss should pass seg_comp/seg_bnd/seg_point gains into single_mask_loss."""
    from ultralytics.utils.loss import v8SegmentationLoss

    torch.manual_seed(2)
    criterion = object.__new__(v8SegmentationLoss)
    criterion.overlap = True
    criterion.hyp = {
        "seg_comp": 1.0,
        "seg_bnd": 1.0,
        "seg_point": 1.0,
        "seg_point_num": 8,
        "seg_point_oversample": 2,
        "seg_point_importance": 0.75,
    }
    batch_size, anchors, channels, h, w = 1, 2, 32, 16, 16
    fg_mask = torch.tensor([[True, False]])
    target_gt_idx = torch.zeros(batch_size, anchors, dtype=torch.long)
    target_bboxes = torch.tensor([[[2.0, 2.0, 12.0, 12.0], [0.0, 0.0, 0.0, 0.0]]])
    masks = torch.zeros(batch_size, h, w)
    masks[:, 2:12, 2:12] = 1.0
    proto = torch.randn(batch_size, channels, h, w)
    pred_masks = torch.randn(batch_size, anchors, channels, requires_grad=True)
    imgsz = torch.tensor([h, w], dtype=torch.float32)

    loss = criterion.calculate_segmentation_loss(
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        torch.zeros(1, 1),
        proto,
        pred_masks,
        imgsz,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert pred_masks.grad is not None
    assert torch.isfinite(pred_masks.grad).all()


def test_segmentation_loss_boundary_roi_path_is_finite():
    """seg_point_boundary=True forces ROI on and biases sampling toward the GT Sobel band."""
    from ultralytics.utils.loss import v8SegmentationLoss

    torch.manual_seed(3)
    criterion = object.__new__(v8SegmentationLoss)
    criterion.overlap = True
    criterion.hyp = {
        "seg_point": 1.0,
        "seg_point_num": 16,
        "seg_point_oversample": 2,
        "seg_point_importance": 0.75,
        "seg_point_roi": -1.0,  # legacy full-grid by itself...
        "seg_point_boundary": True,  # ...but boundary_w forces ROI on (eff margin = 0).
    }
    batch_size, anchors, channels, h, w = 1, 2, 32, 16, 16
    fg_mask = torch.tensor([[True, False]])
    target_gt_idx = torch.zeros(batch_size, anchors, dtype=torch.long)
    target_bboxes = torch.tensor([[[2.0, 2.0, 12.0, 12.0], [0.0, 0.0, 0.0, 0.0]]])
    masks = torch.zeros(batch_size, h, w)
    masks[:, 4:12, 4:12] = 1.0  # block region -> Sobel responds at the 4/12 boundary (inside bbox)
    proto = torch.randn(batch_size, channels, h, w)
    pred_masks = torch.randn(batch_size, anchors, channels, requires_grad=True)
    imgsz = torch.tensor([h, w], dtype=torch.float32)

    loss = criterion.calculate_segmentation_loss(
        fg_mask, masks, target_gt_idx, target_bboxes, torch.zeros(1, 1), proto, pred_masks, imgsz
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert pred_masks.grad is not None
    assert torch.isfinite(pred_masks.grad).all()


def test_segmentation_loss_e2e_point_branch_controls():
    """E2E one2one can downweight point loss or force Lite point logits independently."""
    from ultralytics.utils.loss import v8SegmentationLoss

    criterion = object.__new__(v8SegmentationLoss)
    criterion.hyp = {
        "seg_point": 0.5,
        "seg_point_o2o": 0.0,
        "seg_point_refine": True,
        "seg_point_refine_o2o": False,
    }
    criterion.loss_branch = "one2many"
    assert criterion._point_loss_gain() == 0.5
    assert criterion._point_refine_enabled() is True

    criterion.loss_branch = "one2one"
    assert criterion._point_loss_gain() == 0.0
    assert criterion._point_refine_enabled() is False


def test_segment26_forward_point_refine_dummy_and_detach_contract():
    """Segment26 training forward should expose DDP dummy while keeping one2one features/proto detached."""
    from ultralytics.nn.tasks import SegmentationModel

    model = SegmentationModel("yolo26n-seg-pointrend.yaml", ch=3, nc=80, verbose=False)
    model.train()
    preds = model(torch.randn(1, 3, 64, 64))

    assert set(preds) >= {"one2many", "one2one"}
    o2m, o2o = preds["one2many"], preds["one2one"]
    o2m_proto = o2m["proto"] if isinstance(o2m["proto"], tuple) else (o2m["proto"],)
    o2o_proto = o2o["proto"] if isinstance(o2o["proto"], tuple) else (o2o["proto"],)
    assert o2m["feats"][0].requires_grad is True
    assert o2o["feats"][0].requires_grad is False
    assert all(p.requires_grad for p in o2m_proto)
    assert not any(p.requires_grad for p in o2o_proto)
    assert "point_refine_dummy" in o2m and "point_refine_dummy" in o2o
    assert o2m["point_refine_dummy"] is o2o["point_refine_dummy"]
    assert o2m["point_refine_dummy"].requires_grad is True
    assert float(o2m["point_refine_dummy"].detach()) == 0.0

    model.eval()
    eval_outputs = model(torch.randn(1, 3, 64, 64))
    eval_preds = eval_outputs[1]
    assert "point_refine_dummy" not in eval_preds["one2many"]
    assert "point_refine_dummy" not in eval_preds["one2one"]


def test_e2e_point_refine_backward_branch_gradient_routes():
    """one2many should shape backbone/proto, while one2one stays on detached features/proto."""
    from ultralytics.nn.tasks import SegmentationModel

    def grad_sum(parameters) -> float:
        return sum(float(p.grad.detach().abs().sum()) for p in parameters if p.grad is not None)

    def make_batch(imgsz: int = 128) -> dict[str, torch.Tensor]:
        mask_hw = imgsz // 4
        masks = torch.zeros(2, mask_hw, mask_hw)
        masks[:, mask_hw // 4 : 3 * mask_hw // 4, mask_hw // 4 : 3 * mask_hw // 4] = 1.0
        sem_masks = torch.zeros(2, mask_hw, mask_hw, dtype=torch.long)
        sem_masks[:, mask_hw // 4 : 3 * mask_hw // 4, mask_hw // 4 : 3 * mask_hw // 4] = 1
        return {
            "img": torch.rand(2, 3, imgsz, imgsz),
            "batch_idx": torch.arange(2, dtype=torch.float32),
            "cls": torch.zeros(2),
            "bboxes": torch.tensor([[0.5, 0.5, 0.45, 0.45], [0.5, 0.5, 0.45, 0.45]]),
            "masks": masks,
            "sem_masks": sem_masks,
        }

    model = SegmentationModel("yolo26n-seg-pointrend.yaml", ch=3, nc=80, verbose=False).train()
    args = get_cfg(DEFAULT_CFG)
    args.overlap_mask = False
    args.seg_point = 1.0
    args.seg_point_refine = True
    args.seg_point_num = 16
    args.seg_point_oversample = 2
    args.seg_point_boundary = True
    args.seg_bnd = 0.1
    args.seg_point_o2o = 1.0
    args.seg_point_refine_o2o = True
    model.args = args
    criterion = model.init_criterion()
    batch = make_batch()
    head = model.model[-1]
    backbone_params = tuple(model.model[0].parameters())
    proto_params = tuple(head.proto.parameters())
    o2m_mask_params = tuple(head.cv4.parameters())
    o2o_mask_params = tuple(head.one2one_cv4.parameters())
    point_params = tuple(head.point_head.parameters())

    preds = model(batch["img"])
    model.zero_grad(set_to_none=True)
    criterion.one2many.loss(preds["one2many"], batch)[0].sum().backward()
    assert grad_sum(backbone_params) > 0.0
    assert grad_sum(proto_params) > 0.0
    assert grad_sum(o2m_mask_params) > 0.0
    assert grad_sum(o2o_mask_params) == 0.0
    assert grad_sum(point_params) > 0.0

    preds = model(batch["img"])
    model.zero_grad(set_to_none=True)
    criterion.one2one.loss(preds["one2one"], batch)[0].sum().backward()
    assert grad_sum(backbone_params) == 0.0
    assert grad_sum(proto_params) == 0.0
    assert grad_sum(o2m_mask_params) == 0.0
    assert grad_sum(o2o_mask_params) > 0.0
    assert grad_sum(point_params) > 0.0


def test_process_mask_pointrend_basic():
    """process_mask_pointrend returns localized binary masks at the target shape for K=1 and K=3."""
    from ultralytics.nn.modules import PointHeadMLP
    from ultralytics.utils.ops import process_mask, process_mask_pointrend

    torch.manual_seed(0)
    n, md, mh, mw = 2, 32, 16, 16
    proto = torch.randn(md, mh, mw)
    masks_in = torch.randn(n, md)
    bboxes = torch.tensor([[2.0, 2.0, 30.0, 30.0], [4.0, 4.0, 28.0, 28.0]])
    shape = (32, 32)
    feats = torch.randn(1, 64, 8, 8)  # P3 fine feature (bilinearly sampled at point coords)
    baseline = process_mask(proto, masks_in, bboxes, shape, upsample=True)
    for subdivisions in (1, 3):
        for zero_init in (True, False):
            head = PointHeadMLP(in_channels=64, hidden_channels=16)
            if not zero_init:
                for p in head.parameters():
                    p.data.normal_(0, 0.1)
            masks = process_mask_pointrend(
                proto, masks_in, bboxes, shape, head, feats, num_points=16,
                oversample_ratio=2, importance_ratio=0.75, subdivisions=subdivisions,
            )
            assert masks.shape == (n, *shape), (masks.shape, subdivisions, zero_init)
            assert masks.dtype == torch.uint8
            assert masks.ge(0).all() and masks.le(1).all()  # binary {0, 1}
            if zero_init:
                assert torch.equal(masks, baseline), "zero-init PointRend inference must be a no-op vs process_mask"
            else:
                # ROI delta scatter must not introduce extra changes outside each predicted bbox.
                changed = masks != baseline
                for i, (x1, y1, x2, y2) in enumerate(bboxes.int().tolist()):
                    outside = (
                        changed[i, :y1, :].sum()
                        + changed[i, y2:, :].sum()
                        + changed[i, :, :x1].sum()
                        + changed[i, :, x2:].sum()
                    )
                    assert outside == 0, (i, outside)
    # empty detections -> (0, H, W) without crashing
    empty = process_mask_pointrend(proto, torch.zeros(0, md), torch.zeros(0, 4), shape, head, feats, subdivisions=3)
    assert empty.shape == (0, *shape)


def test_pointrend_infer_subdivision_smoke(tmp_path):
    """seg_point_refine_infer=True routes the predictor through process_mask_pointrend (PyTorch-only)."""
    import numpy as np

    from ultralytics import YOLO

    torch.manual_seed(0)
    model = YOLO("yolo26n-seg-pointrend.yaml")
    img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    on = model.predict(
        img, imgsz=96, conf=0.0, seg_point_refine_infer=True, retina_masks=False, verbose=False, save=False
    )[0]
    assert on.masks is not None, "subdivision path must produce masks for end2end top-k detections"
    assert on.masks.data.shape[0] == on.boxes.data.shape[0]
    assert on.masks.data.shape[1:] == (96, 96)
    # default-off path still produces masks (standard process_mask) and the same detection count
    off = model.predict(img, imgsz=96, conf=0.0, verbose=False, save=False)[0]
    assert off.masks is not None and off.masks.data.shape[0] == off.boxes.data.shape[0]


def test_segmentation_validator_pointrend_postprocess():
    """seg_point_refine_infer=True routes SegmentationValidator.postprocess through process_mask_pointrend.

    The default validation path keeps legacy proto-resolution masks; enabling PointRend validation compares
    full-resolution refined masks and prepares GT masks at the same resolution.
    """
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.ops import process_mask

    torch.manual_seed(0)
    model = SegmentationModel("yolo26n-seg-pointrend.yaml", ch=3, nc=2, verbose=False)
    model.eval()
    with torch.no_grad():
        raw = model(torch.randn(1, 3, 96, 96))
    # Segment26 eval return: ((det, proto), feats_dict) for the PyTorch backend.
    assert isinstance(raw, (list, tuple)) and isinstance(raw[0], tuple)
    assert isinstance(raw[1], dict)

    base = dict(
        conf=0.0,
        iou=0.7,
        max_det=300,
        imgsz=96,
        save_json=False,
        save_txt=False,
        task="segment",
        seg_point_num=8,
        seg_point_oversample=1,
        seg_point_importance=1.0,
        seg_point_subdiv_k=1,
        seg_point_roi=0.0,
    )

    def make(infer):
        v = SegmentationValidator(args=get_cfg(DEFAULT_CFG, overrides={**base, "seg_point_refine_infer": infer}))
        v.data = {}
        v.init_metrics(model)  # resolves self.point_head via _resolve_point_head and sets self.process
        return v

    v_on = make(True)
    assert v_on.point_head is not None, "init_metrics must resolve the Segment26 point head on a raw model"
    assert v_on.process is process_mask  # non-native path -> subdivision eligible
    preds_on = v_on.postprocess(raw)
    masks_on = preds_on[0]["masks"]
    assert masks_on.shape[0] > 0, "test needs at least one detection to exercise the mask branch"
    assert masks_on.shape[1:] == (96, 96)

    v_off = make(False)
    assert v_off._uses_pointrend() is False
    masks_off = v_off.postprocess(raw)[0]["masks"]
    assert masks_off.shape[0] == masks_on.shape[0]
    assert masks_off.shape[1:] == (24, 24)

    batch = {
        "img": torch.zeros(1, 3, 96, 96),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "ori_shape": [(96, 96)],
        "ratio_pad": [((1.0, 1.0), (0.0, 0.0))],
        "im_file": ["synthetic.jpg"],
        "masks": torch.ones(1, 96, 96),
    }
    assert v_on._prepare_batch(0, batch)["masks"].shape[1:] == (96, 96)
    assert v_off._prepare_batch(0, batch)["masks"].shape[1:] == (24, 24)


def test_segmentation_loss_cfg_overrides_are_accepted():
    """New optional seg loss and DALI args should pass cfg type validation."""
    args = get_cfg(
        DEFAULT_CFG,
        overrides={
            "dali": True,
            "seg_comp": 1,
            "seg_bnd": 0.5,
            "seg_point": 1.0,
            "seg_point_num": 64,
            "seg_point_oversample": 2,
            "seg_point_importance": 0.5,
            "seg_point_refine": True,
            "seg_point_o2o": 0.25,
            "seg_point_refine_o2o": False,
            "seg_point_roi": 0.1,
            "seg_point_boundary": True,
            "seg_point_refine_infer": True,
            "seg_point_subdiv_k": 2,
            "e2e_final_o2m": 0.3,
        },
    )

    assert args.dali is True
    assert args.seg_comp == 1.0
    assert args.seg_bnd == 0.5
    assert args.seg_point == 1.0
    assert args.seg_point_num == 64
    assert args.seg_point_oversample == 2
    assert args.seg_point_importance == 0.5
    assert args.seg_point_refine is True
    assert args.seg_point_o2o == 0.25
    assert args.seg_point_refine_o2o is False
    assert args.seg_point_roi == 0.1
    assert args.seg_point_boundary is True
    assert args.seg_point_refine_infer is True
    assert args.seg_point_subdiv_k == 2
    assert args.e2e_final_o2m == 0.3


def test_yolo26_pointrend_yaml_builds_optional_point_head():
    """The PointRend YAML builds Segment26 with a point head without changing the legacy YAML."""
    from ultralytics.nn.tasks import SegmentationModel

    model = SegmentationModel("yolo26n-seg-pointrend.yaml", ch=3, nc=80, verbose=False)
    legacy = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80, verbose=False)

    assert model.model[-1].point_head is not None
    assert model.model[-1].point_head.hidden_channels == 32
    assert legacy.model[-1].point_head is None


def test_model_load_unwraps_distillation_student_checkpoint():
    """Loading a distillation-wrapped checkpoint into a new model should use the student weights."""
    from ultralytics.nn.tasks import SegmentationModel

    source = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80, verbose=False)
    target = SegmentationModel("yolo26n-seg-pointrend.yaml", ch=3, nc=80, verbose=False)
    wrapper = torch.nn.Module()
    wrapper.student_model = source
    with torch.no_grad():
        source.model[0].conv.weight.fill_(0.123)
        target.model[0].conv.weight.zero_()

    target.load({"model": wrapper}, verbose=False)

    assert torch.allclose(target.model[0].conv.weight, source.model[0].conv.weight)
    assert target.model[-1].point_head is not None


def test_preprocess_batch_normalizes_dali_images_like_standard_images():
    """DALI decode/resize tensors still need the same /255 normalization as the CPU path."""
    trainer = object.__new__(detect.DetectionTrainer)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(multi_scale=0.0)
    batch = {"img": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8), "dali": True}

    out = trainer.preprocess_batch(batch)

    assert "dali" not in out
    assert out["img"].dtype == torch.float32
    assert torch.allclose(out["img"], torch.ones_like(out["img"]))
