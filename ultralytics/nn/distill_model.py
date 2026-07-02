# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import copy_attr

from .tasks import load_checkpoint


class FeatureHook:
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        """Initialize the hook with the shared feature dict and the layer index to store outputs under."""
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        """Store the layer's forward output into the shared feature dict under its index.

        The output is a tensor for neck layers but a tuple/dict for the Detect head, so it is left untyped.
        """
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """YOLO knowledge distillation model.

    This class wraps a teacher-student pair for knowledge distillation training. Features are extracted from both models
    via forward hooks for distillation.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        feats_idx (list): Layer indices for feature extraction.
        projector (nn.ModuleList): MLP projector aligning student features to teacher dimensions.
        dis (float): Distillation loss weight factor.

    Methods:
        get_distill_layers: Auto-detect distillation feature layers from the Detect head.
        forward: Run the student model, or compute the combined loss when given a training batch.
        loss: Compute combined detection and distillation loss.
        loss_sl2: Compute score-weighted L2 distillation loss for a feature pair.
        decouple_outputs: Normalize teacher/student head outputs across train/val formats.
        train: Set training mode while keeping teacher frozen.

    Examples:
        Train a student model with knowledge distillation from a larger teacher (the trainer builds the
        DistillationModel internally when the ``distill_model`` argument is set)
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="coco8.yaml", distill_model="yolo26s.pt")
    """

    def __init__(self, teacher_model: str | Path | nn.Module, student_model: nn.Module):
        """Initialize the distillation model with teacher, student, and feature extraction hooks.

        Args:
            teacher_model (str | Path | nn.Module): Teacher model checkpoint path or module.
            student_model (nn.Module): Student model module to be trained.
        """
        super().__init__()
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.feats_idx = self.get_distill_layers(student_model)

        # Hook-based feature capture: identical for teacher and student
        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()
        self.teacher_class_indices = self._resolve_teacher_class_indices()

        # Get feature dimensions via dummy forward pass (hooks capture outputs)
        imgsz = student_model.args.imgsz
        student_model.eval()
        with torch.no_grad():
            self._forward_teacher_for_distillation(torch.zeros(2, 3, imgsz, imgsz).to(device))
            student_model(torch.zeros(2, 3, imgsz, imgsz).to(device))
        student_model.train()
        teacher_output = [self._teacher_feats[idx] for idx in self.feats_idx]
        student_output = [self._student_feats[idx] for idx in self.feats_idx]

        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis
        self.dis_proto = float(getattr(self.student_model.args, "dis_proto", 0.0) or 0.0)
        self.distill_loss_clip = float(getattr(self.student_model.args, "distill_loss_clip", 10.0) or 0.0)
        self.distill_warmup_factor = 1.0
        self._proto_channel_mismatch_warned = False
        projectors = []
        for student_out, teacher_out in zip(student_output[:-1], teacher_output[:-1]):
            student_dim = self.decouple_outputs(student_out).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out).shape[1]
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)

    def set_distill_warmup_factor(self, factor: float) -> None:
        """Set the current distillation warmup factor in [0, 1]."""
        self.distill_warmup_factor = max(0.0, min(1.0, float(factor)))

    def sanitize_distill_loss(self, loss: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """Replace non-finite distillation loss values and optionally cap abnormal spikes before gradients."""
        clip = getattr(self, "distill_loss_clip", None)
        student_args = getattr(self.student_model, "args", None)
        if isinstance(student_args, dict):
            clip = student_args.get("distill_loss_clip", clip)
        elif student_args is not None:
            clip = getattr(student_args, "distill_loss_clip", clip)
        clip = float(10.0 if clip is None else clip or 0.0)
        posinf = clip if clip > 0 else 0.0
        loss = torch.nan_to_num(loss, nan=0.0, posinf=posinf, neginf=0.0).clamp_min(0.0)
        return loss.clamp(max=clip) if clamp and clip > 0 else loss

    @staticmethod
    def _normalize_names(names) -> dict[int, str]:
        """Normalize YOLO class names stored as a dict/list into a lowercase lookup."""
        if isinstance(names, dict):
            items = names.items()
        elif isinstance(names, (list, tuple)):
            items = enumerate(names)
        else:
            items = []
        return {int(k): str(v).strip().lower() for k, v in items}

    def _resolve_teacher_class_indices(self) -> torch.Tensor | None:
        """Return teacher class indices that overlap the student names, or None to use every teacher class."""
        if self.teacher_model is None:
            return None
        teacher_names = self._normalize_names(getattr(self.teacher_model, "names", {}))
        student_names = self._normalize_names(getattr(self.student_model, "names", {}))
        if not teacher_names or not student_names:
            return None
        student_name_set = set(student_names.values())
        indices = [idx for idx, name in teacher_names.items() if name in student_name_set]
        matched_names = {teacher_names[idx] for idx in indices}
        unmatched_students = sorted(student_name_set - matched_names)
        if unmatched_students:
            # Matching is exact lowercase string comparison, so synonyms/formatting differences drop silently.
            # Surface the dropped classes so a bad names mapping is caught before a long training run.
            LOGGER.warning(
                f"Distillation: {len(unmatched_students)} student classes have no teacher match and receive no "
                f"distillation signal from teacher scores: {unmatched_students}"
            )
        if not indices or len(indices) == len(teacher_names):
            return None
        device = next(self.student_model.parameters()).device
        LOGGER.info(
            f"Distillation: using {len(indices)}/{len(teacher_names)} teacher classes that match student names."
        )
        return torch.tensor(indices, device=device, dtype=torch.long)

    def __getstate__(self):
        """Return a copy of state for pickling without captured features or hook handles.

        Clears the feature dicts in place (rather than replacing the attributes) because the registered
        FeatureHooks share these exact dict objects; otherwise a deepcopy/pickle of a mid-training model would
        still reach the hook-held tensors (which carry grad_fn and cannot be deep-copied).
        """
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        """Clear stale features and hooks, and re-register forward hooks after unpickling."""
        self.__dict__.update(state)
        self.dis_proto = float(getattr(self, "dis_proto", 0.0) or 0.0)
        self.distill_loss_clip = float(getattr(self, "distill_loss_clip", 10.0) or 0.0)
        self.distill_warmup_factor = float(getattr(self, "distill_warmup_factor", 1.0))
        self._proto_channel_mismatch_warned = bool(getattr(self, "_proto_channel_mismatch_warned", False))
        if not hasattr(self, "teacher_class_indices"):
            self.teacher_class_indices = self._resolve_teacher_class_indices()
        self._teacher_feats = {}
        self._student_feats = {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self) -> None:
        """Remove any previously registered feature-capture hooks."""
        for handle in self._student_hooks:
            handle.remove()
        self._student_hooks.clear()
        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()
            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        """Remove any FeatureHook instances from a module's forward hooks."""
        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    def _register_feature_hooks(self) -> None:
        """Register feature-capture hooks, removing stale FeatureHook instances first."""
        self._remove_feature_hooks()
        for idx in self.feats_idx:
            self._clear_feature_hooks(self.student_model.model[idx])
            self._student_hooks.append(
                self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))
            )
            if self.teacher_model is not None:
                self._clear_feature_hooks(self.teacher_model.model[idx])
                self._teacher_hooks.append(
                    self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
                )

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        """Auto-detect distillation feature layers from the model's Detect head.

        Returns the Detect head's input layer indices plus the head layer index itself.
        E.g. YOLO26 -> [16, 19, 22, 23], YOLOv8 -> [15, 18, 21, 22].
        """
        for m in model.model:
            if isinstance(m, Detect):
                return [*list(m.f), m.i]
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self):
        """Keep teacher fixed for distillation."""
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """Set model train mode while keeping teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def _forward_teacher_for_distillation(self, imgs: torch.Tensor) -> None:
        """Run the frozen teacher with raw Detect/Segment head outputs for feature distillation."""
        teacher_head = self.teacher_model.model[self.feats_idx[-1]]
        head_was_training = teacher_head.training
        try:
            # Only the head's own flag is toggled. Child modules stay eval, so BN stats are not updated.
            # This makes Detect/Segment return raw train-format preds/proto and skips inference postprocess/topk,
            # the path that previously triggered a CUDA launch timeout during teacher distillation.
            # Segment proto submodules remain eval; extract_proto handles both eval tensor and train tuple outputs.
            teacher_head.training = True
            with torch.no_grad():
                self.teacher_model(imgs)
        finally:
            teacher_head.training = head_was_training

    def forward(self, x, *args, **kwargs):
        """Forward pass through the student model."""
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        """Compute loss.

        The distillation part is returned as two components (feature distillation and proto distillation) so they
        can be monitored independently as `dis_feat`/`dis_proto` in the training logs.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not self.training:  # for loss calculation during validation while training
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
            # validator does not compute distillation losses; keep zero placeholders for shape consistency
            loss_distill = torch.zeros(2, device=batch["img"].device)
            return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, loss_distill])

        # Clear feature dicts before forward passes
        self._teacher_feats.clear()
        self._student_feats.clear()

        self._forward_teacher_for_distillation(batch["img"])  # hooks capture teacher features
        preds = self.student_model(batch["img"])  # hooks capture student features

        regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
        teacher_head_feat = self._teacher_feats[self.feats_idx[-1]]
        student_head_feat = self._student_feats[self.feats_idx[-1]]
        teacher_scores = (
            self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
            + self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
        ) / 2
        if self.teacher_class_indices is not None:
            teacher_scores = teacher_scores.index_select(1, self.teacher_class_indices.to(teacher_scores.device))
        # neck feature sizes vary per batch (e.g. multi_scale), so split scores by the live teacher feats
        neck_feats = [self._teacher_feats[idx] for idx in self.feats_idx[:-1]]
        parts = torch.split(teacher_scores, [f.shape[-2] * f.shape[-1] for f in neck_feats], dim=-1)
        teacher_scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        loss_feat = torch.zeros(1, device=batch["img"].device)
        loss_proto = torch.zeros(1, device=batch["img"].device)
        for i, feat_idx in enumerate(self.feats_idx[:-1]):
            teacher_feat = self.decouple_outputs(self._teacher_feats[feat_idx])
            student_feat = self.projector[i](self.decouple_outputs(self._student_feats[feat_idx]))
            loss_feat += self.sanitize_distill_loss(
                self.loss_sl2(student_feat, teacher_feat, feat_idx=i, teacher_scores=teacher_scores) * self.dis,
                clamp=False,
            )
        if self.dis_proto > 0:
            loss_proto += self.sanitize_distill_loss(
                self.loss_proto(student_head_feat, teacher_head_feat, teacher_scores) * self.dis_proto,
                clamp=False,
            )

        loss_distill = torch.cat(
            [
                self.sanitize_distill_loss(loss_feat * self.distill_warmup_factor),
                self.sanitize_distill_loss(loss_proto * self.distill_warmup_factor),
            ]
        )
        distill_loss_detach = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, distill_loss_detach])

    def loss_sl2(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
    ) -> torch.Tensor:
        """Compute score-weighted L2 distillation loss for a feature pair.

        Args:
            student_feat (torch.Tensor): Student feature tensor of shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature tensor of shape (N, C, H, W).
            feat_idx (int): Index of the feature level for selecting teacher scores.
            teacher_scores (tuple): Tuple of score tensors for each feature level.

        Returns:
            (torch.Tensor): The computed score-weighted L2 loss.
        """
        teacher_score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)
        return weighted_mse

    def extract_proto(self, preds, branch: str = "one2many") -> torch.Tensor | None:
        """Extract the mask prototype tensor from segmentation head outputs."""
        preds = self.decouple_outputs(preds, branch=branch)
        if not isinstance(preds, dict):
            return None
        proto = preds.get("proto")
        if isinstance(proto, (list, tuple)):
            proto = proto[0] if proto else None
        return proto if isinstance(proto, torch.Tensor) else None

    @staticmethod
    def normalize_proto(proto: torch.Tensor) -> torch.Tensor:
        """Normalize proto maps per sample/channel before distillation."""
        proto = proto.float()
        mean = proto.mean(dim=(-2, -1), keepdim=True)
        std = proto.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (proto - mean) / std

    def loss_proto(
        self, student_head_feat, teacher_head_feat, teacher_scores: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute foreground-weighted prototype distillation loss for segmentation models."""
        teacher_proto = self.extract_proto(teacher_head_feat, branch="one2many")
        student_proto = self.extract_proto(student_head_feat, branch="one2many")
        if teacher_proto is None or student_proto is None:
            return torch.zeros(1, device=next(self.student_model.parameters()).device)
        if student_proto.shape[-2:] != teacher_proto.shape[-2:]:
            student_proto = F.interpolate(
                student_proto, size=teacher_proto.shape[-2:], mode="bilinear", align_corners=False
            )
        if student_proto.shape[1] != teacher_proto.shape[1]:
            if not getattr(self, "_proto_channel_mismatch_warned", False):
                LOGGER.warning(
                    "Proto distillation disabled: student/teacher proto channel mismatch "
                    f"({student_proto.shape[1]} vs {teacher_proto.shape[1]})."
                )
                self._proto_channel_mismatch_warned = True
            return torch.zeros(1, device=student_proto.device)

        student_proto = self.normalize_proto(student_proto)
        teacher_proto = self.normalize_proto(teacher_proto.detach())
        mse = F.mse_loss(student_proto, teacher_proto, reduction="none")

        weight = None
        if teacher_scores:
            p3_feat = self._teacher_feats[self.feats_idx[0]]
            score = teacher_scores[0].view(p3_feat.shape[0], 1, p3_feat.shape[-2], p3_feat.shape[-1])
            weight = F.interpolate(score.float(), size=teacher_proto.shape[-2:], mode="bilinear", align_corners=False)
            weight = 1.0 + weight
        if weight is None:
            return mse.mean()
        return (mse * weight).sum() / (weight.sum() * mse.shape[1] + 1e-9)

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Set value for student criterion."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        """Expose student end-to-end mode for validator/predictor control."""
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Forward end-to-end mode update to the student model."""
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        """Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model."""
        self.student_model.set_head_attr(**kwargs)

    def decouple_outputs(self, preds, branch: str = "one2one"):
        """Decouple outputs for teacher/student models.

        This method handles different output formats from YOLO models, including
        tuple outputs (train/val mode), dict outputs with branches (one2one/one2many),
        and direct tensor outputs.

        Args:
            preds (torch.Tensor | tuple | dict): Model predictions in various formats.
            branch (str): Which branch to extract from dict outputs ("one2one" or "one2many").

        Returns:
            (torch.Tensor | dict): The decoupled predictions.
        """
        if isinstance(preds, tuple):  # decouple for val mode
            preds = preds[1]
        if isinstance(preds, dict):
            if branch in preds:
                preds = preds[branch]
        return preds
