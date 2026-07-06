# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Differentiable boundary losses for segmentation masks.

These helpers provide the torch-only Sobel boundary term used as the optional
``seg_bnd`` sub-gain in ``v8SegmentationLoss.single_mask_loss``. The default
configuration keeps the term disabled, so old checkpoints resume through the
legacy dense BCE path until the gain is explicitly enabled.
"""

from __future__ import annotations

import torch

from ultralytics.utils.ops import crop_mask, sobel_magnitude

__all__ = ["boundary_l2_loss", "boundary_l2_loss_per_instance", "sobel_magnitude"]


def _as_single_channel(x: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    """Return ``x`` as (N, 1, H, W) float32 and whether a channel dim was added."""
    if x.ndim == 3:
        return x.float().unsqueeze(1), True
    if x.ndim == 4 and x.shape[1] == 1:
        return x.float(), False
    raise ValueError(f"{name} must have shape (N,H,W) or (N,1,H,W), got {tuple(x.shape)}")


def boundary_l2_loss_per_instance(
    pred: torch.Tensor,
    target: torch.Tensor,
    boxes: torch.Tensor | None = None,
    *,
    from_logits: bool = True,
    band_weight: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-instance Sobel L2 boundary alignment loss.

    Args:
        pred (torch.Tensor): Predicted mask logits/probabilities, shape (N,H,W) or (N,1,H,W).
        target (torch.Tensor): Target mask values, shape matching ``pred``.
        boxes (torch.Tensor | None): Optional xyxy boxes in mask pixel coordinates, shape (N,4).
        from_logits (bool): Apply sigmoid to ``pred`` before boundary extraction.
        band_weight (bool): If True, weight squared error by the GT Sobel magnitude.
        eps (float): Numerical stabilizer for divisions.

    Returns:
        (torch.Tensor): Per-instance loss, shape (N,).
    """
    pred4, _ = _as_single_channel(pred, "pred")
    target4, _ = _as_single_channel(target, "target")
    if pred4.shape != target4.shape:
        raise ValueError(f"pred and target shapes must match, got {tuple(pred4.shape)} and {tuple(target4.shape)}")

    prob = pred4.sigmoid() if from_logits else pred4
    gp = sobel_magnitude(prob)
    gg = sobel_magnitude(target4).detach()
    diff = (gp - gg).square()

    if boxes is not None:
        if boxes.shape != (diff.shape[0], 4):
            raise ValueError(f"boxes must have shape ({diff.shape[0]},4), got {tuple(boxes.shape)}")
        diff = crop_mask(diff.clone(), boxes)
        gg = crop_mask(gg.clone(), boxes)

    if band_weight:
        weight = gg
        return (diff * weight).sum(dim=(1, 2)) / (weight.sum(dim=(1, 2)) + eps)

    if boxes is None:
        denom = diff.new_full((diff.shape[0],), float(diff.shape[1] * diff.shape[2]))
    else:
        denom = crop_mask(torch.ones_like(diff), boxes).sum(dim=(1, 2))
    return diff.sum(dim=(1, 2)) / (denom + eps)


def boundary_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    boxes: torch.Tensor | None = None,
    *,
    from_logits: bool = True,
    band_weight: bool = True,
    reduction: str = "sum",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute reduced Sobel L2 boundary alignment loss."""
    loss = boundary_l2_loss_per_instance(pred, target, boxes, from_logits=from_logits, band_weight=band_weight, eps=eps)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"reduction must be one of 'none', 'sum', or 'mean', got {reduction!r}")
