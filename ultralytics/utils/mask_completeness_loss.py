# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Mask completeness loss for instance segmentation (FN-asymmetric Focal-Tversky).

This is a region-level loss that penalizes false negatives (holes inside the predicted mask)
more strongly than false positives. It is intentionally **relative to the GT mask**: pixels
where ``gt == 0`` (including the legitimate holes preserved in COCONut v2 labels by
``RETR_CCOMP`` + ``merge_multi_segment``) are left alone, so it does not blindly fill holes.

Used as a sub-gain inside ``v8SegmentationLoss.single_mask_loss`` (``seg_comp``), default 0.
"""

from __future__ import annotations

import torch

from ultralytics.utils.ops import crop_mask

__all__ = ["tversky_loss_per_instance"]


def tversky_loss_per_instance(
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    xyxy: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-instance Focal-Tversky loss restricted to each instance's bbox.

    Args:
        pred_logits (torch.Tensor): (N, H, W) predicted mask logits.
        gt_mask (torch.Tensor): (N, H, W) binary ground-truth mask.
        xyxy (torch.Tensor): (N, 4) bboxes in mask pixel coordinates (used to crop to the box).
        alpha (float): False-positive weight (low => do not encourage outward expansion).
        beta (float): False-negative weight (> alpha => encourage filling interior FN).
        gamma (float): Focal exponent applied to (1 - TI); 1.0 => plain Tversky.
        eps (float): Numerical stabilizer.

    Returns:
        (torch.Tensor): Per-instance loss of shape (N,), each in [0, 1].
    """
    n = pred_logits.shape[0]
    if n == 0:
        return pred_logits.new_zeros(0)
    pm = pred_logits.float()
    gm = gt_mask.float()
    prob = torch.sigmoid(pm)
    # crop_mask mutates small CPU tensors in place, so clone autograd outputs first.
    prob = crop_mask(prob.clone(), xyxy)
    gm = crop_mask(gm.clone(), xyxy)
    tp = (prob * gm).sum(dim=(1, 2))
    fp = (prob * (1.0 - gm)).sum(dim=(1, 2))
    fn = ((1.0 - prob) * gm).sum(dim=(1, 2))
    ti = tp / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - ti).clamp(min=eps) ** gamma
