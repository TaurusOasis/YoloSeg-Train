# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""PointRend-style mask point sampling (adapted from Detectron2 PointRend / SAM3 mask_sampling).

Tools here follow the Detectron2 PointRend ``point_features`` fork shipped in
``sam3/train/loss/mask_sampling.py``: ``point_sample`` (bilinear grid_sample wrapper),
``calculate_uncertainty`` (``-|logit|`` for class-agnostic masks, from Mask2Former), and
``get_uncertain_point_coords_with_randomness`` (full-grid [0,1]^2 uncertainty sampling).

The per-instance point losses mirror ``sam3/train/loss/loss_fns._sampled_loss`` /
Mask2Former / kmaxdeeplab: a sigmoid focal loss plus a dice loss on the sampled points, with
no point-head MLP and no inference-time subdivision.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    """Sample features at normalized point coordinates via bilinear grid_sample.

    Args:
        input (torch.Tensor): (N, C, H, W) feature map.
        point_coords (torch.Tensor): (N, P, 2) coordinates in [0, 1] x [0, 1].

    Returns:
        (torch.Tensor): (N, C, P) sampled values.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    normalized_point_coords = 2.0 * point_coords - 1.0
    output = F.grid_sample(input, normalized_point_coords, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """Uncertainty as -|logit| for class-agnostic (N, 1, P) mask logits."""
    assert logits.shape[1] == 1
    return -torch.abs(logits)


def get_uncertain_point_coords_with_randomness(
    logits: torch.Tensor,
    uncertainty_func: Callable[[torch.Tensor], torch.Tensor],
    num_points: int,
    oversample_ratio: int,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """Sample points in [0, 1]^2 using uncertainty-based importance sampling (full-grid variant)."""
    assert oversample_ratio >= 1
    assert 0.0 <= importance_sample_ratio <= 1.0
    num_boxes = logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=logits.device)
    point_logits = point_sample(logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    k = min(num_uncertain_points, num_sampled)
    if k > 0:
        idx = torch.topk(point_uncertainties[:, 0, :], k=k, dim=1)[1]
        shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx = idx + shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, k, 2)
    else:
        point_coords = logits.new_zeros(num_boxes, 0, 2)
    if num_random_points > 0:
        point_coords = torch.cat(
            [point_coords, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
            dim=1,
        )
    return point_coords


def _rand_in_roi(
    num_boxes: int, num: int, boxes_norm: torch.Tensor, margin: float, device: torch.device
) -> torch.Tensor:
    """Draw ``num`` random points per instance inside its (margin-expanded) normalized bbox.

    Args:
        num_boxes (int): Number of instances.
        num (int): Points per instance.
        boxes_norm (torch.Tensor): (N, 4) xyxy in [0, 1].
        margin (float): Fractional expansion of each bbox side (clamped to [0, 1]).
        device (torch.device): Device for new tensors.

    Returns:
        (torch.Tensor): (N, num, 2) random coords in [0, 1]. Degenerate boxes (x2<=x1 or y2<=y1 after clamping) fall
            back to full-grid [0, 1]^2 sampling for those rows.
    """
    x1 = (boxes_norm[:, 0] - margin).clamp(0.0, 1.0)
    y1 = (boxes_norm[:, 1] - margin).clamp(0.0, 1.0)
    x2 = (boxes_norm[:, 2] + margin).clamp(0.0, 1.0)
    y2 = (boxes_norm[:, 3] + margin).clamp(0.0, 1.0)
    rand = torch.rand(num_boxes, num, 2, device=device)
    xs = x1[:, None] + rand[..., 0] * (x2 - x1)[:, None]
    ys = y1[:, None] + rand[..., 1] * (y2 - y1)[:, None]
    degenerate = (x2 <= x1) | (y2 <= y1)
    if degenerate.any():
        xs[degenerate] = rand[degenerate, :, 0]
        ys[degenerate] = rand[degenerate, :, 1]
    return torch.stack([xs, ys], dim=-1)


def _weighted_rand_in_roi(
    num_boxes: int,
    num: int,
    boxes_norm: torch.Tensor,
    margin: float,
    weight_map: torch.Tensor,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw ``num`` points per instance inside its bbox, biased by a per-instance weight map.

    A boundary-band-weighted companion to :func:`_rand_in_roi`: instead of a uniform draw inside the (margin-expanded,
    clamped) bbox, candidate pixels are drawn with probability proportional to ``weight_map`` restricted to the bbox —
    so points concentrate on the true GT boundary band when ``weight_map`` is a Sobel magnitude map. Sampling is via
    ``torch.multinomial`` (with
    replacement) and is non-differentiable by design; callers wrap it in ``torch.no_grad``.

    Fallbacks (so a missing boundary signal never starves the point loss):
    - degenerate bbox (x2<=x1 or y2<=y1 after clamping) -> uniform full-grid [0, 1]^2 for that row
        (matches :func:`_rand_in_roi`).
    - valid bbox but zero total weight inside (e.g. a constant GT region with no Sobel response)
        -> uniform within the bbox for that row.

    Args:
        num_boxes (int): Number of instances.
        num (int): Points per instance.
        boxes_norm (torch.Tensor): (N, 4) xyxy in [0, 1].
        margin (float): Fractional expansion of each bbox side (clamped to [0, 1]).
        weight_map (torch.Tensor): (N, H, W) non-negative per-instance weights (e.g. Sobel |grad|).
        H (int): Mask height (matches ``weight_map`` / the loss-space logits).
        W (int): Mask width.
        device (torch.device): Device for new tensors.

    Returns:
        (torch.Tensor): (N, num, 2) coords in [0, 1]. Pixel indices are mapped to pixel-center coords ``(idx + 0.5) /
            size`` to match ``point_sample(..., align_corners=False)``.
    """
    if num <= 0:
        return weight_map.new_zeros(num_boxes, 0, 2)
    x1 = (boxes_norm[:, 0] - margin).clamp(0.0, 1.0)
    y1 = (boxes_norm[:, 1] - margin).clamp(0.0, 1.0)
    x2 = (boxes_norm[:, 2] + margin).clamp(0.0, 1.0)
    y2 = (boxes_norm[:, 3] + margin).clamp(0.0, 1.0)
    px1 = (x1 * W).long().clamp(0, W)
    px2 = (x2 * W).long().clamp(0, W)
    py1 = (y1 * H).long().clamp(0, H)
    py2 = (y2 * H).long().clamp(0, H)
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    in_x = (xs[None] >= px1[:, None, None]) & (xs[None] < px2[:, None, None])
    in_y = (ys[None] >= py1[:, None, None]) & (ys[None] < py2[:, None, None])
    bbox_mask = (in_x & in_y).to(weight_map.dtype)  # (N, H, W)
    probs = (weight_map.float() * bbox_mask).reshape(num_boxes, -1)  # (N, H*W)
    rowsum = probs.sum(dim=1)
    bbox_degenerate = (px2 <= px1) | (py2 <= py1)
    weight_empty = (~bbox_degenerate) & (rowsum <= 0)
    if bbox_degenerate.any():
        # Degenerate bbox: fall back to uniform full-grid (same as _rand_in_roi).
        probs[bbox_degenerate] = 1.0 / (H * W)
    if weight_empty.any():
        # Valid bbox but no boundary weight inside: fall back to uniform within the bbox.
        probs[weight_empty] = bbox_mask.reshape(num_boxes, -1)[weight_empty]
    probs = probs + 1e-12  # guard against any residual all-zero row
    probs = probs / probs.sum(dim=1, keepdim=True)
    idx = torch.multinomial(probs, num, replacement=True)  # (N, num) flat pixel indices
    rows = idx // W
    cols = idx % W
    xs_c = (cols.to(weight_map.dtype) + 0.5) / W
    ys_c = (rows.to(weight_map.dtype) + 0.5) / H
    return torch.stack([xs_c, ys_c], dim=-1)


def get_uncertain_point_coords_in_roi(
    logits: torch.Tensor,
    uncertainty_func: Callable[[torch.Tensor], torch.Tensor],
    num_points: int,
    oversample_ratio: int,
    importance_sample_ratio: float,
    boxes_norm: torch.Tensor,
    margin: float = 0.0,
    weight_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample uncertain points restricted to each instance's bbox (PointRend ROI variant).

    Unlike ``get_uncertain_point_coords_with_randomness`` (full-grid [0, 1]^2), both the oversample draw and the random
    remainder are confined to the per-instance bbox (expanded by ``margin`` and clamped to [0, 1]), so uncertain points
    land on the object/boundary instead of on background — fixing the small-object background-sampling weakness of the
    lite seg_point.

    When ``weight_map`` (N, H, W) is provided, the oversample candidate pool is a 50/50 blend of a boundary-weighted
    draw (see :func:`_weighted_rand_in_roi`; pass a GT Sobel magnitude map) and a uniform-in-bbox draw (see
    :func:`_rand_in_roi`). The pred-uncertainty top-k then operates over this *combined* pool, so interior
    wrong-but-uncertain points (false-positive regions away from the GT boundary, where the Sobel weight ≈ 0) stay
    reachable by the importance selection — they are not crowded out by the boundary-weighted half. The boundary is
    still over-represented (~5× its pixel share) for boundary-focused point supervision, while interior FP/FN regions
    keep a path into the top-k. The 25% random remainder is *always* uniform-in-bbox (never boundary weighted), matching
    the legacy remainder and guaranteeing interior coverage. When ``weight_map`` is ``None``, the draw is uniform inside
    the bbox and the behavior is pure ROI-uniform.
    """
    assert oversample_ratio >= 1
    assert 0.0 <= importance_sample_ratio <= 1.0
    num_boxes = logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    if weight_map is not None:
        # Blended candidate pool: half boundary-weighted (Sobel) + half uniform-in-bbox, so the
        # pred-uncertainty top-k can still surface interior wrong-but-uncertain points instead of
        # being crowded out by the boundary-weighted half.
        H, W = logits.shape[-2], logits.shape[-1]
        num_bnd = num_sampled // 2
        num_uni = num_sampled - num_bnd
        bnd = _weighted_rand_in_roi(num_boxes, num_bnd, boxes_norm, margin, weight_map, H, W, logits.device)
        uni = _rand_in_roi(num_boxes, num_uni, boxes_norm, margin, logits.device)
        point_coords = torch.cat([bnd, uni], dim=1)  # (N, num_sampled, 2)
    else:
        point_coords = _rand_in_roi(num_boxes, num_sampled, boxes_norm, margin, logits.device)
    point_logits = point_sample(logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    k = min(num_uncertain_points, num_sampled)
    if k > 0:
        idx = torch.topk(point_uncertainties[:, 0, :], k=k, dim=1)[1]
        shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx = idx + shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, k, 2)
    else:
        point_coords = logits.new_zeros(num_boxes, 0, 2)
    if num_random_points > 0:
        # Remainder is always uniform-in-bbox (never boundary weighted) so interior FP/FN regions
        # keep coverage; matches the legacy full-grid random remainder semantics.
        rem = _rand_in_roi(num_boxes, num_random_points, boxes_norm, margin, logits.device)
        point_coords = torch.cat([point_coords, rem], dim=1)
    return point_coords


def point_sigmoid_focal_loss_per_instance(
    point_logits: torch.Tensor,
    point_targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Per-instance sigmoid focal loss over sampled points (torch-only, no triton).

    Mirrors ``sam3/train/loss/loss_fns.sigmoid_focal_loss`` (non-triton branch) but returns a per-instance mean over the
    point dimension instead of a global reduction.

    Args:
        point_logits (torch.Tensor): (N, P) predicted mask logits at sampled points (with grad).
        point_targets (torch.Tensor): (N, P) target values in [0, 1] (soft, bilinear-sampled GT).
        alpha (float): Focal balancing factor in [0, 1]; Mask2Former default 0.25.
        gamma (float): Focal exponent; Mask2Former default 2.0.

    Returns:
        (torch.Tensor): Per-instance loss of shape (N,), each the mean focal loss over P points.
    """
    prob = point_logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(point_logits, point_targets, reduction="none")
    p_t = prob * point_targets + (1.0 - prob) * (1.0 - point_targets)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * point_targets + (1.0 - alpha) * (1.0 - point_targets)
        loss = alpha_t * loss
    return loss.mean(dim=1)


def point_dice_loss_per_instance(
    point_logits: torch.Tensor,
    point_targets: torch.Tensor,
) -> torch.Tensor:
    """Per-instance dice loss over sampled points.

    Mirrors ``sam3/train/loss/loss_fns._dice_loss`` (``1 - (2*inter+1)/(denom+1)``) but returns a per-instance value
    instead of a global reduction.

    Args:
        point_logits (torch.Tensor): (N, P) predicted mask logits at sampled points (with grad).
        point_targets (torch.Tensor): (N, P) target values in [0, 1].

    Returns:
        (torch.Tensor): Per-instance dice loss of shape (N,), each in [0, 1) (since 2*inter <= denom).
    """
    prob = point_logits.sigmoid()
    inter = (prob * point_targets).sum(dim=1)
    denom = prob.sum(dim=1) + point_targets.sum(dim=1)
    return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
