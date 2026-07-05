# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.utils.mask_completeness_loss import tversky_loss_per_instance


def test_tversky_loss_penalizes_missing_foreground_more_than_filled_mask():
    """Completeness loss should increase when GT foreground is missing."""
    gt = torch.zeros(1, 16, 16)
    gt[:, 4:12, 4:12] = 1.0
    full = torch.full_like(gt, -8.0)
    full[:, 4:12, 4:12] = 8.0
    hole = full.clone()
    hole[:, 7:9, 7:9] = -8.0
    box = torch.tensor([[4.0, 4.0, 12.0, 12.0]])

    full_loss = tversky_loss_per_instance(full, gt, box)
    hole_loss = tversky_loss_per_instance(hole, gt, box)
    assert hole_loss.item() > full_loss.item()


def test_tversky_loss_ignores_legitimate_gt_holes():
    """GT-zero regions inside the box should not be treated as foreground to fill."""
    gt = torch.zeros(1, 16, 16)
    gt[:, 3:13, 3:13] = 1.0
    gt[:, 6:10, 6:10] = 0.0
    pred_keeps_hole = torch.full_like(gt, -8.0)
    pred_keeps_hole[:, 3:13, 3:13] = 8.0
    pred_keeps_hole[:, 6:10, 6:10] = -8.0
    pred_fills_hole = pred_keeps_hole.clone()
    pred_fills_hole[:, 6:10, 6:10] = 8.0
    box = torch.tensor([[3.0, 3.0, 13.0, 13.0]])

    keep_loss = tversky_loss_per_instance(pred_keeps_hole, gt, box)
    fill_loss = tversky_loss_per_instance(pred_fills_hole, gt, box)
    assert keep_loss.item() <= fill_loss.item()
