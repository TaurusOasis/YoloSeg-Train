# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.utils.mask_boundary_loss import boundary_l2_loss, boundary_l2_loss_per_instance, sobel_magnitude


def test_sobel_magnitude_constant_input_is_zero():
    """Replicate padding should not create artificial image-border edges."""
    x = torch.ones(2, 12, 10)
    assert torch.allclose(sobel_magnitude(x), torch.zeros_like(x))


def test_sobel_magnitude_uses_float32_under_autocast():
    """Boundary gradients stay in fp32 even when an autocast context is active."""
    x = torch.rand(2, 12, 10)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mag = sobel_magnitude(x)
    assert mag.dtype == torch.float32
    assert torch.isfinite(mag).all()


def test_boundary_l2_loss_is_zero_for_matching_masks_and_has_grad():
    """A matching probability mask has no boundary penalty and the helper remains differentiable."""
    target = torch.zeros(1, 16, 16)
    target[:, 4:12, 4:12] = 1.0
    pred_prob = target.clone().requires_grad_(True)

    loss = boundary_l2_loss(pred_prob, target, from_logits=False)
    assert torch.allclose(loss, torch.zeros_like(loss))
    loss.backward()
    assert pred_prob.grad is not None
    assert torch.isfinite(pred_prob.grad).all()


def test_boundary_l2_loss_increases_for_shifted_boundary():
    """A shifted square boundary should be penalized more than a matching one."""
    target = torch.zeros(1, 20, 20)
    target[:, 5:15, 5:15] = 1.0
    shifted = torch.zeros_like(target)
    shifted[:, 5:15, 7:17] = 1.0

    matching = boundary_l2_loss_per_instance(target, target, from_logits=False)
    shifted_loss = boundary_l2_loss_per_instance(shifted, target, from_logits=False)
    assert torch.allclose(matching, torch.zeros_like(matching))
    assert (shifted_loss > matching).all()


def test_boundary_l2_loss_boxes_crop_unweighted_error():
    """Optional boxes restrict the unweighted boundary error to the target instance region."""
    target = torch.zeros(1, 20, 20)
    target[:, 5:15, 5:15] = 1.0
    pred = target.clone()
    pred[:, 0:4, 0:4] = 1.0
    box = torch.tensor([[5.0, 5.0, 15.0, 15.0]])

    cropped = boundary_l2_loss_per_instance(pred, target, box, from_logits=False, band_weight=False)
    uncropped = boundary_l2_loss_per_instance(pred, target, from_logits=False, band_weight=False)
    assert cropped.item() == 0.0
    assert uncropped.item() > cropped.item()
