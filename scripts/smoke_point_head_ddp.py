#!/usr/bin/env python3
r"""2-GPU DDP smoke: verify point_head gradients allreduce correctly (§2.5 / design doc §10.2).

Launch:
  cd ultralytics && conda run -n yolo26-cu133 python -m torch.distributed.run \\
    --nproc_per_node=2 --master_port=29501 scripts/smoke_point_head_ddp.py
  cd ultralytics && conda run -n yolo26-cu133 python -m torch.distributed.run \\
    --nproc_per_node=2 --master_port=29502 scripts/smoke_point_head_ddp.py --compile --boundary

Checks:
  - yolo26n-seg-pointrend builds with point_head under DDP (find_unused_parameters=False).
  - seg_point>0 + seg_point_refine=True: finite loss, point_head params get finite grads.
  - After backward, point_head grads match across ranks (post allreduce).
  - seg_point_refine=False: dummy-only path still DDP-safe (zero grad, no unused-param error).
  - optional --compile --boundary path mirrors trainer compile/DDP settings (static_graph=True) with boundary losses on.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import attempt_compile, unwrap_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compile", action="store_true", help="Compile the model before DDP, matching trainer behavior."
    )
    parser.add_argument(
        "--boundary", action="store_true", help="Enable seg_bnd and boundary-weighted point ROI sampling."
    )
    return parser.parse_args()


def _unwrap(model: DDP | SegmentationModel) -> SegmentationModel:
    return unwrap_model(model)


def _init_dist() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ:
        raise RuntimeError(
            "Launch with: python -m torch.distributed.run --nproc_per_node=2 scripts/smoke_point_head_ddp.py"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _make_batch(device: torch.device, batch_size: int = 2, imgsz: int = 128) -> dict[str, torch.Tensor]:
    """Minimal segment batch with two instances (overlap_mask=False)."""
    mh = mw = imgsz // 4
    masks = torch.zeros(batch_size, mh, mw, device=device)
    masks[:, mh // 4 : 3 * mh // 4, mw // 4 : 3 * mw // 4] = 1.0
    sem_masks = torch.zeros(batch_size, mh, mw, device=device, dtype=torch.long)
    sem_masks[:, mh // 4 : 3 * mh // 4, mw // 4 : 3 * mw // 4] = 1
    return {
        "img": torch.rand(batch_size, 3, imgsz, imgsz, device=device),
        "batch_idx": torch.arange(batch_size, device=device, dtype=torch.float32),
        "cls": torch.zeros(batch_size, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.45, 0.45]] * batch_size, device=device),
        "masks": masks,
        "sem_masks": sem_masks,
    }


def _model_args(**overrides) -> SimpleNamespace:
    base = dict(
        overlap_mask=False,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        epochs=1,
        seg_comp=0.0,
        seg_bnd=0.0,
        seg_point=1.0,
        seg_point_refine=True,
        seg_point_roi=0.0,
        seg_point_num=32,
        seg_point_oversample=3,
        seg_point_importance=0.75,
        seg_point_boundary=False,
        seg_point_o2o=1.0,
        seg_point_refine_o2o=True,
        e2e_final_o2m=0.1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _verify_point_head_grads_sync(model: DDP, rank: int, world_size: int, label: str) -> None:
    head = _unwrap(model).model[-1].point_head
    assert head is not None, f"{label}: point_head missing"
    for name, param in head.named_parameters():
        assert param.grad is not None, f"{label}: {name} grad is None on rank {rank}"
        assert torch.isfinite(param.grad).all(), f"{label}: {name} non-finite grad on rank {rank}"
        gathered = [torch.empty_like(param.grad) for _ in range(world_size)]
        dist.all_gather(gathered, param.grad.contiguous())
        ref = gathered[0]
        for r, g in enumerate(gathered[1:], start=1):
            max_diff = (ref - g).abs().max().item()
            assert max_diff < 1e-5, f"{label}: {name} grad mismatch rank0 vs rank{r}: max_diff={max_diff}"
    if rank == 0:
        print(f"  [{label}] point_head grads synced across {world_size} ranks")


def _run_step(model: DDP, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    preds = model(batch["img"])
    loss, _ = _unwrap(model).loss(batch, preds)
    assert torch.isfinite(loss).all(), "loss not finite"
    (loss.sum() if loss.ndim else loss).backward()
    return loss


def main() -> None:
    args = _parse_args()
    rank, world_size, device = _init_dist()
    torch.manual_seed(42 + rank)

    model = SegmentationModel("yolo26n-seg-pointrend.yaml", ch=3, nc=80, verbose=False)
    model.args = _model_args(
        seg_bnd=0.1 if args.boundary else 0.0,
        seg_point_boundary=args.boundary,
        seg_point_num=16 if args.compile else 32,
    )
    model = model.to(device).train()
    model = attempt_compile(model, device=device, imgsz=128, mode=args.compile)
    model = DDP(
        model,
        device_ids=[device.index],
        static_graph=bool(args.compile),
        find_unused_parameters=False,
    )

    batch = _make_batch(device)
    loss = _run_step(model, batch)
    _verify_point_head_grads_sync(model, rank, world_size, "mlp")

    # Dummy-only path: head exists but MLP not used in criterion.
    _unwrap(model).args = _model_args(
        seg_point_refine=False,
        seg_point=1.0,
        seg_bnd=0.1 if args.boundary else 0.0,
        seg_point_boundary=args.boundary,
        seg_point_num=16 if args.compile else 32,
    )
    _unwrap(model).criterion = None  # force re-init with new args on next loss()
    loss2 = _run_step(model, batch)
    head = _unwrap(model).model[-1].point_head
    for name, param in head.named_parameters():
        g = param.grad
        assert g is not None, f"dummy path: {name} grad None"
        assert torch.isfinite(g).all()
        assert g.abs().max().item() < 1e-12, f"dummy path: {name} grad should be ~0, got max={g.abs().max()}"
    if rank == 0:
        loss2_sum = loss2.sum().detach().item() if loss2.ndim else float(loss2.detach())
        loss_sum = loss.sum().detach().item() if loss.ndim else float(loss.detach())
        print(f"  [dummy] seg_point_refine=False: zero point_head grads OK (loss_sum={loss2_sum:.4f})")
        print(f"OK: 2-GPU point_head DDP smoke (mlp loss_sum={loss_sum:.4f})")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
