#!/usr/bin/env python3
"""Finetune YOLO26s-seg + PointRend (boundary refine) from the recipe200 distilled checkpoint.

Continuation of yolo26s-seg-coconut-b-v2-distill-recipe200 (stopped 107/200, best mask
mAP50-95=0.376). The recipe200 ckpt has no point_head, so this is a *finetune* (new run,
``pretrained=<best.pt>``), never ``resume``: the pointrend YAML adds a zero-init PointHeadMLP
(844/850 params transferred, 6 new), so step-0 behavior matches the recipe200 best exactly.

Defaults follow the recipe200 memory notes: no distillation (marginal returns ~0, frees teacher
VRAM), batch 84 + multi_scale 0.15 (recipe200 died OOM at batch 90 / ms 0.25 on Size=800), and
PointRend supervision on the one2many branch only (one2one feats are detached anyway).

Launch (3 GPUs):
  cd ultralytics && conda run -n yolo26-cu133 python scripts/finetune_yolo26s_seg_pointrend_coconut_b.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must precede torch CUDA init: multi_scale fragments the allocator cache (recipe200 F20/F21 OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

DEFAULT_PRETRAINED = REPO_ROOT / "runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt"
DEFAULT_DATA = Path("/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED, help="Init weights (recipe200 best).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=84, help="Recipe200 OOM'd at 90 w/ distill; 84 leaves headroom.")
    parser.add_argument("--device", default="0,1,2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.003, help="Finetune LR (recipe200 full-run used 0.01).")
    parser.add_argument("--multi-scale", type=float, default=0.15, help="Reduced from 0.25 (OOM mitigation).")
    parser.add_argument("--seg-point", type=float, default=0.5)
    parser.add_argument("--seg-point-num", type=int, default=64)
    parser.add_argument("--seg-bnd", type=float, default=0.0, help="Optional dense Sobel L2 on top of point loss.")
    parser.add_argument("--no-boundary", action="store_true", help="Disable GT Sobel boundary-weighted sampling.")
    parser.add_argument("--e2e-final-o2m", type=float, default=0.1)
    parser.add_argument("--distill", action="store_true", help="Keep yolo26x-seg distillation (off by default).")
    parser.add_argument("--teacher", type=Path, default=REPO_ROOT / "yolo26x-seg.pt")
    parser.add_argument("--name", default="yolo26s-seg-coconut-b-v2-pointrend-ft60")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pretrained.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data}")

    train_args = {
        "data": str(args.data),
        "pretrained": str(args.pretrained),
        "resume": False,  # structure changed (point_head added): finetune, never resume
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": 640,
        "device": args.device,
        "workers": args.workers,
        "project": str(REPO_ROOT / "runs/segment"),
        "name": args.name,
        "task": "segment",
        "seed": args.seed,
        "optimizer": "MuSGD",
        "lr0": args.lr0,
        "lrf": 0.01,
        "cos_lr": True,
        "warmup_epochs": 1.0,
        "close_mosaic": 15,
        "copy_paste": 0.4,
        "mixup": 0.1,
        "multi_scale": args.multi_scale,
        "patience": args.epochs,  # effectively disable early stopping
        "save_period": 5,
        "amp": True,
        # PointRend boundary refine (training side)
        "seg_point": args.seg_point,
        "seg_point_refine": True,
        "seg_point_boundary": not args.no_boundary,
        "seg_point_roi": 0.0,
        "seg_point_num": args.seg_point_num,
        "seg_point_o2o": 0.0,  # one2one feats are detached; keep point supervision on one2many
        "seg_point_refine_o2o": False,
        "e2e_final_o2m": args.e2e_final_o2m,
        "seg_bnd": args.seg_bnd,
    }
    if args.distill:
        if not args.teacher.exists():
            raise FileNotFoundError(f"Teacher not found: {args.teacher}")
        train_args.update(
            {"distill_model": str(args.teacher), "dis": 3.0, "dis_proto": 1.0, "distill_warmup_epochs": 3.0}
        )

    print(
        f"PointRend finetune from {args.pretrained.name}: epochs={args.epochs} batch={args.batch} "
        f"lr0={args.lr0} seg_point={args.seg_point} boundary={not args.no_boundary} "
        f"distill={args.distill} multi_scale={args.multi_scale}"
    )
    YOLO("yolo26s-seg-pointrend.yaml").train(**train_args)


if __name__ == "__main__":
    main()
