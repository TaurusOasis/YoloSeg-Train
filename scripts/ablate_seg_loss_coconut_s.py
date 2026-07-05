#!/usr/bin/env python3
"""COCONut-S 30-epoch ablation for seg_comp / seg_bnd / seg_point (strategy-1 sub-gains).

Groups (gain 1.0 when enabled, else 0):
  G0 baseline | G1 comp | G2 bnd | G3 comp+bnd | G4 point-lite | G4m point-MLP
  | G5 point+bnd | G6 all

G4 uses yolo26s-seg.yaml (Lite, coarse logits). G4m uses yolo26s-seg-pointrend.yaml with
seg_point_refine=True (PointHeadMLP). seg_point_roi defaults to 0.0 (bbox ROI); pass
--seg-point-roi -1 for legacy full-grid sampling (ablation control).

No distillation by default so mask-loss changes are isolated. recipe200-like aug recipe.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

DEFAULT_DATA = Path("/home/genesis/Train/Dataset/COCONut_yolo_seg/coconut-s-seg.yaml")
DEFAULT_STUDENT = "yolo26s-seg.yaml"
POINTREND_STUDENT = "yolo26s-seg-pointrend.yaml"

GROUPS = {
    "G0": {"seg_comp": 0.0, "seg_bnd": 0.0, "seg_point": 0.0},
    "G1": {"seg_comp": 1.0, "seg_bnd": 0.0, "seg_point": 0.0},
    "G2": {"seg_comp": 0.0, "seg_bnd": 1.0, "seg_point": 0.0},
    "G3": {"seg_comp": 1.0, "seg_bnd": 1.0, "seg_point": 0.0},
    "G4": {"seg_comp": 0.0, "seg_bnd": 0.0, "seg_point": 1.0},
    "G4m": {"seg_comp": 0.0, "seg_bnd": 0.0, "seg_point": 1.0, "seg_point_refine": True},
    "G5": {"seg_comp": 0.0, "seg_bnd": 1.0, "seg_point": 1.0},
    "G6": {"seg_comp": 1.0, "seg_bnd": 1.0, "seg_point": 1.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS), default="G0", help="Ablation group preset.")
    parser.add_argument("--student", help="Student checkpoint or YAML (default: seg yaml, or pointrend for G4m).")
    parser.add_argument(
        "--seg-point-roi",
        type=float,
        default=None,
        help="Override seg_point_roi (default 0.0=bbox ROI; -1=legacy full-grid).",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="COCONut-S data YAML.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--device", default="0,1,2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs/segment"))
    parser.add_argument("--name", help="Run name; default ablate-<group>-coconut-s.")
    parser.add_argument("--distill", action="store_true", help="Enable teacher distillation (off by default).")
    parser.add_argument("--teacher", type=Path, default=REPO_ROOT / "yolo26x-seg.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data}")
    gains = GROUPS[args.group]
    student = args.student or (POINTREND_STUDENT if args.group == "G4m" else DEFAULT_STUDENT)
    name = args.name or f"ablate-{args.group.lower()}-coconut-s"
    if args.seg_point_roi is not None:
        gains = {**gains, "seg_point_roi": args.seg_point_roi}
    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": 640,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": name,
        "task": "segment",
        "seed": args.seed,
        "cos_lr": True,
        "close_mosaic": 20,
        "copy_paste": 0.4,
        "mixup": 0.1,
        "multi_scale": 0.25,
        "amp": True,
        **gains,
    }
    if args.distill:
        if not args.teacher.exists():
            raise FileNotFoundError(f"Teacher not found: {args.teacher}")
        train_args.update({"distill_model": str(args.teacher), "dis": 3.0, "dis_proto": 1.0})
    roi = gains.get("seg_point_roi", 0.0)
    refine = gains.get("seg_point_refine", False)
    print(
        f"Ablation {args.group} student={student}: "
        f"seg_comp={gains['seg_comp']} seg_bnd={gains['seg_bnd']} seg_point={gains['seg_point']} "
        f"seg_point_refine={refine} seg_point_roi={roi}"
    )
    YOLO(student).train(**train_args)


if __name__ == "__main__":
    main()
