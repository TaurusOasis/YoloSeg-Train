#!/usr/bin/env python3
"""Train YOLO26s-seg on the LVIS COCO80-overlap subset with a YOLO26x-seg teacher."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/genesis/Train/Dataset/LVIS_coco80_yolo_seg")
DEFAULT_DATA_YAML = DEFAULT_DATA_ROOT / "lvis-coco80-seg.yaml"
DEFAULT_SOURCE_ROOT = Path("/home/genesis/Train/Dataset/LVIS_yolo_seg")
DEFAULT_STUDENT = "yolo26s-seg.yaml"
DEFAULT_TEACHER = REPO_ROOT / "yolo26x-seg.pt"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--student", default=DEFAULT_STUDENT, help="Student model checkpoint or YAML.")
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER, help="Teacher model checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Filtered LVIS COCO80 data YAML.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Filtered dataset root.")
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Original LVIS YOLO dataset root."
    )
    parser.add_argument(
        "--prepare-data", action="store_true", help="Build/update the filtered dataset before training."
    )
    parser.add_argument("--overwrite-data", action="store_true", help="Overwrite filtered labels when preparing data.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs/segment"))
    parser.add_argument("--name", default="yolo26s-seg-lvis-coco80-distill-x-teacher")
    parser.add_argument("--dis", type=float, default=3.0, help="Feature distillation loss weight.")
    parser.add_argument("--optimizer", default="MuSGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--no-val", action="store_true", help="Disable validation during training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing the output run directory.")
    return parser.parse_known_args()


def parse_unknown_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if not item.startswith("--") or "=" not in item:
            raise ValueError(f"Extra overrides must use --key=value format, got {item!r}")
        key, value = item[2:].split("=", 1)
        key = key.replace("-", "_")
        overrides[key] = yaml.safe_load(value)
    return overrides


def prepare_dataset(args: argparse.Namespace) -> None:
    if args.data.exists() and not args.prepare_data:
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_lvis_coco80_seg_subset.py"),
        "--src-root",
        str(args.source_root),
        "--out-root",
        str(args.data_root),
    ]
    if args.overwrite_data:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)


def main() -> None:
    args, unknown = parse_args()
    prepare_dataset(args)
    if not args.data.exists():
        raise FileNotFoundError(f"Filtered data YAML not found: {args.data}")
    if not args.teacher.exists():
        raise FileNotFoundError(f"Teacher model not found: {args.teacher}")

    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "task": "segment",
        "distill_model": str(args.teacher),
        "dis": args.dis,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
        "close_mosaic": args.close_mosaic,
        "save_period": args.save_period,
        "patience": args.patience,
        "val": not args.no_val,
        "exist_ok": args.exist_ok,
        "amp": True,
    }
    train_args.update(parse_unknown_overrides(unknown))

    model = YOLO(str(args.student))
    model.train(**train_args)


if __name__ == "__main__":
    main()
