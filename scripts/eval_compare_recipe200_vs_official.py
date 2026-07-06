#!/usr/bin/env python3
"""Compare recipe200 best.pt vs official yolo26s-seg.pt on shared val sets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "recipe200-best": ROOT / "runs/segment/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt",
    "yolo26s-seg-official": ROOT / "yolo26s-seg.pt",
}
DATASETS = {
    "coconut-v2-val": "/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml",
    "coco-val2017": "/home/genesis/Train/Dataset/coco_val2017_yolo_seg/coco-val2017-seg.yaml",
}
PROJECT = ROOT / "runs/segment/eval_compare_recipe200_vs_official"
IMGSZ = 640


def _metric_row(metrics) -> dict[str, float]:
    return {
        "box_P": round(float(metrics.box.mp), 4),
        "box_R": round(float(metrics.box.mr), 4),
        "box_mAP50": round(float(metrics.box.map50), 4),
        "box_mAP50-95": round(float(metrics.box.map), 4),
        "mask_P": round(float(metrics.seg.mp), 4),
        "mask_R": round(float(metrics.seg.mr), 4),
        "mask_mAP50": round(float(metrics.seg.map50), 4),
        "mask_mAP50-95": round(float(metrics.seg.map), 4),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plots", action="store_true", help="Save val curves and batch prediction images")
    p.add_argument("--device", type=str, default="2", help="CUDA device id or 'cpu'")
    p.add_argument("--batch", type=int, default=8, help="Val batch size")
    p.add_argument("--suffix", type=str, default="", help="Optional run name suffix (e.g. _plots)")
    p.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Evaluate custom checkpoints instead of the built-in pair; repeatable (e.g. "
        "--model pointrend-ft60=runs/segment/.../best.pt).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Line-buffered stdout for nohup logs
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    models = MODELS
    if args.model:
        models = {}
        for item in args.model:
            name, _, path = item.partition("=")
            if not path:
                raise SystemExit(f"--model expects NAME=PATH, got {item!r}")
            models[name] = Path(path)

    PROJECT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict]] = {}
    for model_name, weights in models.items():
        results[model_name] = {}
        model = YOLO(str(weights))
        for data_name, data_yaml in DATASETS.items():
            run_name = f"{model_name}__{data_name}{args.suffix}"
            print(f"\n=== {run_name} (plots={args.plots}) ===", flush=True)
            metrics = model.val(
                data=data_yaml,
                split="val",
                imgsz=IMGSZ,
                batch=args.batch,
                device=args.device,
                project=str(PROJECT),
                name=run_name,
                exist_ok=True,
                verbose=True,
                plots=args.plots,
            )
            row = _metric_row(metrics)
            results[model_name][data_name] = row
            print(json.dumps(row, indent=2), flush=True)
            if args.plots:
                print(f"Plots saved under: {PROJECT / run_name}", flush=True)

    tag = "plots" if args.plots else "metrics"
    out = PROJECT / f"summary_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
