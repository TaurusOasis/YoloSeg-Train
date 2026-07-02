#!/usr/bin/env python3
"""Export YOLO segmentation predictions to COCO/LVIS-style JSON without validation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Image path, directory, or txt manifest.")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--conf", default=0.001, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--max-det", default=300, type=int)
    parser.add_argument("--retina-masks", action="store_true", help="Export masks at original image resolution.")
    parser.add_argument("--progress-every", default=250, type=int)
    return parser.parse_args()


def encode_mask(mask: torch.Tensor) -> dict:
    arr = mask.detach().to("cpu").numpy()
    arr = np.asfortranarray((arr > 0.5).astype(np.uint8))
    rle = mask_utils.encode(arr)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def image_id_from_path(path: str | Path) -> int | str:
    stem = Path(path).stem
    return int(stem) if stem.isnumeric() else stem


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model.expanduser().resolve()))
    out_json = args.out_json.expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    image_count = 0
    first = True
    with out_json.open("w", encoding="utf-8") as f:
        f.write("[")
        results = model.predict(
            source=str(args.source.expanduser().resolve()),
            stream=True,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            retina_masks=args.retina_masks,
            verbose=False,
        )
        for result in results:
            image_count += 1
            boxes = result.boxes
            masks = result.masks
            if boxes is None or masks is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.detach().to("cpu").numpy()
            conf = boxes.conf.detach().to("cpu").numpy()
            cls = boxes.cls.detach().to("cpu").numpy().astype(int)
            mask_data = masks.data
            image_id = image_id_from_path(result.path)
            file_name = Path(result.path).name

            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                item = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "category_id": int(cls[i]) + 1,
                    "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                    "score": round(float(conf[i]), 5),
                    "segmentation": encode_mask(mask_data[i]),
                }
                if not first:
                    f.write(",")
                json.dump(item, f, separators=(",", ":"))
                first = False
                count += 1

            if args.progress_every and image_count % args.progress_every == 0:
                f.flush()
                print(f"processed_images={image_count} predictions={count}", flush=True)
        f.write("]")

    print(f"Wrote {count} predictions for {image_count} images to {out_json}", flush=True)


if __name__ == "__main__":
    main()
