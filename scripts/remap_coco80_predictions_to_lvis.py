#!/usr/bin/env python3
"""Remap COCO80 YOLO prediction JSON category ids to LVIS category ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COCO80_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


COCO80_TO_LVIS_ID = {
    "person": 793,
    "bicycle": 94,
    "car": 207,
    "motorcycle": 703,
    "airplane": 3,
    "bus": 173,
    "train": 1115,
    "truck": 1123,
    "boat": 118,
    "traffic light": 1112,
    "fire hydrant": 445,
    "stop sign": 1019,
    "parking meter": 766,
    "bench": 90,
    "bird": 99,
    "cat": 225,
    "dog": 378,
    "horse": 569,
    "sheep": 943,
    "cow": 80,
    "elephant": 422,
    "bear": 76,
    "zebra": 1202,
    "giraffe": 496,
    "backpack": 34,
    "umbrella": 1133,
    "handbag": 35,
    "tie": 716,
    "suitcase": 36,
    "frisbee": 474,
    "skis": 964,
    "snowboard": 976,
    "sports ball": 41,
    "kite": 611,
    "baseball bat": 58,
    "baseball glove": 60,
    "skateboard": 962,
    "surfboard": 1037,
    "tennis racket": 1079,
    "bottle": 133,
    "wine glass": 1190,
    "cup": 344,
    "fork": 469,
    "knife": 615,
    "spoon": 1000,
    "bowl": 139,
    "banana": 45,
    "apple": 12,
    "sandwich": 912,
    "orange": 735,
    "broccoli": 154,
    "carrot": 217,
    # No direct LVIS v1 category for COCO "hot dog"; leave unmapped.
    "pizza": 816,
    "donut": 387,
    "cake": 183,
    "chair": 232,
    "couch": 982,
    # LVIS has plant-related parts/containers, but no direct potted plant category.
    "bed": 77,
    "dining table": 367,
    "toilet": 1097,
    "tv": 1077,
    "laptop": 631,
    "mouse": 705,
    "remote": 881,
    "keyboard": 296,
    "cell phone": 230,
    "microwave": 687,
    "oven": 739,
    "toaster": 1095,
    "sink": 961,
    "refrigerator": 421,
    "book": 127,
    "clock": 271,
    "vase": 1139,
    "scissors": 923,
    "teddy bear": 1071,
    "hair drier": 534,
    "toothbrush": 1102,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-json", required=True, type=Path, help="YOLO predictions.json with category_id=COCO index+1."
    )
    parser.add_argument("--out-json", required=True, type=Path, help="Output predictions JSON with LVIS category ids.")
    parser.add_argument("--report-json", type=Path, help="Optional mapping summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = json.loads(args.pred_json.expanduser().read_text(encoding="utf-8"))
    mapped = []
    counts: dict[str, int] = {}
    dropped: dict[str, int] = {}

    for item in preds:
        coco_idx = int(item["category_id"]) - 1
        if coco_idx < 0 or coco_idx >= len(COCO80_NAMES):
            dropped[f"unknown:{item['category_id']}"] = dropped.get(f"unknown:{item['category_id']}", 0) + 1
            continue
        name = COCO80_NAMES[coco_idx]
        lvis_id = COCO80_TO_LVIS_ID.get(name)
        if lvis_id is None:
            dropped[name] = dropped.get(name, 0) + 1
            continue
        out = dict(item)
        out["category_id"] = lvis_id
        mapped.append(out)
        counts[name] = counts.get(name, 0) + 1

    args.out_json.expanduser().write_text(json.dumps(mapped), encoding="utf-8")
    report = {
        "input": str(args.pred_json),
        "output": str(args.out_json),
        "input_predictions": len(preds),
        "mapped_predictions": len(mapped),
        "dropped_predictions": len(preds) - len(mapped),
        "mapped_classes": len(counts),
        "dropped_by_class": dict(sorted(dropped.items())),
        "mapped_by_class": dict(sorted(counts.items())),
        "mapping": COCO80_TO_LVIS_ID,
    }
    if args.report_json:
        args.report_json.expanduser().write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
