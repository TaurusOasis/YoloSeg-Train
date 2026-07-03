#!/usr/bin/env python3
"""Build an Ultralytics YOLO segmentation dataset from local COCONut panoptic masks.

COCONut stores one RGB panoptic mask PNG per image. Each pixel encodes a segment id with COCO panoptic's
``R + 256 * G + 256 * 256 * B`` convention, while the JSON annotation stores ``segments_info`` with category ids. This
script keeps only COCO thing categories and writes YOLO segment labels next to symlinked COCO images. Each thing
segment becomes exactly one label line: hole boundaries (RETR_CCOMP child contours) and disconnected fragments are
merged into a single polygon via thin bridges, so holes stay unfilled and occlusion-split instances are not
duplicated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from ultralytics.data.converter import merge_multi_segment


DEFAULT_COCONUT_ROOT = Path("/home/genesis/Train/Dataset/coconut")
DEFAULT_IMAGE_ROOT = Path("/home/genesis/Train/Dataset/coco2017")
DEFAULT_OUT_ROOT = Path("/home/genesis/Train/Dataset/COCONut_yolo_seg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--coconut-root", type=Path, default=DEFAULT_COCONUT_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--train-split", choices=["coconut_s", "coconut_b"], default="coconut_s")
    parser.add_argument("--val-split", default="relabeled_coco_val")
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 8) // 2)))
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--min-area", type=int, default=4, help="Minimum contour area in mask pixels.")
    parser.add_argument(
        "--approx-epsilon",
        type=float,
        default=0.001,
        help="Douglas-Peucker epsilon as a fraction of contour perimeter. Use 0 to disable.",
    )
    parser.add_argument("--drop-empty", action="store_true", help="Do not include images without thing segments.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing labels before conversion.")
    parser.add_argument("--limit", type=int, help="Debug limit per split.")
    return parser.parse_args()


def rgb_to_id(mask: np.ndarray) -> np.ndarray:
    """Convert a panoptic RGB mask to integer segment ids."""
    if mask.ndim == 2:
        return mask.astype(np.int32)
    mask = mask.astype(np.int32)
    return mask[:, :, 0] + 256 * mask[:, :, 1] + 256 * 256 * mask[:, :, 2]


def ensure_symlink(src: Path, dst: Path) -> None:
    """Create or validate a directory symlink."""
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise FileExistsError(f"{dst} already exists and is not a symlink to {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst, target_is_directory=True)


def load_coco80_categories(categories: list[dict[str, Any]]) -> tuple[list[str], dict[int, int], dict[int, dict[str, Any]]]:
    """Return COCO thing names and category-id to dense-index map."""
    thing_categories = [c for c in categories if int(c.get("isthing", 0)) == 1]
    names = [str(c["name"]) for c in thing_categories]
    category_to_index = {int(c["id"]): i for i, c in enumerate(thing_categories)}
    category_by_id = {int(c["id"]): c for c in categories}
    return names, category_to_index, category_by_id


def simplify_contour(contour: np.ndarray, approx_epsilon: float) -> np.ndarray | None:
    """Simplify one OpenCV contour with Douglas-Peucker and return its (N, 2) points, or None if degenerate."""
    if approx_epsilon > 0:
        epsilon = approx_epsilon * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
    points = contour.reshape(-1, 2)
    return points if len(points) >= 3 else None


def segment_to_line(binary: np.ndarray, cls: int, width: int, height: int, min_area: int, approx_epsilon: float) -> tuple[str | None, int]:
    """Convert one instance's binary mask to a single YOLO segment label line.

    Uses RETR_CCOMP so interior holes are kept (as child contours), and merges all parts — disconnected fragments
    of an occlusion-split instance plus hole boundaries — into ONE polygon via thin zero-width bridges
    (`merge_multi_segment`). This fixes the two systematic label defects of the previous RETR_EXTERNAL +
    one-line-per-contour approach: filled-in holes and one instance being split into several label instances.

    Returns:
        (line, n_contours): The label line (or None when no valid polygon remains) and the number of kept contours.
    """
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None, 0
    polygons: list[np.ndarray] = []
    for contour, info in zip(contours, hierarchy[0]):
        # Holes inside dropped tiny fragments are even smaller, so one area filter covers outers and holes alike.
        if len(contour) < 3 or cv2.contourArea(contour) < min_area:
            continue
        points = simplify_contour(contour, approx_epsilon)
        if points is not None:
            polygons.append(points)
    if not polygons:
        return None, 0
    merged = polygons[0] if len(polygons) == 1 else np.concatenate(merge_multi_segment(polygons), axis=0)
    if len(merged) < 3:
        return None, 0
    values: list[str] = [str(cls)]
    for x, y in merged:
        xn = min(1.0, max(0.0, float(x) / width))
        yn = min(1.0, max(0.0, float(y) / height))
        values.append(f"{xn:.6f}")
        values.append(f"{yn:.6f}")
    return " ".join(values), len(polygons)


def convert_one(task: dict[str, Any]) -> dict[str, Any]:
    """Convert one panoptic mask to one YOLO label file."""
    image_file = task["image_file"]
    mask_path = Path(task["mask_path"])
    label_path = Path(task["label_path"])
    image_path = Path(task["image_path"])
    category_to_index = task["category_to_index"]
    width = int(task["width"])
    height = int(task["height"])
    min_area = int(task["min_area"])
    approx_epsilon = float(task["approx_epsilon"])

    result = {
        "image": image_file,
        "image_subdir": task["image_subdir"],
        "written": False,
        "included": False,
        "missing_image": 0,
        "missing_mask": 0,
        "thing_segments": 0,
        "written_segments": 0,
        "skipped_segments": 0,
        "contours": 0,
        "points": 0,
        "per_class": {},
    }
    if not image_path.exists():
        result["missing_image"] = 1
        return result
    if not mask_path.exists():
        result["missing_mask"] = 1
        return result

    mask = rgb_to_id(np.asarray(Image.open(mask_path).convert("RGB")))
    if mask.shape[:2] != (height, width):
        height, width = mask.shape[:2]

    lines: list[str] = []
    for segment in task["segments_info"]:
        if int(segment.get("isthing", 0)) != 1 or int(segment.get("iscrowd", 0)) == 1:
            continue
        cls = category_to_index.get(int(segment["category_id"]))
        if cls is None:
            continue
        result["thing_segments"] += 1
        binary = (mask == int(segment["id"])).astype(np.uint8)
        if not binary.any():
            result["skipped_segments"] += 1
            continue
        line, n_contours = segment_to_line(binary, cls, width, height, min_area, approx_epsilon)
        if line is None:
            result["skipped_segments"] += 1
            continue
        lines.append(line)  # one line per instance, holes and disconnected parts merged
        result["written_segments"] += 1
        result["contours"] += n_contours
        result["points"] += (len(line.split()) - 1) // 2
        cls_key = str(cls)
        result["per_class"][cls_key] = result["per_class"].get(cls_key, 0) + 1

    label_path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["written"] = True
    else:
        label_path.write_text("", encoding="utf-8")
    result["included"] = bool(lines) or not task["drop_empty"]
    return result


def resolve_image_path(image_root: Path, image_file: str, image_subdirs: list[str]) -> tuple[Path | None, str | None]:
    """Resolve a COCO image under one of the allowed subdirectories."""
    for subdir in image_subdirs:
        candidate = image_root / subdir / image_file
        if candidate.exists():
            return candidate, subdir
    return None, None


def build_tasks(
    *,
    annotations: list[dict[str, Any]],
    images_by_id: dict[int, dict[str, Any]],
    split_name: str,
    image_subdirs: list[str],
    coconut_root: Path,
    image_root: Path,
    out_root: Path,
    category_to_index: dict[int, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for ann in annotations:
        image = images_by_id.get(int(ann["image_id"]))
        if image is None:
            continue
        image_file = str(image["file_name"])
        image_path, image_subdir = resolve_image_path(image_root, image_file, image_subdirs)
        if image_path is None or image_subdir is None:
            continue
        label_file = Path(image_file).with_suffix(".txt").name
        tasks.append(
            {
                "image_file": image_file,
                "image_subdir": image_subdir,
                "image_path": str(image_path),
                "mask_path": str(coconut_root / split_name / ann["file_name"]),
                "label_path": str(out_root / "labels" / image_subdir / label_file),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "segments_info": ann.get("segments_info", []),
                "category_to_index": category_to_index,
                "min_area": args.min_area,
                "approx_epsilon": args.approx_epsilon,
                "drop_empty": args.drop_empty,
            }
        )
        if args.limit and len(tasks) >= args.limit:
            break
    return tasks


def convert_split(tasks: list[dict[str, Any]], manifest_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "input_images": len(tasks),
        "output_images": 0,
        "missing_images": 0,
        "missing_masks": 0,
        "images_with_labels": 0,
        "empty_images": 0,
        "thing_segments": 0,
        "written_segments": 0,
        "skipped_segments": 0,
        "contours": 0,
        "points": 0,
        "per_class_instances": {},
    }
    manifest: list[str] = []
    if not tasks:
        manifest_path.write_text("", encoding="utf-8")
        return stats

    def consume(result: dict[str, Any]) -> None:
        stats["missing_images"] += int(result["missing_image"])
        stats["missing_masks"] += int(result["missing_mask"])
        stats["thing_segments"] += int(result["thing_segments"])
        stats["written_segments"] += int(result["written_segments"])
        stats["skipped_segments"] += int(result["skipped_segments"])
        stats["contours"] += int(result["contours"])
        stats["points"] += int(result["points"])
        if result["included"]:
            manifest.append(f"./images/{result['image_subdir']}/{result['image']}")
        if result["written"]:
            stats["images_with_labels"] += 1
        elif result["included"]:
            stats["empty_images"] += 1
        for cls, count in result["per_class"].items():
            stats["per_class_instances"][cls] = stats["per_class_instances"].get(cls, 0) + count

    if args.workers <= 1:
        for i, result in enumerate(map(convert_one, tasks), 1):
            consume(result)
            if i % 5000 == 0:
                print(f"converted {i}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i, result in enumerate(executor.map(convert_one, tasks, chunksize=args.chunksize), 1):
                consume(result)
                if i % 5000 == 0:
                    print(f"converted {i}/{len(tasks)}")

    manifest_path.write_text("\n".join(manifest) + ("\n" if manifest else ""), encoding="utf-8")
    stats["output_images"] = len(manifest)
    if stats["contours"]:
        stats["mean_points_per_contour"] = stats["points"] / stats["contours"]
    return stats


def load_panoptic_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    coconut_root = args.coconut_root.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    train_json_path = coconut_root / f"{args.train_split}.json"
    val_json_path = coconut_root / f"{args.val_split}.json"

    if not train_json_path.exists():
        raise FileNotFoundError(train_json_path)
    if not val_json_path.exists():
        raise FileNotFoundError(val_json_path)

    out_root.mkdir(parents=True, exist_ok=True)
    ensure_symlink(image_root / "train2017", out_root / "images" / "train2017")
    ensure_symlink(image_root / "val2017", out_root / "images" / "val2017")
    if (image_root / "unlabeled2017").exists():
        ensure_symlink(image_root / "unlabeled2017", out_root / "images" / "unlabeled2017")
    if args.overwrite and (out_root / "labels").exists():
        shutil.rmtree(out_root / "labels")

    train_data = load_panoptic_json(train_json_path)
    val_data = load_panoptic_json(val_json_path)
    names, category_to_index, category_by_id = load_coco80_categories(train_data["categories"])

    train_image_subdirs = ["train2017", "unlabeled2017"] if args.train_split == "coconut_b" else ["train2017"]
    train_tasks = build_tasks(
        annotations=train_data["annotations"],
        images_by_id={int(x["id"]): x for x in train_data["images"]},
        split_name=args.train_split,
        image_subdirs=train_image_subdirs,
        coconut_root=coconut_root,
        image_root=image_root,
        out_root=out_root,
        category_to_index=category_to_index,
        args=args,
    )
    val_tasks = build_tasks(
        annotations=val_data["annotations"],
        images_by_id={int(x["id"]): x for x in val_data["images"]},
        split_name=args.val_split,
        image_subdirs=["val2017"],
        coconut_root=coconut_root,
        image_root=image_root,
        out_root=out_root,
        category_to_index=category_to_index,
        args=args,
    )

    print(f"converting train split {args.train_split}: {len(train_tasks)} masks")
    train_stats = convert_split(train_tasks, out_root / "train.txt", args)
    print(f"converting val split {args.val_split}: {len(val_tasks)} masks")
    val_stats = convert_split(val_tasks, out_root / "val.txt", args)

    data_yaml = {
        "path": str(out_root),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(names),
        "names": {i: name for i, name in enumerate(names)},
    }
    yaml_path = out_root / f"{args.train_split.replace('_', '-')}-seg.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    summary = {
        "coconut_root": str(coconut_root),
        "image_root": str(image_root),
        "output_root": str(out_root),
        "yaml": str(yaml_path),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "classes": len(names),
        "thing_categories": {
            str(i): {"name": name, "category_id": int(cat_id), "source": category_by_id[int(cat_id)]}
            for cat_id, i in category_to_index.items()
            for name in [names[i]]
        },
        "args": {
            "workers": args.workers,
            "chunksize": args.chunksize,
            "min_area": args.min_area,
            "approx_epsilon": args.approx_epsilon,
            "drop_empty": args.drop_empty,
            "limit": args.limit,
        },
        "splits": {"train": train_stats, "val": val_stats},
    }
    (out_root / "coconut_yolo_seg_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
