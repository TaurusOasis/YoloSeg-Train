#!/usr/bin/env python3
"""Build an LVIS segmentation subset that keeps only COCO80-overlap classes.

The generated dataset reuses the original LVIS images through an ``images`` symlink and writes fresh YOLO segment label
files with class ids remapped to a dense 0..N-1 COCO-name order. COCO classes without a direct LVIS v1 category
(``hot dog`` and ``potted plant``) are intentionally skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

try:
    from remap_coco80_predictions_to_lvis import COCO80_NAMES, COCO80_TO_LVIS_ID
except ImportError:  # pragma: no cover - supports running as `python -m scripts...`
    from scripts.remap_coco80_predictions_to_lvis import COCO80_NAMES, COCO80_TO_LVIS_ID


DEFAULT_SRC_ROOT = Path("/home/genesis/Train/Dataset/LVIS_yolo_seg")
DEFAULT_OUT_ROOT = Path("/home/genesis/Train/Dataset/LVIS_coco80_yolo_seg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-root", type=Path, default=DEFAULT_SRC_ROOT, help="Source LVIS YOLO segment dataset root."
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output filtered dataset root.")
    parser.add_argument("--src-yaml", type=Path, help="Source LVIS data YAML. Defaults to SRC_ROOT/lvis-seg.yaml.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to convert.")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Keep images whose labels become empty after filtering. Default keeps only images with valid labels.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output labels before conversion.")
    return parser.parse_args()


def resolve_item(root: Path, item: str) -> Path:
    path = Path(item).expanduser()
    return path if path.is_absolute() else (root / path).absolute()


def rel_to_root(root: Path, path: Path) -> Path:
    try:
        return path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise ValueError(f"Image path {path} is outside source root {root}") from exc


def image_to_label(root: Path, image_path: Path) -> Path:
    rel = rel_to_root(root, image_path)
    parts = list(rel.parts)
    if not parts or parts[0] != "images":
        raise ValueError(f"Expected image path under {root / 'images'}, got {image_path}")
    parts[0] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


def convert_label_file(src_label: Path, dst_label: Path, class_map: dict[int, int]) -> tuple[int, int]:
    kept, dropped = 0, 0
    out_lines: list[str] = []
    if src_label.exists():
        for line in src_label.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                old_cls = int(float(parts[0]))
            except ValueError:
                dropped += 1
                continue
            new_cls = class_map.get(old_cls)
            if new_cls is None:
                dropped += 1
                continue
            out_lines.append(" ".join([str(new_cls), *parts[1:]]))
            kept += 1

    if out_lines:
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return kept, dropped


def ensure_image_link(src_root: Path, out_root: Path) -> None:
    src_images = src_root / "images"
    dst_images = out_root / "images"
    if dst_images.exists() or dst_images.is_symlink():
        if dst_images.is_symlink() and dst_images.resolve() == src_images.resolve():
            return
        raise FileExistsError(f"{dst_images} already exists and is not the expected symlink to {src_images}")
    out_root.mkdir(parents=True, exist_ok=True)
    os.symlink(src_images, dst_images, target_is_directory=True)


def load_split_items(src_root: Path, src_yaml: dict[str, Any], split: str) -> list[str]:
    split_value = src_yaml.get(split)
    if split_value is None:
        raise KeyError(f"Split {split!r} not found in source YAML")
    split_path = resolve_item(src_root, str(split_value))
    if not split_path.is_file():
        raise FileNotFoundError(f"Expected split manifest file, got {split_path}")
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_subset(args: argparse.Namespace) -> dict[str, Any]:
    src_root = args.src_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    src_yaml_path = (args.src_yaml or (src_root / "lvis-seg.yaml")).expanduser().resolve()
    src_yaml = yaml.safe_load(src_yaml_path.read_text(encoding="utf-8")) or {}

    mapped_names = [name for name in COCO80_NAMES if name in COCO80_TO_LVIS_ID]
    lvis_class_to_subset = {int(COCO80_TO_LVIS_ID[name]) - 1: i for i, name in enumerate(mapped_names)}
    {int(COCO80_TO_LVIS_ID[name]): i for i, name in enumerate(mapped_names)}

    out_root.mkdir(parents=True, exist_ok=True)
    ensure_image_link(src_root, out_root)
    if args.overwrite and (out_root / "labels").exists():
        shutil.rmtree(out_root / "labels")

    summary: dict[str, Any] = {
        "source_root": str(src_root),
        "output_root": str(out_root),
        "source_yaml": str(src_yaml_path),
        "classes": len(mapped_names),
        "dropped_coco_classes": [name for name in COCO80_NAMES if name not in COCO80_TO_LVIS_ID],
        "mapping": {
            name: {
                "subset_index": i,
                "coco80_index": COCO80_NAMES.index(name),
                "lvis_category_id": int(COCO80_TO_LVIS_ID[name]),
                "source_lvis_class_index": int(COCO80_TO_LVIS_ID[name]) - 1,
            }
            for i, name in enumerate(mapped_names)
        },
        "splits": {},
    }

    for split in args.splits:
        items = load_split_items(src_root, src_yaml, split)
        out_items: list[str] = []
        split_stats = {
            "input_images": len(items),
            "output_images": 0,
            "kept_instances": 0,
            "dropped_instances": 0,
            "empty_after_filter": 0,
            "missing_source_labels": 0,
            "per_class_instances": {str(i): 0 for i in range(len(mapped_names))},
        }
        for item in items:
            image_path = resolve_item(src_root, item)
            src_label = image_to_label(src_root, image_path)
            if not src_label.exists():
                split_stats["missing_source_labels"] += 1

            rel = rel_to_root(src_root, image_path)
            dst_label = (out_root / rel).with_suffix(".txt")
            dst_label = Path(str(dst_label).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
            kept, dropped = convert_label_file(src_label, dst_label, lvis_class_to_subset)
            split_stats["kept_instances"] += kept
            split_stats["dropped_instances"] += dropped
            if kept:
                for line in dst_label.read_text(encoding="utf-8").splitlines():
                    cls = line.split(maxsplit=1)[0]
                    split_stats["per_class_instances"][cls] += 1
            else:
                split_stats["empty_after_filter"] += 1
                if args.include_empty:
                    dst_label.parent.mkdir(parents=True, exist_ok=True)
                    dst_label.write_text("", encoding="utf-8")
                else:
                    continue

            out_items.append("./" + rel.as_posix())

        (out_root / f"{split}.txt").write_text("\n".join(out_items) + ("\n" if out_items else ""), encoding="utf-8")
        split_stats["output_images"] = len(out_items)
        summary["splits"][split] = split_stats

    data_yaml = {
        "path": str(out_root),
        "train": "train.txt" if "train" in args.splits else None,
        "val": "val.txt" if "val" in args.splits else None,
        "names": {i: name for i, name in enumerate(mapped_names)},
    }
    data_yaml = {k: v for k, v in data_yaml.items() if v is not None}
    (out_root / "lvis-coco80-seg.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (out_root / "coco80_lvis_mapping.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = build_subset(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
