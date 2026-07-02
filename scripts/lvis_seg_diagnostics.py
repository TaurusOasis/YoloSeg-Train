#!/usr/bin/env python3
"""LVIS segmentation diagnostics for YOLO prediction JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-json", required=True, type=Path, help="YOLO/COCO-format predictions.json path.")
    parser.add_argument("--anno-json", required=True, type=Path, help="LVIS annotation JSON path.")
    parser.add_argument("--data", type=Path, help="YOLO data YAML used to resolve val image IDs and names.")
    parser.add_argument("--split", default="val", help="Dataset split key in the YOLO data YAML.")
    parser.add_argument("--top-k", default=20, type=int, help="Number of top/bottom classes to include.")
    parser.add_argument("--out-json", type=Path, help="Output diagnostic JSON path.")
    parser.add_argument("--out-md", type=Path, help="Output Markdown report path.")
    parser.add_argument("--verbose", action="store_true", help="Print faster-coco-eval progress.")
    return parser.parse_args()


def as_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute() and base is not None:
        p = base / p
    return p.resolve()


def load_data_yaml(data_path: Path | None) -> dict[str, Any]:
    if not data_path:
        return {}
    with data_path.expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def resolve_split_items(data: dict[str, Any], split: str) -> list[str]:
    value = data.get(split)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for part in value:
            items.extend(resolve_split_path(data, str(part)))
        return items
    return resolve_split_path(data, str(value))


def resolve_split_path(data: dict[str, Any], value: str) -> list[str]:
    root = as_path(data.get("path", ".")) if data.get("path") else Path.cwd()
    split_path = as_path(value, root)
    if split_path.is_file():
        return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split_path.is_dir():
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return [str(p) for p in sorted(split_path.rglob("*")) if p.suffix.lower() in suffixes]
    return [value]


def load_img_ids(data: dict[str, Any], split: str) -> list[int]:
    ids = []
    for item in resolve_split_items(data, split):
        stem = Path(item).stem
        try:
            ids.append(int(stem))
        except ValueError:
            continue
    return sorted(set(ids))


def finite_mean(values: np.ndarray) -> float:
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def metric_at_iou(precision: np.ndarray, iou_thrs: np.ndarray, cat_idx: int, iou: float) -> float:
    iou_idx = np.where(np.isclose(iou_thrs, iou))[0]
    if iou_idx.size == 0:
        return float("nan")
    return finite_mean(precision[iou_idx, :, cat_idx, 0, -1])


def per_class_ap(eval_obj: Any, categories: dict[int, dict[str, Any]], names: dict[int, str]) -> list[dict[str, Any]]:
    precision = eval_obj.eval["precision"]
    cat_ids = list(eval_obj.params.catIds)
    iou_thrs = np.asarray(eval_obj.params.iouThrs)
    rows = []
    for cat_idx, cat_id in enumerate(cat_ids):
        cat = categories.get(int(cat_id), {})
        class_index = int(cat_id) - 1
        ap = finite_mean(precision[:, :, cat_idx, 0, -1])
        ap50 = metric_at_iou(precision, iou_thrs, cat_idx, 0.50)
        ap75 = metric_at_iou(precision, iou_thrs, cat_idx, 0.75)
        rows.append(
            {
                "cat_id": int(cat_id),
                "class_index": class_index,
                "name": cat.get("name") or names.get(class_index, str(cat_id)),
                "yaml_name": names.get(class_index),
                "frequency": cat.get("frequency"),
                "image_count": cat.get("image_count"),
                "instance_count": cat.get("instance_count"),
                "AP": ap,
                "AP50": ap50,
                "AP75": ap75,
            }
        )
    return rows


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def compact_stats(stats: dict[str, Any]) -> dict[str, float | None]:
    keys = ["AP_all", "AP_50", "AP_75", "APr", "APc", "APf", "AR_all", "AR_50", "AR_75"]
    return {k: safe_float(stats.get(k)) for k in keys if k in stats}


def run_eval(
    anno: Any,
    pred: Any,
    iou_type: str,
    img_ids: list[int],
    categories: dict[int, dict[str, Any]],
    names: dict[int, str],
    verbose: bool,
) -> dict[str, Any]:
    from faster_coco_eval import COCOeval_faster

    printer = print if verbose else (lambda *_args, **_kwargs: None)
    evaluator = COCOeval_faster(anno, pred, iouType=iou_type, lvis_style=True, print_function=printer)
    if img_ids:
        evaluator.params.imgIds = img_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    rows = per_class_ap(evaluator, categories, names)
    valid = [r for r in rows if safe_float(r["AP"]) is not None]
    top = sorted(valid, key=lambda x: x["AP"], reverse=True)
    bottom = sorted(valid, key=lambda x: x["AP"])
    nonzero = [r for r in bottom if r["AP"] > 0]
    return {
        "stats": compact_stats(evaluator.stats_as_dict),
        "per_class": rows,
        "top": top,
        "bottom": bottom,
        "bottom_nonzero": nonzero,
    }


def pct(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value * 100:.2f}"


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def limited(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return rows[: max(0, top_k)]


def table_rows(rows: list[dict[str, Any]]) -> str:
    lines = ["| rank | cat_id | idx | freq | name | AP | AP50 | AP75 | inst |", "|---:|---:|---:|:---:|---|---:|---:|---:|---:|"]
    for i, row in enumerate(rows, 1):
        name = str(row.get("name") or "").replace("|", "/")
        lines.append(
            f"| {i} | {row.get('cat_id')} | {row.get('class_index')} | {row.get('frequency') or ''} | "
            f"{name} | {pct(safe_float(row.get('AP')))} | {pct(safe_float(row.get('AP50')))} | "
            f"{pct(safe_float(row.get('AP75')))} | {row.get('instance_count') or ''} |"
        )
    return "\n".join(lines)


def build_interpretation(report: dict[str, Any]) -> list[str]:
    box = report["bbox"]["stats"]
    mask = report["segm"]["stats"]
    notes = []

    box_ap = box.get("AP_all")
    mask_ap = mask.get("AP_all")
    mask_box_ratio = ratio(mask_ap, box_ap)
    if mask_box_ratio is not None:
        if mask_box_ratio >= 0.75:
            notes.append(
                f"Mask/Box mAP50-95 ratio is {mask_box_ratio:.3f}; mask quality is trailing box, but not enough to be the primary bottleneck."
            )
        else:
            notes.append(
                f"Mask/Box mAP50-95 ratio is only {mask_box_ratio:.3f}; mask proto or mask decoding quality needs priority inspection."
            )

    for label, stats in [("Box", box), ("Mask", mask)]:
        ap50 = stats.get("AP_50")
        ap75 = stats.get("AP_75")
        if ap50 is not None and ap75 is not None:
            notes.append(f"{label} AP50 to AP75 drop is {(ap50 - ap75) * 100:.2f} points.")

    for label, stats in [("Box", box), ("Mask", mask)]:
        apr, apc, apf = stats.get("APr"), stats.get("APc"), stats.get("APf")
        if apr is not None and apf is not None:
            gap = (apf - apr) * 100
            if gap > 5:
                notes.append(f"{label} APf-APr gap is {gap:.2f} points; LVIS long-tail learning is a major limiter.")
            else:
                notes.append(f"{label} APf-APr gap is {gap:.2f} points; long-tail imbalance is not the dominant split-level issue.")
        if apr is not None and apc is not None and apf is not None:
            worst = min([("rare", apr), ("common", apc), ("frequent", apf)], key=lambda x: x[1])
            notes.append(f"{label} weakest frequency split is {worst[0]} at {worst[1] * 100:.2f} AP.")

    return notes


def build_markdown(report: dict[str, Any], top_k: int) -> str:
    box = report["bbox"]["stats"]
    mask = report["segm"]["stats"]
    lines = [
        "# LVIS YOLO26 Seg Diagnostics",
        "",
        f"- predictions: `{report['inputs']['pred_json']}`",
        f"- annotations: `{report['inputs']['anno_json']}`",
        f"- evaluated images: {report['inputs']['num_img_ids'] or 'annotation default'}",
        "",
        "## Summary",
        "",
        "| metric | Box | Mask | Mask-Box | Mask/Box |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["AP_all", "AP_50", "AP_75", "APr", "APc", "APf"]:
        b = box.get(key)
        m = mask.get(key)
        lines.append(
            f"| {key} | {pct(b)} | {pct(m)} | {pct(delta(m, b))} | "
            f"{(ratio(m, b) or float('nan')):.3f} |"
        )

    lines.extend(["", "## Interpretation", ""])
    lines.extend([f"- {note}" for note in report["interpretation"]])

    for section, title in [("bbox", "Box"), ("segm", "Mask")]:
        lines.extend(
            [
                "",
                f"## {title} Per-Class Top {top_k}",
                "",
                table_rows(limited(report[section]["top"], top_k)),
                "",
                f"## {title} Per-Class Bottom {top_k}",
                "",
                table_rows(limited(report[section]["bottom"], top_k)),
                "",
                f"## {title} Per-Class Bottom Nonzero {top_k}",
                "",
                table_rows(limited(report[section]["bottom_nonzero"], top_k)),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pred_json = args.pred_json.expanduser().resolve()
    anno_json = args.anno_json.expanduser().resolve()
    out_json = args.out_json.expanduser().resolve() if args.out_json else pred_json.with_name("lvis_diagnostics.json")
    out_md = args.out_md.expanduser().resolve() if args.out_md else pred_json.with_name("lvis_diagnostics.md")

    if not pred_json.is_file():
        raise FileNotFoundError(f"prediction JSON not found: {pred_json}")
    if not anno_json.is_file():
        raise FileNotFoundError(f"annotation JSON not found: {anno_json}")

    from faster_coco_eval import COCO

    data = load_data_yaml(args.data)
    names = normalize_names(data.get("names"))
    img_ids = load_img_ids(data, args.split)
    anno = COCO(anno_json)
    pred = anno.loadRes(pred_json)
    categories = {int(c["id"]): c for c in anno.dataset.get("categories", [])}

    report = {
        "inputs": {
            "pred_json": str(pred_json),
            "anno_json": str(anno_json),
            "data": str(args.data.expanduser().resolve()) if args.data else None,
            "split": args.split,
            "num_img_ids": len(img_ids),
        },
        "bbox": run_eval(anno, pred, "bbox", img_ids, categories, names, args.verbose),
        "segm": run_eval(anno, pred, "segm", img_ids, categories, names, args.verbose),
    }
    report["interpretation"] = build_interpretation(report)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report, args.top_k), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
