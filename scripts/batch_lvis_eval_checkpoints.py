#!/usr/bin/env python3
"""Batch LVIS JSON evaluation for saved YOLO segmentation checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", required=True, type=Path, help="Directory containing .pt checkpoints.")
    parser.add_argument("--data", required=True, type=Path, help="YOLO data YAML.")
    parser.add_argument("--anno-json", required=True, type=Path, help="LVIS annotation JSON.")
    parser.add_argument("--project", required=True, type=Path, help="Output project directory.")
    parser.add_argument("--summary-csv", type=Path, help="Aggregate CSV path.")
    parser.add_argument("--weights", nargs="*", help="Specific checkpoint filenames or paths to evaluate.")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--top-k", default=30, type=int)
    parser.add_argument("--yolo-bin", default=shutil.which("yolo") or "yolo")
    parser.add_argument("--keep-predictions", action="store_true", help="Keep predictions.json files.")
    parser.add_argument("--force", action="store_true", help="Re-run checkpoints with existing diagnostics.")
    return parser.parse_args()


def epoch_key(path: Path) -> tuple[int, int, str]:
    if path.stem == "best":
        return (1, 10_000, path.stem)
    if path.stem == "last":
        return (2, 10_001, path.stem)
    m = re.fullmatch(r"epoch(\d+)", path.stem)
    if m:
        return (0, int(m.group(1)), path.stem)
    return (3, 10_002, path.stem)


def resolve_weights(weights_dir: Path, names: list[str] | None) -> list[Path]:
    if names:
        paths = []
        for name in names:
            p = Path(name).expanduser()
            if not p.is_absolute():
                p = weights_dir / p
            paths.append(p.resolve())
    else:
        paths = sorted(weights_dir.glob("epoch*.pt"), key=epoch_key)
        for special in ("best.pt", "last.pt"):
            p = weights_dir / special
            if p.is_file():
                paths.append(p)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    return paths


def checkpoint_label(path: Path) -> str:
    key = epoch_key(path)
    if key[0] == 0:
        return f"epoch{key[1]:03d}"
    return path.stem


def run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_stats(diag_json: Path) -> dict[str, Any]:
    report = json.loads(diag_json.read_text(encoding="utf-8"))
    row: dict[str, Any] = {"checkpoint": diag_json.parent.name.replace("lvis-eval-", "")}
    for section, prefix in (("bbox", "box"), ("segm", "mask")):
        stats = report[section]["stats"]
        for key in ("AP_all", "AP_50", "AP_75", "APr", "APc", "APf", "AR_all", "AR_50", "AR_75"):
            row[f"{prefix}_{key}"] = stats.get(key)
    box_ap = row.get("box_AP_all")
    mask_ap = row.get("mask_AP_all")
    row["mask_box_ratio"] = (mask_ap / box_ap) if box_ap else None
    return row


def write_summary(rows: list[dict[str, Any]], summary_csv: Path) -> None:
    if not rows:
        return
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    weights_dir = args.weights_dir.expanduser().resolve()
    data = args.data.expanduser().resolve()
    anno_json = args.anno_json.expanduser().resolve()
    project = args.project.expanduser().resolve()
    summary_csv = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv
        else project / "lvis_batch_eval_summary.csv"
    )
    project.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for weight in resolve_weights(weights_dir, args.weights):
        label = checkpoint_label(weight)
        name = f"lvis-eval-{label}"
        out_dir = project / name
        pred_json = out_dir / "predictions.json"
        diag_json = out_dir / "lvis_diagnostics.json"
        diag_md = out_dir / "lvis_diagnostics.md"

        if not diag_json.is_file() or args.force:
            run(
                [
                    args.yolo_bin,
                    "segment",
                    "val",
                    f"model={weight}",
                    f"data={data}",
                    "split=val",
                    f"imgsz={args.imgsz}",
                    f"batch={args.batch}",
                    f"device={args.device}",
                    f"workers={args.workers}",
                    "save_json=True",
                    "plots=False",
                    f"project={project}",
                    f"name={name}",
                    "exist_ok=True",
                ],
                repo,
            )
            run(
                [
                    sys.executable,
                    str(repo / "scripts/lvis_seg_diagnostics.py"),
                    "--pred-json",
                    str(pred_json),
                    "--anno-json",
                    str(anno_json),
                    "--data",
                    str(data),
                    "--split",
                    "val",
                    "--top-k",
                    str(args.top_k),
                    "--out-json",
                    str(diag_json),
                    "--out-md",
                    str(diag_md),
                ],
                repo,
            )
            if pred_json.is_file() and not args.keep_predictions:
                pred_json.unlink()
        else:
            print(f"skip existing {diag_json}", flush=True)

        rows.append(load_stats(diag_json))
        write_summary(rows, summary_csv)
        print(f"updated {summary_csv}", flush=True)

    print(f"done: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
