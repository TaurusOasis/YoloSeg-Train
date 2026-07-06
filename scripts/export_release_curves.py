#!/usr/bin/env python3
"""Export training curves from results.csv for GitHub Releases documentation.

Curves match SwanLab-tracked val metrics (Ultralytics logs the same columns to SwanLab
when ULTRALYTICS_SWANLAB is enabled). recipe200 was trained without a local swanlab/
folder but metrics are identical to what SwanLab would record.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs" / "segment"
OUT = REPO_ROOT / "docs" / "releases" / "curves"

STAGES = [
    {
        "id": "stage-a-lvis-pretrain",
        "title": "Stage A · LVIS Pretrain",
        "run": "yolo26s-seg-lvis-b48-bf16-swanlab",
        "swanlab": True,
        "val_note": "LVIS val (1203 classes)",
    },
    {
        "id": "stage-b-lvis-coco80-distill",
        "title": "Stage B · LVIS→COCO80 Distill",
        "run": "yolo26s-seg-lvis-coco80-distill-x-teacher-b80-2gpu",
        "swanlab": True,
        "val_note": "LVIS COCO80-subset val",
    },
    {
        "id": "stage-c-coconut-v1-distill",
        "title": "Stage C · COCONut-B v1 Distill",
        "run": "yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu",
        "swanlab": True,
        "val_note": "COCONut-B v1 val",
    },
    {
        "id": "stage-d-recipe200-v2",
        "title": "Stage D · COCONut-B v2 Recipe200",
        "run": "yolo26s-seg-coconut-b-v2-distill-recipe200",
        "swanlab": False,
        "val_note": "COCONut-B v2 val (5000 img)",
    },
    {
        "id": "stage-e-pointrend-ft60",
        "title": "Stage E · PointRend Finetune (in progress)",
        "run": "yolo26s-seg-coconut-b-v2-pointrend-ft60",
        "swanlab": True,
        "val_note": "COCONut-B v2 val; run ongoing at export time",
    },
]


def load_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return list(csv.DictReader(path.open(encoding="utf-8")))


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda r: float(r["metrics/mAP50-95(M)"]))


def export_curves() -> list[dict]:
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for stage in STAGES:
        run_dir = RUNS / stage["run"]
        rows = load_rows(run_dir)
        best = best_row(rows)
        epochs = [int(r["epoch"]) for r in rows]
        mask5095 = [float(r["metrics/mAP50-95(M)"]) for r in rows]
        mask50 = [float(r["metrics/mAP50(M)"]) for r in rows]
        box5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        seg_loss = [float(r["train/seg_loss"]) for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(stage["title"], fontsize=12, fontweight="bold")

        ax = axes[0]
        ax.plot(epochs, mask5095, "o-", label="Mask mAP50-95", linewidth=2, markersize=3)
        ax.plot(epochs, mask50, "s-", label="Mask mAP50", linewidth=1.5, markersize=3, alpha=0.85)
        ax.plot(epochs, box5095, "^-", label="Box mAP50-95", linewidth=1.5, markersize=3, alpha=0.85)
        be = int(best["epoch"])
        ax.axvline(be, color="crimson", linestyle="--", alpha=0.5, label=f"best ep{be}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_title(f"Val metrics · {stage['val_note']}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

        ax = axes[1]
        ax.plot(epochs, seg_loss, "o-", color="tab:orange", linewidth=2, markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("train seg_loss")
        ax.set_title("Training seg_loss (incl. point sub-loss when enabled)")
        ax.grid(True, alpha=0.3)

        out_png = OUT / f"{stage['id']}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        entry = {
            **stage,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "epochs_completed": len(rows),
            "best_epoch": int(best["epoch"]),
            "best_mask_map50_95": round(float(best["metrics/mAP50-95(M)"]), 5),
            "best_mask_map50": round(float(best["metrics/mAP50(M)"]), 5),
            "best_box_map50_95": round(float(best["metrics/mAP50-95(B)"]), 5),
            "curve_png": str(out_png.relative_to(REPO_ROOT)),
            "best_pt_local": str((run_dir / "weights" / "best.pt").relative_to(REPO_ROOT)),
            "best_pt_size_mb": round((run_dir / "weights" / "best.pt").stat().st_size / 1024 / 1024, 1),
        }
        manifest.append(entry)
        print(f"Wrote {out_png.name}  best_mask5095={entry['best_mask_map50_95']} ep{entry['best_epoch']}")

    # Pipeline overview
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [s["id"].replace("stage-", "").replace("-", "\n") for s in manifest]
    vals = [s["best_mask_map50_95"] for s in manifest]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best Mask mAP50-95")
    ax.set_title("YOLO26s-seg COCONut pipeline · best checkpoint per stage")
    ax.set_ylim(0, max(vals) * 1.15)
    for bar, v, s in zip(bars, vals, manifest):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}\nep{s['best_epoch']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(True, axis="y", alpha=0.3)
    overview = OUT / "pipeline-overview.png"
    fig.tight_layout()
    fig.savefig(overview, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {overview.name}")

    manifest_path = REPO_ROOT / "docs" / "releases" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    export_curves()
