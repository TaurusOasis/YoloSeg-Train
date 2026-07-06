#!/usr/bin/env python3
"""Val-twice acceptance for PointRend inference subdivision (design doc §2.8).

Run validation on the same checkpoint twice:
  - indirect: seg_point_refine_infer=False  -> standard process_mask (deterministic baseline)
  - direct:   seg_point_refine_infer=True   -> process_mask_pointrend subdivision (stochastic)

The delta (direct - indirect) on Mask mAP50-95 / mAP75 / AP95 isolates the PointRend point-head's
*direct* inference benefit. Training-time val keeps seg_point_refine_infer=False so best.pt fitness
is not polluted by subdivision randomness; this script measures the post-hoc direct benefit.

Subdivision samples points with torch.rand, so direct metrics are stochastic. Pass --seeds N>1 to
average over N seeds and report mean ± std. Box metrics are collected as a sanity check (they must
not depend on the mask postprocess, so indirect == direct within noise).

Examples:
  # Single seed, GPU 0
  python scripts/val_twice_pointrend.py \
      --ckpt runs/segment/yolo26s-seg-coconut-b-v2-pointrend-ft60/weights/best.pt \
      --data /home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2/coconut-b-seg.yaml \
      --device 0 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ultralytics import YOLO


def _seg_metrics(m) -> dict[str, float]:
    """Return the mask metrics compared across the indirect/direct fork."""
    import numpy as np

    all_ap = getattr(m.seg, "all_ap", None)
    all_ap = np.asarray(all_ap) if all_ap is not None else np.empty(0)
    ap95 = float(all_ap[:, 9].mean()) if all_ap.size and all_ap.shape[1] > 9 else float("nan")
    return {
        "map50-95": float(m.seg.map),
        "map50": float(m.seg.map50),
        "map75": float(m.seg.map75),
        "ap95": ap95,
    }


def _box_metrics(m) -> dict[str, float]:
    """Return box metrics as a sanity check (must be invariant to mask postprocess)."""
    return {"box_map50-95": float(m.box.map), "box_map50": float(m.box.map50), "box_map75": float(m.box.map75)}


def _val_once(ckpt: str, data: str, imgsz: int, device: str, seed: int, direct: bool, subdiv_k: int):
    """Run one validation pass with a fixed seed and return the metrics namespace."""
    torch.manual_seed(seed)
    return YOLO(ckpt).val(
        data=data,
        imgsz=imgsz,
        device=device,
        seg_point_refine_infer=direct,
        seg_point_subdiv_k=subdiv_k,
        project="runs/segment",
        name=f"val_twice_{'direct' if direct else 'indirect'}_s{seed}",
        save_json=False,
        save_txt=False,
        verbose=False,
    )


def _aggregate(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate a list of metric dicts into {key: {mean, std, n}}."""
    out = {}
    keys = records[0].keys()
    for k in keys:
        vals = [r[k] for r in records if r[k] == r[k]]  # drop NaN
        out[k] = {
            "mean": statistics.fmean(vals) if vals else float("nan"),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def main() -> None:
    """Run the val-twice acceptance protocol and print + dump the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Path to the trained checkpoint (best.pt / last.pt).")
    parser.add_argument("--data", required=True, help="Dataset yaml path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--device", default="0", help="Device spec passed to YOLO.val (e.g. 0, cpu).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to average direct metrics over.")
    parser.add_argument("--subdiv-k", type=int, default=3, help="Inference subdivision passes (seg_point_subdiv_k).")
    parser.add_argument("--out", default=None, help="JSON path for the results dump (default: beside ckpt).")
    args = parser.parse_args()

    ckpt = args.ckpt
    out_path = Path(args.out) if args.out else Path(ckpt).parent.parent / f"val_twice_{Path(ckpt).stem}.json"

    print(f"val_twice: ckpt={ckpt}  data={args.data}  imgsz={args.imgsz}  device={args.device}  seeds={args.seeds}")

    # Indirect (deterministic): a single seed is enough, but run the first seed for the record.
    m_ind = _val_once(ckpt, args.data, args.imgsz, args.device, args.seeds[0], direct=False, subdiv_k=args.subdiv_k)
    ind_seg = _seg_metrics(m_ind)
    ind_box = _box_metrics(m_ind)
    print(
        "[indirect] map50-95={map50-95:.5f} map50={map50:.5f} map75={map75:.5f} ap95={ap95:.5f}".format(**ind_seg)
    )

    # Direct (stochastic): average over seeds.
    dir_seg_records, dir_box_records = [], []
    for s in args.seeds:
        m_dir = _val_once(ckpt, args.data, args.imgsz, args.device, s, direct=True, subdiv_k=args.subdiv_k)
        dir_seg_records.append(_seg_metrics(m_dir))
        dir_box_records.append(_box_metrics(m_dir))
        print(
            f"[direct s={s}] map50-95={dir_seg_records[-1]['map50-95']:.5f} "
            f"map50={dir_seg_records[-1]['map50']:.5f} map75={dir_seg_records[-1]['map75']:.5f} "
            f"ap95={dir_seg_records[-1]['ap95']:.5f}"
        )

    dir_seg_agg = _aggregate(dir_seg_records)
    dir_box_agg = _aggregate(dir_box_records)

    print("\n=== Δ direct − indirect (MLP direct inference benefit) ===")
    for k in ind_seg:
        d_mean = dir_seg_agg[k]["mean"]
        d_std = dir_seg_agg[k]["std"]
        delta = d_mean - ind_seg[k]
        print(f"  {k:10s}: indirect={ind_seg[k]:.5f}  direct={d_mean:.5f}±{d_std:.5f}  Δ={delta:+.5f}")

    print("\n=== sanity: box metrics (must be invariant) ===")
    print(f"  indirect box_map50-95={ind_box['box_map50-95']:.5f}  "
          f"direct box_map50-95={dir_box_agg['box_map50-95']['mean']:.5f}±{dir_box_agg['box_map50-95']['std']:.5f}")

    result = {
        "ckpt": ckpt,
        "data": args.data,
        "imgsz": args.imgsz,
        "device": args.device,
        "seeds": args.seeds,
        "subdiv_k": args.subdiv_k,
        "indirect_seg": ind_seg,
        "indirect_box": ind_box,
        "direct_seg_agg": dir_seg_agg,
        "direct_box_agg": dir_box_agg,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()