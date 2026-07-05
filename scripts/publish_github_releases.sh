#!/usr/bin/env bash
# Publish stage best.pt + SwanLab curve PNGs to GitHub Releases.
# Requires: gh auth with `repo` scope (contents + releases write).
# Usage: bash scripts/publish_github_releases.sh [--dry-run]

set -euo pipefail
cd "$(dirname "$0")/.."
DRY=${1:-}

RUNS=runs/segment
CURVES=docs/releases/curves
REPO=TaurusOasis/YoloSeg-Train

publish() {
  local tag=$1 title=$2 notes=$3 pt_path=$4 pt_name=$5 curve=$6
  echo "=== $tag ==="
  if [[ ! -f $pt_path ]]; then
    echo "MISSING $pt_path" >&2
    return 1
  fi
  if [[ "$DRY" == "--dry-run" ]]; then
    echo "gh release create $tag --repo $REPO --title '$title'"
    echo "  attach: $pt_path#$pt_name, $curve, $CURVES/pipeline-overview.png"
    return 0
  fi
  gh release create "$tag" \
    --repo "$REPO" \
    --title "$title" \
    --notes "$notes" \
    "$pt_path#$pt_name" \
    "$curve" \
    "$CURVES/pipeline-overview.png"
}

publish stage-a-lvis-pretrain \
  "Stage A: LVIS Pretrain (best ep100)" \
  "Stage A · LVIS pretrain. Best ep100 mask mAP50-95=0.071 (LVIS val). See docs/releases/RELEASES.md." \
  "$RUNS/yolo26s-seg-lvis-b48-bf16-swanlab/weights/best.pt" \
  yolo26s-seg-lvis-b48-best.pt \
  "$CURVES/stage-a-lvis-pretrain.png"

publish stage-b-lvis-coco80-distill \
  "Stage B: LVIS→COCO80 Distill (best ep100)" \
  "Stage B · LVIS COCO80 distill from yolo26x teacher. Best ep100 mask mAP50-95=0.315." \
  "$RUNS/yolo26s-seg-lvis-coco80-distill-x-teacher-b80-2gpu/weights/best.pt" \
  yolo26s-seg-lvis-coco80-distill-best.pt \
  "$CURVES/stage-b-lvis-coco80-distill.png"

publish stage-c-coconut-v1-distill \
  "Stage C: COCONut-B v1 Distill (best ep99)" \
  "Stage C · COCONut-B v1 labels, 100 epoch distill. Best ep99 mask mAP50-95=0.354 (v1 val)." \
  "$RUNS/yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu/weights/best.pt" \
  yolo26s-seg-coconut-v1-distill-best.pt \
  "$CURVES/stage-c-coconut-v1-distill.png"

publish stage-d-recipe200-v2 \
  "Stage D: COCONut-B v2 Recipe200 (best ep107)" \
  "Stage D · v2 labels recipe200 (107/200 ep, interrupted). Best ep107 mask mAP50-95=0.376 on COCONut-B v2 val. Primary dense-mask checkpoint before PointRend." \
  "$RUNS/yolo26s-seg-coconut-b-v2-distill-recipe200/weights/best.pt" \
  yolo26s-seg-coconut-v2-recipe200-best.pt \
  "$CURVES/stage-d-recipe200-v2.png"

publish stage-e-pointrend-ft60 \
  "Stage E: PointRend Finetune interim (best ep12+)" \
  "Stage E · PointRend finetune from Stage D (interim snapshot). Model: yolo26s-seg-pointrend.yaml. Re-run this script after 60-epoch completion to refresh." \
  "$RUNS/yolo26s-seg-coconut-b-v2-pointrend-ft60/weights/best.pt" \
  yolo26s-seg-pointrend-ft60-best.pt \
  "$CURVES/stage-e-pointrend-ft60.png"

echo "Done."
