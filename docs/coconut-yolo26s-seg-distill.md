# COCONut YOLO26s-seg Distillation Dataset

## Raw Dataset Layout

Local root: `/home/genesis/Train/Dataset/coconut`

COCONut is stored as COCO panoptic-style annotations, not COCO instance polygon JSON. Each JSON has one annotation per image and points to an RGB PNG mask. The mask pixel value encodes the panoptic segment id as `R + 256 * G + 256 * 256 * B`; the JSON `segments_info` maps each segment id to `category_id`, `isthing`, and `iscrowd`.

Available splits:

| split     | JSON                      | mask dir              | images | categories | thing classes |
| --------- | ------------------------- | --------------------- | -----: | ---------: | ------------: |
| COCONut-S | `coconut_s.json`          | `coconut_s/`          | 118200 |        133 |            80 |
| COCONut-B | `coconut_b.json`          | `coconut_b/`          | 241602 |        133 |            80 |
| val       | `relabeled_coco_val.json` | `relabeled_coco_val/` |   5000 |        133 |            80 |

Image coverage under `/home/genesis/Train/Dataset/coco2017`:

| split     | image dirs                   | present | missing |
| --------- | ---------------------------- | ------: | ------: |
| COCONut-S | `train2017`                  |  118200 |       0 |
| COCONut-B | `train2017`, `unlabeled2017` |  241602 |       0 |
| val       | `val2017`                    |    5000 |       0 |

## Conversion Rule

The conversion script is [scripts/build_coconut_yolo_seg.py](/home/genesis/Train/Code/ultralytics/scripts/build_coconut_yolo_seg.py).

It builds an Ultralytics YOLO segmentation dataset by:

- keeping only the 80 COCO `isthing=1` categories;
- skipping stuff categories and `iscrowd=1` segments;
- decoding each RGB panoptic PNG into segment ids;
- extracting external contours for each thing segment with OpenCV;
- simplifying polygons with `approx_epsilon=0.001`;
- writing normalized YOLO segment rows: `class x1 y1 x2 y2 ...`;
- symlinking images instead of copying the COCO JPGs.

Empty images are kept with empty label files so the train/val manifests remain image-complete.

## Generated Training Datasets

COCONut-S, COCO-train2017 aligned:

- YAML: `/home/genesis/Train/Dataset/COCONut_yolo_seg/coconut-s-seg.yaml`
- train images: 118200
- val images: 5000
- train labeled images: 116910
- train empty images: 1290
- train segment rows: 1290092
- val segment rows: 57220
- size on disk: 1.5G

COCONut-B, larger train2017 + unlabeled2017 split:

- YAML: `/home/genesis/Train/Dataset/COCONut_b_yolo_seg/coconut-b-seg.yaml`
- train images: 241602
- val images: 5000
- train labeled images: 238931
- train empty images: 2671
- train segment rows: 2365072
- val segment rows: 57220
- size on disk: 2.8G

Both YAML files parse through `ultralytics.data.utils.check_det_dataset()` as `nc=80`, with standard COCO80 class order from `person` to `toothbrush`.

Full label scans passed:

| dataset   | label rows | empty label files | class range | bad rows |
| --------- | ---------: | ----------------: | ----------: | -------: |
| COCONut-S |    1347312 |              1355 |        0-79 |        0 |
| COCONut-B |    2422292 |              2736 |        0-79 |        0 |

## Rebuild Commands

```bash
/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python scripts/build_coconut_yolo_seg.py \
  --out-root /home/genesis/Train/Dataset/COCONut_yolo_seg \
  --train-split coconut_s \
  --workers 16 \
  --overwrite
```

```bash
/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python scripts/build_coconut_yolo_seg.py \
  --out-root /home/genesis/Train/Dataset/COCONut_b_yolo_seg \
  --train-split coconut_b \
  --workers 16 \
  --overwrite
```

## Distillation Entry

The training helper is [scripts/train_yolo26s_seg_coconut_distill.py](/home/genesis/Train/Code/ultralytics/scripts/train_yolo26s_seg_coconut_distill.py). It uses COCONut-B, `yolo26x-seg.pt` as teacher, and the previous LVIS COCO80-distilled `best.pt` as student initialization by default.

Recommended COCONut-B run:

```bash
/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python scripts/train_yolo26s_seg_coconut_distill.py \
  --data /home/genesis/Train/Dataset/COCONut_b_yolo_seg/coconut-b-seg.yaml \
  --data-root /home/genesis/Train/Dataset/COCONut_b_yolo_seg \
  --train-split coconut_b \
  --teacher /home/genesis/Train/Code/ultralytics/yolo26x-seg.pt \
  --student /home/genesis/Train/Code/ultralytics/runs/segment/yolo26s-seg-lvis-coco80-distill-x-teacher-b80-2gpu/weights/best.pt \
  --epochs 100 \
  --batch 150 \
  --device 0,1,2 \
  --name yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu \
  --dis 3.0 \
  --dis-proto 1.0 \
  --distill-warmup-epochs 3.0 \
  --seed 0 \
  --patience 100 \
  --swanlab-watch \
  --exist-ok
```

Training defaults and intent:

- `patience=100` with `epochs=100` intentionally means "run the full 100 epochs"; early stopping is effectively disabled. Use `--patience 30` to `--patience 50` if early stopping is desired.
- `dis=3.0` is intentionally lower than the framework default `6.0`; for segmentation, the teacher neck-feature loss can otherwise dominate box/cls/mask losses too early.
- `dis_proto=1.0` enables segmentation-specific proto distillation. It distills normalized teacher/student mask prototype maps and uses the teacher P3 foreground score as a light spatial weight.
- `distill_warmup_epochs=3.0` linearly ramps feature/proto distillation during the first 3 epochs to reduce projector cold-start noise.
- Keep validation enabled during distillation tuning. `--no-val` removes mAP tracking and makes `best.pt` selection much less meaningful.
- `--seed` is explicit for reproducibility. The command above uses `seed=0`.
- If `--prepare-data` is used, the helper passes `--prep-workers 16` to the conversion script by default.

By default the helper enables SwanLab local logging:

- mode: `local`
- project: `yolo26s-seg-coconut-distill`
- log dir: `<project>/swanlab/<name>`, for example `/home/genesis/Train/Code/ultralytics/runs/segment/swanlab/yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu`
- dashboard command: `swanlab watch <log-dir> --host 127.0.0.1 --port 5092`

The SwanLab directory is intentionally outside the Ultralytics `save_dir`, because DDP startup removes `save_dir` for fresh runs. Use `--exist-ok` for stable one-to-one naming between the Ultralytics run and the SwanLab local dashboard; without it, Ultralytics may auto-increment the training directory name.

Open or restart the local dashboard without training:

```bash
/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python scripts/train_yolo26s_seg_coconut_distill.py \
  --name yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu \
  --swanlab-watch-only
```

Resume the same run after interruption:

```bash
/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python scripts/train_yolo26s_seg_coconut_distill.py \
  --resume /home/genesis/Train/Code/ultralytics/runs/segment/yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu/weights/last.pt \
  --name yolo26s-seg-coconut-b-distill-x-teacher-lvispretrain-b150-3gpu \
  --swanlab-watch \
  --exist-ok
```

Use COCONut-S for a COCO-train2017-only baseline:

```bash
/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python scripts/train_yolo26s_seg_coconut_distill.py \
  --data /home/genesis/Train/Dataset/COCONut_yolo_seg/coconut-s-seg.yaml \
  --data-root /home/genesis/Train/Dataset/COCONut_yolo_seg \
  --train-split coconut_s \
  --teacher /home/genesis/Train/Code/ultralytics/yolo26x-seg.pt \
  --student yolo26s-seg.yaml \
  --epochs 100 \
  --batch 150 \
  --device 0,1,2 \
  --name yolo26s-seg-coconut-s-distill-x-teacher-b150-3gpu \
  --exist-ok
```
