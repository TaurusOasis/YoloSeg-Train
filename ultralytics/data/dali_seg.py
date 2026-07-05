# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Experimental NVIDIA DALI GPU preprocessing for YOLO segmentation training.

Accelerates JPEG decode + square resize on GPU. Mask rasterization (polygon2mask) stays on CPU
via the existing ``Format`` transform. Enable with ``dali=True`` (requires ``nvidia-dali``).

Limitations (experimental):
  - Does not run mosaic / mixup / copy_paste on the image branch.
  - Uses stretch resize to imgsz×imgsz (not LetterBox); pair with short smoke / ablation runs first.
  - Val dataloader still uses the standard CPU path.

See: ``docs/en/guides/nvidia-dali.md`` for inference-oriented DALI usage.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Iterator

import numpy as np
import torch

from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances

try:
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False


def dali_supported() -> bool:
    """Return True when the NVIDIA DALI Python package is importable."""
    return DALI_AVAILABLE


def _check_dali_augment_compat(hyp) -> None:
    """Warn when CPU-heavy augmentations are non-zero alongside DALI."""
    for key in ("mosaic", "mixup", "copy_paste"):
        if float(getattr(hyp, key, 0) or 0) > 0:
            LOGGER.warning(
                f"DALI path: {key}={getattr(hyp, key)} is not applied on the GPU image branch; "
                f"use {key}=0 for consistent experiments."
            )
    # bgr>0 makes the standard path randomly keep BGR; DALI always decodes RGB, so the channel-flip
    # augmentation is silently dropped on the GPU branch.
    if float(getattr(hyp, "bgr", 0.0) or 0.0) > 0:
        LOGGER.warning(f"DALI path: bgr={hyp.bgr} is ignored (DALI always decodes RGB); use bgr=0.0 for parity.")


def _scale_label_to_square(label: dict, imgsz: int) -> dict:
    """Map normalized labels to imgsz×imgsz stretch (matches DALI resize)."""
    h0, w0 = label["shape"]
    bboxes = label["bboxes"].copy()
    segments = [np.asarray(s, dtype=np.float32).copy() for s in label.get("segments", [])]
    if label.get("normalized", True):
        bboxes[:, [0, 2]] *= w0
        bboxes[:, [1, 3]] *= h0
        for seg in segments:
            seg[:, 0] *= w0
            seg[:, 1] *= h0
    sx, sy = imgsz / w0, imgsz / h0
    bboxes[:, [0, 2]] *= sx
    bboxes[:, [1, 3]] *= sy
    for seg in segments:
        seg[:, 0] *= sx
        seg[:, 1] *= sy
    keypoints = label["keypoints"].copy() if label.get("keypoints") is not None else None
    if keypoints is not None and label.get("normalized", True):
        keypoints[..., 0] *= w0
        keypoints[..., 1] *= h0
        keypoints[..., 0] *= sx
        keypoints[..., 1] *= sy
    out = deepcopy(label)
    out["bboxes"] = bboxes
    out["segments"] = segments
    out["keypoints"] = keypoints
    out["normalized"] = False
    out["shape"] = (imgsz, imgsz)
    return out


def _hyp_from_dataset(dataset: YOLODataset, hyp=None):
    """Resolve hyperparameters for mask formatting (dataset does not store hyp)."""
    if hyp is not None:
        return hyp
    fmt = next((t for t in getattr(dataset.transforms, "transforms", []) if isinstance(t, Format)), None)
    if fmt is None:
        from ultralytics.utils import DEFAULT_CFG

        return DEFAULT_CFG

    class _Hyp:
        pass

    proxy = _Hyp()
    proxy.mask_ratio = fmt.mask_ratio
    proxy.overlap_mask = fmt.mask_overlap
    proxy.bgr = fmt.bgr
    proxy.mosaic = 0.0
    proxy.mixup = 0.0
    proxy.copy_paste = 0.0
    return proxy


class _DALISamplerStub:
    """Minimal sampler API for BaseTrainer DDP epoch hooks."""

    def __init__(self, source: _SegExternalSource | None = None, base_seed: int = 0):
        self._source = source
        self._base_seed = base_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if self._source is not None:
            self._source.rng = np.random.default_rng(self._base_seed + epoch)
            self._source._shuffle_epoch()


def _format_seg_label(label: dict, formatter: Format) -> dict:
    """Build mask/bbox tensors for one sample (image tensor added later from DALI)."""
    h = w = label["shape"][0]
    label_work = {
        "img": np.zeros((h, w, 3), dtype=np.uint8),
        "cls": np.asarray(label["cls"]).copy(),
        "instances": Instances(
            bboxes=np.asarray(label["bboxes"], dtype=np.float32).copy(),
            segments=[np.asarray(s, dtype=np.float32).copy() for s in label.get("segments", [])],
            keypoints=(
                np.asarray(label["keypoints"], dtype=np.float32).copy()
                if label.get("keypoints") is not None
                else None
            ),
            bbox_format=label.get("bbox_format", "xywh"),
            normalized=label.get("normalized", False),
        ),
    }
    params = formatter.get_params(label_work)
    formatted: dict[str, Any] = {}
    formatter.apply_instances(formatted, params)
    return formatted


class _SegExternalSource:
    """Feed JPEG bytes and dataset indices as aligned DALI external-source batches."""

    def __init__(self, files: list[str], batch_size: int, shard_id: int, num_shards: int, seed: int):
        self.files = files
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.rng = np.random.default_rng(seed)
        # Pad the strided shard so every rank owns exactly shard_len items. Without this, ranks whose
        # strided shard is shorter would report a __len__ that exceeds their real shard and wrap into the
        # next epoch's data (repeating their few images, leaking across epoch boundaries, and
        # desynchronising DDP). Padding cyclically repeats a handful of indices on short shards
        # (<1 sample on large datasets), matching DALI's own pad_last_batch convention.
        shard_len = math.ceil(len(files) / max(num_shards, 1))
        base = list(range(shard_id, len(files), num_shards))
        if base:
            self._order = [base[i % len(base)] for i in range(shard_len)]
        else:  # degenerate: more ranks than files; keep __len__ consistent to avoid DDP hang
            self._order = [0] * shard_len
        self._cursor = 0
        self._shuffle_epoch()

    def _shuffle_epoch(self) -> None:
        self.rng.shuffle(self._order)
        self._cursor = 0

    def reset(self) -> None:
        self._shuffle_epoch()

    def __call__(self) -> tuple[list[np.ndarray], np.ndarray]:
        jpegs: list[np.ndarray] = []
        indices: list[int] = []
        for _ in range(self.batch_size):
            if self._cursor >= len(self._order):
                break
            idx = self._order[self._cursor]
            self._cursor += 1
            with open(self.files[idx], "rb") as f:
                jpegs.append(np.frombuffer(f.read(), dtype=np.uint8))
            indices.append(idx)
        if not jpegs:
            raise StopIteration
        return jpegs, np.asarray(indices, dtype=np.int32)


def _build_image_pipeline(
    source: _SegExternalSource,
    batch_size: int,
    imgsz: int,
    device_id: int,
):
    """Construct and build a DALI pipeline (batch_size set at build time)."""

    @dali.pipeline_def(batch_size=batch_size, num_threads=4, device_id=device_id)
    def _pipe():
        jpegs, indices = fn.external_source(
            source=source,
            num_outputs=2,
            batch=True,
            dtype=[types.UINT8, types.INT32],
        )
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_x=imgsz, resize_y=imgsz, interp_type=types.INTERP_LINEAR)
        images = fn.transpose(images, perm=[2, 0, 1])
        return images, indices

    pipe = _pipe()
    pipe.build()
    return pipe


class YOLOSegDALILoader:
    """DALI GPU image decode + CPU YOLO seg label/mask formatting."""

    def __init__(
        self,
        dataset: YOLODataset,
        batch_size: int,
        rank: int = -1,
        device_id: int | None = None,
        hyp=None,
    ):
        if not DALI_AVAILABLE:
            raise ImportError(
                "nvidia-dali is not installed. Example: "
                "pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda130"
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.imgsz = dataset.imgsz
        self.hyp = _hyp_from_dataset(dataset, hyp)
        self.num_workers = 0
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0 if rank < 0 else rank
            self.world_size = 1
        nd = max(torch.cuda.device_count(), 1)
        self.device_id = device_id if device_id is not None else self.rank % nd
        self.formatter = Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=True,
            return_keypoint=dataset.use_keypoints,
            return_obb=dataset.use_obb,
            mask_ratio=self.hyp.mask_ratio,
            mask_overlap=self.hyp.overlap_mask,
            batch_idx=True,
            bgr=getattr(self.hyp, "bgr", 0.0),
        )
        _check_dali_augment_compat(self.hyp)
        files = [str(f) for f in dataset.im_files]
        seed = 6148914691236517205 + self.rank
        self._source = _SegExternalSource(files, batch_size, self.rank, self.world_size, seed)
        self.sampler = _DALISamplerStub(self._source, base_seed=seed)
        self._pipe = _build_image_pipeline(self._source, batch_size, self.imgsz, self.device_id)
        self._dali_iter = DALIGenericIterator(
            [self._pipe],
            ["images", "label_idx"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=False,
        )
        self.iterator: Iterator[dict] = iter(self._batch_stream())
        shard_len = math.ceil(len(files) / max(self.world_size, 1))
        self._batch_count = math.ceil(shard_len / batch_size)

    def __len__(self) -> int:
        return self._batch_count

    def reset(self) -> None:
        self._source.reset()
        self._dali_iter.reset()
        self.iterator = iter(self._batch_stream())

    def _batch_stream(self) -> Iterator[dict]:
        while True:
            try:
                for dali_batch in self._dali_iter:
                    yield self._collate_dali_batch(dali_batch)
            except StopIteration:
                break
            self._source.reset()
            self._dali_iter.reset()

    def __iter__(self) -> Iterator[dict]:
        for _ in range(len(self)):
            yield next(self.iterator)

    def _parse_label_indices(self, raw_labels) -> list[int]:
        """Convert DALI int32 label batch to Python indices."""
        if isinstance(raw_labels, torch.Tensor):
            raw_labels = raw_labels.cpu().numpy()
        return [int(x) for x in np.asarray(raw_labels).reshape(-1)]

    def _collate_dali_batch(self, dali_batch) -> dict:
        images = dali_batch[0]["images"]
        label_raw = dali_batch[0]["label_idx"]
        if images.dim() == 3:
            images = images.unsqueeze(0)
        indices = self._parse_label_indices(label_raw)
        label_batch = [
            _format_seg_label(_scale_label_to_square(self.dataset.labels[i], self.imgsz), self.formatter)
            for i in indices
        ]
        collated = YOLODataset.collate_fn(label_batch)
        collated["img"] = images
        collated["dali"] = True
        return collated


def build_dali_seg_dataloader(
    dataset: YOLODataset, batch: int, rank: int = -1, hyp=None
) -> YOLOSegDALILoader:
    """Build experimental DALI-backed training loader for YOLO segmentation."""
    LOGGER.info(
        f"{dataset.prefix}NVIDIA DALI GPU decode/resize enabled (experimental). "
        "Image-side mosaic/mixup/copy_paste are skipped."
    )
    return YOLOSegDALILoader(dataset, batch_size=batch, rank=rank, hyp=hyp)
