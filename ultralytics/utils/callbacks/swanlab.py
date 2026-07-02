# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""SwanLab logging for Ultralytics YOLO training."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from ultralytics.utils import LOGGER, RANK, TESTS_RUNNING, colorstr

PREFIX = colorstr("SwanLab: ")
_processed_plots: dict[Path, float] = {}

try:
    assert not TESTS_RUNNING  # do not log pytest
    import swanlab

    assert hasattr(swanlab, "__version__")  # verify package is not a local directory
    _IMPORT_ERROR = None
except Exception as e:
    swanlab = None
    _IMPORT_ERROR = e


def _env_enabled() -> bool:
    """Return whether SwanLab logging is enabled for this process."""
    value = os.getenv("ULTRALYTICS_SWANLAB", "true").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def _clean_value(value: Any) -> Any:
    """Convert common non-serializable metric/config values to SwanLab-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _clean_value(value.tolist())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None  # NaN/Inf break swanboard JSON serialization
    return value


def _clean_dict(values: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a dict before sending it to SwanLab."""
    return {str(k): _clean_value(v) for k, v in values.items() if v is not None}


def _log(data: dict[str, Any], step: int) -> bool:
    """Log scalar/image data and disable SwanLab on failure so training can continue."""
    if not data:
        return True
    try:
        swanlab.log(_clean_dict(data), step=step)
        return True
    except Exception as e:
        LOGGER.warning(f"{PREFIX}metric logging failed, disabling tracking for this run: {e}")
        try:
            swanlab.finish(state="crashed", error=str(e))
        except Exception:
            pass
        return False


def _log_plots(plots: dict[Path, dict[str, Any]], step: int, prefix: str) -> bool:
    """Log newly-created plot images to SwanLab."""
    images = []
    for path, params in plots.copy().items():
        timestamp = params.get("timestamp", path.stat().st_mtime if path.exists() else 0)
        if _processed_plots.get(path) == timestamp or not path.exists():
            continue
        try:
            images.append(swanlab.Image(str(path), caption=path.stem))
            _processed_plots[path] = timestamp
        except Exception as e:
            LOGGER.debug(f"{PREFIX}failed to prepare plot {path}: {e}")
    if images:
        return _log({prefix: images}, step=step)
    return True


def on_pretrain_routine_start(trainer) -> None:
    """Initialize SwanLab at the start of training."""
    if not _env_enabled() or RANK not in {-1, 0}:
        return
    if not swanlab:
        if _IMPORT_ERROR is not None:
            LOGGER.warning(f"{PREFIX}not initialized because SwanLab import failed: {_IMPORT_ERROR}")
        return
    mode = os.getenv("ULTRALYTICS_SWANLAB_MODE", os.getenv("SWANLAB_MODE", "local"))
    if mode.strip().lower() == "disabled":
        return

    trainer._swanlab_active = False
    try:
        log_dir = os.getenv("ULTRALYTICS_SWANLAB_LOG_DIR") or str(Path(trainer.save_dir) / "swanlab")
        project = os.getenv("ULTRALYTICS_SWANLAB_PROJECT") or str(trainer.args.project or "Ultralytics")
        name = os.getenv("ULTRALYTICS_SWANLAB_RUN_NAME") or str(trainer.args.name or Path(trainer.save_dir).name)
        kwargs = {
            "project": project.replace("/", "-"),
            "name": name.replace("/", "-"),
            "log_dir": log_dir,
            "mode": mode,
            "config": _clean_dict(vars(trainer.args)),
            "tags": ["ultralytics", str(trainer.args.task)],
        }
        workspace = os.getenv("ULTRALYTICS_SWANLAB_WORKSPACE")
        if workspace:
            kwargs["workspace"] = workspace

        # Reuse the previous run id on resume so curves continue in one run instead of splitting per restart.
        # Local mode still creates a new run directory, but runs share the id and cloud/offline modes truly resume.
        run_id_file = Path(log_dir) / ".swanlab_run_id"
        if getattr(trainer.args, "resume", False) and run_id_file.exists():
            previous_id = run_id_file.read_text(encoding="utf-8").strip()
            if previous_id:
                kwargs["id"] = previous_id
                kwargs["resume"] = "allow"
                LOGGER.info(f"{PREFIX}resuming previous run id {previous_id}")

        run = swanlab.init(**kwargs)
        run_id = getattr(run, "id", None)
        if run_id:
            run_id_file.parent.mkdir(parents=True, exist_ok=True)
            run_id_file.write_text(str(run_id), encoding="utf-8")
        trainer._swanlab_active = True
        LOGGER.info(f"{PREFIX}logging to {log_dir} in {mode!r} mode")
        LOGGER.info(f"{PREFIX}disable with 'ULTRALYTICS_SWANLAB=false' or 'ULTRALYTICS_SWANLAB_MODE=disabled'")
    except Exception as e:
        trainer._swanlab_active = False
        LOGGER.warning(f"{PREFIX}failed to initialize, not logging this run: {e}")


def on_train_epoch_end(trainer) -> None:
    """Log train losses and learning rates at the end of each epoch."""
    if swanlab and getattr(trainer, "_swanlab_active", False) and RANK in {-1, 0}:
        step = trainer.epoch + 1
        trainer._swanlab_active = _log(
            {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.lr}, step=step
        )


def on_fit_epoch_end(trainer) -> None:
    """Log validation metrics and plots at the end of each epoch."""
    if swanlab and getattr(trainer, "_swanlab_active", False) and RANK in {-1, 0}:
        step = trainer.epoch + 1
        trainer._swanlab_active = _log(trainer.metrics, step=step)
        if not trainer._swanlab_active:
            return
        trainer._swanlab_active = _log_plots(trainer.plots, step=step, prefix="plots/train")
        if not trainer._swanlab_active:
            return
        if getattr(trainer, "validator", None) is not None:
            trainer._swanlab_active = _log_plots(trainer.validator.plots, step=step, prefix="plots/val")


def on_train_end(trainer) -> None:
    """Log final plots and close the SwanLab run."""
    if not swanlab or not getattr(trainer, "_swanlab_active", False) or RANK not in {-1, 0}:
        return
    step = trainer.epoch + 1
    try:
        _log_plots(trainer.plots, step=step, prefix="plots/train_end")
        if getattr(trainer, "validator", None) is not None:
            _log_plots(trainer.validator.plots, step=step, prefix="plots/val_end")
        swanlab.finish()
        LOGGER.info(f"{PREFIX}run finished")
    except Exception as e:
        LOGGER.warning(f"{PREFIX}failed to finish run cleanly: {e}")
    finally:
        trainer._swanlab_active = False
        _processed_plots.clear()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if swanlab
    else {}
)
