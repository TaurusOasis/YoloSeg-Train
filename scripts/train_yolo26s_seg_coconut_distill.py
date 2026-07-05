#!/usr/bin/env python3
"""Train YOLO26s-seg on converted COCONut data with a YOLO26x-seg teacher."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Must be set before torch initializes the CUDA allocator: multi_scale training resizes every batch and
# fragments the cache, which is what pushed resumed DDP runs into mid-epoch OOM (see F20/F21 in
# docs/yolo26s-seg-distill-training-flow.md). DDP workers inherit this via the environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SPLIT = "coconut_b"
DEFAULT_DATA_ROOTS = {
    "coconut_s": Path("/home/genesis/Train/Dataset/COCONut_yolo_seg"),
    "coconut_b": Path("/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2"),
}
DEFAULT_COCONUT_ROOT = Path("/home/genesis/Train/Dataset/coconut")
DEFAULT_IMAGE_ROOT = Path("/home/genesis/Train/Dataset/coco2017")
DEFAULT_STUDENT = REPO_ROOT / "runs/segment/yolo26s-seg-lvis-coco80-distill-x-teacher-b80-2gpu/weights/best.pt"
DEFAULT_STUDENT_FALLBACK = "yolo26s-seg.yaml"
DEFAULT_TEACHER = REPO_ROOT / "yolo26x-seg.pt"


def is_explicit_resume_path(resume) -> bool:
    """Return True when --resume was given as a checkpoint path instead of bare --resume."""
    return bool(resume) and resume is not True


def resume_save_dir(resume) -> Path | None:
    """Infer the Ultralytics save_dir from a resume checkpoint path."""
    if not is_explicit_resume_path(resume):
        return None
    path = Path(resume)
    return path.parent.parent if path.parent.name == "weights" else path.parent


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--student",
        default=DEFAULT_STUDENT,
        help=(
            "Student model checkpoint or YAML. Defaults to the previous LVIS COCO80-distilled best.pt, "
            f"falling back to {DEFAULT_STUDENT_FALLBACK} when that checkpoint is unavailable."
        ),
    )
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER, help="Teacher model checkpoint.")
    parser.add_argument("--data", type=Path, help="COCONut YOLO segment data YAML.")
    parser.add_argument("--data-root", type=Path, help="Converted COCONut YOLO dataset root.")
    parser.add_argument("--coconut-root", type=Path, default=DEFAULT_COCONUT_ROOT, help="Raw COCONut root.")
    parser.add_argument(
        "--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="COCO image root with train2017/val2017."
    )
    parser.add_argument("--train-split", choices=["coconut_s", "coconut_b"], default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Force building/updating converted labels. The default data YAML is also auto-built when missing.",
    )
    parser.add_argument("--overwrite-data", action="store_true", help="Overwrite converted labels when preparing data.")
    parser.add_argument(
        "--prep-workers", type=int, default=16, help="Workers for COCONut conversion when --prepare-data is set."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0,1,2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs/segment"))
    parser.add_argument("--name", help="Experiment name. Defaults to yolo26s-seg-<train-split>-distill-x-teacher.")
    parser.add_argument(
        "--dis",
        type=float,
        default=3.0,
        help=(
            "Feature distillation loss weight. Uses 3.0 instead of the framework default 6.0 so the teacher feature "
            "loss does not dominate segmentation losses."
        ),
    )
    parser.add_argument(
        "--dis-proto",
        type=float,
        default=1.0,
        help="Segmentation proto distillation weight, logged separately as dis_proto.",
    )
    parser.add_argument(
        "--distill-warmup-epochs",
        type=float,
        default=3.0,
        help="Linearly ramp feature/proto distillation over the first N epochs to avoid projector cold-start noise.",
    )
    parser.add_argument(
        "--distill-loss-clip",
        type=float,
        default=10.0,
        help="Clamp each feature/proto distillation loss component after NaN/Inf to zero; set 0 to disable finite clipping.",
    )
    parser.add_argument("--optimizer", default="MuSGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early-stopping patience. Default 100 intentionally disables early stopping for the default 100 epochs.",
    )
    parser.add_argument(
        "--resume", nargs="?", const=True, default=False, help="Resume from the latest run or a checkpoint path."
    )
    parser.add_argument("--seed", type=int, default=0, help="Training random seed.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation during training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing the output run directory.")
    parser.add_argument(
        "--swanlab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SwanLab logging through the local Ultralytics callback.",
    )
    parser.add_argument("--swanlab-mode", default="local", choices=["local", "offline", "online", "disabled"])
    parser.add_argument("--swanlab-project", default="yolo26s-seg-coconut-distill")
    parser.add_argument(
        "--swanlab-run-name", help="SwanLab run name. Defaults to the checkpoint run name when resuming."
    )
    parser.add_argument(
        "--swanlab-log-dir",
        type=Path,
        help=(
            "Local SwanLab log directory. Defaults to <project>/swanlab/<name>, or to the resume checkpoint's "
            "sibling swanlab/<run> directory when resuming."
        ),
    )
    parser.add_argument("--swanlab-watch", action="store_true", help="Start a local SwanLab dashboard in tmux.")
    parser.add_argument("--swanlab-watch-only", action="store_true", help="Start the SwanLab dashboard and exit.")
    parser.add_argument("--swanlab-watch-host", default="127.0.0.1")
    parser.add_argument("--swanlab-watch-port", type=int, default=5092)
    parser.add_argument("--swanlab-watch-session", help="tmux session name for the local SwanLab dashboard.")
    args, unknown = parser.parse_known_args()
    args.data_explicit = args.data is not None
    args.name_explicit = args.name is not None
    args.swanlab_log_dir_explicit = args.swanlab_log_dir is not None
    resume_dir = resume_save_dir(args.resume)
    if args.data_root is None:
        args.data_root = DEFAULT_DATA_ROOTS[args.train_split]
    if args.data is None:
        args.data = args.data_root / f"{args.train_split.replace('_', '-')}-seg.yaml"
    if args.name is None:
        args.name = f"yolo26s-seg-{args.train_split.replace('_', '-')}-distill-x-teacher-lvispretrain"
    if args.swanlab_run_name is None:
        args.swanlab_run_name = resume_dir.name if resume_dir else args.name
    if args.swanlab_log_dir is None:
        args.swanlab_log_dir = (
            (resume_dir.parent if resume_dir else Path(args.project))
            / "swanlab"
            / (resume_dir.name if resume_dir else args.name)
        )
    if args.swanlab_watch_session is None:
        args.swanlab_watch_session = f"swanlab-{args.swanlab_log_dir.name}"[:80]
    return args, unknown


def parse_unknown_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if not item.startswith("--") or "=" not in item:
            raise ValueError(f"Extra overrides must use --key=value format, got {item!r}")
        key, value = item[2:].split("=", 1)
        overrides[key.replace("-", "_")] = yaml.safe_load(value)
    return overrides


def resolve_student_model(args: argparse.Namespace) -> str:
    """Return a clear student model source, falling back only when the built-in default is missing."""
    student = args.student
    if student == DEFAULT_STUDENT and not DEFAULT_STUDENT.exists():
        print(
            "WARNING: default LVIS-distilled student checkpoint not found: "
            f"{DEFAULT_STUDENT}. Falling back to {DEFAULT_STUDENT_FALLBACK}."
        )
        return DEFAULT_STUDENT_FALLBACK

    student_path = Path(student)
    is_yaml = student_path.suffix.lower() in {".yaml", ".yml"}
    is_checkpoint = student_path.suffix.lower() in {".pt", ".pth"}
    is_ultralytics_yaml_alias = is_yaml and len(student_path.parts) == 1 and student_path.name.startswith("yolo")
    if (is_yaml or is_checkpoint) and not student_path.exists() and not is_ultralytics_yaml_alias:
        raise FileNotFoundError(
            f"Student model not found: {student}. Pass --student to an existing checkpoint/YAML, "
            f"or omit --student to use {DEFAULT_STUDENT_FALLBACK} when the default checkpoint is unavailable."
        )
    return str(student)


def resolve_model_source(args: argparse.Namespace) -> str:
    """Return the checkpoint/YAML to instantiate, avoiding redundant student loads during explicit resume."""
    if is_explicit_resume_path(args.resume):
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        return str(resume_path)
    return resolve_student_model(args)


def prepare_dataset(args: argparse.Namespace) -> None:
    if args.data.exists() and not args.prepare_data:
        return
    if args.data_explicit and not args.prepare_data:
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_coconut_yolo_seg.py"),
        "--coconut-root",
        str(args.coconut_root),
        "--image-root",
        str(args.image_root),
        "--out-root",
        str(args.data_root),
        "--train-split",
        args.train_split,
        "--workers",
        str(args.prep_workers),
    ]
    if args.overwrite_data:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)


def configure_swanlab(args: argparse.Namespace) -> None:
    """Configure the Ultralytics SwanLab callback from explicit script arguments."""
    if not args.swanlab or args.swanlab_mode == "disabled":
        os.environ["ULTRALYTICS_SWANLAB"] = "false"
        os.environ["ULTRALYTICS_SWANLAB_MODE"] = "disabled"
        return

    args.swanlab_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ULTRALYTICS_SWANLAB"] = "true"
    os.environ["ULTRALYTICS_SWANLAB_MODE"] = args.swanlab_mode
    os.environ["ULTRALYTICS_SWANLAB_PROJECT"] = args.swanlab_project
    os.environ["ULTRALYTICS_SWANLAB_RUN_NAME"] = args.swanlab_run_name
    os.environ["ULTRALYTICS_SWANLAB_LOG_DIR"] = str(args.swanlab_log_dir)
    print(f"SwanLab local log dir: {args.swanlab_log_dir}")


def print_run_notes(args: argparse.Namespace) -> None:
    """Print short notes for settings that are easy to misread in logs."""
    if args.resume:
        inherited = []
        if not cli_provided("--patience"):
            inherited.append("patience")
        if not cli_provided("--epochs"):
            inherited.append("epochs")
        if inherited:
            print(f"Resume: inheriting {', '.join(inherited)} from the checkpoint train_args.")
    elif args.patience >= args.epochs:
        print(f"Early stopping is effectively disabled: patience={args.patience}, epochs={args.epochs}.")
    if args.distill_warmup_epochs > 0:
        print(
            f"Distillation warmup enabled: feature dis={args.dis}, proto dis={args.dis_proto}, "
            f"warmup={args.distill_warmup_epochs} epochs."
        )
    if args.no_val:
        print("WARNING: --no-val disables mAP tracking and reliable best.pt selection during distillation.")
    if args.resume and args.name_explicit:
        print("Resume uses the checkpoint save_dir; --name does not change Ultralytics resume output.")


def maybe_start_swanlab_watch(args: argparse.Namespace) -> None:
    """Start a detached local SwanLab dashboard for the configured log directory."""
    if not (args.swanlab_watch or args.swanlab_watch_only) or not args.swanlab or args.swanlab_mode == "disabled":
        return
    tmux = shutil.which("tmux")
    swanlab = shutil.which("swanlab") or str(Path(sys.executable).with_name("swanlab"))
    if not tmux or not Path(swanlab).exists():
        print(
            "SwanLab watch command:\n"
            f"  {swanlab} watch {args.swanlab_log_dir} --host {args.swanlab_watch_host} "
            f"--port {args.swanlab_watch_port}"
        )
        return

    has_session = subprocess.run(
        [tmux, "has-session", "-t", args.swanlab_watch_session],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if has_session.returncode == 0:
        print(f"SwanLab watch tmux session already running: {args.swanlab_watch_session}")
        return

    cmd = [
        swanlab,
        "watch",
        str(args.swanlab_log_dir),
        "--host",
        args.swanlab_watch_host,
        "--port",
        str(args.swanlab_watch_port),
    ]
    subprocess.run(
        [tmux, "new-session", "-d", "-s", args.swanlab_watch_session, " ".join(shlex.quote(x) for x in cmd)],
        check=True,
    )
    print(
        f"SwanLab dashboard started: http://{args.swanlab_watch_host}:{args.swanlab_watch_port} "
        f"(tmux session: {args.swanlab_watch_session})"
    )


def cli_provided(*flags: str) -> bool:
    """Return True when any of the given option strings was explicitly typed on the command line."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:] for flag in flags)


def build_train_args(args: argparse.Namespace, data_yaml: Path, unknown: list[str]) -> dict[str, Any]:
    """Build Ultralytics train overrides, keeping resume overrides intentionally narrow."""
    extra_overrides = parse_unknown_overrides(unknown)
    if args.resume:
        # Only forward arguments the user explicitly typed. Argparse defaults must not reach the trainer's
        # resume whitelist, or they silently overwrite the checkpoint's train_args (e.g. a run started with
        # --patience 200 would resume with the script default 100).
        resume_overridable = {
            "device": ("--device",),
            "workers": ("--workers",),
            "batch": ("--batch",),
            "imgsz": ("--imgsz",),
            "epochs": ("--epochs",),
            "save_period": ("--save-period",),
            "patience": ("--patience",),
            "dis_proto": ("--dis-proto",),
            "distill_warmup_epochs": ("--distill-warmup-epochs",),
            "distill_loss_clip": ("--distill-loss-clip",),
        }
        train_args = {"resume": args.resume, "data": str(data_yaml)}
        if args.no_val:
            train_args["val"] = False
        for key, flags in resume_overridable.items():
            if cli_provided(*flags):
                train_args[key] = getattr(args, key)
        if "distill_model" in extra_overrides:
            train_args["distill_model"] = extra_overrides.pop("distill_model")
        train_args.update(extra_overrides)
        dropped = sorted(set(resume_overridable) - set(train_args))
        if dropped:
            print(f"Resume: inheriting {', '.join(dropped)} from the checkpoint train_args (not overridden).")
        return train_args

    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "task": "segment",
        "distill_model": str(args.teacher),
        "dis": args.dis,
        "dis_proto": args.dis_proto,
        "distill_warmup_epochs": args.distill_warmup_epochs,
        "distill_loss_clip": args.distill_loss_clip,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
        "close_mosaic": args.close_mosaic,
        "save_period": args.save_period,
        "patience": args.patience,
        "resume": args.resume,
        "seed": args.seed,
        "val": not args.no_val,
        "exist_ok": args.exist_ok,
        "amp": True,
    }
    train_args.update(extra_overrides)
    return train_args


def main() -> None:
    args, unknown = parse_args()
    configure_swanlab(args)
    maybe_start_swanlab_watch(args)
    if args.swanlab_watch_only:
        return
    print_run_notes(args)
    prepare_dataset(args)
    data_yaml = args.data
    if not data_yaml.exists():
        if args.data_explicit:
            raise FileNotFoundError(f"Explicit --data YAML not found: {data_yaml}")
        raise FileNotFoundError(f"Converted COCONut data YAML not found after dataset preparation: {data_yaml}")
    if not args.teacher.exists():
        raise FileNotFoundError(f"Teacher model not found: {args.teacher}")

    model_source = resolve_model_source(args)
    train_args = build_train_args(args, data_yaml, unknown)

    model = YOLO(model_source)
    model.train(**train_args)


if __name__ == "__main__":
    main()
