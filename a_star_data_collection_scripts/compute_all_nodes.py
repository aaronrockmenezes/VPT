#!/usr/bin/env python3
"""Compile all A* raw per-GPU node outputs into per-node staged dirs.

This is the CPU-only bulk version of compile_tasks.py. It scans a collector
BASE_PATH, groups raw GPU folders by logical node, verifies successful envs,
and copies valid RGB/Semantic/cam rollouts into data_node{node_key}_compiled
folders that compile_a_star_dataset.py can consume in --mode staged.

Supported raw directory names under {BASE_PATH}/data:
  data_node{node_id}_gpu0
  data_node{node_id}_gpu1
  data_{job_id}_node{node_id}_gpu0
  data_{job_id}_node{node_id}_gpu1

Repeated env folder numbers across GPUs are safe: staged env names include
node key, GPU id, and source folder idx, e.g. env_n123_node0_g1_f42.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Keep this script mostly self-contained so --dry_run and --no_cam_check work
# in lean CPU environments. OpenCV/numpy are imported lazily for cam checks.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
TYPES = ["RGB", "Semantic", "cam"]
CAM_FINAL_NAME = "final_cam_semantic.png"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
REF_SIDE = 256
REF_AREA = REF_SIDE * REF_SIDE
# Cam semantic Yes check. 125 px @ 256x256 is equivalent to the old
# 500 px @ 512x512 threshold by image area.
REF_RED_THRESH = 125
REF_NO_RED_MAX = 0
REF_CONTOUR_MIN = 50
cv2 = None
np = None

REASONS = ("in_view", "occluded", "outside_fov")
LABELS = ("Yes", "No")

# Current repo/HPC style: data_node{node_id}_gpu{gpu}
NODE_GPU_RE = re.compile(r"^data_node(?P<node>.+)_gpu(?P<gpu>\d+)$")

# User-described style: data_{job_id}_node{node_id}_gpu{gpu}
JOB_NODE_GPU_RE = re.compile(
    r"^data_(?P<job>.+)_node(?P<node>[^_]+)_gpu(?P<gpu>\d+)$")


def load_validation_helpers(do_cam_check: bool) -> None:
    """Import OpenCV/numpy only when cam semantic checks are enabled."""
    global cv2, np
    if not do_cam_check or (cv2 is not None and np is not None):
        return
    try:
        import cv2 as cv2_mod
        import numpy as np_mod
    except Exception as exc:  # noqa: BLE001
        print(
            "[ERR] --no_cam_check was not set, but cv2/numpy could "
            "not be imported. Activate the compile env or rerun with "
            "--no_cam_check. "
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    cv2 = cv2_mod
    np = np_mod


def list_image_files(path: Path) -> list[str]:
    """List image filenames in a directory.

    Parameters
    ----------
    path
        Directory to inspect.

    Returns
    -------
    list[str]
        Image filenames directly under ``path``. Missing directories return an
        empty list.
    """
    if not path.exists():
        return []
    return [
        f for f in os.listdir(path) if Path(f).suffix.lower() in IMAGE_EXTS
    ]


def verify_env(gpu_root: Path, label: str, folder_idx: int, min_frames: int):
    """Return (ok, reason). Checks required dirs/files and frame counts."""
    rgb_dir = gpu_root / "RGB" / label / "rollout" / f"env_{folder_idx}"
    sem_dir = gpu_root / "Semantic" / label / "rollout" / f"env_{folder_idx}"
    cam_dir = gpu_root / "cam" / label / "rollout" / f"env_{folder_idx}"

    if not (rgb_dir.is_dir() and sem_dir.is_dir() and cam_dir.is_dir()):
        return False, "missing_dir"
    if not (cam_dir / CAM_FINAL_NAME).exists():
        return False, "missing_cam_final"
    if not (cam_dir / "meta.json").exists():
        return False, "missing_meta"
    if not (cam_dir / "actions.txt").exists():
        return False, "missing_actions"

    rgb_n = len(list_image_files(rgb_dir))
    sem_n = len(list_image_files(sem_dir))
    if rgb_n != sem_n:
        return False, f"rgb/sem count mismatch ({rgb_n}!={sem_n})"
    if rgb_n < min_frames:
        return False, f"too_few_frames ({rgb_n}<{min_frames})"
    return True, "ok"


def _scale_to_img(img_bgr, ref_count: int) -> int:
    """Scale a reference pixel threshold to an image area.

    Parameters
    ----------
    img_bgr
        OpenCV BGR image.
    ref_count
        Pixel threshold calibrated at ``REF_SIDE`` square resolution.

    Returns
    -------
    int
        Area-scaled threshold, lower-bounded at one pixel.
    """
    if ref_count <= 0:
        return 0
    h, w = img_bgr.shape[:2]
    return max(1, int(round(ref_count * (h * w) / REF_AREA)))


def red_pixel_count(img_bgr) -> int:
    """Count strict semantic red pixels in a cam-POV image.

    Parameters
    ----------
    img_bgr
        OpenCV BGR image.

    Returns
    -------
    int
        Strict-red pixel count.
    """
    if img_bgr is None:
        return 0
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return int(((r >= 0.95) & (g <= 0.05) & (b <= 0.05)).sum())


def has_red(img_bgr, threshold=None) -> bool:
    """Return True when strict-red count exceeds the Yes threshold."""
    if img_bgr is None:
        return False
    if threshold is None:
        threshold = _scale_to_img(img_bgr, REF_RED_THRESH)
    return red_pixel_count(img_bgr) > threshold


def has_circle(img_bgr, fill_thresh=0.80, min_area=None) -> bool:
    """Detect any unlabeled circular blob in a semantic image.

    Parameters
    ----------
    img_bgr
        OpenCV BGR image.
    fill_thresh
        Minimum contour-to-enclosing-circle fill ratio.
    min_area
        Optional contour area threshold. If omitted, scales
        ``REF_CONTOUR_MIN`` by image area.

    Returns
    -------
    bool
        True if a sufficiently circular blob is present.
    """
    if img_bgr is None:
        return False
    if min_area is None:
        min_area = _scale_to_img(img_bgr, REF_CONTOUR_MIN)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    for contour in cnts:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        (_, _), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        if circle_area and (area / circle_area) > fill_thresh:
            return True
    return False


def cam_label_matches(cam_img, label: str) -> bool:
    """Match compile_a_star_dataset.py's cam-POV semantic sanity check.

    Yes requires more than ``REF_RED_THRESH`` red pixels. No permits at most
    ``REF_NO_RED_MAX`` red pixels plus the existing no-circle check, so
    REF_NO_RED_MAX+1..threshold is a deadzone.
    """
    if cam_img is None:
        return False
    red_count = red_pixel_count(cam_img)
    yes_thresh = _scale_to_img(cam_img, REF_RED_THRESH)
    no_max = _scale_to_img(cam_img, REF_NO_RED_MAX)
    if label == "Yes":
        return red_count > yes_thresh
    if label == "No":
        return (red_count <= no_max) and (not has_circle(cam_img))
    return False


def parse_raw_gpu_dir(path: Path,
                      allowed_gpus: set[int]) -> tuple[str, int] | None:
    """Return (node_key, gpu_id) if path is a supported raw GPU dir."""
    name = path.name

    m = JOB_NODE_GPU_RE.match(name)
    if m:
        gpu_id = int(m.group("gpu"))
        if gpu_id not in allowed_gpus:
            return None
        # Keep the job id in the key to avoid cross-submission collisions.
        return f"{m.group('job')}_node{m.group('node')}", gpu_id

    m = NODE_GPU_RE.match(name)
    if m:
        gpu_id = int(m.group("gpu"))
        if gpu_id not in allowed_gpus:
            return None
        return m.group("node"), gpu_id

    return None


def discover_nodes(data_dir: Path,
                   allowed_gpus: set[int]) -> dict[str, dict[int, Path]]:
    """Group raw GPU directories by logical node key."""
    nodes: dict[str, dict[int, Path]] = defaultdict(dict)
    for child in sorted(data_dir.iterdir() if data_dir.exists() else []):
        if not child.is_dir():
            continue
        parsed = parse_raw_gpu_dir(child, allowed_gpus)
        if parsed is None:
            continue
        node_key, gpu_id = parsed
        if gpu_id in nodes[node_key]:
            print(
                f"[WARN] duplicate gpu{gpu_id} for node {node_key}: "
                f"keeping {nodes[node_key][gpu_id]}, ignoring {child}",
                file=sys.stderr,
            )
            continue
        nodes[node_key][gpu_id] = child
    return dict(nodes)


def load_tracker(gpu_root: Path) -> list[dict[str, Any]]:
    """Load successful environment records for one raw GPU output.

    Parameters
    ----------
    gpu_root
        Raw ``data_node*_gpu*`` directory.

    Returns
    -------
    list[dict[str, Any]]
        Records from ``successful_envs.json``. Malformed or missing trackers
        return an empty list and emit a warning.
    """
    tracker = gpu_root / "successful_envs.json"
    if not tracker.exists():
        print(f"[WARN] missing tracker: {tracker}", file=sys.stderr)
        return []
    try:
        data = json.loads(tracker.read_text())
    except Exception as exc:  # noqa: BLE001 - keep batch compile alive.
        print(f"[WARN] cannot parse {tracker}: {exc}", file=sys.stderr)
        return []
    envs = data.get("envs", [])
    if not isinstance(envs, list):
        print(f"[WARN] tracker envs is not a list: {tracker}", file=sys.stderr)
        return []
    return envs


def stage_one_env(
    gpu_root: Path,
    label: str,
    folder_idx: int,
    staged_root: Path,
    node_key: str,
    gpu_id: int,
    reason: str,
) -> str | None:
    """Copy one env's RGB/Semantic/cam dirs to staged location."""
    uniq = f"env_n{node_key}_g{gpu_id}_f{folder_idx}"
    staged_paths: list[Path] = []

    for typ in TYPES:
        src = gpu_root / typ / label / "rollout" / f"env_{folder_idx}"
        dst = staged_root / typ / label / reason / uniq
        if not src.is_dir():
            return None
        staged_paths.append(dst)

    for typ, dst in zip(TYPES, staged_paths, strict=True):
        src = gpu_root / typ / label / "rollout" / f"env_{folder_idx}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    return uniq


def normalize_reject(reason: str) -> str:
    """Normalize verbose validation messages into compact reject keys.

    Parameters
    ----------
    reason
        Raw rejection reason.

    Returns
    -------
    str
        Stable summary key for task summaries.
    """
    if reason.startswith("rgb/sem count mismatch"):
        return "rgb_sem_mismatch"
    if reason.startswith("too_few_frames"):
        return "too_few_frames"
    return reason


def process_one_entry(
    gpu_root: Path,
    gpu_id: int,
    node_key: str,
    staged_root: Path,
    entry: dict[str, Any],
    min_frames: int,
    do_cam_check: bool,
) -> tuple[str, str, dict[str, Any] | None]:
    """Validate and stage one tracker entry.

    Returns (status, reason, kept_record). status is "ok" or "reject".
    """
    folder_idx = entry.get("folder_idx")
    label = entry.get("label")
    reason = entry.get("reason")

    if folder_idx is None or label not in LABELS or reason not in REASONS:
        return "reject", "bad_meta", None

    try:
        folder_idx = int(folder_idx)
    except (TypeError, ValueError):
        return "reject", "bad_folder_idx", None

    ok, why = verify_env(gpu_root, label, folder_idx, min_frames)
    if not ok:
        return "reject", normalize_reject(why), None

    if do_cam_check:
        cam_path = (gpu_root / "cam" / label / "rollout" /
                    f"env_{folder_idx}" / CAM_FINAL_NAME)
        cam_img = cv2.imread(str(cam_path), cv2.IMREAD_COLOR)
        if not cam_label_matches(cam_img, label):
            return "reject", "bad_cam", None

    staged_env = stage_one_env(
        gpu_root=gpu_root,
        label=label,
        folder_idx=folder_idx,
        staged_root=staged_root,
        node_key=node_key,
        gpu_id=gpu_id,
        reason=reason,
    )
    if staged_env is None:
        return "reject", "bad_struct", None

    kept = {
        **entry,
        "staged_env": staged_env,
        "task_id": node_key,
        "node_key": node_key,
        "gpu_id": gpu_id,
        "src_gpu_root": str(gpu_root),
        "src_folder_idx": folder_idx,
    }
    return "ok", reason, kept


def build_work_items(
    node_key: str,
    gpu_dirs: dict[int, Path],
) -> tuple[list[tuple[Path, int, dict[str, Any]]], Counter]:
    """Read trackers and return deduplicated compile work items."""
    items: list[tuple[Path, int, dict[str, Any]]] = []
    rejects: Counter = Counter()
    seen: set[tuple[int, int, str, str]] = set()

    for gpu_id, gpu_root in sorted(gpu_dirs.items()):
        envs = load_tracker(gpu_root)
        for entry in envs:
            folder_idx = entry.get("folder_idx")
            label = entry.get("label")
            reason = entry.get("reason")
            try:
                folder_idx_int = int(folder_idx)
            except (TypeError, ValueError):
                rejects["bad_folder_idx"] += 1
                continue
            key = (gpu_id, folder_idx_int, str(label), str(reason))
            if key in seen:
                rejects["duplicate_tracker_entry"] += 1
                continue
            seen.add(key)
            items.append((gpu_root, gpu_id, entry))

    if not items:
        print(f"[WARN] node {node_key}: no tracker entries found",
              file=sys.stderr)
    return items, rejects


def compile_node(
    base_path: Path,
    node_key: str,
    gpu_dirs: dict[int, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Compile all available GPUs for one logical node."""
    started = time.time()
    data_dir = base_path / "data"
    staged_root = data_dir / f"data_node{node_key}_compiled"

    missing = sorted(set(args.gpus) - set(gpu_dirs))
    if missing:
        print(f"[WARN] node {node_key}: missing GPUs {missing}; "
              "compiling available GPUs")

    if staged_root.exists():
        if args.overwrite:
            print(
                f"[node {node_key}] removing existing compiled dir: {staged_root}"
            )
            shutil.rmtree(staged_root)
        else:
            print(f"[SKIP] node {node_key}: compiled dir exists "
                  f"({staged_root}); use --overwrite")
            return {
                "node_key": node_key,
                "skipped": True,
                "reason": "compiled_exists",
                "staged_root": str(staged_root),
            }

    work_items, pre_rejects = build_work_items(node_key, gpu_dirs)
    print(
        f"[node {node_key}] GPUs={sorted(gpu_dirs)} entries={len(work_items)} "
        f"-> {staged_root}")

    if args.dry_run:
        return {
            "node_key": node_key,
            "dry_run": True,
            "gpu_dirs": {str(k): str(v)
                         for k, v in sorted(gpu_dirs.items())},
            "entries": len(work_items),
            "pre_rejected": dict(pre_rejects),
            "staged_root": str(staged_root),
        }

    do_cam_check = not args.no_cam_check
    load_validation_helpers(do_cam_check)

    counts: Counter = Counter(pre_rejects)
    by_reason: Counter = Counter({reason: 0 for reason in REASONS})
    kept_records: list[dict[str, Any]] = []

    staged_root.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, int(args.workers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_one_entry,
                gpu_root,
                gpu_id,
                node_key,
                staged_root,
                entry,
                args.min_frames,
                do_cam_check,
            ) for gpu_root, gpu_id, entry in work_items
        ]
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                status, reason, kept = fut.result()
            except Exception as exc:  # noqa: BLE001 - continue batch.
                counts["exception"] += 1
                print(f"[WARN] node {node_key}: worker exception: {exc}",
                      file=sys.stderr)
                continue

            if status == "ok" and kept is not None:
                counts["ok"] += 1
                by_reason[reason] += 1
                kept_records.append(kept)
            else:
                counts[reason] += 1

            if args.progress_every and done % args.progress_every == 0:
                print(f"[node {node_key}] progress {done}/{len(work_items)} "
                      f"kept={counts['ok']} rejected="
                      f"{sum(v for k, v in counts.items() if k != 'ok')}")

    kept_records.sort(key=lambda rec: (
        int(rec.get("gpu_id", -1)),
        int(rec.get("src_folder_idx", -1)),
        str(rec.get("staged_env", "")),
    ))

    tracker = {
        "total": int(counts["ok"]),
        "by_reason": {reason: int(by_reason[reason])
                      for reason in REASONS},
        "envs": kept_records,
    }
    (staged_root / "successful_envs.json").write_text(
        json.dumps(tracker, indent=2))

    summary = {
        "node_key": node_key,
        "kept": int(counts["ok"]),
        "rejected":
        {k: int(v)
         for k, v in sorted(counts.items()) if k != "ok"},
        "by_reason": {reason: int(by_reason[reason])
                      for reason in REASONS},
        "gpu_dirs": {str(k): str(v)
                     for k, v in sorted(gpu_dirs.items())},
        "staged_root": str(staged_root),
        "elapsed_sec": round(time.time() - started, 3),
        "cam_check": do_cam_check,
        "min_frames": args.min_frames,
    }
    (staged_root / "task_summary.json").write_text(
        json.dumps(summary, indent=2))

    print(f"[node {node_key}] kept={summary['kept']} "
          f"rejected={sum(summary['rejected'].values())} "
          f"reasons={summary['by_reason']} "
          f"elapsed={summary['elapsed_sec']:.1f}s")

    if args.clean and counts["ok"] > 0:
        print(f"[node {node_key}] --clean enabled; removing raw GPU dirs")
        for gpu_root in gpu_dirs.values():
            shutil.rmtree(gpu_root, ignore_errors=True)

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI settings for bulk node compilation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_path",
        required=True,
        help="Collector BASE_PATH, i.e. parent of the data/ directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Thread workers for validation/copy inside each node.",
    )
    parser.add_argument(
        "--node_workers",
        type=int,
        default=1,
        help="How many node groups to compile concurrently. Total copy "
        "threads can reach workers * node_workers.",
    )
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument(
        "--cam_red_thresh",
        type=int,
        default=125,
        help="Minimum red pixels for cam-POV Yes at 256x256. "
        "Auto-scales by image area. Default: 125.",
    )
    parser.add_argument(
        "--cam_no_red_max",
        type=int,
        default=0,
        help="Maximum red pixels allowed for cam-POV No at 256x256. "
        "Auto-scales by image area. Default: 0.",
    )
    parser.add_argument(
        "--no_cam_check",
        action="store_true",
        help="Skip cam-POV semantic sanity check.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover node/GPU groups and tracker sizes without copying.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild compiled node dirs if they already exist.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=
        "Delete raw per-GPU dirs after a node is compiled. Default keeps raw.",
    )
    parser.add_argument(
        "--limit_nodes",
        type=int,
        default=0,
        help=
        "Compile only the first N discovered nodes, useful for smoke tests.",
    )
    parser.add_argument(
        "--only_node",
        action="append",
        default=[],
        help="Compile only this exact node key. Can be passed multiple times.",
    )
    parser.add_argument(
        "--node_prefix",
        action="append",
        default=[],
        help="Compile only node keys starting with this prefix, e.g. a SLURM "
        "job id like 2451255_. Can be passed multiple times.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0, 1],
        help="GPU ids to consider. Default: 0 1.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=250,
        help=
        "Print per-node progress every N envs. 0 disables progress prints.",
    )
    return parser.parse_args()


def main() -> None:
    """Run bulk raw-GPU to per-node compiled staging pipeline."""
    global REF_RED_THRESH, REF_NO_RED_MAX
    args = parse_args()
    REF_RED_THRESH = args.cam_red_thresh
    REF_NO_RED_MAX = args.cam_no_red_max
    if REF_NO_RED_MAX >= REF_RED_THRESH:
        print("[ERR] --cam_no_red_max must be less than --cam_red_thresh",
              file=sys.stderr)
        sys.exit(2)
    base_path = Path(args.base_path).expanduser().resolve()
    data_dir = base_path / "data"

    if not data_dir.is_dir():
        print(f"[ERR] data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    allowed_gpus = set(args.gpus)
    nodes = discover_nodes(data_dir, allowed_gpus)
    if args.only_node:
        wanted = set(args.only_node)
        nodes = {node: gpus for node, gpus in nodes.items() if node in wanted}
    if args.node_prefix:
        prefixes = tuple(args.node_prefix)
        nodes = {
            node: gpus
            for node, gpus in nodes.items() if node.startswith(prefixes)
        }

    node_items = sorted(nodes.items(), key=lambda item: item[0])
    if args.limit_nodes and args.limit_nodes > 0:
        node_items = node_items[:args.limit_nodes]

    if not node_items:
        print(
            f"[ERR] no raw GPU dirs found under {data_dir} "
            f"for GPUs {sorted(allowed_gpus)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] base_path={base_path}\n"
          f"[INFO] nodes={len(node_items)} workers={args.workers} "
          f"node_workers={args.node_workers} gpus={sorted(allowed_gpus)} "
          f"dry_run={args.dry_run} clean={args.clean}")

    all_summaries = []
    started = time.time()
    node_workers = max(1, int(args.node_workers))
    if node_workers == 1:
        for node_key, gpu_dirs in node_items:
            all_summaries.append(
                compile_node(base_path, node_key, gpu_dirs, args))
    else:
        with ThreadPoolExecutor(max_workers=node_workers) as executor:
            futures = [
                executor.submit(compile_node, base_path, node_key, gpu_dirs,
                                args) for node_key, gpu_dirs in node_items
            ]
            for fut in as_completed(futures):
                all_summaries.append(fut.result())

    total_kept = sum(int(s.get("kept", 0)) for s in all_summaries)
    total_rejected = sum(
        sum(int(v) for v in (s.get("rejected") or {}).values())
        for s in all_summaries)
    print("-" * 80)
    print(f"[DONE] nodes={len(all_summaries)} kept={total_kept} "
          f"rejected={total_rejected} elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
