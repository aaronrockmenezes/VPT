#!/usr/bin/env python3
"""Carve a small VPT1-style linear-probe dataset from A* rollouts.

This is the fast "test right now" path for partially collected A* data. It
borrows the A* compiler discovery/QC logic, samples an exact 50/25/25 episode
mix, then writes:

    out/
    ├── visibility_labels.json
    ├── master_labels.json
    ├── RGB/{Yes,No}/env_*/
    ├── Semantic/{Yes,No}/env_*/
    ├── cam/{Yes,No}/env_*/
    ├── train/{Yes,No}/env_*/      # RGB-only view for VPT1 probe loaders
    └── test/{Yes,No}/env_*/       # RGB-only view for VPT1 probe loaders

By default it symlinks env directories for speed and low disk use. The VPT1
``train/`` and ``test/`` views are first-frame-only by default so linear probes
use one image per episode. Pass ``--all_split_frames`` to expose all rollout
frames in those split folders. Pass ``--copy`` if the training code cannot
follow symlinks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
ASTAR_SCRIPT_DIR = REPO_ROOT / "VPTnav_code" / "cube_game" / "scripts"
sys.path.insert(0, str(ASTAR_SCRIPT_DIR))

import compile_a_star_dataset as astar  # noqa: E402

DEFAULT_REASON_FRACTIONS = {
    "in_view": 0.50,
    "occluded": 0.25,
    "outside_fov": 0.25,
}


def _allocate_counts(total: int, fractions: dict[str,
                                                 float]) -> dict[str, int]:
    """Return integer reason counts summing exactly to ``total``."""
    counts = {k: int(round(total * v)) for k, v in fractions.items()}
    diff = total - sum(counts.values())
    counts["in_view"] += diff
    return counts


def _label_for_reason(reason: str) -> str:
    """Map A* visibility reason to VPT visibility label."""
    return "Yes" if reason == "in_view" else "No"


def _make_dirs(out: Path) -> None:
    """Create canonical A* dirs plus VPT1 train/test RGB split dirs."""
    for typ in astar.TYPES:
        for label in ("Yes", "No"):
            (out / typ / label).mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for label in ("Yes", "No"):
            (out / split / label).mkdir(parents=True, exist_ok=True)


def _safe_replace_dir(dst: Path) -> None:
    """Remove an existing destination dir or symlink."""
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)


def _materialize_dir(src: Path, dst: Path, copy: bool) -> None:
    """Copy or symlink one source directory to ``dst``."""
    _safe_replace_dir(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src, dst, target_is_directory=True)


def _materialize_first_frame(src: Path, dst: Path, copy: bool) -> None:
    """Write a split env dir containing only ``step_00000_start.png``."""
    _safe_replace_dir(dst)
    dst.mkdir(parents=True, exist_ok=True)
    src_file = src / "step_00000_start.png"
    if not src_file.exists():
        raise FileNotFoundError(f"missing first frame: {src_file}")
    dst_file = dst / src_file.name
    if copy:
        shutil.copy2(src_file, dst_file)
    else:
        os.symlink(src_file, dst_file)


def _src_dir(item: dict, typ: str, mode: str) -> Path:
    """Return source dir for one selected A* env item."""
    label = item["label"]
    if mode == "staged":
        return (Path(item["src_task_dir"]) / typ / label / item["reason"] /
                item["env_name"])
    return (Path(item["src_gpu_root"]) / typ / label / "rollout" /
            f"env_{item['folder_idx']}")


def _write_one_env(
    args_tuple: tuple[int, dict, Path, str, bool, int,
                      bool]) -> tuple[int, dict]:
    """Write canonical dirs and RGB split view for one selected env."""
    new_id, item, out, mode, copy, train_count, first_frame_only = args_tuple
    label = item["label"]
    split = "train" if new_id < train_count else "test"

    for typ in astar.TYPES:
        _materialize_dir(_src_dir(item, typ, mode),
                         out / typ / label / f"env_{new_id}", copy)

    rgb_src = _src_dir(item, "RGB", mode)
    split_dst = out / split / label / f"env_{new_id}"
    if first_frame_only:
        _materialize_first_frame(rgb_src, split_dst, copy)
    else:
        _materialize_dir(rgb_src, split_dst, copy)
    return new_id, item


def _discover(args: argparse.Namespace) -> tuple[dict[str, list], dict]:
    """Discover raw or staged A* envs using the shared compiler QC."""
    src_root = Path(args.src_root)
    if args.mode == "staged":
        return astar.discover_staged_envs(src_root,
                                          job_id=args.job_id,
                                          do_cam_check=not args.no_cam_check)
    return astar.discover_envs(src_root, args.min_frames,
                               not args.no_cam_check)


def _build_ordered(pool: dict[str, list], total: int, seed: int) -> list[dict]:
    """Sample exact 50/25/25 globally and exactly half per train/test."""
    rng = random.Random(seed)
    train_total = total // 2
    test_total = total - train_total
    train_targets = _allocate_counts(train_total, DEFAULT_REASON_FRACTIONS)
    test_targets = _allocate_counts(test_total, DEFAULT_REASON_FRACTIONS)
    total_targets = {
        k: train_targets[k] + test_targets[k]
        for k in DEFAULT_REASON_FRACTIONS
    }

    for reason, need in total_targets.items():
        have = len(pool.get(reason, []))
        if have < need:
            raise SystemExit(
                f"[FATAL] not enough {reason}: need {need}, have {have}")

    train_batch: list[dict] = []
    test_batch: list[dict] = []
    for reason in DEFAULT_REASON_FRACTIONS:
        items = list(pool[reason])
        rng.shuffle(items)
        n_train = train_targets[reason]
        n_test = test_targets[reason]
        train_batch.extend(items[:n_train])
        test_batch.extend(items[n_train:n_train + n_test])

    rng.shuffle(train_batch)
    rng.shuffle(test_batch)
    return train_batch + test_batch


def _build_master(copied: list[tuple[int, dict]], args: argparse.Namespace,
                  train_count: int) -> dict:
    """Build both A* master and VPT1-compatible label metadata."""
    envs = {}
    stats = {
        "total_environments": len(copied),
        "yes_count": 0,
        "no_count": 0,
        "by_reason": {k: 0
                      for k in DEFAULT_REASON_FRACTIONS},
        "splits": {
            "train": train_count,
            "test": len(copied) - train_count,
        },
        "split_ranges": {
            "train": [0, train_count - 1],
            "test": [train_count, len(copied) - 1],
        },
        "src_root": str(Path(args.src_root)),
        "mode": args.mode,
        "seed": args.seed,
        "symlink": not args.copy,
        "first_frame_only_splits": not args.all_split_frames,
    }

    for new_id, item in copied:
        split = "train" if new_id < train_count else "test"
        label = item["label"]
        reason = item["reason"]
        rec = {
            "label": label,
            "reason": reason,
            "split": split,
            "n_frames": int(item.get("n_frames", 0) or 0),
            "src": item.get("src_task_dir") or item.get("src_gpu_root"),
            "src_id": item.get("env_name") or f"env_{item.get('folder_idx')}",
            "original": item.get("original") or item.get("meta") or {},
        }
        envs[str(new_id)] = rec
        stats["by_reason"][reason] += 1
        stats["yes_count" if label == "Yes" else "no_count"] += 1

    return {"environments": envs, "statistics": stats}


def _print_balance(name: str, records: list[dict]) -> None:
    """Print label/reason counts for a set of records."""
    labels = Counter(r["label"] for r in records)
    reasons = Counter(r["reason"] for r in records)
    print(f"[BALANCE] {name}: n={len(records)} "
          f"labels={dict(labels)} reasons={dict(reasons)}")


def main() -> None:
    """Run CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src_root",
                    required=True,
                    help="A* collector root, parent of data/.")
    ap.add_argument("--out_dir", required=True, help="Output dataset root.")
    ap.add_argument("--total",
                    type=int,
                    default=1024,
                    help="Total envs to carve. Default: 1024.")
    ap.add_argument("--mode",
                    choices=("raw", "staged"),
                    default="raw",
                    help="raw scans data_node*_gpu*; staged scans compiled.")
    ap.add_argument("--job_id",
                    default=None,
                    help="Optional staged job_id filter.")
    ap.add_argument("--min_frames",
                    type=int,
                    default=10,
                    help="Raw-mode min RGB frame count. Default: 10.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers",
                    type=int,
                    default=32,
                    help="Parallel copy/symlink workers. Default: 32.")
    ap.add_argument("--copy",
                    action="store_true",
                    help="Copy dirs instead of symlinking.")
    ap.add_argument("--all_split_frames",
                    action="store_true",
                    help="Expose every rollout RGB frame in train/test. "
                    "Default is first-frame-only splits.")
    ap.add_argument("--overwrite",
                    action="store_true",
                    help="Delete out_dir before writing.")
    ap.add_argument("--no_cam_check",
                    action="store_true",
                    help="Skip cam semantic label QC.")
    ap.add_argument("--cam_red_thresh",
                    type=int,
                    default=125,
                    help="Yes strict-red threshold at 256x256.")
    ap.add_argument("--cam_no_red_max",
                    type=int,
                    default=0,
                    help="No max strict-red count at 256x256.")
    args = ap.parse_args()

    if args.total % 4 != 0:
        raise SystemExit("[FATAL] --total must be divisible by 4 for 50/25/25")

    astar.REF_RED_THRESH = args.cam_red_thresh
    astar.REF_NO_RED_MAX = args.cam_no_red_max
    if astar.REF_NO_RED_MAX >= astar.REF_RED_THRESH:
        raise SystemExit("[FATAL] --cam_no_red_max must be < --cam_red_thresh")

    out = Path(args.out_dir)
    if out.exists() and args.overwrite:
        print(f"[INFO] removing existing out_dir: {out}")
        shutil.rmtree(out)
    elif out.exists():
        raise SystemExit(f"[FATAL] out_dir exists; use --overwrite: {out}")
    _make_dirs(out)

    print(f"[INFO] discovering src={args.src_root} mode={args.mode}")
    pool, rejected = _discover(args)
    print("[INFO] pool sizes:")
    for reason in DEFAULT_REASON_FRACTIONS:
        print(f"  {reason:>12}: {len(pool.get(reason, []))}")
    nonzero_rej = {k: v for k, v in rejected.items() if v}
    if nonzero_rej:
        print(f"[INFO] rejected={nonzero_rej}")

    ordered = _build_ordered(pool, args.total, args.seed)
    train_count = args.total // 2
    _print_balance("ALL", ordered)
    _print_balance("TRAIN", ordered[:train_count])
    _print_balance("TEST", ordered[train_count:])

    copy_mode = "copy" if args.copy else "symlink"
    split_mode = "all frames" if args.all_split_frames else "first frame only"
    print(f"[INFO] writing {len(ordered)} envs to {out} "
          f"({copy_mode}, split view={split_mode})")
    jobs = [(i, item, out, args.mode, args.copy, train_count,
             not args.all_split_frames) for i, item in enumerate(ordered)]

    copied = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_write_one_env, job) for job in jobs]
        for fut in tqdm(as_completed(futs),
                        total=len(futs),
                        desc=f"Writing ({args.workers} workers)"):
            copied.append(fut.result())

    copied.sort(key=lambda x: x[0])
    master = _build_master(copied, args, train_count)
    (out / "master_labels.json").write_text(json.dumps(master, indent=2))
    (out / "visibility_labels.json").write_text(json.dumps(master, indent=2))

    print("[DONE] wrote dataset")
    print(f"  out_dir: {out}")
    print(f"  stats: {master['statistics']}")


if __name__ == "__main__":
    main()
