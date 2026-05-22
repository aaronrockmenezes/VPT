#!/usr/bin/env python3
"""Per-task compile: verify + sanity-check envs, copy to staging, clean raw.

Cleanup uses ThreadPoolExecutor to parallel-rm leaf env_{N} dirs across
all 6 GPU subtrees — Lustre rm is per-syscall, GIL releases on unlink,
so threads scale well.


Run after each array task finishes to:
  1. Walk {base_path}/data/data_node{node_id}_gpu*/
  2. Validate every env from successful_envs.json (dirs, frame counts,
     cam-POV semantic consistent with label).
  3. Copy good envs to {base_path}/data/data_node{node_id}_compiled/.
  4. Remove the raw data_node{node_id}_gpu*/ dirs.

Final cross-task merge happens later via compile_a_star_dataset.py.
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

# Reuse sanity-check helpers from full compiler (relative import).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compile_a_star_dataset as compiler  # noqa: E402
from compile_a_star_dataset import (  # noqa: E402
    cam_label_matches, list_image_files, verify_env, CAM_FINAL_NAME, TYPES,
)


def stage_one_env(gpu_root: Path, label: str, folder_idx: int,
                  staged_root: Path, node_id: str, gpu_id: int,
                  reason: str) -> str | None:
    """Copy one env's RGB/Semantic/cam dirs to staged location.
    Returns the unique staged env name on success, else None.
    """
    uniq = f"env_n{node_id}_g{gpu_id}_f{folder_idx}"
    for t in TYPES:
        src = gpu_root / t / label / "rollout" / f"env_{folder_idx}"
        dst = staged_root / t / label / reason / uniq
        if not src.is_dir():
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path",
                    required=True,
                    help="Collector BASE_PATH (parent of `data/`).")
    # Either --node_id (full namespaced id like "12345_5") or legacy
    # --task_id (just int, treated as node_id).
    ap.add_argument("--node_id",
                    type=str,
                    default=None,
                    help="Full NODE_ID like '{job}_{task}'. Used for both "
                    "raw glob and compiled output dir.")
    ap.add_argument("--task_id",
                    type=int,
                    default=None,
                    help="Legacy. Used only if --node_id not given.")
    ap.add_argument("--min_frames", type=int, default=10)
    ap.add_argument("--cam_red_thresh",
                    type=int,
                    default=125,
                    help="Yes requires more than this many strict-red pixels "
                    "at 256x256. No requires zero red pixels.")
    ap.add_argument("--cam_no_red_max",
                    type=int,
                    default=0,
                    help="No permits at most this many strict-red pixels at "
                    "256x256 before entering deadzone.")
    ap.add_argument("--no_cam_check", action="store_true")
    ap.add_argument("--no_clean",
                    action="store_true",
                    help="Skip removing raw data dirs after staging.")
    ap.add_argument("--rm_workers",
                    type=int,
                    default=16,
                    help="Parallel rm thread pool size for cleanup.")
    args = ap.parse_args()
    compiler.REF_RED_THRESH = args.cam_red_thresh
    compiler.REF_NO_RED_MAX = args.cam_no_red_max
    if args.cam_no_red_max >= args.cam_red_thresh:
        print("[ERR] --cam_no_red_max must be less than --cam_red_thresh",
              file=sys.stderr)
        sys.exit(2)

    base = Path(args.base_path)
    data_dir = base / "data"

    # Resolve node identifier: prefer --node_id, fall back to --task_id.
    if args.node_id is not None:
        node_id = args.node_id
    elif args.task_id is not None:
        node_id = str(args.task_id)
    else:
        print("[ERR] need either --node_id or --task_id", file=sys.stderr)
        sys.exit(1)

    # Compiled output sits alongside raw GPU dirs so count_successful_envs.py
    # picks it up via the same glob pattern.
    staged_root = data_dir / f"data_node{node_id}_compiled"

    # Find this task's GPU dirs (filter out .txt logs).
    gpu_dirs = sorted(p for p in data_dir.glob(f"data_node{node_id}_gpu*")
                      if p.is_dir())
    if not gpu_dirs:
        print(f"[ERR] no data_node{node_id}_gpu* dirs under {data_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[node {node_id}] found {len(gpu_dirs)} GPU dirs")

    counts = {"ok": 0, "bad_struct": 0, "bad_cam": 0, "no_meta": 0}
    by_reason = {"in_view": 0, "occluded": 0, "outside_fov": 0}
    kept_records = []  # list of dicts; aggregated into successful_envs.json

    for gpu_root in gpu_dirs:
        try:
            gpu_id = int(gpu_root.name.split("_gpu")[-1])
        except ValueError:
            continue  # skip stray non-numeric matches
        tracker = gpu_root / "successful_envs.json"
        if not tracker.exists():
            continue
        try:
            envs = json.loads(tracker.read_text()).get("envs", [])
        except Exception as e:
            print(f"[WARN] can't read {tracker}: {e}")
            continue

        for entry in envs:
            folder_idx = entry.get("folder_idx")
            label = entry.get("label")
            reason = entry.get("reason")
            if folder_idx is None or label not in ("Yes", "No") \
                    or reason not in by_reason:
                counts["no_meta"] += 1
                continue

            ok, _ = verify_env(gpu_root, label, folder_idx, args.min_frames)
            if not ok:
                counts["bad_struct"] += 1
                continue

            if not args.no_cam_check:
                cam_path = (gpu_root / "cam" / label / "rollout" /
                            f"env_{folder_idx}" / CAM_FINAL_NAME)
                cam_img = cv2.imread(str(cam_path), cv2.IMREAD_COLOR)
                if not cam_label_matches(cam_img, label):
                    counts["bad_cam"] += 1
                    continue

            uniq = stage_one_env(gpu_root, label, folder_idx, staged_root,
                                 node_id, gpu_id, reason)
            if uniq:
                counts["ok"] += 1
                by_reason[reason] += 1
                kept_records.append({
                    **entry,
                    "staged_env": uniq,
                    "task_id": node_id,
                    "gpu_id": gpu_id,
                    "src_gpu_root": str(gpu_root),
                    "src_folder_idx": folder_idx,
                })
            else:
                counts["bad_struct"] += 1

    staged_root.mkdir(parents=True, exist_ok=True)

    # successful_envs.json — same shape as collector's tracker, plus the
    # extra src fields so count_successful_envs.py picks this up.
    tracker = {
        "total": counts["ok"],
        "by_reason": by_reason,
        "envs": kept_records,
    }
    (staged_root / "successful_envs.json").write_text(
        json.dumps(tracker, indent=2))

    # Compile-time meta.
    summary = {
        "task_id": node_id,
        "kept": counts["ok"],
        "rejected": {k: v
                     for k, v in counts.items() if k != "ok"},
        "by_reason": by_reason,
        "staged_root": str(staged_root),
    }
    (staged_root / "task_summary.json").write_text(
        json.dumps(summary, indent=2))

    print(f"[task {node_id}] kept={counts['ok']} "
          f"rejected={sum(v for k,v in counts.items() if k != 'ok')} "
          f"reasons={by_reason}")

    # Cleanup raw GPU dirs in parallel.
    if not args.no_clean:
        # Collect all leaf env_* dirs (deepest level — most files live here).
        leaf_dirs = []
        for gpu_root in gpu_dirs:
            for env_d in gpu_root.rglob("env_*"):
                if env_d.is_dir():
                    leaf_dirs.append(env_d)
        # Also catch the per-GPU JSON/txt files via parent dir cleanup later.

        print(f"[task {node_id}] parallel rm of {len(leaf_dirs)} env "
              f"leaf dirs with {args.rm_workers} workers...")
        with ThreadPoolExecutor(max_workers=args.rm_workers) as ex:
            list(
                ex.map(lambda d: shutil.rmtree(d, ignore_errors=True),
                       leaf_dirs))

        # Sweep up the (now mostly empty) parent gpu dirs.
        for gpu_root in gpu_dirs:
            shutil.rmtree(gpu_root, ignore_errors=True)
        print(f"[task {node_id}] cleanup done")


if __name__ == "__main__":
    main()
