#!/usr/bin/env python3
"""Audit A* rollout starts for VPTnav-style valid-viewpoint invariants.

This is intentionally metadata-first and fast. It checks each saved rollout's
``cam/.../meta.json`` for the hard guarantees written by
``prepare_astar_valid_start`` and verifies the first-frame image files exist.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def iter_meta_paths(root: Path):
    """Yield rollout meta files under raw, staged, or final dataset roots."""
    search_root = root / "data" if (root / "data").is_dir() else root
    yield from search_root.glob("**/cam/*/rollout/env_*/meta.json")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Audit A* valid-viewpoint start metadata.")
    parser.add_argument("root", type=Path, help="A* base path or data root.")
    parser.add_argument("--goal_px", type=int, default=500)
    parser.add_argument("--camera_px", type=int, default=800)
    parser.add_argument("--cam_goal_px", type=int, default=500)
    parser.add_argument("--start_half_extent", type=float, default=6.0)
    parser.add_argument("--start_deadzone", type=float, default=3.0)
    parser.add_argument("--allow_random_start", action="store_true")
    return parser.parse_args()


def image_paths_for_meta(meta_path: Path):
    """Return expected RGB/Semantic start-image paths for a rollout meta."""
    env_dir = meta_path.parent
    label = env_dir.parent.parent.name
    rollout_dir = env_dir.parent.name
    env_name = env_dir.name
    cam_root = env_dir.parents[3]
    rgb = cam_root / "RGB" / label / rollout_dir / env_name / "step_00000_start.png"
    sem = cam_root / "Semantic" / label / rollout_dir / env_name / "step_00000_start.png"
    return rgb, sem


def check_one(meta_path: Path, args):
    """Return list of failure reasons for one meta file."""
    failures = []
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        return [f"bad_json:{exc}"]

    start = meta.get("start") or {}
    source = start.get("start_source")
    if source != "valid_viewpoint_0" and not args.allow_random_start:
        failures.append(f"bad_start_source:{source}")
    if not start.get("start_valid", False):
        failures.append(f"start_invalid:{start.get('start_fail_reason', '')}")

    goal_px = int(start.get("agent_first_view_goal_px", -1))
    camera_px = int(start.get("agent_first_view_camera_px", -1))
    cam_goal_px = int(start.get("camera_pov_goal_px", -1))
    if goal_px < args.goal_px:
        failures.append(f"goal_px:{goal_px}")
    if camera_px < args.camera_px:
        failures.append(f"camera_px:{camera_px}")

    reason = meta.get("reason", "unknown")
    if reason == "in_view" and cam_goal_px < args.cam_goal_px:
        failures.append(f"cam_goal_px_in_view:{cam_goal_px}")
    if reason in ("occluded", "outside_fov") and cam_goal_px >= args.cam_goal_px:
        failures.append(f"cam_goal_px_no:{cam_goal_px}")

    dx, dy = start.get("start_delta_xy_from_camera", [None, None])
    if dx is None or dy is None:
        failures.append("missing_start_delta")
    else:
        if abs(float(dx)) > args.start_half_extent or abs(
                float(dy)) > args.start_half_extent:
            failures.append(f"outside_square:{dx},{dy}")
        if abs(float(dx)) < args.start_deadzone and abs(
                float(dy)) < args.start_deadzone:
            failures.append(f"inside_deadzone:{dx},{dy}")

    rgb, sem = image_paths_for_meta(meta_path)
    if not rgb.exists():
        failures.append(f"missing_rgb_start:{rgb}")
    if not sem.exists():
        failures.append(f"missing_sem_start:{sem}")
    return failures


def main():
    """Run audit and exit nonzero on invariant violations."""
    args = parse_args()
    root = args.root.expanduser().resolve()
    paths = sorted(iter_meta_paths(root))
    if not paths:
        raise SystemExit(f"[ERR] no rollout meta.json files found under {root}")

    reason_counts = Counter()
    failures = {}
    for path in paths:
        try:
            meta = json.loads(path.read_text())
            reason_counts[meta.get("reason", "unknown")] += 1
        except Exception:
            reason_counts["bad_json"] += 1
        bad = check_one(path, args)
        if bad:
            failures[str(path)] = bad

    print(f"[INFO] root={root}")
    print(f"[INFO] checked={len(paths)}")
    print(f"[INFO] reason_counts={dict(reason_counts)}")
    if failures:
        print(f"[FAIL] bad_rollouts={len(failures)}")
        for idx, (path, bad) in enumerate(failures.items()):
            if idx >= 20:
                print(f"... {len(failures) - idx} more")
                break
            print(path)
            print("  " + "; ".join(bad))
        raise SystemExit(1)
    print("[OK] all A* starts passed metadata audit")


if __name__ == "__main__":
    main()
