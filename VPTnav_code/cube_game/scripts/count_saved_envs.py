"""Count successfully saved camera-move envs.

A env counts as "successfully saved" when its config JSON exists — the env writes
the config last, after every frame image, so config presence == complete env.

With --verify, also checks the RGB image dir exists and has as many image_*d.png
files as the config's trajectory length (catches partial/corrupt saves).

Also reports saved-frame label/reason counts from:
  camera_move_collection.trajectory[].label
  camera_move_collection.trajectory[].reason

Layout (multi-GPU/multi-node):
  {base}/data/data_node*_gpu*/configs/env_*_config.json
  {base}/data/data_node*_gpu*/RGB/Mixed/env_*/image_*d.png

Usage:
  python scripts/count_saved_envs.py --base_path /oscar/scratch/arock3/VPT_DATA_CAM_MOVE/v1
  python scripts/count_saved_envs.py --base_path <p> --verify
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict


def shard_of(config_path):
    """Return the data_node*_gpu* shard name for a config path, or '.' if flat."""
    # .../<shard>/configs/env_*_config.json
    parts = os.path.normpath(config_path).split(os.sep)
    try:
        return parts[parts.index("configs") - 1]
    except (ValueError, IndexError):
        return "."


def add_trajectory_counts(cfg, label_counts, reason_counts):
    """Accumulate per-frame labels/reasons from a camera-move config."""
    traj = cfg.get("camera_move_collection", {}).get("trajectory", [])
    for frame in traj:
        label = frame.get("label") or "unknown"
        reason = frame.get("reason") or "unknown"
        label_counts[label] += 1
        reason_counts[reason] += 1
    return len(traj)


def print_breakdown(title, counts, preferred_order, total):
    print(f"  {title}:")
    shown = set()
    for key in preferred_order:
        value = counts.get(key, 0)
        shown.add(key)
        pct = (100.0 * value / total) if total else 0.0
        print(f"    {key:12s} {value:8d}  ({pct:5.1f}%)")
    for key in sorted(k for k in counts if k not in shown):
        value = counts[key]
        pct = (100.0 * value / total) if total else 0.0
        print(f"    {key:12s} {value:8d}  ({pct:5.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="Count successfully saved camera-move envs.")
    ap.add_argument("--base_path", required=True,
                    help="Run root (contains data/data_node*_gpu*/configs/).")
    ap.add_argument("--verify", action="store_true",
                    help="Also check RGB image count matches config trajectory length.")
    args = ap.parse_args()

    patterns = [
        os.path.join(args.base_path, "data", "*", "configs", "env_*_config.json"),
        os.path.join(args.base_path, "**", "configs", "env_*_config.json"),
        os.path.join(args.base_path, "configs", "env_*_config.json"),
    ]
    seen = set()
    configs = []
    for pat in patterns:
        for fp in glob.glob(pat, recursive=True):
            rp = os.path.realpath(fp)
            if rp not in seen:
                seen.add(rp)
                configs.append(fp)

    if not configs:
        print(f"[count] no configs found under {args.base_path}")
        return

    per_shard = defaultdict(int)
    label_counts = Counter()
    reason_counts = Counter()
    total_frames = 0
    incomplete = []  # (config_path, reason)
    unreadable = []  # (config_path, reason)

    for fp in configs:
        shard = shard_of(fp)

        try:
            with open(fp) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if args.verify:
                incomplete.append((fp, f"unreadable config: {e}"))
            else:
                per_shard[shard] += 1
                unreadable.append((fp, f"unreadable config: {e}"))
            continue

        traj = cfg.get("camera_move_collection", {}).get("trajectory", [])
        n_traj = len(traj)

        if args.verify:
            folder_idx = cfg.get("metadata", {}).get("folder_idx")

            # configs/ -> shard root -> RGB/Mixed/env_{folder_idx}
            shard_root = os.path.dirname(os.path.dirname(fp))
            rgb_dir = os.path.join(shard_root, "RGB", "Mixed", f"env_{folder_idx}")
            if not os.path.isdir(rgb_dir):
                incomplete.append((fp, f"missing RGB dir env_{folder_idx}"))
                continue
            n_img = len([x for x in os.listdir(rgb_dir)
                         if re.fullmatch(r"image_-?\d+d_\w+\.png", x)])
            if n_img != n_traj:
                incomplete.append((fp, f"img/traj mismatch: {n_img} png vs {n_traj} frames"))
                continue

        per_shard[shard] += 1
        total_frames += add_trajectory_counts(cfg, label_counts, reason_counts)

    total = sum(per_shard.values())
    print("=" * 52)
    print(f"[count] base_path={args.base_path}")
    print(f"[count] mode: {'verified (config + image check)' if args.verify else 'config-only'}")
    print("-" * 52)
    for shard in sorted(per_shard):
        print(f"  {shard:32s} {per_shard[shard]:6d}")
    print("-" * 52)
    print(f"  TOTAL successfully saved envs: {total}")
    print(f"  TOTAL saved trajectory frames: {total_frames}")
    print("-" * 52)
    print_breakdown("LABELS", label_counts, ["Yes", "No", "unknown"], total_frames)
    print("-" * 52)
    print_breakdown("REASONS", reason_counts,
                    ["in_view", "occluded", "outside_fov", "unknown"],
                    total_frames)
    if unreadable:
        print("-" * 52)
        print(f"  UNREADABLE configs counted as saved envs, excluded from frame breakdown: {len(unreadable)}")
        for fp, reason in unreadable[:20]:
            print(f"    {os.path.basename(fp)}: {reason}")
        if len(unreadable) > 20:
            print(f"    ... and {len(unreadable) - 20} more")
    if args.verify and incomplete:
        print("-" * 52)
        print(f"  INCOMPLETE / corrupt: {len(incomplete)}")
        for fp, reason in incomplete[:20]:
            print(f"    {os.path.basename(fp)}: {reason}")
        if len(incomplete) > 20:
            print(f"    ... and {len(incomplete) - 20} more")
    print("=" * 52)


if __name__ == "__main__":
    main()
