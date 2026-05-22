#!/usr/bin/env python3
"""Fast sanity check for final compiled A* dataset.

Checks canonical tree:
  out_dir/
    master_labels.json
    RGB/{Yes,No}/env_*
    Semantic/{Yes,No}/env_*
    cam/{Yes,No}/env_*

No train/val/test folders should exist; splits live only in master_labels.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

from compile_a_star_dataset import CAM_FINAL_NAME, cam_label_reason

TYPES = ("RGB", "Semantic", "cam")
LABELS = ("Yes", "No")
SPLITS = ("train", "val", "test")
REASONS = ("in_view", "occluded", "outside_fov")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def fail(msg: str, errors: list[str]) -> None:
    """Record and print a sanity-check error.

    Parameters
    ----------
    msg
        Human-readable failure message.
    errors
        Mutable error accumulator.
    """
    errors.append(msg)
    print(f"[ERR] {msg}")


def check_range(name: str, expected_ids: set[int], records: dict[str, dict],
                errors: list[str]) -> None:
    """Validate that a split owns exactly its declared env IDs.

    Parameters
    ----------
    name
        Split name, e.g. ``train``.
    expected_ids
        Env IDs that should belong to the split.
    records
        ``master_labels.json`` environment records.
    errors
        Mutable error accumulator.
    """
    seen = {int(k) for k, v in records.items() if v.get("split") == name}
    missing = expected_ids - seen
    extra = seen - expected_ids
    if missing:
        fail(f"split {name}: missing ids in declared range, n={len(missing)}",
             errors)
    if extra:
        fail(f"split {name}: ids outside declared range, n={len(extra)}",
             errors)


def has_any_file(path: Path) -> bool:
    """Check whether a directory has at least one direct file.

    Parameters
    ----------
    path
        Directory to inspect.

    Returns
    -------
    bool
        True if ``path`` exists and contains a direct file.
    """
    return path.is_dir() and any(p.is_file() for p in path.iterdir())


def count_image_files(path: Path) -> int:
    """Count direct image files in a directory.

    Parameters
    ----------
    path
        Directory to inspect.

    Returns
    -------
    int
        Number of direct files with known image extensions.
    """
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    """Run compiled dataset sanity checks from the command line."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("compiled_path",
                    help="Final compiled dataset dir, e.g. v18_compiled_15k")
    ap.add_argument("--sample",
                    type=int,
                    default=20,
                    help="Number of envs to inspect for files in fast mode.")
    ap.add_argument("--deep",
                    action="store_true",
                    help="Check every env dir exists and has files.")
    ap.add_argument("--count_images",
                    action="store_true",
                    help="Count image files for checked envs and compare "
                    "RGB/Semantic counts. With --deep, counts all envs.")
    ap.add_argument(
        "--cam_check",
        action="store_true",
        help="Validate final cam semantic labels for checked envs.")
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
    args = ap.parse_args()

    import compile_a_star_dataset as compiler
    compiler.REF_RED_THRESH = args.cam_red_thresh
    compiler.REF_NO_RED_MAX = args.cam_no_red_max
    if args.cam_no_red_max >= args.cam_red_thresh:
        fail("--cam_no_red_max must be less than --cam_red_thresh", errors)
        sys.exit(2)

    root = Path(args.compiled_path).expanduser().resolve()
    errors: list[str] = []

    if not root.is_dir():
        fail(f"compiled path not found: {root}", errors)
        sys.exit(1)

    master_path = root / "master_labels.json"
    if not master_path.exists():
        fail(f"missing master_labels.json: {master_path}", errors)
        sys.exit(1)

    try:
        master = json.loads(master_path.read_text())
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot parse master_labels.json: {exc}", errors)
        sys.exit(1)

    records = master.get("environments") or {}
    stats = master.get("statistics") or {}
    total = int(stats.get("total_environments", len(records)))
    print(f"[INFO] root={root}")
    print(f"[INFO] total={total} records={len(records)}")

    # No split folders.
    for split in SPLITS:
        if (root / split).exists():
            fail(f"stale split folder exists: {root / split}", errors)

    # Canonical roots exist.
    for typ in TYPES:
        for label in LABELS:
            path = root / typ / label
            if not path.is_dir():
                fail(f"missing canonical dir: {path}", errors)

    # Env IDs continuous.
    ids = sorted(int(k) for k in records.keys())
    expected = list(range(total))
    if ids != expected:
        fail("environment ids not continuous 0..total-1", errors)

    # Split metadata.
    split_counts = Counter(v.get("split") for v in records.values())
    stats_splits = stats.get("splits") or {}
    print(f"[INFO] split_counts={dict(split_counts)}")
    if stats_splits:
        for split in SPLITS:
            if split_counts.get(split, 0) != int(stats_splits.get(split, 0)):
                fail(
                    f"split count mismatch for {split}: records "
                    f"{split_counts.get(split, 0)} vs stats "
                    f"{stats_splits.get(split, 0)}", errors)
    else:
        fail("missing statistics.splits", errors)

    ranges = stats.get("split_ranges") or {}
    if ranges:
        for split in SPLITS:
            rng = ranges.get(split, [])
            if not rng:
                check_range(split, set(), records, errors)
                continue
            start, end = int(rng[0]), int(rng[1])
            check_range(split, set(range(start, end + 1)), records, errors)
    else:
        fail("missing statistics.split_ranges", errors)

    # Label/reason counts.
    label_counts = Counter(v.get("label") for v in records.values())
    reason_counts = Counter(v.get("reason") for v in records.values())
    print(f"[INFO] label_counts={dict(label_counts)}")
    print(f"[INFO] reason_counts={dict(reason_counts)}")
    for reason in REASONS:
        stat_n = int((stats.get("by_reason") or {}).get(reason, 0))
        if reason_counts.get(reason, 0) != stat_n:
            fail(
                f"reason count mismatch {reason}: records "
                f"{reason_counts.get(reason, 0)} vs stats {stat_n}", errors)

    yes = label_counts.get("Yes", 0)
    no = label_counts.get("No", 0)
    if yes != int(stats.get("yes_count", 0)):
        fail(
            f"yes_count mismatch: records {yes} vs stats "
            f"{stats.get('yes_count')}", errors)
    if no != int(stats.get("no_count", 0)):
        fail(
            f"no_count mismatch: records {no} vs stats "
            f"{stats.get('no_count')}", errors)

    # File checks. Fast mode samples evenly across ID range.
    if args.deep:
        check_ids = expected
    else:
        if total <= args.sample:
            check_ids = expected
        else:
            step = max(1, total // args.sample)
            check_ids = sorted(
                set(expected[::step][:args.sample] + [0, total - 1]))
    count_images = args.count_images or args.deep
    cam_check = args.cam_check or args.deep
    print(f"[INFO] checking env files: {len(check_ids)} "
          f"({'deep' if args.deep else 'sample'}, "
          f"{'counting images' if count_images else 'existence only'}, "
          f"{'cam check' if cam_check else 'no cam check'})")

    image_totals = Counter()
    expected_rgb_sem = 0

    for env_id in check_ids:
        rec = records.get(str(env_id))
        if not rec:
            fail(f"missing record env_{env_id}", errors)
            continue
        label = rec.get("label")
        if label not in LABELS:
            fail(f"bad label env_{env_id}: {label}", errors)
            continue
        for typ in TYPES:
            env_dir = root / typ / label / f"env_{env_id}"
            if not has_any_file(env_dir):
                fail(f"missing/empty env dir: {env_dir}", errors)
                continue
            if count_images:
                image_totals[typ] += count_image_files(env_dir)

        if count_images:
            rgb_dir = root / "RGB" / label / f"env_{env_id}"
            sem_dir = root / "Semantic" / label / f"env_{env_id}"
            cam_dir = root / "cam" / label / f"env_{env_id}"
            rgb_n = count_image_files(rgb_dir)
            sem_n = count_image_files(sem_dir)
            cam_n = count_image_files(cam_dir)
            expected_n = int(rec.get("n_frames", 0) or 0)
            expected_rgb_sem += expected_n
            if rgb_n != sem_n:
                fail(
                    f"RGB/Semantic count mismatch env_{env_id}: "
                    f"{rgb_n}!={sem_n}", errors)
            if expected_n and rgb_n != expected_n:
                fail(
                    f"n_frames mismatch env_{env_id}: RGB {rgb_n} vs "
                    f"master {expected_n}", errors)
            if cam_n < 1:
                fail(f"cam image missing env_{env_id}: {cam_dir}", errors)

        if cam_check:
            cam_path = root / "cam" / label / f"env_{env_id}" / CAM_FINAL_NAME
            cam_img = cv2.imread(str(cam_path), cv2.IMREAD_COLOR)
            ok, why, red_count = cam_label_reason(cam_img, label)
            if not ok:
                fail(
                    f"cam label mismatch env_{env_id}: label={label} "
                    f"red_px={red_count} reason={why}", errors)

    if count_images:
        total_pngs = sum(image_totals.values())
        print(f"[INFO] image_counts={dict(image_totals)}")
        print(f"[INFO] total_images_checked={total_pngs:,}")
        if args.deep:
            expected_total = (2 * expected_rgb_sem) + total
            print(f"[INFO] expected_total_images_from_master="
                  f"{expected_total:,}")
            if total_pngs != expected_total:
                fail(
                    f"total image count mismatch: actual {total_pngs} vs "
                    f"expected {expected_total}", errors)

    if errors:
        print(f"\n[FAIL] {len(errors)} sanity errors")
        sys.exit(1)

    print("\n[OK] compiled dataset sanity check passed")


if __name__ == "__main__":
    main()
