#!/usr/bin/env python3
"""Audit cam-POV semantic red-pixel deadzone on compiled A* node dirs.

This scans ``{base_path}/data/data_node*_compiled`` by default and checks each
staged env's final camera semantic image against the compile rule:

* ``Yes`` passes only when strict-red pixel count is greater than threshold.
* ``No`` passes only when strict-red pixel count is <= ``--cam_no_red_max``
  and no circle is detected by the existing semantic sanity check.
* Any image with ``cam_no_red_max+1..cam_red_thresh`` red pixels is a deadzone
  rejection.

Use this before final merge to calibrate ``--cam_red_thresh`` without copying
the whole dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

import compile_a_star_dataset as compiler
from compile_a_star_dataset import CAM_FINAL_NAME, cam_label_reason


def load_compiled_records(base_path: Path, job_id: str | None) -> list[tuple]:
    """Load ``(task_dir, label, reason, env_name, original)`` records.

    Parameters
    ----------
    base_path
        Collector ``BASE_PATH`` containing ``data/``.
    job_id
        Optional SLURM job id prefix filter.

    Returns
    ------
    list[tuple]
        Task directory, label, reason, staged env name, and tracker record
        tuples. A materialized list lets tqdm show an accurate total.
    """
    data_dir = base_path / "data"
    pattern = f"data_node{job_id}_*_compiled" if job_id else "data_node*_compiled"
    task_dirs = sorted(p for p in data_dir.glob(pattern) if p.is_dir())
    if not task_dirs:
        print(f"[ERR] no {pattern} under {data_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    for task_dir in task_dirs:
        tracker_path = task_dir / "successful_envs.json"
        if not tracker_path.exists():
            continue
        try:
            tracker = json.loads(tracker_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] skip bad tracker {tracker_path}: {exc}",
                  file=sys.stderr)
            continue
        for rec in tracker.get("envs", []) or []:
            env_name = rec.get("staged_env")
            label = rec.get("label")
            reason = rec.get("reason")
            if not env_name or label not in ("Yes", "No") or reason not in (
                    "in_view", "occluded", "outside_fov"):
                continue
            records.append((task_dir, label, reason, env_name, rec))
    return records


def red_bucket(red_count: int, no_max: int, threshold: int) -> str:
    """Bucket red-pixel count for compact calibration reporting."""
    if red_count <= no_max:
        return f"<= {no_max}"
    if red_count <= threshold:
        return f"{no_max + 1}..{threshold}"
    return f">{threshold}"


def audit_one(record: tuple, no_max: int, threshold: int) -> tuple:
    """Audit one staged env camera semantic image.

    Parameters
    ----------
    record
        Tuple from :func:`load_compiled_records`.
    no_max
        Maximum red-pixel count allowed for No images.
    threshold
        Red-pixel threshold used only for report bucketing.

    Returns
    -------
    tuple
        ``(label, reason, ok, why, red_count, bucket, cam_path)``.
    """
    task_dir, label, reason, env_name, _ = record
    cam_path = task_dir / "cam" / label / reason / env_name / CAM_FINAL_NAME
    cam_img = cv2.imread(str(cam_path), cv2.IMREAD_COLOR)
    ok, why, red_count = cam_label_reason(cam_img, label)
    return (label, reason, ok, why, red_count,
            red_bucket(red_count, no_max, threshold), str(cam_path))


def main() -> None:
    """Run cam deadzone audit CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base_path", help="Collector BASE_PATH, parent of data/.")
    ap.add_argument("--job_id",
                    default=None,
                    help="Only scan data_node{job_id}_*_compiled dirs.")
    ap.add_argument("--cam_red_thresh",
                    type=int,
                    default=125,
                    help="Yes threshold at 256x256. No must have zero red.")
    ap.add_argument("--cam_no_red_max",
                    type=int,
                    default=0,
                    help="No permits at most this many strict-red pixels at "
                    "256x256 before entering deadzone.")
    ap.add_argument("--limit",
                    type=int,
                    default=0,
                    help="Scan only first N envs for quick smoke test.")
    ap.add_argument("--examples",
                    type=int,
                    default=8,
                    help="Print up to N failing examples.")
    ap.add_argument("--workers",
                    type=int,
                    default=32,
                    help="Parallel image-check workers. Use 128 on a big CPU "
                    "node if filesystem can handle it.")
    args = ap.parse_args()

    compiler.REF_RED_THRESH = args.cam_red_thresh
    compiler.REF_NO_RED_MAX = args.cam_no_red_max
    if args.cam_no_red_max >= args.cam_red_thresh:
        print("[ERR] --cam_no_red_max must be less than --cam_red_thresh",
              file=sys.stderr)
        sys.exit(2)
    base_path = Path(args.base_path).expanduser().resolve()

    total = 0
    pass_count = 0
    fail_count = 0
    by_label_reason = defaultdict(Counter)
    by_fail_reason = Counter()
    by_red_bucket = defaultdict(Counter)
    examples = []

    records = load_compiled_records(base_path, args.job_id)
    if args.limit:
        records = records[:args.limit]

    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(audit_one, rec, args.cam_no_red_max,
                            args.cam_red_thresh) for rec in records
        ]
        iterator = tqdm(as_completed(futures),
                        total=len(futures),
                        desc=f"Auditing cam ({workers} workers)",
                        unit="env")
        for future in iterator:
            label, reason, ok, why, red_count, bucket, cam_path = future.result(
            )
            total += 1
            key = (label, reason)
            by_red_bucket[key][bucket] += 1
            if ok:
                pass_count += 1
                by_label_reason[key]["pass"] += 1
            else:
                fail_count += 1
                by_label_reason[key]["fail"] += 1
                by_fail_reason[why] += 1
                if len(examples) < args.examples:
                    examples.append((label, reason, red_count, why, cam_path))

    print(f"[INFO] base_path={base_path}")
    print(f"[INFO] cam_red_thresh={args.cam_red_thresh} "
          f"No red max={args.cam_no_red_max} "
          f"deadzone={args.cam_no_red_max + 1}..{args.cam_red_thresh}")
    print(f"[INFO] scanned={total} pass={pass_count} fail={fail_count}")

    print(
        "\nlabel  reason          pass     fail  red<=NoMax   red=deadzone   red>thresh"
    )
    print("-" * 78)
    for label in ("Yes", "No"):
        for reason in ("in_view", "occluded", "outside_fov"):
            key = (label, reason)
            stats = by_label_reason[key]
            buckets = by_red_bucket[key]
            if not stats and not buckets:
                continue
            print(
                f"{label:<5}  {reason:<12}  {stats['pass']:>6} "
                f"{stats['fail']:>7} "
                f"{buckets[f'<= {args.cam_no_red_max}']:>11} "
                f"{buckets[f'{args.cam_no_red_max + 1}..{args.cam_red_thresh}']:>14} "
                f"{buckets[f'>{args.cam_red_thresh}']:>12}")

    if by_fail_reason:
        print("\nfail reasons:")
        for reason, count in by_fail_reason.most_common():
            print(f"  {reason:<28} {count}")

    if examples:
        print("\nexamples:")
        for label, reason, red_count, why, path in examples:
            print(f"  label={label:<3} reason={reason:<12} "
                  f"red_px={red_count:<5} why={why:<24} {path}")


if __name__ == "__main__":
    main()
