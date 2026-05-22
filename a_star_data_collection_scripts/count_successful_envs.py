#!/usr/bin/env python3
"""Count successful collected envs across all per-GPU output dirs.

Usage:
  python count_successful_envs.py BASE_PATH

BASE_PATH is the same value used by the data collector (e.g.
`/oscar/scratch/arock3/VPT_DATA_A_STAR/v18_data_collector_run1`).
The script walks `{BASE_PATH}/data/data_node*_gpu*/successful_envs.json`,
sums totals, and computes wall-clock throughput from min/max timestamps.

Reports:
  - total successes
  - per-reason breakdown (in_view / occluded / outside_fov)
  - elapsed time across the whole run (first → last save, max across GPUs)
  - average sec/env (elapsed / total)
  - per-GPU breakdown
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone


def _parse_iso(ts):
    """Parse ISO 8601 timestamp with trailing Z."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _fmt_dt(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    ap = argparse.ArgumentParser(
        description="Count successful envs across per-GPU outputs.")
    ap.add_argument("base_path",
                    help="Collector BASE_PATH (parent of `data/`).")
    ap.add_argument("--target", type=int, default=15000,
                    help="Target final dataset size for balance check.")
    ap.add_argument("--compiled_only", action="store_true",
                    help="Only count post-compile data_node*_compiled outputs; "
                         "ignore raw data_node*_gpu* trackers.")
    args = ap.parse_args()

    # 50/25/25 fractions per project convention
    fractions = {"in_view": 0.50, "occluded": 0.25, "outside_fov": 0.25}
    target_per_cat = {c: int(round(args.target * f))
                      for c, f in fractions.items()}

    # Pick up both raw per-GPU dirs (data_node{T}_gpu{G}) AND post-compile
    # per-task dirs (data_node{T}_compiled), unless --compiled_only is set.
    compiled_pattern = os.path.join(args.base_path, "data",
                                    "data_node*_compiled",
                                    "successful_envs.json")
    raw_pattern = os.path.join(args.base_path, "data",
                               "data_node*_gpu*",
                               "successful_envs.json")
    patterns = [compiled_pattern] if args.compiled_only else [
        raw_pattern,
        compiled_pattern,
    ]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})

    if not paths:
        print(f"[ERR] no successful_envs.json under any of:\n  "
              + "\n  ".join(patterns), file=sys.stderr)
        sys.exit(1)

    grand_total = 0
    grand_by_reason = {"in_view": 0, "occluded": 0, "outside_fov": 0}
    grand_steps = 0
    grand_steps_by_reason = {"in_view": 0, "occluded": 0, "outside_fov": 0}
    all_timestamps = []
    per_gpu_rows = []

    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}", file=sys.stderr)
            continue

        total = data.get("total", 0)
        by_reason = data.get("by_reason", {}) or {}
        envs = data.get("envs", []) or []

        ts = []
        steps_sum = 0
        for e in envs:
            t = e.get("timestamp")
            if t:
                try:
                    ts.append(_parse_iso(t))
                except Exception:
                    pass
            steps = int(e.get("total_steps", 0) or 0)
            steps_sum += steps
            r = e.get("reason")
            if r in grand_steps_by_reason:
                grand_steps_by_reason[r] += steps

        span_sec = 0.0
        sec_per_env = 0.0
        if len(ts) >= 2:
            span_sec = (max(ts) - min(ts)).total_seconds()
            sec_per_env = span_sec / max(1, total - 1)

        label = os.path.basename(os.path.dirname(p))
        per_gpu_rows.append((label, total, dict(by_reason),
                             span_sec, sec_per_env, steps_sum))

        grand_total += total
        grand_steps += steps_sum
        for k in grand_by_reason:
            grand_by_reason[k] += int(by_reason.get(k, 0))
        all_timestamps.extend(ts)

    # ── per-GPU table ──
    print(f"\nFound {len(per_gpu_rows)} GPU output dirs under {args.base_path}\n")
    print(f"{'GPU':<22}{'total':>8}{'in_view':>10}{'occl':>8}{'out_fov':>10}"
          f"{'span':>14}{'sec/env':>12}{'frames':>10}")
    print("-" * 94)
    for label, tot, br, span, spe, steps in per_gpu_rows:
        print(f"{label:<22}{tot:>8}{br.get('in_view',0):>10}"
              f"{br.get('occluded',0):>8}{br.get('outside_fov',0):>10}"
              f"{_fmt_dt(span):>14}{spe:>11.2f}s{steps:>10}")
    print("-" * 94)

    # ── grand totals ──
    # Per env: 2*(steps+1)+1 PNGs
    #   = 2× step frames (RGB+Sem) + 2× final pose (RGB+Sem) + 1 cam-POV final
    total_pngs = 2 * (grand_steps + grand_total) + grand_total
    print(f"\nTOTAL successes: {grand_total}")
    for cat in ("in_view", "occluded", "outside_fov"):
        have = grand_by_reason[cat]
        want = target_per_cat[cat]
        deficit = max(0, want - have)
        excess = max(0, have - want)
        marker = "✓" if have >= want else "✗"
        bar = (f"deficit {deficit}" if deficit else
               (f"excess +{excess}" if excess else "exact"))
        print(f"  {marker} {cat:<12}: {have:>5} / {want:<5} ({bar})")

    # Max compileable at 50/25/25 given current category-limited pool
    max_compileable = min(grand_by_reason[c] / fractions[c]
                          for c in fractions)
    print(f"\nMax balanced dataset (50/25/25) from current pool: "
          f"{int(max_compileable)} envs")
    print(f"  bottleneck: "
          f"{min(fractions, key=lambda c: grand_by_reason[c]/fractions[c])}")
    if grand_total < args.target:
        # rough est: each task collects ~50% in_view × N envs/GPU × 2 GPUs
        in_view_deficit = max(0, target_per_cat['in_view']
                                - grand_by_reason['in_view'])
        if in_view_deficit > 0:
            tasks_needed = (in_view_deficit + 99) // 100
            print(f"\nTo hit {args.target} balanced: need "
                  f"~{tasks_needed} more array tasks "
                  f"(at ~100 in_view per task).")
    print(f"\nTOTAL action frames (sum of total_steps): {grand_steps:,}")
    print(f"  avg frames/env: "
          f"{grand_steps/max(1,grand_total):.1f}")
    for cat in ("in_view", "occluded", "outside_fov"):
        n_envs = grand_by_reason[cat]
        n_steps = grand_steps_by_reason[cat]
        avg = (n_steps / n_envs) if n_envs else 0
        print(f"  {cat:<12}: {n_steps:>10,} frames "
              f"({n_envs} envs, avg {avg:.1f})")

    # Balanced frame count: limited by the smallest (frames_in_cat / fraction)
    balanced_frames = int(min(
        grand_steps_by_reason[c] / fractions[c] for c in fractions))
    bottleneck_frames = min(
        fractions, key=lambda c: grand_steps_by_reason[c] / fractions[c])
    print(f"\nMax balanced frames (50/25/25): {balanced_frames:,}")
    print(f"  bottleneck: {bottleneck_frames}")
    for cat in ("in_view", "occluded", "outside_fov"):
        share = int(round(balanced_frames * fractions[cat]))
        print(f"  {cat:<12}: take {share:,} of "
              f"{grand_steps_by_reason[cat]:,}")

    print(f"\nTOTAL saved PNGs (RGB + Semantic + cam-POV): "
          f"{total_pngs:,}")

    if len(all_timestamps) >= 2:
        global_span = (max(all_timestamps) -
                       min(all_timestamps)).total_seconds()
        # Average across whole job: total successes / wall-clock span,
        # but each save is parallel across GPUs. Use span / total for
        # effective sec/env throughput.
        avg = global_span / max(1, grand_total - 1)
        print(f"\nWall-clock span: {_fmt_dt(global_span)} "
              f"({global_span:.0f}s)")
        print(f"Avg sec/env (global): {avg:.2f}s")
        # Effective throughput across N GPUs:
        n_gpus = len(per_gpu_rows)
        per_gpu_avg = sum(r[4] for r in per_gpu_rows if r[4] > 0)
        per_gpu_avg = (per_gpu_avg / max(1, sum(1 for r in per_gpu_rows
                                                 if r[4] > 0)))
        print(f"Avg sec/env (per-GPU mean): {per_gpu_avg:.2f}s "
              f"(across {n_gpus} GPUs)")
    else:
        print("\n[INFO] not enough timestamps for time stats.")


if __name__ == "__main__":
    main()
