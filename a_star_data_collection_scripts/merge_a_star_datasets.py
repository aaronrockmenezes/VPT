#!/usr/bin/env python3
"""Merge multiple staged A* collector roots into one compiled dataset.

This is the manual dynamics-training merge path. By default it does not
enforce 50/25/25 reason balance or 50/50 label balance. Every passing staged
env from every source root is kept, shuffled once, and assigned to
train/val/test by fixed counts. Use --balance_vpt for probe-specific artifacts
where every split must be exactly 50/25/25 over in_view/occluded/outside_fov.
"""

import argparse
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

compiler = None


def _load_compiler():
    """Import the existing compiler after argparse handles --help."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import compile_a_star_dataset as compiler_mod  # noqa: E402

    return compiler_mod


def _parse_source_name(arg: str, idx: int) -> tuple[str, Path]:
    """Parse NAME=PATH or PATH into a stable source name and path."""
    if "=" in arg:
        name, path = arg.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty source name in {arg!r}")
        return name, Path(path).expanduser()
    path = Path(arg).expanduser()
    return f"source_{idx}", path


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _discover_sources(args):
    """Return merged env items and source-level stats."""
    all_items = []
    source_stats = {}

    for idx, src_arg in enumerate(args.src_root):
        name, root = _parse_source_name(src_arg, idx)
        print(f"\n--- Discovering {name}: {root} ---")
        pool, rejected = compiler.discover_staged_envs(
            root,
            job_id=args.job_id,
            do_cam_check=not args.no_cam_check,
        )

        stats = {
            "src_root": str(root),
            "accepted": 0,
            "by_reason": {k: len(v) for k, v in pool.items()},
            "rejected": rejected,
        }
        source_stats[name] = stats

        for reason, items in pool.items():
            for item in items:
                item["source_dataset"] = name
                item["source_root"] = str(root)
                item["reason"] = reason
                all_items.append(item)
                stats["accepted"] += 1

        print(f"[{name}] accepted {stats['accepted']} envs")
        for reason, n in stats["by_reason"].items():
            print(f"  {reason:>12}: {n:>6}")
        for reason, n in sorted(rejected.items(), key=lambda kv: -kv[1]):
            if n:
                print(f"  rejected {reason}: {n}")

    return all_items, source_stats


def _split_for_id(env_id: int, n_train: int, n_val: int) -> str:
    if env_id < n_train:
        return "train"
    if env_id < n_train + n_val:
        return "val"
    return "test"


def _order_unbalanced(items, args):
    """Shuffle all items and optionally cap the merged pool."""
    ordered = list(items)
    random.shuffle(ordered)
    if args.limit:
        if args.limit < args.val_count + args.test_count:
            raise SystemExit(
                "[FATAL] --limit must be at least val_count + test_count")
        ordered = ordered[:args.limit]
    return ordered


def _take_label_balanced(pool_by_label, n_per_label: int, split_name: str):
    """Take n_per_label Yes and No examples for one split."""
    out = []
    for label in ("Yes", "No"):
        available = len(pool_by_label[label])
        if available < n_per_label:
            raise SystemExit(
                f"[FATAL] not enough {label} examples for {split_name}: "
                f"need {n_per_label}, have {available}")
        out.extend(pool_by_label[label][:n_per_label])
        del pool_by_label[label][:n_per_label]
    random.shuffle(out)
    return out


def _order_label_balanced(items, args):
    """Return ordered train+val+test items with exact 50/50 labels."""
    if args.val_count % 2 or args.test_count % 2:
        raise SystemExit(
            "[FATAL] --balance_labels requires even val_count and test_count")
    if args.limit and args.limit % 2:
        raise SystemExit("[FATAL] --balance_labels requires an even --limit")

    pool_by_label = {"Yes": [], "No": []}
    for item in items:
        label = item.get("label")
        if label not in pool_by_label:
            continue
        pool_by_label[label].append(item)
    for bucket in pool_by_label.values():
        random.shuffle(bucket)

    balanced_total = min(len(pool_by_label["Yes"]), len(pool_by_label["No"])) * 2
    total = args.limit or balanced_total
    if total > balanced_total:
        raise SystemExit(
            f"[FATAL] requested balanced total {total}, but pool supports "
            f"only {balanced_total} ({len(pool_by_label['Yes'])} Yes, "
            f"{len(pool_by_label['No'])} No)")
    if total < args.val_count + args.test_count:
        raise SystemExit(
            "[FATAL] balanced total is smaller than val_count + test_count")

    n_train = total - args.val_count - args.test_count
    if n_train % 2:
        raise SystemExit(
            "[FATAL] --balance_labels requires train count to be even; adjust "
            "--limit, --val_count, or --test_count")

    train = _take_label_balanced(pool_by_label, n_train // 2, "train")
    val = _take_label_balanced(pool_by_label, args.val_count // 2, "val")
    test = _take_label_balanced(pool_by_label, args.test_count // 2, "test")
    return train + val + test


def _take_vpt_balanced(pool_by_reason, n_total: int, split_name: str):
    """Take exact 50/25/25 in_view/occluded/outside_fov for one split."""
    if n_total % 4:
        raise SystemExit(
            f"[FATAL] {split_name} count must be divisible by 4 for "
            "50/25/25 VPT balance")
    targets = {
        "in_view": n_total // 2,
        "occluded": n_total // 4,
        "outside_fov": n_total // 4,
    }
    out = []
    for reason, need in targets.items():
        available = len(pool_by_reason[reason])
        if available < need:
            raise SystemExit(
                f"[FATAL] not enough {reason} examples for {split_name}: "
                f"need {need}, have {available}")
        out.extend(pool_by_reason[reason][:need])
        del pool_by_reason[reason][:need]
    random.shuffle(out)
    return out


def _order_vpt_balanced(items, args):
    """Return ordered train+val+test items with exact 50/25/25 reason balance."""
    for name, count in (("val", args.val_count), ("test", args.test_count)):
        if count % 4:
            raise SystemExit(
                f"[FATAL] --balance_vpt requires {name}_count divisible by 4")
    if args.limit and args.limit % 4:
        raise SystemExit("[FATAL] --balance_vpt requires --limit divisible by 4")

    pool_by_reason = {c: [] for c in ("in_view", "occluded", "outside_fov")}
    for item in items:
        reason = item.get("reason")
        if reason in pool_by_reason:
            pool_by_reason[reason].append(item)
    for bucket in pool_by_reason.values():
        random.shuffle(bucket)

    balanced_total = min(
        len(pool_by_reason["in_view"]) * 2,
        len(pool_by_reason["occluded"]) * 4,
        len(pool_by_reason["outside_fov"]) * 4,
    )
    balanced_total -= balanced_total % 4
    total = args.limit or balanced_total
    if total > balanced_total:
        raise SystemExit(
            f"[FATAL] requested VPT-balanced total {total}, but pool supports "
            f"only {balanced_total} "
            f"(in_view={len(pool_by_reason['in_view'])}, "
            f"occluded={len(pool_by_reason['occluded'])}, "
            f"outside_fov={len(pool_by_reason['outside_fov'])})")
    if total < args.val_count + args.test_count:
        raise SystemExit(
            "[FATAL] VPT-balanced total is smaller than "
            "val_count + test_count")

    n_train = total - args.val_count - args.test_count
    if n_train % 4:
        raise SystemExit(
            "[FATAL] --balance_vpt requires train count divisible by 4; "
            "adjust --limit, --val_count, or --test_count")

    train = _take_vpt_balanced(pool_by_reason, n_train, "train")
    val = _take_vpt_balanced(pool_by_reason, args.val_count, "val")
    test = _take_vpt_balanced(pool_by_reason, args.test_count, "test")
    return train + val + test


def _write_dataset(args, ordered, source_stats):
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists():
        if args.overwrite_out:
            print(f"[INFO] removing existing out_dir: {out_dir}")
            _remove_path(out_dir)
        else:
            raise SystemExit(
                f"[FATAL] {out_dir} exists. Use --overwrite_out to replace it.")

    compiler.make_dirs(out_dir)

    n_total = len(ordered)
    if args.val_count + args.test_count > n_total:
        raise SystemExit(
            f"[FATAL] val_count + test_count exceeds total envs: "
            f"{args.val_count} + {args.test_count} > {n_total}")
    n_val = args.val_count
    n_test = args.test_count
    n_train = n_total - n_val - n_test

    split_ranges = {
        "train": [0, n_train - 1] if n_train else [],
        "val": [n_train, n_train + n_val - 1] if n_val else [],
        "test": [n_train + n_val, n_total - 1] if n_test else [],
    }

    print(f"\n--- Writing {n_total} envs to {out_dir} ---")
    print(f"split: train={n_train} val={n_val} test={n_test}")
    copy_mode = "copy" if args.copy else "symlink"
    print(f"storage: {copy_mode}")

    def copy_one(pair):
        new_id, item = pair
        compiler.copy_env(
            item,
            out_dir,
            new_id,
            link_only=not args.copy,
            mode="staged",
        )
        return new_id, item

    copied = []
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(copy_one, pair) for pair in enumerate(ordered)]
        for fut in as_completed(futures):
            copied.append(fut.result())
            if len(copied) % 1000 == 0:
                print(f"  copied {len(copied):>6}/{n_total}", flush=True)

    copied.sort(key=lambda x: x[0])
    if len(copied) != n_total:
        raise SystemExit(f"[FATAL] copied {len(copied)} / {n_total} envs")

    master = {
        "environments": {},
        "statistics": {
            "total_environments": n_total,
            "yes_count": 0,
            "no_count": 0,
            "by_reason": {c: 0 for c in compiler.DEFAULT_FRACTIONS},
            "by_source": {
                name: {
                    "count": 0,
                    "by_reason": {c: 0 for c in compiler.DEFAULT_FRACTIONS},
                }
                for name in source_stats
            },
            "splits": {
                "train": n_train,
                "val": n_val,
                "test": n_test,
            },
            "split_ranges": split_ranges,
            "source_stats": source_stats,
            "seed": args.seed,
            "merge_policy": (
                "manual_sources_vpt_50_25_25_balanced"
                if args.balance_vpt else
                "manual_sources_label_balanced"
                if args.balance_labels else
                "manual_all_sources_no_balance"
            ),
            "storage": copy_mode,
            "cam_check": {
                "enabled": not args.no_cam_check,
                "yes_red_threshold_px_at_256": compiler.REF_RED_THRESH,
                "no_red_max_px_at_256": compiler.REF_NO_RED_MAX,
            },
        },
    }

    for new_id, item in copied:
        split = _split_for_id(new_id, n_train, n_val)
        source = item["source_dataset"]
        reason = item["reason"]
        label = item["label"]

        master["environments"][str(new_id)] = {
            "label": label,
            "reason": reason,
            "n_frames": item["n_frames"],
            "split": split,
            "source_dataset": source,
            "src_root": item["source_root"],
            "src": item["src_task_dir"],
            "src_id": item["env_name"],
            "original": item.get("original", {}),
        }
        master["statistics"]["by_reason"][reason] += 1
        master["statistics"]["by_source"][source]["count"] += 1
        master["statistics"]["by_source"][source]["by_reason"][reason] += 1
        if label == "Yes":
            master["statistics"]["yes_count"] += 1
        else:
            master["statistics"]["no_count"] += 1

    (out_dir / "master_labels.json").write_text(json.dumps(master, indent=2))

    print("\n--- Final stats ---")
    print(f"total: {n_total}")
    print(f"yes:   {master['statistics']['yes_count']}")
    print(f"no:    {master['statistics']['no_count']}")
    for reason, n in master["statistics"]["by_reason"].items():
        print(f"{reason:>12}: {n:>6}")
    for source, stats in master["statistics"]["by_source"].items():
        print(f"{source:>18}: {stats['count']:>6}")
    print(f"\n[DONE] wrote merged dataset to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src_root",
        action="append",
        required=True,
        help="Staged collector root. Use NAME=PATH to set provenance name. "
        "Pass once per dataset.",
    )
    ap.add_argument("--out_dir", required=True, help="Merged output dataset.")
    ap.add_argument("--val_count", type=int, default=1000)
    ap.add_argument("--test_count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap after shuffle. 0 keeps every discovered env.",
    )
    ap.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Optional job filter passed to staged discovery for every root.",
    )
    ap.add_argument("--no_cam_check", action="store_true")
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy env dirs instead of symlinking them.",
    )
    ap.add_argument(
        "--balance_labels",
        action="store_true",
        help="Downsample to exact 50/50 Yes/No in train, val, and test. "
        "Use for VPT probe artifacts, not max-data dynamics pretraining.",
    )
    ap.add_argument(
        "--balance_vpt",
        action="store_true",
        help="Downsample to exact 50/25/25 in_view/occluded/outside_fov in "
        "train, val, and test. This is the preferred VPT probe balance.",
    )
    ap.add_argument("--overwrite_out", action="store_true")
    args = ap.parse_args()

    global compiler
    compiler = _load_compiler()

    random.seed(args.seed)
    all_items, source_stats = _discover_sources(args)
    if not all_items:
        raise SystemExit("[FATAL] no envs discovered")

    if args.balance_vpt and args.balance_labels:
        raise SystemExit(
            "[FATAL] use only one of --balance_vpt or --balance_labels")

    if args.balance_vpt:
        all_items = _order_vpt_balanced(all_items, args)
    elif args.balance_labels:
        all_items = _order_label_balanced(all_items, args)
    else:
        all_items = _order_unbalanced(all_items, args)

    _write_dataset(args, all_items, source_stats)


if __name__ == "__main__":
    main()
