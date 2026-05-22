#!/usr/bin/env python3
"""Create ImageFolder train/test views from a compiled A* dataset.

The linear-probe runner expects::

    dataset/
    ├── train/{Yes,No}/...
    └── test/{Yes,No}/...

Compiled A* datasets are canonical episode trees instead::

    dataset/
    ├── master_labels.json
    ├── RGB/{Yes,No}/env_*/
    ├── Semantic/{Yes,No}/env_*/
    └── cam/{Yes,No}/env_*/

This script creates train/test ImageFolder views in place by symlinking the
first RGB frame from each selected episode. It also writes
``visibility_labels.json`` for downstream reason-wise analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
REASONS = ("in_view", "occluded", "outside_fov")


def load_master(root: Path) -> dict:
    """Load ``master_labels.json`` from ``root``."""
    path = root / "master_labels.json"
    if not path.exists():
        raise SystemExit(f"[FATAL] missing {path}")
    return json.loads(path.read_text())


def records_from_master(master: dict) -> list[dict]:
    """Return sorted environment records with integer ``env_id``."""
    envs = master.get("environments") or {}
    records = []
    for env_id, rec in envs.items():
        item = dict(rec)
        item["env_id"] = int(env_id)
        item.setdefault("label", "Yes" if item.get("reason") == "in_view" else "No")
        records.append(item)
    return sorted(records, key=lambda item: item["env_id"])


def first_rgb_file(root: Path, rec: dict) -> Path:
    """Return first RGB image for one compiled env record."""
    env_id = rec["env_id"]
    label = rec["label"]
    env_dir = root / "RGB" / label / f"env_{env_id}"
    if not env_dir.is_dir():
        raise FileNotFoundError(f"missing RGB env dir: {env_dir}")
    preferred = env_dir / "step_00000_start.png"
    if preferred.exists():
        return preferred
    imgs = sorted(p for p in env_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        raise FileNotFoundError(f"no images in {env_dir}")
    return imgs[0]


def safe_clear(path: Path) -> None:
    """Remove an existing split directory."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def materialize(src: Path, dst: Path, copy: bool) -> None:
    """Copy or symlink one image file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def split_records(records: list[dict], train_frac: float, seed: int,
                  stratify: bool) -> tuple[list[dict], list[dict]]:
    """Split records reproducibly, optionally preserving label/reason mix."""
    rng = random.Random(seed)
    if not stratify:
        shuffled = list(records)
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * train_frac))
        return shuffled[:n_train], shuffled[n_train:]

    train, test = [], []
    buckets: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        key = (rec.get("label", ""), rec.get("reason", ""))
        buckets.setdefault(key, []).append(rec)
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_train = int(round(len(bucket) * train_frac))
        train.extend(bucket[:n_train])
        test.extend(bucket[n_train:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_visibility_labels(root: Path, train: list[dict], test: list[dict],
                            seed: int, train_frac: float) -> None:
    """Write VPT analysis-compatible visibility labels."""
    envs = {}
    stats = {
        "total_environments": len(train) + len(test),
        "yes_count": 0,
        "no_count": 0,
        "by_reason": Counter(),
        "splits": {
            "train": len(train),
            "test": len(test),
        },
        "seed": seed,
        "train_frac": train_frac,
        "source": "make_vpt_probe_split.py",
    }
    for split, items in (("train", train), ("test", test)):
        for rec in items:
            env_id = str(rec["env_id"])
            label = rec["label"]
            reason = rec.get("reason", "unknown")
            envs[env_id] = {
                "label": label,
                "reason": reason,
                "split": split,
            }
            if label == "Yes":
                stats["yes_count"] += 1
            elif label == "No":
                stats["no_count"] += 1
            stats["by_reason"][reason] += 1
    stats["by_reason"] = dict(stats["by_reason"])
    (root / "visibility_labels.json").write_text(
        json.dumps({"environments": envs, "statistics": stats}, indent=2))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Compiled A* dataset root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink.")
    parser.add_argument("--overwrite", action="store_true", help="Replace train/test if present.")
    parser.add_argument("--no_stratify", action="store_true", help="Pure shuffle split instead of label/reason-stratified split.")
    args = parser.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"[FATAL] dataset root missing: {root}")
    if not (0.0 < args.train_frac < 1.0):
        raise SystemExit("[FATAL] --train_frac must be between 0 and 1")

    train_dir = root / "train"
    test_dir = root / "test"
    if (train_dir.exists() or test_dir.exists()) and not args.overwrite:
        raise SystemExit("[FATAL] train/test already exist; pass --overwrite")
    if args.overwrite:
        safe_clear(train_dir)
        safe_clear(test_dir)

    records = records_from_master(load_master(root))
    if not records:
        raise SystemExit("[FATAL] no records in master_labels.json")
    train, test = split_records(records, args.train_frac, args.seed,
                                not args.no_stratify)

    for split_name, items in (("train", train), ("test", test)):
        for rec in items:
            src = first_rgb_file(root, rec)
            label = rec["label"]
            env_id = rec["env_id"]
            dst = root / split_name / label / f"env_{env_id}_{src.name}"
            materialize(src, dst, args.copy)

    write_visibility_labels(root, train, test, args.seed, args.train_frac)

    print(f"[OK] wrote train/test views in {root}")
    print(f"[INFO] train={len(train)} test={len(test)} seed={args.seed} copy={args.copy}")
    print(f"[INFO] label counts train={dict(Counter(r['label'] for r in train))} test={dict(Counter(r['label'] for r in test))}")
    print(f"[INFO] reason counts train={dict(Counter(r.get('reason') for r in train))} test={dict(Counter(r.get('reason') for r in test))}")


if __name__ == "__main__":
    main()
