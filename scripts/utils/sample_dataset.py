"""
sample_dataset.py
-----------------
Draws a random human-study sample from a built VPT dataset.

Modes
-----
depth
    Source : train_depth/ and test_depth/
    Folders: {split}/Yes/env_{idx}/ (depth=1) and {split}/No/env_{idx}/ (depth=0)
    Labels : reason from visibility_labels.json
    Balance: 2:1:1 reason (in_view:outside_fov:occluded)
             within each reason bucket: strict 50/50 depth 0/1
             → also gives 50/50 Yes/No globally
    n must be divisible by 8

vpt1
    Source : train/ and test/
    Folders: {split}/Yes/env_{idx}/ and {split}/No/env_{idx}/
    Labels : reason from visibility_labels.json
    Balance: 2:1:1 reason (in_view:outside_fov:occluded)
             → 50/50 Yes/No falls out of reason proportions
    n must be divisible by 4

vpt2
    Source : train/ and test/
    Folders: {split}/left/env_{idx}/ and {split}/right/env_{idx}/
    Balance: 50/50 left/right
    n must be divisible by 2

Output
------
    output_dir/train/img_0000.png
    output_dir/test/img_0000.png
    output_dir/train_labels.json
    output_dir/test_labels.json
    output_dir/train_labels.csv
    output_dir/test_labels.csv
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
REASONS          = ("in_view", "outside_fov", "occluded")


# ── METADATA ──────────────────────────────────────────────────────────────────

def load_env_meta(dataset_dir: Path) -> dict[int, dict]:
    path = dataset_dir / "visibility_labels.json"
    if not path.exists():
        raise FileNotFoundError(f"visibility_labels.json not found in {dataset_dir}")
    with open(path) as f:
        data = json.load(f)
    return {
        int(k): {"label": v["label"], "reason": v["reason"]}
        for k, v in data["environments"].items()
    }


# ── COLLECTORS ────────────────────────────────────────────────────────────────

def collect_depth(split_dir: Path, env_meta: dict[int, dict]) -> dict[str, dict[int, list]]:
    buckets: dict[str, dict[int, list]] = {r: {0: [], 1: []} for r in REASONS}
    unknown = 0

    for vis_label_str, depth_val in (("Yes", 1), ("No", 0)):
        label_dir = split_dir / vis_label_str
        if not label_dir.exists():
            print(f"  Warning: {label_dir} does not exist, skipping.")
            continue

        for env_dir in sorted(label_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            try:
                env_idx = int(env_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            meta = env_meta.get(env_idx)
            if meta is None or meta["reason"] not in buckets:
                unknown += 1
                continue

            reason = meta["reason"]
            for img in sorted(env_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTENSIONS:
                    buckets[reason][depth_val].append({
                        "path":      img,
                        "label":     depth_val,
                        "vis_label": vis_label_str,
                        "reason":    reason,
                    })

    if unknown:
        print(f"  Warning: {unknown} envs skipped (missing/unknown reason).")
    return buckets


def collect_vpt1(split_dir: Path, env_meta: dict[int, dict]) -> dict[str, list]:
    buckets: dict[str, list] = {r: [] for r in REASONS}
    unknown = 0

    for vis_label_str in ("Yes", "No"):
        label_dir = split_dir / vis_label_str
        if not label_dir.exists():
            print(f"  Warning: {label_dir} does not exist, skipping.")
            continue

        for env_dir in sorted(label_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            try:
                env_idx = int(env_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            meta = env_meta.get(env_idx)
            if meta is None or meta["reason"] not in buckets:
                unknown += 1
                continue

            reason = meta["reason"]
            for img in sorted(env_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTENSIONS:
                    buckets[reason].append({
                        "path":      img,
                        "vis_label": vis_label_str,
                        "reason":    reason,
                    })

    if unknown:
        print(f"  Warning: {unknown} envs skipped (missing/unknown reason).")
    return buckets


def collect_vpt2(split_dir: Path) -> dict[str, list]:
    buckets: dict[str, list] = {"left": [], "right": []}

    for label_str in ("left", "right"):
        label_dir = split_dir / label_str
        if not label_dir.exists():
            print(f"  Warning: {label_dir} does not exist, skipping.")
            continue

        for env_dir in sorted(label_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            for img in sorted(env_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTENSIONS:
                    buckets[label_str].append({"path": img, "label": label_str})

    return buckets


# ── SAMPLERS ──────────────────────────────────────────────────────────────────

def sample_depth(buckets: dict[str, dict[int, list]], n: int) -> list[dict]:
    if n % 8 != 0:
        raise ValueError(f"depth mode: n must be divisible by 8, got {n}")

    quarter = n // 4
    reason_counts = {
        "in_view":     2 * quarter,
        "outside_fov":     quarter,
        "occluded":        quarter,
    }

    selected = []
    for reason, count in reason_counts.items():
        half = count // 2
        pool_1 = buckets[reason][1][:]
        pool_0 = buckets[reason][0][:]

        if len(pool_1) < half:
            raise ValueError(f"Not enough depth=1 in '{reason}': need {half}, have {len(pool_1)}")
        if len(pool_0) < half:
            raise ValueError(f"Not enough depth=0 in '{reason}': need {half}, have {len(pool_0)}")

        random.shuffle(pool_1)
        random.shuffle(pool_0)
        selected.extend(pool_1[:half])
        selected.extend(pool_0[:half])

    random.shuffle(selected)
    return selected


def sample_vpt1(buckets: dict[str, list], n: int) -> list[dict]:
    if n % 4 != 0:
        raise ValueError(f"vpt1 mode: n must be divisible by 4, got {n}")

    quarter = n // 4
    reason_counts = {
        "in_view":     2 * quarter,
        "outside_fov":     quarter,
        "occluded":        quarter,
    }

    selected = []
    for reason, count in reason_counts.items():
        pool = buckets[reason][:]
        if len(pool) < count:
            raise ValueError(f"Not enough '{reason}' images: need {count}, have {len(pool)}")
        random.shuffle(pool)
        selected.extend(pool[:count])

    random.shuffle(selected)
    return selected


def sample_vpt2(buckets: dict[str, list], n: int) -> list[dict]:
    if n % 2 != 0:
        raise ValueError(f"vpt2 mode: n must be divisible by 2, got {n}")

    half = n // 2
    selected = []
    for label_str in ("left", "right"):
        pool = buckets[label_str][:]
        if len(pool) < half:
            raise ValueError(f"Not enough '{label_str}' images: need {half}, have {len(pool)}")
        random.shuffle(pool)
        selected.extend(pool[:half])

    random.shuffle(selected)
    return selected


# ── WRITE ─────────────────────────────────────────────────────────────────────

def copy_or_resize(src: Path, dst: Path, size: int | None) -> None:
    """Copy image to dst, optionally resizing to size x size."""
    if size is None:
        shutil.copy2(src, dst)
    else:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {src}")
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), resized)


def write_split(samples: list[dict], out_dir: Path, size: int | None) -> list[dict]:
    """
    Copies (and optionally resizes) images into out_dir as img_XXXX.png (flat).
    Returns list of {filename, original_path, ...metadata} records.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for i, entry in enumerate(tqdm(samples, desc=f"  → {out_dir.name}")):
        dst_name = f"img_{i:04d}.png"
        copy_or_resize(entry["path"], out_dir / dst_name, size)
        rec = {"filename": dst_name, "original_path": str(entry["path"])}
        for k, v in entry.items():
            if k not in ("path", "original_path"):
                rec[k] = v
        records.append(rec)

    return records


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def write_labels(records: list[dict], out_dir: Path, split: str, extra_cols: list[str]) -> None:
    """Writes {split}_labels.json and {split}_labels.csv for one split."""

    # JSON
    labels_dict = {
        r["filename"]: {k: v for k, v in r.items() if k != "filename"}
        for r in records
    }
    json_path = out_dir / f"{split}_labels.json"
    json_path.write_text(json.dumps(labels_dict, indent=4))
    print(f"  Wrote {json_path.name}")

    # CSV
    csv_path = out_dir / f"{split}_labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + extra_cols)
        for rec in sorted(records, key=lambda r: r["filename"]):
            writer.writerow([rec["filename"]] + [rec[c] for c in extra_cols])
    print(f"  Wrote {csv_path.name}")


def print_summary(name: str, records: list[dict], mode: str) -> None:
    print(f"\n  [{name}] {len(records)} images")

    if mode == "depth":
        ones  = sum(1 for r in records if r["label"] == 1)
        zeros = sum(1 for r in records if r["label"] == 0)
        yes   = sum(1 for r in records if r["vis_label"] == "Yes")
        no    = sum(1 for r in records if r["vis_label"] == "No")
        print(f"    depth 1/0   : {ones}/{zeros}")
        print(f"    Yes/No      : {yes}/{no}")
        for reason in REASONS:
            count = sum(1 for r in records if r["reason"] == reason)
            print(f"    {reason:15s}: {count}")

    elif mode == "vpt1":
        yes = sum(1 for r in records if r["vis_label"] == "Yes")
        no  = sum(1 for r in records if r["vis_label"] == "No")
        print(f"    Yes/No      : {yes}/{no}")
        for reason in REASONS:
            count = sum(1 for r in records if r["reason"] == reason)
            print(f"    {reason:15s}: {count}")

    elif mode == "vpt2":
        left  = sum(1 for r in records if r["label"] == "left")
        right = sum(1 for r in records if r["label"] == "right")
        print(f"    left/right  : {left}/{right}")


# ── LLM SUBSAMPLER ────────────────────────────────────────────────────────────

def subsample_records(records: list[dict], n: int, mode: str) -> list[dict]:
    """
    Stratified subsample of n records from an already-balanced human-study record list.
    Preserves the same balance constraints as the original samplers.
    """
    if mode == "depth":
        if n % 8 != 0:
            raise ValueError(f"depth mode: llm n must be divisible by 8, got {n}")
        quarter = n // 4
        reason_counts = {"in_view": 2 * quarter, "outside_fov": quarter, "occluded": quarter}
        selected = []
        for reason, count in reason_counts.items():
            half = count // 2
            pool_1 = [r for r in records if r["reason"] == reason and r["label"] == 1]
            pool_0 = [r for r in records if r["reason"] == reason and r["label"] == 0]
            if len(pool_1) < half:
                raise ValueError(f"Not enough depth=1 '{reason}' in human set: need {half}, have {len(pool_1)}")
            if len(pool_0) < half:
                raise ValueError(f"Not enough depth=0 '{reason}' in human set: need {half}, have {len(pool_0)}")
            random.shuffle(pool_1); random.shuffle(pool_0)
            selected.extend(pool_1[:half]); selected.extend(pool_0[:half])

    elif mode == "vpt1":
        if n % 4 != 0:
            raise ValueError(f"vpt1 mode: llm n must be divisible by 4, got {n}")
        quarter = n // 4
        reason_counts = {"in_view": 2 * quarter, "outside_fov": quarter, "occluded": quarter}
        selected = []
        for reason, count in reason_counts.items():
            pool = [r for r in records if r["reason"] == reason]
            if len(pool) < count:
                raise ValueError(f"Not enough '{reason}' in human set: need {count}, have {len(pool)}")
            random.shuffle(pool)
            selected.extend(pool[:count])

    elif mode == "vpt2":
        if n % 2 != 0:
            raise ValueError(f"vpt2 mode: llm n must be divisible by 2, got {n}")
        half = n // 2
        selected = []
        for label_str in ("left", "right"):
            pool = [r for r in records if r["label"] == label_str]
            if len(pool) < half:
                raise ValueError(f"Not enough '{label_str}' in human set: need {half}, have {len(pool)}")
            random.shuffle(pool)
            selected.extend(pool[:half])

    random.shuffle(selected)
    return selected


def write_llm_split(records: list[dict], llm_dir: Path) -> list[dict]:
    """
    Symlink-free copy: reuses the already-written human study images (no re-resize).
    Records carry original_path; the human study image is already at the right size.
    Returns new records with updated filenames pointing into llm_dir.
    """
    llm_dir.mkdir(parents=True, exist_ok=True)
    new_records = []
    for i, rec in enumerate(tqdm(records, desc=f"  → {llm_dir.name}")):
        dst_name = f"img_{i:04d}.png"
        # source is the human-study image that was already written
        src = Path(rec["filename"]).parent.parent / rec["filename"] \
            if not Path(rec["filename"]).is_absolute() else Path(rec["filename"])
        # rec["filename"] is just "img_XXXX.png"; we need the full path from the parent split dir
        # It's easier to pass the parent dir in; handled by caller via rec["_src_dir"]
        shutil.copy2(rec["_src_dir"] / rec["filename"], llm_dir / dst_name)
        new_rec = {k: v for k, v in rec.items() if k != "_src_dir"}
        new_rec["filename"] = dst_name
        new_records.append(new_rec)
    return new_records


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample a human-study set from a built VPT dataset."
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Root of the built dataset (contains visibility_labels.json)")
    parser.add_argument("--mode", required=True, choices=("depth", "vpt1", "vpt2"),
                        help="Dataset mode")
    parser.add_argument("--num_train", type=int, default=24,
                        help="Number of train images for the human study (default: 24)")
    parser.add_argument("--num_test",  type=int, default=96,
                        help="Number of test images for the human study (default: 96)")
    parser.add_argument("--output_dir", required=True,
                        help="Root output directory")
    parser.add_argument("--size", type=int, default=256,
                        help="Resize images to size x size (default: 256). "
                             "Pass 0 to skip resizing and keep original resolution.")
    parser.add_argument("--llm_num_train", type=int, default=16,
                        help="LLM train images subsampled from the human study set (default: 16)")
    parser.add_argument("--llm_num_test",  type=int, default=40,
                        help="LLM test images subsampled from the human study set (default: 40)")
    parser.add_argument("--skip_llm", action="store_true",
                        help="Skip LLM_data generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    mode        = args.mode
    size        = args.size if args.size > 0 else None

    if size:
        print(f"Images will be resized to {size}x{size}.")
    else:
        print("Images will be copied at original resolution.")

    if output_dir.exists():
        print(f"Warning: output_dir {output_dir} already exists.")

    # ── Resolve split dirs ────────────────────────────────────────────────────
    if mode == "depth":
        train_subdir, test_subdir = "train_depth", "test_depth"
    else:
        train_subdir, test_subdir = "train", "test"

    train_dir = dataset_dir / train_subdir
    test_dir  = dataset_dir / test_subdir

    # ── Load metadata ─────────────────────────────────────────────────────────
    env_meta = None
    if mode in ("depth", "vpt1"):
        print(f"\nLoading env metadata from: {dataset_dir}")
        env_meta = load_env_meta(dataset_dir)
        print(f"  Loaded {len(env_meta)} environments.")

    # ── Collect ───────────────────────────────────────────────────────────────
    print(f"\nCollecting train images from: {train_dir}")
    print(f"Collecting test images from:  {test_dir}")

    if mode == "depth":
        train_buckets = collect_depth(train_dir, env_meta)
        test_buckets  = collect_depth(test_dir,  env_meta)
        print("\nTrain pool:")
        for r in REASONS:
            print(f"  {r:15s}: depth=1 → {len(train_buckets[r][1])}, depth=0 → {len(train_buckets[r][0])}")
        print("Test pool:")
        for r in REASONS:
            print(f"  {r:15s}: depth=1 → {len(test_buckets[r][1])}, depth=0 → {len(test_buckets[r][0])}")

    elif mode == "vpt1":
        train_buckets = collect_vpt1(train_dir, env_meta)
        test_buckets  = collect_vpt1(test_dir,  env_meta)
        print("\nTrain pool:")
        for r in REASONS:
            print(f"  {r:15s}: {len(train_buckets[r])}")
        print("Test pool:")
        for r in REASONS:
            print(f"  {r:15s}: {len(test_buckets[r])}")

    elif mode == "vpt2":
        train_buckets = collect_vpt2(train_dir)
        test_buckets  = collect_vpt2(test_dir)
        print(f"\nTrain pool: left={len(train_buckets['left'])}, right={len(train_buckets['right'])}")
        print(f"Test pool:  left={len(test_buckets['left'])}, right={len(test_buckets['right'])}")

    # ── Sample ────────────────────────────────────────────────────────────────
    print(f"\nSampling {args.num_train} train and {args.num_test} test images...")

    if mode == "depth":
        train_samples = sample_depth(train_buckets, args.num_train)
        test_samples  = sample_depth(test_buckets,  args.num_test)
    elif mode == "vpt1":
        train_samples = sample_vpt1(train_buckets, args.num_train)
        test_samples  = sample_vpt1(test_buckets,  args.num_test)
    elif mode == "vpt2":
        train_samples = sample_vpt2(train_buckets, args.num_train)
        test_samples  = sample_vpt2(test_buckets,  args.num_test)

    # ── Copy / resize ─────────────────────────────────────────────────────────
    print(f"\nWriting to {output_dir}...")
    train_records = write_split(train_samples, output_dir / "train", size)
    test_records  = write_split(test_samples,  output_dir / "test",  size)

    # ── Labels — separate files per split ─────────────────────────────────────
    if mode == "depth":
        extra_cols = ["label", "vis_label", "reason"]
    elif mode == "vpt1":
        extra_cols = ["vis_label", "reason"]
    elif mode == "vpt2":
        extra_cols = ["label"]

    print(f"\nWriting label files...")
    write_labels(train_records, output_dir, "train", extra_cols)
    write_labels(test_records,  output_dir, "test",  extra_cols)

    # ── LLM_data — subsample from human study records ─────────────────────────
    if not args.skip_llm:
        print(f"\nGenerating LLM_data ({args.llm_num_train} train / {args.llm_num_test} test) ...")
        llm_dir = output_dir / "LLM_data"

        # Tag each record with its source directory so write_llm_split can find the file
        for rec in train_records:
            rec["_src_dir"] = output_dir / "train"
        for rec in test_records:
            rec["_src_dir"] = output_dir / "test"

        llm_train_sub = subsample_records(train_records, args.llm_num_train, mode)
        llm_test_sub  = subsample_records(test_records,  args.llm_num_test,  mode)

        llm_train_records = write_llm_split(llm_train_sub, llm_dir / "train")
        llm_test_records  = write_llm_split(llm_test_sub,  llm_dir / "test")

        # Strip the internal tag from human records now that we're done
        for rec in train_records + test_records:
            rec.pop("_src_dir", None)

        print(f"\nWriting LLM label files...")
        write_labels(llm_train_records, llm_dir, "train", extra_cols)
        write_labels(llm_test_records,  llm_dir, "test",  extra_cols)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nSummary (human study):")
    print_summary("TRAIN", train_records, mode)
    print_summary("TEST",  test_records,  mode)
    if not args.skip_llm:
        print(f"\nSummary (LLM_data):")
        print_summary("TRAIN", llm_train_records, mode)
        print_summary("TEST",  llm_test_records,  mode)
    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()