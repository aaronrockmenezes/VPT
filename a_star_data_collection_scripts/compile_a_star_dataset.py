#!/usr/bin/env python3
"""Compile A* collected envs into a clean labeled dataset.

Walks all per-task per-GPU output dirs, runs sanity checks on the
camera-object POV semantic image, samples to hit category quotas
(50/25/25 in_view/occluded/outside_fov), shuffles, assigns continuous
train/val/test episode splits in master_labels.json, copies to a sequential-ID
canonical output tree.

Usage:
  python compile_a_star_dataset.py \
      --src_root /oscar/scratch/arock3/VPT_DATA_A_STAR/v18_data_collector_run_array \
      --out_dir  /oscar/scratch/arock3/VPT_DATA_A_STAR/v18_compiled \
      --total 15000 --train_frac 0.8 --val_frac 0.1 --test_frac 0.1

CLI overrides for category fractions and validation are available.
"""
import argparse
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_FRACTIONS = {"in_view": 0.50, "occluded": 0.25, "outside_fov": 0.25}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Source per-env structure (relative to data_node{T}_gpu{G}/):
#   RGB/{Yes|No}/rollout/env_{folder_idx}/{step_*.png, final_*.png}
#   Semantic/{Yes|No}/rollout/env_{folder_idx}/...
#   cam/{Yes|No}/rollout/env_{folder_idx}/{final_cam_semantic.png, actions.txt, meta.json}
TYPES = ["RGB", "Semantic", "cam"]

CAM_FINAL_NAME = "final_cam_semantic.png"

# ── sanity checks (adapted from old compiler) ───────────────────────────────
# Thresholds calibrated for 256x256 (65536 px), matches our --img_size=256
# default. Auto-scaled by area for any other input size.
REF_SIDE = 256
REF_AREA = REF_SIDE * REF_SIDE

REF_RED_THRESH = 125  # px @ 256x256; area-equivalent to 500 @ 512
REF_NO_RED_MAX = 0  # px @ 256x256; No means truly no strict-red pixels
REF_GREEN_MIN_PX = 5  # px @ 256x256
REF_CONTOUR_MIN = 50  # px @ 256x256


def _scale_to_img(img_bgr, ref_count):
    """Scale a 256x256-calibrated pixel count to actual image area."""
    if ref_count <= 0:
        return 0
    h, w = img_bgr.shape[:2]
    return max(1, int(round(ref_count * (h * w) / REF_AREA)))


def red_pixel_count(img_bgr):
    """Count strict semantic-red pixels in a cam-POV image."""
    if img_bgr is None:
        return 0
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return int(((r >= 0.95) & (g <= 0.05) & (b <= 0.05)).sum())


def has_red(img_bgr, threshold=None):
    """Strict red mask count must exceed the Yes threshold."""
    if img_bgr is None:
        return False
    if threshold is None:
        threshold = _scale_to_img(img_bgr, REF_RED_THRESH)
    return red_pixel_count(img_bgr) > threshold


def cam_label_reason(cam_img, label):
    """Return ``(ok, reason, red_count)`` for cam-POV semantic label check.

    Yes requires more than ``REF_RED_THRESH`` strict-red pixels. No permits at
    most ``REF_NO_RED_MAX`` strict-red pixels plus the existing circle
    rejection. The interval ``REF_NO_RED_MAX+1..REF_RED_THRESH`` is a deadzone.
    """
    if cam_img is None:
        return False, "cam_missing", 0

    red_count = red_pixel_count(cam_img)
    yes_thresh = _scale_to_img(cam_img, REF_RED_THRESH)
    no_max = _scale_to_img(cam_img, REF_NO_RED_MAX)

    if label == "Yes":
        if red_count > yes_thresh:
            return True, "ok", red_count
        return False, "yes_below_red_threshold", red_count

    if label == "No":
        if red_count > no_max:
            return False, "no_has_red_or_deadzone", red_count
        if has_circle(cam_img):
            return False, "no_has_circle", red_count
        return True, "ok", red_count

    return False, "bad_label", red_count


def has_green(img_bgr, trim=0, threshold=None):
    """Any non-trivial green presence (HSV)."""
    if img_bgr is None:
        return False
    if trim > 0:
        h, w = img_bgr.shape[:2]
        if h <= trim * 2 or w <= trim * 2:
            return False
        img_bgr = img_bgr[trim:-trim, trim:-trim]
    if threshold is None:
        threshold = _scale_to_img(img_bgr, REF_GREEN_MIN_PX)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    return cv2.countNonZero(mask) > threshold


def has_circle(img_bgr, fill_thresh=0.80, min_area=None):
    """True if any contour fills its min-enclosing circle by > fill_thresh."""
    if img_bgr is None:
        return False
    if min_area is None:
        min_area = _scale_to_img(img_bgr, REF_CONTOUR_MIN)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        (_, _), r = cv2.minEnclosingCircle(c)
        ca = np.pi * r * r
        if ca and (a / ca) > fill_thresh:
            return True
    return False


def cam_label_matches(cam_img, label):
    """
    Returns True if cam-POV semantic image is consistent with `label`.
    Yes  → red goal must exceed the Yes pixel threshold.
    No   → at most REF_NO_RED_MAX red pixels AND no unlabeled circle.
    """
    ok, _, _ = cam_label_reason(cam_img, label)
    return ok


# ── env discovery + verification ─────────────────────────────────────────────
def list_image_files(p: Path):
    """List image filenames in a directory.

    Parameters
    ----------
    p
        Directory to scan.

    Returns
    -------
    list[str]
        Image filenames directly inside ``p``. Missing directories return an
        empty list.
    """
    if not p.exists():
        return []
    return [f for f in os.listdir(p) if Path(f).suffix.lower() in IMAGE_EXTS]


def verify_env(gpu_root: Path, label: str, folder_idx: int, min_frames: int):
    """Return (ok, reason_str). Checks dirs exist, file counts consistent."""
    rgb_dir = gpu_root / "RGB" / label / "rollout" / f"env_{folder_idx}"
    sem_dir = gpu_root / "Semantic" / label / "rollout" / f"env_{folder_idx}"
    cam_dir = gpu_root / "cam" / label / "rollout" / f"env_{folder_idx}"

    if not (rgb_dir.is_dir() and sem_dir.is_dir() and cam_dir.is_dir()):
        return False, "missing_dir"
    cam_final = cam_dir / CAM_FINAL_NAME
    meta_json = cam_dir / "meta.json"
    actions_txt = cam_dir / "actions.txt"
    if not cam_final.exists():
        return False, "missing_cam_final"
    if not meta_json.exists():
        return False, "missing_meta"
    if not actions_txt.exists():
        return False, "missing_actions"

    rgb_n = len(list_image_files(rgb_dir))
    sem_n = len(list_image_files(sem_dir))
    if rgb_n != sem_n:
        return False, f"rgb/sem count mismatch ({rgb_n}!={sem_n})"
    if rgb_n < min_frames:
        return False, f"too_few_frames ({rgb_n}<{min_frames})"
    return True, "ok"


def discover_staged_envs(src_root: Path,
                         job_id: str | None = None,
                         do_cam_check: bool = True):
    """Walk {src_root}/data/data_node*_compiled/{RGB,Semantic,cam}/{Yes,No}/
    {reason}/env_*.  Per-task staging already verified each env.

    If `job_id` is set, restricts to dirs `data_node{job_id}_*_compiled` so
    one final merge stays scoped to a single SLURM submission.
    """
    pool = {"in_view": [], "occluded": [], "outside_fov": []}
    rejected = {"staged_dir_missing": 0, "cam_label_mismatch": 0}

    data_dir = src_root / "data"
    if job_id:
        glob_pat = f"data_node{job_id}_*_compiled"
    else:
        glob_pat = "data_node*_compiled"
    task_dirs = sorted(data_dir.glob(glob_pat))
    if not task_dirs:
        print(
            f"[ERR] no {glob_pat} under {data_dir}; "
            f"run compile_tasks.py per task first or use --mode raw.",
            file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(task_dirs)} compiled task dirs (filter: {glob_pat}).")

    for tdir in task_dirs:
        # Load tracker so we can attach original-src info to each env.
        tracker_idx = {}  # env_name → original record
        tracker_p = tdir / "successful_envs.json"
        if tracker_p.exists():
            try:
                t = json.loads(tracker_p.read_text())
                for rec in t.get("envs", []):
                    if "staged_env" in rec:
                        tracker_idx[rec["staged_env"]] = rec
            except Exception as e:
                print(f"[WARN] tracker parse failed at {tracker_p}: {e}")

        for label in ("Yes", "No"):
            for reason in pool.keys():
                rg = tdir / "RGB" / label / reason
                sg = tdir / "Semantic" / label / reason
                cm = tdir / "cam" / label / reason
                if not rg.is_dir():
                    continue
                for env_dir in rg.iterdir():
                    if not env_dir.is_dir():
                        continue
                    env_name = env_dir.name
                    if not (sg / env_name).is_dir() \
                            or not (cm / env_name).is_dir():
                        rejected["staged_dir_missing"] += 1
                        continue
                    if do_cam_check:
                        cam_path = cm / env_name / CAM_FINAL_NAME
                        cam_img = cv2.imread(str(cam_path), cv2.IMREAD_COLOR)
                        ok, why, _ = cam_label_reason(cam_img, label)
                        if not ok:
                            rejected[why] = rejected.get(why, 0) + 1
                            rejected["cam_label_mismatch"] += 1
                            continue
                    original = tracker_idx.get(env_name, {})
                    # Per-task staging already verified frame counts. Avoid
                    # expensive listdir over every candidate env on Lustre.
                    n_frames = int(original.get("total_steps", 0) or 0) + 1
                    pool[reason].append({
                        "src_task_dir": str(tdir),
                        "env_name": env_name,
                        "label": label,
                        "reason": reason,
                        "n_frames": n_frames,
                        "original": original,
                    })

    return pool, rejected


def discover_envs(src_root: Path, min_frames: int, do_cam_check: bool):
    """Walk all per-GPU dirs, collect verified env records (raw mode)."""
    pool = {"in_view": [], "occluded": [], "outside_fov": []}
    rejected = {
        "missing_dir": 0,
        "missing_cam_final": 0,
        "missing_meta": 0,
        "missing_actions": 0,
        "rgb/sem count mismatch": 0,
        "too_few_frames": 0,
        "cam_label_mismatch": 0,
        "unknown_reason": 0,
        "no_meta_load": 0
    }

    gpu_dirs = sorted(p for p in (src_root / "data").glob("data_node*_gpu*")
                      if p.is_dir())
    if not gpu_dirs:
        print(f"[ERR] no data_node*_gpu* dirs under {src_root/'data'}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(gpu_dirs)} GPU output dirs.")

    for gpu_root in tqdm(gpu_dirs, desc="Scanning GPUs"):
        tracker = gpu_root / "successful_envs.json"
        if not tracker.exists():
            continue
        try:
            data = json.loads(tracker.read_text())
        except Exception as e:
            print(f"[WARN] cant load {tracker}: {e}")
            continue

        for entry in data.get("envs", []):
            folder_idx = entry.get("folder_idx")
            label = entry.get("label")
            reason = entry.get("reason")
            if folder_idx is None or label not in ("Yes", "No"):
                rejected["no_meta_load"] += 1
                continue
            if reason not in pool:
                rejected["unknown_reason"] += 1
                continue

            ok, why = verify_env(gpu_root, label, folder_idx, min_frames)
            if not ok:
                key = why.split(" ")[0]
                rejected[key] = rejected.get(key, 0) + 1
                continue

            if do_cam_check:
                cam_path = (gpu_root / "cam" / label / "rollout" /
                            f"env_{folder_idx}" / CAM_FINAL_NAME)
                cam_img = cv2.imread(str(cam_path), cv2.IMREAD_COLOR)
                if not cam_label_matches(cam_img, label):
                    rejected["cam_label_mismatch"] += 1
                    continue

            pool[reason].append({
                "src_gpu_root":
                str(gpu_root),
                "folder_idx":
                folder_idx,
                "label":
                label,
                "reason":
                reason,
                "meta":
                entry,
                "n_frames":
                len(
                    list_image_files(gpu_root / "RGB" / label / "rollout" /
                                     f"env_{folder_idx}")),
            })

    return pool, rejected


# ── output writer ────────────────────────────────────────────────────────────
def make_dirs(out: Path):
    """Create canonical output dirs only; splits live in master_labels.json."""
    for stale_split in ("train", "val", "test"):
        stale_path = out / stale_split
        if stale_path.exists():
            print(f"[INFO] removing stale split dir: {stale_path}")
            if stale_path.is_symlink() or stale_path.is_file():
                stale_path.unlink()
            else:
                shutil.rmtree(stale_path)
    for t in TYPES:
        for lbl in ("Yes", "No"):
            (out / t / lbl).mkdir(parents=True, exist_ok=True)


def copy_env(item, out: Path, new_id: int, link_only: bool, mode: str):
    """Copy (or symlink) one env's RGB/Semantic/cam dirs into out tree."""
    label = item["label"]

    for t in TYPES:
        if mode == "staged":
            src_dir = (Path(item["src_task_dir"]) / t / label /
                       item["reason"] / item["env_name"])
        else:
            src_dir = (Path(item["src_gpu_root"]) / t / label / "rollout" /
                       f"env_{item['folder_idx']}")
        dst_dir = out / t / label / f"env_{new_id}"
        if dst_dir.exists():
            if dst_dir.is_symlink():
                dst_dir.unlink()
            else:
                shutil.rmtree(dst_dir)
        if link_only:
            os.symlink(src_dir, dst_dir, target_is_directory=True)
        else:
            shutil.copytree(src_dir, dst_dir)


def main():
    """Run final A* dataset compilation CLI.

    The command samples a globally balanced pool, assigns continuous
    train/val/test episode ranges in ``master_labels.json``, and writes one
    canonical RGB/Semantic/cam tree.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src_root",
                    required=True,
                    help="Collector BASE_PATH (parent of `data/`).")
    ap.add_argument("--out_dir",
                    required=True,
                    help="Compiled dataset destination.")
    ap.add_argument("--total",
                    type=int,
                    default=15000,
                    help="Total compiled envs after split (default 15000).")
    ap.add_argument("--train_frac",
                    type=float,
                    default=0.8,
                    help="Train episode fraction (default 0.8).")
    ap.add_argument("--val_frac",
                    type=float,
                    default=0.1,
                    help="Validation episode fraction (default 0.1).")
    ap.add_argument("--test_frac",
                    type=float,
                    default=0.1,
                    help="Test episode fraction (default 0.1).")
    ap.add_argument("--min_frames",
                    type=int,
                    default=10,
                    help="Reject envs with fewer than this many RGB frames.")
    ap.add_argument("--no_cam_check",
                    action="store_true",
                    help="Skip cam-POV semantic sanity check (faster).")
    ap.add_argument("--cam_red_thresh",
                    type=int,
                    default=125,
                    help="Yes requires more than this many strict-red pixels "
                    "at 256x256. Auto-scales by area. No requires zero red "
                    "pixels. Default: 125.")
    ap.add_argument("--cam_no_red_max",
                    type=int,
                    default=0,
                    help="No permits at most this many strict-red pixels at "
                    "256x256 before entering deadzone. Auto-scales by area. "
                    "Default: 0.")
    ap.add_argument("--symlink",
                    action="store_true",
                    help="Symlink env dirs instead of copy (saves space).")
    ap.add_argument("--workers",
                    type=int,
                    default=32,
                    help="Parallel env copy workers (default 32).")
    ap.add_argument("--overwrite_out",
                    action="store_true",
                    help="Delete out_dir before writing. Use for reruns after "
                    "interrupted compiles.")
    ap.add_argument("--seed",
                    type=int,
                    default=0,
                    help="Shuffle seed for reproducibility.")
    ap.add_argument("--mode",
                    choices=["staged", "raw"],
                    default="staged",
                    help="'staged': read from {src}/data/data_node*_compiled/ "
                    "(produced by compile_tasks.py). "
                    "'raw': scan raw {src}/data/data_node*_gpu*/ "
                    "and run sanity checks here.")
    ap.add_argument("--job_id",
                    type=str,
                    default=None,
                    help="If set, only merge tasks matching "
                    "data_node{job_id}_*_compiled (one SLURM submission). "
                    "Default: merge all compiled tasks across jobs.")
    args = ap.parse_args()

    global REF_RED_THRESH, REF_NO_RED_MAX
    REF_RED_THRESH = args.cam_red_thresh
    REF_NO_RED_MAX = args.cam_no_red_max
    if REF_NO_RED_MAX >= REF_RED_THRESH:
        print("[FATAL] --cam_no_red_max must be less than "
              "--cam_red_thresh")
        sys.exit(2)

    random.seed(args.seed)
    src_root = Path(args.src_root)
    out_dir = Path(args.out_dir)

    print(f"\n--- Discovering envs under {src_root} (mode={args.mode}, "
          f"job_id={args.job_id or 'ALL'}) ---")
    if args.mode == "staged":
        pool, rejected = discover_staged_envs(
            src_root, job_id=args.job_id, do_cam_check=not args.no_cam_check)
    else:
        pool, rejected = discover_envs(src_root, args.min_frames,
                                       not args.no_cam_check)

    print("\nPool size by category:")
    for cat, items in pool.items():
        print(f"  {cat:>12}: {len(items):>6}")
    print("\nRejected counts:")
    for reason, n in sorted(rejected.items(), key=lambda x: -x[1]):
        if n:
            print(f"  {reason:>22}: {n}")

    # Allocate per category
    fractions = DEFAULT_FRACTIONS
    targets = {c: int(round(args.total * f)) for c, f in fractions.items()}
    diff = args.total - sum(targets.values())
    targets["in_view"] += diff  # absorb rounding into largest bin

    print(f"\nCategory targets (total {args.total}):")
    for c, t in targets.items():
        avail = len(pool[c])
        marker = "✓" if avail >= t else "✗"
        print(f"  {marker} {c:>12}: need {t:>5}, have {avail:>5}")
    if any(len(pool[c]) < targets[c] for c in targets):
        print("\n[FATAL] not enough collected envs to hit targets. "
              "Run more collection or lower --total.")
        sys.exit(2)

    split_fracs = {
        "train": args.train_frac,
        "val": args.val_frac,
        "test": args.test_frac,
    }
    split_total = sum(split_fracs.values())
    if abs(split_total - 1.0) > 1e-6:
        print(f"[FATAL] split fractions must sum to 1.0, got {split_total}")
        sys.exit(2)

    # Sample globally balanced categories, then shuffle once. Splits are by
    # contiguous episode ID ranges, not frame count.
    ordered = []
    for c, items in pool.items():
        random.shuffle(items)
        ordered.extend(items[:targets[c]])
    random.shuffle(ordered)

    n_train = int(round(args.total * args.train_frac))
    n_val = int(round(args.total * args.val_frac))
    n_test = args.total - n_train - n_val
    if n_test < 0:
        print("[FATAL] split rounding produced negative test count")
        sys.exit(2)

    split_ranges = {
        "train": [0, n_train - 1] if n_train else [],
        "val": [n_train, n_train + n_val - 1] if n_val else [],
        "test": [n_train + n_val, args.total - 1] if n_test else [],
    }

    def split_for_id(env_id: int) -> str:
        if env_id < n_train:
            return "train"
        if env_id < n_train + n_val:
            return "val"
        return "test"

    print(f"\n--- Writing {len(ordered)} envs to {out_dir} ---")
    if out_dir.exists():
        if args.overwrite_out:
            print(f"[INFO] removing existing out_dir: {out_dir}")
            shutil.rmtree(out_dir)
        else:
            print(f"[WARN] {out_dir} already exists; merging into it. "
                  "Use --overwrite_out for a clean rerun.")
    make_dirs(out_dir)

    master = {
        "environments": {},
        "statistics": {
            "total_environments": len(ordered),
            "yes_count": 0,
            "no_count": 0,
            "by_reason": {c: 0
                          for c in DEFAULT_FRACTIONS},
            "splits": {
                "train": n_train,
                "val": n_val,
                "test": n_test
            },
            "split_fractions": split_fracs,
            "split_ranges": split_ranges,
            "src_root": str(src_root),
            "seed": args.seed,
            "cam_check": {
                "yes_red_threshold_px_at_256": REF_RED_THRESH,
                "no_red_max_px_at_256": REF_NO_RED_MAX,
                "deadzone_px_at_256": [REF_NO_RED_MAX + 1, REF_RED_THRESH],
            },
        },
    }

    def copy_one(pair):
        new_id, item = pair
        copy_env(item, out_dir, new_id, args.symlink, args.mode)
        return new_id, item

    copied = []
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(copy_one, pair) for pair in enumerate(ordered)]
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc=f"Copying ({workers} workers)"):
            try:
                copied.append(fut.result())
            except Exception as e:
                print(f"[WARN] copy failed: {e}")

    copied.sort(key=lambda x: x[0])
    if len(copied) != len(ordered):
        print(f"[FATAL] copied {len(copied)} / {len(ordered)} envs")
        sys.exit(2)

    for new_id, item in copied:
        split = split_for_id(new_id)

        master["environments"][str(new_id)] = {
            "label": item["label"],
            "reason": item["reason"],
            "n_frames": item["n_frames"],
            "split": split,
            "src": item.get("src_task_dir") or item.get("src_gpu_root"),
            "src_id": item.get("env_name") or f"env_{item.get('folder_idx')}",
            # Full provenance: original GPU dir, folder_idx, seed, etc.
            "original": item.get("original", {}),
        }
        master["statistics"]["by_reason"][item["reason"]] += 1
        if item["label"] == "Yes":
            master["statistics"]["yes_count"] += 1
        else:
            master["statistics"]["no_count"] += 1

    (out_dir / "master_labels.json").write_text(json.dumps(master, indent=2))

    print("\n--- Final stats ---")
    for k, v in master["statistics"].items():
        print(f"  {k}: {v}")
    print(f"\n✅ wrote {len(ordered)} envs to {out_dir}")


if __name__ == "__main__":
    main()
