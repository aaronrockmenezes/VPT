"""
build_vpt2_dataset.py
---------------------
Aggregates VPT2 environments from multiple source directories into a
single re-indexed dataset with train/test splits.

VPT2 extends VPT1 by:
  - Labels: 'left' / 'right' instead of 'Yes' / 'No' (no reason categories)
  - cam_pov QC: both red (goal) AND pink (ref obj) must be visible
  - Semantic QC: every image must have red AND pink AND green >= thresholds
  - 50/50 left/right balance enforced at train, test, and overall level

Pipeline:
  1. QC   - cam_pov checked first; fail → blacklist immediately
           - all semantic images checked next; any single fail → blacklist
  2. Pool  - valid envs bucketed by label (left / right)
  3. Alloc - 50/50 left/right enforced per split, re-indexed
  4. Copy  - RGB, Depth, Semantic, cam, configs → OUTPUT_DIR; master JSON written
  5. Validate - structural correctness verified post-copy

Assumptions:
  - Source dirs match data_node<N>_gpu<N> or data_node<JOB>_<TASK>_gpu<N>
    inside BASE_DIR
  - Labels: 'left' / 'right' only, no reason subcategories
  - cam_pov QC: red and pink must be above scaled pixel thresholds
  - Semantic QC: every image must be readable and mostly keep all VPT2 objects:
      red goal count > scaled SEMANTIC_RED_THRESHOLD
      pink reference count > scaled SEMANTIC_PINK_THRESHOLD
      green camera presence > scaled SEMANTIC_GREEN_MIN_PX
  - Missing semantic folder is a hard reject
  - Train/test split folder contains RGB only
"""

import os, json, shutil, random, re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR = os.getenv(
    "VPT2_BASE_DIR",
    "/users/arock3/scratch/VPT2_DATA/v4/data",
)
DIR_PATTERN = r"^data_node(?:\d+|\d+_\d+)_gpu\d+$"

OUTPUT_DIR = os.getenv("VPT2_OUTPUT_DIR", "/users/arock3/scratch/VPT2_v4")
EXPECTED_IMAGES_PER_ENV = 10
REQUIRED_CAM_FILES = ("cam_pov.png",)

M            = 2 ** 3
TRAIN_COUNT  = 32 * M
TEST_COUNT   = 32 * M
TOTAL_NEEDED = TRAIN_COUNT + TEST_COUNT

# Pixel QC thresholds are calibrated at 256x256 and scaled by image area.
# VPT2 has no No/negative visibility label: both left and right require goal
# and reference object visible in cam_pov. The deadzone is the minimum visible
# red/pink area; tiny object slivers are rejected.
REF_SIDE = 256
REF_AREA = REF_SIDE * REF_SIDE
CAM_RED_THRESHOLD = 125
CAM_PINK_THRESHOLD = 125
SEMANTIC_RED_THRESHOLD = 125
SEMANTIC_PINK_THRESHOLD = 125
SEMANTIC_GREEN_MIN_PX = 5
SEMANTIC_FAIL_TOLERANCE = 1

FOLDERS = {
    "rgb":      "RGB",
    "depth":    "Depth",
    "semantic": "Semantic",
    "cam":      "cam",
    "configs":  "configs",
}

LABELS_FILENAME  = "visibility_labels.json"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

# ── QC HELPERS ────────────────────────────────────────────────────────────────

def _scale_to_img(img_bgr: np.ndarray, ref_count: int) -> int:
    """Scale a 256x256-calibrated pixel threshold to image area."""
    if ref_count <= 0:
        return 0
    h, w = img_bgr.shape[:2]
    return max(1, int(round(ref_count * (h * w) / REF_AREA)))


def _image_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(
        f for f in os.listdir(path)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    )


def _red_pixel_count(img_bgr: np.ndarray) -> int:
    """Count pure-red pixels (goal object). img_bgr is uint8 0-255."""
    if img_bgr is None:
        return 0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (
        (img_rgb[:, :, 0] >= 0.95) &
        (img_rgb[:, :, 1] <= 0.05) &
        (img_rgb[:, :, 2] <= 0.05)
    )
    return int(mask.sum())


def _pink_pixel_count(img_bgr: np.ndarray) -> int:
    """Count pure-pink pixels (reference object). img_bgr is uint8 0-255."""
    if img_bgr is None:
        return 0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (
        (img_rgb[:, :, 0] >= 0.95) &
        (img_rgb[:, :, 1] <= 0.05) &
        (img_rgb[:, :, 2] >= 0.95)
    )
    return int(mask.sum())


def _green_pixel_count(img_bgr: np.ndarray) -> int:
    """Count pure-green pixels (reference object). img_bgr is uint8 0-255."""
    if img_bgr is None:
        return 0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (
        (img_rgb[:, :, 0] <= 0.05) &
        (img_rgb[:, :, 1] >= 0.95) &
        (img_rgb[:, :, 2] <= 0.05)
    )
    return int(mask.sum())


def _has_green_hsv(img_bgr: np.ndarray, threshold: Optional[int] = None) -> bool:
    """A*-style non-trivial green presence check."""
    if img_bgr is None:
        return False
    if threshold is None:
        threshold = _scale_to_img(img_bgr, SEMANTIC_GREEN_MIN_PX)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    return cv2.countNonZero(mask) > threshold


def check_cam_pov(src_root: str, label: str, env_id: str) -> tuple[bool, str]:
    """
    QC on cam_pov.png.
    Both left and right require:
      - red  >= CAM_RED_THRESHOLD
      - pink >= CAM_PINK_THRESHOLD

    Returns (passed: bool, reason: str).
    """
    path = Path(src_root) / FOLDERS["cam"] / label / f"env_{env_id}" / "cam_pov.png"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return False, "cam_pov.png missing or unreadable"

    red = _red_pixel_count(img)
    pink = _pink_pixel_count(img)
    red_thresh = _scale_to_img(img, CAM_RED_THRESHOLD)
    pink_thresh = _scale_to_img(img, CAM_PINK_THRESHOLD)

    if red <= red_thresh:
        return False, f"red={red} <= {red_thresh}"
    if pink <= pink_thresh:
        return False, f"pink={pink} <= {pink_thresh}"
    return True, ""


def check_semantic_images(src_root: str, label: str, env_id: str) -> tuple[bool, str]:
    """
    QC on all semantic images for an env.

    Every image must:
      1. Be readable (not None)
      2. Have red   >= SEMANTIC_RED_THRESHOLD
      3. Have pink  >= SEMANTIC_PINK_THRESHOLD
      4. Have green >= SEMANTIC_GREEN_THRESHOLD

    Any single image failing blacklists the entire env.

    Returns (passed: bool, reason: str).
    """
    sem_dir = Path(src_root) / FOLDERS["semantic"] / label / f"env_{env_id}"

    if not sem_dir.exists():
        return False, "Semantic folder missing"

    files = _image_files(sem_dir)

    if len(files) != EXPECTED_IMAGES_PER_ENV:
        return False, f"Expected {EXPECTED_IMAGES_PER_ENV} semantic images, found {len(files)}"

    threshold_failures = []
    for fname in files:
        img = cv2.imread(str(sem_dir / fname), cv2.IMREAD_COLOR)

        if img is None:
            return False, f"Unreadable: {fname}"

        red_thresh = _scale_to_img(img, SEMANTIC_RED_THRESHOLD)
        red = _red_pixel_count(img)
        if red <= red_thresh:
            threshold_failures.append(f"{fname}: red={red} <= {red_thresh}")
            continue

        pink_thresh = _scale_to_img(img, SEMANTIC_PINK_THRESHOLD)
        pink = _pink_pixel_count(img)
        if pink <= pink_thresh:
            threshold_failures.append(f"{fname}: pink={pink} <= {pink_thresh}")
            continue

        if not _has_green_hsv(img):
            green = _green_pixel_count(img)
            green_thresh = _scale_to_img(img, SEMANTIC_GREEN_MIN_PX)
            threshold_failures.append(
                f"{fname}: green_hsv<=threshold strict_green={green} "
                f"threshold={green_thresh}"
            )

    if len(threshold_failures) > SEMANTIC_FAIL_TOLERANCE:
        return False, threshold_failures[0]

    return True, ""

# ── DATASET HELPERS ───────────────────────────────────────────────────────────

def verify_environment(src_root: str, env_id: str, label: str) -> tuple[bool, str]:
    """
    Returns (passed, reason).
    Checks:
      1. All required folders/files exist
      2. RGB has exactly EXPECTED_IMAGES_PER_ENV images
      3. cam_pov QC (fail → immediate blacklist)
      4. Semantic QC on all images (any fail → blacklist)
    """
    root = Path(src_root)
    p_rgb      = root / FOLDERS["rgb"]      / label / f"env_{env_id}"
    p_depth    = root / FOLDERS["depth"]    / label / f"env_{env_id}"
    p_semantic = root / FOLDERS["semantic"] / label / f"env_{env_id}"
    p_cam      = root / FOLDERS["cam"]      / label / f"env_{env_id}"
    p_conf     = root / FOLDERS["configs"]  / f"env_{env_id}_config.json"

    for p in [p_rgb, p_depth, p_semantic, p_cam, p_conf]:
        if not p.exists():
            return False, f"Missing: {p.name}"

    try:
        rgb_count = len(_image_files(p_rgb))
        if rgb_count != EXPECTED_IMAGES_PER_ENV:
            return False, f"RGB count: {rgb_count} != {EXPECTED_IMAGES_PER_ENV}"
        sem_count = len(_image_files(p_semantic))
        if sem_count != EXPECTED_IMAGES_PER_ENV:
            return False, f"Semantic count: {sem_count} != {EXPECTED_IMAGES_PER_ENV}"
        if rgb_count != sem_count:
            return False, f"RGB/Semantic count mismatch: {rgb_count}!={sem_count}"
    except OSError as e:
        return False, f"OSError reading RGB: {e}"

    cam_ok, cam_reason = check_cam_pov(src_root, label, env_id)
    if not cam_ok:
        return False, f"cam_pov fail: {cam_reason}"

    sem_ok, sem_reason = check_semantic_images(src_root, label, env_id)
    if not sem_ok:
        return False, f"semantic fail: {sem_reason}"

    return True, ""


def check_balance(envs: list, stage: str, expected_total: int) -> None:
    """Asserts exact total and 50/50 left/right split. Raises on failure."""
    total       = len(envs)
    left_count  = sum(1 for e in envs if e["label"] == "left")
    right_count = sum(1 for e in envs if e["label"] == "right")
    half        = expected_total // 2

    print(f"\n[BALANCE CHECK — {stage}]")
    print(f"  Total : {total}  (expected {expected_total})")
    print(f"  Left  : {left_count}  (expected {half})")
    print(f"  Right : {right_count}  (expected {half})")

    errors = []
    if total != expected_total:
        errors.append(f"Total mismatch: got {total}, expected {expected_total}")
    if left_count != half:
        errors.append(f"Left mismatch: got {left_count}, expected {half}")
    if right_count != half:
        errors.append(f"Right mismatch: got {right_count}, expected {half}")

    if errors:
        for e in errors:
            print(f"  ❌ {e}")
        raise ValueError(f"Balance check failed at '{stage}': " + "; ".join(errors))

    print(f"  ✅ 50/50 confirmed.")


def create_output_structure(base: Path) -> None:
    for folder in (FOLDERS["rgb"], FOLDERS["depth"], FOLDERS["semantic"], FOLDERS["cam"]):
        for label in ("left", "right"):
            (base / folder / label).mkdir(parents=True, exist_ok=True)
    (base / FOLDERS["configs"]).mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for label in ("left", "right"):
            (base / split / label).mkdir(parents=True, exist_ok=True)

# ── VALIDATION ────────────────────────────────────────────────────────────────

def validate_folder(base: Path, json_data: dict, scope: str = "root") -> None:
    print(f"\n[Validate] {scope.upper()}")
    errors = []

    all_indices = sorted(int(k) for k in json_data["environments"])
    if scope == "train":
        indices = [i for i in all_indices if i < TRAIN_COUNT]
    elif scope == "test":
        indices = [i for i in all_indices if i >= TRAIN_COUNT]
    else:
        indices = all_indices

    for idx in tqdm(indices, desc=scope):
        label = json_data["environments"][str(idx)]["label"]

        if scope == "root":
            checks = [
                (base / FOLDERS["rgb"]      / label / f"env_{idx}", EXPECTED_IMAGES_PER_ENV),
                (base / FOLDERS["depth"]    / label / f"env_{idx}", EXPECTED_IMAGES_PER_ENV),
                (base / FOLDERS["semantic"] / label / f"env_{idx}", EXPECTED_IMAGES_PER_ENV),
            ]
            cam_path = base / FOLDERS["cam"] / label / f"env_{idx}"
        else:
            checks = [
                (base / scope / label / f"env_{idx}", EXPECTED_IMAGES_PER_ENV),
            ]
            cam_path = None

        for path, expected in checks:
            if not path.exists():
                errors.append(f"Missing: {path}")
                continue
            count = sum(
                1 for f in os.listdir(path)
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS
            )
            if count != expected:
                errors.append(f"Count mismatch: {path} ({count} ≠ {expected})")

        if cam_path is not None:
            if not cam_path.exists():
                errors.append(f"Missing: {cam_path}")
            else:
                for fname in REQUIRED_CAM_FILES:
                    if not (cam_path / fname).exists():
                        errors.append(f"Missing required cam file: {cam_path / fname}")

    if not errors:
        print(f"  ✅ {scope} OK")
    else:
        print(f"  ❌ {scope} — {len(errors)} errors:")
        for e in errors[:10]:
            print(f"     {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # ── 0. DISCOVER source directories ────────────────────────────────────────
    source_dirs = sorted(
        str(Path(BASE_DIR) / d)
        for d in os.listdir(BASE_DIR)
        if re.match(DIR_PATTERN, d) and (Path(BASE_DIR) / d).is_dir()
    )
    if not source_dirs:
        print(f"❌ No directories matching '{DIR_PATTERN}' found in {BASE_DIR}")
        return
    print(f"Found {len(source_dirs)} source dirs:")
    for d in source_dirs:
        print(f"  {d}")

    # ── 1. POOL valid environments by label ───────────────────────────────────
    pool = {"left": [], "right": []}
    qc_rejected = 0

    for src in source_dirs:
        labels_path = Path(src) / LABELS_FILENAME
        if not labels_path.exists():
            print(f"  Skipping {src} (no labels file)")
            continue

        print(f"\nScanning: {src}")
        with open(labels_path) as f:
            vis_data = json.load(f)

        src_accepted = 0
        src_rejected = 0

        for env_id, info in sorted(
            vis_data.get("environments", {}).items(), key=lambda x: int(x[0])
        ):
            label = info.get("label")
            if label not in ("left", "right"):
                print(f"  Warning: unexpected label '{label}' for env_{env_id}, skipping.")
                continue

            passed, fail_reason = verify_environment(src, env_id, label)
            if passed:
                pool[label].append({
                    "src_root": src,
                    "old_id":   env_id,
                    "label":    label,
                    "details":  info,
                })
                src_accepted += 1
            else:
                src_rejected += 1
                qc_rejected  += 1

        print(f"  → Accepted: {src_accepted} | Rejected: {src_rejected}")

    total_left  = len(pool["left"])
    total_right = len(pool["right"])
    print(f"\nPool summary:")
    print(f"  left  : {total_left}")
    print(f"  right : {total_right}")
    print(f"  QC rejected total: {qc_rejected}")

    # ── 2. ALLOCATE train / test with guaranteed 50/50 ────────────────────────
    half_train = TRAIN_COUNT // 2
    half_test  = TEST_COUNT  // 2
    half_total = half_train + half_test

    if total_left < half_total:
        print(f"❌ Need {half_total} left envs, only {total_left} available.")
        return
    if total_right < half_total:
        print(f"❌ Need {half_total} right envs, only {total_right} available.")
        return

    left_envs  = pool["left"][:]
    right_envs = pool["right"][:]
    random.shuffle(left_envs)
    random.shuffle(right_envs)

    train_batch = left_envs[:half_train]           + right_envs[:half_train]
    test_batch  = left_envs[half_train:half_total] + right_envs[half_train:half_total]

    random.shuffle(train_batch)
    random.shuffle(test_batch)
    ordered = train_batch + test_batch

    # ── 3. PRE-COPY BALANCE CHECKS ────────────────────────────────────────────
    check_balance(ordered,     stage="PRE-COPY (ALL)",   expected_total=TOTAL_NEEDED)
    check_balance(train_batch, stage="PRE-COPY (TRAIN)", expected_total=TRAIN_COUNT)
    check_balance(test_batch,  stage="PRE-COPY (TEST)",  expected_total=TEST_COUNT)

    # ── 4. COPY data to output ────────────────────────────────────────────────
    out = Path(OUTPUT_DIR)
    if out.exists():
        print(f"\nWarning: Output dir {out} already exists.")
    create_output_structure(out)

    master_json = {
        "environments": {},
        "statistics": {
            "total_environments": len(ordered),
            "left_count":  0,
            "right_count": 0,
        },
    }

    print(f"\nCopying {len(ordered)} environments → {out}")
    for new_idx, item in enumerate(tqdm(ordered, desc="Copying")):
        src    = Path(item["src_root"])
        old_id = item["old_id"]
        label  = item["label"]
        split  = "train" if new_idx < TRAIN_COUNT else "test"

        shutil.copytree(src / FOLDERS["rgb"]      / label / f"env_{old_id}",
                        out / FOLDERS["rgb"]      / label / f"env_{new_idx}",
                        dirs_exist_ok=True)
        shutil.copytree(src / FOLDERS["depth"]    / label / f"env_{old_id}",
                        out / FOLDERS["depth"]    / label / f"env_{new_idx}",
                        dirs_exist_ok=True)
        shutil.copytree(src / FOLDERS["semantic"] / label / f"env_{old_id}",
                        out / FOLDERS["semantic"] / label / f"env_{new_idx}",
                        dirs_exist_ok=True)
        shutil.copytree(src / FOLDERS["cam"]      / label / f"env_{old_id}",
                        out / FOLDERS["cam"]      / label / f"env_{new_idx}",
                        dirs_exist_ok=True)
        shutil.copy2(src / FOLDERS["configs"] / f"env_{old_id}_config.json",
                     out / FOLDERS["configs"] / f"env_{new_idx}_config.json")

        shutil.copytree(src / FOLDERS["rgb"] / label / f"env_{old_id}",
                        out / split / label / f"env_{new_idx}",
                        dirs_exist_ok=True)

        master_json["environments"][str(new_idx)] = {
            **item["details"],
            "src_root":   item["src_root"],
            "src_env_id": old_id,
        }
        master_json["statistics"]["left_count" if label == "left" else "right_count"] += 1

    (out / LABELS_FILENAME).write_text(json.dumps(master_json, indent=4))

    # ── 5. STRUCTURAL VALIDATION ──────────────────────────────────────────────
    validate_folder(out, master_json, "root")
    validate_folder(out, master_json, "train")
    validate_folder(out, master_json, "test")

    # ── 6. POST-COPY BALANCE CHECKS ───────────────────────────────────────────
    all_written   = [{"label": v["label"]} for v in master_json["environments"].values()]
    train_written = [{"label": v["label"]} for k, v in master_json["environments"].items()
                     if int(k) < TRAIN_COUNT]
    test_written  = [{"label": v["label"]} for k, v in master_json["environments"].items()
                     if int(k) >= TRAIN_COUNT]

    check_balance(all_written,   stage="POST-COPY (ALL)",   expected_total=TOTAL_NEEDED)
    check_balance(train_written, stage="POST-COPY (TRAIN)", expected_total=TRAIN_COUNT)
    check_balance(test_written,  stage="POST-COPY (TEST)",  expected_total=TEST_COUNT)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
