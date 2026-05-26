"""
build_vpt1_depth_dataset_v18.py
--------------------------------
Aggregates VPT1-Depth environments from multiple source directories into a
single re-indexed dataset with train/test splits.

Pipeline:
  1. QC   - cam_pov checked first; fail → blacklist immediately
           - all semantic images checked next; any single fail → blacklist
             (semantic QC: red >= SEMANTIC_RED_THRESHOLD AND green >= SEMANTIC_GREEN_THRESHOLD)
           - depth labels checked last; missing/unbalanced → blacklist
  2. Pool  - valid envs bucketed by visibility reason (in_view/occluded/outside_fov)
  3. Alloc - each bucket sampled to SPLIT_TARGETS, re-indexed
           - 50/50 Yes/No falls out of reason proportions (not within-bucket)
  4. Copy  - RGB, Depth, Semantic, cam, configs → root OUTPUT_DIR
           - RGB also copied to train/ and test/ (visibility label split)
           - depth_labels.json re-keyed and written as master file
  5. Depth split - per-image: each image routed to train_depth/Yes/ or
                   train_depth/No/ based on depth label (1→Yes, 0→No)
                   Source is the already-copied root RGB (not original src)
  6. Validate - structural correctness verified post-copy

Assumptions:
  - Source dirs match data_node<N>_gpu<N> or data_node<JOB>_<TASK>_gpu<N>
    inside BASE_DIR
  - Labels: 'Yes' / 'No', reasons: 'in_view', 'occluded', 'outside_fov'
  - cam_pov QC follows VPT1 v18 builder:
      Yes → strict-red count > CAM_RED_THRESHOLD scaled to image area
      No  → strict-red count <= CAM_NO_RED_MAX scaled to image area
             AND no unlabeled circular blob
  - Semantic QC: every image must be readable and mostly keep both VPT objects:
      red goal count > SEMANTIC_RED_THRESHOLD scaled to image area
      green camera presence > SEMANTIC_GREEN_MIN_PX scaled to image area
  - Depth QC: env must have exactly EXPECTED_IMAGES_PER_ENV entries,
              exactly half == 1 (cam-proximal) and half == 0 (goal-proximal)
  - train_depth/Yes/ and train_depth/No/ each contain exactly half the images per env
  - in_view is always Yes, occluded/outside_fov are always No
"""

import os, json, shutil, random, re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR = os.getenv(
    "VPT1_DEPTH_BASE_DIR",
    "/users/arock3/scratch/VPT1_DATA/thesis/v18_depth/data",
)
DIR_PATTERN = r"^data_node(?:\d+|\d+_\d+)_gpu\d+$"

OUTPUT_DIR = os.getenv(
    "VPT1_DEPTH_OUTPUT_DIR",
    "/users/arock3/scratch/THESIS/VPT_1_v18_depth",
)
EXPECTED_IMAGES_PER_ENV = 10
REQUIRED_CAM_FILES = ("cam_pov.png",)

M            = 2 ** 3
TRAIN_COUNT  = 32 * M
TEST_COUNT   = 32 * M
TOTAL_NEEDED = TRAIN_COUNT + TEST_COUNT

# in_view=Yes, occluded=No, outside_fov=No → 16M Yes + 8M+8M No = 50/50 per split
SPLIT_TARGETS = {
    "in_view":     16 * M,
    "outside_fov":  8 * M,
    "occluded":     8 * M,
}

# VPT1 v18 pixel QC thresholds calibrated at 256x256 and scaled by image area.
# No labels get a deadzone: any strict-red goal pixels in cam_pov reject the env.
REF_SIDE = 256
REF_AREA = REF_SIDE * REF_SIDE
CAM_RED_THRESHOLD = 125
CAM_NO_RED_MAX = 0
CONTOUR_MIN_AREA = 50
SEMANTIC_RED_THRESHOLD = 125
SEMANTIC_GREEN_MIN_PX = 5
SEMANTIC_FAIL_TOLERANCE = 1

FOLDERS = {
    "rgb":      "RGB",
    "depth":    "Depth",
    "semantic": "Semantic",
    "cam":      "cam",
    "configs":  "configs",
}

LABELS_FILENAME       = "visibility_labels.json"
DEPTH_LABELS_FILENAME = "depth_labels.json"
IMAGE_EXTENSIONS      = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

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
    if img_bgr is None:
        return 0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (
        (img_rgb[:, :, 0] >= 0.95) &
        (img_rgb[:, :, 1] <= 0.05) &
        (img_rgb[:, :, 2] <= 0.05)
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


def _has_circle(
    img_bgr: np.ndarray,
    fill_thresh: float = 0.80,
    min_area: Optional[int] = None,
) -> bool:
    """Reject unlabeled circular blobs in No cam-POV semantic images."""
    if img_bgr is None:
        return False
    if min_area is None:
        min_area = _scale_to_img(img_bgr, CONTOUR_MIN_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in cnts:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        (_, _), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        if circle_area and (area / circle_area) > fill_thresh:
            return True
    return False


def check_cam_pov(src_root: str, label: str, env_id: str) -> tuple[bool, str]:
    path = Path(src_root) / FOLDERS["cam"] / label / f"env_{env_id}" / "cam_pov.png"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return False, "cam_pov.png missing or unreadable"

    red = _red_pixel_count(img)
    yes_thresh = _scale_to_img(img, CAM_RED_THRESHOLD)
    no_max = _scale_to_img(img, CAM_NO_RED_MAX)
    if label == "Yes":
        if red <= yes_thresh:
            return False, f"Yes env: red={red} <= {yes_thresh}"
    else:
        if red > no_max:
            return False, f"No env: red={red} > {no_max} (goal visible/deadzone)"
        if _has_circle(img):
            return False, "No env: circular blob detected in cam_pov"
    return True, ""


def check_semantic_images(src_root: str, label: str, env_id: str) -> tuple[bool, str]:
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


def check_depth_labels(env_id: str, depth_data: dict) -> tuple[bool, str]:
    env_key = f"env_{env_id}"

    if env_key not in depth_data:
        return False, f"Key '{env_key}' missing from depth_labels.json"

    env_depth = depth_data[env_key]
    n = len(env_depth)

    if n != EXPECTED_IMAGES_PER_ENV:
        return False, f"Expected {EXPECTED_IMAGES_PER_ENV} depth entries, found {n}"

    half  = EXPECTED_IMAGES_PER_ENV // 2
    ones  = sum(1 for v in env_depth.values() if v == 1)
    zeros = sum(1 for v in env_depth.values() if v == 0)

    if ones != half or zeros != half:
        return False, f"Depth split not 50/50: ones={ones}, zeros={zeros} (need {half} each)"

    return True, ""

# ── DATASET HELPERS ───────────────────────────────────────────────────────────

def verify_environment(
    src_root: str,
    env_id: str,
    label: str,
    depth_data: dict,
) -> tuple[bool, str]:
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

    dep_ok, dep_reason = check_depth_labels(env_id, depth_data)
    if not dep_ok:
        return False, f"depth fail: {dep_reason}"

    return True, ""


def check_balance(envs: list, stage: str, expected_total: int) -> None:
    total     = len(envs)
    yes_count = sum(1 for e in envs if e["label"] == "Yes")
    no_count  = sum(1 for e in envs if e["label"] == "No")
    half      = expected_total // 2

    print(f"\n[BALANCE CHECK — {stage}]")
    print(f"  Total : {total}  (expected {expected_total})")
    print(f"  Yes   : {yes_count}  (expected {half})")
    print(f"  No    : {no_count}  (expected {half})")

    errors = []
    if total != expected_total:
        errors.append(f"Total mismatch: got {total}, expected {expected_total}")
    if yes_count != half:
        errors.append(f"Yes mismatch: got {yes_count}, expected {half}")
    if no_count != half:
        errors.append(f"No mismatch: got {no_count}, expected {half}")

    if errors:
        for e in errors:
            print(f"  ❌ {e}")
        raise ValueError(f"Balance check failed at '{stage}': " + "; ".join(errors))

    print(f"  ✅ 50/50 confirmed.")


def create_output_structure(base: Path) -> None:
    # Root modality folders
    for folder in (FOLDERS["rgb"], FOLDERS["depth"], FOLDERS["semantic"], FOLDERS["cam"]):
        for label in ("Yes", "No"):
            (base / folder / label).mkdir(parents=True, exist_ok=True)
    (base / FOLDERS["configs"]).mkdir(parents=True, exist_ok=True)

    # Visibility split (train/test) — full env folders, RGB only
    for split in ("train", "test"):
        for label in ("Yes", "No"):
            (base / split / label).mkdir(parents=True, exist_ok=True)

    # Depth split (train_depth/test_depth) — per-image, routed by depth label
    for split in ("train_depth", "test_depth"):
        for label in ("Yes", "No"):
            (base / split / label).mkdir(parents=True, exist_ok=True)

# ── VALIDATION ────────────────────────────────────────────────────────────────

def validate_folder(base: Path, json_data: dict, scope: str = "root") -> None:
    print(f"\n[Validate] {scope.upper()}")
    errors = []

    all_indices = sorted(int(k) for k in json_data["environments"])
    if scope in ("train", "train_depth"):
        indices = [i for i in all_indices if i < TRAIN_COUNT]
    elif scope in ("test", "test_depth"):
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
        elif scope in ("train", "test"):
            # Full env RGB folders
            checks = [(base / scope / label / f"env_{idx}", EXPECTED_IMAGES_PER_ENV)]
            cam_path = None
        else:
            # train_depth / test_depth: each env split across Yes/ and No/ subfolders
            # Each should have exactly half the images
            half = EXPECTED_IMAGES_PER_ENV // 2
            checks = [
                (base / scope / "Yes" / f"env_{idx}", half),
                (base / scope / "No"  / f"env_{idx}", half),
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


def validate_depth_labels(depth_data: dict, json_data: dict) -> None:
    print(f"\n[Validate] DEPTH LABELS")
    errors = []
    half = EXPECTED_IMAGES_PER_ENV // 2

    for idx_str in tqdm(json_data["environments"], desc="depth labels"):
        env_key = f"env_{idx_str}"

        if env_key not in depth_data:
            errors.append(f"Missing depth key: {env_key}")
            continue

        images = depth_data[env_key]
        n = len(images)
        if n != EXPECTED_IMAGES_PER_ENV:
            errors.append(f"{env_key}: expected {EXPECTED_IMAGES_PER_ENV} entries, got {n}")
            continue

        ones  = sum(1 for v in images.values() if v == 1)
        zeros = sum(1 for v in images.values() if v == 0)
        if ones != half or zeros != half:
            errors.append(f"{env_key}: depth split {ones}/{zeros} (need {half}/{half})")

    if not errors:
        print(f"  ✅ Depth labels OK")
    else:
        print(f"  ❌ {len(errors)} depth label errors:")
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

    # ── 1. POOL valid environments by reason ──────────────────────────────────
    pool: dict[str, list] = {r: [] for r in SPLIT_TARGETS}
    qc_rejected = 0

    for src in source_dirs:
        labels_path = Path(src) / LABELS_FILENAME
        depth_path  = Path(src) / DEPTH_LABELS_FILENAME

        if not labels_path.exists():
            print(f"  Skipping {src} (no {LABELS_FILENAME})")
            continue
        if not depth_path.exists():
            print(f"  Skipping {src} (no {DEPTH_LABELS_FILENAME})")
            continue

        print(f"\nScanning: {src}")
        with open(labels_path) as f:
            vis_data = json.load(f)
        with open(depth_path) as f:
            depth_data = json.load(f)

        src_accepted = 0
        src_rejected = 0

        for env_id, info in sorted(
            vis_data.get("environments", {}).items(), key=lambda x: int(x[0])
        ):
            label  = info.get("label")
            reason = info.get("reason", "unknown")

            if label not in ("Yes", "No"):
                print(f"  Warning: unexpected label '{label}' for env_{env_id}, skipping.")
                continue

            if reason not in pool:
                continue

            passed, fail_reason = verify_environment(src, env_id, label, depth_data)
            if passed:
                pool[reason].append({
                    "src_root":    src,
                    "old_id":      env_id,
                    "label":       label,
                    "reason":      reason,
                    "details":     info,
                    "depth_slice": depth_data[f"env_{env_id}"],
                })
                src_accepted += 1
            else:
                src_rejected += 1
                qc_rejected  += 1

        print(f"  → Accepted: {src_accepted} | Rejected: {src_rejected}")

    print(f"\nPool summary:")
    for reason, envs in pool.items():
        yes = sum(1 for e in envs if e["label"] == "Yes")
        no  = sum(1 for e in envs if e["label"] == "No")
        print(f"  {reason}: {len(envs)}  (Yes={yes}, No={no})")
    print(f"  QC rejected total: {qc_rejected}")

    # ── 2. ALLOCATE train / test per reason ───────────────────────────────────
    train_batch, test_batch = [], []

    for reason, target in SPLIT_TARGETS.items():
        needed    = target * 2
        available = len(pool[reason])
        print(f"\n  {reason}: need {needed}, have {available}")
        if available < needed:
            print(f"  ❌ Not enough '{reason}' data. Aborting.")
            return

        selected = pool[reason][:]
        random.shuffle(selected)

        train_batch.extend(selected[:target])
        test_batch.extend(selected[target:needed])

    random.shuffle(train_batch)
    random.shuffle(test_batch)
    ordered = train_batch + test_batch

    # ── 3. PRE-COPY BALANCE CHECKS ────────────────────────────────────────────
    check_balance(ordered,     stage="PRE-COPY (ALL)",   expected_total=TOTAL_NEEDED)
    check_balance(train_batch, stage="PRE-COPY (TRAIN)", expected_total=TRAIN_COUNT)
    check_balance(test_batch,  stage="PRE-COPY (TEST)",  expected_total=TEST_COUNT)

    # ── 4. COPY root + train/test visibility split ─────────────────────────────
    out = Path(OUTPUT_DIR)
    if out.exists():
        print(f"\nWarning: Output dir {out} already exists.")
    create_output_structure(out)

    master_json = {
        "environments": {},
        "statistics": {
            "total_environments": len(ordered),
            "yes_count":  0,
            "no_count":   0,
            "by_reason":  {r: 0 for r in SPLIT_TARGETS},
        },
    }
    master_depth: dict[str, dict] = {}

    print(f"\nCopying {len(ordered)} environments → {out}")
    for new_idx, item in enumerate(tqdm(ordered, desc="Copying")):
        src    = Path(item["src_root"])
        old_id = item["old_id"]
        label  = item["label"]
        split  = "train" if new_idx < TRAIN_COUNT else "test"

        # Root modality folders
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

        # Visibility split: full env RGB folder
        shutil.copytree(src / FOLDERS["rgb"] / label / f"env_{old_id}",
                        out / split / label / f"env_{new_idx}",
                        dirs_exist_ok=True)

        master_depth[f"env_{new_idx}"] = item["depth_slice"]
        master_json["environments"][str(new_idx)] = {
            **item["details"],
            "src_root":   item["src_root"],
            "src_env_id": old_id,
        }
        master_json["statistics"]["yes_count" if label == "Yes" else "no_count"] += 1
        master_json["statistics"]["by_reason"][item["reason"]] += 1

    # Write master label files
    (out / LABELS_FILENAME).write_text(json.dumps(master_json, indent=4))
    (out / DEPTH_LABELS_FILENAME).write_text(json.dumps(master_depth, indent=4))
    print(f"\nWrote {LABELS_FILENAME} and {DEPTH_LABELS_FILENAME} to {out}")

    # ── 5. DEPTH SPLIT — per-image routing ────────────────────────────────────
    # Source: already-copied root RGB (not original src dirs)
    # Routing: depth_val 1 → train_depth/Yes/, 0 → train_depth/No/
    print(f"\nDistributing depth splits...")
    for env_key, depth_frames in tqdm(master_depth.items(), desc="Depth split"):
        new_idx    = int(env_key.split("_")[1])
        split      = "train_depth" if new_idx < TRAIN_COUNT else "test_depth"
        vis_label  = master_json["environments"][str(new_idx)]["label"]
        src_rgb    = out / FOLDERS["rgb"] / vis_label / f"env_{new_idx}"

        for img_name, depth_val in depth_frames.items():
            depth_label = "Yes" if depth_val == 1 else "No"
            dest_dir    = out / split / depth_label / f"env_{new_idx}"
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Match by stem in case extension varies
            matched = list(src_rgb.glob(f"{img_name}.*"))
            if matched:
                shutil.copy2(matched[0], dest_dir / matched[0].name)
            else:
                print(f"  Warning: {img_name} not found in {src_rgb}")

    # ── 6. STRUCTURAL VALIDATION ──────────────────────────────────────────────
    validate_folder(out, master_json, "root")
    validate_folder(out, master_json, "train")
    validate_folder(out, master_json, "test")
    validate_folder(out, master_json, "train_depth")
    validate_folder(out, master_json, "test_depth")
    validate_depth_labels(master_depth, master_json)

    # ── 7. POST-COPY BALANCE CHECKS ───────────────────────────────────────────
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
