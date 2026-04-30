import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
BASE_DIR = "/users/arock3/scratch/VPT2_DATA/v2/data"
OUTPUT_DIR = "/users/arock3/scratch/VPT2_v2"

EXPECTED_IMAGES_PER_ENV = 10

MULTIPLIER = 2**3  # Adjust as needed
TRAIN_COUNT = 32 * MULTIPLIER
TEST_COUNT = 32 * MULTIPLIER

FOLDERS = {
    "rgb": "RGB",
    "depth": "Depth",
    "cam": "cam",
    "configs": "configs"
}

LABELS_FILENAME = "visibility_labels.json"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
# =================================================


def check_balance(envs, stage: str, expected_total: int):
    """
    Asserts that `envs` has exactly expected_total entries and a perfect 50/50
    left/right split. Prints a summary and raises on failure so the pipeline
    halts immediately rather than silently producing a skewed dataset.

    Args:
        envs:           list of env dicts, each with a 'label' key ('left'/'right')
        stage:          label for print output, e.g. 'PRE-COPY' or 'POST-COPY'
        expected_total: TRAIN_COUNT + TEST_COUNT
    """
    total      = len(envs)
    left_count  = sum(1 for e in envs if e['label'] == 'left')
    right_count = sum(1 for e in envs if e['label'] == 'right')
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
        raise ValueError(f"Balance check failed at stage '{stage}': " + "; ".join(errors))

    print(f"  ✅ Perfect 50/50 split confirmed.")


def verify_environment(src_root, env_idx, label):
    """
    Checks if an environment:
    1. Has all required folders (rgb, depth, cam, config).
    2. Has EXACTLY 'EXPECTED_IMAGES_PER_ENV' images in the RGB folder.
    """
    p_rgb   = Path(src_root) / FOLDERS["rgb"]     / label / f"env_{env_idx}"
    p_depth = Path(src_root) / FOLDERS["depth"]   / label / f"env_{env_idx}"
    p_cam   = Path(src_root) / FOLDERS["cam"]     / label / f"env_{env_idx}"
    p_conf  = Path(src_root) / FOLDERS["configs"] / f"env_{env_idx}_config.json"

    # 1. Existence check
    if not (p_rgb.exists() and p_depth.exists() and p_cam.exists() and p_conf.exists()):
        return False

    # 2. Image count check (must be exactly EXPECTED_IMAGES_PER_ENV)
    try:
        rgb_files = [f for f in os.listdir(p_rgb) if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        if len(rgb_files) != EXPECTED_IMAGES_PER_ENV:
            return False
    except OSError:
        return False

    return True


def create_folder_structure(base_path):
    base = Path(base_path)
    for folder in [FOLDERS["rgb"], FOLDERS["depth"], FOLDERS["cam"]]:
        (base / folder / "left").mkdir(parents=True, exist_ok=True)
        (base / folder / "right").mkdir(parents=True, exist_ok=True)
    (base / FOLDERS["configs"]).mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        (base / split / "left").mkdir(parents=True, exist_ok=True)
        (base / split / "right").mkdir(parents=True, exist_ok=True)


def validate_folder(base_path, json_data, scope="root"):
    print(f"\n[INFO] Validating {scope.upper()} structure...")
    errors = []

    all_indices = sorted([int(k) for k in json_data["environments"].keys()])
    if scope == "train":
        target_indices = [i for i in all_indices if i < TRAIN_COUNT]
    elif scope == "test":
        target_indices = [i for i in all_indices if i >= TRAIN_COUNT]
    else:
        target_indices = all_indices

    for env_idx in tqdm(target_indices, desc=f"Checking {scope}"):
        info = json_data["environments"][str(env_idx)]
        label = info['label']

        paths_to_check = []
        if scope == "root":
            paths_to_check.append((base_path / FOLDERS["rgb"]   / label / f"env_{env_idx}", EXPECTED_IMAGES_PER_ENV))
            paths_to_check.append((base_path / FOLDERS["depth"] / label / f"env_{env_idx}", EXPECTED_IMAGES_PER_ENV))
            paths_to_check.append((base_path / FOLDERS["cam"]   / label / f"env_{env_idx}", 1))
        elif scope in ["train", "test"]:
            paths_to_check.append((base_path / scope / label / f"env_{env_idx}", EXPECTED_IMAGES_PER_ENV))

        for path, expected_count in paths_to_check:
            if not path.exists():
                errors.append(f"Missing: {path}")
                continue

            count = len([f for f in os.listdir(path) if Path(f).suffix.lower() in IMAGE_EXTENSIONS])
            if count != expected_count:
                errors.append(f"Count Error: {path} has {count} (Expected {expected_count})")

    if not errors:
        print(f"✅ {scope.upper()} VALID.")
    else:
        print(f"❌ {scope.upper()} ERRORS:")
        for e in errors[:5]:
            print(f"   - {e}")


def main():
    # Dynamically find all data_* subdirectories inside BASE_DIR
    source_dirs = [
        os.path.join(BASE_DIR, d) for d in sorted(os.listdir(BASE_DIR)) 
        if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("data_")
    ]
    
    if not source_dirs:
        print(f"❌ No valid 'data_' directories found in {BASE_DIR}")
        return

    print(f"--- Scanning {len(source_dirs)} directories for {LABELS_FILENAME} ---")
    valid_pool = []

    for src in source_dirs:
        labels_path = Path(src) / LABELS_FILENAME

        if not labels_path.exists():
            print(f"Skipping {src} (No labels file)")
            continue

        with open(labels_path, 'r') as f:
            data = json.load(f)

        sorted_envs = sorted(data.get("environments", {}).items(), key=lambda x: int(x[0]))

        for env_id, info in sorted_envs:
            label = info['label']

            if label not in ("left", "right"):
                print(f"Warning: Unexpected label '{label}' for env_{env_id} in {src}, skipping.")
                continue

            if verify_environment(src, env_id, label):
                valid_pool.append({
                    "src_root": src,
                    "old_id": env_id,
                    "label": label,
                    "details": info
                })

    print(f"\nTotal valid environments found: {len(valid_pool)}")

    needed = TRAIN_COUNT + TEST_COUNT
    if len(valid_pool) < needed:
        print(f"❌ CRITICAL ERROR: Need {needed} environments, but only {len(valid_pool)} are valid.")
        return

    # --- SHUFFLE AND SPLIT ---
    # Enforce 50/50 *within each split* by drawing left/right halves independently
    # for train and test. A global shuffle + slice is NOT sufficient — the split
    # boundary can land anywhere and produce e.g. 130 left / 126 right in test.
    half_train = TRAIN_COUNT // 2  # left envs needed for train
    half_test  = TEST_COUNT  // 2  # left envs needed for test
    half_total = half_train + half_test

    left_envs  = [e for e in valid_pool if e['label'] == 'left']
    right_envs = [e for e in valid_pool if e['label'] == 'right']

    random.shuffle(left_envs)
    random.shuffle(right_envs)

    if len(left_envs) < half_total:
        print(f"❌ CRITICAL ERROR: Need {half_total} left envs, only {len(left_envs)} available.")
        return
    if len(right_envs) < half_total:
        print(f"❌ CRITICAL ERROR: Need {half_total} right envs, only {len(right_envs)} available.")
        return

    # Build each split with guaranteed exact half/half, then shuffle within
    train_batch = left_envs[:half_train] + right_envs[:half_train]
    test_batch  = left_envs[half_train:half_total] + right_envs[half_train:half_total]

    random.shuffle(train_batch)
    random.shuffle(test_batch)

    ordered_envs = train_batch + test_batch

    # --- PRE-COPY BALANCE CHECK (overall + per-split) ---
    check_balance(ordered_envs,  stage="PRE-COPY (ALL)",   expected_total=TRAIN_COUNT + TEST_COUNT)
    check_balance(train_batch,   stage="PRE-COPY (TRAIN)", expected_total=TRAIN_COUNT)
    check_balance(test_batch,    stage="PRE-COPY (TEST)",  expected_total=TEST_COUNT)

    # --- OUTPUT ---
    output_base = Path(OUTPUT_DIR)
    if output_base.exists():
        print(f"Warning: Output dir {output_base} already exists.")

    create_folder_structure(output_base)

    master_json = {
        "environments": {},
        "statistics": {
            "total_environments": len(ordered_envs),
            "left_count": 0,
            "right_count": 0,
        }
    }

    print(f"\n--- Processing {len(ordered_envs)} Environments ---")

    for new_idx, item in enumerate(tqdm(ordered_envs, desc="Copying")):
        src_root = Path(item['src_root'])
        old_id   = item['old_id']
        label    = item['label']

        # A. Copy to root folders
        shutil.copytree(src_root / FOLDERS["rgb"]   / label / f"env_{old_id}", output_base / FOLDERS["rgb"]   / label / f"env_{new_idx}", dirs_exist_ok=True)
        shutil.copytree(src_root / FOLDERS["depth"] / label / f"env_{old_id}", output_base / FOLDERS["depth"] / label / f"env_{new_idx}", dirs_exist_ok=True)
        shutil.copytree(src_root / FOLDERS["cam"]   / label / f"env_{old_id}", output_base / FOLDERS["cam"]   / label / f"env_{new_idx}", dirs_exist_ok=True)
        shutil.copy2(src_root / FOLDERS["configs"] / f"env_{old_id}_config.json", output_base / FOLDERS["configs"] / f"env_{new_idx}_config.json")

        # B. Copy RGB to split folder
        split_folder = "train" if new_idx < TRAIN_COUNT else "test"
        shutil.copytree(src_root / FOLDERS["rgb"] / label / f"env_{old_id}", output_base / split_folder / label / f"env_{new_idx}", dirs_exist_ok=True)

        # C. Update JSON
        master_json["environments"][str(new_idx)] = item['details']
        if label == "left":
            master_json["statistics"]["left_count"] += 1
        else:
            master_json["statistics"]["right_count"] += 1

    with open(output_base / LABELS_FILENAME, "w") as f:
        json.dump(master_json, f, indent=4)

    # --- STRUCTURAL VALIDATION ---
    print("\n--- Starting Validation ---")
    validate_folder(output_base, master_json, scope="root")
    validate_folder(output_base, master_json, scope="train")
    validate_folder(output_base, master_json, scope="test")

    # --- POST-COPY BALANCE CHECK (overall + per-split) ---
    all_written   = [{"label": v["label"]} for v in master_json["environments"].values()]
    train_written = [{"label": v["label"]} for k, v in master_json["environments"].items() if int(k) < TRAIN_COUNT]
    test_written  = [{"label": v["label"]} for k, v in master_json["environments"].items() if int(k) >= TRAIN_COUNT]

    check_balance(all_written,   stage="POST-COPY (ALL)",   expected_total=TRAIN_COUNT + TEST_COUNT)
    check_balance(train_written, stage="POST-COPY (TRAIN)", expected_total=TRAIN_COUNT)
    check_balance(test_written,  stage="POST-COPY (TEST)",  expected_total=TEST_COUNT)

    print("\n✅ JOB DONE.")


if __name__ == "__main__":
    main()