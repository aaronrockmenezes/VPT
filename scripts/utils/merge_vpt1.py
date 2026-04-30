import os
import json
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_DIRS = [
    "/users/arock3/scratch/VPT1_DATA/v17_5/data/data_node0_gpu4",
    "/users/arock3/scratch/VPT1_DATA/v17_5/data/data_node0_gpu5",
    "/users/arock3/scratch/VPT1_DATA/v17_5/data/data_node0_gpu6",
    "/users/arock3/scratch/VPT1_DATA/v17_5/data/data_node0_gpu7",
]

OUTPUT_DIR = "/users/arock3/scratch/VPT_v17_v5"
EXPECTED_IMAGES_PER_ENV = 10 

MULTIPLIER = 2**3  # Adjust as needed
# TRAIN_COUNT = 128*MULTIPLIER
TRAIN_COUNT = 32*MULTIPLIER
TEST_COUNT = 32*MULTIPLIER

SPLIT_TARGETS = {
    "in_view": 16*MULTIPLIER,
    "outside_fov": 8*MULTIPLIER,
    "occluded": 8*MULTIPLIER
}

FOLDERS = {
    "rgb": "RGB",
    "depth": "Depth",
    "cam": "cam",
    "configs": "configs"
}

LABELS_FILENAME = "visibility_labels.json"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
# =================================================

def check_occlusion(base_path: str):
    """
    Scans 'Yes' and 'No' folders.
    
    1. GLOBAL REJECT: If ANY Green is found (using HSV), mismatch immediately.
    2. 'No' Folder: Mismatch if ANY Red (Strict 0.95) OR ANY Circle (Hough) is found.
    3. 'Yes' Folder: Mismatch if Red (Strict 0.95) is missing.
    """

    def check_3_is_green_present(img_bgr, trim=0) -> bool:
        """
        Returns True if ANY Green pixels are found using HSV.
        Args:
            trim: Number of pixels to crop from each side (default 0).
        """
        if img_bgr is None: return False
        
        # Crop if trim is requested
        if trim > 0:
            h, w = img_bgr.shape[:2]
            # Safety check: ensure image is larger than the crop amount
            if h <= trim * 2 or w <= trim * 2:
                return False
            img_bgr = img_bgr[trim:-trim, trim:-trim]
        
        # Convert to HSV (Hue, Saturation, Value)
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Green range (covers your Hue ~45)
        lower_green = np.array([35, 50, 50]) 
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv_img, lower_green, upper_green)
        
        return cv2.countNonZero(mask) > 5

    def check_1_is_red_present(img_bgr) -> bool:
        """
        Returns True if Red pixels count > 100.
        Uses Strict 0.95 logic on normalized image.
        """
        if img_bgr is None: return False
        
        GOAL_THRESHOLD = 500
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 float
        if img_rgb.max() > 1.0:
            img_norm = img_rgb.astype(np.float32) / 255.0
        else:
            img_norm = img_rgb.astype(np.float32)
        
        r = img_norm[:, :, 0]
        g = img_norm[:, :, 1]
        b = img_norm[:, :, 2]

        # Strict 0.95 / 0.05 Logic
        red_mask = ((r >= 0.95) & (g <= 0.05) & (b <= 0.05))
        
        return red_mask.sum().item() > GOAL_THRESHOLD

    def check_2_is_circle_present(img_bgr) -> bool:
        """
        Returns True if a circular object is found using Fill Ratio.
        Method: Area of Object / Area of Min Enclosing Circle.
        Circle ~= 0.9 - 1.0
        Square ~= 0.64
        Threshold > 0.8 is extremely safe.
        """
        if img_bgr is None: return False
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get all shapes
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Use CHAIN_APPROX_NONE for more accurate area calculation
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Ignore small noise
            if area < 50: 
                continue
            
            # Find Minimum Enclosing Circle
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius ** 2)
            
            if circle_area == 0: continue
            
            # Calculate Fill Ratio
            fill_ratio = area / circle_area
            
            # 0.80 allows for significant pixelation/distortion while rejecting squares (0.64)
            if fill_ratio > 0.80:
                return True
                
        return False

    mismatches = []
    
    for subfolder in ["No", "Yes"]:
        path = os.path.join(base_path, subfolder)
        if not os.path.exists(path):
            continue

        env_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("env_")]
        
        for env_dir in env_dirs:
            sem_img_path = os.path.join(path, env_dir, "cam_pov.png")
            img_bgr = cv2.imread(sem_img_path, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                print(f"Warning: Image not found: {sem_img_path}")
                mismatches.append(env_dir)
                continue

            # --- CHECK 1: GREEN GATEKEEPER ---
            if check_3_is_green_present(img_bgr):
                mismatches.append(env_dir)
                continue 

            # --- CHECK 2 & 3: LOGIC CHECKS ---
            red_exists = check_1_is_red_present(img_bgr)
            
            is_mismatch = False

            if subfolder == "No":
                # FOLDER NO: Strict Empty Check
                if red_exists:
                    is_mismatch = True
                elif check_2_is_circle_present(img_bgr):
                    is_mismatch = True
                    print(f"Mismatch in 'No' - {env_dir}: Found Unlabeled Circle")

            elif subfolder == "Yes":
                # FOLDER YES: Strict Red Check
                if not red_exists:
                    is_mismatch = True

            if is_mismatch:
                mismatches.append(env_dir)

    return mismatches

def verify_environment(src_root, env_idx, label):
    """
    Checks if an environment:
    1. Has all required folders (rgb, depth, cam, config).
    2. Has EXACTLY 'EXPECTED_IMAGES_PER_ENV' images in the RGB folder.
    """
    p_rgb = Path(src_root) / FOLDERS["rgb"] / label / f"env_{env_idx}"
    p_depth = Path(src_root) / FOLDERS["depth"] / label / f"env_{env_idx}"
    p_cam = Path(src_root) / FOLDERS["cam"] / label / f"env_{env_idx}"
    p_conf = Path(src_root) / FOLDERS["configs"] / f"env_{env_idx}_config.json"
    
    # 1. Existence Check
    if not (p_rgb.exists() and p_depth.exists() and p_cam.exists() and p_conf.exists()):
        return False
        
    # 2. Count Check (Must be exactly EXPECTED_IMAGES_PER_ENV)
    try:
        rgb_files = [f for f in os.listdir(p_rgb) if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        if len(rgb_files) != EXPECTED_IMAGES_PER_ENV:
            print(f"Skipping {env_idx}: Image count {len(rgb_files)} != {EXPECTED_IMAGES_PER_ENV}")
            return False
    except OSError:
        return False
        
    return True

def create_folder_structure(base_path):
    base = Path(base_path)
    # Root folders
    for folder in [FOLDERS["rgb"], FOLDERS["depth"], FOLDERS["cam"]]:
        (base / folder / "Yes").mkdir(parents=True, exist_ok=True)
        (base / folder / "No").mkdir(parents=True, exist_ok=True)
    (base / FOLDERS["configs"]).mkdir(parents=True, exist_ok=True)

    # Split folders
    for split in ["train", "test"]:
        (base / split / "Yes").mkdir(parents=True, exist_ok=True)
        (base / split / "No").mkdir(parents=True, exist_ok=True)

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
            paths_to_check.append((base_path / FOLDERS["rgb"] / label / f"env_{env_idx}", EXPECTED_IMAGES_PER_ENV))
            paths_to_check.append((base_path / FOLDERS["depth"] / label / f"env_{env_idx}", EXPECTED_IMAGES_PER_ENV))
            paths_to_check.append((base_path / FOLDERS["cam"] / label / f"env_{env_idx}", 1)) 
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
        for e in errors[:5]: print(f"   - {e}")

def main():
    print(f"--- Scanning for {LABELS_FILENAME} ---")
    valid_pool = {"in_view": [], "outside_fov": [], "occluded": []}
    
    for src in SOURCE_DIRS:
        labels_path = Path(src) / LABELS_FILENAME
        cam_root = Path(src) / FOLDERS["cam"]
        
        if not labels_path.exists():
            print(f"Skipping {src} (No labels file)")
            continue

        # --- RUN OCCLUSION CHECK FOR THIS SOURCE ---
        print(f"Checking occlusion integrity for {src}...")
        mismatched_envs = check_occlusion(str(cam_root))
        
        # Convert ["env_1", "env_2"] -> set("1", "2") for fast lookup
        mismatched_ids = set()
        for m in mismatched_envs:
            try:
                # Extracts '1' from 'env_1'
                mid = m.split('_')[1]
                mismatched_ids.add(mid)
            except IndexError:
                pass
        
        if mismatched_ids:
            print(f"   -> Found {len(mismatched_ids)} mismatches. These will be ignored.")

        # --- LOAD JSON AND FILTER ---
        with open(labels_path, 'r') as f:
            data = json.load(f)
        
        sorted_envs = sorted(data.get("environments", {}).items(), key=lambda x: int(x[0]))

        for env_id, info in sorted_envs:
            # FILTER 1: Skip if this ID was flagged as a mismatch
            if env_id in mismatched_ids:
                continue

            reason = info['reason']
            label = info['label']
            
            # FILTER 2: Verify folder structure AND image count (must be 20)
            if reason in valid_pool and verify_environment(src, env_id, label):
                valid_pool[reason].append({
                    "src_root": src,
                    "old_id": env_id,
                    "label": label,
                    "reason": reason,
                    "details": info
                })

    print("\n--- Allocating Data ---")
    train_batch = []
    test_batch = []
    
    # Gather data based on strict quotas
    for reason in ["in_view", "outside_fov", "occluded"]:
        target = SPLIT_TARGETS[reason]
        needed = target * 2
        available = len(valid_pool[reason])
        print(f"Category '{reason}': Need {needed}, Have {available}")
        
        if available < needed:
            print(f"❌ CRITICAL ERROR: Not enough '{reason}' data after filtering.")
            return
        
        subset = valid_pool[reason][:needed]
        train_batch.extend(subset[:target])
        test_batch.extend(subset[target:])
        
    # --- SHUFFLE LOGIC ---
    print("🔀 Shuffling Train and Test batches...")
    random.shuffle(train_batch) 
    random.shuffle(test_batch)
    
    # Combine: Indices 0-255 will be the mixed Train set, 256-511 will be mixed Test set
    ordered_envs = train_batch + test_batch

    # Process
    output_base = Path(OUTPUT_DIR)
    if output_base.exists():
        print(f"Warning: Output dir {output_base} already exists.")
        
    create_folder_structure(output_base)
    
    master_json = {
        "environments": {},
        "statistics": {
            "total_environments": len(ordered_envs),
            "yes_count": 0, "no_count": 0,
            "by_reason": {k: 0 for k in SPLIT_TARGETS.keys()}
        }
    }

    print(f"\n--- Processing {len(ordered_envs)} Environments ---")
    
    for new_idx, item in enumerate(tqdm(ordered_envs, desc="Copying")):
        src_root = Path(item['src_root'])
        old_id = item['old_id']
        label = item['label']
        
        # A. COPY TO ROOT
        shutil.copytree(src_root / FOLDERS["rgb"] / label / f"env_{old_id}", output_base / FOLDERS["rgb"] / label / f"env_{new_idx}", dirs_exist_ok=True)
        shutil.copytree(src_root / FOLDERS["depth"] / label / f"env_{old_id}", output_base / FOLDERS["depth"] / label / f"env_{new_idx}", dirs_exist_ok=True)
        shutil.copytree(src_root / FOLDERS["cam"] / label / f"env_{old_id}", output_base / FOLDERS["cam"] / label / f"env_{new_idx}", dirs_exist_ok=True)
        shutil.copy2(src_root / FOLDERS["configs"] / f"env_{old_id}_config.json", output_base / FOLDERS["configs"] / f"env_{new_idx}_config.json")

        # B. COPY TO SPLIT
        split_folder = "train" if new_idx < TRAIN_COUNT else "test"
        shutil.copytree(src_root / FOLDERS["rgb"] / label / f"env_{old_id}", output_base / split_folder / label / f"env_{new_idx}", dirs_exist_ok=True)

        # C. UPDATE JSON
        master_json["environments"][str(new_idx)] = item['details']
        master_json["statistics"]["by_reason"][item['reason']] += 1
        if label == "Yes":
            master_json["statistics"]["yes_count"] += 1
        else:
            master_json["statistics"]["no_count"] += 1

    with open(output_base / LABELS_FILENAME, "w") as f:
        json.dump(master_json, f, indent=4)

    # Validation
    print("\n--- Starting Validation ---")
    validate_folder(output_base, master_json, scope="root")
    validate_folder(output_base, master_json, scope="train")
    validate_folder(output_base, master_json, scope="test")
    print("\n✅ JOB DONE.")

if __name__ == "__main__":
    main()