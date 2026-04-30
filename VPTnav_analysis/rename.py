import os
import shutil
import numpy as np
import cv2
import json
from collections import Counter

def rename_envs(base_path: str, starting_idx: int):
    """
    Given a base path, go through No and Yes folders and rename the env_x folders to env_<starting_idx + x> 
    """

    for subfolder in ["No", "Yes"]:
        path = os.path.join(base_path, subfolder)
        if os.path.exists(path):
            env_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("env_")]
            for env_dir in env_dirs:
                try:
                    x = int(env_dir.split("_")[1])
                    new_name = f"env_{starting_idx + x}"
                    old_path = os.path.join(path, env_dir)
                    new_path = os.path.join(path, new_name)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print("Renamed:", old_path, "to", new_path)
                except (ValueError, IndexError):
                    print("Skipping invalid directory name:", env_dir)
                    pass

def find_largest_env_idx(base_path: str):
    """
    Given a base path, find the largest env_x index ammong No and Yes folders
    """
    max_idx = -1
    for subfolder in ["No", "Yes"]:
        path = os.path.join(base_path, subfolder)
        if os.path.exists(path):
            env_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("env_")]
            for env_dir in env_dirs:
                try:
                    x = int(env_dir.split("_")[1])
                    if x > max_idx:
                        max_idx = x
                except (ValueError, IndexError):
                    pass
    print("Largest env index found:", max_idx)
    return max_idx

def find_missing_indicies(base_path: str, start_idx: int = 0):
    """
    Given base path, go through No and Yes folders, collect all env_x indices and find missing ones. 
    Assume indices start from <starting_idx> and increase by 1.
    """
    existing_indices = set()
    for subfolder in ["No", "Yes"]:
        path = os.path.join(base_path, subfolder)
        if os.path.exists(path):
            env_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("env_")]
            for env_dir in env_dirs:
                try:
                    x = int(env_dir.split("_")[1])
                    existing_indices.add(x)
                except (ValueError, IndexError):
                    pass
    if existing_indices:
        max_idx = max(existing_indices)
        missing_indices = [i for i in range(start_idx, max_idx + 1) if i not in existing_indices]
    else:
        missing_indices = []
    print("Missing env indices:", missing_indices)
    original_missing_indices = [i - start_idx for i in missing_indices]
    print("Original Missing env indices:", original_missing_indices)
    return missing_indices

def check_occlusion(base_path: str):
    """
    Given base path, go through No and Yes folders, check occlusion for env_x and match with label folder. 
    """
    # Logic for occlusion check
    def is_occluded(img_path: str, return_red_count: bool = False) -> bool:
        """
        Return True if the red object is occluded in the semantic image.
        """
        GOAL_THRESHOLD = 100
        sem_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        sem_img = cv2.cvtColor(sem_img, cv2.COLOR_BGR2RGB)
        r = sem_img[:, :, 0]
        g = sem_img[:, :, 1]
        b = sem_img[:, :, 2]

        red_mask = ((r >= 0.95) & (g <= 0.05) & (b <= 0.05))
        red_count = red_mask.sum().item()

        if return_red_count:
            return red_count < GOAL_THRESHOLD, red_count
        return red_count < GOAL_THRESHOLD

    mismatches = []
    for subfolder in ["No", "Yes"]:
        path = os.path.join(base_path, subfolder)
        if os.path.exists(path):
            env_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("env_")]
            for env_dir in env_dirs:
                sem_img_path = os.path.join(path, env_dir, "cam_pov.png")
                if os.path.exists(sem_img_path):
                    occluded, red_count = is_occluded(sem_img_path, return_red_count=True)
                    expected = (subfolder == "No")  # Update: No folder means occluded
                    if occluded != expected:
                        mismatches.append(env_dir)
                        # print(f"Mismatch in {env_dir}: Expected {expected}, Found {occluded}. Red Count = {red_count}")
                else:
                    print(f"Semantic image not found for {env_dir}")
    return mismatches

def count_yes_no(base_path: str):
    """
    Return count of envs in Yes and No folders
    """
    counts = {"Yes": 0, "No": 0}
    for subfolder in ["No", "Yes"]:
        path = os.path.join(base_path, subfolder)
        if os.path.exists(path):
            env_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("env_")]
            counts[subfolder] = len(env_dirs)
    print("Counts:", counts)
    return counts

def get_visibility_counts(base_path: str) -> dict:
    """
    Scans base_path/RGB for physical env folders and counts their 
    corresponding reasons from base_path/visibility_labels.json.
    """
    json_path = os.path.join(base_path, "visibility_labels.json")
    rgb_path = os.path.join(base_path, "RGB")
    
    counts = {}
    
    if not os.path.exists(json_path):
        return counts

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Adjust based on your root structure, usually it's under "environments"
            env_labels = data.get("environments", {})
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return counts

    for subfolder in ["Yes", "No"]:
        path = os.path.join(rgb_path, subfolder)
        if not os.path.exists(path):
            continue
            
        env_dirs = [d for d in os.listdir(path) if d.startswith("env_")]
        
        for env_dir in env_dirs:
            try:
                # Extract index string (e.g. "0", "1")
                idx = str(int(env_dir.split("_")[1]))
                
                if idx in env_labels:
                    entry = env_labels[idx]
                    
                    # FIX: Handle if entry is a dictionary (e.g. {"reason": "occluded"})
                    if isinstance(entry, dict):
                        # Change "reason" to whatever key holds the text if this fails
                        reason = entry.get("reason", "missing_reason_key")
                    else:
                        # Handle if entry is just a string
                        reason = str(entry)
                        
                    counts[reason] = counts.get(reason, 0) + 1
                    
            except (ValueError, IndexError):
                continue
                
    return counts

# rename_envs("/home/arock3/Documents/data_v5_copy/RGB", 336)
# find_largest_env_idx("/home/arock3/Documents/data_v5_copy/RGB")
# find_missing_indicies("/home/arock3/vpt_v6_sanity/VPTnav_v6_sanity_2/test", 256)

# rename_envs("/home/arock3/Documents/data_2 copy/RGB", 436)
# find_largest_env_idx("/home/arock3/Documents/data_2 copy/RGB")
# find_missing_indicies("/home/arock3/Documents/data_2 copy/RGB", 436)

# rename_envs("/home/arock3/Documents/data_5 copy/RGB", 492)
# find_largest_env_idx("/home/arock3/Documents/data_5 copy/RGB")
# find_missing_indicies("/home/arock3/Documents/data_5 copy/RGB", 492)

# find_missing_indicies("/media/data_cifs_lrs/projects/prj_robotics/VPTnav_v6_600_envs/RGB")
# check_occlusion("/media/data_cifs_lrs/projects/prj_robotics/VPTnav_v6_100_envs/cam")

# rename_envs("/home/arock3/vpt_v6_data_compiling/VPTnav_v6/RGB", 487)
# find_largest_env_idx("/home/arock3/vpt_v6_data_compiling/VPTnav_v6_600_envs/RGB")
# find_missing_indicies("/home/arock3/vpt_v6_data_compiling/VPTnav_v6_600_envs/RGB", 0)


base_path1 = "/users/arock3/scratch/VPTnav_data/v17_new/data_v17_reload"
base_path2 = "/users/arock3/scratch/VPTnav_data/v17_new/data_v17_reload2"
base_path3 = "/users/arock3/scratch/VPTnav_data/v17_new/data_v17_reload3"
count_total = 0
count_yes, count_no = 0, 0
reason_counts = Counter()  # Initialize as a Counter
for j in [base_path1, base_path2, base_path3]:
    base_path = j
    for i in range(0, 5):
        print(f"--------------------------- Data {i} ---------------------------")
        counts = count_yes_no(os.path.join(base_path, f"data_{i}/RGB"))
        print(f"Data {i} num_envs {counts['Yes'] + counts['No']}")
        
        count_total += counts['Yes'] + counts['No']
        count_yes += counts['Yes']
        count_no += counts['No']
        
        reasons = get_visibility_counts(os.path.join(base_path, f"data_{i}"))
        # print(reasons)
        
        reason_counts.update(reasons)  # Automatically adds the values of matching keys
        # print(dict(reason_counts))     # Cast to dict for clean printing
        
        find_largest_env_idx(os.path.join(base_path, f"data_{i}/RGB"))
        missing_ids = find_missing_indicies(os.path.join(base_path, f"data_{i}/RGB"), 0)
        print(f"Data {i} missing {len(missing_ids)} ids\n")

print(f"Total num_envs across all datasets: {count_total}")
print(f"Total Yes: {count_yes}, Total No: {count_no}")
print("Reason counts:", dict(reason_counts))



# count_yes_no("/home/arock3/VPTnav_v12/test")
# find_missing_indicies("/home/arock3/VPTnav_v10/RGB", 0)
# find_largest_env_idx("/home/arock3/VPTnav_v10/RGB")
# count_yes_no("/home/arock3/VPTnav_v8/test")
# count_yes_no("/home/arock3/VPTnav_v8/train")

# mis = check_occlusion("/home/arock3/VPTnav_v12_LATEST/cam")
# print(len(mis))

# for i in range(0, 5):
#     mis = check_occlusion(f"/home/arock3/data_v12_new/data_{i}/cam")
#     print(len(mis))

