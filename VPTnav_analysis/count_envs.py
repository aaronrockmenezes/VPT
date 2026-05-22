import os
import glob

def count_left_right_envs(parent_dir: str, min_images: int):
    """
    Given a parent directory containing data_x folders, count the number of 
    valid 'No' and 'Yes' environments that contain at least 'min_images' .png files.
    """
    if not os.path.exists(parent_dir):
        print(f"Error: Parent directory '{parent_dir}' does not exist.")
        return

    # Initialize cumulative counters
    total_left = 0
    total_right = 0

    # Find all data_* directories in the parent directory and sort them
    data_dirs = sorted([d for d in os.listdir(parent_dir) 
                        if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith("data_")])

    if not data_dirs:
        print(f"No 'data_' directories found in {parent_dir}")
        return

    for data_dir in data_dirs:
        data_path = os.path.join(parent_dir, data_dir)
        rgb_path = os.path.join(data_path, "RGB")
        
        local_left_count = 0
        local_right_count = 0

        # Process both 'No' and 'Yes' folders
        for direction in ["No", "Yes"]:
            dir_path = os.path.join(rgb_path, direction)
            
            if os.path.exists(dir_path):
                # Find all env_* folders
                env_dirs = [e for e in os.listdir(dir_path) 
                            if os.path.isdir(os.path.join(dir_path, e)) and e.startswith("env_")]
                
                for env_dir in env_dirs:
                    env_path = os.path.join(dir_path, env_dir)
                    # Count .png images in the env folder
                    png_files = glob.glob(os.path.join(env_path, "*.png"))
                    
                    if len(png_files) >= min_images:
                        if direction == "No":
                            local_left_count += 1
                        else:
                            local_right_count += 1
                    else:
                        print(f"Warning: {data_dir}/RGB/{direction}/{env_dir} only has {len(png_files)} images (Expected >= {min_images}). Skipping.")

        # Add to cumulative totals
        total_left += local_left_count
        total_right += local_right_count

        # Print per-dataset report
        # print(f"--- {data_dir} ---")
        # print(f"  No envs:  {local_left_count}")
        # print(f"  Right envs: {local_right_count}")
        # print(f"  Total envs: {local_left_count + local_right_count}\n")

    # Print cumulative report
    print("========================================")
    print("FINAL CUMULATIVE COUNTS")
    print("========================================")
    print(f"Total No envs:  {total_left}")
    print(f"Total Right envs: {total_right}")
    print(f"Grand Total envs: {total_left + total_right}")
    print("========================================")

# Example usage:
PARENT_DIRECTORY = "/users/arock3/scratch/VPT1_DATA/camera/v18_camera/data"
EXPECTED_MIN_IMAGES = 10
count_left_right_envs(PARENT_DIRECTORY, EXPECTED_MIN_IMAGES)