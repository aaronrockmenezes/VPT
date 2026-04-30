import pandas as pd
import numpy as np
import json
import os
import glob
import shutil
import argparse
from typing import Dict

# ==========================================
# DEFAULT CONFIGURATION
# ==========================================
DEFAULT_DATASET_PATH = "/users/arock3/scratch/VPT_v17_v4_FULL/VPTnav_v17_high_depth_clip"
DEFAULT_FOLDER_SEARCH = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_robotics/VPT/logs/linear_probe_preds/*_depth_lp.csv"
DEFAULT_IMAGENET_CSV_PATH = "/users/arock3/data/arock3/VPT/results-imagenet.csv"
# ==========================================

def filter_csv_files(folder_pattern: str, dataset_subset_string: str) -> list:
    """Filter CSV files where the path column contains the dataset_subset_string."""
    filtered_files = []
    files = glob.glob(folder_pattern)
    
    print(f"Found {len(files)} files matching pattern. Filtering for content...")
    
    for csv_file in files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty and "path" in df.columns:
                if df["path"].astype(str).str.contains(dataset_subset_string, regex=False).any():
                    filtered_files.append(csv_file)
        except Exception as e:
            pass
    return filtered_files

def read_csv(csv_file_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_file_path)

def analyze_csv(csv_path: str, env_labels: dict) -> Dict:
    file_name = os.path.splitext(os.path.basename(csv_path))[0]
    # Clean up the model name (handling both perspective and depth naming conventions just in case)
    clean_model_name = file_name.replace("_preds_depth_lp", "").replace("_preds_lp", "")
    
    df = read_csv(csv_path)
    
    # Extract env_id and image name from the path
    # e.g., .../env_0/image_0000.png -> env_id: env_0, img_name: image_0000
    try:
        df["env_id"] = df["path"].apply(lambda x: x.split("/")[-2]) 
        df["img_name"] = df["path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    except IndexError:
        print(f"Warning: Unexpected path format in {csv_path}")
        return {}

    pos_correct, pos_total = 0, 0
    neg_correct, neg_total = 0, 0

    for _, row in df.iterrows():
        env_id = row["env_id"]
        img_name = row["img_name"]
        
        # Ensure env_id matches the JSON key format (e.g., "env_0")
        env_key = f"env_{env_id}" if str(env_id).isdigit() else env_id

        # Check if the specific image exists in our ground truth JSON
        if env_key in env_labels and img_name in env_labels[env_key]:
            target_label = env_labels[env_key][img_name]
            pred = row["pred"]
            
            is_prob = (0 <= pred <= 1)
            threshold = 0.5 if is_prob else 0.0
            pred_binary = 1 if pred > threshold else 0
            
            if target_label == 1:
                pos_total += 1
                if pred_binary == 1:
                    pos_correct += 1
            else:
                neg_total += 1
                if pred_binary == 0:
                    neg_correct += 1

    overall_total = pos_total + neg_total
    overall_acc = (pos_correct + neg_correct) / overall_total if overall_total > 0 else 0.0
    pos_acc = pos_correct / pos_total if pos_total > 0 else 0.0
    neg_acc = neg_correct / neg_total if neg_total > 0 else 0.0

    return {
        "model_name": clean_model_name,
        "target_depth_acc": np.round(pos_acc, 3),
        "background_depth_acc": np.round(neg_acc, 3),
        "average": np.round(overall_acc, 3)
    }

def main(dataset_path, folder_search, imagenet_csv_path):
    # --- Infer Paths Automatically ---
    # Load the new merged depth labels JSON
    json_file_path = os.path.join(dataset_path, "merged_depth_labels.json")
    
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    # dest_path = os.path.join("/home/arock3/Documents/Depth/MonoVPT_Depth/logs", dataset_name)
    dest_path = os.path.join("/users/arock3/data/arock3/VPT/v17_v4_logs/depth/analysis", dataset_name)

    print(f"--- Configuration ---")
    print(f"Dataset Path:    {dataset_path}")
    print(f"Inferred JSON:   {json_file_path}")
    print(f"Inferred Dest:   {dest_path}")
    print(f"Search Pattern:  {folder_search}")
    print(f"---------------------")

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    print(f"Loading depth labels...")
    with open(json_file_path, 'r') as f:
        env_labels = json.load(f)
    
    print(f"Filtering CSVs...")
    valid_csvs = filter_csv_files(folder_search, dataset_path)
    
    if not valid_csvs:
        print("No valid CSV files found matching criteria.")
        return

    os.makedirs(dest_path, exist_ok=True)

    print(f"Copying {len(valid_csvs)} filtered CSV files to {dest_path}")
    for csv_file in valid_csvs:
        try:
            shutil.copy(csv_file, dest_path)
        except Exception as e:
            print(f"Error copying file {csv_file}: {e}")

    imagenet_df = None
    if imagenet_csv_path and os.path.exists(imagenet_csv_path):
        try:
            imagenet_df = read_csv(imagenet_csv_path)
            if "top1" in imagenet_df.columns and "model" in imagenet_df.columns:
                imagenet_df = imagenet_df.sort_values(by="top1", ascending=False)
                imagenet_df = imagenet_df.drop_duplicates(subset=["model"], keep="first")
            imagenet_df = imagenet_df.set_index("model")
        except Exception as e:
             print(f"Warning: Could not process ImageNet CSV: {e}")

    print("Analyzing files...")
    results = []
    for csv_path in valid_csvs:
        try:
            res = analyze_csv(csv_path, env_labels)
            if res:  # Only append if valid results were returned
                results.append(res)
        except Exception as e:
            print(f"ERROR processing {csv_path}: {e}")

    if results:
        summary_df = pd.DataFrame(results)
        
        if imagenet_df is not None:
            summary_df["imagenet_top1"] = summary_df["model_name"].map(
                lambda x: round(imagenet_df.loc[x, "top1"], 3) if x in imagenet_df.index else None
            )

        # Updated columns for the depth task
        cols = ["model_name", "target_depth_acc", "background_depth_acc", "average", "imagenet_top1"]
        existing_cols = [c for c in cols if c in summary_df.columns]
        summary_df = summary_df[existing_cols]
        
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns.tolist()
        avg_values = summary_df[numeric_cols].mean().round(3).to_dict()
        avg_values["model_name"] = "AVERAGE"
        
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_values])], ignore_index=True)
        
        output_csv = os.path.join(dest_path, "analysis_depth_summary.csv")
        summary_df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully saved depth analysis to {output_csv}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Linear Probe Depth Prediction CSVs")

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH,
                        help="Root path of the dataset (used to find JSON and naming output dir)")
    
    parser.add_argument("--folder_search", type=str, default=DEFAULT_FOLDER_SEARCH,
                        help="Glob pattern to find CSV files")
    
    parser.add_argument("--imagenet_csv", type=str, default=DEFAULT_IMAGENET_CSV_PATH,
                        help="Path to ImageNet results CSV (optional)")

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        folder_search=args.folder_search,
        imagenet_csv_path=args.imagenet_csv
    )