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
DEFAULT_DATASET_PATH = "/users/arock3/scratch/VPT1_v18"
DEFAULT_FOLDER_SEARCH = "/users/arock3/data/arock3/VPT/VPT_code/VPT/logs/linear_probe_preds/*_preds_lp.csv"
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
    clean_model_name = file_name.replace("_preds_lp", "")
    
    df = read_csv(csv_path)
    
    # Extract env_id from path
    try:
        df["env_id"] = df["path"].apply(lambda x: x.split("/")[-2]) 
    except IndexError:
        df["env_id"] = df["path"]

    env_accuracies = {}
    env_reasons = {}

    for env_id_key, info in env_labels.items():
        target_env_folder = f"env_{env_id_key}"
        env_df = df[df['env_id'] == target_env_folder]
        
        if env_df.empty:
            continue
            
        target_label = 1 if info["label"] == "Yes" else 0
        reason = info["reason"]
        
        preds = env_df["pred"].tolist()
        if not preds:
            continue

        is_prob = all(0 <= p <= 1 for p in preds)
        threshold = 0.5 if is_prob else 0.0
        
        correct_count = sum(1 for p in preds if (1 if p > threshold else 0) == target_label)
        acc = correct_count / len(preds)
        
        env_accuracies[target_env_folder] = acc
        env_reasons[target_env_folder] = reason

    outside_scores = []
    occluded_scores = []
    in_view_scores = []

    for env_name, acc in env_accuracies.items():
        reason_str = str(env_reasons[env_name]).lower()
        
        if "occluded" in reason_str:
            occluded_scores.append(acc)
        elif any(x in reason_str for x in ["outside", "out of view", "not visible"]):
            outside_scores.append(acc)
        elif any(x in reason_str for x in ["in_view", "visible", "in view"]):
            in_view_scores.append(acc)

    mean_outside = np.mean(outside_scores) if outside_scores else 0.0
    mean_occluded = np.mean(occluded_scores) if occluded_scores else 0.0
    mean_in_view = np.mean(in_view_scores) if in_view_scores else 0.0

    negatives = outside_scores + occluded_scores
    mean_negatives = np.mean(negatives) if negatives else 0.0
    
    if negatives and in_view_scores:
        final_score = (mean_negatives + mean_in_view) / 2.0
    elif in_view_scores:
        final_score = mean_in_view
    else:
        final_score = mean_negatives

    return {
        "model_name": clean_model_name,
        "outside_view": np.round(mean_outside, 3),
        "in_view": np.round(mean_in_view, 3),
        "occluded": np.round(mean_occluded, 3),
        "average": np.round(final_score, 3)
    }

def main(dataset_path, folder_search, imagenet_csv_path):
    # --- Infer Paths Automatically ---
    # 1. JSON is inside the dataset folder
    json_file_path = os.path.join(dataset_path, "visibility_labels.json")
    
    # 2. Output logs go to /home/arock3/logs/<dataset_name>
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    dest_path = os.path.join("/users/arock3/data/arock3/VPT/v18_logs/perspective/analysis", dataset_name)

    print(f"--- Configuration ---")
    print(f"Dataset Path:    {dataset_path}")
    print(f"Inferred JSON:   {json_file_path}")
    print(f"Inferred Dest:   {dest_path}")
    print(f"Search Pattern:  {folder_search}")
    print(f"---------------------")

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    print(f"Loading labels...")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        env_labels = data.get("environments", {})
    
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
            results.append(res)
        except Exception as e:
            print(f"ERROR processing {csv_path}: {e}")

    if results:
        summary_df = pd.DataFrame(results)
        
        if imagenet_df is not None:
            summary_df["imagenet_top1"] = summary_df["model_name"].map(
                lambda x: round(imagenet_df.loc[x, "top1"], 3) if x in imagenet_df.index else None
            )

        cols = ["model_name", "outside_view", "in_view", "occluded", "average", "imagenet_top1"]
        existing_cols = [c for c in cols if c in summary_df.columns]
        summary_df = summary_df[existing_cols]
        
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns.tolist()
        avg_values = summary_df[numeric_cols].mean().round(3).to_dict()
        avg_values["model_name"] = "AVERAGE"
        
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_values])], ignore_index=True)
        
        output_csv = os.path.join(dest_path, "analysis_summary.csv")
        summary_df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully saved analysis to {output_csv}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Linear Probe Prediction CSVs")

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