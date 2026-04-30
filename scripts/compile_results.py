import os
import csv
import json
import glob
import argparse
import numpy as np


def compile_results(results_dir, output_dir, num_runs=3):
    """
    Read per-model JSON files from results_dir and compile into a single CSV.

    Expected JSON format (produced by run_accel_latest.py):
    {
        "model": "resnet50.a1_in1k",
        "task": "perspective",
        "runs": [
            {"run": 1, "train_acc": 0.892, "test_acc": 0.834},
            {"run": 2, "train_acc": 0.901, "test_acc": 0.841},
            ...
        ],
        "avg_test_acc": 0.835,
        ...
    }
    """
    csv_path = os.path.join(output_dir, "compiled_results.csv")

    # Dynamic headers based on num_runs
    acc_headers = [f"Acc {i+1}" for i in range(num_runs)]
    headers = ["Model Name"] + acc_headers + ["Avg Acc"]

    # Find all JSON files (exclude anything that isn't a per-model result)
    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))

    print(f"Found {len(json_files)} JSON result files in {results_dir}")
    print(f"Expecting {num_runs} runs per model")

    rows = []
    column_storage = [[] for _ in range(num_runs)]
    all_avg_storage = []

    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            model_name = data.get('model', os.path.splitext(os.path.basename(json_path))[0])
            runs = data.get('runs', [])

            if not runs:
                print(f"\tWarning: No runs found in {os.path.basename(json_path)}, skipping.")
                continue

            # Extract test accuracies per run
            accuracies = [round(r['test_acc'], 3) for r in runs]

            # Slice if more than requested
            accuracies = accuracies[:num_runs]

            # Pad with None if fewer than requested
            while len(accuracies) < num_runs:
                accuracies.append(None)

            # Row average (ignoring None)
            valid_accs = [a for a in accuracies if a is not None]
            avg_acc = round(np.mean(valid_accs), 3) if valid_accs else 0.0

            row = [model_name] + accuracies + [avg_acc]
            rows.append(row)

            # Store for column averages
            for i in range(num_runs):
                val = accuracies[i] if accuracies[i] is not None else np.nan
                column_storage[i].append(val)

            all_avg_storage.append(avg_acc)

        except Exception as e:
            print(f"Error processing {os.path.basename(json_path)}: {e}")

    # Total average row
    if rows:
        total_row_values = []
        for i in range(num_runs):
            col_avg = round(np.nanmean(column_storage[i]), 3)
            total_row_values.append(col_avg)

        total_avg_col = round(np.nanmean(all_avg_storage), 3)
        total_row = ["Total Average"] + total_row_values + [total_avg_col]
        rows.append(total_row)

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"\nSuccessfully compiled {len(rows) - 1} models to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile per-model JSON results into a summary CSV.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing per-model .json files (output_dir/results/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save compiled_results.csv")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs to collect and average (default: 3)")

    args = parser.parse_args()
    compile_results(args.results_dir, args.output_dir, args.num_runs)