import os
import re
import csv
import argparse
import glob
import numpy as np

def compile_results(results_dir, output_dir, num_runs=3):
    # output csv path
    csv_path = os.path.join(output_dir, "compiled_results.csv")
    
    # Dynamic Headers based on num_runs
    acc_headers = [f"Acc {i+1}" for i in range(num_runs)]
    headers = ["Model Name"] + acc_headers + ["Avg Acc"]
    
    # Find all .txt files in the results directory
    txt_files = glob.glob(os.path.join(results_dir, "*.txt"))
    txt_files.sort()
    
    rows = []
    
    # Storage for calculating column-wise averages
    # Create a list of lists to store values for each accuracy column dynamically
    column_storage = [[] for _ in range(num_runs)]
    all_avg_storage = []

    print(f"Found {len(txt_files)} log files in {results_dir}")
    print(f"Processing for {num_runs} runs per model...")

    for file_path in txt_files:
        # Extract model name from filename (remove .txt extension)
        filename = os.path.basename(file_path)
        model_name = os.path.splitext(filename)[0]
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Regex to find 'Best acc test <number>'
            pattern = r"Best acc test\s+([0-9.]+)"
            matches = re.findall(pattern, content)
            
            if not matches:
                print(f"\tWarning: No 'Best acc test' found in {filename}, skipping.")
                continue

            # Convert matches to floats and round to 3
            accuracies = [round(float(acc), 3) for acc in matches]
            
            # 1. Slice if we have more than requested
            accuracies = accuracies[:num_runs]
            
            # 2. Pad with None if we have fewer than requested
            while len(accuracies) < num_runs:
                accuracies.append(None)
            
            # Calculate Row Average (ignoring None)
            valid_accs = [a for a in accuracies if a is not None]
            avg_acc = round(np.mean(valid_accs), 3) if valid_accs else 0.0
            
            # Prepare row for CSV
            row = [model_name] + accuracies + [avg_acc]
            rows.append(row)
            
            # Store values for column averages (convert None to np.nan for math)
            for i in range(num_runs):
                val = accuracies[i] if accuracies[i] is not None else np.nan
                column_storage[i].append(val)
            
            all_avg_storage.append(avg_acc)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # --- Calculate Total Averages ---
    if rows:
        total_row_values = []
        
        # Calculate average for each 'Acc N' column
        for i in range(num_runs):
            col_avg = round(np.nanmean(column_storage[i]), 3)
            total_row_values.append(col_avg)
            
        # Calculate average of the averages
        total_avg_col = round(np.nanmean(all_avg_storage), 3)
        
        # Create the final summary row
        total_row = ["Total Average"] + total_row_values + [total_avg_col]
        rows.append(total_row)

    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"\nSuccessfully compiled results to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile results to CSV with Total Average.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing the .txt result files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs to collect and average (default: 3)")
    
    args = parser.parse_args()
    
    compile_results(args.results_dir, args.output_dir, args.num_runs)

    # results_dir = "/home/arock3/Documents/Depth/VPTnav_v17_rl_high_v4_depth_small"
    # output_dir = "/home/arock3/Documents/Depth/VPTnav_v17_rl_high_v4_depth_small"
    # num_runs = 3
    # compile_results(results_dir, output_dir, num_runs)