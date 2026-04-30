#!/bin/bash

# ================= CONFIGURATION =================

# 1. Path to your existing script (the one you provided)
#    Assumes it is in the same directory. If not, provide full path.
INNER_SCRIPT="/home/arock3/cube_game/scripts/test_models_accel_args.sh"

# 2. Define your Data Directories here (in order)
data_dirs=(
    "/home/arock3/VPTnav_v15_low_var_small"
    "/home/arock3/VPTnav_v17_rl_low_high_small"
    "/home/arock3/VPTnav_v17_rl_high_low_small"
    "/home/arock3/VPTnav_v17_rl_high_v4_small"
)

# 3. Define your Output Directories here (must match the order above)
output_dirs=(
    "/home/arock3/Documents/Mono_VPT/v17/Low_Low"
    "/home/arock3/Documents/Mono_VPT/v17/Low_High"
    "/home/arock3/Documents/Mono_VPT/v17/High_Low"
    "/home/arock3/Documents/Mono_VPT/v17/High_High"
)

# ================= EXECUTION =================

# Check if script exists
if [ ! -f "$INNER_SCRIPT" ]; then
    echo "Error: Could not find $INNER_SCRIPT"
    exit 1
fi

# Ensure arrays are same length
if [ "${#data_dirs[@]}" -ne "${#output_dirs[@]}" ]; then
    echo "Error: Number of data directories (${#data_dirs[@]}) does not match number of output directories (${#output_dirs[@]})."
    exit 1
fi

# Make inner script executable just in case
chmod +x "$INNER_SCRIPT"

echo "=========================================="
echo "Starting Master Run: 4 Batches Scheduled"
echo "=========================================="

# Loop through the arrays by index
for i in "${!data_dirs[@]}"; do
    current_data="${data_dirs[$i]}"
    current_output="${output_dirs[$i]}"
    batch_num=$((i+1))

    echo ""
    echo "###################################################"
    echo "  BATCH $batch_num / ${#data_dirs[@]}"
    echo "  Data:   $current_data"
    echo "  Output: $current_output"
    echo "###################################################"
    echo ""

    # Call the inner script and wait for it to finish
    # We use "bash" to invoke it to ensure strictly new shell execution
    bash "$INNER_SCRIPT" "$current_data" "$current_output"

    # Check exit status of the run
    if [ $? -eq 0 ]; then
        echo ">> Batch $batch_num completed successfully."
    else
        echo ">> Batch $batch_num finished with errors (or was interrupted)."
        # Optional: Uncomment 'exit 1' below if you want the whole loop to stop on error
        # exit 1 
    fi

    # Small pause between large runs (optional, good for log readability)
    sleep 2
done

echo ""
echo "=========================================="
echo "All batches completed."
echo "=========================================="