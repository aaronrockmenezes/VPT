#!/bin/bash

version=17_reload2
NUM_GPUS=6 # Set this to your desired number of GPUs

DATA_DIR="/oscar/scratch/arock3/VPTnav_data/v17_new/data_v${version}"
AGENT_SCRIPT="/oscar/scratch/arock3/VPTnav_code/cube_game/scripts/keyboard_agent.py"

# Update this path to wherever you save the Python launcher code
LAUNCHER_SCRIPT="/oscar/scratch/arock3/VPTnav_code/cube_game/scripts/launcher_2.py" 
ISAAC_SH="/workspace/isaaclab/isaaclab.sh"

if [ ! -d "$DATA_DIR" ]; then
    echo "Creating directory $DATA_DIR..."
    mkdir -p "$DATA_DIR"
fi

echo "Launching master Python script via single Apptainer instance..."
echo "--------------------------------------------------------------"

# 1. Define the command as an array for safe execution
CMD=(
    "$ISAAC_SH" "-p" "$LAUNCHER_SCRIPT"
    "--num_gpus" "$NUM_GPUS"
    "--data_dir" "$DATA_DIR"
    "--script_path" "$AGENT_SCRIPT"
)

# 2. Print the exact command it is about to run
echo "Executing:"
echo "${CMD[@]}"
echo "--------------------------------------------------------------"

# 3. Actually run the command
"${CMD[@]}"