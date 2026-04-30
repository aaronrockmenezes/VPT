#!/bin/bash

NODE_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATA_DIR="${BASE_PATH}/data"

AGENT_SCRIPT="/mnt/VPT/VPTnav_code/cube_game/scripts/keyboard_agent.py"
# AGENT_SCRIPT="/mnt/VPT/VPTnav_code/cube_game/scripts/vpt2_keyboard_agent.py"
LAUNCHER_SCRIPT="/mnt/VPT/VPTnav_code/cube_game/job_array/launcher.py"
ISAAC_SH="/workspace/isaaclab/isaaclab.sh"

mkdir -p "$DATA_DIR"

echo "Launching master Python script via single Apptainer instance..."
echo "Node ID: $NODE_ID | GPUs: $NUM_GPUS | Task: $TASK | Data Dir: $DATA_DIR"
echo "--------------------------------------------------------------"

CMD=(
    "$ISAAC_SH" "-p" "$LAUNCHER_SCRIPT"
    "--num_gpus" "$NUM_GPUS"
    "--data_dir" "$DATA_DIR"
    "--script_path" "$AGENT_SCRIPT"
    "--task" "$TASK"
)

echo "Executing:"
echo "${CMD[@]}"
echo "--------------------------------------------------------------"

"${CMD[@]}"