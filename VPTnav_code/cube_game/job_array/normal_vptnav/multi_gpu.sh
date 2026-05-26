#!/bin/bash

# Prefer composite NODE_ID from worker ({job}_{task}); fall back to bare task id.
NODE_ID="${NODE_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
DATA_DIR="${BASE_PATH}/data"

AGENT_SCRIPT="${AGENT_SCRIPT:-/mnt/VPT/VPTnav_code/cube_game/scripts/vptnav/keyboard_agent.py}"
# override via env: AGENT_SCRIPT=/mnt/VPT/.../camera_move_collector.py
LAUNCHER_SCRIPT="/mnt/VPT/VPTnav_code/cube_game/job_array/normal_vptnav/launcher.py"
ISAAC_SH="/workspace/isaaclab/isaaclab.sh"

mkdir -p "$DATA_DIR"

echo "Launching master Python script via single Apptainer instance..."
echo "Node ID: $NODE_ID | GPUs: $NUM_GPUS | Task: $TASK | Data Dir: $DATA_DIR"
echo "Agent script: $AGENT_SCRIPT"
echo "--------------------------------------------------------------"

CMD=(
    "$ISAAC_SH" "-p" "$LAUNCHER_SCRIPT"
    "--num_gpus"    "$NUM_GPUS"
    "--data_dir"    "$DATA_DIR"
    "--script_path" "$AGENT_SCRIPT"
    "--task"        "$TASK"
)

# Forward optional per-script args if set
[ -n "${NUM_ENVS:-}" ] && CMD+=("--num_envs" "$NUM_ENVS")

echo "Executing:"
echo "${CMD[@]}"
echo "--------------------------------------------------------------"

"${CMD[@]}"
