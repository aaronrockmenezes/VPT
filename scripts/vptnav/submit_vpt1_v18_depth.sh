#!/bin/bash
# Submit normal VPTnav VPT1 v18 depth generation.
# This is not A*. It runs keyboard_agent.py on task VPT-v18-Depth.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_DIR="${REPO_ROOT}/VPTnav_code/cube_game/job_array/normal_vptnav"

BASE_PATH="${BASE_PATH:-/users/arock3/scratch/VPT1_DATA/thesis/v18_depth}"
NUM_GPUS="${NUM_GPUS:-8}"
TASK="${TASK:-VPT-v18-Depth}"
NUM_NODES="${NUM_NODES:-30}"
NUM_ENVS="${NUM_ENVS:-96}"
AGENT_SCRIPT="${AGENT_SCRIPT:-/mnt/VPT/VPTnav_code/cube_game/scripts/vptnav/keyboard_agent.py}"

LOG_DIR="${BASE_PATH}/logs"
DATA_DIR="${BASE_PATH}/data"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

echo "[VPTNAV] VPT1 v18 depth generation"
echo "  base=${BASE_PATH}"
echo "  task=${TASK} nodes=${NUM_NODES} gpus/node=${NUM_GPUS} envs/gpu=${NUM_ENVS}"
echo "  agent=${AGENT_SCRIPT}"

cd "${JOB_DIR}"
sbatch --array=0-$((NUM_NODES - 1)) \
  --gres=gpu:${NUM_GPUS} \
  --output=${LOG_DIR}/worker_%A_%a.out \
  --error=${LOG_DIR}/worker_%A_%a.err \
  --export=ALL,BASE_PATH=${BASE_PATH},NUM_GPUS=${NUM_GPUS},TASK=${TASK},NUM_ENVS=${NUM_ENVS},AGENT_SCRIPT=${AGENT_SCRIPT} \
  generation_worker.sh
