#!/bin/bash
# Submit normal VPTnav VPT2 generation.
# Defaults to VPT2-v4 and vpt2_keyboard_agent.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_DIR="${REPO_ROOT}/VPTnav_code/cube_game/job_array/normal_vptnav"

BASE_PATH="${BASE_PATH:-/users/arock3/scratch/VPT2_DATA/v4}"
NUM_GPUS="${NUM_GPUS:-8}"
TASK="${TASK:-VPT2-v4}"
NUM_NODES="${NUM_NODES:-30}"
NUM_ENVS="${NUM_ENVS:-36}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
AGENT_SCRIPT="${AGENT_SCRIPT:-/mnt/VPT/VPTnav_code/cube_game/scripts/vptnav/vpt2_keyboard_agent.py}"

LOG_DIR="${BASE_PATH}/logs"
DATA_DIR="${BASE_PATH}/data"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

echo "[VPTNAV] VPT2 generation"
echo "  base=${BASE_PATH}"
echo "  task=${TASK} nodes=${NUM_NODES} gpus/node=${NUM_GPUS} envs/gpu=${NUM_ENVS} time=${TIME_LIMIT}"
echo "  agent=${AGENT_SCRIPT}"

cd "${JOB_DIR}"
sbatch --array=0-$((NUM_NODES - 1)) \
  --gres=gpu:${NUM_GPUS} \
  --time=${TIME_LIMIT} \
  --output=${LOG_DIR}/worker_%A_%a.out \
  --error=${LOG_DIR}/worker_%A_%a.err \
  --export=ALL,BASE_PATH=${BASE_PATH},NUM_GPUS=${NUM_GPUS},TASK=${TASK},NUM_ENVS=${NUM_ENVS},AGENT_SCRIPT=${AGENT_SCRIPT} \
  generation_worker.sh
