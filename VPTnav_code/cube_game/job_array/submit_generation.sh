#!/bin/bash
# ============================================
#  CONFIG — Edit only this block
# ============================================
BASE_PATH="/oscar/scratch/arock3/VPT1_DATA/v18_rgb_cam"
NUM_GPUS=4
TASK="VPT-v18"
NUM_NODES=4
# ============================================

LOG_DIR="${BASE_PATH}/logs"
DATA_DIR="${BASE_PATH}/data"

mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"

sbatch --array=0-$((NUM_NODES - 1)) \
  --gres=gpu:${NUM_GPUS} \
  --output=${LOG_DIR}/worker_%A_%a.out \
  --error=${LOG_DIR}/worker_%A_%a.err \
  --export=ALL,BASE_PATH=${BASE_PATH},NUM_GPUS=${NUM_GPUS},TASK=${TASK} \
  generation_worker.sh