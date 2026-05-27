#!/bin/bash
# ============================================================
#  Submit A* data-collection SLURM array
# ============================================================
#  Single point of config. Edit, then `bash submit_a_star.sh`.
# ============================================================

# ── User config ─────────────────────────────────────────────
BASE_PATH="/oscar/scratch/arock3/VPT_DATA_A_STAR/v18_data_collector_run_full_new"
NUM_GPUS=6                              # GPUs per node
NUM_NODES=2                             # SLURM array size
TASK="VPT-v18-A-star"

# Collector hyperparams
TARGET_TOTAL=15000                      # successes across ALL GPUs
NUM_ENVS=96                             # envs per GPU process
PLAN_WORKERS=16                         # threads for parallel A*
MAX_TOTAL_STEPS=150
IMG_SIZE=256
SAVE_MODE="pass"                        # "pass" or "all"
SEED_BASE=42                            # per-GPU seed = SEED_BASE + node*NG + gpu

# SLURM resources
ACCOUNT="carney-tserre-condo2"
CONSTRAINT="blackwell"
PARTITION="gpu-he"
CPUS_PER_TASK=75                        # per node total
MEM="40G"                               # per node total
TIME="24:00:00"
# ─────────────────────────────────────────────────────────────

LOG_DIR="${BASE_PATH}/logs"
DATA_DIR="${BASE_PATH}/data"

mkdir -p "$LOG_DIR" "$DATA_DIR"

sbatch --array=0-$((NUM_NODES - 1)) \
  --account="${ACCOUNT}" \
  --constraint="${CONSTRAINT}" \
  --partition="${PARTITION}" \
  --gres=gpu:${NUM_GPUS} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --mem=${MEM} \
  --time=${TIME} \
  --output=${LOG_DIR}/worker_%A_%a.out \
  --error=${LOG_DIR}/worker_%A_%a.err \
  --export=ALL,BASE_PATH="${BASE_PATH}",NUM_GPUS=${NUM_GPUS},NUM_NODES=${NUM_NODES},TASK="${TASK}",TARGET_TOTAL=${TARGET_TOTAL},NUM_ENVS=${NUM_ENVS},PLAN_WORKERS=${PLAN_WORKERS},MAX_TOTAL_STEPS=${MAX_TOTAL_STEPS},IMG_SIZE=${IMG_SIZE},SAVE_MODE="${SAVE_MODE}",SEED_BASE=${SEED_BASE} \
  a_star_worker.sh
