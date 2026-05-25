#!/bin/bash
# ============================================================
#  A* data collection — SLURM job array.
#  Each array task is INDEPENDENT: 1 node, NUM_GPUS GPUs,
#  collects TASK_TARGET successes (~1000 each) into its own
#  subdir. SLURM runs as many in parallel as cluster permits.
#  Total tasks = ceil(FULL_TARGET / TASK_TARGET) + BUFFER_TASKS.
# ============================================================

set -euo pipefail

# ── User config ─────────────────────────────────────────────
BASE_PATH="/oscar/scratch/arock3/VPT_DATA_A_STAR/v18_data_collector_v1"
NUM_GPUS=8                              # GPUs per task/node
TASK="VPT-v18-A-star"

FULL_TARGET=30000
ENVS_PER_GPU_TARGET=48                 # successes each GPU collects per task
BUFFER_TASKS=5

# Collector hyperparams
NUM_ENVS=96                             # parallel envs running on each GPU
PLAN_WORKERS=1
MAX_TOTAL_STEPS=150
IMG_SIZE=256
SETTLE_STEPS=30                         # v18 VPTnav-style render settle before capture
START_MODE="valid_viewpoint"            # verified camera+goal visible first frame
START_HALF_EXTENT=6.0                   # 12x12 m camera-centered start square
START_DEADZONE=3.0                      # reject starts inside 3x3 m square deadzone
CAM_NO_RED_MAX=0                        # No labels require zero red px in cam POV
SAVE_MODE="pass"
SEED_BASE=42
RESET_BASE_PATH=0                       # 1 = delete old run contents before sbatch
USE_GLOBAL_REWEIGHT=0                   # 0 = use FRAC_* directly; good for top-up jobs
DYNAMIC_BALANCE_ALPHA=0.9               # 0=final deficits only, 1=current-ratio catch-up only
FRAC_IN_VIEW=0.25                       # Yes
FRAC_OCCLUDED=0.5                       # No, goal occluded by camera body
FRAC_OUTSIDE_FOV=0.25                   # No, goal outside cam FOV
COMPILE_MIN_FRAMES=30
COMPILE_CAM_RED_THRESH=125              # Yes >125 red px; No == 0 red px
COMPILE_CAM_NO_RED_MAX=0                # set 25 to allow tiny red speckle in No
COMPILE_RM_WORKERS=6
COMPILE_NO_CLEAN=0                      # 0 = delete raw GPU dirs after compile

# SLURM resources (per task)
ACCOUNT="carney-tserre-condo2"
CONSTRAINT="blackwell"
PARTITION="gpu-he"
CPUS_PER_TASK=12                         # 4 collector procs + light compile/cleanup
MEM="180G"                              # ~24G/GPU process
TIME_PER_TASK="06:00:00"
MAX_TOTAL_CPUS=120                      # cap across concurrently running array tasks
MAX_TOTAL_GPUS=60                       # cap across concurrently running array tasks

# Auto-derived
TASK_TARGET=$(( ENVS_PER_GPU_TARGET * NUM_GPUS ))
if [ "${USE_GLOBAL_REWEIGHT}" = "1" ]; then
  GLOBAL_TARGET_FOR_COLLECTOR="${FULL_TARGET}"
else
  GLOBAL_TARGET_FOR_COLLECTOR=0
fi
# ─────────────────────────────────────────────────────────────

# Compute task count: ceil(full / task_target) + buffer
REQUIRED_TASKS=$(( (FULL_TARGET + TASK_TARGET - 1) / TASK_TARGET ))
NUM_TASKS=$(( REQUIRED_TASKS + BUFFER_TASKS ))
ARRAY_HI=$(( NUM_TASKS - 1 ))
MAX_PARALLEL_BY_CPU=$(( MAX_TOTAL_CPUS / CPUS_PER_TASK ))
MAX_PARALLEL_BY_GPU=$(( MAX_TOTAL_GPUS / NUM_GPUS ))
MAX_PARALLEL_NODES="${MAX_PARALLEL_BY_CPU}"
if [ "${MAX_PARALLEL_BY_GPU}" -lt "${MAX_PARALLEL_NODES}" ]; then
  MAX_PARALLEL_NODES="${MAX_PARALLEL_BY_GPU}"
fi
if [ "${MAX_PARALLEL_NODES}" -lt 1 ]; then
  echo "[ERR] resource caps allow zero parallel tasks: cpu_cap=${MAX_TOTAL_CPUS}, gpu_cap=${MAX_TOTAL_GPUS}, cpus_per_task=${CPUS_PER_TASK}, gpus_per_task=${NUM_GPUS}" >&2
  exit 1
fi

LOG_DIR="${BASE_PATH}/logs"
DATA_DIR="${BASE_PATH}/data"

if [ "${RESET_BASE_PATH}" = "1" ]; then
  case "${BASE_PATH}" in
    /oscar/scratch/arock3/VPT_DATA_A_STAR/*|/users/arock3/scratch/VPT_DATA_A_STAR/*)
      echo "[RESET] deleting old contents under ${BASE_PATH}"
      mkdir -p "${BASE_PATH}"
      find "${BASE_PATH}" -mindepth 1 -maxdepth 1 -print0 \
        | xargs -0 -r -P 8 -I{} rm -rf "{}"
      ;;
    *)
      echo "[ERR] refusing RESET_BASE_PATH for unsafe BASE_PATH=${BASE_PATH}" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "$LOG_DIR" "$DATA_DIR"

echo "==================================================="
echo "[ARRAY] full_target=$FULL_TARGET  task_target=$TASK_TARGET"
echo "[ARRAY] required=$REQUIRED_TASKS  buffer=$BUFFER_TASKS  total_tasks=$NUM_TASKS"
echo "[ARRAY] array=0-$ARRAY_HI%$MAX_PARALLEL_NODES  base_path=$BASE_PATH"
echo "[ARRAY] resource_caps cpus=$MAX_TOTAL_CPUS gpus=$MAX_TOTAL_GPUS -> max_parallel_nodes=$MAX_PARALLEL_NODES"
echo "[ARRAY] fractions in_view=$FRAC_IN_VIEW occluded=$FRAC_OCCLUDED outside_fov=$FRAC_OUTSIDE_FOV global_target=$GLOBAL_TARGET_FOR_COLLECTOR"
echo "[ARRAY] dynamic_balance_alpha=$DYNAMIC_BALANCE_ALPHA"
echo "[ARRAY] start_mode=$START_MODE start_half_extent=$START_HALF_EXTENT start_deadzone=$START_DEADZONE cam_no_red_max=$CAM_NO_RED_MAX settle=$SETTLE_STEPS"
echo "==================================================="

sbatch \
  --array=0-${ARRAY_HI}%${MAX_PARALLEL_NODES} \
  --account="${ACCOUNT}" \
  --constraint="${CONSTRAINT}" \
  --partition="${PARTITION}" \
  --gres=gpu:${NUM_GPUS} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --mem=${MEM} \
  --time=${TIME_PER_TASK} \
  --output=${LOG_DIR}/task_%A_%a.out \
  --error=${LOG_DIR}/task_%A_%a.err \
  --export=ALL,BASE_PATH="${BASE_PATH}",NUM_GPUS=${NUM_GPUS},TASK="${TASK}",TASK_TARGET=${TASK_TARGET},NUM_ENVS=${NUM_ENVS},PLAN_WORKERS=${PLAN_WORKERS},MAX_TOTAL_STEPS=${MAX_TOTAL_STEPS},IMG_SIZE=${IMG_SIZE},SETTLE_STEPS=${SETTLE_STEPS},START_MODE="${START_MODE}",START_HALF_EXTENT=${START_HALF_EXTENT},START_DEADZONE=${START_DEADZONE},CAM_NO_RED_MAX=${CAM_NO_RED_MAX},SAVE_MODE="${SAVE_MODE}",SEED_BASE=${SEED_BASE},GLOBAL_TARGET=${GLOBAL_TARGET_FOR_COLLECTOR},DYNAMIC_BALANCE_ALPHA=${DYNAMIC_BALANCE_ALPHA},FRAC_IN_VIEW=${FRAC_IN_VIEW},FRAC_OCCLUDED=${FRAC_OCCLUDED},FRAC_OUTSIDE_FOV=${FRAC_OUTSIDE_FOV},COMPILE_MIN_FRAMES=${COMPILE_MIN_FRAMES},COMPILE_CAM_RED_THRESH=${COMPILE_CAM_RED_THRESH},COMPILE_CAM_NO_RED_MAX=${COMPILE_CAM_NO_RED_MAX},COMPILE_RM_WORKERS=${COMPILE_RM_WORKERS},COMPILE_NO_CLEAN=${COMPILE_NO_CLEAN} \
  a_star_worker.sh
