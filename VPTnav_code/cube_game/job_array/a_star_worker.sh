#!/bin/bash
# ============================================================
#  Per-node SLURM worker for A* data collection.
#  Launches apptainer container that runs `a_star_multi_gpu.sh`
#  inside, which spawns NUM_GPUS GPU processes.
# ============================================================

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-0}"
# Compose unique NODE_ID across submissions: {sbatch_job}_{array_idx}
NODE_ID="${JOB_ID}_${TASK_ID}"
echo "[TASK ${TASK_ID}] job=${JOB_ID} node_id=${NODE_ID} on $(hostname)"
echo "  BASE_PATH=${BASE_PATH}"
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  TASK=${TASK}  TASK_TARGET=${TASK_TARGET}"

CACHE_ROOT="/oscar/scratch/arock3/isaac_caches/task${TASK_ID}"
mkdir -p \
  "${CACHE_ROOT}/dot_cache" \
  "${CACHE_ROOT}/dot_nv" \
  "${CACHE_ROOT}/omniverse" \
  "${CACHE_ROOT}/local_share_ov" \
  "${CACHE_ROOT}/kit_cache" \
  "${CACHE_ROOT}/tmp"

OVERLAY_DIR="/oscar/scratch/arock3/isaac_overlays"
OVERLAY_POOL_SIZE=32                      # >= max parallel tasks (lots of headroom)
OVERLAY_SIZE_MB=512
POOL_IDX=$(( TASK_ID % OVERLAY_POOL_SIZE ))
OVERLAY_FILE="${OVERLAY_DIR}/pool${POOL_IDX}.img"
mkdir -p "$OVERLAY_DIR"
if [ ! -f "$OVERLAY_FILE" ]; then
  echo "[OVERLAY] creating ${OVERLAY_FILE} (${OVERLAY_SIZE_MB} MB)"
  # Stage in /tmp to avoid path collision with `apptainer overlay create`'s
  # scaffolding logic (which mkdirs the image's scaffold under target's
  # parent path; can collide with existing /users/.../scratch dirs).
  TMP_OVL="/tmp/pool${POOL_IDX}_$$.img"
  apptainer overlay create --size "$OVERLAY_SIZE_MB" "$TMP_OVL"
  mv "$TMP_OVL" "$OVERLAY_FILE"
fi
echo "[OVERLAY] task=${TASK_ID} -> pool slot ${POOL_IDX}"

apptainer exec --nv --cleanenv \
  --overlay "$OVERLAY_FILE" \
  --pwd /mnt/VPT/VPTnav_code/cube_game/scripts \
  --bind /oscar/scratch/arock3/isaac_kit_cache:/isaac-sim/kit/data \
  --bind ${CACHE_ROOT}/kit_cache:/isaac-sim/kit/cache \
  --bind ${CACHE_ROOT}/dot_cache:/root/.cache \
  --bind ${CACHE_ROOT}/dot_nv:/root/.nv \
  --bind ${CACHE_ROOT}/omniverse:/root/.nvidia-omniverse \
  --bind ${CACHE_ROOT}/local_share_ov:/root/.local/share/ov \
  --bind ${CACHE_ROOT}/tmp:/tmp \
  --bind /oscar/scratch/arock3/isaac_experiments/skrl_logs:/workspace/isaaclab/logs \
  --bind /oscar/scratch/arock3/isaac_experiments/hydra_outputs:/workspace/isaaclab/outputs \
  --bind /users/arock3/scratch/conda/envs/isaac_external_pkgs/lib/python3.11/site-packages:/opt/user_packages \
  --bind /oscar/home/arock3/data/arock3/VPT:/mnt/VPT \
  --bind /oscar/scratch/arock3 \
  --env PYTHONPATH=/opt/user_packages:/mnt/VPT/VPTnav_code/cube_game/source/cube_game \
  --env PYTHONNOUSERSITE=1 \
  --env SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}" \
  --env SLURM_ARRAY_JOB_ID="${JOB_ID}" \
  --env TASK_ID="${TASK_ID}" \
  --env JOB_ID="${JOB_ID}" \
  --env NODE_ID="${NODE_ID}" \
  --env BASE_PATH="${BASE_PATH}" \
  --env NUM_GPUS="${NUM_GPUS}" \
  --env TASK="${TASK}" \
  --env TASK_TARGET="${TASK_TARGET}" \
  --env NUM_ENVS="${NUM_ENVS}" \
  --env PLAN_WORKERS="${PLAN_WORKERS}" \
  --env MAX_TOTAL_STEPS="${MAX_TOTAL_STEPS}" \
  --env IMG_SIZE="${IMG_SIZE}" \
  --env SETTLE_STEPS="${SETTLE_STEPS:-30}" \
  --env START_MODE="${START_MODE:-valid_viewpoint}" \
  --env START_HALF_EXTENT="${START_HALF_EXTENT:-6.0}" \
  --env START_DEADZONE="${START_DEADZONE:-3.0}" \
  --env CAM_NO_RED_MAX="${CAM_NO_RED_MAX:-0}" \
  --env SAVE_MODE="${SAVE_MODE}" \
  --env SEED_BASE="${SEED_BASE}" \
  --env GLOBAL_TARGET="${GLOBAL_TARGET:-0}" \
  --env DYNAMIC_BALANCE_ALPHA="${DYNAMIC_BALANCE_ALPHA:-0.5}" \
  --env FRAC_IN_VIEW="${FRAC_IN_VIEW:-0.50}" \
  --env FRAC_OCCLUDED="${FRAC_OCCLUDED:-0.25}" \
  --env FRAC_OUTSIDE_FOV="${FRAC_OUTSIDE_FOV:-0.25}" \
  /oscar/home/arock3/data/arock3/VPT/isaac-lab.simg \
  /mnt/VPT/VPTnav_code/cube_game/job_array/a_star_multi_gpu.sh
APPTAINER_RC=$?

echo "[TASK ${TASK_ID}] apptainer exit code: ${APPTAINER_RC}"

# ── Per-task compile + cleanup ─────────────────────────────────────────────
# Verifies envs, copies passing ones to {BASE_PATH}/data/data_node*_compiled/,
# then removes the raw data_node*_gpu*/ dirs to free scratch.
if [ "$APPTAINER_RC" -eq 0 ]; then
  echo "[TASK ${TASK_ID}] compiling and cleaning up..."
  set +u                                  # conda init touches unset vars
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" \
    || source /users/arock3/.conda/etc/profile.d/conda.sh
  conda activate eval_env
  set -u

  COMPILE_SCRIPT="/users/arock3/data/arock3/VPT/a_star_data_collection_scripts/compile_tasks.py"
  if [ ! -f "${COMPILE_SCRIPT}" ]; then
    COMPILE_SCRIPT="/users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/compile_tasks.py"
  fi

  if [ ! -f "${COMPILE_SCRIPT}" ]; then
    echo "[TASK ${TASK_ID}] compile_tasks.py not found; raw data left in place." >&2
  elif [ "${COMPILE_NO_CLEAN:-0}" = "1" ]; then
    python "${COMPILE_SCRIPT}" \
      --base_path "${BASE_PATH}" \
      --node_id "${NODE_ID}" \
      --min_frames "${COMPILE_MIN_FRAMES:-10}" \
      --cam_red_thresh "${COMPILE_CAM_RED_THRESH:-125}" \
      --cam_no_red_max "${COMPILE_CAM_NO_RED_MAX:-0}" \
      --rm_workers "${COMPILE_RM_WORKERS:-16}" \
      --no_clean \
      || echo "[TASK ${TASK_ID}] compile_tasks.py failed; raw data left in place."
  else
    python "${COMPILE_SCRIPT}" \
      --base_path "${BASE_PATH}" \
      --node_id "${NODE_ID}" \
      --min_frames "${COMPILE_MIN_FRAMES:-10}" \
      --cam_red_thresh "${COMPILE_CAM_RED_THRESH:-125}" \
      --cam_no_red_max "${COMPILE_CAM_NO_RED_MAX:-0}" \
      --rm_workers "${COMPILE_RM_WORKERS:-16}" \
      || echo "[TASK ${TASK_ID}] compile_tasks.py failed; raw data left in place."
  fi
  set +u
  conda deactivate
  set -u
else
  echo "[TASK ${TASK_ID}] apptainer non-zero exit; skipping compile, leaving raw."
fi
