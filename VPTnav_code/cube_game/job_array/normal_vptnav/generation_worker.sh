#!/bin/bash
#SBATCH --account=carney-tserre-condo2
#SBATCH --constraint=blackwell
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=50G
#SBATCH --time=36:00:00
# ============================================================
#  Per-node SLURM worker for generation / camera-move collection.
#  Uses the overlay-pool apptainer activation (same as a_star_worker.sh),
#  then runs `multi_gpu.sh` inside, which spawns NUM_GPUS GPU processes.
# ============================================================

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-0}"
# Unique NODE_ID across submissions: {sbatch_job}_{array_idx}
NODE_ID="${JOB_ID}_${TASK_ID}"
echo "[TASK ${TASK_ID}] job=${JOB_ID} node_id=${NODE_ID} on $(hostname)"
echo "  BASE_PATH=${BASE_PATH}"
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  TASK=${TASK}"
echo "  AGENT_SCRIPT=${AGENT_SCRIPT:-<default>}  NUM_ENVS=${NUM_ENVS:-<default>}"

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
  --env SLURM_ARRAY_TASK_ID="${TASK_ID}" \
  --env SLURM_ARRAY_JOB_ID="${JOB_ID}" \
  --env TASK_ID="${TASK_ID}" \
  --env JOB_ID="${JOB_ID}" \
  --env NODE_ID="${NODE_ID}" \
  --env BASE_PATH="${BASE_PATH}" \
  --env NUM_GPUS="${NUM_GPUS}" \
  --env TASK="${TASK}" \
  --env AGENT_SCRIPT="${AGENT_SCRIPT:-}" \
  --env NUM_ENVS="${NUM_ENVS:-}" \
  /oscar/home/arock3/data/arock3/VPT/isaac-lab.simg \
  /mnt/VPT/VPTnav_code/cube_game/job_array/normal_vptnav/multi_gpu.sh
APPTAINER_RC=$?

echo "[TASK ${TASK_ID}] apptainer exit code: ${APPTAINER_RC}"
