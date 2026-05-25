#!/bin/bash
# ============================================================
#  Inside-container entry point. Calls a_star_launcher.py which
#  spawns NUM_GPUS subprocesses (one per GPU) for ONE array task.
# ============================================================

set -euo pipefail

DATA_DIR="${BASE_PATH}/data"
mkdir -p "$DATA_DIR"

LAUNCHER="/mnt/VPT/VPTnav_code/cube_game/job_array/a_star/a_star_launcher.py"
ISAAC_SH="/workspace/isaaclab/isaaclab.sh"

echo "[CONTAINER] task=${TASK_ID}  GPUs=${NUM_GPUS}  isaac_task=${TASK}"
echo "[CONTAINER] task_target=${TASK_TARGET}  num_envs=${NUM_ENVS}  save=${SAVE_MODE}  settle=${SETTLE_STEPS:-30}"
echo "[CONTAINER] start_mode=${START_MODE:-valid_viewpoint}  start_half_extent=${START_HALF_EXTENT:-6.0}  start_deadzone=${START_DEADZONE:-3.0}"
echo "[CONTAINER] cam_no_red_max=${CAM_NO_RED_MAX:-0}"
echo "[CONTAINER] fractions in_view=${FRAC_IN_VIEW:-0.50} occluded=${FRAC_OCCLUDED:-0.25} outside_fov=${FRAC_OUTSIDE_FOV:-0.25} global_target=${GLOBAL_TARGET:-0}"
echo "[CONTAINER] dynamic_balance_alpha=${DYNAMIC_BALANCE_ALPHA:-0.5}"

exec "$ISAAC_SH" -p "$LAUNCHER" \
    --task_id        "$TASK_ID" \
    --job_id         "${JOB_ID:-0}" \
    --num_gpus       "$NUM_GPUS" \
    --data_dir       "$DATA_DIR" \
    --task           "$TASK" \
    --task_target    "$TASK_TARGET" \
    --num_envs       "$NUM_ENVS" \
    --plan_workers   "$PLAN_WORKERS" \
    --max_total_steps "$MAX_TOTAL_STEPS" \
    --img_size       "$IMG_SIZE" \
    --settle_steps   "${SETTLE_STEPS:-30}" \
    --start_mode     "${START_MODE:-valid_viewpoint}" \
    --start_half_extent "${START_HALF_EXTENT:-6.0}" \
    --start_deadzone "${START_DEADZONE:-3.0}" \
    --cam_no_red_max "${CAM_NO_RED_MAX:-0}" \
    --save           "$SAVE_MODE" \
    --seed_base      "$SEED_BASE" \
    --global_target  "${GLOBAL_TARGET:-0}" \
    --dynamic_balance_alpha "${DYNAMIC_BALANCE_ALPHA:-0.5}" \
    --frac_in_view   "${FRAC_IN_VIEW:-0.50}" \
    --frac_occluded  "${FRAC_OCCLUDED:-0.25}" \
    --frac_outside_fov "${FRAC_OUTSIDE_FOV:-0.25}"
