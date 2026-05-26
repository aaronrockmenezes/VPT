#!/bin/bash
# =============================================================
#  compile_dataset.sh
#  Stage raw per-GPU dirs → per-node compiled dirs, then merge
#  into a single canonical dataset with train/val/test split.
#
#  Staged mode (default):
#    bash compile_dataset.sh
#
#  Raw mode (skip staging, read raw GPU dirs directly):
#    bash compile_dataset.sh --raw
#
#  Other overrides:
#    --random          natural collection ratio instead of 50/25/25
#    --total N         number of compiled envs (default: see config)
#    --out_dir PATH    output dataset dir (default: see config)
#    --num_gpus N      GPUs per node (must match submit config)
#    --overwrite_out   delete out_dir before writing (clean rerun)
#    --seed N          shuffle seed
#    --train_frac F    train fraction (default 0.8)
#    --val_frac F      val fraction (default 0.1)
#    --test_frac F     test fraction (default 0.1)
# =============================================================

set -euo pipefail

# ══════════════════════════════════════════════════════════════
# EDIT THESE
# ══════════════════════════════════════════════════════════════
BASE_PATH="/oscar/scratch/arock3/VPT_DATA_A_STAR/v18_data_collector_v1"
TOTAL=10000
OUT_DIR="/oscar/scratch/arock3/VPT_DATA_A_STAR/v18_compiled_10_new"
# ══════════════════════════════════════════════════════════════

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILE_ALL_NODES="/users/arock3/data/arock3/VPT/a_star_data_collection_scripts/compile_all_nodes.py"
COMPILE_DATASET="/users/arock3/data/arock3/VPT/a_star_data_collection_scripts/compile_a_star_dataset.py"

# Resource config — 128 CPUs, 80G
# NODE_WORKERS * STAGE_WORKERS = total threads during staging
NODE_WORKERS=16       # nodes staged in parallel
STAGE_WORKERS=8       # copy threads per node  (16 * 8 = 128)
MERGE_WORKERS=120     # threads for final merge (leave OS headroom)

# Dataset defaults
NUM_GPUS=4
TRAIN_FRAC=0.8
VAL_FRAC=0.1
TEST_FRAC=0.1
SEED=0
MIN_FRAMES=10
CAM_RED_THRESH=125
CAM_NO_RED_MAX=0

# Flags
MODE="staged"
RANDOM_FLAG=""
OVERWRITE_OUT=""

# ── Parse CLI ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw)           MODE="raw"; shift ;;
        --random)        RANDOM_FLAG="--random"; shift ;;
        --overwrite_out) OVERWRITE_OUT="--overwrite_out"; shift ;;
        --total)         TOTAL="$2"; shift 2 ;;
        --out_dir)       OUT_DIR="$2"; shift 2 ;;
        --num_gpus)      NUM_GPUS="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --train_frac)    TRAIN_FRAC="$2"; shift 2 ;;
        --val_frac)      VAL_FRAC="$2"; shift 2 ;;
        --test_frac)     TEST_FRAC="$2"; shift 2 ;;
        -h|--help)       sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^#  \?//'; exit 0 ;;
        *) echo "[ERR] unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Build --gpus arg list for compile_all_nodes.py
GPU_ARGS=()
for g in $(seq 0 $((NUM_GPUS - 1))); do GPU_ARGS+=("$g"); done

echo "==================================================="
echo "[CONFIG] base_path=${BASE_PATH}"
echo "[CONFIG] total=${TOTAL}  out_dir=${OUT_DIR}"
echo "[CONFIG] mode=${MODE}  random=${RANDOM_FLAG:-no}  seed=${SEED}"
echo "[CONFIG] split  train=${TRAIN_FRAC}  val=${VAL_FRAC}  test=${TEST_FRAC}"
echo "[CONFIG] gpus=0...$((NUM_GPUS-1))  node_workers=${NODE_WORKERS}  stage_workers=${STAGE_WORKERS}  merge_workers=${MERGE_WORKERS}"
echo "==================================================="

# ── STAGED MODE ───────────────────────────────────────────────
if [ "$MODE" = "staged" ]; then

    DATA_DIR="${BASE_PATH}/data"

    # Pre-pass: for nodes already compiled, delete matching raw GPU dirs.
    # Handles mid-run node shutdown: compiled dir present = staging done.
    echo ""
    echo "[RESUME] checking for already-compiled nodes..."
    cleaned=0
    for compiled_dir in "${DATA_DIR}"/data_node*_compiled/; do
        [ -d "$compiled_dir" ] || continue
        node_key=$(basename "$compiled_dir" | sed 's/^data_node//; s/_compiled$//')
        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            raw_dir="${DATA_DIR}/data_node${node_key}_gpu${gpu_id}"
            if [ -d "$raw_dir" ]; then
                echo "  [DEL] $(basename "$raw_dir")  (compiled dir exists)"
                rm -rf "$raw_dir"
                cleaned=$((cleaned + 1))
            fi
        done
    done
    echo "[RESUME] removed ${cleaned} raw GPU dirs from already-compiled nodes."

    # Stage remaining raw dirs → compiled. --clean deletes raw dirs after
    # each successful node compile, so a crash leaves no partial raw dirs
    # for nodes that finished.
    raw_remaining=$(find "${DATA_DIR}" -maxdepth 1 -type d -name "data_node*_gpu*" 2>/dev/null | wc -l)
    if [ "$raw_remaining" -eq 0 ]; then
        echo "[STAGE] no raw GPU dirs remaining — all nodes already compiled. skipping."
    else
        echo ""
        echo "[STAGE] staging ${raw_remaining} raw GPU dirs..."
        python "$COMPILE_ALL_NODES" \
            --base_path      "$BASE_PATH" \
            --gpus           "${GPU_ARGS[@]}" \
            --workers        "$STAGE_WORKERS" \
            --node_workers   "$NODE_WORKERS" \
            --min_frames     "$MIN_FRAMES" \
            --cam_red_thresh "$CAM_RED_THRESH" \
            --cam_no_red_max "$CAM_NO_RED_MAX" \
            --clean
        echo "[STAGE] done."
    fi

fi

# ── MERGE ─────────────────────────────────────────────────────
echo ""
echo "[MERGE] compiling final dataset → ${OUT_DIR}"
python "$COMPILE_DATASET" \
    --src_root       "$BASE_PATH" \
    --out_dir        "$OUT_DIR" \
    --total          "$TOTAL" \
    --mode           "$MODE" \
    --train_frac     "$TRAIN_FRAC" \
    --val_frac       "$VAL_FRAC" \
    --test_frac      "$TEST_FRAC" \
    --workers        "$MERGE_WORKERS" \
    --seed           "$SEED" \
    --min_frames     "$MIN_FRAMES" \
    --cam_red_thresh "$CAM_RED_THRESH" \
    --cam_no_red_max "$CAM_NO_RED_MAX" \
    ${RANDOM_FLAG} \
    ${OVERWRITE_OUT}

echo ""
echo "==================================================="
echo "[DONE] dataset → ${OUT_DIR}"
echo "==================================================="
