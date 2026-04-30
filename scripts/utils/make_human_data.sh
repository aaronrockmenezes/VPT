#!/usr/bin/env bash
# run_human_study_sampling.sh
# Generates human study samples for all 3 dataset modes.
# Outputs land in /users/arock3/scratch/HUMAN_DATASETS/{depth,vpt1,vpt2}

set -e  # exit on any error

SCRIPT="$(dirname "$0")/sample_dataset.py"
BASE="/users/arock3/scratch/HUMAN_DATASETS_w_paths"

DEPTH_DATASET="/users/arock3/scratch/VPT1_depth_v18_A"
VPT1_DATASET="/users/arock3/scratch/VPT1_v18"
VPT2_DATASET="/users/arock3/scratch/VPT2_v3"

NUM_TRAIN=24
NUM_TEST=96
LLM_NUM_TRAIN=16
LLM_NUM_TEST=40
SIZE=256
SEED=5

echo "============================================"
echo " Human Study Sampling"
echo " num_train=${NUM_TRAIN} | num_test=${NUM_TEST} | size=${SIZE}x${SIZE}"
echo " llm_num_train=${LLM_NUM_TRAIN} | llm_num_test=${LLM_NUM_TEST}"
echo "============================================"

# ── Depth ─────────────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Depth..."
python "$SCRIPT" \
    --dataset_dir   "$DEPTH_DATASET" \
    --mode          depth \
    --num_train     "$NUM_TRAIN" \
    --num_test      "$NUM_TEST" \
    --llm_num_train "$LLM_NUM_TRAIN" \
    --llm_num_test  "$LLM_NUM_TEST" \
    --size          "$SIZE" \
    --seed          "$SEED" \
    --output_dir    "$BASE/depth"

# ── VPT1 ──────────────────────────────────────────────────────────────────────
echo ""
echo "[2/3] VPT1..."
python "$SCRIPT" \
    --dataset_dir   "$VPT1_DATASET" \
    --mode          vpt1 \
    --num_train     "$NUM_TRAIN" \
    --num_test      "$NUM_TEST" \
    --llm_num_train "$LLM_NUM_TRAIN" \
    --llm_num_test  "$LLM_NUM_TEST" \
    --size          "$SIZE" \
    --seed          "$SEED" \
    --output_dir    "$BASE/vpt1"

# ── VPT2 ──────────────────────────────────────────────────────────────────────
echo ""
echo "[3/3] VPT2..."
python "$SCRIPT" \
    --dataset_dir   "$VPT2_DATASET" \
    --mode          vpt2 \
    --num_train     "$NUM_TRAIN" \
    --num_test      "$NUM_TEST" \
    --llm_num_train "$LLM_NUM_TRAIN" \
    --llm_num_test  "$LLM_NUM_TEST" \
    --size          "$SIZE" \
    --seed          "$SEED" \
    --output_dir    "$BASE/vpt2"

echo ""
echo "============================================"
echo " All done. Outputs in $BASE"
echo "============================================"