#!/bin/bash
# =============================================================================
# ft_config.sh - fine-tune array defaults for thesis VPT1 v18.
# =============================================================================

# Task: "perspective" | "depth" | "vpt2"
TASK="perspective"

# Parallel arrays - data_dirs[i] pairs with output_dirs[i].
data_dirs=(
    "/users/arock3/scratch/THESIS/VPT_1_v18_tmp_count"
)

output_dirs=(
    "/users/arock3/scratch/VPT_logs/thesis/VPT_1_v18_tmp_count/ft_logs"
)

# Full fine-tune sweep. Override with FT_MODELS_FILE=ft_models.txt for smoke tests.
FT_MODELS_FILE="${FT_MODELS_FILE:-models.txt}"

FT_NUM_RUNS=3
FT_EPOCHS=30
FT_BATCH_SIZE=16
FT_EXTRACT_BATCH_SIZE=64
FT_LEARNING_RATE=5e-5
FT_WEIGHT_DECAY=1e-4
FT_NUM_WORKERS=2

# Resource policy: 2 CPU cores per GPU.
FT_GPUS_PER_TASK=4
FT_CPUS_PER_GPU=2
FT_MEM="80G"
FT_TIME="03:00:00"
FT_MAX_CONCURRENT=10

# One-shot overrides for launch commands. These avoid editing this file when
# launching a single dataset from the shell.
TASK="${FT_TASK:-$TASK}"
if [ -n "${FT_DATA_DIR:-}" ]; then
    data_dirs=("${FT_DATA_DIR}")
fi
if [ -n "${FT_OUTPUT_DIR:-}" ]; then
    output_dirs=("${FT_OUTPUT_DIR}")
fi
FT_MEM="${FT_MEM_OVERRIDE:-$FT_MEM}"
FT_MAX_CONCURRENT="${FT_MAX_CONCURRENT_OVERRIDE:-$FT_MAX_CONCURRENT}"
