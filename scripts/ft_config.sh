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

# Fine-tuning is heavier than LP; keep model list explicit.
FT_MODELS_FILE="${FT_MODELS_FILE:-ft_models.txt}"

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
FT_MEM="120G"
FT_TIME="03:00:00"
FT_MAX_CONCURRENT=10
