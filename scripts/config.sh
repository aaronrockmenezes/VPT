#!/bin/bash
# =============================================================================
# config.sh — EDIT THIS FILE, then run ./launch.sh
# =============================================================================

# Task: "perspective" | "depth" | "vpt2"
TASK="perspective"

# Parallel arrays — data_dirs[i] pairs with output_dirs[i]
data_dirs=(
    "/users/arock3/scratch/THESIS/VPT_1_v18_tmp_count"
    # "/users/arock3/scratch/VPT1_v18_v3_finer"
    # "/users/arock3/scratch/VPT1_depth_v18_B"
    # "/users/arock3/scratch/VPT2_v4"
    # "/users/arock3/scratch/VPT_v17_v4_FULL/VPTnav_v17_high_depth_clip"
    # "/users/arock3/scratch/VPT_v17_v4_FULL/VPTnav_v17_high_low_depth_clip"
    # "/users/arock3/scratch/VPT_v17_v4_FULL/VPTnav_v17_low_depth_clip"
    # "/users/arock3/scratch/VPT_v17_v4_FULL/VPTnav_v17_low_high_depth_clip"
)

output_dirs=(
    "/users/arock3/scratch/VPT_logs/thesis/VPT_1_v18_tmp_count/lp_logs"
    # "/users/arock3/scratch/VPT_logs/perspective_new/v18_v3_finer/lp_logs"
    # "/users/arock3/scratch/VPT_logs/depth_new/v18_depth_B/lp_logs"
    # "/users/arock3/scratch/VPT_logs/vpt2/v4/lp_logs"
    # "/users/arock3/scratch/VPT_logs/depth_new/High_High/lp_logs"
    # "/users/arock3/scratch/VPT_logs/depth_new/High_Low/lp_logs"
    # "/users/arock3/scratch/VPT_logs/depth_new/Low_Low/lp_logs"
    # "/users/arock3/scratch/VPT_logs/depth_new/Low_High/lp_logs"
)

# Number of sequential repeats per model
NUM_RUNS=3

# Max concurrent array tasks (8 GPUs/node × N nodes you want)
MAX_CONCURRENT=40
