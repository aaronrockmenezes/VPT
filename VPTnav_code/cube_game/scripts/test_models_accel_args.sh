#!/bin/bash

# 1. Get the directory where this bash script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Parse command-line arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_dir> <output_dir>"
    exit 1
fi

data_dir="$1"
output_dir="$2"

# 2. Define your list of model names here
models=(
#   --- Self-Supervised Small/Tiny (DINO/MAE style) ---
  "vit_small_patch16_224.dino"
  "vit_small_patch14_dinov2.lvd142m"
  "vit_small_patch14_reg4_dinov2.lvd142m"
  "vit_small_patch16_dinov3.lvd1689m"
  "vit_small_patch16_dinov3_qkvb.lvd1689m"
  "vit_base_patch14_dinov2.lvd142m"
  "vit_base_patch14_reg4_dinov2.lvd142m"
  "vit_large_patch14_dinov2.lvd142m"
  "vit_large_patch14_reg4_dinov2.lvd142m"
  "vit_base_patch16_224.dino"
  "vit_large_patch16_224.dino"
  "vit_base_patch16_224.mae"
  "vit_large_patch16_224.mae"
  "convnext_base.dinov3_lvd1689m"
  "convnext_large.dinov3_lvd1689m"

  # --- Modern Base/Small/Tiny Variants ---
  "vit_small_patch16_224.augreg_in21k_ft_in1k"
  "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
  "vit_base_patch16_clip_224.openai"
  "vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b"
  "swin_small_patch4_window7_224.ms_in1k"
  "swin_base_patch4_window7_224.ms_in1k"
  "convnextv2_base.fcmae_ft_in22k_in1k"
  "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
  "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384"
  "convnext_large.fb_in22k_ft_in1k_384"
  "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384"
  "eva02_small_patch14_336.mim_in22k_ft_in1k"
  "eva02_base_patch14_448.mim_in22k_ft_in1k"
  "deit3_base_patch16_224.fb_in1k"
  "deit3_medium_patch16_224.fb_in1k"
  "maxvit_tiny_tf_224.in1k"
  "maxvit_small_tf_224.in1k"
  "pvt_v2_b0.in1k"
  "pvt_v2_b2.in1k"
  "xcit_large_24_p16_224.fb_dist_in1k"
  "xcit_small_12_p16_224.fb_dist_in1k"
  "cait_m36_384.fb_dist_in1k"
  "cait_m48_448.fb_dist_in1k"
  "efficientnet_b7.ra3_in1k"
  "efficientnet_l2.ra_in1k"
  "mobilenetv3_large_100.miil_in21k_ft_in1k"
  "mobilenetv3_small_050.lamb_in1k"

  # --- BEiT (v1/v2/v3 - Scales to Large) ---
  "beit_base_patch16_224.in22k_ft_in22k_in1k"
  "beit_large_patch16_512.in22k_ft_in22k_in1k"

  # --- Conventional/Classic Baselines ---
  "resnet18.a1_in1k"
  "resnet34.a1_in1k"
  "vgg16.tv_in1k"
  "densenet121.ra_in1k"
  "inception_v3.tv_in1k"
  "xception.tf_in1k"
  "resnext50_32x4d.a1_in1k"
  "wide_resnet50_2.tv_in1k"
)

# Number of runs per model
num_runs=3

# Create the directory if it doesn't exist
mkdir -p "$output_dir"

# Navigate to project root
cd /media/data_cifs_lrs/projects/prj_robotics/VPT || { echo "Error: Could not change directory to project root. Exiting."; exit 1; }

echo "Successfully changed directory to $(pwd)"

eval "$(conda shell.bash hook)"
conda activate vpt_env
echo "Active Conda env: $CONDA_DEFAULT_ENV"

export TQDM_DISABLE=1

echo "Environment ready. Starting loops..."

unset CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=5

# 3. Loop over each model name
for model in "${models[@]}"; do
    
    echo "Processing model: $model"
    output_file="${output_dir}/${model}.txt"

    # Clear the file if it already exists for a fresh log
    > "$output_file"

    # Loop 5 times
    for ((i=1; i<=num_runs; i++)); do
        echo "  - Starting run $i for $model..."
        
        # Add a visual separator in the text file
        echo "==============================" >> "$output_file"
        echo "RUN #$i - DATE: $(date)" >> "$output_file"
        echo "==============================" >> "$output_file"

        # Run the command
        accelerate launch \
            --multi_gpu \
            --num_processes=5 \
            --gpu_ids="0,1,2,3,4" \
            --main_process_port 0 \
            run_accelerate_new.py \
            --task perspective \
            --data_dir "$data_dir" \
            --model_name "$model" \
            >> "$output_file" 2>&1

    done

    echo "Completed $model. Log saved to $output_file"
    echo "-----------------------------------"
done

# --- COMPILE RESULTS ---
echo "All models processed. Compiling results into CSV..."

if [ -f "$SCRIPT_DIR/compile_results.py" ]; then
    python "$SCRIPT_DIR/compile_results.py" \
        --results_dir "$output_dir" \
        --output_dir "$output_dir" \
        --num_runs "$num_runs"
    echo "Done."
else
    echo "Error: compile_results.py not found in $SCRIPT_DIR"
fi

# --- COMPILE CSVs ---
echo "Copying and analyzing CSVs..."

if [ -f "/home/arock3/visualization/analysis_args.py" ]; then
    python "/home/arock3/visualization/analysis_args.py" \
        --dataset_path "$data_dir"
    echo "Done."
else
    echo "Error: analysis_args.py not found in /home/arock3/visualization/"
fi
