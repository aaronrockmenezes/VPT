#!/bin/bash

# 1. Get the directory where this bash script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 2. Define your list of model names here
models=(
  # --- Self-Supervised Small/Tiny (DINO/MAE style) ---
  "vit_small_patch16_224.dino"
  "vit_small_patch14_dinov2.lvd142m"
  "vit_small_patch14_reg4_dinov2.lvd142m"
  "vit_small_patch16_dinov3.lvd1689m"
  "vit_small_patch16_dinov3_qkvb.lvd1689m"
  "convnext_small.dinov3_lvd1689m"
  "convnextv2_tiny.fcmae_ft_in22k_in1k"

  # --- Modern Base/Small/Tiny Variants ---
  "vit_small_patch16_224.augreg_in21k_ft_in1k"
  "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
  "swin_small_patch4_window7_224.ms_in1k"
  "swin_base_patch4_window7_224.ms_in1k"
  "convnextv2_atto.fcmae_ft_in1k"
  "convnextv2_base.fcmae_ft_in22k_in1k"
  "eva02_small_patch14_336.mim_in22k_ft_in1k"
  "eva02_base_patch14_448.mim_in22k_ft_in1k"
  "deit3_base_patch16_224.fb_in1k"
  "deit3_medium_patch16_224.fb_in1k"
  "maxvit_tiny_tf_224.in1k"
  "maxvit_small_tf_224.in1k"
  "pvt_v2_b0.in1k"
  "pvt_v2_b2.in1k"
  "xcit_tiny_12_p16_224.fb_dist_in1k"
  "xcit_small_12_p16_224.fb_dist_in1k"
  "cait_xxs24_224.fb_dist_in1k"
  "cait_s24_224.fb_dist_in1k"
  "efficientnet_b2.ra_in1k"

  # --- Conventional/Classic Baselines ---
  "resnet18.a1_in1k"
  "resnet34.a1_in1k"
  "vgg16.tv_in1k"
  "densenet121.ra_in1k"
  "inception_v3.tv_in1k"
  "xception.tf_in1k"
  "mobilenetv2_100.ra_in1k"
  "resnext50_32x4d.a1_in1k"
  "wide_resnet50_2.tv_in1k"
)

# Define the output directory
data_dir="/home/arock3/VPTnav_v17_rl_high_v4_depth_small" 
output_dir="/home/arock3/Documents/VPT_Model_Zoo/VPTnav_v17_rl_high_v4_depth_small"
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
            run_accelerate.py \
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