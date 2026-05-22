#!/bin/bash

# 1. Define your list of model names here
models=("mobilenetv3_small_050.lamb_in1k" "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384" "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k" "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k" "beit_large_patch16_512.in22k_ft_in22k_in1k" "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384" "convnext_xxlarge.clip_laion2b_soup_ft_in1k" "convnext_large.fb_in22k_ft_in1k_384")
# models=("convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384" "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384" "convnext_xxlarge.clip_laion2b_soup_ft_in1k" "convnext_large.fb_in22k_ft_in1k_384")

# Define the output directory
# data_dir="/media/data_cifs_lrs/projects/prj_robotics/VPTnav_v14" 
data_dir="/home/arock3/VPTnav_v15_new" 
output_dir="/home/arock3/Documents/VPTnav_v15_new"

# Create the directory if it doesn't exist
mkdir -p "$output_dir"

cd /media/data_cifs_lrs/projects/prj_robotics/VPT || { echo "Error: Could not change directory to /media/data_cifs_lrs/projects/prj_robotics/VPT. Exiting."; exit 1; }

echo "Successfully changed directory to $(pwd)"

eval "$(conda shell.bash hook)"
conda activate vpt_env
echo "Active Conda env: $CONDA_DEFAULT_ENV"

export TQDM_DISABLE=1

echo "Environment ready. Starting loops..."

# 2. Loop over each model name
for model in "${models[@]}"; do
    
    echo "Processing model: $model"
    output_file="${output_dir}/${model}.txt"

    # Optional: Clear the file if it already exists so you get a fresh log for these 3 runs
    # If you prefer to keep old data, remove the line below.
    > "$output_file"

    # Loop 3 times
    for i in {1..3}; do
        echo "  - Starting run $i for $model..."
        
        # Add a visual separator in the text file for readability
        echo "==============================" >> "$output_file"
        echo "RUN #$i - DATE: $(date)" >> "$output_file"
        echo "==============================" >> "$output_file"

        # Run the command
        # ">>" appends to the file
        # "2>&1" ensures errors are also saved to the text file
        python run_linear_probe.py \
            --task perspective \
            --data_dir "$data_dir" \
            --model_name "$model" \
            >> "$output_file" 2>&1

    done

    echo "Completed $model. Log saved to $output_file"
    echo "-----------------------------------"
done