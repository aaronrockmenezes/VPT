#!/bin/bash

# 1. Get the directory where this bash script is located
#    This ensures we can find compile_results.py even after changing directories.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 2. Define your list of model names here
models=("swinv2_tiny_window8_256.ms_in1k" "caformer_s18.sail_in22k_ft_in1k" "mobilenetv3_small_050.lamb_in1k" "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384" "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k" "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k" "beit_large_patch16_512.in22k_ft_in22k_in1k" "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384" "convnext_xxlarge.clip_laion2b_soup_ft_in1k" "convnext_large.fb_in22k_ft_in1k_384", "eva02_tiny_patch14_336.mim_in22k_ft_in1k" "mobilenetv3_large_100.miil_in21k_ft_in1k" "resnet50.a1_in1k" "convnext_small.fb_in22k_ft_in1k_384" "convnext_tiny.fb_in1k" "deit3_small_patch16_224.fb_in1k" "eva_giant_patch14_560.m30m_ft_in22k_in1k")
# models=("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")

# Define the output directory
data_dir="/home/arock3/VPTnav_v15_high2_full_2048" 
output_dir="/home/arock3/Documents/Mono_VPT/VPTnav_v15_high2_full_2048"

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

# 3. Loop over each model name
for model in "${models[@]}"; do
    
    echo "Processing model: $model"
    output_file="${output_dir}/${model}.txt"

    # Clear the file if it already exists for a fresh log
    > "$output_file"

    # Loop 3 times
    for i in {1..3}; do
        echo "  - Starting run $i for $model..."
        
        # Add a visual separator in the text file
        echo "==============================" >> "$output_file"
        echo "RUN #$i - DATE: $(date)" >> "$output_file"
        echo "==============================" >> "$output_file"

        # Run the command
        python run_linear_probe.py \
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

# We use "$SCRIPT_DIR" to find the python file in the same folder as this script
if [ -f "$SCRIPT_DIR/compile_results.py" ]; then
    python "$SCRIPT_DIR/compile_results.py" \
        --results_dir "$output_dir" \
        --output_dir "$output_dir"
    echo "Done."
else
    echo "Error: compile_results.py not found in $SCRIPT_DIR"
fi