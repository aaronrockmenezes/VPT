#!/bin/bash

# 1. Get the directory where this bash script is located
#    This ensures we can find compile_results.py even after changing directories.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 2. Define your list of model names here
# models=(
#   "beit3_base_patch16_224.in22k_ft_in1k"
#   "beit3_large_patch16_224.in22k_ft_in1k"
#   "beit_base_patch16_224.in22k_ft_in22k_in1k"
#   "beit_large_patch16_224.in22k_ft_in22k_in1k"
#   "beit_large_patch16_384.in22k_ft_in22k_in1k"
#   "beit_large_patch16_512.in22k_ft_in22k_in1k"
#   "beitv2_base_patch16_224.in1k_ft_in22k_in1k"
#   "beitv2_large_patch16_224.in1k_ft_in22k_in1k"
#   "caformer_s18.sail_in22k_ft_in1k"
#   "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384"
#   "convnext_large.fb_in22k_ft_in1k_384"
#   "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384"
#   "convnext_small.fb_in22k_ft_in1k_384"
#   "convnext_tiny.fb_in1k"
#   "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
#   "deit3_small_patch16_224.fb_in1k"
#   "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
#   "eva02_tiny_patch14_336.mim_in22k_ft_in1k"
#   "eva_giant_patch14_560.m30m_ft_in22k_in1k"
#   "mobilenetv3_large_100.miil_in21k_ft_in1k"
#   "mobilenetv3_small_050.lamb_in1k"
#   "resnet50.a1_in1k"
#   "swinv2_tiny_window8_256.ms_in1k"
#   "vit_base_mci_224.apple_mclip2_dfndr2b"
#   "vit_base_patch14_dinov2.lvd142m"
#   "vit_base_patch14_reg4_dinov2.lvd142m"
#   "vit_base_patch16_224.dino"
#   "vit_base_patch16_224.mae"
#   "vit_base_patch16_clip_224.openai"
#   "vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b"
#   "vit_base_patch16_clip_quickgelu_224.metaclip_400m"
#   "vit_base_patch16_dinov3.lvd1689m"
#   "vit_base_patch16_siglip_224.v2_webli"
#   "vit_base_patch16_siglip_256.v2_webli"
#   "vit_base_patch16_siglip_384.v2_webli"
#   "vit_base_patch16_siglip_512.v2_webli"
#   "vit_base_patch32_clip_quickgelu_224.metaclip_2pt5b"
#   "vit_base_patch32_clip_quickgelu_224.metaclip_400m"
#   "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"
#   "vit_pe_spatial_base_patch16_512.fb"
#   "convnext_large.dinov3_lvd1689m"
#   "convnext_base.dinov3_lvd1689m"
#   "convnext_small.dinov3_lvd1689m"
#   "convnext_tiny.dinov3_lvd1689m"
# )

models=(
  # --- BEiT (v1/v2/v3) ---
  "beit_base_patch16_224.in22k_ft_in22k_in1k"
  "beit_large_patch16_224.in22k_ft_in22k_in1k"
  "beit_large_patch16_384.in22k_ft_in22k_in1k"
  "beit_large_patch16_512.in22k_ft_in22k_in1k"
  "beitv2_base_patch16_224.in1k_ft_in22k_in1k"
  "beitv2_large_patch16_224.in1k_ft_in22k_in1k"
  "beit3_base_patch16_224.in22k_ft_in1k"
  "beit3_large_patch16_224.in22k_ft_in1k"
  
  # --- CaiT ---
  "cait_xxs24_224.fb_dist_in1k"
  "cait_s24_224.fb_dist_in1k"
  "cait_m36_384.fb_dist_in1k"
  "cait_m48_448.fb_dist_in1k"
  
  # --- CLIP & SigLIP ---
  "vit_base_patch16_clip_224.openai"
  "vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b"
  "vit_base_patch32_clip_quickgelu_224.metaclip_2pt5b"
  "vit_base_patch16_siglip_224.v2_webli"
  "vit_base_patch16_siglip_512.v2_webli"
  "vit_large_patch14_clip_336.openai"
  "vit_large_patch16_siglip_384.webli"
  "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"
  "vit_giant_patch14_clip_224.laion2b"
  
  # --- ConvNeXt (Original) ---
  "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384"
  "convnext_large.fb_in22k_ft_in1k_384"
  "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384"
  "convnext_xlarge.fb_in22k_ft_in1k_384"
  "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
  
  # --- ConvNeXt V2 ---
  "convnextv2_atto.fcmae_ft_in1k"
  "convnextv2_tiny.fcmae_ft_in22k_in1k"
  "convnextv2_base.fcmae_ft_in22k_in1k"
  "convnextv2_large.fcmae_ft_in22k_in1k"
  "convnextv2_huge.fcmae_ft_in22k_in1k"
  
  # --- DeiT III ---
  "deit3_base_patch16_224.fb_in1k"
  "deit3_medium_patch16_224.fb_in1k"
  "deit3_large_patch16_224.fb_in1k"
  "deit3_huge_patch14_224.fb_in1k"
  
  # --- DenseNet ---
  "densenet121.ra_in1k"
  
  # --- DINOv2 ---
  "vit_small_patch16_224.dino"
  "vit_small_patch14_dinov2.lvd142m"
  "vit_small_patch14_reg4_dinov2.lvd142m"
  "vit_base_patch16_224.dino"
  "vit_base_patch14_dinov2.lvd142m"
  "vit_base_patch14_reg4_dinov2.lvd142m"
  "vit_large_patch16_224.dino"
  "vit_large_patch14_dinov2.lvd142m"
  "vit_large_patch14_reg4_dinov2.lvd142m"
  
  # --- DINOv3 ---
  "vit_small_patch16_dinov3.lvd1689m"
  "vit_small_patch16_dinov3_qkvb.lvd1689m"
  "convnext_small.dinov3_lvd1689m"
  "vit_base_patch16_dinov3.lvd1689m"
  "convnext_base.dinov3_lvd1689m"
  "vit_large_patch16_dinov3.lvd1689m"
  "convnext_large.dinov3_lvd1689m"
  
  # --- EfficientNet ---
  "efficientnet_b2.ra_in1k"
  
  # --- EVA & EVA02 ---
  "eva02_small_patch14_336.mim_in22k_ft_in1k"
  "eva02_base_patch14_448.mim_in22k_ft_in1k"
  "eva02_large_patch14_448.mim_in22k_ft_in1k"
  "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
  "eva_giant_patch14_560.m30m_ft_in22k_in1k"
  
  # --- Inception ---
  "inception_v3.tv_in1k"
  
  # --- MaxViT ---
  "maxvit_tiny_tf_224.in1k"
  "maxvit_small_tf_224.in1k"
  "maxvit_base_tf_224.in1k"
  "maxvit_large_tf_224.in1k"
  "maxvit_xlarge_tf_224.in21k_ft_in1k"
  
  # --- MobileNet ---
  "mobilenetv3_small_050.lamb_in1k"
  "mobilenetv2_100.ra_in1k"
  "mobilenetv3_large_100.miil_in21k_ft_in1k"
  
  # --- PVT v2 ---
  "pvt_v2_b0.in1k"
  "pvt_v2_b2.in1k"
  "pvt_v2_b4.in1k"
  "pvt_v2_b5.in1k"
  
  # --- ResNet ---
  "resnet18.a1_in1k"
  "resnet34.a1_in1k"
  
  # --- ResNeXt ---
  "resnext50_32x4d.a1_in1k"
  
  # --- Swin Transformer ---
  "swin_small_patch4_window7_224.ms_in1k"
  "swin_base_patch4_window7_224.ms_in1k"
  
  # --- Swin Transformer V2 ---
  "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"
  "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"
  
  # --- VGG ---
  "vgg16.tv_in1k"
  
  # --- ViT (Other Variants) ---
  "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
  "vit_small_patch16_224.augreg_in21k_ft_in1k"
  "vit_base_mci_224.apple_mclip2_dfndr2b"
  "vit_base_patch16_224.mae"
  "vit_large_patch16_224.mae"
  "vit_pe_spatial_base_patch16_512.fb"
  
  # --- Wide ResNet ---
  "wide_resnet50_2.tv_in1k"
  
  # --- XCiT ---
  "xcit_tiny_12_p16_224.fb_dist_in1k"
  "xcit_small_12_p16_224.fb_dist_in1k"
  "xcit_large_24_p16_224.fb_dist_in1k"
  
  # --- Xception ---
  "xception.tf_in1k"
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

# We use "$SCRIPT_DIR" to find the python file in the same folder as this script
if [ -f "$SCRIPT_DIR/compile_results.py" ]; then
    python "$SCRIPT_DIR/compile_results.py" \
        --results_dir "$output_dir" \
        --output_dir "$output_dir" \
        --num_runs "$num_runs"
    echo "Done."
else
    echo "Error: compile_results.py not found in $SCRIPT_DIR"
fi