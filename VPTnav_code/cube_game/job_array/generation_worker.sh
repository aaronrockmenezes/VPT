#!/bin/bash
#SBATCH --account=carney-tserre-condo
#SBATCH --constraint=l40s|a6000|a40
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=03:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate env_isaaclab

cd ~

apptainer exec --nv --writable-tmpfs \
  --bind /oscar/scratch/arock3/isaac_kit_cache:/isaac-sim/kit/data \
  --bind /oscar/scratch/arock3/isaac_experiments/skrl_logs:/workspace/isaaclab/logs \
  --bind /oscar/scratch/arock3/isaac_experiments/hydra_outputs:/workspace/isaaclab/outputs \
  --bind /oscar/scratch/arock3/isaac_pip_packages:/opt/user_packages \
  --bind /oscar/home/arock3/data/arock3/VPT:/mnt/VPT \
  --bind /oscar/scratch/arock3 \
  --env PYTHONPATH=/opt/user_packages:/mnt/VPT/VPTnav_code/cube_game/source/cube_game \
  --env SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID \
  --env BASE_PATH=$BASE_PATH \
  --env NUM_GPUS=$NUM_GPUS \
  --env TASK=$TASK \
  --env LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  /oscar/home/arock3/data/arock3/VPT/isaac-lab.simg \
  /mnt/VPT/VPTnav_code/cube_game/job_array/multi_gpu.sh