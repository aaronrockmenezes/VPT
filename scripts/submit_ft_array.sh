#!/bin/bash
#SBATCH --job-name=ft-array
#SBATCH --output=slurm_logs/%A_ft_%a.out
#SBATCH --error=slurm_logs/%A_ft_%a.err
#SBATCH --account=carney-tserre-condo2
#SBATCH --partition=gpu-he
#SBATCH --constraint=blackwell
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

SCRIPT_DIR="${FT_SCRIPT_DIR:?'ERROR: FT_SCRIPT_DIR not set. Run via launch_ft.sh, not directly.'}"
source "${SCRIPT_DIR}/ft_config.sh"

MODELS_FILE="${FT_MODELS_FILE}"
if [[ "${MODELS_FILE}" != /* ]]; then
    MODELS_FILE="${SCRIPT_DIR}/${MODELS_FILE}"
fi

PROJECT_ROOT="/users/arock3/data/arock3/VPT/VPT_code/VPT"
PYTHON_SCRIPT="${PROJECT_ROOT}/run_accel_finetune.py"

case "$TASK" in
    perspective) LABEL_MAP="no:0,yes:1" ;;
    depth)       LABEL_MAP="no:0,yes:1" ;;
    vpt2)        LABEL_MAP="left:0,right:1" ;;
    *)           echo "FATAL: Unknown task '$TASK'"; exit 1 ;;
esac

mapfile -t MODELS < <(grep -v '^\s*$' "$MODELS_FILE")
NUM_MODELS=${#MODELS[@]}
NUM_CONFIGS=${#data_dirs[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$(( TASK_ID / NUM_MODELS ))
MODEL_IDX=$(( TASK_ID % NUM_MODELS ))
MODEL_NAME="${MODELS[$MODEL_IDX]}"

if [ "$CONFIG_IDX" -ge "$NUM_CONFIGS" ] || [ "$MODEL_IDX" -ge "$NUM_MODELS" ]; then
    echo "FATAL [model=${MODEL_NAME:-unknown}]: TASK_ID=$TASK_ID out of range (configs=$NUM_CONFIGS, models=$NUM_MODELS)"
    exit 1
fi

DATA_DIR="${data_dirs[$CONFIG_IDX]}"
OUTPUT_DIR="${output_dirs[$CONFIG_IDX]}"

echo "============================================"
echo "FT Array Task ID: $TASK_ID"
echo "  Task:      $TASK"
echo "  Model:     $MODEL_NAME"
echo "  Data:      $DATA_DIR"
echo "  Output:    $OUTPUT_DIR"
echo "  Label map: $LABEL_MAP"
echo "  Runs:      $FT_NUM_RUNS"
echo "  Epochs:    $FT_EPOCHS"
echo "  GPUs:      ${SLURM_GPUS_ON_NODE:-unknown}"
echo "  CUDA:      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  Node:      $(hostname)"
echo "  Start:     $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT" || { echo "FATAL [model=$MODEL_NAME]: Cannot cd to $PROJECT_ROOT"; exit 1; }

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vpt_env

export TORCH_HOME="/users/arock3/scratch/.cache/torch"
export HF_HOME="/users/arock3/scratch/.cache/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME"

export TQDM_DISABLE=1

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}.txt"

accelerate launch \
    --num_processes="${FT_GPUS_PER_TASK}" \
    --main_process_port $(( 30500 + TASK_ID % 1000 )) \
    "$PYTHON_SCRIPT" \
    --task "$TASK" \
    --data_dir "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --label_map "$LABEL_MAP" \
    --output_dir "$OUTPUT_DIR" \
    --num_runs "$FT_NUM_RUNS" \
    --epochs "$FT_EPOCHS" \
    --batch_size "$FT_BATCH_SIZE" \
    --extract_batch_size "$FT_EXTRACT_BATCH_SIZE" \
    --learning_rate "$FT_LEARNING_RATE" \
    --weight_decay "$FT_WEIGHT_DECAY" \
    --num_workers "$FT_NUM_WORKERS" \
    > "$LOG_FILE" 2>&1

echo "[model=$MODEL_NAME] FT finished at $(date)"
