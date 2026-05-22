#!/bin/bash
#SBATCH --job-name=lp-array
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --account=carney-tserre-condo2
#SBATCH --partition=gpu-he
#SBATCH --constraint=h100|a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# =============================================================================
# Load config — LP_SCRIPT_DIR is exported by launch.sh
# =============================================================================
SCRIPT_DIR="${LP_SCRIPT_DIR:?'ERROR: LP_SCRIPT_DIR not set. Run via launch.sh, not directly.'}"
source "${SCRIPT_DIR}/config.sh"

MODELS_FILE="${SCRIPT_DIR}/models.txt"
PROJECT_ROOT="/users/arock3/data/arock3/VPT/VPT_code/VPT"
PYTHON_SCRIPT="${PROJECT_ROOT}/run_accel_latest_new.py"

# =============================================================================
# Resolve label map from task name
# =============================================================================
case "$TASK" in
    perspective) LABEL_MAP="no:0,yes:1" ;;
    depth)       LABEL_MAP="no:0,yes:1" ;;
    vpt2)        LABEL_MAP="left:0,right:1" ;;
    *)           echo "FATAL: Unknown task '$TASK'"; exit 1 ;;
esac

# =============================================================================
# Load models list
# =============================================================================
mapfile -t MODELS < <(grep -v '^\s*$' "$MODELS_FILE")
NUM_MODELS=${#MODELS[@]}
NUM_CONFIGS=${#data_dirs[@]}

# =============================================================================
# Decode SLURM_ARRAY_TASK_ID → (config_idx, model_idx)
# Layout: task_id = config_idx * NUM_MODELS + model_idx
# =============================================================================
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
echo "Array Task ID: $TASK_ID"
echo "  Task:      $TASK"
echo "  Model:     $MODEL_NAME"
echo "  Data:      $DATA_DIR"
echo "  Output:    $OUTPUT_DIR"
echo "  Label map: $LABEL_MAP"
echo "  Runs:      $NUM_RUNS"
echo "  GPU:       $CUDA_VISIBLE_DEVICES"
echo "  Node:      $(hostname)"
echo "  Start:     $(date)"
echo "============================================"

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT" || { echo "FATAL [model=$MODEL_NAME]: Cannot cd to $PROJECT_ROOT"; exit 1; }

# Activate your env — uncomment/adjust:
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vpt_env
# module load cuda/12.x

# Cache pretrained weights on scratch — download once, reuse everywhere
export TORCH_HOME="/users/arock3/scratch/.cache/torch"
export HF_HOME="/users/arock3/scratch/.cache/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME"

export TQDM_DISABLE=1

# =============================================================================
# Run — single launch, python handles multi-run internally
# =============================================================================
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}.txt"

echo "[model=$MODEL_NAME] Starting $NUM_RUNS runs"

accelerate launch \
    --num_processes=1 \
    --main_process_port $(( 29500 + TASK_ID % 1000 )) \
    "$PYTHON_SCRIPT" \
    --task "$TASK" \
    --data_dir "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --label_map "$LABEL_MAP" \
    --output_dir "$OUTPUT_DIR" \
    --num_runs "$NUM_RUNS" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAILED [model=$MODEL_NAME] [task=$TASK] [config=$CONFIG_IDX] exited with code $EXIT_CODE" | tee -a "$LOG_FILE" >&2
fi

echo "[model=$MODEL_NAME] Finished at $(date)"