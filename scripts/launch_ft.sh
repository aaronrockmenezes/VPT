#!/bin/bash
# launch_ft.sh - submit fine-tune model array from ft_config.sh.
#
# Usage:
#   bash launch_ft.sh
#   bash launch_ft.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/ft_config.sh"

MODELS_FILE="${FT_MODELS_FILE}"
if [[ "${MODELS_FILE}" != /* ]]; then
    MODELS_FILE="${SCRIPT_DIR}/${MODELS_FILE}"
fi

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

NUM_MODELS=$(grep -cv '^\s*$' "$MODELS_FILE")
NUM_CONFIGS=${#data_dirs[@]}

if [ "$NUM_CONFIGS" -ne "${#output_dirs[@]}" ]; then
    echo "ERROR: data_dirs (${NUM_CONFIGS}) and output_dirs (${#output_dirs[@]}) length mismatch."
    exit 1
fi

for data_dir in "${data_dirs[@]}"; do
    case "$TASK" in
        depth)
            train_split="train_depth"
            test_split="test_depth"
            ;;
        perspective|vpt2)
            train_split="train"
            test_split="test"
            ;;
        *)
            echo "ERROR: unknown TASK '$TASK'"
            exit 1
            ;;
    esac

    if [ ! -d "${data_dir}/${train_split}" ] || [ ! -d "${data_dir}/${test_split}" ]; then
        echo "ERROR: dataset split missing for ${data_dir}"
        echo "  expected: ${data_dir}/${train_split}"
        echo "  expected: ${data_dir}/${test_split}"
        exit 1
    fi
done

TOTAL_TASKS=$(( NUM_CONFIGS * NUM_MODELS ))
MAX_IDX=$(( TOTAL_TASKS - 1 ))
FT_CPUS_PER_TASK=$(( FT_GPUS_PER_TASK * FT_CPUS_PER_GPU ))

echo "========================================"
echo "  FT task:      $TASK"
echo "  Models:       $NUM_MODELS"
echo "  Datasets:     $NUM_CONFIGS"
echo "  Runs/model:   $FT_NUM_RUNS"
echo "  Epochs/run:   $FT_EPOCHS"
echo "  GPUs/task:    $FT_GPUS_PER_TASK"
echo "  CPUs/task:    $FT_CPUS_PER_TASK"
echo "  Mem/task:     $FT_MEM"
echo "  Time:         $FT_TIME"
echo "  Total tasks:  $TOTAL_TASKS"
echo "  Array range:  0-${MAX_IDX}%${FT_MAX_CONCURRENT}"
echo "========================================"
echo ""
for i in "${!data_dirs[@]}"; do
    echo "  [$i] ${data_dirs[$i]}"
    echo "    -> ${output_dirs[$i]}"
done
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit:"
    echo "  1. sbatch --array=0-${MAX_IDX}%${FT_MAX_CONCURRENT} --gres=gpu:${FT_GPUS_PER_TASK} --cpus-per-task=${FT_CPUS_PER_TASK} --mem=${FT_MEM} --time=${FT_TIME} ${SCRIPT_DIR}/submit_ft_array.sh"
    echo "  2. sbatch --dependency=afterany:<JOB_ID> ${SCRIPT_DIR}/compile_ft_job.sh"
    exit 0
fi

mkdir -p "${SCRIPT_DIR}/slurm_logs"

export FT_SCRIPT_DIR="$SCRIPT_DIR"

JOB_ID=$(sbatch --array=0-${MAX_IDX}%${FT_MAX_CONCURRENT} \
    --gres=gpu:${FT_GPUS_PER_TASK} \
    --cpus-per-task=${FT_CPUS_PER_TASK} \
    --mem=${FT_MEM} \
    --time=${FT_TIME} \
    --export=ALL \
    --parsable \
    "${SCRIPT_DIR}/submit_ft_array.sh")

echo "Submitted FT job array: $JOB_ID"

COMPILE_ID=$(sbatch --dependency=afterany:${JOB_ID} \
    --export=ALL \
    --parsable \
    "${SCRIPT_DIR}/compile_ft_job.sh")

echo "Submitted FT compile job: $COMPILE_ID (runs after $JOB_ID completes)"
echo ""
echo "  Monitor:  squeue -j $JOB_ID,$COMPILE_ID"
echo "  Logs:     ${SCRIPT_DIR}/slurm_logs/${JOB_ID}_ft_<task_id>.{out,err}"
echo "  Compile:  ${SCRIPT_DIR}/slurm_logs/${COMPILE_ID}_ft_compile.{out,err}"
echo "  Status:   sacct -j $JOB_ID --format=JobID%30,State%15,Elapsed,MaxRSS,NodeList"
