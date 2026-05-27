#!/bin/bash
# launch.sh — computes array range from config.sh and submits the SLURM job
#
# Usage:
#   ./launch.sh              # submit
#   ./launch.sh --dry-run    # just print what would happen

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/config.sh"

MODELS_FILE="${SCRIPT_DIR}/models.txt"
DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

# Validate
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

echo "========================================"
echo "  Task:        $TASK"
echo "  Models:      $NUM_MODELS"
echo "  Datasets:    $NUM_CONFIGS"
echo "  Runs/model:  $NUM_RUNS"
echo "  Total tasks: $TOTAL_TASKS"
echo "  Array range: 0-${MAX_IDX}%${MAX_CONCURRENT}"
echo "========================================"
echo ""
for i in "${!data_dirs[@]}"; do
    echo "  [$i] ${data_dirs[$i]}"
    echo "    -> ${output_dirs[$i]}"
done
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit:"
    echo "  1. sbatch --array=0-${MAX_IDX}%${MAX_CONCURRENT} ${SCRIPT_DIR}/submit_array.sh"
    echo "  2. sbatch --dependency=afterany:<JOB_ID> ${SCRIPT_DIR}/compile_job.sh"
    exit 0
fi

mkdir -p "${SCRIPT_DIR}/slurm_logs"

export LP_SCRIPT_DIR="$SCRIPT_DIR"

JOB_ID=$(sbatch --array=0-${MAX_IDX}%${MAX_CONCURRENT} \
    --export=ALL,LP_SCRIPT_DIR="${SCRIPT_DIR}",LP_TASK="${LP_TASK:-}",LP_DATA_DIR="${LP_DATA_DIR:-}",LP_OUTPUT_DIR="${LP_OUTPUT_DIR:-}",LP_MAX_CONCURRENT="${LP_MAX_CONCURRENT:-}" \
    --parsable \
    "${SCRIPT_DIR}/submit_array.sh")

echo "Submitted job array: $JOB_ID"

# Submit compile job — runs after all array tasks finish
COMPILE_ID=$(sbatch --dependency=afterany:${JOB_ID} \
    --export=ALL \
    --parsable \
    "${SCRIPT_DIR}/compile_job.sh")

echo "Submitted compile job: $COMPILE_ID (runs after $JOB_ID completes)"
echo ""
echo "  Monitor:  squeue -j $JOB_ID,$COMPILE_ID"
echo "  Logs:     ${SCRIPT_DIR}/slurm_logs/${JOB_ID}_<task_id>.{out,err}"
echo "  Compile:  ${SCRIPT_DIR}/slurm_logs/${COMPILE_ID}_compile.{out,err}"
echo "  Status:   sacct -j $JOB_ID --format=JobID%30,State%15,Elapsed,MaxRSS,NodeList"
