#!/bin/bash
#SBATCH --job-name=lp-compile
#SBATCH --output=slurm_logs/%A_compile.out
#SBATCH --error=slurm_logs/%A_compile.err
#SBATCH --account=carney-tserre-condo2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00

# =============================================================================
# Load config — LP_SCRIPT_DIR is exported by launch.sh
# =============================================================================
SCRIPT_DIR="${LP_SCRIPT_DIR:?'ERROR: LP_SCRIPT_DIR not set. Run via launch.sh, not directly.'}"
source "${SCRIPT_DIR}/config.sh"

COMPILE_SCRIPT="${SCRIPT_DIR}/compile_results.py"

# Activate your env — uncomment/adjust (must match submit_array.sh):
# source /path/to/conda/etc/profile.d/conda.sh && conda activate vpt
# module load cuda/12.x

echo "============================================"
echo "Compiling results: $(date)"
echo "  Task:       $TASK"
echo "  Num runs:   $NUM_RUNS"
echo "  Output dirs: ${#output_dirs[@]}"
echo "============================================"

for i in "${!output_dirs[@]}"; do
    OUTPUT_DIR="${output_dirs[$i]}"
    RESULTS_DIR="${OUTPUT_DIR}/results"

    echo ""
    echo "--- Compiling: $OUTPUT_DIR ---"

    if [ ! -d "$RESULTS_DIR" ]; then
        echo "  WARNING: $RESULTS_DIR does not exist, skipping."
        continue
    fi

    python "$COMPILE_SCRIPT" \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_runs "$NUM_RUNS"
done

echo ""
echo "All compilations done at $(date)"