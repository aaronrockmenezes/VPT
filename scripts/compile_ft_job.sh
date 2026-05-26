#!/bin/bash
#SBATCH --job-name=ft-compile
#SBATCH --output=slurm_logs/%A_ft_compile.out
#SBATCH --error=slurm_logs/%A_ft_compile.err
#SBATCH --account=carney-tserre-condo2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00

set -euo pipefail

SCRIPT_DIR="${FT_SCRIPT_DIR:?'ERROR: FT_SCRIPT_DIR not set. Run via launch_ft.sh, not directly.'}"
source "${SCRIPT_DIR}/ft_config.sh"

COMPILE_SCRIPT="${SCRIPT_DIR}/compile_results.py"

echo "============================================"
echo "Compiling FT results: $(date)"
echo "  Task:       $TASK"
echo "  Num runs:   $FT_NUM_RUNS"
echo "  Output dirs: ${#output_dirs[@]}"
echo "============================================"

for i in "${!output_dirs[@]}"; do
    OUTPUT_DIR="${output_dirs[$i]}"
    RESULTS_DIR="${OUTPUT_DIR}/results"

    echo ""
    echo "--- Compiling FT: $OUTPUT_DIR ---"

    if [ ! -d "$RESULTS_DIR" ]; then
        echo "  WARNING: $RESULTS_DIR does not exist, skipping."
        continue
    fi

    python "$COMPILE_SCRIPT" \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_runs "$FT_NUM_RUNS"
done

echo ""
echo "All FT compilations done at $(date)"
