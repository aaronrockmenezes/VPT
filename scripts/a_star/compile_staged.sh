#!/bin/bash
# Final A* balanced compile from staged data_node*_compiled dirs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_ROOT="${1:?usage: compile_staged.sh SRC_ROOT OUT_DIR [TOTAL]}"
OUT_DIR="${2:?usage: compile_staged.sh SRC_ROOT OUT_DIR [TOTAL]}"
TOTAL="${3:-20000}"

python "${REPO_ROOT}/a_star_data_collection_scripts/compile_a_star_dataset.py" \
  --src_root "${SRC_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --total "${TOTAL}" \
  --mode staged \
  --min_frames "${MIN_FRAMES:-30}" \
  --workers "${WORKERS:-64}" \
  --seed "${SEED:-0}" \
  ${OVERWRITE_OUT:+--overwrite_out}
