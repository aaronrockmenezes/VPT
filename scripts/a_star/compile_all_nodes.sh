#!/bin/bash
# Compile raw A* per-GPU node dirs into staged data_node*_compiled dirs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_PATH="${1:?usage: compile_all_nodes.sh BASE_PATH}"

python "${REPO_ROOT}/a_star_data_collection_scripts/compile_all_nodes.py" \
  --base_path "${BASE_PATH}" \
  --gpus ${GPUS:-0 1 2 3 4 5 6 7} \
  --min_frames "${MIN_FRAMES:-30}" \
  --workers "${WORKERS:-16}" \
  --node_workers "${NODE_WORKERS:-2}" \
  ${OVERWRITE:+--overwrite}
