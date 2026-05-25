#!/bin/bash
# Count A* raw/staged pool and report balanced capacity.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_PATH="${1:?usage: count_pool.sh BASE_PATH [TARGET]}"
TARGET="${2:-20000}"

python "${REPO_ROOT}/a_star_data_collection_scripts/count_successful_envs.py" \
  "${BASE_PATH}" \
  --target "${TARGET}"
