#!/bin/bash
# Sanity-check a final compiled A* dataset.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPILED_PATH="${1:?usage: sanity_check.sh COMPILED_PATH}"

python "${REPO_ROOT}/a_star_data_collection_scripts/sanity_check_compiled_a_star.py" \
  "${COMPILED_PATH}" \
  --deep \
  --count_images \
  --cam_check
