#!/bin/bash
# Create first-frame ImageFolder train/test views for A* compiled data.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${1:?usage: make_probe_split.sh DATASET [TRAIN_FRAC]}"
TRAIN_FRAC="${2:-0.5}"

python "${REPO_ROOT}/a_star_data_collection_scripts/make_vpt_probe_split.py" \
  "${DATASET}" \
  --train_frac "${TRAIN_FRAC}" \
  --seed "${SEED:-42}" \
  --overwrite
