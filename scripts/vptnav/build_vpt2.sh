#!/bin/bash
# Build VPT2 dataset from generated VPT2 shards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python scripts/utils/build_vpt2_dataset.py
