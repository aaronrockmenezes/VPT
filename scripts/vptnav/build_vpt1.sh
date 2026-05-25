#!/bin/bash
# Build normal VPT1 dataset from generated VPT-v18 shards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python scripts/utils/build_vpt1_dataset.py
