#!/bin/bash
# Build normal VPT1 depth dataset from generated VPT-v18-Depth shards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python scripts/utils/build_depth_dataset.py
