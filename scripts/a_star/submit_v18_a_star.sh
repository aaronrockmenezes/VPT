#!/bin/bash
# Submit VPT-v18 A* rollout collection.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/VPTnav_code/cube_game/job_array/a_star"

bash submit_a_star_array.sh
