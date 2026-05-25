# Normal VPTnav Workflows

Use these wrappers for non-A* VPTnav data generation and dataset builds.

This directory owns:

- VPT1 normal v18 generation (`VPT-v18`)
- VPT1 depth v18 generation (`VPT-v18-Depth`)
- VPT2 generation (`VPT2-v1` through `VPT2-v4`)
- VPT1/VPT1-depth/VPT2 dataset build wrappers

A* rollout data lives under `scripts/a_star/`,
`VPTnav_code/cube_game/job_array/a_star/`, and
`a_star_data_collection_scripts/`. Do not mix A* compile scripts with normal
VPTnav datasets.

## Examples

```bash
cd /users/arock3/data/arock3/VPT

# Thesis VPT1 v18 generation
bash scripts/vptnav/submit_vpt1_v18.sh

# VPT1 v18 depth generation
bash scripts/vptnav/submit_vpt1_v18_depth.sh

# VPT2 v4 generation
bash scripts/vptnav/submit_vpt2.sh

# Build datasets after generation
bash scripts/vptnav/build_vpt1.sh
bash scripts/vptnav/build_vpt1_depth.sh
bash scripts/vptnav/build_vpt2.sh
```

All submit wrappers accept environment-variable overrides for `BASE_PATH`,
`NUM_GPUS`, `NUM_NODES`, `NUM_ENVS`, and `TASK`.
