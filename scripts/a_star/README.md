# A* Rollout Workflows

Use these wrappers only for A* rollout data. Normal VPTnav VPT1/VPT2 generation
lives under `scripts/vptnav/`.

This directory owns:

- A* SLURM submission
- A* pool counting
- A* staged/raw compile
- A* final sanity checks
- A* first-frame probe carveouts

The underlying implementation scripts live in `a_star_data_collection_scripts/`
and `VPTnav_code/cube_game/job_array/a_star/`.

## Examples

```bash
cd /users/arock3/data/arock3/VPT

bash scripts/a_star/submit_v18_a_star.sh
bash scripts/a_star/count_pool.sh /users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1 20000
bash scripts/a_star/compile_staged.sh /users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1 /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k 20000
bash scripts/a_star/sanity_check.sh /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k
```
