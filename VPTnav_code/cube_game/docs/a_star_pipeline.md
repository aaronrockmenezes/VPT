# A* Dataset Pipeline

## Generation

Entry point:

```bash
cd /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/job_array/a_star
bash submit_a_star_array.sh
```

Current continuation-safe submit settings:

```text
BASE_PATH=/oscar/scratch/arock3/VPT_DATA_A_STAR/v18_data_collector_v1
NUM_GPUS=8
FULL_TARGET=30000
ENVS_PER_GPU_TARGET=48
BUFFER_TASKS=5
NUM_ENVS=96
PLAN_WORKERS=1
SETTLE_STEPS=30
START_MODE=valid_viewpoint
START_HALF_EXTENT=6.0
START_DEADZONE=3.0
CAM_NO_RED_MAX=0
RESET_BASE_PATH=0
USE_GLOBAL_REWEIGHT=0
DYNAMIC_BALANCE_ALPHA=0.9
FRAC_IN_VIEW=0.25
FRAC_OCCLUDED=0.5
FRAC_OUTSIDE_FOV=0.25
COMPILE_MIN_FRAMES=30
CPUS_PER_TASK=12
MEM=180G
TIME_PER_TASK=06:00:00
MAX_TOTAL_CPUS=120
MAX_TOTAL_GPUS=60
```

The job array launches one task per node. Each task runs one process per GPU through `a_star_launcher.py`; each GPU writes raw data under:

```text
{BASE_PATH}/data/data_node{SLURM_ARRAY_JOB_ID}_{SLURM_ARRAY_TASK_ID}_gpu{GPU_ID}/
```

After the container exits successfully, `a_star_worker.sh` runs `compile_tasks.py` and writes staged/compiled node output:

```text
{BASE_PATH}/data/data_node{SLURM_ARRAY_JOB_ID}_{SLURM_ARRAY_TASK_ID}_compiled/
```

If compile fails, raw data is left in place.

## Dynamic Category Balancing

Collector category requests are adapted from current saved counts.

- `GLOBAL_TARGET` tells each process the final target count.
- `FRAC_*` define the desired final reason distribution.
- `DYNAMIC_BALANCE_ALPHA` blends between final remaining deficits and current-ratio catch-up.

Recommended values:

```text
0.0 = use only final deficits
0.5 = moderate catch-up
0.7 = aggressive catch-up for skewed pools
1.0 = maximum catch-up for severe bottlenecks
```

This only affects what future env categories are requested. Final compile still enforces exact balance.

## Counting Current Pool

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/count_successful_envs.py \
  /users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1
```

Use `--compiled_only` if inspecting only staged/compiled node dirs.

## Final Canonical Compile

Canonical output tree:

```text
out_dir/
├── master_labels.json
├── RGB/{Yes,No}/env_*/
├── Semantic/{Yes,No}/env_*/
└── cam/{Yes,No}/env_*/
```

Compile command pattern:

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/compile_a_star_dataset.py \
  --src_root /users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1 \
  --out_dir /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k \
  --total 20000 \
  --mode staged \
  --workers 64 \
  --seed 0
```

If raw GPU dirs remain and staged dirs are incomplete, use `--mode raw`.

## Sanity Check Final Dataset

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/sanity_check_compiled_a_star.py \
  /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k \
  --deep \
  --count_images \
  --breakdown
```

Expected checks:

- `master_labels.json` record count matches env dirs.
- Split metadata is continuous by env id.
- `RGB`, `Semantic`, `cam` dirs exist for every env.
- Image counts match expected totals from metadata.
- Reason/label image breakdown is reported.

## WebDataset Conversion For World Model

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/compile_a_star_webdataset.py \
  --root /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k \
  --out_dir /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k_wds \
  --total 20000 \
  --num_shards 100 \
  --episodes_per_shard 200 \
  --format npy \
  --workers 32 \
  --seed 0
```

Each shard should contain approximately:

```text
100 in_view
 50 occluded
 50 outside_fov
```

Payload per env:

```text
{key}.frames.npy
{key}.actions.npy
{key}.meta.json
{key}.json
```

Run the matching sanity checker after conversion.
