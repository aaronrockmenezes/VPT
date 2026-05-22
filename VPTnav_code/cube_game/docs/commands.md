# Command Reference

## Camera-Move Collection (`VPT-v18-camera-move`)

Edit `job_array/submit_generation.sh` config block (`TASK="VPT-v18-camera-move"`,
`NUM_ENVS`, `BASE_PATH`, `NUM_NODES`, `NUM_GPUS`), then:

```bash
cd /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/job_array
bash submit_generation.sh
```

Local single-GPU dry run:

```bash
isaaclab.sh -p scripts/keyboard_agent.py --task VPT-v18-camera-move --num_envs 8
```

Monitor Yes/No balance + 50/50 feasibility:

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/monitor_camera_move.py \
  --base_path /oscar/scratch/arock3/VPT1_DATA/camera/v18_4 \
  --target 100
# add --watch 30 for a live-refreshing view
```

Count successfully saved envs:

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/count_saved_envs.py \
  --base_path /oscar/scratch/arock3/VPT1_DATA/camera/v18_4 \
  --verify
```

> Server not yet synced with the local camera-move changes — see `docs/server_sync_todo.md`.

## Continue A* Generation

```bash
cd /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/job_array
bash submit_a_star_array.sh
```

Before running continuation jobs:

```bash
grep -E 'BASE_PATH|RESET_BASE_PATH|DYNAMIC_BALANCE_ALPHA|FULL_TARGET|BUFFER_TASKS|NUM_GPUS|CPUS_PER_TASK|TIME_PER_TASK' submit_a_star_array.sh
```

`RESET_BASE_PATH` must be `0` unless intentionally wiping the run.

## Count Collection Pool

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/count_successful_envs.py \
  /users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1
```

## Carve 1024 First-Frame VPT Probe

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/carve_astar_vpt1_firstframe_probe.py \
  --src_root /users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1 \
  --out_dir /users/arock3/scratch/VPT_DATA_A_STAR/v18_vpt1_probe_1024_firstframe_v2 \
  --total 1024 \
  --mode raw \
  --workers 32 \
  --seed 0 \
  --overwrite
```

Expected first-frame split image counts:

```bash
find /users/arock3/scratch/VPT_DATA_A_STAR/v18_vpt1_probe_1024_firstframe_v2/train -name '*.png' | wc -l
find /users/arock3/scratch/VPT_DATA_A_STAR/v18_vpt1_probe_1024_firstframe_v2/test -name '*.png' | wc -l
```

Both should be `512`.

## Run Linear Probe

Use H100 unless PyTorch is rebuilt for B200.

```bash
cd /users/arock3/data/arock3/VPT/scripts
./launch.sh
```

Reason-wise analysis:

```bash
python /users/arock3/data/arock3/VPT/scripts/analysis_args.py \
  --dataset_path /users/arock3/scratch/VPT_DATA_A_STAR/v18_vpt1_probe_1024_firstframe_v2 \
  --folder_search "/users/arock3/scratch/VPT_logs/perspective_astar/v18_vpt1_probe_1024_firstframe_v2/lp_logs/results/*_preds.csv"
```

## Final Compile

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/compile_a_star_dataset.py \
  --src_root /users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1 \
  --out_dir /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k \
  --total 20000 \
  --mode staged \
  --workers 64 \
  --seed 0
```

## Sanity Check Final Compile

```bash
python /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/scripts/sanity_check_compiled_a_star.py \
  /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k \
  --deep \
  --count_images \
  --breakdown
```

## Convert To WebDataset

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
