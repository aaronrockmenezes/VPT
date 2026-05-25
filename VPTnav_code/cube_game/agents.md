# agents.md — VPTnav Cube Game

Keep this file as the agent-facing project map. For detailed handoff, read:

- `MEMORY.md`
- `docs/wiki.md`
- `docs/a_star_pipeline.md`
- `docs/camera_move_handoff.md`
- `docs/server_sync_todo.md`
- `docs/world_model_handoff.md`
- `docs/commands.md`
- `docs/server_tree.md`

## Current Priority

Continue and validate the v18 A* dataset for world-model training. The dataset must preserve v18 VPT visual logic and use verified VPT-valid first-frame starts.

Secondary active pipeline: `VPT-v18-camera-move` sweep collection — see
`docs/camera_move_handoff.md`. **The Oscar server is not yet synced with the local
camera-move changes — `docs/server_sync_todo.md` must be completed before launching it
on the cluster.**

## Do Not Assume Cross-Chat Memory

All durable state should be written to markdown in this repo. A new chat should read `AGENTS.md`, this file, `MEMORY.md`, and `docs/world_model_handoff.md` before making changes.

## Active Server Paths

```text
/users/arock3/data/arock3/VPT/VPTnav_code/cube_game
/users/arock3/data/arock3/VPT/a_star_data_collection_scripts
/oscar/scratch/arock3/VPT_DATA_A_STAR/v18_data_collector_v1
/users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1
```

## Main Environments

| File | Use |
|---|---|
| `source/cube_game/cube_game/tasks/direct/cube_game/vpt_env.py` | V18 visual source of truth. |
| `source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_A_star.py` | A* rollout environment. Must use v18 visuals plus valid-viewpoint starts. |
| `source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera_move.py` | Camera-move sweep environment (`VPT-v18-camera-move`). Fixed agent, camera sweeps a right-half arc around the goal. See `docs/camera_move_handoff.md`. |
| `source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v17_alekh.py` | Historical/reference env. Use only for RL constraint ideas, not visual distribution. |

## Main Scripts

| File | Use |
|---|---|
| `scripts/A_star_data_collector.py` | Production A* rollout collector. |
| `scripts/keyboard_agent.py` | Generation driver. Camera-move branch: `"camera-move"` in task name → sends `action=5` (soft reset) every step. |
| `scripts/compile_tasks.py` | Per-node validation/staging/cleanup. |
| `scripts/compile_all_nodes.py` | Compile existing raw node dirs. |
| `scripts/count_successful_envs.py` | Count raw/compiled A* collection pool. |
| `scripts/count_saved_envs.py` | Count successfully saved camera-move envs (per shard + total). |
| `scripts/monitor_camera_move.py` | Camera-move Yes/No progress + 50/50 feasibility monitor. |
| `scripts/compile_a_star_dataset.py` | Final canonical dataset compiler. |
| `scripts/sanity_check_compiled_a_star.py` | Final dataset sanity checker. |
| `scripts/carve_astar_vpt1_firstframe_probe.py` | Fast first-frame VPT probe carveout. |
| `scripts/compile_a_star_webdataset.py` | JEPA/world-model WebDataset conversion. |
| `job_array/a_star/submit_a_star_array.sh` | Active SLURM submit file (A*). |
| `job_array/normal_vptnav/submit_generation.sh` | SLURM submit file for normal VPTnav / camera-move generation runs. |

## A* Data Contract

A valid episode must satisfy:

- `start_source == valid_viewpoint_0` or equivalent verified valid viewpoint metadata.
- First saved agent frame sees both camera object and goal.
- Start is within `abs(dx), abs(dy) <= 6.0` from camera object.
- Start is outside the `abs(dx) < 3.0 and abs(dy) < 3.0` deadzone.
- `No` camera POV has strict red pixel count `<= CAM_NO_RED_MAX`, currently `0`.
- `Yes` camera POV red pixel count is above threshold, currently `125` at `256x256`.
- Final yaw uses corrected camera sensor yaw.

## Dynamic Collection Balance

Final dataset balance is enforced at compile time. During generation, category requests can be reweighted:

```text
USE_GLOBAL_REWEIGHT=1
DYNAMIC_BALANCE_ALPHA=0.7
FRAC_IN_VIEW=0.50
FRAC_OCCLUDED=0.25
FRAC_OUTSIDE_FOV=0.25
```

Set `DYNAMIC_BALANCE_ALPHA=1.0` only for severe catch-up/top-up runs.

## Current Risk

Occluded yield is low under the corrected first-frame-valid constraints. Keep monitoring with `count_successful_envs.py`; final compile will bottleneck on occluded.

## Camera-Move Pipeline

`VPT-v18-camera-move`: fixed agent, camera object sweeps `0..180°` around the goal (angle
measured from the goal→agent direction; `0°` collinear on the agent's side, `90°` agent's
right, `180°` collinear opposite). Each angle captured from agent POV + camera-object POV,
labeled `Yes`/`No` (goal in_view / occluded from camera POV). Label is in the filename
(`image_{angle}d_{Yes|No}.png`) and the config JSON. No per-env balance; global balance is
a future carve-time step.

Runs through `job_array/normal_vptnav/submit_generation.sh` →
`generation_worker.sh` → `multi_gpu.sh` → `launcher.py` →
`keyboard_agent.py`. Full handoff: `docs/camera_move_handoff.md`.

## TODO

- **Sync the Oscar server with the local camera-move changes** before launching
  `VPT-v18-camera-move` on the cluster. Checklist: `docs/server_sync_todo.md`.
- `VPT-v18-strategy`: `SETTLE_STEPS` default is temporarily `3` for fast smoke tests. Restore/tune to about `30` before production collection to avoid render/lighting artifacts.
- Camera-move: global `in_view`/`occluded` 50/50 balance is not enforced at collection. Use `monitor_camera_move.py` and carve a balanced subset afterward.

## Operational Rules

- Do not submit with `RESET_BASE_PATH=1` unless intentionally wiping the run.
- Do not move server scripts while SLURM jobs are running.
- Do not use B200 for the old linear-probe PyTorch env; use H100 unless the environment is rebuilt.
- Before deleting raw dirs, verify compiled output and recount.
- `job_array/a_star/submit_a_star_array.sh` derives SLURM array concurrency from `MAX_TOTAL_CPUS=120` and `MAX_TOTAL_GPUS=60`; with `CPUS_PER_TASK=12` and `NUM_GPUS=8`, this submits at most 7 concurrent array tasks.
