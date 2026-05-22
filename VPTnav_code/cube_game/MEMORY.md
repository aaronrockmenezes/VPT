# VPTnav Memory

This file is durable cross-chat memory. New Codex chats should read this before touching the project.

## Current Priority

Continue A* dataset generation and prepare world-model training data. The A* dataset must start from a verified VPT-valid first-person viewpoint where frame 0 sees both goal and camera object.

Secondary active pipeline: `VPT-v18-camera-move` sweep collection (see `## Camera-Move Pipeline` below and `docs/camera_move_handoff.md`).

## Server Sync — PENDING

The camera-move pipeline was built locally; the Oscar server copy has drifted. **Do not
launch `VPT-v18-camera-move` on the cluster until `docs/server_sync_todo.md` is done.**
Local is the source of truth. Update this section once the server is in sync.

## Current Active Dataset

```text
BASE_PATH=/oscar/scratch/arock3/VPT_DATA_A_STAR/v18_data_collector_v1
mirror/path often used=/users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1
```

Recent status snapshot while run was still active:

```text
TOTAL successes: 5849
in_view     : 4350 / 7500
occluded    :  417 / 3750
outside_fov : 1082 / 3750
Max balanced dataset: 1668 envs
Bottleneck: occluded
```

Do not assume this is final; recount before acting.

## Current Submit Config

`VPTnav_code/cube_game/job_array/submit_a_star_array.sh` is configured for continuation:

```text
NUM_GPUS=4
FULL_TARGET=20000
ENVS_PER_GPU_TARGET=128
BUFFER_TASKS=10
NUM_ENVS=96
PLAN_WORKERS=1
SETTLE_STEPS=30
START_MODE=valid_viewpoint
START_HALF_EXTENT=6.0
START_DEADZONE=3.0
CAM_NO_RED_MAX=0
RESET_BASE_PATH=0
USE_GLOBAL_REWEIGHT=1
DYNAMIC_BALANCE_ALPHA=0.7
COMPILE_MIN_FRAMES=30
CPUS_PER_TASK=6
MEM=95G
TIME_PER_TASK=06:00:00
```

Never submit continuation jobs with `RESET_BASE_PATH=1`.

## A* Correctness Requirements

- Preserve v18 visual logic from `vpt_env.py`.
- Use `vpt_env_v17_alekh.py` only as reference for RL-style valid-start constraints.
- Start A* from a verified valid viewpoint, not the initial reset pose.
- First saved agent frame must see goal and camera object.
- Start must satisfy camera-centered `12x12 m` square with `3x3 m` deadzone rejection.
- Camera POV label validation: `Yes > 125` red pixels at 256x256; `No <= 0` strict-red pixels.
- Final yaw target is corrected camera sensor yaw, not raw camera object yaw.

## Important Files

```text
VPTnav_code/cube_game/docs/wiki.md
VPTnav_code/cube_game/docs/a_star_pipeline.md
VPTnav_code/cube_game/docs/camera_move_handoff.md
VPTnav_code/cube_game/docs/server_sync_todo.md
VPTnav_code/cube_game/docs/world_model_handoff.md
VPTnav_code/cube_game/docs/commands.md
VPTnav_code/cube_game/docs/server_tree.md
VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env.py
VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_A_star.py
VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera_move.py
VPTnav_code/cube_game/scripts/A_star_data_collector.py
VPTnav_code/cube_game/scripts/keyboard_agent.py
VPTnav_code/cube_game/job_array/submit_a_star_array.sh
VPTnav_code/cube_game/job_array/submit_generation.sh
```

## Camera-Move Pipeline

`VPT-v18-camera-move` — fixed agent, camera object sweeps a fixed-radius right-half arc
around the goal. Captures agent POV + camera-object POV at each angle, labels `Yes`/`No`.

- Angles `0..180°` (`range(0, 181, 15)`, 13 angles), measured from the **goal→agent
  direction**: `0°` collinear on the agent's side (camera looks at goal, same dir as agent),
  `90°` agent's right, `180°` collinear diametrically opposite.
- `radius = min(initial_cam_to_goal_dist, agent_to_goal_dist)`.
- Per-angle drop filters: out-of-bounds, outside agent FOV (half-HFOV minus `7°` buffer),
  OBB collision, agent-view-invalid.
- Label = strict-red pixel count in the camera-object's occlusion-camera semantic seg.
  Stored in BOTH the filename (`image_{angle}d_{Yes|No}.png`) and the config JSON
  (`camera_move_collection.trajectory[].label`).
- No per-env 50/50 balance; global balance is a future carve-time step.
- Collection is batch-parallel (`_build_fixed_sweep_trajectory_batch`): all envs processed
  per angle together, ~`num_envs`× fewer sim steps.
- Runs via `submit_generation.sh` → `generation_worker.sh` (overlay-pool apptainer) →
  `multi_gpu.sh` → `launcher.py` → `keyboard_agent.py` (camera-move branch sends `action=5`).
- Output: `{BASE_PATH}/data/data_node{NODE_ID}_gpu{G}/{RGB,Depth,Semantic,cam_Semantic,cam_RGB,cam_RGB_norm}/Mixed/env_{N}/` + `configs/env_{N}_config.json`.
- Monitor: `scripts/monitor_camera_move.py`, `scripts/count_saved_envs.py`.

Full handoff: `docs/camera_move_handoff.md`. Server sync checklist: `docs/server_sync_todo.md`.

## World Model Plan

Use compiled A* rollouts for JEPA/world-model training. Minimal payload should be RGB frames, actions, and metadata. Use WebDataset shard conversion after final compile.

Primary handoff: `docs/world_model_handoff.md`.

## Known Operational Notes

- Existing `vpt_env` conda/PyTorch stack does not support B200; use H100 for linear probes unless rebuilding.
- Dynamic balancing improves category request mix but cannot force occluded yield if scene constraints make occlusion rare.
- Active SLURM jobs may leave raw GPU dirs if compile fails. Count raw plus compiled before deleting anything.
- Do not physically move scripts while jobs are running; document server mappings instead.
