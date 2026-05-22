# VPTnav Project Wiki

This wiki is the durable handoff layer for Codex chats. It tracks the current A* dataset collection, VPT linear-probe bridge, and world-model direction.

## Current Objective

Build a high-quality A* rollout dataset where the agent starts from a VPT-valid first-person viewpoint: the agent's first frame sees both the red goal and the green camera object. The rollout then navigates to the camera object's pose and aligns to the corrected camera yaw. This dataset is intended for JEPA/world-model training and for VPT linear-probe experiments.

## Secondary Pipeline: Camera-Move Sweep

`VPT-v18-camera-move` — a separate collection mode where the agent is fixed and a camera
object sweeps a fixed-radius right-half arc (`0..180°`) around the goal, capturing agent
POV + camera-object POV at each angle and labeling `Yes`/`No`. Built and run locally; the
Oscar server is **not yet synced**. Full handoff in `docs/camera_move_handoff.md`; sync
checklist in `docs/server_sync_todo.md`.

## Current Server Layout

See also `docs/server_tree.md` for the exact VS Code server tree.

Primary server paths used in active work:

```text
/users/arock3/data/arock3/VPT/
├── VPTnav_code/cube_game/
│   ├── source/cube_game/cube_game/tasks/direct/cube_game/
│   │   ├── vpt_env.py
│   │   ├── vpt_env_v18_A_star.py
│   │   └── vpt_env_v17_alekh.py
│   ├── scripts/
│   │   ├── A_star_data_collector.py
│   │   ├── compile_a_star_dataset.py
│   │   ├── compile_a_star_webdataset.py
│   │   ├── compile_all_nodes.py
│   │   ├── compile_tasks.py
│   │   ├── count_successful_envs.py
│   │   ├── sanity_check_compiled_a_star.py
│   │   ├── audit_cam_deadzone.py
│   │   └── carve_astar_vpt1_firstframe_probe.py
│   └── job_array/
│       ├── submit_a_star_array.sh
│       ├── a_star_worker.sh
│       ├── a_star_multi_gpu.sh
│       └── a_star_launcher.py
├── a_star_data_collection_scripts/
│   └── optional operational mirror of the scripts above
└── scripts/
    ├── launch.sh
    ├── submit_array.sh
    ├── config.sh
    └── analysis_args.py
```

Scratch paths used in recent runs:

```text
/oscar/scratch/arock3/VPT_DATA_A_STAR/v18_data_collector_v1
/users/arock3/scratch/VPT_DATA_A_STAR/v18_data_collector_v1
/users/arock3/scratch/VPT_DATA_A_STAR/v18_vpt1_probe_1024_firstframe_v2
/users/arock3/scratch/VPT_logs/perspective_astar/v18_vpt1_probe_1024_firstframe_v2/lp_logs
```

`/oscar/scratch` and `/users/arock3/scratch` may resolve to the same storage namespace on the cluster depending on node/mount.

## Local Repo Layout

```text
VPT/
├── AGENTS.md
├── VPTnav_code/cube_game/
│   ├── README.md
│   ├── agents.md
│   ├── MEMORY.md
│   ├── docs/
│   │   ├── wiki.md
│   │   ├── a_star_pipeline.md
│   │   ├── world_model_handoff.md
│   │   └── commands.md
│   ├── job_array/
│   ├── scripts/
│   └── source/cube_game/cube_game/tasks/direct/cube_game/
├── a_star_data_collection_scripts/
├── VPT_code/VPT/
└── scripts/
```

## Important Source Files

| File | Role |
|---|---|
| `source/.../vpt_env.py` | V18 visual/data-generation truth. Contains camera/goal/object randomization, viewpoint generation, yaw jitter, camera-object orientation/sensor correction assumptions. |
| `source/.../vpt_env_v18_A_star.py` | A* variant. Should preserve v18 visual logic while using one valid VPT viewpoint as the A* start. |
| `source/.../vpt_env_v18_camera_move.py` | Camera-move sweep variant (`VPT-v18-camera-move`). Fixed agent, camera sweeps a right-half arc around the goal. See `docs/camera_move_handoff.md`. |
| `source/.../vpt_env_v17_alekh.py` | Reference only. Useful for older RL constraints and safe placement patterns, not visual truth. |
| `scripts/A_star_data_collector.py` | Production collector. Spawns valid starts, uses cached A* plans, saves rollouts. |
| `job_array/submit_a_star_array.sh` | SLURM submit config for active A* collection. |
| `scripts/compile_tasks.py` | Per-array-task validation, staging, and cleanup. |
| `scripts/compile_a_star_dataset.py` | Final canonical dataset compiler. |
| `scripts/carve_astar_vpt1_firstframe_probe.py` | Fast 1024-env first-frame VPT linear-probe carveout. |
| `scripts/compile_a_star_webdataset.py` | Converts compiled canonical dataset to JEPA-friendly sharded WebDataset with `.npy` payloads. |

## Current A* Collection Requirements

Hard requirements for valid A* data:

- First saved agent frame must see both goal and camera object.
- A* start must come from `valid_viewpoint_0`, not initial reset pose.
- Valid start must satisfy camera-centered square constraints: `abs(dx) <= 6`, `abs(dy) <= 6`.
- Reject start deadzone: `abs(dx) < 3` and `abs(dy) < 3`.
- Camera POV label check: `Yes` means red pixels `> 125` at 256x256; `No` means strict red pixels `<= 0` unless intentionally relaxed.
- Final alignment target must use corrected camera sensor yaw, not raw camera-object yaw.
- Invalid starts retry and must not be saved as almost-valid examples.

## Dataset Ratios

Final target remains globally balanced by reason:

```text
in_view     50%
occluded    25%
outside_fov 25%
```

Collection can be dynamically reweighted to catch up underrepresented categories. This only changes future requested categories; it does not relabel data or affect final compile quality.

Relevant flags:

```text
USE_GLOBAL_REWEIGHT=1
DYNAMIC_BALANCE_ALPHA=0.7  # 0=final deficits only, 1=current-ratio catch-up only
FRAC_IN_VIEW=0.50
FRAC_OCCLUDED=0.25
FRAC_OUTSIDE_FOV=0.25
```

## Current Collection Status Snapshot

Recent `count_successful_envs.py` output for `v18_data_collector_v1`:

```text
TOTAL successes: 5849
in_view     : 4350 / 7500
occluded    :  417 / 3750
outside_fov : 1082 / 3750
Max balanced dataset: 1668 envs
Bottleneck: occluded
```

This run was still active when this handoff was written. Occluded was improving, so do not assume final counts from this snapshot.

## Linear Probe Snapshot

A fast 1024-env first-frame probe carveout was created to test whether first-frame A* images solve VPT1. Results were not strong enough to declare solved. Important conclusion: the current first-frame distribution still needs careful validation and likely world-model/VPT bridging rather than relying on probe performance alone.

Probe carve command pattern:

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

## World Model Direction

The world model should train on A* rollouts where frame 0 has both goal and camera visible. Input is the agent view; action sequence should enable prediction/navigation toward the camera object's view. For JEPA-style training, the useful minimal payload is:

```text
frames.npy   # RGB rollout frames, uint8
actions.npy  # action ids from actions.txt
meta.json    # env metadata, reason, split, source, counts, start info
```

Use `compile_a_star_webdataset.py` after final compile to create sharded data.

## Non-Negotiable Safety Notes

- Do not run `submit_a_star_array.sh` with `RESET_BASE_PATH=1` unless intentionally deleting the current run.
- Do not move server-side scripts during active SLURM runs.
- Do not assume B200 works with the existing `vpt_env` PyTorch stack; use H100 for linear probes unless the env is rebuilt.
- Do not rely on hidden cross-chat memory; read `MEMORY.md` and `docs/world_model_handoff.md`.
