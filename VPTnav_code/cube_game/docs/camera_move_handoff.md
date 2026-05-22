# Camera-Move Pipeline Handoff

Handoff doc for the `VPT-v18-camera-move` data collection pipeline. A new chat should
read this before touching anything camera-move related. For the A* pipeline, see
`docs/a_star_pipeline.md` and `docs/world_model_handoff.md` instead.

## Purpose

Generate a dataset where the **agent is fixed** and a **camera object sweeps a fixed-radius
arc around the goal**. At each sweep angle the env captures the agent's POV and the camera
object's POV, and labels the frame `Yes`/`No` depending on whether the goal is visible from
the camera object's POV.

This is a different question from A*: instead of navigating to the camera, we directly
sample many camera viewpoints per scene and ask "can the camera see the goal from here?".

## Angle Convention (important — was buggy, now fixed)

The sweep angle is measured from the **goal → agent direction**, not the agent's heading:

- `base_angle = atan2(agent_xy - goal_xy)`
- `world_angle = base_angle + angle_deg`
- `camera_xy = goal_xy + radius * (cos(world_angle), sin(world_angle))`

| angle | camera position | meaning |
|---|---|---|
| `0°`   | collinear, on the agent's side of the goal | camera looks at the goal in the **same direction** the agent looks |
| `90°`  | on the agent's **right** | side view |
| `180°` | collinear, diametrically opposite the agent across the goal | camera and agent both look at the goal from opposite sides |

Angles swept: `0, 15, 30, ..., 180` (13 angles, `range(0, 181, 15)`). Right semicircle only.

`radius = min(initial_camera_to_goal_dist, agent_to_goal_dist)` so the camera circle never
places the camera behind the agent.

## Per-Angle Filtering

For each angle, the camera position is dropped (not saved) if any of:

1. **Out of bounds** — outside the env's `center_to_boundary` square.
2. **Outside agent FOV** — camera center must be within the agent's horizontal FOV minus a
   `7°` buffer (`_agent_fov_buffer_deg`), so the camera object does not clip the agent's
   frame edge. `_agent_half_hfov_deg` is derived from the camera cfg aperture/focal.
3. **OBB collision** — camera object's oriented bounding box collides with an obstacle.
4. **Agent view invalid** — after placement, the agent must still see both goal and camera
   object (batch pixel check).

Surviving angles are captured and labeled. The default local collection mode now keeps
only balanced saved frames per accepted env: at least one `Yes` and one `No`, then an
equal number of each. This makes the saved-frame dataset 50/50 by construction, but it
does not increase occluded yield; all-visible scenes are rejected and retried.

## Labeling

Each saved frame is labeled by `_camera_goal_visible_from_current_pov`: counts strict-red
pixels in the occlusion camera's semantic segmentation. `red_count >= goal_pixel_threshold_occlusion`
→ `Yes` (`in_view`), else `No` (`occluded`).

The label is stored in **two places**:
- the **filename**: `image_{angle}d_{Yes|No}.png`
- the **config JSON**: `camera_move_collection.trajectory[].label` (+ `reason`, `red_count`).

## Execution Flow

```
submit_generation.sh                  # edit config block: TASK, NUM_ENVS, BASE_PATH, NUM_NODES, NUM_GPUS
   │  sbatch --array, --export=...
   ▼
generation_worker.sh                  # SLURM worker — overlay-pool apptainer activation
   │  apptainer exec --cleanenv --overlay (32-slot pool, per-task CACHE_ROOT)
   ▼
multi_gpu.sh                          # in-container — picks AGENT_SCRIPT, forwards --num_envs
   ▼
launcher.py                           # spawns NUM_GPUS subprocesses, one per GPU, 30s stagger
   ▼
keyboard_agent.py                     # camera-move branch: sends action=5 to all envs every step
   ▼
vpt_env_v18_camera_move.py            # action=5 → soft reset → _reset_idx(rl_reset=False) → collection
```

`keyboard_agent.py` auto-detects the task: if `"camera-move"` is in the task name it skips
the keyboard/planner logic and just sends `action=5` (soft reset) to every env each step.
No new collector script — the old `camera_move_collector.py` was removed.

## Batch / Parallel Collection

`_build_fixed_sweep_trajectory_batch` processes **all envs simultaneously for each sweep
angle**: vectorized bounds/FOV/OBB filters, one batch camera placement, one 30-step settle
per angle for the whole batch, batch agent-view check, then per-slot pixel read + tensor
clone into `self._cached_frames`. `_reset_idx` then writes those cached tensors to disk via
`_save_slot_from_cache` with zero extra sim steps.

Result: ~`num_envs`× fewer sim steps per reset cycle (e.g. 8 envs, 13 angles: ~4160 → ~390
sim steps). The old per-env sequential path (`_build_fixed_sweep_trajectory`,
`_collect_images_for_slot`) is kept as a fallback but is not used when the cache is present.

## Output Layout

```
{BASE_PATH}/data/data_node{NODE_ID}_gpu{GPU_ID}/
├── RGB/Mixed/env_{N}/image_{angle}d_{Yes|No}.png            # agent POV
├── Depth/Mixed/env_{N}/image_{angle}d_{Yes|No}.png
├── Semantic/Mixed/env_{N}/image_{angle}d_{Yes|No}.png
├── cam_Semantic/Mixed/env_{N}/image_{angle}d_{Yes|No}.png   # camera-object POV
├── cam_RGB/Mixed/env_{N}/image_{angle}d_{Yes|No}.png
├── cam_RGB_norm/Mixed/env_{N}/image_{angle}d_{Yes|No}.png
└── configs/env_{N}_config.json
```

`NODE_ID = {sbatch_job}_{array_task}` (composite, unique across submissions). All angles for
an env go into the same `Mixed` dir — the per-frame label is in the filename and JSON, not
the path. Camera-object POV gets three own dirs (was previously one flat `cam/`).

## How To Run

Edit `job_array/submit_generation.sh` config block:

```bash
BASE_PATH="/oscar/scratch/arock3/VPT1_DATA/camera/v18_4"
NUM_GPUS=4
TASK="VPT-v18-camera-move"
NUM_NODES=8
NUM_ENVS=32          # parallel envs per GPU — tunable
```

Then:

```bash
cd /users/arock3/data/arock3/VPT/VPTnav_code/cube_game/job_array
bash submit_generation.sh
```

It runs until the SLURM time limit kills it (no `total_envs` stop condition).

## Monitoring

```bash
# Yes/No balance: overall, per-env avg, 50/50 feasibility for an N-env target
python scripts/monitor_camera_move.py --base_path <BASE_PATH> --target 100
python scripts/monitor_camera_move.py --base_path <BASE_PATH> --watch 30   # live refresh

# Count successfully saved envs (config JSON present == complete env)
python scripts/count_saved_envs.py --base_path <BASE_PATH>
python scripts/count_saved_envs.py --base_path <BASE_PATH> --verify         # cross-check images
```

## Key Files

| File | Role |
|---|---|
| `source/.../tasks/direct/cube_game/vpt_env_v18_camera_move.py` | The camera-move env. All collection logic lives here. |
| `source/.../tasks/direct/cube_game/__init__.py` | Registers gym id `VPT-v18-camera-move`. |
| `scripts/keyboard_agent.py` | Driver — camera-move branch sends `action=5` every step. |
| `job_array/submit_generation.sh` | SLURM submit — edit the config block to launch. |
| `job_array/generation_worker.sh` | SLURM worker — overlay-pool apptainer activation. |
| `job_array/multi_gpu.sh` | In-container launcher dispatch. |
| `job_array/launcher.py` | Spawns one process per GPU. |
| `scripts/monitor_camera_move.py` | Yes/No progress + 50/50 feasibility monitor. |
| `scripts/count_saved_envs.py` | Counts successfully saved envs. |

## Known Gaps / TODO

- **Occluded yield can still be low.** Balanced mode rejects all-visible scenes and
  downsamples excess `Yes` frames; it does not make occlusions more common.
- **No compile step.** Unlike A*, the camera-move worker has no post-task compile/cleanup.
  Raw `data_node*_gpu*` dirs are the final output. A compile/carve script may be needed.
- **`outside_fov` is not a separate label here.** The FOV gate just *drops* those angles —
  they are never saved. Only `Yes`/`No` (in_view/occluded) labels exist.
- **Balanced mode trades throughput for label balance.** With
  `CAMERA_MOVE_BALANCE_SAVED_FRAMES=1` (default), accepted envs must have at least
  `CAMERA_MOVE_MIN_PAIRS_PER_ENV=1` Yes/No pair. Set
  `CAMERA_MOVE_BALANCE_SAVED_FRAMES=0` only if you intentionally want the older
  save-all-safe-frames behavior.

## Critical: Server ↔ Local Sync

The camera-move pipeline was developed locally and the Oscar server copy has drifted.
**Before launching any camera-move job on the server, sync the files** listed in
`docs/server_sync_todo.md`. This has caused repeated headaches — local is the source of truth.
