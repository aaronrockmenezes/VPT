# agents.md — cube_game VPT Benchmark

> Keep this file updated before every git push.

---

## Task: Visual Perspective Taking (VPT) Level 1

**Goal:** Collect images to benchmark whether pre-trained vision models can reason about 3D spatial perspective via linear probing.

**Question per image:** Given a 3rd-person view, can the camera object (green) see the red ball (goal)?

**Labels:**
- `Yes` → `in_view`: camera sees ball (no occlusion, in FOV)
- `No` → `occluded`: ball hidden behind obstacle
- `No` → `outside_fov`: camera not pointing toward ball

**Label distribution (v18):**
- 50% `in_view`, 25% `occluded`, 25% `outside_fov`
- Set in `_preallocate_visibility_labels()` via deterministic assignment

**Data layout:** `{base_path}/RGB|Depth|Semantic|cam/Yes|No/env_{idx}/image_{i:04d}.png`

---

## Architecture: VPTEnv (vpt_env_v18.py)

**Class:** `VPTEnv(DirectRLEnv)` — Isaac Lab environment

**Key config fields (VPTEnvCfg):**
- `images_per_env`: images collected per spawn slot
- `num_vpt_objs`: obstacle count per env
- `base_path`: output data directory
- `save_camera_pov`: whether to save camera-object POV images
- `verbose`: 0/1/2 logging level

**Actors:**
- `_agent`: robot with tiled RGB camera mounted on top
- `_camera_obj`: green camera-shaped object (the "third person")
- `_goal`: red ball
- `_vpt_objects`: obstacle objects (USDAssets, MeshCuboid, MeshCylinder, MeshCone)

**Cameras:**
- `_rgb_tiled_camera`: main data collection camera (agent POV)
- `_occlusion_camera`: camera-object POV (for validation / `save_camera_pov`)

---

## Spawn Pipeline (`_reset_idx_internal`)

Up to 20 attempts per reset batch. Stages:

1. **`initial_spawn_loop`** — place `_camera_obj` and `_goal` with geometric separation
2. **Z-check** — validate both objects above floor
3. **`outside_fov_camera_movement`** — move camera object to face away from ball (for `outside_fov` slots) — has TODO: use `_update_camera_poses`
4. **`occlusion_validation_check`** — verify occlusion actually holds (has `# TODO: Clean up later`)
5. **Geometric check (`_geometric_check`)** — distance/angle constraints
6. **Camera POV check (`_check_target_in_img`)** — pixel threshold on semantic segmentation
7. **`generate_valid_circle_points`** → `_sample_fov_candidates` — find agent viewpoints
8. **`_collect_images_for_slot`** — teleport agent to each viewpoint, step 30 sim frames, save images

---

## Viewpoint Generation (`generate_valid_circle_points`)

Generates agent camera positions on a circle around the `_goal`:
- Radius = `(d/2) / tan(FOV/2)` where `d` = camera-goal distance; scaled by `[1.1, 1.5]` random factor + ±20% radial jitter
- `angle_step=2°` → 180 candidate angles
- Stage 1: geometric validity (`_is_point_valid_batch`, no FOV check)
- Stage 2: greedy displacement filtering (min `min_viewpoint_distance` spacing)
- Stage 3: FOV check only on displacement-filtered candidates
- Need ≥ `MIN_CANDIDATES_FOR_FOV=40` before FOV check; ≥ `images_per_env` valid points to pass

**`_collect_images_for_slot`:**
- Teleports agent to each viewpoint; yaw jitter constrained so camera object stays in FOV
- Steps physics 30 frames at each pose before saving
- Saves: RGB, Depth (normalized uint8), Semantic, optional cam POV

---

## Validation Methods

| Method | Purpose |
|---|---|
| `_check_occlusion` | Semantic seg pixel count (red ≥500, green ≥800); uses `sim.step()` for "lazy sync" — **perf concern** |
| `_check_target_in_img` | Pixel threshold only — Hough Circle detection is **fully commented out** (dead code) |
| `_is_point_valid_batch` | Batch geometric + optional FOV check for agent viewpoints |
| `_check_collisions_vectorized` | 2D SAT + wall boundary check for object placement |
| `place_object_safely` | Shotgun sampling (1000 candidates) with SAT collision check |

---

## Randomization

| Method | What it does |
|---|---|
| `randomize_shape_scale` | Per-object scale via USD Sdf.ChangeBlock; recomputes BBoxCache bounds; handles USDAsset, Cuboid, Cylinder, Cone with type-specific z-offset |
| `randomize_shape_color` | Fast path (standard hierarchy) + dynamic shader search fallback; optional roughness/metallic |
| `randomize_spherical_lights` | Intensity [40k–75k], color temp [2k–8k], position (inner/outer ring), min separation 8.0m |
| `randomize_material` | Applies random MDL material from pre-loaded pool (`mat_material_paths` / `vpt_material_paths`) |
| `get_color` | Generates pastel color excluding red, green, blue, pink — used for obstacle color |

**Scale drift fix:** `_cached_mesh_points` dict prevents scale accumulation across resets.

---

## OBB / Collision System

- `_cache_base_dims()`: caches per-object base dimensions from config (USDAsset uses scale, primitives use size/radius/height)
- `get_obb_hitbox()`: vectorized analytic OBB corners `(Batch, NumObjs, 8, 3)` using cached dims + current pose
- `update_obb_cache()`: lazy-init `obb_corners_cache` tensor, updates on reset
- `_check_collisions_vectorized()`: 2D SAT (XY only) — 4-axis test (2 agent + 2 obstacle)
- `_get_object_corners()`: hardcoded half-extents per object type (agent=0.1, cam_obj=[0.55,0.575,0.45], goal=0.2)
- `debug_plot_analytic_obbs()`: saves `debug_obb_env_0.png` matplotlib plot — debug utility, not part of pipeline

---

## Dead Code

| Symbol | Location | Notes |
|---|---|---|
| `initial_spawn_loop_old` | ~line 900 | Shapely-based serial spawn; superseded by vectorized `initial_spawn_loop` |
| `moving_ball_loop` | defined, commented in `_reset_idx_internal` | Ball motion during data collection; never used |
| Hough Circle detection | `_check_target_in_img` | Fully commented out; only pixel threshold used |
| All `EnvTimer.start_timer/stop_timer` calls | Throughout | Commented out; `env_timer.py` still imported |
| `_collect_images_for_slot` OLD yaw block | ~line 3380 | Comment says `--- OLD ---`; replaced by camera-constrained jitter |

---

## Active Imports (NOT orphaned)

From `vpt_env_v18.py` lines 29–30:
```python
from .spawn_boundary import get_vpt_material_paths, get_mat_material_paths
from .env_timer import EnvTimer
```

`spawn_boundary.py` and `env_timer.py` are **used** — do NOT delete.

---

## Optimization Opportunities

1. **`_check_occlusion` lazy sync:** calls `self.sim.step()` on every occlusion check — expensive; consider batching
2. **30 sim steps per image** in `_collect_images_for_slot`: may be reducible; depends on physics settle time
3. **`randomize_shape_color` Strategy 2** (dynamic shader search): depth-first `Usd.PrimRange` is slow for complex assets; precompute shader paths once
4. **`get_color` busy-loop rejection sampling:** could precompute a palette
5. **`_sample_fov_candidates` random permutation:** only applied when `total > max_s`; consistent behavior fine
6. **`place_object_safely` fixed BATCH_SIZE=1000:** may over/under-sample; tune per scene density

---

## File Structure

```
cube_game/
├── agents.md                          # this file
├── assets/                            # USD walls, room geometry
├── mass_assets/                       # obstacle USD files
├── job_array/
│   ├── submit_generation.sh           # SLURM array submit (4 vars: BASE_PATH, NUM_GPUS, TASK, NUM_NODES)
│   ├── generation_worker.sh           # SLURM worker: conda + Apptainer + multi_gpu.sh
│   ├── multi_gpu.sh                   # reads SLURM_ARRAY_TASK_ID, calls launcher.py
│   └── launcher.py                    # spawns per-GPU subprocesses, 30s stagger, SIGINT/SIGTERM cleanup
├── scripts/
│   ├── test_models_accel_args.sh      # accelerate-based eval runner
│   ├── test_multiple.sh               # multi-run test wrapper
│   ├── compile_results.py
│ ├── keyboard_agent.py / random_agent.py / zero_agent.py
│ ├── vpt2_keyboard_agent.py
│ ├── A_star_agent.py # interactive A* debug agent
│ ├── A_star_automatic_agent.py # batch A* navigation agent
│ ├── astar_viz.py # offline A* plan visualizer
│ ├── list_envs.py
│   ├── sb3/                           # SB3 RL (untouched)
│   └── skrl/                          # SKRL RL (untouched)
└── source/cube_game/cube_game/tasks/direct/cube_game/
 ├── vpt_env_v18.py # primary data collection env (current)
 ├── vpt_env_v18_A_star.py # A* navigation variant
 ├── vpt_env_v18_depth.py # depth variant
    ├── vpt_env_v19.py                 # WIP next version
    ├── vpt_env_cfg_v15_rl.py          # RL config
    ├── vpt_env_cfg_v17.py             # current data collection config
    ├── vpt2_env_v1-v4.py              # VPT2 task envs
    ├── vpt2_env_cfg_v1-v2.py
    ├── spawn_boundary.py              # get_vpt_material_paths, get_mat_material_paths
    ├── env_timer.py                   # EnvTimer class
    ├── geometry_utils.py              # pending import verification
    ├── timing_utils.py                # pending import verification
    ├── utils.py                       # pending import verification
    ├── check_collisions.py            # pending import verification
    ├── check_collisions_new.py        # pending import verification
    ├── spawn_boundary_old.py          # pending import verification
    └── __init__.py
```

---

## A* Navigation (vpt_env_v18_A_star.py)

**Class:** `VPTEnvAStar(VPTEnv)` — extends data collection env with A* pathfinding.

**Three-action space:** `0=forward`, `2=left turn`, `3=right turn`. No backward/stop.

**Planning pipeline:**
1. Build 2D occupancy grid from OBB footprints + inflation
2. Run A* on grid (8-connected, heading-binned cost) from agent to camera object
3. Convert grid path → action sequence (forward/turn)
4. Execute actions; yaw alignment done at runtime (greedy left/right spamming, NOT in plan)

**Navigation state machine (shared by both agent scripts):**
- `NAVIGATING` → executing queued plan actions
- `ALIGNING_YAW` → queue drained, position OK, greedy yaw correction
- `DONE_STATE` → both position and yaw within tolerance

**Greedy yaw alignment:**
- After plan queue drains, check if position within `pos_tol_m`
- If yes: spam turn commands toward target yaw
- 3 consecutive yaw-error increases → give up (overshoot detection)
- Yaw tolerance: 11.46 deg (matches heading bin resolution)

**Collision-revert detection:**
- Before each forward step, record agent XY position
- After step, compute displacement
- If displacement < 0.01m → agent hit wall/obstacle → clear queue, re-plan

**Inflation fallback schedule (both agent scripts):**
1. `None` (use env default `planner_inflation_m=0.12`)
2. `0.06` (50% of default)
3. `0.03` (25% of default)
4. `0.0` (no inflation — last resort)

**Key constants (synced across env + astar_viz.py):**
| Constant | Value | Notes |
|---|---|---|
| `PLANNER_INFLATION_M` | 0.12 | 0.1m half-extent + 0.02m margin |
| `HEADING_BIN_RAD` | π/24 (7.5°) | A* heading discretization |
| `ASTAR_MAX_EXPANSIONS` | 250000 | Max nodes explored |
| `FORWARD_STEP_M` | 0.15 | Grid resolution = forward step |
| `POS_TOL_M` | 0.2 | Goal position tolerance |

**OBB inflation method (edge-intersection, both files identical):**
1. Compute each edge's outward normal (RIGHT perp `(ey,-ex)` for CCW polygon)
2. Shift each edge outward by `normal * inflate`
3. Intersect consecutive shifted edges → true Minkowski sum vertices
4. Scanline rasterize inflated polygon → mark occupancy grid

**Agent scripts:**
| Script | Purpose |
|---|---|
| `A_star_agent.py` | Interactive debug: step-by-step plan/execute/align + frame capture |
| `A_star_automatic_agent.py` | Batch: auto-navigate fixed set of envs, save all frames, JSON metrics |
| `A_star_data_collector.py` | **Scalable data collection**: runs until `--target_successes` (default 10k). Saves only successful navigations. RL-resets failed/overflow envs. Enforces 50/25/25 split. |
| `astar_viz.py` | Offline: visualize A* plans + inflation on static scenes |

**A_star_data_collector.py key design:**
- On success within quota → save frames + `actions.txt` + `meta.json`, RL-reset, continue
- On failure or quota overflow → discard frames, RL-reset immediately
- RL reset spawns agent in 10×10 square around camera with 4×4 deadzone (collision-checked, 200 attempts)
- 50/25/25 enforced via per-category quotas; stops when all three are full
- Frames staged in `_tmp_{i}_{ep_id}/` then atomically moved to `ep_{N:06d}_env_{i}/` on save

**Output structure (mirrors data-collection pipeline + `rollout/` subfolder):**

```
{base_path}/
├── RGB/{Yes|No}/rollout/env_{folder_idx}/
│   ├── step_{N:05d}_{action}.png    # agent POV RGB per action
│   └── final_pos_*.png              # agent POV at final pose
├── Semantic/{Yes|No}/rollout/env_{folder_idx}/
│   ├── step_{N:05d}_{action}.png    # agent POV semantic per action
│   └── final_pos_*.png              # agent POV semantic at final pose
├── cam/{Yes|No}/rollout/env_{folder_idx}/
│   ├── final_cam_semantic.png       # camera-obj POV semantic (final)
│   ├── actions.txt                  # space-separated action sequence
│   └── meta.json                    # folder_idx, seed, reason, label, metrics
└── successful_envs.json             # cumulative tracker, flushed per success
```

**Env API additions (`vpt_env_v18_A_star.py`):**
- `save_rollout_step(env_slot, folder_idx, step_n, action_label)` — writes one RGB + semantic frame
- `save_rollout_final(env_slot, folder_idx, pos_err, yaw_err)` — writes final agent RGB + semantic + camera-obj POV semantic
- `_rollout_dirs(folder_idx)` — returns dict of label-aware dir paths
- `_to_uint8_rgb(tensor)` — tensor→numpy conversion helper

**Folder advancement on RL reset:** Collector calls `_advance_slot` which manually pops the next label from `visibility_label_pool` and advances `slot_folder_indices`. Pool is auto-refilled (50/25/25) when empty so 10k+ targets work despite default `total_envs_to_sim=2000`.

**CLI args (data collector only):**
| Arg | Default | Notes |
|---|---|---|
| `--target_successes` | 10000 | Stop after this many saves |
| `--spawn_half_extent` | 5.0 | Half-side of spawn square (metres) |
| `--spawn_deadzone` | 2.0 | Half-side of no-spawn zone at camera centre |
| `--max_total_steps` | 150 | Per-env step budget |

---

## Pending TODOs

- [ ] Verify if `geometry_utils.py`, `timing_utils.py`, `utils.py`, `check_collisions.py`, `check_collisions_new.py`, `spawn_boundary_old.py` are imported anywhere — if not, delete
- [ ] `outside_fov_camera_movement`: replace with `_update_camera_poses`
- [ ] `occlusion_validation_check`: clean up marked TODO
- [ ] `check_batch_object_visibility`: consider merging with `_check_occlusion`
- [ ] `_cache_base_dims` TODO comment: verify object list matches current USD assets
