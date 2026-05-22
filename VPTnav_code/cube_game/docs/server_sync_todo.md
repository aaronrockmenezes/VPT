# Server ↔ Local Sync TODO

**Status: PENDING — do this before launching any camera-move job on Oscar.**

The camera-move pipeline was built locally. The Oscar server copy is behind. Aaron wants to
pull the server copy down and treat **local as the source of truth** going forward. This
file is the actionable checklist; `docs/camera_move_handoff.md` has the full context.

Server repo root: `/users/arock3/data/arock3/VPT/VPTnav_code/cube_game`

## Files to push local → server

### Edited

- [ ] `source/cube_game/cube_game/tasks/direct/cube_game/__init__.py`
      — registered gym id `VPT-v18-camera-move`.
- [ ] `source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera_move.py`
      — the camera-move env (large). 180° sweep, sweep measured from goal→agent direction,
      agent-FOV gate, batch-parallel collection (`_build_fixed_sweep_trajectory_batch`,
      `_save_slot_from_cache`, `_cached_frames`), label in filename, cam POV split into
      `cam_Semantic`/`cam_RGB`/`cam_RGB_norm`. NOTE: this file is untracked in git locally.
- [ ] `scripts/keyboard_agent.py`
      — added `camera_move_mode` branch: `"camera-move"` in task name → send `action=5`
      to all envs every step.
- [ ] `job_array/launcher.py`
      — `parse_args` → `parse_known_args`, forwards extra args (`--num_envs`) to the script;
      prefers composite `NODE_ID` env over bare `SLURM_ARRAY_TASK_ID`.
- [ ] `job_array/multi_gpu.sh`
      — `AGENT_SCRIPT` env-overridable (default `keyboard_agent.py`); appends
      `--num_envs $NUM_ENVS` when set; prefers composite `NODE_ID`.
- [ ] `job_array/generation_worker.sh`
      — REWRITTEN to the overlay-pool apptainer activation (same as `a_star_worker.sh`):
      `--cleanenv --overlay` from a 32-slot pool, per-task `CACHE_ROOT`, conda
      site-packages bind, composite `NODE_ID={job}_{task}`. Dropped old `--writable-tmpfs`
      + host-conda method. No compile step.
- [ ] `job_array/submit_generation.sh`
      — config block set to `TASK="VPT-v18-camera-move"`, added `NUM_ENVS=32`, added
      `NUM_ENVS` to `--export`.
- [ ] `plan.md`
      — added handoff/sync section (informational).

### Created (new files — copy to server)

- [ ] `scripts/monitor_camera_move.py`
      — Yes/No progress monitor: overall %, per-env avg, 50/50 feasibility, `--watch`.
- [ ] `scripts/count_saved_envs.py`
      — counts successfully saved envs per shard + total; `--verify` cross-checks images.
- [ ] `docs/camera_move_handoff.md`
      — full camera-move pipeline handoff.
- [ ] `docs/server_sync_todo.md`
      — this file.

## Files NOT changed by camera-move work (do not attribute / do not touch)

These showed as dirty in the local worktree but predate this work:
`README.md`, `agents.md`, `scripts/A_star_data_collector.py`,
`source/.../vpt_env.py`, `source/.../vpt_env_cfg.py`, `source/.../vpt_env_v18_A_star.py`.
(README.md and agents.md WERE later updated to reference the camera-move pipeline — see
their diffs; those edits are intentional.)

## Removed locally

- `scripts/camera_move_collector.py` — was a standalone collector; replaced by the
  `keyboard_agent.py` camera-move branch. Delete it on the server too if present.

## Sync method (suggested)

1. On the server, stash or back up any local-only server changes first.
2. `rsync` or `scp` the files above from local → server, preserving paths.
3. Verify `submit_generation.sh` config block points at the intended `BASE_PATH`.
4. Dry-run: `NUM_NODES=1`, small `NUM_ENVS`, check one shard's output tree + a config JSON.
5. Confirm `monitor_camera_move.py` and `count_saved_envs.py` read the output correctly.
6. Scale up `NUM_NODES` / `NUM_ENVS`.

## After sync

Mark this file done (or delete it) and update `MEMORY.md` to note the server is in sync.
