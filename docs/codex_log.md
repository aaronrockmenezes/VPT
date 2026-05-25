# Codex Project Log

## 2026-05-12

**Task:** Establish root VPT agent instructions and project log.

**Files changed:**

- `AGENTS.md`
- `VPTnav_code/cube_game/agents.md`
- `docs/codex_log.md`

**Summary:**

- Promoted root `AGENTS.md` as the first file for future Codex chats.
- Added the rule to work on VPT project files only and not edit Obsidian/PERMANENT_MEMORY content unless explicitly requested.
- Added root `docs/codex_log.md` as the project log location.
- Recorded the latest A* array concurrency cap: 140 CPUs and 70 GPUs total, deriving a 17-task SLURM array throttle with current per-task resources.

**Open questions:**

- None.

**Next actions:**

- Keep future durable progress updates in `docs/codex_log.md`.
- Update `AGENTS.md` when operational assumptions or active paths change.

## 2026-05-14

**Task:** Linear probe baseline on v18_compiled_10k_test with DINOv2-reg and ViT Tiny.

**Files changed:**

- None (inference only, results written to `logs/perspective_results.json` on Oscar)

**Summary:**

Ran linear probes on `v18_compiled_10k_test` (10k envs, first-frame RGB only).
Split created with `make_vpt_probe_split.py --train_frac 0.5 --seed 42` → 5k train / 5k test, stratified by label+reason.
Used `run_linear_probe.py` (single GPU, not accelerate). 30 epochs, AdamW, BCE loss.

| Model | Train acc | Val/Test acc |
|---|---:|---:|
| `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | 55.58% | 55.05% |
| `vit_small_patch14_reg4_dinov2.lvd142m` | 60.65% | 61.20% |

DINOv2-reg outperforms ViT Tiny by ~6 points. Both near chance (50%) baseline, suggesting task is hard for first-frame-only linear readout at this scale.

**Comparison note:**

Prior experiments used 512 envs, 50/50 split, 10 frames/env. Not directly comparable to this run (scale, representation differ). These 10k results should be treated as a new baseline.

**Open questions:**

- Is ~61% DINOv2 signal or near-chance noise? Need to check against a shuffled-label control.
- Would using all rollout frames (not just first) improve readout?
- Accelerate multi-run version not yet tested — single run here, no variance estimate.

**Next actions:**

- Run shuffled-label control to confirm signal.
- Run `run_accel_latest_new.py` with `--num_runs 3` for variance.
- Consider probing on full rollout (all frames) vs first-frame.

## 2026-05-14 (2)

**Task:** Document v18_compiled_20k dataset and prep world model handoff.

**Files changed:**

- `VPTnav_code/cube_game/docs/world_model_handoff.md` — full rewrite with active dataset, split config, probe baselines, WM launch commands, known risks, and next steps for new agent chat
- `AaronVault/.../VPTnav Results and Evidence.md` — added 20k probe baselines section

**Summary:**

Active canonical dataset is `v18_compiled_20k` (20k envs, 50/25/25 balance, 80/10/10 split in master_labels.json). ImageFolder probe split is 80/20 via `make_vpt_probe_split.py`. World model handoff doc now contains everything a new agent needs to start WM training cold: dataset path, format, split, baselines, SLURM commands, and known risks.

**Open questions:**

- None new. See handoff doc.

**Next actions:**

- World models agent: read `docs/world_model_handoff.md` + run smoke test against v18_compiled_20k before full training.

## 2026-05-17

**Task:** Add manual A* dataset merge script for dynamics-training data.

**Files changed:**

- `VPTnav_code/cube_game/scripts/merge_a_star_datasets.py`
- `docs/codex_log.md`

**Summary:**

Added a direct multi-root staged A* merge script that keeps every passing env from
each source root, does not enforce 50/25/25 reason balance by default, and
assigns fixed-count splits (`1000` val, `1000` test by default, rest train).
The script preserves label/reason metadata and records `source_dataset`/`src_root`
provenance in `master_labels.json`, so dynamics training can use both the old
`v18_data_collector_run_array_test` data and corrected `v18_data_collector_v1`
data. It also has `--balance_vpt` for VPT probe-specific artifacts; that mode
downsamples each split to exact 50/25/25 over `in_view`/`occluded`/`outside_fov`
(therefore 50/50 Yes/No, with No split evenly across occluded and outside-FOV).

**Open questions:**

- Whether the world-model loader should filter eval/probe splits to
  `source_dataset == "v1"` or use a separate v1-only probe artifact.

**Next actions:**

- Run the merge script on Oscar under an environment with OpenCV available.
- Sanity-check the merged output before WebDataset conversion or training.

## 2026-05-19

**Task:** Add label/reason breakdowns to camera-move saved-env counter.

**Files changed:**

- `VPTnav_code/cube_game/scripts/count_saved_envs.py`
- `docs/codex_log.md`

**Summary:**

Extended `count_saved_envs.py` so it still reports saved env totals per shard, but now
also reads each complete config's `camera_move_collection.trajectory[]` entries and
prints saved-frame totals by label (`Yes`/`No`) and reason (`in_view`, `occluded`,
`outside_fov`, plus `unknown`/future values). In `--verify` mode, the breakdown only
includes envs that pass the RGB image-count check.

**Open questions:**

- Camera-move docs say `outside_fov` frames are normally dropped rather than saved, so
  this bucket should usually be zero unless future collection logic changes.

**Next actions:**

- Run on the Oscar camera-move output root after sync/collection to confirm global
  Yes/No and reason balance.

## 2026-05-19 (2)

**Task:** Enforce balanced saved frames for camera-move collection.

**Files changed:**

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera_move.py`
- `VPTnav_code/cube_game/docs/camera_move_handoff.md`
- `docs/codex_log.md`

**Summary:**

The first camera-move count showed `1282` Yes / `31` No saved frames, because the
active batch sweep saved every safe angle and did not enforce label balance. Added
default balanced saved-frame selection: after classifying sweep frames, each accepted
env must have at least one `Yes` and one `No`, and the saved trajectory is downsampled
to equal counts. This applies to both the batch path and sequential fallback. The mode
is controlled by `CAMERA_MOVE_BALANCE_SAVED_FRAMES` (default on) and
`CAMERA_MOVE_MIN_PAIRS_PER_ENV` (default `1`).

**Open questions:**

- Occluded yield is still intrinsically low under the current scene/sweep constraints;
  balanced saving discards excess `Yes` frames but does not create more `No` frames.

**Next actions:**

- Sync this env file to Oscar before the next camera-move run.
- Re-run `count_saved_envs.py` on a fresh output root and verify `Yes`/`No` are 50/50.

## 2026-05-19 (3)

**Task:** Write camera-move generation/checking optimization report.

**Files changed:**

- `VPTnav_code/cube_game/docs/camera_move_generation_optimization_report.md`
- `docs/codex_log.md`

**Summary:**

Added a repo-local report on optimizing camera-move env generation. The report maps the
current path (`_reset_idx_internal`, geometric filters, FOV/render checks, batch sweep,
collision checks), identifies bottlenecks, and recommends a staged roadmap: instrumentation,
batched camera POV label counting, vectorized segment-vs-OBB occlusion precheck, rendering
only balanced candidate subsets, and gating production debug plotting.

**Open questions:**

- Exact false-positive/false-negative rate of an analytic 2D occlusion precheck versus
  final semantic-pixel labels needs measurement on a small run.

**Next actions:**

- Implement counters first, then add the conservative geometric occlusion precheck.

## 2026-05-19 (4)

**Task:** Fix optimized camera-move geometric precheck crash and relax fixed-agent FOV count.

**Files changed:**

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera_move_optimized.py`
- `VPTnav_code/cube_game/docs/camera_move_generation_optimization_report.md`
- `docs/codex_log.md`

**Summary:**

Fixed the optimized copy's 2D segment-vs-OBB helper after an Oscar smoke run hit a tensor
broadcast error in the inside-OBB check. `start_proj`/`end_proj` are shaped
`[candidate, active_obj, 2]`, so they now compare directly against `extents` with the same
shape instead of `extents[:, :, None, :]`. Also changed fixed-agent candidate validation
for the optimized camera-move path to require one FOV-valid agent viewpoint by default via
`CAMERA_MOVE_MIN_AGENT_VIEWPOINTS=1`, because only the first valid point is used before
the camera sweep. The optimized file keeps the server-style `vpt_env_cfg_v15_rl` import.

**Open questions:**

- Need another Oscar smoke run to measure whether geometric precheck predictions line up
  with final semantic-pixel labels.

**Next actions:**

- Re-sync the optimized file to the server path used by the registered test task and rerun
  a small job.

## 2026-05-19 (5)

**Task:** Score fixed-agent viewpoints by geometric camera-sweep feasibility.

**Files changed:**

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera_move_optimized.py`
- `VPTnav_code/cube_game/docs/camera_move_generation_optimization_report.md`
- `docs/codex_log.md`

**Summary:**

Updated the optimized camera-move copy to collect up to
`CAMERA_MOVE_MAX_AGENT_VIEWPOINTS=10` FOV-valid fixed-agent candidates, then score each
candidate with a pure geometric 13-angle camera sweep. A candidate is eligible only if the
geometric sweep has at least one predicted visible camera point and one predicted occluded
camera point. Among eligible candidates, the code keeps only candidates tied for the max
number of valid geometric sweep positions, with predicted balanced pairs and occluded count
as tie-breakers. The existing FOV loop already stops stepping envs that hit the target,
so envs freeze at 10 valid agent viewpoints and only remaining envs continue.

**Open questions:**

- Need smoke-run stats to verify the 2D OBB occlusion prediction is neither too strict nor
  too loose relative to semantic-pixel labels.

**Next actions:**

- Re-run the optimized task on a small fresh output root and compare saved envs/minute plus
  final `Yes`/`No` balance.

## 2026-05-19 (6)

**Task:** Restore 12-frame camera-move save target with at least one Yes and one No.

**Files changed:**

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera_move_optimized.py`
- `VPTnav_code/cube_game/docs/camera_move_generation_optimization_report.md`
- `docs/codex_log.md`

**Summary:**

Corrected the optimized camera-move path after over-tightening it to equal Yes/No pairs.
The optimized env now keeps the 15-degree right-half sweep candidates (`0..180`, 13
possible angles), renders valid sweep candidates, and saves `CAMERA_MOVE_TARGET_FRAMES_PER_ENV=12`
frames per accepted env. Acceptance requires at least one final semantic-pixel `Yes` and
one final semantic-pixel `No`, but no longer forces a 50/50 split. Removed the confusing
pair-capped render knob from the active optimized path.

**Open questions:**

- Whether to use all 13 candidate angles and drop one, or switch the candidate sweep itself
  to exactly 12 angles (`0..165`). Current optimized behavior keeps 13 candidates and saves
  12 frames.

**Next actions:**

- Smoke run the optimized env and verify `TOTAL saved trajectory frames == 12 * saved envs`
  with nonzero Yes and No counts.

## 2026-05-19 (7)

**Task:** Reduce optimized camera-move saved modalities.

**Files changed:**

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera_move_optimized.py`
- `docs/codex_log.md`

**Summary:**

Updated the optimized camera-move env to save only three modalities per frame:
agent `RGB`, agent `Semantic`, and camera-object `cam_Semantic`. Removed saved depth,
`cam_RGB`, and `cam_RGB_norm` outputs from the optimized copy, including cached-frame
payloads and save-function arguments.

**Open questions:**

- None.

**Next actions:**

- Smoke run and confirm each accepted env has 12 files in each of `RGB`, `Semantic`, and
  `cam_Semantic`, with no new `Depth`, `cam_RGB`, or `cam_RGB_norm` dirs.

## 2026-05-22

**Task:** Establish server tarball as local git base and drop runtime artifacts.

**Files changed:**

- `.gitignore`
- server-sourced VPT files under `VPT_code/`, `VPTnav_code/`, `VPTnav_analysis/`,
  `a_star_data_collection_scripts/`, and `scripts/`
- `AGENTS.md`, `docs/codex_log.md`, and cube-game docs restored as repo governance docs
- dropped runtime/log/cache/binary artifacts from git tracking

**Summary:**

Imported `/Users/aaronrockmenezes/Downloads/VPT.tar.gz` as the server-base tree after
extracting it to `/private/tmp/vpt_server_extract/VPT`. The import intentionally excluded
nested `.git` directories, `.simg`/`.sif` binaries, logs, caches, W&B/Hydra outputs,
checkpoints, `v17_v4_logs/`, `v18_logs/`, and the duplicate `VPTnav_code/cube_game copy/`
directory. Root repo is now the one true git-tracked source for the server snapshot.

**Open questions:**

- Whether any excluded logs/checkpoints need separate archival outside git.

**Next actions:**

- Push the cleaned server-base history to `origin/main`.
- Use this repo as the canonical source for future local/server sync.

## 2026-05-25

**Task:** Add deferred Oscar git migration TODO.

**Files changed:**

- `VPTnav_code/cube_game/docs/server_sync_todo.md`

**Summary:**

Recorded the next server-side step: migrate Oscar's active
`/users/arock3/data/arock3/VPT` tree to the canonical GitHub-backed repo instead
of continuing ad hoc rsync/tarball sync. The old camera-move checklist remains
for context, but the canonical path is now clone/pull from `origin/main` and keep
heavy runtime artifacts outside git.

**Open questions:**

- When to schedule the Oscar-side tree backup and cutover.

**Next actions:**

- Start current VPT1/VPT2 data/runs first.
- Later, migrate Oscar to the canonical GitHub repo and update `MEMORY.md` after
  verification.

## 2026-05-25 (2)

**Task:** Add cleanup/wiki/FAQ backlog.

**Files changed:**

- `docs/project_todo.md`
- `AGENTS.md`
- `docs/codex_log.md`

**Summary:**

Added a repo-local project TODO for a future full cleanup and documentation pass.
The backlog explicitly calls for a private local wiki plus FAQ covering normal
VPTnav/VPT1 generation, VPT2 generation, A* rollouts, camera-move generation,
dataset count/compile/validation, linear probing, fine-tuning, server migration,
and generated-artifact hygiene.

**Open questions:**

- Exact structure of the local private wiki and whether it should live entirely
  under `docs/` or split root docs from `VPTnav_code/cube_game/docs/`.

**Next actions:**

- After current runs are launched, do the cleanup/doc pass before starting more
  large workflow changes.
