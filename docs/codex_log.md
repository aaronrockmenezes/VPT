# Codex Project Log

## 2026-05-27

**Task:** Add exact v18 config save and single-env replay.

**Files changed:**

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18.py`
- `VPTnav_code/cube_game/scripts/vptnav/keyboard_agent.py`
- `VPTnav_code/cube_game/scripts/vptnav/generate_one_v18_config.py`
- `VPTnav_code/cube_game/scripts/vptnav/generate_n_v18_configs.py`
- `VPTnav_code/cube_game/scripts/vptnav/replay_random_capture.py`
- `VPTnav_code/cube_game/scripts/vptnav/replay_many_random_capture.py`
- `docs/codex_log.md`

**Summary:**

Extended v18 saved configs with replay metadata for randomized object scales,
final bounds, material `.mdl` paths/names, floor material, wall shader values,
and light parameters. Added a loader that restores saved object poses, active
VPT indices, visual randomization, sensors, and OBB cache. `keyboard_agent.py`
now accepts `--config_file` and forces single-env replay for that mode. Added
action `7` to save the current agent RGB view using the A* debug-frame pattern,
plus small scripts to generate one config and replay 10 random actions with
captures. Added batch scripts for the current sample workflow: generate 10 v18
configs in one Isaac process, then replay those configs one by one and capture
10 random forward/left/right frames per sample.

**Open questions:**

- Runtime visual equivalence still needs an Isaac Lab smoke replay against a
  freshly generated v18 config.

**Next actions:**

- Generate one new v18 env, inspect the saved `scene_randomization` block, then
  replay it with `replay_random_capture.py --config_file`.
- Run the 10-sample batch workflow and visually inspect
  `replay_random_10/sample_*/env_0`.

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

- `VPTnav_code/cube_game/scripts/vptnav/count_saved_envs.py`
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

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_camera.py`
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

## 2026-05-25 (3)

**Task:** Align normal VPT1 builder QC with A* color constraints.

**Files changed:**

- `scripts/utils/build_vpt1_dataset.py`
- `docs/codex_log.md`

**Summary:**

Updated the normal VPT1 dataset builder to use A*-style semantic validation:
256-calibrated area-scaled strict-red thresholds, `No` cam-POV requiring zero
strict-red pixels plus circular-blob rejection, HSV green-presence checks for
agent semantic frames, and explicit RGB/Semantic image-count matching before
copying environments into the final train/test dataset.

**Open questions:**

- Whether `SEMANTIC_FAIL_TOLERANCE=1` should stay permissive or be tightened to
  `0` for production v18 builds.

**Next actions:**

- Run the builder on the current v18 pool and inspect accepted/rejected counts,
  especially `cam_red`, `cam_circle`, `sem_red`, and `sem_green`.

## 2026-05-25 (4)

**Task:** Add normal VPT-v18 camera deadzone for thesis dataset regeneration.

**Files changed:**

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18.py`
- `docs/codex_log.md`

**Summary:**

Added a normal VPT1/VPT-v18 agent-viewpoint deadzone matching the A* square
deadzone behavior. Candidate collection viewpoints are rejected when they fall
inside a camera-centered square with `abs(dx) < 3.0` and `abs(dy) < 3.0` by
default. The threshold can be overridden with `VPT1_AGENT_CAMERA_DEADZONE`, and
the value/metric are written into per-env config JSON.

**Open questions:**

- Whether thesis production should keep the default `3.0` square deadzone or
  sweep a stricter value after visual QC.

**Next actions:**

- Regenerate normal VPT-v18 data for thesis and rebuild with
  `scripts/utils/build_vpt1_dataset.py`.

## 2026-05-25 (5)

**Task:** Split A* and normal VPTnav workflow entrypoints.

**Files changed:**

- `scripts/vptnav/`
- `scripts/a_star/`
- `VPTnav_code/cube_game/job_array/normal_vptnav/`
- `VPTnav_code/cube_game/job_array/a_star/`
- `AGENTS.md`
- `VPTnav_code/cube_game/docs/commands.md`
- `docs/project_todo.md`
- `docs/codex_log.md`

**Summary:**

Separated active SLURM job-array scripts inside `VPTnav_code/cube_game/job_array/`:
normal VPTnav generation now lives under `normal_vptnav/`, and A* generation now
lives under `a_star/`. Root-level wrappers remain split under `scripts/vptnav/`
and `scripts/a_star/` for quick commands and dataset-build helpers.

**Open questions:**

- Whether to add compatibility shims at the old `job_array/*.sh` paths after
  server sync, if any old notes or scripts still call those paths directly.

**Next actions:**

- Use `scripts/vptnav/submit_vpt1_v18.sh` for the thesis v18 remake, or edit
  `VPTnav_code/cube_game/job_array/normal_vptnav/submit_generation.sh` directly.

## 2026-05-26

**Task:** Fix missing USD source assets in clean server clone.

**Files changed:**

- `VPTnav_code/cube_game/.gitignore`
- `VPTnav_code/cube_game/assets/*.usd`
- `VPTnav_code/cube_game/assets/*/*.usd`
- `VPTnav_code/cube_game/mass_assets/*.usd`
- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/*.usd`
- `docs/codex_log.md`

**Summary:**

Server smoke test failed because `VPT-v18` loads
`/mnt/VPT/VPTnav_code/cube_game/assets/new_cam_latest.usd`, but the clean Git
clone did not include USD assets. The old cube-game `.gitignore` blocked all
USD files. Added exceptions for source USD assets needed by registered VPT
tasks and staged the local source assets so `git pull` restores them on Oscar.

**Open questions:**

- Whether to later prune legacy USDs after a full file-map audit.

**Next actions:**

- Pull on Oscar and rerun the VPT-v18 smoke job.

## 2026-05-26 (2)

**Task:** Split cube-game scripts by workflow and remove old probe launchers.

**Files changed:**

- `VPTnav_code/cube_game/scripts/`
- `VPTnav_code/cube_game/job_array/a_star/a_star_launcher.py`
- `VPTnav_code/cube_game/job_array/normal_vptnav/multi_gpu.sh`
- `scripts/vptnav/submit_vpt1_v18.sh`
- `scripts/vptnav/submit_vpt1_v18_depth.sh`
- `scripts/vptnav/submit_vpt2.sh`
- `VPTnav_code/cube_game/agents.md`
- `VPTnav_code/cube_game/MEMORY.md`
- `VPTnav_code/cube_game/docs/`

**Summary:**

Separated active cube-game scripts into `scripts/a_star/` and
`scripts/vptnav/`. Removed stale probe-era launchers (`test_models*.sh`,
`old_test_models.sh`, `test_multiple.sh`, `compile_results.py`) plus generated
probe/debug artifacts from the cube-game scripts folder. Updated job-array and
root launch wrappers to point at the new agent paths.

**Open questions:**

- Whether to later archive or remove legacy SB3/SKRL scripts after confirming
  they are no longer useful.

**Next actions:**

- Pull on Oscar before launching new VPTnav or A* jobs so the script paths match.

## 2026-05-26 (3)

**Task:** Archive inactive direct-task env modules.

**Files changed:**

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/__init__.py`
- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/legacy/`
- `VPTnav_code/cube_game/agents.md`
- `VPTnav_code/cube_game/MEMORY.md`
- `VPTnav_code/cube_game/docs/`
- `docs/codex_log.md`

**Summary:**

Trimmed cube-game task registration to the active task set: template cube game,
normal `VPT-v18`, `VPT-v18-Depth`, VPT2 v1-v4, A* v18, A* strategy, and v18
camera/camera-optim tasks. Moved old VPT env/config variants, copy-suffixed
files, old parallel envs, and inactive simulate/train helpers into
`legacy/`. Kept active utility/import support in place.

**Open questions:**

- Whether to eventually delete the legacy folder after thesis runs and docs are
  stable.

**Next actions:**

- Pull on Oscar before new generation jobs so active task registration matches
  the cleaned direct-task layout.

## 2026-05-26 (4)

**Task:** Fix VPT1 builder validation for multi-file cam folders.

**Files changed:**

- `scripts/utils/build_vpt1_dataset.py`
- `docs/codex_log.md`

**Summary:**

The VPT1 builder copied full `cam/env_*` directories correctly, but root
validation still expected exactly one cam image per env. Active v18 writes
`cam_pov.png` plus RGB/normalized camera views, so successful 512-env builds
printed 512 false validation errors. Updated validation to require
`cam_pov.png` by name and allow additional cam artifacts.

**Open questions:**

- Whether to apply the same named cam-file validation rule to older VPT2/depth
  merge utilities if they are reused.

**Next actions:**

- Pull on Oscar and rerun `scripts/vptnav/build_vpt1.sh`; existing compiled
  output is balanced, but rerunning removes the false root validation errors.

## 2026-05-27

**Task:** Capture and analyze VPT1 v18 LP/FT result tables.

**Files changed:**

- `docs/results/vpt1_v18/vpt1_v18_linear_probe_results.csv`
- `docs/results/vpt1_v18/vpt1_v18_finetune_results.csv`
- `docs/results/vpt1_v18/vpt1_v18_lp_vs_ft_comparison.csv`
- `docs/results/vpt1_v18/vpt1_v18_lp_ft_imagenet_matched.csv`
- `docs/results/vpt1_v18/vpt1_v18_lp_ft_vs_imagenet.html`
- `docs/results/vpt1_v18/vpt1_v18_lp_ft_analysis.md`
- `scripts/analyze_lp_ft_results.py`
- `scripts/plot_lp_ft_vs_imagenet.py`
- `docs/project_todo.md`
- `docs/codex_log.md`

**Summary:**

Recovered the pasted VPT1 v18 linear-probe and fine-tune compiled result
tables from the local Codex session log, normalized them into CSVs, and added a
small comparison script. The FT table averages 57.294% versus LP at 55.209%.
Across 493 shared models, FT improves by +2.105 percentage points on average
and beats LP for 379/493 models, but LP/FT rank correlation is weak. The
fine-tune result should be treated as model-search/debug evidence because the
current FT script selects the best epoch on test accuracy. Added a Plotly
scatter comparing LP and FT accuracy against ImageNet top-1 for 466 matched
models; both correlations are moderate and similar (LP r=0.429, FT r=0.427).
Added the remaining LP/FT cleanup work to `docs/project_todo.md`.

**Open questions:**

- Whether to rerun FT with a train/val/test protocol before using the table in
  thesis-facing claims.

**Next actions:**

- Add a clean FT mode that selects checkpoints on validation accuracy and only
  reports held-out test once.

## 2026-05-27 (2)

**Task:** Cap SLURM memory defaults at 40G.

**Files changed:**

- `VPTnav_code/cube_game/job_array/normal_vptnav/generation_worker.sh`
- `VPTnav_code/cube_game/job_array/a_star/submit_a_star_array.sh`
- `VPTnav_code/cube_game/job_array/a_star/submit_a_star.sh`
- `scripts/ft_config.sh`
- `docs/codex_log.md`

**Summary:**

Set the normal VPTnav generation worker, A* submit wrappers, and FT launcher
defaults to request `40G` node memory. LP and compile jobs were already below
that cap (`30G` and `4G` respectively), so they were left unchanged.

**Open questions:**

- Whether any future large FT/A* jobs need an explicit one-off override after
  validating the 40G cap.

**Next actions:**

- Pull on Oscar before submitting new jobs so the memory cap is active there.

## 2026-05-27 (3)

**Task:** Add one-shot launch overrides for depth LP/FT and VPT2 time.

**Files changed:**

- `scripts/config.sh`
- `scripts/ft_config.sh`
- `scripts/vptnav/submit_vpt2.sh`
- `docs/codex_log.md`

**Summary:**

Added shell override support for single-dataset LP/FT launches:
`LP_TASK`, `LP_DATA_DIR`, `LP_OUTPUT_DIR`, `LP_MAX_CONCURRENT`,
`FT_TASK`, `FT_DATA_DIR`, `FT_OUTPUT_DIR`, `FT_MEM_OVERRIDE`, and
`FT_MAX_CONCURRENT_OVERRIDE`. Set the FT default memory request to `80G` while
keeping the existing 4 GPU / 8 CPU task shape. Added `TIME_LIMIT` support to
the VPT2 generation wrapper so VPT2 jobs can request 4h without hand-editing
the worker script.

**Open questions:**

- Whether VPT2 generation should eventually get its own default time above 1h
  after the current thesis run.

**Next actions:**

- Pull on Oscar before using the override-based launch commands.

## 2026-05-27 (4)

**Task:** Record active Oscar LP/FT/VPT2 jobs and next thesis run order.

**Files changed:**

- `VPTnav_code/cube_game/MEMORY.md`
- `docs/project_todo.md`
- `docs/codex_log.md`

**Summary:**

Captured the current Oscar queue snapshot from `squeue -u arock3`: VPT1 depth
LP array `2873958_[0-499%40]` is running the first 40 tasks with compile job
`2873959` pending; VPT1 depth FT array `2874004_[0-499%10]` and compile job
`2874005` are submitted but pending on `QOSMaxMemoryPerUser`; VPT2-v4
generation array `2874008_[0-29]` is also pending on `QOSMaxMemoryPerUser`.
Recorded the next thesis order: analyze VPT1 depth, build VPT2, run/analyze
VPT2 LP+FT, then generate/build/analyze VPT1 Strategy.

**Open questions:**

- Whether to let LP consume the memory cap until slots finish or reduce active
  LP concurrency so FT/VPT2 generation can begin sooner.

**Next actions:**

- Re-check `squeue -u arock3` and result counts before canceling or relaunching
  anything.

## 2026-05-28

**Task:** Log VPT2-v4 LP/FT thesis results and VPT1 depth FT compile status.

**Files changed:**

- `docs/results/vpt2_v4/vpt2_v4_linear_probe_results.csv`
- `docs/results/vpt2_v4/vpt2_v4_finetune_results.csv`
- `docs/codex_log.md`

**Summary:**

Saved the pasted VPT2-v4 linear probe and fine-tune result tables as CSVs.
VPT2 LP total average row is `51.947, 51.853, 51.955, 51.918`; VPT2 FT total
average row is `86.617, 86.496, 86.996, 86.703`. VPT1 depth FT array appears
done, but compile job `2874005` is held/requeued with
`user env retrieval failed requeued held`.

**Open questions:**

- Whether the held SLURM compile job should be canceled after manually compiling
  VPT1 depth FT results.

**Next actions:**

- Manually compile VPT1 depth FT results from the existing result shards.

## 2026-05-28 (2)

**Task:** Save VPT1 depth FT results and analyze current thesis LP/FT tables.

**Files changed:**

- `docs/results/vpt1_v18_depth/vpt1_v18_depth_finetune_results.csv`
- `scripts/analyze_thesis_results.py`
- `docs/results/thesis_analysis/`
- `docs/codex_log.md`

**Summary:**

Saved the VPT1 v18 depth FT table. Added a reusable thesis analysis script and
generated matched result CSVs, Plotly HTML plots, static PNG plots, and a
summary markdown report across VPT1 v18, VPT1 v18 depth, and VPT2 v4. Headline
totals: VPT1 v18 LP/FT `55.209/57.294`, VPT1 depth LP/FT `78.210/88.934`,
and VPT2 v4 LP/FT `51.918/86.703`.

**Open questions:**

- Whether FT numbers are final-reportable or should be rerun with explicit
  train/val checkpoint selection and one held-out test evaluation.

**Next actions:**

- Use the analysis outputs to pick candidate models for reruns and thesis
  figures.

## 2026-05-28 (3)

**Task:** Add config replay support for VPT strategy/depth/VPT2 workflows.

**Files changed:**

- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/env_config_replay.py`
- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_A_star_strategy.py`
- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18_depth.py`
- `VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt2_env_v4.py`
- `VPTnav_code/cube_game/scripts/vptnav/vpt2_keyboard_agent.py`
- `docs/codex_log.md`

**Summary:**

VPT1 strategy is the proxy LOS test set: freeze the agent/observer and move
only camera plus goal, with each accepted scene saving five `Yes` and five `No`
camera-view images. Added per-scene replay configs for this strategy set. The
expected thesis hypothesis is that this is hard: prior VPT1 FT strategy-like
performance did not cross 70%, while human VPT1 accuracy is roughly 76-83%.

Added a shared config replay helper for VPT-style envs, wired it into VPT1
depth and VPT2-v4 testing mode, and made VPT2 configs save/replay the active
pink reference object. VPT2 keyboard replay now accepts `--config_file`.

**Open questions:**

- Whether VPT2 strategy should be a separate registered env with `left/right`
  balanced rail settings, or generated by a wrapper around VPT2-v4 once the
  exact reference-object placement rule is finalized.

**Next actions:**

- Generate a small VPT1 v18 strategy test set, replay a few configs locally on
  Oscar, then run VPT1 LP/FT checkpoints against it.
