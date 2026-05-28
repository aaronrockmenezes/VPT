# Claude Handoff - 2026-05-27

## Read First

- Root repo entry point: `AGENTS.md`
- Cube-game map: `VPTnav_code/cube_game/agents.md`
- Current durable memory: `VPTnav_code/cube_game/MEMORY.md`
- Current backlog: `docs/project_todo.md`
- Result summary: `VPTnav_code/cube_game/docs/vpt_thesis_results_and_plan.md`

Do not edit Aaron's Obsidian/PERMANENT_MEMORY vault unless explicitly asked.
Keep durable notes in this repo, especially `docs/codex_log.md`.

## Current Code State

Recent work updated normal `VPT-v18` config replay:

- `source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18.py`
  now saves exact randomization metadata in each `env_*_config.json`:
  object scales, final bbox dims, z-offset ratios, shape ids, VPT/floor
  material prim paths plus `.mdl` paths/names, wall colors, roughness/metallic
  values, and spherical-light position/intensity/radius/color temperature.
- The same env now has `load_env_config_from_json(...)` to restore one saved
  config into an env slot, including object poses, active indices, visuals,
  camera poses, sensors, and OBB cache.
- Action `7` in `VPT-v18` saves the current agent RGB view to PNG using the
  A* debug-frame pattern (`obs -> step_00000_capture.png`).
- `scripts/vptnav/keyboard_agent.py` accepts `--config_file` and forces
  `--num_envs 1` for replay mode.

Helper scripts added:

- `scripts/vptnav/generate_one_v18_config.py`
- `scripts/vptnav/generate_n_v18_configs.py`
- `scripts/vptnav/replay_random_capture.py`
- `scripts/vptnav/replay_many_random_capture.py`

Static checks passed:

```bash
python -m py_compile \
  VPTnav_code/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/vpt_env_v18.py \
  VPTnav_code/cube_game/scripts/vptnav/generate_n_v18_configs.py \
  VPTnav_code/cube_game/scripts/vptnav/replay_many_random_capture.py
git diff --check
```

Runtime Isaac smoke still needs to be run.

## 10-Sample Replay Smoke

Run from `VPTnav_code/cube_game`:

```bash
RUN_ROOT="/tmp/vpt_v18_samples_$(date +%Y%m%d_%H%M%S)"

isaaclab.sh -p scripts/vptnav/generate_n_v18_configs.py \
  --task VPT-v18 \
  --base_path "$RUN_ROOT" \
  --target_configs 10 \
  --max_attempts 2000

isaaclab.sh -p scripts/vptnav/replay_many_random_capture.py \
  --task VPT-v18 \
  --config_glob "$RUN_ROOT/data/data_node*_gpu*/configs/env_*_config.json" \
  --capture_dir "$RUN_ROOT/replay_random_10" \
  --limit 10 \
  --steps 10 \
  --seed 0

find "$RUN_ROOT/replay_random_10" -name '*.png' | sort
```

Expected behavior: generate 10 saved `VPT-v18` configs, exit, then replay them
one by one. For each config, perform 10 random actions from forward/left/right
and save 10 agent-view RGB frames under `sample_*/env_0`.

## Result Context

VPT1 v18 RGB:

- LP mean: 55.209% over 497 model rows.
- FT mean: 57.294% over 494 model rows.
- Shared-model FT-LP delta: +2.105 percentage points.
- FT beats LP for 379/493 shared models.
- Caveat: current FT selection is test-set-selected, so treat as model-search
  evidence until val-selected FT is implemented.

VPT1 v18 depth:

- LP mean: 78.210% over 498 model rows.
- Best depth LP: `tf_efficientnet_b7.ap_in1k` at 85.521%.
- Depth LP beats RGB LP for all 497 shared models; mean depth-RGB LP delta is
  +23.004 percentage points.
- Depth FT has not been analyzed yet and is a high-priority follow-up.

## Next Work

1. Run the 10-sample replay smoke above and visually inspect captures.
2. Re-check Oscar queue/results before acting on cluster jobs.
3. Analyze VPT1 depth FT once complete; compare against depth LP.
4. Build VPT2 dataset after generation completes, then run LP + FT and analyze.
5. Generate VPT1 Strategy and VPT2 Strategy datasets, build them, then evaluate
   the same LP and FT model suites on both.
6. For thesis-grade FT evidence, add or use a clean train/val/test FT protocol
   with checkpoint selection on validation accuracy only.

