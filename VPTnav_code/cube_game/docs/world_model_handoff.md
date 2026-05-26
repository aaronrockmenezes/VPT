# World Model Handoff

## Problem

The VPT task asks whether a green camera object can see a red goal from the camera object's perspective. Linear probes on third-person views are weak because the model must infer another viewpoint. The world-model idea is to train a navigable model that can use the agent's first-person observation plus actions to move to the camera object's pose and approximate its view.

## Intended Bridge

1. Generate envs with the same visual distribution as VPTnav v18.
2. Ensure first agent frame sees both goal and camera object.
3. Use A* to produce action rollouts from this valid first viewpoint to the camera object's pose.
4. Train a world model on `(frame_t, action_t) -> future representation/frame`.
5. At inference/probing time, search or optimize actions in the learned model toward the camera pose/view.
6. Use the predicted/assumed camera view for VPT linear probing.

## Why The A* Dataset Was Fixed

The early A* dataset used the initial reset pose or random near-camera spawn, which did not guarantee the camera object was visible from frame 0. That breaks the world-model premise and makes first-frame VPT probing invalid.

The corrected A* start logic:

- Uses `vpt_env_v18.py` as the visual source of truth.
- Runs the VPT valid-viewpoint pipeline for A* starts.
- Requires at least one valid viewpoint.
- Tests all valid candidates for A* viability.
- Chooses the first candidate with a valid A* path.
- Starts rollout from that verified candidate.

## Visual Logic To Preserve

From `vpt_env_v18.py` / v18:

- Camera and goal spawn/randomization distribution.
- Obstacle randomization/material/color/light setup.
- Viewpoint generation radius/jitter.
- Agent yaw jitter constrained to keep the camera object inside FOV.
- Camera-object quaternion plus `+90 deg` sensor yaw offset.
- 30 sim/render settle steps before first saved frame.
- Semantic-pixel validation thresholds.

Do not replace this with v17 visuals.

## RL Constraint Borrowed From v17

Only borrow the valid-start constraint style:

```text
req_valid_viewpoints = 1
start_source = valid_viewpoint_0
abs(start_x - cam_x) <= 6
abs(start_y - cam_y) <= 6
reject if abs(dx) < 3 and abs(dy) < 3
collision/geometry checks must pass
```

## Active Canonical Dataset

**Path on Oscar:**

```text
/users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k
```

**Size:** 20,000 environments.

**Label balance (enforced at compile):**

| Reason | Label | Count |
|---|---|---:|
| `in_view` | Yes | 10,000 |
| `occluded` | No | 5,000 |
| `outside_fov` | No | 5,000 |

Binary: 50% Yes / 50% No.

**Split config (in `master_labels.json`):**

```text
train_frac: 0.8  →  16,000 envs
val_frac:   0.1  →   2,000 envs
test_frac:  0.1  →   2,000 envs
seed: 0
```

Split metadata lives in `master_labels.json["environments"][env_id]["split"]`.
The world-model loader (`CompiledAStarIndex` / `WMSequenceDataset`) reads this directly — no separate train/test folders needed.

**ImageFolder split (for linear probes only):**

A separate 80/20 probe split was created with:

```bash
python scripts/make_vpt_probe_split.py \
  /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k \
  --train_frac 0.8 --seed 42 --overwrite
```

This creates symlinked `train/` and `test/` dirs (first RGB frame per env) plus `visibility_labels.json`.
These are probe-only artifacts. The world model should ignore them and use the full rollout frames.

## Dataset Format

```text
root/
├── master_labels.json
├── visibility_labels.json      ← probe split only, ignore for WM
├── train/{Yes,No}/env_*/       ← probe split only, ignore for WM
├── test/{Yes,No}/env_*/        ← probe split only, ignore for WM
├── RGB/{Yes,No}/env_*/step_*.png
├── Semantic/{Yes,No}/env_*/step_*.png
└── cam/{Yes,No}/env_*/
    ├── actions.txt
    ├── meta.json
    └── final_cam_semantic.png
```

World model loader uses:

- `master_labels.json`
- `RGB/{Yes,No}/env_*/` (full rollout frames)
- `cam/{Yes,No}/env_*/actions.txt`

Does not require `Semantic/`, `final_cam_semantic.png`, or `meta.json` for training.

## Metadata Fields To Use

Per-env fields in `master_labels.json["environments"][env_id]`:

- `label`: `Yes` or `No`.
- `reason`: `in_view`, `occluded`, or `outside_fov`.
- `split`: `train`, `val`, or `test`.
- `n_frames`: rollout frame count (not counting `final_*.png`).
- `src`, `src_id`: provenance from original compiled node dir.

Per-env action file: `cam/{Yes,No}/env_{id}/actions.txt`.
Action token mapping used by world model loader:

```text
fwd / forward / 0      -> 0
left / turn_left / 2   -> 1
right / turn_right / 3 -> 2
```

## Linear Probe Baselines (2026-05-14)

First-frame RGB only, 80/20 probe split, single run, `run_linear_probe.py`.

| Model | Train acc | Test acc |
|---|---:|---:|
| `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | 55.6% | 55.1% |
| `vit_small_patch14_reg4_dinov2.lvd142m` | 60.7% | 61.2% |

Both near chance (50%). First-frame linear readout is weak — validates the world-model motivation.
These are single-run, no variance estimate. Treat as rough baseline only.

## World Model Launch (Oscar)

Repo:

```text
/users/arock3/data/arock3/world_models/world_action_jepa
```

World model training:

```bash
sbatch --export=ALL,\
REPO_DIR=/users/arock3/data/arock3/world_models/world_action_jepa,\
DATA_ROOT=/users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k,\
CONFIG=configs/vit_tiny.yaml \
slurm/train_wm_4gpu.slurm
```

Smoke test first (400-env subset):

```bash
bash scripts/compile_400_env_smoke.sh
bash scripts/run_wm_smoke_5ep.sh
```

Checkpoint root:

```text
/users/arock3/scratch/world_models_new/checkpoints/
```

## Known Risks

- `occluded` examples were historically rarer under corrected first-frame-visible start constraints. An occluded-heavy top-up was run; verify breakdown in `master_labels.json` before training.
- First viable A* candidate selection may bias start distribution toward lower candidate indices.
- `sanity_check_compiled_a_star.py --count_images` can report false `n_frames` mismatches because it counts `final_*.png` in RGB dirs. Use `--deep` only (without `--count_images`) to avoid this.
- `train/`, `test/` symlink dirs are first-frame probe artifacts — do not confuse with full rollout data.

## Immediate Next Steps For New Chat

1. Read `docs/wiki.md`, `docs/a_star_pipeline.md`, and this file.
2. Verify dataset breakdown: `python scripts/sanity_check_compiled_a_star.py /users/arock3/scratch/VPT_DATA_A_STAR/v18_compiled_20k --deep --breakdown`.
3. Run smoke test against v18_compiled_20k.
4. Launch full WM training with `vit_tiny.yaml` on 4 GPUs.
5. After WM training, run linear probe from dreamed latent and compare against first-frame baselines above.
