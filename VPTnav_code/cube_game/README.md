# VPTnav — Visual Perspective Taking Data Collection

Isaac Lab extension for generating Visual Perspective Taking (VPT) datasets in simulation. Spawns randomized 3D scenes with obstacles, samples viewpoints around a goal object, and saves RGB / depth / semantic images with visibility labels.

This project is **data-collection only**. RL training code has been removed; the environment classes still inherit from `DirectRLEnv` for Isaac Lab compatibility, but no agent is trained here.

## Requirements

- NVIDIA GPU (Isaac Sim / Isaac Lab requirement)
- Isaac Lab installed (see [Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- Singularity image `isaac-lab.simg` available on the HPC for containerized runs

## Tasks

Two gym IDs are registered (see `source/cube_game/cube_game/tasks/direct/cube_game/__init__.py`):

| Task ID         | Entry point        | Notes                                                                 |
|-----------------|--------------------|-----------------------------------------------------------------------|
| `VPT-v0`        | `vpt_env.py`       | Main env. RGB + semantic + depth, single viewpoint pool.              |
| `VPT-Depth-v0`  | `vpt_env_depth.py` | Depth-aware variant. 50/50 camera-proximal vs goal-proximal viewpoints, depth label per image. |

Both share `vpt_env_cfg.py` for scene / camera / arena config.

## Install

```bash
# Use the isaaclab Python interpreter (not system python)
python -m pip install -e source/cube_game
```

Verify registration:

```bash
python scripts/list_envs.py
```

## Run data collection

`scripts/keyboard_agent.py` runs an env with action `-1` (no-op) so the env's `_reset_idx` loop drives data collection. With keyboard control:

| Key       | Action |
|-----------|--------|
| `W` / `↑` | forward |
| `S` / `↓` | backward |
| `A` / `←` | turn left |
| `D` / `→` | turn right |
| no input  | soft reset (default; triggers data collection step) |

```bash
python scripts/keyboard_agent.py --task=VPT-v0 --num_envs=30
python scripts/keyboard_agent.py --task=VPT-Depth-v0 --num_envs=30
```

## Output structure

Each env writes to `{base_path}/{RGB|Depth|Semantic}/{Yes|No}/env_{idx}/` plus a JSON of visibility labels:

```json
{
  "environments": {
    "0": {"label": "Yes", "reason": "in_view"},
    "1": {"label": "No",  "reason": "occluded"}
  },
  "statistics": {
    "total_environments": N,
    "yes_count": X,
    "no_count": Y,
    "by_reason": {"in_view": ..., "occluded": ..., "outside_fov": ...}
  }
}
```

Labels are pre-allocated at the dataset level in a 50 / 25 / 25 ratio — `in_view` / `occluded` / `outside_fov`. Each env is assigned the next slot from the pool when reset.

`VPT-Depth-v0` additionally writes a per-image depth label (`1` = agent closer to camera, `0` = closer to goal).

## Configuration

Key knobs live in `vpt_env_cfg.py`:

- `num_envs` — parallel envs (default 32)
- Camera resolution: 512×512, RGB + semantic + depth
- Arena: 20 × 20 m
- Goal: 0.25 m red sphere
- Object pool: 32 VPT objects, 16 active per env (cross, L, cuboid, cylinder, cone, table_A, table_B, A, H, I, Z, Bench)
- Agent camera pitch: 15°

Output paths are still hardcoded inside `vpt_env.py` / `vpt_env_depth.py` (search for `/home/` and `/mnt/`). Edit those before running.

## Source layout

```
cube_game/
├── source/cube_game/cube_game/tasks/direct/cube_game/
│   ├── __init__.py            # gym.register entries
│   ├── vpt_env.py             # VPT-v0
│   ├── vpt_env_depth.py       # VPT-Depth-v0
│   ├── vpt_env_cfg.py         # shared cfg (VPTEnvCfg)
│   ├── spawn_boundary.py      # arena walls, materials, object configs
│   └── env_timer.py           # per-env timing utility
├── scripts/
│   ├── keyboard_agent.py
│   ├── list_envs.py
│   ├── compile_results.py
│   └── ...
├── job_array/                 # SLURM job array submission
├── assets/, mass_assets/      # USD assets
└── isaac-lab.simg             # Singularity image (HPC)
```

## Code formatting

```bash
pip install pre-commit
pre-commit run --all-files
```
