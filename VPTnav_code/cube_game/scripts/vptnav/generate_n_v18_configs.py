"""Generate N VPT-v18 saved env configs in one Isaac process, then exit."""

from __future__ import annotations

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate N VPT-v18 configs.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="VPT-v18")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--target_configs", type=int, default=10)
parser.add_argument("--max_attempts", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.num_envs = 1
args_cli.enable_cameras = True
args_cli.headless = True
os.environ["BASE_PATH"] = args_cli.base_path
os.environ.setdefault("NODE_ID", "sample")
os.environ.setdefault("GPU_ID", os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import cube_game.tasks  # noqa: F401


def _configs():
    return sorted(
        glob.glob(
            os.path.join(
                args_cli.base_path,
                "data",
                "data_node*_gpu*",
                "configs",
                "env_*_config.json",
            )
        )
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    seen = set(_configs())
    generated = []
    for attempt in range(args_cli.max_attempts):
        actions = torch.full((1,), 5, dtype=torch.long, device=env.unwrapped.device)
        env.step(actions)

        new_configs = [path for path in _configs() if path not in seen]
        for path in new_configs:
            seen.add(path)
            generated.append(path)
            print(f"[CONFIG {len(generated):02d}/{args_cli.target_configs}] {path}", flush=True)
            if len(generated) >= args_cli.target_configs:
                env.close()
                return

        if not new_configs:
            print(f"[WAIT] attempt={attempt + 1}/{args_cli.max_attempts}", flush=True)

    env.close()
    raise RuntimeError(
        f"Generated {len(generated)}/{args_cli.target_configs} configs "
        f"after {args_cli.max_attempts} attempts."
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
