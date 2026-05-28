"""Generate one VPT-v18 saved env config, then exit."""

from __future__ import annotations

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate one VPT-v18 config.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="VPT-v18")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--max_attempts", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.num_envs = 1
args_cli.enable_cameras = True
args_cli.headless = True
os.environ["BASE_PATH"] = args_cli.base_path
os.environ.setdefault("NODE_ID", "replay")
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

    before = {path: os.path.getmtime(path) for path in _configs()}
    for attempt in range(args_cli.max_attempts):
        actions = torch.full((1,), 5, dtype=torch.long, device=env.unwrapped.device)
        env.step(actions)
        changed_configs = [
            path for path in _configs()
            if path not in before or os.path.getmtime(path) > before[path]
        ]
        if changed_configs:
            newest = max(changed_configs, key=os.path.getmtime)
            print(newest, flush=True)
            env.close()
            return
        print(f"[WAIT] attempt={attempt + 1}/{args_cli.max_attempts}", flush=True)

    env.close()
    raise RuntimeError(f"No config generated after {args_cli.max_attempts} attempts.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
