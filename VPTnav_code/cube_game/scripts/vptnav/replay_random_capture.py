"""Replay one saved VPT-v18 config, run random actions, and capture frames."""

from __future__ import annotations

import argparse
import os
import random

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay VPT-v18 config and capture random rollout frames.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="VPT-v18")
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--capture_dir", type=str, required=True)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.num_envs = 1
args_cli.enable_cameras = True
args_cli.headless = True
os.environ["VPT_REPLAY_CAPTURE_DIR"] = args_cli.capture_dir

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import cube_game.tasks  # noqa: F401


def main():
    if args_cli.seed is not None:
        random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.config_file = args_cli.config_file
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    labels = {0: "forward", 1: "backward", 2: "left", 3: "right"}
    for step_idx in range(args_cli.steps):
        action = random.randint(0, 3)
        env.step(torch.tensor([action], dtype=torch.long, device=env.unwrapped.device))
        env.step(torch.tensor([7], dtype=torch.long, device=env.unwrapped.device))
        print(f"[STEP {step_idx + 1:02d}] action={action} {labels[action]}", flush=True)

    print(os.path.join(args_cli.capture_dir, "env_0"), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
