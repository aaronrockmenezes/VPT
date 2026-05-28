"""Replay many VPT-v18 configs one by one and capture random action frames."""

from __future__ import annotations

import argparse
import glob
import os
import random

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay VPT-v18 configs and capture random rollout frames.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="VPT-v18")
parser.add_argument("--config_glob", type=str, required=True)
parser.add_argument("--capture_dir", type=str, required=True)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--limit", type=int, default=10)
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


def _action_label(action: int) -> str:
    return {0: "forward", 2: "left", 3: "right"}.get(action, f"a{action}")


def main():
    if args_cli.seed is not None:
        random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)

    config_files = sorted(glob.glob(args_cli.config_glob))[:args_cli.limit]
    if not config_files:
        raise FileNotFoundError(f"No configs matched: {args_cli.config_glob}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.config_file = config_files[0]
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped

    actions_pool = [0, 2, 3]
    for cfg_idx, config_path in enumerate(config_files):
        if cfg_idx > 0:
            u.load_env_config_from_json(config_path, target_env_id=0)

        env_capture_dir = os.path.join(args_cli.capture_dir, f"sample_{cfg_idx:02d}")
        os.environ["VPT_REPLAY_CAPTURE_DIR"] = env_capture_dir
        u.capture_frame_counters[0] = 0

        for step_idx in range(args_cli.steps):
            action = random.choice(actions_pool)
            env.step(torch.tensor([action], dtype=torch.long, device=u.device))
            env.step(torch.tensor([7], dtype=torch.long, device=u.device))
            print(
                f"[SAMPLE {cfg_idx:02d} STEP {step_idx + 1:02d}] "
                f"action={action} {_action_label(action)}",
                flush=True,
            )

        print(f"[CAPTURED] {env_capture_dir}/env_0", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
