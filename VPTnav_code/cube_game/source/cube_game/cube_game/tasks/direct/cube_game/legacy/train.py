import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="PPO training with CNN-LSTM and camera images")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Cube-Game-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import parse_env_cfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg:dict):
    num_envs = 9

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    agent_cfg = process_sb3_cfg(agent_cfg)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")
    env_cfg = parse_env_cfg(
        args_cli.task, device="cuda", num_envs=args_cli.num_envs
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    env = Sb3VecEnvWrapper(env)

    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)

    agent.learn(
        total_timesteps=n_timesteps,
        progress_bar=True
        )
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()