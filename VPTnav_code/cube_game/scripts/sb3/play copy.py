# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Cube-Game-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import RecurrentPPO

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

import cube_game.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    """Play with stable-baselines agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_root_path = os.path.join("logs_new", "sb3", train_task_name)
    log_root_path = os.path.abspath(log_root_path)
    # checkpoint and log_dir stuff
    if args_cli.use_pretrained_checkpoint:
        checkpoint_path = get_published_pretrained_checkpoint("sb3", train_task_name)
        if not checkpoint_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # FIXME: last checkpoint doesn't seem to really use the last one'
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint, sort_alpha=False)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    vec_norm_path = checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
    vec_norm_path = Path(vec_norm_path)

    # normalize environment (if needed)
    if vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
    elif "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        )

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")

    policy_arch = agent_cfg.pop("policy")

    if policy_arch not in ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]:
        agent = RecurrentPPO.load(checkpoint_path, env, print_system_info=True)
    else:
        agent = PPO.load(checkpoint_path, env, print_system_info=True)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    timestep = 0
    episode_starts = np.ones((args_cli.num_envs,), dtype=bool)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        if policy_arch not in ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]:
            lstm_states = None
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if policy_arch not in ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]:
                actions, lstm_states = agent.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                timestep += 1
            else:
                actions, _ = agent.predict(obs, deterministic=True)
                # print(actions[0])
            # env stepping
            obs, rew, done, info = env.step(actions)
            episode_starts = done

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
