# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with keyboard control agent."""

"""Launch Isaac Sim Simulator first."""


import signal
import os
import sys

def force_exit(signum, frame):
    print(f"\n Force killing self (PID: {os.getpid()})...")
    os.kill(os.getpid(), signal.SIGKILL)

# Register handlers
signal.signal(signal.SIGINT, force_exit)   # Ctrl+C
signal.signal(signal.SIGTSTP, force_exit)  # Ctrl+Z

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=36, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="VPT2-v3", help="Name of the task.")
parser.add_argument(
    "--config_file",
    type=str,
    default=None,
    help="Replay one saved environment config JSON instead of sampling a new env.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
    
"""Rest everything follows."""

import gymnasium as gym
import torch


print(f"Running with force-kill enabled (PID: {os.getpid()})")

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

import cube_game.tasks  # noqa: F401


def main():
    """Keyboard control agent with Isaac Lab environment."""
    if args_cli.config_file:
        print(f"[INFO]: Loading config file: {args_cli.config_file}")
        args_cli.num_envs = 1

    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.config_file:
        env_cfg.config_file = args_cli.config_file

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # create keyboard interface
    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.5, rot_sensitivity=0.5))
    keyboard.reset()
    
    print("\n[INFO]: Keyboard Controls:")
    print("  W / Up Arrow    -> Forward (action 0)")
    print("  S / Down Arrow  -> Backward (action 1)")
    print("  A / Left Arrow  -> Turn Left (action 2)")
    print("  D / Right Arrow -> Turn Right (action 3)")
    print("  ESC             -> Quit\n")
    
    # reset environment
    env.reset()
    steps = 0
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard input
            delta_pose = keyboard.advance()
            
            # Map keyboard input to actions
            # delta_pose is [x, y, z, roll, pitch, yaw]
            action = -1  # Default: no action
            # if steps < 100:
            #     steps += 1

            # if steps == 101:
            #     sys.exit()
            
            # Forward: W or Up (positive x)
            if delta_pose[0] > 0:
                action = 0
            # Backward: S or Down (negative x)
            elif delta_pose[0] < 0:
                action = 1
            # Turn Left: A or Left (positive yaw)
            elif delta_pose[1] > 0:
                action = 2
            # Turn Right: D or Right (negative yaw)
            elif delta_pose[1] < 0:
                action = 3
            # RL only Reset: Q (positive z)
            elif delta_pose[2] > 0:
                action = 5
            # Data collection Reset: E (negative z)
            elif delta_pose[2] < 0:
                action = 6
            # Create action tensor for all environments
            # Use the same action for all envs (controlled by keyboard)
            if action >= 0:
                actions = torch.full((env.unwrapped.num_envs,), action, 
                                    dtype=torch.long, device=env.unwrapped.device)
            else:
                # No key pressed - data collection
                actions = torch.full((env.unwrapped.num_envs,), 5, 
                                    dtype=torch.long, device=env.unwrapped.device)
                # actions = torch.randint(0, 6, (env.unwrapped.num_envs,), 
                        # device=env.unwrapped.device)
            
            # if steps == 100:
            #     actions = torch.full((env.unwrapped.num_envs,), 5, 
            #                         dtype=torch.long, device=env.unwrapped.device)
            #     steps = 0
            
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
