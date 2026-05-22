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
parser.add_argument("--num_envs", type=int, default=30, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="VPT-v18-camera", help="Name of the task.")

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
import traceback

import cube_game.tasks

def main():
    """Keyboard control agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
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
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard input
            delta_pose = keyboard.advance()
            
            # Map keyboard input to actions
            # delta_pose is [x, y, z, roll, pitch, yaw]
            action = -1  # Default: no action
            
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
            elif delta_pose[2] < 0:
                action = 5
            elif delta_pose[2] > 0:
                action = 6
            # Create action tensor for all environments
            # Use the same action for all envs (controlled by keyboard)
            if action >= 0:
                actions = torch.full((env.unwrapped.num_envs,), action, 
                                    dtype=torch.long, device=env.unwrapped.device)
            else:
                # No key pressed - stay still (you can use action 0 or define a "no-op" action)
                actions = torch.full((env.unwrapped.num_envs,), 5, 
                                    dtype=torch.long, device=env.unwrapped.device)
            
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
    finally:
        # close sim app
        simulation_app.close()
