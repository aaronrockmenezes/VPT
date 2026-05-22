# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Cube-Game-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10_000, help="Interval between video recordings (in steps).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint. If None, train from scratch.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video

args_cli.enable_cameras = True
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
import time
from datetime import datetime
import wandb

import omni
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy

from sb3_contrib import RecurrentPPO

# Add WandB callback import
from wandb.integration.sb3 import WandbCallback

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

from isaaclab.utils.dict import print_dict
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import cube_game.tasks  # noqa: F401
from cnn_feat_extractor import CustomCNN  # noqa: F401

class VideoUploadCallback(BaseCallback):
    """Custom callback for uploading videos to WandB."""
    
    def __init__(self, video_folder, check_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.video_folder = video_folder
        self.check_freq = check_freq
        self.uploaded_video = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._upload_videos()
        return True
    
    def _upload_videos(self):
        """Upload any new videos to wandb."""
        if os.path.exists(self.video_folder):
            # Get all mp4 files
            video_files = [f for f in os.listdir(self.video_folder) if f.endswith(".mp4")]
            
            if video_files:
                # Sort by creation time to get the latest
                video_files.sort(key=lambda f: os.path.getctime(os.path.join(self.video_folder, f)))
                
                # Upload all videos that haven't been uploaded yet
                for video_file in video_files:
                    video_path = os.path.join(self.video_folder, video_file)
                    
                    # Check if this video is new (not uploaded before)
                    if video_file not in self.uploaded_video:
                        # Wait a bit to ensure video is fully written
                        time.sleep(0.5)
                        
                        try:
                            # Upload to wandb
                            wandb.log({
                                "training_video": wandb.Video(video_path, format="mp4"),
                                "global_step": self.num_timesteps
                            })
                            
                            if self.verbose > 0:
                                print(f"[VideoUploadCallback] Uploaded video: {video_file}")
                            
                            self.uploaded_video.append(video_file)
                        except Exception as e:
                            if self.verbose > 0:
                                print(f"[VideoUploadCallback] Failed to upload video {video_file}: {e}")

class CustomLogCallback(LogEveryNTimesteps):
    def __init__(self, n_steps):
        super().__init__(n_steps)
    
    def _on_step(self) -> bool:
        # Call parent to handle the regular logging timing
        should_log = super()._on_step()
        
        # If it's time to log, add our custom metrics
        if should_log and len(self.locals.get("infos", [])) > 0:
            infos = self.locals["infos"]
            
            # Extract our three key metrics from info
            if "distance_to_goal" in infos[0]:
                avg_distance = np.mean([info.get("distance_to_goal", 0) for info in infos])
                self.logger.record("custom/distance_to_goal", avg_distance)
            
            if "success_rate" in infos[0]:
                success_rate = np.mean([info.get("success_rate", 0) for info in infos])
                self.logger.record("custom/success_rate", success_rate)
            
            if "timeout_rate" in infos[0]:
                timeout_rate = np.mean([info.get("timeout_rate", 0) for info in infos])
                self.logger.record("custom/timeout_rate", timeout_rate)
            
            if "kill_zone_rate" in infos[0]:
                kill_zone_rate = np.mean([info.get("kill_zone_rate", 0) for info in infos])
                self.logger.record("custom/kill_zone_rate", kill_zone_rate)

        return should_log

class EntropySchedulerCallback(BaseCallback):
    """Custom callback for adjusting entropy coefficient during training."""
    
    def __init__(self, 
                 initial_entropy_coef=0.01, 
                 final_entropy_coef=0.001, 
                 decay_type="linear",  # "linear", "exponential", "step", "custom"
                 decay_steps=None,     # Total steps for decay (None = use total_timesteps)
                 step_schedule=None,   # For step decay: [(step, entropy_coef), ...]
                 custom_schedule_fn=None,  # Custom function: step -> entropy_coef
                 verbose=1):
        super().__init__(verbose)
        self.initial_entropy_coef = initial_entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.step_schedule = step_schedule or []
        self.custom_schedule_fn = custom_schedule_fn
        self.current_entropy_coef = initial_entropy_coef
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        if self.decay_steps is None:
            # Use total timesteps from the model
            self.decay_steps = self.model.learn_total_timesteps if hasattr(self.model, 'learn_total_timesteps') else 1_000_000
        
        if self.verbose > 0:
            print(f"[EntropyScheduler] Starting entropy coefficient: {self.initial_entropy_coef}")
            print(f"[EntropyScheduler] Target entropy coefficient: {self.final_entropy_coef}")
            print(f"[EntropyScheduler] Decay type: {self.decay_type}")
            print(f"[EntropyScheduler] Decay steps: {self.decay_steps}")
        
        # Set initial entropy coefficient
        self.model.ent_coef = self.initial_entropy_coef
        self.current_entropy_coef = self.initial_entropy_coef
        
    def _on_step(self) -> bool:
        """Called at each step to update entropy coefficient."""
        progress = min(self.num_timesteps / self.decay_steps, 1.0)
        
        if self.decay_type == "linear":
            new_entropy_coef = self._linear_decay(progress)
        elif self.decay_type == "exponential":
            new_entropy_coef = self._exponential_decay(progress)
        elif self.decay_type == "step":
            new_entropy_coef = self._step_decay()
        elif self.decay_type == "custom" and self.custom_schedule_fn:
            new_entropy_coef = self.custom_schedule_fn(self.num_timesteps)
        else:
            new_entropy_coef = self.current_entropy_coef
        
        # Update model's entropy coefficient
        if abs(new_entropy_coef - self.current_entropy_coef) > 1e-8:
            self.model.ent_coef = new_entropy_coef
            self.current_entropy_coef = new_entropy_coef
            
            if self.verbose > 1:
                print(f"[EntropyScheduler] Step {self.num_timesteps}: Updated entropy coefficient to {new_entropy_coef:.6f}")
        
        # Log to tensorboard/wandb
        self.logger.record("entropy/entropy_coef", self.current_entropy_coef)
        
        return True
    
    def _linear_decay(self, progress: float) -> float:
        """Linear decay from initial to final entropy coefficient."""
        return self.initial_entropy_coef + progress * (self.final_entropy_coef - self.initial_entropy_coef)
    
    def _exponential_decay(self, progress: float) -> float:
        """Exponential decay from initial to final entropy coefficient."""
        import math
        decay_rate = math.log(self.final_entropy_coef / self.initial_entropy_coef)
        return self.initial_entropy_coef * math.exp(decay_rate * progress)
    
    def _step_decay(self) -> float:
        """Step-wise decay based on predefined schedule."""
        current_entropy = self.initial_entropy_coef
        for step, entropy_coef in self.step_schedule:
            if self.num_timesteps >= step:
                current_entropy = entropy_coef
            else:
                break
        return current_entropy


def episode_record(episode_idx):
    print(episode_idx)
    if episode_idx % 4 == 0:
        return True
    return False

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")
    
    policy_kwargs = {
        "normalize_images": False,
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {
            "features_dim": 512
        },
        "lstm_hidden_size": 512,
        "net_arch": {
            "pi": [128, 128],
            "vf": [128, 128]
        }
    }

    # agent_cfg["policy_kwargs"] = policy_kwargs
    
    is_recurrent = False
    if policy_arch not in ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]:
        is_recurrent = True

    # Create log directory
    if is_recurrent:
        log_dir = f"./logs_new/cube_game_LSTM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        log_dir = f"./logs_new/cube_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % (50_000 // args_cli.num_envs) == 0,
            # "episode_trigger": episode_record,
            "video_length": 200,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=False)

    print(f"Action Space = {env.action_space}, Observation Space = {env.observation_space}")
    env = VecNormalize(env, training=True)

    # Initialize Weights & Biases
    run_name = f"cube_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        entity="aaron_rock_menezes-private",  # Your wandb entity
        # project="isaaclab",
        project="cube_game",
        name=run_name,
        config={
            "notes": "pre-trained agent + 0.25 overlap + 0.8 thickness stick (mat size)",
            "algorithm": "PPO" if not is_recurrent else "RecurrentPPO",
            "env_name": args_cli.task,
            "num_envs": env_cfg.scene.num_envs,
            "total_timesteps": n_timesteps,
            "seed": agent_cfg["seed"],
            "policy_architecture": policy_arch,
            **agent_cfg  # Include all agent configuration
        },
        sync_tensorboard=True,  # Sync tensorboard logs to wandb
        monitor_gym=True,
        save_code=True
    )
    
    if is_recurrent:
        agent = RecurrentPPO(
            policy_arch,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **agent_cfg
        )
    else:
        # create agent with tensorboard logging
        agent = PPO(
            policy_arch, 
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **agent_cfg
        )
    
    # Check if checkpoint exists, train from scratch if it doesn't and log it
    if args_cli.checkpoint is not None:
        if Path(args_cli.checkpoint).exists():
            print(f"Loading checkpoint from: {args_cli.checkpoint}")
            agent = agent.load(args_cli.checkpoint, env, print_system_info=True)
        else:
            print(f"[WARNING] Checkpoint path {args_cli.checkpoint} does not exist. Training from scratch.")

    # # Create WandB callback
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models_new/{run_name}",
        verbose=2,
    )
    
    video_callback = VideoUploadCallback(
        video_folder=os.path.join(log_dir, "videos", "train"),
        check_freq=200_000 // args_cli.num_envs,
        verbose=1
    )

    entropy_callback = EntropySchedulerCallback(
        initial_entropy_coef=agent_cfg['ent_coef'] if 'ent_coef' in agent_cfg else 0.01,
        decay_type="step",
        step_schedule=[     
            (0, 0.05),          # Start with 0.01
            (200_000, 0.025),   # At 200k steps, reduce to 0.025
            (800_000, 0.01),   # At 800k steps, reduce to 0.01
        ],
        verbose=1
    )

    callbacks = [
        CustomLogCallback(n_steps=20_000),  # Replaces both LogEveryNTimesteps and CustomInfoCallback
        wandb_callback,
        video_callback,
        # entropy_callback,
        # LogEveryNTimesteps(n_steps=10_000),  # Keep this for regular logging
        CheckpointCallback(
            save_freq=500_000,
            save_path=log_dir,
            name_prefix="cube_game_model",
            save_vecnormalize=True,
        )
    ]

    # train the agent
    agent.learn(
        total_timesteps=n_timesteps,
        progress_bar=True,
        callback=callbacks,
        log_interval=1,  # Log after every policy update (this is key!)
    )

    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    # close the simulator
    env.close()
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
