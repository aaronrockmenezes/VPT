# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import random  # added

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, sample_gaussian
from isaaclab.utils import math as math_utils  # added

from .cube_game_env_cfg import CubeGameEnvCfg
from .check_collisions import check_collisions_batched  # added

import gymnasium as gym
from gymnasium.spaces import Discrete, Box


class CubeGameEnv(DirectRLEnv):

    cfg: CubeGameEnvCfg

    def __init__(
        self, cfg: CubeGameEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # New
        self.action_scale = self.cfg.action_scale
        self.boundary_limits = self.cfg.boundary_limits
        self.center_to_boundary = torch.abs(torch.tensor(self.boundary_limits).view(-1)[0])
        self.goal_mat_size = 0.6  # side length of the goal mat
        # Fix: Use self.num_envs instead of cfg.num_envs to get actual number of environments
        self.env_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_reasons = ["" for _ in range(self.num_envs)]  # Match actual num_envs
        
        # Episode tracking for info dictionary
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Add action counting for each environment
        self.episode_action_counts = torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device)  # 4 actions: forward, backward, left, right
        
        # Verbose control - set to 0 by default
        self.verbose = 0  # 0 = no prints, 1 = some prints, 2 = all prints

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The Cube Game environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        """Setup the scene with the cartpole and camera."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(1000,1000)))
        # self._cartpole = Articulation(self.cfg.robot_cfg)
        self._agent = RigidObject(self.cfg.agent)
        self._goal = RigidObject(self.cfg.goal_ball)
        self._mat = RigidObject(self.cfg.goal_mat)
        self._wall = RigidObject(self.cfg.wall)
        self._boundary_top = RigidObject(self.cfg.top_wall)
        self._boundary_bottom = RigidObject(self.cfg.bottom_wall)
        self._boundary_left = RigidObject(self.cfg.left_wall)
        self._boundary_right = RigidObject(self.cfg.right_wall)


        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation and sensors to scene
        # self.scene.articulations["cartpole"] = self._cartpole
        self.scene.rigid_objects["agent"] = self._agent
        self.scene.rigid_objects["goal"] = self._goal
        self.scene.rigid_objects["mat"] = self._mat
        self.scene.rigid_objects["wall"] = self._wall
        self.scene.rigid_objects["boundary_top"] = self._boundary_top
        self.scene.rigid_objects["boundary_bottom"] = self._boundary_bottom
        self.scene.rigid_objects["boundary_left"] = self._boundary_left
        self.scene.rigid_objects["boundary_right"] = self._boundary_right

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    
    def move_agent(self, actions, env_ids: Sequence[int] | None = None):
        """Apply random movement to each agent using action-based logic.
        Actions: 0=forward, 1=backward, 2=turn_left, 3=turn_right
        """
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        # Convert list to tensor if needed
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        device = self._agent.device
        num_envs = len(env_ids)
        max_velocity = 3.0

        # Rotation quaternions for yaw movement (15 degrees per step)
        theta = math.pi / 12  # 15 degrees in radians
        half_theta = theta / 2
        left_rot_quat = torch.tensor(
            [math.cos(half_theta), 0.0, 0.0, math.sin(half_theta)],
            device=device,
        )
        right_rot_quat = torch.tensor(
            [math.cos(half_theta), 0.0, 0.0, -math.sin(half_theta)],
            device=device,
        )

        # Current state for selected envs
        current_pos = self._agent.data.root_pos_w[env_ids].clone()
        current_quat = self._agent.data.root_quat_w[env_ids].clone()

        # Initialize new states
        new_quat = current_quat.clone()
        desired_vel = torch.zeros((num_envs, 6), device=device)

        # Process one random action per environment
        for i, _ in enumerate(env_ids):
            action = actions[i]

            if action == 2:  # Turn left
                new_quat[i] = math_utils.quat_mul(
                    current_quat[i].unsqueeze(0), left_rot_quat.unsqueeze(0)
                ).squeeze(0)
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 3:  # Turn right
                new_quat[i] = math_utils.quat_mul(
                    current_quat[i].unsqueeze(0), right_rot_quat.unsqueeze(0)
                ).squeeze(0)
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            else:
                # 0: forward, 1: backward
                forward_input = 1.0 if action == 0 else -1.0
                local_movement = torch.tensor([0.0, forward_input, 0.0], device=device)
                world_velocity = math_utils.quat_apply(
                    current_quat[i].unsqueeze(0), local_movement.unsqueeze(0)
                ).squeeze(0) * max_velocity
                desired_vel[i, :3] = world_velocity
                desired_vel[i, 3:6] = 0.0

            # Lock Z movement
            desired_vel[i, 2] = 0.0

        # Write pose (updated orientation) and velocity back to sim
        self._agent.write_root_pose_to_sim(torch.cat([current_pos, new_quat], dim=1), env_ids)
        self._agent.write_root_velocity_to_sim(desired_vel, env_ids)
        self._agent.reset()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # Track action counts before applying actions
        for i in range(self.num_envs):
            action = int(self.actions[i].item())
            if 0 <= action <= 3:  # Valid action range
                self.episode_action_counts[i, action] += 1
        
        self.move_agent(self.actions)

    def _get_observations(self) -> dict:
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        if "rgb" in self.cfg.tiled_camera.data_types:
            camera_data = self._tiled_camera.data.output[data_type] / 255.0
            # normalize the camera data for better training results
            mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            camera_data -= mean_tensor
        elif "depth" in self.cfg.tiled_camera.data_types:
            camera_data = self._tiled_camera.data.output[data_type]
            camera_data[camera_data == float("inf")] = 0
        
        # Ensure camera data is of shape (num_envs, height, width, num_channels=3)
        # camera_data = camera_data.view(self.num_envs, self._tiled_camera.height, self._tiled_camera.width, -1)
        # print(f"Old data = {camera_data.shape}")
        camera_data = camera_data.permute(0, 3, 1, 2)
        # print(f"New data = {camera_data.shape}")
        observations = {"policy": camera_data.clone()}

        if self.cfg.write_image_to_file:
            save_images_to_file(observations["policy"], f"cartpole_{data_type}.png")

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards for the current step."""
        collision_mask, collision_types = self._check_collisions()
        
        agent_positions = self._agent.data.root_pos_w.clone()
        goal_positions = self._goal.data.root_pos_w.clone()
        
        total_reward = compute_rewards(
            collision_mask=collision_mask,
            collision_types=collision_types,
            reset_terminated=self.reset_terminated,
            reset_time_outs=self.reset_time_outs,
            agent_positions=agent_positions,
            goal_positions=goal_positions,
            device=self.device,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination and timeout masks."""
        # Reset episode tracking
        self.episode_success.fill_(False)
        self.episode_timeout.fill_(False)
        
        # Check collisions and handle them
        collision_mask, collision_types = self._check_collisions()
        self._handle_collisions(collision_mask, collision_types)
        
        # Get current termination and timeout states
        terminated = self.env_done.clone()
        time_outs = (self.episode_length_buf >= self.max_episode_length)
        
        # Track episode outcomes for info dictionary
        for i in range(self.num_envs):
            if collision_mask[i] and collision_types[i] == "goal":
                self.episode_success[i] = True
            elif time_outs[i] and not terminated[i]:
                self.episode_timeout[i] = True
                if i < len(self.reset_reasons):
                    self.reset_reasons[i] = "timeout"
        
        return terminated, time_outs

    def _get_extras(self) -> dict:
        """Return info dictionary with episode statistics."""
        info = {
            "episode_success": self.episode_success.cpu().numpy(),
            "episode_timeout": self.episode_timeout.cpu().numpy(),
            "episode_action_counts": self.episode_action_counts.cpu().numpy(),  # Shape: (num_envs, 4)
            "episode_forward_count": self.episode_action_counts[:, 0].cpu().numpy(),
            "episode_backward_count": self.episode_action_counts[:, 1].cpu().numpy(), 
            "episode_left_count": self.episode_action_counts[:, 2].cpu().numpy(),
            "episode_right_count": self.episode_action_counts[:, 3].cpu().numpy(),
            "episode_total_actions": self.episode_action_counts.sum(dim=1).cpu().numpy(),
        }
        return info

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        # Convert list to tensor if needed
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # Reset action counts for the environments being reset
        self.episode_action_counts[env_ids] = 0

        super()._reset_idx(env_ids)

        device = self._agent.device
        num_envs = len(env_ids)
        safe_x_range = self.center_to_boundary - 0.25
        safe_y_range = self.center_to_boundary - 0.25
        
        # 1. Respawn mat at default positions
        mat_default_state = self._mat.data.default_root_state[env_ids].clone()
        mat_default_state[:, :3] += self.scene.env_origins[env_ids]
        # spawn the mat at z = 0.001 to avoid clipping
        mat_default_state[:, 2] = 0.001

        self._mat.write_root_pose_to_sim(mat_default_state[:, :7], env_ids)
        self._mat.write_root_velocity_to_sim(torch.zeros_like(mat_default_state[:, 7:]), env_ids)
        
        # 2. Respawn goal at default positions
        goal_default_state = self._goal.data.default_root_state[env_ids].clone()
        goal_default_state[:, :3] += self.scene.env_origins[env_ids]
        goal_default_state[:, 2] = 0.005
        self._goal.write_root_pose_to_sim(goal_default_state[:, :7], env_ids)
        self._goal.write_root_velocity_to_sim(torch.zeros_like(goal_default_state[:, 7:]), env_ids)
        
        # Get mat and goal positions for constraint checking
        mat_positions = mat_default_state[:, :3]
        goal_positions = goal_default_state[:, :3]
        
        # 3. Respawn agent and wall with noise and constraints (merged loop)
        agent_default_state = self._agent.data.default_root_state[env_ids].clone()
        # Don't add env_origins here - we'll handle positioning in the loop
        
        wall_default_state = self._wall.data.default_root_state[env_ids].clone()
        # Don't add env_origins here - we'll handle positioning in the loop
        
        # Get boundary limits as relative offsets from center
        # x_min, x_max = self.boundary_limits[0]
        # y_min, y_max = self.boundary_limits[1]

        # Create safe sampling range (centered around 0 with buffer)
        # border_buffer = 1.0
        
        for i in range(num_envs):
            valid_positions = False
            attempts = 0
            max_attempts = 100
            
            while not valid_positions and attempts < max_attempts:
                # Sample agent position as offsets from environment center
                agent_offset_x = sample_uniform(-safe_x_range, safe_x_range, (1,), device)
                agent_offset_y = sample_uniform(-safe_y_range, safe_y_range, (1,), device)
                agent_new_pos = agent_default_state[i, :3].clone()
                agent_new_pos[0] = self.scene.env_origins[env_ids[i], 0] + agent_offset_x
                agent_new_pos[1] = self.scene.env_origins[env_ids[i], 1] + agent_offset_y
                
                # Sample wall position as offsets from environment center
                wall_offset_x = sample_uniform(-safe_x_range, safe_x_range, (1,), device)
                wall_offset_y = sample_uniform(-safe_y_range, safe_y_range, (1,), device)
                wall_new_pos = wall_default_state[i, :3].clone()
                wall_new_pos[0] = self.scene.env_origins[env_ids[i], 0] + wall_offset_x
                wall_new_pos[1] = self.scene.env_origins[env_ids[i], 1] + wall_offset_y
                
                # Check constraints for both objects together
                mat_center = mat_positions[i, :2]
                mat_half_size = self.goal_mat_size / 2.0
                
                # Agent constraints
                agent_outside_mat = (torch.abs(agent_new_pos[:2] - mat_center) > mat_half_size).any()
                agent_far_from_goal = torch.norm(agent_new_pos[:2] - goal_positions[i, :2]) >= mat_half_size
                
                # Wall constraints
                wall_outside_mat = (torch.abs(wall_new_pos[:2] - mat_center) > mat_half_size).any()
                wall_far_from_goal = torch.norm(wall_new_pos[:2] - goal_positions[i, :2]) >= mat_half_size
                
                if agent_outside_mat and agent_far_from_goal and wall_outside_mat and wall_far_from_goal:
                    agent_default_state[i, :3] = agent_new_pos
                    wall_default_state[i, :3] = wall_new_pos
                    valid_positions = True
                
                attempts += 1
            
            if attempts >= max_attempts and self.verbose >= 1:
                print(f"Warning: Could not find valid positions for env {env_ids[i]} after {max_attempts} attempts")
        
        # Random orientation for wall only
        random_yaw = sample_uniform(0, 1 * math.pi, (num_envs,), device)
        wall_default_state[:, 3] = torch.cos(random_yaw / 2)  # w
        wall_default_state[:, 4] = 0.0  # x
        wall_default_state[:, 5] = 0.0  # y
        wall_default_state[:, 6] = torch.sin(random_yaw / 2)  # z
        
        # Write states to simulation
        self._agent.write_root_pose_to_sim(agent_default_state[:, :7], env_ids)
        self._agent.write_root_velocity_to_sim(torch.zeros_like(agent_default_state[:, 7:]), env_ids)
        
        self._wall.write_root_pose_to_sim(wall_default_state[:, :7], env_ids)
        self._wall.write_root_velocity_to_sim(torch.zeros_like(wall_default_state[:, 7:]), env_ids)
        
        # Safe printing with bounds checking
        if self.verbose >= 1:
            valid_env_ids = [env_id for env_id in env_ids.tolist() if env_id < len(self.reset_reasons)]
            reset_reasons_for_envs = [self.reset_reasons[env_id] if env_id < len(self.reset_reasons) else "unknown" 
                                     for env_id in env_ids.tolist()]
            # print(f"[INFO] Reset environments: {env_ids.tolist()} for reasons: {reset_reasons_for_envs}")
            num_goal_collision = reset_reasons_for_envs.count("goal_collision")
            if num_goal_collision >= 1:
                print(f"Reset {num_goal_collision} envs due to goal collision")

        # Reset environment state with bounds checking
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            if env_id_item < len(self.env_done):
                self.env_done[env_id_item] = False
            if env_id_item < len(self.reset_reasons):
                self.reset_reasons[env_id_item] = ""

    def _check_collisions(self):
        """Check for collisions across all environments."""
        agents = self._agent
        goal_balls = self._goal
        goal_mats = self._mat
        
        collision_mask, collision_types = check_collisions_batched(
            agents=agents,
            goal_balls=goal_balls,
            goal_mats=goal_mats,
            boundary_limits=self.boundary_limits,
            mat_side=self.goal_mat_size
        )
        
        return collision_mask, collision_types
    
    def _handle_collisions(self, collision_mask, collision_types):
        """Handle collision events and reset environments."""
        if not collision_mask.any():
            return
        
        # Get environments that need resetting
        envs_to_reset = collision_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        
        # Mark environments as done and record reasons
        for env_id in envs_to_reset:
            if env_id < len(self.env_done):
                self.env_done[env_id] = True
                collision_type = collision_types[env_id] if env_id < len(collision_types) else "unknown"
                if env_id < len(self.reset_reasons):
                    self.reset_reasons[env_id] = f"{collision_type}_collision"
        # print(f"Envs to reset = {envs_to_reset}")
        # print(f"Reset reasons = {[self.reset_reasons[env_id] if env_id < len(self.reset_reasons) else 'unknown' for env_id in envs_to_reset]}")

        # Reset all colliding environments
        # self._reset_idx(envs_to_reset)

    def step(self, actions):
        """Override step to include info from _get_extras()."""
        # Call parent step method
        obs, rewards, terminated, truncated, info = super().step(actions)
        
        # Add our custom info
        extras = self._get_extras()
        info.update(extras)
        # print(info)
        
        return obs, rewards, terminated, truncated, info


def compute_rewards(
    collision_mask: torch.Tensor,
    collision_types: list[str],
    reset_terminated: torch.Tensor,
    reset_time_outs: torch.Tensor,
    agent_positions: torch.Tensor,  # Add agent positions
    goal_positions: torch.Tensor,   # Add goal positions
    device: torch.device,
):
    """Compute rewards based on game events - collision rewards applied last."""
    batch_size = collision_mask.shape[0]
    
    # Reward configuration
    REWARD_CHART = {
        "goal_collision": 100.0,
        "mat_collision": -20.0,
        "timeout": -50.0,
        "step_penalty": 0.0,
        "distance_constant": 1.0,  # Use as 't' in {t/(t+dist)}
    }
    
    # Start with step penalty for all environments
    total_reward = torch.full((batch_size,), REWARD_CHART["step_penalty"], device=device)
    
    # Add distance-based reward for non-terminal states
    distances = torch.norm(agent_positions[:, :2] - goal_positions[:, :2], dim=1)
    t = REWARD_CHART["distance_constant"]
    distance_reward = t / (t + distances)  # This gives values between 0 and 1

    # Apply distance reward to all environments initially
    total_reward += distance_reward
    
    # Apply timeout penalty (overrides step penalty + distance reward)
    timeout_mask = reset_time_outs.bool()
    total_reward = torch.where(
        timeout_mask,
        torch.tensor(REWARD_CHART["timeout"], device=device),
        total_reward
    )
    
    # Apply collision rewards LAST (overrides everything else)
    for i in range(batch_size):
        if collision_mask[i]:
            collision_type = collision_types[i]
            if collision_type == "goal":
                total_reward[i] = REWARD_CHART["goal_collision"]
                print(f"Goal reward +100 for env {i}")
            elif collision_type == "mat":
                total_reward[i] = REWARD_CHART["mat_collision"]
                # print(f"Mat penalty -20 for env {i}")
    
    # Debug logging
    # if total_reward.max() > 0:
        # print(f"Rewards: {total_reward}, Mean: {total_reward.mean().item():.4f}")
        # print(f"Distances: {distances}, Distance rewards: {distance_reward}")
    
    return total_reward