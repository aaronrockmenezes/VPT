# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import random  # added
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, RayCaster, save_images_to_file
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, sample_gaussian, quat_from_euler_xyz
from isaaclab.utils import math as math_utils  # added

from .vpt_env_cfg import VPTEnvCfg
from .check_collisions_new import check_collisions_batched  # added


class VPTEnv(DirectRLEnv):

    cfg: VPTEnvCfg

    def __init__(self,
                 cfg: VPTEnvCfg,
                 render_mode: str | None = None,
                 **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # New
        self.action_scale = self.cfg.action_scale
        self.boundary_limits = self.cfg.boundary_limits
        self.num_objs = self.cfg.num_vpt_objs
        self.center_to_boundary = torch.abs(
            torch.tensor(self.boundary_limits).view(-1)[0])
        # Fix: Use self.num_envs instead of cfg.num_envs to get actual number of environments
        self.env_done = torch.zeros(self.num_envs,
                                    dtype=torch.bool,
                                    device=self.device)
        self.reset_reasons = ["" for _ in range(self.num_envs)
                              ]  # Match actual num_envs

        # Episode tracking for info dictionary - these should accumulate until reset
        self.episode_success = torch.zeros(self.num_envs,
                                           dtype=torch.bool,
                                           device=self.device)
        self.episode_timeout = torch.zeros(self.num_envs,
                                           dtype=torch.bool,
                                           device=self.device)
        self.episode_kill_zone = torch.zeros(self.num_envs,
                                             dtype=torch.bool,
                                             device=self.device)

        # Add cumulative statistics tracking
        self.total_episodes_completed = torch.zeros(self.num_envs,
                                                    dtype=torch.long,
                                                    device=self.device)
        self.total_successes = torch.zeros(self.num_envs,
                                           dtype=torch.long,
                                           device=self.device)
        self.total_timeouts = torch.zeros(self.num_envs,
                                          dtype=torch.long,
                                          device=self.device)
        self.total_kill_zones = torch.zeros(self.num_envs,
                                            dtype=torch.long,
                                            device=self.device)

        # Add action counting for each environment
        self.episode_action_counts = torch.zeros(
            (self.num_envs, 4), dtype=torch.long,
            device=self.device)  # 4 actions: forward, backward, left, right

        # Verbose control - set to 0 by default
        self.verbose = 2  # 0 = no prints, 1 = some prints, 2 = all prints

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The Cube Game environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}")

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        """Setup the scene with the cartpole and camera."""
        spawn_ground_plane(prim_path="/World/ground",
                           cfg=GroundPlaneCfg(size=(1000, 1000)))
        # self._cartpole = Articulation(self.cfg.robot_cfg)
        self._agent = RigidObject(self.cfg.agent)
        self._goal = RigidObject(self.cfg.goal_ball)
        self._boundary_top = RigidObject(self.cfg.top_wall)
        self._boundary_bottom = RigidObject(self.cfg.bottom_wall)
        self._boundary_left = RigidObject(self.cfg.left_wall)
        self._boundary_right = RigidObject(self.cfg.right_wall)
        self._camera_obj = RigidObject(self.cfg.camera_obj)
        self._vpt_objects = RigidObjectCollection(self.cfg.vpt_objects)
        self._mat = RigidObject(self.cfg.mat)
        # self._raycaster = RayCaster(self.cfg.ray_caster)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation and sensors to scene
        self.scene.rigid_objects["agent"] = self._agent
        self.scene.rigid_objects["goal"] = self._goal
        self.scene.rigid_objects["boundary_top"] = self._boundary_top
        self.scene.rigid_objects["boundary_bottom"] = self._boundary_bottom
        self.scene.rigid_objects["boundary_left"] = self._boundary_left
        self.scene.rigid_objects["boundary_right"] = self._boundary_right
        self.scene.rigid_objects["camera_object"] = self._camera_obj
        self.scene.rigid_objects["mat"] = self._mat
        self.scene.rigid_object_collections["vpt_objects"] = self._vpt_objects

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        
        self._occlusion_camera = TiledCamera(self.cfg.occlusion_camera)
        self.scene.sensors["occlusion_camera"] = self._occlusion_camera
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0,
                                           color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def move_agent(self, actions, env_ids: Sequence[int] | None = None):
        """Apply random movement to each agent using action-based logic.
        Actions: 0=forward, 1=backward, 2=turn_left, 3=turn_right
        """
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        # Convert list to tensor if needed
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids,
                                   dtype=torch.long,
                                   device=self.device)

        device = self._agent.device
        num_envs = len(env_ids)
        max_velocity = 3.0

        # Rotation quaternions for yaw movement (15 degrees per step)
        theta = math.pi / 12  # 15 degrees in radians
        half_theta = theta / 2
        left_rot_quat = torch.tensor(
            [math.cos(half_theta), 0.0, 0.0,
             math.sin(half_theta)],
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
                    current_quat[i].unsqueeze(0),
                    left_rot_quat.unsqueeze(0)).squeeze(0)
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 3:  # Turn right
                new_quat[i] = math_utils.quat_mul(
                    current_quat[i].unsqueeze(0),
                    right_rot_quat.unsqueeze(0)).squeeze(0)
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            else:
                # 0: forward, 1: backward
                forward_input = 1.0 if action == 0 else -1.0
                local_movement = torch.tensor(
                    [forward_input, 0.0, 0.0],
                    device=device)  # Changed from [0.0, forward_input, 0.0]
                world_velocity = math_utils.quat_apply(
                    current_quat[i].unsqueeze(0),
                    local_movement.unsqueeze(0)).squeeze(0) * max_velocity
                desired_vel[i, :3] = world_velocity
                desired_vel[i, 3:6] = 0.0

            # Lock Z movement
            desired_vel[i, 2] = 0.0

        # Write pose (updated orientation) and velocity back to sim
        self._agent.write_root_pose_to_sim(
            torch.cat([current_pos, new_quat], dim=1), env_ids)
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
        
        # Lock agent to upright position after movement
        self._lock_agent_upright()

    def _lock_agent_upright(self):
        """Lock agent to upright position, only allowing yaw rotation."""
        # Get current state
        current_pos = self._agent.data.root_pos_w.clone()
        current_quat = self._agent.data.root_quat_w.clone()
        current_vel = self._agent.data.root_vel_w.clone()
        
        # Lock Z position to default height
        current_pos[:, 2] = self._agent.data.default_root_state[:, 2]
        
        # Extract only yaw component from quaternion (remove roll and pitch)
        # Keep w and z components for yaw-only rotation
        w = current_quat[:, 0]
        z = current_quat[:, 3]
        magnitude = torch.sqrt(w**2 + z**2)
        
        upright_quat = current_quat.clone()
        upright_quat[:, 0] = w / magnitude  # w
        upright_quat[:, 1] = 0.0  # x (roll) = 0
        upright_quat[:, 2] = 0.0  # y (pitch) = 0
        upright_quat[:, 3] = z / magnitude  # z (yaw only)
        
        # Lock velocities to prevent toppling
        locked_vel = current_vel.clone()
        locked_vel[:, 2] = 0.0  # No vertical velocity
        locked_vel[:, 3] = 0.0  # No angular velocity around X (roll)
        locked_vel[:, 4] = 0.0  # No angular velocity around Y (pitch)
        
        # Write corrected state back to simulation
        self._agent.write_root_pose_to_sim(
            torch.cat([current_pos, upright_quat], dim=1))
        self._agent.write_root_velocity_to_sim(locked_vel)

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
            save_images_to_file(observations["policy"],
                                f"cartpole_{data_type}.png")

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards for the current step."""
        collision_mask, collision_types = self._check_collisions()

        total_reward = compute_rewards(
            collision_mask=collision_mask,
            collision_types=collision_types,
            reset_terminated=self.reset_terminated,
            reset_time_outs=self.reset_time_outs,
            agent=self._agent,
            goal=self._goal,
            device=self.device,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination and timeout masks."""
        # DON'T reset episode tracking here - let it accumulate

        # Check collisions and handle them
        collision_mask, collision_types = self._check_collisions()
        self._handle_collisions(collision_mask, collision_types)

        # Get current termination and timeout states
        terminated = self.env_done.clone()
        time_outs = (self.episode_length_buf >= self.max_episode_length)

        # Track episode outcomes for info dictionary (accumulate, don't overwrite)
        for i in range(self.num_envs):
            if collision_mask[i] and collision_types[i] == "goal":
                self.episode_success[
                    i] = True  # Once successful, stays successful
            elif collision_mask[i] and collision_types[i] == "mat":
                self.episode_kill_zone[
                    i] = True  # Once hit kill zone, stays hit
            elif time_outs[i] and not terminated[i]:
                self.episode_timeout[
                    i] = True  # Once timed out, stays timed out
                if i < len(self.reset_reasons):
                    self.reset_reasons[i] = "timeout"

        return terminated, time_outs

    def _get_extras(self) -> dict:
        """Return info dictionary with episode statistics."""
        # Current distances from agent to goal
        agent_positions = self._agent.data.root_pos_w
        goal_positions = self._goal.data.root_pos_w
        distances_to_goal = torch.norm(agent_positions[:, :2] -
                                       goal_positions[:, :2],
                                       dim=1)

        # Calculate cumulative success, timeout, and kill zone rates
        success_rates = torch.where(
            self.total_episodes_completed > 0,
            self.total_successes.float() /
            self.total_episodes_completed.float(),
            torch.zeros_like(self.total_successes, dtype=torch.float))

        timeout_rates = torch.where(
            self.total_episodes_completed > 0,
            self.total_timeouts.float() /
            self.total_episodes_completed.float(),
            torch.zeros_like(self.total_timeouts, dtype=torch.float))

        kill_zone_rates = torch.where(
            self.total_episodes_completed > 0,
            self.total_kill_zones.float() /
            self.total_episodes_completed.float(),
            torch.zeros_like(self.total_kill_zones, dtype=torch.float))

        info = {
            "distance_to_goal": distances_to_goal.cpu().numpy(),
            "success_rate": success_rates.cpu().numpy(),
            "timeout_rate": timeout_rates.cpu().numpy(),
            "kill_zone_rate": kill_zone_rates.cpu().numpy(),
        }

        return info

    def _reset_idx(self,
                   env_ids: Sequence[int] | None,
                   randomize_objects: bool = True) -> None:
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        # Convert list to tensor if needed
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids,
                                   dtype=torch.long,
                                   device=self.device)

        # Update cumulative statistics before resetting episode tracking
        for env_id in env_ids:
            if self.episode_success[env_id]:
                self.total_successes[env_id] += 1
            if self.episode_timeout[env_id]:
                self.total_timeouts[env_id] += 1
            if self.episode_kill_zone[env_id]:
                self.total_kill_zones[env_id] += 1

            # Only increment if episode actually completed (success, timeout, or kill zone)
            if self.episode_success[env_id] or self.episode_timeout[
                    env_id] or self.episode_kill_zone[env_id]:
                self.total_episodes_completed[env_id] += 1

        # Reset episode tracking for environments being reset
        self.episode_success[env_ids] = False
        self.episode_timeout[env_ids] = False
        self.episode_kill_zone[env_ids] = False

        # Reset action counts for the environments being reset
        self.episode_action_counts[env_ids] = 0

        super()._reset_idx(env_ids)

        device = self._agent.device
        num_envs = len(env_ids)
        safe_x_range = self.center_to_boundary - 1.5
        safe_y_range = self.center_to_boundary - 1.5

        # Default states
        goal_default_state = self._goal.data.default_root_state[
            env_ids].clone()
        agent_default_state = self._agent.data.default_root_state[
            env_ids].clone()
        camera_obj_default_state = self._camera_obj.data.default_root_state[
            env_ids].clone()
        vpt_obj_default_state = self._vpt_objects.data.default_object_state[
            env_ids].clone()
        # print(goal_default_state.shape, vpt_obj_default_state.shape)

        for i in range(num_envs):
            valid_positions = False
            attempts = 0
            max_attempts = 100

            if i % 4 == 0 or i % 4 == 1:
                target_occluded = False
                target_outside_fov = False
            # Every 3rd object is occluded 
            elif i % 4 == 2:
                target_occluded = True
                target_outside_fov = False
            # Every 4th object is outside camera FOV 
            elif i % 4 == 3:
                target_occluded = False
                target_outside_fov = True
                # target_outside_fov = False

            target_occluded = False
            target_outside_fov = False

            while not valid_positions and attempts < max_attempts:
                # Add 45-degree right roll (negative roll) as a tensor
                roll = torch.tensor(-math.pi / 6, device=device)
                # roll = torch.tensor(0, device=device)
                
                if randomize_objects:
                    # 1. Sample goal position randomly within safe range
                    goal_offset_x = sample_uniform(-safe_x_range, safe_x_range,
                                                   (1, ), device)
                    goal_offset_y = sample_uniform(-safe_y_range, safe_y_range,
                                                   (1, ), device)
                    goal_new_pos = goal_default_state[i, :3].clone()
                    goal_new_pos[0] = self.scene.env_origins[env_ids[i],
                                                             0] + goal_offset_x
                    goal_new_pos[1] = self.scene.env_origins[env_ids[i],
                                                             1] + goal_offset_y
                    goal_new_pos[2] += self.scene.env_origins[env_ids[i], 2]

                    camera_offset_x = sample_uniform(-safe_x_range,
                                                     safe_x_range, (1, ),
                                                     device)
                    camera_offset_y = sample_uniform(-safe_y_range,
                                                     safe_y_range, (1, ),
                                                     device)
                    camera_new_pos = camera_obj_default_state[i, :3].clone()
                    camera_new_pos[0] = self.scene.env_origins[
                        env_ids[i], 0] + camera_offset_x
                    camera_new_pos[1] = self.scene.env_origins[
                        env_ids[i], 1] + camera_offset_y

                    # ALWAYS point camera at goal by default
                    direction_to_goal = goal_new_pos[:2] - camera_new_pos[:2]
                    yaw = torch.atan2(direction_to_goal[1], direction_to_goal[0])
                    
                    # Add 90-degree offset to account for camera's local forward axis
                    yaw = yaw - math.radians(90)

                    # Debug: Print for specific environment if verbose
                    if self.verbose >= 2:  # Change i == 0 to the problematic env index
                        print(f"Env {env_ids[i]}: Camera at {camera_new_pos[:2]}, Goal at {goal_new_pos[:2]}")
                        print(f"Direction: {direction_to_goal}, Yaw: {yaw}")
                    
                    horizontal_distance = torch.norm(direction_to_goal)
                    vertical_distance = goal_new_pos[2] - camera_new_pos[2]
                    pitch = torch.atan2(vertical_distance, horizontal_distance)
                    
                    quaternion = quat_from_euler_xyz(
                        roll,
                        # pitch,
                        torch.tensor(0, device=device),
                        yaw
                        # torch.tensor(0, device=device),
                    )
                    camera_obj_default_state[i, 3:7] = quaternion

                    # Sample object positions randomly within safe range
                    vpt_obj_offset_x = sample_uniform(-safe_x_range,
                                                      safe_x_range,
                                                      (self.num_objs, ),
                                                      device)
                    vpt_obj_offset_y = sample_uniform(-safe_y_range,
                                                      safe_y_range,
                                                      (self.num_objs, ),
                                                      device)
                    vpt_obj_new_pos = vpt_obj_default_state[i, :, :3].clone()
                    vpt_obj_new_pos[:, 0] = self.scene.env_origins[
                        env_ids[i], 0] + vpt_obj_offset_x
                    vpt_obj_new_pos[:, 1] = self.scene.env_origins[
                        env_ids[i], 1] + vpt_obj_offset_y
                    vpt_obj_new_pos[:, 2] += self.scene.env_origins[env_ids[i], 2]

                    # Handle special cases based on flags
                    if target_occluded:
                        # Try to ensure occlusion by checking VPT objects block line of sight
                        occlusion_attempts = 0
                        max_occlusion_attempts = 100
                        goal_is_occluded = False
                        
                        while not goal_is_occluded and occlusion_attempts < max_occlusion_attempts:
                            # Re-randomize VPT object positions
                            vpt_obj_offset_x = sample_uniform(-safe_x_range,
                                                              safe_x_range,
                                                              (self.num_objs, ),
                                                              device)
                            vpt_obj_offset_y = sample_uniform(-safe_y_range,
                                                              safe_y_range,
                                                              (self.num_objs, ),
                                                              device)
                            vpt_obj_new_pos[:, 0] = self.scene.env_origins[
                                env_ids[i], 0] + vpt_obj_offset_x
                            vpt_obj_new_pos[:, 1] = self.scene.env_origins[
                                env_ids[i], 1] + vpt_obj_offset_y
                            
                            # Check if camera position is still valid (min 1 unit from VPT objects)
                            camera_distances_from_vpt = torch.norm(
                                camera_new_pos[:2].unsqueeze(0) - vpt_obj_new_pos[:, :2],
                                dim=1)
                            camera_min_distance_ok = torch.all(
                                camera_distances_from_vpt >= 1.0)
                            
                            if not camera_min_distance_ok:
                                occlusion_attempts += 1
                                continue
                            
                            # Check if any VPT object occludes the goal using raycast
                            goal_is_occluded = self._check_occlusion_raycast(
                                camera_new_pos, goal_new_pos, env_ids[i]
                            )
                            
                            occlusion_attempts += 1
                        
                        if not goal_is_occluded and self.verbose >= 1:
                            print(f"Warning: Could not create occlusion for env {env_ids[i]} after {max_occlusion_attempts} attempts")
                    
                    elif target_outside_fov:
                        # Point camera away from goal (180 degrees opposite)
                        yaw_away = yaw + math.pi
                        quaternion_away = quat_from_euler_xyz(
                            roll,
                            pitch,
                            # torch.tensor(0, device=device),
                            # torch.tensor(0, device=device),
                            yaw_away
                        )
                        camera_obj_default_state[i, 3:7] = quaternion_away

                else:
                    # Keep goal at its original default position + env_origins
                    goal_new_pos = goal_default_state[i, :3].clone()
                    goal_new_pos[:2] += self.scene.env_origins[env_ids[i], :2]
                    goal_new_pos[2] += self.scene.env_origins[env_ids[i], 2]

                    camera_new_pos = camera_obj_default_state[i, :3].clone()
                    camera_new_pos[:2] += self.scene.env_origins[
                        env_ids[i], :2]
                    camera_new_pos[2] += self.scene.env_origins[env_ids[i], 2]
                    
                    # Point camera at goal by default even when not randomizing
                    direction_to_goal = goal_new_pos[:2] - camera_new_pos[:2]
                    yaw = torch.atan2(direction_to_goal[1], direction_to_goal[0])
                    horizontal_distance = torch.norm(direction_to_goal)
                    vertical_distance = goal_new_pos[2] - camera_new_pos[2]
                    pitch = torch.atan2(vertical_distance, horizontal_distance)

                    quaternion = quat_from_euler_xyz(
                        roll,
                        pitch,
                        # torch.tensor(0, device=device),
                        yaw
                        # torch.tensor(0, device=device),
                    )
                    camera_obj_default_state[i, 3:7] = quaternion

                # 2. Sample agent position (ensuring minimum distance of 2 units from goal)
                agent_offset_x = sample_uniform(-safe_x_range, safe_x_range,
                                                (1, ), device)
                agent_offset_y = sample_uniform(-safe_y_range, safe_y_range,
                                                (1, ), device)
                agent_new_pos = agent_default_state[i, :3].clone()
                agent_new_pos[0] = self.scene.env_origins[env_ids[i],
                                                          0] + agent_offset_x
                agent_new_pos[1] = self.scene.env_origins[env_ids[i],
                                                          1] + agent_offset_y

                # Constraints: minimum distance of 2 units from goal
                agent_distance_from_goal = torch.norm(agent_new_pos[:2] -
                                                      goal_new_pos[:2])
                agent_min_distance_ok = agent_distance_from_goal >= 2.0

                # Constraint: minimum distance of camera is 1 unit from all VPT objects
                camera_distances_from_vpt = torch.norm(
                    camera_new_pos[:2].unsqueeze(0) - vpt_obj_new_pos[:, :2],
                    dim=1)
                camera_min_distance_ok = torch.all(
                    camera_distances_from_vpt >= 1.0)
                

                if agent_min_distance_ok and camera_min_distance_ok:
                    goal_default_state[i, :3] = goal_new_pos
                    agent_default_state[i, :3] = agent_new_pos
                    camera_obj_default_state[i, :3] = camera_new_pos
                    vpt_obj_default_state[i, :, :3] = vpt_obj_new_pos
                    valid_positions = True

                attempts += 1

            if attempts >= max_attempts and self.verbose >= 1:
                print(
                    f"Warning: Could not find valid positions for env {env_ids[i]} after {max_attempts} attempts"
                )

        # Random orientation for agent (full 360 degrees)
        random_yaw_agent = sample_uniform(0, 2 * math.pi, (num_envs, ), device)
        agent_default_state[:, 3] = torch.cos(random_yaw_agent / 2)  # w
        agent_default_state[:, 4] = 0.0  # x
        agent_default_state[:, 5] = 0.0  # y
        agent_default_state[:, 6] = torch.sin(random_yaw_agent / 2)  # z

        # Write all states to simulation
        self._goal.write_root_pose_to_sim(goal_default_state[:, :7], env_ids)
        self._goal.write_root_velocity_to_sim(
            torch.zeros_like(goal_default_state[:, 7:]), env_ids)

        self._camera_obj.write_root_pose_to_sim(
            camera_obj_default_state[:, :7], env_ids)
        self._camera_obj.write_root_velocity_to_sim(
            torch.zeros_like(camera_obj_default_state[:, 7:]), env_ids)

        # Copy position and quat of camera obj
        # Move camera back and to the left in local space
        backward_offset = -0.55  # Adjust this value to move camera back
        left_offset = -0.040      # Adjust this value to move camera left
        up_offset = 0.12
        
        # Get camera's forward and left directions from its orientation
        camera_forward = math_utils.quat_apply(
            camera_obj_default_state[:, 3:7],
            torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)
        )
        camera_left = math_utils.quat_apply(
            camera_obj_default_state[:, 3:7],
            torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)
        )
        camera_up = math_utils.quat_apply(
            camera_obj_default_state[:, 3:7],
            torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(num_envs, -1)
        )
        
        # Calculate new camera position (move backward and left)
        occlusion_camera_pos = (camera_obj_default_state[:, :3].clone() + 
                               backward_offset * camera_forward + 
                               left_offset * camera_left +
                               up_offset * camera_up)
        
        # Create 90-degree left rotation quaternion (rotation around Z-axis)
        theta_left = math.pi / 2  # 90 degrees
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)],
            device=device
        )
        
        # Apply rotation to camera orientations for left-facing view
        rotated_orientations = math_utils.quat_mul(
            camera_obj_default_state[:, 3:7],
            left_90_quat.unsqueeze(0).expand(num_envs, -1)
        )
        
        self._occlusion_camera.set_world_poses(
            positions=occlusion_camera_pos,
            orientations=rotated_orientations,
            env_ids=env_ids.tolist(),
            convention="world"
        )

        self._agent.write_root_pose_to_sim(agent_default_state[:, :7], env_ids)
        self._agent.write_root_velocity_to_sim(
            torch.zeros_like(agent_default_state[:, 7:]), env_ids)

        # print(f"Goal write to pos: {goal_default_state[:, :7].shape}")
        # print(f"VPT New pos list: {vpt_obj_default_state.shape} | {vpt_obj_default_state} | [:, :, :7] {vpt_obj_default_state[:, :, :7].shape}")
        self._vpt_objects.write_object_pose_to_sim(
            vpt_obj_default_state[:, :, :7], env_ids)
        self._vpt_objects.write_object_velocity_to_sim(
            torch.zeros_like(vpt_obj_default_state[:, :, 7:]), env_ids)

        # Safe printing with bounds checking
        if self.verbose >= 1:
            valid_env_ids = [
                env_id for env_id in env_ids.tolist()
                if env_id < len(self.reset_reasons)
            ]
            reset_reasons_for_envs = [
                self.reset_reasons[env_id]
                if env_id < len(self.reset_reasons) else "unknown"
                for env_id in env_ids.tolist()
            ]
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
        # Remove goal_mats and wall
        collision_mask, collision_types = check_collisions_batched(
            agents=agents,
            goal_balls=goal_balls,
            boundary_limits=self.boundary_limits)
        return collision_mask, collision_types

    def _handle_collisions(self, collision_mask, collision_types):
        """Handle collision events and reset environments."""
        if not collision_mask.any():
            return

        # Get environments that need resetting
        envs_to_reset = collision_mask.nonzero(
            as_tuple=False).squeeze(-1).tolist()

        # Mark environments as done and record reasons
        for env_id in envs_to_reset:
            if env_id < len(self.env_done):
                self.env_done[env_id] = True
                collision_type = collision_types[env_id] if env_id < len(
                    collision_types) else "unknown"
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

    def _check_occlusion_raycast(self, camera_pos, goal_pos, env_id):
        """
        Check if goal (red sphere with radius 0.15) is visible using depth camera.
        Checks entire FOV since camera may be perturbed and goal might not be centered.
        
        Args:
            camera_pos: Camera position (3D tensor)
            goal_pos: Goal position (center of sphere) (3D tensor)
            env_id: Environment ID (index in the batch)
            
        Returns:
            bool: True if goal is occluded (NOT visible), False if visible
        """
        # Calculate expected distance from camera to goal center
        expected_distance = torch.norm(goal_pos - camera_pos)
        
        if expected_distance < 1e-6:
            return False
        
        # Get depth data from occlusion camera for this environment
        depth_data = self._occlusion_camera.data.output["distance_to_camera"][env_id]
        
        # Goal sphere has radius 0.15, add small buffer for safety
        goal_radius = 0.15
        buffer = 0.05  # Small buffer for numerical stability
        effective_radius = goal_radius + buffer
        
        # Expected distance range where we should see the goal
        # Goal surface is closer than center by radius
        min_expected_depth = expected_distance - effective_radius
        max_expected_depth = expected_distance + effective_radius
        
        # Check ENTIRE image since goal could be anywhere in FOV
        # Filter out invalid depths (inf values)
        valid_depths = depth_data[depth_data != float("inf")]
        
        if len(valid_depths) == 0:
            if self.verbose >= 2:
                print(f"Env {env_id}: No valid depth readings in entire FOV")
            return True  # No valid readings = assume occluded
        
        # Check if any depth readings fall within the expected goal range
        depths_in_goal_range = ((valid_depths >= min_expected_depth) & 
                               (valid_depths <= max_expected_depth))
        num_goal_pixels = depths_in_goal_range.sum().item()
        
        # If we see enough pixels at the goal distance, it's visible
        # Need more pixels than center-only check since we're looking at full FOV
        min_pixels_for_visibility = 5  # Adjust based on testing
        goal_is_visible = num_goal_pixels >= min_pixels_for_visibility
        
        # Also check if there's something significantly closer blocking the view
        # across the entire image
        min_depth = valid_depths.min()
        something_blocking = min_depth < (min_expected_depth - 0.1)
        
        if self.verbose >= 2:
            print(f"Env {env_id}:")
            print(f"  Expected distance: {expected_distance:.2f}")
            print(f"  Goal range: [{min_expected_depth:.2f}, {max_expected_depth:.2f}]")
            print(f"  Min depth in FOV: {min_depth:.2f}")
            print(f"  Pixels in goal range: {num_goal_pixels}/{len(valid_depths)}")
            print(f"  Something blocking: {something_blocking}")
            print(f"  Goal visible: {goal_is_visible}")
        
        # Goal is occluded if either:
        # 1. Not enough pixels at goal distance, OR
        # 2. Something is blocking (much closer than goal)
        goal_is_occluded = (not goal_is_visible) or something_blocking
        
        return goal_is_occluded


def compute_rewards(
    collision_mask: torch.Tensor,
    collision_types: list[str],
    reset_terminated: torch.Tensor,
    reset_time_outs: torch.Tensor,
    agent: object,  # Agent rigid object
    goal: object,  # Goal rigid object
    device: torch.device,
):
    """Compute rewards based on game events - collision rewards applied last."""
    # Extract positions from objects
    agent_positions = agent.data.root_pos_w.clone()
    goal_positions = goal.data.root_pos_w.clone()
    goal_default_positions = goal.data.default_root_state[:, :3].clone()

    batch_size = collision_mask.shape[0]

    # Reward configuration
    REWARD_CHART = {
        "goal_collision": 100.0,
        "timeout": -50.0,
        "step_penalty": 0.0,
        "distance_constant": 1.0,  # Use as 't' in {t/(t+dist)}
        "goal_displacement":
        0.0,  # Max reward for moving goal from default position
        "wall_collision": 0.5
    }

    # Start with step penalty for all environments
    total_reward = torch.full((batch_size, ),
                              REWARD_CHART["step_penalty"],
                              device=device)

    # Add distance-based reward for non-terminal states
    distances = torch.norm(agent_positions[:, :2] - goal_positions[:, :2],
                           dim=1)
    t = REWARD_CHART["distance_constant"]
    distance_reward = t / (t + distances)  # This gives values between 0 and 1

    # Apply distance reward to all environments initially
    total_reward += distance_reward

    # Add goal displacement reward based on movement from default position
    goal_displacement_distances = torch.norm(goal_positions[:, :2] -
                                             goal_default_positions[:, :2],
                                             dim=1)

    # Scale displacement reward with clamp between 0 and 1
    max_displacement = 1.0  # Maximum displacement to consider for full reward
    displacement_scale = torch.clamp(
        goal_displacement_distances / max_displacement, 0.0, 1.0)
    displacement_reward = displacement_scale * REWARD_CHART["goal_displacement"]

    total_reward += displacement_reward

    # Apply timeout penalty (overrides step penalty + distance reward + displacement reward)
    timeout_mask = reset_time_outs.bool()
    total_reward = torch.where(
        timeout_mask, torch.tensor(REWARD_CHART["timeout"], device=device),
        total_reward)

    # Apply collision rewards LAST (overrides everything else)
    for i in range(batch_size):
        if collision_mask[i]:
            collision_type = collision_types[i]
            if collision_type == "goal":
                total_reward[i] = REWARD_CHART["goal_collision"]
            elif collision_type == "wall":
                total_reward[i] += REWARD_CHART["wall_collision"]

    return total_reward
