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
from typing import List, Dict, Tuple, Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, RayCaster, save_images_to_file, Camera
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
        self.obstacle_metadata = self.cfg.objects_metadata
        self.agent_height = self.cfg.agent_height
        self.agent_camera_pitch = self.cfg.agent_camera_pitch
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

        # Initialize storage for valid viewpoint poses
        # Will be populated in _reset_idx: shape (num_envs, num_poses, 3)
        self.valid_viewpoint_poses = None
        
        # Initialize counter for iterating through valid viewpoint poses
        # Tracks current index for each environment
        self.viewpoint_pose_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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
        self.scene.sensors["semantic_tiled_camera"] = self._tiled_camera

        self._rgb_tiled_camera = TiledCamera(self.cfg.rgb_tiled_camera)
        self.scene.sensors["rgb_tiled_camera"] = self._rgb_tiled_camera

        self._distance_tiled_camera = TiledCamera(self.cfg.distance_tiled_camera)
        self.scene.sensors["distance_tiled_camera"] = self._distance_tiled_camera

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

        # LOCK AGENT UPRIGHT: Extract only yaw component, remove roll and pitch
        w = current_quat[:, 0]
        z = current_quat[:, 3]
        magnitude = torch.sqrt(w**2 + z**2)

        upright_quat = current_quat.clone()
        upright_quat[:, 0] = w / magnitude  # w
        upright_quat[:, 1] = 0.0  # x (roll) = 0
        upright_quat[:, 2] = 0.0  # y (pitch) = 0
        upright_quat[:, 3] = z / magnitude  # z (yaw only)

        # Lock Z position to default height
        current_pos[:, 2] = self._agent.data.default_root_state[env_ids, 2]

        # Initialize new states with upright orientation
        new_quat = upright_quat.clone()
        desired_vel = torch.zeros((num_envs, 6), device=device)

        # Process one random action per environment
        for i, _ in enumerate(env_ids):
            action = actions[i]

            if action == 2:  # Turn left
                new_quat[i] = math_utils.quat_mul(
                    upright_quat[i].unsqueeze(0),
                    left_rot_quat.unsqueeze(0)).squeeze(0)
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 3:  # Turn right
                new_quat[i] = math_utils.quat_mul(
                    upright_quat[i].unsqueeze(0),
                    right_rot_quat.unsqueeze(0)).squeeze(0)
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 4:  # Do nothing / Stay still
                # Keep current orientation and zero velocity
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 5:  # Move to next valid viewpoint (iterating through poses)
                # Teleport agent to the next valid viewpoint in sequence
                env_id_item = env_ids[i].item() if torch.is_tensor(env_ids[i]) else env_ids[i]
                
                if (self.valid_viewpoint_poses is not None and 
                    env_id_item < len(self.valid_viewpoint_poses) and
                    self.valid_viewpoint_poses[env_id_item] is not None and 
                    len(self.valid_viewpoint_poses[env_id_item]) > 0):
                    
                    # Get current counter value for this environment
                    current_idx = self.viewpoint_pose_counter[env_id_item].item()
                    num_valid_poses = len(self.valid_viewpoint_poses[env_id_item])
                    
                    # Use counter to select pose (wraps around if counter exceeds available poses)
                    pose_idx = current_idx % num_valid_poses
                    target_pos = self.valid_viewpoint_poses[env_id_item][pose_idx].to(device)
                    
                    # Increment counter for next time
                    self.viewpoint_pose_counter[env_id_item] += 1
                    
                    if self.verbose >= 1 and env_id_item == 0:
                        print(f"[Action 5] Moving to valid pose {pose_idx}/{num_valid_poses-1} for env {env_id_item}")
                    
                    # Calculate midpoint between camera and goal (in world coordinates)
                    camera_pos_3d = self._camera_obj.data.root_pos_w[env_ids[i]]
                    goal_pos_3d = self._goal.data.root_pos_w[env_ids[i]]
                    midpoint = (camera_pos_3d + goal_pos_3d) / 2.0
                    
                    # Calculate yaw to look at midpoint
                    direction = midpoint[:2] - target_pos[:2]
                    if torch.norm(direction) > 1e-6:
                        yaw = torch.atan2(direction[1], direction[0])
                    else:
                        yaw = torch.tensor(0.0, device=device)
                    
                    # Create quaternion for pure yaw rotation (no roll/pitch)
                    # Format: [w, x, y, z] for rotation around Z-axis
                    new_quat[i] = torch.tensor(
                        [math.cos(yaw.item() / 2), 0.0, 0.0, math.sin(yaw.item() / 2)],
                        device=device,
                        dtype=torch.float32
                    )
                    
                    # Update position
                    current_pos[i, :3] = target_pos
                else:
                    # Fallback: stay in current position if no valid poses available
                    if self.verbose >= 1:
                        print(f"Warning: No valid viewpoint poses available for env {env_id_item}, staying in place")
                
                # Set velocity to zero
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 6:  # Move to random valid viewpoint (random selection from valid poses)
                # Teleport agent to a randomly selected valid viewpoint
                env_id_item = env_ids[i].item() if torch.is_tensor(env_ids[i]) else env_ids[i]
                
                if (self.valid_viewpoint_poses is not None and 
                    env_id_item < len(self.valid_viewpoint_poses) and
                    self.valid_viewpoint_poses[env_id_item] is not None and 
                    len(self.valid_viewpoint_poses[env_id_item]) > 0):
                    
                    # Randomly select a pose index
                    num_valid_poses = len(self.valid_viewpoint_poses[env_id_item])
                    random_idx = torch.randint(0, num_valid_poses, (1,), device=device).item()
                    target_pos = self.valid_viewpoint_poses[env_id_item][random_idx].to(device)
                    
                    if self.verbose >= 1 and env_id_item == 0:
                        print(f"[Action 6] Moving to random valid pose {random_idx}/{num_valid_poses-1} for env {env_id_item}")
                    
                    # Calculate midpoint between camera and goal (in world coordinates, 2D only)
                    camera_pos_3d = self._camera_obj.data.root_pos_w[env_ids[i]]
                    goal_pos_3d = self._goal.data.root_pos_w[env_ids[i]]
                    midpoint = (camera_pos_3d[:2] + goal_pos_3d[:2]) / 2.0  # XY only
                    
                    # Calculate yaw to look at midpoint (XY plane only)
                    direction = midpoint - target_pos[:2]
                    if torch.norm(direction) > 1e-6:
                        yaw = torch.atan2(direction[1], direction[0])
                    else:
                        yaw = torch.tensor(0.0, device=device)
                    
                    # Create quaternion for pure yaw rotation (no roll/pitch)
                    # Format: [w, x, y, z] for rotation around Z-axis
                    new_quat[i] = torch.tensor(
                        [math.cos(yaw.item() / 2), 0.0, 0.0, math.sin(yaw.item() / 2)],
                        device=device,
                        dtype=torch.float32
                    )
                    
                    # Update position
                    current_pos[i, :3] = target_pos
                else:
                    # Fallback: stay in current position if no valid poses available
                    if self.verbose >= 1:
                        print(f"Warning: No valid viewpoint poses available for env {env_id_item}, staying in place")
                
                # Set velocity to zero
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            else:
                # 0: forward, 1: backward
                forward_input = 1.0 if action == 0 else -1.0
                local_movement = torch.tensor([forward_input, 0.0, 0.0],
                                              device=device)
                world_velocity = math_utils.quat_apply(
                    upright_quat[i].unsqueeze(0),
                    local_movement.unsqueeze(0)).squeeze(0) * max_velocity
                desired_vel[i, :3] = world_velocity
                desired_vel[i, 3:6] = 0.0

            # Lock Z velocity and angular velocities around X and Y
            desired_vel[i, 2] = 0.0  # No vertical velocity
            desired_vel[i, 3] = 0.0  # No angular velocity around X (roll)
            desired_vel[i, 4] = 0.0  # No angular velocity around Y (pitch)

        # Write pose (updated orientation) and velocity back to sim
        # Use current_pos which has been updated with target positions for action==5
        self._agent.write_root_com_pose_to_sim(
            torch.cat([current_pos, new_quat], dim=1), env_ids)
        self._agent.write_root_com_velocity_to_sim(desired_vel, env_ids)
        self._agent.reset()

        # Update tiled camera to follow agent
        # Camera position: same XY as agent, but at agent_height
        camera_pos = current_pos.clone()
        camera_pos[:, 2] = self.agent_height

        # Camera orientation: same yaw as agent, but with -30 degree pitch
        pitch_angle = -math.pi / 6  # -30 degrees
        half_pitch = pitch_angle / 2

        # Create pitch quaternion (rotation around Y-axis in local frame)
        # For downward pitch, we rotate around Y
        pitch_quat = torch.tensor(
            [math.cos(half_pitch), 0.0,
             math.sin(half_pitch), 0.0],
            device=device)

        # Combine agent's yaw with camera's pitch
        camera_quat = torch.zeros((num_envs, 4), device=device)
        for i in range(num_envs):
            camera_quat[i] = math_utils.quat_mul(
                new_quat[i].unsqueeze(0), pitch_quat.unsqueeze(0)).squeeze(0)

        # # Set camera pose
        # self._tiled_camera.set_world_poses(
        #     positions=camera_pos,
        #     orientations=camera_quat,
        #     env_ids=env_ids.tolist(),
        #     convention="world"
        # )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # Track action counts before applying actions
        for i in range(self.num_envs):
            action = int(self.actions[i].item())
            if 0 <= action <= 3:  # Valid action range
                self.episode_action_counts[i, action] += 1

        # Print action info for env 0 BEFORE applying action (skip HOLD action to avoid spam)
        if self.verbose >= 1:
            env_id = 0
            action = int(self.actions[env_id].item())
            
            # Skip logging for HOLD action (action 4) to avoid spam
            if action != 4:
                # Action names for readability
                action_names = {
                    0: "FORWARD",
                    1: "BACKWARD", 
                    2: "TURN LEFT",
                    3: "TURN RIGHT",
                    4: "HOLD",
                    5: "TELEPORT TO VALID VIEWPOINT",
                    6: "TELEPORT RANDOM"
                }
                action_name = action_names.get(action, f"UNKNOWN({action})")
                
                # Get agent state
                agent_pos = self._agent.data.root_pos_w[env_id]
                agent_quat = self._agent.data.root_quat_w[env_id]
                
                # Get camera and goal positions
                camera_pos = self._camera_obj.data.root_pos_w[env_id]
                goal_pos = self._goal.data.root_pos_w[env_id]
                
                # Calculate distances
                dist_to_camera = torch.norm(agent_pos[:2] - camera_pos[:2]).item()
                dist_to_goal = torch.norm(agent_pos[:2] - goal_pos[:2]).item()
                
                # Calculate min distance to VPT objects
                vpt_positions = self._vpt_objects.data.object_pos_w[env_id, :, :2]
                distances_to_vpt = torch.norm(agent_pos[:2].unsqueeze(0) - vpt_positions, dim=1)
                min_dist_to_vpt = distances_to_vpt.min().item()
                
                # Calculate midpoint between camera and goal
                midpoint = (camera_pos + goal_pos) / 2.0
                
                # Calculate additional distances
                dist_agent_to_midpoint = torch.norm(agent_pos[:2] - midpoint[:2]).item()
                dist_camera_to_goal = torch.norm(camera_pos[:2] - goal_pos[:2]).item()
                
                print(f"\n{'='*60}")
                print(f"[ENV 0] ACTION: {action_name}")
                print(f"{'='*60}")
                print(f"Agent Position: [{agent_pos[0]:.3f}, {agent_pos[1]:.3f}, {agent_pos[2]:.3f}]")
                print(f"Agent Quaternion: [{agent_quat[0]:.3f}, {agent_quat[1]:.3f}, {agent_quat[2]:.3f}, {agent_quat[3]:.3f}]")
                print(f"\nDistances:")
                print(f"  Distance to Camera:      {dist_to_camera:.3f}")
                print(f"  Distance to Goal:        {dist_to_goal:.3f}")
                print(f"  Distance to Midpoint:    {dist_agent_to_midpoint:.3f}")
                print(f"  Distance Cam to Goal:    {dist_camera_to_goal:.3f}")
                print(f"  Min Dist to VPT Obj:     {min_dist_to_vpt:.3f}")
                print(f"\nMidpoint (Camera + Goal): [{midpoint[0]:.3f}, {midpoint[1]:.3f}, {midpoint[2]:.3f}]")
                
                # Validate agent's current position
                print(f"\n[AGENT POSITION VALIDATION]")
                agent_pos_2d = agent_pos[:2]
                
                # 1. Check bounds
                env_origin = self.scene.env_origins[env_id, :2]
                boundary_limit = self.center_to_boundary.item() if torch.is_tensor(self.center_to_boundary) else self.center_to_boundary
                min_bound = env_origin - boundary_limit
                max_bound = env_origin + boundary_limit
                in_bounds = torch.all(agent_pos_2d >= min_bound) and torch.all(agent_pos_2d <= max_bound)
                print(f"  Bounds: {'✓ VALID' if in_bounds else '✗ INVALID'} (limit: ±{boundary_limit:.2f})")
                
                # 2. Check distance from VPT objects
                min_obstacle_dist = 0.4
                vpt_clear = min_dist_to_vpt >= min_obstacle_dist
                print(f"  VPT Objects: {'✓ CLEAR' if vpt_clear else '✗ TOO CLOSE'} (min: {min_dist_to_vpt:.3f}, required: {min_obstacle_dist:.2f})")
                
                # 3. Check distance from camera
                camera_clear = dist_to_camera >= min_obstacle_dist
                print(f"  Camera: {'✓ CLEAR' if camera_clear else '✗ TOO CLOSE'} (dist: {dist_to_camera:.3f}, required: {min_obstacle_dist:.2f})")
                
                # 4. Check distance from goal
                goal_clear = dist_to_goal >= min_obstacle_dist
                print(f"  Goal: {'✓ CLEAR' if goal_clear else '✗ TOO CLOSE'} (dist: {dist_to_goal:.3f}, required: {min_obstacle_dist:.2f})")
                
                # Overall validation
                is_valid = in_bounds and vpt_clear and camera_clear and goal_clear
                if is_valid:
                    print(f"\n  ✅ Agent position is VALID (excluding visibility check)")
                else:
                    reasons = []
                    if not in_bounds: reasons.append("OUT OF BOUNDS")
                    if not vpt_clear: reasons.append("TOO CLOSE TO VPT")
                    if not camera_clear: reasons.append("TOO CLOSE TO CAMERA")
                    if not goal_clear: reasons.append("TOO CLOSE TO GOAL")
                    print(f"\n  ❌ Agent position is INVALID: {', '.join(reasons)}")
                
                print(f"{'='*60}\n")

        self.move_agent(self.actions)

        # Lock agent to upright position after movement
        # self._lock_agent_upright()

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
        """
        Get observations for all environments.
        
        For each environment:
        1. Ensure folder structure exists: data/env_{idx}/RGB and data/env_{idx}/Depth
        2. Check if camera and target are within FOV
        3. If within FOV AND recent action was 6 (random teleport), save RGB and Depth images
        4. Return RGB image as observation (regardless of FOV or action)
        
        Returns:
            dict: Observations with "policy" key containing RGB images
        """

        # Force the simulation to take 3 steps with no movement to stabilize everything
        for _ in range(3):
            self.sim.step()
            self._rgb_tiled_camera.update(self.sim.cfg.dt)
            self._distance_tiled_camera.update(self.sim.cfg.dt)

        import os
        
        # Ensure each env has it's own folder at data/env_{idx}
        env_ids = self._agent._ALL_INDICES
        
        # Get RGB and depth data from tiled camera
        rgb_data = self._rgb_tiled_camera.data.output["rgb"]  # Shape: (num_envs, height, width, 4) - RGBA
        depth_data = self._distance_tiled_camera.data.output["distance_to_camera"]  # Shape: (num_envs, height, width, 1)
        
        # Process each environment
        for env_id in env_ids:
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            
            # 1. Ensure folder structure exists
            base_path = "/home/arock3/Documents/isaaclab_game/isaac_game/cube_game/source/cube_game/data"
            env_folder = f"{base_path}/env_{env_id_item}"
            rgb_folder = f"{env_folder}/RGB"
            depth_folder = f"{env_folder}/Depth"
            
            os.makedirs(env_folder, exist_ok=True)
            os.makedirs(rgb_folder, exist_ok=True)
            os.makedirs(depth_folder, exist_ok=True)
            
            # Count existing images
            num_rgb_images = len([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
            num_depth_images = len([f for f in os.listdir(depth_folder) if f.endswith('.png')])
            
            # Stop saving if we already have 10 images
            if num_rgb_images >= 10 and num_depth_images >= 10:
                continue

            # 2. Check if camera and target are within FOV
            goal_visible, camera_visible = self.check_object_visibility(env_id_item)
            both_in_fov = goal_visible and camera_visible
            
            # 3. Check if recent action was 6 (random teleport)
            recent_action = int(self.actions[env_id_item].item()) if hasattr(self, 'actions') else -1
            
            # Save images if both conditions met
            if both_in_fov and recent_action == 6:
                # Use image count for filename
                rgb_filename = f"{rgb_folder}/image_{num_rgb_images:04d}.png"
                depth_filename = f"{depth_folder}/image_{num_depth_images:04d}.png"
                
                # Save RGB image (remove alpha channel, keep RGB only)
                rgb_img = rgb_data[env_id_item, :, :, :3]  # Remove alpha channel
                
                if rgb_img.max() <= 1.0:
                    # Convert to numpy and save (RGB values are in [0, 1])
                    rgb_np = (rgb_img.cpu().numpy() * 255.0).astype(np.uint8)
                else:
                    rgb_np = (rgb_img.cpu().numpy()).astype(np.uint8)
                import cv2
                cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
                
                # Save Depth image
                depth_img = depth_data[env_id_item, :, :, 0]  # Remove channel dimension
                
                # Normalize depth to [0, 255] for visualization
                # Replace inf values with max finite value
                depth_np = depth_img.cpu().numpy()
                depth_np[np.isinf(depth_np)] = depth_np[~np.isinf(depth_np)].max() if depth_np[~np.isinf(depth_np)].size > 0 else 0
                
                # Normalize to [0, 255]
                if depth_np.max() > depth_np.min():
                    depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)
                
                cv2.imwrite(depth_filename, depth_normalized)
                
                if self.verbose >= 1 and env_id_item == 0:
                    print(f"\n[ENV {env_id_item}] Saved images {num_rgb_images + 1}/10 (Action 6, both objects in FOV):")
                    print(f"  RGB: {rgb_filename}")
                    print(f"  Depth: {depth_filename}")
        
        # 4. Return RGB images as observations (regardless of FOV or action)
        # Remove alpha channel from RGB data and rearrange as (num_envs, 3, h, w)
        rgb_data = rgb_data.permute(0, 3, 1, 2)[:, :3, :, :]
        observations = {
            "policy": rgb_data.clone()  # Shape: (num_envs, height, width, 3)
        }
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards for the current step."""
        collision_mask, collision_types = self._check_collisions()

        return 0

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

    def _reset_idx(self,
                   env_ids: Sequence[int] | None,
                   randomize_objects: bool = True) -> None:
        """Wrapper that retries reset until at least 20 valid viewpoints are found."""
        MIN_VALID_VIEWPOINTS = 20
        max_reset_attempts = 10
        
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES
        
        # Convert list to tensor if needed
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        
        # Track which environments need retry
        envs_needing_retry = set(range(len(env_ids)))
        
        for reset_attempt in range(max_reset_attempts):
            if not envs_needing_retry:
                break
            
            if self.verbose >= 1 and reset_attempt > 0:
                print(f"\n[Reset Attempt {reset_attempt + 1}] Retrying {len(envs_needing_retry)} environments...")
            
            # Call the original reset logic
            self._reset_idx_internal(env_ids, randomize_objects)
            
            # Check which environments have enough valid viewpoints
            envs_needing_retry.clear()
            for i, env_id in enumerate(env_ids):
                env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
                
                if (self.valid_viewpoint_poses is None or 
                    env_id_item >= len(self.valid_viewpoint_poses) or
                    self.valid_viewpoint_poses[env_id_item] is None or 
                    len(self.valid_viewpoint_poses[env_id_item]) < MIN_VALID_VIEWPOINTS):
                    envs_needing_retry.add(i)
                    if self.verbose >= 1:
                        num_poses = 0 if self.valid_viewpoint_poses is None else len(self.valid_viewpoint_poses[env_id_item]) if env_id_item < len(self.valid_viewpoint_poses) and self.valid_viewpoint_poses[env_id_item] is not None else 0
                        print(f"  ✗ Env {env_id_item}: Only {num_poses}/{MIN_VALID_VIEWPOINTS} valid viewpoints - will retry")
                else:
                    if self.verbose >= 1 and reset_attempt > 0:
                        print(f"  ✓ Env {env_id_item}: {len(self.valid_viewpoint_poses[env_id_item])} valid viewpoints")
        
        # Final warning if any environments still failed
        if envs_needing_retry and self.verbose >= 1:
            failed_env_ids = [env_ids[i].item() if torch.is_tensor(env_ids[i]) else env_ids[i] for i in envs_needing_retry]
            print(f"\n⚠️  WARNING: After {max_reset_attempts} attempts, environments {failed_env_ids} still have insufficient valid viewpoints!")

    def _reset_idx_internal(self,
                       env_ids: Sequence[int] | None,
                       randomize_objects: bool = True) -> None:
        """Original reset logic - renamed from _reset_idx."""
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
        
        # Reset viewpoint pose counter for the environments being reset
        self.viewpoint_pose_counter[env_ids] = 0

        super()._reset_idx(env_ids)

        device = self._agent.device
        num_envs = len(env_ids)
        safe_x_range = self.center_to_boundary - 1.5
        safe_y_range = self.center_to_boundary - 1.5

        # Default states
        goal_default_state = self._goal.data.default_root_state[env_ids].clone()
        agent_default_state = self._agent.data.default_root_state[env_ids].clone()
        camera_obj_default_state = self._camera_obj.data.default_root_state[env_ids].clone()
        vpt_obj_default_state = self._vpt_objects.data.default_object_state[env_ids].clone()
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

            # target_occluded = False
            # target_outside_fov = False

            while not valid_positions and attempts < max_attempts:
                # Add 45-degree right roll (negative roll) as a tensor
                roll = torch.tensor(-math.radians(self.agent_camera_pitch),
                                    device=device)
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
                    yaw = torch.atan2(direction_to_goal[1],
                                      direction_to_goal[0])

                    # Add 90-degree offset to account for camera's local forward axis
                    yaw = yaw - math.radians(90)

                    # Debug: Print for specific environment if verbose
                    if self.verbose >= 2:  # Change i == 0 to the problematic env index
                        print(
                            f"Env {env_ids[i]}: Camera at {camera_new_pos[:2]}, Goal at {goal_new_pos[:2]}"
                        )
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
                    vpt_obj_new_pos[:, 2] += self.scene.env_origins[env_ids[i],
                                                                    2]

                    # Handle special cases based on flags
                    if target_occluded:
                        # Try to ensure occlusion by checking VPT objects block line of sight
                        occlusion_attempts = 0
                        max_occlusion_attempts = 100
                        goal_is_occluded = False

                        while not goal_is_occluded and occlusion_attempts < max_occlusion_attempts:
                            # Re-randomize VPT object positions
                            vpt_obj_offset_x = sample_uniform(
                                -safe_x_range, safe_x_range, (self.num_objs, ),
                                device)
                            vpt_obj_offset_y = sample_uniform(
                                -safe_y_range, safe_y_range, (self.num_objs, ),
                                device)
                            vpt_obj_new_pos[:, 0] = self.scene.env_origins[
                                env_ids[i], 0] + vpt_obj_offset_x
                            vpt_obj_new_pos[:, 1] = self.scene.env_origins[
                                env_ids[i], 1] + vpt_obj_offset_y

                            # Check if camera position is still valid (min 1 unit from VPT objects)
                            camera_distances_from_vpt = torch.norm(
                                camera_new_pos[:2].unsqueeze(0) -
                                vpt_obj_new_pos[:, :2],
                                dim=1)
                            camera_min_distance_ok = torch.all(
                                camera_distances_from_vpt >= 1.0)

                            if not camera_min_distance_ok:
                                occlusion_attempts += 1
                                continue

                            # Check if any VPT object occludes the goal using raycast
                            goal_is_occluded = self._check_occlusion_raycast(
                                camera_new_pos, goal_new_pos, env_ids[i])

                            occlusion_attempts += 1

                        if not goal_is_occluded and self.verbose >= 1:
                            print(
                                f"Warning: Could not create occlusion for env {env_ids[i]} after {max_occlusion_attempts} attempts"
                            )

                    elif target_outside_fov:
                        # Point camera away from goal (180 degrees opposite)
                        yaw_away = yaw + math.pi
                        quaternion_away = quat_from_euler_xyz(
                            roll,
                            # pitch,
                            torch.tensor(0, device=device),
                            # torch.tensor(0, device=device),
                            yaw_away)
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
                    yaw = torch.atan2(direction_to_goal[1],
                                      direction_to_goal[0])
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
                
                # Constraint: camera and goal must be within 3 units of each other
                camera_goal_distance = torch.norm(camera_new_pos[:2] - goal_new_pos[:2])
                camera_goal_distance_ok = camera_goal_distance <= 3.0

                if agent_min_distance_ok and camera_min_distance_ok and camera_goal_distance_ok:
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
        left_offset = -0.040  # Adjust this value to move camera left
        up_offset = 0.12

        # Get camera's forward and left directions from its orientation
        camera_forward = math_utils.quat_apply(
            camera_obj_default_state[:, 3:7],
            torch.tensor([0.0, 1.0, 0.0],
                         device=device).unsqueeze(0).expand(num_envs, -1))
        camera_left = math_utils.quat_apply(
            camera_obj_default_state[:, 3:7],
            torch.tensor([1.0, 0.0, 0.0],
                         device=device).unsqueeze(0).expand(num_envs, -1))
        camera_up = math_utils.quat_apply(
            camera_obj_default_state[:, 3:7],
            torch.tensor([0.0, 0.0, 1.0],
                         device=device).unsqueeze(0).expand(num_envs, -1))

        # Calculate new camera position (move backward and left)
        occlusion_camera_pos = (camera_obj_default_state[:, :3].clone() +
                                backward_offset * camera_forward +
                                left_offset * camera_left +
                                up_offset * camera_up)

        # Create 90-degree left rotation quaternion (rotation around Z-axis)
        theta_left = math.pi / 2  # 90 degrees
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0,
             math.sin(half_theta_left)],
            device=device)

        # Apply rotation to camera orientations for left-facing view
        rotated_orientations = math_utils.quat_mul(
            camera_obj_default_state[:, 3:7],
            left_90_quat.unsqueeze(0).expand(num_envs, -1))

        self._occlusion_camera.set_world_poses(
            positions=occlusion_camera_pos,
            orientations=rotated_orientations,
            env_ids=env_ids.tolist(),
            convention="world")

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

        # Generate valid circle viewpoint poses for these environments
        if self.verbose >= 1:
            print(f"\nGenerating valid circle viewpoint poses for {num_envs} environment(s)...")
        
        # Generate circle points with 2-degree step (180 points per circle)
        # Will try multiple radii (optimal to 1.3x optimal) with max 300 attempts per env
        all_valid_points = self.generate_valid_circle_points(
            env_ids=env_ids,
            angle_step=2.0,  # 180 points around the circle
            max_attempts=1000  # Maximum validation attempts per environment
        )
        
        # Initialize or update the valid_viewpoint_poses storage
        # Store as list of tensors since each env may have different number of valid points
        if self.valid_viewpoint_poses is None:
            self.valid_viewpoint_poses = [None] * self.num_envs
        
        # Update poses for the specified environments
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            
            # Convert 2D points (XY) to 3D poses (XYZ) with agent's default Z height
            valid_points_2d = all_valid_points[i]
            
            if valid_points_2d.shape[0] > 0:
                # Create 3D positions by adding Z coordinate
                agent_z = self._agent.data.default_root_state[env_id, 2]
                valid_points_3d = torch.zeros((valid_points_2d.shape[0], 3), device=device)
                valid_points_3d[:, :2] = valid_points_2d  # XY from circle
                valid_points_3d[:, 2] = agent_z  # Z from default height
                
                self.valid_viewpoint_poses[env_id_item] = valid_points_3d
                
        # Call parent step method
        obs, rewards, terminated, truncated, info = super().step(actions)
        
        # Check visibility for movement actions (0-3) and Action 5 (teleport) in environment 0
        # This happens AFTER physics step and camera update
        for env_id in range(self.num_envs):
            # Only check env 0
            if env_id != 0:
                continue
            
            action = int(actions[env_id].item())
            
            # Check for movement actions (0-3) OR Action 5 (teleport to valid viewpoint)
            if 0 <= action <= 3 or action == 5:
                # Check visibility of both objects (camera is already updated by parent step)
                self.check_object_visibility(env_id)
        
        return obs, rewards, terminated, truncated, info

    def _check_occlusion_raycast(self, camera_pos, goal_pos, env_id, camera=None):
        """
        Check if goal (red sphere with radius 0.15) is visible using depth camera.
        Checks entire FOV since camera may be perturbed and goal might not be centered.
        
        Args:
            camera_pos: Camera position (3D tensor)
            goal_pos: Goal position (center of sphere) (3D tensor)
            env_id: Environment ID (index in the batch)
            camera: Camera sensor to use (default: self._occlusion_camera)
            
        Returns:
            bool: True if goal is occluded (NOT visible), False if visible
        """
        # Use provided camera or default to occlusion camera
        if camera is None:
            camera = self._occlusion_camera
            
        # Calculate expected distance from camera to goal center
        expected_distance = torch.norm(goal_pos - camera_pos)

        if expected_distance < 1e-6:
            return False

        # Get depth data from specified camera for this environment
        depth_data = camera.data.output["distance_to_camera"][env_id]

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
        min_pixels_for_visibility = 15  # Adjust based on testing
        goal_is_visible = num_goal_pixels >= min_pixels_for_visibility

        # Also check if there's something significantly closer blocking the view
        # across the entire image
        min_depth = valid_depths.min()
        something_blocking = min_depth < (min_expected_depth - 0.1)

        if self.verbose >= 2:
            print(f"Env {env_id}:")
            print(f"  Expected distance: {expected_distance:.2f}")
            print(
                f"  Goal range: [{min_expected_depth:.2f}, {max_expected_depth:.2f}]"
            )
            print(f"  Min depth in FOV: {min_depth:.2f}")
            print(
                f"  Pixels in goal range: {num_goal_pixels}/{len(valid_depths)}"
            )
            print(f"  Something blocking: {something_blocking}")
            print(f"  Goal visible: {goal_is_visible}")

        # Goal is occluded if either:
        # 1. Not enough pixels at goal distance, OR
        # 2. Something is blocking (much closer than goal)
        goal_is_occluded = (not goal_is_visible) or something_blocking

        return goal_is_occluded
    
    def check_object_visibility(self, env_id: int, print_agent_state: bool = False) -> tuple[bool, bool]:
        """
        Check if goal ball (RED) and camera object (GREEN) are visible using semantic segmentation.
        
        Args:
            env_id: Environment ID
            print_agent_state: Whether to print agent position/quaternion (for debugging)
            
        Returns:
            tuple: (goal_visible, camera_visible)
        """
        # Thresholds for pixel counts
        GOAL_THRESHOLD = 10    # Red pixels for goal ball
        CAMERA_THRESHOLD = 10  # Green pixels for camera object
        
        # Get semantic segmentation image (RGBA format)
        sem_img = self._tiled_camera.data.output["semantic_segmentation"][env_id]
        
        # Extract RGB channels (ignore alpha channel)
        r = sem_img[:, :, 0]
        g = sem_img[:, :, 1]
        b = sem_img[:, :, 2]
        
        # Detect PURE RED pixels (goal ball) - R=1.0, G=0.0, B=0.0
        red_mask = (
            (r >= 0.95) &     # Pure red (very high)
            (g <= 0.05) &     # No green
            (b <= 0.05)       # No blue
        )
        red_count = red_mask.sum().item()
        
        # Detect PURE GREEN pixels (camera object) - R=0.0, G=1.0, B=0.0
        green_mask = (
            (r <= 0.05) &     # No red
            (g >= 0.95) &     # Pure green (very high)
            (b <= 0.05)       # No blue
        )
        green_count = green_mask.sum().item()
        
        # Check visibility based on thresholds
        goal_visible = red_count >= GOAL_THRESHOLD
        camera_visible = green_count >= CAMERA_THRESHOLD
        
        # Only log results for env 0
        # if env_id == 0:
        #     print(f"\n[ENV {env_id}] Object Visibility Check:")
        #     print(f"  Agent position: {self._agent.data.root_pos_w[env_id, :3]}")
        #     print(f"  Agent quaternion: {self._agent.data.root_quat_w[env_id]}")
        #     print(f"  Goal (RED) pixels: {red_count} (threshold: {GOAL_THRESHOLD}) -> {'VISIBLE' if goal_visible else 'NOT VISIBLE'}")
        #     print(f"  Camera (GREEN) pixels: {green_count} (threshold: {CAMERA_THRESHOLD}) -> {'VISIBLE' if camera_visible else 'NOT VISIBLE'}")
        
        return goal_visible, camera_visible
    
    def _calculate_optimal_radius(self, midpoint: torch.Tensor, camera_pos: torch.Tensor, 
                                   goal_pos: torch.Tensor, horizontal_fov_degrees: float = 35.0) -> float:
        """
        Calculate the optimal radius where agent can see both camera and goal objects.
        Uses FOV geometry to determine distance from midpoint.
        
        Args:
            midpoint: Midpoint between camera and goal (2D or 3D, only XY used)
            camera_pos: Camera object position (2D or 3D, only XY used)
            goal_pos: Goal ball position (2D or 3D, only XY used)
            horizontal_fov_degrees: Camera's horizontal FOV in degrees
            
        Returns:
            float: Optimal radius for the circle
        """
        # Calculate distance from midpoint to camera/goal (should be equal)
        # Use XY plane ONLY
        dist_to_camera = torch.norm(camera_pos[:2] - midpoint[:2]).item()
        dist_to_goal = torch.norm(goal_pos[:2] - midpoint[:2]).item()
        half_span = max(dist_to_camera, dist_to_goal)
        
        # FOV geometry: tan(fov/2) = half_span / distance
        # Solving for distance: distance = half_span / tan(fov/2)
        import math
        half_fov_radians = math.radians(horizontal_fov_degrees / 2.0)
        optimal_radius = half_span / math.tan(half_fov_radians)
        
        # Add 20% safety margin to ensure visibility
        radius_with_margin = optimal_radius * 1
        
        return radius_with_margin
    
    def _get_circle_point(self, center: torch.Tensor, radius: float, 
                          theta_degrees: float, device: torch.device, env_id: int = None) -> torch.Tensor:
        """
        Get a point on a circle given center, radius, and angle.
        
        Args:
            center: Circle center (XY coordinates in world space)
            radius: Circle radius
            theta_degrees: Angle in degrees (0-360)
            device: Torch device
            env_id: Environment ID (optional, for getting env origin)
            
        Returns:
            torch.Tensor: Point on circle (XY coordinates in world space)
        """
        # Convert theta to radians
        import math
        theta_radians = math.radians(theta_degrees)
        
        # Calculate x and y using trigonometry (center is already in world space)
        x = center[0] + radius * math.cos(theta_radians)
        y = center[1] + radius * math.sin(theta_radians)
        
        # Return as 2D tensor (already in world coordinates)
        return torch.tensor([x, y], device=device, dtype=torch.float32)

    def _is_point_valid(self, point: torch.Tensor, env_id: int, 
                        min_obstacle_distance: float = 0.4, 
                        min_camera_target_distance: float = 1.0,
                        print_details: bool = False) -> bool:
        """
        Validate if a point meets requirements:
        - Within environment bounds
        - Minimum distance from VPT obstacles
        - Minimum distance from camera (0.4 units)
        - Minimum distance from goal (0.4 units)
        
        Args:
            point: Point to validate (XY coordinates)
            env_id: Environment ID
            min_obstacle_distance: Minimum distance from obstacles
            print_details: Whether to print validation details (for debugging)
            
        Returns:
            bool: True if point is valid
        """
        if print_details:
            print(f"\n  Validating point: {point.cpu().numpy()}")
        
        # 1. Check if point is within bounds
        env_origin = self.scene.env_origins[env_id, :2]
        boundary_limit = self.center_to_boundary.item() if torch.is_tensor(self.center_to_boundary) else self.center_to_boundary
        
        # Point must be within env_origin ± boundary_limit
        min_bound = env_origin - boundary_limit
        max_bound = env_origin + boundary_limit
        
        if not (torch.all(point >= min_bound) and torch.all(point <= max_bound)):
            if print_details:
                print(f"    ❌ OUT OF BOUNDS (limit: ±{boundary_limit:.2f} from origin)")
            return False
        
        if print_details:
            print(f"    ✓ Within bounds")
        
        # 2. Check distance from VPT objects
        vpt_positions = self._vpt_objects.data.object_pos_w[env_id, :, :2]
        distances = torch.norm(point.unsqueeze(0) - vpt_positions, dim=1)
        min_vpt_dist = distances.min().item()
        
        if torch.any(distances < min_obstacle_distance):
            if print_details:
                print(f"    ❌ TOO CLOSE to VPT obstacles (min dist: {min_vpt_dist:.2f}, required: {min_obstacle_distance:.2f})")
            return False
        
        if print_details:
            print(f"    ✓ Clear of VPT obstacles (min dist: {min_vpt_dist:.2f})")
        
        # 3. Check distance from camera object
        camera_pos = self._camera_obj.data.root_pos_w[env_id, :2]
        camera_distance = torch.norm(point - camera_pos).item()
        
        if camera_distance < min_camera_target_distance:
            if print_details:
                print(f"    ❌ TOO CLOSE to camera (dist: {camera_distance:.2f}, required: {min_camera_target_distance:.2f})")
            return False
        
        if print_details:
            print(f"    ✓ Clear of camera (dist: {camera_distance:.2f})")
        
        # 4. Check distance from goal
        goal_pos = self._goal.data.root_pos_w[env_id, :2]
        goal_distance = torch.norm(point - goal_pos).item()
        
        if goal_distance < min_camera_target_distance:
            if print_details:
                print(f"    ❌ TOO CLOSE to goal (dist: {goal_distance:.2f}, required: {min_camera_target_distance:.2f})")
            return False
        
        if print_details:
            print(f"    ✓ Clear of goal (dist: {goal_distance:.2f})")
            print(f"    ✅ VALID")

        return True

    def generate_valid_circle_points(self, env_ids: torch.Tensor,
                                     angle_step: float = 2.0,
                                     max_attempts: int = 300) -> torch.Tensor:
        """
        Generate valid viewpoint positions on a circle around camera/goal midpoint.
        
        Main function that orchestrates the circle point generation:
        1. Calculate midpoint between camera and goal
        2. Calculate optimal (minimum) radius using FOV geometry
        3. Try multiple radii from optimal to 1.3x optimal
        4. Generate points at 2-degree intervals (180 points per circle)
        5. Validate each point (bounds, obstacles, visibility)
        6. For env 0, randomly select and print 10 sample points
        
        Args:
            env_ids: Environment IDs to generate points for
            angle_step: Angular step size in degrees (default: 2.0)
            max_attempts: Maximum validation attempts per environment (default: 300)
            
        Returns:
            list: List of tensors, one per environment with valid circle points (num_valid_points, 2)
        """
        import random
        device = self._agent.device
        num_envs = len(env_ids)
        
        # Generate angles from 0 to 360 at 2-degree step (180 points)
        angles = torch.arange(0, 360, angle_step, device=device)
        
        # Store valid points for each environment (list of tensors)
        all_valid_points = []
        
        for i, env_id in enumerate(env_ids):
            # 1. Get camera and goal positions (already in world coordinates)
            camera_pos = self._camera_obj.data.root_pos_w[env_id]
            goal_pos = self._goal.data.root_pos_w[env_id]
            
            # 2. Calculate midpoint (XY plane ONLY, in world coordinates)
            midpoint = (camera_pos[:2] + goal_pos[:2]) / 2.0  # Use only XY components
            
            # 3. Calculate optimal (MINIMUM) radius using FOV geometry
            # Pass only XY components to avoid Z affecting the calculation
            camera_pos_2d = camera_pos[:2]
            goal_pos_2d = goal_pos[:2]
            min_radius = self._calculate_optimal_radius(
                torch.cat([midpoint, torch.zeros(1, device=midpoint.device)]),  # Add dummy Z for compatibility
                torch.cat([camera_pos_2d, torch.zeros(1, device=midpoint.device)]),
                torch.cat([goal_pos_2d, torch.zeros(1, device=midpoint.device)])
            )
            
            # Create multiple radii to try (e.g., 5 steps from min to max)
            num_radius_steps = 10
            radii_to_try = torch.linspace(min_radius, min_radius * 1.3, num_radius_steps, device=device)
            
            if self.verbose >= 1:
                print(f"\n[Env {env_id}] Generating circle points:")
                print(f"  Camera pos: {camera_pos[:2]}")
                print(f"  Goal pos: {goal_pos[:2]}")
                print(f"  Midpoint: {midpoint[:2]}")
                print(f"  Min radius (optimal): {min_radius:.2f}")
                print(f"  Max radius (1.3x): {min_radius * 1.3:.2f}")
                print(f"  Testing {num_radius_steps} radii with {len(angles)} angles each")
            
            # 4. Generate and validate circle points across multiple radii
            valid_points_for_env = []
            attempts = 0
            
            # Try each radius until we reach max_attempts or run out of radii
            for radius in torch.Tensor([min_radius]):
                if attempts >= max_attempts:
                    break
                    
                for angle_idx, angle in enumerate(angles):
                    if attempts >= max_attempts:
                        break
                    
                    # Get circle point (XY coordinates in world space)
                    point = self._get_circle_point(
                        midpoint[:2],
                        radius.item(),
                        angle.item(),
                        device,
                        env_id
                    )
                    
                    # Validate the point (no printing yet)
                    is_valid = self._is_point_valid(point, env_id, print_details=False)
                    attempts += 1
                    
                    if is_valid:
                        valid_points_for_env.append(point)
            
            if self.verbose >= 1:
                print(f"  Total attempts: {attempts}")
                print(f"  Valid points found: {len(valid_points_for_env)}")
            
            # 5. For env 0, randomly select and print 10 sample points with details
            if env_id == 0 and len(valid_points_for_env) > 0:
                num_samples = min(10, len(valid_points_for_env))
                sample_indices = random.sample(range(len(valid_points_for_env)), num_samples)
                
                print(f"\n[ENV 0] Randomly selected {num_samples} points for detailed validation:")
                print(f"  Agent current position: {self._agent.data.root_pos_w[env_id, :3]}")
                print(f"  Agent current quaternion: {self._agent.data.root_quat_w[env_id]}")
                
                for sample_idx in sample_indices:
                    sample_point = valid_points_for_env[sample_idx]
                    # Re-validate with printing enabled
                    self._is_point_valid(sample_point, env_id, print_details=False)
            
            # 6. Store valid points for this environment
            if len(valid_points_for_env) > 0:
                # Convert list to tensor
                valid_points_tensor = torch.stack(valid_points_for_env)
            else:
                # If no valid points found, return empty tensor
                valid_points_tensor = torch.zeros((0, 2), device=device)
                if self.verbose >= 1:
                    print(f"  WARNING: No valid circle points found for env {env_id}")
            
            all_valid_points.append(valid_points_tensor)
        
        # Return list of tensors (one per environment)
        # Note: Can't easily convert to single tensor since each env may have different number of valid points
        return all_valid_points


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
