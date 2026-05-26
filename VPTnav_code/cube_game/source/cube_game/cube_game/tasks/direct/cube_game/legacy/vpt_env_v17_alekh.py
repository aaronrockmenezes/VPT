from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import random
import numpy as np
import os
import cv2
import re
from typing import List
import time
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils

import isaacsim.core.utils.bounds as bounds_utils
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.api import World
from pxr import Gf, Sdf, UsdGeom, Usd, UsdLux, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.utils import math as math_utils

from .vpt_env_cfg_v17 import VPTEnvCfg
from .spawn_boundary import get_vpt_material_paths, get_mat_material_paths
from .env_timer import EnvTimer


class VPTEnv(DirectRLEnv):

    cfg: VPTEnvCfg

    def __init__(self,
                 cfg: VPTEnvCfg,
                 render_mode: str | None = None,
                 **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Configuration parameters from cfg
        self.action_scale = self.cfg.action_scale
        self.boundary_limits = self.cfg.boundary_limits
        self.agent_height = self.cfg.agent_height
        self.agent_camera_pitch = self.cfg.agent_camera_pitch
        self.goal_radius = self.cfg.goal_radius
        self.config_file = self.cfg.config_file

        # VPT parameters
        self.num_objs = self.cfg.num_vpt_objs
        self.active_vpt_objs = self.cfg.objects_per_env
        self.storage_position = torch.tensor([250.0, 250.0, 0.0])
        self.active_vpt_indices = [
            None
        ] * self.num_envs  # Track which 20 are active per env

        # Derived environment parameters
        self.center_to_boundary = torch.abs(
            torch.tensor(self.boundary_limits).view(-1)[0])

        # Verbosity and visibility thresholds
        self.verbose = 0
        self.goal_pixel_threshold = 50  # Minimum pixels for goal visibility
        self.camera_pixel_threshold = 500  # Minimum pixels for camera visibility

        # Data collection parameters
        self.images_per_env = 20  # Number of images to collect per environment
        self.min_viewpoint_distance = 0.1  # Minimum distance between viewpoints (meters)
        self.save_camera_pov = True

        # Viewpoint and collection state variables
        self.valid_viewpoint_poses = [None] * self.num_envs
        self.selected_viewpoints_for_collection = [None] * self.num_envs
        self.current_collection_index = torch.zeros(self.num_envs,
                                                    dtype=torch.long,
                                                    device=self.device)
        self.viewpoint_pose_counter = torch.zeros(self.num_envs,
                                                  dtype=torch.long,
                                                  device=self.device)
        self.used_viewpoint_indices = [set() for _ in range(self.num_envs)]
        self.moved_vpt_objs = [list() for _ in range(self.num_envs)]

        # Environment management state
        self.next_env_folder_idx = 0
        self.env_visibility_labels = {}
        self.env_visibility_reasons = {}
        self._reset_called = False

        # Get active GPU ID (there will be only 1)
        self.GPU_ID = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

        print("*" * 50)
        print(
            f"🚀 Initializing VPTEnv on GPU {self.GPU_ID} with {self.num_envs} environments..."
        )
        print("*" * 50)

        # File paths
        self.base_path = f"/home/arock3/data_v17_rl_high/data_{self.GPU_ID}"
        # self.base_path = "/media/data_cifs_lrs/projects/prj_robotics/VPTnav_v6_1k_envs"
        self.visibility_labels_json_path = f"{self.base_path}/visibility_labels.json"

        # Mode determination
        if self.config_file is not None and os.path.exists(self.config_file):
            self.mode = "testing"
        else:
            self.mode = "data_collection"

        self.total_envs_to_sim = 1000
        self.slot_to_env_id = list(range(self.num_envs))
        self.next_env_id = self.num_envs
        self.completed_envs = set()
        self.slot_attempt_counts = [0] * self.num_envs
        self.max_attempts_per_slot = 20 * 50  # Full resets * Inner resets

        self.used_vpt_objects = set()
        self._preallocate_visibility_labels()
        self.verbose = 2

        #rotation angle
        self.theta = math.pi / 24
        self.half_theta = self.theta / 2
        # Shape (1, 4) for broadcasting
        self.rot_q_left = torch.tensor(
            [[math.cos(self.half_theta), 0., 0., math.sin(self.half_theta)]],
            device=self.device)
        self.rot_q_right = torch.tensor(
            [[math.cos(self.half_theta), 0., 0., -math.sin(self.half_theta)]],
            device=self.device)
        
        self.move_speed = 1.2

        self.times = {}

    def close(self):
        super().close()

    def _cache_valid_shapes(self):
        """Cache boolean mask for objects that are valid for ball placement (Cylinders/Cuboids)."""
        # Create a boolean mask of size [num_total_vpt_objs]
        self.valid_shape_mask = torch.zeros(self.num_objs,
                                            dtype=torch.bool,
                                            device=self.device)

        vpt_keys = list(self.cfg.vpt_objects.rigid_objects.keys())
        for i, key in enumerate(vpt_keys):
            spawn_cfg = self.cfg.vpt_objects.rigid_objects[key].spawn
            if isinstance(spawn_cfg,
                          (sim_utils.CylinderCfg, sim_utils.CuboidCfg)):
                self.valid_shape_mask[i] = True

    def _preallocate_visibility_labels(self) -> None:
        """Pre-allocate visibility labels for all environments in 50/25/25 proportion."""
        total = self.total_envs_to_sim
        num_in_view = total // 2
        num_occluded = total // 4
        num_outside_fov = total - num_in_view - num_occluded
        # num_in_view = 1
        # num_occluded = total - 2
        # num_outside_fov = 1

        # Create list of all labels
        all_labels = (["in_view"] * num_in_view + ["occluded"] * num_occluded +
                      ["outside_fov"] * num_outside_fov)
        random.shuffle(all_labels)

        # Store as a list to pop from
        self.visibility_label_pool = all_labels

        if self.verbose >= 1:
            print(f"📋 Pre-allocated {total} visibility labels:")
            print(f"   - in_view: {num_in_view}")
            print(f"   - occluded: {num_occluded}")
            print(f"   - outside_fov: {num_outside_fov}")

    def _assign_next_visibility_label(self, folder_idx: int) -> str:
        """Assign the next visibility label from the pre-allocated pool."""
        if not self.visibility_label_pool:
            raise RuntimeError("Visibility label pool exhausted!")

        category = self.visibility_label_pool.pop(0)

        if category == "in_view":
            self.env_visibility_labels[folder_idx] = "Yes"
            self.env_visibility_reasons[folder_idx] = "in_view"
        elif category == "occluded":
            self.env_visibility_labels[folder_idx] = "No"
            self.env_visibility_reasons[folder_idx] = "occluded"
        else:  # outside_fov
            self.env_visibility_labels[folder_idx] = "No"
            self.env_visibility_reasons[folder_idx] = "outside_fov"

        return category

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground",
                           cfg=GroundPlaneCfg(size=(1000, 1000)))
        self._agent = RigidObject(self.cfg.agent)
        self._goal = RigidObject(self.cfg.goal_ball)
        self._boundary_top = RigidObject(self.cfg.top_wall)
        self._boundary_bottom = RigidObject(self.cfg.bottom_wall)
        self._boundary_left = RigidObject(self.cfg.left_wall)
        self._boundary_right = RigidObject(self.cfg.right_wall)
        self._camera_obj = RigidObject(self.cfg.camera_obj)
        self._vpt_objects = RigidObjectCollection(self.cfg.vpt_objects)
        self._mat = RigidObject(self.cfg.mat)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.rigid_objects["agent"] = self._agent
        self.scene.rigid_objects["goal"] = self._goal
        self.scene.rigid_objects["boundary_top"] = self._boundary_top
        self.scene.rigid_objects["boundary_bottom"] = self._boundary_bottom
        self.scene.rigid_objects["boundary_left"] = self._boundary_left
        self.scene.rigid_objects["boundary_right"] = self._boundary_right
        self.scene.rigid_objects["camera_object"] = self._camera_obj
        self.scene.rigid_objects["mat"] = self._mat
        self.scene.rigid_object_collections["vpt_objects"] = self._vpt_objects

        # TODO: COMMENT THIS
        # self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # self.scene.sensors["semantic_tiled_camera"] = self._tiled_camera

        self._rgb_tiled_camera = TiledCamera(self.cfg.rgb_tiled_camera)
        self.scene.sensors["rgb_tiled_camera"] = self._rgb_tiled_camera

        # TODO: COMMENT THIS
        # self._distance_tiled_camera = TiledCamera(
        #     self.cfg.distance_tiled_camera)
        # self.scene.sensors[
        #     "distance_tiled_camera"] = self._distance_tiled_camera

        self._occlusion_camera = TiledCamera(self.cfg.occlusion_camera)
        self.scene.sensors["occlusion_camera"] = self._occlusion_camera

        light_cfg_a = sim_utils.SphereLightCfg(intensity=1000.0,
                                               color=(0.75, 0.75, 0.75))

        # Spawn them in the template environment
        light_cfg_a.func("/World/envs/env_0/Light_A", light_cfg_a)

        self.mat_material_paths = []
        self.vpt_material_paths = []
        self.mat_material_configs = self.get_material_configs(
            material_type="mat")
        self.vpt_material_configs = self.get_material_configs(
            material_type="vpt")
        for idx, material in enumerate(self.mat_material_configs):
            material.func(f"/World/Looks/mat_material_{idx}", material)
            self.mat_material_paths.append(f"/World/Looks/mat_material_{idx}")

        for idx, material in enumerate(self.vpt_material_configs):
            material.func(f"/World/Looks/vpt_material_{idx}", material)
            self.vpt_material_paths.append(f"/World/Looks/vpt_material_{idx}")

    def check_batch_object_visibility(
            self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Check object visibility for a batch of environments in parallel."""
        sem_imgs = self._rgb_tiled_camera.data.output["semantic_segmentation"][
            env_ids]

        # Extract channels
        r = sem_imgs[..., 0]
        g = sem_imgs[..., 1]
        b = sem_imgs[..., 2]
        a = sem_imgs[..., 3]

        # Exact match for Red (255, 0, 0, 255)
        red_mask = (r == 255) & (g == 0) & (b == 0) & (a == 255)

        # Exact match for Green (0, 255, 0, 255)
        green_mask = (r == 0) & (g == 255) & (b == 0) & (a == 255)

        # Count pixels
        red_counts = red_mask.sum(dim=(1, 2))
        green_counts = green_mask.sum(dim=(1, 2))

        # Check against thresholds
        goal_visible = red_counts >= self.goal_pixel_threshold
        camera_visible = green_counts >= self.camera_pixel_threshold

        # Debug printing for matches
        both_visible_mask = goal_visible & camera_visible
        ids_with_both = env_ids[both_visible_mask]

        if len(ids_with_both) > 0:
            print(f"Envs with both visible: {ids_with_both.tolist()}")

        return goal_visible, camera_visible

    def move_agent(self, actions, env_ids: Sequence[int] | None = None):
        """
        Moves the agent kinematically (teleportation) based on actions.
        Removed velocity writing for performance optimization.
        Removed upright quaternion normalization (uses raw current orientation).
        """
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids,
                                   dtype=torch.long,
                                   device=self.device)

        # Fix: Ensure actions matches the shape of env_ids and is STRICTLY 1D
        if len(env_ids) != len(actions):
            current_actions = actions[env_ids].flatten()
        else:
            current_actions = actions.flatten()

        dt = self.cfg.sim.dt

        # --- 1. Separate Physics Actions from Resets ---
        reset_mask_5 = (current_actions == 5)
        reset_mask_6 = (current_actions == 6)
        physics_mask = ~(reset_mask_5 | reset_mask_6)

        # --- 2. Handle Resets / Debug ---
        # Notice we can remove .squeeze() here now!
        if reset_mask_5.any():
            self._reset_idx(env_ids[reset_mask_5], rl_reset=False)

        if reset_mask_6.any():
            print("+" * 50)
            self._reset_idx(env_ids[reset_mask_6], rl_reset=True)

        reset_time = time.time()

        # If no agents are performing physics actions, exit early
        if not physics_mask.any():
            return

        # Filter indices and actions for physics updates
        # Notice we can remove .squeeze() here too!
        phys_ids = env_ids[physics_mask]
        phys_actions = current_actions[physics_mask]

        # --- 3. Calculate "Proposed" State ---
        # Clone tentative_pos as we modify it in-place
        tentative_pos = self._agent.data.root_pos_w[phys_ids].clone()
        current_quat = self._agent.data.root_quat_w[phys_ids]

        # A. Calculate Rotation (Sparse)
        mask_left = (phys_actions == 2)
        mask_right = (phys_actions == 3)

        # Initialize new_quat directly from the current orientation
        new_quat = current_quat.clone()

        if mask_left.any():
            n_left = mask_left.sum()
            # Apply rotation to the CURRENT orientation
            new_quat[mask_left] = math_utils.quat_mul(
                current_quat[mask_left], self.rot_q_left.expand(n_left, -1))

        if mask_right.any():
            n_right = mask_right.sum()
            new_quat[mask_right] = math_utils.quat_mul(
                current_quat[mask_right], self.rot_q_right.expand(n_right, -1))

        # B. Calculate Position Update (Sparse)
        mask_fwd = (phys_actions == 0)
        mask_bwd = (phys_actions == 1)
        moving_mask = mask_fwd | mask_bwd

        if moving_mask.any():
            n_moving = moving_mask.sum()
            local_move_subset = torch.zeros((n_moving, 3), device=self.device)

            moving_actions = phys_actions[moving_mask]
            local_move_subset[moving_actions == 0, 0] = 1.0
            local_move_subset[moving_actions == 1, 0] = -1.0

            # Rotate local vector by CURRENT orientation to get world frame vector
            world_vel_3d = math_utils.quat_apply(current_quat[moving_mask],
                                                 local_move_subset)

            # Apply displacement: (Direction * Speed * dt)
            displacement = world_vel_3d * self.move_speed * dt
            tentative_pos[moving_mask] += displacement

        # Enforce height constraint
        tentative_pos[:, 2] = self._agent.data.default_root_state[phys_ids, 2]

        # --- 4. Collision Guard ---
        if moving_mask.any():
            collision_mask = self._check_collisions_vectorized(
                phys_ids, tentative_pos, new_quat)
            collision_mask = collision_mask & moving_mask

            if collision_mask.any():
                colliding_indices = torch.where(collision_mask)[0]
                # Revert state for colliders
                tentative_pos[colliding_indices] = self._agent.data.root_pos_w[
                    phys_ids[colliding_indices]]
                new_quat[colliding_indices] = current_quat[colliding_indices]

        # --- 5. Apply Final State ---
        self._agent.write_root_com_pose_to_sim(
            torch.cat([tentative_pos, new_quat], dim=1), phys_ids)

        self._agent.reset()

    # def _update_camera_poses(self, env_ids):
    #     """Helper to handle the camera/occlusion update logic."""
    #     camera_obj_pos = self._camera_obj.data.root_pos_w[env_ids].clone()
    #     camera_obj_quat = self._camera_obj.data.root_quat_w[env_ids].clone()

    #     # Rotate 90 degrees left
    #     half_theta = (math.pi / 2) / 2
    #     left_90_quat = torch.tensor(
    #         [math.cos(half_theta), 0.0, 0.0,
    #          math.sin(half_theta)],
    #         device=self.device)

    #     rotated_orientations = math_utils.quat_mul(
    #         camera_obj_quat, left_90_quat.expand(len(env_ids), -1))

    #     self._occlusion_camera.set_world_poses(
    #         positions=camera_obj_pos,
    #         orientations=rotated_orientations,
    #         env_ids=env_ids.tolist(),
    #         convention="world")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.old_actions = self.actions.clone() if hasattr(
            self, 'actions') else torch.zeros_like(actions)
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.move_agent(self.actions)

    def _get_observations(self, mode=None) -> dict:
        if not self._reset_called:
            raise RuntimeError(
                "ERROR: _get_observations called before _reset_idx! "
                "Environment initialization must call reset first.")
        if mode is None:
            mode = self.mode

        # Update cameras for normal observation
        self._rgb_tiled_camera.update(self.sim.cfg.dt)

        rgb_data = self._rgb_tiled_camera.data.output["rgb"]
        # Permute, cast to float, and normalize to [0, 1]
        rgb_data = rgb_data.permute(0, 3, 1, 2)[:, :3, :, :].float()
        observations = {"policy": rgb_data.clone()}

        self.obs = rgb_data

        return observations

    def _get_rewards(self) -> torch.Tensor:
        distance = (self._camera_obj.data.root_pos_w[:, :2] - self._agent.data.root_pos_w[:, :2]) ** 2
        distance = torch.sqrt(distance.sum(dim=1))
        reward = -1 * distance

        # angle_to_camera = (self._camera_obj.data.root_quat_w * self._agent.data.root_quat_w).sum(dim=1) ** 2
        # import pdb; pdb.set_trace()

        yaw_green_cam = math_utils.yaw_quat(self._camera_obj.data.root_quat_w + torch.tensor((math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)), device=self.device))
        yaw_agent = math_utils.yaw_quat(self._agent.data.root_quat_w)
        angle_to_camera = math_utils.quat_error_magnitude(yaw_green_cam, yaw_agent)

        reward[distance < 0.4] = reward[distance < 0.4] + 1 * angle_to_camera[distance < 0.4]
        
        #print(f"Distance: {distance.cpu().numpy()}, Angle to Camera: {angle_to_camera.cpu().numpy()}, Reward: {reward.cpu().numpy()}")
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance = (self._camera_obj.data.root_pos_w[:, :2] - self._agent.data.root_pos_w[:, :2]) ** 2
        distance = torch.sqrt(distance.sum(dim=1))

        yaw_green_cam = math_utils.yaw_quat(self._camera_obj.data.root_quat_w + torch.tensor((math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)), device=self.device))
        yaw_agent = math_utils.yaw_quat(self._agent.data.root_quat_w)
        angle_to_camera = math_utils.quat_error_magnitude(yaw_green_cam, yaw_agent)

        goal_reached = (distance <= 0.2) & (angle_to_camera <= 0.1)

        terminated = torch.zeros(self.num_envs,
                                 dtype=torch.bool,
                                 device=self.device)
        terminated = terminated #| goal_reached

        time_outs = (self.episode_length_buf >= self.max_episode_length)

        return terminated, time_outs

    def _validate_env_state(self, env_id, folder_idx, min_viewpoints):
        """Validate single environment's state after reset attempt."""
        env_id_item = env_id.item()

        if (self.valid_viewpoint_poses is None
                or env_id_item >= len(self.valid_viewpoint_poses)
                or self.valid_viewpoint_poses[env_id_item] is None or len(
                    self.valid_viewpoint_poses[env_id_item]) < min_viewpoints):
            num_poses = 0 if self.valid_viewpoint_poses is None or self.valid_viewpoint_poses[
                env_id_item] is None else len(
                    self.valid_viewpoint_poses[env_id_item])
            return False, f"insufficient viewpoints: {num_poses}/{min_viewpoints}"

        visibility_reason = self.env_visibility_reasons.get(
            folder_idx, "unknown")
        if visibility_reason in ["in_view", "occluded"]:
            camera_pos = self._camera_obj.data.root_pos_w[env_id]
            goal_pos = self._goal.data.root_pos_w[env_id]
            is_occluded = self._check_occlusion_raycast(
                camera_pos, goal_pos, env_id)
            expected_occluded = (visibility_reason == "occluded")

            if is_occluded != expected_occluded:
                return False, f"occlusion mismatch: expected {'occluded' if expected_occluded else 'visible'}, got {'occluded' if is_occluded else 'visible'}"

        return True, ""

    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)
        return obs, rewards, terminated, truncated, info

    def render(self):

        frame = self.obs[0].permute(1, 2, 0).cpu().numpy()
        #draw the action from 1st env on the frame
        action = self.actions[0].cpu().numpy()
        
        if action == 0:
            action_str = "Forward"
        elif action == 1:
            action_str = "Backward"
        elif action == 2:
            action_str = "Left"
        elif action == 3:
            action_str = "Right"
        else:
            action_str = "action " + str(action)
        cv2.putText(frame, action_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # #write agent pos and quat on the frame upto 4 decimal places in human readable format
        # agent_pos = self._agent.data.root_pos_w[0].cpu().numpy()
        # agent_quat = self._agent.data.root_quat_w[0].cpu().numpy()
        # #agent_pos is an array of unknown length, so we convert it to string
        # #rounded to 4 decimal places
        # pos_string = ", ".join([f"{p:.4f}" for p in agent_pos])
        # quat_string = ", ".join([f"{q:.4f}" for q in agent_quat])
        # cv2.putText(frame, f"Agnt Pos: {pos_string}", (10, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
        # cv2.putText(frame, f"Agnt Quat: {quat_string}", (10, 80),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # #save frame as image
        # cv2.imwrite(f"logs/frame_{self.episode_length_buf[0].cpu().numpy()}_{action_str}.png", frame)

        return frame

    def _check_occlusion_raycast(self,
                                 camera_pos,
                                 goal_pos,
                                 env_id,
                                 camera=None):
        if camera is None:
            camera = self._occlusion_camera

        camera_obj_pos = self._camera_obj.data.root_pos_w[env_id]
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_id]
        occlusion_cam_pos = camera.data.pos_w[env_id]

        pos_diff = torch.norm(camera_obj_pos - occlusion_cam_pos).item()

        if pos_diff > 0.1:
            device = camera_obj_pos.device
            theta_left = math.pi / 2
            half_theta_left = theta_left / 2
            left_90_quat = torch.tensor([
                math.cos(half_theta_left), 0.0, 0.0,
                math.sin(half_theta_left)
            ],
                                        device=device)

            rotated_orientation = math_utils.quat_mul(
                camera_obj_quat.unsqueeze(0),
                left_90_quat.unsqueeze(0)).squeeze(0)

            camera.set_world_poses(
                positions=camera_obj_pos.unsqueeze(0),
                orientations=rotated_orientation.unsqueeze(0),
                env_ids=[env_id.item() if torch.is_tensor(env_id) else env_id],
                convention="world")

            for _ in range(1):
                self.sim.step()

            new_occlusion_cam_pos = camera.data.pos_w[env_id]
            new_pos_diff = torch.norm(camera_obj_pos -
                                      new_occlusion_cam_pos).item()
            # print(f"    ✓ After fix: Distance = {new_pos_diff:.4f}")

        GOAL_THRESHOLD = self.goal_pixel_threshold
        sem_img = camera.data.output["semantic_segmentation"][env_id]
        r = sem_img[:, :, 0]
        g = sem_img[:, :, 1]
        b = sem_img[:, :, 2]

        red_mask = ((r >= 0.95) & (g <= 0.05) & (b <= 0.05))
        red_count = red_mask.sum().item()

        return red_count < GOAL_THRESHOLD

    def _save_visibility_labels(self):
        """Save visibility labels to JSON file."""
        import json

        os.makedirs(os.path.dirname(self.visibility_labels_json_path),
                    exist_ok=True)

        env_details = {}
        for folder_idx in self.env_visibility_labels.keys():
            label = self.env_visibility_labels[folder_idx]
            reason = self.env_visibility_reasons.get(folder_idx, "unknown")
            env_details[str(folder_idx)] = {"label": label, "reason": reason}

        reason_counts = {
            "in_view": 0,
            "occluded": 0,
            "outside_fov": 0,
            "unknown": 0
        }

        for reason in self.env_visibility_reasons.values():
            if reason in reason_counts:
                reason_counts[reason] += 1
            else:
                reason_counts["unknown"] += 1

        labels_data = {
            "environments": env_details,
            "statistics": {
                "total_environments":
                len(self.env_visibility_labels),
                "yes_count":
                sum(1 for v in self.env_visibility_labels.values()
                    if v == "Yes"),
                "no_count":
                sum(1 for v in self.env_visibility_labels.values()
                    if v == "No"),
                "by_reason":
                reason_counts,
                "next_env_folder_idx":
                self.next_env_folder_idx
            }
        }

        with open(self.visibility_labels_json_path, 'w') as f:
            json.dump(labels_data, f, indent=2)

    def _check_target_in_img(self,
                             file_name: str,
                             cam_pov,
                             return_red_count: bool = False) -> bool:
        cv2.imwrite(file_name, cv2.cvtColor(cam_pov, cv2.COLOR_RGB2BGR))

        # Load image back and count red pixels
        loaded_img = cv2.imread(file_name)
        loaded_img_rgb = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)

        r = loaded_img_rgb[:, :, 0]
        g = loaded_img_rgb[:, :, 1]
        b = loaded_img_rgb[:, :, 2]
        max_r = 0.95 * 255
        min_gb = 0.05 * 255

        red_mask = ((r >= max_r) & (g <= min_gb) & (b <= min_gb)
                    )  # 0.95*255=242, 0.05*255=13
        red_count = red_mask.sum().item()

        target_visible_in_camera = red_count >= self.goal_pixel_threshold

        # Check for solid filled circles using HoughCircles
        gray = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray,
                                   cv2.HOUGH_GRADIENT,
                                   dp=1,
                                   minDist=10,
                                   param1=100,
                                   param2=12,
                                   minRadius=5,
                                   maxRadius=200)

        has_solid_circles = circles is not None and len(circles[0]) > 0

        if return_red_count:
            return target_visible_in_camera or has_solid_circles, red_count
        return target_visible_in_camera or has_solid_circles

    def _save_env_config_to_json(self, env_id: int, folder_idx: int):
        """
        Save complete environment configuration to JSON file using cfg data only.
        
        Saves:
        - Goal ball (position, orientation, spawn cfg)
        - Camera object (position, orientation, spawn cfg)
        - Agent (position, orientation, spawn cfg)
        - VPT objects (positions, orientations, spawn cfg per object) - ONLY ACTIVE ONES
        - Active VPT indices
        - Visibility label and reason
        - Valid viewpoint poses
        - Environment settings from cfg
        
        Args:
            env_id: Environment ID (index in batch)
            folder_idx: Folder index for this environment
        """
        import json

        device = self._agent.device

        # Get visibility info
        label = self.env_visibility_labels.get(folder_idx, "UNKNOWN")
        reason = self.env_visibility_reasons.get(folder_idx, "unknown")

        # Get goal ball info (position + cfg spawn parameters)
        goal_pos = self._goal.data.root_pos_w[env_id].cpu().numpy().tolist()
        goal_quat = self._goal.data.root_quat_w[env_id].cpu().numpy().tolist()
        goal_spawn_cfg = {
            "radius": float(self.cfg.goal_ball.spawn.radius),
            "rigid_props": {
                "disable_gravity":
                bool(self.cfg.goal_ball.spawn.rigid_props.disable_gravity)
            },
            "mass_props": {
                "mass": float(self.cfg.goal_ball.spawn.mass_props.mass)
            },
            "visual_material": {
                "diffuse_color":
                list(self.cfg.goal_ball.spawn.visual_material.diffuse_color)
            }
        }

        # Get camera object info (position + cfg spawn parameters)
        camera_pos = self._camera_obj.data.root_pos_w[env_id].cpu().numpy(
        ).tolist()
        camera_quat = self._camera_obj.data.root_quat_w[env_id].cpu().numpy(
        ).tolist()
        camera_spawn_cfg = {
            "rigid_props": {
                "disable_gravity":
                bool(self.cfg.camera_obj.spawn.rigid_props.disable_gravity)
            },
            "mass_props": {
                "mass": float(self.cfg.camera_obj.spawn.mass_props.mass)
            },
            "visual_material": {
                "diffuse_color":
                list(self.cfg.camera_obj.spawn.visual_material.diffuse_color)
            }
        }

        # Get agent info (position + cfg spawn parameters)
        agent_pos = self._agent.data.root_pos_w[env_id].cpu().numpy().tolist()
        agent_quat = self._agent.data.root_quat_w[env_id].cpu().numpy().tolist(
        )
        agent_spawn_cfg = {
            "size": list(self.cfg.agent.spawn.size),
            "rigid_props": {
                "disable_gravity":
                bool(self.cfg.agent.spawn.rigid_props.disable_gravity)
            },
            "mass_props": {
                "mass": float(self.cfg.agent.spawn.mass_props.mass)
            },
            "visual_material": {
                "diffuse_color":
                list(self.cfg.agent.spawn.visual_material.diffuse_color)
            }
        }

        # ========== CHANGE #1: Get active VPT indices ==========
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        active_indices = self.active_vpt_indices[env_id_item]
        active_indices_list = active_indices.cpu().numpy().tolist()
        # =======================================================

        # ========== CHANGE #2: Get VPT objects info - ONLY ACTIVE ONES ==========
        vpt_positions = self._vpt_objects.data.object_pos_w[env_id].cpu(
        ).numpy().tolist()
        vpt_orientations = self._vpt_objects.data.object_quat_w[env_id].cpu(
        ).numpy().tolist()
        # =========================================================================

        # Get valid viewpoint poses
        valid_viewpoints = []
        if (self.valid_viewpoint_poses is not None
                and env_id_item < len(self.valid_viewpoint_poses)
                and self.valid_viewpoint_poses[env_id_item] is not None):
            valid_viewpoints = self.valid_viewpoint_poses[env_id_item].cpu(
            ).numpy().tolist()

        # Build configuration dictionary
        config = {
            "metadata": {
                "env_id": env_id_item,
                "folder_idx": folder_idx,
                "visibility_label": label,
                "visibility_reason": reason,
                "cfg_version": "1.0"
            },
            "environment_settings": {
                "boundary_limits": list(self.cfg.boundary_limits),
                "agent_height": float(self.cfg.agent_height),
                "agent_camera_pitch": float(self.cfg.agent_camera_pitch),
                "action_scale": float(self.cfg.action_scale),
                "num_vpt_objs": int(self.cfg.num_vpt_objs)
            },
            "goal_ball": {
                "position": goal_pos,
                "orientation": goal_quat,
                "spawn_cfg": goal_spawn_cfg
            },
            "camera_object": {
                "position": camera_pos,
                "orientation": camera_quat,
                "spawn_cfg": camera_spawn_cfg
            },
            "agent": {
                "position": agent_pos,
                "orientation": agent_quat,
                "spawn_cfg": agent_spawn_cfg
            },
            # ========== CHANGE #1: Add active VPT metadata ==========
            "vpt_objects": {
                "total_count": self.num_objs,
                "active_count": self.active_vpt_objs,
                "active_indices": active_indices_list,
                "objects": []
            },
            # ========================================================
            "valid_viewpoints": {
                "count": len(valid_viewpoints),
                "positions": valid_viewpoints
            },
            "collected_viewpoints": {
                "count":
                len(self.selected_viewpoints_for_collection[env_id_item])
                if self.selected_viewpoints_for_collection[env_id_item]
                is not None else 0,
                "positions":
                self.selected_viewpoints_for_collection[env_id_item].cpu(
                ).numpy().tolist()
                if self.selected_viewpoints_for_collection[env_id_item]
                is not None else []
            }
        }

        # ========== CHANGE #2: Add only active VPT objects ==========
        # Add each ACTIVE VPT object with cfg metadata extracted at runtime
        for local_idx, obj_idx in enumerate(active_indices):
            obj_idx_item = obj_idx.item() if torch.is_tensor(
                obj_idx) else obj_idx

            # Extract spawn cfg directly from the VPT object's configuration
            vpt_spawn_cfg = self.cfg.vpt_objects.rigid_objects[list(
                self.cfg.vpt_objects.rigid_objects.keys())[obj_idx_item]].spawn

            # Get rigid props
            rigid_props = {}
            if hasattr(vpt_spawn_cfg, 'rigid_props'):
                rigid_props['disable_gravity'] = bool(
                    vpt_spawn_cfg.rigid_props.disable_gravity)

            # Get mass props
            mass_props = {}
            if hasattr(vpt_spawn_cfg, 'mass_props'):
                mass_props['mass'] = float(vpt_spawn_cfg.mass_props.mass)

            # Get visual material
            visual_material = {}
            if hasattr(vpt_spawn_cfg, 'visual_material'):
                visual_material['diffuse_color'] = list(
                    vpt_spawn_cfg.visual_material.diffuse_color)

            # Get size/dimensions based on spawn type
            size_info = {}
            if hasattr(vpt_spawn_cfg, 'size'):
                size_info['size'] = list(vpt_spawn_cfg.size)
            elif hasattr(vpt_spawn_cfg, 'radius'):
                size_info['radius'] = float(vpt_spawn_cfg.radius)
            elif hasattr(vpt_spawn_cfg, 'height') and hasattr(
                    vpt_spawn_cfg, 'radius'):
                size_info['height'] = float(vpt_spawn_cfg.height)
                size_info['radius'] = float(vpt_spawn_cfg.radius)

            vpt_obj = {
                "index": obj_idx_item,
                "position": vpt_positions[obj_idx_item],
                "orientation": vpt_orientations[obj_idx_item],
                "spawn_cfg": {
                    **size_info, "rigid_props": rigid_props,
                    "mass_props": mass_props,
                    "visual_material": visual_material
                }
            }
            config["vpt_objects"]["objects"].append(vpt_obj)
        # ============================================================

        # Create config directory if it doesn't exist
        config_dir = f"{self.base_path}/configs"
        os.makedirs(config_dir, exist_ok=True)

        # Save to JSON file
        config_filepath = f"{config_dir}/env_{folder_idx}_config.json"
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)

        # if self.verbose >= 2:
        # print(f"  💾 Saved config: {config_filepath}")
        # print(
        #     f"     Active VPT objects: {self.active_vpt_objs}/{self.num_objs}"
        # )

    def _select_active_vpt_indices(self, env_ids: torch.Tensor) -> None:
        """Randomly select 20 active VPT object indices for each environment.
        
        Args:
            env_ids: Tensor of environment IDs to select active indices for
        """
        device = self._agent.device

        for env_id in env_ids:
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id

            # Randomly select 20 indices from 0-199
            active_indices = torch.randperm(
                self.num_objs, device=device)[:self.active_vpt_objs]

            # Store for this environment
            self.active_vpt_indices[env_id_item] = active_indices

    def _store_inactive_vpt_objects(
            self, env_ids: torch.Tensor,
            vpt_obj_default_state: torch.Tensor) -> torch.Tensor:
        """Move inactive VPT objects (180 out of 200) to storage position far away.
        
        Args:
            env_ids: Tensor of environment IDs to store inactive objects for
            vpt_obj_default_state: The default state tensor to modify (shape: [num_envs, num_objs, state_dim])
            
        Returns:
            Modified vpt_obj_default_state tensor
        """
        device = self._agent.device

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id

            # Get active indices for this env
            active_indices = self.active_vpt_indices[env_id_item]

            # Create set of inactive indices (all indices not in active)
            all_indices = set(range(self.num_objs))
            active_indices_set = set(active_indices.cpu().numpy().tolist())
            inactive_indices = torch.tensor(list(all_indices -
                                                 active_indices_set),
                                            dtype=torch.long,
                                            device=device)

            # Move all inactive objects to storage position
            for inactive_idx in inactive_indices:
                vpt_obj_default_state[i, inactive_idx,
                                      0] = self.storage_position[0]
                vpt_obj_default_state[i, inactive_idx,
                                      1] = self.storage_position[1]
                vpt_obj_default_state[i, inactive_idx,
                                      2] = self.storage_position[2]

                # Zero out velocities
                vpt_obj_default_state[i, inactive_idx, 7:13] = 0.0

        return vpt_obj_default_state

    def _get_active_vpt_dims(self, env_ids) -> torch.Tensor:
        """Get dimensions of active VPT objects for given environment(s)."""
        # 1. Standardize input to Tensor
        if isinstance(env_ids, int):
            env_ids = torch.tensor([env_ids], device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device)

        # Ensure 1D [batch_size]
        if env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)
        elif env_ids.ndim > 1:
            env_ids = env_ids.flatten()

        # 2. Handle List vs Tensor for active_vpt_indices lookup
        if isinstance(self.active_vpt_indices, list):
            if len(self.active_vpt_indices) > 0 and isinstance(
                    self.active_vpt_indices[0], torch.Tensor):
                active_indices_tensor = torch.stack(
                    self.active_vpt_indices).to(self.device)
            else:
                active_indices_tensor = torch.tensor(self.active_vpt_indices,
                                                     device=self.device,
                                                     dtype=torch.long)
        else:
            active_indices_tensor = self.active_vpt_indices

        # 3. Get Active Indices
        # active_indices_tensor: [num_envs, max_objs]
        batch_indices = active_indices_tensor[env_ids]  # [batch, active_objs]

        # 4. Lookup Dimensions
        # Expand env_ids to [batch, active_objs]
        env_ids_expanded = env_ids.unsqueeze(1).expand_as(batch_indices)

        # Fetch dims: [batch, active_objs, 3]
        batch_dims = self.all_vpt_dims[env_ids_expanded, batch_indices, :]

        return batch_dims

    def _get_active_vpt_positions(
            self,
            env_ids,
            base_pivoted: bool = False,
            return_full_pose: bool = False) -> torch.Tensor:
        """
        Get positions (and optionally orientation) of only the active VPT objects for given environment(s).
        
        Args:
            env_ids: Tensor or int of environment indices.
            base_pivoted (bool): If True, adjusts Z to be at the base of the object.
            return_full_pose (bool): If True, returns [x, y, z, w, x, y, z] (7 dims). 
                                    If False, returns [x, y, z] (3 dims).
        """

        # --- 1. Ensure env_ids is a GPU Tensor ---
        if torch.is_tensor(env_ids):
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids = torch.tensor(env_ids,
                                   device=self.device,
                                   dtype=torch.long)

        if env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)

        # --- 2. Standardize active_vpt_indices to Tensor ---
        if isinstance(self.active_vpt_indices, list):
            if len(self.active_vpt_indices) > 0 and isinstance(
                    self.active_vpt_indices[0], torch.Tensor):
                active_indices_full = torch.stack(self.active_vpt_indices).to(
                    self.device)
            else:
                active_indices_full = torch.tensor(self.active_vpt_indices,
                                                   device=self.device,
                                                   dtype=torch.long)
        else:
            active_indices_full = self.active_vpt_indices.to(self.device)

        # --- 3. Get active indices for the requested batch ---
        # Shape: [batch_size, num_active_per_env]
        batch_active_indices = active_indices_full[env_ids]

        # --- 4. Advanced Indexing Setup ---
        # Expand env_ids: [batch_size, 1] -> [batch_size, num_active_per_env]
        env_ids_expanded = env_ids.unsqueeze(1).expand_as(batch_active_indices)

        # --- 5. Fetch Positions ---
        # CRITICAL: Use .clone() to avoid modifying the simulation state in place
        active_positions = self._vpt_objects.data.object_pos_w[
            env_ids_expanded, batch_active_indices].clone()
        # print(f"From Get active VPT positions: {active_positions[:, :, 2]}")

        # Apply Base Pivot Adjustment (modifies Z)
        if base_pivoted:
            # Get dimensions: [batch_size, num_active_per_env]
            active_heights = self.all_vpt_dims[env_ids_expanded,
                                               batch_active_indices, 2]

            # [FIXED] Get ratios using 2D Indexing: [batch_size, num_active_per_env]
            # Since self.vpt_z_offset_ratios is [num_envs, num_objs]
            active_ratios = self.vpt_z_offset_ratios[env_ids_expanded,
                                                     batch_active_indices]

            z_adjustment = active_heights * active_ratios

            # Subtract adjustment to go from COM -> Floor
            active_positions[:, :, 2] -= z_adjustment

        # --- 6. Return Logic ---
        if return_full_pose:
            # Fetch Quaternions: [batch, num_active, 4]
            active_quats = self._vpt_objects.data.object_quat_w[
                env_ids_expanded, batch_active_indices].clone()

            # Concatenate: [batch, num_active, 7]
            return torch.cat([active_positions, active_quats], dim=-1)

        return active_positions

    def _select_viewpoints_for_collection(self, env_id: int) -> bool:
        """Select viewpoints for a single environment slot.
        
        Args:
            env_id: Environment slot index (0-7)
            
        Returns:
            True if selection successful, False otherwise
        """
        if (self.valid_viewpoint_poses is None
                or env_id >= len(self.valid_viewpoint_poses)
                or self.valid_viewpoint_poses[env_id] is None or len(
                    self.valid_viewpoint_poses[env_id]) < self.images_per_env):
            return False

        all_viewpoints = self.valid_viewpoint_poses[env_id]
        selected_points = [all_viewpoints[0]]

        for point_idx in range(1, len(all_viewpoints)):
            candidate = all_viewpoints[point_idx]
            distances = torch.norm(torch.stack(selected_points) -
                                   candidate.unsqueeze(0),
                                   dim=1)

            if torch.all(distances >= self.min_viewpoint_distance):
                selected_points.append(candidate)

                if len(selected_points) == self.images_per_env:
                    break

        if len(selected_points) == self.images_per_env:
            self.selected_viewpoints_for_collection[env_id] = torch.stack(
                selected_points)
            # if self.verbose >= 2:
            #     print(
            #         f"    ✅ Slot {env_id}: Selected {self.images_per_env} viewpoints for collection"
            #     )
            return True
        else:
            # if self.verbose >= 2:
            #     print(
            #         f"    ⚠️  Slot {env_id}: Only {len(selected_points)} viewpoints available (need {self.images_per_env})"
            #     )
            return False

    def _reset_idx(self,
                   env_ids: Sequence[int] | None,
                   rl_reset: bool = True) -> None:
        """
        Reset environments. 
        If rl_reset=True: Only performs physics reset and randomization.
        If rl_reset=False: Performs reset, validation, data collection, and slot replenishment.
        """
        MIN_VALID_VIEWPOINTS = self.images_per_env

        if env_ids is None:
            env_ids = self._agent._ALL_INDICES
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids,
                                   dtype=torch.long,
                                   device=self.device)

        # --- 1. Lazy Initialization of Slot State (Run Once) ---
        if not hasattr(self, "slot_folder_indices"):
            num_all_envs = self.num_envs
            self.slot_folder_indices = [
                self.next_env_folder_idx + i for i in range(num_all_envs)
            ]
            self.slot_visibility_categories = []
            self.slot_attempt_counts = [0] * num_all_envs

            if self.verbose >= 1:
                print("🔒 Initializing slot states and visibility labels...")
            for i in range(num_all_envs):
                cat = self._assign_next_visibility_label(
                    self.slot_folder_indices[i])
                self.slot_visibility_categories.append(cat)

            # self._save_visibility_labels()

        # --- 2. Randomize Scene (Procedural Generation) ---
        self._cache_base_dims()
        self._randomize_scene_props(env_ids)

        # Randomly select 25% of envs to have the goal on top of an object
        num_to_select = max(1, int(0.25 * len(env_ids)))
        selected_indices = torch.randperm(len(env_ids),
                                          device=self.device)[:num_to_select]
        self.envs_to_move_ball = env_ids[selected_indices]

        # --- 3. Prepare Batch Data for Reset ---
        reset_folder_indices = []
        reset_visibility_categories = []
        active_slots = env_ids.tolist()

        for slot_idx in active_slots:
            reset_folder_indices.append(self.slot_folder_indices[slot_idx])
            reset_visibility_categories.append(
                self.slot_visibility_categories[slot_idx])

        # --- 4. Physics Reset (Teleport Agents/Objects) ---
        if self.verbose >= 1:
            print(
                f"🔄 Resetting {len(active_slots)} environments (RL Reset: {rl_reset})..."
            )

        self._reset_idx_internal(
            env_ids,
            rl_reset=rl_reset,
            folder_indices=reset_folder_indices,
            visibility_categories=reset_visibility_categories)

        # Step simulation to let physics settle
        for _ in range(1):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.step_dt)

        # --- EARLY EXIT FOR RL TRAINING ---
        if rl_reset:
            self._reset_called = True
            return

        # ==========================================================
        #  DATA COLLECTION PIPELINE (Only runs if rl_reset=False)
        # ==========================================================

        # --- 5. Validate & Collect ---
        valid_slots = []
        failed_slots = []
        exceeded_slots = []

        for slot_idx in active_slots:
            env_id = self.slot_to_env_id[slot_idx]

            # Skip if already completed
            if env_id in self.completed_envs:
                continue

            folder_idx = self.slot_folder_indices[slot_idx]

            is_valid, reason = self._validate_env_state(
                torch.tensor([slot_idx], device=self.device), folder_idx,
                MIN_VALID_VIEWPOINTS)

            if is_valid:
                valid_slots.append(slot_idx)
                # if self.verbose >= 1:
                #     print(f"  ✅ Slot {slot_idx} | Env {env_id} VALIDATED")
            else:
                self.slot_attempt_counts[slot_idx] += 1

                if self.slot_attempt_counts[
                        slot_idx] >= self.max_attempts_per_slot:
                    exceeded_slots.append(slot_idx)
                    if self.verbose >= 1:
                        print(
                            f"  ⚠️ Slot {slot_idx} | Env {env_id} EXCEEDED max attempts, skipping."
                        )
                else:
                    failed_slots.append(slot_idx)
                    if self.verbose >= 2:
                        print(
                            f"  ❌ Slot {slot_idx} | Env {env_id} Attempt {self.slot_attempt_counts[slot_idx]}: {reason}"
                        )

        # --- 6. Data Collection (Only for Valid Slots) ---
        if valid_slots:
            # if self.verbose >= 1:
            #     print(f"📸 Collecting images for {len(valid_slots)} slots...")

            for slot_idx in valid_slots:
                env_id = self.slot_to_env_id[slot_idx]
                folder_idx = self.slot_folder_indices[slot_idx]

                if self._select_viewpoints_for_collection(slot_idx):
                    self._collect_images_for_slot(
                        torch.tensor([slot_idx], device=self.device),
                        folder_idx)
                    self.completed_envs.add(env_id)

        # --- 7. Replenish / Advance Slots ---
        slots_to_replace = valid_slots + exceeded_slots

        for slot_idx in slots_to_replace:
            if self.next_env_id < self.total_envs_to_sim:
                old_env_id = self.slot_to_env_id[slot_idx]

                new_env_id = self.next_env_id
                self.next_env_id += 1
                new_folder_idx = self.next_env_folder_idx + new_env_id

                self.slot_to_env_id[slot_idx] = new_env_id
                self.slot_folder_indices[slot_idx] = new_folder_idx
                self.slot_attempt_counts[slot_idx] = 0

                new_visibility = self._assign_next_visibility_label(
                    new_folder_idx)
                self.slot_visibility_categories[slot_idx] = new_visibility

                if self.verbose >= 1:
                    print(
                        f"  🔄 Slot {slot_idx}: Replaced {old_env_id} -> {new_env_id}"
                    )

        if slots_to_replace:
            self._save_visibility_labels()

        # if len(self.completed_envs) >= self.total_envs_to_sim:
        #     print(f"\n🎉 SUCCESS: Completed all {self.total_envs_to_sim} environments!")
        #     sys.exit(0) # Or handle completion gracefully

        self._reset_called = True

    def _randomize_scene_props(self, env_ids):
        """Helper to handle all the randomization logic for specific environments."""

        # Ensure we have a standard list of integers
        if isinstance(env_ids, torch.Tensor):
            target_ids = env_ids.tolist()
        else:
            target_ids = list(env_ids)

        # 1. Rescale VPT Objects
        vpt_prim_paths = []
        for env_id in target_ids:
            for idx in range(self.cfg.num_vpt_objs):
                vpt_prim_paths.append(f"/World/envs/env_{env_id}/obs_{idx}")

        # if vpt_prim_paths:
        #     self.randomize_shape_scale(prim_path_expr=vpt_prim_paths,
        #                                is_random=True)

        self.apply_randomizations(env_ids=target_ids,
                                  randomize_scale=True,
                                  randomize_color=False,
                                  randomize_light=True)


        # 2. Randomize Materials (Unified Calls)
        # A. Floor Materials ("mat")
        floor_paths = [f"/World/envs/env_{i}/mat" for i in target_ids]
        if floor_paths:
            self.randomize_material(prim_paths=floor_paths,
                                    material_type="mat")

        # B. VPT Materials ("vpt")
        if vpt_prim_paths:
            self.randomize_material(prim_paths=vpt_prim_paths,
                                    material_type="vpt")

        # SPHERICAL LIGHTS HERE
        # light_paths = []
        # for env_id in target_ids:
        #     light_paths.extend([
        #         f"/World/envs/env_{env_id}/Light_A",
        #         # f"/World/envs/env_{env_id}/Light_B",
        #     ])
        # self.randomize_spherical_lights(prim_paths=light_paths, )

    def initial_spawn_loop(self,
                           env_ids,
                           envs_need_spawn_retry,
                           safe_range: float,
                           states,
                           allow_clipping: bool = False,
                           device=None):
        import math
        import random
        import torch
        # from shapely.geometry import Point, box
        # from shapely import affinity

        if device is None:
            device = self._agent.device

        # --- Unpack States (These are LOCAL subsets, size = len(env_ids)) ---
        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        # --- Setup Retry Batching ---
        retry_mask = envs_need_spawn_retry.clone()
        retry_indices = torch.where(retry_mask)[0]
        global_retry_env_ids = env_ids[retry_indices]
        batch_size = retry_indices.numel()

        if batch_size == 0:
            return envs_need_spawn_retry, states

        safe_x_range = safe_range - 4.0
        safe_x_range_obstacles = float(safe_range - 3.0)

        # Subset of origins
        env_origins = self.scene.env_origins[global_retry_env_ids]

        # --- 1. INITIAL SAMPLING (Goal, Camera, Agent) ---
        goal_offsets = sample_uniform(-safe_x_range, safe_x_range,
                                      (batch_size, 2), device)
        camera_offsets = sample_uniform(-safe_x_range, safe_x_range,
                                        (batch_size, 2), device)
        agent_offsets = sample_uniform(-safe_x_range, safe_x_range,
                                       (batch_size, 2), device)
        goal_perturb_offsets = sample_uniform(-2, 2, (batch_size, 2), device)

        # Apply initial positions
        goal_default_state[retry_indices,
                           0] = env_origins[:, 0] + goal_offsets[:, 0]
        goal_default_state[retry_indices,
                           1] = env_origins[:, 1] + goal_offsets[:, 1]
        goal_default_state[retry_indices, 2] = env_origins[:, 2]

        camera_obj_default_state[retry_indices,
                                 0] = env_origins[:, 0] + camera_offsets[:, 0]
        camera_obj_default_state[retry_indices,
                                 1] = env_origins[:, 1] + camera_offsets[:, 1]

        # --- 2. ENFORCE CAMERA-GOAL DISTANCE (>= 4.0) ---
        max_dist_retries = 20
        for _ in range(max_dist_retries):
            cam_pos_subset = camera_obj_default_state[retry_indices, :2]
            goal_pos_subset = goal_default_state[retry_indices, :2]
            dists = torch.norm(cam_pos_subset - goal_pos_subset, dim=1)

            bad_mask = (dists < 3.0) | (dists > 18.0)
            if not bad_mask.any():
                break

            num_bad = bad_mask.sum().item()
            bad_sub_indices = torch.where(bad_mask)[0]
            bad_local_indices = retry_indices[bad_sub_indices]

            new_goal_offsets = sample_uniform(-safe_x_range, safe_x_range,
                                              (num_bad, 2), device)
            new_cam_offsets = sample_uniform(-safe_x_range, safe_x_range,
                                             (num_bad, 2), device)

            current_origins = env_origins[bad_sub_indices]

            goal_default_state[bad_local_indices,
                               0] = current_origins[:, 0] + new_goal_offsets[:,
                                                                             0]
            goal_default_state[bad_local_indices,
                               1] = current_origins[:, 1] + new_goal_offsets[:,
                                                                             1]

            camera_obj_default_state[
                bad_local_indices,
                0] = current_origins[:, 0] + new_cam_offsets[:, 0]
            camera_obj_default_state[
                bad_local_indices,
                1] = current_origins[:, 1] + new_cam_offsets[:, 1]

        # --- 3. ORIENTATION & FINAL SETUP ---
        direction_to_goal = goal_default_state[
            retry_indices, :2] - camera_obj_default_state[retry_indices, :2]
        yaw = torch.atan2(direction_to_goal[:, 1],
                          direction_to_goal[:, 0]) - math.radians(90)
        roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
        zero = torch.zeros_like(yaw)
        quaternion = quat_from_euler_xyz(roll, zero, yaw)
        camera_obj_default_state[retry_indices, 3:7] = quaternion

        goal_default_state[retry_indices, 0] += goal_perturb_offsets[:, 0]
        goal_default_state[retry_indices, 1] += goal_perturb_offsets[:, 1]

        agent_default_state[retry_indices,
                            0] = env_origins[:, 0] + agent_offsets[:, 0]
        agent_default_state[retry_indices,
                            1] = env_origins[:, 1] + agent_offsets[:, 1]

        # --- 4. ROBUST VPT LOOP ---
        # def create_rotated_rect(x, y, w, l, yaw_rad):
        #     poly = box(-w / 2.0, -l / 2.0, w / 2.0, l / 2.0)
        #     poly = affinity.rotate(poly, yaw_rad, use_radians=True)
        #     poly = affinity.translate(poly, x, y)
        #     return poly

        MARGIN = 0.1
        MAX_ATTEMPTS = 50
        NUM_CANDIDATES = 20

        for batch_idx, local_idx in enumerate(retry_indices):

            global_env_id = env_ids[local_idx]
            global_env_id_item = global_env_id.item() if torch.is_tensor(
                global_env_id) else global_env_id

            # Global & Local setup
            cam_global_x = camera_obj_default_state[local_idx, 0].item()
            cam_global_y = camera_obj_default_state[local_idx, 1].item()
            goal_global_x = goal_default_state[local_idx, 0].item()
            goal_global_y = goal_default_state[local_idx, 1].item()

            origin_x = env_origins[batch_idx, 0].item()
            origin_y = env_origins[batch_idx, 1].item()

            if not allow_clipping:
                cam_local_p = Point(cam_global_x - origin_x,
                                    cam_global_y - origin_y)
                goal_local_p = Point(goal_global_x - origin_x,
                                    goal_global_y - origin_y)

            active_indices = self.active_vpt_indices[global_env_id_item]
            active_dims = self.all_vpt_dims[global_env_id, active_indices, :3]

            placed_polys = []
            placement_failed = False

            for k, obj_idx in enumerate(active_indices):
                obj_w = active_dims[k, 0].item()
                obj_l = active_dims[k, 1].item()

                coll_w = obj_w + MARGIN
                coll_l = obj_l + MARGIN

                found = False

                for _ in range(MAX_ATTEMPTS):

                    if allow_clipping:
                        # --- FAST PATH: Random Placement (Clipping Allowed) ---
                        # Skip Shapely, skip overlap checks, skip best candidate selection.
                        rx = (random.random() * 2 *
                              safe_x_range_obstacles) - safe_x_range_obstacles
                        ry = (random.random() * 2 *
                              safe_x_range_obstacles) - safe_x_range_obstacles
                        r_yaw = random.random() * 2 * math.pi

                        # Analytic Distance Checks (Camera: 4.0, Goal: radius + 0.1)
                        dx_cam = rx - (cam_global_x - origin_x)
                        dy_cam = ry - (cam_global_y - origin_y)
                        dist_cam = math.sqrt(dx_cam**2 + dy_cam**2)

                        dx_goal = rx - (goal_global_x - origin_x)
                        dy_goal = ry - (goal_global_y - origin_y)
                        dist_goal = math.sqrt(dx_goal**2 + dy_goal**2)

                        if dist_cam < 5.0:
                            # print(f"dist cam = {dist_cam}")
                            continue
                        
                        if dist_goal < (self.goal_radius + 3.0 + 0.05):
                            # print(f"dist goal = {dist_goal}")
                            continue
                        

                        # --- Assign and Break ---
                        cand_global_x = origin_x + rx
                        cand_global_y = origin_y + ry
                        vpt_obj_default_state[local_idx, obj_idx,
                                              0] = cand_global_x
                        vpt_obj_default_state[local_idx, obj_idx,
                                              1] = cand_global_y

                        r_yaw_tensor = torch.tensor(r_yaw, device=device)
                        zero_t = torch.tensor(0.0, device=device)
                        quat = quat_from_euler_xyz(zero_t, zero_t,
                                                   r_yaw_tensor)
                        vpt_obj_default_state[local_idx, obj_idx, 3:7] = quat

                        found = True
                        break

                    else:
                        # --- ROBUST PATH: No Clipping (Shapely + Best Candidate) ---
                        # --- A. BEST CANDIDATE SELECTION ---
                        best_candidate = None
                        max_isolation_dist = -1.0

                        candidates = []
                        for _ in range(NUM_CANDIDATES):
                            raw_rx = (random.random() * 2 *
                                      safe_x_range_obstacles
                                      ) - safe_x_range_obstacles
                            raw_ry = (random.random() * 2 *
                                      safe_x_range_obstacles
                                      ) - safe_x_range_obstacles
                            r_yaw = random.random() * 2 * math.pi
                            candidates.append(
                                (float(raw_rx), float(raw_ry), float(r_yaw)))

                        for cand in candidates:
                            cand_p = Point(cand[0], cand[1])
                            d_cam = cand_p.distance(cam_local_p)
                            d_goal = cand_p.distance(goal_local_p)
                            current_min_dist = min(d_cam, d_goal)

                            for poly in placed_polys:
                                d_obj = poly.distance(cand_p)
                                if d_obj < current_min_dist:
                                    current_min_dist = d_obj

                            if current_min_dist > max_isolation_dist:
                                max_isolation_dist = current_min_dist
                                best_candidate = cand

                        # --- B. VALIDATE WINNER ---
                        rx, ry, r_yaw = best_candidate
                        collision_poly = create_rotated_rect(
                            rx, ry, coll_w, coll_l, r_yaw)

                        if collision_poly.distance(cam_local_p) < 4.0:
                            # print(f"Cam to obj dist: {collision_poly.distance(cam_local_p)}")
                            continue
                        
                        if collision_poly.distance(goal_local_p) < (
                                self.goal_radius + 0.1):
                            # print(f"Goal to obj dist: {collision_poly.distance(goal_local_p)}")
                            continue
                        

                        minx, miny, maxx, maxy = collision_poly.bounds
                        if (minx < -self.center_to_boundary
                                or miny < -self.center_to_boundary
                                or maxx > self.center_to_boundary
                                or maxy > self.center_to_boundary):
                            # print("out of bounds")
                            continue

                        overlap = False
                        for other_poly in placed_polys:
                            if collision_poly.intersects(other_poly):
                                overlap = True
                                # print("----------Overlap----------")
                                break

                        if not overlap:
                            placed_polys.append(collision_poly)

                            cand_global_x = origin_x + rx
                            cand_global_y = origin_y + ry

                            vpt_obj_default_state[local_idx, obj_idx,
                                                  0] = cand_global_x
                            vpt_obj_default_state[local_idx, obj_idx,
                                                  1] = cand_global_y

                            r_yaw_tensor = torch.tensor(r_yaw, device=device)
                            zero_t = torch.tensor(0.0, device=device)
                            quat = quat_from_euler_xyz(zero_t, zero_t,
                                                       r_yaw_tensor)
                            vpt_obj_default_state[local_idx, obj_idx,
                                                  3:7] = quat

                            found = True
                            break

                if not found:
                    placement_failed = True
                    break

            # --- VERBOSITY CHECK ---
            if placement_failed:
                continue
            else:
                envs_need_spawn_retry[local_idx] = False
        # print(f"Envs need respawn - {envs_need_spawn_retry}")
        vpt_obj_default_state[
            retry_indices] = self._store_inactive_vpt_objects(
                env_ids[retry_indices], vpt_obj_default_state[retry_indices])

        #make a 2D occupancy map for visualization/debugging using the positions of the camera, goal, and VPT objects
        occupancy_map = np.zeros((512, 512), dtype=np.uint8)
        scale = 10.0  # Scale factor to convert world coordinates to map coordinates
        # import pdb; pdb.set_trace()


        return envs_need_spawn_retry, [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state
        ]

    def moving_ball_loop(self,
                         env_ids,
                         moved_vpt_for_ball,
                         move_ball_indices,
                         states,
                         safe_range: float,
                         device=None):

        if device is None:
            device = self._agent.device

        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        safe_x_range_obstacles = safe_range - 3.5

        if len(move_ball_indices) == 0:
            return moved_vpt_for_ball, states

        # --- 1. PREPARE BATCH DATA ---
        target_env_ids = env_ids[move_ball_indices]
        batch_size = len(target_env_ids)

        if isinstance(self.active_vpt_indices, torch.Tensor):
            active_indices_tensor = self.active_vpt_indices
        elif isinstance(self.active_vpt_indices, list):
            if len(self.active_vpt_indices) > 0 and isinstance(
                    self.active_vpt_indices[0], torch.Tensor):
                active_indices_tensor = torch.stack(
                    self.active_vpt_indices).to(device)
            else:
                active_indices_tensor = torch.tensor(self.active_vpt_indices,
                                                     device=device,
                                                     dtype=torch.long)
        else:
            active_indices_tensor = self.active_vpt_indices

        batch_active_indices = active_indices_tensor[target_env_ids]

        # Get Dimensions: [batch_size, active_vpt_objs, 3]
        batch_dims = self._get_active_vpt_dims(target_env_ids)
        # STRICT: Use True Height from USD Bounding Box
        batch_heights = batch_dims[:, :, 2]

        # --- 2. BUILD CANDIDATE MASK ---
        height_mask = batch_heights < 0.75
        shape_mask = self.valid_shape_mask[batch_active_indices]
        candidate_mask = height_mask & shape_mask
        has_candidates_mask = candidate_mask.any(dim=1)

        if not has_candidates_mask.all():
            skipped_count = (~has_candidates_mask).sum().item()
            # if skipped_count > 0:
            #     print(
            #         f"  ❌ Skipping ball move for {skipped_count} envs (no valid objects found)."
            #     )

            target_env_ids = target_env_ids[has_candidates_mask]
            move_ball_indices = move_ball_indices[has_candidates_mask]
            batch_active_indices = batch_active_indices[has_candidates_mask]
            batch_heights = batch_heights[has_candidates_mask]
            candidate_mask = candidate_mask[has_candidates_mask]
            batch_size = len(target_env_ids)

        if batch_size == 0:
            return moved_vpt_for_ball, states

        # --- 3. SELECT OBJECTS ---
        weights = candidate_mask.float() + 1e-6
        selected_local_indices = torch.multinomial(weights, 1).squeeze(-1)

        selected_global_indices = torch.gather(
            batch_active_indices, 1,
            selected_local_indices.unsqueeze(1)).squeeze(-1)

        selected_heights = torch.gather(
            batch_heights, 1, selected_local_indices.unsqueeze(1)).squeeze(-1)

        # --- 4. UPDATE POSES ---
        target_goal_pos = goal_default_state[move_ball_indices, :3]
        target_env_origins = self.scene.env_origins[target_env_ids]

        # [REMOVED] Z-calculation for object. Handled by universal grounding loop.
        # [REMOVED] selected_ratios retrieval.

        # STRICT: Goal Z = Origin + True Height + Radius + Margin
        # (Goal is center-pivoted, so we add radius to sit on top)
        new_goal_z = target_env_origins[:,
                                        2] + selected_heights + self.goal_radius

        vpt_obj_default_state[move_ball_indices, selected_global_indices,
                              0] = target_goal_pos[:, 0]
        vpt_obj_default_state[move_ball_indices, selected_global_indices,
                              1] = target_goal_pos[:, 1]

        # [FIXED] Do NOT write Z here. Universal loop handles it.

        goal_default_state[move_ball_indices, 2] = new_goal_z

        move_ball_cpu = move_ball_indices.cpu().numpy()
        selected_global_cpu = selected_global_indices.cpu().numpy()

        # Loop to populate dict and print ID
        for i, env_idx in enumerate(move_ball_cpu):
            obj_id = selected_global_cpu[i]
            moved_vpt_for_ball[env_idx] = obj_id

            if self.verbose >= 2:
                # Find the global env ID corresponding to this local index
                global_env_id = env_ids[env_idx].item() if torch.is_tensor(
                    env_ids) else env_ids[env_idx]
                print(
                    f"  Env {global_env_id}: Mounting ball on Object ID {obj_id}"
                )

        # --- 5. CONFLICT RESOLUTION ---
        all_active_pos = vpt_obj_default_state[move_ball_indices.unsqueeze(1),
                                               batch_active_indices, :2]

        dists = torch.norm(all_active_pos -
                           target_goal_pos[:, :2].unsqueeze(1),
                           dim=2)
        selection_one_hot = torch.nn.functional.one_hot(
            selected_local_indices, num_classes=self.active_vpt_objs).bool()

        conflict_mask = (dists < 1.5) & (~selection_one_hot)

        if conflict_mask.any():
            num_conflicts = conflict_mask.sum()

            new_x = sample_uniform(-safe_x_range_obstacles,
                                   safe_x_range_obstacles, (num_conflicts, ),
                                   device)
            new_y = sample_uniform(-safe_x_range_obstacles,
                                   safe_x_range_obstacles, (num_conflicts, ),
                                   device)

            expanded_env_indices = move_ball_indices.unsqueeze(1).expand_as(
                conflict_mask)
            conflict_env_idxs = expanded_env_indices[conflict_mask]
            conflict_obj_idxs = batch_active_indices[conflict_mask]
            conflict_origins = self.scene.env_origins[
                env_ids[conflict_env_idxs]]

            vpt_obj_default_state[conflict_env_idxs, conflict_obj_idxs,
                                  0] = conflict_origins[:, 0] + new_x
            vpt_obj_default_state[conflict_env_idxs, conflict_obj_idxs,
                                  1] = conflict_origins[:, 1] + new_y

            if self.verbose >= 1:
                print(
                    f"  ⚠️ Resolved {num_conflicts} conflicts in ball movement batch."
                )

        return moved_vpt_for_ball, [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state
        ]

    def move_vpt_objects(self,
                         env_ids,
                         valid_indices,
                         visibility_categories,
                         moved_vpt_for_ball,
                         states,
                         in_view_displaced=None,
                         outside_fov_displaced=None,
                         device=None):
        """
        Unified function to displace VPT objects.
        - "occluded": Always moves.
        - "in_view": Moves ONLY if in `in_view_displaced`.
        - "outside_fov": Moves ONLY if in `outside_fov_displaced`.
        """
        if device is None:
            device = self._agent.device

        valid_env_ids = env_ids[valid_indices]

        # Unpack states: [goal, camera, agent, vpt]
        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        vpt_obj_default_state = states[3]

        camera_positions = camera_obj_default_state[valid_indices, :3]

        # Optimization: Pre-convert displacement lists to a Python Set for O(1) lookup
        indices_to_move = set()
        if in_view_displaced is not None:
            indices_to_move.update(in_view_displaced.tolist())
        if outside_fov_displaced is not None:
            indices_to_move.update(outside_fov_displaced.tolist())

        for local_idx, env_idx in enumerate(valid_indices):
            category = visibility_categories[env_idx]
            current_env_idx_val = env_idx.item()

            # --- 1. Filter Logic (The Gatekeeper) ---
            should_move = False

            if category == "occluded":
                should_move = True  # Always move occluded
            elif current_env_idx_val in indices_to_move:
                should_move = True  # Move if in the random subset

            if not should_move:
                continue

            # --- 2. Determine Range based on Category ---
            if category == "in_view":
                t_min, t_max = 0.3, 0.7
            else:
                t_min, t_max = 0.2, 0.8

            env_id_item = valid_env_ids[local_idx].item()
            camera_pos = camera_positions[local_idx]
            goal_pos = goal_default_state[env_idx, :3]

            # --- 3. Select Object (Exclude Ball-Moved Object) ---
            active_indices = self.active_vpt_indices[env_id_item]
            moved_idx = moved_vpt_for_ball[env_idx.item()]

            if moved_idx is not None:
                candidates = [
                    i for i in range(self.active_vpt_objs)
                    if active_indices[i].item() != moved_idx
                ]
                if not candidates:
                    continue
                random_local_idx = random.choice(candidates)
            else:
                random_local_idx = random.randint(0, self.active_vpt_objs - 1)

            random_obj_idx = active_indices[random_local_idx].item()

            # [UPDATED] Print logic for Dragging vs Conflict Avoidance
            if self.verbose >= 2:
                conflict_msg = ""
                if moved_idx is not None:
                    conflict_msg = f" (Avoided ball-mount Obj {moved_idx})"
                # print(f"  Env {env_id_item}: Dragging Object {random_obj_idx} for '{category}'{conflict_msg}")

            # --- 4. Calculate New Position (LERP + Jitter) ---
            direction_vec = goal_pos[:2] - camera_pos[:2]
            dist = torch.norm(direction_vec)

            if dist > 1e-6:
                direction_vec = direction_vec / dist
                t = random.uniform(t_min, t_max)
                new_pos_2d = camera_pos[:2] + direction_vec * (dist * t)
                random_offset = sample_uniform(-0.4, 0.4, (2, ), device=device)

                vpt_obj_default_state[env_idx, random_obj_idx,
                                      0] = new_pos_2d[0] + random_offset[0]
                vpt_obj_default_state[env_idx, random_obj_idx,
                                      1] = new_pos_2d[1] + random_offset[1]

        return states

    def outside_fov_camera_movement(self, valid_env_ids, valid_indices,
                                    visibility_categories, states, device):

        if device is None:
            device = self._agent.device

        goal_default_state = states[0]
        camera_obj_default_state = states[1]

        # 1. Identify "Outside FOV" cases
        current_categories = [
            visibility_categories[i] for i in valid_indices.cpu().tolist()
        ]
        outside_fov_mask = torch.tensor(
            [c == "outside_fov" for c in current_categories],
            device=device,
            dtype=torch.bool)

        # 2. Handle "Outside FOV" Logic
        if outside_fov_mask.any():
            # Get subset of indices
            outside_fov_global_idxs = valid_indices[outside_fov_mask]

            camera_pos_batch = camera_obj_default_state[
                outside_fov_global_idxs, :3]
            goal_pos_batch = goal_default_state[outside_fov_global_idxs, :3]

            # Calculate Look-Away Rotation
            # Safety: Add tiny epsilon to avoid 0,0 vector issues
            direction_to_goal = (goal_pos_batch[:, :2] -
                                 camera_pos_batch[:, :2]) + 1e-6

            yaw = torch.atan2(direction_to_goal[:, 1],
                              direction_to_goal[:, 0]) - math.radians(90)

            yaw_offset_magnitude = sample_uniform(
                math.radians(60),
                math.pi, (len(outside_fov_global_idxs), ),
                device=device)

            signs = torch.randint(0,
                                  2, (len(outside_fov_global_idxs), ),
                                  device=device).float() * 2 - 1

            yaw_away = yaw + (yaw_offset_magnitude * signs)
            roll = torch.full((len(outside_fov_global_idxs), ),
                              -math.radians(self.agent_camera_pitch),
                              device=device)
            zero = torch.zeros_like(roll)

            # Create quaternion
            quaternion_away = quat_from_euler_xyz(roll, zero, yaw_away)

            # Update State Tensor
            camera_obj_default_state[outside_fov_global_idxs,
                                     3:7] = quaternion_away

        # ==================== CRITICAL FIX ====================
        # 3. Sanitize Quaternions BEFORE Sim Step
        # Get all quaternions we are about to write
        subset_quats = camera_obj_default_state[valid_indices, 3:7]

        # A. Force Normalization (Fixes "Device-side assert" in PhysX)
        subset_quats = torch.nn.functional.normalize(subset_quats, p=2, dim=-1)

        # B. Check for NaNs (Nuclear Option replacement)
        nan_mask = torch.isnan(subset_quats).any(dim=1)
        if nan_mask.any():
            print(
                f"⚠️ FATAL: Found {nan_mask.sum()} NaN quaternions! Resetting to identity."
            )
            # Set to identity [1, 0, 0, 0] to prevent crash
            subset_quats[nan_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                                  device=device)

        # Write sanitized quats back to state
        camera_obj_default_state[valid_indices, 3:7] = subset_quats
        # ======================================================

        # 4. Write to Sim
        self._camera_obj.write_root_pose_to_sim(
            camera_obj_default_state[valid_indices, :7], valid_env_ids)

        # # 5. Update Sensor
        # camera_positions = camera_obj_default_state[valid_indices, :3]
        # camera_orientations = camera_obj_default_state[valid_indices, 3:7]

        # # Apply 90-degree offset for sensor
        # theta_left = math.pi / 2
        # half_theta_left = theta_left / 2
        # left_90_quat = torch.tensor(
        #     [math.cos(half_theta_left), 0.0, 0.0,
        #      math.sin(half_theta_left)],
        #     device=device)

        # rotated_orientations = math_utils.quat_mul(
        #     camera_orientations,
        #     left_90_quat.unsqueeze(0).expand(len(valid_env_ids), -1))

        # # Normalize sensor quats too
        # rotated_orientations = torch.nn.functional.normalize(
        #     rotated_orientations, p=2, dim=-1)

        # self._occlusion_camera.set_world_poses(
        #     positions=camera_positions,
        #     orientations=rotated_orientations,
        #     env_ids=valid_env_ids.tolist(),
        #     convention="world")

        # 6. Step Simulation
        for _ in range(1):
            self.sim.step()
            self._occlusion_camera.update(self.sim.cfg.dt)

    def check_z_bounds(self,
                       env_ids,
                       valid_indices,
                       states,
                       envs_need_spawn_retry,
                       tolerance=5e-2):
        """
        Verifies Z-heights and returns the updated retry mask.
        """
        updated_retry_mask = envs_need_spawn_retry.clone()
        goal_pos, camera_pos, agent_pos, vpt_pos = states
        for local_idx, env_idx in enumerate(valid_indices):
            env_id_val = env_ids[env_idx].item()
            failure_reasons = []
            # 1. Goal Check
            goal_z = goal_pos[local_idx, 2]
            if not (-tolerance <= goal_z <= 1.0 + tolerance):
                failure_reasons.append(f"Goal Z out of bounds: {goal_z:.6f}")
            # 2. Camera Check
            cam_z = camera_pos[local_idx, 2]
            if not (0.0 <= cam_z <= 1.0):
                failure_reasons.append(f"Camera Z out of bounds: {cam_z:.4f}")
            # 3. Agent Check
            agent_z = agent_pos[local_idx, 2]
            if not (0.0 <= agent_z <= 1.0):
                failure_reasons.append(f"Agent Z out of bounds: {agent_z:.4f}")
            # 4. VPT Check (Detailed)
            og_current_vpt_z = vpt_pos[
                local_idx, :,
                2]  # This is already filtered to active objects (size 16)
            env_active_indices = self.active_vpt_indices[env_id_val]
            z_offsets = self.vpt_z_offset_ratios[env_id_val,
                                                 env_active_indices]
            current_vpt_z = og_current_vpt_z * z_offsets
            # Create a mask for valid objects
            # Z must be >= -tolerance AND <= 0.1 + tolerance
            valid_obj_mask = (current_vpt_z >= -tolerance) & (
                current_vpt_z <= 0.1 + tolerance)
            if not torch.all(valid_obj_mask):
                # Find exactly which objects failed
                failed_obj_indices = torch.where(~valid_obj_mask)[0]
                # [UPDATED] Retrieve Global IDs
                env_active_indices = self.active_vpt_indices[env_id_val]
                for obj_idx in failed_obj_indices:
                    # Map active subset index -> Global Object ID
                    global_obj_id = env_active_indices[obj_idx].item()
                    bad_z = current_vpt_z[obj_idx].item()
                    failure_reasons.append(
                        f"VPT Object {global_obj_id} (Active #{obj_idx.item()}) Z out of bounds: {bad_z:.6f}"
                    )
            # Update the mask if failures found
            if failure_reasons:
                print(f"Env {env_id_val} Z-Check Failed:")
                for reason in failure_reasons:
                    print(f"   {reason}")
                updated_retry_mask[env_idx] = True
        return updated_retry_mask

    def occlusion_validation_check(self, final_valid_env_ids, valid_indices,
                                   visibility_categories,
                                   envs_need_spawn_retry, env_dict, states,
                                   device):
        if device is None:
            device = self._agent.device

        valid_env_ids = final_valid_env_ids
        goal_default_state = states[0]
        camera_obj_default_state = states[1]

        # Check Camera State for NaNs (Common cause of sim crashes)
        if torch.isnan(camera_obj_default_state).any():
            print("❌ FATAL: Camera state contains NaNs!")

        camera_positions = camera_obj_default_state[valid_indices, :3]

        occlusion_valid_mask = torch.ones(len(valid_indices),
                                          dtype=torch.bool,
                                          device=device)

        for local_idx, env_idx in enumerate(valid_indices):
            # [DEBUG] Barrier 2: Before accessing item()
            # If it crashes here, the PREVIOUS iteration's raycast killed it.
            # torch.cuda.synchronize()

            env_id = valid_env_ids[local_idx]
            env_id_item = env_id.item()

            visibility_category = visibility_categories[env_idx]
            camera_pos = camera_positions[local_idx]
            goal_pos = goal_default_state[env_idx, :3]

            if visibility_category in ["in_view", "occluded", "outside_fov"]:
                # [DEBUG] Barrier 3: Before Raycast
                # print(f"DEBUG: Checking raycast for Env {env_id_item}...")

                is_occluded = self._check_occlusion_raycast(
                    camera_pos, goal_pos, env_id)

                # [DEBUG] Barrier 4: After Raycast
                # If it crashes here, '_check_occlusion_raycast' is the killer.
                # torch.cuda.synchronize()

                expected_occluded = (visibility_category == "occluded"
                                     or visibility_category == "outside_fov")
                occlusion_valid = (is_occluded == expected_occluded)
                occlusion_valid_mask[local_idx] = occlusion_valid

                if not occlusion_valid:
                    envs_need_spawn_retry[env_idx] = True
                else:
                    if env_id_item not in env_dict:
                        env_dict[env_id_item] = 0.0

        return occlusion_valid_mask, envs_need_spawn_retry, env_dict, states

    def geometric_occlusion_check(self, env_ids, valid_indices,
                                  occlusion_valid_mask, envs_need_spawn_retry,
                                  device):

        occlusion_passed_env_ids = env_ids[occlusion_valid_mask]
        # Track which environments passed geometric validation
        geometric_valid_mask = occlusion_valid_mask.clone(
        )  # Start with occlusion results

        if occlusion_passed_env_ids.numel() > 0:
            # Generate candidate points as in generate_valid_circle_points
            num_envs_passed = len(occlusion_passed_env_ids)
            fov_deg = 30.0
            fov_rad = math.radians(fov_deg)
            camera_pos = self._camera_obj.data.root_pos_w[
                occlusion_passed_env_ids, :2]
            goal_pos = self._goal.data.root_pos_w[occlusion_passed_env_ids, :2]
            d = torch.norm(camera_pos - goal_pos, dim=1)
            half_fov = torch.tensor(fov_rad / 2, device=device)
            radii = (d / 2) / torch.tan(half_fov)
            radii = radii * 1.2
            radii = radii.unsqueeze(1)
            num_angles = int(360.0 / 2.0)
            angles = torch.linspace(0, 2 * math.pi, num_angles, device=device)
            angles_expanded = angles.unsqueeze(0).expand(num_envs_passed, -1)
            all_x = goal_pos[:, 0].unsqueeze(
                1) + radii * torch.cos(angles_expanded)
            all_y = goal_pos[:, 1].unsqueeze(
                1) + radii * torch.sin(angles_expanded)
            total_points = num_envs_passed * num_angles
            all_points_batch = torch.stack([all_x, all_y],
                                           dim=2).reshape(total_points, 2)
            env_ids_batch = occlusion_passed_env_ids.unsqueeze(1).expand(
                -1, num_angles).reshape(total_points)

            # Geometric validation
            geometric_valid = self._is_point_valid_batch(
                points=all_points_batch,
                env_ids=env_ids_batch,
                check_agent_fov=False)
            geometric_valid_per_env = geometric_valid.reshape(
                num_envs_passed, num_angles)

            MIN_GEOMETRIC_VALID_POINTS = 40
            # Check if at least MIN_GEOMETRIC_VALID_POINTS valid points exist
            for i, env_id in enumerate(occlusion_passed_env_ids):
                # Find the original local_idx in final_valid_indices
                local_idx = (env_ids == env_id).nonzero(
                    as_tuple=True)[0].item()
                env_idx = valid_indices[local_idx]

                valid_mask = geometric_valid_per_env[i]
                if valid_mask.sum().item() < MIN_GEOMETRIC_VALID_POINTS:
                    # print(
                    #     f"    ❌ Env {env_id.item()}: Geometric viewpoint check FAILED ({valid_mask.sum().item()}/{MIN_GEOMETRIC_VALID_POINTS} valid points)"
                    # )
                    # Not enough viewpoints, mark for retry
                    envs_need_spawn_retry[env_idx] = True
                    geometric_valid_mask[local_idx] = False
                else:
                    if self.verbose >= 2:
                        # print(
                        #     f"    ✅ Env {env_id.item()}: Geometric viewpoint check PASSED ({valid_mask.sum().item()}/{MIN_GEOMETRIC_VALID_POINTS} valid points)"
                        # )
                        pass
        return geometric_valid_mask, envs_need_spawn_retry

    def camera_pov_validation(self, env_ids, valid_indices,
                              geometric_valid_mask, visibility_categories,
                              envs_need_spawn_retry, folder_indices,
                              spawn_attempt):
        # 3. For those that passed BOTH occlusion AND geometric checks, do camera POV validation
        for local_idx, env_idx in enumerate(valid_indices):
            # Skip if failed any previous test
            if not geometric_valid_mask[local_idx]:
                continue

            env_id = env_ids[local_idx]
            env_id_item = env_id.item()
            visibility_category = visibility_categories[env_idx]
            folder_idx = folder_indices[env_idx]

            # Save camera POV to debug folder FIRST
            debug_folder = f"{self.base_path}/debug_camera_pov"
            os.makedirs(debug_folder, exist_ok=True)
            sem_img = self._occlusion_camera.data.output[
                "semantic_segmentation"][env_id]
            cam_pov_img = sem_img[:, :, :3]
            if cam_pov_img.max() <= 1.0:
                cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(
                    np.uint8)
            else:
                cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)
            debug_filename = f"{debug_folder}/env_{env_id_item}_folder_{folder_idx}_attempt_{spawn_attempt}.png"
            target_visible_in_camera, red_count = self._check_target_in_img(
                file_name=debug_filename,
                cam_pov=cam_pov_np,
                return_red_count=True)

            camera_validation_passed = False
            if visibility_category == "in_view":
                camera_validation_passed = target_visible_in_camera
                if not camera_validation_passed:
                    if self.verbose >= 1:
                        print(
                            f"    ❌ Env {env_id_item} (folder {folder_idx}): Camera validation FAILED"
                        )
                        print(
                            f"       Expected: in_view (target visible), Got: target NOT visible (red pixels: {red_count}/{self.goal_pixel_threshold})"
                        )
                        print(f"       Debug image saved: {debug_filename}")
                else:
                    if self.verbose >= 2:
                        print(
                            f"    ✅ Env {env_id_item} (folder {folder_idx}): Camera validation PASSED - in_view (red pixels: {red_count})"
                        )
                        # pass
            elif visibility_category == "occluded":
                camera_validation_passed = not target_visible_in_camera
                if not camera_validation_passed:
                    if self.verbose >= 1:
                        print(
                            f"    ❌ Env {env_id_item} (folder {folder_idx}): Camera validation FAILED"
                        )
                        print(
                            f"       Expected: occluded (target NOT visible), Got: target visible (red pixels: {red_count}/{self.goal_pixel_threshold})"
                        )
                        print(f"       Debug image saved: {debug_filename}")
                else:
                    if self.verbose >= 2:
                        print(
                            f"    ✅ Env {env_id_item} (folder {folder_idx}): Camera validation PASSED - occluded (red pixels: {red_count})"
                        )
                        # pass
            elif visibility_category == "outside_fov":
                camera_validation_passed = not target_visible_in_camera
                if not camera_validation_passed:
                    if self.verbose >= 1:
                        print(
                            f"    ❌ Env {env_id_item} (folder {folder_idx}): Camera validation FAILED"
                        )
                        print(
                            f"       Expected: outside_fov (target NOT visible), Got: target visible (red pixels: {red_count}/{self.goal_pixel_threshold})"
                        )
                        print(f"       Debug image saved: {debug_filename}")
                else:
                    if self.verbose >= 2:
                        print(
                            f"    ✅ Env {env_id_item} (folder {folder_idx}): Camera validation PASSED - outside_fov (red pixels: {red_count})"
                        )
                        # pass

            if not camera_validation_passed:
                envs_need_spawn_retry[env_idx] = True

        return envs_need_spawn_retry

    def _reset_idx_internal(self,
                            env_ids: Sequence[int] | None,
                            rl_reset: bool = False,
                            folder_indices: List[int] = None,
                            visibility_categories: List[str] = None) -> None:
        """Internal reset logic - spawn objects and generate viewpoints."""

        # --- 1. Initialization & Sanitization ---
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids,
                                   dtype=torch.long,
                                   device=self.device)

        num_envs = len(env_ids)
        if folder_indices is None:
            folder_indices = [
                self.next_env_folder_idx + i for i in range(num_envs)
            ]

        if visibility_categories is None:
            raise RuntimeError(
                "visibility_categories must be provided to _reset_idx_internal!"
            )

        # Reset viewpoint cache
        for env_id in env_ids:
            eid = env_id.item() if torch.is_tensor(env_id) else env_id
            self.used_viewpoint_indices[eid].clear()

            # Validate labels
            global_folder_idx = folder_indices[env_ids.tolist().index(eid)]
            if global_folder_idx not in self.env_visibility_labels:
                raise RuntimeError(
                    f"Labels not set for folder {global_folder_idx}!")

        # --- 2. Geometry Caching & State Prep ---
        self._cache_valid_shapes()
        self._cache_base_dims()
        self._select_active_vpt_indices(env_ids)

        self.viewpoint_pose_counter[env_ids] = 0
        super()._reset_idx(env_ids)

        device = self._agent.device

        # Clone default states to create working copies
        goal_state = self._goal.data.default_root_state[env_ids].clone()
        agent_state = self._agent.data.default_root_state[env_ids].clone()
        camera_state = self._camera_obj.data.default_root_state[env_ids].clone(
        )
        vpt_state = self._vpt_objects.data.default_object_state[env_ids].clone(
        )

        # Retry Management
        max_spawn_attempts = 20
        envs_need_spawn_retry = torch.ones(num_envs,
                                           dtype=torch.bool,
                                           device=device)

        # --- 3. Timer Initialization ---
        if rl_reset:
            task_keys = [
                "writing_spawn_pose_time", "moving_ball_time",
                "vpt_displacement_movement_time", "camera_posing_time",
                "occlusion_raycast_time"
            ]
        else:
            task_keys = [
                "writing_spawn_pose_time", "moving_ball_time",
                "vpt_displacement_movement_time", "camera_posing_time",
                "occlusion_raycast_time", "geometric_check_time",
                "camera_validation_time", "circle_validation_time"
            ]

        timer = EnvTimer(num_envs=self.num_envs,
                         slot_to_env_id=self.slot_to_env_id,
                         task_keys=task_keys,
                         verbose=(self.verbose >= 1))

        # Pre-calculate displacement subsets
        valid_indices = torch.arange(num_envs, device=device)

        in_view_indices = [
            i for i in range(num_envs) if visibility_categories[i] == "in_view"
        ]
        rand_iv = torch.randperm(len(in_view_indices))[:len(in_view_indices) //
                                                       2]
        in_view_displaced = torch.tensor(
            in_view_indices,
            device=device)[rand_iv] if in_view_indices else torch.tensor(
                [], device=device)

        outside_fov_indices = [
            i for i in range(num_envs)
            if visibility_categories[i] == "outside_fov"
        ]
        rand_of = torch.randperm(
            len(outside_fov_indices))[:len(outside_fov_indices) // 2]
        outside_fov_displaced = torch.tensor(
            outside_fov_indices,
            device=device)[rand_of] if outside_fov_indices else torch.tensor(
                [], device=device)

        # =================================================================
        # MAIN SPAWN LOOP
        # =================================================================
        for spawn_attempt in range(max_spawn_attempts):
            if not envs_need_spawn_retry.any():
                break

            # --- A. Initial Spawn ---
            # timer.start_timer("writing_spawn_pose_time")

            retry_mask = envs_need_spawn_retry.clone()
            envs_need_spawn_retry, states = self.initial_spawn_loop(
                env_ids=env_ids,
                envs_need_spawn_retry=envs_need_spawn_retry,
                safe_range=self.center_to_boundary,
                states=[goal_state, camera_state, agent_state, vpt_state],
                allow_clipping=True,
                device=device,
            )
            goal_state, camera_state, agent_state, vpt_state = states

            valid_mask = retry_mask & ~envs_need_spawn_retry
            if not valid_mask.any():
                # timer.stop_timer("writing_spawn_pose_time", spawn_attempt, envs_need_spawn_retry)
                continue

            valid_indices = torch.where(valid_mask)[0]
            valid_env_ids = env_ids[valid_indices]

            # timer.stop_timer("writing_spawn_pose_time", spawn_attempt, envs_need_spawn_retry)

            # --- B. Move Ball ---
            # timer.start_timer("moving_ball_time")

            moved_vpt_for_ball = {i: None for i in range(num_envs)}
            move_ball_indices = torch.where(
                torch.isin(env_ids, self.envs_to_move_ball))[0]

            # moved_vpt_for_ball, states = self.moving_ball_loop(
            #     env_ids=env_ids,
            #     moved_vpt_for_ball=moved_vpt_for_ball,
            #     move_ball_indices=move_ball_indices,
            #     states=[goal_state, camera_state, agent_state, vpt_state],
            #     safe_range=self.center_to_boundary,
            #     device=device
            # )
            # goal_state, camera_state, agent_state, vpt_state = states

            # move_ball_env_ids = env_ids[move_ball_indices]

            # timer.stop_timer("moving_ball_time", spawn_attempt, envs_need_spawn_retry)

            # --- C. Move VPT Objects + Z-Check ---
            # timer.start_timer("vpt_displacement_movement_time")

            states = self.move_vpt_objects(
                env_ids=env_ids,
                valid_indices=valid_indices,
                visibility_categories=visibility_categories,
                moved_vpt_for_ball=moved_vpt_for_ball,
                states=[goal_state, camera_state, agent_state, vpt_state],
                in_view_displaced=in_view_displaced,
                outside_fov_displaced=outside_fov_displaced,
                device=device,
            )
            goal_state, camera_state, agent_state, vpt_state = states

            for local_idx, env_id in enumerate(env_ids):
                env_id_item = env_id.item() if torch.is_tensor(
                    env_id) else env_id

                # 1. Env Origin
                origin_z = self.scene.env_origins[env_id, 2]

                # 2. Get Active Objects info
                active_indices = self.active_vpt_indices[env_id_item]

                # 3. Get Scaled Heights & Ratios (from randomize_shape_scale)
                # Ratio is 0.5 for Prims (half height) and 0.0 for USDs (base)
                heights = self.all_vpt_dims[env_id, active_indices, 2]
                ratios = self.vpt_z_offset_ratios[env_id, active_indices]

                # 4. Calculate & Write
                # Z = Origin + (Height * Ratio)
                safe_z = origin_z + (heights * ratios)
                vpt_state[local_idx, active_indices, 2] = safe_z
                # print(f"Env {local_idx} - VPT obj {active_indices}:\n SafeZ {safe_z} |\n Heights {heights} |\n Ratios {ratios}")
            # ========================================================

            self.write_pose_to_sim(env_ids=valid_env_ids,
                                   indices=valid_indices,
                                   goal_default_state=goal_state,
                                   camera_obj_default_state=camera_state,
                                   agent_default_state=agent_state,
                                   vpt_obj_default_state=vpt_state)

            # Validation: Z-Bounds (Physics Check)
            goal_new_pos = self._goal.data.root_pos_w[valid_env_ids]
            camera_new_pos = self._camera_obj.data.root_pos_w[valid_env_ids]
            agent_new_pos = self._agent.data.root_pos_w[valid_env_ids]
            vpt_new_pos = self._get_active_vpt_positions(valid_env_ids,
                                                         base_pivoted=True)
            # print(f"VPT pos base pivot = {vpt_new_pos[:,:,2]}")

            envs_need_spawn_retry = self.check_z_bounds(
                env_ids=env_ids,
                valid_indices=valid_indices,
                states=(goal_new_pos, camera_new_pos, agent_new_pos,
                        vpt_new_pos),
                envs_need_spawn_retry=envs_need_spawn_retry,
                tolerance=5e-2)

            # timer.stop_timer("vpt_displacement_movement_time", spawn_attempt, envs_need_spawn_retry)

            # Update valid mask after Z-check failures
            final_valid_mask = valid_mask & ~envs_need_spawn_retry
            if not final_valid_mask.any():
                continue

            final_valid_indices = torch.where(final_valid_mask)[0]
            final_valid_env_ids = env_ids[final_valid_indices]

            # --- D. Camera Posing ---
            # timer.start_timer("camera_posing_time")

            self.outside_fov_camera_movement(
                valid_env_ids=final_valid_env_ids,
                valid_indices=final_valid_indices,
                visibility_categories=visibility_categories,
                states=[goal_state, camera_state, agent_state, vpt_state],
                device=device,
            )

            # timer.stop_timer("camera_posing_time", spawn_attempt, envs_need_spawn_retry)

            # --- E. Occlusion Validation (Raycast) ---
            # timer.start_timer("occlusion_raycast_time")

            occlusion_valid_mask, envs_need_spawn_retry, _, states = self.occlusion_validation_check(
                final_valid_env_ids=final_valid_env_ids,
                valid_indices=final_valid_indices,
                visibility_categories=visibility_categories,
                envs_need_spawn_retry=envs_need_spawn_retry,
                env_dict={},
                states=[goal_state, camera_state, agent_state, vpt_state],
                device=device)

            # timer.stop_timer("occlusion_raycast_time", spawn_attempt, envs_need_spawn_retry)

            # --- OPTIONAL: Full Data Collection Steps ---
            if not rl_reset:
                # --- F. Geometric Validation ---
                # timer.start_timer("geometric_check_time")

                geometric_valid_mask, envs_need_spawn_retry = self.geometric_occlusion_check(
                    env_ids=final_valid_env_ids,
                    valid_indices=final_valid_indices,
                    occlusion_valid_mask=occlusion_valid_mask,
                    envs_need_spawn_retry=envs_need_spawn_retry,
                    device=device)

                # timer.stop_timer("geometric_check_time", spawn_attempt, envs_need_spawn_retry)

                # --- G. Camera POV Validation ---
                # timer.start_timer("camera_validation_time")

                envs_need_spawn_retry = self.camera_pov_validation(
                    env_ids=final_valid_env_ids,
                    valid_indices=final_valid_indices,
                    geometric_valid_mask=geometric_valid_mask,
                    visibility_categories=visibility_categories,
                    envs_need_spawn_retry=envs_need_spawn_retry,
                    folder_indices=folder_indices,
                    spawn_attempt=spawn_attempt)

                # timer.stop_timer("camera_validation_time", spawn_attempt, envs_need_spawn_retry)

            # Update Status for this Attempt
            # timer.update_status(spawn_attempt, envs_need_spawn_retry)

        # =================================================================
        # POST-LOOP FINALIZATION
        # =================================================================

        # 1. Finalize Agent Orientation & Position (Shotgun Placement)
        # This internally updates the OBB cache and ensures the agent starts collision-free.
        agent_state = self.place_agent_safely(
            env_ids=env_ids,
            agent_state=agent_state,
            vpt_state=vpt_state,
            safe_range=self.center_to_boundary - 1.0)

        # self.debug_plot_analytic_obbs(vpt_state)

        # 2. Final Write
        # Push the finalized, safe states to the physics engine.
        self.write_pose_to_sim(env_ids=env_ids,
                               indices=torch.arange(len(env_ids),
                                                    device=device),
                               vpt_obj_default_state=vpt_state,
                               agent_default_state=agent_state)

        # --- H. Circle Generation (SKIPPED IF RL_RESET) ---
        if not rl_reset:
            # timer.start_timer("circle_validation_time")

            success_mask = ~envs_need_spawn_retry
            successful_env_ids = env_ids[success_mask]

            subset_valid_points = []
            if len(successful_env_ids) > 0:
                subset_valid_points = self.generate_valid_circle_points(
                    env_ids=successful_env_ids,
                    angle_step=2.0,
                    max_attempts=100)

            if self.valid_viewpoint_poses is None:
                self.valid_viewpoint_poses = [None] * self.num_envs

            # Clear failed envs
            failed_env_ids = env_ids[envs_need_spawn_retry]
            for env_id in failed_env_ids:
                eid = env_id.item() if torch.is_tensor(env_id) else env_id
                self.valid_viewpoint_poses[eid] = torch.zeros((0, 3),
                                                              device=device)

            # Assign success envs
            for i, env_id in enumerate(successful_env_ids):
                eid = env_id.item() if torch.is_tensor(env_id) else env_id
                points_2d = subset_valid_points[i]

                if points_2d.shape[0] >= self.images_per_env:
                    agent_z = self._agent.data.default_root_state[env_id, 2]
                    points_3d = torch.zeros((points_2d.shape[0], 3),
                                            device=device)
                    points_3d[:, :2] = points_2d
                    points_3d[:, 2] = agent_z
                    self.valid_viewpoint_poses[eid] = points_3d
                else:
                    self.valid_viewpoint_poses[eid] = torch.zeros(
                        (0, 3), device=device)

            # timer.stop_timer("circle_validation_time", spawn_attempt, envs_need_spawn_retry)

            # Final update to catch any completion times
            timer.update_status(spawn_attempt, envs_need_spawn_retry)

        # timer.print_summary(spawn_attempt)

    def _is_point_valid_batch(self,
                              points: torch.Tensor,
                              env_ids: torch.Tensor,
                              min_obstacle_distance: float = 0.3,
                              min_camera_obstacle_distance: float = 0.4,
                              min_camera_target_distance: float = 1.0,
                              min_target_obstacle_distance: float = None,
                              check_agent_fov: bool = False,
                              min_required_points: int = None) -> torch.Tensor:
        """Pipeline: Geometric Checks (Fast) -> FOV Checks (Slow/Simulated)."""
        if min_required_points is None:
            min_required_points = self.images_per_env

        if min_target_obstacle_distance is None:
            min_target_obstacle_distance = self.goal_radius + 0.01

        # 1. Fast Geometric Checks
        valid_mask = self._check_geometric_validity(
            points, env_ids, min_obstacle_distance,
            min_camera_obstacle_distance, min_camera_target_distance,
            min_target_obstacle_distance)

        if not check_agent_fov or not valid_mask.any():
            return valid_mask

        # 2. Slow FOV Checks (Physics Simulation)
        valid_mask = self._check_fov_validity(points, env_ids, valid_mask,
                                              min_required_points)

        return valid_mask

    def _check_geometric_validity(self, points, env_ids, min_obs_dist,
                                  min_cam_obs_dist, min_cam_target_dist,
                                  min_target_obs_dist):
        """Validates boundaries, obstacle proximity, and camera clearance."""
        device = points.device
        valid_mask = torch.ones(points.shape[0],
                                dtype=torch.bool,
                                device=device)

        # Boundary Check
        env_origins = self.scene.env_origins[env_ids, :2]
        in_bounds = torch.all(
            (points >= env_origins - self.center_to_boundary) &
            (points <= env_origins + self.center_to_boundary),
            dim=1)
        valid_mask &= in_bounds

        if not valid_mask.any(): return valid_mask

        # Active Obstacle Positions
        active_obs_pos = self._get_active_obstacle_positions(env_ids)

        # Point (Goal) -> Obstacle Check
        # Calculates distance from every candidate point to every obstacle
        dist_pt_obs = torch.norm(points.unsqueeze(1) - active_obs_pos, dim=2)

        # Enforce min_target_obs_dist here
        valid_mask &= (dist_pt_obs.min(dim=1)[0] >= min_target_obs_dist)

        # Camera Checks
        cam_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
        valid_mask &= (torch.norm(points - cam_pos, dim=1)
                       >= min_cam_target_dist)

        dist_cam_obs = torch.norm(cam_pos.unsqueeze(1) - active_obs_pos, dim=2)
        valid_mask &= (dist_cam_obs.min(dim=1)[0] >= min_cam_obs_dist)

        return valid_mask

    def _get_active_obstacle_positions(self, env_ids):
        """Gather active VPT object positions for specific envs."""
        device = env_ids.device
        all_pos = self._vpt_objects.data.object_pos_w[env_ids, :, :2]

        if isinstance(self.active_vpt_indices, list):
            if len(self.active_vpt_indices) > 0 and isinstance(
                    self.active_vpt_indices[0], torch.Tensor):
                idx = torch.stack(self.active_vpt_indices).to(dtype=torch.long,
                                                              device=device)
            else:
                idx = torch.tensor(self.active_vpt_indices,
                                   device=device,
                                   dtype=torch.long)
        else:
            idx = self.active_vpt_indices

        # Gather: [num_points, n_active, 2]
        batch_idx = idx[env_ids].unsqueeze(-1).expand(-1, -1, 2)
        return torch.gather(all_pos, 1, batch_idx)

    def _check_fov_validity(self, points, env_ids, valid_mask, min_req_points):
        """Simulates view to check visibility. Handles sampling and logging."""
        device = points.device
        points_to_check = torch.where(valid_mask)[0]

        if points_to_check.numel() == 0: return valid_mask

        # Save State
        saved_pos = self._agent.data.root_pos_w[env_ids].clone()
        saved_quat = self._agent.data.root_quat_w[env_ids].clone()

        # Sample & queue points
        env_queues, env_status = self._sample_fov_candidates(points,
                                                             env_ids,
                                                             points_to_check,
                                                             max_s=120)

        fov_valid_global = torch.zeros_like(valid_mask)
        max_steps = max(len(q['points']) for q in env_queues.values())

        # Simulation Loop
        for step in range(max_steps):
            if self.verbose >= 2 and step % 10 == 0:
                print(f"    🔄 FOV progress: {step+1}/{max_steps} points")

            # Batch construction
            step_ids, step_pts, step_idxs = [], [], []

            for eid, data in env_queues.items():
                # Skip satisfied envs
                if env_status[eid]['count'] >= min_req_points: continue

                if step < len(data['points']):
                    step_ids.append(torch.tensor(eid, device=device))
                    step_pts.append(data['points'][step])
                    step_idxs.append(data['indices'][step])

            if not step_ids:
                if self.verbose >= 1 and all(s['count'] >= min_req_points
                                             for s in env_status.values()):
                    print(
                        f"    ✅ All environments have {min_req_points}+ valid points. Early stop."
                    )
                break

            # Vectorize
            b_envs = torch.stack(step_ids)
            b_pts = torch.stack(step_pts)
            b_idxs = torch.stack(step_idxs)

            # Teleport -> Step -> Check
            self._teleport_and_step(b_envs, b_pts)
            g_vis, c_vis = self.check_batch_object_visibility(b_envs)
            is_vis = g_vis & c_vis

            # Update results
            fov_valid_global[b_idxs] = is_vis

            # Update counts & Log
            for i, eid in enumerate(b_envs.tolist()):
                if is_vis[i]:
                    env_status[eid]['count'] += 1
                    if env_status[eid][
                            'count'] == min_req_points:  # Just hit threshold
                        if self.verbose >= 1:
                            print(
                                f"    🎯 Env {eid}: Reached {min_req_points} valid points."
                            )

        # Restore State
        restore = torch.cat([saved_pos, saved_quat], dim=1)
        self._agent.write_root_com_pose_to_sim(restore, env_ids)

        return valid_mask & fov_valid_global

    def _sample_fov_candidates(self, points, env_ids, indices, max_s):
        """Groups by env, downsamples, and logs stats."""
        unique_envs = torch.unique(env_ids[indices])
        queues, status = {}, {}

        for eid in unique_envs:
            eid_item = eid.item()
            env_indices = indices[env_ids[indices] == eid]
            total = len(env_indices)

            if total > max_s:
                perm = torch.randperm(total, device=points.device)[:max_s]
                env_indices = env_indices[perm]
                if self.verbose >= 2:
                    print(
                        f"  🎲 Env {eid_item}: Sampled {max_s}/{total} candidates."
                    )
            else:
                if self.verbose >= 2:
                    print(f"  🎲 Env {eid_item}: Using all {total} candidates.")

            queues[eid_item] = {
                'points': points[env_indices],
                'indices': env_indices
            }
            status[eid_item] = {'count': 0}

        return queues, status

    def _teleport_and_step(self, env_ids, points):
        """Teleports agents, orients to midpoints, and steps physics."""
        cam_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
        goal_pos = self._goal.data.root_pos_w[env_ids, :2]

        dirs = ((cam_pos + goal_pos) / 2.0) - points
        yaws = torch.atan2(dirs[:, 1], dirs[:, 0])

        # Pose construction
        pos = torch.zeros((len(env_ids), 3), device=points.device)
        pos[:, :2] = points
        pos[:, 2] = self._agent.data.default_root_state[env_ids, 2]

        quat = torch.zeros((len(env_ids), 4), device=points.device)
        quat[:, 0] = torch.cos(yaws / 2)
        quat[:, 3] = torch.sin(yaws / 2)

        self._agent.write_root_com_pose_to_sim(torch.cat([pos, quat], dim=1),
                                               env_ids)

        # Step Physics
        self.sim.step()
        # self._tiled_camera.update(self.sim.cfg.dt)
        self._agent.update(self.sim.cfg.dt)
        self._camera_obj.update(self.sim.cfg.dt)
        self._goal.update(self.sim.cfg.dt)
        self._vpt_objects.update(self.sim.cfg.dt)

    def generate_valid_circle_points(
            self,
            env_ids: torch.Tensor,
            angle_step: float = 2.0,
            max_attempts: int = 300) -> List[torch.Tensor]:
        """Generate valid viewpoint positions in parallel for all environments."""
        device = self.device
        num_envs = len(env_ids)

        MIN_REQUIRED_POINTS = self.images_per_env

        # Generate all candidate angles
        num_angles = int(360.0 / angle_step)
        angles = torch.linspace(0, 2 * math.pi, num_angles, device=device)

        # Get radii for all environments
        fov_deg = 30.0
        fov_rad = math.radians(fov_deg)
        camera_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
        goal_pos = self._goal.data.root_pos_w[env_ids, :2]
        d = torch.norm(camera_pos - goal_pos, dim=1)

        half_fov = torch.tensor(fov_rad / 2, device=self.device)
        radii = (d / 2) / torch.tan(half_fov)
        radii = radii * 1.3
        radii = radii.unsqueeze(1)

        # Generate all points for all envs
        angles_expanded = angles.unsqueeze(0).expand(num_envs, -1)
        all_x = self._goal.data.root_pos_w[env_ids, 0].unsqueeze(
            1) + radii * torch.cos(angles_expanded)
        all_y = self._goal.data.root_pos_w[env_ids, 1].unsqueeze(
            1) + radii * torch.sin(angles_expanded)

        total_points = num_envs * num_angles
        all_points_batch = torch.stack([all_x, all_y],
                                       dim=2).reshape(total_points, 2)
        env_ids_batch = env_ids.unsqueeze(1).expand(
            -1, num_angles).reshape(total_points)

        # Step 1: Geometric validation
        geometric_valid = self._is_point_valid_batch(points=all_points_batch,
                                                     env_ids=env_ids_batch,
                                                     check_agent_fov=False)

        geometric_valid_per_env = geometric_valid.reshape(num_envs, num_angles)

        if self.verbose >= 2:
            for i, env_id in enumerate(env_ids):
                env_id_item = env_id.item()
                # print(
                #     f"  Env {env_id_item}: {geometric_valid_per_env[i].sum().item()} geometric candidates"
                # )

        # Step 2: Vectorized displacement filtering across all environments
        displacement_filtered_points = []
        displacement_filtered_env_ids = []
        displacement_filtered_indices = []

        MIN_CANDIDATES_FOR_FOV = 40  # Require at least this many candidates before FOV check

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()
            valid_mask = geometric_valid_per_env[i]

            if not valid_mask.any():
                if self.verbose >= 2:
                    print(f"  Env {env_id_item}: No geometric candidates")
                continue

            num_geometric = valid_mask.sum().item()

            # Early check: need enough geometric candidates
            if num_geometric < MIN_CANDIDATES_FOR_FOV:
                if self.verbose >= 1:
                    print(
                        f"  Env {env_id_item}: ❌ Only {num_geometric} geometric candidates, need {MIN_CANDIDATES_FOR_FOV}. Skipping."
                    )
                continue

            valid_points = all_points_batch[i * num_angles:(i + 1) *
                                            num_angles][valid_mask]

            # Sort by angle for consistent ordering
            goal_pos_2d = self._goal.data.root_pos_w[env_id, :2]
            relative_pos = valid_points - goal_pos_2d
            angles_rad = torch.atan2(relative_pos[:, 1], relative_pos[:, 0])
            sorted_indices = torch.argsort(angles_rad)
            sorted_points = valid_points[sorted_indices]

            # Vectorized greedy displacement filtering
            num_points = len(sorted_points)
            if num_points == 0:
                continue

            # Compute pairwise distances for all points at once
            # Shape: (num_points, num_points)
            diff = sorted_points.unsqueeze(0) - sorted_points.unsqueeze(1)
            pairwise_distances = torch.norm(diff, dim=2)

            # Greedy selection using vectorized operations
            selected_mask = torch.zeros(num_points,
                                        dtype=torch.bool,
                                        device=device)
            selected_mask[0] = True  # Always select first point

            for idx in range(1, num_points):
                # Check if this point is far enough from all selected points
                distances_to_selected = pairwise_distances[idx, selected_mask]
                if torch.all(
                        distances_to_selected >= self.min_viewpoint_distance):
                    selected_mask[idx] = True

            filtered_candidates = sorted_points[selected_mask]

            num_before = len(sorted_points)
            num_after = len(filtered_candidates)

            # Check: need enough candidates after displacement filtering
            if num_after < MIN_CANDIDATES_FOR_FOV:
                if self.verbose >= 1:
                    rejection_rate = (1 - num_after / num_before
                                      ) * 100 if num_before > 0 else 0
                    print(
                        f"  Env {env_id_item}: ❌ Only {num_after} candidates after displacement ({rejection_rate:.1f}% rejected), need {MIN_CANDIDATES_FOR_FOV}. Skipping FOV check."
                    )
                continue

            if self.verbose >= 2:
                rejection_rate = (
                    1 - num_after / num_before) * 100 if num_before > 0 else 0
                print(
                    f"  Env {env_id_item}: Displacement filter: {num_after}/{num_before} kept ({rejection_rate:.1f}% rejected)"
                )

            # Add these pre-filtered candidates for FOV checking
            displacement_filtered_points.append(filtered_candidates)
            displacement_filtered_env_ids.extend([env_id.item()] *
                                                 len(filtered_candidates))
            displacement_filtered_indices.extend([i] *
                                                 len(filtered_candidates))

        if len(displacement_filtered_points) == 0:
            if self.verbose >= 1:
                print(
                    f"  ❌ No candidates passed displacement filter for any environment"
                )
            return [
                torch.zeros((0, 2), device=device) for _ in range(num_envs)
            ]

        # Concatenate all displacement-filtered candidates
        all_candidates = torch.cat(displacement_filtered_points, dim=0)
        all_candidates_env_ids = torch.tensor(displacement_filtered_env_ids,
                                              dtype=torch.long,
                                              device=device)
        all_candidates_indices = torch.tensor(displacement_filtered_indices,
                                              dtype=torch.long,
                                              device=device)

        if self.verbose >= 2:
            total_geometric = geometric_valid.sum().item()
            total_after_displacement = len(all_candidates)
            saved_compute = ((total_geometric - total_after_displacement) /
                             total_geometric *
                             100) if total_geometric > 0 else 0
            print(
                f"  💡 FOV candidates: {total_after_displacement}/{total_geometric} ({saved_compute:.1f}% compute saved)"
            )

        # Store original agent state
        original_agent_pos = self._agent.data.root_pos_w[env_ids].clone()
        original_agent_quat = self._agent.data.root_quat_w[env_ids].clone()

        # Step 3: FOV check ONLY on displacement-filtered candidates
        fov_valid_mask = self._is_point_valid_batch(
            points=all_candidates,
            env_ids=all_candidates_env_ids,
            check_agent_fov=True,
            min_required_points=
            MIN_REQUIRED_POINTS  # Pass min required to enable early stopping
        )

        # Restore original agent positions
        self._agent.write_root_pose_to_sim(torch.cat(
            [original_agent_pos, original_agent_quat], dim=-1),
                                           env_ids=env_ids)

        # Step 4: Collect final valid points per environment
        all_valid_points = []

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()

            # Get FOV-valid points for this env
            env_mask = all_candidates_indices == i
            env_fov_valid = fov_valid_mask[env_mask]

            num_candidates = env_mask.sum().item()
            num_fov_valid = env_fov_valid.sum().item()

            if not env_fov_valid.any():
                all_valid_points.append(torch.zeros((0, 2), device=device))
                if self.verbose >= 1:
                    print(
                        f"  Env {env_id_item}: ❌ 0/{num_candidates} passed FOV check"
                    )
                continue

            valid_points_tensor = all_candidates[env_mask][env_fov_valid]

            fov_rejection_rate = (1 - num_fov_valid / num_candidates
                                  ) * 100 if num_candidates > 0 else 0

            if len(valid_points_tensor) >= MIN_REQUIRED_POINTS:
                all_valid_points.append(valid_points_tensor)
                if self.verbose >= 2:
                    print(
                        f"  Env {env_id_item}: ✅ {len(valid_points_tensor)}/{num_candidates} passed FOV ({fov_rejection_rate:.1f}% rejected)"
                    )
            else:
                all_valid_points.append(torch.zeros((0, 2), device=device))
                if self.verbose >= 1:
                    print(
                        f"  Env {env_id_item}: ❌ Only {len(valid_points_tensor)}/{MIN_REQUIRED_POINTS} points ({fov_rejection_rate:.1f}% FOV rejection)"
                    )

        return all_valid_points

    def _collect_images_for_slot(self, env_id: torch.Tensor,
                                 folder_idx: int) -> None:
        """Collect images for a specific slot/environment.
        
        Args:
            env_id: Tensor representing the environment SLOT index (0-7)
            folder_idx: Global folder index for saving images
        """
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        device = self.device

        # env_id_item is the SLOT index, not the global env number
        global_env_id = self.slot_to_env_id[env_id_item]

        if self.verbose >= 1:
            print(
                f"    📸 Collecting {self.images_per_env} images for slot {env_id_item}, env {global_env_id}, folder {folder_idx}"
            )

        # Get the selected viewpoints for this SLOT
        viewpoints = self.selected_viewpoints_for_collection[env_id_item]
        if viewpoints is None:
            # Debug print
            print(
                f"[ERROR]: No viewpoints selected for slot {env_id_item} (env {global_env_id})"
            )
            print(
                f"  valid_viewpoint_poses length: {len(self.valid_viewpoint_poses[env_id_item]) if self.valid_viewpoint_poses[env_id_item] else 'None'}"
            )
            raise RuntimeError(
                f"No viewpoints selected for slot {env_id_item} (env {global_env_id})"
            )

        # Create single-env tensor
        single_env_tensor = torch.tensor([env_id_item],
                                         dtype=torch.long,
                                         device=device)

        # Collect images from all viewpoints
        for viewpoint_idx in range(self.images_per_env):
            # Get target position for this viewpoint
            target_pos_2d = viewpoints[viewpoint_idx]
            if target_pos_2d.shape[-1] == 3:
                target_pos_2d = target_pos_2d[:2]

            # Create 3D position
            target_pos_3d = torch.zeros(3, device=device)
            target_pos_3d[:2] = target_pos_2d
            target_pos_3d[2] = self._agent.data.default_root_state[env_id_item,
                                                                   2]

            # Calculate orientation to look at midpoint between camera and goal
            camera_pos_3d = self._camera_obj.data.root_pos_w[env_id_item]
            goal_pos_3d = self._goal.data.root_pos_w[env_id_item]
            midpoint = (camera_pos_3d[:2] + goal_pos_3d[:2]) / 2.0
            direction = midpoint - target_pos_3d[:2]
            yaw = torch.atan2(direction[1], direction[0])

            # Create quaternion
            quat = torch.zeros(4, device=device)
            quat[0] = torch.cos(yaw / 2)
            quat[3] = torch.sin(yaw / 2)

            # Write pose for this single agent
            pose = torch.cat([target_pos_3d, quat]).unsqueeze(0)
            self._agent.write_root_com_pose_to_sim(pose, single_env_tensor)
            self._agent.write_root_com_velocity_to_sim(
                torch.zeros((1, 6), device=device), single_env_tensor)

            # Update cameras
            for _ in range(1):
                self.sim.step()
                self._rgb_tiled_camera.update(self.sim.cfg.dt)
                # self._distance_tiled_camera.update(self.sim.cfg.dt)
                if self.save_camera_pov:
                    self._occlusion_camera.update(self.sim.cfg.dt)

            # Get camera data for this env
            rgb_data = self._rgb_tiled_camera.data.output["rgb"]
            depth_data = self._rgb_tiled_camera.data.output[
                "distance_to_camera"]
            camera_pov_data = self._occlusion_camera.data.output[
                "semantic_segmentation"] if self.save_camera_pov else None

            # Save this single image
            self._save_single_image(env_id_item, folder_idx, rgb_data,
                                    depth_data, camera_pov_data, viewpoint_idx)

        # Save env config
        self._save_env_config_to_json(env_id_item, folder_idx)

        # Clear viewpoints for this env
        self.selected_viewpoints_for_collection[env_id_item] = None

        if self.verbose >= 1:
            print(
                f"    ✅ Collected and saved {self.images_per_env} images for folder {folder_idx}"
            )

    def _save_single_image(self,
                           env_id_item: int,
                           folder_idx: int,
                           rgb_data: torch.Tensor,
                           depth_data: torch.Tensor,
                           camera_pov_data: torch.Tensor = None,
                           image_idx: int = 0) -> None:
        """Save a single image for an environment."""

        if folder_idx not in self.env_visibility_labels:
            raise RuntimeError(
                f"CRITICAL ERROR: No visibility label found for folder_idx {folder_idx}!"
            )

        visibility_label = self.env_visibility_labels[folder_idx]

        if visibility_label not in ["Yes", "No"]:
            raise RuntimeError(
                f"CRITICAL ERROR: Invalid visibility label '{visibility_label}' for folder_idx {folder_idx}!"
            )

        # Create directories
        rgb_base = f"{self.base_path}/RGB/{visibility_label}"
        rgb_env_folder = f"{rgb_base}/env_{folder_idx}"
        cam_base = f"{self.base_path}/cam/{visibility_label}"
        cam_env_folder = f"{cam_base}/env_{folder_idx}"
        depth_base = f"{self.base_path}/Depth/{visibility_label}"
        depth_env_folder = f"{depth_base}/env_{folder_idx}"

        os.makedirs(rgb_env_folder, exist_ok=True)
        os.makedirs(cam_env_folder, exist_ok=True)
        os.makedirs(depth_env_folder, exist_ok=True)

        # Save RGB
        rgb_filename = f"{rgb_env_folder}/image_{image_idx:04d}.png"
        rgb_img = rgb_data[env_id_item, :, :, :3]

        if rgb_img.max() <= 1.0:
            rgb_np = (rgb_img.cpu().numpy() * 255.0).astype(np.uint8)
        else:
            rgb_np = rgb_img.cpu().numpy().astype(np.uint8)

        cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Save Depth
        depth_filename = f"{depth_env_folder}/image_{image_idx:04d}.png"
        depth_img = depth_data[env_id_item, :, :, :]
        depth_np = depth_img.cpu().numpy()

        depth_np[np.isinf(depth_np)] = depth_np[~np.isinf(depth_np)].max(
        ) if depth_np[~np.isinf(depth_np)].size > 0 else 0

        if depth_np.max() > depth_np.min():
            depth_normalized = ((depth_np - depth_np.min()) /
                                (depth_np.max() - depth_np.min()) *
                                255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)

        cv2.imwrite(depth_filename, depth_normalized)

        # Save camera POV (only once)
        if self.save_camera_pov and camera_pov_data is not None and image_idx == 0:
            cam_pov_filename = f"{cam_env_folder}/cam_pov.png"
            if not os.path.exists(cam_pov_filename):
                cam_pov_img = camera_pov_data[env_id_item, :, :, :3]

                if cam_pov_img.max() <= 1.0:
                    cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(
                        np.uint8)
                else:
                    cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)

                cv2.imwrite(cam_pov_filename,
                            cv2.cvtColor(cam_pov_np, cv2.COLOR_RGB2BGR))

    def _cache_base_dims(self):
        """
        Caches the base dimensions of all rigid objects from the config.
        For UsdAssets use scale.
        For Cuboids use size [x, y, z].
        For Cylinders and Cones use radius and height (x=2r, y=2r, z=h).
        """
        self.vpt_base_dims = []

        # Iterate over the config dictionary to preserve order matching object indices
        # We assume self.cfg.vpt_objects.rigid_objects is an ordered dict or similar structure
        for key, obj_cfg in self.cfg.vpt_objects.rigid_objects.items():
            spawn_cfg = obj_cfg.spawn
            dims = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                # UsdAssets use scale. Default to (1,1,1) if None, but usually present in cfg
                scale = getattr(spawn_cfg, "scale", (1.0, 1.0, 1.0))
                if scale is None:
                    print("Scale not found")
                    scale = (1.0, 1.0, 1.0)

                filename = spawn_cfg.usd_path.split("/")[-1].split(".")[0]
                if filename.endswith(('X', 'L', 'T')):
                    # For specific shapes, we might want to adjust scale handling
                    if filename.endswith(("X")):
                        dims = torch.tensor(
                            [1.0 * scale[0], 0.25 * scale[1], 1.0 * scale[2]])
                    elif filename.endswith("L"):
                        dims = torch.tensor(
                            [1.0 * scale[0], 0.25 * scale[1], 1.0 * scale[2]])
                    else:
                        dims = torch.tensor([1.0, 1.0, 1.0],
                                            device=self.device)
                    dims = torch.tensor(scale, device=self.device)

            elif isinstance(spawn_cfg, sim_utils.CuboidCfg):
                # Cuboids use size (x, y, z)
                dims = torch.tensor(spawn_cfg.size, device=self.device)

            elif isinstance(spawn_cfg,
                            (sim_utils.ConeCfg, sim_utils.CylinderCfg)):
                # Cylinders/Cones use radius and height
                # x = 2*r, y = 2*r, z = h
                r = spawn_cfg.radius
                h = spawn_cfg.height
                dims = torch.tensor([2 * r, 2 * r, h], device=self.device)

            else:
                # Fallback
                dims = torch.tensor([1.0, 1.0, 1.0], device=self.device)

            self.vpt_base_dims.append(dims)

        # Stack into a single tensor of shape (num_objs, 3)
        self.vpt_base_dims = torch.stack(self.vpt_base_dims)

        if self.verbose >= 1:
            print(
                f"📦 Cached base dimensions for {len(self.vpt_base_dims)} objects."
            )

    def randomize_shape_scale(self,
                              prim_path_expr: str | list,
                              is_random: bool = True):
        """
        Refined Randomization:
        1. Identifies object type (Mesh, Primitive, or Xform).
        2. Calculates Scale and Z-Position analytically.
        3. Updates GEOMETRY (Points/Radius/Height) where possible, falls back to Xform Scale.
        4. Calculates and stores precise bounding box info.
        """
        world = World.instance()
        if world.is_playing():
            world.pause()

        stage = get_current_stage()

        # Initialize Mesh Cache to prevent "Scale Drift" (accumulating scale over iterations)
        if not hasattr(self, "_cached_mesh_points"):
            self._cached_mesh_points = {}

        # We need bbox_cache to get ACCURATE final visual bounds/offsets
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                       [UsdGeom.Tokens.default_])

        # 1. Initialize Standard Dims
        if not hasattr(
                self,
                "all_vpt_dims") or self.all_vpt_dims.shape[0] != self.num_envs:
            self.all_vpt_dims = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs, 3), device=self.device)
            self.vpt_obj_default_state = torch.zeros(
                (self.num_envs, self.num_objs, 3), device=self.device)

        # 2. Initialize BB Storage
        if not hasattr(
                self,
                "all_vpt_bb") or self.all_vpt_bb.shape[0] != self.num_envs:
            self.all_vpt_bb = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs, 4), device=self.device)

        # 3. Initialize Offset Ratios
        if not hasattr(
                self,
                "vpt_z_offset_ratios") or self.vpt_z_offset_ratios.shape != (
                    self.num_envs, self.cfg.num_vpt_objs):
            self.vpt_z_offset_ratios = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs), device=self.device)

        if not hasattr(self,
                       "vpt_shapes") or self.vpt_z_offset_ratios.shape != (
                           self.num_envs, self.cfg.num_vpt_objs):
            self.vpt_shapes = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs), device=self.device)

        # Resolve paths
        if isinstance(prim_path_expr, str):
            prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
        elif isinstance(prim_path_expr, list):
            prim_paths = []
            for expr in prim_path_expr:
                prim_paths.extend(sim_utils.find_matching_prim_paths(expr))

        print(
            f"\n[Randomizing Scale & Geometry] Processing {len(prim_paths)} objects..."
        )
        obj_configs = list(self.cfg.vpt_objects.rigid_objects.values())

        with Sdf.ChangeBlock():
            for prim_path in prim_paths:
                root_prim = stage.GetPrimAtPath(prim_path)
                if not root_prim.IsValid(): continue

                # Parse Indices (Env/Obj)
                try:
                    path_str = root_prim.GetPath().pathString
                    path_parts = path_str.split("/")
                    env_part = next(p for p in path_parts
                                    if p.startswith("env_"))
                    env_idx = int(env_part.split("_")[-1])
                    obs_part = next(p for p in path_parts
                                    if p.startswith("obs_"))
                    obj_idx = int(obs_part.split("_")[-1])
                except:
                    continue

                if obj_idx >= len(obj_configs): continue
                obj_cfg = obj_configs[obj_idx]
                spawn_cfg = obj_cfg.spawn

                # Get/Create Transform Ops
                xform = UsdGeom.Xformable(root_prim)
                scale_op = None
                translate_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = op
                    elif op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op

                if scale_op is None:
                    scale_op = xform.AddScaleOp(
                        UsdGeom.XformOp.PrecisionDouble)
                if translate_op is None:
                    translate_op = xform.AddTranslateOp(
                        UsdGeom.XformOp.PrecisionDouble)

                if is_random:
                    base_dim = self.vpt_base_dims[obj_idx].cpu().numpy(
                    )  # [x, y, z]
                    final_scale_vec = Gf.Vec3d(1, 1, 1)
                    final_z_pos = 0.0
                    z_offset_multiplier = 0.0
                    shape_name = -1  # Unknown

                    # --- LOGIC BRANCHING ---
                    # 1. USD Files (Complex Meshes / Xforms)
                    if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                        filename = spawn_cfg.usd_path.split("/")[-1].split(
                            ".")[0]

                        if filename.endswith(
                            ('X', 'L', 'T', 'A', 'H', 'I', 'Z', 'Table_A',
                             'Table_B', 'Bench')):
                            # Special USDs: Base Pivoted
                            z_offset_multiplier = 0.0

                            if filename.endswith(("X")):
                                s_xz = random.uniform(0.5, 3.0)
                                s_y = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_xz, base_dim[1] * s_y,
                                    base_dim[2] * s_xz)
                            elif filename.endswith(("L")):
                                s_factor = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_factor,
                                    base_dim[1] * s_factor,
                                    base_dim[2] * s_factor)
                            elif filename.endswith(("H")):
                                s_xz = random.uniform(0.5, 3.0)
                                s_y = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_xz, base_dim[1] * s_y,
                                    base_dim[2] * s_xz)
                            elif filename.endswith(("I")):
                                s_xz = random.uniform(0.5, 3.0)
                                s_y = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_xz, base_dim[1] * s_y,
                                    base_dim[2] * s_xz)
                            elif filename.endswith(("Z")):
                                s_xz = random.uniform(0.5, 3.0)
                                s_y = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_xz, base_dim[1] * s_y,
                                    base_dim[2] * s_xz)
                            elif filename.endswith(("Table_A")):
                                s_x = random.uniform(0.5, 3.0)
                                s_y = random.uniform(0.5, 3.0)
                                s_z = random.uniform(1.0, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_x, base_dim[1] * s_y,
                                    base_dim[2] * s_z)
                            elif filename.endswith(("Table_B")):
                                s_factor = random.uniform(1.0, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_factor,
                                    base_dim[1] * s_factor,
                                    base_dim[2] * s_factor)
                            elif filename.endswith(("Bench")):
                                s_factor = random.uniform(1.0, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_factor,
                                    base_dim[1] * s_factor,
                                    base_dim[2] * s_factor)
                            elif filename.endswith(("A")):
                                s_xz = random.uniform(0.5, 3.0)
                                s_y = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_xz, base_dim[1] * s_y,
                                    base_dim[2] * s_xz)

                            # Calculation: Base is at 0, so Z=0
                            final_z_pos = 0.0
                        else:
                            z_offset_multiplier = 0.0
                            s_xy = random.uniform(0.5, 2.5)
                            s_z = random.uniform(0.5, 2.5)
                            final_scale_vec = Gf.Vec3d(base_dim[0] * s_xy,
                                                       base_dim[1] * s_xy,
                                                       base_dim[2] * s_z)
                            shape_name = -1

                    # 2. Cuboids
                    elif isinstance(spawn_cfg, sim_utils.MeshCuboidCfg):
                        shape_name = 2
                        z_offset_multiplier = 0.5
                        s_x = random.uniform(0.5, 2.5)
                        s_y = random.uniform(0.5, 2.5)
                        s_z = random.uniform(0.5, 2.5)
                        final_scale_vec = Gf.Vec3d(s_x, s_y, s_z)
                        total_height = base_dim[2] * s_z
                        final_z_pos = total_height * z_offset_multiplier

                    # 3. Cylinders / Cones
                    elif isinstance(
                            spawn_cfg,
                        (sim_utils.MeshCylinderCfg, sim_utils.MeshConeCfg)):
                        if isinstance(spawn_cfg, sim_utils.MeshCylinderCfg):
                            shape_name = 3
                            z_offset_multiplier = 0.5
                        elif isinstance(spawn_cfg, sim_utils.MeshConeCfg):
                            shape_name = 4
                            z_offset_multiplier = 0.0
                        s_r = random.uniform(0.75, 1.0)
                        s_h = random.uniform(0.75, 2.5)
                        # Store the factors, not just the result, to apply to radius/height attrs
                        final_scale_vec = Gf.Vec3d(s_r, s_r, s_h)
                        total_height = base_dim[2] * s_h
                        final_z_pos = total_height * z_offset_multiplier

                    scale_op.Set(final_scale_vec)

                    # Update Translate (Z-Position)
                    current_trans = translate_op.Get()
                    translate_op.Set(
                        Gf.Vec3d(current_trans[0], current_trans[1],
                                 final_z_pos))

                    # Update Internal State Records
                    self.vpt_obj_default_state[env_idx, obj_idx,
                                               2] = final_z_pos
                    self.vpt_z_offset_ratios[env_idx,
                                             obj_idx] = z_offset_multiplier
                    self.vpt_shapes[env_idx, obj_idx] = shape_name

                    # --- COMPUTE BOUNDS ---
                    # Because we modified geometry/scale, we must re-compute bounds
                    bbox_cache.Clear()
                    local_bound = bbox_cache.ComputeLocalBound(
                        root_prim).GetRange()

                    final_w = local_bound.GetMax()[0] - local_bound.GetMin()[0]
                    final_l = local_bound.GetMax()[1] - local_bound.GetMin()[1]
                    final_h = local_bound.GetMax()[2] - local_bound.GetMin()[2]

                    self.all_vpt_dims[env_idx, obj_idx, 0] = final_w
                    self.all_vpt_dims[env_idx, obj_idx, 1] = final_l
                    self.all_vpt_dims[env_idx, obj_idx, 2] = final_h

                    b_min = local_bound.GetMin()
                    b_max = local_bound.GetMax()

                    self.all_vpt_bb[env_idx, obj_idx, 0] = b_min[0]
                    self.all_vpt_bb[env_idx, obj_idx, 1] = b_min[1]
                    self.all_vpt_bb[env_idx, obj_idx, 2] = b_max[0]
                    self.all_vpt_bb[env_idx, obj_idx, 3] = b_max[1]

        world.play()
        self.sim.step(render=False)

    def randomize_shape_color(self,
                              prim_path_expr: str | list,
                              random_roughness: bool = False,
                              random_metallic: bool = False):
        """
        Randomize the color, and optionally roughness and metallic attributes of the geometry.
        
        Args:
            prim_path_expr (str | list): The prim path expression(s) to modify.
            random_roughness (bool): If True, randomizes inputs:roughness (0.0 to 1.0).
            random_metallic (bool): If True, randomizes inputs:metallic (0.0 to 1.0).
        """
        stage = get_current_stage()

        if isinstance(prim_path_expr, str):
            prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
        elif isinstance(prim_path_expr, list):
            prim_paths = []
            for expr in prim_path_expr:
                prim_paths.extend(sim_utils.find_matching_prim_paths(expr))

        with Sdf.ChangeBlock():
            for prim_path in prim_paths:
                # --- Generate Random Values ---
                rand_color = Gf.Vec3f(self.get_color())

                rand_roughness = random.random()
                rand_metallic = random.random()

                # Helper to set attribute on a shader spec
                def _set_shader_attr(shader_spec, attr_name, value, type_name):
                    # Try to find existing attribute first
                    attr_spec = shader_spec.GetAttributeAtPath(
                        shader_spec.path.AppendProperty(attr_name))
                    if not attr_spec:
                        # Create if it doesn't exist
                        attr_spec = Sdf.AttributeSpec(shader_spec, attr_name,
                                                      type_name)
                    attr_spec.default = value

                # --- STRATEGY 1: FAST PATH (Standard Assets) ---
                # We assume the standard hierarchy: /geometry/material/Shader
                standard_shader_path = prim_path + "/geometry/material/Shader"

                # We use the root layer to modify/override specs
                layer = stage.GetRootLayer()

                # Check if this standard path exists by looking for the Color attribute (as a proxy)
                # We check the attribute specifically to avoid creating empty Prims if the path is wrong
                check_prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
                color_check = check_prim_spec.GetAttributeAtPath(
                    standard_shader_path + ".inputs:diffuseColor")

                if color_check:
                    # If the standard path is valid, get the spec for the Shader itself for easier updates
                    shader_spec = Sdf.CreatePrimInLayer(
                        layer, standard_shader_path)

                    # Set Color
                    color_check.default = rand_color

                    # Set Roughness
                    if random_roughness:
                        _set_shader_attr(shader_spec, "inputs:roughness",
                                         rand_roughness,
                                         Sdf.ValueTypeNames.Float)

                    # Set Metallic
                    if random_metallic:
                        _set_shader_attr(shader_spec, "inputs:metallic",
                                         rand_metallic,
                                         Sdf.ValueTypeNames.Float)

                    continue

                # --- STRATEGY 2: DYNAMIC SEARCH (Custom Assets) ---
                # If the fast path failed, we search the stage for the shader.

                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                target_shader_path = None

                # Usd.PrimRange iterates depth-first through all children
                for child in Usd.PrimRange(prim):
                    if child.GetTypeName() == "Shader":
                        target_shader_path = child.GetPath().pathString
                        break

                if target_shader_path:
                    shader_spec = Sdf.CreatePrimInLayer(
                        layer, target_shader_path)

                    # Set Color
                    # Handle potential naming differences for color (usually inputs:diffuseColor)
                    color_attr_name = "inputs:diffuseColor"
                    _set_shader_attr(shader_spec, color_attr_name, rand_color,
                                     Sdf.ValueTypeNames.Color3f)

                    # Set Roughness
                    if random_roughness:
                        _set_shader_attr(shader_spec, "inputs:roughness",
                                         rand_roughness,
                                         Sdf.ValueTypeNames.Float)

                    # Set Metallic
                    if random_metallic:
                        _set_shader_attr(shader_spec, "inputs:metallic",
                                         rand_metallic,
                                         Sdf.ValueTypeNames.Float)

    def randomize_spherical_lights(self, prim_paths, random_light_off=False):
        """
        Randomizes Spherical Lights with a minimum separation distance check.
        """
        stage = get_current_stage()

        limit = self.center_to_boundary * 0.8
        active_paths = set(prim_paths)

        # Track positions per environment to ensure separation
        # Structure: { env_idx: [(x, y), (x, y)] }
        env_light_positions = {}
        min_separation = 8.0

        if random_light_off and len(prim_paths) > 0:
            num_to_keep = random.randint(1, len(prim_paths))
            active_paths = set(random.sample(prim_paths, num_to_keep))

        for path in prim_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue

            # 1. EXTRACT ENV INDEX
            match = re.search(r"env_(\d+)", path)
            if not match:
                continue
            env_idx = int(match.group(1))

            # Initialize position list for this environment if new
            if env_idx not in env_light_positions:
                env_light_positions[env_idx] = []

            # (Optional) Extract origin if needed for global coords,
            # but currently using local offsets based on your previous snippet.
            # origin_data = self.scene.env_origins[env_idx].tolist()

            # 2. INTENSITY
            if path not in active_paths:
                prim.GetAttribute("inputs:intensity").Set(0.0)
                continue

            rand_intensity = random.uniform(40_000.0, 75_000.0)
            prim.GetAttribute("inputs:intensity").Set(rand_intensity)

            # 3. POSITION WITH SEPARATION CHECK
            rand_z_offset = random.uniform(7.5, 15.0)

            # Define the allowed magnitude ranges (percentages of center_to_boundary)
            valid_ranges = [
                (0.1, 0.3),  # Inner ring
                (0.7, 0.9)  # Outer corners
            ]

            cand_x, cand_y = 0.0, 0.0
            max_retries = 20

            for _ in range(max_retries):
                # GENERATE X
                # 1. Pick a range (Inner or Outer)
                rx_min, rx_max = random.choice(valid_ranges)
                # 2. Sample magnitude and apply random sign
                mag_x = random.uniform(rx_min,
                                       rx_max) * self.center_to_boundary
                cand_x = mag_x * random.choice([-1, 1])

                # GENERATE Y
                # 1. Pick a range (Inner or Outer)
                ry_min, ry_max = random.choice(valid_ranges)
                # 2. Sample magnitude and apply random sign
                mag_y = random.uniform(ry_min,
                                       ry_max) * self.center_to_boundary
                cand_y = mag_y * random.choice([-1, 1])

                collision = False
                for (ex, ey) in env_light_positions[env_idx]:
                    # Euclidean distance check
                    dist = math.hypot(cand_x - ex, cand_y - ey)
                    if dist < min_separation:
                        collision = True
                        break

                if not collision:
                    break  # Found a valid spot

            # Register this spot so the next light in this env avoids it
            env_light_positions[env_idx].append((cand_x, cand_y))

            # Explicit float cast for safety
            final_x = float(cand_x)
            final_y = float(cand_y)
            final_z = float(rand_z_offset)

            new_pos = Gf.Vec3d(final_x, final_y, final_z)

            xform = UsdGeom.Xformable(prim)
            translate_op = None

            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            if translate_op is None:
                translate_op = xform.AddTranslateOp()

            translate_op.Set(new_pos)

            # 4. SHADOW SOFTNESS
            rand_radius = random.uniform(1.0, 2.0)
            prim.GetAttribute("inputs:radius").Set(rand_radius)

            # 5. COLOR TEMPERATURE
            prim.GetAttribute("inputs:enableColorTemperature").Set(True)
            rand_temp = random.uniform(2500.0, 7500.0)
            prim.GetAttribute("inputs:colorTemperature").Set(rand_temp)
            # print(
            #     f"Position of Light = {new_pos}, Temp = {rand_temp}, Intensity = {rand_intensity}, Radius = {rand_radius}"
            # )

    def get_color(self):
        """
        Generate a random pastel color that is not similar to red, green, blue, or pink.
        Returns r, g, b values.
        """
        # Define forbidden colors (Reference points)
        forbidden_colors = [
            np.array([1.0, 0.0, 0.0]),  # Pure Red
            np.array([0.0, 1.0, 0.0]),  # Pure Green
            np.array([0.2, 0.8, 0.2]),  # Lime/Forest Green
            np.array([0.0, 0.0, 1.0]),  # Pure Blue
            np.array([0.8, 0.0, 0.0]),  # Pinkish-red
            np.array([1.0, 0.75, 0.8]),  # Pink
            np.array([1.0, 0.2, 0.6]),  # Pink
            np.array([0.9, 0.1, 0.5]),  # Pink
            np.array([0.95, 0.3, 0.6]),  # Pink
            np.array([0.0, 0.0, 0.0])  # Black
        ]

        # Increased threshold slightly to push colors further away from the forbidden list
        threshold = 0.25

        valid = False
        base = None

        while not valid:
            # Generate random values.
            # Note: For "pastel", values usually shouldn't be too close to 0 (which makes them dark/muddy).
            # We use 0.2 to 0.9 to ensure they aren't pitch black or pure white.
            base = np.array([random.uniform(0.2, 0.9) for _ in range(3)])

            # 1. Euclidean Distance Check
            too_close = any(
                np.linalg.norm(base - fc) < threshold
                for fc in forbidden_colors)

            if too_close:
                continue

            # 2. Dominant Green Check (Strict Green Ban)
            r, g, b = base[0], base[1], base[2]

            # Logic: If Green is the highest value, AND it exceeds the others by a margin, reject it.
            # This kills lime, forest green, grass green, etc.
            is_green_dominant = (g > r + 0.05) and (g > b + 0.05)

            # Logic: If Green is very high (>0.6) and Red/Blue are low (<0.4), it is definitely green.
            is_bright_green = (g > 0.6) and (r < 0.4) and (b < 0.4)

            if is_green_dominant or is_bright_green:
                continue

            valid = True

        return float(base[0]), float(base[1]), float(base[2])

    def write_pose_to_sim(self,
                          env_ids: torch.Tensor,
                          indices: torch.Tensor,
                          goal_default_state: torch.Tensor = None,
                          camera_obj_default_state: torch.Tensor = None,
                          agent_default_state: torch.Tensor = None,
                          vpt_obj_default_state: torch.Tensor = None,
                          device: torch.device = None):
        """Writes the default poses to the simulation for specified environments.
        
        Args:
            env_ids: Tensor of environment IDs to update.
            goal_default_state: Optional tensor of default goal states.
            camera_obj_default_state: Optional tensor of default camera object states.
            agent_default_state: Optional tensor of default agent states.
            vpt_obj_default_state: Optional tensor of default VPT object states.
            device: Device to use for tensor operations.
        """
        if device is None:
            device = self._agent.device

        if goal_default_state is not None:
            goal_poses = goal_default_state[indices, :7]
            self._goal.write_root_com_pose_to_sim(goal_poses, env_ids)
            self._goal.write_root_com_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=device), env_ids)

        if camera_obj_default_state is not None:
            camera_poses = camera_obj_default_state[indices, :7]
            self._camera_obj.write_root_com_pose_to_sim(camera_poses, env_ids)
            self._camera_obj.write_root_com_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=device), env_ids)

        if agent_default_state is not None:
            agent_poses = agent_default_state[indices, :7]
            self._agent.write_root_com_pose_to_sim(agent_poses, env_ids)
            self._agent.write_root_com_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=device), env_ids)

        if vpt_obj_default_state is not None:
            vpt_obj_poses = vpt_obj_default_state[indices, :, :7]
            self._vpt_objects.write_object_pose_to_sim(vpt_obj_poses, env_ids)
            self._vpt_objects.write_object_velocity_to_sim(
                torch.zeros((len(env_ids), self.num_objs, 6), device=device),
                env_ids)

        for _ in range(1):
            self.sim.step()
            self._vpt_objects.update(self.sim.cfg.dt)

    def get_material_configs(self,
                             material_type: str) -> list[sim_utils.MdlFileCfg]:
        """
        Retrieves material paths, selects a subset, and wraps them in MdlFileCfg objects.
        Args:
            material_type (str): "mat" (for floor) or "vpt" (for obstacles).
        """
        # 1. Determine constraints based on type
        if material_type == "mat":
            raw_paths = get_mat_material_paths()  # Imported function
            tex_scale = (10000.0, 10000.0)
        elif material_type == "vpt":
            raw_paths = get_vpt_material_paths()  # Imported function
            tex_scale = (2.0, 2.0)
        else:
            raise ValueError(f"Unknown material type: {material_type}")

        # 2. Limit sample size (Logic preserved: Limit 100)
        num_to_select = min(len(raw_paths), 100)
        selected_paths = random.sample(raw_paths, num_to_select)

        print(
            f"[INFO] {material_type.upper()} Config: {len(selected_paths)} materials selected."
        )

        # 3. Wrap in MdlFileCfg
        configs_list = []
        for path in selected_paths:
            material = sim_utils.MdlFileCfg(
                mdl_path=path,
                project_uvw=True,
                texture_scale=tex_scale,
            )
            configs_list.append(material)

        return configs_list

    def randomize_material(self, prim_paths: list, material_type: str):
        """
        Applies a random material from the pre-loaded pool to the given prims.
        Args:
            prim_paths (list): List of prim paths to modify.
            material_type (str): "mat" or "vpt" to select the correct material pool.
        """
        # Select the correct pool from self
        if material_type == "mat":
            material_pool = self.mat_material_paths
        elif material_type == "vpt":
            material_pool = self.vpt_material_paths
        else:
            print(
                f"⚠️ Unknown material type '{material_type}', skipping randomization."
            )
            return

        if not material_pool:
            return

        for prim in prim_paths:
            rand_material = random.choice(material_pool)
            sim_utils.bind_visual_material(prim, rand_material)

    def get_obb_hitbox(self, env_ids: torch.Tensor,
                       vpt_state: torch.Tensor) -> torch.Tensor:
        """
        Analytically computes world corners for obstacles using cached dims.
        Replaces slow USD stage reads with vectorized tensor math.
        
        Args:
            env_ids: Tensor of environment indices (Batch,)
            vpt_state: Tensor containing poses (Batch, NumObjs, 13) [pos:0-3, rot:3-7]
            
        Returns:
            corners: Tensor (Batch, NumObjs, 8, 3)
        """
        # 1. Setup Signs for the 8 corners of a cube (Broadcasting ready)
        # Shape: (1, 1, 8, 3)
        signs = torch.tensor(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1],
             [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            device=self.device,
            dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 2. Get Dimensions & Calculate Half Extents
        # self.all_vpt_dims: (NumEnvs, NumObjs, 3) -> Select subset (Batch, NumObjs, 3)
        dims = self.all_vpt_dims[env_ids]

        # Apply padding reduction (shrink box slightly for safety)
        padding_reduction = 0.05
        half_extents = (dims / 2.0) - padding_reduction
        # Ensure non-negative dimensions
        half_extents = torch.max(half_extents,
                                 torch.tensor(0.01, device=self.device))

        # 3. Create Local Scaled Corners
        # (Batch, NumObjs, 1, 3) * (1, 1, 8, 3) -> (Batch, NumObjs, 8, 3)
        local_corners = half_extents.unsqueeze(2) * signs

        # 4. Get Rotations & Positions
        # vpt_state is (Batch, NumObjs, ...). Indices 3:7 are (w, x, y, z)
        quats = vpt_state[..., 3:7]
        pos = vpt_state[..., :3]

        # 5. Rotate
        # Convert quats to rotation matrices: (Batch, NumObjs, 3, 3)
        rot_mats = math_utils.matrix_from_quat(quats)

        # Matrix Multiply: (Batch, NumObjs, 8, 3) @ (Batch, NumObjs, 3, 3)^T
        # This rotates the 8 local corner vectors
        rotated_corners = torch.matmul(local_corners,
                                       rot_mats.transpose(-1, -2))

        # 6. Translate
        # Add position: (Batch, NumObjs, 8, 3) + (Batch, NumObjs, 1, 3)
        world_corners = rotated_corners + pos.unsqueeze(2)

        return world_corners

    def update_obb_cache(self, env_ids: torch.Tensor, vpt_state: torch.Tensor):
        """
        Updates the GPU tensor cache for specific environments using the analytic hitbox function.
        
        Args:
            env_ids: Tensor of environment indices to update.
            vpt_state: Tensor containing the CURRENT poses of the objects.
        """
        # 1. Lazy Initialization of the Master Tensor
        if not hasattr(self, "obb_corners_cache"):
            num_envs = self.num_envs
            num_objs = self.cfg.num_vpt_objs
            self.obb_corners_cache = torch.zeros((num_envs, num_objs, 8, 3),
                                                 device=self.device,
                                                 dtype=torch.float32)

        # 2. Compute Analytic Corners
        # This returns corners for ALL objects in the batch (active & inactive)
        corners = self.get_obb_hitbox(env_ids, vpt_state)

        # 3. Update Cache
        # Writes the computed corners directly into the master cache
        self.obb_corners_cache[env_ids] = corners

    def _get_agent_corners(self, pos, quat):
        """
        Generates world-space corners for agents.
        Args:
            pos: (N, 3) agent positions
            quat: (N, 4) agent rotations
        Returns:
            (N, 1, 8, 3) tensor matching the shape structure of the obstacle cache
        """
        # --- CONFIG: AGENT SIZE ---
        # Format: [x_radius, y_radius, z_radius]
        half_extents = torch.tensor([0.1, 0.1, 0.1], device=self.device)

        # Define the 8 local corners of a box (canonical order)
        # Shape: (8, 3)
        signs = torch.tensor(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],  # Bottom
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1]  # Top
            ],
            device=self.device)

        # Scale unit cube by half_extents -> (8, 3)
        local_corners = half_extents * signs

        # Get Rotation Matrices: (N, 3, 3) -> expand to (N, 1, 3, 3)
        # REPLACEMENT HERE:
        rot_mat = math_utils.matrix_from_quat(quat).unsqueeze(1)

        # Rotate Corners: R * local
        # We treat local_corners as (1, 1, 8, 3) for broadcasting
        # Matmul: (N, 1, 8, 3) x (N, 1, 3, 3)^T
        rotated_corners = torch.matmul(
            local_corners.unsqueeze(0).unsqueeze(0), rot_mat.transpose(-1, -2))

        # Translate: (N, 1, 8, 3) + (N, 1, 1, 3)
        world_corners = rotated_corners + pos.unsqueeze(1).unsqueeze(2)

        return world_corners

    def _check_collisions_vectorized(self, env_ids, proposed_pos,
                                     proposed_quat):
        """
        Pure 2D SAT implementation (XY Plane only) + Wall Boundary Check.
        Ignores Z-height entirely.
        Returns: Boolean Tensor (N,) where True = Collision Detected.
        """
        # 1. Get Agent Corners at PROPOSED position
        # Shape: (N, 1, 8, 3)
        agent_corners = self._get_agent_corners(proposed_pos, proposed_quat)

        # Slice to keep only X, Y for SAT and Bounds check
        ac_xy = agent_corners[..., :2]  # (N, 1, 8, 2)

        # --- A. WALL BOUNDARY CHECK ---
        # Get origins for these environments
        # self.scene.env_origins is usually (NumEnvs, 3), we need (N, 2)
        env_origins_xy = self.scene.env_origins[env_ids][:, :2]

        # Broadcast origins to match corners shape: (N, 1, 1, 2)
        origins_expanded = env_origins_xy.unsqueeze(1).unsqueeze(1)

        limit = self.center_to_boundary

        # Check if any corner is outside [origin - limit, origin + limit]
        # logic: coord < min OR coord > max
        min_bounds = origins_expanded - limit
        max_bounds = origins_expanded + limit

        # (N, 1, 8, 2) boolean mask
        out_of_bounds = (ac_xy < min_bounds) | (ac_xy > max_bounds)

        # If ANY corner, in ANY dimension (x or y) is out, it's a wall collision
        # Collapse all dimensions except batch (N)
        is_colliding_wall = out_of_bounds.any(dim=-1).any(dim=-1).squeeze(
            -1)  # (N,)

        # --- B. OBJECT COLLISION CHECK (SAT) ---
        # 2. Get Obstacle Corners
        # Shape: (N, NumObs, 8, 3)
        obs_corners = self.obb_corners_cache[env_ids]
        num_obs = obs_corners.shape[1]

        oc_xy = obs_corners[..., :2]  # (N, NumObs, 8, 2)

        # Generate Axes (Normals)
        # Agent Axes: Normal to edge 0->1 and 0->3
        a_edge1 = ac_xy[..., 1, :] - ac_xy[..., 0, :]
        a_edge2 = ac_xy[..., 3, :] - ac_xy[..., 0, :]
        a_axes = torch.stack([a_edge1, a_edge2], dim=2)  # (N, 1, 2, 2)
        a_axes = a_axes / (torch.norm(a_axes, dim=-1, keepdim=True) + 1e-6)

        # Obstacle Axes
        o_edge1 = oc_xy[..., 1, :] - oc_xy[..., 0, :]
        o_edge2 = oc_xy[..., 3, :] - oc_xy[..., 0, :]
        o_axes = torch.stack([o_edge1, o_edge2], dim=2)  # (N, NumObs, 2, 2)
        o_axes = o_axes / (torch.norm(o_axes, dim=-1, keepdim=True) + 1e-6)

        # Combine Axes: (N, NumObs, 4, 2)
        all_axes = torch.cat([a_axes.expand(-1, num_obs, -1, -1), o_axes],
                             dim=2)

        # --- Projection ---
        # Project all 8 corners onto the 4 axes (2 from agent, 2 from obs)
        # (N, 1, 8, 2) @ (N, NumObs, 2, 4) -> (N, NumObs, 8, 4)
        axes_T = all_axes.transpose(-1, -2)
        proj_a = torch.matmul(ac_xy.expand(-1, num_obs, -1, -1), axes_T)
        proj_o = torch.matmul(oc_xy, axes_T)

        # --- Overlap Test ---
        # Get Min/Max of projection intervals
        min_a = proj_a.min(dim=2).values  # (N, NumObs, 4)
        max_a = proj_a.max(dim=2).values
        min_o = proj_o.min(dim=2).values
        max_o = proj_o.max(dim=2).values

        # SAT Condition: Intervals overlap if MaxA >= MinB AND MaxB >= MinA
        overlap_axes = (max_a >= min_o) & (max_o >= min_a)  # (N, NumObs, 4)

        # Collision = Overlap on ALL 4 axes (Pure 2D)
        is_colliding_2d = overlap_axes.all(dim=2)  # (N, NumObs)

        # --- COMBINE CHECKS ---
        # Return True if agent collides with ANY object OR the wall
        return is_colliding_2d.any(dim=1) | is_colliding_wall

    def debug_plot_analytic_obbs(self, vpt_state):
        """
        Analytically computes OBB corners from state and plots Env 0.
        Args:
            vpt_state: Tensor (num_envs, num_objs, 7) -> [x, y, z, qx, qy, qz, qw]
                       Note: Ensure quat order matches your math_utils (xyzw vs wxyz).
        """

        # 1. Setup (Unit Cube Corners)
        # Shape: (8, 3)
        signs = torch.tensor(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1],
             [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            device=self.device,
            dtype=torch.float32)

        # 2. Select Env 0 for Plotting
        env_id = 0

        # Get Active Indices & Dims for Env 0
        # Assuming self.active_vpt_indices is a list of lists or similar structure
        active_indices = self.active_vpt_indices[env_id]
        if len(active_indices) == 0:
            print("No active objects in Env 0 to plot.")
            return

        # Get Dimensions for active objects: (NumActive, 3)
        # self.all_vpt_dims is likely (NumEnvs, NumObjs, 3)
        dims = self.all_vpt_dims[env_id, active_indices, :3]
        half_extents = dims / 2.0

        # 3. Get Pose & Quat from Input State
        # vpt_state is (NumEnvs, NumObjs, 7)
        # positions: (NumActive, 3)
        positions = vpt_state[env_id, active_indices, :3]
        # quats: (NumActive, 4)
        quats = vpt_state[env_id, active_indices, 3:7]
        # print(f"Quats = {quats}")

        # 4. Analytic Calculation
        # A. Local Scaled Corners: (NumActive, 1, 3) * (1, 8, 3) -> (NumActive, 8, 3)
        local_corners = half_extents.unsqueeze(1) * signs.unsqueeze(0)

        # B. Rotation Matrix: (NumActive, 3, 3)
        rot_mats = math_utils.matrix_from_quat(quats)

        # C. Rotate: (NumActive, 8, 3) @ (NumActive, 3, 3)^T
        # We assume local_corners is row vectors, so we multiply by Transpose of Rot
        rotated_corners = torch.matmul(local_corners,
                                       rot_mats.transpose(-1, -2))

        # D. Translate: (NumActive, 8, 3) + (NumActive, 1, 3)
        world_corners = rotated_corners + positions.unsqueeze(1)

        # 5. Plotting (XY Plane Projection)
        corners_np = world_corners.cpu().numpy()
        pos_np = positions.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_title(f"Analytic OBB Verification (Env {env_id})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Plot Env Boundary (assuming center_to_boundary is defined)
        limit = self.center_to_boundary
        boundary = plt.Rectangle((-limit, -limit),
                                 2 * limit,
                                 2 * limit,
                                 fill=False,
                                 linestyle='--',
                                 color='k')
        ax.add_patch(boundary)

        # Plot Objects
        for i in range(len(active_indices)):
            c = corners_np[i]  # (8, 3)

            # Draw Base (First 4 points are usually bottom face in canonical order)
            # Connecting 0-1, 1-2, 2-3, 3-0
            base_idx = [0, 1, 2, 3, 0]
            ax.plot(c[base_idx, 0], c[base_idx, 1], 'b-', alpha=0.7)

            # Draw Top (Next 4 points)
            top_idx = [4, 5, 6, 7, 4]
            ax.plot(c[top_idx, 0], c[top_idx, 1], 'r-', alpha=0.7)

            # Draw Centroid
            ax.plot(pos_np[i, 0], pos_np[i, 1], 'ko', markersize=3)

        plt.grid(True)
        # Save Plot
        output_filename = f"debug_obb_env_{env_id}.png"
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved analytic OBB plot to: {output_filename}")

        plt.close(fig)  # Close to release memory

    def place_agent_safely(self, env_ids: torch.Tensor,
                           agent_state: torch.Tensor, vpt_state: torch.Tensor,
                           safe_range: float) -> torch.Tensor:
        """
        Updates the obstacle cache and finds a collision-free spawn pose for the agent 
        using vectorized batch rejection sampling ("Shotgun" method).

        Args:
            env_ids (torch.Tensor): Tensor of environment indices involved in this spawn.
            agent_state (torch.Tensor): The agent state tensor to update (N, 13).
            vpt_state (torch.Tensor): The obstacle state tensor (N, NumObjs, 13).
            safe_range (float): Maximum +/- deviation from the environment origin.

        Returns:
            torch.Tensor: The updated agent_state with valid positions and random orientations.
        """
        device = self.device
        num_envs = len(env_ids)
        BATCH_SIZE = 1000

        # 1. Update Obstacle Cache
        # Crucial: The physics engine hasn't stepped yet, so we must manually compute
        # where the obstacles are based on the tensors we just generated.
        self.update_obb_cache(env_ids, vpt_state)

        # 2. Fetch Origins
        env_origins = self.scene.env_origins[env_ids]

        # 3. Generate Batch Candidates (Pos & Rot)
        # Shape: (NumEnvs, BatchSize, ...)
        rand_xy = sample_uniform(-safe_range, safe_range,
                                 (num_envs, BATCH_SIZE, 2), device)
        rand_yaw = sample_uniform(0, 2 * math.pi, (num_envs, BATCH_SIZE),
                                  device)

        cands_pos = torch.zeros((num_envs, BATCH_SIZE, 3), device=device)
        cands_pos[..., :2] = env_origins[:, :2].unsqueeze(1) + rand_xy
        cands_pos[..., 2] = agent_state[0, 2]  # Preserve original height

        zeros = torch.zeros_like(rand_yaw)
        cands_quat = quat_from_euler_xyz(zeros, zeros, rand_yaw)

        # 4. Vectorized Collision Check
        # Flatten batch dims for the check function: (N * BatchSize)
        flat_ids = env_ids.repeat_interleave(BATCH_SIZE)
        flat_cols = self._check_collisions_vectorized(
            flat_ids, cands_pos.reshape(-1, 3),
            cands_quat.reshape(-1, 4)).reshape(num_envs, BATCH_SIZE)

        # 5. Select First Valid Candidate
        # argmax on 'False' (0) gives index 0 if all are True (collision).
        # So we cast ~collisions to int (Safe=1) and find first 1.
        valid_indices = torch.argmax((~flat_cols).int(), dim=1)

        # Fallback check: Did we actually find a safe spot?
        # If collision at [idx] is True, we failed.
        unsafe_mask = torch.gather(flat_cols, 1,
                                   valid_indices.unsqueeze(1)).squeeze(1)
        if unsafe_mask.any():
            print(
                f"⚠️ Shotgun Fail: {unsafe_mask.sum().item()} envs using unsafe fallback."
            )

        # 6. Extract Winners & Update State
        batch_idx = torch.arange(num_envs, device=device)
        agent_state[:, :3] = cands_pos[batch_idx, valid_indices]
        agent_state[:, 3:7] = cands_quat[batch_idx, valid_indices]

        return agent_state

    def calc_shape_scale_data(self, env_ids: list[int]):
        """
        Calculates scale/Z, updates self.* variables, returns apply data.
        """

        if not hasattr(
                self,
                "all_vpt_dims") or self.all_vpt_dims.shape[0] != self.num_envs:
            self.all_vpt_dims = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs, 3), device=self.device)
            self.vpt_obj_default_state = torch.zeros(
                (self.num_envs, self.num_objs, 3), device=self.device)

        if not hasattr(
                self,
                "vpt_z_offset_ratios") or self.vpt_z_offset_ratios.shape != (
                    self.num_envs, self.cfg.num_vpt_objs):
            self.vpt_z_offset_ratios = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs), device=self.device)

        if not hasattr(self, "vpt_shapes") or self.vpt_shapes.shape != (
                self.num_envs, self.cfg.num_vpt_objs):
            self.vpt_shapes = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs), device=self.device)

        updates = []
        # Pre-fetch configs
        obj_cfgs = list(self.cfg.vpt_objects.rigid_objects.values())
        max_objs = min(32, len(obj_cfgs))

        for env_idx in env_ids:
            for obj_idx in range(max_objs):
                spawn_cfg = obj_cfgs[obj_idx].spawn
                base = self.vpt_base_dims[obj_idx].cpu().numpy()

                # Defaults
                s, z_pos, z_mult, shape_id = Gf.Vec3d(1, 1, 1), 0.0, 0.0, -1

                # --- 1. USD FILES ---
                if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                    fname = spawn_cfg.usd_path.split("/")[-1].split(".")[0]

                    if fname.endswith(('L', 'Table_B', 'Bench')):
                        f = random.uniform(0.5, 3.0) if fname.endswith(
                            'L') else random.uniform(1.0, 3.0)
                        s = Gf.Vec3d(base[0] * f, base[1] * f, base[2] * f)
                    elif fname.endswith('Table_A'):
                        s = Gf.Vec3d(base[0] * random.uniform(0.5, 3.0),
                                     base[1] * random.uniform(0.5, 3.0),
                                     base[2] * random.uniform(1.0, 3.0))
                    elif fname.endswith(('X', 'A', 'H', 'I', 'Z')):
                        xz, y = random.uniform(0.5,
                                               3.0), random.uniform(0.5, 3.0)
                        s = Gf.Vec3d(base[0] * xz, base[1] * y, base[2] * xz)
                    else:  # Generic
                        xy, z = random.uniform(0.5,
                                               2.5), random.uniform(0.5, 2.5)
                        s = Gf.Vec3d(base[0] * xy, base[1] * xy, base[2] * z)

                # --- 2. PRIMITIVES ---
                elif isinstance(spawn_cfg, sim_utils.MeshCuboidCfg):
                    shape_id, z_mult = 2, 0.5
                    s = Gf.Vec3d(random.uniform(0.5, 2.5),
                                 random.uniform(0.5, 2.5),
                                 random.uniform(0.5, 2.5))

                elif isinstance(
                        spawn_cfg,
                    (sim_utils.MeshCylinderCfg, sim_utils.MeshConeCfg)):
                    shape_id = 3 if isinstance(
                        spawn_cfg, sim_utils.MeshCylinderCfg) else 4
                    z_mult = 0.5 if shape_id == 3 else 0.0
                    sr, sh = random.uniform(0.75,
                                            1.0), random.uniform(0.75, 2.5)
                    s = Gf.Vec3d(sr, sr, sh)

                # Final Z Calculation
                z_pos = (base[2] * s[2]) * z_mult

                # --- UPDATE SELF STATE ---
                self.vpt_obj_default_state[env_idx, obj_idx, 2] = z_pos
                self.vpt_z_offset_ratios[env_idx, obj_idx] = z_mult
                self.vpt_shapes[env_idx, obj_idx] = shape_id
                self.all_vpt_dims[env_idx, obj_idx] = torch.tensor(
                    [base[0] * s[0], base[1] * s[1], base[2] * s[2]],
                    device=self.device)

                updates.append({
                    "prim_path": f"/World/envs/env_{env_idx}/obs_{obj_idx}",
                    "scale": s,
                    "z_pos": z_pos
                })
        return updates

    def calc_shape_color_data(self, env_ids: list[int], random_roughness: bool,
                              random_metallic: bool):
        """
        Calculates random colors/materials.
        """
        updates = []
        for env_idx in env_ids:
            for obj_idx in range(32):
                # Fast path assumption based on your correction
                updates.append({
                    "shader_path":
                    f"/World/envs/env_{env_idx}/obs_{obj_idx}/geometry/material/Shader",
                    "diffuse_color":
                    Gf.Vec3f(self.get_color()),
                    "roughness":
                    random.random() if random_roughness else None,
                    "metallic":
                    random.random() if random_metallic else None
                })
        return updates

    def calc_spherical_light_data(self, env_ids: list[int],
                                  random_light_off: bool):
        """
        Calculates light params.
        """
        updates = []
        ranges = [(0.1, 0.3), (0.7, 0.9)]

        for env_idx in env_ids:
            # Off Logic
            if random_light_off and random.random() < 0.5:
                updates.append({
                    "light_path": f"/World/envs/env_{env_idx}/Light_A",
                    "intensity": 0.0
                })
                continue

            # Coordinate Logic
            get_c = lambda: random.uniform(*random.choice(
                ranges)) * self.center_to_boundary * random.choice([-1, 1])

            updates.append({
                "light_path":
                f"/World/envs/env_{env_idx}/Light_A",
                "translate":
                Gf.Vec3d(float(get_c()), float(get_c()), float(random.uniform(7.5, 15.0))),
                "intensity":
                random.uniform(40_000.0, 75_000.0),
                "radius":
                random.uniform(1.0, 2.0),
                "temp":
                random.uniform(2500.0, 7500.0)
            })
        return updates

    def apply_randomizations(self,
                             env_ids: list[int],
                             randomize_scale: bool = True,
                             randomize_color: bool = False,
                             randomize_light: bool = True):
        """
        Main wrapper. Calls calcs, then applies in ONE Sdf.ChangeBlock.
        """
        # 1. GENERATE DATA
        scale_data = self.calc_shape_scale_data(
            env_ids) if randomize_scale else []
        color_data = self.calc_shape_color_data(
            env_ids, True, True) if randomize_color else []
        light_data = self.calc_spherical_light_data(
            env_ids, False) if randomize_light else []

        stage = get_current_stage()
        layer = stage.GetRootLayer()

        # 2. APPLY USD CHANGES
        with Sdf.ChangeBlock():

            # --- SCALE & GEOMETRY ---
            for d in scale_data:
                prim = stage.GetPrimAtPath(d['prim_path'])
                if not prim.IsValid(): continue

                xform = UsdGeom.Xformable(prim)
                scale_op, translate_op = None, None

                for op in xform.GetOrderedXformOps():
                    t = op.GetOpType()
                    if t == UsdGeom.XformOp.TypeScale: scale_op = op
                    elif t == UsdGeom.XformOp.TypeTranslate: translate_op = op

                if not scale_op: scale_op = xform.AddScaleOp()
                if not translate_op: translate_op = xform.AddTranslateOp()

                scale_op.Set(d['scale'])

                # Fetch CURRENT X/Y, set NEW Z
                cur = translate_op.Get()
                translate_op.Set(Gf.Vec3d(cur[0], cur[1], d['z_pos']))

            # --- COLOR (Sdf Spec Logic) ---
            for d in color_data:
                # Create Prim Spec in Layer (Fast Path)
                shader_spec = Sdf.CreatePrimInLayer(layer, d['shader_path'])

                # Helper to set attr default
                def _set(name, val, type_name):
                    attr = shader_spec.GetAttributeAtPath(
                        shader_spec.path.AppendProperty(name))
                    if not attr:
                        attr = Sdf.AttributeSpec(shader_spec, name, type_name)
                    attr.default = val

                _set("inputs:diffuseColor", d['diffuse_color'],
                     Sdf.ValueTypeNames.Color3f)

                if d['roughness'] is not None:
                    _set("inputs:roughness", d['roughness'],
                         Sdf.ValueTypeNames.Float)
                if d['metallic'] is not None:
                    _set("inputs:metallic", d['metallic'],
                         Sdf.ValueTypeNames.Float)

            # --- LIGHTS ---
            for d in light_data:
                prim = stage.GetPrimAtPath(d['light_path'])
                if not prim.IsValid(): continue

                prim.GetAttribute("inputs:intensity").Set(d['intensity'])

                if d['intensity'] > 0.0:
                    # Transform
                    xform = UsdGeom.Xformable(prim)
                    for op in xform.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            op.Set(d['translate'])
                            break

                    prim.GetAttribute("inputs:radius").Set(d['radius'])
                    prim.GetAttribute("inputs:enableColorTemperature").Set(
                        True)
                    prim.GetAttribute("inputs:colorTemperature").Set(d['temp'])