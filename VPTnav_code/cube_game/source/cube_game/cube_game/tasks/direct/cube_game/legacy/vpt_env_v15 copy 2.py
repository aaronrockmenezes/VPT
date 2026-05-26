from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import random
import numpy as np
import os
import cv2
from typing import List, Dict, Tuple, Optional
from PIL import Image
import json
import time
import sys
from shapely.geometry import Polygon, box
from shapely import affinity
from shapely.geometry import Point

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.api import World
from pxr import Gf, Sdf, UsdGeom, Usd, UsdLux

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, RayCaster, save_images_to_file, Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, sample_gaussian, quat_from_euler_xyz
from isaaclab.utils import math as math_utils

from .vpt_env_cfg_v15 import VPTEnvCfg
from .spawn_boundary import get_vpt_material_paths, get_mat_material_paths


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
        self.verbose = 2
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
        self.base_path = f"/home/arock3/data_v15_24_mbc/data_{self.GPU_ID}"
        # self.base_path = "/media/data_cifs_lrs/projects/prj_robotics/VPTnav_v6_1k_envs"
        self.visibility_labels_json_path = f"{self.base_path}/visibility_labels.json"

        # Mode determination
        if self.config_file is not None and os.path.exists(self.config_file):
            self.mode = "testing"
        else:
            self.mode = "data_collection"

        self.total_envs_to_sim = 550
        self.slot_to_env_id = list(range(self.num_envs))
        self.next_env_id = self.num_envs
        self.completed_envs = set()
        self.slot_attempt_counts = [0] * self.num_envs
        self.max_attempts_per_slot = 20 * 50  # Full resets * Inner resets

        self.used_vpt_objects = set()
        self._preallocate_visibility_labels()

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

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["semantic_tiled_camera"] = self._tiled_camera

        self._rgb_tiled_camera = TiledCamera(self.cfg.rgb_tiled_camera)
        self.scene.sensors["rgb_tiled_camera"] = self._rgb_tiled_camera

        self._distance_tiled_camera = TiledCamera(
            self.cfg.distance_tiled_camera)
        self.scene.sensors[
            "distance_tiled_camera"] = self._distance_tiled_camera

        self._occlusion_camera = TiledCamera(self.cfg.occlusion_camera)
        self.scene.sensors["occlusion_camera"] = self._occlusion_camera

        light_cfg = sim_utils.DistantLightCfg(intensity=1000.0,
                                              color=(0.75, 0.75, 0.75))
        light_cfg_a = sim_utils.DistantLightCfg(intensity=1000.0,
                                              color=(0.75, 0.75, 0.75))
        light_cfg_b = sim_utils.DistantLightCfg(intensity=1000.0,
                                              color=(0.75, 0.75, 0.75))
        light_cfg_c = sim_utils.DistantLightCfg(intensity=1000.0,
                                              color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        light_cfg_a.func("/World/Light_A", light_cfg_a)
        light_cfg_b.func("/World/Light_B", light_cfg_b)
        light_cfg_c.func("/World/Light_C", light_cfg_c)

        self.mat_material_paths = []
        self.vpt_material_paths = []
        self.mat_material_configs = self.get_mat_material_configs()
        self.vpt_material_configs = self.get_vpt_material_configs()
        for idx, material in enumerate(self.mat_material_configs):
            material.func(f"/World/Looks/mat_material_{idx}", material)
            self.mat_material_paths.append(f"/World/Looks/mat_material_{idx}")
        
        for idx, material in enumerate(self.vpt_material_configs):
            material.func(f"/World/Looks/vpt_material_{idx}", material)
            self.vpt_material_paths.append(f"/World/Looks/vpt_material_{idx}")
        
        print("-"*50)
        print(self.mat_material_paths)
        print(self.vpt_material_paths)
        print("-"*50)

    def check_batch_object_visibility(
            self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Check object visibility for a batch of environments in parallel."""
        sem_imgs = self._tiled_camera.data.output["semantic_segmentation"][env_ids]
        
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
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        # Ensure env_ids is a tensor
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_envs = len(env_ids)
        device = self.device  # Assuming self.device is available directly
        
        # --- 1. Separate Physics Actions (0-4) from Resets (5-6) ---
        # Masks for different action types
        reset_mask_5 = (actions == 5)
        reset_mask_6 = (actions == 6)
        physics_mask = ~(reset_mask_5 | reset_mask_6) # Actions 0, 1, 2, 3, 4

        # --- 2. Handle Resets (Actions 5 & 6) ---
        if reset_mask_5.any():
            self.reset(env_ids[reset_mask_5])
            
        if reset_mask_6.any():
            self.reset(env_ids[reset_mask_6], rl_reset=True)

        # If no agents need physics updates, we can exit early or just skip the block
        if not physics_mask.any():
            return

        # Filter ids and actions for physics updates
        phys_ids = env_ids[physics_mask]
        phys_actions = actions[physics_mask]
        num_phys = len(phys_ids)

        # --- 3. Vectorized Physics Calculation (Actions 0-4) ---
        # Fetch current states
        current_pos = self._agent.data.root_pos_w[phys_ids].clone()
        current_quat = self._agent.data.root_quat_w[phys_ids].clone()

        # Calculate "Upright" Quaternion (remove pitch/roll)
        w, z = current_quat[:, 0], current_quat[:, 3]
        magnitude = torch.sqrt(w**2 + z**2)
        
        upright_quat = current_quat.clone()
        upright_quat[:, 0] = w / magnitude
        upright_quat[:, 1] = 0.0
        upright_quat[:, 2] = 0.0
        upright_quat[:, 3] = z / magnitude

        # Prepare outputs
        new_quat = upright_quat.clone()
        desired_vel = torch.zeros((num_phys, 6), device=device)
        max_velocity = 3.0

        # A. Handle Rotations (Actions 2 & 3)
        # 2: Left, 3: Right
        theta = math.pi / 12
        half_theta = theta / 2
        
        # Create a batch of identity quats [1, 0, 0, 0]
        rot_quats = torch.zeros((num_phys, 4), device=device)
        rot_quats[:, 0] = 1.0 

        # Mask for rotations
        mask_left = (phys_actions == 2)
        mask_right = (phys_actions == 3)

        if mask_left.any():
            rot_quats[mask_left] = torch.tensor(
                [math.cos(half_theta), 0.0, 0.0, math.sin(half_theta)], device=device
            )
        if mask_right.any():
            rot_quats[mask_right] = torch.tensor(
                [math.cos(half_theta), 0.0, 0.0, -math.sin(half_theta)], device=device
            )

        # Apply rotation only where needed (optimization: skip if no rotations)
        if mask_left.any() or mask_right.any():
            new_quat = math_utils.quat_mul(upright_quat, rot_quats)

        # B. Handle Movement (Actions 0 & 1)
        # 0: Forward, 1: Backward
        local_move = torch.zeros((num_phys, 3), device=device)
        
        mask_fwd = (phys_actions == 0)
        mask_bwd = (phys_actions == 1)
        
        local_move[mask_fwd, 0] = 1.0
        local_move[mask_bwd, 0] = -1.0

        # Calculate world velocity
        # Only compute for moving agents to save compute
        moving_mask = mask_fwd | mask_bwd
        if moving_mask.any():
            world_vel_3d = math_utils.quat_apply(upright_quat[moving_mask], local_move[moving_mask])
            desired_vel[moving_mask, :3] = world_vel_3d * max_velocity

        # C. Apply Z-axis correction (keep height constant)
        current_pos[:, 2] = self._agent.data.default_root_state[phys_ids, 2]

        # --- 4. Apply State to Simulation ---
        self._agent.write_root_com_pose_to_sim(
            torch.cat([current_pos, new_quat], dim=1), phys_ids)
        self._agent.write_root_com_velocity_to_sim(desired_vel, phys_ids)
        
        # Note: Original code called reset() here. 
        # If that was a "flush state" call, keep it. 
        # If it was a "reset env" call, it might be redundant for phys_ids.
        self._agent.reset() 

        # --- 5. Update Camera / Occlusion ---
        self._update_camera_poses(phys_ids)

    def _update_camera_poses(self, env_ids):
        """Helper to handle the camera/occlusion update logic."""
        camera_obj_pos = self._camera_obj.data.root_pos_w[env_ids].clone()
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_ids].clone()
        
        # Rotate 90 degrees left
        half_theta = (math.pi / 2) / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta), 0.0, 0.0, math.sin(half_theta)],
            device=self.device
        )
        
        rotated_orientations = math_utils.quat_mul(
            camera_obj_quat, left_90_quat.expand(len(env_ids), -1)
        )
        
        self._occlusion_camera.set_world_poses(
            positions=camera_obj_pos,
            orientations=rotated_orientations,
            env_ids=env_ids.tolist(),
            convention="world"
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.move_agent(self.actions)

        env_ids = self._agent._ALL_INDICES
        camera_pose = self._camera_obj.data.root_pose_w[env_ids].clone()

        self._camera_obj.write_root_pose_to_sim(camera_pose[:, :7], env_ids)
        self._camera_obj.write_root_velocity_to_sim(
            torch.zeros_like(self._camera_obj.data.root_vel_w[env_ids]),
            env_ids)

    def _get_observations(self, mode=None) -> dict:
        if not self._reset_called:
            raise RuntimeError(
                "ERROR: _get_observations called before _reset_idx! "
                "Environment initialization must call reset first.")
        if mode is None:
            mode = self.mode

        env_ids = self._agent._ALL_INDICES
        device = self._agent.device

        if mode == "testing":
            env_ids = self._agent._ALL_INDICES
            env_id_item = env_ids[0].item()

            # Write all objects to their place
            self._goal.write_root_pose_to_sim(
                self._goal.data.default_root_state[env_ids], env_ids)
            self._agent.write_root_pose_to_sim(
                self._agent.data.default_root_state[env_ids], env_ids)
            self._vpt_objects.write_object_com_pose_to_sim(
                self._vpt_objects.data.default_root_state[env_ids], env_ids)

            if self.selected_viewpoints_for_collection[
                    env_id_item] is not None:
                viewpoints = self.selected_viewpoints_for_collection[
                    env_id_item]
                for viewpoint_idx in range(len(viewpoints)):
                    print(
                        f"  📍 Testing viewpoint {viewpoint_idx + 1}/{len(viewpoints)}..."
                    )
                    target_pos = viewpoints[viewpoint_idx]
                    target_pos_3d = torch.zeros(3, device=self.device)
                    target_pos_3d[:2] = target_pos[:2]
                    target_pos_3d[2] = self._agent.data.default_root_state[
                        env_ids[0], 2]
                    camera_pos_3d = self._camera_obj.data.root_pos_w[
                        env_ids[0]]
                    goal_pos_3d = self._goal.data.root_pos_w[env_ids[0]]
                    midpoint = (camera_pos_3d + goal_pos_3d) / 2.0
                    direction = midpoint[:2] - target_pos[:2]
                    yaw = torch.atan2(
                        direction[1], direction[0]) if torch.norm(
                            direction) > 1e-6 else torch.tensor(0.0)
                    quat = torch.tensor([
                        math.cos(yaw.item() / 2), 0.0, 0.0,
                        math.sin(yaw.item() / 2)
                    ],
                                        device=self.device)
                    pose = torch.cat([target_pos_3d, quat])
                    self._agent.write_root_com_pose_to_sim(
                        pose.unsqueeze(0), env_ids)
                    for _ in range(3):
                        self.sim.step()
                        self._rgb_tiled_camera.update(self.sim.cfg.dt)
                    rgb_data = self._rgb_tiled_camera.data.output["rgb"]
                    rgb_data = rgb_data.permute(0, 3, 1, 2)[:, :3, :, :]
                return {"policy": rgb_data.clone()}
            else:
                print(
                    f"⚠️  Env {env_id_item}: No selected viewpoints for testing mode"
                )
                rgb_data = self._rgb_tiled_camera.data.output["rgb"]
                rgb_data = rgb_data.permute(0, 3, 1, 2)[:, :3, :, :]
                return {"policy": rgb_data.clone()}

        # Update cameras for normal observation
        for _ in range(3):
            self.sim.step()
            self._rgb_tiled_camera.update(self.sim.cfg.dt)
            self._distance_tiled_camera.update(self.sim.cfg.dt)
            self._occlusion_camera.update(self.sim.cfg.dt)

        rgb_data = self._rgb_tiled_camera.data.output["rgb"]
        rgb_data = rgb_data.permute(0, 3, 1, 2)[:, :3, :, :]
        observations = {"policy": rgb_data.clone()}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs,
                                 dtype=torch.bool,
                                 device=self.device)
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

        if pos_diff > 0.01:
            print(
                f"⚠️  Env {env_id}: Occlusion camera misaligned! Distance: {pos_diff:.4f}"
            )
            print(f"    Camera Obj: {camera_obj_pos.cpu().numpy()}")
            print(f"    Occlusion Cam: {occlusion_cam_pos.cpu().numpy()}")
            print(f"    Forcing update...")

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
            print(f"    ✓ After fix: Distance = {new_pos_diff:.4f}")

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

        if self.verbose >= 2:
            print(f"  💾 Saved config: {config_filepath}")
            print(
                f"     Active VPT objects: {self.active_vpt_objs}/{self.num_objs}"
            )


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

            if self.verbose >= 2:
                print(
                    f"  🎲 Env {env_id_item}: Selected {self.active_vpt_objs} active VPT indices from {self.num_objs} total"
                )

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

            # if self.verbose >= 2:
            #     print(f"  📦 Env {env_id_item}: Stored {len(inactive_indices)} inactive VPT objects at {self.storage_position.cpu().numpy()}")

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
            if len(self.active_vpt_indices) > 0 and isinstance(self.active_vpt_indices[0], torch.Tensor):
                active_indices_tensor = torch.stack(self.active_vpt_indices).to(self.device)
            else:
                active_indices_tensor = torch.tensor(self.active_vpt_indices, device=self.device, dtype=torch.long)
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

    def _get_active_vpt_positions(self, env_ids, base_pivoted: bool = False, return_full_pose: bool = False) -> torch.Tensor:
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
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        if env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)

        # --- 2. Standardize active_vpt_indices to Tensor ---
        if isinstance(self.active_vpt_indices, list):
            if len(self.active_vpt_indices) > 0 and isinstance(self.active_vpt_indices[0], torch.Tensor):
                active_indices_full = torch.stack(self.active_vpt_indices).to(self.device)
            else:
                active_indices_full = torch.tensor(self.active_vpt_indices, device=self.device, dtype=torch.long)
        else:
            active_indices_full = self.active_vpt_indices.to(self.device)

        # --- 3. Get active indices for the requested batch ---
        batch_active_indices = active_indices_full[env_ids]

        # --- 4. Advanced Indexing Setup ---
        # Expand env_ids: [batch_size, 1] -> [batch_size, num_active_per_env]
        env_ids_expanded = env_ids.unsqueeze(1).expand_as(batch_active_indices)
        
        # --- 5. Fetch Positions ---
        active_positions = self._vpt_objects.data.object_pos_w[env_ids_expanded, batch_active_indices].clone()

        # Apply Base Pivot Adjustment (modifies Z)
        if base_pivoted:
            # Get the SCALED heights we stored in randomize_shape_scale
            active_heights = self.all_vpt_dims[env_ids_expanded, batch_active_indices, 2]
            
            # Get the exact ratio used during placement (0.0 or 0.5)
            active_ratios = self.vpt_z_offset_ratios[batch_active_indices]
            
            # Calculate adjustment: Center (Sim Pos) -> Base
            z_adjustment = active_heights * active_ratios
            
            active_positions[:, :, 2] -= z_adjustment

        # --- 6. Return Logic ---
        if return_full_pose:
            # Fetch Quaternions: [batch, num_active, 4] (w, x, y, z)
            active_quats = self._vpt_objects.data.object_quat_w[env_ids_expanded, batch_active_indices].clone()
            
            # Concatenate: [batch, num_active, 3] + [batch, num_active, 4] -> [batch, num_active, 7]
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
            if self.verbose >= 2:
                print(
                    f"    ✅ Slot {env_id}: Selected {self.images_per_env} viewpoints for collection"
                )
            return True
        else:
            if self.verbose >= 2:
                print(
                    f"    ⚠️  Slot {env_id}: Only {len(selected_points)} viewpoints available (need {self.images_per_env})"
                )
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
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # --- 1. Lazy Initialization of Slot State (Run Once) ---
        if not hasattr(self, "slot_folder_indices"):
            num_all_envs = self.num_envs
            self.slot_folder_indices = [self.next_env_folder_idx + i for i in range(num_all_envs)]
            self.slot_visibility_categories = []
            self.slot_attempt_counts = [0] * num_all_envs
            
            if self.verbose >= 1:
                print("🔒 Initializing slot states and visibility labels...")
            for i in range(num_all_envs):
                cat = self._assign_next_visibility_label(self.slot_folder_indices[i])
                self.slot_visibility_categories.append(cat)
            
            self._save_visibility_labels()

        # --- 2. Randomize Scene (Procedural Generation) ---
        self._randomize_scene_props(env_ids)

        # --- 3. Prepare Batch Data for Reset ---
        reset_folder_indices = []
        reset_visibility_categories = []
        active_slots = env_ids.tolist()

        for slot_idx in active_slots:
            reset_folder_indices.append(self.slot_folder_indices[slot_idx])
            reset_visibility_categories.append(self.slot_visibility_categories[slot_idx])

        # --- 4. Physics Reset (Teleport Agents/Objects) ---
        if self.verbose >= 1:
            print(f"🔄 Resetting {len(active_slots)} environments (RL Reset: {rl_reset})...")

        self._reset_idx_internal(
            env_ids,
            rl_reset=rl_reset,
            folder_indices=reset_folder_indices,
            visibility_categories=reset_visibility_categories
        )

        # Step simulation to let physics settle
        for _ in range(2):
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
                torch.tensor([slot_idx], device=self.device), 
                folder_idx, 
                MIN_VALID_VIEWPOINTS
            )

            if is_valid:
                valid_slots.append(slot_idx)
                if self.verbose >= 1:
                    print(f"  ✅ Slot {slot_idx} | Env {env_id} VALIDATED")
            else:
                self.slot_attempt_counts[slot_idx] += 1
                
                if self.slot_attempt_counts[slot_idx] >= self.max_attempts_per_slot:
                    exceeded_slots.append(slot_idx)
                    if self.verbose >= 1:
                        print(f"  ⚠️ Slot {slot_idx} | Env {env_id} EXCEEDED max attempts, skipping.")
                else:
                    failed_slots.append(slot_idx)
                    if self.verbose >= 2:
                        print(f"  ❌ Slot {slot_idx} | Env {env_id} Attempt {self.slot_attempt_counts[slot_idx]}: {reason}")

        # --- 6. Data Collection (Only for Valid Slots) ---
        if valid_slots:
            if self.verbose >= 1:
                print(f"📸 Collecting images for {len(valid_slots)} slots...")

            for slot_idx in valid_slots:
                env_id = self.slot_to_env_id[slot_idx]
                folder_idx = self.slot_folder_indices[slot_idx]

                if self._select_viewpoints_for_collection(slot_idx):
                    self._collect_images_for_slot(torch.tensor([slot_idx], device=self.device), folder_idx)
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
                
                new_visibility = self._assign_next_visibility_label(new_folder_idx)
                self.slot_visibility_categories[slot_idx] = new_visibility

                if self.verbose >= 1:
                    print(f"  🔄 Slot {slot_idx}: Replaced {old_env_id} -> {new_env_id}")

        if slots_to_replace:
            self._save_visibility_labels()

        if len(self.completed_envs) >= self.total_envs_to_sim:
            print(f"\n🎉 SUCCESS: Completed all {self.total_envs_to_sim} environments!")
            sys.exit(0) # Or handle completion gracefully

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

        if vpt_prim_paths:
            self.randomize_shape_scale(prim_path_expr=vpt_prim_paths, is_random=True)

        # 2. Randomize Materials (Unified Calls)
        # A. Floor Materials ("mat")
        floor_paths = [f"/World/envs/env_{i}/mat" for i in target_ids]
        if floor_paths:
            self.randomize_material(prim_paths=floor_paths, mat_type="mat")
        
        # B. VPT Materials ("vpt")
        if vpt_prim_paths:
            self.randomize_material(prim_paths=vpt_prim_paths, mat_type="vpt")
        
        # 3. Randomize Wall Colors
        wall_paths = []
        for env_id in target_ids:
            wall_paths.extend([
                f"/World/envs/env_{env_id}/bottom_wall",
                f"/World/envs/env_{env_id}/right_wall",
                f"/World/envs/env_{env_id}/left_wall", 
                f"/World/envs/env_{env_id}/top_wall"
            ])
        
        if wall_paths:
            self.randomize_shape_color(prim_path_expr=wall_paths)

        # 4. Randomize Lights (Global)
        self.randomize_global_lights(
            prim_paths=["/World/Light", "/World/Light_A", "/World/Light_B"], 
            random_light_off=True
        )

    def initial_spawn_loop_old(self,
                           env_ids,
                           envs_need_spawn_retry,
                           safe_range: float,
                           states,
                           device=None):
        import math
        import random
        import torch
        from shapely.geometry import Point, box
        from shapely import affinity

        if device is None:
            device = self._agent.device

        # --- Unpack States (These are LOCAL subsets, size = len(env_ids)) ---
        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        # --- Setup Retry Batching ---
        # retry_indices: Local indices (e.g., 0, 2) relative to the passed 'states' tensors
        retry_mask = envs_need_spawn_retry.clone()
        retry_indices = torch.where(retry_mask)[0]
        
        # global_retry_env_ids: Global indices (e.g., 10, 12) for looking up Global Env Data (origins, dims)
        global_retry_env_ids = env_ids[retry_indices]

        batch_size = retry_indices.numel()

        if batch_size == 0:
            return envs_need_spawn_retry, states

        safe_x_range = safe_range - 4.0
        safe_x_range_obstacles = float(safe_range - 2.0)

        # Subset of origins (Requires GLOBAL IDs because env_origins is a global buffer)
        env_origins = self.scene.env_origins[global_retry_env_ids]

        # --- 1. INITIAL SAMPLING (Goal, Camera, Agent) ---
        goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        camera_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        goal_perturb_offsets = sample_uniform(-0.4, 0.4, (batch_size, 2), device)

        # Apply initial positions
        # [CRITICAL FIX]: Use LOCAL 'retry_indices' for states, but global 'env_origins' data
        goal_default_state[retry_indices, 0] = env_origins[:, 0] + goal_offsets[:, 0]
        goal_default_state[retry_indices, 1] = env_origins[:, 1] + goal_offsets[:, 1]
        goal_default_state[retry_indices, 2] = env_origins[:, 2]
        
        camera_obj_default_state[retry_indices, 0] = env_origins[:, 0] + camera_offsets[:, 0]
        camera_obj_default_state[retry_indices, 1] = env_origins[:, 1] + camera_offsets[:, 1]

        # --- 2. ENFORCE CAMERA-GOAL DISTANCE (>= 4.0) ---
        max_dist_retries = 20
        for _ in range(max_dist_retries):
            # Check subsets using LOCAL indices
            cam_pos_subset = camera_obj_default_state[retry_indices, :2]
            goal_pos_subset = goal_default_state[retry_indices, :2]
            dists = torch.norm(cam_pos_subset - goal_pos_subset, dim=1)
            
            bad_mask = (dists < 3.0) | (dists > 15.0)
            if not bad_mask.any():
                break 
            
            num_bad = bad_mask.sum().item()
            
            # bad_sub_indices: indices relative to the 'retry_indices' subset
            bad_sub_indices = torch.where(bad_mask)[0]
            
            # bad_local_indices: indices relative to 'states' (0..num_envs)
            bad_local_indices = retry_indices[bad_sub_indices]
            
            new_goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (num_bad, 2), device)
            new_cam_offsets = sample_uniform(-safe_x_range, safe_x_range, (num_bad, 2), device)
            
            # Map back to specific origins for the bad indices
            # We need to subset 'env_origins' using 'bad_sub_indices' because 'env_origins' matches 'retry_indices'
            current_origins = env_origins[bad_sub_indices]

            goal_default_state[bad_local_indices, 0] = current_origins[:, 0] + new_goal_offsets[:, 0]
            goal_default_state[bad_local_indices, 1] = current_origins[:, 1] + new_goal_offsets[:, 1]
            
            camera_obj_default_state[bad_local_indices, 0] = current_origins[:, 0] + new_cam_offsets[:, 0]
            camera_obj_default_state[bad_local_indices, 1] = current_origins[:, 1] + new_cam_offsets[:, 1]

        # --- 3. ORIENTATION & FINAL SETUP ---
        # Use LOCAL indices for states
        direction_to_goal = goal_default_state[retry_indices, :2] - camera_obj_default_state[retry_indices, :2]
        yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0]) - math.radians(90)
        roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
        zero = torch.zeros_like(yaw)
        quaternion = quat_from_euler_xyz(roll, zero, yaw)
        camera_obj_default_state[retry_indices, 3:7] = quaternion

        # Apply small perturbation to goal
        goal_default_state[retry_indices, 0] += goal_perturb_offsets[:, 0]
        goal_default_state[retry_indices, 1] += goal_perturb_offsets[:, 1]

        # Set Agent Position
        agent_default_state[retry_indices, 0] = env_origins[:, 0] + agent_offsets[:, 0]
        agent_default_state[retry_indices, 1] = env_origins[:, 1] + agent_offsets[:, 1]

        self._cache_z_offsets()

        # --- 4. ROBUST VPT LOOP (Surface Distance Check) ---
        def create_rotated_rect(x, y, w, l, yaw_rad):
            # Creates a polygon centered at (x,y)
            poly = box(-w/2.0, -l/2.0, w/2.0, l/2.0)
            poly = affinity.rotate(poly, yaw_rad, use_radians=True)
            poly = affinity.translate(poly, x, y)
            return poly

        MARGIN = 0.1
        MAX_ATTEMPTS = 100
        
        # batch_idx: iterator (0, 1, 2...)
        # local_idx: the index in 'states' (e.g. 0, 5, 8)
        for batch_idx, local_idx in enumerate(retry_indices):
            
            # Get Global ID for Global Buffer Lookup
            global_env_id = env_ids[local_idx]
            global_env_id_item = global_env_id.item() if torch.is_tensor(global_env_id) else global_env_id

            # 1. Get Global Positions (from LOCAL states)
            cam_global_x = camera_obj_default_state[local_idx, 0].item()
            cam_global_y = camera_obj_default_state[local_idx, 1].item()
            goal_global_x = goal_default_state[local_idx, 0].item()
            goal_global_y = goal_default_state[local_idx, 1].item()

            # 2. Convert to Local Positions (Relative to Env Origin)
            origin_x = env_origins[batch_idx, 0].item()
            origin_y = env_origins[batch_idx, 1].item()

            cam_local_p = Point(cam_global_x - origin_x, cam_global_y - origin_y)
            goal_local_p = Point(goal_global_x - origin_x, goal_global_y - origin_y)

            # Look up dimensions using GLOBAL ID
            active_indices = self.active_vpt_indices[global_env_id_item]
            active_dims = self.all_vpt_dims[global_env_id, active_indices, :3]
            
            placed_polys = [] 
            placement_failed = False
            
            for k, obj_idx in enumerate(active_indices):
                obj_w = active_dims[k, 0].item()
                obj_l = active_dims[k, 1].item()
                obj_h = active_dims[k, 2].item()
                
                z_ratio = self.vpt_z_offset_ratios[obj_idx].item()

                coll_w = obj_w + MARGIN
                coll_l = obj_l + MARGIN
                
                found = False
                
                for _ in range(MAX_ATTEMPTS):
                    raw_rx = (random.random() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
                    raw_ry = (random.random() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
                    rx = float(raw_rx)
                    ry = float(raw_ry)
                    r_yaw = float(random.random() * 2 * math.pi)

                    # --- A. Create Polygon First ---
                    collision_poly = create_rotated_rect(rx, ry, coll_w, coll_l, r_yaw)

                    # --- B. Check Distance from Surface ---
                    if collision_poly.distance(cam_local_p) < 4.0:
                        continue 

                    if collision_poly.distance(goal_local_p) < (self.goal_radius + 0.1):
                        continue
                    
                    # --- C. Check Bounds & Collisions ---
                    minx, miny, maxx, maxy = collision_poly.bounds
                    if (minx < -safe_x_range_obstacles or miny < -safe_x_range_obstacles or 
                        maxx > safe_x_range_obstacles or maxy > safe_x_range_obstacles):
                        continue 
                        
                    overlap = False
                    for other_poly in placed_polys:
                        if collision_poly.intersects(other_poly):
                            overlap = True
                            break
                    
                    if not overlap:
                        placed_polys.append(collision_poly)
                        
                        cand_global_x = origin_x + rx
                        cand_global_y = origin_y + ry

                        # Write to LOCAL states using 'local_idx'
                        vpt_obj_default_state[local_idx, obj_idx, 0] = cand_global_x
                        vpt_obj_default_state[local_idx, obj_idx, 1] = cand_global_y
                        vpt_obj_default_state[local_idx, obj_idx, 2] = obj_h * z_ratio
                        
                        r_yaw_tensor = torch.tensor(r_yaw, device=device)
                        zero_t = torch.tensor(0.0, device=device)
                        quat = quat_from_euler_xyz(zero_t, zero_t, r_yaw_tensor)
                        vpt_obj_default_state[local_idx, obj_idx, 3:7] = quat
                        
                        found = True
                        break
                
                if not found:
                    placement_failed = True
                    break
            
            if placement_failed:
                # Keep retry=True
                continue
            else:
                # Mark as successful
                envs_need_spawn_retry[local_idx] = False

        # --- Final cleanup ---
        # Call store_inactive with GLOBAL IDs, but update LOCAL states
        vpt_obj_default_state[retry_indices] = self._store_inactive_vpt_objects(
                env_ids[retry_indices], vpt_obj_default_state[retry_indices])

        return envs_need_spawn_retry, [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state
        ]

    def initial_spawn_loop(self,
                           env_ids,
                           envs_need_spawn_retry,
                           safe_range: float,
                           states,
                           device=None):
        import math
        import random
        import torch
        from shapely.geometry import Point, box
        from shapely import affinity

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
        safe_x_range_obstacles = float(safe_range - 2.0)

        # Subset of origins
        env_origins = self.scene.env_origins[global_retry_env_ids]

        # --- 1. INITIAL SAMPLING (Goal, Camera, Agent) ---
        goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        camera_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        goal_perturb_offsets = sample_uniform(-0.4, 0.4, (batch_size, 2), device)

        # Apply initial positions
        goal_default_state[retry_indices, 0] = env_origins[:, 0] + goal_offsets[:, 0]
        goal_default_state[retry_indices, 1] = env_origins[:, 1] + goal_offsets[:, 1]
        goal_default_state[retry_indices, 2] = env_origins[:, 2]
        
        camera_obj_default_state[retry_indices, 0] = env_origins[:, 0] + camera_offsets[:, 0]
        camera_obj_default_state[retry_indices, 1] = env_origins[:, 1] + camera_offsets[:, 1]

        # --- 2. ENFORCE CAMERA-GOAL DISTANCE (>= 4.0) ---
        max_dist_retries = 20
        for _ in range(max_dist_retries):
            cam_pos_subset = camera_obj_default_state[retry_indices, :2]
            goal_pos_subset = goal_default_state[retry_indices, :2]
            dists = torch.norm(cam_pos_subset - goal_pos_subset, dim=1)
            
            bad_mask = (dists < 3.0) | (dists > 15.0)
            if not bad_mask.any():
                break 
            
            num_bad = bad_mask.sum().item()
            bad_sub_indices = torch.where(bad_mask)[0]
            bad_local_indices = retry_indices[bad_sub_indices]
            
            new_goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (num_bad, 2), device)
            new_cam_offsets = sample_uniform(-safe_x_range, safe_x_range, (num_bad, 2), device)
            
            current_origins = env_origins[bad_sub_indices]

            goal_default_state[bad_local_indices, 0] = current_origins[:, 0] + new_goal_offsets[:, 0]
            goal_default_state[bad_local_indices, 1] = current_origins[:, 1] + new_goal_offsets[:, 1]
            
            camera_obj_default_state[bad_local_indices, 0] = current_origins[:, 0] + new_cam_offsets[:, 0]
            camera_obj_default_state[bad_local_indices, 1] = current_origins[:, 1] + new_cam_offsets[:, 1]

        # --- 3. ORIENTATION & FINAL SETUP ---
        direction_to_goal = goal_default_state[retry_indices, :2] - camera_obj_default_state[retry_indices, :2]
        yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0]) - math.radians(90)
        roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
        zero = torch.zeros_like(yaw)
        quaternion = quat_from_euler_xyz(roll, zero, yaw)
        camera_obj_default_state[retry_indices, 3:7] = quaternion

        goal_default_state[retry_indices, 0] += goal_perturb_offsets[:, 0]
        goal_default_state[retry_indices, 1] += goal_perturb_offsets[:, 1]

        agent_default_state[retry_indices, 0] = env_origins[:, 0] + agent_offsets[:, 0]
        agent_default_state[retry_indices, 1] = env_origins[:, 1] + agent_offsets[:, 1]

        self._cache_z_offsets()

        # --- 4. ROBUST VPT LOOP (MITCHELL'S BEST CANDIDATE) ---
        def create_rotated_rect(x, y, w, l, yaw_rad):
            poly = box(-w/2.0, -l/2.0, w/2.0, l/2.0)
            poly = affinity.rotate(poly, yaw_rad, use_radians=True)
            poly = affinity.translate(poly, x, y)
            return poly

        MARGIN = 0.1
        MAX_ATTEMPTS = 50       # Reduced from 100 because each attempt now does 10 checks
        NUM_CANDIDATES = 10     # Number of candidates to sample per attempt (Best-Candidate param)
        
        for batch_idx, local_idx in enumerate(retry_indices):
            
            global_env_id = env_ids[local_idx]
            global_env_id_item = global_env_id.item() if torch.is_tensor(global_env_id) else global_env_id

            # Global & Local setup
            cam_global_x = camera_obj_default_state[local_idx, 0].item()
            cam_global_y = camera_obj_default_state[local_idx, 1].item()
            goal_global_x = goal_default_state[local_idx, 0].item()
            goal_global_y = goal_default_state[local_idx, 1].item()

            origin_x = env_origins[batch_idx, 0].item()
            origin_y = env_origins[batch_idx, 1].item()

            cam_local_p = Point(cam_global_x - origin_x, cam_global_y - origin_y)
            goal_local_p = Point(goal_global_x - origin_x, goal_global_y - origin_y)

            active_indices = self.active_vpt_indices[global_env_id_item]
            active_dims = self.all_vpt_dims[global_env_id, active_indices, :3]
            
            placed_polys = [] 
            placement_failed = False
            
            for k, obj_idx in enumerate(active_indices):
                obj_w = active_dims[k, 0].item()
                obj_l = active_dims[k, 1].item()
                obj_h = active_dims[k, 2].item()
                
                z_ratio = self.vpt_z_offset_ratios[obj_idx].item()
                coll_w = obj_w + MARGIN
                coll_l = obj_l + MARGIN
                
                found = False
                
                for _ in range(MAX_ATTEMPTS):
                    # --- A. BEST CANDIDATE SELECTION ---
                    # Generate 'NUM_CANDIDATES' points and pick the one with max separation
                    best_candidate = None
                    max_isolation_dist = -1.0

                    candidates = []
                    # 1. Bulk Generate (x, y, yaw)
                    for _ in range(NUM_CANDIDATES):
                        raw_rx = (random.random() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
                        raw_ry = (random.random() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
                        r_yaw = random.random() * 2 * math.pi
                        candidates.append((float(raw_rx), float(raw_ry), float(r_yaw)))
                    
                    # 2. Score Candidates
                    for cand in candidates:
                        cand_p = Point(cand[0], cand[1])
                        
                        # Distance to Critical Entities (Camera, Goal)
                        d_cam = cand_p.distance(cam_local_p)
                        d_goal = cand_p.distance(goal_local_p)
                        current_min_dist = min(d_cam, d_goal)

                        # Distance to Already Placed Objects
                        # (Using Shapely distance from Point to existing Polygons)
                        for poly in placed_polys:
                            d_obj = poly.distance(cand_p)
                            if d_obj < current_min_dist:
                                current_min_dist = d_obj
                        
                        # We want to maximize the minimum distance (Blue Noise property)
                        if current_min_dist > max_isolation_dist:
                            max_isolation_dist = current_min_dist
                            best_candidate = cand

                    # --- B. VALIDATE WINNER ---
                    # Now we perform the strict validity checks ONLY on the best candidate
                    rx, ry, r_yaw = best_candidate
                    collision_poly = create_rotated_rect(rx, ry, coll_w, coll_l, r_yaw)

                    # 1. Hard Constraints (Must not be too close to camera/goal despite being "best")
                    if collision_poly.distance(cam_local_p) < 4.0:
                        continue 
                    if collision_poly.distance(goal_local_p) < (self.goal_radius + 0.1):
                        continue
                    
                    # 2. Check Bounds
                    minx, miny, maxx, maxy = collision_poly.bounds
                    if (minx < -safe_x_range_obstacles or miny < -safe_x_range_obstacles or 
                        maxx > safe_x_range_obstacles or maxy > safe_x_range_obstacles):
                        continue 
                        
                    # 3. Check Overlap (Intersection)
                    overlap = False
                    for other_poly in placed_polys:
                        if collision_poly.intersects(other_poly):
                            overlap = True
                            break
                    
                    if not overlap:
                        placed_polys.append(collision_poly)
                        
                        cand_global_x = origin_x + rx
                        cand_global_y = origin_y + ry

                        vpt_obj_default_state[local_idx, obj_idx, 0] = cand_global_x
                        vpt_obj_default_state[local_idx, obj_idx, 1] = cand_global_y
                        vpt_obj_default_state[local_idx, obj_idx, 2] = obj_h * z_ratio
                        
                        r_yaw_tensor = torch.tensor(r_yaw, device=device)
                        zero_t = torch.tensor(0.0, device=device)
                        quat = quat_from_euler_xyz(zero_t, zero_t, r_yaw_tensor)
                        vpt_obj_default_state[local_idx, obj_idx, 3:7] = quat
                        
                        found = True
                        break
                
                if not found:
                    placement_failed = True
                    break
            
            if placement_failed:
                continue
            else:
                envs_need_spawn_retry[local_idx] = False

        vpt_obj_default_state[retry_indices] = self._store_inactive_vpt_objects(
                env_ids[retry_indices], vpt_obj_default_state[retry_indices])

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
            if skipped_count > 0:
                print(
                    f"  ❌ Skipping ball move for {skipped_count} envs (no valid objects found)."
                )

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

        selected_ratios = self.vpt_z_offset_ratios[selected_global_indices]
        new_obj_z = target_env_origins[:, 2] + (selected_heights * selected_ratios)

        # STRICT: Goal Z = Origin + True Height + Radius + Margin
        # (Goal is center-pivoted, so we add radius to sit on top)
        new_goal_z = target_env_origins[:, 2] + selected_heights + self.goal_radius

        vpt_obj_default_state[move_ball_indices, selected_global_indices,
                              0] = target_goal_pos[:, 0]
        vpt_obj_default_state[move_ball_indices, selected_global_indices,
                              1] = target_goal_pos[:, 1]
        vpt_obj_default_state[move_ball_indices, selected_global_indices,
                              2] = new_obj_z

        goal_default_state[move_ball_indices, 2] = new_goal_z

        move_ball_cpu = move_ball_indices.cpu().numpy()
        selected_global_cpu = selected_global_indices.cpu().numpy()
        for i, env_idx in enumerate(move_ball_cpu):
            moved_vpt_for_ball[env_idx] = selected_global_cpu[i]

        # --- 5. CONFLICT RESOLUTION ---
        all_active_pos = vpt_obj_default_state[
            move_ball_indices.unsqueeze(1), batch_active_indices, :2]

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

            print(f"  ⚠️ Resolved {num_conflicts} conflicts in ball movement batch.")

        return moved_vpt_for_ball, [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state
        ]

    def move_vpt_objects(self, env_ids, valid_indices, visibility_categories,
                         moved_vpt_for_ball, states, 
                         in_view_displaced=None, outside_fov_displaced=None, # <--- Added back
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
        # This is much faster than checking "if index in tensor" inside a loop
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
                should_move = True # Always move occluded
            elif current_env_idx_val in indices_to_move:
                should_move = True # Move if in the random subset
            
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

            # --- 4. Calculate New Position (LERP + Jitter) ---
            direction_vec = goal_pos[:2] - camera_pos[:2]
            dist = torch.norm(direction_vec)

            if dist > 1e-6:
                direction_vec = direction_vec / dist
                t = random.uniform(t_min, t_max)
                new_pos_2d = camera_pos[:2] + direction_vec * (dist * t)
                random_offset = sample_uniform(-0.4, 0.4, (2,), device=device)

                vpt_obj_default_state[env_idx, random_obj_idx, 0] = new_pos_2d[0] + random_offset[0]
                vpt_obj_default_state[env_idx, random_obj_idx, 1] = new_pos_2d[1] + random_offset[1]

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

            camera_pos_batch = camera_obj_default_state[outside_fov_global_idxs, :3]
            goal_pos_batch = goal_default_state[outside_fov_global_idxs, :3]

            # Calculate Look-Away Rotation
            # Safety: Add tiny epsilon to avoid 0,0 vector issues
            direction_to_goal = (goal_pos_batch[:, :2] - camera_pos_batch[:, :2]) + 1e-6
            
            yaw = torch.atan2(direction_to_goal[:, 1],
                              direction_to_goal[:, 0]) - math.radians(90)

            yaw_offset_magnitude = sample_uniform(
                math.radians(60),
                math.pi, (len(outside_fov_global_idxs), ),
                device=device)

            signs = torch.randint(0, 2, (len(outside_fov_global_idxs), ),
                                  device=device).float() * 2 - 1

            yaw_away = yaw + (yaw_offset_magnitude * signs)
            roll = torch.full((len(outside_fov_global_idxs), ),
                              -math.radians(self.agent_camera_pitch),
                              device=device)
            zero = torch.zeros_like(roll)
            
            # Create quaternion
            quaternion_away = quat_from_euler_xyz(roll, zero, yaw_away)

            # Update State Tensor
            camera_obj_default_state[outside_fov_global_idxs, 3:7] = quaternion_away

        # ==================== CRITICAL FIX ====================
        # 3. Sanitize Quaternions BEFORE Sim Step
        # Get all quaternions we are about to write
        subset_quats = camera_obj_default_state[valid_indices, 3:7]
        
        # A. Force Normalization (Fixes "Device-side assert" in PhysX)
        subset_quats = torch.nn.functional.normalize(subset_quats, p=2, dim=-1)
        
        # B. Check for NaNs (Nuclear Option replacement)
        nan_mask = torch.isnan(subset_quats).any(dim=1)
        if nan_mask.any():
            print(f"⚠️ FATAL: Found {nan_mask.sum()} NaN quaternions! Resetting to identity.")
            # Set to identity [1, 0, 0, 0] to prevent crash
            subset_quats[nan_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        # Write sanitized quats back to state
        camera_obj_default_state[valid_indices, 3:7] = subset_quats
        # ======================================================

        # 4. Write to Sim
        self._camera_obj.write_root_pose_to_sim(
            camera_obj_default_state[valid_indices, :7], valid_env_ids)

        # 5. Update Sensor
        camera_positions = camera_obj_default_state[valid_indices, :3]
        camera_orientations = camera_obj_default_state[valid_indices, 3:7]

        # Apply 90-degree offset for sensor
        theta_left = math.pi / 2
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)],
            device=device)

        rotated_orientations = math_utils.quat_mul(
            camera_orientations,
            left_90_quat.unsqueeze(0).expand(len(valid_env_ids), -1))
        
        # Normalize sensor quats too
        rotated_orientations = torch.nn.functional.normalize(rotated_orientations, p=2, dim=-1)

        self._occlusion_camera.set_world_poses(
            positions=camera_positions,
            orientations=rotated_orientations,
            env_ids=valid_env_ids.tolist(),
            convention="world")

        # 6. Step Simulation
        for _ in range(1):
            self.sim.step()
            self._occlusion_camera.update(self.sim.cfg.dt)

    def check_z_bounds(self, env_ids, valid_indices, states, envs_need_spawn_retry, tolerance=1e-3):
        """
        Verifies Z-heights and returns the updated retry mask.
        """
        # Clone the mask if you want to be purely functional, 
        # otherwise we are still modifying the reference, but returning it is good practice.
        updated_retry_mask = envs_need_spawn_retry.clone()
        
        goal_pos, camera_pos, agent_pos, vpt_pos = states

        for local_idx, env_idx in enumerate(valid_indices):
            env_id_val = env_ids[env_idx].item()
            failure_reasons = []

            # 1. Goal Check
            goal_z = goal_pos[local_idx, 2]
            if not (-tolerance <= goal_z <= 1.0 + tolerance):
                failure_reasons.append(f"❌ Goal Z out of bounds: {goal_z:.6f}")

            # 2. Camera Check
            cam_z = camera_pos[local_idx, 2]
            if not (0.0 <= cam_z <= 1.0):
                failure_reasons.append(f"❌ Camera Z out of bounds: {cam_z:.4f}")

            # 3. Agent Check
            agent_z = agent_pos[local_idx, 2]
            if not (0.0 <= agent_z <= 1.0):
                failure_reasons.append(f"❌ Agent Z out of bounds: {agent_z:.4f}")

            # 4. VPT Check
            current_vpt_z = vpt_pos[local_idx, :, 2]
            if not torch.all((current_vpt_z >= -tolerance) & (current_vpt_z <= 0.1 + tolerance)):
                failure_reasons.append(f"❌ VPT Object Zs out of bounds")

            # Update the mask if failures found
            if failure_reasons:
                print(f"⚠️ Env {env_id_val} Z-Check Failed:")
                for reason in failure_reasons:
                    print(f"   {reason}")
                
                updated_retry_mask[env_idx] = True

        return updated_retry_mask

    def occlusion_validation_check(self, final_valid_env_ids, valid_indices,
                             visibility_categories, envs_need_spawn_retry,
                             env_dict, states, device):
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
                    print(
                        f"    ❌ Env {env_id.item()}: Geometric viewpoint check FAILED ({valid_mask.sum().item()}/{MIN_GEOMETRIC_VALID_POINTS} valid points)"
                    )
                    # Not enough viewpoints, mark for retry
                    envs_need_spawn_retry[env_idx] = True
                    geometric_valid_mask[local_idx] = False
                else:
                    if self.verbose >= 2:
                        print(
                            f"    ✅ Env {env_id.item()}: Geometric viewpoint check PASSED ({valid_mask.sum().item()}/{MIN_GEOMETRIC_VALID_POINTS} valid points)"
                        )
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

            if not camera_validation_passed:
                envs_need_spawn_retry[env_idx] = True

        return envs_need_spawn_retry

    def _reset_idx_internal(self,
                            env_ids: Sequence[int] | None,
                            rl_reset: bool = False,
                            folder_indices: List[int] = None,
                            visibility_categories: List[str] = None) -> None:
        """Internal reset logic - spawn objects and generate viewpoints."""
        reset_internal_start_time = time.time()
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

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            self.used_viewpoint_indices[env_id_item].clear()

        for i in range(num_envs):
            global_folder_idx = folder_indices[i]
            if global_folder_idx not in self.env_visibility_labels:
                raise RuntimeError(
                    f"Labels not set for folder {global_folder_idx} before _reset_idx_internal!"
                )

        # self._cache_vpt_object_dims()
        self._cache_valid_shapes()

        # ========== CHANGE #1: Call _select_active_vpt_indices ==========
        self._select_active_vpt_indices(env_ids)
        # ================================================================
        # [ADD THIS] Calculate offsets and randomize shapes BEFORE positioning
        self._cache_base_dims()
        self._cache_z_offsets()
        # ================================================================

        self.viewpoint_pose_counter[env_ids] = 0
        super()._reset_idx(env_ids)

        device = self._agent.device

        goal_default_state = self._goal.data.default_root_state[env_ids].clone(
        )
        agent_default_state = self._agent.data.default_root_state[
            env_ids].clone()
        camera_obj_default_state = self._camera_obj.data.default_root_state[
            env_ids].clone()
        vpt_obj_default_state = self._vpt_objects.data.default_object_state[
            env_ids].clone()

        max_spawn_attempts = 20
        envs_need_spawn_retry = torch.ones(num_envs,
                                           dtype=torch.bool,
                                           device=device)

        valid_indices = torch.where(envs_need_spawn_retry)[0]
        in_view_indices = [
            idx for idx in valid_indices
            if visibility_categories[idx] == "in_view"
        ]
        random_indices = torch.randperm(
            len(in_view_indices))[:len(in_view_indices) // 2]
        in_view_displaced = torch.tensor(in_view_indices,
                                         device=device)[random_indices]
        outside_fov_indices = [
            idx for idx in valid_indices
            if visibility_categories[idx] == "outside_fov"
        ]
        random_indices = torch.randperm(
            len(outside_fov_indices))[:len(outside_fov_indices) // 2]
        outside_fov_displaced = torch.tensor(outside_fov_indices,
                                             device=device)[random_indices]

        env_dict = {
            self.slot_to_env_id[env]: {
                "attempts": 0,
                "completion_status": "incomplete",
                "writing_spawn_pose_time": {},
                "moving_ball_time": {},
                "vpt_occlusion_time": {},
                "vpt_displacement_movement_time": {},
                "camera_posing_time": {},
                "occlusion_raycast_time": {},
                "geometric_check_time": {},
                "camera_validation_time": {},
                "circle_validation_time": {},
                "total_time": 0,
                "last_attempt_setup_time": 0,
                "final_setup_time": 0
            }
            for env in range(self.num_envs)
        }

        for spawn_attempt in range(max_spawn_attempts):
            spawn_start_time = time.time()

            # Update dict with spawn attempt
            for idx in range(self.num_envs):
                if envs_need_spawn_retry[idx].item() is True:
                    env_dict[
                        self.slot_to_env_id[idx]]["attempts"] = spawn_attempt
                    env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] = "incomplete"
                elif envs_need_spawn_retry[idx].item() is False:
                    if env_dict[self.slot_to_env_id[idx]][
                            "completion_status"] != "complete":
                        env_dict[self.slot_to_env_id[idx]][
                            "completion_status"] = "complete"
                        # if env_dict[self.slot_to_env_id[idx]]["total_time"] == 0:
                        #     env_dict[self.slot_to_env_id[idx]]["total_time"] = np.round(time.time() - reset_internal_start_time, 3)

                # Round off times to 4 decimal places for readability
                for key in [
                        "writing_spawn_pose_time", "moving_ball_time",
                        "camera_posing_time", "vpt_occlusion_time",
                        "vpt_displacement_movement_time",
                        "occlusion_raycast_time", "geometric_check_time",
                        "camera_validation_time", "circle_validation_time"
                ]:
                    if key in env_dict[
                            self.slot_to_env_id[idx]] and isinstance(
                                env_dict[self.slot_to_env_id[idx]][key], dict):
                        for sub_key in env_dict[self.slot_to_env_id[idx]][key]:
                            if isinstance(
                                    env_dict[self.slot_to_env_id[idx]][key]
                                [sub_key], (float, int)):
                                env_dict[self.slot_to_env_id[idx]][key][
                                    sub_key] = np.round(
                                        env_dict[self.slot_to_env_id[idx]][key]
                                        [sub_key], 3)

            if not envs_need_spawn_retry.any():
                break

            # =========== DEBUG: Time spawn attempt ==========
            start_time = time.time()
            # ================================================

            retry_mask = envs_need_spawn_retry.clone()

            envs_need_spawn_retry, states = self.initial_spawn_loop(
                env_ids=env_ids,
                envs_need_spawn_retry=envs_need_spawn_retry,
                safe_range=self.center_to_boundary,
                states=[
                    goal_default_state, camera_obj_default_state,
                    agent_default_state, vpt_obj_default_state
                ],
                device=device)

            goal_default_state = states[0]
            camera_obj_default_state = states[1]
            agent_default_state = states[2]
            vpt_obj_default_state = states[3]

            valid_mask = retry_mask & ~envs_need_spawn_retry
            if not valid_mask.any():
                continue

            valid_indices = torch.where(valid_mask)[0]
            valid_env_ids = env_ids[valid_indices]

            writing_spawn_start_time = time.time()

            self.write_pose_to_sim(
                env_ids=valid_env_ids,
                indices=valid_indices,
                goal_default_state=goal_default_state,
                camera_obj_default_state=camera_obj_default_state,
                agent_default_state=agent_default_state,
                vpt_obj_default_state=vpt_obj_default_state)

            writing_spawn_end_time = time.time()
            # print("AFTER INIT SPAWN LOOP")
            # print(f"Z of active VPT objects: {self._get_active_vpt_positions(valid_env_ids, base_pivoted=True)[:,:,2]}")
            # print(self._vpt_objects.data.object_pos_w.shape, self._vpt_objects.data.object_pos_w[valid_env_ids].shape)
            # print(f"Z of active VPT objects: {self._vpt_objects.data.object_pos_w[valid_env_ids][:,:,2]}")
            # Update dicts for all incomplete envs
            for idx in range(num_envs):
                if env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] == "incomplete":
                    if envs_need_spawn_retry[idx].item() is False:
                        env_dict[self.slot_to_env_id[idx]][
                            "writing_spawn_pose_time"][
                                spawn_attempt] = writing_spawn_end_time - writing_spawn_start_time
                    if envs_need_spawn_retry[idx].item() is True:
                        env_dict[self.slot_to_env_id[idx]][
                            "writing_spawn_pose_time"][spawn_attempt] = "F"

            # Initialize dict for tracking (kept for compatibility)
            moved_vpt_for_ball = {i: None for i in range(num_envs)}

            # Identify indices relative to total envs
            move_ball_indices = torch.where(
                torch.isin(env_ids, self.envs_to_move_ball))[0]

            ball_movement_start_time = time.time()

            moved_vpt_for_ball, states = self.moving_ball_loop(
                env_ids=env_ids,
                moved_vpt_for_ball=moved_vpt_for_ball,
                move_ball_indices=move_ball_indices,
                states=[
                    goal_default_state, camera_obj_default_state,
                    agent_default_state, vpt_obj_default_state
                ],
                safe_range=self.center_to_boundary,
                device=device)

            goal_default_state = states[0]
            camera_obj_default_state = states[1]
            agent_default_state = states[2]
            vpt_obj_default_state = states[3]

            move_ball_env_ids = env_ids[move_ball_indices]
            self.write_pose_to_sim(env_ids=move_ball_env_ids,
                                   indices=move_ball_indices,
                                   goal_default_state=goal_default_state,
                                   vpt_obj_default_state=vpt_obj_default_state)

            moving_ball_end_time = time.time()

            # print("AFTER MOVING BALL LOOP")
            # print(f"Z of active VPT objects: {self._get_active_vpt_positions(valid_env_ids, base_pivoted=True)[:,:,2]}")
            # print(f"Z of active VPT objects: {self._vpt_objects.data.object_pos_w[valid_env_ids][:,:,2]}")

            # Update dict
            for idx in range(num_envs):
                if env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] == "incomplete":
                    if envs_need_spawn_retry[idx].item() is False:
                        env_dict[self.slot_to_env_id[idx]]["moving_ball_time"][
                            spawn_attempt] = moving_ball_end_time - ball_movement_start_time
                    if envs_need_spawn_retry[idx].item() is True:
                        env_dict[self.slot_to_env_id[idx]]["moving_ball_time"][
                            spawn_attempt] = "F"

            vpt_move_start_time = time.time()

            states = self.move_vpt_objects(
                env_ids=env_ids,
                valid_indices=valid_indices,
                visibility_categories=visibility_categories,
                moved_vpt_for_ball=moved_vpt_for_ball,
                states=[
                    goal_default_state, camera_obj_default_state,
                    agent_default_state, vpt_obj_default_state
                ],
                in_view_displaced=in_view_displaced,         # <--- Passed back in
                outside_fov_displaced=outside_fov_displaced, # <--- Passed back in
                device=device,
            )

            # Unpack states
            goal_default_state = states[0]
            camera_obj_default_state = states[1]
            agent_default_state = states[2]
            vpt_obj_default_state = states[3]

            # Write poses to sim immediately after calculating
            self.write_pose_to_sim(env_ids=valid_env_ids,
                                   indices=valid_indices,
                                   vpt_obj_default_state=vpt_obj_default_state)

            vpt_move_end_time = time.time()
            total_duration = vpt_move_end_time - vpt_move_start_time

            # Update dict with new single key
            for idx in range(num_envs):
                env_slot_id = self.slot_to_env_id[idx]
                
                if env_dict[env_slot_id]["completion_status"] == "incomplete":
                    retry_needed = envs_need_spawn_retry[idx].item()
                    
                    if retry_needed is False:
                        env_dict[env_slot_id]["vpt_displacement_movement_time"][spawn_attempt] = total_duration
                        
                    elif retry_needed is True:
                        env_dict[env_slot_id]["vpt_displacement_movement_time"][spawn_attempt] = "F"
            
            # Fetch latest positions
            goal_new_pos = self._goal.data.root_pos_w[valid_env_ids]
            camera_new_pos = self._camera_obj.data.root_pos_w[valid_env_ids]
            agent_new_pos = self._agent.data.root_pos_w[valid_env_ids]
            vpt_new_pos = self._get_active_vpt_positions(valid_env_ids, base_pivoted=True)

            # Perform Z-Checks and CAPTURE the return
            envs_need_spawn_retry = self.check_z_bounds(
                env_ids=env_ids,
                valid_indices=valid_indices,
                states=(goal_new_pos, camera_new_pos, agent_new_pos, vpt_new_pos),
                envs_need_spawn_retry=envs_need_spawn_retry,
                tolerance=1e-3
            )

            # Check if any envs remain valid
            final_valid_mask = valid_mask & ~envs_need_spawn_retry
            if not final_valid_mask.any():
                continue

            camera_writing_start_time = time.time()
            
            final_valid_indices = torch.where(final_valid_mask)[0]
            final_valid_env_ids = env_ids[final_valid_indices]

            self.outside_fov_camera_movement(
                valid_env_ids=final_valid_env_ids,
                valid_indices=final_valid_indices,
                visibility_categories=visibility_categories,
                states=[
                    goal_default_state, camera_obj_default_state,
                    agent_default_state, vpt_obj_default_state
                ],
                device=device,
            )

            camera_writing_end_time = time.time()

            # Update dict
            for idx in range(num_envs):
                if env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] == "incomplete":
                    if envs_need_spawn_retry[idx].item() is False:
                        env_dict[self.slot_to_env_id[idx]]["camera_posing_time"][
                            spawn_attempt] = camera_writing_end_time - camera_writing_start_time
                    if envs_need_spawn_retry[idx].item() is True:
                        env_dict[self.slot_to_env_id[idx]][
                            "camera_posing_time"][spawn_attempt] = "F"

            # Camera POV validation with image saving
            occlusion_raycast_start_time = time.time()
            occlusion_valid_mask, envs_need_spawn_retry, env_dict, states = self.occlusion_validation_check(
                final_valid_env_ids=final_valid_env_ids,
                valid_indices=final_valid_indices,
                visibility_categories=visibility_categories,
                envs_need_spawn_retry=envs_need_spawn_retry,
                env_dict=env_dict,
                states=[
                    goal_default_state, camera_obj_default_state,
                    agent_default_state, vpt_obj_default_state
                ],
                device=device)

            occlusion_raycast_end_time = time.time()

            # Update dict
            for idx in range(num_envs):
                if env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] == "incomplete":
                    if envs_need_spawn_retry[idx].item() is False:
                        env_dict[self.slot_to_env_id[idx]][
                            "occlusion_raycast_time"][
                                spawn_attempt] = occlusion_raycast_end_time - occlusion_raycast_start_time
                    if envs_need_spawn_retry[idx].item() is True:
                        env_dict[self.slot_to_env_id[idx]][
                            "occlusion_raycast_time"][spawn_attempt] = "F"

            geometric_check_start_time = time.time()

            geometric_valid_mask, envs_need_spawn_retry = self.geometric_occlusion_check(
                env_ids=final_valid_env_ids,
                valid_indices=final_valid_indices,
                occlusion_valid_mask=occlusion_valid_mask,
                envs_need_spawn_retry=envs_need_spawn_retry,
                device=device)

            geometric_check_end_time = time.time()

            # Update dict
            for idx in range(num_envs):
                if env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] == "incomplete":
                    if envs_need_spawn_retry[idx].item() is False:
                        env_dict[
                            self.slot_to_env_id[idx]]["geometric_check_time"][
                                spawn_attempt] = geometric_check_end_time - geometric_check_start_time
                    if envs_need_spawn_retry[idx].item() is True:
                        env_dict[self.slot_to_env_id[idx]][
                            "geometric_check_time"][spawn_attempt] = "F"

            occlusion_geometric_start_time = time.time()

            envs_need_spawn_retry = self.camera_pov_validation(
                env_ids=final_valid_env_ids,
                valid_indices=final_valid_indices,
                geometric_valid_mask=geometric_valid_mask,
                visibility_categories=visibility_categories,
                envs_need_spawn_retry=envs_need_spawn_retry,
                folder_indices=folder_indices,
                spawn_attempt=spawn_attempt
            )

            occlusion_geometric_end_time = time.time()

            # Update dict
            for idx in range(num_envs):
                if env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] == "incomplete":
                    if envs_need_spawn_retry[idx].item() is False:
                        env_dict[self.slot_to_env_id[idx]][
                            "camera_validation_time"][
                                spawn_attempt] = occlusion_geometric_end_time - occlusion_geometric_start_time
                        env_dict[self.slot_to_env_id[idx]][
                            "final_setup_time"] = np.round(
                                occlusion_geometric_end_time -
                                reset_internal_start_time, 3)

                    if envs_need_spawn_retry[idx].item() is True:
                        env_dict[self.slot_to_env_id[idx]][
                            "camera_validation_time"][spawn_attempt] = "F"

                    # Update time before circle validation
                    current_time = time.time()
                    env_dict[self.slot_to_env_id[idx]][
                        "total_time_wo_FOV"] = np.round(
                            current_time - reset_internal_start_time, 3)

        random_yaw_agent = sample_uniform(0, 2 * math.pi, (num_envs, ), device)
        agent_default_state[:, 3] = torch.cos(random_yaw_agent / 2)
        agent_default_state[:, 4] = 0.0
        agent_default_state[:, 5] = 0.0
        agent_default_state[:, 6] = torch.sin(random_yaw_agent / 2)

        # Ensure all VPT obstacles on the floor
        for i, env_id in enumerate(env_ids):
            # Get the global environment ID
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            
            # Identify active objects
            active_indices = self.active_vpt_indices[env_id_item]
            
            # Retrieve the immutable 'resting Z' from class storage (set in randomize_shape_scale)
            # self.vpt_obj_default_state is the "Source of Truth" for dimensions/scale
            safe_z_values = self.vpt_obj_default_state[env_id, active_indices, 2]
            
            # Force the local state tensor to match this Z
            vpt_obj_default_state[i, active_indices, 2] = safe_z_values

        # 2. Force Write to Sim
        #    We must push this state to the physics engine immediately so the 
        #    subsequent Circle Validation doesn't detect objects in the air.
        self.write_pose_to_sim(
            env_ids=env_ids,
            indices=torch.arange(len(env_ids), device=device),
            vpt_obj_default_state=vpt_obj_default_state,
            agent_default_state=agent_default_state
        )

        # ========== CHANGE: Filter Envs for Circle Generation ==========
        circle_validation_start_time = time.time()

        # 1. Generate points ONLY for successful envs (Saves Compute)
        success_mask = ~envs_need_spawn_retry
        successful_env_ids = env_ids[success_mask]
        
        subset_valid_points = []
        if len(successful_env_ids) > 0:
            subset_valid_points = self.generate_valid_circle_points(
                env_ids=successful_env_ids,
                angle_step=2.0,
                max_attempts=100
            )

        circle_validation_end_time = time.time()

        # 2. Assign results correctly
        if self.valid_viewpoint_poses is None:
            self.valid_viewpoint_poses = [None] * self.num_envs

        # A. Clear/Zero out failed environments
        failed_env_ids = env_ids[envs_need_spawn_retry]
        for env_id in failed_env_ids:
            eid = env_id.item() if torch.is_tensor(env_id) else env_id
            self.valid_viewpoint_poses[eid] = torch.zeros((0, 3), device=device)

        # B. Map successful results to their specific global env IDs
        for i, env_id in enumerate(successful_env_ids):
            eid = env_id.item() if torch.is_tensor(env_id) else env_id
            points_2d = subset_valid_points[i]

            if points_2d.shape[0] >= self.images_per_env:
                # Convert 2D -> 3D
                agent_z = self._agent.data.default_root_state[env_id, 2]
                points_3d = torch.zeros((points_2d.shape[0], 3), device=device)
                points_3d[:, :2] = points_2d
                points_3d[:, 2] = agent_z
                self.valid_viewpoint_poses[eid] = points_3d
            else:
                self.valid_viewpoint_poses[eid] = torch.zeros((0, 3), device=device)
        # ===============================================================

        # Update dict (Time logging)
        for idx in range(num_envs):
            if env_dict[self.slot_to_env_id[idx]][
                    "completion_status"] == "incomplete":
                if envs_need_spawn_retry[idx].item() is False:
                    env_dict[self.slot_to_env_id[idx]]["circle_validation_time"][
                        spawn_attempt] = circle_validation_end_time - circle_validation_start_time
                if envs_need_spawn_retry[idx].item() is True:
                    env_dict[self.slot_to_env_id[idx]][
                        "circle_validation_time"][spawn_attempt] = "F"
            elif env_dict[self.slot_to_env_id[idx]][
                    "completion_status"] == "complete":
                env_dict[self.slot_to_env_id[idx]]["circle_validation_time"][
                    spawn_attempt] = circle_validation_end_time - circle_validation_start_time

        # --- DELETED THE OLD LOOP HERE THAT USED all_valid_points ---

        reset_internal_end_time = time.time()

        # TODO: Clear this mess below, make an update_env_dict function
        # Update dict with spawn attempt
        for idx in range(self.num_envs):
            if envs_need_spawn_retry[idx].item() is True:
                env_dict[self.slot_to_env_id[idx]]["attempts"] = spawn_attempt
                env_dict[self.slot_to_env_id[idx]][
                    "completion_status"] = "incomplete"
            elif envs_need_spawn_retry[idx].item() is False:
                if env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] != "complete":
                    env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] = "complete"
                    if env_dict[self.slot_to_env_id[idx]]["total_time"] == 0:
                        env_dict[
                            self.slot_to_env_id[idx]]["total_time"] = np.round(
                                reset_internal_end_time -
                                reset_internal_start_time, 3)
                # Round off times to 4 decimal places for readability
            for key in [
                    "writing_spawn_pose_time", "moving_ball_time",
                    "camera_posing_time", "vpt_occlusion_time",
                    "vpt_displacement_movement_time",
                    "occlusion_raycast_time", "geometric_check_time",
                    "camera_validation_time", "circle_validation_time"
            ]:
                if key in env_dict[self.slot_to_env_id[idx]] and isinstance(
                        env_dict[self.slot_to_env_id[idx]][key], dict):
                    for sub_key in env_dict[self.slot_to_env_id[idx]][key]:
                        if isinstance(
                                env_dict[self.slot_to_env_id[idx]][key]
                            [sub_key], (float, int)):
                            env_dict[self.slot_to_env_id[idx]][key][
                                sub_key] = np.round(
                                    env_dict[self.slot_to_env_id[idx]][key]
                                    [sub_key], 3)

            if env_dict[self.slot_to_env_id[idx]][
                    "completion_status"] == "complete":
                env_dict[
                    self.slot_to_env_id[idx]]["last_attempt_setup_time"] = (
                        list(env_dict[self.slot_to_env_id[idx]]
                             ["writing_spawn_pose_time"].values())[-1] +
                        list(env_dict[self.slot_to_env_id[idx]]
                             ["moving_ball_time"].values())[-1] +
                        list(env_dict[self.slot_to_env_id[idx]]
                             ["camera_posing_time"].values())[-1] +
                        list(env_dict[self.slot_to_env_id[idx]]
                             ["vpt_displacement_movement_time"].values())[-1] +
                        list(env_dict[self.slot_to_env_id[idx]]
                             ["occlusion_raycast_time"].values())[-1] +
                        list(env_dict[self.slot_to_env_id[idx]]
                             ["camera_validation_time"].values())[-1])

                # Round up
                env_dict[
                    self.slot_to_env_id[idx]]["final_setup_time"] = np.round(
                        env_dict[self.slot_to_env_id[idx]]["final_setup_time"],
                        3)
                env_dict[self.slot_to_env_id[idx]][
                    "last_attempt_setup_time"] = np.round(
                        env_dict[self.slot_to_env_id[idx]]
                        ["last_attempt_setup_time"], 3)

        print(f"\n--- Spawn Attempt {spawn_attempt} ---")
        print(f"-" * 50)
        print(f"Env dict below:")
        for key, value in env_dict.items():
            print(f"Env {key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
        print(f"-" * 50)
        print(
            f"Total reset time for spawn attempts = {spawn_attempt}: {reset_internal_end_time - reset_internal_start_time:.3f} seconds"
        )
        print("\n")

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
            min_camera_obstacle_distance, min_camera_target_distance, min_target_obstacle_distance
        )

        if not check_agent_fov or not valid_mask.any():
            return valid_mask

        # 2. Slow FOV Checks (Physics Simulation)
        valid_mask = self._check_fov_validity(
            points, env_ids, valid_mask, min_required_points
        )

        return valid_mask

    def _check_geometric_validity(self, points, env_ids, min_obs_dist,
                                  min_cam_obs_dist, min_cam_target_dist, 
                                  min_target_obs_dist):
        """Validates boundaries, obstacle proximity, and camera clearance."""
        device = points.device
        valid_mask = torch.ones(points.shape[0], dtype=torch.bool, device=device)

        # Boundary Check
        env_origins = self.scene.env_origins[env_ids, :2]
        in_bounds = torch.all(
            (points >= env_origins - self.center_to_boundary) & 
            (points <= env_origins + self.center_to_boundary), dim=1)
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
        valid_mask &= (torch.norm(points - cam_pos, dim=1) >= min_cam_target_dist)

        dist_cam_obs = torch.norm(cam_pos.unsqueeze(1) - active_obs_pos, dim=2)
        valid_mask &= (dist_cam_obs.min(dim=1)[0] >= min_cam_obs_dist)

        return valid_mask

    def _get_active_obstacle_positions(self, env_ids):
        """Gather active VPT object positions for specific envs."""
        device = env_ids.device
        all_pos = self._vpt_objects.data.object_pos_w[env_ids, :, :2]

        if isinstance(self.active_vpt_indices, list):
            if len(self.active_vpt_indices) > 0 and isinstance(self.active_vpt_indices[0], torch.Tensor):
                idx = torch.stack(self.active_vpt_indices).to(dtype=torch.long, device=device)
            else:
                idx = torch.tensor(self.active_vpt_indices, device=device, dtype=torch.long)
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
        env_queues, env_status = self._sample_fov_candidates(
            points, env_ids, points_to_check, max_s=120
        )
        
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
                if self.verbose >= 1 and all(s['count'] >= min_req_points for s in env_status.values()):
                    print(f"    ✅ All environments have {min_req_points}+ valid points. Early stop.")
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
                    if env_status[eid]['count'] == min_req_points: # Just hit threshold
                        if self.verbose >= 1:
                            print(f"    🎯 Env {eid}: Reached {min_req_points} valid points.")

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
                    print(f"  🎲 Env {eid_item}: Sampled {max_s}/{total} candidates.")
            else:
                if self.verbose >= 2:
                    print(f"  🎲 Env {eid_item}: Using all {total} candidates.")

            queues[eid_item] = {'points': points[env_indices], 'indices': env_indices}
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

        self._agent.write_root_com_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)
        
        # Step Physics
        self.sim.step()
        self._tiled_camera.update(self.sim.cfg.dt)
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
                print(
                    f"  Env {env_id_item}: {geometric_valid_per_env[i].sum().item()} geometric candidates"
                )

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
            for _ in range(2):
                self.sim.step()
                self._rgb_tiled_camera.update(self.sim.cfg.dt)
                self._distance_tiled_camera.update(self.sim.cfg.dt)
                if self.save_camera_pov:
                    self._occlusion_camera.update(self.sim.cfg.dt)

            # Get camera data for this env
            rgb_data = self._rgb_tiled_camera.data.output["rgb"]
            depth_data = self._distance_tiled_camera.data.output[
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

    def _cache_z_offsets(self):
        """
        Creates a tensor of Z-offset ratios based on object type.
        - Primitives (Cuboid, Cone, Cylinder) -> 0.5 (Center pivoted)
        - Specific USD Shapes (X, L, T ending) -> 0.0 (Base pivoted)
        """
        offsets = []
        for key, obj_cfg in self.cfg.vpt_objects.rigid_objects.items():
            spawn_cfg = obj_cfg.spawn

            # Case 1: Primitives (Always 0.5)
            if isinstance(spawn_cfg, (sim_utils.CuboidCfg, sim_utils.ConeCfg,
                                      sim_utils.CylinderCfg)):
                offsets.append(0.5)

            # Case 2: USD Files (Check filename for X, L, T)
            elif isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                usd_path = spawn_cfg.usd_path
                filename = usd_path.split("/")[-1].split(".")[0]
                # If filename ends with X, L, or T, assume base pivot (0.0)
                if filename.endswith(('X', 'L', 'T')):
                    offsets.append(0.0)
                else:
                    print("0.5 used")
                    offsets.append(0.5)  # Default center pivoted
            else:
                offsets.append(0.5)

        self.vpt_z_offset_ratios = torch.tensor(offsets, device=self.device)

    def randomize_shape_scale(self,
                              prim_path_expr: str | list,
                              is_random: bool = True):
        """
        Refined Randomization:
        1. Identifies object type.
        2. Calculates Scale and Z-Position analytically based on known base dims.
        3. Updates Physics/Visuals.
        4. Calculates and stores precise bounding box info for placement.
        5. Updates offset ratios dynamically for the getter function.
        """
        world = World.instance()
        if world.is_playing():
            world.pause()

        stage = get_current_stage()
        
        # We need bbox_cache to get ACCURATE final visual bounds/offsets
        # Note: ComputeLocalBound gives UNSCALED bounds of the geometry
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                       [UsdGeom.Tokens.default_])

        # 1. Initialize Standard Dims (for Observations/Rewards) [Width, Length, Height]
        if not hasattr(
                self,
                "all_vpt_dims") or self.all_vpt_dims.shape[0] != self.num_envs:
            self.all_vpt_dims = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs, 3), device=self.device)
            self.vpt_obj_default_state = torch.zeros(
                (self.num_envs, self.num_objs, 3), device=self.device)

        # 2. Initialize Bounding Box Storage (for Placement) [min_x, min_y, max_x, max_y]
        if not hasattr(self, "all_vpt_bb") or self.all_vpt_bb.shape[0] != self.num_envs:
            self.all_vpt_bb = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs, 4), device=self.device)

        # 3. Initialize Offset Ratios (for Getter Function)
        if not hasattr(self, "vpt_z_offset_ratios") or self.vpt_z_offset_ratios.shape[0] != self.cfg.num_vpt_objs:
            self.vpt_z_offset_ratios = torch.zeros((self.cfg.num_vpt_objs), device=self.device)

        # Resolve paths
        if isinstance(prim_path_expr, str):
            prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
        elif isinstance(prim_path_expr, list):
            prim_paths = []
            for expr in prim_path_expr:
                prim_paths.extend(sim_utils.find_matching_prim_paths(expr))

        print(f"\n[Randomizing Scale] Processing {len(prim_paths)} objects...")
        obj_configs = list(self.cfg.vpt_objects.rigid_objects.values())

        with Sdf.ChangeBlock():
            for prim_path in prim_paths:
                root_prim = stage.GetPrimAtPath(prim_path)
                if not root_prim.IsValid(): continue

                # Parse Indices
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

                # Get/Create Ops
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
                    base_dim = self.vpt_base_dims[obj_idx].cpu().numpy()  # [x, y, z]
                    final_scale_vec = Gf.Vec3d(1, 1, 1)
                    final_z_pos = 0.0
                    
                    # Store the ratio used here so getter stays in sync
                    z_offset_multiplier = 0.0 

                    # --- LOGIC BRANCHING ---

                    if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                        filename = spawn_cfg.usd_path.split("/")[-1].split(".")[0]

                        if filename.endswith(('X', 'L', 'T')):
                            # Special USDs: Base Pivoted
                            z_offset_multiplier = 0.0
                            
                            if filename.endswith(("X")):
                                s_xz = random.uniform(0.5, 2.5)
                                s_y = random.uniform(0.5, 2.5)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_xz, base_dim[1] * s_y,
                                    base_dim[2] * s_xz)
                            elif filename.endswith(("L")):
                                s_factor = random.uniform(0.5, 2.5)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_factor,
                                    base_dim[1] * s_factor,
                                    base_dim[2] * s_factor)
                                
                            # Calculation: Base is at 0, so Z=0
                            final_z_pos = 0.0 
                        else:
                            # Generic USD: Treat as Base Pivoted (0.0) usually safest for USDs
                            # If your generic USDs are center pivoted, change this to 0.5
                            z_offset_multiplier = 0.0
                            
                            s_xy = random.uniform(0.5, 2.5)
                            s_z = random.uniform(0.5, 2.5)
                            final_scale_vec = Gf.Vec3d(base_dim[0] * s_xy,
                                                       base_dim[1] * s_xy,
                                                       base_dim[2] * s_z)
                            final_z_pos = 0.0

                    elif isinstance(spawn_cfg, sim_utils.CuboidCfg):
                        # Primitives: Center Pivoted
                        z_offset_multiplier = 0.5
                        
                        s_x = random.uniform(0.5, 2.5)
                        s_y = random.uniform(0.5, 2.5)
                        s_z = random.uniform(0.5, 2.5)
                        final_scale_vec = Gf.Vec3d(s_x, s_y, s_z)

                        # Calculation: Lift by half height
                        total_height = base_dim[2] * s_z
                        final_z_pos = total_height * z_offset_multiplier

                    elif isinstance(spawn_cfg, (sim_utils.CylinderCfg, sim_utils.ConeCfg)):
                        # Primitives: Center Pivoted
                        z_offset_multiplier = 0.5
                        
                        s_r = random.uniform(0.75, 1.0)
                        s_h = random.uniform(0.75, 2.5)
                        final_scale_vec = Gf.Vec3d(s_r, s_r, s_h)

                        # Calculation: Lift by half height
                        total_height = base_dim[2] * s_h
                        final_z_pos = total_height * z_offset_multiplier

                    # -----------------------

                    # Apply Scale
                    scale_op.Set(final_scale_vec)

                    # Apply Translation
                    current_trans = translate_op.Get()
                    translate_op.Set(
                        Gf.Vec3d(current_trans[0], current_trans[1],
                                 final_z_pos))

                    self.vpt_obj_default_state[env_idx, obj_idx, 2] = final_z_pos
                    
                    # Update Source of Truth for Getter
                    self.vpt_z_offset_ratios[obj_idx] = z_offset_multiplier

                    # --- COMPUTE BOUNDS ---
                    bbox_cache.Clear()
                    
                    # ComputeLocalBound ALREADY includes the scale you just set!
                    local_bound = bbox_cache.ComputeLocalBound(root_prim).GetRange()
                    
                    # These are the FINAL world-space dimensions (assuming parent has identity scale)
                    final_w = local_bound.GetMax()[0] - local_bound.GetMin()[0]
                    final_l = local_bound.GetMax()[1] - local_bound.GetMin()[1]
                    final_h = local_bound.GetMax()[2] - local_bound.GetMin()[2]

                    # 1. Update standard dims - DO NOT MULTIPLY BY SCALE AGAIN
                    self.all_vpt_dims[env_idx, obj_idx, 0] = final_w
                    self.all_vpt_dims[env_idx, obj_idx, 1] = final_l
                    self.all_vpt_dims[env_idx, obj_idx, 2] = final_h
                    # print(f"Env {env_idx} - Obj {obj_idx} Z = {final_h}")
                    
                    # 2. Update Bounding Box - DO NOT MULTIPLY BY SCALE AGAIN
                    # These coordinates are already in the scaled local space
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

    def randomize_global_lights(self, prim_paths=["/World/Light_A", "/World/Light_B"], random_light_off=False):
        """
        Randomizes Distant Lights to cover full day cycles (Morning -> Noon -> Evening)
        and varying weather conditions (Dim -> Bright).
        """
        stage = get_current_stage()
        
        assigned_azimuths = []
        min_separation = 30.0

        # --- 0. DETERMINE ACTIVE LIGHTS ---
        active_paths = set(prim_paths)

        if random_light_off and len(prim_paths) > 0:
            # Keep at least 2 light on, turn off a random subset of the rest
            num_to_keep = random.randint(2, len(prim_paths))
            active_paths = set(random.sample(prim_paths, num_to_keep))

        for path in prim_paths:
            prim = stage.GetPrimAtPath(path)

            if not prim.IsValid():
                print(f"⚠️ Warning: Light prim not found at {path}")
                continue

            # --- 1. INTENSITY (Dim to Bright) ---
            if path not in active_paths:
                prim.GetAttribute("inputs:intensity").Set(0.0)
                continue 
            
            # 300 (Dim/Overcast) to 3000 (Blazing Sun)
            # With 4 lights, max total energy could be ~12000, min ~300.
            rand_intensity = random.uniform(400.0, 1000.0)
            prim.GetAttribute("inputs:intensity").Set(rand_intensity)

            # --- 2. ROTATION (Time of Day) ---
            # 15 deg = Early Morning / Late Evening (Long shadows)
            # 85 deg = High Noon (Short shadows)
            rand_elevation = random.uniform(15, 85)
            
            # Azimuth Separation Logic
            valid_azimuth = False
            candidate_azimuth = 0.0
            max_retries = 50 

            for _ in range(max_retries):
                candidate_azimuth = random.uniform(0, 360) # Covers East (Morning) and West (Evening)
                collision = False
                for existing_azimuth in assigned_azimuths:
                    diff = abs(candidate_azimuth - existing_azimuth)
                    dist = min(diff, 360.0 - diff)
                    if dist < min_separation:
                        collision = True
                        break
                if not collision:
                    valid_azimuth = True
                    break
            
            assigned_azimuths.append(candidate_azimuth)

            xform = UsdGeom.Xformable(prim)
            rotate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                    rotate_op = op
                    break
            if rotate_op is None:
                rotate_op = xform.AddRotateXYZOp()

            # -X is elevation in standard Z-up orientation
            rotate_op.Set(Gf.Vec3d(-rand_elevation, 0, candidate_azimuth))

            # --- 3. SHADOW SOFTNESS ---
            # 0.5 (Sharp/Clear Sky) to 5.0 (Very Soft/Overcast)
            rand_angle = random.uniform(0.5, 5.0) 
            prim.GetAttribute("inputs:angle").Set(rand_angle)

            # --- 4. COLOR TEMPERATURE ---
            prim.GetAttribute("inputs:enableColorTemperature").Set(True)
            # 3500K (Warm/Golden Hour) to 8500K (Cool/Overcast)
            rand_temp = random.uniform(3500.0, 8500.0)
            prim.GetAttribute("inputs:colorTemperature").Set(rand_temp)

    def get_color(self):
        """
        Generate a random pastel color that is not similar to red, green, blue, or pink.
        Returns r, g, b values.
        """
        # Define forbidden colors (Reference points)
        forbidden_colors = [
            np.array([1.0, 0.0, 0.0]),   # Pure Red
            np.array([0.0, 1.0, 0.0]),   # Pure Green
            np.array([0.2, 0.8, 0.2]),   # Lime/Forest Green
            np.array([0.0, 0.0, 1.0]),   # Pure Blue
            np.array([0.8, 0.0, 0.0]),   # Pinkish-red
            np.array([1.0, 0.75, 0.8]),  # Pink
            np.array([1.0, 0.2, 0.6]),   # Pink
            np.array([0.9, 0.1, 0.5]),   # Pink
            np.array([0.95, 0.3, 0.6]),  # Pink
            np.array([0.0, 0.0, 0.0])    # Black
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

        return base[0], base[1], base[2]

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
    
    def get_material_configs(self, material_type: str) -> list[sim_utils.MdlFileCfg]:
        """
        Retrieves material paths, selects a subset, and wraps them in MdlFileCfg objects.
        Args:
            material_type (str): "mat" (for floor) or "vpt" (for obstacles).
        """
        # 1. Determine constraints based on type
        if material_type == "mat":
            raw_paths = get_mat_material_paths() # Imported function
            tex_scale = (1000.0, 1000.0)
        elif material_type == "vpt":
            raw_paths = get_vpt_material_paths() # Imported function
            tex_scale = (2.0, 2.0)
        else:
            raise ValueError(f"Unknown material type: {material_type}")

        # 2. Limit sample size (Logic preserved: Limit 100)
        num_to_select = min(len(raw_paths), 100)
        selected_paths = random.sample(raw_paths, num_to_select)

        if self.verbose >= 1:
            print(f"[INFO] {material_type.upper()} Config: {len(selected_paths)} materials selected.")

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
            print(f"⚠️ Unknown material type '{material_type}', skipping randomization.")
            return

        if not material_pool:
            return

        for prim in prim_paths:
            rand_material = random.choice(material_pool)
            sim_utils.bind_visual_material(prim, rand_material)
