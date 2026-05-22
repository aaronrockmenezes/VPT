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

from .vpt2_env_cfg_v2 import VPTEnvCfg
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
        self.storage_position = torch.tensor([250.0, 250.0, 0.0], device=self.device)
        self.active_vpt_indices = [
            None
        ] * self.num_envs  # Track which 20 are active per env

        # Derived environment parameters
        self.center_to_boundary = torch.abs(
            torch.tensor(self.boundary_limits).view(-1)[0])

        # Verbosity and visibility thresholds
        self.verbose = 0
        self.goal_pixel_threshold_agent = 300  # Minimum pixels for goal visibility
        self.goal_pixel_threshold_cam = 300  # Minimum pixels for goal visibility
        # self.camera_pixel_threshold = 500  # Minimum pixels for camera visibility
        self.camera_pixel_threshold_agent = 1000  # Minimum pixels for camera visibility
        self.ref_obj_pixel_threshold_agent = 300
        self.ref_obj_pixel_threshold_cam = 300

        # Data collection parameters
        self.images_per_env = 10  # Number of images to collect per environment
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
        self.save_fov_debug_images = False  # Enable only when debugging FOV validation

        # File paths
        self.GPU_ID = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.NODE_ID = os.getenv("NODE_ID", os.getenv("SLURM_ARRAY_TASK_ID", "0"))
        base = os.getenv("BASE_PATH", "/oscar/scratch/arock3/VPT2_DATA/v3")
        self.base_path = f"{base}/data/data_node{self.NODE_ID}_gpu{self.GPU_ID}"

        print("*" * 50)
        print(f"🚀 Initializing VPTEnv on Node {self.NODE_ID} GPU {self.GPU_ID}...")
        print("*" * 50)

        self.visibility_labels_json_path = f"{self.base_path}/visibility_labels.json"

        # Mode determination
        self.mode = "rl"
        # self.mode = "rl_data_collection"

        self.total_envs_to_sim = 1000
        self.slot_to_env_id = list(range(self.num_envs))
        self.next_env_id = self.num_envs
        self.completed_envs = set()
        self.slot_attempt_counts = [0] * self.num_envs
        self.max_attempts_per_slot = 20 * 50  # Full resets * Inner resets
        self.active_ref_idx = [None] * self.num_envs  # 0=cone, 1=cuboid, 2=cylinder

        self.used_vpt_objects = set()
        self._preallocate_visibility_labels()
        self.verbose = 2

        #rotation angle
        self.theta = self.cfg.agent_turn_angle
        self.half_theta = self.theta / 2
        # Shape (1, 4) for broadcasting
        self.rot_q_left = torch.tensor(
            [[math.cos(self.half_theta), 0., 0., math.sin(self.half_theta)]],
            device=self.device)
        self.rot_q_right = torch.tensor(
            [[math.cos(self.half_theta), 0., 0., -math.sin(self.half_theta)]],
            device=self.device)
        
        self.move_speed = self.cfg.agent_speed


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
        """Pre-allocate visibility labels for all environments in 50/50 proportion."""
        total = self.total_envs_to_sim
        num_left = total // 2
        num_right = total - num_left

        # Create list of all labels
        all_labels = (["left"] * num_left + ["right"] * num_right)
        random.shuffle(all_labels)

        # Store as a list to pop from
        self.visibility_label_pool = all_labels

        if self.verbose >= 1:
            print(f"📋 Pre-allocated {total} visibility labels:")
            print(f"   - left: {num_left}")
            print(f"   - right: {num_right}")

    def _assign_next_visibility_label(self, folder_idx: int) -> str:
        """Assign the next visibility label from the pre-allocated pool."""
        if not self.visibility_label_pool:
            raise RuntimeError("Visibility label pool exhausted!")

        category = self.visibility_label_pool.pop(0)

        if category == "left":
            self.env_visibility_labels[folder_idx] = "left"
            self.env_visibility_reasons[folder_idx] = "left"
        elif category == "right":
            self.env_visibility_labels[folder_idx] = "right"
            self.env_visibility_reasons[folder_idx] = "right"

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
        self._ref_objs = [
            RigidObject(self.cfg.reference_obj_cone),
            RigidObject(self.cfg.reference_obj_cuboid),
            RigidObject(self.cfg.reference_obj_cylinder),
        ]
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
        for i, ref_obj in enumerate(self._ref_objs):
            self.scene.rigid_objects[f"ref_obj_{i}"] = ref_obj

        self._rgb_tiled_camera = TiledCamera(self.cfg.rgb_tiled_camera)
        self.scene.sensors["rgb_tiled_camera"] = self._rgb_tiled_camera

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
        self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Check visibility of goal (red), camera (green), ref obj (pink) for a batch of envs."""
        sem_imgs = self._rgb_tiled_camera.data.output["semantic_segmentation"][env_ids]

        # Optional debug image saving (disabled by default — can generate thousands of images)
        if self.save_fov_debug_images:
            save_dir = os.path.expanduser("~/images_debug/")
            os.makedirs(save_dir, exist_ok=True)
            if not hasattr(self, '_debug_img_counter'):
                self._debug_img_counter = 0
            for idx in range(sem_imgs.shape[0]):
                img_np = sem_imgs[idx].cpu().numpy().astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(os.path.join(save_dir, f"img_{self._debug_img_counter}.png"), img_bgr)
                self._debug_img_counter += 1

        r = sem_imgs[..., 0]
        g = sem_imgs[..., 1]
        b = sem_imgs[..., 2]
        a = sem_imgs[..., 3]

        red_mask  = (r == 255) & (g == 0)   & (b == 0)   & (a == 255)
        green_mask = (r == 0)  & (g == 255) & (b == 0)   & (a == 255)
        pink_mask  = (r == 255) & (g == 0)  & (b == 255) & (a == 255)

        red_counts   = red_mask.sum(dim=(1, 2))
        green_counts = green_mask.sum(dim=(1, 2))
        pink_counts  = pink_mask.sum(dim=(1, 2))

        goal_visible    = red_counts   >= self.goal_pixel_threshold_agent
        camera_visible  = green_counts >= self.camera_pixel_threshold_agent
        ref_obj_visible = pink_counts  >= self.ref_obj_pixel_threshold_agent

        both_visible_mask = goal_visible & camera_visible & ref_obj_visible
        ids_with_both = env_ids[both_visible_mask]
        if len(ids_with_both) > 0:
            print(f"Envs with both goal and camera visible: {ids_with_both.tolist()}")

        return goal_visible, camera_visible, ref_obj_visible

    def move_agent(self, actions, env_ids: Sequence[int] | None = None):
        """
        Moves the agent kinematically (teleportation) based on actions.
        Removed velocity writing for performance optimization.
        Removed upright quaternion normalization (uses raw current orientation).
        """
        # start_time = time.time()
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids,
                                   dtype=torch.long,
                                   device=self.device)

        dt = self.cfg.sim.dt

        # --- 1. Separate Physics Actions from Resets ---
        reset_mask_5 = (actions == 5)
        reset_mask_6 = (actions == 6)
        physics_mask = ~(reset_mask_5 | reset_mask_6)

        # --- 2. Handle Resets / Debug ---
        if reset_mask_5.any():
            # print(f"Nothing to do for action 5.")
            self._reset_idx(env_ids[reset_mask_5], rl_reset=False)

        if reset_mask_6.any():
            print("+" * 50)
            self._reset_idx(env_ids[reset_mask_6], rl_reset=True)

        reset_time = time.time()

        # If no agents are performing physics actions, exit early
        if not physics_mask.any():
            return

        # Filter indices and actions for physics updates
        phys_ids = env_ids[physics_mask]
        phys_actions = actions[physics_mask]

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
        rgb_data = rgb_data.permute(0, 3, 1, 2)[:, :3, :, :]
        observations = {"policy": rgb_data.clone()}

        self.obs = rgb_data

        return observations

    def _update_camera_poses(self, env_ids: torch.Tensor) -> None:
        """Aligns the occlusion camera sensor with the camera rigid body, rotated 90° left."""
        camera_obj_pos = self._camera_obj.data.root_pos_w[env_ids].clone()
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_ids].clone()

        half_theta = (math.pi / 2) / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta), 0.0, 0.0, math.sin(half_theta)],
            device=self.device
        ).unsqueeze(0).expand(len(env_ids), -1)

        rotated_orientations = math_utils.quat_mul(camera_obj_quat, left_90_quat)

        self._occlusion_camera.set_world_poses(
            positions=camera_obj_pos,
            orientations=rotated_orientations,
            env_ids=env_ids.tolist(),
            convention="world"
        )

    def _get_rewards(self) -> torch.Tensor:
        distance = (self._camera_obj.data.root_pos_w[:, :2] - self._agent.data.root_pos_w[:, :2]) ** 2
        distance = torch.sqrt(distance.sum(dim=1))
        reward = -1 * distance

        # angle_to_camera = (self._camera_obj.data.root_quat_w * self._agent.data.root_quat_w).sum(dim=1) ** 2
        # import pdb; pdb.set_trace()

        # yaw_green_cam = math_utils.yaw_quat(self._camera_obj.data.root_quat_w) # + torch.tensor((math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)), device=self.device))
        yaw_green_cam = math_utils.yaw_quat(self._occlusion_camera.data.quat_w_world)
        yaw_agent = math_utils.yaw_quat(self._agent.data.root_quat_w)
        angle_to_camera = math_utils.quat_error_magnitude(yaw_green_cam, yaw_agent)

        # reward[distance < 0.4] = reward[distance < 0.4] - 1 * angle_to_camera[distance < 0.4]
        reward = reward - 1 * angle_to_camera
        
        #print(f"Distance: {distance.cpu().numpy()}, Angle to Camera: {angle_to_camera.cpu().numpy()}, Reward: {reward.cpu().numpy()}")
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance = (self._camera_obj.data.root_pos_w[:, :2] - self._agent.data.root_pos_w[:, :2]) ** 2
        distance = torch.sqrt(distance.sum(dim=1))

        # yaw_green_cam = math_utils.yaw_quat(self._camera_obj.data.root_quat_w) # + torch.tensor((math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)), device=self.device))
        yaw_green_cam = math_utils.yaw_quat(self._occlusion_camera.data.quat_w_world)
        yaw_agent = math_utils.yaw_quat(self._agent.data.root_quat_w)
        angle_to_camera = math_utils.quat_error_magnitude(yaw_green_cam, yaw_agent)

        goal_reached = (distance <= self.cfg.agent_success_distance) & (angle_to_camera <= self.cfg.agent_success_angle)

        self.extras["is_success"] = goal_reached.clone().cpu()

        # print(f"Distance: {distance.cpu().numpy()}, Angle to Camera: {angle_to_camera.cpu().numpy()}, Goal Reached: {goal_reached.cpu().numpy()}")

        terminated = torch.zeros(self.num_envs,
                                 dtype=torch.bool,
                                 device=self.device)
        terminated = terminated | goal_reached

        time_outs = (self.episode_length_buf >= self.max_episode_length)

        return terminated, time_outs

    def _validate_env_state(self, env_id, folder_idx, min_viewpoints):
        """
        Validate single environment's state after reset attempt.
        Strictly enforces visibility for 'left' and 'right' categories.
        """
        # env_id here is the physical SLOT INDEX (0-17)
        slot_idx = env_id.item() if torch.is_tensor(env_id) else env_id

        # --- 1. VIEWPOINT VALIDATION ---
        # Checks if we successfully generated enough viewpoints for this physical slot
        if (self.valid_viewpoint_poses is None
                or slot_idx >= len(self.valid_viewpoint_poses)
                or self.valid_viewpoint_poses[slot_idx] is None):
            return False, "viewpoint cache is empty (None)"

        num_poses = len(self.valid_viewpoint_poses[slot_idx])
        if num_poses < min_viewpoints:
            return False, f"insufficient viewpoints: {num_poses}/{min_viewpoints}"

        # --- 2. DIRECTIONAL VISIBILITY VALIDATION ---
        visibility_category = self.env_visibility_labels.get(folder_idx, "unknown")
        
        # In your current setup, both 'left' and 'right' demand a clear line of sight
        if visibility_category in ["left", "right"]:
            camera_pos = self._camera_obj.data.root_pos_w[slot_idx]
            goal_pos = self._goal.data.root_pos_w[slot_idx]
            
            # Fire the raycast to check for physical obstructions (VPT objects)
            is_occluded = self._check_occlusion_raycast(camera_pos, goal_pos, slot_idx)
            
            if is_occluded:
                return False, f"occlusion mismatch: '{visibility_category}' requires clear visibility, but path is blocked."

        else:
            # If a label somehow isn't left/right, we reject it to prevent corrupted data collection
            return False, f"invalid category: '{visibility_category}' (expected 'left' or 'right')"

        # Passed all checks
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

        GOAL_THRESHOLD = self.goal_pixel_threshold_cam
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
            "left": 0,
            "right": 0,
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
                "left_count":
                sum(1 for v in self.env_visibility_labels.values()
                    if v == "left"),
                "right_count":
                sum(1 for v in self.env_visibility_labels.values()
                    if v == "right"),
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
                             return_counts: bool = False,
                             count_ref_obj: bool = False):
        """
        Checks for target (red) and reference (pink) object visibility in an image.

        Parameters
        ----------
        file_name : str
            Path to save and reload the image for pixel counting.
        cam_pov : numpy.ndarray
            RGB camera image array to be analyzed.
        return_counts : bool, optional
            If True, returns the exact pixel counts for the evaluated objects. Default is False.
        count_ref_obj : bool, optional
            If True, evaluates visibility of the pink reference object. Default is False.

        Returns
        -------
        target_visible_in_camera : bool
            True if target red pixel count meets the threshold.
        ref_obj_visible : bool, optional
            Returned if `count_ref_obj` is True. True if pink pixel count meets the threshold.
        red_count : int, optional
            Returned if `return_counts` is True.
        pink_count : int, optional
            Returned if `count_ref_obj` and `return_counts` are True.
        """
        cv2.imwrite(file_name, cv2.cvtColor(cam_pov, cv2.COLOR_RGB2BGR))

        # Load image back and count exact pixels
        loaded_img = cv2.imread(file_name)
        loaded_img_rgb = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)

        r = loaded_img_rgb[:, :, 0]
        g = loaded_img_rgb[:, :, 1]
        b = loaded_img_rgb[:, :, 2]

        # Exact match for Red (255, 0, 0)
        red_mask = (r == 255) & (g == 0) & (b == 0)
        red_count = red_mask.sum().item()

        target_visible_in_camera = red_count >= self.goal_pixel_threshold_cam

        if count_ref_obj:
            # Exact match for Pink (255, 0, 255)
            pink_mask = (r == 255) & (g == 0) & (b == 255)
            pink_count = pink_mask.sum().item()
            
            # Assuming a camera-specific threshold exists for the reference object
            ref_obj_visible = pink_count >= self.ref_obj_pixel_threshold_cam

            if return_counts:
                return target_visible_in_camera, ref_obj_visible, red_count, pink_count
            return target_visible_in_camera, ref_obj_visible

        if return_counts:
            return target_visible_in_camera, red_count
            
        return target_visible_in_camera

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

    def _get_batch_active_indices(self, env_ids: int | list | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standardizes env_ids and retrieves their active object indices.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (env_ids_flat, batch_indices)
            - env_ids_flat: 1D tensor of environment IDs.
            - batch_indices: 2D tensor [batch_size, num_active].
        """
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.view(-1)

        if isinstance(self.active_vpt_indices, list):
            if self.active_vpt_indices and isinstance(self.active_vpt_indices[0], torch.Tensor):
                full_indices = torch.stack(self.active_vpt_indices).to(self.device)
            else:
                full_indices = torch.tensor(self.active_vpt_indices, device=self.device, dtype=torch.long)
        else:
            full_indices = self.active_vpt_indices.to(self.device)

        return env_ids, full_indices[env_ids]

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

    def _get_active_vpt_dims(self, env_ids: int | torch.Tensor) -> torch.Tensor:
        """Retrieves dimensions for the active objects in the given environments."""
        ids, batch_indices = self._get_batch_active_indices(env_ids)
        env_ids_expanded = ids.view(-1, 1).expand_as(batch_indices)
        return self.all_vpt_dims[env_ids_expanded, batch_indices, :]

    def _get_active_vpt_positions(self,
                                env_ids: int | torch.Tensor,
                                base_pivoted: bool = False,
                                return_full_pose: bool = False) -> torch.Tensor:
        """
        Retrieves world positions (and optionally orientation) of active objects.
        """
        ids, batch_indices = self._get_batch_active_indices(env_ids)
        env_ids_expanded = ids.view(-1, 1).expand_as(batch_indices)

        active_pos = self._vpt_objects.data.object_pos_w[env_ids_expanded, batch_indices].clone()

        if base_pivoted:
            heights = self.all_vpt_dims[env_ids_expanded, batch_indices, 2]
            ratios = self.vpt_z_offset_ratios[env_ids_expanded, batch_indices]
            active_pos[:, :, 2] -= (heights * ratios)

        if return_full_pose:
            active_quat = self._vpt_objects.data.object_quat_w[env_ids_expanded, batch_indices].clone()
            return torch.cat([active_pos, active_quat], dim=-1)

        return active_pos

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

    def _ensure_slot_initialization(self) -> None:
        """Lazy initializes slot states and visibility labels on the first run."""
        if hasattr(self, "slot_folder_indices"):
            return

        if self.verbose >= 1:
            print("🔒 Initializing slot states...")

        self.slot_folder_indices = [self.next_env_folder_idx + i for i in range(self.num_envs)]
        self.slot_attempt_counts = [0] * self.num_envs

        self.slot_visibility_categories = []
        for idx in self.slot_folder_indices:
            self.slot_visibility_categories.append(self._assign_next_visibility_label(idx))

        self._save_visibility_labels()
    
    def _validate_slots(self, active_slots: list[int]) -> tuple[list[int], list[int]]:
        """
        Validates the state of active slots against viewpoint requirements.

        Returns
        -------
        valid_slots : list[int]
        exceeded_slots : list[int]
        """
        valid_slots = []
        exceeded_slots = []
        min_viewpoints = self.images_per_env

        for slot_idx in active_slots:
            env_id = self.slot_to_env_id[slot_idx]

            if env_id in self.completed_envs:
                continue

            folder_idx = self.slot_folder_indices[slot_idx]
            is_valid, reason = self._validate_env_state(
                torch.tensor([slot_idx], device=self.device),
                folder_idx,
                min_viewpoints
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
                        print(f"  ⚠️ Slot {slot_idx} | Env {env_id} EXCEEDED attempts.")
                elif self.verbose >= 2:
                    print(f"  ❌ Slot {slot_idx} | Env {env_id}: {reason}")

        return valid_slots, exceeded_slots

    def _replenish_slots(self, slots_to_replace: list[int]) -> None:
        """Advances the environment ID for slots that completed or failed too many times."""
        if not slots_to_replace:
            return

        for slot_idx in slots_to_replace:
            if self.next_env_id >= self.total_envs_to_sim:
                continue

            old_env = self.slot_to_env_id[slot_idx]
            new_env = self.next_env_id
            self.next_env_id += 1

            self.slot_to_env_id[slot_idx] = new_env
            self.slot_folder_indices[slot_idx] = self.next_env_folder_idx + new_env
            self.slot_attempt_counts[slot_idx] = 0
            self.slot_visibility_categories[slot_idx] = self._assign_next_visibility_label(
                self.slot_folder_indices[slot_idx]
            )

            if self.verbose >= 1:
                print(f"  🔄 Slot {slot_idx}: Replaced {old_env} -> {new_env}")

        self._save_visibility_labels()

    def _reset_idx(self, env_ids: Sequence[int] | None, rl_reset: bool = True) -> None:
        """
        Resets specified environments.

        Modes:
        - rl_reset=True: Fast physics reset and randomization (Training).
        - rl_reset=False: Full pipeline with validation, data collection, slot replenishment.
        """
        # --- 1. Standardization ---
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        active_slots_list = env_ids.tolist()

        # --- 2. Lazy Init ---
        self._ensure_slot_initialization()

        # --- 3. Scene Randomization ---
        self._cache_base_dims()
        self._randomize_scene_props(active_slots_list)

        num_targets = max(1, int(0.25 * len(env_ids)))
        subset_idx = torch.randperm(len(env_ids), device=self.device)[:num_targets]
        self.envs_to_move_ball = env_ids[subset_idx]

        # --- 4. Physics Reset ---
        reset_folder_indices = [self.slot_folder_indices[i] for i in active_slots_list]
        reset_visibility_cats = [self.slot_visibility_categories[i] for i in active_slots_list]

        if self.verbose >= 1:
            print(f"🔄 Resetting {len(active_slots_list)} envs (RL: {rl_reset})")

        self._reset_idx_internal(
            env_ids,
            rl_reset=rl_reset,
            folder_indices=reset_folder_indices,
            visibility_categories=reset_visibility_cats
        )

        # Settle physics
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.step_dt)

        if rl_reset:
            self._reset_called = True
            return

        # --- 5. Data Collection Pipeline ---
        valid_slots, exceeded_slots = self._validate_slots(active_slots_list)

        for slot_idx in valid_slots:
            folder_idx = self.slot_folder_indices[slot_idx]
            if self._select_viewpoints_for_collection(slot_idx):
                self._collect_images_for_slot(
                    torch.tensor([slot_idx], device=self.device),
                    folder_idx
                )
                self.completed_envs.add(self.slot_to_env_id[slot_idx])

        self._replenish_slots(valid_slots + exceeded_slots)
        self._reset_called = True

    def _randomize_scene_props(self, env_ids):
        """Helper to handle all the randomization logic for specific environments."""
        if isinstance(env_ids, torch.Tensor):
            target_ids = env_ids.tolist()
        else:
            target_ids = list(env_ids)

        vpt_prim_paths = []
        for env_id in target_ids:
            for idx in range(self.cfg.num_vpt_objs):
                vpt_prim_paths.append(f"/World/envs/env_{env_id}/obs_{idx}")

        self.apply_randomizations(env_ids=target_ids,
                                randomize_scale=True,
                                randomize_color=False,
                                randomize_light=True)

        floor_paths = [f"/World/envs/env_{i}/mat" for i in target_ids]
        if floor_paths:
            self.randomize_material(prim_paths=floor_paths, material_type="mat")

        if vpt_prim_paths:
            self.randomize_material(prim_paths=vpt_prim_paths, material_type="vpt")

        # --- NEW: Randomize wall colors ---
        wall_paths = []
        for env_id in target_ids:
            wall_paths.extend([
                f"/World/envs/env_{env_id}/bottom_wall",
                f"/World/envs/env_{env_id}/right_wall",
                f"/World/envs/env_{env_id}/left_wall",
                f"/World/envs/env_{env_id}/top_wall",
            ])
        if wall_paths:
            self.randomize_shape_color(prim_path_expr=wall_paths)

    def initial_spawn_loop(self,
                           env_ids,
                           envs_need_spawn_retry,
                           safe_range: float,
                           states,
                           allow_clipping: bool = False,
                           device=None):
        import math
        import torch
        # Assuming math_utils is imported in your file, or import it here:
        # from isaaclab.utils import math as math_utils

        if device is None:
            device = self._agent.device

        # --- Unpack States ---
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

        safe_range_objects = float(safe_range - 4.0)
        safe_range_obstacles = float(safe_range - 3.0)

        env_origins = self.scene.env_origins[global_retry_env_ids]

        # ==========================================================
        # 1. SCATTER VPT OBSTACLES (Vectorized, same as RL version)
        # ==========================================================
        num_vpt_objs = vpt_obj_default_state.shape[1]

        rxs = (torch.rand((batch_size, num_vpt_objs), device=device) * 2 * safe_range_obstacles) - safe_range_obstacles
        rys = (torch.rand((batch_size, num_vpt_objs), device=device) * 2 * safe_range_obstacles) - safe_range_obstacles
        r_yaws = torch.rand((batch_size, num_vpt_objs), device=device) * 2 * math.pi

        vpt_obj_default_state[retry_indices, :, 0] = env_origins[:, 0].unsqueeze(1) + rxs
        vpt_obj_default_state[retry_indices, :, 1] = env_origins[:, 1].unsqueeze(1) + rys

        zero_t = torch.zeros_like(r_yaws)
        quats = quat_from_euler_xyz(zero_t, zero_t, r_yaws)
        vpt_obj_default_state[retry_indices, :, 3:7] = quats

        # Store inactive objects at storage position
        vpt_obj_default_state[retry_indices] = self._store_inactive_vpt_objects(
            global_retry_env_ids, vpt_obj_default_state[retry_indices])

        # ==========================================================
        # 2. PLACE GOAL (Collision-free via shotgun)
        # ==========================================================
        goal_default_state[retry_indices] = self.place_object_safely(
            env_ids=global_retry_env_ids,
            object_state=goal_default_state[retry_indices],
            vpt_state=vpt_obj_default_state[retry_indices],
            safe_range=safe_range_objects,
            object_type='goal'
        )

        # ==========================================================
        # 3. PLACE CAMERA (Collision-free via shotgun)
        # ==========================================================
        camera_obj_default_state[retry_indices] = self.place_object_safely(
            env_ids=global_retry_env_ids,
            object_state=camera_obj_default_state[retry_indices],
            vpt_state=vpt_obj_default_state[retry_indices],
            safe_range=safe_range_objects,
            object_type='cam_obj'
        )

        # ==========================================================
        # 4. ENFORCE CAMERA-GOAL DISTANCE [3.0, 18.0]
        # ==========================================================
        max_dist_retries = 50
        for _ in range(max_dist_retries):
            dists = torch.norm(
                camera_obj_default_state[retry_indices, :2] - goal_default_state[retry_indices, :2],
                dim=1
            )

            bad_mask = (dists < 3.0) | (dists > 18.0)
            if not bad_mask.any():
                break

            bad_sub_indices = torch.where(bad_mask)[0]
            bad_local_indices = retry_indices[bad_sub_indices]
            bad_global_env_ids = global_retry_env_ids[bad_sub_indices]

            # Re-roll only failed cameras
            camera_obj_default_state[bad_local_indices] = self.place_object_safely(
                env_ids=bad_global_env_ids,
                object_state=camera_obj_default_state[bad_local_indices],
                vpt_state=vpt_obj_default_state[bad_local_indices],
                safe_range=safe_range_objects,
                object_type='cam_obj'
            )

        # ==========================================================
        # 5. PERTURB GOAL POSITION (MOVED UP)
        # ==========================================================
        # Must happen BEFORE orientation so the camera looks at the final, perturbed spot.
        goal_perturb = sample_uniform(-2, 2, (batch_size, 2), device)
        goal_default_state[retry_indices, 0] += goal_perturb[:, 0]
        goal_default_state[retry_indices, 1] += goal_perturb[:, 1]

        # ==========================================================
        # 6. ORIENT CAMERA TOWARD GOAL (CLEAN EULER METHOD)
        # ==========================================================
        direction_to_goal = goal_default_state[retry_indices, :2] - camera_obj_default_state[retry_indices, :2]
        
        # Subtract pi/2 (90 degrees) directly from the yaw so the Y-facing lens aligns with the target
        yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0]) - (math.pi / 2)
        
        # Pitch goes on the X-axis to cleanly tilt the Y-lens up/down. No roll induced.
        pitch = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
        zero = torch.zeros_like(yaw)
        
        # One single conversion. No quaternion multiplication needed.
        camera_obj_default_state[retry_indices, 3:7] = quat_from_euler_xyz(pitch, zero, yaw)

        # ==========================================================
        # 7. MARK ALL AS SPAWNED
        # ==========================================================
        envs_need_spawn_retry[retry_indices] = False

        return envs_need_spawn_retry, [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state
        ]


    def initial_spawn_loop_rl(self,
                              env_ids,
                              envs_need_spawn_retry,
                              safe_range: float,
                              states,
                              visibility_categories,  # Added this to feed in the labels
                              allow_clipping: bool = False,
                              device=None,
                              mode: str = None):
        if device is None:
            device = self._agent.device
            
        if mode is None:
            mode = self.mode

        # --- Unpack States (Expected size: 5) ---
        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]
        ref_obj_default_state = states[4]

        # --- Setup Retry Batching ---
        retry_mask = envs_need_spawn_retry.clone()
        retry_indices = torch.where(retry_mask)[0]
        global_retry_env_ids = env_ids[retry_indices]
        batch_size = retry_indices.numel()

        if batch_size == 0:
            return envs_need_spawn_retry, states

        # Safe ranges for spawning
        safe_range_objects = float(safe_range - 4.0)
        safe_x_range_obstacles = float(safe_range - 3.0)

        # Subset of origins
        env_origins = self.scene.env_origins[global_retry_env_ids]

        # ==========================================================
        # 1. RANDOMLY SCATTER ALL VPT OBSTACLES FIRST
        # ==========================================================
        num_vpt_objs = vpt_obj_default_state.shape[1]
        
        rxs = (torch.rand((batch_size, num_vpt_objs), device=device) * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
        rys = (torch.rand((batch_size, num_vpt_objs), device=device) * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
        r_yaws = torch.rand((batch_size, num_vpt_objs), device=device) * 2 * math.pi
        
        # Apply to XY
        vpt_obj_default_state[retry_indices, :, 0] = env_origins[:, 0].unsqueeze(1) + rxs
        vpt_obj_default_state[retry_indices, :, 1] = env_origins[:, 1].unsqueeze(1) + rys
        
        # Apply random Yaw
        zero_t = torch.zeros_like(r_yaws)
        quats = quat_from_euler_xyz(zero_t, zero_t, r_yaws)
        vpt_obj_default_state[retry_indices, :, 3:7] = quats

        # Store inactive objects
        vpt_obj_default_state[retry_indices] = self._store_inactive_vpt_objects(
            global_retry_env_ids, vpt_obj_default_state[retry_indices])

        goal_default_state[retry_indices] = self.place_object_safely(
            env_ids=global_retry_env_ids,
            object_state=goal_default_state[retry_indices],
            vpt_state=vpt_obj_default_state[retry_indices],
            safe_range=safe_range_objects,
            object_type='goal'
        )

        # Initial placement for all cameras
        camera_obj_default_state[retry_indices] = self.place_object_safely(
            env_ids=global_retry_env_ids,
            object_state=camera_obj_default_state[retry_indices],
            vpt_state=vpt_obj_default_state[retry_indices],
            safe_range=safe_range_objects,
            object_type='cam_obj'
        )

        # ==========================================================
        # 2. REJECTION LOOP FOR CAMERA-GOAL DISTANCE
        # ==========================================================
        max_dist_retries = 50
        for _ in range(max_dist_retries):
            # Calculate distance between camera and goal for all active envs
            dists = torch.norm(
                camera_obj_default_state[retry_indices, :2] - goal_default_state[retry_indices, :2], 
                dim=1
            )

            # Mask out the ones that violate the constraint
            bad_mask = (dists < 2.0) | (dists > 20.0)
            if not bad_mask.any():
                break  

            # Isolate the environments that failed
            bad_sub_indices = torch.where(bad_mask)[0]
            bad_local_indices = retry_indices[bad_sub_indices]
            bad_global_env_ids = global_retry_env_ids[bad_sub_indices]

            # Re-roll ONLY the failed environments
            camera_obj_default_state[bad_local_indices] = self.place_object_safely(
                env_ids=bad_global_env_ids,
                object_state=camera_obj_default_state[bad_local_indices],
                vpt_state=vpt_obj_default_state[bad_local_indices],
                safe_range=safe_range_objects,
                object_type='cam_obj'
            )

        # ==========================================================
        # 3. PLACE REFERENCE OBJECT RELATIVE TO *FINAL* GOAL AND CAMERA
        # ==========================================================
        cam_pos_xy = camera_obj_default_state[retry_indices, :2]
        goal_pos_xy = goal_default_state[retry_indices, :2]

        direction_to_goal = goal_pos_xy - cam_pos_xy
        dists_to_goal = torch.norm(direction_to_goal, dim=1, keepdim=True)
        dists_to_goal = torch.clamp(dists_to_goal, min=1e-5)
        forward_dir = direction_to_goal / dists_to_goal

        perp_dir = torch.stack([-forward_dir[:, 1], forward_dir[:, 0]], dim=1)

        # -- NEW LOGIC: Dynamic side multipliers based on vis labels --
        multipliers = []
        for env_id in global_retry_env_ids:
            env_idx = env_id.item()
            # Fetch the label, defaulting to None if lists are mismatched
            cat = visibility_categories[env_idx] if visibility_categories else None
            
            if cat == 'left':
                multipliers.append(1.0)
            elif cat == 'right':
                multipliers.append(-1.0)
            else:
                # Fallback for 'occluded', 'in_view', or random initialization
                multipliers.append(float(random.choice([1.0, -1.0])))

        # Convert to correctly shaped tensor for batch multiplication
        side_multiplier = torch.tensor(multipliers, device=device, dtype=torch.float32).view(-1, 1)

        # Calculate exact center point of the patch (0.5 units to the chosen side)
        ref_patch_centers = goal_pos_xy + (perp_dir * side_multiplier * 1.5)

        patch_size_radius = 0.5
        ref_obj_default_state[retry_indices, :2] = ref_patch_centers
        # print(global_retry_env_ids)
        
        ref_obj_default_state[retry_indices] = self.place_object_safely(
            env_ids=global_retry_env_ids,
            object_state=ref_obj_default_state[retry_indices],
            vpt_state=vpt_obj_default_state[retry_indices],
            safe_range=patch_size_radius, 
            object_type='ref_obj',
            custom_centers=ref_patch_centers
        )

        # ==========================================================
        # 4. ORIENT CAMERA TOWARD GOAL (CLEAN EULER METHOD)
        # ==========================================================
        # if mode == "rl_data_collection":
        if True:
            final_direction_to_goal = goal_default_state[retry_indices, :2] - camera_obj_default_state[retry_indices, :2]
            yaw = torch.atan2(final_direction_to_goal[:, 1], final_direction_to_goal[:, 0]) - (math.pi / 2)
            
            pitch = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
            zero = torch.zeros_like(yaw)
            
            camera_obj_default_state[retry_indices, 3:7] = quat_from_euler_xyz(pitch, zero, yaw)

        # ==========================================================
        # 5. MARK ALL AS SPAWNED
        # ==========================================================
        envs_need_spawn_retry[retry_indices] = False

        return envs_need_spawn_retry, [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state, ref_obj_default_state
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
            print("OUTSIDE")
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
                                   device=None, rl_mode: bool = False):
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
            env_id = valid_env_ids[local_idx]
            env_id_item = env_id.item()

            visibility_category = visibility_categories[env_idx]
            camera_pos = camera_positions[local_idx]
            goal_pos = goal_default_state[env_idx, :3]

            if rl_mode:
                # RL MODE: Just fire the raycast. No constraints, no rejections.
                _ = self._check_occlusion_raycast(camera_pos, goal_pos, env_id)
                
                # Register the env as successful
                if env_id_item not in env_dict:
                    env_dict[env_id_item] = 0.0
            else:
                # NORMAL MODE: Enforce strict visibility categories
                # --- FIX: Added "left" and "right" to the allowed list ---
                if visibility_category in ["in_view", "left", "right", "occluded", "outside_fov"]:
                    is_occluded = self._check_occlusion_raycast(
                        camera_pos, goal_pos, env_id)

                    expected_occluded = (visibility_category == "occluded"
                                         or visibility_category == "outside_fov")
                    occlusion_valid = (is_occluded == expected_occluded)
                    occlusion_valid_mask[local_idx] = occlusion_valid

                    if not occlusion_valid:
                        # print(f"Category = {visibility_category} | Got {is_occluded}")
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
                cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(np.uint8)
            else:
                cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)
                
            debug_filename = f"{debug_folder}/env_{env_id_item}_folder_{folder_idx}_attempt_{spawn_attempt}.png"
            
            goal_visible_in_camera, ref_obj_visible, red_count, pink_count = self._check_target_in_img(
                file_name=debug_filename,
                cam_pov=cam_pov_np,
                return_counts=True,
                count_ref_obj=True
            )
            
            target_visible_in_camera = goal_visible_in_camera and ref_obj_visible
            camera_validation_passed = False
            
            # Since there will only be "left" or "right", they both require full visibility
            if visibility_category in ["left", "right"]:
                camera_validation_passed = target_visible_in_camera
                
                if not camera_validation_passed:
                    if self.verbose >= 1:
                        # Determine exactly why it failed
                        fail_reasons = []
                        if not goal_visible_in_camera:
                            fail_reasons.append(f"Goal missing (Red: {red_count}/{self.goal_pixel_threshold_cam})")
                        if not ref_obj_visible:
                            fail_reasons.append(f"Ref Obj missing (Pink: {pink_count}/{self.ref_obj_pixel_threshold_cam})")
                            
                        reason_str = " AND ".join(fail_reasons)
                        
                        print(
                            f"    ❌ Env {env_id_item} (folder {folder_idx}): Camera validation FAILED"
                        )
                        print(
                            f"       Expected: {visibility_category} (both visible), Reason: {reason_str}"
                        )
                        print(f"       Debug image saved: {debug_filename}")
                else:
                    if self.verbose >= 2:
                        print(
                            f"    ✅ Env {env_id_item} (folder {folder_idx}): Camera validation PASSED - {visibility_category} (Red: {red_count}, Pink: {pink_count})"
                        )
            else:
                if self.verbose >= 1:
                    print(f"    ⚠️ Env {env_id_item}: Unknown category '{visibility_category}', marking as failed.")
                camera_validation_passed = False

            # If validation failed, flag it for a retry
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
        camera_state = self._camera_obj.data.default_root_state[env_ids].clone()
        vpt_state = self._vpt_objects.data.default_object_state[env_ids].clone()
        # Clone default states for all 3 ref obj variants
        ref_states = [obj.data.default_root_state[env_ids].clone() for obj in self._ref_objs]

        # Pick 1 random ref obj per env, build the single state tensor that flows through spawn logic
        ref_obj_state = ref_states[0].clone()  # same shape, will be overwritten
        for local_idx, env_id in enumerate(env_ids):
            eid = env_id.item() if torch.is_tensor(env_id) else env_id
            choice = random.randint(0, 2)
            self.active_ref_idx[eid] = choice
            ref_obj_state[local_idx] = ref_states[choice][local_idx]

        # Retry Management
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

        # NO environments are displaced
        in_view_displaced = torch.tensor([], device=device, dtype=torch.long)

        # No environments are outside the FOV
        outside_fov_displaced = torch.tensor([], device=device, dtype=torch.long)

        # =================================================================
        # MAIN SPAWN LOOP
        # =================================================================
        spawn_attempt = -1
        while envs_need_spawn_retry.any():
            spawn_attempt += 1

            # --- A. Initial Spawn ---
            retry_mask = envs_need_spawn_retry.clone()
            if not rl_reset or self.mode == "rl_data_collection":
                envs_need_spawn_retry, states = self.initial_spawn_loop_rl(
                    env_ids=env_ids,
                    envs_need_spawn_retry=envs_need_spawn_retry,
                    safe_range=self.center_to_boundary,
                    states=[goal_state, camera_state, agent_state, vpt_state, ref_obj_state],
                    visibility_categories=visibility_categories,
                    allow_clipping=True,
                    device=device,
                )
            elif rl_reset and self.mode != "rl_data_collection":
                envs_need_spawn_retry, states = self.initial_spawn_loop_rl(
                    env_ids=env_ids,
                    envs_need_spawn_retry=envs_need_spawn_retry,
                    safe_range=self.center_to_boundary,
                    states=[goal_state, camera_state, agent_state, vpt_state, ref_obj_state],
                    visibility_categories=visibility_categories,
                    allow_clipping=True,
                    device=device,
                )

            goal_state, camera_state, agent_state, vpt_state, ref_obj_state = states
            # print(envs_need_spawn_retry)
            # print("INITIAL SPAWN LOOP")

            valid_mask = retry_mask & ~envs_need_spawn_retry
            if not valid_mask.any():
                continue

            valid_indices = torch.where(valid_mask)[0]
            valid_env_ids = env_ids[valid_indices]

            # --- B. Move Ball (Skipped in provided snippet, kept for structure) ---
            moved_vpt_for_ball = {i: None for i in range(num_envs)}
            move_ball_indices = torch.where(
                torch.isin(env_ids, getattr(self, "envs_to_move_ball", [])))[0]

            # --- C. Move VPT Objects + Z-Check ---
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
                env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id

                # 1. Env Origin
                origin_z = self.scene.env_origins[env_id, 2]

                # 2. Get Active Objects info
                active_indices = self.active_vpt_indices[env_id_item]

                # 3. Get Scaled Heights & Ratios
                heights = self.all_vpt_dims[env_id, active_indices, 2]
                ratios = self.vpt_z_offset_ratios[env_id, active_indices]

                # 4. Calculate & Write
                safe_z = origin_z + (heights * ratios)
                vpt_state[local_idx, active_indices, 2] = safe_z

            self.write_pose_to_sim(env_ids=valid_env_ids,
                                   indices=valid_indices,
                                   goal_default_state=goal_state,
                                   camera_obj_default_state=camera_state,
                                   agent_default_state=agent_state,
                                   vpt_obj_default_state=vpt_state,
                                   ref_obj_default_state=ref_obj_state)

            # Validation: Z-Bounds (Physics Check)
            goal_new_pos = self._goal.data.root_pos_w[valid_env_ids]
            camera_new_pos = self._camera_obj.data.root_pos_w[valid_env_ids]
            agent_new_pos = self._agent.data.root_pos_w[valid_env_ids]
            vpt_new_pos = self._get_active_vpt_positions(valid_env_ids,
                                                         base_pivoted=True)

            envs_need_spawn_retry = self.check_z_bounds(
                env_ids=env_ids,
                valid_indices=valid_indices,
                states=(goal_new_pos, camera_new_pos, agent_new_pos,
                        vpt_new_pos),
                envs_need_spawn_retry=envs_need_spawn_retry,
                tolerance=5e-2)
            # print(envs_need_spawn_retry)
            # print("Z BOUNDS")

            # Update valid mask after Z-check failures
            final_valid_mask = valid_mask & ~envs_need_spawn_retry
            if not final_valid_mask.any():
                continue

            final_valid_indices = torch.where(final_valid_mask)[0]
            final_valid_env_ids = env_ids[final_valid_indices]

            # --- D. Camera Posing ---
            self.outside_fov_camera_movement(
                valid_env_ids=final_valid_env_ids,
                valid_indices=final_valid_indices,
                visibility_categories=visibility_categories,
                states=[goal_state, camera_state, agent_state, vpt_state],
                device=device,
            )

            # --- E. Occlusion Validation (Raycast) ---
            occlusion_valid_mask, envs_need_spawn_retry, _, states = self.occlusion_validation_check(
                final_valid_env_ids=final_valid_env_ids,
                valid_indices=final_valid_indices,
                visibility_categories=visibility_categories,
                envs_need_spawn_retry=envs_need_spawn_retry,
                env_dict={},
                states=[goal_state, camera_state, agent_state, vpt_state],
                device=device,
                rl_mode=rl_reset and self.mode != "rl_data_collection")
            # print(occlusion_valid_mask)
            # print("OCCLUSION")

            # --- OPTIONAL: Full Data Collection Steps ---
            if not rl_reset or self.mode == "rl_data_collection":
                # --- F. Geometric Validation ---
                geometric_valid_mask, envs_need_spawn_retry = self.geometric_occlusion_check(
                    env_ids=final_valid_env_ids,
                    valid_indices=final_valid_indices,
                    occlusion_valid_mask=occlusion_valid_mask,
                    envs_need_spawn_retry=envs_need_spawn_retry,
                    device=device)
                # print(envs_need_spawn_retry)
                # print("GEOMTRIC")

                # --- G. Camera POV Validation ---
                envs_need_spawn_retry = self.camera_pov_validation(
                    env_ids=final_valid_env_ids,
                    valid_indices=final_valid_indices,
                    geometric_valid_mask=geometric_valid_mask,
                    visibility_categories=visibility_categories,
                    envs_need_spawn_retry=envs_need_spawn_retry,
                    folder_indices=folder_indices,
                    spawn_attempt=spawn_attempt)
                # print(envs_need_spawn_retry)
                # print("CAM POV")

            # =================================================================
            # FINAL VALIDATION (INSIDE WHILE LOOP)
            # =================================================================
            
            # FIX 1: Filter success to ONLY the envs that passed during THIS attempt
            success_mask = retry_mask & ~envs_need_spawn_retry
            # print(f"Retry Mask: {retry_mask}")
            # print(f"Env retry NOT: {~envs_need_spawn_retry}")
            
            successful_env_ids = env_ids[success_mask]
            successful_local_indices = torch.where(success_mask)[0]
            # print(successful_env_ids)

            if len(successful_env_ids) > 0:
                # 1. Finalize Agent Orientation & Position for successful envs
                restricted_agent_positioning = True
                if restricted_agent_positioning:
                    camera_positions = camera_state[successful_local_indices, :2] - self.scene.env_origins[successful_env_ids][:, :2]
                else:
                    camera_positions = None
                
                # FIX 2: Slice the states being passed into shotgun to avoid global dimension broadcasting
                agent_state[successful_local_indices] = self.place_object_safely(
                    env_ids=successful_env_ids,
                    object_state=agent_state[successful_local_indices],
                    vpt_state=vpt_state[successful_local_indices],
                    safe_range=(self.center_to_boundary - 5.0) if restricted_agent_positioning else self.center_to_boundary - 1.0,
                    range_offsets=camera_positions,
                    object_type='agent')
                # print("AGENT PLACED")

                # --- H. Circle Generation ---
                if not rl_reset or getattr(self, "mode", "rl") == "rl_data_collection":
                    # print("CIRCLE START")
                    subset_valid_points = self.generate_valid_circle_points(
                        env_ids=successful_env_ids,
                        angle_step=2.0,
                        max_attempts=100)
                    # print("CIRCLES")

                    if getattr(self, "valid_viewpoint_poses", None) is None:
                        self.valid_viewpoint_poses = [None] * self.num_envs

                    # req_points = 1 if getattr(self, "mode", "rl") == "rl_data_collection" else getattr(self, "images_per_env", 1)
                    req_points = self.images_per_env

                    # Assign success envs
                    for i, env_id in enumerate(successful_env_ids):
                        eid = env_id.item() if torch.is_tensor(env_id) else env_id
                        points_2d = subset_valid_points[i]
                        local_idx = successful_local_indices[i]

                        if points_2d.shape[0] >= req_points:
                            agent_z = self._agent.data.default_root_state[env_id, 2]
                            points_3d = torch.zeros((points_2d.shape[0], 3), device=device)
                            points_3d[:, :2] = points_2d
                            points_3d[:, 2] = agent_z
                            self.valid_viewpoint_poses[local_idx] = points_3d
                            
                            if getattr(self, "mode", "rl") == "rl_data_collection":
                                target_pos_2d = points_2d[0]
                                agent_state[local_idx, :2] = target_pos_2d
                                
                                camera_pos = camera_state[local_idx, :2]
                                goal_pos = goal_state[local_idx, :2]
                                midpoint = (camera_pos + goal_pos) / 2.0
                                
                                direction = midpoint - target_pos_2d
                                yaw = torch.atan2(direction[1], direction[0])
                                
                                zero_t = torch.zeros_like(yaw)
                                quat = quat_from_euler_xyz(zero_t, zero_t, yaw)
                                agent_state[local_idx, 3:7] = quat
                        else:
                            self.valid_viewpoint_poses[local_idx] = torch.zeros((0, 3), device=device)
                            
                            # Triggers a retry since we are still inside the while loop!
                            if getattr(self, "mode", "rl") == "rl_data_collection":
                                envs_need_spawn_retry[local_idx] = True

            # Clear failed envs for this attempt
            failed_mask = envs_need_spawn_retry
            # --- FIX: Get the local indices of the failed environments ---
            failed_local_indices = torch.where(failed_mask)[0]

            if getattr(self, "valid_viewpoint_poses", None) is not None:
                for local_idx in failed_local_indices:
                    # --- FIX: Wipe the local_idx slot ---
                    self.valid_viewpoint_poses[local_idx.item()] = torch.zeros((0, 3), device=device)

            # === SINGLE-PASS GUARD ===
            if self.mode != "rl_data_collection":
                break

            # FIX 3: Update this mask so we don't redundantly write poses for older envs that already finished
            if getattr(self, "mode", "rl") == "rl_data_collection" and success_mask.any():
                self.write_pose_to_sim(env_ids=env_ids[success_mask],
                                       indices=torch.where(success_mask)[0],
                                       agent_default_state=agent_state,
                                       device=device)

        # =================================================================
        # POST-LOOP FINALIZATION
        # =================================================================
        # 2. Final Write (Executes only once the while loop completes for ALL environments)
        self.write_pose_to_sim(env_ids=env_ids,
                            indices=torch.arange(len(env_ids), device=device),
                            vpt_obj_default_state=vpt_state,
                            agent_default_state=agent_state,
                            ref_obj_default_state=ref_obj_state)

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

        # Camera Checks
        cam_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
        
        # NEW: +/- 5 unit square from camera position
        # in_camera_square = torch.all(torch.abs(points - cam_pos) <= 5.0, dim=1)
        # valid_mask &= in_camera_square
        
        # if not valid_mask.any(): return valid_mask

        # Minimum distance from camera
        valid_mask &= (torch.norm(points - cam_pos, dim=1) >= min_cam_target_dist)
        
        if not valid_mask.any(): return valid_mask

        # Active Obstacle Positions
        active_obs_pos = self._get_active_obstacle_positions(env_ids)

        # Point (Goal) -> Obstacle Check
        # Calculates distance from every candidate point to every obstacle
        dist_pt_obs = torch.norm(points.unsqueeze(1) - active_obs_pos, dim=2)

        # Enforce min_target_obs_dist here
        valid_mask &= (dist_pt_obs.min(dim=1)[0] >= min_target_obs_dist)

        # Camera -> Obstacle Check
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
            g_vis, c_vis, r_vis = self.check_batch_object_visibility(b_envs)
            is_vis = g_vis & c_vis & r_vis

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
        """Teleports agents, orients to midpoints with jitter, and steps physics."""
        cam_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
        goal_pos = self._goal.data.root_pos_w[env_ids, :2]

        dirs = ((cam_pos + goal_pos) / 2.0) - points
        yaws = torch.atan2(dirs[:, 1], dirs[:, 0])

        # ±15° yaw jitter for viewpoint diversity during FOV validation
        yaw_jitter = (torch.rand(len(env_ids), device=points.device) - 0.5) * 2 * math.radians(15)
        yaws = yaws + yaw_jitter

        pos = torch.zeros((len(env_ids), 3), device=points.device)
        pos[:, :2] = points
        pos[:, 2] = self._agent.data.default_root_state[env_ids, 2]

        quat = torch.zeros((len(env_ids), 4), device=points.device)
        quat[:, 0] = torch.cos(yaws / 2)
        quat[:, 3] = torch.sin(yaws / 2)

        self._agent.write_root_com_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)

        self.sim.step(render=True)
        self._occlusion_camera.update(self.sim.cfg.dt)
        self._rgb_tiled_camera.update(self.sim.cfg.dt)
        self._agent.update(self.sim.cfg.dt)
        self._camera_obj.update(self.sim.cfg.dt)
        self._goal.update(self.sim.cfg.dt)
        self._vpt_objects.update(self.sim.cfg.dt)


    def generate_valid_circle_points(
            self,
            env_ids: torch.Tensor,
            angle_step: float = 2.0,
            max_attempts: int = 300,
            mode: str = None) -> List[torch.Tensor]:
        """Generate valid viewpoint positions in parallel for all environments."""
        device = self.device
        num_envs = len(env_ids)

        if mode is None:
            mode = self.mode

        # Dynamically set thresholds based on mode
        MIN_REQUIRED_POINTS = 1 if mode == "rl_data_collection" else self.images_per_env
        MIN_CANDIDATES_FOR_FOV = 5 if mode == "rl_data_collection" else 40

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

        # Variable scale factor [1.1, 1.5] for viewpoint diversity
        scale_factor = (torch.rand(num_envs, device=device) * 0.4) + 1.1
        radii = radii * scale_factor
        radii = radii.unsqueeze(1)  # [num_envs, 1]

        angles_expanded = angles.unsqueeze(0).expand(num_envs, -1)

        # Radial jitter: ±20% of radius for natural spread
        radial_jitter = (torch.rand(num_envs, num_angles, device=device) * 0.4 - 0.2)
        jitter_scale = 1.0 + radial_jitter  # [0.8, 1.2] multiplier

        all_x = self._goal.data.root_pos_w[env_ids, 0].unsqueeze(1) + (radii * jitter_scale) * torch.cos(angles_expanded)
        all_y = self._goal.data.root_pos_w[env_ids, 1].unsqueeze(1) + (radii * jitter_scale) * torch.sin(angles_expanded)

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
            min_required_points=MIN_REQUIRED_POINTS  # This dynamically scales!
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
                # In rl_data_collection mode, this returns exactly 1 point due to early stopping!
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


    def _collect_images_for_slot(self, env_id: torch.Tensor, folder_idx: int) -> None:
        """Collect images for a specific slot/environment."""
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        device = self.device
        global_env_id = self.slot_to_env_id[env_id_item]

        if self.verbose >= 1:
            print(f"    📸 Collecting {self.images_per_env} images for slot {env_id_item}, env {global_env_id}, folder {folder_idx}")

        viewpoints = self.selected_viewpoints_for_collection[env_id_item]
        if viewpoints is None:
            raise RuntimeError(f"No viewpoints selected for slot {env_id_item} (env {global_env_id})")

        single_env_tensor = torch.tensor([env_id_item], dtype=torch.long, device=device)
        zero_velocity = torch.zeros((1, 6), device=device)

        cam_pos_2d = self._camera_obj.data.root_pos_w[env_id_item, :2]
        goal_pos_2d = self._goal.data.root_pos_w[env_id_item, :2]
        midpoint = (cam_pos_2d + goal_pos_2d) / 2.0
        agent_z = self._agent.data.default_root_state[env_id_item, 2]

        for viewpoint_idx in range(self.images_per_env):
            target_pos_2d = viewpoints[viewpoint_idx]
            if target_pos_2d.shape[-1] == 3:
                target_pos_2d = target_pos_2d[:2]

            target_pos_3d = torch.zeros(3, device=device)
            target_pos_3d[:2] = target_pos_2d
            target_pos_3d[2] = agent_z

            direction = midpoint - target_pos_3d[:2]
            yaw = torch.atan2(direction[1], direction[0])

            # ±20° yaw jitter for collection diversity
            yaw_jitter = (torch.rand(1, device=device) - 0.5) * 2 * math.radians(20)
            yaw = yaw + yaw_jitter.squeeze()

            quat = torch.zeros(4, device=device)
            quat[0] = torch.cos(yaw / 2)
            quat[3] = torch.sin(yaw / 2)

            pose = torch.cat([target_pos_3d, quat]).unsqueeze(0)
            self._agent.write_root_com_pose_to_sim(pose, single_env_tensor)
            self._agent.write_root_com_velocity_to_sim(zero_velocity, single_env_tensor)

            for _ in range(10):
                self.sim.step()
                self._rgb_tiled_camera.update(self.sim.cfg.dt)
                if self.save_camera_pov:
                    self._occlusion_camera.update(self.sim.cfg.dt)

            rgb_data = self._rgb_tiled_camera.data.output["rgb"]
            depth_data = self._rgb_tiled_camera.data.output["distance_to_camera"]
            semantic_data = self._rgb_tiled_camera.data.output["semantic_segmentation"]
            camera_pov_data = self._occlusion_camera.data.output["semantic_segmentation"] if self.save_camera_pov else None

            self._save_single_image(env_id_item, folder_idx, rgb_data,
                                    depth_data, semantic_data, camera_pov_data, viewpoint_idx)

        self._save_env_config_to_json(env_id_item, folder_idx)
        self.selected_viewpoints_for_collection[env_id_item] = None

        if self.verbose >= 1:
            print(f"    ✅ Collected and saved {self.images_per_env} images for folder {folder_idx}")


    def _save_single_image(self,
                        env_id_item: int,
                        folder_idx: int,
                        rgb_data: torch.Tensor,
                        depth_data: torch.Tensor,
                        semantic_data: torch.Tensor,
                        camera_pov_data: torch.Tensor = None,
                        image_idx: int = 0) -> None:
        """Save RGB, Depth, Semantic, and optional cam_pov images."""
        if folder_idx not in self.env_visibility_labels:
            raise RuntimeError(f"No visibility label found for folder_idx {folder_idx}!")

        visibility_label = self.env_visibility_labels[folder_idx]
        if visibility_label not in ["left", "right"]:
            raise RuntimeError(f"Invalid visibility label '{visibility_label}' for folder_idx {folder_idx}!")

        rgb_env_folder = f"{self.base_path}/RGB/{visibility_label}/env_{folder_idx}"
        depth_env_folder = f"{self.base_path}/Depth/{visibility_label}/env_{folder_idx}"
        semantic_env_folder = f"{self.base_path}/Semantic/{visibility_label}/env_{folder_idx}"
        cam_env_folder = f"{self.base_path}/cam/{visibility_label}/env_{folder_idx}"

        # Create dirs once per env on first image
        if image_idx == 0:
            os.makedirs(rgb_env_folder, exist_ok=True)
            os.makedirs(depth_env_folder, exist_ok=True)
            os.makedirs(semantic_env_folder, exist_ok=True)
            if self.save_camera_pov:
                os.makedirs(cam_env_folder, exist_ok=True)

        # --- RGB ---
        rgb_img = rgb_data[env_id_item, :, :, :3]
        rgb_np = rgb_img.cpu().numpy()
        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255.0).astype(np.uint8)
        else:
            rgb_np = rgb_np.astype(np.uint8)
        cv2.imwrite(f"{rgb_env_folder}/image_{image_idx:04d}.png", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # --- Depth ---
        depth_img = depth_data[env_id_item, :, :, :]
        depth_np = depth_img.cpu().numpy()
        valid_mask = ~np.isinf(depth_np)
        depth_np[~valid_mask] = depth_np[valid_mask].max() if valid_mask.any() else 0
        if depth_np.max() > depth_np.min():
            depth_normalized = ((depth_np - depth_np.min()) /
                                (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)
        cv2.imwrite(f"{depth_env_folder}/image_{image_idx:04d}.png", depth_normalized)

        # --- Semantic ---
        if semantic_data is not None:
            sem_img = semantic_data[env_id_item, :, :, :3]
            sem_np = sem_img.cpu().numpy()
            if sem_np.max() <= 1.0:
                sem_np = (sem_np * 255.0).astype(np.uint8)
            else:
                sem_np = sem_np.astype(np.uint8)
            cv2.imwrite(f"{semantic_env_folder}/image_{image_idx:04d}.png", cv2.cvtColor(sem_np, cv2.COLOR_RGB2BGR))

        # --- Camera POV (once per env) ---
        if self.save_camera_pov and camera_pov_data is not None and image_idx == 0:
            cam_pov_path = f"{cam_env_folder}/cam_pov.png"
            if not os.path.exists(cam_pov_path):
                cam_pov_img = camera_pov_data[env_id_item, :, :, :3]
                cam_pov_np = cam_pov_img.cpu().numpy()
                if cam_pov_np.max() <= 1.0:
                    cam_pov_np = (cam_pov_np * 255.0).astype(np.uint8)
                else:
                    cam_pov_np = cam_pov_np.astype(np.uint8)
                cv2.imwrite(cam_pov_path, cv2.cvtColor(cam_pov_np, cv2.COLOR_RGB2BGR))

    def _cache_base_dims(self):
        """
        Caches the base dimensions of all rigid objects from the config.
        For UsdAssets use scale.
        For MeshCuboids use size [x, y, z].
        For MeshCylinders and MeshCones use radius and height (x=2r, y=2r, z=h).
        """
        self.vpt_base_dims = []

        # Iterate over the config dictionary to preserve order matching object indices
        for key, obj_cfg in self.cfg.vpt_objects.rigid_objects.items():
            spawn_cfg = obj_cfg.spawn
            
            # Default fallback
            dims = torch.tensor([1.0, 1.0, 1.0], device=self.device)

            if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                # UsdAssets use scale. Default to (1,1,1) if None
                scale = getattr(spawn_cfg, "scale", (1.0, 1.0, 1.0))
                if scale is None:
                    scale = [1.0, 1.0, 1.0]
                
                # Extract filename to check for special scaling rules
                filename = spawn_cfg.usd_path.split("/")[-1].split(".")[0]

                # Special Case 1: Furniture (Table_A, Table_B, Bench)
                # These use standard 1.0 scaling
                if filename.endswith(('Table_A', 'Table_B', 'Bench')):
                    dims = torch.tensor(scale, device=self.device)
                
                # Special Case 2: Letters (X, L, T, I, A, H, Z)
                # These use a 0.25 multiplier on the Y axis
                elif filename.endswith(('X', 'L', 'T', 'I', 'H', 'A', 'Z')):
                    dims = torch.tensor(
                        [1.0 * scale[0], 0.25 * scale[1], 1.0 * scale[2]], 
                        device=self.device
                    )
                # Fallback for any other USDs
                else:
                    dims = torch.tensor(scale, device=self.device)

            # Updated Primitive Names: MeshCuboid, MeshCone, MeshCylinder
            elif isinstance(spawn_cfg, sim_utils.MeshCuboidCfg):
                # Cuboids use size (x, y, z)
                dims = torch.tensor(spawn_cfg.size, device=self.device)

            elif isinstance(spawn_cfg, (sim_utils.MeshConeCfg, sim_utils.MeshCylinderCfg)):
                # Cylinders/Cones use radius and height
                # x = 2*r, y = 2*r, z = h
                r = spawn_cfg.radius
                h = spawn_cfg.height
                dims = torch.tensor([2 * r, 2 * r, h], device=self.device)

            self.vpt_base_dims.append(dims)

        # Stack into a single tensor of shape (num_objs, 3)
        if len(self.vpt_base_dims) > 0:
            self.vpt_base_dims = torch.stack(self.vpt_base_dims)
        else:
            self.vpt_base_dims = torch.empty((0, 3), device=self.device)

        if self.verbose >= 1:
            print(f"📦 Cached base dimensions for {len(self.vpt_base_dims)} objects.")

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

        return base[0], base[1], base[2]

    def write_pose_to_sim(self,
                          env_ids: torch.Tensor,
                          indices: torch.Tensor,
                          goal_default_state: torch.Tensor = None,
                          camera_obj_default_state: torch.Tensor = None,
                          agent_default_state: torch.Tensor = None,
                          vpt_obj_default_state: torch.Tensor = None,
                          ref_obj_default_state: torch.Tensor = None,
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
        
        if ref_obj_default_state is not None:
            # Build per-variant state: active gets placed position, inactive get storage
            storage_pose = torch.tensor(
                [self.storage_position[0], self.storage_position[1], self.storage_position[2],
                1.0, 0.0, 0.0, 0.0], device=device)

            for variant_idx, ref_obj in enumerate(self._ref_objs):
                variant_poses = storage_pose.unsqueeze(0).expand(len(env_ids), -1).clone()
                for batch_i, env_id in enumerate(env_ids):
                    eid = env_id.item() if torch.is_tensor(env_id) else env_id
                    if self.active_ref_idx[eid] == variant_idx:
                        variant_poses[batch_i] = ref_obj_default_state[indices[batch_i], :7]

                ref_obj.write_root_com_pose_to_sim(variant_poses, env_ids)
                ref_obj.write_root_com_velocity_to_sim(
                    torch.zeros((len(env_ids), 6), device=device), env_ids)

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

    def _get_object_corners(self, pos, quat, object_type='agent'):
        """
        Generates world-space corners for various objects.
        Args:
            pos: (N, 3) object positions
            quat: (N, 4) object rotations
            object_type: str, type of object ('agent', 'cam_obj', 'goal')
        Returns:
            (N, 1, 8, 3) tensor matching the shape structure of the obstacle cache
        """
        # --- CONFIG: OBJECT SIZE ---
        if object_type == 'agent':
            half_extents = torch.tensor([0.1, 0.1, 0.1], device=self.device)
        elif object_type == 'cam_obj':
            half_extents = torch.tensor([0.55, 0.575, 0.45], device=self.device)
        elif object_type == 'goal':
            # A bounding box for a sphere uses the radius for all half-extents
            half_extents = torch.tensor([0.2, 0.2, 0.2], device=self.device)
        elif object_type == 'ref_obj':
            # Adjust these half-extents to match your reference object's dimensions
            half_extents = torch.tensor([0.2, 0.2, 0.25], device=self.device)
        else:
            raise ValueError(f"Unknown object type: {object_type}")

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
                                     proposed_quat, object_type='agent'):
        """
        Pure 2D SAT implementation (XY Plane only) + Wall Boundary Check.
        Ignores Z-height entirely.
        Returns: Boolean Tensor (N,) where True = Collision Detected.
        """
        # 1. Get Agent Corners at PROPOSED position
        # Shape: (N, 1, 8, 3)
        agent_corners = self._get_object_corners(proposed_pos, proposed_quat, object_type=object_type)

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

    def place_object_safely_old(self, env_ids: torch.Tensor,
                           object_state: torch.Tensor, vpt_state: torch.Tensor,
                           safe_range: float, range_offsets: torch.Tensor = None, object_type: str = 'agent') -> torch.Tensor:
        """
        Updates the obstacle cache and finds a collision-free spawn pose for the object 
        using vectorized batch rejection sampling ("Shotgun" method).

        Args:
            env_ids (torch.Tensor): Tensor of environment indices involved in this spawn.
            object_state (torch.Tensor): The state tensor to update (N, 13).
            vpt_state (torch.Tensor): The obstacle state tensor (N, NumObjs, 13).
            safe_range (float): Maximum +/- deviation from the environment origin.
            object_type (str): The type of object being placed ('agent', 'cam_obj', 'goal').

        Returns:
            torch.Tensor: The updated state with valid positions and random orientations.
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
        
        if range_offsets is not None:
            rand_xy += range_offsets.unsqueeze(1)
            #clip to ensure we don't go out of bounds of the environment
            rand_xy = torch.clamp(rand_xy, -self.center_to_boundary.item()-1.0, self.center_to_boundary.item()-1.0)
        
        rand_yaw = sample_uniform(0, 2 * math.pi, (num_envs, BATCH_SIZE),
                                  device)

        cands_pos = torch.zeros((num_envs, BATCH_SIZE, 3), device=device)
        cands_pos[..., :2] = env_origins[:, :2].unsqueeze(1) + rand_xy
        cands_pos[..., 2] = object_state[0, 2]  # Preserve original height

        zeros = torch.zeros_like(rand_yaw)
        cands_quat = quat_from_euler_xyz(zeros, zeros, rand_yaw)

        # 4. Vectorized Collision Check
        # Flatten batch dims for the check function: (N * BatchSize)
        flat_ids = env_ids.repeat_interleave(BATCH_SIZE)
        
        # EXPLICITLY PASSING OBJECT TYPE HERE
        flat_cols = self._check_collisions_vectorized(
            flat_ids, 
            cands_pos.reshape(-1, 3),
            cands_quat.reshape(-1, 4),
            object_type=object_type
        ).reshape(num_envs, BATCH_SIZE)

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
                f"⚠️ Shotgun Fail: {unsafe_mask.sum().item()} envs using unsafe fallback for {object_type}."
            )

        # 6. Extract Winners & Update State
        batch_idx = torch.arange(num_envs, device=device)
        object_state[:, :3] = cands_pos[batch_idx, valid_indices]
        object_state[:, 3:7] = cands_quat[batch_idx, valid_indices]

        return object_state

    def place_object_safely(self, env_ids: torch.Tensor,
                           object_state: torch.Tensor, vpt_state: torch.Tensor,
                           safe_range: float, range_offsets: torch.Tensor = None, 
                           object_type: str = 'agent',
                           custom_centers: torch.Tensor = None) -> torch.Tensor:
        """
        Updates the obstacle cache and finds a collision-free spawn pose for the object 
        using vectorized batch rejection sampling ("Shotgun" method).
        """
        device = self.device
        num_envs = len(env_ids)
        BATCH_SIZE = 1000

        # 1. Update Obstacle Cache
        self.update_obb_cache(env_ids, vpt_state)

        # 2. Determine base coordinates for the scattering
        if custom_centers is not None:
            # Shape expected: (num_envs, 2) or (num_envs, 3)
            base_xy = custom_centers[:, :2].unsqueeze(1)
        else:
            base_xy = self.scene.env_origins[env_ids][:, :2].unsqueeze(1)

        if range_offsets is not None:
            base_xy += range_offsets.unsqueeze(1)

        # 3. Generate Batch Candidates (Pos & Rot)
        rand_xy = sample_uniform(-safe_range, safe_range,
                                 (num_envs, BATCH_SIZE, 2), device)
        rand_yaw = sample_uniform(0, 2 * math.pi, (num_envs, BATCH_SIZE),
                                  device)

        cands_pos = torch.zeros((num_envs, BATCH_SIZE, 3), device=device)
        cands_pos[..., :2] = base_xy + rand_xy
        cands_pos[..., 2] = object_state[0, 2]  # Preserve original height

        zeros = torch.zeros_like(rand_yaw)
        cands_quat = quat_from_euler_xyz(zeros, zeros, rand_yaw)

        # 4. Vectorized Collision Check
        flat_ids = env_ids.repeat_interleave(BATCH_SIZE)
        
        flat_cols = self._check_collisions_vectorized(
            flat_ids, 
            cands_pos.reshape(-1, 3),
            cands_quat.reshape(-1, 4),
            object_type=object_type
        ).reshape(num_envs, BATCH_SIZE)

        # 5. Select First Valid Candidate
        valid_indices = torch.argmax((~flat_cols).int(), dim=1)

        unsafe_mask = torch.gather(flat_cols, 1,
                                   valid_indices.unsqueeze(1)).squeeze(1)
        if unsafe_mask.any():
            print(
                f"⚠️ Shotgun Fail: {unsafe_mask.sum().item()} envs using unsafe fallback for {object_type}."
            )

        # 6. Extract Winners & Update State
        batch_idx = torch.arange(num_envs, device=device)
        object_state[:, :3] = cands_pos[batch_idx, valid_indices]
        object_state[:, 3:7] = cands_quat[batch_idx, valid_indices]

        return object_state

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
    
        world = World.instance()
        if world.is_playing():
            world.pause()

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
        
        world.play()
        self.sim.step(render=False)