from __future__ import annotations

import hashlib
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
import json
import matplotlib.pyplot as plt

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.api import World
from pxr import Gf, Sdf, UsdGeom, Usd

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.utils import math as math_utils

from .vpt_env_cfg_v15_rl import VPTEnvCfg
from .spawn_boundary import get_vpt_material_paths, get_mat_material_paths
from .env_timer import EnvTimer


class VPTEnv(DirectRLEnv):

    cfg: VPTEnvCfg

    def __init__(self, cfg: VPTEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- Configuration ---
        self.action_scale = self.cfg.action_scale
        self.boundary_limits = self.cfg.boundary_limits
        self.agent_height = self.cfg.agent_height
        self.agent_camera_pitch = self.cfg.agent_camera_pitch
        self.goal_radius = self.cfg.goal_radius
        self.config_file = self.cfg.config_file
        self.center_to_boundary = torch.abs(torch.tensor(self.boundary_limits).view(-1)[0])

        # --- VPT Object State ---
        self.num_objs = self.cfg.num_vpt_objs
        self.active_vpt_objs = self.cfg.objects_per_env
        self.storage_position = torch.tensor([250.0, 250.0, 0.0], device=self.device)
        self.active_vpt_indices = [None] * self.num_envs
        self.moved_vpt_objs = [[] for _ in range(self.num_envs)]
        self.used_vpt_objects = set()

        # --- Data Collection Settings ---
        self.mode = "testing" if (self.config_file and os.path.exists(self.config_file)) else "data_collection"
        self.images_per_env = 10
        self.min_viewpoint_distance = 0.1
        self.save_camera_pov = True
        self.goal_pixel_threshold = 500
        self.goal_pixel_threshold_occlusion = 500
        self.camera_pixel_threshold = 1000
        self.verbose = 2

        # --- Collection Counters & State ---
        self.valid_viewpoint_poses = [None] * self.num_envs
        self.selected_viewpoints_for_collection = [None] * self.num_envs
        self.used_viewpoint_indices = [set() for _ in range(self.num_envs)]

        # Pre-allocated tensors on device
        self.current_collection_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.viewpoint_pose_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # --- Environment Management ---
        self.total_envs_to_sim = 2000
        self.slot_to_env_id = list(range(self.num_envs))
        self.next_env_id = self.num_envs
        self.completed_envs = set()
        self.slot_attempt_counts = [0] * self.num_envs
        self.max_attempts_per_slot = 1000
        self.next_env_folder_idx = 0
        self.env_visibility_labels = {}
        self.env_visibility_reasons = {}
        self._reset_called = False
        self.times = {}

        # --- Depth / Metadata Tracking ---
        self.env_metadata_cache = {}
        self.depth_labels_data = {}

        self._preallocate_visibility_labels()

        # --- Paths & GPU ---
        self.GPU_ID = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.NODE_ID = os.getenv("NODE_ID", os.getenv("SLURM_ARRAY_TASK_ID", "0"))
        base = os.getenv("BASE_PATH", "/oscar/scratch/arock3/VPT1_DATA/v18_2_depth")
        self.base_path = f"{base}/data/data_node{self.NODE_ID}_gpu{self.GPU_ID}"

        print("*" * 50)
        print(f"🚀 Initializing VPTEnv (v18_depth) on Node {self.NODE_ID} GPU {self.GPU_ID}...")
        print("*" * 50)

        self.visibility_labels_json_path = os.path.join(self.base_path, "visibility_labels.json")
        self.depth_labels_json_path = os.path.join(self.base_path, "depth_labels.json")

        # --- Precomputed Math ---
        self.theta = math.pi / 12
        self.half_theta = self.theta / 2

        c, s = math.cos(self.half_theta), math.sin(self.half_theta)
        self.rot_q_left = torch.tensor([[c, 0., 0., s]], device=self.device)
        self.rot_q_right = torch.tensor([[c, 0., 0., -s]], device=self.device)

    def close(self):
        super().close()

    # =========================================================================
    # SCENE SETUP
    # =========================================================================

    def _cache_valid_shapes(self):
        """Cache boolean mask for objects valid for ball placement (Cylinders/Cuboids)."""
        self.valid_shape_mask = torch.zeros(self.num_objs, dtype=torch.bool, device=self.device)
        vpt_keys = list(self.cfg.vpt_objects.rigid_objects.keys())
        for i, key in enumerate(vpt_keys):
            spawn_cfg = self.cfg.vpt_objects.rigid_objects[key].spawn
            if isinstance(spawn_cfg, (sim_utils.CylinderCfg, sim_utils.CuboidCfg)):
                self.valid_shape_mask[i] = True

    def _preallocate_visibility_labels(self) -> None:
        """Pre-allocate visibility labels for all environments in 50/25/25 proportion."""
        if hasattr(self, 'visibility_label_pool') and self.visibility_label_pool:
            return
        total = self.total_envs_to_sim
        num_in_view = total // 2
        num_occluded = total // 4
        num_outside_fov = total - num_in_view - num_occluded

        all_labels = (["in_view"] * num_in_view + ["occluded"] * num_occluded +
                      ["outside_fov"] * num_outside_fov)
        random.shuffle(all_labels)
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
        elif category == "outside_fov":
            self.env_visibility_labels[folder_idx] = "No"
            self.env_visibility_reasons[folder_idx] = "outside_fov"

        return category

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(1000, 1000)))

        self._agent = RigidObject(self.cfg.agent)
        self._goal = RigidObject(self.cfg.goal_ball)
        self._mat = RigidObject(self.cfg.mat)
        self._camera_obj = RigidObject(self.cfg.camera_obj)
        self._vpt_objects = RigidObjectCollection(self.cfg.vpt_objects)

        self._boundary_top = RigidObject(self.cfg.top_wall)
        self._boundary_bottom = RigidObject(self.cfg.bottom_wall)
        self._boundary_left = RigidObject(self.cfg.left_wall)
        self._boundary_right = RigidObject(self.cfg.right_wall)

        self.scene.rigid_objects.update({
            "agent": self._agent,
            "goal": self._goal,
            "mat": self._mat,
            "camera_object": self._camera_obj,
            "boundary_top": self._boundary_top,
            "boundary_bottom": self._boundary_bottom,
            "boundary_left": self._boundary_left,
            "boundary_right": self._boundary_right,
        })
        self.scene.rigid_object_collections["vpt_objects"] = self._vpt_objects

        self._rgb_tiled_camera = TiledCamera(self.cfg.rgb_tiled_camera)
        self._occlusion_camera = TiledCamera(self.cfg.occlusion_camera)

        self.scene.sensors.update({
            "rgb_tiled_camera": self._rgb_tiled_camera,
            "occlusion_camera": self._occlusion_camera,
        })

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.SphereLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/envs/env_0/Light_A", light_cfg)

        self.mat_material_configs = self.get_material_configs(material_type="mat")
        self.vpt_material_configs = self.get_material_configs(material_type="vpt")

        self.mat_material_paths = []
        for idx, material in enumerate(self.mat_material_configs):
            path = f"/World/Looks/mat_material_{idx}"
            material.func(path, material)
            self.mat_material_paths.append(path)

        self.vpt_material_paths = []
        for idx, material in enumerate(self.vpt_material_configs):
            path = f"/World/Looks/vpt_material_{idx}"
            material.func(path, material)
            self.vpt_material_paths.append(path)

    # =========================================================================
    # CORE ENV METHODS
    # =========================================================================

    def check_batch_object_visibility(
            self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Check object visibility for a batch of environments in parallel."""
        sem_imgs = self._rgb_tiled_camera.data.output["semantic_segmentation"][env_ids]

        r = sem_imgs[..., 0]
        g = sem_imgs[..., 1]
        b = sem_imgs[..., 2]
        a = sem_imgs[..., 3]

        red_mask = (r == 255) & (g == 0) & (b == 0) & (a == 255)
        green_mask = (r == 0) & (g == 255) & (b == 0) & (a == 255)

        red_counts = red_mask.sum(dim=(1, 2))
        green_counts = green_mask.sum(dim=(1, 2))

        goal_visible = red_counts >= self.goal_pixel_threshold
        camera_visible = green_counts >= self.camera_pixel_threshold

        both_visible_mask = goal_visible & camera_visible
        ids_with_both = env_ids[both_visible_mask]
        if len(ids_with_both) > 0:
            print(f"Envs with both visible: {ids_with_both.tolist()}")

        return goal_visible, camera_visible

    def move_agent(self, actions: torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        reset_mask_5 = (actions == 5)
        reset_mask_6 = (actions == 6)
        physics_mask = ~(reset_mask_5 | reset_mask_6)

        if reset_mask_5.any():
            self._reset_idx(env_ids[reset_mask_5], rl_reset=False)

        if reset_mask_6.any():
            print("+" * 50)
            self._reset_idx(env_ids[reset_mask_6], rl_reset=True)

        if not physics_mask.any():
            return

        phys_ids = env_ids[physics_mask]
        phys_actions = actions[physics_mask]
        dt = self.cfg.sim.dt

        current_pos = self._agent.data.root_pos_w[phys_ids]
        current_quat = self._agent.data.root_quat_w[phys_ids]

        tentative_pos = current_pos.clone()
        new_quat = current_quat.clone()

        mask_left = (phys_actions == 2)
        mask_right = (phys_actions == 3)

        if mask_left.any():
            new_quat[mask_left] = math_utils.quat_mul(
                current_quat[mask_left],
                self.rot_q_left.expand(mask_left.sum(), -1))

        if mask_right.any():
            new_quat[mask_right] = math_utils.quat_mul(
                current_quat[mask_right],
                self.rot_q_right.expand(mask_right.sum(), -1))

        mask_fwd = (phys_actions == 0)
        mask_bwd = (phys_actions == 1)
        moving_mask = mask_fwd | mask_bwd

        if moving_mask.any():
            n_moving = moving_mask.sum()
            local_move = torch.zeros((n_moving, 3), device=self.device)
            moving_actions = phys_actions[moving_mask]
            local_move[moving_actions == 0, 0] = 1.0
            local_move[moving_actions == 1, 0] = -1.0
            world_vel = math_utils.quat_apply(current_quat[moving_mask], local_move)
            tentative_pos[moving_mask] += (world_vel * 2.0 * dt)

        tentative_pos[:, 2] = self._agent.data.default_root_state[phys_ids, 2]

        if moving_mask.any():
            is_collision = self._check_collisions_vectorized(phys_ids, tentative_pos, new_quat)
            collision_mask = is_collision & moving_mask
            if collision_mask.any():
                revert_indices = torch.where(collision_mask)[0]
                tentative_pos[revert_indices] = current_pos[revert_indices]
                new_quat[revert_indices] = current_quat[revert_indices]

        new_pose = torch.cat([tentative_pos, new_quat], dim=1)
        self._agent.write_root_com_pose_to_sim(new_pose, phys_ids)
        self._agent.reset()

    def _update_camera_poses(self, env_ids):
        camera_obj_pos = self._camera_obj.data.root_pos_w[env_ids].clone()
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_ids].clone()

        half_theta = (math.pi / 2) / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta), 0.0, 0.0, math.sin(half_theta)],
            device=self.device)

        rotated_orientations = math_utils.quat_mul(
            camera_obj_quat, left_90_quat.expand(len(env_ids), -1))

        self._occlusion_camera.set_world_poses(
            positions=camera_obj_pos,
            orientations=rotated_orientations,
            env_ids=env_ids.tolist(),
            convention="world")

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

        self._rgb_tiled_camera.update(self.sim.cfg.dt)

        rgb_data = self._rgb_tiled_camera.data.output["rgb"]
        rgb_data = rgb_data.permute(0, 3, 1, 2)[:, :3, :, :]
        observations = {"policy": rgb_data.clone()}
        self.obs = rgb_data

        return observations

    def _get_rewards(self) -> torch.Tensor:
        distance = (self._camera_obj.data.root_pos_w[:, :2] -
                    self._agent.data.root_pos_w[:, :2])**2
        distance = -1 * torch.sqrt(distance.sum(dim=1))
        return distance

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance = (self._camera_obj.data.root_pos_w[:, :2] -
                    self._agent.data.root_pos_w[:, :2])**2
        distance = torch.sqrt(distance.sum(dim=1))
        goal_reached = distance <= 0.2
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminated = terminated | goal_reached
        time_outs = (self.episode_length_buf >= self.max_episode_length)
        return terminated, time_outs

    def _validate_env_state(self, env_id: torch.Tensor, folder_idx: int, min_viewpoints: int) -> tuple[bool, str]:
        env_idx = env_id.item()

        poses = self.valid_viewpoint_poses[env_idx] if self.valid_viewpoint_poses else None
        num_poses = len(poses) if poses is not None else 0

        if num_poses < min_viewpoints:
            return False, f"insufficient viewpoints: {num_poses}/{min_viewpoints}"

        label = self.env_visibility_reasons.get(folder_idx, "unknown")

        if label in ["in_view", "occluded"]:
            cam_pos = self._camera_obj.data.root_pos_w[env_id]
            goal_pos = self._goal.data.root_pos_w[env_id]

            is_occluded = self._check_occlusion(cam_pos, goal_pos, env_id)
            expected_occluded = (label == "occluded")

            if is_occluded != expected_occluded:
                exp_str = "occluded" if expected_occluded else "visible"
                got_str = "occluded" if is_occluded else "visible"
                return False, f"occlusion mismatch: expected {exp_str}, got {got_str}"

        return True, ""

    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)
        return obs, rewards, terminated, truncated, info

    def render(self):
        frame = self.obs[0].permute(1, 2, 0).cpu().numpy()
        action = self.old_actions[0].cpu().numpy()

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
        cv2.putText(frame, action_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(f"frame_{self.step_count}_{action_str}.png", frame)
        return frame

    # =========================================================================
    # OCCLUSION / VISIBILITY
    # =========================================================================

    def _check_occlusion(self,
                         camera_pos: torch.Tensor,
                         goal_pos: torch.Tensor,
                         env_id: int | torch.Tensor,
                         camera=None) -> bool:
        if camera is None:
            camera = self._occlusion_camera

        target_pos = self._camera_obj.data.root_pos_w[env_id]
        sensor_pos = camera.data.pos_w[env_id]

        if torch.norm(target_pos - sensor_pos).item() > 0.01:
            angle = math.pi / 4
            rot_correction = torch.tensor(
                [math.cos(angle), 0.0, 0.0, math.sin(angle)],
                device=target_pos.device)

            body_quat = self._camera_obj.data.root_quat_w[env_id]
            new_orient = math_utils.quat_mul(
                body_quat.unsqueeze(0),
                rot_correction.unsqueeze(0)).squeeze(0)

            idx_list = [env_id.item()] if hasattr(env_id, "item") else [env_id]

            camera.set_world_poses(
                positions=target_pos.unsqueeze(0),
                orientations=new_orient.unsqueeze(0),
                env_ids=idx_list,
                convention="world")
            self.sim.step()

        sem_img = camera.data.output["semantic_segmentation"][env_id]
        r, g, b = sem_img[..., 0], sem_img[..., 1], sem_img[..., 2]

        red_mask = (r >= 0.95) & (g <= 0.05) & (b <= 0.05)
        visible_pixels = red_mask.sum().item()

        return visible_pixels < self.goal_pixel_threshold_occlusion

    # =========================================================================
    # SAVE / LOAD UTILITIES
    # =========================================================================

    def _save_visibility_labels(self):
        os.makedirs(os.path.dirname(self.visibility_labels_json_path), exist_ok=True)

        env_details = {
            str(idx): {
                "label": label,
                "reason": self.env_visibility_reasons.get(idx, "unknown")
            }
            for idx, label in self.env_visibility_labels.items()
        }

        reason_counts = {"in_view": 0, "occluded": 0, "outside_fov": 0, "unknown": 0}
        for reason in self.env_visibility_reasons.values():
            if reason in reason_counts:
                reason_counts[reason] += 1
            else:
                reason_counts["unknown"] += 1

        labels_data = {
            "environments": env_details,
            "statistics": {
                "total_environments": len(self.env_visibility_labels),
                "yes_count": sum(1 for v in self.env_visibility_labels.values() if v == "Yes"),
                "no_count": sum(1 for v in self.env_visibility_labels.values() if v == "No"),
                "by_reason": reason_counts,
                "next_env_folder_idx": self.next_env_folder_idx
            }
        }

        with open(self.visibility_labels_json_path, 'w') as f:
            json.dump(labels_data, f, indent=2)

    def _save_depth_labels(self):
        """Saves depth proximity labels to a JSON file (updates existing file)."""
        os.makedirs(os.path.dirname(self.depth_labels_json_path), exist_ok=True)

        existing_data = {}

        if os.path.exists(self.depth_labels_json_path):
            with open(self.depth_labels_json_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    pass

        for env_key, images in self.depth_labels_data.items():
            if env_key not in existing_data:
                existing_data[env_key] = {}
            existing_data[env_key].update(images)

        with open(self.depth_labels_json_path, 'w') as f:
            json.dump(existing_data, f, indent=2)

    def _check_target_in_img(self,
                             file_name: str,
                             cam_pov: np.ndarray,
                             return_red_count: bool = False) -> bool | tuple[bool, int]:
        cv2.imwrite(file_name, cv2.cvtColor(cam_pov, cv2.COLOR_RGB2BGR))

        r, g, b = cam_pov[..., 0], cam_pov[..., 1], cam_pov[..., 2]
        red_mask = (r >= 242) & (g <= 13) & (b <= 13)
        red_count = red_mask.sum().item()

        target_visible = red_count >= self.goal_pixel_threshold_occlusion
        is_detected = target_visible

        if return_red_count:
            return is_detected, red_count
        return is_detected

    def _generate_env_dedup_hash(self, active_obj_ids, obj_positions, obj_quats,
                                  obj_scales, obj_textures, cam_pos, cam_quat,
                                  goal_pos, goal_quat, viewpoint_radius) -> str:
        """Generates a hash for deduplicating environments."""
        content = json.dumps({
            "active_obj_ids": active_obj_ids,
            "obj_positions": [[round(v, 3) for v in p] for p in obj_positions],
            "obj_quats": [[round(v, 3) for v in q] for q in obj_quats],
            "obj_scales": [[round(v, 3) for v in s] for s in obj_scales],
            "obj_textures": obj_textures,
            "cam_pos": [round(v, 3) for v in cam_pos],
            "cam_quat": [round(v, 3) for v in cam_quat],
            "goal_pos": [round(v, 3) for v in goal_pos],
            "goal_quat": [round(v, 3) for v in goal_quat],
            "viewpoint_radius": round(viewpoint_radius, 3)
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _save_env_config_to_json(self, env_id: int, folder_idx: int):
        """Save environment configuration to JSON (v1.0 — world coords)."""
        import json

        label = self.env_visibility_labels.get(folder_idx, "UNKNOWN")
        reason = self.env_visibility_reasons.get(folder_idx, "unknown")

        goal_pos = self._goal.data.root_pos_w[env_id].cpu().numpy().tolist()
        goal_quat = self._goal.data.root_quat_w[env_id].cpu().numpy().tolist()
        goal_spawn_cfg = {
            "radius": float(self.cfg.goal_ball.spawn.radius),
            "rigid_props": {"disable_gravity": bool(self.cfg.goal_ball.spawn.rigid_props.disable_gravity)},
            "mass_props": {"mass": float(self.cfg.goal_ball.spawn.mass_props.mass)},
            "visual_material": {"diffuse_color": list(self.cfg.goal_ball.spawn.visual_material.diffuse_color)}
        }

        camera_pos = self._camera_obj.data.root_pos_w[env_id].cpu().numpy().tolist()
        camera_quat = self._camera_obj.data.root_quat_w[env_id].cpu().numpy().tolist()
        camera_spawn_cfg = {
            "rigid_props": {"disable_gravity": bool(self.cfg.camera_obj.spawn.rigid_props.disable_gravity)},
            "mass_props": {"mass": float(self.cfg.camera_obj.spawn.mass_props.mass)},
            "visual_material": {"diffuse_color": list(self.cfg.camera_obj.spawn.visual_material.diffuse_color)}
        }

        agent_pos = self._agent.data.root_pos_w[env_id].cpu().numpy().tolist()
        agent_quat = self._agent.data.root_quat_w[env_id].cpu().numpy().tolist()
        agent_spawn_cfg = {
            "size": list(self.cfg.agent.spawn.size),
            "rigid_props": {"disable_gravity": bool(self.cfg.agent.spawn.rigid_props.disable_gravity)},
            "mass_props": {"mass": float(self.cfg.agent.spawn.mass_props.mass)},
            "visual_material": {"diffuse_color": list(self.cfg.agent.spawn.visual_material.diffuse_color)}
        }

        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        active_indices = self.active_vpt_indices[env_id_item]
        active_indices_list = active_indices.cpu().numpy().tolist()

        vpt_positions = self._vpt_objects.data.object_pos_w[env_id].cpu().numpy().tolist()
        vpt_orientations = self._vpt_objects.data.object_quat_w[env_id].cpu().numpy().tolist()

        valid_viewpoints = []
        if (self.valid_viewpoint_poses is not None
                and env_id_item < len(self.valid_viewpoint_poses)
                and self.valid_viewpoint_poses[env_id_item] is not None):
            valid_viewpoints = self.valid_viewpoint_poses[env_id_item].cpu().numpy().tolist()

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
            "goal_ball": {"position": goal_pos, "orientation": goal_quat, "spawn_cfg": goal_spawn_cfg},
            "camera_object": {"position": camera_pos, "orientation": camera_quat, "spawn_cfg": camera_spawn_cfg},
            "agent": {"position": agent_pos, "orientation": agent_quat, "spawn_cfg": agent_spawn_cfg},
            "vpt_objects": {
                "total_count": self.num_objs,
                "active_count": self.active_vpt_objs,
                "active_indices": active_indices_list,
                "objects": []
            },
            "valid_viewpoints": {"count": len(valid_viewpoints), "positions": valid_viewpoints},
            "collected_viewpoints": {
                "count": len(self.selected_viewpoints_for_collection[env_id_item])
                    if self.selected_viewpoints_for_collection[env_id_item] is not None else 0,
                "positions": self.selected_viewpoints_for_collection[env_id_item].cpu().numpy().tolist()
                    if self.selected_viewpoints_for_collection[env_id_item] is not None else []
            }
        }

        for local_idx, obj_idx in enumerate(active_indices):
            obj_idx_item = obj_idx.item() if torch.is_tensor(obj_idx) else obj_idx
            vpt_spawn_cfg = self.cfg.vpt_objects.rigid_objects[
                list(self.cfg.vpt_objects.rigid_objects.keys())[obj_idx_item]].spawn

            rigid_props = {}
            if hasattr(vpt_spawn_cfg, 'rigid_props'):
                rigid_props['disable_gravity'] = bool(vpt_spawn_cfg.rigid_props.disable_gravity)
            mass_props = {}
            if hasattr(vpt_spawn_cfg, 'mass_props'):
                mass_props['mass'] = float(vpt_spawn_cfg.mass_props.mass)
            visual_material = {}
            if hasattr(vpt_spawn_cfg, 'visual_material'):
                visual_material['diffuse_color'] = list(vpt_spawn_cfg.visual_material.diffuse_color)
            size_info = {}
            if hasattr(vpt_spawn_cfg, 'size'):
                size_info['size'] = list(vpt_spawn_cfg.size)
            elif hasattr(vpt_spawn_cfg, 'radius'):
                size_info['radius'] = float(vpt_spawn_cfg.radius)
            elif hasattr(vpt_spawn_cfg, 'height') and hasattr(vpt_spawn_cfg, 'radius'):
                size_info['height'] = float(vpt_spawn_cfg.height)
                size_info['radius'] = float(vpt_spawn_cfg.radius)

            config["vpt_objects"]["objects"].append({
                "index": obj_idx_item,
                "position": vpt_positions[obj_idx_item],
                "orientation": vpt_orientations[obj_idx_item],
                "spawn_cfg": {**size_info, "rigid_props": rigid_props,
                              "mass_props": mass_props, "visual_material": visual_material}
            })

        config_dir = f"{self.base_path}/configs"
        os.makedirs(config_dir, exist_ok=True)
        config_filepath = f"{config_dir}/env_{folder_idx}_config.json"
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)

    def _save_env_config_to_json_new(self, env_id: int, folder_idx: int):
        """Save environment configuration to JSON (v2.0 — local coords, dedup hash)."""
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id

        # Fetch and clear cache for this env
        env_cache = getattr(self, "env_metadata_cache", {}).pop(env_id_item, {})
        textures_cache = env_cache.get("textures", {})
        vpt_textures_cache = textures_cache.get("vpt_objects", {})

        label = self.env_visibility_labels.get(folder_idx, "UNKNOWN")
        reason = self.env_visibility_reasons.get(folder_idx, "unknown")

        env_origin = self.scene.env_origins[env_id_item].cpu().numpy()

        def to_local_pos(world_pos_tensor):
            return (world_pos_tensor.cpu().numpy() - env_origin).tolist()

        goal_pos_local = to_local_pos(self._goal.data.root_pos_w[env_id_item])
        goal_quat = self._goal.data.root_quat_w[env_id_item].cpu().numpy().tolist()

        camera_pos_local = to_local_pos(self._camera_obj.data.root_pos_w[env_id_item])
        camera_quat = self._camera_obj.data.root_quat_w[env_id_item].cpu().numpy().tolist()

        agent_pos_local = to_local_pos(self._agent.data.root_pos_w[env_id_item])
        agent_quat = self._agent.data.root_quat_w[env_id_item].cpu().numpy().tolist()

        viewpoint_radius = env_cache.get("viewpoint_radius", 0.0)

        # Viewpoints in local coords
        valid_viewpoints_local = []
        if (self.valid_viewpoint_poses is not None
                and env_id_item < len(self.valid_viewpoint_poses)
                and self.valid_viewpoint_poses[env_id_item] is not None):
            valid_vp_world = self.valid_viewpoint_poses[env_id_item].cpu().numpy()
            if len(valid_vp_world) > 0:
                valid_viewpoints_local = (valid_vp_world - env_origin).tolist()

        collected_viewpoints_local = []
        if self.selected_viewpoints_for_collection[env_id_item] is not None:
            collected_vp_world = self.selected_viewpoints_for_collection[env_id_item].cpu().numpy()
            if len(collected_vp_world) > 0:
                collected_viewpoints_local = (collected_vp_world - env_origin).tolist()

        # VPT objects
        active_indices = self.active_vpt_indices[env_id_item]
        active_indices_list = active_indices.cpu().numpy().tolist()

        vpt_objects_data = []
        hash_obj_positions = []
        hash_obj_quats = []
        hash_obj_scales = []
        hash_obj_textures = []

        for local_idx, obj_idx in enumerate(active_indices):
            obj_idx_item = obj_idx.item() if torch.is_tensor(obj_idx) else obj_idx

            world_pos = self._vpt_objects.data.object_pos_w[env_id_item, local_idx]
            pos_local = to_local_pos(world_pos)
            quat = self._vpt_objects.data.object_quat_w[env_id_item, local_idx].cpu().numpy().tolist()
            bbox_dims = self.all_vpt_dims[env_id_item, local_idx].cpu().numpy().tolist()

            texture_path = vpt_textures_cache.get(obj_idx_item, "unknown")

            vpt_objects_data.append({
                "obj_index": obj_idx_item,
                "pos_local": pos_local,
                "quat_wxyz": quat,
                "dimensions_bbox": bbox_dims,
                "aesthetics": {"material_path": texture_path}
            })

            hash_obj_positions.append(pos_local)
            hash_obj_quats.append(quat)
            hash_obj_scales.append(bbox_dims)
            hash_obj_textures.append(texture_path)

        dedup_hash = self._generate_env_dedup_hash(
            active_obj_ids=active_indices_list,
            obj_positions=hash_obj_positions,
            obj_quats=hash_obj_quats,
            obj_scales=hash_obj_scales,
            obj_textures=hash_obj_textures,
            cam_pos=camera_pos_local,
            cam_quat=camera_quat,
            goal_pos=goal_pos_local,
            goal_quat=goal_quat,
            viewpoint_radius=viewpoint_radius
        )

        lighting_data = env_cache.get("lighting", [])
        floor_texture = textures_cache.get("floor", "unknown")

        config = {
            "metadata": {
                "env_id": env_id_item,
                "folder_idx": folder_idx,
                "visibility_label": label,
                "visibility_reason": reason,
                "dedup_hash": dedup_hash,
                "cfg_version": "2.0"
            },
            "environment_settings": {
                "boundary_limits": list(self.cfg.boundary_limits),
                "agent_height": float(self.cfg.agent_height),
                "agent_camera_pitch": float(self.cfg.agent_camera_pitch),
                "viewpoint_radius": viewpoint_radius
            },
            "entities": {
                "agent": {"pos_local": agent_pos_local, "quat_wxyz": agent_quat},
                "goal": {"pos_local": goal_pos_local, "quat_wxyz": goal_quat},
                "camera": {"pos_local": camera_pos_local, "quat_wxyz": camera_quat}
            },
            "vpt_objects": {
                "total_count": self.num_objs,
                "active_count": self.active_vpt_objs,
                "active_indices": active_indices_list,
                "objects": vpt_objects_data
            },
            "environment_aesthetics": {"floor_material_path": floor_texture},
            "lighting": lighting_data,
            "viewpoints": {
                "valid_count": len(valid_viewpoints_local),
                "valid_positions_local": valid_viewpoints_local,
                "collected_count": len(collected_viewpoints_local),
                "collected_positions_local": collected_viewpoints_local
            }
        }

        config_dir = os.path.join(self.base_path, "configs")
        os.makedirs(config_dir, exist_ok=True)
        config_filepath = os.path.join(config_dir, f"env_{folder_idx}_config.json")
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)

    # =========================================================================
    # VPT OBJECT HELPERS
    # =========================================================================

    def _get_batch_active_indices(self, env_ids: int | list | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _select_active_vpt_indices(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.view(-1) if torch.is_tensor(env_ids) else torch.tensor(env_ids)
        for env_id in env_ids:
            active = torch.randperm(self.num_objs, device=self.device)[:self.active_vpt_objs]
            self.active_vpt_indices[env_id.item()] = active

    def _store_inactive_vpt_objects(self, env_ids: torch.Tensor, vpt_obj_default_state: torch.Tensor) -> torch.Tensor:
        ids, active_indices = self._get_batch_active_indices(env_ids)
        num_batch = len(ids)

        inactive_mask = torch.ones((num_batch, self.num_objs), dtype=torch.bool, device=self.device)
        inactive_mask.scatter_(1, active_indices, False)

        local_row_indices = torch.arange(num_batch, device=self.device).view(-1, 1).expand(-1, self.num_objs)
        col_indices = torch.arange(self.num_objs, device=self.device).expand(num_batch, -1)

        target_rows = local_row_indices[inactive_mask]
        target_cols = col_indices[inactive_mask]

        if len(target_rows) > 0:
            vpt_obj_default_state[target_rows, target_cols, 0] = self.storage_position[0]
            vpt_obj_default_state[target_rows, target_cols, 1] = self.storage_position[1]
            vpt_obj_default_state[target_rows, target_cols, 2] = self.storage_position[2]
            vpt_obj_default_state[target_rows, target_cols, 7:13] = 0.0

        return vpt_obj_default_state

    def _get_active_vpt_dims(self, env_ids: int | torch.Tensor) -> torch.Tensor:
        ids, batch_indices = self._get_batch_active_indices(env_ids)
        env_ids_expanded = ids.view(-1, 1).expand_as(batch_indices)
        return self.all_vpt_dims[env_ids_expanded, batch_indices, :]

    def _get_active_vpt_positions(self,
                                  env_ids: int | torch.Tensor,
                                  base_pivoted: bool = False,
                                  return_full_pose: bool = False) -> torch.Tensor:
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

    # =========================================================================
    # VIEWPOINT SELECTION — 50/50 DEPTH SPLIT (from v17)
    # =========================================================================

    def _select_viewpoints_for_collection(self, env_id: int) -> bool:
        """Select viewpoints ensuring a strict 50/50 split based on distance to camera vs goal."""
        if (self.valid_viewpoint_poses is None
                or env_id >= len(self.valid_viewpoint_poses)
                or self.valid_viewpoint_poses[env_id] is None
                or len(self.valid_viewpoint_poses[env_id]) < self.images_per_env):
            return False

        all_viewpoints = self.valid_viewpoint_poses[env_id]

        cam_pos_2d = self._camera_obj.data.root_pos_w[env_id, :2]
        goal_pos_2d = self._goal.data.root_pos_w[env_id, :2]

        dist_cam = torch.norm(all_viewpoints[:, :2] - cam_pos_2d, dim=1)
        dist_goal = torch.norm(all_viewpoints[:, :2] - goal_pos_2d, dim=1)

        pool_a = all_viewpoints[dist_cam < dist_goal]   # closer to cam -> label 1
        pool_b = all_viewpoints[dist_goal < dist_cam]   # closer to goal -> label 0

        half_req = self.images_per_env // 2

        def select_from_pool(pool, req_count):
            if len(pool) == 0:
                return []
            selected = [pool[0]]
            for pt_idx in range(1, len(pool)):
                candidate = pool[pt_idx]
                distances = torch.norm(
                    torch.stack(selected)[:, :2] - candidate[:2].unsqueeze(0), dim=1)
                if torch.all(distances >= self.min_viewpoint_distance):
                    selected.append(candidate)
                if len(selected) == req_count:
                    break
            return selected

        selected_a = select_from_pool(pool_a, half_req)
        selected_b = select_from_pool(pool_b, half_req)

        if len(selected_a) == half_req and len(selected_b) == half_req:
            self.selected_viewpoints_for_collection[env_id] = torch.stack(selected_a + selected_b)
            return True
        else:
            if self.verbose >= 2:
                print(f"  ⚠️  Slot {env_id}: 50/50 split failed "
                      f"(cam-proximal: {len(selected_a)}, goal-proximal: {len(selected_b)}, need {half_req} each)")
            return False

    # =========================================================================
    # RESET PIPELINE
    # =========================================================================

    def _reset_idx(self, env_ids: Sequence[int] | None, rl_reset: bool = True) -> None:
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        active_slots_list = env_ids.tolist()

        self._ensure_slot_initialization()

        self._cache_base_dims()
        self._randomize_scene_props(active_slots_list)

        num_targets = max(1, int(0.25 * len(env_ids)))
        subset_idx = torch.randperm(len(env_ids), device=self.device)[:num_targets]
        self.envs_to_move_ball = env_ids[subset_idx]

        reset_folder_indices = [self.slot_folder_indices[i] for i in active_slots_list]
        reset_visibility_cats = [self.slot_visibility_categories[i] for i in active_slots_list]

        if self.verbose >= 1:
            print(f"🔄 Resetting {len(active_slots_list)} envs (RL: {rl_reset})")

        self._reset_idx_internal(
            env_ids,
            rl_reset=rl_reset,
            folder_indices=reset_folder_indices,
            visibility_categories=reset_visibility_cats)

        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.step_dt)

        if rl_reset:
            self._reset_called = True
            return

        valid_slots, exceeded_slots = self._validate_slots(active_slots_list)

        for slot_idx in valid_slots:
            folder_idx = self.slot_folder_indices[slot_idx]
            if self._select_viewpoints_for_collection(slot_idx):
                self._collect_images_for_slot(
                    torch.tensor([slot_idx], device=self.device),
                    folder_idx)
                self.completed_envs.add(self.slot_to_env_id[slot_idx])

        self._replenish_slots(valid_slots + exceeded_slots)
        self._reset_called = True

    def _ensure_slot_initialization(self) -> None:
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
                min_viewpoints)

            if is_valid:
                valid_slots.append(slot_idx)
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
                self.slot_folder_indices[slot_idx])

            if self.verbose >= 1:
                print(f"  🔄 Slot {slot_idx}: Replaced {old_env} -> {new_env}")

        self._save_visibility_labels()

    def _randomize_scene_props(self, env_ids: list[int]) -> None:
        if not env_ids:
            return

        vpt_paths = [
            f"/World/envs/env_{eid}/obs_{oid}"
            for eid in env_ids
            for oid in range(self.cfg.num_vpt_objs)
        ]

        if vpt_paths:
            self.randomize_shape_scale(prim_path_expr=vpt_paths, is_random=True)
            self.randomize_material(prim_paths=vpt_paths, material_type="vpt")

        floor_paths = [f"/World/envs/env_{i}/mat" for i in env_ids]
        if floor_paths:
            self.randomize_material(prim_paths=floor_paths, material_type="mat")

        self.randomize_shape_color(prim_path_expr=[
            "/World/envs/env_.*/bottom_wall",
            "/World/envs/env_.*/right_wall",
            "/World/envs/env_.*/left_wall",
            "/World/envs/env_.*/top_wall"
        ])

        light_paths = [f"/World/envs/env_{i}/Light_A" for i in env_ids]
        if light_paths:
            self.randomize_spherical_lights(prim_paths=light_paths)

    # =========================================================================
    # SPAWN LOOPS
    # =========================================================================

    def initial_spawn_loop_old(self, env_ids, envs_need_spawn_retry, safe_range, states,
                                allow_clipping=False, device=None):
        """Legacy Shapely-based spawn loop. Kept for reference."""
        import math
        import random
        import torch
        from shapely.geometry import Point, box
        from shapely import affinity

        if device is None:
            device = self._agent.device

        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        retry_mask = envs_need_spawn_retry.clone()
        retry_indices = torch.where(retry_mask)[0]
        global_retry_env_ids = env_ids[retry_indices]
        batch_size = retry_indices.numel()

        if batch_size == 0:
            return envs_need_spawn_retry, states

        safe_x_range = safe_range - 4.0
        safe_x_range_obstacles = float(safe_range - 3.0)
        env_origins = self.scene.env_origins[global_retry_env_ids]

        goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        camera_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        goal_perturb_offsets = sample_uniform(-2, 2, (batch_size, 2), device)

        goal_default_state[retry_indices, 0] = env_origins[:, 0] + goal_offsets[:, 0]
        goal_default_state[retry_indices, 1] = env_origins[:, 1] + goal_offsets[:, 1]
        goal_default_state[retry_indices, 2] = env_origins[:, 2]

        camera_obj_default_state[retry_indices, 0] = env_origins[:, 0] + camera_offsets[:, 0]
        camera_obj_default_state[retry_indices, 1] = env_origins[:, 1] + camera_offsets[:, 1]

        max_dist_retries = 20
        for _ in range(max_dist_retries):
            cam_pos_subset = camera_obj_default_state[retry_indices, :2]
            goal_pos_subset = goal_default_state[retry_indices, :2]
            dists = torch.norm(cam_pos_subset - goal_pos_subset, dim=1)
            bad_mask = (dists < 3.5) | (dists > 15.0)
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

        # (VPT scattering loop omitted for brevity — same as v18 original)
        envs_need_spawn_retry[retry_indices] = False

        return envs_need_spawn_retry, [
            goal_default_state, camera_obj_default_state, agent_default_state, vpt_obj_default_state
        ]

    def initial_spawn_loop(self, env_ids, envs_need_spawn_retry, safe_range, states,
                           allow_clipping=False, device=None):
        """Vectorized spawn loop (v18 base)."""
        import math
        import torch

        if device is None:
            device = self._agent.device

        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        retry_mask = envs_need_spawn_retry.clone()
        retry_indices = torch.where(retry_mask)[0]
        global_retry_env_ids = env_ids[retry_indices]
        batch_size = retry_indices.numel()

        if batch_size == 0:
            return envs_need_spawn_retry, states

        safe_x_range = safe_range - 4.0
        safe_x_range_obstacles = float(safe_range - 3.0)
        env_origins = self.scene.env_origins[global_retry_env_ids]

        # 1. Scatter VPT obstacles
        num_vpt_objs = vpt_obj_default_state.shape[1]
        rxs = (torch.rand((batch_size, num_vpt_objs), device=device) * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
        rys = (torch.rand((batch_size, num_vpt_objs), device=device) * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
        r_yaws = torch.rand((batch_size, num_vpt_objs), device=device) * 2 * math.pi

        vpt_obj_default_state[retry_indices, :, 0] = env_origins[:, 0].unsqueeze(1) + rxs
        vpt_obj_default_state[retry_indices, :, 1] = env_origins[:, 1].unsqueeze(1) + rys

        zero_t = torch.zeros_like(r_yaws)
        quats = quat_from_euler_xyz(zero_t, zero_t, r_yaws)
        vpt_obj_default_state[retry_indices, :, 3:7] = quats

        vpt_obj_default_state[retry_indices] = self._store_inactive_vpt_objects(
            global_retry_env_ids, vpt_obj_default_state[retry_indices])

        # 2. Place goal with OBB check
        goal_default_state[retry_indices] = self.place_object_safely(
            env_ids=global_retry_env_ids,
            object_state=goal_default_state[retry_indices],
            vpt_state=vpt_obj_default_state[retry_indices],
            safe_range=float(safe_x_range),
            object_type='goal')

        # 3. Place camera with OBB check
        camera_obj_default_state[retry_indices] = self.place_object_safely(
            env_ids=global_retry_env_ids,
            object_state=camera_obj_default_state[retry_indices],
            vpt_state=vpt_obj_default_state[retry_indices],
            safe_range=float(safe_x_range),
            object_type='cam_obj')

        # 4. Enforce camera-goal distance [3.5, 15.0]
        max_dist_retries = 20
        for _ in range(max_dist_retries):
            cam_pos_subset = camera_obj_default_state[retry_indices, :2]
            goal_pos_subset = goal_default_state[retry_indices, :2]
            dists = torch.norm(cam_pos_subset - goal_pos_subset, dim=1)
            bad_mask = (dists < 3.5) | (dists > 15.0)
            if not bad_mask.any():
                break
            bad_sub_indices = torch.where(bad_mask)[0]
            bad_local_indices = retry_indices[bad_sub_indices]
            bad_global_env_ids = global_retry_env_ids[bad_sub_indices]
            camera_obj_default_state[bad_local_indices] = self.place_object_safely(
                env_ids=bad_global_env_ids,
                object_state=camera_obj_default_state[bad_local_indices],
                vpt_state=vpt_obj_default_state[bad_local_indices],
                safe_range=float(safe_x_range),
                object_type='cam_obj')

        # 5. Perturb goal
        goal_perturb_offsets = sample_uniform(-2, 2, (batch_size, 2), device)
        goal_default_state[retry_indices, 0] += goal_perturb_offsets[:, 0]
        goal_default_state[retry_indices, 1] += goal_perturb_offsets[:, 1]

        # 6. Agent random placement
        agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        agent_default_state[retry_indices, 0] = env_origins[:, 0] + agent_offsets[:, 0]
        agent_default_state[retry_indices, 1] = env_origins[:, 1] + agent_offsets[:, 1]

        # 7. Orient camera toward goal with yaw jitter
        direction_to_goal = (goal_default_state[retry_indices, :2] -
                             camera_obj_default_state[retry_indices, :2])
        exact_yaw = (torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
                     - math.radians(90))
        half_fov_rad = math.radians(30 * 0.8)
        yaw_jitter = sample_uniform(-half_fov_rad, half_fov_rad, (batch_size,), device)
        yaw = exact_yaw + yaw_jitter

        roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
        zero = torch.zeros_like(yaw)
        camera_obj_default_state[retry_indices, 3:7] = quat_from_euler_xyz(roll, zero, yaw)

        envs_need_spawn_retry[retry_indices] = False

        return envs_need_spawn_retry, [
            goal_default_state, camera_obj_default_state,
            agent_default_state, vpt_obj_default_state
        ]

    def moving_ball_loop(self, env_ids, moved_vpt_for_ball, move_ball_indices,
                         states, safe_range, device=None):
        if device is None:
            device = self._agent.device

        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        safe_x_range_obstacles = safe_range - 3.5

        if len(move_ball_indices) == 0:
            return moved_vpt_for_ball, states

        target_env_ids = env_ids[move_ball_indices]
        batch_size = len(target_env_ids)

        if isinstance(self.active_vpt_indices, list):
            if len(self.active_vpt_indices) > 0 and isinstance(self.active_vpt_indices[0], torch.Tensor):
                active_indices_tensor = torch.stack(self.active_vpt_indices).to(device)
            else:
                active_indices_tensor = torch.tensor(self.active_vpt_indices, device=device, dtype=torch.long)
        else:
            active_indices_tensor = self.active_vpt_indices

        batch_active_indices = active_indices_tensor[target_env_ids]
        batch_dims = self._get_active_vpt_dims(target_env_ids)
        batch_heights = batch_dims[:, :, 2]

        height_mask = batch_heights < 0.75
        shape_mask = self.valid_shape_mask[batch_active_indices]
        candidate_mask = height_mask & shape_mask
        has_candidates_mask = candidate_mask.any(dim=1)

        if not has_candidates_mask.all():
            target_env_ids = target_env_ids[has_candidates_mask]
            move_ball_indices = move_ball_indices[has_candidates_mask]
            batch_active_indices = batch_active_indices[has_candidates_mask]
            batch_heights = batch_heights[has_candidates_mask]
            candidate_mask = candidate_mask[has_candidates_mask]
            batch_size = len(target_env_ids)

        if batch_size == 0:
            return moved_vpt_for_ball, states

        weights = candidate_mask.float() + 1e-6
        selected_local_indices = torch.multinomial(weights, 1).squeeze(-1)
        selected_global_indices = torch.gather(batch_active_indices, 1, selected_local_indices.unsqueeze(1)).squeeze(-1)
        selected_heights = torch.gather(batch_heights, 1, selected_local_indices.unsqueeze(1)).squeeze(-1)

        target_goal_pos = goal_default_state[move_ball_indices, :3]
        target_env_origins = self.scene.env_origins[target_env_ids]
        new_goal_z = target_env_origins[:, 2] + selected_heights + self.goal_radius

        vpt_obj_default_state[move_ball_indices, selected_global_indices, 0] = target_goal_pos[:, 0]
        vpt_obj_default_state[move_ball_indices, selected_global_indices, 1] = target_goal_pos[:, 1]
        goal_default_state[move_ball_indices, 2] = new_goal_z

        for i, env_idx in enumerate(move_ball_indices.cpu().numpy()):
            obj_id = selected_global_indices.cpu().numpy()[i]
            moved_vpt_for_ball[env_idx] = obj_id

        all_active_pos = vpt_obj_default_state[move_ball_indices.unsqueeze(1), batch_active_indices, :2]
        dists = torch.norm(all_active_pos - target_goal_pos[:, :2].unsqueeze(1), dim=2)
        selection_one_hot = torch.nn.functional.one_hot(selected_local_indices, num_classes=self.active_vpt_objs).bool()
        conflict_mask = (dists < 1.5) & (~selection_one_hot)

        if conflict_mask.any():
            num_conflicts = conflict_mask.sum()
            new_x = sample_uniform(-safe_x_range_obstacles, safe_x_range_obstacles, (num_conflicts,), device)
            new_y = sample_uniform(-safe_x_range_obstacles, safe_x_range_obstacles, (num_conflicts,), device)
            expanded_env_indices = move_ball_indices.unsqueeze(1).expand_as(conflict_mask)
            conflict_env_idxs = expanded_env_indices[conflict_mask]
            conflict_obj_idxs = batch_active_indices[conflict_mask]
            conflict_origins = self.scene.env_origins[env_ids[conflict_env_idxs]]
            vpt_obj_default_state[conflict_env_idxs, conflict_obj_idxs, 0] = conflict_origins[:, 0] + new_x
            vpt_obj_default_state[conflict_env_idxs, conflict_obj_idxs, 1] = conflict_origins[:, 1] + new_y

        return moved_vpt_for_ball, [
            goal_default_state, camera_obj_default_state, agent_default_state, vpt_obj_default_state
        ]

    def move_vpt_objects(self, env_ids, valid_indices, visibility_categories,
                         moved_vpt_for_ball, states, in_view_displaced=None,
                         outside_fov_displaced=None, device=None):
        if device is None:
            device = self._agent.device

        goal_state = states[0]
        camera_state = states[1]
        vpt_state = states[3]

        indices_to_move = set()
        if in_view_displaced is not None:
            indices_to_move.update(in_view_displaced.tolist())
        if outside_fov_displaced is not None:
            indices_to_move.update(outside_fov_displaced.tolist())

        for env_idx in valid_indices:
            env_idx_int = env_idx.item()
            category = visibility_categories[env_idx_int]

            should_move = (category == "occluded") or (env_idx_int in indices_to_move)
            if not should_move:
                continue

            global_env_id = env_ids[env_idx_int].item()
            active_indices = self.active_vpt_indices[global_env_id]
            ball_mount_idx = moved_vpt_for_ball[env_idx_int]

            candidates = [
                idx.item() for idx in active_indices
                if ball_mount_idx is None or idx.item() != ball_mount_idx
            ]

            if not candidates:
                continue

            target_obj_idx = random.choice(candidates)

            cam_pos = camera_state[env_idx, :3]
            goal_pos = goal_state[env_idx, :3]
            vec_cam_to_goal = goal_pos[:2] - cam_pos[:2]
            dist = torch.norm(vec_cam_to_goal)

            if dist > 1e-6:
                t_min, t_max = (0.3, 0.7) if category == "in_view" else (0.2, 0.8)
                t = random.uniform(t_min, t_max)
                jitter = (torch.rand(2, device=device) * 0.8) - 0.4
                new_pos = cam_pos[:2] + (vec_cam_to_goal * t) + jitter
                vpt_state[env_idx, target_obj_idx, 0] = new_pos[0]
                vpt_state[env_idx, target_obj_idx, 1] = new_pos[1]

        return states

    def outside_fov_camera_movement(self, valid_env_ids, valid_indices,
                                    visibility_categories, states, device):
        if device is None:
            device = self._agent.device

        goal_default_state = states[0]
        camera_obj_default_state = states[1]

        current_categories = [visibility_categories[i] for i in valid_indices.cpu().tolist()]
        outside_fov_mask = torch.tensor(
            [c == "outside_fov" for c in current_categories], device=device, dtype=torch.bool)

        if outside_fov_mask.any():
            outside_fov_global_idxs = valid_indices[outside_fov_mask]
            camera_pos_batch = camera_obj_default_state[outside_fov_global_idxs, :3]
            goal_pos_batch = goal_default_state[outside_fov_global_idxs, :3]

            direction_to_goal = (goal_pos_batch[:, :2] - camera_pos_batch[:, :2]) + 1e-6
            yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0]) - math.radians(90)

            yaw_offset_magnitude = sample_uniform(
                math.radians(60), math.pi, (len(outside_fov_global_idxs),), device=device)
            signs = torch.randint(0, 2, (len(outside_fov_global_idxs),), device=device).float() * 2 - 1
            yaw_away = yaw + (yaw_offset_magnitude * signs)
            roll = torch.full((len(outside_fov_global_idxs),), -math.radians(self.agent_camera_pitch), device=device)
            zero = torch.zeros_like(roll)
            quaternion_away = quat_from_euler_xyz(roll, zero, yaw_away)
            camera_obj_default_state[outside_fov_global_idxs, 3:7] = quaternion_away

        subset_quats = camera_obj_default_state[valid_indices, 3:7]
        subset_quats = torch.nn.functional.normalize(subset_quats, p=2, dim=-1)
        nan_mask = torch.isnan(subset_quats).any(dim=1)
        if nan_mask.any():
            print(f"⚠️ FATAL: Found {nan_mask.sum()} NaN quaternions! Resetting to identity.")
            subset_quats[nan_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        camera_obj_default_state[valid_indices, 3:7] = subset_quats

        self._camera_obj.write_root_pose_to_sim(camera_obj_default_state[valid_indices, :7], valid_env_ids)

        camera_positions = camera_obj_default_state[valid_indices, :3]
        camera_orientations = camera_obj_default_state[valid_indices, 3:7]

        theta_left = math.pi / 2
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)], device=device)
        rotated_orientations = math_utils.quat_mul(
            camera_orientations, left_90_quat.unsqueeze(0).expand(len(valid_env_ids), -1))
        rotated_orientations = torch.nn.functional.normalize(rotated_orientations, p=2, dim=-1)

        self._occlusion_camera.set_world_poses(
            positions=camera_positions,
            orientations=rotated_orientations,
            env_ids=valid_env_ids.tolist(),
            convention="world")

        for _ in range(1):
            self.sim.step()
            self._occlusion_camera.update(self.sim.cfg.dt)

    def check_z_bounds(self, env_ids, valid_indices, states, envs_need_spawn_retry, tolerance=5e-2):
        updated_retry_mask = envs_need_spawn_retry.clone()
        goal_pos, camera_pos, agent_pos, vpt_pos = states

        for local_idx, env_idx in enumerate(valid_indices):
            env_id_val = env_ids[env_idx].item()
            failure_reasons = []

            goal_z = goal_pos[local_idx, 2].item()
            if not (-tolerance <= goal_z <= 1.0 + tolerance):
                failure_reasons.append(f"Goal Z: {goal_z:.6f}")

            cam_z = camera_pos[local_idx, 2].item()
            if not (0.0 <= cam_z <= 1.0):
                failure_reasons.append(f"Camera Z: {cam_z:.4f}")

            agent_z = agent_pos[local_idx, 2].item()
            if not (0.0 <= agent_z <= 1.0):
                failure_reasons.append(f"Agent Z: {agent_z:.4f}")

            active_indices = self.active_vpt_indices[env_id_val]
            raw_z = vpt_pos[local_idx, :, 2]
            offsets = self.vpt_z_offset_ratios[env_id_val, active_indices]
            adjusted_z = raw_z * offsets

            valid_obj_mask = (adjusted_z >= -tolerance) & (adjusted_z <= 0.1 + tolerance)
            if not torch.all(valid_obj_mask):
                failed_indices = torch.where(~valid_obj_mask)[0]
                for idx in failed_indices:
                    global_id = active_indices[idx].item()
                    bad_z = adjusted_z[idx].item()
                    failure_reasons.append(f"VPT Obj {global_id} Z: {bad_z:.6f}")

            if failure_reasons:
                print(f"⚠️ Env {env_id_val} Z-Check Failed:")
                for reason in failure_reasons:
                    print(f"   - {reason}")
                updated_retry_mask[env_idx] = True

        return updated_retry_mask

    def occlusion_validation_check(self, final_valid_env_ids, valid_indices, visibility_categories,
                                   envs_need_spawn_retry, env_dict, states, device):
        if device is None:
            device = self._agent.device

        valid_env_ids = final_valid_env_ids
        goal_default_state = states[0]
        camera_obj_default_state = states[1]

        if torch.isnan(camera_obj_default_state).any():
            print("❌ FATAL: Camera state contains NaNs!")

        camera_positions = camera_obj_default_state[valid_indices, :3]
        occlusion_valid_mask = torch.ones(len(valid_indices), dtype=torch.bool, device=device)

        for local_idx, env_idx in enumerate(valid_indices):
            env_id = valid_env_ids[local_idx]
            env_id_item = env_id.item()
            visibility_category = visibility_categories[env_idx]
            camera_pos = camera_positions[local_idx]
            goal_pos = goal_default_state[env_idx, :3]

            if visibility_category in ["in_view", "occluded", "outside_fov"]:
                is_occluded = self._check_occlusion(camera_pos, goal_pos, env_id)
                expected_occluded = (visibility_category == "occluded" or visibility_category == "outside_fov")
                occlusion_valid = (is_occluded == expected_occluded)
                occlusion_valid_mask[local_idx] = occlusion_valid

                if not occlusion_valid:
                    envs_need_spawn_retry[env_idx] = True
                else:
                    if env_id_item not in env_dict:
                        env_dict[env_id_item] = 0.0

        return occlusion_valid_mask, envs_need_spawn_retry, env_dict, states

    def geometric_occlusion_check(self, env_ids, valid_indices, occlusion_valid_mask,
                                  envs_need_spawn_retry, device):
        FOV_DEG = 30.0
        MIN_GEOMETRIC_VALID_POINTS = 40
        NUM_ANGLES = 180

        geometric_valid_mask = occlusion_valid_mask.clone()
        passed_env_ids = env_ids[occlusion_valid_mask]

        if passed_env_ids.numel() == 0:
            return geometric_valid_mask, envs_need_spawn_retry

        num_envs = len(passed_env_ids)

        cam_pos = self._camera_obj.data.root_pos_w[passed_env_ids, :2]
        goal_pos = self._goal.data.root_pos_w[passed_env_ids, :2]
        dist_to_goal = torch.norm(cam_pos - goal_pos, dim=1)
        half_fov = torch.tensor(math.radians(FOV_DEG) / 2, device=device)
        radii = ((dist_to_goal / 2) / torch.tan(half_fov)) * 1.2
        radii = radii.unsqueeze(1)

        angles = torch.linspace(0, 2 * math.pi, NUM_ANGLES, device=device)
        angles_expanded = angles.unsqueeze(0).expand(num_envs, -1)

        circle_x = goal_pos[:, 0].unsqueeze(1) + radii * torch.cos(angles_expanded)
        circle_y = goal_pos[:, 1].unsqueeze(1) + radii * torch.sin(angles_expanded)

        total_points = num_envs * NUM_ANGLES
        flat_points = torch.stack([circle_x, circle_y], dim=2).reshape(total_points, 2)
        flat_env_ids = passed_env_ids.unsqueeze(1).expand(-1, NUM_ANGLES).reshape(total_points)

        is_valid_flat = self._is_point_valid_batch(points=flat_points, env_ids=flat_env_ids, check_agent_fov=False)
        valid_counts = is_valid_flat.reshape(num_envs, NUM_ANGLES).sum(dim=1)

        for i, global_env_id in enumerate(passed_env_ids):
            count = valid_counts[i].item()
            local_idx = (env_ids == global_env_id).nonzero(as_tuple=True)[0].item()
            batch_idx = valid_indices[local_idx]

            if count < MIN_GEOMETRIC_VALID_POINTS:
                envs_need_spawn_retry[batch_idx] = True
                geometric_valid_mask[local_idx] = False

        return geometric_valid_mask, envs_need_spawn_retry

    def camera_pov_validation(self, env_ids, valid_indices, geometric_valid_mask,
                              visibility_categories, envs_need_spawn_retry,
                              folder_indices, spawn_attempt):
        debug_folder = os.path.join(self.base_path, "debug_camera_pov")
        os.makedirs(debug_folder, exist_ok=True)

        for local_idx, env_idx in enumerate(valid_indices):
            if not geometric_valid_mask[local_idx]:
                continue

            env_id = env_ids[local_idx]
            env_id_val = env_id.item()
            category = visibility_categories[env_idx]
            folder_idx = folder_indices[env_idx]

            sem_img = self._occlusion_camera.data.output["semantic_segmentation"][env_id]
            cam_pov_img = sem_img[..., :3]

            if cam_pov_img.max() <= 1.0:
                cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(np.uint8)
            else:
                cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)

            debug_filename = os.path.join(
                debug_folder,
                f"env_{env_id_val}_folder_{folder_idx}_attempt_{spawn_attempt}.png")

            target_visible, red_count = self._check_target_in_img(
                file_name=debug_filename, cam_pov=cam_pov_np, return_red_count=True)

            expected_visible = (category == "in_view")
            is_valid = (target_visible == expected_visible)

            if is_valid:
                if self.verbose >= 2:
                    print(f"    ✅ Env {env_id_val}: Camera valid ({category}) | Red: {red_count}")
            else:
                envs_need_spawn_retry[env_idx] = True
                if self.verbose >= 1:
                    status = "visible" if target_visible else "NOT visible"
                    print(f"    ❌ Env {env_id_val}: Camera check FAILED")
                    print(f"       Expected: {category}, Got: {status} (Red: {red_count})")

        return envs_need_spawn_retry

    def _reset_idx_internal(self, env_ids, rl_reset=False, folder_indices=None,
                            visibility_categories=None) -> None:
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_envs = len(env_ids)
        if folder_indices is None:
            folder_indices = [self.next_env_folder_idx + i for i in range(num_envs)]

        if visibility_categories is None:
            raise RuntimeError("visibility_categories must be provided to _reset_idx_internal!")

        for env_id in env_ids:
            eid = env_id.item() if torch.is_tensor(env_id) else env_id
            self.used_viewpoint_indices[eid].clear()
            global_folder_idx = folder_indices[env_ids.tolist().index(eid)]
            if global_folder_idx not in self.env_visibility_labels:
                raise RuntimeError(f"Labels not set for folder {global_folder_idx}!")

        self._cache_valid_shapes()
        self._cache_base_dims()
        self._select_active_vpt_indices(env_ids)

        self.viewpoint_pose_counter[env_ids] = 0
        super()._reset_idx(env_ids)

        device = self._agent.device

        goal_state = self._goal.data.default_root_state[env_ids].clone()
        agent_state = self._agent.data.default_root_state[env_ids].clone()
        camera_state = self._camera_obj.data.default_root_state[env_ids].clone()
        vpt_state = self._vpt_objects.data.default_object_state[env_ids].clone()

        max_spawn_attempts = 20
        envs_need_spawn_retry = torch.ones(num_envs, dtype=torch.bool, device=device)

        if rl_reset:
            task_keys = ["writing_spawn_pose_time", "moving_ball_time",
                         "vpt_displacement_movement_time", "camera_posing_time", "occlusion_raycast_time"]
        else:
            task_keys = ["writing_spawn_pose_time", "moving_ball_time",
                         "vpt_displacement_movement_time", "camera_posing_time",
                         "occlusion_raycast_time", "geometric_check_time",
                         "camera_validation_time", "circle_validation_time"]

        timer = EnvTimer(num_envs=self.num_envs, slot_to_env_id=self.slot_to_env_id,
                         task_keys=task_keys, verbose=(self.verbose >= 1))

        valid_indices = torch.arange(num_envs, device=device)

        in_view_indices = [i for i in range(num_envs) if visibility_categories[i] == "in_view"]
        rand_iv = torch.randperm(len(in_view_indices))[:len(in_view_indices) // 2]
        in_view_displaced = (torch.tensor(in_view_indices, device=device)[rand_iv]
                             if in_view_indices else torch.tensor([], device=device))

        outside_fov_indices = [i for i in range(num_envs) if visibility_categories[i] == "outside_fov"]
        rand_of = torch.randperm(len(outside_fov_indices))[:len(outside_fov_indices) // 2]
        outside_fov_displaced = (torch.tensor(outside_fov_indices, device=device)[rand_of]
                                  if outside_fov_indices else torch.tensor([], device=device))

        # =================================================================
        # MAIN SPAWN LOOP
        # =================================================================
        for spawn_attempt in range(max_spawn_attempts):
            if not envs_need_spawn_retry.any():
                break

            retry_mask = envs_need_spawn_retry.clone()
            envs_need_spawn_retry, states = self.initial_spawn_loop(
                env_ids=env_ids,
                envs_need_spawn_retry=envs_need_spawn_retry,
                safe_range=self.center_to_boundary,
                states=[goal_state, camera_state, agent_state, vpt_state],
                allow_clipping=False,
                device=device)
            goal_state, camera_state, agent_state, vpt_state = states

            valid_mask = retry_mask & ~envs_need_spawn_retry
            if not valid_mask.any():
                continue

            valid_indices = torch.where(valid_mask)[0]
            valid_env_ids = env_ids[valid_indices]

            moved_vpt_for_ball = {i: None for i in range(num_envs)}

            states = self.move_vpt_objects(
                env_ids=env_ids,
                valid_indices=valid_indices,
                visibility_categories=visibility_categories,
                moved_vpt_for_ball=moved_vpt_for_ball,
                states=[goal_state, camera_state, agent_state, vpt_state],
                in_view_displaced=in_view_displaced,
                outside_fov_displaced=outside_fov_displaced,
                device=device)
            goal_state, camera_state, agent_state, vpt_state = states

            for local_idx, env_id in enumerate(env_ids):
                env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
                origin_z = self.scene.env_origins[env_id, 2]
                active_indices = self.active_vpt_indices[env_id_item]
                heights = self.all_vpt_dims[env_id, active_indices, 2]
                ratios = self.vpt_z_offset_ratios[env_id, active_indices]
                safe_z = origin_z + (heights * ratios)
                vpt_state[local_idx, active_indices, 2] = safe_z

            self.write_pose_to_sim(env_ids=valid_env_ids, indices=valid_indices,
                                   goal_default_state=goal_state,
                                   camera_obj_default_state=camera_state,
                                   agent_default_state=agent_state,
                                   vpt_obj_default_state=vpt_state)

            goal_new_pos = self._goal.data.root_pos_w[valid_env_ids]
            camera_new_pos = self._camera_obj.data.root_pos_w[valid_env_ids]
            agent_new_pos = self._agent.data.root_pos_w[valid_env_ids]
            vpt_new_pos = self._get_active_vpt_positions(valid_env_ids, base_pivoted=True)

            envs_need_spawn_retry = self.check_z_bounds(
                env_ids=env_ids, valid_indices=valid_indices,
                states=(goal_new_pos, camera_new_pos, agent_new_pos, vpt_new_pos),
                envs_need_spawn_retry=envs_need_spawn_retry, tolerance=5e-2)

            final_valid_mask = valid_mask & ~envs_need_spawn_retry
            if not final_valid_mask.any():
                continue

            final_valid_indices = torch.where(final_valid_mask)[0]
            final_valid_env_ids = env_ids[final_valid_indices]

            self.outside_fov_camera_movement(
                valid_env_ids=final_valid_env_ids, valid_indices=final_valid_indices,
                visibility_categories=visibility_categories,
                states=[goal_state, camera_state, agent_state, vpt_state], device=device)

            occlusion_valid_mask, envs_need_spawn_retry, _, states = self.occlusion_validation_check(
                final_valid_env_ids=final_valid_env_ids, valid_indices=final_valid_indices,
                visibility_categories=visibility_categories,
                envs_need_spawn_retry=envs_need_spawn_retry, env_dict={},
                states=[goal_state, camera_state, agent_state, vpt_state], device=device)

            if not rl_reset:
                geometric_valid_mask, envs_need_spawn_retry = self.geometric_occlusion_check(
                    env_ids=final_valid_env_ids, valid_indices=final_valid_indices,
                    occlusion_valid_mask=occlusion_valid_mask,
                    envs_need_spawn_retry=envs_need_spawn_retry, device=device)

                envs_need_spawn_retry = self.camera_pov_validation(
                    env_ids=final_valid_env_ids, valid_indices=final_valid_indices,
                    geometric_valid_mask=geometric_valid_mask,
                    visibility_categories=visibility_categories,
                    envs_need_spawn_retry=envs_need_spawn_retry,
                    folder_indices=folder_indices, spawn_attempt=spawn_attempt)

        # =================================================================
        # POST-LOOP FINALIZATION
        # =================================================================

        agent_state = self.place_object_safely(
            env_ids=env_ids,
            object_state=agent_state,
            vpt_state=vpt_state,
            safe_range=self.center_to_boundary - 1.0,
            object_type='agent')

        self.debug_plot_analytic_obbs(vpt_state)

        self.write_pose_to_sim(
            env_ids=env_ids,
            indices=torch.arange(len(env_ids), device=device),
            vpt_obj_default_state=vpt_state,
            agent_default_state=agent_state)

        if not rl_reset:
            success_mask = ~envs_need_spawn_retry
            successful_env_ids = env_ids[success_mask]

            subset_valid_points = []
            if len(successful_env_ids) > 0:
                subset_valid_points = self.generate_valid_circle_points(
                    env_ids=successful_env_ids, angle_step=2.0, max_attempts=100)

            if self.valid_viewpoint_poses is None:
                self.valid_viewpoint_poses = [None] * self.num_envs

            failed_env_ids = env_ids[envs_need_spawn_retry]
            for env_id in failed_env_ids:
                eid = env_id.item() if torch.is_tensor(env_id) else env_id
                self.valid_viewpoint_poses[eid] = torch.zeros((0, 3), device=device)

            for i, env_id in enumerate(successful_env_ids):
                eid = env_id.item() if torch.is_tensor(env_id) else env_id
                points_2d = subset_valid_points[i]

                if points_2d.shape[0] >= self.images_per_env:
                    agent_z = self._agent.data.default_root_state[env_id, 2]
                    points_3d = torch.zeros((points_2d.shape[0], 3), device=device)
                    points_3d[:, :2] = points_2d
                    points_3d[:, 2] = agent_z
                    self.valid_viewpoint_poses[eid] = points_3d
                else:
                    self.valid_viewpoint_poses[eid] = torch.zeros((0, 3), device=device)

            timer.update_status(spawn_attempt, envs_need_spawn_retry)

    # =========================================================================
    # FOV / GEOMETRIC CHECKS
    # =========================================================================

    def _is_point_valid_batch(self, points, env_ids, min_obstacle_distance=0.3,
                              min_camera_obstacle_distance=0.4, min_camera_target_distance=1.0,
                              min_target_obstacle_distance=None, check_agent_fov=False,
                              min_required_points=None) -> torch.Tensor:
        if min_required_points is None:
            min_required_points = self.images_per_env
        if min_target_obstacle_distance is None:
            min_target_obstacle_distance = self.goal_radius + 0.01

        valid_mask = self._check_geometric_validity(
            points, env_ids, min_obstacle_distance,
            min_camera_obstacle_distance, min_camera_target_distance,
            min_target_obstacle_distance)

        if not check_agent_fov or not valid_mask.any():
            return valid_mask

        valid_mask = self._check_fov_validity(points, env_ids, valid_mask, min_required_points)
        return valid_mask

    def _check_geometric_validity(self, points, env_ids, min_obs_dist,
                                  min_cam_obs_dist, min_cam_target_dist, min_target_obs_dist):
        device = points.device
        valid_mask = torch.ones(points.shape[0], dtype=torch.bool, device=device)

        env_origins = self.scene.env_origins[env_ids, :2]
        in_bounds = torch.all(
            (points >= env_origins - self.center_to_boundary) &
            (points <= env_origins + self.center_to_boundary), dim=1)
        valid_mask &= in_bounds

        if not valid_mask.any():
            return valid_mask

        active_obs_pos = self._get_active_obstacle_positions(env_ids)
        dist_pt_obs = torch.norm(points.unsqueeze(1) - active_obs_pos, dim=2)
        valid_mask &= (dist_pt_obs.min(dim=1)[0] >= min_target_obs_dist)

        cam_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
        valid_mask &= (torch.norm(points - cam_pos, dim=1) >= min_cam_target_dist)

        dist_cam_obs = torch.norm(cam_pos.unsqueeze(1) - active_obs_pos, dim=2)
        valid_mask &= (dist_cam_obs.min(dim=1)[0] >= min_cam_obs_dist)

        return valid_mask

    def _get_active_obstacle_positions(self, env_ids):
        device = env_ids.device
        all_pos = self._vpt_objects.data.object_pos_w[env_ids, :, :2]

        if isinstance(self.active_vpt_indices, list):
            if len(self.active_vpt_indices) > 0 and isinstance(self.active_vpt_indices[0], torch.Tensor):
                idx = torch.stack(self.active_vpt_indices).to(dtype=torch.long, device=device)
            else:
                idx = torch.tensor(self.active_vpt_indices, device=device, dtype=torch.long)
        else:
            idx = self.active_vpt_indices

        batch_idx = idx[env_ids].unsqueeze(-1).expand(-1, -1, 2)
        return torch.gather(all_pos, 1, batch_idx)

    def _check_fov_validity(self, points, env_ids, valid_mask, min_req_points):
        """Simulates view to check visibility. Enforces 50/50 cam/goal proximity split."""
        device = points.device
        points_to_check = torch.where(valid_mask)[0]

        if points_to_check.numel() == 0:
            return valid_mask

        saved_pos = self._agent.data.root_pos_w[env_ids].clone()
        saved_quat = self._agent.data.root_quat_w[env_ids].clone()

        env_queues, env_status = self._sample_fov_candidates(points, env_ids, points_to_check, max_s=120)

        # Override status to dual-quota format
        half_req = min_req_points // 2
        for eid in env_status:
            env_status[eid] = {'c_count': 0, 'g_count': 0, 'done': False}

        fov_valid_global = torch.zeros_like(valid_mask)
        max_steps = max(len(q['points']) for q in env_queues.values())

        for step in range(max_steps):
            if self.verbose >= 2 and step % 10 == 0:
                print(f"    🔄 FOV progress: {step+1}/{max_steps} points")

            step_ids, step_pts, step_idxs = [], [], []

            for eid, data in env_queues.items():
                if env_status[eid]['done']:
                    continue
                if step < len(data['points']):
                    step_ids.append(torch.tensor(eid, device=device))
                    step_pts.append(data['points'][step])
                    step_idxs.append(data['indices'][step])

            if not step_ids:
                if self.verbose >= 1 and all(s['done'] for s in env_status.values()):
                    print(f"    ✅ All environments achieved 50/50 split ({half_req} each). Early stop.")
                break

            b_envs = torch.stack(step_ids)
            b_pts = torch.stack(step_pts)
            b_idxs = torch.stack(step_idxs)

            self._teleport_and_step(b_envs, b_pts)
            g_vis, c_vis = self.check_batch_object_visibility(b_envs)
            is_vis = g_vis & c_vis

            # Classify by proximity and enforce quota
            b_cam_pos = self._camera_obj.data.root_pos_w[b_envs, :2]
            b_goal_pos = self._goal.data.root_pos_w[b_envs, :2]
            dist_cam = torch.norm(b_pts - b_cam_pos, dim=1)
            dist_goal = torch.norm(b_pts - b_goal_pos, dim=1)
            is_c_batch = dist_cam < dist_goal

            for i, eid in enumerate(b_envs.tolist()):
                if is_vis[i]:
                    is_c = is_c_batch[i].item()
                    if is_c:
                        if env_status[eid]['c_count'] < half_req:
                            env_status[eid]['c_count'] += 1
                        else:
                            is_vis[i] = False  # quota full
                    else:
                        if env_status[eid]['g_count'] < half_req:
                            env_status[eid]['g_count'] += 1
                        else:
                            is_vis[i] = False  # quota full

                    if (env_status[eid]['c_count'] >= half_req and
                            env_status[eid]['g_count'] >= half_req):
                        env_status[eid]['done'] = True
                        if self.verbose >= 1:
                            print(f"    🎯 Env {eid}: Reached 50/50 target ({half_req} C, {half_req} G).")

            fov_valid_global[b_idxs] = is_vis

        restore = torch.cat([saved_pos, saved_quat], dim=1)
        self._agent.write_root_com_pose_to_sim(restore, env_ids)

        return valid_mask & fov_valid_global

    def _sample_fov_candidates(self, points, env_ids, indices, max_s):
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
        cam_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
        goal_pos = self._goal.data.root_pos_w[env_ids, :2]

        dirs = ((cam_pos + goal_pos) / 2.0) - points
        yaws = torch.atan2(dirs[:, 1], dirs[:, 0])
        yaw_jitter = (torch.rand(len(env_ids), device=points.device) - 0.5) * 2 * math.radians(15)
        yaws = yaws + yaw_jitter

        pos = torch.zeros((len(env_ids), 3), device=points.device)
        pos[:, :2] = points
        pos[:, 2] = self._agent.data.default_root_state[env_ids, 2]

        quat = torch.zeros((len(env_ids), 4), device=points.device)
        quat[:, 0] = torch.cos(yaws / 2)
        quat[:, 3] = torch.sin(yaws / 2)

        self._agent.write_root_com_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids)

        self.sim.step()
        self._rgb_tiled_camera.update(self.sim.cfg.dt)
        self._agent.update(self.sim.cfg.dt)
        self._camera_obj.update(self.sim.cfg.dt)
        self._goal.update(self.sim.cfg.dt)
        self._vpt_objects.update(self.sim.cfg.dt)

    # =========================================================================
    # CIRCLE GENERATION — 50/50 POOL SEGREGATION (from v17)
    # =========================================================================

    def generate_valid_circle_points(self, env_ids: torch.Tensor, angle_step: float = 2.0,
                                     max_attempts: int = 300) -> List[torch.Tensor]:
        """Generate valid viewpoint positions enforcing 50/50 cam-proximal/goal-proximal split."""
        device = self.device
        num_envs = len(env_ids)
        MIN_REQUIRED_POINTS = self.images_per_env

        num_angles = int(360.0 / angle_step)
        angles = torch.linspace(0, 2 * math.pi, num_angles, device=device)

        fov_deg = 30.0
        fov_rad = math.radians(fov_deg)
        camera_pos = self._camera_obj.data.root_pos_w[env_ids, :2]
        goal_pos = self._goal.data.root_pos_w[env_ids, :2]
        d = torch.norm(camera_pos - goal_pos, dim=1)

        half_fov = torch.tensor(fov_rad / 2, device=self.device)
        radii = (d / 2) / torch.tan(half_fov)
        scale_factor = (torch.rand(num_envs, device=device) * 0.75) + 0.75
        radii = radii * scale_factor

        # --- Save radius to metadata cache ---
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            if env_id_item not in self.env_metadata_cache:
                self.env_metadata_cache[env_id_item] = {}
            self.env_metadata_cache[env_id_item]["viewpoint_radius"] = radii[i].item()

        radii = radii.unsqueeze(1)

        angles_expanded = angles.unsqueeze(0).expand(num_envs, -1)

        jitter_x = (torch.rand((num_envs, num_angles), device=device) * 2.0) - 1.0
        jitter_y = (torch.rand((num_envs, num_angles), device=device) * 2.0) - 1.0

        # Dynamic center: random point along cam->goal line
        direction_vectors = goal_pos - camera_pos
        t = (torch.rand(num_envs, 1, device=device) * 0.4) + 0.3
        centers = camera_pos + t * direction_vectors

        all_x = centers[:, 0].unsqueeze(1) + radii * torch.cos(angles_expanded) + jitter_x
        all_y = centers[:, 1].unsqueeze(1) + radii * torch.sin(angles_expanded) + jitter_y

        total_points = num_envs * num_angles
        all_points_batch = torch.stack([all_x, all_y], dim=2).reshape(total_points, 2)
        env_ids_batch = env_ids.unsqueeze(1).expand(-1, num_angles).reshape(total_points)

        # Step 1: Geometric validation
        geometric_valid = self._is_point_valid_batch(
            points=all_points_batch, env_ids=env_ids_batch, check_agent_fov=False)
        geometric_valid_per_env = geometric_valid.reshape(num_envs, num_angles)

        # Step 2: Pool segregation + displacement filtering
        displacement_filtered_points = []
        displacement_filtered_env_ids = []
        displacement_filtered_indices = []

        MIN_CANDIDATES_FOR_FOV = 40
        min_half_candidates = MIN_CANDIDATES_FOR_FOV // 2

        def filter_pool(points, center_pos):
            """Greedy spatial filter sorted by angle from center."""
            num_pts = len(points)
            if num_pts == 0:
                return points
            rel_pos = points - center_pos
            angles_rad = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
            sorted_points = points[torch.argsort(angles_rad)]
            diff = sorted_points.unsqueeze(0) - sorted_points.unsqueeze(1)
            pairwise_distances = torch.norm(diff, dim=2)
            selected_mask = torch.zeros(num_pts, dtype=torch.bool, device=device)
            selected_mask[0] = True
            for idx in range(1, num_pts):
                if torch.all(pairwise_distances[idx, selected_mask] >= self.min_viewpoint_distance):
                    selected_mask[idx] = True
            return sorted_points[selected_mask]

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()
            valid_mask = geometric_valid_per_env[i]

            if not valid_mask.any():
                if self.verbose >= 2:
                    print(f"  Env {env_id_item}: No geometric candidates")
                continue

            valid_points = all_points_batch[i * num_angles:(i + 1) * num_angles][valid_mask]

            cam_pos_2d = self._camera_obj.data.root_pos_w[env_id, :2]
            goal_pos_2d = self._goal.data.root_pos_w[env_id, :2]
            center_2d = centers[i]

            dist_cam = torch.norm(valid_points - cam_pos_2d, dim=1)
            dist_goal = torch.norm(valid_points - goal_pos_2d, dim=1)

            # Radial deadzone
            valid_dist_mask = (dist_cam >= 1.5) & (dist_goal >= 1.5)
            valid_points = valid_points[valid_dist_mask]
            dist_cam = dist_cam[valid_dist_mask]
            dist_goal = dist_goal[valid_dist_mask]

            if len(valid_points) == 0:
                if self.verbose >= 2:
                    print(f"  Env {env_id_item}: No candidates survived the deadzone filter.")
                continue

            # Segregate into cam-proximal and goal-proximal pools
            pool_a_mask = dist_cam < dist_goal   # label 1 (closer to cam)
            pool_b_mask = dist_goal < dist_cam   # label 0 (closer to goal)

            pool_a_points = valid_points[pool_a_mask]
            pool_b_points = valid_points[pool_b_mask]

            filtered_a = filter_pool(pool_a_points, center_2d)
            filtered_b = filter_pool(pool_b_points, center_2d)

            num_a = len(filtered_a)
            num_b = len(filtered_b)

            if num_a < min_half_candidates or num_b < min_half_candidates:
                if self.verbose >= 1:
                    print(f"  Env {env_id_item}: ❌ Failed 50/50 split criteria "
                          f"(Cam-proximal: {num_a}, Goal-proximal: {num_b}, need {min_half_candidates} each). "
                          f"Skipping FOV check.")
                continue

            filtered_candidates = torch.cat([filtered_a, filtered_b], dim=0)

            if self.verbose >= 2:
                print(f"  Env {env_id_item}: Split filter passed. A:{num_a}, B:{num_b} "
                      f"(Total: {len(filtered_candidates)})")

            displacement_filtered_points.append(filtered_candidates)
            displacement_filtered_env_ids.extend([env_id_item] * len(filtered_candidates))
            displacement_filtered_indices.extend([i] * len(filtered_candidates))

        if len(displacement_filtered_points) == 0:
            if self.verbose >= 1:
                print("  ❌ No candidates passed displacement filter for any environment")
            return [torch.zeros((0, 2), device=device) for _ in range(num_envs)]

        all_candidates = torch.cat(displacement_filtered_points, dim=0)
        all_candidates_env_ids = torch.tensor(displacement_filtered_env_ids, dtype=torch.long, device=device)
        all_candidates_indices = torch.tensor(displacement_filtered_indices, dtype=torch.long, device=device)

        original_agent_pos = self._agent.data.root_pos_w[env_ids].clone()
        original_agent_quat = self._agent.data.root_quat_w[env_ids].clone()

        # Step 3: FOV check on pool-segregated candidates
        fov_valid_mask = self._is_point_valid_batch(
            points=all_candidates,
            env_ids=all_candidates_env_ids,
            check_agent_fov=True,
            min_required_points=MIN_REQUIRED_POINTS)

        self._agent.write_root_pose_to_sim(
            torch.cat([original_agent_pos, original_agent_quat], dim=-1), env_ids=env_ids)

        # Step 4: Collect final valid points per env
        all_valid_points = []

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()

            env_mask = all_candidates_indices == i
            env_fov_valid = fov_valid_mask[env_mask]

            num_candidates = env_mask.sum().item()
            num_fov_valid = env_fov_valid.sum().item()

            if not env_fov_valid.any():
                all_valid_points.append(torch.zeros((0, 2), device=device))
                if self.verbose >= 1:
                    print(f"  Env {env_id_item}: ❌ 0/{num_candidates} passed FOV check")
                continue

            valid_points_tensor = all_candidates[env_mask][env_fov_valid]
            fov_rejection_rate = (1 - num_fov_valid / num_candidates) * 100 if num_candidates > 0 else 0

            if len(valid_points_tensor) >= MIN_REQUIRED_POINTS:
                all_valid_points.append(valid_points_tensor)
                if self.verbose >= 2:
                    print(f"  Env {env_id_item}: ✅ {len(valid_points_tensor)}/{num_candidates} "
                          f"passed FOV ({fov_rejection_rate:.1f}% rejected)")
            else:
                all_valid_points.append(torch.zeros((0, 2), device=device))
                if self.verbose >= 1:
                    print(f"  Env {env_id_item}: ❌ Only {len(valid_points_tensor)}/{MIN_REQUIRED_POINTS} "
                          f"points ({fov_rejection_rate:.1f}% FOV rejection)")

        return all_valid_points

    # =========================================================================
    # IMAGE COLLECTION — WITH DEPTH LABELS (from v17)
    # =========================================================================

    def _collect_images_for_slot(self, env_id: torch.Tensor, folder_idx: int) -> None:
        """
        Teleports the agent to selected viewpoints and collects sensor data.
        Tracks depth label per image: 1 if camera is closer, 0 if goal is closer.
        """
        env_slot = env_id.item() if torch.is_tensor(env_id) else env_id
        global_env_id = self.slot_to_env_id[env_slot]
        viewpoints = self.selected_viewpoints_for_collection[env_slot]

        if viewpoints is None:
            raise RuntimeError(f"No viewpoints selected for Slot {env_slot} (Env {global_env_id})")

        if self.verbose >= 1:
            print(f"    📸 Collecting {self.images_per_env} images | Slot {env_slot} -> Folder {folder_idx}")

        single_env_tensor = torch.tensor([env_slot], dtype=torch.long, device=self.device)
        zero_velocity = torch.zeros((1, 6), device=self.device)

        cam_pos_2d = self._camera_obj.data.root_pos_w[env_slot, :2]
        goal_pos_2d = self._goal.data.root_pos_w[env_slot, :2]
        midpoint = (cam_pos_2d + goal_pos_2d) / 2.0

        agent_z = self._agent.data.default_root_state[env_slot, 2]

        # Initialize depth label dict for this env
        env_key = f"env_{folder_idx}"
        if env_key not in self.depth_labels_data:
            self.depth_labels_data[env_key] = {}

        for i in range(self.images_per_env):
            target_pos_2d = viewpoints[i][:2]

            # --- Depth label: 1 if closer to camera, 0 if closer to goal ---
            dist_cam = torch.norm(target_pos_2d - cam_pos_2d).item()
            dist_goal = torch.norm(target_pos_2d - goal_pos_2d).item()
            depth_label = 1 if dist_cam < dist_goal else 0
            self.depth_labels_data[env_key][f"image_{i:04d}"] = depth_label

            # --- OLD ---
            # direction = midpoint - target_pos_2d
            # yaw = torch.atan2(direction[1], direction[0])
            # yaw_jitter = (torch.rand(1, device=self.device) - 0.5) * 2 * math.radians(20)
            # yaw = yaw + yaw_jitter.squeeze()

            # --- NEW ---
            direction = midpoint - target_pos_2d
            base_yaw = torch.atan2(direction[1], direction[0])

            # Compute angular offset from base_yaw to the camera object
            cam_dir = cam_pos_2d - target_pos_2d
            angle_to_cam = torch.atan2(cam_dir[1], cam_dir[0])
            delta_to_cam = angle_to_cam - base_yaw
            # Normalize to [-pi, pi]
            delta_to_cam = (delta_to_cam + math.pi) % (2 * math.pi) - math.pi

            # Jitter window: camera must land within this many degrees of center
            HALF_FOV      = math.radians(15.0)  # true FOV half-angle
            SAFETY_MARGIN = math.radians(5.0)   # keep tip this far inside the edge
            DESIRED_MAX   = math.radians(20.0)  # your original jitter budget

            # Allowed jitter = intersection of [desired_max] and [cam stays in FOV]
            delta_val   = delta_to_cam.item()
            jitter_lo   = max(delta_val - HALF_FOV + SAFETY_MARGIN, -DESIRED_MAX)
            jitter_hi   = min(delta_val + HALF_FOV - SAFETY_MARGIN,  DESIRED_MAX)

            if jitter_lo < jitter_hi:
                jitter_val = random.uniform(jitter_lo, jitter_hi)
            else:
                # Camera is so far off-center (shouldn't happen post-FOV-check),
                # just look straight at it
                jitter_val = delta_val

            yaw = base_yaw + jitter_val

            half_yaw = yaw / 2
            quat = torch.tensor([torch.cos(half_yaw), 0.0, 0.0, torch.sin(half_yaw)], device=self.device)

            pose = torch.tensor([
                target_pos_2d[0], target_pos_2d[1], agent_z,
                quat[0], quat[1], quat[2], quat[3]
            ], device=self.device).unsqueeze(0)

            self._agent.write_root_com_pose_to_sim(pose, single_env_tensor)
            self._agent.write_root_com_velocity_to_sim(zero_velocity, single_env_tensor)

            for _ in range(10):
                self.sim.step()
                self._rgb_tiled_camera.update(self.sim.cfg.dt)
                if self.save_camera_pov:
                    self._occlusion_camera.update(self.sim.cfg.dt)

            rgb = self._rgb_tiled_camera.data.output["rgb"][env_slot]
            depth = self._rgb_tiled_camera.data.output["distance_to_camera"][env_slot]
            semantic = self._rgb_tiled_camera.data.output["semantic_segmentation"][env_slot]
            cam_pov = (self._occlusion_camera.data.output["semantic_segmentation"][env_slot]
                    if self.save_camera_pov else None)
            cam_pov_rgb = (self._occlusion_camera.data.output["rgb"][env_slot]
                        if self.save_camera_pov else None)

            self._save_single_image(env_slot, folder_idx, rgb, depth, semantic, cam_pov, image_idx=i, cam_pov_rgb=cam_pov_rgb)

        # Save depth labels and config, then reset per-env state
        self._save_depth_labels()
        self._save_env_config_to_json_new(env_slot, folder_idx)
        self.selected_viewpoints_for_collection[env_slot] = None
        self.depth_labels_json_path = os.path.join(self.base_path, "depth_labels.json")
        self.depth_labels_data = {}

        if self.verbose >= 1:
            print(f"    ✅ Saved Folder {folder_idx}")

    def _save_single_image(self, env_slot, folder_idx, rgb_data, depth_data,
                        semantic_data, camera_pov_data=None, image_idx=0,
                        cam_pov_rgb=None) -> None:
        label = self.env_visibility_labels.get(folder_idx)
        if label not in ["Yes", "No"]:
            raise RuntimeError(f"Invalid visibility label '{label}' for folder {folder_idx}")

        base_dir = f"{self.base_path}/{{}}/{label}/env_{folder_idx}"
        rgb_dir = base_dir.format("RGB")
        depth_dir = base_dir.format("Depth")
        semantic_dir = base_dir.format("Semantic")

        if image_idx == 0:
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(semantic_dir, exist_ok=True)
            if self.save_camera_pov:
                os.makedirs(base_dir.format("cam"), exist_ok=True)

        # RGB
        rgb_np = rgb_data.cpu().numpy()
        if rgb_np.dtype != np.uint8:
            rgb_np = (rgb_np * 255.0).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]
        cv2.imwrite(f"{rgb_dir}/image_{image_idx:04d}.png", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Depth
        depth_np = depth_data.cpu().numpy()
        valid_mask = ~np.isinf(depth_np)
        max_val = depth_np[valid_mask].max() if valid_mask.any() else 0
        depth_np[~valid_mask] = max_val
        if depth_np.max() > depth_np.min():
            depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            depth_norm = np.zeros_like(depth_np, dtype=np.uint8)
        cv2.imwrite(f"{depth_dir}/image_{image_idx:04d}.png", depth_norm)

        # Semantic
        if semantic_data is not None:
            sem_np = semantic_data.cpu().numpy()
            if sem_np.dtype != np.uint8:
                sem_np = (sem_np * 255.0).astype(np.uint8) if sem_np.max() <= 1.0 else sem_np.astype(np.uint8)
            if sem_np.shape[-1] == 4:
                sem_np = sem_np[..., :3]
            cv2.imwrite(f"{semantic_dir}/image_{image_idx:04d}.png", cv2.cvtColor(sem_np, cv2.COLOR_RGB2BGR))

        # Camera POV (once per env)
        if self.save_camera_pov and camera_pov_data is not None and image_idx == 0:
            cam_dir = base_dir.format("cam")
            cam_pov_np = camera_pov_data.cpu().numpy()
            if cam_pov_np.dtype != np.uint8:
                cam_pov_np = (cam_pov_np * 255.0).astype(np.uint8) if cam_pov_np.max() <= 1.0 else cam_pov_np.astype(np.uint8)
            if cam_pov_np.shape[-1] == 4:
                cam_pov_np = cam_pov_np[..., :3]
            cv2.imwrite(f"{cam_dir}/cam_pov.png", cv2.cvtColor(cam_pov_np, cv2.COLOR_RGB2BGR))
        
        # ADD THIS:
        if self.save_camera_pov and cam_pov_rgb is not None and image_idx == 0:
            cam_rgb_dir = base_dir.format("cam_rgb")
            os.makedirs(cam_rgb_dir, exist_ok=True)
            cam_rgb_np = cam_pov_rgb.cpu().numpy()
            if cam_rgb_np.dtype != np.uint8:
                cam_rgb_np = (cam_rgb_np * 255.0).astype(np.uint8) if cam_rgb_np.max() <= 1.0 else cam_rgb_np.astype(np.uint8)
            if cam_rgb_np.shape[-1] == 4:
                cam_rgb_np = cam_rgb_np[..., :3]
            cv2.imwrite(f"{cam_rgb_dir}/cam_pov_rgb.png", cv2.cvtColor(cam_rgb_np, cv2.COLOR_RGB2BGR))

    # =========================================================================
    # RANDOMIZATION
    # =========================================================================

    def _cache_base_dims(self):
        self.vpt_base_dims = []

        for key, obj_cfg in self.cfg.vpt_objects.rigid_objects.items():
            spawn_cfg = obj_cfg.spawn
            dims = torch.tensor([1.0, 1.0, 1.0], device=self.device)

            if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                scale = getattr(spawn_cfg, "scale", (1.0, 1.0, 1.0))
                if scale is None:
                    scale = (1.0, 1.0, 1.0)
                filename = spawn_cfg.usd_path.split("/")[-1].split(".")[0]

                if filename.endswith(('Table_A', 'Table_B', 'Bench')):
                    dims = torch.tensor(scale, device=self.device)
                elif filename.endswith(('X', 'L', 'T', 'I', 'A', 'H', 'Z')):
                    dims = torch.tensor([1.0 * scale[0], 0.25 * scale[1], 1.0 * scale[2]], device=self.device)
                else:
                    dims = torch.tensor(scale, device=self.device)

            elif isinstance(spawn_cfg, sim_utils.MeshCuboidCfg):
                dims = torch.tensor(spawn_cfg.size, device=self.device)

            elif isinstance(spawn_cfg, (sim_utils.MeshConeCfg, sim_utils.MeshCylinderCfg)):
                r = spawn_cfg.radius
                h = spawn_cfg.height
                dims = torch.tensor([2 * r, 2 * r, h], device=self.device)

            self.vpt_base_dims.append(dims)

        if len(self.vpt_base_dims) > 0:
            self.vpt_base_dims = torch.stack(self.vpt_base_dims)
        else:
            self.vpt_base_dims = torch.empty((0, 3), device=self.device)

        if self.verbose >= 1:
            print(f"📦 Cached base dimensions for {len(self.vpt_base_dims)} objects.")

    def randomize_shape_scale(self, prim_path_expr: str | list, is_random: bool = True):
        world = World.instance()
        if world.is_playing():
            world.pause()

        stage = get_current_stage()

        if not hasattr(self, "_cached_mesh_points"):
            self._cached_mesh_points = {}

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        if not hasattr(self, "all_vpt_dims") or self.all_vpt_dims.shape[0] != self.num_envs:
            self.all_vpt_dims = torch.zeros((self.num_envs, self.cfg.num_vpt_objs, 3), device=self.device)
            self.vpt_obj_default_state = torch.zeros((self.num_envs, self.num_objs, 3), device=self.device)

        if not hasattr(self, "vpt_z_offset_ratios") or self.vpt_z_offset_ratios.shape != (self.num_envs, self.cfg.num_vpt_objs):
            self.vpt_z_offset_ratios = torch.zeros((self.num_envs, self.cfg.num_vpt_objs), device=self.device)

        if not hasattr(self, "vpt_shapes") or self.vpt_z_offset_ratios.shape != (self.num_envs, self.cfg.num_vpt_objs):
            self.vpt_shapes = torch.zeros((self.num_envs, self.cfg.num_vpt_objs), device=self.device)

        if isinstance(prim_path_expr, str):
            prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
        elif isinstance(prim_path_expr, list):
            prim_paths = []
            for expr in prim_path_expr:
                prim_paths.extend(sim_utils.find_matching_prim_paths(expr))

        print(f"\n[Randomizing Scale & Geometry] Processing {len(prim_paths)} objects...")
        obj_configs = list(self.cfg.vpt_objects.rigid_objects.values())

        with Sdf.ChangeBlock():
            for prim_path in prim_paths:
                root_prim = stage.GetPrimAtPath(prim_path)
                if not root_prim.IsValid():
                    continue

                try:
                    path_str = root_prim.GetPath().pathString
                    path_parts = path_str.split("/")
                    env_part = next(p for p in path_parts if p.startswith("env_"))
                    env_idx = int(env_part.split("_")[-1])
                    obs_part = next(p for p in path_parts if p.startswith("obs_"))
                    obj_idx = int(obs_part.split("_")[-1])
                except:
                    continue

                if obj_idx >= len(obj_configs):
                    continue
                obj_cfg = obj_configs[obj_idx]
                spawn_cfg = obj_cfg.spawn

                xform = UsdGeom.Xformable(root_prim)
                scale_op = None
                translate_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = op
                    elif op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op

                if scale_op is None:
                    scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                if translate_op is None:
                    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

                if is_random:
                    base_dim = self.vpt_base_dims[obj_idx].cpu().numpy()
                    final_scale_vec = Gf.Vec3d(1, 1, 1)
                    final_z_pos = 0.0
                    z_offset_multiplier = 0.0
                    shape_name = -1

                    if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                        filename = spawn_cfg.usd_path.split("/")[-1].split(".")[0]
                        if filename.endswith(('X', 'L', 'T', 'A', 'H', 'I', 'Z', 'Table_A', 'Table_B', 'Bench')):
                            z_offset_multiplier = 0.0
                            if filename.endswith(("L",)):
                                s_factor = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(base_dim[0]*s_factor, base_dim[1]*s_factor, base_dim[2]*s_factor)
                            elif filename.endswith(("Table_B", "Bench")):
                                s_factor = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(base_dim[0]*s_factor, base_dim[1]*s_factor, base_dim[2]*s_factor)
                            elif filename.endswith(("Table_A",)):
                                s_x = random.uniform(0.5, 3.0)
                                s_y = random.uniform(0.5, 3.0)
                                s_z = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(base_dim[0]*s_x, base_dim[1]*s_y, base_dim[2]*s_z)
                            elif filename.endswith(("X", "A", "H", "I", "Z")):
                                s_xz = random.uniform(0.5, 3.0)
                                s_y = random.uniform(1.0, 5.0)
                                final_scale_vec = Gf.Vec3d(base_dim[0]*s_xz, base_dim[1]*s_y, base_dim[2]*s_xz)
                            final_z_pos = 0.0
                        else:
                            z_offset_multiplier = 0.0
                            s_xy = random.uniform(0.5, 3.0)
                            s_z = random.uniform(0.5, 3.0)
                            final_scale_vec = Gf.Vec3d(base_dim[0]*s_xy, base_dim[1]*s_xy, base_dim[2]*s_z)
                            shape_name = -1

                    elif isinstance(spawn_cfg, sim_utils.MeshCuboidCfg):
                        shape_name = 2
                        z_offset_multiplier = 0.5
                        s_x = random.uniform(0.5, 3.0)
                        s_y = random.uniform(0.5, 3.0)
                        s_z = random.uniform(0.5, 3.0)
                        final_scale_vec = Gf.Vec3d(s_x, s_y, s_z)
                        total_height = base_dim[2] * s_z
                        final_z_pos = total_height * z_offset_multiplier

                    elif isinstance(spawn_cfg, (sim_utils.MeshCylinderCfg, sim_utils.MeshConeCfg)):
                        if isinstance(spawn_cfg, sim_utils.MeshCylinderCfg):
                            shape_name = 3
                            z_offset_multiplier = 0.5
                        elif isinstance(spawn_cfg, sim_utils.MeshConeCfg):
                            shape_name = 4
                            z_offset_multiplier = 0.0
                        s_r = random.uniform(0.75, 1.0)
                        s_h = random.uniform(0.75, 3.0)
                        final_scale_vec = Gf.Vec3d(s_r, s_r, s_h)
                        total_height = base_dim[2] * s_h
                        final_z_pos = total_height * z_offset_multiplier

                    scale_op.Set(final_scale_vec)
                    current_trans = translate_op.Get()
                    translate_op.Set(Gf.Vec3d(current_trans[0], current_trans[1], final_z_pos))

                    self.vpt_obj_default_state[env_idx, obj_idx, 2] = final_z_pos
                    self.vpt_z_offset_ratios[env_idx, obj_idx] = z_offset_multiplier
                    self.vpt_shapes[env_idx, obj_idx] = shape_name

                    bbox_cache.Clear()
                    local_bound = bbox_cache.ComputeLocalBound(root_prim).GetRange()
                    final_w = local_bound.GetMax()[0] - local_bound.GetMin()[0]
                    final_l = local_bound.GetMax()[1] - local_bound.GetMin()[1]
                    final_h = local_bound.GetMax()[2] - local_bound.GetMin()[2]
                    self.all_vpt_dims[env_idx, obj_idx, 0] = final_w
                    self.all_vpt_dims[env_idx, obj_idx, 1] = final_l
                    self.all_vpt_dims[env_idx, obj_idx, 2] = final_h

        world.play()
        self.sim.step(render=False)

    def randomize_shape_color(self, prim_path_expr: str | list,
                              random_roughness: bool = False, random_metallic: bool = False):
        stage = get_current_stage()

        if isinstance(prim_path_expr, str):
            prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
        elif isinstance(prim_path_expr, list):
            prim_paths = []
            for expr in prim_path_expr:
                prim_paths.extend(sim_utils.find_matching_prim_paths(expr))

        with Sdf.ChangeBlock():
            for prim_path in prim_paths:
                rand_color = Gf.Vec3f(self.get_color())
                rand_roughness = random.random()
                rand_metallic = random.random()

                def _set_shader_attr(shader_spec, attr_name, value, type_name):
                    attr_spec = shader_spec.GetAttributeAtPath(
                        shader_spec.path.AppendProperty(attr_name))
                    if not attr_spec:
                        attr_spec = Sdf.AttributeSpec(shader_spec, attr_name, type_name)
                    attr_spec.default = value

                standard_shader_path = prim_path + "/geometry/material/Shader"
                layer = stage.GetRootLayer()
                check_prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
                color_check = check_prim_spec.GetAttributeAtPath(
                    standard_shader_path + ".inputs:diffuseColor")

                if color_check:
                    shader_spec = Sdf.CreatePrimInLayer(layer, standard_shader_path)
                    color_check.default = rand_color
                    if random_roughness:
                        _set_shader_attr(shader_spec, "inputs:roughness", rand_roughness, Sdf.ValueTypeNames.Float)
                    if random_metallic:
                        _set_shader_attr(shader_spec, "inputs:metallic", rand_metallic, Sdf.ValueTypeNames.Float)
                    continue

                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                target_shader_path = None
                for child in Usd.PrimRange(prim):
                    if child.GetTypeName() == "Shader":
                        target_shader_path = child.GetPath().pathString
                        break

                if target_shader_path:
                    shader_spec = Sdf.CreatePrimInLayer(layer, target_shader_path)
                    _set_shader_attr(shader_spec, "inputs:diffuseColor", rand_color, Sdf.ValueTypeNames.Color3f)
                    if random_roughness:
                        _set_shader_attr(shader_spec, "inputs:roughness", rand_roughness, Sdf.ValueTypeNames.Float)
                    if random_metallic:
                        _set_shader_attr(shader_spec, "inputs:metallic", rand_metallic, Sdf.ValueTypeNames.Float)

    def randomize_spherical_lights(self, prim_paths, random_light_off=False):
        stage = get_current_stage()
        active_paths = set(prim_paths)
        env_light_positions = {}
        min_separation = 8.0

        if random_light_off and len(prim_paths) > 0:
            num_to_keep = random.randint(1, len(prim_paths))
            active_paths = set(random.sample(prim_paths, num_to_keep))

        for path in prim_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue

            match = re.search(r"env_(\d+)", path)
            if not match:
                continue
            env_idx = int(match.group(1))

            if env_idx not in env_light_positions:
                env_light_positions[env_idx] = []

            if path not in active_paths:
                prim.GetAttribute("inputs:intensity").Set(0.0)
                continue

            rand_intensity = random.uniform(40_000.0, 75_000.0)
            prim.GetAttribute("inputs:intensity").Set(rand_intensity)

            rand_z_offset = random.uniform(7.5, 15.0)
            valid_ranges = [(0.1, 0.3), (0.7, 0.9)]
            cand_x, cand_y = 0.0, 0.0

            for _ in range(20):
                rx_min, rx_max = random.choice(valid_ranges)
                mag_x = random.uniform(rx_min, rx_max) * self.center_to_boundary
                cand_x = mag_x * random.choice([-1, 1])
                ry_min, ry_max = random.choice(valid_ranges)
                mag_y = random.uniform(ry_min, ry_max) * self.center_to_boundary
                cand_y = mag_y * random.choice([-1, 1])

                collision = any(math.hypot(cand_x - ex, cand_y - ey) < min_separation
                                for (ex, ey) in env_light_positions[env_idx])
                if not collision:
                    break

            env_light_positions[env_idx].append((cand_x, cand_y))
            new_pos = Gf.Vec3d(float(cand_x), float(cand_y), float(rand_z_offset))

            xform = UsdGeom.Xformable(prim)
            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xform.AddTranslateOp()
            translate_op.Set(new_pos)

            prim.GetAttribute("inputs:radius").Set(random.uniform(0.75, 2.25))
            prim.GetAttribute("inputs:enableColorTemperature").Set(True)
            prim.GetAttribute("inputs:colorTemperature").Set(random.uniform(2000.0, 8000.0))

    def get_color(self):
        forbidden_colors = [
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            np.array([0.2, 0.8, 0.2]), np.array([0.0, 0.0, 1.0]),
            np.array([0.8, 0.0, 0.0]), np.array([1.0, 0.75, 0.8]),
            np.array([1.0, 0.2, 0.6]), np.array([0.9, 0.1, 0.5]),
            np.array([0.95, 0.3, 0.6]), np.array([0.0, 0.0, 0.0])
        ]
        threshold = 0.15
        valid = False
        base = None

        while not valid:
            base = np.array([random.uniform(0.2, 0.9) for _ in range(3)])
            too_close = any(np.linalg.norm(base - fc) < threshold for fc in forbidden_colors)
            if too_close:
                continue
            r, g, b = base[0], base[1], base[2]
            is_green_dominant = (g > r + 0.05) and (g > b + 0.05)
            is_bright_green = (g > 0.6) and (r < 0.4) and (b < 0.4)
            if is_green_dominant or is_bright_green:
                continue
            valid = True

        return base[0], base[1], base[2]

    # =========================================================================
    # SIM WRITE / MATERIAL
    # =========================================================================

    def write_pose_to_sim(self, env_ids, indices, goal_default_state=None,
                          camera_obj_default_state=None, agent_default_state=None,
                          vpt_obj_default_state=None, device=None):
        if device is None:
            device = self._agent.device

        if goal_default_state is not None:
            self._goal.write_root_com_pose_to_sim(goal_default_state[indices, :7], env_ids)
            self._goal.write_root_com_velocity_to_sim(torch.zeros((len(env_ids), 6), device=device), env_ids)

        if camera_obj_default_state is not None:
            self._camera_obj.write_root_com_pose_to_sim(camera_obj_default_state[indices, :7], env_ids)
            self._camera_obj.write_root_com_velocity_to_sim(torch.zeros((len(env_ids), 6), device=device), env_ids)

        if agent_default_state is not None:
            self._agent.write_root_com_pose_to_sim(agent_default_state[indices, :7], env_ids)
            self._agent.write_root_com_velocity_to_sim(torch.zeros((len(env_ids), 6), device=device), env_ids)

        if vpt_obj_default_state is not None:
            self._vpt_objects.write_object_pose_to_sim(vpt_obj_default_state[indices, :, :7], env_ids)
            self._vpt_objects.write_object_velocity_to_sim(
                torch.zeros((len(env_ids), self.num_objs, 6), device=device), env_ids)

        for _ in range(1):
            self.sim.step()
            self._vpt_objects.update(self.sim.cfg.dt)

    def get_material_configs(self, material_type: str) -> list[sim_utils.MdlFileCfg]:
        if material_type == "mat":
            raw_paths = get_mat_material_paths()
            tex_scale = (1000.0, 1000.0)
        elif material_type == "vpt":
            raw_paths = get_vpt_material_paths()
            tex_scale = (2.0, 2.0)
        else:
            raise ValueError(f"Unknown material type: {material_type}")

        num_to_select = min(len(raw_paths), 100)
        selected_paths = random.sample(raw_paths, num_to_select)
        print(f"[INFO] {material_type.upper()} Config: {len(selected_paths)} materials selected.")

        return [
            sim_utils.MdlFileCfg(mdl_path=path, project_uvw=True, texture_scale=tex_scale)
            for path in selected_paths
        ]

    def randomize_material(self, prim_paths: list, material_type: str):
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

    # =========================================================================
    # OBB / COLLISION
    # =========================================================================

    def get_obb_hitbox(self, env_ids: torch.Tensor, vpt_state: torch.Tensor) -> torch.Tensor:
        signs = torch.tensor(
            [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
            device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        dims = self.all_vpt_dims[env_ids]
        padding_reduction = 0.05
        half_extents = (dims / 2.0) - padding_reduction
        half_extents = torch.max(half_extents, torch.tensor(0.01, device=self.device))

        local_corners = half_extents.unsqueeze(2) * signs
        quats = vpt_state[..., 3:7]
        pos = vpt_state[..., :3]

        rot_mats = math_utils.matrix_from_quat(quats)
        rotated_corners = torch.matmul(local_corners, rot_mats.transpose(-1, -2))
        world_corners = rotated_corners + pos.unsqueeze(2)
        return world_corners

    def update_obb_cache(self, env_ids: torch.Tensor, vpt_state: torch.Tensor):
        if not hasattr(self, "obb_corners_cache"):
            self.obb_corners_cache = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs, 8, 3),
                device=self.device, dtype=torch.float32)

        corners = self.get_obb_hitbox(env_ids, vpt_state)
        self.obb_corners_cache[env_ids] = corners

    def _get_object_corners(self, pos, quat, object_type='agent'):
        if object_type == 'agent':
            half_extents = torch.tensor([0.1, 0.1, 0.1], device=self.device)
        elif object_type == 'cam_obj':
            half_extents = torch.tensor([0.55, 0.575, 0.45], device=self.device)
        elif object_type == 'goal':
            half_extents = torch.tensor([0.2, 0.2, 0.2], device=self.device)
        else:
            half_extents = torch.tensor([0.1, 0.1, 0.1], device=self.device)

        signs = torch.tensor([
            [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
            [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]
        ], device=self.device)

        local_corners = half_extents * signs
        rot_mat = math_utils.matrix_from_quat(quat).unsqueeze(1)
        rotated_corners = torch.matmul(
            local_corners.unsqueeze(0).unsqueeze(0), rot_mat.transpose(-1, -2))
        world_corners = rotated_corners + pos.unsqueeze(1).unsqueeze(2)
        return world_corners

    def _check_collisions_vectorized(self, env_ids, proposed_pos, proposed_quat, object_type='agent'):
        agent_corners = self._get_object_corners(proposed_pos, proposed_quat, object_type=object_type)
        ac_xy = agent_corners[..., :2]

        env_origins_xy = self.scene.env_origins[env_ids][:, :2]
        origins_expanded = env_origins_xy.unsqueeze(1).unsqueeze(1)
        limit = self.center_to_boundary
        out_of_bounds = (ac_xy < origins_expanded - limit) | (ac_xy > origins_expanded + limit)
        is_colliding_wall = out_of_bounds.any(dim=-1).any(dim=-1).squeeze(-1)

        obs_corners = self.obb_corners_cache[env_ids]
        num_obs = obs_corners.shape[1]
        oc_xy = obs_corners[..., :2]

        a_edge1 = ac_xy[..., 1, :] - ac_xy[..., 0, :]
        a_edge2 = ac_xy[..., 3, :] - ac_xy[..., 0, :]
        a_axes = torch.stack([a_edge1, a_edge2], dim=2)
        a_axes = a_axes / (torch.norm(a_axes, dim=-1, keepdim=True) + 1e-6)

        o_edge1 = oc_xy[..., 1, :] - oc_xy[..., 0, :]
        o_edge2 = oc_xy[..., 3, :] - oc_xy[..., 0, :]
        o_axes = torch.stack([o_edge1, o_edge2], dim=2)
        o_axes = o_axes / (torch.norm(o_axes, dim=-1, keepdim=True) + 1e-6)

        all_axes = torch.cat([a_axes.expand(-1, num_obs, -1, -1), o_axes], dim=2)
        axes_T = all_axes.transpose(-1, -2)
        proj_a = torch.matmul(ac_xy.expand(-1, num_obs, -1, -1), axes_T)
        proj_o = torch.matmul(oc_xy, axes_T)

        min_a = proj_a.min(dim=2).values
        max_a = proj_a.max(dim=2).values
        min_o = proj_o.min(dim=2).values
        max_o = proj_o.max(dim=2).values

        overlap_axes = (max_a >= min_o) & (max_o >= min_a)
        is_colliding_2d = overlap_axes.all(dim=2)

        return is_colliding_2d.any(dim=1) | is_colliding_wall

    def debug_plot_analytic_obbs(self, vpt_state):
        signs = torch.tensor(
            [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
            device=self.device, dtype=torch.float32)

        env_id = 0
        active_indices = self.active_vpt_indices[env_id]
        if len(active_indices) == 0:
            print("No active objects in Env 0 to plot.")
            return

        dims = self.all_vpt_dims[env_id, active_indices, :3]
        half_extents = dims / 2.0
        positions = vpt_state[env_id, active_indices, :3]
        quats = vpt_state[env_id, active_indices, 3:7]

        local_corners = half_extents.unsqueeze(1) * signs.unsqueeze(0)
        rot_mats = math_utils.matrix_from_quat(quats)
        rotated_corners = torch.matmul(local_corners, rot_mats.transpose(-1, -2))
        world_corners = rotated_corners + positions.unsqueeze(1)

        corners_np = world_corners.cpu().numpy()
        pos_np = positions.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_title(f"Analytic OBB Verification (Env {env_id})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        limit = self.center_to_boundary
        boundary = plt.Rectangle((-limit, -limit), 2*limit, 2*limit, fill=False, linestyle='--', color='k')
        ax.add_patch(boundary)

        for i in range(len(active_indices)):
            c = corners_np[i]
            ax.plot(c[[0,1,2,3,0], 0], c[[0,1,2,3,0], 1], 'b-', alpha=0.7)
            ax.plot(c[[4,5,6,7,4], 0], c[[4,5,6,7,4], 1], 'r-', alpha=0.7)
            ax.plot(pos_np[i, 0], pos_np[i, 1], 'ko', markersize=3)

        plt.grid(True)
        output_filename = f"debug_obb_env_{env_id}.png"
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved analytic OBB plot to: {output_filename}")
        plt.close(fig)

    def place_object_safely(self, env_ids, object_state, vpt_state, safe_range,
                            range_offsets=None, object_type='agent', custom_centers=None):
        device = self.device
        num_envs = len(env_ids)
        BATCH_SIZE = 1000

        self.update_obb_cache(env_ids, vpt_state)

        if custom_centers is not None:
            base_xy = custom_centers[:, :2].unsqueeze(1)
        else:
            base_xy = self.scene.env_origins[env_ids][:, :2].unsqueeze(1)

        if range_offsets is not None:
            base_xy = base_xy + range_offsets.unsqueeze(1)

        rand_xy = sample_uniform(-safe_range, safe_range, (num_envs, BATCH_SIZE, 2), device)
        rand_yaw = sample_uniform(0, 2 * math.pi, (num_envs, BATCH_SIZE), device)

        cands_pos = torch.zeros((num_envs, BATCH_SIZE, 3), device=device)
        cands_pos[..., :2] = base_xy + rand_xy
        cands_pos[..., 2] = object_state[0, 2]

        zeros = torch.zeros_like(rand_yaw)
        cands_quat = quat_from_euler_xyz(zeros, zeros, rand_yaw)

        flat_ids = env_ids.repeat_interleave(BATCH_SIZE)
        flat_cols = self._check_collisions_vectorized(
            flat_ids,
            cands_pos.reshape(-1, 3),
            cands_quat.reshape(-1, 4),
            object_type=object_type
        ).reshape(num_envs, BATCH_SIZE)

        valid_indices = torch.argmax((~flat_cols).int(), dim=1)
        unsafe_mask = torch.gather(flat_cols, 1, valid_indices.unsqueeze(1)).squeeze(1)
        if unsafe_mask.any():
            print(f"⚠️ Shotgun Fail: {unsafe_mask.sum().item()} envs for {object_type}.")

        batch_idx = torch.arange(num_envs, device=device)
        object_state[:, :3] = cands_pos[batch_idx, valid_indices]
        object_state[:, 3:7] = cands_quat[batch_idx, valid_indices]
        return object_state