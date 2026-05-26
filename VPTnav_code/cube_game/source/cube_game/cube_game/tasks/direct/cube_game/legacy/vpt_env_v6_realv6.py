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

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, RayCaster, save_images_to_file, Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, sample_gaussian, quat_from_euler_xyz
from isaaclab.utils import math as math_utils

from .vpt_env_cfg_v5 import VPTEnvCfg


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
        self.storage_position = torch.tensor([150.0, 150.0, 0.0]) 
        self.active_vpt_indices = [None] * self.num_envs  # Track which 20 are active per env

        # Derived environment parameters
        self.center_to_boundary = torch.abs(
            torch.tensor(self.boundary_limits).view(-1)[0])

        # Verbosity and visibility thresholds
        self.verbose = 2
        self.goal_pixel_threshold = 100  # Minimum pixels for goal visibility
        self.camera_pixel_threshold = 1200  # Minimum pixels for camera visibility

        # Data collection parameters
        self.images_per_env = 20  # Number of images to collect per environment
        self.min_viewpoint_distance = 0.1  # Minimum distance between viewpoints (meters)
        self.save_camera_pov = True

        # Viewpoint and collection state variables
        self.valid_viewpoint_poses = [None] * self.num_envs
        self.selected_viewpoints_for_collection = [None] * self.num_envs
        self.current_collection_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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


        # File paths
        self.base_path = f"/home/arock3/Documents/data_v6"
        # self.base_path = "/media/data_cifs_lrs/projects/prj_robotics/VPTnav_v6"
        self.visibility_labels_json_path = f"{self.base_path}/visibility_labels.json"

        # Mode determination
        if self.config_file is not None and os.path.exists(self.config_file):
            self.mode = "testing"
        else:
            self.mode = "data_collection"
        
        self.total_envs_to_sim = 100
        self.slot_to_env_id = list(range(self.num_envs))
        self.next_env_id = self.num_envs
        self.completed_envs= set()
        self.slot_attempt_counts = [0] * self.num_envs
        self.max_attempts_per_slot = 20 * 50    # Full resets * Inner resets
        
        self._preallocate_visibility_labels()

    def close(self):
        super().close()

    def _preallocate_visibility_labels(self) -> None:
        """Pre-allocate visibility labels for all environments in 50/25/25 proportion."""
        total = self.total_envs_to_sim
        num_in_view = total // 2
        num_occluded = total // 4
        num_outside_fov = total - num_in_view - num_occluded
        
        # Create list of all labels
        all_labels = (["in_view"] * num_in_view +
                    ["occluded"] * num_occluded +
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

        self._distance_tiled_camera = TiledCamera(self.cfg.distance_tiled_camera)
        self.scene.sensors["distance_tiled_camera"] = self._distance_tiled_camera

        self._occlusion_camera = TiledCamera(self.cfg.occlusion_camera)
        self.scene.sensors["occlusion_camera"] = self._occlusion_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def check_batch_object_visibility(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Check object visibility for a batch of environments in parallel."""
        sem_imgs = self._tiled_camera.data.output["semantic_segmentation"][env_ids]
        r = sem_imgs[..., 0]
        g = sem_imgs[..., 1]
        b = sem_imgs[..., 2]

        red_mask = (r >= 0.95) & (g <= 0.05) & (b <= 0.05)
        green_mask = (r <= 0.05) & (g >= 0.95) & (b <= 0.05)

        red_counts = red_mask.sum(dim=(1, 2))
        green_counts = green_mask.sum(dim=(1, 2))

        goal_visible = red_counts >= self.goal_pixel_threshold
        camera_visible = green_counts >= self.camera_pixel_threshold

        return goal_visible, camera_visible

    def move_agent(self, actions, env_ids: Sequence[int] | None = None):
        if (env_ids is None):
            env_ids = self._agent._ALL_INDICES

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        device = self._agent.device
        num_envs = len(env_ids)
        max_velocity = 3.0

        theta = math.pi / 12
        half_theta = theta / 2
        left_rot_quat = torch.tensor([math.cos(half_theta), 0.0, 0.0, math.sin(half_theta)], device=device)
        right_rot_quat = torch.tensor([math.cos(half_theta), 0.0, 0.0, -math.sin(half_theta)], device=device)

        current_pos = self._agent.data.root_pos_w[env_ids].clone()
        current_quat = self._agent.data.root_quat_w[env_ids].clone()

        w = current_quat[:, 0]
        z = current_quat[:, 3]
        magnitude = torch.sqrt(w**2 + z**2)

        upright_quat = current_quat.clone()
        upright_quat[:, 0] = w / magnitude
        upright_quat[:, 1] = 0.0
        upright_quat[:, 2] = 0.0
        upright_quat[:, 3] = z / magnitude

        current_pos[:, 2] = self._agent.data.default_root_state[env_ids, 2]

        new_quat = upright_quat.clone()
        desired_vel = torch.zeros((num_envs, 6), device=device)

        action_6_mask = (actions == 6)

        for i in range(num_envs):
            if action_6_mask[i]:
                desired_vel[i, :] = 0.0
                continue

            action = actions[i]
            if action == 2:
                new_quat[i] = math_utils.quat_mul(upright_quat[i], left_rot_quat)
                desired_vel[i, :] = 0.0
            elif action == 3:
                new_quat[i] = math_utils.quat_mul(upright_quat[i], right_rot_quat)
                desired_vel[i, :] = 0.0
            elif action == 4:
                desired_vel[i, :] = 0.0
            elif action == 5:
                env_id_item = env_ids[i].item()
                if (self.valid_viewpoint_poses is not None
                        and env_id_item < len(self.valid_viewpoint_poses)
                        and self.valid_viewpoint_poses[env_id_item] is not None
                        and len(self.valid_viewpoint_poses[env_id_item]) > 0):

                    current_idx = self.viewpoint_pose_counter[env_id_item].item()
                    num_valid_poses = len(self.valid_viewpoint_poses[env_id_item])
                    pose_idx = current_idx % num_valid_poses
                    target_pos = self.valid_viewpoint_poses[env_id_item][pose_idx].to(device)
                    self.viewpoint_pose_counter[env_id_item] += 1

                    camera_pos_3d = self._camera_obj.data.root_pos_w[env_ids[i]]
                    goal_pos_3d = self._goal.data.root_pos_w[env_ids[i]]
                    midpoint = (camera_pos_3d + goal_pos_3d) / 2.0
                    direction = midpoint[:2] - target_pos[:2]
                    yaw = torch.atan2(direction[1], direction[0]) if torch.norm(direction) > 1e-6 else torch.tensor(0.0)
                    
                    new_quat[i] = torch.tensor([math.cos(yaw.item() / 2), 0.0, 0.0, math.sin(yaw.item() / 2)], device=device)
                    current_pos[i, :3] = target_pos
                desired_vel[i, :] = 0.0
            else:
                forward_input = 1.0 if action == 0 else -1.0
                local_movement = torch.tensor([forward_input, 0.0, 0.0], device=device)
                world_velocity = math_utils.quat_apply(upright_quat[i], local_movement) * max_velocity
                desired_vel[i, :3] = world_velocity
                desired_vel[i, 3:6] = 0.0

        # Action 6: Select viewpoints for collection (no movement, just selection)
        if action_6_mask.any():
            action_6_indices = action_6_mask.nonzero(as_tuple=True)[0]
            action_6_env_ids = env_ids[action_6_indices]

            for i, env_id_item in enumerate(action_6_env_ids.tolist()):
                if (self.valid_viewpoint_poses is not None 
                    and env_id_item < len(self.valid_viewpoint_poses)
                    and self.valid_viewpoint_poses[env_id_item] is not None 
                    and len(self.valid_viewpoint_poses[env_id_item]) >= self.images_per_env):
                    
                    all_viewpoints = self.valid_viewpoint_poses[env_id_item]
                    
                    selected_points = [all_viewpoints[0]]
                    
                    for point_idx in range(1, len(all_viewpoints)):
                        candidate = all_viewpoints[point_idx]
                        distances = torch.norm(
                            torch.stack(selected_points) - candidate.unsqueeze(0),
                            dim=1
                        )
                        
                        if torch.all(distances >= self.min_viewpoint_distance):
                            selected_points.append(candidate)
                            
                            if len(selected_points) == self.images_per_env:
                                break
                    
                    if len(selected_points) == self.images_per_env:
                        self.selected_viewpoints_for_collection[env_id_item] = torch.stack(selected_points)
                        if self.verbose >= 1:
                            print(f"✅ Env {env_id_item}: Selected {self.images_per_env} viewpoints for collection")
                    else:
                        if self.verbose >= 1:
                            print(f"⚠️  Env {env_id_item}: Only {len(selected_points)} viewpoints available (need {self.images_per_env})")
                else:
                    if self.verbose >= 1:
                        print(f"⚠️  Env {env_id_item}: No valid viewpoints available")

        self._agent.write_root_com_pose_to_sim(torch.cat([current_pos, new_quat], dim=1), env_ids)
        self._agent.write_root_com_velocity_to_sim(desired_vel, env_ids)
        self._agent.reset()

        camera_obj_pos = self._camera_obj.data.root_pos_w[env_ids].clone()
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_ids].clone()
        theta_left = math.pi / 2
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor([math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)], device=device)
        rotated_orientations = math_utils.quat_mul(camera_obj_quat, left_90_quat.expand(num_envs, -1))
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
            torch.zeros_like(self._camera_obj.data.root_vel_w[env_ids]), env_ids)

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
            self._goal.write_root_pose_to_sim(self._goal.data.default_root_state[env_ids], env_ids)
            self._agent.write_root_pose_to_sim(self._agent.data.default_root_state[env_ids], env_ids)
            self._vpt_objects.write_object_com_pose_to_sim(self._vpt_objects.data.default_root_state[env_ids], env_ids)
            
            
            if self.selected_viewpoints_for_collection[env_id_item] is not None:
                viewpoints = self.selected_viewpoints_for_collection[env_id_item]
                for viewpoint_idx in range(len(viewpoints)):
                    print(f"  📍 Testing viewpoint {viewpoint_idx + 1}/{len(viewpoints)}...")
                    target_pos = viewpoints[viewpoint_idx]
                    target_pos_3d = torch.zeros(3, device=self.device)
                    target_pos_3d[:2] = target_pos[:2]
                    target_pos_3d[2] = self._agent.data.default_root_state[env_ids[0], 2]
                    camera_pos_3d = self._camera_obj.data.root_pos_w[env_ids[0]]
                    goal_pos_3d = self._goal.data.root_pos_w[env_ids[0]]
                    midpoint = (camera_pos_3d + goal_pos_3d) / 2.0
                    direction = midpoint[:2] - target_pos[:2]
                    yaw = torch.atan2(direction[1], direction[0]) if torch.norm(direction) > 1e-6 else torch.tensor(0.0)
                    quat = torch.tensor([math.cos(yaw.item() / 2), 0.0, 0.0, math.sin(yaw.item() / 2)], device=self.device)
                    pose = torch.cat([target_pos_3d, quat])
                    self._agent.write_root_com_pose_to_sim(pose.unsqueeze(0), env_ids)
                    for _ in range(10):
                        self.sim.step()
                        self._rgb_tiled_camera.update(self.sim.cfg.dt)
                    rgb_data = self._rgb_tiled_camera.data.output["rgb"]
                    rgb_data = rgb_data.permute(0, 3, 1, 2)[:, :3, :, :]
                return {"policy": rgb_data.clone()}
            else:
                print(f"⚠️  Env {env_id_item}: No selected viewpoints for testing mode")
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
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = (self.episode_length_buf >= self.max_episode_length)
        return terminated, time_outs

    def _validate_env_state(self, env_id, folder_idx, min_viewpoints):
        """Validate single environment's state after reset attempt."""
        env_id_item = env_id.item()

        if (self.valid_viewpoint_poses is None or
                env_id_item >= len(self.valid_viewpoint_poses) or
                self.valid_viewpoint_poses[env_id_item] is None or
                len(self.valid_viewpoint_poses[env_id_item]) < min_viewpoints):
            num_poses = 0 if self.valid_viewpoint_poses is None or self.valid_viewpoint_poses[env_id_item] is None else len(self.valid_viewpoint_poses[env_id_item])
            return False, f"insufficient viewpoints: {num_poses}/{min_viewpoints}"

        visibility_reason = self.env_visibility_reasons.get(folder_idx, "unknown")
        if visibility_reason in ["in_view", "occluded"]:
            camera_pos = self._camera_obj.data.root_pos_w[env_id]
            goal_pos = self._goal.data.root_pos_w[env_id]
            is_occluded = self._check_occlusion_raycast(camera_pos, goal_pos, env_id)
            expected_occluded = (visibility_reason == "occluded")

            if is_occluded != expected_occluded:
                return False, f"occlusion mismatch: expected {'occluded' if expected_occluded else 'visible'}, got {'occluded' if is_occluded else 'visible'}"

        return True, ""

    def _perform_final_validation(self, env_ids, original_folder_indices):
        """Perform final comprehensive validation check on all environments."""
        validation_results = []
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()
            folder_idx = original_folder_indices[i]
            visibility_reason = self.env_visibility_reasons.get(folder_idx, "unknown")
            
            result = {
                "env_id": env_id_item, "folder_idx": folder_idx,
                "label": self.env_visibility_labels.get(folder_idx, "UNKNOWN"),
                "reason": visibility_reason, "valid": True
            }

            if visibility_reason in ["in_view", "occluded"]:
                camera_pos = self._camera_obj.data.root_pos_w[env_id]
                goal_pos = self._goal.data.root_pos_w[env_id]
                is_occluded = self._check_occlusion_raycast(camera_pos, goal_pos, env_id)
                expected_occluded = (visibility_reason == "occluded")
                
                result.update({
                    "expected_occluded": expected_occluded,
                    "actual_occluded": is_occluded,
                    "valid": is_occluded == expected_occluded
                })
            
            result["status"] = "✅" if result["valid"] else "❌"
            validation_results.append(result)
            
        return validation_results

    def _print_validation_summary(self, validation_results: list[dict], num_envs: int):
        """Print formatted and detailed summary of validation results."""
        num_valid = sum(1 for r in validation_results if r["valid"])

        print(f"\n{'='*80}")
        print(f"📊 VALIDATION SUMMARY: {num_valid}/{len(validation_results)} environments correct")
        print(f"{'='*80}")

        in_view_results = [r for r in validation_results if r["reason"] == "in_view"]
        occluded_results = [r for r in validation_results if r["reason"] == "occluded"]
        outside_fov_results = [r for r in validation_results if r["reason"] == "outside_fov"]

        if in_view_results:
            print(f"\n📍 IN_VIEW ({len(in_view_results)} envs - {len(in_view_results)/num_envs*100:.0f}%):")
            for r in in_view_results:
                print(f"  {r['status']} Env {r['env_id']} (folder {r['folder_idx']}): "
                    f"Label={r['label']}, Occluded={r.get('actual_occluded', 'N/A')}")

        if occluded_results:
            print(f"\n🚫 OCCLUDED ({len(occluded_results)} envs - {len(occluded_results)/num_envs*100:.0f}%):")
            for r in occluded_results:
                print(f"  {r['status']} Env {r['env_id']} (folder {r['folder_idx']}): "
                    f"Label={r['label']}, Occluded={r.get('actual_occluded', 'N/A')}")

        if outside_fov_results:
            print(f"\n👁️  OUTSIDE_FOV ({len(outside_fov_results)} envs - {len(outside_fov_results)/num_envs*100:.0f}%):")
            for r in outside_fov_results:
                print(f"  {r['status']} Env {r['env_id']} (folder {r['folder_idx']}): Label={r['label']}")

        num_invalid = len(validation_results) - num_valid
        if num_invalid > 0:
            print(f"\n{'='*80}")
            print(f"⚠️  WARNING: {num_invalid} environment(s) have mismatched occlusion status!")
            for r in validation_results:
                if not r["valid"]:
                    print(f"  ❌ Env {r['env_id']} (folder {r['folder_idx']}): "
                        f"Expected {'occluded' if r['expected_occluded'] else 'visible'}, "
                        f"got {'occluded' if r['actual_occluded'] else 'visible'}")
        
        print(f"\n{'='*80}\n")

    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)
        return obs, rewards, terminated, truncated, info

    def _check_occlusion_raycast(self, camera_pos, goal_pos, env_id, camera=None):
        if camera is None:
            camera = self._occlusion_camera

        camera_obj_pos = self._camera_obj.data.root_pos_w[env_id]
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_id]
        occlusion_cam_pos = camera.data.pos_w[env_id]
        
        pos_diff = torch.norm(camera_obj_pos - occlusion_cam_pos).item()
        
        if pos_diff > 0.01:
            print(f"⚠️  Env {env_id}: Occlusion camera misaligned! Distance: {pos_diff:.4f}")
            print(f"    Camera Obj: {camera_obj_pos.cpu().numpy()}")
            print(f"    Occlusion Cam: {occlusion_cam_pos.cpu().numpy()}")
            print(f"    Forcing update...")
            
            device = camera_obj_pos.device
            theta_left = math.pi / 2
            half_theta_left = theta_left / 2
            left_90_quat = torch.tensor(
                [math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)],
                device=device)
            
            rotated_orientation = math_utils.quat_mul(
                camera_obj_quat.unsqueeze(0),
                left_90_quat.unsqueeze(0)).squeeze(0)
            
            camera.set_world_poses(
                positions=camera_obj_pos.unsqueeze(0),
                orientations=rotated_orientation.unsqueeze(0),
                env_ids=[env_id.item() if torch.is_tensor(env_id) else env_id],
                convention="world")
            
            
            new_occlusion_cam_pos = camera.data.pos_w[env_id]
            new_pos_diff = torch.norm(camera_obj_pos - new_occlusion_cam_pos).item()
            print(f"    ✓ After fix: Distance = {new_pos_diff:.4f}")

        GOAL_THRESHOLD = self.goal_pixel_threshold
        sem_img = camera.data.output["semantic_segmentation"][env_id]
        r = sem_img[:, :, 0]
        g = sem_img[:, :, 1]
        b = sem_img[:, :, 2]

        red_mask = ((r >= 0.95) & (g <= 0.05) & (b <= 0.05))
        red_count = red_mask.sum().item()

        return red_count < GOAL_THRESHOLD

    def check_object_visibility(self, env_id: int) -> tuple[bool, bool]:

        sem_img = self._tiled_camera.data.output["semantic_segmentation"][env_id]
        r = sem_img[:, :, 0]
        g = sem_img[:, :, 1]
        b = sem_img[:, :, 2]

        red_mask = ((r >= 0.95) & (g <= 0.05) & (b <= 0.05))
        red_count = red_mask.sum().item()

        green_mask = ((r <= 0.05) & (g >= 0.95) & (b <= 0.05))
        green_count = green_mask.sum().item()

        goal_visible = red_count >= self.goal_pixel_threshold
        camera_visible = green_count >= self.camera_pixel_threshold

        return goal_visible, camera_visible

    def _save_visibility_labels(self):
        """Save visibility labels to JSON file."""
        import json

        os.makedirs(os.path.dirname(self.visibility_labels_json_path), exist_ok=True)

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
                "total_environments": len(self.env_visibility_labels),
                "yes_count": sum(1 for v in self.env_visibility_labels.values() if v == "Yes"),
                "no_count": sum(1 for v in self.env_visibility_labels.values() if v == "No"),
                "by_reason": reason_counts,
                "next_env_folder_idx": self.next_env_folder_idx
            }
        }

        with open(self.visibility_labels_json_path, 'w') as f:
            json.dump(labels_data, f, indent=2)
    
    def _check_target_in_img(self, file_name: str, cam_pov, return_red_count: bool = False) -> bool:
        cv2.imwrite(file_name, cv2.cvtColor(cam_pov, cv2.COLOR_RGB2BGR))

        # Load image back and count red pixels
        loaded_img = cv2.imread(file_name)
        loaded_img_rgb = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)

        r = loaded_img_rgb[:, :, 0]
        g = loaded_img_rgb[:, :, 1]
        b = loaded_img_rgb[:, :, 2]
        max_r = 0.95 * 255
        min_gb = 0.05 * 255

        red_mask = ((r >= max_r) & (g <= min_gb) & (b <= min_gb))  # 0.95*255=242, 0.05*255=13
        red_count = red_mask.sum().item()

        target_visible_in_camera = red_count >= self.goal_pixel_threshold

        # Check for solid filled circles using HoughCircles
        gray = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=100,
            param2=12,
            minRadius=5,
            maxRadius=200
        )
        
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
                "disable_gravity": bool(self.cfg.goal_ball.spawn.rigid_props.disable_gravity)
            },
            "mass_props": {
                "mass": float(self.cfg.goal_ball.spawn.mass_props.mass)
            },
            "visual_material": {
                "diffuse_color": list(self.cfg.goal_ball.spawn.visual_material.diffuse_color)
            }
        }
        
        # Get camera object info (position + cfg spawn parameters)
        camera_pos = self._camera_obj.data.root_pos_w[env_id].cpu().numpy().tolist()
        camera_quat = self._camera_obj.data.root_quat_w[env_id].cpu().numpy().tolist()
        camera_spawn_cfg = {
            "rigid_props": {
                "disable_gravity": bool(self.cfg.camera_obj.spawn.rigid_props.disable_gravity)
            },
            "mass_props": {
                "mass": float(self.cfg.camera_obj.spawn.mass_props.mass)
            },
            "visual_material": {
                "diffuse_color": list(self.cfg.camera_obj.spawn.visual_material.diffuse_color)
            }
        }
        
        # Get agent info (position + cfg spawn parameters)
        agent_pos = self._agent.data.root_pos_w[env_id].cpu().numpy().tolist()
        agent_quat = self._agent.data.root_quat_w[env_id].cpu().numpy().tolist()
        agent_spawn_cfg = {
            "size": list(self.cfg.agent.spawn.size),
            "rigid_props": {
                "disable_gravity": bool(self.cfg.agent.spawn.rigid_props.disable_gravity)
            },
            "mass_props": {
                "mass": float(self.cfg.agent.spawn.mass_props.mass)
            },
            "visual_material": {
                "diffuse_color": list(self.cfg.agent.spawn.visual_material.diffuse_color)
            }
        }
        
        # ========== CHANGE #1: Get active VPT indices ==========
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        active_indices = self.active_vpt_indices[env_id_item]
        active_indices_list = active_indices.cpu().numpy().tolist()
        # =======================================================
        
        # ========== CHANGE #2: Get VPT objects info - ONLY ACTIVE ONES ==========
        vpt_positions = self._vpt_objects.data.object_pos_w[env_id].cpu().numpy().tolist()
        vpt_orientations = self._vpt_objects.data.object_quat_w[env_id].cpu().numpy().tolist()
        # =========================================================================
        
        # Get valid viewpoint poses
        valid_viewpoints = []
        if (self.valid_viewpoint_poses is not None 
            and env_id_item < len(self.valid_viewpoint_poses)
            and self.valid_viewpoint_poses[env_id_item] is not None):
            valid_viewpoints = self.valid_viewpoint_poses[env_id_item].cpu().numpy().tolist()
        
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
            "count": len(self.selected_viewpoints_for_collection[env_id_item]) if self.selected_viewpoints_for_collection[env_id_item] is not None else 0,
            "positions": self.selected_viewpoints_for_collection[env_id_item].cpu().numpy().tolist() if self.selected_viewpoints_for_collection[env_id_item] is not None else []
            }
        }
        
        # ========== CHANGE #2: Add only active VPT objects ==========
        # Add each ACTIVE VPT object with cfg metadata extracted at runtime
        for local_idx, obj_idx in enumerate(active_indices):
            obj_idx_item = obj_idx.item() if torch.is_tensor(obj_idx) else obj_idx
            
            # Extract spawn cfg directly from the VPT object's configuration
            vpt_spawn_cfg = self.cfg.vpt_objects.rigid_objects[list(self.cfg.vpt_objects.rigid_objects.keys())[obj_idx_item]].spawn
            
            # Get rigid props
            rigid_props = {}
            if hasattr(vpt_spawn_cfg, 'rigid_props'):
                rigid_props['disable_gravity'] = bool(vpt_spawn_cfg.rigid_props.disable_gravity)
            
            # Get mass props
            mass_props = {}
            if hasattr(vpt_spawn_cfg, 'mass_props'):
                mass_props['mass'] = float(vpt_spawn_cfg.mass_props.mass)
            
            # Get visual material
            visual_material = {}
            if hasattr(vpt_spawn_cfg, 'visual_material'):
                visual_material['diffuse_color'] = list(vpt_spawn_cfg.visual_material.diffuse_color)
            
            # Get size/dimensions based on spawn type
            size_info = {}
            if hasattr(vpt_spawn_cfg, 'size'):
                size_info['size'] = list(vpt_spawn_cfg.size)
            elif hasattr(vpt_spawn_cfg, 'radius'):
                size_info['radius'] = float(vpt_spawn_cfg.radius)
            elif hasattr(vpt_spawn_cfg, 'height') and hasattr(vpt_spawn_cfg, 'radius'):
                size_info['height'] = float(vpt_spawn_cfg.height)
                size_info['radius'] = float(vpt_spawn_cfg.radius)
            
            vpt_obj = {
            "index": obj_idx_item,
            "position": vpt_positions[obj_idx_item],
            "orientation": vpt_orientations[obj_idx_item],
            "spawn_cfg": {
                **size_info,
                "rigid_props": rigid_props,
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
            print(f"     Active VPT objects: {self.active_vpt_objs}/{self.num_objs}")

    def _load_env_config_from_json(self, config_filepath: str, target_env_id: int = 0):
        """
        Load environment configuration from JSON file and apply to specified environment.
        Restores positions, orientations, and validates spawn cfg matches current cfg.
        
        Args:
            config_filepath: Path to JSON configuration file
            target_env_id: Environment ID to load configuration into
        """
        import json
        
        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f"Config file not found: {config_filepath}")
        
        # Load configuration
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        
        device = self._agent.device
        
        # Convert target_env_id to tensor if needed
        if isinstance(target_env_id, int):
            env_ids = torch.tensor([target_env_id], dtype=torch.long, device=device)
        else:
            env_ids = torch.tensor([target_env_id.item()], dtype=torch.long, device=device)
        
        env_id_item = target_env_id if isinstance(target_env_id, int) else target_env_id.item()
        
        # Validate environment settings match current cfg
        env_settings = config.get("environment_settings", {})
        if env_settings:
            if env_settings.get("num_vpt_objs") != self.num_objs:
                print(f"⚠️  Warning: Config has {env_settings.get('num_vpt_objs')} VPT objects, "
                    f"but current cfg has {self.num_objs}")
        
        # Extract configuration data
        goal_pos = torch.tensor(config["goal_ball"]["position"], device=device, dtype=torch.float32)
        goal_quat = torch.tensor(config["goal_ball"]["orientation"], device=device, dtype=torch.float32)
        
        camera_pos = torch.tensor(config["camera_object"]["position"], device=device, dtype=torch.float32)
        camera_quat = torch.tensor(config["camera_object"]["orientation"], device=device, dtype=torch.float32)
        
        agent_pos = torch.tensor(config["agent"]["position"], device=device, dtype=torch.float32)
        agent_quat = torch.tensor(config["agent"]["orientation"], device=device, dtype=torch.float32)
        
        # ========== CHANGE #3 & #4: Load active indices and restore them ==========
        active_indices = torch.tensor(config["vpt_objects"]["active_indices"], 
                                    dtype=torch.long, device=device)
        self.active_vpt_indices[env_id_item] = active_indices
        # ==========================================================================
        
        # Build VPT object states - initialize all to zeros
        vpt_count = config["vpt_objects"]["total_count"]
        vpt_positions_full = torch.zeros((vpt_count, 3), device=device, dtype=torch.float32)
        vpt_orientations_full = torch.zeros((vpt_count, 4), device=device, dtype=torch.float32)
        vpt_orientations_full[:, 0] = 1.0  # Default quaternion (w=1, x=y=z=0)
        vpt_colors_full = torch.zeros((vpt_count, 3), device=device, dtype=torch.float32)
        
        # ========== CHANGE #5: Set inactive objects to storage position ==========
        # First, set all objects to storage position
        vpt_positions_full[:, :] = self.storage_position
        # =========================================================================
        
        # Now load ACTIVE object data from config
        for obj_data in config["vpt_objects"]["objects"]:
            obj_idx = obj_data["index"]
            vpt_positions_full[obj_idx] = torch.tensor(obj_data["position"], device=device, dtype=torch.float32)
            vpt_orientations_full[obj_idx] = torch.tensor(obj_data["orientation"], device=device, dtype=torch.float32)
            vpt_colors_full[obj_idx] = torch.tensor(obj_data["spawn_cfg"]["visual_material"]["diffuse_color"], 
                                                    device=device, dtype=torch.float32)
        
        # Apply goal ball configuration
        goal_full_state = torch.cat([goal_pos, goal_quat, torch.zeros(6, device=device)], dim=-1)
        self._goal.data.default_root_state[env_ids] = goal_full_state.unsqueeze(0)
        self._goal.write_data_to_sim()
        
        # Apply camera object configuration
        camera_pose = torch.cat([camera_pos.unsqueeze(0), camera_quat.unsqueeze(0), torch.zeros((1, 6), device=device)], dim=1)
        self._camera_obj.data.default_root_state[env_ids] = camera_pose
        self._camera_obj.write_data_to_sim()
        
        # Apply agent configuration
        agent_pose = torch.cat([agent_pos.unsqueeze(0), agent_quat.unsqueeze(0), torch.zeros((1, 6), device=device)], dim=1)
        self._agent.data.default_root_state[env_ids] = agent_pose
        self._agent.write_data_to_sim()
        
        # Apply VPT objects configuration (all 200: 20 active + 180 in storage)
        vpt_poses = torch.cat([vpt_positions_full.unsqueeze(0), vpt_orientations_full.unsqueeze(0), 
                            torch.zeros((1, vpt_count, 6), device=device)], dim=2)
        self._vpt_objects.data.default_object_state[env_ids] = vpt_poses
        self._vpt_objects.write_data_to_sim()
        
        # Apply colors to all VPT objects
        for obj_idx, obj_key in enumerate(self.cfg.vpt_objects.rigid_objects.keys()):
            color = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(vpt_colors_full[obj_idx].cpu().numpy()))
            self._vpt_objects.cfg.rigid_objects[obj_key].spawn.visual_material = color
            self._vpt_objects.write_data_to_sim()
        
        # Update occlusion camera to match camera object
        theta_left = math.pi / 2
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)],
            device=device)
        
        rotated_orientation = math_utils.quat_mul(
            camera_quat.unsqueeze(0),
            left_90_quat.unsqueeze(0))
        
        self._occlusion_camera.set_world_poses(
            positions=camera_pos.unsqueeze(0),
            orientations=rotated_orientation,
            env_ids=env_ids.tolist(),
            convention="world")
        
        # Load valid viewpoints
        if "valid_viewpoints" in config and config["valid_viewpoints"]["count"] > 0:
            valid_viewpoints = torch.tensor(config["valid_viewpoints"]["positions"], 
                                        device=device, dtype=torch.float32)
            
            if self.valid_viewpoint_poses is None:
                self.valid_viewpoint_poses = [None] * self.num_envs
            
            self.valid_viewpoint_poses[env_id_item] = valid_viewpoints
        
        # Load collected viewpoints if present
        if "collected_viewpoints" in config and config["collected_viewpoints"]["count"] > 0:
            collected_viewpoints = torch.tensor(config["collected_viewpoints"]["positions"],
                                            device=device, dtype=torch.float32)
            self.selected_viewpoints_for_collection[env_id_item] = collected_viewpoints
        
        # Update visibility labels
        folder_idx = config["metadata"]["folder_idx"]
        self.env_visibility_labels[folder_idx] = config["metadata"]["visibility_label"]
        self.env_visibility_reasons[folder_idx] = config["metadata"]["visibility_reason"]
        
        # Simulate a few steps to stabilize
        for _ in range(3):
            self.sim.step()
        
        if self.verbose >= 1:
            print(f"✅ Loaded environment configuration from: {config_filepath}")
            print(f"   → Env {target_env_id}, Folder {folder_idx}, Label: {config['metadata']['visibility_label']}")
            print(f"   → Reason: {config['metadata']['visibility_reason']}")
            print(f"   → Valid Viewpoints: {config['valid_viewpoints']['count']}")
            print(f"   → Active VPT objects: {len(active_indices)}/{vpt_count}")

        self.mode = "testing"

    def _check_target_vpt_distance(self, env_idx, target_position, vpt_positions, min_distance=1.0):
        """Check if VPT object surfaces are at least min_distance away from target position."""
        vpt_dims = self._get_vpt_dims(env_idx)
        
        # Get VPT center positions in 3D (positions are at bottom center, so add half height)
        vpt_positions_3d = torch.zeros((self.num_objs, 3), device=self._agent.device)
        vpt_positions_3d[:, :2] = vpt_positions
        vpt_positions_3d[:, 2] = vpt_dims[:, 2] / 2.0  # Add half Z-dimension to get center
        
        # Get target position in 3D
        goal_radius = self.goal_radius + 0.01  # (0.15 radius + 1 cm buffer)
        target_position_3d = torch.zeros((3), device=self._agent.device)
        target_position_3d[:2] = target_position[:2]
        target_position_3d[2] = goal_radius
        
        # Compute closest point on each VPT cuboid surface to the target
        target_expanded = target_position_3d.unsqueeze(0).expand_as(vpt_positions_3d)
        
        # Half-extents of the cuboids
        half_extents = vpt_dims / 2.0
        
        # Vector from VPT centers to target
        diff = target_expanded - vpt_positions_3d
        
        # Clamp to cuboid surface (closest point on cuboid to target)
        closest_point_local = torch.clamp(diff, -half_extents, half_extents)
        
        # Get closest point in world coordinates
        closest_point_world = vpt_positions_3d + closest_point_local
        
        # Distance from cuboid surface to target (accounting for target radius)
        dists = torch.norm(target_expanded - closest_point_world, dim=1)
        
        return torch.all(dists >= min_distance).item()

    def _select_active_vpt_indices(self, env_ids: torch.Tensor) -> None:
        """Randomly select 20 active VPT object indices for each environment.
        
        Args:
            env_ids: Tensor of environment IDs to select active indices for
        """
        device = self._agent.device
        
        for env_id in env_ids:
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            
            # Randomly select 20 indices from 0-199
            active_indices = torch.randperm(self.num_objs, device=device)[:self.active_vpt_objs]
            
            # Store for this environment
            self.active_vpt_indices[env_id_item] = active_indices
            
            if self.verbose >= 2:
                print(f"  🎲 Env {env_id_item}: Selected {self.active_vpt_objs} active VPT indices from {self.num_objs} total")
    
    def _store_inactive_vpt_objects(self, env_ids: torch.Tensor, vpt_obj_default_state: torch.Tensor) -> torch.Tensor:
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
            inactive_indices = torch.tensor(list(all_indices - active_indices_set), 
                                        dtype=torch.long, device=device)
            
            # Move all inactive objects to storage position
            for inactive_idx in inactive_indices:
                vpt_obj_default_state[i, inactive_idx, 0] = self.storage_position[0]
                vpt_obj_default_state[i, inactive_idx, 1] = self.storage_position[1]
                vpt_obj_default_state[i, inactive_idx, 2] = self.storage_position[2]
                
                # Zero out velocities
                vpt_obj_default_state[i, inactive_idx, 7:13] = 0.0
            
            # if self.verbose >= 2:
            #     print(f"  📦 Env {env_id_item}: Stored {len(inactive_indices)} inactive VPT objects at {self.storage_position.cpu().numpy()}")
    
        return vpt_obj_default_state

    def _get_active_vpt_dims(self, env_id) -> torch.Tensor:
        """Get dimensions of only the 20 active VPT objects for given environment.
        
        Args:
            env_id: Environment ID (can be int or tensor)
            
        Returns:
            Tensor of shape (active_vpt_objs, 3) containing X, Y, Z dimensions
        """
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        device = self._agent.device
        
        # Get active indices for this env
        active_indices = self.active_vpt_indices[env_id_item]
        
        # Allocate dims tensor for active objects only
        dims = torch.zeros((self.active_vpt_objs, 3), device=device)
        
        # Get dimensions for each active object
        vpt_keys = list(self.cfg.vpt_objects.rigid_objects.keys())
        for local_idx, obj_idx in enumerate(active_indices):
            vpt_size = self._vpt_objects.cfg.rigid_objects[vpt_keys[obj_idx]].spawn.size
            dims[local_idx] = torch.tensor(vpt_size, device=device)
        
        return dims

    def _get_active_vpt_positions(self, env_id) -> torch.Tensor:
        """Get positions of only the 20 active VPT objects for given environment.
        
        Args:
            env_id: Environment ID (can be int or tensor)
            
        Returns:
            Tensor of shape (active_vpt_objs, 3) containing 3D positions
        """
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        
        # Get active indices for this env
        active_indices = self.active_vpt_indices[env_id_item]
        
        # Get positions of active objects only
        active_positions = self._vpt_objects.data.object_pos_w[env_id, active_indices]
        
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
            or self.valid_viewpoint_poses[env_id] is None 
            or len(self.valid_viewpoint_poses[env_id]) < self.images_per_env):
            return False
        
        all_viewpoints = self.valid_viewpoint_poses[env_id]
        selected_points = [all_viewpoints[0]]
        
        for point_idx in range(1, len(all_viewpoints)):
            candidate = all_viewpoints[point_idx]
            distances = torch.norm(
                torch.stack(selected_points) - candidate.unsqueeze(0),
                dim=1
            )
            
            if torch.all(distances >= self.min_viewpoint_distance):
                selected_points.append(candidate)
                
                if len(selected_points) == self.images_per_env:
                    break
        
        if len(selected_points) == self.images_per_env:
            self.selected_viewpoints_for_collection[env_id] = torch.stack(selected_points)
            if self.verbose >= 2:
                print(f"    ✅ Slot {env_id}: Selected {self.images_per_env} viewpoints for collection")
            return True
        else:
            if self.verbose >= 2:
                print(f"    ⚠️  Slot {env_id}: Only {len(selected_points)} viewpoints available (need {self.images_per_env})")
            return False

    def _reset_idx(self, env_ids: Sequence[int] | None, randomize_objects: bool = True) -> None:
        """Reset environments by selectively retrying only the failed ones."""
        # check for config in kwargs
        if self.mode == "testing":
            config_filepath = self.config_file
            if config_filepath is not None:
                if env_ids is None:
                    env_ids = self._agent._ALL_INDICES
                for env_id in env_ids:
                    self._load_env_config_from_json(config_filepath, env_id)
                self._reset_called = True
                return

        MIN_VALID_VIEWPOINTS = self.images_per_env

        if env_ids is None:
            env_ids = self._agent._ALL_INDICES
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_slots = len(env_ids)
        # Prepare initial batch: assign folder indices and visibility labels
        initial_folder_indices = [self.next_env_folder_idx + i for i in range(num_slots)]

        if self.verbose >= 1:
            print("🔒 Assigning visibility labels from pre-allocated pool...")
        
        visibility_categories = []
        for i in range(num_slots):
            folder_idx = initial_folder_indices[i]
            category = self._assign_next_visibility_label(folder_idx)
            visibility_categories.append(category)
        
        self._save_visibility_labels()
        
        # Randomly select 25% of envs to have the goal on top of an object
        num_to_select = max(1, int(0.25 * len(env_ids)))
        selected_indices = torch.randperm(len(env_ids), device=self.device)[:num_to_select]
        self.envs_to_move_ball = env_ids[selected_indices]
        
        # Track folder indices and visibility per slot
        slot_folder_indices = initial_folder_indices.copy()
        slot_visibility_categories = visibility_categories.copy()
        
        # Main streaming loop
        iteration = 0
        while self.next_env_id < self.total_envs_to_sim or len(self.completed_envs) < self.total_envs_to_sim:
            iteration += 1
            
            if self.verbose >= 1:
                print(f"\n{'='*60}")
                print(f"🔄 Iteration {iteration} | Envs in slots: {self.slot_to_env_id}")
                print(f"{'='*60}")
            
            # 1. Reset ALL active slots (failed from previous iteration + new envs)
            reset_env_ids = []
            reset_folder_indices = []
            reset_visibility_categories = []
            
            for slot_idx in range(num_slots):
                env_id = self.slot_to_env_id[slot_idx]
                
                # Skip if already completed
                if env_id in self.completed_envs:
                    continue
                
                reset_env_ids.append(env_ids[slot_idx].item())
                reset_folder_indices.append(slot_folder_indices[slot_idx])
                reset_visibility_categories.append(slot_visibility_categories[slot_idx])
            
            if reset_env_ids:
                if self.verbose >= 1:
                    print(f"🔄 Resetting {len(reset_env_ids)} active slot(s)...")
                
                self._reset_idx_internal(
                    torch.tensor(reset_env_ids, dtype=torch.long, device=self.device),
                    randomize_objects,
                    folder_indices=reset_folder_indices,
                    visibility_categories=reset_visibility_categories
                )
                
                # Step simulation
                for _ in range(5):
                    self.scene.write_data_to_sim()
                    self.sim.step(render=False)
                    self.scene.update(dt=self.step_dt)
            
            # 2. Validate each slot
            valid_slots = []
            failed_slots = []
            exceeded_slots = []
            
            for slot_idx in range(num_slots):
                env_id = self.slot_to_env_id[slot_idx]
                
                # Skip if already completed
                if env_id in self.completed_envs:
                    continue
                
                folder_idx = slot_folder_indices[slot_idx]
                is_valid, reason = self._validate_env_state(env_ids[slot_idx], folder_idx, MIN_VALID_VIEWPOINTS)
                
                if is_valid:
                    valid_slots.append(slot_idx)
                    if self.verbose >= 1:
                        print(f"  ✅ Slot {slot_idx} | Env {env_id} | Folder {folder_idx} VALIDATED")
                else:
                    self.slot_attempt_counts[slot_idx] += 1
                    
                    if self.slot_attempt_counts[slot_idx] >= self.max_attempts_per_slot:
                        exceeded_slots.append(slot_idx)
                        if self.verbose >= 1:
                            print(f"  ⚠️ Slot {slot_idx} | Env {env_id} | EXCEEDED max attempts ({self.max_attempts_per_slot}), giving up")
                    else:
                        failed_slots.append(slot_idx)
                        if self.verbose >= 2:
                            print(f"  ❌ Slot {slot_idx} | Env {env_id} | Attempt {self.slot_attempt_counts[slot_idx]}/{self.max_attempts_per_slot}: {reason}")
            
            # 3. Collect images for valid slots
            if valid_slots:
                if self.verbose >= 1:
                    print(f"\n📸 Collecting images for {len(valid_slots)} valid slot(s): {valid_slots}")
                
                for slot_idx in valid_slots:
                    env_id = self.slot_to_env_id[slot_idx]
                    folder_idx = slot_folder_indices[slot_idx]
                    
                    # Select viewpoints for this slot
                    if not self._select_viewpoints_for_collection(slot_idx):
                        if self.verbose >= 1:
                            print(f"    ⚠️  Failed to select viewpoints for slot {slot_idx}")
                        continue
                    
                    # Collect images for this slot
                    self._collect_images_for_slot(env_ids[slot_idx], folder_idx)
                    
                    # Mark as completed
                    self.completed_envs.add(env_id)
            
            # 4. Replace valid and exceeded slots with new envs
            slots_to_replace = valid_slots + exceeded_slots
            
            if slots_to_replace and self.next_env_id < self.total_envs_to_sim:
                for slot_idx in slots_to_replace:
                    if self.next_env_id >= self.total_envs_to_sim:
                        break
                    
                    old_env_id = self.slot_to_env_id[slot_idx]
                    new_env_id = self.next_env_id
                    new_folder_idx = self.next_env_folder_idx + new_env_id
                    
                    # Assign visibility label for new env
                    new_visibility = self._assign_next_visibility_label(new_folder_idx)
                    
                    # Update tracking (reset happens next iteration)
                    self.slot_to_env_id[slot_idx] = new_env_id
                    self.slot_attempt_counts[slot_idx] = 0
                    slot_folder_indices[slot_idx] = new_folder_idx
                    slot_visibility_categories[slot_idx] = new_visibility
                    
                    self.next_env_id += 1
                    
                    if self.verbose >= 1:
                        print(f"  🔄 Slot {slot_idx}: Replacing env {old_env_id} → env {new_env_id} (folder {new_folder_idx})")
                
                self._save_visibility_labels()
        
            # 5. Check if done
            if len(self.completed_envs) >= self.total_envs_to_sim:
                if self.verbose >= 1:
                    print(f"\n🎉 SUCCESS: Completed all {self.total_envs_to_sim} environments!")
                break
            
            # Progress update
            if self.verbose >= 1:
                print(f"\n⏳ Progress | Completed: {len(self.completed_envs)}/{self.total_envs_to_sim} | Next env: {self.next_env_id}")
            
        
        # Final validation steps
        for _ in range(5):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.step_dt)

        self._reset_called = True

    def _reset_idx_internal(self, env_ids: Sequence[int] | None, randomize_objects: bool = True,
                            folder_indices: List[int] = None, visibility_categories: List[str] = None) -> None:
        """Internal reset logic - spawn objects and generate viewpoints."""
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_envs = len(env_ids)
        if folder_indices is None:
            folder_indices = [self.next_env_folder_idx + i for i in range(num_envs)]

        if visibility_categories is None:
            raise RuntimeError("visibility_categories must be provided to _reset_idx_internal!")

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            self.used_viewpoint_indices[env_id_item].clear()

        for i in range(num_envs):
            global_folder_idx = folder_indices[i]
            if global_folder_idx not in self.env_visibility_labels:
                raise RuntimeError(f"Labels not set for folder {global_folder_idx} before _reset_idx_internal!")

        # ========== CHANGE #1: Call _select_active_vpt_indices ==========
        self._select_active_vpt_indices(env_ids)
        # ================================================================

        self.viewpoint_pose_counter[env_ids] = 0
        super()._reset_idx(env_ids)

        device = self._agent.device
        safe_x_range = self.center_to_boundary - 4.0
        safe_x_range_obstacles = self.center_to_boundary - 2.5

        goal_default_state = self._goal.data.default_root_state[env_ids].clone()
        agent_default_state = self._agent.data.default_root_state[env_ids].clone()
        camera_obj_default_state = self._camera_obj.data.default_root_state[env_ids].clone()
        vpt_obj_default_state = self._vpt_objects.data.default_object_state[env_ids].clone()

        max_spawn_attempts = 20
        envs_need_spawn_retry = torch.ones(num_envs, dtype=torch.bool, device=device)
        
        valid_indices = torch.where(envs_need_spawn_retry)[0]
        in_view_indices = [idx for idx in valid_indices if visibility_categories[idx] == "in_view"]
        random_indices = torch.randperm(len(in_view_indices))[:len(in_view_indices) // 2]
        in_view_displaced = torch.tensor(in_view_indices, device=device)[random_indices]
        outside_fov_indices = [idx for idx in valid_indices if visibility_categories[idx] == "outside_fov"]
        random_indices = torch.randperm(len(outside_fov_indices))[:len(outside_fov_indices) // 4]
        outside_fov_displaced = torch.tensor(outside_fov_indices, device=device)[random_indices]
        
        
        for spawn_attempt in range(max_spawn_attempts):
            if not envs_need_spawn_retry.any():
                break
            
            retry_mask = envs_need_spawn_retry.clone()
            batch_size = retry_mask.sum().item()
            retry_indices = torch.where(retry_mask)[0]
            
            goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
            camera_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
            agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
            vpt_offsets = sample_uniform(-safe_x_range_obstacles, safe_x_range_obstacles, (batch_size, self.active_vpt_objs, 2), device)
            # vpt_offsets_valid = False
            # while not vpt_offsets_valid:
            #     vpt_offsets = sample_uniform(-safe_x_range_obstacles, safe_x_range_obstacles, (batch_size, self.active_vpt_objs, 2), device)
            #     # Ensure all VPT objects are at least 1.0 units apart from each other
            #     vpt_offsets_flat = vpt_offsets.reshape(batch_size * self.active_vpt_objs, 2)
            #     dists = torch.cdist(vpt_offsets_flat, vpt_offsets_flat)
            #     # Ignore self-distances by filling diagonal with a large value
            #     dists.fill_diagonal_(float('inf'))
            #     min_dist = dists.min().item()
            #     vpt_offsets_valid = min_dist >= 0.5

            goal_perturb_offsets = sample_uniform(-0.4, 0.4, (batch_size, 2), device)
            
            # Parallelized version of the spawn logic for all retry_indices at once
            env_ids_batch = env_ids[retry_indices]
            env_origins = self.scene.env_origins[env_ids_batch]  # (batch_size, 3)

            # Goal positions
            goal_default_state[retry_indices, 0] = env_origins[:, 0] + goal_offsets[:, 0]
            goal_default_state[retry_indices, 1] = env_origins[:, 1] + goal_offsets[:, 1]
            # goal_default_state[retry_indices, 2] = env_origins[:, 2] + goal_height_offset[:, 0]
            goal_default_state[retry_indices, 2] = env_origins[:, 2]

            # Camera positions
            camera_obj_default_state[retry_indices, 0] = env_origins[:, 0] + camera_offsets[:, 0]
            camera_obj_default_state[retry_indices, 1] = env_origins[:, 1] + camera_offsets[:, 1]

            # Camera orientation (look at goal, with pitch)
            direction_to_goal = goal_default_state[retry_indices, :2] - camera_obj_default_state[retry_indices, :2]
            yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0]) - math.radians(90)
            roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
            zero = torch.zeros_like(yaw)
            quaternion = quat_from_euler_xyz(roll, zero, yaw)
            camera_obj_default_state[retry_indices, 3:7] = quaternion

            # Add small random perturbation to goal position after camera is set
            goal_default_state[retry_indices, 0] += goal_perturb_offsets[:, 0]
            goal_default_state[retry_indices, 1] += goal_perturb_offsets[:, 1]

            # Agent positions
            agent_default_state[retry_indices, 0] = env_origins[:, 0] + agent_offsets[:, 0]
            agent_default_state[retry_indices, 1] = env_origins[:, 1] + agent_offsets[:, 1]

            # ========== CHANGE #4: Loop over active_indices instead of all objects ==========
            # VPT object positions - only for active objects
            for batch_idx, env_idx in enumerate(retry_indices):
                env_id = env_ids[env_idx]
                env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
                active_indices = self.active_vpt_indices[env_id_item]
                
                for local_idx, obj_idx in enumerate(active_indices):
                    vpt_obj_default_state[env_idx, obj_idx, 0] = env_origins[batch_idx, 0] + vpt_offsets[batch_idx, local_idx, 0]
                    vpt_obj_default_state[env_idx, obj_idx, 1] = env_origins[batch_idx, 1] + vpt_offsets[batch_idx, local_idx, 1]
                    vpt_obj_default_state[env_idx, obj_idx, 2] = self._vpt_objects.data.default_object_state[env_id, obj_idx, 2] + env_origins[batch_idx, 2]
            # =================================================================================
            
            # ========== CHANGE #5: Call _store_inactive_vpt_objects ==========
            vpt_obj_default_state[retry_indices] = self._store_inactive_vpt_objects(env_ids[retry_indices], vpt_obj_default_state[retry_indices])
            # =================================================================
            
            
            for batch_idx, env_idx in enumerate(retry_indices):
                agent_pos = agent_default_state[env_idx, :2]
                goal_pos = goal_default_state[env_idx, :2]
                camera_pos = camera_obj_default_state[env_idx, :2]
                
                camera_goal_distance = torch.norm(camera_pos - goal_pos)
                
                if camera_goal_distance > 15.0 or camera_goal_distance < 2.0:
                    continue
                
                # vpt_positions = vpt_obj_default_state[env_idx, :, :2]
                # camera_distances_from_vpt = torch.norm(camera_pos.unsqueeze(0) - vpt_positions, dim=1)
                
                # ========== CHANGE #6: Only check distances for active VPT objects ==========
                env_id = env_ids[env_idx]
                env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
                active_indices = self.active_vpt_indices[env_id_item]
                vpt_positions = vpt_obj_default_state[env_idx, active_indices, :2]
                camera_distances_from_vpt = torch.norm(camera_pos.unsqueeze(0) - vpt_positions, dim=1)
                # ============================================================================
                
                if not torch.all(camera_distances_from_vpt >= 2.0):
                    continue
                
                envs_need_spawn_retry[env_idx] = False
            
            valid_mask = retry_mask & ~envs_need_spawn_retry
            if not valid_mask.any():
                continue
            
            
            valid_indices = torch.where(valid_mask)[0]
            valid_env_ids = env_ids[valid_indices]
            
            
            self._goal.write_root_pose_to_sim(goal_default_state[valid_indices, :7], valid_env_ids)
            self._goal.write_root_velocity_to_sim(torch.zeros((len(valid_env_ids), 6), device=device), valid_env_ids)
            
            self._camera_obj.write_root_pose_to_sim(camera_obj_default_state[valid_indices, :7], valid_env_ids)
            self._camera_obj.write_root_velocity_to_sim(torch.zeros((len(valid_env_ids), 6), device=device), valid_env_ids)
            
            self._agent.write_root_pose_to_sim(agent_default_state[valid_indices, :7], valid_env_ids)
            self._agent.write_root_velocity_to_sim(torch.zeros((len(valid_env_ids), 6), device=device), valid_env_ids)
            
            self._vpt_objects.write_object_pose_to_sim(vpt_obj_default_state[valid_indices, :, :7], valid_env_ids)
            self._vpt_objects.write_object_velocity_to_sim(torch.zeros((len(valid_env_ids), self.num_objs, 6), device=device), valid_env_ids)
            
            for _ in range(5):
                self.sim.step()
            
            moved_vpt_for_ball = {i: None for i in range(num_envs)}
            move_ball_indices = torch.where(torch.isin(env_ids, self.envs_to_move_ball))[0]

            for local_idx, env_idx in enumerate(move_ball_indices):
                print(f"Moving ball for env {env_ids[env_idx].item()}")
                # We'll pick a random vpt obstacle and place it under the goal
                env_origin = self.scene.env_origins[env_ids[env_idx]]  # Fixed: use env_idx to get correct origin
                env_id = env_ids[env_idx]
                env_id_item = env_id.item()
                folder_idx = folder_indices[env_idx]
                goal_pos = goal_default_state[env_idx, :3]
                
                # ========== CHANGE #7: Select from active_indices only ==========
                active_indices = self.active_vpt_indices[env_id_item]
                # ================================================================
                
                # Pick a random VPT object that's not too tall
                tries = 0
                max_tries = self.active_vpt_objs * 2
                while True:
                    # ========== CHANGE #8: Use _get_active_vpt_dims and select from active_indices ==========
                    random_local_idx = random.randint(0, self.active_vpt_objs - 1)
                    random_obj_idx = active_indices[random_local_idx].item()
                    vpt_dims = self._get_active_vpt_dims(env_idx)
                    vpt_height = vpt_dims[random_local_idx, 2]
                    # ========================================================================================
                    
                    # print(f"Height of chosen object = {vpt_height.item()} - Index = {random_obj_idx}")
                    tries += 1
                    if vpt_height < 0.4:
                        break
                    if tries >= max_tries:
                        # Fail the env and break out
                        print(f"  ❌ Failed to find suitable VPT object under height limit for env {env_ids[env_idx].item()} after {max_tries} tries. Skipping ball move.")
                        random_obj_idx = None
                        break

                if random_obj_idx is None:
                    continue

                # Check if any other VPT object is within 1.5 units in x or y from goal_pos
                # ========== CHANGE #9: Only check conflicts with active VPT objects ==========
                vpt_positions = vpt_obj_default_state[env_idx, active_indices, :2]
                distances_xy = torch.norm(vpt_positions - goal_pos[:2], dim=1)
                # Exclude the current object itself
                current_obj_local_idx = (active_indices == random_obj_idx).nonzero(as_tuple=True)[0].item()
                distances_xy[current_obj_local_idx] = float('inf')
                conflicting_mask = distances_xy < 1.5
                conflicting_local_indices = torch.where(conflicting_mask)[0]
                # =============================================================================
                
                if len(conflicting_local_indices) > 0:
                    print(f"  Conflict detected for env {env_ids[env_idx].item()} when placing VPT object {random_obj_idx} under goal. Re-sampling position for conflicting objects.")
                
                # ========== CHANGE #10: When moving conflicting objects, use active_indices ==========
                for conf_local_idx in conflicting_local_indices:
                    conf_idx = active_indices[conf_local_idx].item()
                    # Move the conflicting object to a new random position
                    print(f"Moving conflicting VPT object {conf_idx} for env {env_ids[env_idx].item()}")
                    new_x = sample_uniform(-safe_x_range_obstacles, safe_x_range_obstacles, (1,), device=device)
                    new_y = sample_uniform(-safe_x_range_obstacles, safe_x_range_obstacles, (1,), device=device)
                    vpt_obj_default_state[env_idx, conf_idx, 0] = env_origin[0] + new_x
                    vpt_obj_default_state[env_idx, conf_idx, 1] = env_origin[1] + new_y
                # ===================================================================================
                
                # Place the selected object under the goal
                moved_vpt_for_ball[env_idx.item()] = random_obj_idx
                
                vpt_obj_default_state[env_idx, random_obj_idx, 0] = goal_pos[0]
                vpt_obj_default_state[env_idx, random_obj_idx, 1] = goal_pos[1]
                vpt_obj_default_state[env_idx, random_obj_idx, 2] = env_origin[2] + (vpt_height / 2.0)
                
                goal_height = vpt_obj_default_state[env_idx, random_obj_idx, 2] + (vpt_height / 2.0) + (self.goal_radius + 0.01)  # 0.15 radius + 1 cm buffer
                
                goal_default_state[env_idx, 2] = goal_height
            
                # print(f"Goal position = {goal_default_state[env_idx, :3].cpu().numpy()}")
                # print(f"VPT Object moved new pos = {vpt_obj_default_state[env_idx, random_obj_idx, :3].cpu().numpy()}")
            
            move_ball_env_ids = env_ids[move_ball_indices]
            self._goal.write_root_pose_to_sim(goal_default_state[move_ball_indices, :7], move_ball_env_ids)
            self._goal.write_root_velocity_to_sim(torch.zeros((len(move_ball_env_ids), 6), device=device), move_ball_env_ids)
            
            self._vpt_objects.write_object_pose_to_sim(vpt_obj_default_state[move_ball_indices, :, :7], move_ball_env_ids)
            self._vpt_objects.write_object_velocity_to_sim(torch.zeros((len(move_ball_env_ids), self.num_objs, 6), device=device), move_ball_env_ids)

                
            camera_positions = camera_obj_default_state[valid_indices, :3]
            camera_orientations = camera_obj_default_state[valid_indices, 3:7]
            
            # For occluded cases, place one of the top 3 tallest objects between camera and goal
            for local_idx, env_idx in enumerate(valid_indices):
                if visibility_categories[env_idx] != "occluded":
                    continue
                
                print(f"Triggered VPT displacement for env {env_ids[env_idx].item()} due to occlusion requirement")
                
                env_origin = env_origins[local_idx]
                env_id = valid_env_ids[local_idx]
                env_id_item = env_id.item()
                folder_idx = folder_indices[env_idx]
                camera_pos = camera_positions[local_idx]
                goal_pos = goal_default_state[env_idx, :3]
                
                # ========== CHANGE #11: Select random_obj_idx from active_indices ==========
                active_indices = self.active_vpt_indices[env_id_item]
                moved_idx = moved_vpt_for_ball[env_idx.item()]
                if moved_idx is not None:
                    # ========== CHANGE #12: Pick from active_indices excluding moved one ==========
                    available_local_indices = [i for i in range(self.active_vpt_objs) if active_indices[i].item() != moved_idx]
                    random_local_idx = random.choice(available_local_indices)
                    random_obj_idx = active_indices[random_local_idx].item()
                    # ===============================================================================
                else:
                    random_local_idx = random.randint(0, self.active_vpt_objs - 1)
                    random_obj_idx = active_indices[random_local_idx].item()
                # ===============================================================================
                
                # Direction vector from camera to goal
                direction_cam_to_goal = goal_pos[:2] - camera_pos[:2]
                distance_cam_to_goal = torch.norm(direction_cam_to_goal)
                
                if distance_cam_to_goal > 1e-6:
                    # Normalize direction
                    direction_cam_to_goal = direction_cam_to_goal / distance_cam_to_goal
                    
                    # Place object at random point between 30-70% along the line from camera to goal
                    # (centered range for occlusion)
                    t = random.uniform(0.2, 0.8)
                    new_pos = camera_pos[:2] + direction_cam_to_goal * (distance_cam_to_goal * t)
                    
                    vpt_pos = vpt_obj_default_state[env_idx, random_obj_idx, :3]
                    # print(f"Occlusion: Old pos = {vpt_pos[:2].cpu().numpy()}, New pos = {new_pos.cpu().numpy()}")
                    random_offset = sample_uniform(-0.4, 0.4, (2,), device=device)
                    
                    vpt_obj_default_state[env_idx, random_obj_idx, 0] = new_pos[0] + random_offset[0]
                    vpt_obj_default_state[env_idx, random_obj_idx, 1] = new_pos[1] + random_offset[1]
                    vpt_obj_default_state[env_idx, random_obj_idx, 2] = env_origin[2]
            
            
            for local_idx, env_idx in enumerate(valid_indices):
                if env_idx not in in_view_displaced and env_idx not in outside_fov_displaced:
                    continue
                
                print(f"Triggered VPT displacement for env {env_ids[env_idx].item()} due to {visibility_categories[env_idx]} requirement")
                
                env_origin = env_origins[local_idx]
                
                env_id = valid_env_ids[local_idx]
                env_id_item = env_id.item()
                folder_idx = folder_indices[env_idx]
                camera_pos = camera_positions[local_idx]
                goal_pos = goal_default_state[env_idx, :3]
                
                # ========== CHANGE #13: Select random_obj_idx from active_indices ==========
                active_indices = self.active_vpt_indices[env_id_item]
                moved_idx = moved_vpt_for_ball[env_idx.item()]
                if moved_idx is not None:
                    # ========== CHANGE #14: Pick from active_indices excluding moved one ==========
                    available_local_indices = [i for i in range(self.active_vpt_objs) if active_indices[i].item() != moved_idx]
                    random_local_idx = random.choice(available_local_indices)
                    random_obj_idx = active_indices[random_local_idx].item()
                    # ===============================================================================
                else:
                    random_local_idx = random.randint(0, self.active_vpt_objs - 1)
                    random_obj_idx = active_indices[random_local_idx].item()
                # ===============================================================================
                
                # Direction vector from camera to goal
                direction_cam_to_goal = goal_pos[:2] - camera_pos[:2]
                distance_cam_to_goal = torch.norm(direction_cam_to_goal)
                
                if distance_cam_to_goal > 1e-6:
                    # Normalize direction
                    direction_cam_to_goal = direction_cam_to_goal / distance_cam_to_goal
                    
                    # Place object at random point between 10-40% along the line from camera to goal
                    # t = random.uniform(0.1, 0.6)  # Interpolation factor
                    t = random.uniform(0.2, 0.8)  # Interpolation factor
                    new_pos = camera_pos[:2] + direction_cam_to_goal * (distance_cam_to_goal * t)
                    
                    vpt_pos = vpt_obj_default_state[env_idx, random_obj_idx, :3]
                    # print(f"Old pos = {vpt_pos[:2].cpu().numpy()}, New pos = {new_pos.cpu().numpy()}")
                    # print(f"Placed at {t*100:.1f}% along camera-to-goal line")
                    random_offset = sample_uniform(-0.4, 0.4, (2,), device=device)
                    
                    vpt_obj_default_state[env_idx, random_obj_idx, 0] = new_pos[0] + random_offset[0]
                    vpt_obj_default_state[env_idx, random_obj_idx, 1] = new_pos[1] + random_offset[1]
                    vpt_obj_default_state[env_idx, random_obj_idx, 2] = env_origin[2]
            
            self._vpt_objects.write_object_pose_to_sim(vpt_obj_default_state[valid_indices, :, :7], valid_env_ids)
            self._vpt_objects.write_object_velocity_to_sim(torch.zeros((len(valid_env_ids), self.num_objs, 6), device=device), valid_env_ids)
            
            goal_new_pos = self._goal.data.root_pos_w[valid_env_ids]
            camera_new_pos = self._camera_obj.data.root_pos_w[valid_env_ids]
            agent_new_pos = self._agent.data.root_pos_w[valid_env_ids]
            vpt_new_pos = self._vpt_objects.data.object_pos_w[valid_env_ids]
            
            
            for local_idx, env_idx in enumerate(valid_indices):
                z_valid = (
                    (goal_new_pos[local_idx, 2] >= 0.0) and (goal_new_pos[local_idx, 2] <= 1.0) and
                    (camera_new_pos[local_idx, 2] >= 0.0) and (camera_new_pos[local_idx, 2] <= 1.0) and
                    (agent_new_pos[local_idx, 2] >= 0.0) and (agent_new_pos[local_idx, 2] <= 1.0) and
                    torch.all((vpt_new_pos[local_idx, :, 2] >= 0.0) & (vpt_new_pos[local_idx, :, 2] <= 1.0))
                )
                
                if not z_valid:
                    envs_need_spawn_retry[env_idx] = True
            
            for _ in range(5):
                self.sim.step()
            
            # goal_radius = 0.15 + 0.01   # (0.15 radius + 1 cm buffer)
            # for local_idx, env_idx in enumerate(valid_indices):
            #     env_id = valid_env_ids[local_idx]
            #     env_id_item = env_id.item()
            #     folder_idx = folder_indices[env_idx]
            #     camera_pos = self._camera_obj.data.root_pos_w[env_id]
            #     goal_pos = self._goal.data.root_pos_w[env_id]
            #     if not self._check_target_vpt_distance(env_idx, goal_pos, vpt_new_pos[local_idx, :, :2], min_distance=goal_radius):
            #         print(f"    ❌ Env {env_id_item} (folder {folder_idx}): VPT objects too close to target")
            #         envs_need_spawn_retry[env_idx] = True
            #     else:
            #         if self.verbose >= 2:
            #             print(f"    ✅ Env {env_id_item} (folder {folder_idx}): VPT objects valid distance from target")
            
            final_valid_mask = valid_mask & ~envs_need_spawn_retry
            if not final_valid_mask.any():
                continue
            
            final_valid_indices = torch.where(final_valid_mask)[0]
            final_valid_env_ids = env_ids[final_valid_indices]
            
            camera_positions = camera_obj_default_state[final_valid_indices, :3]
            camera_orientations = camera_obj_default_state[final_valid_indices, 3:7]
            
            theta_left = math.pi / 2
            half_theta_left = theta_left / 2
            left_90_quat = torch.tensor([math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)], device=device)
            rotated_orientations = math_utils.quat_mul(camera_orientations, left_90_quat.unsqueeze(0).expand(len(final_valid_env_ids), -1))
            
            self._occlusion_camera.set_world_poses(
                positions=camera_positions,
                orientations=rotated_orientations,
                env_ids=final_valid_env_ids.tolist(),
                convention="world")
            
            
            # Parallelize camera orientation update for all "outside_fov" environments
            outside_fov_mask = torch.tensor(
                [visibility_categories[env_idx] == "outside_fov" for env_idx in final_valid_indices],
                device=device
            )
            if outside_fov_mask.any():
                # Gather indices and data for all outside_fov envs
                outside_fov_indices = final_valid_indices[outside_fov_mask]
                camera_pos_batch = camera_positions[outside_fov_mask]
                goal_pos_batch = goal_default_state[outside_fov_indices, :3]

                direction_to_goal = goal_pos_batch[:, :2] - camera_pos_batch[:, :2]
                yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0]) - math.radians(90)
                # Sample yaw_away from a range of math.pi ± 30 degrees (in radians)
                yaw_away = yaw + sample_uniform(
                    math.pi - math.radians(30), math.pi + math.radians(30), (len(outside_fov_indices),), device=device
                )
                roll = torch.full((len(outside_fov_indices),), -math.radians(self.agent_camera_pitch), device=device)
                zero = torch.zeros_like(roll)
                quaternion_away = quat_from_euler_xyz(roll, zero, yaw_away)

                camera_obj_default_state[outside_fov_indices, 3:7] = quaternion_away
                self._camera_obj.write_root_pose_to_sim(
                    camera_obj_default_state[outside_fov_indices, :7],
                    env_ids[outside_fov_indices]
                )
                
            camera_positions = camera_obj_default_state[final_valid_indices, :3]
            camera_orientations = camera_obj_default_state[final_valid_indices, 3:7]
            
            theta_left = math.pi / 2
            half_theta_left = theta_left / 2
            left_90_quat = torch.tensor([math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)], device=device)
            rotated_orientations = math_utils.quat_mul(camera_orientations, left_90_quat.unsqueeze(0).expand(len(final_valid_env_ids), -1))
            
            self._occlusion_camera.set_world_poses(
                positions=camera_positions,
                orientations=rotated_orientations,
                env_ids=final_valid_env_ids.tolist(),
                convention="world")
            
            for _ in range(10):
                self.sim.step()
                self._occlusion_camera.update(self.sim.cfg.dt)
            
            # Camera POV validation with image saving
            # 1. First, do occlusion raycast validation for all envs
            occlusion_valid_mask = torch.ones(len(final_valid_indices), dtype=torch.bool, device=device)
            for local_idx, env_idx in enumerate(final_valid_indices):
                env_id = final_valid_env_ids[local_idx]
                env_id_item = env_id.item()
                visibility_category = visibility_categories[env_idx]
                folder_idx = folder_indices[env_idx]
                camera_pos = camera_positions[local_idx]
                goal_pos = goal_default_state[env_idx, :3]
                
                if visibility_category in ["in_view", "occluded"]:
                    is_occluded = self._check_occlusion_raycast(camera_pos, goal_pos, env_id)
                    expected_occluded = (visibility_category == "occluded")
                    occlusion_valid = (is_occluded == expected_occluded)
                    occlusion_valid_mask[local_idx] = occlusion_valid
                    if not occlusion_valid:
                        if self.verbose >= 1:
                            print(f"    ❌ Env {env_id_item} (folder {folder_idx}): Occlusion raycast validation FAILED")
                            print(f"       Expected: {'occluded' if expected_occluded else 'visible'}, Got: {'occluded' if is_occluded else 'visible'}")
                        envs_need_spawn_retry[env_idx] = True
                    else:
                        if self.verbose >= 2:
                            print(f"    ✅ Env {env_id_item} (folder {folder_idx}): Occlusion raycast validation PASSED")

            # 2. For those that passed occlusion, generate circle points without agent fov checks
            occlusion_passed_mask = occlusion_valid_mask
            occlusion_passed_env_ids = final_valid_env_ids[occlusion_passed_mask]

            # Track which environments passed geometric validation
            geometric_valid_mask = occlusion_valid_mask.clone()  # Start with occlusion results

            if occlusion_passed_env_ids.numel() > 0:
                # Generate candidate points as in generate_valid_circle_points
                num_envs_passed = len(occlusion_passed_env_ids)
                fov_deg = 30.0
                fov_rad = math.radians(fov_deg)
                camera_pos = self._camera_obj.data.root_pos_w[occlusion_passed_env_ids, :2]
                goal_pos = self._goal.data.root_pos_w[occlusion_passed_env_ids, :2]
                d = torch.norm(camera_pos - goal_pos, dim=1)
                half_fov = torch.tensor(fov_rad / 2, device=device)
                radii = (d / 2) / torch.tan(half_fov)
                radii = radii * 1.2
                radii = radii.unsqueeze(1)
                num_angles = int(360.0 / 2.0)
                angles = torch.linspace(0, 2 * math.pi, num_angles, device=device)
                angles_expanded = angles.unsqueeze(0).expand(num_envs_passed, -1)
                all_x = goal_pos[:, 0].unsqueeze(1) + radii * torch.cos(angles_expanded)
                all_y = goal_pos[:, 1].unsqueeze(1) + radii * torch.sin(angles_expanded)
                total_points = num_envs_passed * num_angles
                all_points_batch = torch.stack([all_x, all_y], dim=2).reshape(total_points, 2)
                env_ids_batch = occlusion_passed_env_ids.unsqueeze(1).expand(-1, num_angles).reshape(total_points)
                
                # Geometric validation
                geometric_valid = self._is_point_valid_batch(
                    points=all_points_batch,
                    env_ids=env_ids_batch,
                    check_agent_fov=False
                )
                geometric_valid_per_env = geometric_valid.reshape(num_envs_passed, num_angles)
                
                MIN_GEOMETRIC_VALID_POINTS = 60
                # Check if at least MIN_GEOMETRIC_VALID_POINTS valid points exist
                for i, env_id in enumerate(occlusion_passed_env_ids):
                    # Find the original local_idx in final_valid_indices
                    local_idx = (final_valid_env_ids == env_id).nonzero(as_tuple=True)[0].item()
                    env_idx = final_valid_indices[local_idx]
                    
                    valid_mask = geometric_valid_per_env[i]
                    if valid_mask.sum().item() < MIN_GEOMETRIC_VALID_POINTS:
                        print(f"    ❌ Env {env_id.item()} (folder {folder_indices[env_idx]}): Geometric viewpoint check FAILED ({valid_mask.sum().item()}/{MIN_GEOMETRIC_VALID_POINTS} valid points)")
                        # Not enough viewpoints, mark for retry
                        envs_need_spawn_retry[env_idx] = True
                        geometric_valid_mask[local_idx] = False
                    else:
                        if self.verbose >= 2:
                            print(f"    ✅ Env {env_id.item()} (folder {folder_indices[env_idx]}): Geometric viewpoint check PASSED ({valid_mask.sum().item()}/{MIN_GEOMETRIC_VALID_POINTS} valid points)")

            # 3. For those that passed BOTH occlusion AND geometric checks, do camera POV validation
            for local_idx, env_idx in enumerate(final_valid_indices):
                # Skip if failed any previous test
                if not geometric_valid_mask[local_idx]:
                    continue

                env_id = final_valid_env_ids[local_idx]
                env_id_item = env_id.item()
                visibility_category = visibility_categories[env_idx]
                folder_idx = folder_indices[env_idx]
                camera_pos = camera_positions[local_idx]
                goal_pos = goal_default_state[env_idx, :3]

                # Save camera POV to debug folder FIRST
                debug_folder = f"{self.base_path}/debug_camera_pov"
                os.makedirs(debug_folder, exist_ok=True)
                sem_img = self._occlusion_camera.data.output["semantic_segmentation"][env_id]
                cam_pov_img = sem_img[:, :, :3]
                if cam_pov_img.max() <= 1.0:
                    cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(np.uint8)
                else:
                    cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)
                debug_filename = f"{debug_folder}/env_{env_id_item}_folder_{folder_idx}_attempt_{spawn_attempt}.png"
                target_visible_in_camera, red_count = self._check_target_in_img(file_name=debug_filename, cam_pov=cam_pov_np, return_red_count=True)

                camera_validation_passed = False
                if visibility_category == "in_view":
                    camera_validation_passed = target_visible_in_camera
                    if not camera_validation_passed:
                        if self.verbose >= 1:
                            print(f"    ❌ Env {env_id_item} (folder {folder_idx}): Camera validation FAILED")
                            print(f"       Expected: in_view (target visible), Got: target NOT visible (red pixels: {red_count}/{self.goal_pixel_threshold})")
                            print(f"       Debug image saved: {debug_filename}")
                    else:
                        if self.verbose >= 2:
                            print(f"    ✅ Env {env_id_item} (folder {folder_idx}): Camera validation PASSED - in_view (red pixels: {red_count})")
                elif visibility_category == "occluded":
                    camera_validation_passed = not target_visible_in_camera
                    if not camera_validation_passed:
                        if self.verbose >= 1:
                            print(f"    ❌ Env {env_id_item} (folder {folder_idx}): Camera validation FAILED")
                            print(f"       Expected: occluded (target NOT visible), Got: target visible (red pixels: {red_count}/{self.goal_pixel_threshold})")
                            print(f"       Debug image saved: {debug_filename}")
                    else:
                        if self.verbose >= 2:
                            print(f"    ✅ Env {env_id_item} (folder {folder_idx}): Camera validation PASSED - occluded (red pixels: {red_count})")
                elif visibility_category == "outside_fov":
                    camera_validation_passed = not target_visible_in_camera
                    if not camera_validation_passed:
                        if self.verbose >= 1:
                            print(f"    ❌ Env {env_id_item} (folder {folder_idx}): Camera validation FAILED")
                            print(f"       Expected: outside_fov (target NOT visible), Got: target visible (red pixels: {red_count}/{self.goal_pixel_threshold})")
                            print(f"       Debug image saved: {debug_filename}")
                    else:
                        if self.verbose >= 2:
                            print(f"    ✅ Env {env_id_item} (folder {folder_idx}): Camera validation PASSED - outside_fov (red pixels: {red_count})")

                if not camera_validation_passed:
                    envs_need_spawn_retry[env_idx] = True

        random_yaw_agent = sample_uniform(0, 2 * math.pi, (num_envs,), device)
        agent_default_state[:, 3] = torch.cos(random_yaw_agent / 2)
        agent_default_state[:, 4] = 0.0
        agent_default_state[:, 5] = 0.0
        agent_default_state[:, 6] = torch.sin(random_yaw_agent / 2)
        self._agent.write_root_pose_to_sim(agent_default_state[:, :7], env_ids)

        all_valid_points = self.generate_valid_circle_points(env_ids=env_ids, angle_step=2.0, max_attempts=100)

        if self.valid_viewpoint_poses is None:
            self.valid_viewpoint_poses = [None] * self.num_envs

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            valid_points_2d = all_valid_points[i]

            if valid_points_2d.shape[0] > 0:
                agent_z = self._agent.data.default_root_state[env_id, 2]
                valid_points_3d = torch.zeros((valid_points_2d.shape[0], 3), device=device)
                valid_points_3d[:, :2] = valid_points_2d
                valid_points_3d[:, 2] = agent_z
                self.valid_viewpoint_poses[env_id_item] = valid_points_3d if valid_points_3d.shape[0] >= self.images_per_env else torch.zeros((0, 3), device=device)
            else:
                self.valid_viewpoint_poses[env_id_item] = torch.zeros((0, 3), device=device)

    def _is_point_valid_batch(self, points: torch.Tensor, env_ids: torch.Tensor, 
                          min_obstacle_distance: float = 0.5,
                          min_camera_obstacle_distance: float = 0.4,
                          min_camera_target_distance: float = 1.0,
                          check_agent_fov: bool = False,
                          min_required_points: int = None) -> torch.Tensor:
        """Batch validate multiple points across multiple environments in parallel.
        
        Args:
            min_required_points: If provided, stops FOV checking for an env once it has this many valid points.
        """
        device = points.device
        num_points = points.shape[0]
        
        if min_required_points is None:
            min_required_points = self.images_per_env
        
        valid_mask = torch.ones(num_points, dtype=torch.bool, device=device)
        
        env_origins = self.scene.env_origins[env_ids, :2]
        boundary_limit = self.center_to_boundary.item() if torch.is_tensor(
            self.center_to_boundary) else self.center_to_boundary
        
        # Boundary Check
        min_bounds = env_origins - boundary_limit
        max_bounds = env_origins + boundary_limit
        
        in_bounds = torch.all((points >= min_bounds) & (points <= max_bounds), dim=1)
        valid_mask &= in_bounds
        print(f"  🌐 Boundary check: {valid_mask.sum().item()}/{num_points} points valid")
        if not valid_mask.any():
            return valid_mask
        
        # ========== CHANGE #1 & #2: VPT Distance check - only for active objects ==========
        # Build a tensor of active VPT positions for each point's environment using helper method
        vpt_positions_list = []
        
        for point_idx in range(num_points):
            env_id = env_ids[point_idx]
            # Get active VPT positions using helper method (shape: [active_vpt_objs, 3])
            active_vpt_pos = self._get_active_vpt_positions(env_id)[:, :2]  # Take only x, y
            vpt_positions_list.append(active_vpt_pos)
        
        # Stack into tensor (shape: [num_points, active_vpt_objs, 2])
        vpt_positions = torch.stack(vpt_positions_list, dim=0)
        
        # Compute distances from each point to its env's active VPT objects
        distances_to_vpt = torch.norm(points.unsqueeze(1) - vpt_positions, dim=2)
        min_vpt_distances = distances_to_vpt.min(dim=1)[0]
        valid_mask &= (min_vpt_distances >= min_obstacle_distance)
        # ===================================================================================
        print(f"  🛑 VPT obstacle check: {valid_mask.sum().item()}/{num_points} points valid")
        
        if not valid_mask.any():
            return valid_mask
        
        # Camera distance check
        camera_positions = self._camera_obj.data.root_pos_w[env_ids, :2]
        camera_distances = torch.norm(points - camera_positions, dim=1)
        valid_mask &= (camera_distances >= min_camera_target_distance)
        print(f"  📷 Camera point distance check: {valid_mask.sum().item()}/{num_points} points valid")
        
        if not valid_mask.any():
            return valid_mask
        
        # Camera - VPT Distance check
        # camera_vpt_distances = torch.norm(vpt_positions - camera_positions.unsqueeze(1), dim=2)
        # min_camera_vpt_distances = camera_vpt_distances.min(dim=1)[0]
        # valid_mask &= (min_camera_vpt_distances >= min_camera_obstacle_distance)
        
        # ========== CHANGE #3: Camera - VPT Distance check - only for active objects ==========
        # Build camera-VPT distances using active objects only via helper method
        camera_vpt_distances_list = []
        
        for point_idx in range(num_points):
            env_id = env_ids[point_idx]
            
            # Get active VPT positions for this env using helper method
            active_vpt_pos = self._get_active_vpt_positions(env_id)[:, :2]  # Take only x, y
            camera_pos = camera_positions[point_idx]
            
            # Compute distances from camera to active VPT objects
            distances = torch.norm(active_vpt_pos - camera_pos.unsqueeze(0), dim=1)
            min_distance = distances.min()
            camera_vpt_distances_list.append(min_distance)
        
        min_camera_vpt_distances = torch.stack(camera_vpt_distances_list)
        valid_mask &= (min_camera_vpt_distances >= min_camera_obstacle_distance)
        # =====================================================================================
        print(f"  🤖 Camera-VPT obstacle check: {valid_mask.sum().item()}/{num_points} points valid")
        
        if not valid_mask.any():
            return valid_mask
        
        
        if not check_agent_fov:
            return valid_mask
        
        # FOV validation - process in parallel across environments
        points_to_check = torch.where(valid_mask)[0]
        
        if points_to_check.numel() == 0:
            return valid_mask
        
        # Store original agent state
        current_agent_pos = self._agent.data.root_pos_w[env_ids].clone()
        current_agent_quat = self._agent.data.root_quat_w[env_ids].clone()
        
        # Group points by environment and randomly sample up to MAX_SAMPLES_PER_ENV
        unique_env_ids = torch.unique(env_ids[points_to_check])
        env_to_points = {}
        env_to_indices = {}
        env_completed = {}  # Track which envs have enough valid points
        env_valid_counts = {}  # Track valid point count per env
        
        MAX_SAMPLES_PER_ENV = 120
        
        for env_id in unique_env_ids:
            env_mask = env_ids[points_to_check] == env_id
            env_points_indices = points_to_check[env_mask]
            env_points = points[env_points_indices]
            
            # Randomly sample up to MAX_SAMPLES_PER_ENV points
            num_points_for_env = len(env_points)
            if num_points_for_env > MAX_SAMPLES_PER_ENV:
                # Random sampling without replacement
                sample_indices = torch.randperm(num_points_for_env, device=device)[:MAX_SAMPLES_PER_ENV]
                env_to_points[env_id.item()] = env_points[sample_indices]
                env_to_indices[env_id.item()] = env_points_indices[sample_indices]
                if self.verbose >= 2:
                    print(f"  🎲 Env {env_id.item()}: Sampled {MAX_SAMPLES_PER_ENV}/{num_points_for_env} points for FOV validation")
            else:
                env_to_points[env_id.item()] = env_points
                env_to_indices[env_id.item()] = env_points_indices
                if self.verbose >= 2:
                    print(f"  🎲 Env {env_id.item()}: Using all {num_points_for_env} points for FOV validation")
            
            env_completed[env_id.item()] = False
            env_valid_counts[env_id.item()] = 0
                
        
        # Find maximum number of points any environment has (after sampling)
        max_points_per_env = max(len(pts) for pts in env_to_points.values())
        
        fov_valid = torch.zeros(num_points, dtype=torch.bool, device=device)
        
        # Process one point index at a time across all environments
        for point_idx in range(max_points_per_env):
            if self.verbose >= 2 and point_idx % 10 == 0:
                print(f"    🔄 FOV validation progress: {point_idx+1}/{max_points_per_env} points")
            
            # Collect environments and points for this iteration (skip completed envs)
            batch_env_ids = []
            batch_points = []
            batch_global_indices = []
            
            for env_id in unique_env_ids:
                env_id_item = env_id.item()
                
                # Skip this env if it already has enough valid points
                if min_required_points is not None and env_completed[env_id_item]:
                    continue
                
                env_points = env_to_points[env_id_item]
                env_indices = env_to_indices[env_id_item]
                
                # Check if this environment has a point at this index
                if point_idx < len(env_points):
                    batch_env_ids.append(env_id)
                    batch_points.append(env_points[point_idx])
                    batch_global_indices.append(env_indices[point_idx])
            
            if len(batch_env_ids) == 0:
                # All envs either completed or out of points
                if min_required_points is not None and all(env_completed.values()):
                    if self.verbose >= 1:
                        print(f"    ✅ All environments have {min_required_points}+ valid points, stopping FOV check early")
                    break
                continue
            
            batch_env_ids = torch.stack(batch_env_ids)
            batch_points = torch.stack(batch_points)
            batch_global_indices = torch.stack(batch_global_indices)
            
            # Teleport all agents to their respective points
            temp_agent_pos = torch.zeros((len(batch_env_ids), 3), device=device)
            temp_agent_pos[:, :2] = batch_points
            temp_agent_pos[:, 2] = self._agent.data.default_root_state[batch_env_ids, 2]
            
            camera_pos_3d = self._camera_obj.data.root_pos_w[batch_env_ids]
            goal_pos_3d = self._goal.data.root_pos_w[batch_env_ids]
            midpoints_3d = (camera_pos_3d[:, :2] + goal_pos_3d[:, :2]) / 2.0
            
            directions = midpoints_3d - batch_points
            yaws = torch.atan2(directions[:, 1], directions[:, 0])
            
            temp_agent_quat = torch.zeros((len(batch_env_ids), 4), device=device)
            temp_agent_quat[:, 0] = torch.cos(yaws / 2)
            temp_agent_quat[:, 3] = torch.sin(yaws / 2)
            
            temp_poses = torch.cat([temp_agent_pos, temp_agent_quat], dim=1)
            self._agent.write_root_com_pose_to_sim(temp_poses, batch_env_ids)
            
            # Update simulation
            for _ in range(3):
                self.sim.step()
                self._tiled_camera.update(self.sim.cfg.dt)
                self._agent.update(self.sim.cfg.dt)
                self._camera_obj.update(self.sim.cfg.dt)
                self._goal.update(self.sim.cfg.dt)
            
            # Check visibility for all agents in this batch in parallel
            batch_goal_visible, batch_camera_visible = self.check_batch_object_visibility(batch_env_ids)
            batch_fov_valid = batch_goal_visible & batch_camera_visible
            
            # Assign results back to global fov_valid tensor
            fov_valid[batch_global_indices] = batch_fov_valid
            
            # Update valid counts and check for completion
            if min_required_points is not None:
                for i, env_id in enumerate(batch_env_ids):
                    env_id_item = env_id.item()
                    if batch_fov_valid[i]:
                        env_valid_counts[env_id_item] += 1
                        if env_valid_counts[env_id_item] >= min_required_points and not env_completed[env_id_item]:
                            env_completed[env_id_item] = True
                            if self.verbose >= 1:
                                print(f"    🎯 Env {env_id_item}: Reached {min_required_points} valid points, stopping FOV checks for this env")
        
        # Restore original agent positions
        restore_poses = torch.cat([current_agent_pos, current_agent_quat], dim=1)
        self._agent.write_root_com_pose_to_sim(restore_poses, env_ids)
        
        valid_mask &= fov_valid
        
        return valid_mask

    def generate_valid_circle_points(self, env_ids: torch.Tensor, angle_step: float = 2.0,
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
        all_x = self._goal.data.root_pos_w[env_ids, 0].unsqueeze(1) + radii * torch.cos(angles_expanded)
        all_y = self._goal.data.root_pos_w[env_ids, 1].unsqueeze(1) + radii * torch.sin(angles_expanded)
        
        total_points = num_envs * num_angles
        all_points_batch = torch.stack([all_x, all_y], dim=2).reshape(total_points, 2)
        env_ids_batch = env_ids.unsqueeze(1).expand(-1, num_angles).reshape(total_points)
        
        # Step 1: Geometric validation
        geometric_valid = self._is_point_valid_batch(
            points=all_points_batch,
            env_ids=env_ids_batch,
            check_agent_fov=False
        )
        
        geometric_valid_per_env = geometric_valid.reshape(num_envs, num_angles)
        
        if self.verbose >= 2:
            for i, env_id in enumerate(env_ids):
                env_id_item = env_id.item()
                print(f"  Env {env_id_item}: {geometric_valid_per_env[i].sum().item()} geometric candidates")
        
        # Step 2: Vectorized displacement filtering across all environments
        displacement_filtered_points = []
        displacement_filtered_env_ids = []
        displacement_filtered_indices = []
        
        MIN_CANDIDATES_FOR_FOV = 60  # Require at least this many candidates before FOV check
        
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
                    print(f"  Env {env_id_item}: ❌ Only {num_geometric} geometric candidates, need {MIN_CANDIDATES_FOR_FOV}. Skipping.")
                continue
            
            valid_points = all_points_batch[i * num_angles:(i + 1) * num_angles][valid_mask]
            
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
            selected_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
            selected_mask[0] = True  # Always select first point
            
            for idx in range(1, num_points):
                # Check if this point is far enough from all selected points
                distances_to_selected = pairwise_distances[idx, selected_mask]
                if torch.all(distances_to_selected >= self.min_viewpoint_distance):
                    selected_mask[idx] = True
            
            filtered_candidates = sorted_points[selected_mask]
            
            num_before = len(sorted_points)
            num_after = len(filtered_candidates)
            
            # Check: need enough candidates after displacement filtering
            if num_after < MIN_CANDIDATES_FOR_FOV:
                if self.verbose >= 1:
                    rejection_rate = (1 - num_after/num_before) * 100 if num_before > 0 else 0
                    print(f"  Env {env_id_item}: ❌ Only {num_after} candidates after displacement ({rejection_rate:.1f}% rejected), need {MIN_CANDIDATES_FOR_FOV}. Skipping FOV check.")
                continue
            
            if self.verbose >= 2:
                rejection_rate = (1 - num_after/num_before) * 100 if num_before > 0 else 0
                print(f"  Env {env_id_item}: Displacement filter: {num_after}/{num_before} kept ({rejection_rate:.1f}% rejected)")
            
            # Add these pre-filtered candidates for FOV checking
            displacement_filtered_points.append(filtered_candidates)
            displacement_filtered_env_ids.extend([env_id.item()] * len(filtered_candidates))
            displacement_filtered_indices.extend([i] * len(filtered_candidates))
        
        if len(displacement_filtered_points) == 0:
            if self.verbose >= 1:
                print(f"  ❌ No candidates passed displacement filter for any environment")
            return [torch.zeros((0, 2), device=device) for _ in range(num_envs)]
        
        # Concatenate all displacement-filtered candidates
        all_candidates = torch.cat(displacement_filtered_points, dim=0)
        all_candidates_env_ids = torch.tensor(displacement_filtered_env_ids, dtype=torch.long, device=device)
        all_candidates_indices = torch.tensor(displacement_filtered_indices, dtype=torch.long, device=device)
        
        if self.verbose >= 2:
            total_geometric = geometric_valid.sum().item()
            total_after_displacement = len(all_candidates)
            saved_compute = ((total_geometric - total_after_displacement) / total_geometric * 100) if total_geometric > 0 else 0
            print(f"  💡 FOV candidates: {total_after_displacement}/{total_geometric} ({saved_compute:.1f}% compute saved)")
        
        # Store original agent state
        original_agent_pos = self._agent.data.root_pos_w[env_ids].clone()
        original_agent_quat = self._agent.data.root_quat_w[env_ids].clone()
        
        # Step 3: FOV check ONLY on displacement-filtered candidates
        fov_valid_mask = self._is_point_valid_batch(
            points=all_candidates,
            env_ids=all_candidates_env_ids,
            check_agent_fov=True,
            min_required_points=MIN_REQUIRED_POINTS  # Pass min required to enable early stopping
        )
        
        # Restore original agent positions
        self._agent.write_root_pose_to_sim(
            torch.cat([original_agent_pos, original_agent_quat], dim=-1),
            env_ids=env_ids
        )
        
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
                    print(f"  Env {env_id_item}: ❌ 0/{num_candidates} passed FOV check")
                continue
            
            valid_points_tensor = all_candidates[env_mask][env_fov_valid]
            
            fov_rejection_rate = (1 - num_fov_valid/num_candidates) * 100 if num_candidates > 0 else 0
            
            if len(valid_points_tensor) >= MIN_REQUIRED_POINTS:
                all_valid_points.append(valid_points_tensor)
                if self.verbose >= 2:
                    print(f"  Env {env_id_item}: ✅ {len(valid_points_tensor)}/{num_candidates} passed FOV ({fov_rejection_rate:.1f}% rejected)")
            else:
                all_valid_points.append(torch.zeros((0, 2), device=device))
                if self.verbose >= 1:
                    print(f"  Env {env_id_item}: ❌ Only {len(valid_points_tensor)}/{MIN_REQUIRED_POINTS} points ({fov_rejection_rate:.1f}% FOV rejection)")
        
        return all_valid_points

    def _collect_images_for_slot(self, env_id: torch.Tensor, folder_idx: int) -> None:
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
            print(f"    📸 Collecting {self.images_per_env} images for slot {env_id_item}, env {global_env_id}, folder {folder_idx}")
        
        # Get the selected viewpoints for this SLOT
        viewpoints = self.selected_viewpoints_for_collection[env_id_item]
        if viewpoints is None:
            # Debug print
            print(f"[ERROR]: No viewpoints selected for slot {env_id_item} (env {global_env_id})")
            print(f"  valid_viewpoint_poses length: {len(self.valid_viewpoint_poses[env_id_item]) if self.valid_viewpoint_poses[env_id_item] else 'None'}")
            raise RuntimeError(f"No viewpoints selected for slot {env_id_item} (env {global_env_id})")
        
        # Create single-env tensor
        single_env_tensor = torch.tensor([env_id_item], dtype=torch.long, device=device)
        
        # Collect images from all viewpoints
        for viewpoint_idx in range(self.images_per_env):
            # Get target position for this viewpoint
            target_pos_2d = viewpoints[viewpoint_idx]
            if target_pos_2d.shape[-1] == 3:
                target_pos_2d = target_pos_2d[:2]
            
            # Create 3D position
            target_pos_3d = torch.zeros(3, device=device)
            target_pos_3d[:2] = target_pos_2d
            target_pos_3d[2] = self._agent.data.default_root_state[env_id_item, 2]
            
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
                torch.zeros((1, 6), device=device),
                single_env_tensor
            )
            
            # Update cameras
            for _ in range(3):
                self.sim.step()
                self._rgb_tiled_camera.update(self.sim.cfg.dt)
                self._distance_tiled_camera.update(self.sim.cfg.dt)
                if self.save_camera_pov:
                    self._occlusion_camera.update(self.sim.cfg.dt)
            
            # Get camera data for this env
            rgb_data = self._rgb_tiled_camera.data.output["rgb"]
            depth_data = self._distance_tiled_camera.data.output["distance_to_camera"]
            camera_pov_data = self._occlusion_camera.data.output["semantic_segmentation"] if self.save_camera_pov else None
            
            # Save this single image
            self._save_single_image(env_id_item, folder_idx, rgb_data, depth_data, camera_pov_data, viewpoint_idx)
        
        # Save env config
        self._save_env_config_to_json(env_id_item, folder_idx)
        
        # Clear viewpoints for this env
        self.selected_viewpoints_for_collection[env_id_item] = None
        
        if self.verbose >= 1:
            print(f"    ✅ Collected and saved {self.images_per_env} images for folder {folder_idx}")

    def _save_single_image(self, env_id_item: int, folder_idx: int, rgb_data: torch.Tensor, 
                        depth_data: torch.Tensor, camera_pov_data: torch.Tensor = None,
                        image_idx: int = 0) -> None:
        """Save a single image for an environment."""
        
        if folder_idx not in self.env_visibility_labels:
            raise RuntimeError(f"CRITICAL ERROR: No visibility label found for folder_idx {folder_idx}!")
        
        visibility_label = self.env_visibility_labels[folder_idx]
        
        if visibility_label not in ["Yes", "No"]:
            raise RuntimeError(f"CRITICAL ERROR: Invalid visibility label '{visibility_label}' for folder_idx {folder_idx}!")
        
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
        
        depth_np[np.isinf(depth_np)] = depth_np[~np.isinf(depth_np)].max() if depth_np[~np.isinf(depth_np)].size > 0 else 0
        
        if depth_np.max() > depth_np.min():
            depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)
        
        cv2.imwrite(depth_filename, depth_normalized)
        
        # Save camera POV (only once)
        if self.save_camera_pov and camera_pov_data is not None and image_idx == 0:
            cam_pov_filename = f"{cam_env_folder}/cam_pov.png"
            if not os.path.exists(cam_pov_filename):
                cam_pov_img = camera_pov_data[env_id_item, :, :, :3]
                
                if cam_pov_img.max() <= 1.0:
                    cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(np.uint8)
                else:
                    cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)
                
                cv2.imwrite(cam_pov_filename, cv2.cvtColor(cam_pov_np, cv2.COLOR_RGB2BGR))