from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import random
import numpy as np
from typing import List, Dict, Tuple, Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, RayCaster, save_images_to_file, Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, sample_gaussian, quat_from_euler_xyz
from isaaclab.utils import math as math_utils

from .vpt_env_cfg import VPTEnvCfg


class VPTEnv(DirectRLEnv):

    cfg: VPTEnvCfg

    def __init__(self,
                 cfg: VPTEnvCfg,
                 render_mode: str | None = None,
                 **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.boundary_limits = self.cfg.boundary_limits
        self.obstacle_metadata = self.cfg.objects_metadata
        self.agent_height = self.cfg.agent_height
        self.agent_camera_pitch = self.cfg.agent_camera_pitch
        self.num_objs = self.cfg.num_vpt_objs
        self.center_to_boundary = torch.abs(
            torch.tensor(self.boundary_limits).view(-1)[0])

        self.verbose = 2

        self.valid_viewpoint_poses = None

        self.viewpoint_pose_counter = torch.zeros(self.num_envs,
                                                  dtype=torch.long,
                                                  device=self.device)

        self.next_env_folder_idx = 0

        self.env_visibility_labels = {}
        self.env_visibility_reasons = {}
        self.base_path = "/home/arock3/Documents/data"
        # self.base_path = "/media/data_cifs_lrs/projects/prj_robotics/VPTnav_v1"

        self.visibility_labels_json_path = f"{self.base_path}/visibility_labels.json"

        self._reset_called = False
        self.max_rgb_images = 1_000
        self.save_camera_pov = True
        self.imgs_saved_per_env = 20

        self.used_viewpoint_indices = [set() for _ in range(self.num_envs)]

    def close(self):
        super().close()

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

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0,
                                           color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def check_batch_object_visibility(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Checks object visibility for a batch of environments in a fully parallelized manner.

        This function leverages PyTorch's tensor operations to process all semantic
        segmentation images from the specified environments at once, avoiding any
        sequential loops.

        Args:
            env_ids: A tensor of environment indices to check.

        Returns:
            A tuple containing two boolean tensors:
            - goal_visible: A boolean tensor indicating if the goal is visible in each environment.
            - camera_visible: A boolean tensor indicating if the camera is visible in each environment.
        """
        GOAL_THRESHOLD = 20
        CAMERA_THRESHOLD = 10

        # 1. Index the semantic data tensor to get a batch of images
        # Shape: (num_check_envs, height, width, 4)
        sem_imgs = self._tiled_camera.data.output["semantic_segmentation"][env_ids]

        # 2. Perform vectorized color masking across the entire batch
        # The ellipsis (...) broadcasts the operation across the batch dimension
        r = sem_imgs[..., 0]
        g = sem_imgs[..., 1]
        b = sem_imgs[..., 2]

        red_mask = (r >= 0.95) & (g <= 0.05) & (b <= 0.05)
        green_mask = (r <= 0.05) & (g >= 0.95) & (b <= 0.05)

        # 3. Sum pixel counts over the height and width dimensions (1 and 2) for each image
        # This results in a 1D tensor of counts, one for each environment
        red_counts = red_mask.sum(dim=(1, 2))
        green_counts = green_mask.sum(dim=(1, 2))

        # 4. Perform vectorized comparison against thresholds
        goal_visible = red_counts >= GOAL_THRESHOLD
        camera_visible = green_counts >= CAMERA_THRESHOLD

        return goal_visible, camera_visible


    def move_agent(self, actions, env_ids: Sequence[int] | None = None):
        if env_ids is None:
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

        # Process non-action 6 environments
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
            else: # Actions 0 and 1
                forward_input = 1.0 if action == 0 else -1.0
                local_movement = torch.tensor([forward_input, 0.0, 0.0], device=device)
                world_velocity = math_utils.quat_apply(upright_quat[i], local_movement) * max_velocity
                desired_vel[i, :3] = world_velocity
                desired_vel[i, 3:6] = 0.0

        if action_6_mask.any():
            action_6_indices = action_6_mask.nonzero(as_tuple=True)[0]
            action_6_env_ids = env_ids[action_6_indices]
            num_action_6_envs = len(action_6_env_ids)

            final_pos = current_pos[action_6_indices].clone()
            final_quat = new_quat[action_6_indices].clone()
            found_valid_mask = torch.zeros(num_action_6_envs, dtype=torch.bool, device=device)

            available_indices_per_env = []
            can_search_mask = torch.ones(num_action_6_envs, dtype=torch.bool, device=device)

            for i, env_id_item in enumerate(action_6_env_ids.tolist()):
                if (self.valid_viewpoint_poses is not None and env_id_item < len(self.valid_viewpoint_poses)
                        and self.valid_viewpoint_poses[env_id_item] is not None and len(self.valid_viewpoint_poses[env_id_item]) > 0):
                    num_poses = len(self.valid_viewpoint_poses[env_id_item])
                    available = list(set(range(num_poses)) - self.used_viewpoint_indices[env_id_item])
                    if not available:
                        if self.verbose >= 1: print(f"[Action 6] Env {env_id_item}: All {num_poses} viewpoints used, resetting")
                        self.used_viewpoint_indices[env_id_item].clear()
                        available = list(range(num_poses))
                    available_indices_per_env.append(available)
                else:
                    can_search_mask[i] = False
                    available_indices_per_env.append([])
                    if self.verbose >= 1: print(f"Warning: No valid viewpoints for env {env_id_item}, staying in place")

            still_searching_indices = can_search_mask.nonzero(as_tuple=True)[0]
            max_attempts = 10

            for attempt in range(max_attempts):
                if len(still_searching_indices) == 0:
                    break

                candidate_pose_indices, candidate_positions, valid_sample_mask = [], [], []
                for i in still_searching_indices.tolist():
                    if available_indices_per_env[i]:
                        pose_idx = random.choice(available_indices_per_env[i])
                        env_id_item = action_6_env_ids[i].item()
                        candidate_pose_indices.append(pose_idx)
                        candidate_positions.append(self.valid_viewpoint_poses[env_id_item][pose_idx])
                        valid_sample_mask.append(True)
                    else:
                        valid_sample_mask.append(False)
                
                if not any(valid_sample_mask): break

                valid_sample_mask = torch.tensor(valid_sample_mask, dtype=torch.bool, device=device)
                indices_this_attempt = still_searching_indices[valid_sample_mask]
                env_ids_this_attempt = action_6_env_ids[indices_this_attempt]
                candidate_pos_tensor = torch.stack(candidate_positions).to(device)

                camera_pos = self._camera_obj.data.root_pos_w[env_ids_this_attempt, :2]
                goal_pos = self._goal.data.root_pos_w[env_ids_this_attempt, :2]
                direction = (camera_pos + goal_pos) / 2.0 - candidate_pos_tensor[:, :2]
                yaw = torch.atan2(direction[:, 1], direction[:, 0])
                yaw[torch.norm(direction, dim=1) < 1e-6] = 0.0

                temp_quat = torch.zeros(len(indices_this_attempt), 4, device=device)
                temp_quat[:, 0] = torch.cos(yaw / 2.0)
                temp_quat[:, 3] = torch.sin(yaw / 2.0)
                
                temp_pos = final_pos[indices_this_attempt].clone()
                temp_pos[:, :3] = candidate_pos_tensor
                
                original_poses = torch.cat([final_pos[indices_this_attempt], final_quat[indices_this_attempt]], dim=1)
                self._agent.write_root_com_pose_to_sim(torch.cat([temp_pos, temp_quat], dim=1), env_ids_this_attempt)
                
                for _ in range(3): self.sim.step(); self._tiled_camera.update(self.sim.cfg.dt)

                # MODIFIED: Call the new batch function and use the resulting tensors directly
                goal_vis, cam_vis = self.check_batch_object_visibility(env_ids_this_attempt)
                goal_vis = torch.as_tensor(goal_vis, device=device)
                cam_vis = torch.as_tensor(cam_vis, device=device)

                success_mask = goal_vis & cam_vis   
                
                self._agent.write_root_com_pose_to_sim(original_poses, env_ids_this_attempt)

                successful_indices = indices_this_attempt[success_mask]
                if successful_indices.numel() > 0:
                    found_valid_mask[successful_indices] = True
                    final_pos[successful_indices] = temp_pos[success_mask]
                    final_quat[successful_indices] = temp_quat[success_mask]
                    
                    successful_pose_idxs = torch.tensor(candidate_pose_indices, device=device)[valid_sample_mask][success_mask]
                    for i, local_idx in enumerate(successful_indices.tolist()):
                        self.used_viewpoint_indices[action_6_env_ids[local_idx].item()].add(successful_pose_idxs[i].item())

                failed_pose_idxs = torch.tensor(candidate_pose_indices, device=device)[valid_sample_mask][~success_mask]
                failed_indices = indices_this_attempt[~success_mask]
                for i, local_idx in enumerate(failed_indices.tolist()):
                    available_indices_per_env[local_idx].remove(failed_pose_idxs[i].item())

                still_searching_indices = (~found_valid_mask & can_search_mask).nonzero(as_tuple=True)[0]
            
            if not found_valid_mask.all():
                failed_envs = action_6_env_ids[~found_valid_mask & can_search_mask].tolist()
                if self.verbose >= 1 and failed_envs:
                    print(f"[Action 6] Failed to find a visible viewpoint for envs: {failed_envs}")

            current_pos[action_6_indices] = final_pos
            new_quat[action_6_indices] = final_quat

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

    def check_batch_object_visibility(self, env_ids: list[int]) -> tuple[list[bool], list[bool]]:
        """
        Checks visibility for a batch of environments.

        NOTE: This is a placeholder implementation that calls the single-environment
        check in a loop. For maximum performance, this function's internals should be
        replaced with a true batch raycasting API if one is available in your simulator,
        as that would remove the final sequential bottleneck.
        """
        goal_visible_list = []
        camera_visible_list = []
        for env_id in env_ids:
            goal_visible, camera_visible = self.check_object_visibility(env_id)
            goal_visible_list.append(goal_visible)
            camera_visible_list.append(camera_visible)
        return goal_visible_list, camera_visible_list

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.move_agent(self.actions)
        
        # Lock camera object in place after agent moves
        # This prevents physics from moving the camera
        env_ids = self._agent._ALL_INDICES
        
        # Get current camera pose (should be what we set during reset)
        camera_pose = self._camera_obj.data.root_pose_w[env_ids].clone()
        
        # Re-write camera pose with zero velocity to keep it locked
        self._camera_obj.write_root_pose_to_sim(camera_pose[:, :7], env_ids)
        self._camera_obj.write_root_velocity_to_sim(
            torch.zeros_like(self._camera_obj.data.root_vel_w[env_ids]), env_ids)

    def _save_images(self, env_ids: torch.Tensor, rgb_data: torch.Tensor, 
                     depth_data: torch.Tensor, camera_pov_data: torch.Tensor = None) -> List[int]:
        """
        Save RGB, depth, and camera POV images for environments.
        
        Args:
            env_ids: Tensor of environment IDs to save images for
            rgb_data: RGB image data from tiled camera
            depth_data: Depth image data from distance camera
            camera_pov_data: Optional camera POV data from occlusion camera
            
        Returns:
            List of environment IDs that have reached 20 images
        """
        import os
        import cv2

        current_folder_indices = [
            self.next_env_folder_idx + i for i in range(len(env_ids))
        ]
        envs_with_20_images = []

        for idx, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            
            folder_idx = current_folder_indices[idx]

            # Validate visibility label exists
            if folder_idx not in self.env_visibility_labels:
                raise RuntimeError(
                    f"CRITICAL ERROR: No visibility label found for folder_idx {folder_idx}!")

            visibility_label = self.env_visibility_labels[folder_idx]

            if visibility_label not in ["Yes", "No"]:
                raise RuntimeError(
                    f"CRITICAL ERROR: Invalid visibility label '{visibility_label}' for folder_idx {folder_idx}!")

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

            # Count existing images
            num_rgb_images = len([
                f for f in os.listdir(rgb_env_folder)
                if f.endswith('.png') and not f.startswith('cam_pov')
            ])
            num_depth_images = len([
                f for f in os.listdir(depth_env_folder) if f.endswith('.png')
            ])

            # Check if already have 20 images
            if num_rgb_images >= 20 and num_depth_images >= 20:
                envs_with_20_images.append(env_id_item)
                continue

            # Check if both objects are in FOV and recent action was teleport
            goal_visible, camera_visible = self.check_object_visibility(env_id_item)
            both_in_fov = goal_visible and camera_visible
            recent_action = int(self.actions[env_id_item].item()) if hasattr(
                self, 'actions') else -1

            # Only save if both in FOV and action was teleport (6)
            if both_in_fov and recent_action == 6:
                # Save RGB image
                rgb_filename = f"{rgb_env_folder}/image_{num_rgb_images:04d}.png"
                rgb_img = rgb_data[env_id_item, :, :, :3]

                if rgb_img.max() <= 1.0:
                    rgb_np = (rgb_img.cpu().numpy() * 255.0).astype(np.uint8)
                else:
                    rgb_np = rgb_img.cpu().numpy().astype(np.uint8)
                
                cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

                # Save depth image
                depth_filename = f"{depth_env_folder}/image_{num_depth_images:04d}.png"
                depth_img = depth_data[env_id_item, :, :, :]
                depth_np = depth_img.cpu().numpy()
                
                # Handle infinite values
                depth_np[np.isinf(depth_np)] = depth_np[~np.isinf(depth_np)].max(
                ) if depth_np[~np.isinf(depth_np)].size > 0 else 0

                # Normalize depth
                if depth_np.max() > depth_np.min():
                    depth_normalized = ((depth_np - depth_np.min()) /
                                        (depth_np.max() - depth_np.min()) *
                                        255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)

                cv2.imwrite(depth_filename, depth_normalized)

                # Save camera POV if enabled and not already saved
                if self.save_camera_pov and camera_pov_data is not None:
                    cam_pov_filename = f"{cam_env_folder}/cam_pov.png"
                    if not os.path.exists(cam_pov_filename):
                        cam_pov_img = camera_pov_data[env_id_item, :, :, :3]

                        if cam_pov_img.max() <= 1.0:
                            cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(np.uint8)
                        else:
                            cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)

                        cv2.imwrite(cam_pov_filename, cv2.cvtColor(cam_pov_np, cv2.COLOR_RGB2BGR))
                
                # Save environment configuration once (when first image is saved)
                if num_rgb_images == 0:
                    self._save_env_config_to_json(env_id, folder_idx)

        return envs_with_20_images

    def _check_all_envs_have_images(self) -> bool:
        """
        Check if all current environments have 20 images saved.
        
        Returns:
            True if all environments have 20+ images, False otherwise
        """
        import os
        
        # Get current folder indices for all environments
        env_ids = self._agent._ALL_INDICES
        current_folder_indices = [
            self.next_env_folder_idx + i for i in range(len(env_ids))
        ]
        
        for folder_idx in current_folder_indices:
            # Get visibility label
            if folder_idx not in self.env_visibility_labels:
                return False
            
            visibility_label = self.env_visibility_labels[folder_idx]
            
            # Check RGB images
            rgb_env_folder = f"{self.base_path}/RGB/{visibility_label}/env_{folder_idx}"
            if not os.path.exists(rgb_env_folder):
                return False
            
            num_rgb_images = len([
                f for f in os.listdir(rgb_env_folder)
                if f.endswith('.png') and not f.startswith('cam_pov')
            ])
            
            if num_rgb_images < self.imgs_saved_per_env:
                return False
        
        return True

    def _get_observations(self) -> dict:
        if not self._reset_called:
            raise RuntimeError(
                "ERROR: _get_observations called before _reset_idx! "
                "Environment initialization must call reset first.")

        # Update cameras
        for _ in range(3):
            self.sim.step()
            self._rgb_tiled_camera.update(self.sim.cfg.dt)
            self._distance_tiled_camera.update(self.sim.cfg.dt)
            self._occlusion_camera.update(self.sim.cfg.dt)

        # Get camera data
        env_ids = self._agent._ALL_INDICES
        rgb_data = self._rgb_tiled_camera.data.output["rgb"]
        depth_data = self._distance_tiled_camera.data.output["distance_to_camera"]
        camera_pov_data = self._occlusion_camera.data.output["semantic_segmentation"] if self.save_camera_pov else None

        # Save images
        self._save_images(env_ids, rgb_data, depth_data, camera_pov_data)

        # Check if all environments have 20 images
        if self._check_all_envs_have_images():
            import os
            
            yes_rgb_path = f"{self.base_path}/RGB/Yes"
            no_rgb_path = f"{self.base_path}/RGB/No"
            total_rgb_count = 0

            # Count total RGB images across all environments
            for base_path in [yes_rgb_path, no_rgb_path]:
                if os.path.exists(base_path):
                    for env_folder in os.listdir(base_path):
                        env_folder_path = os.path.join(base_path, env_folder)
                        if os.path.isdir(env_folder_path):
                            rgb_images = [
                                f for f in os.listdir(env_folder_path) if
                                f.endswith('.png') and not f.startswith('cam_pov')
                            ]
                            total_rgb_count += len(rgb_images)

            # Check if max images reached
            if total_rgb_count >= self.max_rgb_images:
                print(f"\n{'='*80}")
                print(f"✅ MAX RGB IMAGES REACHED: {total_rgb_count}/{self.max_rgb_images}")
                print(f"{'='*80}\n")
                self.close()
                exit(0)
            else:
                # Print progress every 200 images
                if total_rgb_count > 0 and total_rgb_count % 200 == 0:
                    print(f"\n{'='*80}")
                    print(f"📊 PROGRESS UPDATE: {total_rgb_count}/{self.max_rgb_images} RGB images collected")
                    print(f"{'='*80}\n")

            # Reset ALL environments with new folder indices
            print(f"\n{'='*80}")
            print(f"🔄 ALL ENVIRONMENTS HAVE 20 IMAGES - RESETTING ALL")
            print(f"{'='*80}\n")
            
            self.next_env_folder_idx += len(env_ids)
            self._reset_idx(env_ids.tolist())

        # Prepare observations
        rgb_data = rgb_data.permute(0, 3, 1, 2)[:, :3, :, :]
        observations = {"policy": rgb_data.clone()}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = (self.episode_length_buf >= self.max_episode_length)
        return terminated, time_outs

    # Assume self, and other necessary class members are defined elsewhere.
    def _reset_idx(self, env_ids: Sequence[int] | None, randomize_objects: bool = True) -> None:
        """
        Reset environments by selectively retrying only the failed ones, with fixed visibility labels.
        """
        MIN_VALID_VIEWPOINTS = 20
        max_inner_attempts = 50  # Attempts for the subset in each outer loop
        max_full_attempts = 10   # Max number of times to gather failures and retry them

        if env_ids is None:
            env_ids = self._agent._ALL_INDICES
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_envs = len(env_ids)
        original_folder_indices = [self.next_env_folder_idx + i for i in range(num_envs)]

        # --- Step 1: Assign and LOCK visibility categories for all environments ONCE ---
        if self.verbose >= 1:
            print("🔒 Locking in visibility assignments for all environments...")
        num_in_view = num_envs // 2
        num_occluded = num_envs // 4
        num_outside_fov = num_envs - num_in_view - num_occluded
        visibility_categories = (["in_view"] * num_in_view +
                                ["occluded"] * num_occluded +
                                ["outside_fov"] * num_outside_fov)
        random.shuffle(visibility_categories)

        for i in range(num_envs):
            folder_idx = original_folder_indices[i]
            category = visibility_categories[i]
            if category == "in_view":
                self.env_visibility_labels[folder_idx] = "Yes"
                self.env_visibility_reasons[folder_idx] = "in_view"
            elif category == "occluded":
                self.env_visibility_labels[folder_idx] = "No"
                self.env_visibility_reasons[folder_idx] = "occluded"
            else:
                self.env_visibility_labels[folder_idx] = "No"
                self.env_visibility_reasons[folder_idx] = "outside_fov"
        self._save_visibility_labels()
        
        # Master mask to track which envs are not yet successfully configured
        batch_needs_retry = torch.ones(num_envs, dtype=torch.bool, device=self.device)

        # --- Step 2: MASTER LOOP to iteratively fix failing environments ---
        for full_attempt in range(max_full_attempts):
            if not batch_needs_retry.any():
                print(f"\n🎉 SUCCESS: All {num_envs} environments correctly configured!")
                break

            global_indices_to_fix = torch.where(batch_needs_retry)[0]

            print(f"\n{'='*40}")
            print(f"🚀 Master Attempt {full_attempt + 1}/{max_full_attempts}: Targeting {batch_needs_retry.sum().item()} environment(s).")
            print(f"{'='*40}")

            # Get the specific envs and their properties for this retry attempt
            retry_env_ids = env_ids[global_indices_to_fix]
            retry_folder_indices = [original_folder_indices[i] for i in global_indices_to_fix]
            retry_visibility_categories = [visibility_categories[i] for i in global_indices_to_fix]

            # --- Inner loop to reset the CURRENT SUBSET of failed envs ---
            subset_needs_retry = torch.ones_like(retry_env_ids, dtype=torch.bool)
            for inner_attempt in range(max_inner_attempts):
                if not subset_needs_retry.any():
                    break # All envs in this subset are now fixed

                # Further narrow down to only those in the subset that still need a reset
                ids_to_reset_now = retry_env_ids[subset_needs_retry]
                folders_to_reset_now = [retry_folder_indices[i] for i, needs_retry in enumerate(subset_needs_retry) if needs_retry]
                vis_to_reset_now = [retry_visibility_categories[i] for i, needs_retry in enumerate(subset_needs_retry) if needs_retry]

                self._reset_idx_internal(
                    ids_to_reset_now,
                    randomize_objects,
                    folder_indices=folders_to_reset_now,
                    visibility_categories=vis_to_reset_now
                )

                # Validate the subset that was just reset
                for i, needs_retry in enumerate(subset_needs_retry):
                    if needs_retry:
                        is_valid, reason = self._validate_env_state(retry_env_ids[i], retry_folder_indices[i], MIN_VALID_VIEWPOINTS)
                        if is_valid:
                            subset_needs_retry[i] = False # This one is good now
                        elif self.verbose >= 1 and inner_attempt == max_inner_attempts - 1:
                            # Print failure reason only on the last attempt to avoid spam
                            env_id_item = retry_env_ids[i].item()
                            print(f"  ❌ Env {env_id_item} (folder {retry_folder_indices[i]}) failed inner reset: {reason}")
            
            # --- Step 3: Update the master mask after inner loop finishes ---
            # Re-validate the entire subset one last time to be sure
            for i, global_idx in enumerate(global_indices_to_fix):
                is_valid, _ = self._validate_env_state(retry_env_ids[i], retry_folder_indices[i], MIN_VALID_VIEWPOINTS)
                if is_valid:
                    batch_needs_retry[global_idx] = False # Mark as successfully configured

        # --- Step 4: Final validation and reporting ---
        if self.verbose >= 1:
            print("\n🔍 Performing final comprehensive validation on the entire batch...")
        
        # Force camera updates before final check
        for _ in range(5):
            self.sim.step()
            self._occlusion_camera.update(self.sim.cfg.dt)
            
        validation_results = self._perform_final_validation(env_ids, original_folder_indices)
        self._print_validation_summary(validation_results, num_envs)

        if batch_needs_retry.any():
            num_failed = batch_needs_retry.sum().item()
            raise RuntimeError(f"CRITICAL FAILURE: {num_failed} environment(s) could not be configured correctly after {max_full_attempts} master attempts.")

        self._reset_called = True
    # Helper methods (to be part of your class) for better organization
    def _validate_env_state(self, env_id, folder_idx, min_viewpoints):
        """Validates a single environment's state after a reset attempt."""
        env_id_item = env_id.item()

        # Check viewpoint count
        if (self.valid_viewpoint_poses is None or
                env_id_item >= len(self.valid_viewpoint_poses) or
                self.valid_viewpoint_poses[env_id_item] is None or
                len(self.valid_viewpoint_poses[env_id_item]) < min_viewpoints):
            num_poses = 0 if self.valid_viewpoint_poses is None or self.valid_viewpoint_poses[env_id_item] is None else len(self.valid_viewpoint_poses[env_id_item])
            return False, f"insufficient viewpoints: {num_poses}/{min_viewpoints}"

        # Validate occlusion matches label
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
        """Performs a final, comprehensive validation check on all environments."""
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
        """
        Prints a formatted and detailed summary of the validation results.

        Args:
            validation_results: A list of dictionaries, where each dict contains
                                the validation info for one environment.
            num_envs: The total number of environments in the batch.
        """
        # Calculate overall statistics
        num_valid = sum(1 for r in validation_results if r["valid"])
        num_invalid = len(validation_results) - num_valid

        # Print main header
        print(f"\n{'='*80}")
        print(f"📊 VALIDATION SUMMARY: {num_valid}/{len(validation_results)} environments correct")
        print(f"{'='*80}")

        # --- Group results by their assigned category ---
        in_view_results = [r for r in validation_results if r["reason"] == "in_view"]
        occluded_results = [r for r in validation_results if r["reason"] == "occluded"]
        outside_fov_results = [r for r in validation_results if r["reason"] == "outside_fov"]

        # --- Print details for IN_VIEW environments ---
        if in_view_results:
            print(f"\n📍 IN_VIEW ({len(in_view_results)} envs - {len(in_view_results)/num_envs*100:.0f}%):")
            for r in in_view_results:
                # Use .get() for 'actual_occluded' in case it's not present (e.g., outside_fov)
                print(f"  {r['status']} Env {r['env_id']} (folder {r['folder_idx']}): "
                    f"Label={r['label']}, Occluded={r.get('actual_occluded', 'N/A')}, Occlusion Expected? = {r.get('expected_occluded', 'N/A')}")

        # --- Print details for OCCLUDED environments ---
        if occluded_results:
            print(f"\n🚫 OCCLUDED ({len(occluded_results)} envs - {len(occluded_results)/num_envs*100:.0f}%):")
            for r in occluded_results:
                print(f"  {r['status']} Env {r['env_id']} (folder {r['folder_idx']}): "
                    f"Label={r['label']}, Occluded={r.get('actual_occluded', 'N/A')}, Occlusion Expected? = {r.get('expected_occluded', 'N/A')}")

        # --- Print details for OUTSIDE_FOV environments ---
        if outside_fov_results:
            print(f"\n👁️  OUTSIDE_FOV ({len(outside_fov_results)} envs - {len(outside_fov_results)/num_envs*100:.0f}%):")
            for r in outside_fov_results:
                print(f"  {r['status']} Env {r['env_id']} (folder {r['folder_idx']}): Label={r['label']}")

        # --- Print a separate, final summary of any failures ---
        if num_invalid > 0:
            print(f"\n{'='*80}")
            print(f"⚠️  WARNING: {num_invalid} environment(s) have mismatched occlusion status!")
            for r in validation_results:
                if not r["valid"]:
                    print(f"  ❌ Env {r['env_id']} (folder {r['folder_idx']}): "
                        f"Expected {'occluded' if r['expected_occluded'] else 'visible'}, "
                        f"got {'occluded' if r['actual_occluded'] else 'visible'}")
        
        print(f"\n{'='*80}\n")

    def _reset_idx_internal(self, env_ids: Sequence[int] | None, randomize_objects: bool = True,
                            folder_indices: List[int] = None, visibility_categories: List[str] = None) -> None:
        """Internal reset logic - spawn objects and generate viewpoints (PARALLELIZED)."""
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_envs = len(env_ids)
        if folder_indices is None:
            folder_indices = [self.next_env_folder_idx + i for i in range(num_envs)]

        # Categories should ALWAYS be provided - never generate new ones here
        if visibility_categories is None:
            raise RuntimeError("visibility_categories must be provided to _reset_idx_internal!")

        # Reset used viewpoint tracking
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            self.used_viewpoint_indices[env_id_item].clear()

        # Labels should already be set by _reset_idx - just verify they exist
        for i in range(num_envs):
            global_folder_idx = folder_indices[i]
            if global_folder_idx not in self.env_visibility_labels:
                raise RuntimeError(f"Labels not set for folder {global_folder_idx} before _reset_idx_internal!")

        self.viewpoint_pose_counter[env_ids] = 0
        super()._reset_idx(env_ids)

        device = self._agent.device
        safe_x_range = self.center_to_boundary - 1.5
        safe_y_range = self.center_to_boundary - 1.5

        # Get default states for all environments
        goal_default_state = self._goal.data.default_root_state[env_ids].clone()
        agent_default_state = self._agent.data.default_root_state[env_ids].clone()
        camera_obj_default_state = self._camera_obj.data.default_root_state[env_ids].clone()
        vpt_obj_default_state = self._vpt_objects.data.default_object_state[env_ids].clone()

        # Batch spawn objects with retry logic - ALL ENVIRONMENTS AT ONCE
        max_spawn_attempts = 500
        envs_need_spawn_retry = torch.ones(num_envs, dtype=torch.bool, device=device)
        
        for spawn_attempt in range(max_spawn_attempts):
            if not envs_need_spawn_retry.any():
                break
            
            # Process all environments needing retry together
            retry_mask = envs_need_spawn_retry.clone()
            batch_size = retry_mask.sum().item()
            retry_indices = torch.where(retry_mask)[0]
            
            # Generate random positions for ALL retry environments in one batch
            goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
            camera_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
            agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
            vpt_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, self.num_objs, 2), device)
            
            # Apply positions for all retry environments
            for batch_idx, env_idx in enumerate(retry_indices):
                env_id = env_ids[env_idx]
                env_origin = self.scene.env_origins[env_id]
                
                # Goal position
                goal_default_state[env_idx, 0] = env_origin[0] + goal_offsets[batch_idx, 0]
                goal_default_state[env_idx, 1] = env_origin[1] + goal_offsets[batch_idx, 1]
                goal_default_state[env_idx, 2] = self._goal.data.default_root_state[env_id, 2] + env_origin[2]
                
                # Camera position and orientation
                camera_obj_default_state[env_idx, 0] = env_origin[0] + camera_offsets[batch_idx, 0]
                camera_obj_default_state[env_idx, 1] = env_origin[1] + camera_offsets[batch_idx, 1]
                
                direction_to_goal = goal_default_state[env_idx, :2] - camera_obj_default_state[env_idx, :2]
                yaw = torch.atan2(direction_to_goal[1], direction_to_goal[0]) - math.radians(90)
                roll = torch.tensor(-math.radians(self.agent_camera_pitch), device=device)
                quaternion = quat_from_euler_xyz(roll, torch.tensor(0, device=device), yaw)
                camera_obj_default_state[env_idx, 3:7] = quaternion
                
                # Agent position
                agent_default_state[env_idx, 0] = env_origin[0] + agent_offsets[batch_idx, 0]
                agent_default_state[env_idx, 1] = env_origin[1] + agent_offsets[batch_idx, 1]
                
                # VPT objects positions
                vpt_obj_default_state[env_idx, :, 0] = env_origin[0] + vpt_offsets[batch_idx, :, 0]
                vpt_obj_default_state[env_idx, :, 1] = env_origin[1] + vpt_offsets[batch_idx, :, 1]
                vpt_obj_default_state[env_idx, :, 2] = self._vpt_objects.data.default_object_state[env_id, :, 2] + env_origin[2]
            
            # Vectorized distance validation - check ALL retry environments
            for batch_idx, env_idx in enumerate(retry_indices):
                env_id = env_ids[env_idx]
                
                # Check distance constraints (vectorized per environment)
                agent_pos = agent_default_state[env_idx, :2]
                goal_pos = goal_default_state[env_idx, :2]
                camera_pos = camera_obj_default_state[env_idx, :2]
                
                # agent_distance_from_goal = torch.norm(agent_pos - goal_pos)
                camera_goal_distance = torch.norm(camera_pos - goal_pos)
                
                if camera_goal_distance > 4.0 or camera_goal_distance < 0.5:
                    continue  # Keep retry flag True
                
                # Check VPT distances (vectorized)
                vpt_positions = vpt_obj_default_state[env_idx, :, :2]
                camera_distances_from_vpt = torch.norm(camera_pos.unsqueeze(0) - vpt_positions, dim=1)
                
                if not torch.all(camera_distances_from_vpt >= 0.5):
                    continue  # Keep retry flag True
                
                # Passed distance checks - clear retry flag
                envs_need_spawn_retry[env_idx] = False
            
            # Get valid environments from this batch
            valid_mask = retry_mask & ~envs_need_spawn_retry
            if not valid_mask.any():
                continue
            
            valid_indices = torch.where(valid_mask)[0]
            valid_env_ids = env_ids[valid_indices]
            
            # Batch write poses for ALL valid environments at once
            self._goal.write_root_pose_to_sim(goal_default_state[valid_indices, :7], valid_env_ids)
            self._goal.write_root_velocity_to_sim(torch.zeros((len(valid_env_ids), 6), device=device), valid_env_ids)
            
            self._camera_obj.write_root_pose_to_sim(camera_obj_default_state[valid_indices, :7], valid_env_ids)
            self._camera_obj.write_root_velocity_to_sim(torch.zeros((len(valid_env_ids), 6), device=device), valid_env_ids)
            
            self._agent.write_root_pose_to_sim(agent_default_state[valid_indices, :7], valid_env_ids)
            self._agent.write_root_velocity_to_sim(torch.zeros((len(valid_env_ids), 6), device=device), valid_env_ids)
            
            self._vpt_objects.write_object_pose_to_sim(vpt_obj_default_state[valid_indices, :, :7], valid_env_ids)
            self._vpt_objects.write_object_velocity_to_sim(torch.zeros((len(valid_env_ids), self.num_objs, 6), device=device), valid_env_ids)
            
            # Single simulation step for ALL valid environments together
            for _ in range(2):
                self.sim.step()
            
            # Batch validate Z-bounds for all valid environments
            goal_new_pos = self._goal.data.root_pos_w[valid_env_ids]
            camera_new_pos = self._camera_obj.data.root_pos_w[valid_env_ids]
            agent_new_pos = self._agent.data.root_pos_w[valid_env_ids]
            vpt_new_pos = self._vpt_objects.data.object_pos_w[valid_env_ids]
            
            # Check Z-bounds per environment
            for local_idx, env_idx in enumerate(valid_indices):
                z_valid = (
                    (goal_new_pos[local_idx, 2] >= 0.0) and (goal_new_pos[local_idx, 2] <= 1.0) and
                    (camera_new_pos[local_idx, 2] >= 0.0) and (camera_new_pos[local_idx, 2] <= 1.0) and
                    (agent_new_pos[local_idx, 2] >= 0.0) and (agent_new_pos[local_idx, 2] <= 1.0) and
                    torch.all((vpt_new_pos[local_idx, :, 2] >= 0.0) & (vpt_new_pos[local_idx, :, 2] <= 1.0))
                )
                
                if not z_valid:
                    envs_need_spawn_retry[env_idx] = True
            
            # Batch update occlusion cameras for environments that passed Z-check
            final_valid_mask = valid_mask & ~envs_need_spawn_retry
            if not final_valid_mask.any():
                continue
            
            final_valid_indices = torch.where(final_valid_mask)[0]
            final_valid_env_ids = env_ids[final_valid_indices]
            
            camera_positions = camera_obj_default_state[final_valid_indices, :3]
            camera_orientations = camera_obj_default_state[final_valid_indices, 3:7]
            
            # Apply 90-degree rotation (vectorized)
            theta_left = math.pi / 2
            half_theta_left = theta_left / 2
            left_90_quat = torch.tensor([math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)], device=device)
            rotated_orientations = math_utils.quat_mul(camera_orientations, left_90_quat.unsqueeze(0).expand(len(final_valid_env_ids), -1))
            
            self._occlusion_camera.set_world_poses(
                positions=camera_positions,
                orientations=rotated_orientations,
                env_ids=final_valid_env_ids.tolist(),
                convention="world")
            
            # Single camera update for ALL valid environments
            for _ in range(10):
                self.sim.step()
                self._occlusion_camera.update(self.sim.cfg.dt)
                self._goal.update(self.sim.cfg.dt)
            
            # Batch validate occlusion status for all final valid environments
            for local_idx, env_idx in enumerate(final_valid_indices):
                env_id = final_valid_env_ids[local_idx]
                visibility_category = visibility_categories[env_idx]
                
                camera_pos = camera_positions[local_idx]
                goal_pos = goal_default_state[env_idx, :3]
                
                is_occluded = self._check_occlusion_raycast(camera_pos, goal_pos, env_id)
                
                # Validate occlusion matches category
                occlusion_valid = True
                if visibility_category == "in_view" and is_occluded:
                    occlusion_valid = False
                elif visibility_category == "occluded" and not is_occluded:
                    occlusion_valid = False
                
                if not occlusion_valid:
                    envs_need_spawn_retry[env_idx] = True
                elif visibility_category == "outside_fov":
                    # Point camera away for outside_fov
                    direction_to_goal = goal_pos[:2] - camera_pos[:2]
                    yaw = torch.atan2(direction_to_goal[1], direction_to_goal[0]) - math.radians(90)
                    yaw_away = yaw + math.pi
                    roll = torch.tensor(-math.radians(self.agent_camera_pitch), device=device)
                    quaternion_away = quat_from_euler_xyz(roll, torch.tensor(0, device=device), yaw_away)
                    camera_obj_default_state[env_idx, 3:7] = quaternion_away
                    self._camera_obj.write_root_pose_to_sim(camera_obj_default_state[env_idx:env_idx+1, :7], env_id.unsqueeze(0))

        # Set random agent orientations (vectorized) - ALL environments at once
        random_yaw_agent = sample_uniform(0, 2 * math.pi, (num_envs,), device)
        agent_default_state[:, 3] = torch.cos(random_yaw_agent / 2)
        agent_default_state[:, 4] = 0.0
        agent_default_state[:, 5] = 0.0
        agent_default_state[:, 6] = torch.sin(random_yaw_agent / 2)
        self._agent.write_root_pose_to_sim(agent_default_state[:, :7], env_ids)

        # Generate viewpoints in parallel for ALL environments at once
        all_valid_points = self.generate_valid_circle_points(env_ids=env_ids, angle_step=2.0, max_attempts=1000)

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
                self.valid_viewpoint_poses[env_id_item] = valid_points_3d if valid_points_3d.shape[0] >= 20 else torch.zeros((0, 3), device=device)
            else:
                self.valid_viewpoint_poses[env_id_item] = torch.zeros((0, 3), device=device)

    def _write_and_validate_pose(self, obj, pose, velocity, env_ids, obj_name="object", is_collection=False):
        """Write pose, simulate 2 steps, validate Z bounds and movement."""
        # Store old positions
        if is_collection:
            old_pos = obj.data.object_pos_w[env_ids].clone()
        else:
            old_pos = obj.data.root_pos_w[env_ids].clone()
        
        # Write pose and velocity
        if is_collection:
            obj.write_object_pose_to_sim(pose, env_ids)
            obj.write_object_velocity_to_sim(velocity, env_ids)
        else:
            obj.write_root_pose_to_sim(pose, env_ids)
            obj.write_root_velocity_to_sim(velocity, env_ids)
        
        # Simulate 2 steps
        for _ in range(2):
            self.sim.step()
        
        # Get new positions
        if is_collection:
            new_pos = obj.data.object_pos_w[env_ids]
        else:
            new_pos = obj.data.root_pos_w[env_ids]
        
        # Check Z bounds [0, 1]
        z_valid = torch.all((new_pos[..., 2] >= 0.0) & (new_pos[..., 2] <= 1.0))
        
        # Check objects moved (allow small tolerance)
        moved = torch.norm(new_pos[..., :2] - old_pos[..., :2], dim=-1).max() > 0.01
        
        return z_valid and moved

    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)
        return obs, rewards, terminated, truncated, info

    def _check_occlusion_raycast(self, camera_pos, goal_pos, env_id, camera=None):
        if camera is None:
            camera = self._occlusion_camera

        # VALIDATION: Ensure occlusion camera is at the same position as camera object
        camera_obj_pos = self._camera_obj.data.root_pos_w[env_id]
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_id]
        occlusion_cam_pos = camera.data.pos_w[env_id]
        occlusion_cam_quat = camera.data.quat_w_world[env_id]
        
        # Calculate position and orientation differences
        pos_diff = torch.norm(camera_obj_pos - occlusion_cam_pos).item()
        
        # If positions don't match (threshold: 0.01 units), force update
        if pos_diff > 0.01:
            print(f"⚠️  Env {env_id}: Occlusion camera misaligned! Distance: {pos_diff:.4f}")
            print(f"    Camera Obj: {camera_obj_pos.cpu().numpy()}")
            print(f"    Occlusion Cam: {occlusion_cam_pos.cpu().numpy()}")
            print(f"    Forcing update...")
            
            # Create 90-degree left rotation quaternion
            device = camera_obj_pos.device
            theta_left = math.pi / 2
            half_theta_left = theta_left / 2
            left_90_quat = torch.tensor(
                [math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)],
                device=device)
            
            # Apply rotation to camera orientation
            rotated_orientation = math_utils.quat_mul(
                camera_obj_quat.unsqueeze(0),
                left_90_quat.unsqueeze(0)).squeeze(0)
            
            # Force update occlusion camera position and orientation
            camera.set_world_poses(
                positions=camera_obj_pos.unsqueeze(0),
                orientations=rotated_orientation.unsqueeze(0),
                env_ids=[env_id.item() if torch.is_tensor(env_id) else env_id],
                convention="world")
            
            # Verify the fix worked
            new_occlusion_cam_pos = camera.data.pos_w[env_id]
            new_pos_diff = torch.norm(camera_obj_pos - new_occlusion_cam_pos).item()
            print(f"    ✓ After fix: Distance = {new_pos_diff:.4f}")

        GOAL_THRESHOLD = 20

        sem_img = camera.data.output["semantic_segmentation"][env_id]

        r = sem_img[:, :, 0]
        g = sem_img[:, :, 1]
        b = sem_img[:, :, 2]

        red_mask = ((r >= 0.95) & (g <= 0.05) & (b <= 0.05))
        red_count = red_mask.sum().item()

        if red_count >= GOAL_THRESHOLD:
            return False
        else:
            return True

    def check_object_visibility(self, env_id: int, print_agent_state: bool = False) -> tuple[bool, bool]:
        GOAL_THRESHOLD = 20
        CAMERA_THRESHOLD = 10

        sem_img = self._tiled_camera.data.output["semantic_segmentation"][env_id]

        r = sem_img[:, :, 0]
        g = sem_img[:, :, 1]
        b = sem_img[:, :, 2]

        red_mask = ((r >= 0.95) & (g <= 0.05) & (b <= 0.05))
        red_count = red_mask.sum().item()

        green_mask = ((r <= 0.05) & (g >= 0.95) & (b <= 0.05))
        green_count = green_mask.sum().item()

        goal_visible = red_count >= GOAL_THRESHOLD
        camera_visible = green_count >= CAMERA_THRESHOLD

        return goal_visible, camera_visible

    def _calculate_optimal_radius(self, midpoint: torch.Tensor, camera_pos: torch.Tensor,
                                   goal_pos: torch.Tensor, horizontal_fov_degrees: float = 35.0) -> float:
        dist_to_camera = torch.norm(camera_pos[:2] - midpoint[:2]).item()
        dist_to_goal = torch.norm(goal_pos[:2] - midpoint[:2]).item()
        half_span = max(dist_to_camera, dist_to_goal)

        half_fov_radians = math.radians(horizontal_fov_degrees / 2.0)
        optimal_radius = half_span / math.tan(half_fov_radians)

        radius_with_margin = optimal_radius * 1

        return radius_with_margin

    def _get_circle_point(self, center: torch.Tensor, radius: float, theta_degrees: float,
                          device: torch.device, env_id: int = None) -> torch.Tensor:
        theta_radians = math.radians(theta_degrees)

        x = center[0] + radius * math.cos(theta_radians)
        y = center[1] + radius * math.sin(theta_radians)

        return torch.tensor([x, y], device=device, dtype=torch.float32)

    def _is_point_valid(self, point: torch.Tensor, env_id: int, min_obstacle_distance: float = 0.4,
                        min_camera_target_distance: float = 2.0, print_details: bool = False, check_agent_fov=False) -> bool:
        """Single-point validation wrapper for backward compatibility."""
        result = self._is_point_valid_batch(
            points=point.unsqueeze(0),
            env_ids=torch.tensor([env_id], dtype=torch.long, device=point.device),
            min_obstacle_distance=min_obstacle_distance,
            min_camera_target_distance=min_camera_target_distance,
            check_agent_fov=check_agent_fov
        )
        return result[0].item()

    def _is_point_valid_batch(self, points: torch.Tensor, env_ids: torch.Tensor, 
                              min_obstacle_distance: float = 0.5,
                              min_camera_vpt_distance: float = 1.0,
                              min_camera_target_distance: float = 2.0,
                              check_agent_fov: bool = False) -> torch.Tensor:
        """
        Batch validate multiple points across multiple environments in parallel.
        
        Args:
            points: Tensor of shape (N, 2) containing 2D points to validate
            env_ids: Tensor of shape (N,) containing environment IDs for each point
            min_obstacle_distance: Minimum distance from VPT objects
            min_camera_target_distance: Minimum distance from camera and goal
            check_agent_fov: Whether to validate agent FOV (requires simulation steps)
            
        Returns:
            Boolean tensor of shape (N,) indicating validity of each point
        """
        device = points.device
        num_points = points.shape[0]
        
        # Initialize validity mask (all True initially)
        valid_mask = torch.ones(num_points, dtype=torch.bool, device=device)
        
        # 1. BATCH BOUNDARY CHECK
        env_origins = self.scene.env_origins[env_ids, :2]
        boundary_limit = self.center_to_boundary.item() if torch.is_tensor(
            self.center_to_boundary) else self.center_to_boundary
        
        min_bounds = env_origins - boundary_limit
        max_bounds = env_origins + boundary_limit
        
        # Check if points are within boundaries
        in_bounds = torch.all((points >= min_bounds) & (points <= max_bounds), dim=1)
        # Update valid_mask to only keep points within bounds
        valid_mask = valid_mask & in_bounds
        
        # Early exit if no points pass boundary check
        if not valid_mask.any():
            print(f"Early exit after boundary check: {valid_mask.sum().item()}/{num_points} valid")
            return valid_mask
        
        # 2. BATCH VPT OBJECTS DISTANCE CHECK
        # Get VPT positions for all environments
        vpt_positions = self._vpt_objects.data.object_pos_w[env_ids, :, :2]  # (N, num_objs, 2)
        
        # Calculate distances from each point to all VPT objects in its environment
        # Shape: (N, num_objs)
        distances_to_vpt = torch.norm(
            points.unsqueeze(1) - vpt_positions, 
            dim=2
        )
        
        # Check if any VPT object is too close
        min_vpt_distances = distances_to_vpt.min(dim=1)[0]
        valid_mask &= (min_vpt_distances >= min_obstacle_distance)
        
        # Early exit if no points pass VPT check
        if not valid_mask.any():
            print(f"Early exit after VPT distance check: {valid_mask.sum().item()}/{num_points} valid")
            return valid_mask
        
        # 2.5 BATCH VPT CAMERA DISTANCE CHECK
        camera_positions = self._camera_obj.data.root_pos_w[env_ids, :2]
        camera_distances = torch.norm(camera_positions.unsqueeze(1) - vpt_positions, dim=2)
        valid_mask &= (camera_distances >= min_camera_vpt_distance)
        
        # Early exit if no points pass VPT check
        if not valid_mask.any():
            print(f"Early exit after VPT-Camera distance check: {valid_mask.sum().item()}/{num_points} valid")
            return valid_mask
        
        # 3. BATCH CAMERA DISTANCE CHECK
        camera_positions = self._camera_obj.data.root_pos_w[env_ids, :2]
        camera_distances = torch.norm(points - camera_positions, dim=1)
        valid_mask &= (camera_distances >= min_camera_target_distance)
        
        # Early exit if no points pass camera check
        if not valid_mask.any():
            print(f"Early exit after Camera point distance check: {valid_mask.sum().item()}/{num_points} valid")
            return valid_mask
        
        # 4. BATCH GOAL DISTANCE CHECK
        goal_positions = self._goal.data.root_pos_w[env_ids, :2]
        goal_distances = torch.norm(points - goal_positions, dim=1)
        valid_mask &= (goal_distances >= min_camera_target_distance)
        
        # Early exit if no points pass camera check
        if not valid_mask.any():
            print(f"Early exit after Goal point distance check: {valid_mask.sum().item()}/{num_points} valid")
            return valid_mask
        
        # Skip FOV check if not requested
        if not check_agent_fov:
            return valid_mask
        
        # 5. BATCH AGENT FOV CHECK (requires simulation)
        # Only check points that passed geometric validation
        points_to_check = torch.where(valid_mask)[0]
        
        if points_to_check.numel() == 0:
            return valid_mask
        
        # Save current agent states for ALL environments being checked
        current_agent_pos = self._agent.data.root_pos_w[env_ids].clone()
        current_agent_quat = self._agent.data.root_quat_w[env_ids].clone()
        
        # Process in batches to avoid too many simulation steps
        batch_size = 20
        fov_valid = torch.zeros(num_points, dtype=torch.bool, device=device)
        
        for batch_start in range(0, len(points_to_check), batch_size):
            batch_end = min(batch_start + batch_size, len(points_to_check))
            batch_indices = points_to_check[batch_start:batch_end]
            batch_env_ids = env_ids[batch_indices]
            batch_points = points[batch_indices]
            
            # Create temporary agent poses
            temp_agent_pos = torch.zeros((len(batch_indices), 3), device=device)
            temp_agent_pos[:, :2] = batch_points
            temp_agent_pos[:, 2] = self._agent.data.default_root_state[batch_env_ids, 2]
            
            # Calculate orientations to look at midpoint between camera and goal
            camera_pos_3d = self._camera_obj.data.root_pos_w[batch_env_ids]
            goal_pos_3d = self._goal.data.root_pos_w[batch_env_ids]
            midpoints_2d = (camera_pos_3d[:, :2] + goal_pos_3d[:, :2]) / 2.0
            
            # Vectorized yaw calculation
            directions = midpoints_2d - batch_points
            yaws = torch.atan2(directions[:, 1], directions[:, 0])
            
            # Create quaternions
            temp_agent_quat = torch.zeros((len(batch_indices), 4), device=device)
            temp_agent_quat[:, 0] = torch.cos(yaws / 2)
            temp_agent_quat[:, 3] = torch.sin(yaws / 2)
            
            # Write temporary agent poses
            temp_poses = torch.cat([temp_agent_pos, temp_agent_quat], dim=1)
            self._agent.write_root_com_pose_to_sim(temp_poses, batch_env_ids)
            
            # Update tiled camera
            for _ in range(5):
                self.sim.step()
                self._tiled_camera.update(self.sim.cfg.dt)
                self._agent.update(self.sim.cfg.dt)
            
            # Check visibility for each point in batch
            for local_idx, global_idx in enumerate(batch_indices):
                env_id_item = env_ids[global_idx].item()
                goal_visible, camera_visible = self.check_object_visibility(env_id_item)
                fov_valid[global_idx] = goal_visible and camera_visible
        
        # Restore original agent states
        restore_poses = torch.cat([current_agent_pos, current_agent_quat], dim=1)
        self._agent.write_root_com_pose_to_sim(restore_poses, env_ids)
        
        # Combine geometric and FOV validity
        valid_mask &= fov_valid
        
        return valid_mask

    def generate_valid_circle_points(self, env_ids: torch.Tensor, angle_step: float = 2.0,
                                     max_attempts: int = 300) -> List[torch.Tensor]:
        """Generate valid viewpoint circle points in parallel for all environments."""
        device = self._agent.device
        angles = torch.arange(0, 360, angle_step, device=device)

        # Batch calculate optimal radii for all environments
        camera_positions = self._camera_obj.data.root_pos_w[env_ids]
        goal_positions = self._goal.data.root_pos_w[env_ids]
        midpoints = (camera_positions[:, :2] + goal_positions[:, :2]) / 2.0
        
        # Vectorized radius calculation
        dist_to_camera = torch.norm(camera_positions[:, :2] - midpoints, dim=1)
        dist_to_goal = torch.norm(goal_positions[:, :2] - midpoints, dim=1)
        half_spans = torch.max(dist_to_camera, dist_to_goal)
        
        horizontal_fov_degrees = 35.0
        half_fov_radians = math.radians(horizontal_fov_degrees / 2.0)
        min_radii = half_spans / math.tan(half_fov_radians)

        all_valid_points = []
        num_envs = len(env_ids)

        # FULLY PARALLEL PASS 1: Geometric validation for ALL environments at once
        # Generate all candidate points for all environments
        cos_angles = torch.cos(torch.deg2rad(angles))
        sin_angles = torch.sin(torch.deg2rad(angles))
        
        # Prepare batch data: (num_envs * num_angles, 2) for all points
        num_angles = len(angles)
        total_points = num_envs * num_angles
        
        # Expand midpoints and radii for all angles
        midpoints_expanded = midpoints.unsqueeze(1).expand(-1, num_angles, -1).reshape(total_points, 2)
        radii_expanded = min_radii.unsqueeze(1).expand(-1, num_angles).reshape(total_points)
        
        # Generate all points
        all_x = midpoints_expanded[:, 0] + radii_expanded * cos_angles.repeat(num_envs)
        all_y = midpoints_expanded[:, 1] + radii_expanded * sin_angles.repeat(num_envs)
        all_points_batch = torch.stack([all_x, all_y], dim=1)
        
        # Create environment IDs for each point
        env_ids_batch = env_ids.unsqueeze(1).expand(-1, num_angles).reshape(total_points)
        
        # BATCH GEOMETRIC VALIDATION (no FOV check)
        geometric_valid = self._is_point_valid_batch(
            points=all_points_batch,
            env_ids=env_ids_batch,
            check_agent_fov=False
        )
        
        # Reshape results back to (num_envs, num_angles)
        geometric_valid_per_env = geometric_valid.reshape(num_envs, num_angles)
        
        # PASS 2: Sample and validate with agent FOV (per environment)
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            
            # Get valid points for this environment
            valid_mask = geometric_valid_per_env[i]
            candidate_points = all_points_batch[i * num_angles:(i + 1) * num_angles][valid_mask]
            
            if self.verbose >= 2:
                print(f"  Env {env_id_item}: {valid_mask.sum().item()} geometric candidates")
            
            # Sample points for FOV check
            if len(candidate_points) > 50:
                indices = torch.randperm(len(candidate_points), device=device)[:50]
                points_to_check = candidate_points[indices]
            else:
                points_to_check = candidate_points
            
            if len(points_to_check) == 0:
                all_valid_points.append(torch.zeros((0, 2), device=device))
                continue
            
            # BATCH FOV VALIDATION for sampled points
            env_ids_for_check = torch.full((len(points_to_check),), env_id.item(), 
                                          dtype=torch.long, device=device)
            
            fov_valid = self._is_point_valid_batch(
                points=points_to_check,
                env_ids=env_ids_for_check,
                check_agent_fov=True
            )
            
            valid_points_tensor = points_to_check[fov_valid]
            # PASS 3: Filter points to ensure minimum displacement of 0.15 units
            MIN_DISPLACEMENT = 0.1
            MIN_REQUIRED_POINTS = 30
            
            if len(valid_points_tensor) > 0:
                # Sort points by angle to ensure diverse coverage
                center = midpoints[i]
                angles_rad = torch.atan2(
                    valid_points_tensor[:, 1] - center[1],
                    valid_points_tensor[:, 0] - center[0]
                )
                sorted_indices = torch.argsort(angles_rad)
                sorted_points = valid_points_tensor[sorted_indices]
                
                # Greedy selection with minimum displacement
                filtered_points = [sorted_points[0]]
                
                for point_idx in range(1, len(sorted_points)):
                    candidate = sorted_points[point_idx]
                    
                    # Check distance to all previously selected points
                    distances = torch.norm(
                        torch.stack(filtered_points) - candidate.unsqueeze(0),
                        dim=1
                    )
                    
                    # Only add if far enough from all existing points
                    if torch.all(distances >= MIN_DISPLACEMENT):
                        filtered_points.append(candidate)
                
                filtered_points_tensor = torch.stack(filtered_points) if len(filtered_points) > 0 else torch.zeros((0, 2), device=device)
                
                if self.verbose >= 2:
                    print(f"  Env {env_id_item}: After FOV: {len(valid_points_tensor)}, "
                          f"After displacement filter: {len(filtered_points_tensor)}")
                
                # Check if we have enough points
                if len(filtered_points_tensor) >= MIN_REQUIRED_POINTS:
                    all_valid_points.append(filtered_points_tensor)
                    if self.verbose >= 2:
                        print(f"  Env {env_id_item}: ✅ {len(filtered_points_tensor)} valid viewpoints (min: {MIN_REQUIRED_POINTS})")
                else:
                    # Not enough points - return empty
                    all_valid_points.append(torch.zeros((0, 2), device=device))
                    if self.verbose >= 1:
                        print(f"  Env {env_id_item}: ❌ Only {len(filtered_points_tensor)} viewpoints, need {MIN_REQUIRED_POINTS}")
            else:
                all_valid_points.append(torch.zeros((0, 2), device=device))
                if self.verbose >= 1:
                    print(f"  Env {env_id_item}: ❌ No valid viewpoints after FOV check")

        return all_valid_points


    def _save_visibility_labels(self):
        import json
        import os

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
                "total_environments": len(self.env_visibility_labels),
                "yes_count": sum(1 for v in self.env_visibility_labels.values() if v == "Yes"),
                "no_count": sum(1 for v in self.env_visibility_labels.values() if v == "No"),
                "by_reason": reason_counts,
                "next_env_folder_idx": self.next_env_folder_idx
            }
        }

        with open(self.visibility_labels_json_path, 'w') as f:
            json.dump(labels_data, f, indent=2)

    def _save_env_config_to_json(self, env_id: int, folder_idx: int):
        """
        Save complete environment configuration to JSON file using cfg data only.
        
        Saves:
        - Goal ball (position, orientation, spawn cfg)
        - Camera object (position, orientation, spawn cfg)
        - Agent (position, orientation, spawn cfg)
        - VPT objects (positions, orientations, spawn cfg per object)
        - Visibility label and reason
        - Valid viewpoint poses
        - Environment settings from cfg
        
        Args:
            env_id: Environment ID (index in batch)
            folder_idx: Folder index for this environment
        """
        import json
        import os
        
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
        
        # Get VPT objects info (positions + cfg spawn parameters per object)
        vpt_positions = self._vpt_objects.data.object_pos_w[env_id].cpu().numpy().tolist()
        vpt_orientations = self._vpt_objects.data.object_quat_w[env_id].cpu().numpy().tolist()
        
        # Get valid viewpoint poses
        valid_viewpoints = []
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
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
            "vpt_objects": {
                "count": self.num_objs,
                "objects": []
            },
            "valid_viewpoints": {
                "count": len(valid_viewpoints),
                "positions": valid_viewpoints
            }
        }
        
        # Add each VPT object with cfg metadata
        for obj_idx in range(self.num_objs):
            # Get spawn cfg for this VPT object from metadata
            obj_metadata = self.obstacle_metadata.get(obj_idx, {})
            
            vpt_obj = {
                "index": obj_idx,
                "position": vpt_positions[obj_idx],
                "orientation": vpt_orientations[obj_idx],
                "spawn_cfg": {
                    "shape": obj_metadata.get("shape", "cube"),
                    "size": obj_metadata.get("size", [0.3, 0.3, 0.3]),
                    "color": obj_metadata.get("color", [0.5, 0.5, 0.5]),
                    "mass": obj_metadata.get("mass", 1.0),
                    "disable_gravity": obj_metadata.get("disable_gravity", True)
                }
            }
            config["vpt_objects"]["objects"].append(vpt_obj)
        
        # Create config directory if it doesn't exist
        config_dir = f"{self.base_path}/configs"
        os.makedirs(config_dir, exist_ok=True)
        
        # Save to JSON file
        config_filepath = f"{config_dir}/env_{folder_idx}_config.json"
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        if self.verbose >= 2:
            print(f"  💾 Saved config: {config_filepath}")

    def _load_env_config_from_json(self, config_filepath: str, target_env_id: int):
        """
        Load environment configuration from JSON file and apply to specified environment.
        Restores positions, orientations, and validates spawn cfg matches current cfg.
        
        Args:
            config_filepath: Path to JSON configuration file
            target_env_id: Environment ID to load configuration into
        """
        import json
        import os
        
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
            env_ids = target_env_id
        
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
        
        # Build VPT object states
        vpt_count = config["vpt_objects"]["count"]
        vpt_positions = []
        vpt_orientations = []
        
        for obj_data in config["vpt_objects"]["objects"]:
            vpt_positions.append(obj_data["position"])
            vpt_orientations.append(obj_data["orientation"])
        
        vpt_positions = torch.tensor(vpt_positions, device=device, dtype=torch.float32)
        vpt_orientations = torch.tensor(vpt_orientations, device=device, dtype=torch.float32)
        
        # Apply goal ball configuration
        goal_pose = torch.cat([goal_pos.unsqueeze(0), goal_quat.unsqueeze(0)], dim=1)
        self._goal.write_root_pose_to_sim(goal_pose, env_ids)
        self._goal.write_root_velocity_to_sim(torch.zeros((1, 6), device=device), env_ids)
        
        # Apply camera object configuration
        camera_pose = torch.cat([camera_pos.unsqueeze(0), camera_quat.unsqueeze(0)], dim=1)
        self._camera_obj.write_root_pose_to_sim(camera_pose, env_ids)
        self._camera_obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device), env_ids)
        
        # Apply agent configuration
        agent_pose = torch.cat([agent_pos.unsqueeze(0), agent_quat.unsqueeze(0)], dim=1)
        self._agent.write_root_pose_to_sim(agent_pose, env_ids)
        self._agent.write_root_velocity_to_sim(torch.zeros((1, 6), device=device), env_ids)
        
        # Apply VPT objects configuration
        vpt_poses = torch.cat([vpt_positions.unsqueeze(0), vpt_orientations.unsqueeze(0)], dim=2)
        self._vpt_objects.write_object_pose_to_sim(vpt_poses, env_ids)
        self._vpt_objects.write_object_velocity_to_sim(torch.zeros((1, vpt_count, 6), device=device), env_ids)
        
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
            
            env_id_item = target_env_id if isinstance(target_env_id, int) else target_env_id.item()
            self.valid_viewpoint_poses[env_id_item] = valid_viewpoints
        
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
