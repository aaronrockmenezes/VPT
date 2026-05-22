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

        self.visibility_labels_json_path = f"{self.base_path}/visibility_labels.json"

        self._reset_called = False

        self.max_rgb_images = 300

        self.save_camera_pov = True

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

        for i, _ in enumerate(env_ids):
            action = actions[i]

            if action == 2:
                new_quat[i] = math_utils.quat_mul(upright_quat[i].unsqueeze(0), left_rot_quat.unsqueeze(0)).squeeze(0)
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 3:
                new_quat[i] = math_utils.quat_mul(upright_quat[i].unsqueeze(0), right_rot_quat.unsqueeze(0)).squeeze(0)
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 4:
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 5:
                env_id_item = env_ids[i].item() if torch.is_tensor(env_ids[i]) else env_ids[i]

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
                    if torch.norm(direction) > 1e-6:
                        yaw = torch.atan2(direction[1], direction[0])
                    else:
                        yaw = torch.tensor(0.0, device=device)

                    new_quat[i] = torch.tensor([math.cos(yaw.item() / 2), 0.0, 0.0, math.sin(yaw.item() / 2)],
                                               device=device, dtype=torch.float32)

                    current_pos[i, :3] = target_pos

                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            elif action == 6:  # Move to random valid viewpoint (random selection from valid poses)
                # Teleport agent to a randomly selected valid viewpoint
                env_id_item = env_ids[i].item() if torch.is_tensor(env_ids[i]) else env_ids[i]

                if (self.valid_viewpoint_poses is not None
                        and env_id_item < len(self.valid_viewpoint_poses)
                        and self.valid_viewpoint_poses[env_id_item] is not None
                        and len(self.valid_viewpoint_poses[env_id_item]) > 0):

                    num_valid_poses = len(self.valid_viewpoint_poses[env_id_item])
                    
                    # Get unused indices for this environment
                    used_indices = self.used_viewpoint_indices[env_id_item]
                    available_indices = [idx for idx in range(num_valid_poses) if idx not in used_indices]
                    
                    # If all indices have been used, reset the used set
                    if not available_indices:
                        if self.verbose >= 1:
                            print(f"[Action 6] Env {env_id_item}: All {num_valid_poses} viewpoints used, resetting")
                        self.used_viewpoint_indices[env_id_item].clear()
                        available_indices = list(range(num_valid_poses))
                    
                    # Try to find a viewpoint where both camera and goal are visible
                    max_attempts = min(1, len(available_indices))
                    found_valid = False
                    selected_idx = None
                    selected_target_pos = None
                    selected_yaw = None
                    
                    for attempt in range(max_attempts):
                        # Randomly select from available indices
                        random_idx = random.choice(available_indices)
                        target_pos = self.valid_viewpoint_poses[env_id_item][random_idx].to(device)
                        
                        # Calculate midpoint between camera and goal (in world coordinates, 2D only)
                        camera_pos_3d = self._camera_obj.data.root_pos_w[env_ids[i]]
                        goal_pos_3d = self._goal.data.root_pos_w[env_ids[i]]
                        midpoint = (camera_pos_3d[:2] + goal_pos_3d[:2]) / 2.0
                        
                        # Calculate yaw to look at midpoint (XY plane only)
                        direction = midpoint - target_pos[:2]
                        if torch.norm(direction) > 1e-6:
                            yaw = torch.atan2(direction[1], direction[0])
                        else:
                            yaw = torch.tensor(0.0, device=device)
                        
                        # Check if both objects are in FOV from this position/orientation
                        # Temporarily set agent position and orientation
                        temp_pos = current_pos[i].clone()
                        temp_pos[:3] = target_pos
                        temp_quat = torch.tensor([
                            math.cos(yaw.item() / 2), 0.0, 0.0,
                            math.sin(yaw.item() / 2)
                        ], device=device, dtype=torch.float32)
                        
                        # Write temporary pose
                        self._agent.write_root_com_pose_to_sim(
                            torch.cat([temp_pos.unsqueeze(0), temp_quat.unsqueeze(0)], dim=1), 
                            env_ids[i:i+1])
                        
                        # Update camera
                        for _ in range(3):
                            self.sim.step()
                            self._tiled_camera.update(self.sim.cfg.dt)
                        
                        # Check visibility
                        goal_visible, camera_visible = self.check_object_visibility(env_id_item)
                        
                        if goal_visible and camera_visible:
                            # Mark this index as used
                            self.used_viewpoint_indices[env_id_item].add(random_idx)
                            found_valid = True
                            selected_idx = random_idx
                            selected_target_pos = target_pos
                            selected_yaw = yaw
                            
                            if self.verbose >= 2:
                                print(f"[Action 6] Env {env_id_item}: Found valid viewpoint {random_idx}/{num_valid_poses-1} "
                                      f"on attempt {attempt+1} ({len(self.used_viewpoint_indices[env_id_item])}/{num_valid_poses} used)")
                            break
                        else:
                            # Remove this index from available if not visible
                            available_indices.remove(random_idx)
                            if not available_indices:
                                break
                    
                    if found_valid:
                        # Store selected position and orientation to be written later
                        new_quat[i] = torch.tensor([
                            math.cos(selected_yaw.item() / 2), 0.0, 0.0,
                            math.sin(selected_yaw.item() / 2)
                        ], device=device, dtype=torch.float32)
                        current_pos[i, :3] = selected_target_pos
                    else:
                        if self.verbose >= 1:
                            print(f"[Action 6] Env {env_id_item}: Could not find viewpoint with both objects visible after {max_attempts} attempts")
                else:
                    # Fallback: stay in current position if no valid poses available
                    if self.verbose >= 1:
                        print(f"Warning: No valid viewpoint poses available for env {env_id_item}, staying in place")

                # Set velocity to zero
                desired_vel[i, :3] = 0.0
                desired_vel[i, 3:6] = 0.0

            else:
                forward_input = 1.0 if action == 0 else -1.0
                local_movement = torch.tensor([forward_input, 0.0, 0.0], device=device)
                world_velocity = math_utils.quat_apply(upright_quat[i].unsqueeze(0),
                                                       local_movement.unsqueeze(0)).squeeze(0) * max_velocity
                desired_vel[i, :3] = world_velocity
                desired_vel[i, 3:6] = 0.0

            desired_vel[i, 2] = 0.0
            desired_vel[i, 3] = 0.0
            desired_vel[i, 4] = 0.0

        self._agent.write_root_com_pose_to_sim(torch.cat([current_pos, new_quat], dim=1), env_ids)
        self._agent.write_root_com_velocity_to_sim(desired_vel, env_ids)
        self._agent.reset()

        # Update occlusion camera to match camera object position and orientation
        camera_obj_pos = self._camera_obj.data.root_pos_w[env_ids].clone()
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_ids].clone()

        # Create 90-degree left rotation quaternion
        theta_left = math.pi / 2
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)],
            device=device)

        # Apply rotation to camera orientations for left-facing view
        rotated_orientations = math_utils.quat_mul(
            camera_obj_quat,
            left_90_quat.unsqueeze(0).expand(num_envs, -1))

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
            depth_base = f"{self.base_path}/Depth/{visibility_label}"
            depth_env_folder = f"{depth_base}/env_{folder_idx}"

            os.makedirs(rgb_env_folder, exist_ok=True)
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
                    cam_pov_filename = f"{rgb_env_folder}/cam_pov.png"
                    if not os.path.exists(cam_pov_filename):
                        cam_pov_img = camera_pov_data[env_id_item, :, :, :3]

                        if cam_pov_img.max() <= 1.0:
                            cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(np.uint8)
                        else:
                            cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)

                        cv2.imwrite(cam_pov_filename, cv2.cvtColor(cam_pov_np, cv2.COLOR_RGB2BGR))

        return envs_with_20_images

    def _check_all_envs_have_20_images(self) -> bool:
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
            
            if num_rgb_images < 20:
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
        if self._check_all_envs_have_20_images():
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

    def _reset_idx(self, env_ids: Sequence[int] | None, randomize_objects: bool = True) -> None:
        """Reset environments with validation for viewpoints and occlusion status."""
        MIN_VALID_VIEWPOINTS = 20
        max_reset_attempts = 50

        if env_ids is None:
            env_ids = self._agent._ALL_INDICES

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # Track environments needing reset
        envs_needing_reset = set(range(len(env_ids)))
        original_folder_indices = [self.next_env_folder_idx + i for i in range(len(env_ids))]
        original_visibility_categories = {}
        
        for reset_attempt in range(max_reset_attempts):
            if not envs_needing_reset:
                break

            # Reset only failed environments
            env_ids_to_reset = torch.tensor([env_ids[i] for i in envs_needing_reset], 
                                            dtype=torch.long, device=self.device)
            folder_indices_to_reset = [original_folder_indices[i] for i in envs_needing_reset]
            
            # Preserve visibility categories on retry
            visibility_categories_to_use = None if reset_attempt == 0 else \
                [original_visibility_categories.get(i) for i in envs_needing_reset]
            
            self._reset_idx_internal(env_ids_to_reset, randomize_objects, 
                                    folder_indices=folder_indices_to_reset,
                                    visibility_categories=visibility_categories_to_use)
            
            # Store categories from first attempt
            if reset_attempt == 0:
                for i in envs_needing_reset:
                    folder_idx = original_folder_indices[i]
                    original_visibility_categories[i] = self.env_visibility_reasons.get(folder_idx)

            # Validate viewpoints and occlusion
            envs_still_needing_retry = set()
            for local_idx, global_idx in enumerate(envs_needing_reset):
                env_id = env_ids[global_idx]
                env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
                reason = ""
                
                # Check viewpoint count
                if (self.valid_viewpoint_poses is None or 
                    env_id_item >= len(self.valid_viewpoint_poses) or
                    self.valid_viewpoint_poses[env_id_item] is None):
                    reason = "no viewpoints"
                    envs_still_needing_retry.add(global_idx)
                else:
                    num_poses = len(self.valid_viewpoint_poses[env_id_item])
                    if num_poses < MIN_VALID_VIEWPOINTS:
                        reason = f"insufficient viewpoints: {num_poses}/{MIN_VALID_VIEWPOINTS}"
                        envs_still_needing_retry.add(global_idx)
                
                # Validate occlusion for in_view and occluded cases
                if not reason:
                    folder_idx = original_folder_indices[global_idx]
                    visibility_reason = self.env_visibility_reasons.get(folder_idx, "unknown")
                    
                    if visibility_reason in ["in_view", "occluded"]:
                        camera_pos = self._camera_obj.data.root_pos_w[env_id]
                        goal_pos = self._goal.data.root_pos_w[env_id]
                        is_occluded = self._check_occlusion_raycast(camera_pos, goal_pos, env_id)
                        
                        # in_view should NOT be occluded, occluded should BE occluded
                        expected_occluded = (visibility_reason == "occluded")
                        if is_occluded != expected_occluded:
                            reason = f"occlusion mismatch: {visibility_reason}"
                            envs_still_needing_retry.add(global_idx)
                
                if reason and self.verbose >= 1:
                    print(f"  ❌ Env {env_id_item}: {reason}")
            
            envs_needing_reset = envs_still_needing_retry

        # Final validation: Force camera update and check ALL environments
        if self.verbose >= 1:
            print(f"\n🔍 Final validation: Checking camera raycast for all {len(env_ids)} environments...")
        
        # Force camera updates before final check
        for _ in range(5):
            self.sim.step()
            self._occlusion_camera.update(self.sim.cfg.dt)
        
        final_validation_failures = []
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            folder_idx = original_folder_indices[i]
            visibility_reason = self.env_visibility_reasons.get(folder_idx, "unknown")
            
            # Only validate in_view and occluded (skip outside_fov)
            if visibility_reason in ["in_view", "occluded"]:
                camera_pos = self._camera_obj.data.root_pos_w[env_id]
                goal_pos = self._goal.data.root_pos_w[env_id]
                is_occluded = self._check_occlusion_raycast(camera_pos, goal_pos, env_id)
                
                expected_occluded = (visibility_reason == "occluded")
                if is_occluded != expected_occluded:
                    final_validation_failures.append(env_id_item)
                    if self.verbose >= 1:
                        print(f"  ❌ Final check - Env {env_id_item}: expected {'occluded' if expected_occluded else 'visible'}, "
                              f"got {'occluded' if is_occluded else 'visible'}")
        
        # Log final validation results
        if final_validation_failures:
            if self.verbose >= 1:
                print(f"\n⚠️  Final camera raycast validation failed for {len(final_validation_failures)} environment(s): {final_validation_failures}")
                print(f"     These environments may have incorrect occlusion status")

        # Warn if any failed
        if envs_needing_reset and self.verbose >= 1:
            print(f"\n⚠️  RESET FAILURE after {max_reset_attempts} attempts:")
            for global_idx in envs_needing_reset:
                env_id_item = env_ids[global_idx].item() if torch.is_tensor(env_ids[global_idx]) else env_ids[global_idx]
                print(f"  ❌ Env {env_id_item}")

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

        # Reset used viewpoint tracking
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            self.used_viewpoint_indices[env_id_item].clear()

        # Assign visibility categories
        if visibility_categories is None:
            num_in_view = num_envs // 2
            num_occluded = num_envs // 4
            num_outside_fov = num_envs - num_in_view - num_occluded
            categories_list = ["in_view"] * num_in_view + ["occluded"] * num_occluded + ["outside_fov"] * num_outside_fov
            random.shuffle(categories_list)
            visibility_categories = categories_list

        # Set labels and reasons
        for i in range(num_envs):
            global_folder_idx = folder_indices[i]
            category = visibility_categories[i]
            
            if category == "in_view":
                self.env_visibility_labels[global_folder_idx] = "Yes"
                self.env_visibility_reasons[global_folder_idx] = "in_view"
            elif category == "occluded":
                self.env_visibility_labels[global_folder_idx] = "No"
                self.env_visibility_reasons[global_folder_idx] = "occluded"
            else:
                self.env_visibility_labels[global_folder_idx] = "No"
                self.env_visibility_reasons[global_folder_idx] = "outside_fov"

        self._save_visibility_labels()
        self.viewpoint_pose_counter[env_ids] = 0
        super()._reset_idx(env_ids)

        device = self._agent.device
        safe_x_range = self.center_to_boundary - 1.5
        safe_y_range = self.center_to_boundary - 1.5

        # Get default states
        goal_default_state = self._goal.data.default_root_state[env_ids].clone()
        agent_default_state = self._agent.data.default_root_state[env_ids].clone()
        camera_obj_default_state = self._camera_obj.data.default_root_state[env_ids].clone()
        vpt_obj_default_state = self._vpt_objects.data.default_object_state[env_ids].clone()

        # Spawn objects for each environment
        for i in range(len(env_ids)):
            visibility_category = visibility_categories[i]
            target_occluded = (visibility_category == "occluded")
            target_outside_fov = (visibility_category == "outside_fov")
            
            success = False
            for attempt in range(100):
                # Spawn positions
                roll = torch.tensor(-math.radians(self.agent_camera_pitch), device=device)
                
                goal_offset = sample_uniform(-safe_x_range, safe_x_range, (2,), device)
                goal_new_pos = goal_default_state[i, :3].clone()
                goal_new_pos[0] = self.scene.env_origins[env_ids[i], 0] + goal_offset[0]
                goal_new_pos[1] = self.scene.env_origins[env_ids[i], 1] + goal_offset[1]
                goal_new_pos[2] += self.scene.env_origins[env_ids[i], 2]

                camera_offset = sample_uniform(-safe_x_range, safe_x_range, (2,), device)
                camera_new_pos = camera_obj_default_state[i, :3].clone()
                camera_new_pos[0] = self.scene.env_origins[env_ids[i], 0] + camera_offset[0]
                camera_new_pos[1] = self.scene.env_origins[env_ids[i], 1] + camera_offset[1]

                # Camera orientation towards goal
                direction_to_goal = goal_new_pos[:2] - camera_new_pos[:2]
                yaw = torch.atan2(direction_to_goal[1], direction_to_goal[0]) - math.radians(90)
                quaternion = quat_from_euler_xyz(roll, torch.tensor(0, device=device), yaw)
                camera_obj_default_state[i, 3:7] = quaternion

                agent_offset = sample_uniform(-safe_x_range, safe_x_range, (2,), device)
                agent_new_pos = agent_default_state[i, :3].clone()
                agent_new_pos[0] = self.scene.env_origins[env_ids[i], 0] + agent_offset[0]
                agent_new_pos[1] = self.scene.env_origins[env_ids[i], 1] + agent_offset[1]

                # Check distance constraints
                agent_distance_from_goal = torch.norm(agent_new_pos[:2] - goal_new_pos[:2])
                camera_goal_distance = torch.norm(camera_new_pos[:2] - goal_new_pos[:2])
                if agent_distance_from_goal < 1.0 or camera_goal_distance > 3.0 or camera_goal_distance < 0.5:
                    continue

                # Spawn VPT objects
                vpt_obj_offset = sample_uniform(-safe_x_range, safe_x_range, (self.num_objs, 2), device)
                vpt_obj_new_pos = vpt_obj_default_state[i, :, :3].clone()
                vpt_obj_new_pos[:, 0] = self.scene.env_origins[env_ids[i], 0] + vpt_obj_offset[:, 0]
                vpt_obj_new_pos[:, 1] = self.scene.env_origins[env_ids[i], 1] + vpt_obj_offset[:, 1]
                vpt_obj_new_pos[:, 2] += self.scene.env_origins[env_ids[i], 2]

                # Check VPT distance from camera
                camera_distances_from_vpt = torch.norm(camera_new_pos[:2].unsqueeze(0) - vpt_obj_new_pos[:, :2], dim=1)
                if not torch.all(camera_distances_from_vpt >= 1.0):
                    continue

                # Write and validate poses
                goal_default_state[i, :3] = goal_new_pos
                agent_default_state[i, :3] = agent_new_pos
                camera_obj_default_state[i, :3] = camera_new_pos
                vpt_obj_default_state[i, :, :3] = vpt_obj_new_pos

                single_env_id = env_ids[i:i+1]
                
                goal_valid = self._write_and_validate_pose(
                    self._goal, goal_default_state[i:i+1, :7], 
                    torch.zeros_like(goal_default_state[i:i+1, 7:]), single_env_id, "goal")
                camera_valid = self._write_and_validate_pose(
                    self._camera_obj, camera_obj_default_state[i:i+1, :7],
                    torch.zeros_like(camera_obj_default_state[i:i+1, 7:]), single_env_id, "camera")
                agent_valid = self._write_and_validate_pose(
                    self._agent, agent_default_state[i:i+1, :7],
                    torch.zeros_like(agent_default_state[i:i+1, 7:]), single_env_id, "agent")
                vpt_valid = self._write_and_validate_pose(
                    self._vpt_objects, vpt_obj_default_state[i:i+1, :, :7],
                    torch.zeros_like(vpt_obj_default_state[i:i+1, :, 7:]), single_env_id, "vpt", is_collection=True)
                
                if not (goal_valid and camera_valid and agent_valid and vpt_valid):
                    continue

                # Update occlusion camera
                occlusion_camera_pos = camera_obj_default_state[i:i+1, :3].clone()
                theta_left = math.pi / 2
                half_theta_left = theta_left / 2
                left_90_quat = torch.tensor([math.cos(half_theta_left), 0.0, 0.0, math.sin(half_theta_left)], device=device)
                rotated_orientations = math_utils.quat_mul(camera_obj_default_state[i:i+1, 3:7], left_90_quat.unsqueeze(0))
                
                self._occlusion_camera.set_world_poses(positions=occlusion_camera_pos, orientations=rotated_orientations,
                                                       env_ids=single_env_id.tolist(), convention="world")
                for _ in range(3):
                    self.sim.step()
                    self._occlusion_camera.update(self.sim.cfg.dt)

                # Validate occlusion status
                is_occluded = self._check_occlusion_raycast(camera_new_pos, goal_new_pos, single_env_id[0])
                
                if visibility_category == "in_view" and is_occluded:
                    continue
                elif visibility_category == "occluded" and not is_occluded:
                    continue
                
                # Point camera away for outside_fov
                if target_outside_fov:
                    yaw_away = yaw + math.pi
                    quaternion_away = quat_from_euler_xyz(roll, torch.tensor(0, device=device), yaw_away)
                    camera_obj_default_state[i, 3:7] = quaternion_away
                    self._camera_obj.write_root_pose_to_sim(camera_obj_default_state[i:i+1, :7], single_env_id)
                
                success = True
                break

        # Set random agent orientations
        random_yaw_agent = sample_uniform(0, 2 * math.pi, (num_envs,), device)
        agent_default_state[:, 3] = torch.cos(random_yaw_agent / 2)
        agent_default_state[:, 4] = 0.0
        agent_default_state[:, 5] = 0.0
        agent_default_state[:, 6] = torch.sin(random_yaw_agent / 2)
        self._agent.write_root_pose_to_sim(agent_default_state[:, :7], env_ids)

        # Generate viewpoints
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
            
            # Update camera to apply changes
            for _ in range(5):
                self.sim.step()
                camera.update(self.sim.cfg.dt)
            
            # Verify the fix worked
            new_occlusion_cam_pos = camera.data.pos_w[env_id]
            new_pos_diff = torch.norm(camera_obj_pos - new_occlusion_cam_pos).item()
            print(f"    ✓ After fix: Distance = {new_pos_diff:.4f}")

        GOAL_THRESHOLD = 5

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
        GOAL_THRESHOLD = 10
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
        env_origin = self.scene.env_origins[env_id, :2]
        boundary_limit = self.center_to_boundary.item() if torch.is_tensor(
            self.center_to_boundary) else self.center_to_boundary

        min_bound = env_origin - boundary_limit
        max_bound = env_origin + boundary_limit

        if not (torch.all(point >= min_bound) and torch.all(point <= max_bound)):
            return False

        vpt_positions = self._vpt_objects.data.object_pos_w[env_id, :, :2]
        distances = torch.norm(point.unsqueeze(0) - vpt_positions, dim=1)

        if torch.any(distances < min_obstacle_distance):
            return False

        camera_pos = self._camera_obj.data.root_pos_w[env_id, :2]
        camera_distance = torch.norm(point - camera_pos).item()

        if camera_distance < min_camera_target_distance:
            return False

        goal_pos = self._goal.data.root_pos_w[env_id, :2]
        goal_distance = torch.norm(point - goal_pos).item()

        if goal_distance < min_camera_target_distance:
            return False

        if not check_agent_fov:
            return True
        elif check_agent_fov:
            # NEW: Check if agent can view both camera and goal from this point
            # Temporarily position agent at candidate point and check visibility
            device = self._agent.device
            
            # Save current agent state
            current_agent_pos = self._agent.data.root_pos_w[env_id].clone()
            current_agent_quat = self._agent.data.root_quat_w[env_id].clone()
            
            # Create temporary agent pose at candidate point
            temp_agent_pos = torch.zeros(3, device=device)
            temp_agent_pos[:2] = point
            temp_agent_pos[2] = self._agent.data.default_root_state[env_id, 2]
            
            # Calculate midpoint between camera and goal for agent orientation
            camera_pos_3d = self._camera_obj.data.root_pos_w[env_id]
            goal_pos_3d = self._goal.data.root_pos_w[env_id]
            midpoint = (camera_pos_3d[:2] + goal_pos_3d[:2]) / 2.0
            
            # Calculate yaw to look at midpoint
            direction = midpoint - point
            if torch.norm(direction) > 1e-6:
                yaw = torch.atan2(direction[1], direction[0])
            else:
                yaw = torch.tensor(0.0, device=device)

            # Add environment origin to temporary position
            # temp_agent_pos[:2] += self.scene.env_origins[env_id, :2]
            temp_agent_quat = torch.tensor([
                math.cos(yaw.item() / 2), 0.0, 0.0,
                math.sin(yaw.item() / 2)
            ], device=device, dtype=torch.float32)
            
            # Write temporary agent pose
            temp_pose = torch.cat([temp_agent_pos.unsqueeze(0), temp_agent_quat.unsqueeze(0)], dim=1)
            env_ids_tensor = torch.tensor([env_id], dtype=torch.long, device=device)
            self._agent.write_root_com_pose_to_sim(temp_pose, env_ids_tensor)
            
            # Update tiled camera to get visibility from this position
            for _ in range(1):
                self.sim.step()
                self._tiled_camera.update(self.sim.cfg.dt)
            
            # Check if both camera and goal are visible
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            goal_visible, camera_visible = self.check_object_visibility(env_id_item)
            
            # Restore original agent state
            original_pose = torch.cat([current_agent_pos.unsqueeze(0), current_agent_quat.unsqueeze(0)], dim=1)
            self._agent.write_root_com_pose_to_sim(original_pose, env_ids_tensor)
            
            # Point is only valid if both objects are visible
            return goal_visible and camera_visible

    def generate_valid_circle_points(self, env_ids: torch.Tensor, angle_step: float = 2.0,
                                     max_attempts: int = 300) -> List[torch.Tensor]:
        device = self._agent.device
        angles = torch.arange(0, 360, angle_step, device=device)

        all_valid_points = []

        for i, env_id in enumerate(env_ids):
            camera_pos = self._camera_obj.data.root_pos_w[env_id]
            goal_pos = self._goal.data.root_pos_w[env_id]

            midpoint = (camera_pos[:2] + goal_pos[:2]) / 2.0

            camera_pos_2d = camera_pos[:2]
            goal_pos_2d = goal_pos[:2]
            min_radius = self._calculate_optimal_radius(
                torch.cat([midpoint, torch.zeros(1, device=midpoint.device)]),
                torch.cat([camera_pos_2d, torch.zeros(1, device=midpoint.device)]),
                torch.cat([goal_pos_2d, torch.zeros(1, device=midpoint.device)]))

            # PASS 1: Quick geometric validation (no agent FOV)
            candidate_points = []
            attempts = 0

            for radius in torch.Tensor([min_radius]):
                if attempts >= max_attempts:
                    break

                for angle_idx, angle in enumerate(angles):
                    if attempts >= max_attempts:
                        break

                    point = self._get_circle_point(midpoint[:2], radius.item(), angle.item(), device, env_id)

                    # Quick geometric check WITHOUT agent FOV (fast)
                    is_valid = self._is_point_valid(point, env_id, print_details=False, check_agent_fov=False)
                    attempts += 1

                    if is_valid:
                        candidate_points.append(point)
            
            # PASS 2: Sample candidates and validate with agent FOV (slow but necessary)
            if len(candidate_points) > 50:
                step = len(candidate_points) // 50
                points_to_check = [candidate_points[idx] for idx in range(0, len(candidate_points), step)][:50]
            else:
                points_to_check = candidate_points
            
            if self.verbose >= 2:
                print(f"  Env {env_id.item() if torch.is_tensor(env_id) else env_id}: "
                      f"{len(candidate_points)} geometric candidates, checking {len(points_to_check)} with agent FOV")
            
            valid_points_for_env = []
            for point in points_to_check:
                # Full validation WITH agent FOV check (includes simulation)
                is_valid_with_fov = self._is_point_valid(point, env_id, print_details=False, check_agent_fov=True)
                
                if is_valid_with_fov:
                    valid_points_for_env.append(point)

            if len(valid_points_for_env) > 0:
                valid_points_tensor = torch.stack(valid_points_for_env)
            else:
                valid_points_tensor = torch.zeros((0, 2), device=device)

            all_valid_points.append(valid_points_tensor)
            
            if self.verbose >= 2:
                print(f"  Env {env_id.item() if torch.is_tensor(env_id) else env_id}: "
                      f"Final valid viewpoints: {len(valid_points_for_env)}")

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
