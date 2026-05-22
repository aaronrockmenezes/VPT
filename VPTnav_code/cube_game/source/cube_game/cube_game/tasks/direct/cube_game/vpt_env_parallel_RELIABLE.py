from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import random
import numpy as np
import os
import cv2
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

        # Configurable collection parameters
        self.images_per_env = 20  # Number of images to collect per environment
        self.min_viewpoint_distance = 0.1  # Minimum distance between viewpoints (meters)
        self.goal_pixel_threshold = 400  # Minimum pixels for goal visibility
        self.camera_pixel_threshold = 1200  # Minimum pixels for camera visibility

        self.valid_viewpoint_poses = [None] * self.num_envs
        self.selected_viewpoints_for_collection = [None] * self.num_envs
        self.current_collection_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.viewpoint_pose_counter = torch.zeros(self.num_envs,
                                                  dtype=torch.long,
                                                  device=self.device)
        self.next_env_folder_idx = 0
        self.env_visibility_labels = {}
        self.env_visibility_reasons = {}
        self.base_path = "/home/arock3/data"
        # self.base_path = "/media/data_cifs_lrs/projects/prj_robotics/VPTnav_v2a"
        self.visibility_labels_json_path = f"{self.base_path}/visibility_labels.json"
        self._reset_called = False
        self.max_rgb_images = 20_000
        # self.max_rgb_images = 150
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

        self._distance_tiled_camera = TiledCamera(self.cfg.distance_tiled_camera)
        self.scene.sensors["distance_tiled_camera"] = self._distance_tiled_camera

        self._occlusion_camera = TiledCamera(self.cfg.occlusion_camera)
        self.scene.sensors["occlusion_camera"] = self._occlusion_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
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

    def _save_images(self, env_ids: torch.Tensor, rgb_data: torch.Tensor, 
                     depth_data: torch.Tensor, camera_pov_data: torch.Tensor = None, semantic_data: torch.Tensor = None) -> List[int]:
        """Save RGB, depth, and camera POV images for environments."""
        # TODO: Add semantic data for agent for debugging.

        current_folder_indices = [
            self.next_env_folder_idx + i for i in range(len(env_ids))
        ]
        envs_with_20_images = []

        for idx, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            folder_idx = current_folder_indices[idx]

            if folder_idx not in self.env_visibility_labels:
                raise RuntimeError(f"CRITICAL ERROR: No visibility label found for folder_idx {folder_idx}!")

            visibility_label = self.env_visibility_labels[folder_idx]

            if visibility_label not in ["Yes", "No"]:
                raise RuntimeError(f"CRITICAL ERROR: Invalid visibility label '{visibility_label}' for folder_idx {folder_idx}!")

            rgb_base = f"{self.base_path}/RGB/{visibility_label}"
            rgb_env_folder = f"{rgb_base}/env_{folder_idx}"
            cam_base = f"{self.base_path}/cam/{visibility_label}"
            cam_env_folder = f"{cam_base}/env_{folder_idx}"
            depth_base = f"{self.base_path}/Depth/{visibility_label}"
            depth_env_folder = f"{depth_base}/env_{folder_idx}"
            

            os.makedirs(rgb_env_folder, exist_ok=True)
            os.makedirs(cam_env_folder, exist_ok=True)
            os.makedirs(depth_env_folder, exist_ok=True)

            num_rgb_images = len([
                f for f in os.listdir(rgb_env_folder)
                if f.endswith('.png') and not f.startswith('cam_pov')
            ])
            num_depth_images = len([
                f for f in os.listdir(depth_env_folder) if f.endswith('.png')
            ])
            if semantic_data:
                semantic_base = f"{self.base_path}/Semantic/{visibility_label}"
                semantic_env_folder = f"{semantic_base}/env_{folder_idx}"
                os.makedirs(semantic_env_folder, exist_ok=True)
                num_semantic_images = len([
                    f for f in os.listdir(semantic_env_folder) if f.endswith('.png')
                ])

            if num_rgb_images >= self.images_per_env and num_depth_images >= self.images_per_env:
                if semantic_data:
                    if num_semantic_images >= self.images_per_env:
                        envs_with_20_images.append(env_id_item)
                else:
                    envs_with_20_images.append(env_id_item)
                continue

            # Save images (no visibility check needed - we trust the selected viewpoints)
            rgb_filename = f"{rgb_env_folder}/image_{num_rgb_images:04d}.png"
            rgb_img = rgb_data[env_id_item, :, :, :3]

            if rgb_img.max() <= 1.0:
                rgb_np = (rgb_img.cpu().numpy() * 255.0).astype(np.uint8)
            else:
                rgb_np = rgb_img.cpu().numpy().astype(np.uint8)
            
            cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

            depth_filename = f"{depth_env_folder}/image_{num_depth_images:04d}.png"
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

            if semantic_data is not None:
                semantic_filename = f"{semantic_env_folder}/image_{num_semantic_images:04d}.png"
                semantic_img = semantic_data[env_id_item, :, :, :3]
                semantic_np = semantic_img.cpu().numpy()

                if semantic_np.max() <= 1.0:
                    semantic_np = (semantic_np * 255.0).astype(np.uint8)
                else:
                    semantic_np = semantic_np.cpu().numpy().astype(np.uint8)
                
                cv2.imwrite(semantic_filename, cv2.cvtColor(semantic_np, cv2.COLOR_RGB2BGR))

            if self.save_camera_pov and camera_pov_data is not None:
                cam_pov_filename = f"{cam_env_folder}/cam_pov.png"
                if not os.path.exists(cam_pov_filename):
                    cam_pov_img = camera_pov_data[env_id_item, :, :, :3]

                    if cam_pov_img.max() <= 1.0:
                        cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(np.uint8)
                    else:
                        cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)

                    cv2.imwrite(cam_pov_filename, cv2.cvtColor(cam_pov_np, cv2.COLOR_RGB2BGR))

        return envs_with_20_images

    def _check_all_envs_have_20_images(self, return_total_count: bool = False):
        """Check if all current environments have required images saved."""
        env_ids = self._agent._ALL_INDICES
        current_folder_indices = [
            self.next_env_folder_idx + i for i in range(len(env_ids))
        ]

        total_rgb_images = 0
        all_have_required = True

        for folder_idx in current_folder_indices:
            if folder_idx not in self.env_visibility_labels:
                if return_total_count:
                    continue
                return False

            visibility_label = self.env_visibility_labels[folder_idx]
            rgb_env_folder = f"{self.base_path}/RGB/{visibility_label}/env_{folder_idx}"
            if not os.path.exists(rgb_env_folder):
                if return_total_count:
                    continue
                return False

            num_rgb_images = len([
                f for f in os.listdir(rgb_env_folder)
                if f.endswith('.png') and not f.startswith('cam_pov')
            ])
            total_rgb_images += num_rgb_images

            if num_rgb_images < self.images_per_env:
                all_have_required = False
                if not return_total_count:
                    return False

        if return_total_count:
            return all_have_required, total_rgb_images
        return all_have_required

    def _get_observations(self) -> dict:
        if not self._reset_called:
            raise RuntimeError(
                "ERROR: _get_observations called before _reset_idx! "
                "Environment initialization must call reset first.")

        env_ids = self._agent._ALL_INDICES
        device = self._agent.device
        
        # Check which envs have selected viewpoints ready for collection
        envs_to_collect = []
        for env_id in env_ids:
            env_id_item = env_id.item()
            if self.selected_viewpoints_for_collection[env_id_item] is not None:
                envs_to_collect.append(env_id_item)
        
        # Collect images for all 20 viewpoints for each env that triggered action 6
        if envs_to_collect:
            print(f"\n{'='*80}")
            print(f"📸 COLLECTING {self.images_per_env} IMAGES FOR {len(envs_to_collect)} ENVIRONMENT(S)")
            print(f"{'='*80}\n")
            
            for viewpoint_idx in range(self.images_per_env):
                print(f"  📍 Collecting from viewpoint {viewpoint_idx + 1}/{self.images_per_env}...")
                
                # Teleport ALL agents to their respective viewpoint_idx in parallel
                envs_to_collect_tensor = torch.tensor(envs_to_collect, dtype=torch.long, device=device)
                num_collecting = len(envs_to_collect)
                
                # Gather all target positions
                target_positions_2d = torch.stack([
                    self.selected_viewpoints_for_collection[env_id_item][viewpoint_idx]
                    for env_id_item in envs_to_collect
                ])
                
                if target_positions_2d.shape[-1] == 3:
                    target_positions_2d = target_positions_2d[:, :2]
                
                # Create full 3D positions
                target_positions_3d = torch.zeros((num_collecting, 3), device=device)
                target_positions_3d[:, :2] = target_positions_2d
                target_positions_3d[:, 2] = self._agent.data.default_root_state[envs_to_collect_tensor, 2]
                
                # Calculate orientations to look at midpoint between camera and goal
                camera_positions_3d = self._camera_obj.data.root_pos_w[envs_to_collect_tensor]
                goal_positions_3d = self._goal.data.root_pos_w[envs_to_collect_tensor]
                midpoints = (camera_positions_3d[:, :2] + goal_positions_3d[:, :2]) / 2.0
                directions = midpoints - target_positions_3d[:, :2]
                yaws = torch.atan2(directions[:, 1], directions[:, 0])
                
                # Create quaternions
                quats = torch.zeros((num_collecting, 4), device=device)
                quats[:, 0] = torch.cos(yaws / 2)
                quats[:, 3] = torch.sin(yaws / 2)
                
                # Write poses for all agents at once
                poses = torch.cat([target_positions_3d, quats], dim=1)
                self._agent.write_root_com_pose_to_sim(poses, envs_to_collect_tensor)
                self._agent.write_root_com_velocity_to_sim(
                    torch.zeros((num_collecting, 6), device=device),
                    envs_to_collect_tensor
                )
                
                # Update cameras after all agents moved
                for _ in range(3):
                    self.sim.step()
                    self._rgb_tiled_camera.update(self.sim.cfg.dt)
                    self._distance_tiled_camera.update(self.sim.cfg.dt)
                    self._occlusion_camera.update(self.sim.cfg.dt)

                # Get camera data
                rgb_data = self._rgb_tiled_camera.data.output["rgb"]
                depth_data = self._distance_tiled_camera.data.output["distance_to_camera"]
                camera_pov_data = self._occlusion_camera.data.output["semantic_segmentation"] if self.save_camera_pov else None

                # Save images for all collecting envs
                envs_to_collect_tensor = torch.tensor(envs_to_collect, dtype=torch.long, device=device)
                self._save_images(envs_to_collect_tensor, rgb_data, depth_data, camera_pov_data)
            
            print(f"\n  ✅ Collected {self.images_per_env} images for {len(envs_to_collect)} environment(s)\n")
            
            # Save env configs using method
            for env_id_item in envs_to_collect:
                folder_idx = self.next_env_folder_idx + env_ids.tolist().index(env_id_item)
                self._save_env_config_to_json(env_id_item, folder_idx)
                
            
            
            # Clear selection and check for reset
            envs_to_reset = []
            for env_id_item in envs_to_collect:
                self.selected_viewpoints_for_collection[env_id_item] = None
                
                # Check if this env now has 20 images
                folder_idx = self.next_env_folder_idx + env_ids.tolist().index(env_id_item)
                
                if folder_idx in self.env_visibility_labels:
                    visibility_label = self.env_visibility_labels[folder_idx]
                    rgb_env_folder = f"{self.base_path}/RGB/{visibility_label}/env_{folder_idx}"
                    
                    if os.path.exists(rgb_env_folder):
                        num_rgb_images = len([
                            f for f in os.listdir(rgb_env_folder)
                            if f.endswith('.png') and not f.startswith('cam_pov')
                        ])
                        
                        if num_rgb_images >= self.images_per_env:
                            envs_to_reset.append(env_id_item)
            
            # Reset individual envs that have 20 images
            if envs_to_reset:
                print(f"\n{'='*80}")
                print(f"🔄 RESETTING {len(envs_to_reset)} ENVIRONMENT(S) WITH {self.images_per_env} IMAGES: {envs_to_reset}")
                print(f"{'='*80}\n")
                
                for env_id_item in envs_to_reset:
                    self.next_env_folder_idx += 1
                
                self._reset_idx(envs_to_reset)
        
        # Update cameras for normal observation
        for _ in range(3):
            self.sim.step()
            self._rgb_tiled_camera.update(self.sim.cfg.dt)
            self._distance_tiled_camera.update(self.sim.cfg.dt)
            self._occlusion_camera.update(self.sim.cfg.dt)

        # Check global progress
        rgb_check, total_rgb_count = self._check_all_envs_have_20_images(return_total_count=True)

        if total_rgb_count >= self.max_rgb_images:
            print(f"\n{'='*80}")
            print(f"✅ MAX RGB IMAGES REACHED: {total_rgb_count}/{self.max_rgb_images}")
            print(f"{'='*80}\n")
            self.close()
            exit(0)
        else:
            if total_rgb_count > 0 and total_rgb_count % 320 == 0:
                print(f"\n{'='*80}")
                print(f"📊 PROGRESS UPDATE: {total_rgb_count}/{self.max_rgb_images} RGB images collected")
                print(f"{'='*80}\n")

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

    def _reset_idx(self, env_ids: Sequence[int] | None, randomize_objects: bool = True) -> None:
        """Reset environments by selectively retrying only the failed ones."""
        MIN_VALID_VIEWPOINTS = self.images_per_env
        max_inner_attempts = 50
        max_full_attempts = 20

        if env_ids is None:
            env_ids = self._agent._ALL_INDICES
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        num_envs = len(env_ids)
        original_folder_indices = [self.next_env_folder_idx + i for i in range(num_envs)]

        if self.verbose >= 1:
            print("🔒 Locking in visibility assignments for all environments...")
        num_in_view = num_envs // 2
        num_occluded = num_envs // 4
        num_outside_fov = num_envs - num_in_view - num_occluded
        visibility_categories = (["in_view"] * num_in_view +
                                ["occluded"] * num_occluded +
                                ["outside_fov"] * num_outside_fov)
        # visibility_categories = (["in_view"] * num_envs)
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
        
        batch_needs_retry = torch.ones(num_envs, dtype=torch.bool, device=self.device)

        for full_attempt in range(max_full_attempts):
            if not batch_needs_retry.any():
                print(f"\n🎉 SUCCESS: All {num_envs} environments correctly configured!")
                break

            num_to_fix = batch_needs_retry.sum().item()
            global_indices_to_fix = torch.where(batch_needs_retry)[0]

            print(f"\n{'='*40}")
            print(f"🚀 Master Attempt {full_attempt + 1}/{max_full_attempts}: Targeting {num_to_fix} environment(s).")
            print(f"{'='*40}")

            retry_env_ids = env_ids[global_indices_to_fix]
            retry_folder_indices = [original_folder_indices[i] for i in global_indices_to_fix]
            retry_visibility_categories = [visibility_categories[i] for i in global_indices_to_fix]

            subset_needs_retry = torch.ones_like(retry_env_ids, dtype=torch.bool)
            for inner_attempt in range(max_inner_attempts):
                if not subset_needs_retry.any():
                    break

                ids_to_reset_now = retry_env_ids[subset_needs_retry]
                folders_to_reset_now = [retry_folder_indices[i] for i, needs_retry in enumerate(subset_needs_retry) if needs_retry]
                vis_to_reset_now = [retry_visibility_categories[i] for i, needs_retry in enumerate(subset_needs_retry) if needs_retry]

                self._reset_idx_internal(
                    ids_to_reset_now,
                    randomize_objects,
                    folder_indices=folders_to_reset_now,
                    visibility_categories=vis_to_reset_now
                )

                for i, needs_retry in enumerate(subset_needs_retry):
                    if needs_retry:
                        is_valid, reason = self._validate_env_state(retry_env_ids[i], retry_folder_indices[i], MIN_VALID_VIEWPOINTS)
                        if is_valid:
                            subset_needs_retry[i] = False
                        elif self.verbose >= 1 and inner_attempt == max_inner_attempts - 1:
                            env_id_item = retry_env_ids[i].item()
                            print(f"  ❌ Env {env_id_item} (folder {retry_folder_indices[i]}) failed inner reset: {reason}")
            
            for i, global_idx in enumerate(global_indices_to_fix):
                is_valid, _ = self._validate_env_state(retry_env_ids[i], retry_folder_indices[i], MIN_VALID_VIEWPOINTS)
                if is_valid:
                    batch_needs_retry[global_idx] = False

        if self.verbose >= 1:
            print("\n🔍 Performing final comprehensive validation on the entire batch...")
        
        for _ in range(5):
            self.sim.step()
            self._occlusion_camera.update(self.sim.cfg.dt)
            
        validation_results = self._perform_final_validation(env_ids, original_folder_indices)
        self._print_validation_summary(validation_results, num_envs)

        if batch_needs_retry.any():
            num_failed = batch_needs_retry.sum().item()
            raise RuntimeError(f"CRITICAL FAILURE: {num_failed} environment(s) could not be configured correctly after {max_full_attempts} master attempts.")

        self._reset_called = True

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
    
    def _get_vpt_heights(self, env_ids):
        heights = self._vpt_objects.data.object_pos_w[env_ids, :, 2]
        return heights

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
        
        
        for spawn_attempt in range(max_spawn_attempts):
            if not envs_need_spawn_retry.any():
                break
            
            retry_mask = envs_need_spawn_retry.clone()
            batch_size = retry_mask.sum().item()
            retry_indices = torch.where(retry_mask)[0]
            
            goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
            camera_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
            agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
            vpt_offsets = sample_uniform(-safe_x_range_obstacles, safe_x_range_obstacles, (batch_size, self.num_objs, 2), device)
            # vpt_offsets_valid = False
            # while not vpt_offsets_valid:
            #     vpt_offsets = sample_uniform(-safe_x_range_obstacles, safe_x_range_obstacles, (batch_size, self.num_objs, 2), device)
            #     # Ensure all VPT objects are at least 1.0 units apart from each other
            #     vpt_offsets_flat = vpt_offsets.reshape(batch_size * self.num_objs, 2)
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
            goal_default_state[retry_indices, 2] = self._goal.data.default_root_state[env_ids_batch, 2] + env_origins[:, 2]

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

            # VPT object positions
            vpt_obj_default_state[retry_indices, :, 0] = env_origins[:, 0].unsqueeze(1) + vpt_offsets[:, :, 0]
            vpt_obj_default_state[retry_indices, :, 1] = env_origins[:, 1].unsqueeze(1) + vpt_offsets[:, :, 1]
            vpt_obj_default_state[retry_indices, :, 2] = self._vpt_objects.data.default_object_state[env_ids_batch, :, 2] + env_origins[:, 2].unsqueeze(1)
            
            for batch_idx, env_idx in enumerate(retry_indices):
                agent_pos = agent_default_state[env_idx, :2]
                goal_pos = goal_default_state[env_idx, :2]
                camera_pos = camera_obj_default_state[env_idx, :2]
                
                camera_goal_distance = torch.norm(camera_pos - goal_pos)
                
                if camera_goal_distance > 8.0 or camera_goal_distance < 1.0:
                    continue
                
                vpt_positions = vpt_obj_default_state[env_idx, :, :2]
                camera_distances_from_vpt = torch.norm(camera_pos.unsqueeze(0) - vpt_positions, dim=1)
                
                if not torch.all(camera_distances_from_vpt >= 1.0):
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
                
                # Select the tallest object from top 3
                # vpt_heights = vpt_obj_default_state[env_idx, :, 2]
                # top_3_indices = torch.topk(vpt_heights, k=3, largest=True).indices
                # tallest_obj_idx = top_3_indices[0].item()
                # print(f"Picked VPT object {tallest_obj_idx} of heights {vpt_heights[top_3_indices].cpu().numpy()}")
                random_obj_idx = random.randint(0, self.num_objs - 1)
                # print(f"Picked VPT object {random_obj_idx}")
                
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
                    
                    vpt_obj_default_state[env_idx, random_obj_idx, 0] = new_pos[0]
                    vpt_obj_default_state[env_idx, random_obj_idx, 1] = new_pos[1]
                    vpt_obj_default_state[env_idx, random_obj_idx, 2] = env_origin[2]
            
            
            for local_idx, env_idx in enumerate(valid_indices):
                if env_idx not in in_view_displaced:
                    continue
                
                print(f"Triggered VPT displacement for env {env_ids[env_idx].item()} due to in_view requirement")
                
                env_origin = env_origins[local_idx]
                
                env_id = valid_env_ids[local_idx]
                env_id_item = env_id.item()
                folder_idx = folder_indices[env_idx]
                camera_pos = camera_positions[local_idx]
                goal_pos = goal_default_state[env_idx, :3]
                
                # Pick one of the 3 smallest VPT obstacles
                # vpt_heights = vpt_obj_default_state[env_idx, :, 2]
                # bottom_3_indices = torch.topk(vpt_heights, k=3, largest=False).indices
                # random_obj_idx = bottom_3_indices[random.randint(0, 2)].item()
                # print(f"Picked VPT object {random_obj_idx} of heights {vpt_heights[bottom_3_indices].cpu().numpy()}")
                
                # Pick a random VPT obstacle
                random_obj_idx = random.randint(0, self.num_objs - 1)
                # print(f"Picked VPT object {random_obj_idx}")
                
                # Direction vector from camera to goal
                direction_cam_to_goal = goal_pos[:2] - camera_pos[:2]
                distance_cam_to_goal = torch.norm(direction_cam_to_goal)
                
                if distance_cam_to_goal > 1e-6:
                    # Normalize direction
                    direction_cam_to_goal = direction_cam_to_goal / distance_cam_to_goal
                    
                    # Place object at random point between 10-40% along the line from camera to goal
                    t = random.uniform(0.1, 0.6)  # Interpolation factor
                    new_pos = camera_pos[:2] + direction_cam_to_goal * (distance_cam_to_goal * t)
                    
                    vpt_pos = vpt_obj_default_state[env_idx, random_obj_idx, :3]
                    # print(f"Old pos = {vpt_pos[:2].cpu().numpy()}, New pos = {new_pos.cpu().numpy()}")
                    # print(f"Placed at {t*100:.1f}% along camera-to-goal line")
                    
                    vpt_obj_default_state[env_idx, random_obj_idx, 0] = new_pos[0]
                    vpt_obj_default_state[env_idx, random_obj_idx, 1] = new_pos[1]
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
        
        # VPT Distance check
        vpt_positions = self._vpt_objects.data.object_pos_w[env_ids, :, :2]
        distances_to_vpt = torch.norm(points.unsqueeze(1) - vpt_positions, dim=2)
        min_vpt_distances = distances_to_vpt.min(dim=1)[0]
        valid_mask &= (min_vpt_distances >= min_obstacle_distance)
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
        camera_vpt_distances = torch.norm(vpt_positions - camera_positions.unsqueeze(1), dim=2)
        min_camera_vpt_distances = camera_vpt_distances.min(dim=1)[0]
        valid_mask &= (min_camera_vpt_distances >= min_camera_obstacle_distance)
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
            maxRadius=150
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
        - VPT objects (positions, orientations, spawn cfg per object)
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
            },
            "collected_viewpoints": {
            "count": len(self.selected_viewpoints_for_collection[env_id_item]) if self.selected_viewpoints_for_collection[env_id_item] is not None else 0,
            "positions": self.selected_viewpoints_for_collection[env_id_item].cpu().numpy().tolist() if self.selected_viewpoints_for_collection[env_id_item] is not None else []
            }
        }
        
        # Add each VPT object with cfg metadata extracted at runtime
        for obj_idx in range(self.num_objs):
            # Extract spawn cfg directly from the VPT object's configuration
            vpt_spawn_cfg = self.cfg.vpt_objects.rigid_objects[list(self.cfg.vpt_objects.rigid_objects.keys())[obj_idx]].spawn
            
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
            "index": obj_idx,
            "position": vpt_positions[obj_idx],
            "orientation": vpt_orientations[obj_idx],
            "spawn_cfg": {
                **size_info,
                "rigid_props": rigid_props,
                "mass_props": mass_props,
                "visual_material": visual_material
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
