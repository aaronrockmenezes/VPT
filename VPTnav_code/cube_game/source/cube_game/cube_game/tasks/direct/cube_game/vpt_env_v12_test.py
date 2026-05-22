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

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.api import World
from pxr import Gf, Sdf, UsdGeom, Usd

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, RayCaster, save_images_to_file, Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, sample_gaussian, quat_from_euler_xyz
from isaaclab.utils import math as math_utils

from .vpt_env_cfg_v12 import VPTEnvCfg


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
        self.camera_pixel_threshold = 1200  # Minimum pixels for camera visibility

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
        self.base_path = f"/home/arock3/data_v13_latest/data_{self.GPU_ID}"
        # self.base_path = "/media/data_cifs_lrs/projects/prj_robotics/VPTnav_v6_1k_envs"
        self.visibility_labels_json_path = f"{self.base_path}/visibility_labels.json"

        # Mode determination
        if self.config_file is not None and os.path.exists(self.config_file):
            self.mode = "testing"
        else:
            self.mode = "data_collection"

        self.total_envs_to_sim = 800
        self.slot_to_env_id = list(range(self.num_envs))
        self.next_env_id = self.num_envs
        self.completed_envs = set()
        self.slot_attempt_counts = [0] * self.num_envs
        self.max_attempts_per_slot = 20 * 50  # Full resets * Inner resets

        self.used_vpt_objects = set()
        self._preallocate_visibility_labels()

        # Misc
        self.cross_scaling_factors = (1 / 0.06, 1 / 0.06, 1 / 0.020182)
        self.l_scaling_factors = (1 / 0.021433, 1 / 0.036474, 1 / 0.018880)
        self.t_scaling_factors = (1 / 0.020121, 1 / 0.034027, 1 / 0.018880)

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
        light_cfg.func("/World/Light", light_cfg)
        light_cfg_a.func("/World/Light_A", light_cfg_a)
        # light_cfg_a = sim_utils.SphereLightCfg(
        #     prim_type="SphereLight",
        #     intensity=1_000_000.0,
        #     treat_as_point=True,
        #     color=(0.75, 0.75, 0.75)
        # )

        # light_cfg_b = sim_utils.SphereLightCfg(
        #     prim_type="SphereLight",
        #     intensity=1_000_000.0,
        #     treat_as_point=True,
        #     color=(0.75, 0.75, 0.75)
        # )

        # light_cfg_c = sim_utils.SphereLightCfg(
        #     prim_type="SphereLight",
        #     intensity=1_000_000.0,
        #     treat_as_point=True,
        #     color=(0.75, 0.75, 0.75)
        # )

        # light_cfg_d = sim_utils.SphereLightCfg(
        #     prim_type="SphereLight",
        #     intensity=1_000_000.0,
        #     treat_as_point=True,
        #     color=(0.75, 0.75, 0.75)
        # )

        # light_cfg_e = sim_utils.SphereLightCfg(
        #     prim_type="SphereLight",
        #     intensity=1_000_000.0,
        #     treat_as_point=True,
        #     color=(0.75, 0.75, 0.75)
        # )
        # for i in range(0, self.num_envs):
        # Create the path string
        # prim_path = f"/World/envs/env_0/Light_A"
        # light_cfg_a.func(
        #         prim_path,
        #         light_cfg_a,
        #         translation=(2.0, 2.0, 8.5)  # <--- This is where the offset goes
        #     )

        # prim_path = f"/World/envs/env_0/Light_B"
        # light_cfg_b.func(
        #         prim_path,
        #         light_cfg_b,
        #         translation=(0.0, 0.0, 8.5)  # <--- This is where the offset goes
        #     )

        # prim_path = f"/World/envs/env_0/Light_C"
        # light_cfg_c.func(
        #         prim_path,
        #         light_cfg_c,
        #         translation=(0.0, 0.0, 8.5)  # <--- This is where the offset goes
        #     )

        # prim_path = f"/World/envs/env_0/Light_D"
        # light_cfg_d.func(
        #         prim_path,
        #         light_cfg_d,
        #         translation=(0.0, 0.0, 8.5)  # <--- This is where the offset goes
        #     )

        # prim_path = f"/World/envs/env_0/Light_E"
        # light_cfg_e.func(
        #         prim_path,
        #         light_cfg_e,
        #         translation=(0.0, 0.0, 8.5)  # <--- This is where the offset goes
        #     )
        # prim_path = f"/World/envs/env_0/Light_B"
        # self.cfg.light_cfg_b.func(
        #     prim_path,
        #     self.cfg.light_cfg_b,
        #     translation=(0.0, 0.0, 6.5)  # <--- This is where the offset goes
        # )
        for idx in range(0, 32):
            self.check_usd_pivot_alignment(
                prim_path_expr=f"/World/envs/env_0/obs_{idx}")
        
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
        sem_imgs = self._tiled_camera.data.output["semantic_segmentation"][
            env_ids]
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
            env_ids = torch.tensor(env_ids,
                                   dtype=torch.long,
                                   device=self.device)

        device = self._agent.device
        num_envs = len(env_ids)
        max_velocity = 3.0

        theta = math.pi / 12
        half_theta = theta / 2
        left_rot_quat = torch.tensor(
            [math.cos(half_theta), 0.0, 0.0,
             math.sin(half_theta)],
            device=device)
        right_rot_quat = torch.tensor(
            [math.cos(half_theta), 0.0, 0.0, -math.sin(half_theta)],
            device=device)

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
                new_quat[i] = math_utils.quat_mul(upright_quat[i],
                                                  left_rot_quat)
                desired_vel[i, :] = 0.0
            elif action == 3:
                new_quat[i] = math_utils.quat_mul(upright_quat[i],
                                                  right_rot_quat)
                desired_vel[i, :] = 0.0
            elif action == 4:
                desired_vel[i, :] = 0.0
            elif action == 5:
                env_id_item = env_ids[i].item()
                if (self.valid_viewpoint_poses is not None
                        and env_id_item < len(self.valid_viewpoint_poses)
                        and self.valid_viewpoint_poses[env_id_item] is not None
                        and len(self.valid_viewpoint_poses[env_id_item]) > 0):

                    current_idx = self.viewpoint_pose_counter[
                        env_id_item].item()
                    num_valid_poses = len(
                        self.valid_viewpoint_poses[env_id_item])
                    pose_idx = current_idx % num_valid_poses
                    target_pos = self.valid_viewpoint_poses[env_id_item][
                        pose_idx].to(device)
                    self.viewpoint_pose_counter[env_id_item] += 1

                    camera_pos_3d = self._camera_obj.data.root_pos_w[
                        env_ids[i]]
                    goal_pos_3d = self._goal.data.root_pos_w[env_ids[i]]
                    midpoint = (camera_pos_3d + goal_pos_3d) / 2.0
                    direction = midpoint[:2] - target_pos[:2]
                    yaw = torch.atan2(
                        direction[1], direction[0]) if torch.norm(
                            direction) > 1e-6 else torch.tensor(0.0)

                    new_quat[i] = torch.tensor([
                        math.cos(yaw.item() / 2), 0.0, 0.0,
                        math.sin(yaw.item() / 2)
                    ],
                                               device=device)
                    current_pos[i, :3] = target_pos
                desired_vel[i, :] = 0.0
            else:
                forward_input = 1.0 if action == 0 else -1.0
                local_movement = torch.tensor([forward_input, 0.0, 0.0],
                                              device=device)
                world_velocity = math_utils.quat_apply(
                    upright_quat[i], local_movement) * max_velocity
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
                        and len(self.valid_viewpoint_poses[env_id_item])
                        >= self.images_per_env):

                    all_viewpoints = self.valid_viewpoint_poses[env_id_item]

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
                        self.selected_viewpoints_for_collection[
                            env_id_item] = torch.stack(selected_points)
                        if self.verbose >= 1:
                            print(
                                f"✅ Env {env_id_item}: Selected {self.images_per_env} viewpoints for collection"
                            )
                    else:
                        if self.verbose >= 1:
                            print(
                                f"⚠️  Env {env_id_item}: Only {len(selected_points)} viewpoints available (need {self.images_per_env})"
                            )
                else:
                    if self.verbose >= 1:
                        print(
                            f"⚠️  Env {env_id_item}: No valid viewpoints available"
                        )

        self._agent.write_root_com_pose_to_sim(
            torch.cat([current_pos, new_quat], dim=1), env_ids)
        self._agent.write_root_com_velocity_to_sim(desired_vel, env_ids)
        self._agent.reset()

        camera_obj_pos = self._camera_obj.data.root_pos_w[env_ids].clone()
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_ids].clone()
        theta_left = math.pi / 2
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0,
             math.sin(half_theta_left)],
            device=device)
        rotated_orientations = math_utils.quat_mul(
            camera_obj_quat, left_90_quat.expand(num_envs, -1))
        self._occlusion_camera.set_world_poses(
            positions=camera_obj_pos,
            orientations=rotated_orientations,
            env_ids=env_ids.tolist(),
            convention="world")

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

    def _perform_final_validation(self, env_ids, original_folder_indices):
        """Perform final comprehensive validation check on all environments."""
        validation_results = []
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()
            folder_idx = original_folder_indices[i]
            visibility_reason = self.env_visibility_reasons.get(
                folder_idx, "unknown")

            result = {
                "env_id": env_id_item,
                "folder_idx": folder_idx,
                "label": self.env_visibility_labels.get(folder_idx, "UNKNOWN"),
                "reason": visibility_reason,
                "valid": True
            }

            if visibility_reason in ["in_view", "occluded"]:
                camera_pos = self._camera_obj.data.root_pos_w[env_id]
                goal_pos = self._goal.data.root_pos_w[env_id]
                is_occluded = self._check_occlusion_raycast(
                    camera_pos, goal_pos, env_id)
                expected_occluded = (visibility_reason == "occluded")

                result.update({
                    "expected_occluded": expected_occluded,
                    "actual_occluded": is_occluded,
                    "valid": is_occluded == expected_occluded
                })

            result["status"] = "✅" if result["valid"] else "❌"
            validation_results.append(result)

        return validation_results

    def _print_validation_summary(self, validation_results: list[dict],
                                  num_envs: int):
        """Print formatted and detailed summary of validation results."""
        num_valid = sum(1 for r in validation_results if r["valid"])

        print(f"\n{'='*80}")
        print(
            f"📊 VALIDATION SUMMARY: {num_valid}/{len(validation_results)} environments correct"
        )
        print(f"{'='*80}")

        in_view_results = [
            r for r in validation_results if r["reason"] == "in_view"
        ]
        occluded_results = [
            r for r in validation_results if r["reason"] == "occluded"
        ]
        outside_fov_results = [
            r for r in validation_results if r["reason"] == "outside_fov"
        ]

        if in_view_results:
            print(
                f"\n📍 IN_VIEW ({len(in_view_results)} envs - {len(in_view_results)/num_envs*100:.0f}%):"
            )
            for r in in_view_results:
                print(
                    f"  {r['status']} Env {r['env_id']} (folder {r['folder_idx']}): "
                    f"Label={r['label']}, Occluded={r.get('actual_occluded', 'N/A')}"
                )

        if occluded_results:
            print(
                f"\n🚫 OCCLUDED ({len(occluded_results)} envs - {len(occluded_results)/num_envs*100:.0f}%):"
            )
            for r in occluded_results:
                print(
                    f"  {r['status']} Env {r['env_id']} (folder {r['folder_idx']}): "
                    f"Label={r['label']}, Occluded={r.get('actual_occluded', 'N/A')}"
                )

        if outside_fov_results:
            print(
                f"\n👁️  OUTSIDE_FOV ({len(outside_fov_results)} envs - {len(outside_fov_results)/num_envs*100:.0f}%):"
            )
            for r in outside_fov_results:
                print(
                    f"  {r['status']} Env {r['env_id']} (folder {r['folder_idx']}): Label={r['label']}"
                )

        num_invalid = len(validation_results) - num_valid
        if num_invalid > 0:
            print(f"\n{'='*80}")
            print(
                f"⚠️  WARNING: {num_invalid} environment(s) have mismatched occlusion status!"
            )
            for r in validation_results:
                if not r["valid"]:
                    print(
                        f"  ❌ Env {r['env_id']} (folder {r['folder_idx']}): "
                        f"Expected {'occluded' if r['expected_occluded'] else 'visible'}, "
                        f"got {'occluded' if r['actual_occluded'] else 'visible'}"
                    )

        print(f"\n{'='*80}\n")

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

    def _load_env_config_from_json(self,
                                   config_filepath: str,
                                   target_env_id: int = 0):
        """
        Load environment configuration from JSON file and apply to specified environment.
        Restores positions, orientations, and validates spawn cfg matches current cfg.
        
        Args:
            config_filepath: Path to JSON configuration file
            target_env_id: Environment ID to load configuration into
        """
        import json

        if not os.path.exists(config_filepath):
            raise FileNotFoundError(
                f"Config file not found: {config_filepath}")

        # Load configuration
        with open(config_filepath, 'r') as f:
            config = json.load(f)

        device = self._agent.device

        # Convert target_env_id to tensor if needed
        if isinstance(target_env_id, int):
            env_ids = torch.tensor([target_env_id],
                                   dtype=torch.long,
                                   device=device)
        else:
            env_ids = torch.tensor([target_env_id.item()],
                                   dtype=torch.long,
                                   device=device)

        env_id_item = target_env_id if isinstance(
            target_env_id, int) else target_env_id.item()

        # Validate environment settings match current cfg
        env_settings = config.get("environment_settings", {})
        if env_settings:
            if env_settings.get("num_vpt_objs") != self.num_objs:
                print(
                    f"⚠️  Warning: Config has {env_settings.get('num_vpt_objs')} VPT objects, "
                    f"but current cfg has {self.num_objs}")

        # Extract configuration data
        goal_pos = torch.tensor(config["goal_ball"]["position"],
                                device=device,
                                dtype=torch.float32)
        goal_quat = torch.tensor(config["goal_ball"]["orientation"],
                                 device=device,
                                 dtype=torch.float32)

        camera_pos = torch.tensor(config["camera_object"]["position"],
                                  device=device,
                                  dtype=torch.float32)
        camera_quat = torch.tensor(config["camera_object"]["orientation"],
                                   device=device,
                                   dtype=torch.float32)

        agent_pos = torch.tensor(config["agent"]["position"],
                                 device=device,
                                 dtype=torch.float32)
        agent_quat = torch.tensor(config["agent"]["orientation"],
                                  device=device,
                                  dtype=torch.float32)

        # ========== CHANGE #3 & #4: Load active indices and restore them ==========
        active_indices = torch.tensor(config["vpt_objects"]["active_indices"],
                                      dtype=torch.long,
                                      device=device)
        self.active_vpt_indices[env_id_item] = active_indices
        # ==========================================================================

        # Build VPT object states - initialize all to zeros
        vpt_count = config["vpt_objects"]["total_count"]
        vpt_positions_full = torch.zeros((vpt_count, 3),
                                         device=device,
                                         dtype=torch.float32)
        vpt_orientations_full = torch.zeros((vpt_count, 4),
                                            device=device,
                                            dtype=torch.float32)
        vpt_orientations_full[:, 0] = 1.0  # Default quaternion (w=1, x=y=z=0)
        vpt_colors_full = torch.zeros((vpt_count, 3),
                                      device=device,
                                      dtype=torch.float32)

        # ========== CHANGE #5: Set inactive objects to storage position ==========
        # First, set all objects to storage position
        vpt_positions_full[:, :] = self.storage_position
        # =========================================================================

        # Now load ACTIVE object data from config
        for obj_data in config["vpt_objects"]["objects"]:
            obj_idx = obj_data["index"]
            vpt_positions_full[obj_idx] = torch.tensor(obj_data["position"],
                                                       device=device,
                                                       dtype=torch.float32)
            vpt_orientations_full[obj_idx] = torch.tensor(
                obj_data["orientation"], device=device, dtype=torch.float32)
            vpt_colors_full[obj_idx] = torch.tensor(
                obj_data["spawn_cfg"]["visual_material"]["diffuse_color"],
                device=device,
                dtype=torch.float32)

        # Apply goal ball configuration
        goal_full_state = torch.cat(
            [goal_pos, goal_quat,
             torch.zeros(6, device=device)], dim=-1)
        self._goal.data.default_root_state[
            env_ids] = goal_full_state.unsqueeze(0)
        self._goal.write_data_to_sim()

        # Apply camera object configuration
        camera_pose = torch.cat([
            camera_pos.unsqueeze(0),
            camera_quat.unsqueeze(0),
            torch.zeros((1, 6), device=device)
        ],
                                dim=1)
        self._camera_obj.data.default_root_state[env_ids] = camera_pose
        self._camera_obj.write_data_to_sim()

        # Apply agent configuration
        agent_pose = torch.cat([
            agent_pos.unsqueeze(0),
            agent_quat.unsqueeze(0),
            torch.zeros((1, 6), device=device)
        ],
                               dim=1)
        self._agent.data.default_root_state[env_ids] = agent_pose
        self._agent.write_data_to_sim()

        # Apply VPT objects configuration (all 200: 20 active + 180 in storage)
        vpt_poses = torch.cat([
            vpt_positions_full.unsqueeze(0),
            vpt_orientations_full.unsqueeze(0),
            torch.zeros((1, vpt_count, 6), device=device)
        ],
                              dim=2)
        self._vpt_objects.data.default_object_state[env_ids] = vpt_poses
        self._vpt_objects.write_data_to_sim()

        # Apply colors to all VPT objects
        for obj_idx, obj_key in enumerate(
                self.cfg.vpt_objects.rigid_objects.keys()):
            color = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(vpt_colors_full[obj_idx].cpu().numpy()))
            self._vpt_objects.cfg.rigid_objects[
                obj_key].spawn.visual_material = color
            self._vpt_objects.write_data_to_sim()

        # Update occlusion camera to match camera object
        theta_left = math.pi / 2
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0,
             math.sin(half_theta_left)],
            device=device)

        rotated_orientation = math_utils.quat_mul(camera_quat.unsqueeze(0),
                                                  left_90_quat.unsqueeze(0))

        self._occlusion_camera.set_world_poses(
            positions=camera_pos.unsqueeze(0),
            orientations=rotated_orientation,
            env_ids=env_ids.tolist(),
            convention="world")

        # Load valid viewpoints
        if "valid_viewpoints" in config and config["valid_viewpoints"][
                "count"] > 0:
            valid_viewpoints = torch.tensor(
                config["valid_viewpoints"]["positions"],
                device=device,
                dtype=torch.float32)

            if self.valid_viewpoint_poses is None:
                self.valid_viewpoint_poses = [None] * self.num_envs

            self.valid_viewpoint_poses[env_id_item] = valid_viewpoints

        # Load collected viewpoints if present
        if "collected_viewpoints" in config and config["collected_viewpoints"][
                "count"] > 0:
            collected_viewpoints = torch.tensor(
                config["collected_viewpoints"]["positions"],
                device=device,
                dtype=torch.float32)
            self.selected_viewpoints_for_collection[
                env_id_item] = collected_viewpoints

        # Update visibility labels
        folder_idx = config["metadata"]["folder_idx"]
        self.env_visibility_labels[folder_idx] = config["metadata"][
            "visibility_label"]
        self.env_visibility_reasons[folder_idx] = config["metadata"][
            "visibility_reason"]

        # Simulate a few steps to stabilize
        for _ in range(3):
            self.sim.step()

        if self.verbose >= 1:
            print(
                f"✅ Loaded environment configuration from: {config_filepath}")
            print(
                f"   → Env {target_env_id}, Folder {folder_idx}, Label: {config['metadata']['visibility_label']}"
            )
            print(f"   → Reason: {config['metadata']['visibility_reason']}")
            print(
                f"   → Valid Viewpoints: {config['valid_viewpoints']['count']}"
            )
            print(
                f"   → Active VPT objects: {len(active_indices)}/{vpt_count}")

        self.mode = "testing"

    def _check_target_vpt_distance(self,
                                   env_idx,
                                   target_position,
                                   vpt_positions,
                                   min_distance=1.0):
        """Check if VPT object surfaces are at least min_distance away from target position."""
        vpt_dims = self._get_active_vpt_dims(env_idx)

        # Get VPT center positions in 3D (positions are at bottom center, so add half height)
        vpt_positions_3d = torch.zeros((self.num_objs, 3),
                                       device=self._agent.device)
        vpt_positions_3d[:, :2] = vpt_positions
        vpt_positions_3d[:,
                         2] = vpt_dims[:,
                                       2] / 2.0  # Add half Z-dimension to get center

        # Get target position in 3D
        goal_radius = self.goal_radius + 0.001  # (0.15 radius + 1 cm buffer)
        target_position_3d = torch.zeros((3), device=self._agent.device)
        target_position_3d[:2] = target_position[:2]
        target_position_3d[2] = goal_radius

        # Compute closest point on each VPT cuboid surface to the target
        target_expanded = target_position_3d.unsqueeze(0).expand_as(
            vpt_positions_3d)

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
        """Get dimensions of active VPT objects for given environment(s).
        
        Args:
            env_ids: int, single-element Tensor, or list/Tensor of batch indices.
            
        Returns:
            Tensor of shape (batch_size, active_vpt_objs, 3) containing X, Y, Z dimensions
        """
        # 1. Standardize input to Tensor
        if isinstance(env_ids, int):
            env_ids = torch.tensor([env_ids], device=self.device)
        elif torch.is_tensor(env_ids) and env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device)

        # 2. Handle List vs Tensor for active_vpt_indices lookup
        if isinstance(self.active_vpt_indices, list):
            # Check if it's a list of tensors or list of ints
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

        # 3. Get Active Indices for these envs
        # active_indices_tensor shape: [num_envs, active_objs]
        # env_ids shape: [batch_size]
        batch_indices = active_indices_tensor[
            env_ids]  # Shape: [batch_size, active_objs]

        # 4. Lookup Dimensions using Advanced Indexing (UPDATED)
        # self.all_vpt_dims shape is now: [num_envs, num_objs, 3]
        # We need to match Env IDs with Object IDs to pull the correct randomized size.

        # Expand env_ids to match batch_indices shape:
        # [batch_size] -> [batch_size, 1] -> [batch_size, active_objs]
        env_ids_expanded = env_ids.unsqueeze(1).expand_as(batch_indices)

        # Advanced Indexing: Select specific (Env, Object) pairs
        batch_dims = self.all_vpt_dims[env_ids_expanded, batch_indices, :]
        # Result Shape: [batch_size, active_objs, 3]

        return batch_dims

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
        active_positions = self._vpt_objects.data.object_pos_w[env_id,
                                                               active_indices]

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

    def _check_camera_vpt_clearance(self, env_id_item: int,
                                    camera_pos_3d: torch.Tensor,
                                    vpt_pos_all: torch.Tensor,
                                    min_dist: float) -> bool:
        """
        Checks if camera is at least min_dist away from any active VPT object surface.
        Calculates distance from Camera Point to Object Cuboid Surface.
        """
        active_indices = self.active_vpt_indices[env_id_item]

        # Get active positions and dimensions
        active_vpt_pos = vpt_pos_all[active_indices]  # [N_active, 3]
        vpt_dims = self.all_vpt_dims[env_id_item,
                                     active_indices, :]  # [N_active, 3]

        # Calculate Physical Centers of objects
        # Center_Z = Pos_Z + Height * (0.5 - Pivot_Ratio)
        # Ratio is 0.0 for base-pivoted (Center=Pos+H/2), 0.5 for center-pivoted (Center=Pos)
        ratios = self.vpt_z_offset_ratios[active_indices]
        centers_z = active_vpt_pos[:, 2] + vpt_dims[:, 2] * (0.5 - ratios)
        vpt_centers = torch.stack(
            [active_vpt_pos[:, 0], active_vpt_pos[:, 1], centers_z], dim=1)

        # Expand Camera Position to match objects
        cam_expanded = camera_pos_3d.unsqueeze(0).expand_as(vpt_centers)

        # Calculate vector from Object Center to Camera
        diff = cam_expanded - vpt_centers
        half_extents = vpt_dims / 2.0

        # Clamp to get closest point on cuboid surface
        closest_local = torch.clamp(diff, -half_extents, half_extents)
        closest_world = vpt_centers + closest_local

        # Calculate distance
        dists = torch.norm(cam_expanded - closest_world, dim=1)

        return torch.all(dists >= min_dist).item()

    def _sample_valid_initial_poses(
        self,
        env_ids: torch.Tensor,
        max_attempts: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """
        Samples valid initial poses with collision and distance checks.
        Returns: (goal_state, cam_state, agent_state, vpt_state, success_mask)
        """
        device = self._agent.device
        num_envs = len(env_ids)

        safe_x = self.center_to_boundary - 4.0
        safe_obs = self.center_to_boundary - 3.5
        env_origins = self.scene.env_origins[env_ids]

        goal_state = self._goal.data.default_root_state[env_ids].clone()
        cam_state = self._camera_obj.data.default_root_state[env_ids].clone()
        agent_state = self._agent.data.default_root_state[env_ids].clone()
        vpt_state = self._vpt_objects.data.default_object_state[env_ids].clone(
        )

        envs_need_retry = torch.ones(num_envs, dtype=torch.bool, device=device)

        for _ in range(max_attempts):
            if not envs_need_retry.any(): break

            retry_idxs = torch.where(envs_need_retry)[0]
            batch_size = len(retry_idxs)
            curr_env_ids = env_ids[retry_idxs]

            # Sample random offsets
            goal_off = sample_uniform(-safe_x, safe_x, (batch_size, 2), device)
            cam_off = sample_uniform(-safe_x, safe_x, (batch_size, 2), device)
            agent_off = sample_uniform(-safe_x, safe_x, (batch_size, 2),
                                       device)
            goal_perturb = sample_uniform(-0.4, 0.4, (batch_size, 2), device)

            # Apply Goal & Camera Poses
            goal_state[retry_idxs,
                       0] = env_origins[retry_idxs,
                                        0] + goal_off[:, 0] + goal_perturb[:,
                                                                           0]
            goal_state[retry_idxs,
                       1] = env_origins[retry_idxs,
                                        1] + goal_off[:, 1] + goal_perturb[:,
                                                                           1]
            goal_state[retry_idxs, 2] = env_origins[retry_idxs, 2]

            cam_state[retry_idxs,
                      0] = env_origins[retry_idxs, 0] + cam_off[:, 0]
            cam_state[retry_idxs,
                      1] = env_origins[retry_idxs, 1] + cam_off[:, 1]

            # Look at Goal
            dir_to_goal = goal_state[retry_idxs, :2] - cam_state[
                retry_idxs, :2]
            yaw = torch.atan2(dir_to_goal[:, 1],
                              dir_to_goal[:, 0]) - math.radians(90)
            roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
            cam_state[retry_idxs,
                      3:7] = quat_from_euler_xyz(roll, torch.zeros_like(yaw),
                                                 yaw)

            agent_state[retry_idxs,
                        0] = env_origins[retry_idxs, 0] + agent_off[:, 0]
            agent_state[retry_idxs,
                        1] = env_origins[retry_idxs, 1] + agent_off[:, 1]

            # VPT Placement (Greedy Non-Overlapping)
            self._cache_z_offsets()
            for i, env_idx in enumerate(retry_idxs):
                active_indices = self.active_vpt_indices[
                    curr_env_ids[i].item()]
                active_dims = self.all_vpt_dims[env_idx, active_indices, :2]
                placed_pos, placed_dims = [], []
                placement_failed, margin = False, 0.05

                for k, obj_idx in enumerate(active_indices):
                    curr_w, curr_l = active_dims[k, 0], active_dims[k, 1]
                    found = False
                    for _ in range(20):
                        rx = (torch.rand(1, device=device).item() * 2 *
                              safe_obs) - safe_obs
                        ry = (torch.rand(1, device=device).item() * 2 *
                              safe_obs) - safe_obs

                        # Collision check against previously placed
                        collision = any(
                            abs(rx - px) < (curr_w + pw) / 2.0 +
                            margin and abs(ry -
                                           py) < (curr_l + pl) / 2.0 + margin
                            for px, py, (
                                pw,
                                pl) in zip([p[0] for p in placed_pos],
                                           [p[1]
                                            for p in placed_pos], placed_dims))

                        if not collision:
                            placed_pos.append((rx, ry))
                            placed_dims.append((curr_w, curr_l))
                            vpt_state[env_idx, obj_idx,
                                      0] = env_origins[i, 0] + rx
                            vpt_state[env_idx, obj_idx,
                                      1] = env_origins[i, 1] + ry
                            found = True
                            break
                    if not found:
                        placement_failed = True
                        break
                if placement_failed: continue

            # Move inactive to storage and validate distances
            vpt_state[retry_idxs] = self._store_inactive_vpt_objects(
                curr_env_ids, vpt_state[retry_idxs])

            cam_pos = cam_state[retry_idxs, :2]
            dist_goal = torch.norm(cam_pos - goal_state[retry_idxs, :2], dim=1)
            valid_goal = (dist_goal >= 2.0) & (dist_goal <= 15.0)

            # Check min distance to any active VPT object using utility function (1.5 units)
            valid_vpt = torch.zeros(batch_size,
                                    dtype=torch.bool,
                                    device=device)
            cam_pos_3d = cam_state[retry_idxs, :
                                   3]  # 3D position for precise check

            for k, r_idx in enumerate(retry_idxs):
                env_id_curr = curr_env_ids[k].item()
                # Check clearance for this specific env
                is_clear = self._check_camera_vpt_clearance(env_id_curr,
                                                            cam_pos_3d[k],
                                                            vpt_state[r_idx],
                                                            min_dist=1.5)
                valid_vpt[k] = is_clear

            envs_need_retry[retry_idxs[valid_goal & valid_vpt]] = False

        return goal_state, cam_state, agent_state, vpt_state, ~envs_need_retry

    def _reset_idx(self,
                   env_ids: Sequence[int] | None,
                   randomize_objects: bool = True) -> None:
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
            env_ids = torch.tensor(env_ids,
                                   dtype=torch.long,
                                   device=self.device)

        num_slots = len(env_ids)
        # Prepare initial batch: assign folder indices and visibility labels
        initial_folder_indices = [
            self.next_env_folder_idx + i for i in range(num_slots)
        ]

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
        selected_indices = torch.randperm(len(env_ids),
                                          device=self.device)[:num_to_select]
        self.envs_to_move_ball = env_ids[selected_indices]

        # Track folder indices and visibility per slot
        slot_folder_indices = initial_folder_indices.copy()
        slot_visibility_categories = visibility_categories.copy()

        vpt_prim_paths = [
            f"/World/envs/env_.*/obs_{idx}"
            for idx in range(self.cfg.num_vpt_objs)
        ]

        vpt_mat_prim_paths = []
        for env_idx in range(self.num_envs):
            for obj_idx in range(self.cfg.num_vpt_objs):
                vpt_mat_prim_paths.append(f"/World/envs/env_{env_idx}/obs_{obj_idx}")

        # Main streaming loop
        iteration = 0
        self._cache_base_dims()
        while self.next_env_id < self.total_envs_to_sim or len(
                self.completed_envs) < self.total_envs_to_sim:
            iteration += 1

            if self.verbose >= 1:
                print(f"\n{'='*60}")
                print(
                    f"🔄 Iteration {iteration} | Envs in slots: {self.slot_to_env_id}"
                )
                print(f"{'='*60}")

            randomize_start_time = time.time()
            print(
                f"Rescaling VPT obstacles for {self.cfg.num_vpt_objs} objects..."
            )
            self.randomize_shape_scale(prim_path_expr=vpt_prim_paths,
                                       is_random=True)
            # print(
            #     f"Recoloring VPT obstacles for {self.cfg.num_vpt_objs} objects..."
            # )
            # self.randomize_shape_color(prim_path_expr=vpt_prim_paths)
            # print(f"Randomizing environment floor color...")
            # self.randomize_shape_color(
            #     prim_path_expr=["/World/envs/env_.*/mat"])
            self.randomize_mat_material(prim_paths=[f"/World/envs/env_{i}/mat" for i in range(self.num_envs)])
            self.randomize_vpt_material(prim_paths=vpt_mat_prim_paths)
            self.randomize_shape_color(prim_path_expr=[
                "/World/envs/env_.*/bottom_wall",
                "/World/envs/env_.*/right_wall",
                "/World/envs/env_.*/left_wall", "/World/envs/env_.*/top_wall"
            ])
            print(f"Resetting lighting")
            self.randomize_global_light(prim_path="/World/Light")
            self.randomize_global_light(prim_path="/World/Light_A")
            # self.randomize_sphere_lights(light_names=["Light_A", "Light_B"], z_heights=[None, None])
            # self.randomize_sphere_lights()
            randomize_end_time = time.time()
            print(
                f"Randomization time for size and color = {np.round(randomize_end_time - randomize_start_time, 3) / self.cfg.num_vpt_objs}"
            )

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
                reset_visibility_categories.append(
                    slot_visibility_categories[slot_idx])

            if reset_env_ids:
                if self.verbose >= 1:
                    print(
                        f"🔄 Resetting {len(reset_env_ids)} active slot(s)...")

                self._reset_idx_internal(
                    torch.tensor(reset_env_ids,
                                 dtype=torch.long,
                                 device=self.device),
                    randomize_objects,
                    folder_indices=reset_folder_indices,
                    visibility_categories=reset_visibility_categories)

                # Step simulation
                for _ in range(2):
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
                is_valid, reason = self._validate_env_state(
                    env_ids[slot_idx], folder_idx, MIN_VALID_VIEWPOINTS)

                if is_valid:
                    valid_slots.append(slot_idx)
                    if self.verbose >= 1:
                        print(
                            f"  ✅ Slot {slot_idx} | Env {env_id} | Folder {folder_idx} VALIDATED"
                        )
                else:
                    self.slot_attempt_counts[slot_idx] += 1

                    if self.slot_attempt_counts[
                            slot_idx] >= self.max_attempts_per_slot:
                        exceeded_slots.append(slot_idx)
                        if self.verbose >= 1:
                            print(
                                f"  ⚠️ Slot {slot_idx} | Env {env_id} | EXCEEDED max attempts ({self.max_attempts_per_slot}), giving up"
                            )
                    else:
                        failed_slots.append(slot_idx)
                        if self.verbose >= 2:
                            print(
                                f"  ❌ Slot {slot_idx} | Env {env_id} | Attempt {self.slot_attempt_counts[slot_idx]}/{self.max_attempts_per_slot}: {reason}"
                            )

            # 3. Collect images for valid slots
            if valid_slots:
                if self.verbose >= 1:
                    print(
                        f"\n📸 Collecting images for {len(valid_slots)} valid slot(s): {valid_slots}"
                    )

                for slot_idx in valid_slots:
                    env_id = self.slot_to_env_id[slot_idx]
                    folder_idx = slot_folder_indices[slot_idx]

                    # Select viewpoints for this slot
                    if not self._select_viewpoints_for_collection(slot_idx):
                        if self.verbose >= 1:
                            print(
                                f"    ⚠️  Failed to select viewpoints for slot {slot_idx}"
                            )
                        continue

                    # Collect images for this slot
                    self._collect_images_for_slot(env_ids[slot_idx],
                                                  folder_idx)

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
                    new_visibility = self._assign_next_visibility_label(
                        new_folder_idx)

                    # Update tracking (reset happens next iteration)
                    self.slot_to_env_id[slot_idx] = new_env_id
                    self.slot_attempt_counts[slot_idx] = 0
                    slot_folder_indices[slot_idx] = new_folder_idx
                    slot_visibility_categories[slot_idx] = new_visibility

                    self.next_env_id += 1

                    if self.verbose >= 1:
                        print(
                            f"  🔄 Slot {slot_idx}: Replacing env {old_env_id} → env {new_env_id} (folder {new_folder_idx})"
                        )

                self._save_visibility_labels()

            # 5. Check if done
            if len(self.completed_envs) >= self.total_envs_to_sim:
                if self.verbose >= 1:
                    print(
                        f"\n🎉 SUCCESS: Completed all {self.total_envs_to_sim} environments!"
                    )
                    sys.exit(1)
                break

            # Progress update
            if self.verbose >= 1:
                print(
                    f"\n⏳ Progress | Completed: {len(self.completed_envs)}/{self.total_envs_to_sim} | Next env: {self.next_env_id}"
                )

        # Final validation steps
        for _ in range(2):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.step_dt)

        self._reset_called = True

    def initial_spawn_loop(self,
                           env_ids,
                           envs_need_spawn_retry,
                           safe_range: float,
                           states,
                           device=None):
        if device is None:
            device = self._agent.device

        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        retry_mask = envs_need_spawn_retry.clone()
        batch_size = retry_mask.sum().item()
        retry_indices = torch.where(retry_mask)[0]

        safe_x_range = safe_range - 4.0
        safe_x_range_obstacles = safe_range - 3.5

        # --- Standard Pre-calculation ---
        goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        camera_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        goal_perturb_offsets = sample_uniform(-0.8, 0.8, (batch_size, 2), device)

        env_ids_batch = env_ids[retry_indices]
        env_origins = self.scene.env_origins[env_ids_batch]

        # Goal Setup
        goal_default_state[retry_indices, 0] = env_origins[:, 0] + goal_offsets[:, 0]
        goal_default_state[retry_indices, 1] = env_origins[:, 1] + goal_offsets[:, 1]
        goal_default_state[retry_indices, 2] = env_origins[:, 2]

        # Camera Setup
        camera_obj_default_state[retry_indices, 0] = env_origins[:, 0] + camera_offsets[:, 0]
        camera_obj_default_state[retry_indices, 1] = env_origins[:, 1] + camera_offsets[:, 1]

        # Camera Orientation
        direction_to_goal = goal_default_state[retry_indices, :2] - camera_obj_default_state[retry_indices, :2]
        yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0]) - math.radians(90)
        roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
        zero = torch.zeros_like(yaw)
        quaternion = quat_from_euler_xyz(roll, zero, yaw)
        camera_obj_default_state[retry_indices, 3:7] = quaternion

        # Goal Perturbation
        goal_default_state[retry_indices, 0] += goal_perturb_offsets[:, 0]
        goal_default_state[retry_indices, 1] += goal_perturb_offsets[:, 1]

        # Agent Setup
        agent_default_state[retry_indices, 0] = env_origins[:, 0] + agent_offsets[:, 0]
        agent_default_state[retry_indices, 1] = env_origins[:, 1] + agent_offsets[:, 1]

        self._cache_z_offsets()

        # [CRITICAL FIX] 1. Hide Inactive Objects FIRST
        # We do this before the placement loop so we don't accidentally overwrite our new positions later.
        vpt_obj_default_state[retry_indices] = self._store_inactive_vpt_objects(
                env_ids[retry_indices], vpt_obj_default_state[retry_indices])

        # 2. ROBUST VPT PLACEMENT LOOP
        for batch_idx, env_idx in enumerate(retry_indices):
            env_id = env_ids[env_idx]
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id

            active_indices = self.active_vpt_indices[env_id_item]
            # Ensure these are the correct Ground Plane dimensions (X and Y)
            active_dims = self.all_vpt_dims[env_idx, active_indices, :2]

            placed_pos = []  
            placed_radii = [] 
            placement_failed = False

            margin = 0.1

            for i, obj_idx in enumerate(active_indices):
                curr_w = active_dims[i, 0]
                curr_l = active_dims[i, 1]

                # Calculate Safe Radius (covers object at any rotation)
                curr_radius = math.hypot(curr_w, curr_l) * 0.6
                
                found_valid_pos = False

                for _ in range(20):
                    # Random Yaw
                    obj_yaw = (torch.rand(1, device=device).item() * 2 * math.pi) - math.pi
                    
                    # Random Position (Standard Range)
                    rx = (torch.rand(1, device=device).item() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
                    ry = (torch.rand(1, device=device).item() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles

                    # Collision Check
                    collision = False
                    for p_idx, (px, py) in enumerate(placed_pos):
                        p_radius = placed_radii[p_idx]
                        dist = math.hypot(rx - px, ry - py)
                        min_dist = curr_radius + p_radius + margin
                        if dist < min_dist:
                            collision = True
                            break

                    if not collision:
                        placed_pos.append((rx, ry))
                        placed_radii.append(curr_radius)
                        found_valid_pos = True

                        # Assign Position
                        vpt_obj_default_state[env_idx, obj_idx, 0] = env_origins[batch_idx, 0] + rx
                        vpt_obj_default_state[env_idx, obj_idx, 1] = env_origins[batch_idx, 1] + ry
                        vpt_obj_default_state[env_idx, obj_idx, 2] = self.vpt_obj_default_state[env_idx, obj_idx, 2]
                        
                        # Assign Rotation
                        q_roll = torch.tensor([0.0], device=device)
                        q_pitch = torch.tensor([0.0], device=device)
                        q_yaw = torch.tensor([obj_yaw], device=device)
                        
                        quat = quat_from_euler_xyz(q_roll, q_pitch, q_yaw)
                        if quat.shape[0] == 1 and len(quat.shape) == 2:
                            quat = quat.squeeze(0)
                        vpt_obj_default_state[env_idx, obj_idx, 3:7] = quat
                        break
                    
                if not found_valid_pos:
                    placement_failed = True
                    break

            if placement_failed:
                continue

        # 3. DEBUG VALIDATION LOOP
        # print(f"\n[Debug] Starting Validation. Safe Range limit: {safe_range} (Wall) -> Checking against {safe_range - 0.1}")
        
        for batch_idx, env_idx in enumerate(retry_indices):
            env_id = env_ids[env_idx]
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            
            # 1. Get Active Indices & Positions
            active_indices = self.active_vpt_indices[env_id_item]
            current_env_origin = env_origins[batch_idx, :2]
            vpt_positions = vpt_obj_default_state[env_idx, active_indices, :2]
            
            # 2. Check Bounds
            vpt_local_positions = vpt_positions - current_env_origin
            out_of_bounds_mask = torch.any(torch.abs(vpt_local_positions) > (safe_range - 0.1), dim=1)
            
            if torch.any(out_of_bounds_mask):
                failed_local_indices = torch.where(out_of_bounds_mask)[0]
                failed_global_indices = active_indices[failed_local_indices]
                failed_coords = vpt_local_positions[failed_local_indices]
                
                print(f"  ❌ Env {env_id_item} FAILED Bounds Check.")
                # If these print as 0s, placement loop failed entirely or didn't run.
                for i, idx in enumerate(failed_local_indices):
                    g_idx = failed_global_indices[i].item()
                    coords = failed_coords[i].cpu().numpy()
                    print(f"     - Obj ID {g_idx} Local Pos: {coords}")
                continue

            # 3. Check Camera Logic
            agent_pos = agent_default_state[env_idx, :2]
            goal_pos = goal_default_state[env_idx, :2]
            camera_pos = camera_obj_default_state[env_idx, :2]

            camera_goal_distance = torch.norm(camera_pos - goal_pos)
            if camera_goal_distance > 15.0 or camera_goal_distance < 2.0:
                continue

            camera_distances_from_vpt = torch.norm(camera_pos.unsqueeze(0) - vpt_positions, dim=1)
            if not torch.all(camera_distances_from_vpt >= 3.0):
                continue

            envs_need_spawn_retry[env_idx] = False
            # print(f"  ✅ Env {env_id_item} passed")

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

        safe_x_range = safe_range - 4.0
        safe_x_range_obstacles = safe_range - 3.5

        if len(move_ball_indices) > 0:
            # 1. Prepare Batch Data
            target_env_ids = env_ids[move_ball_indices]
            batch_size = len(target_env_ids)

            if isinstance(self.active_vpt_indices, list):
                if len(self.active_vpt_indices) > 0 and isinstance(
                        self.active_vpt_indices[0], torch.Tensor):
                    # Case A: List of Tensors -> Stack them
                    active_indices_tensor = torch.stack(
                        self.active_vpt_indices).to(device)
                else:
                    # Case B: List of Ints/Lists -> Convert directly
                    active_indices_tensor = torch.tensor(
                        self.active_vpt_indices,
                        device=device,
                        dtype=torch.long)
            else:
                # Case C: Already a Tensor
                active_indices_tensor = self.active_vpt_indices

            # Get active global indices: [batch_size, active_vpt_objs]
            batch_active_indices = active_indices_tensor[target_env_ids]

            # Get Dimensions: [batch_size, active_vpt_objs, 3]
            batch_dims = self._get_active_vpt_dims(target_env_ids)
            batch_heights = batch_dims[:, :, 2]  # Z dimension

            # 2. Build Candidates Mask
            # Constraint A: Height < 0.5
            height_mask = batch_heights < 0.5

            # Constraint B: Valid Shape (Cylinder/Cuboid)
            # Index into the global shape mask we created
            shape_mask = self.valid_shape_mask[batch_active_indices]

            # Combined Mask: [batch_size, active_vpt_objs]
            candidate_mask = height_mask & shape_mask

            # Check which envs actually have at least one valid object
            has_candidates_mask = candidate_mask.any(dim=1)

            if not has_candidates_mask.all():
                # Filter down to only envs that can actually move a ball
                skipped_indices = move_ball_indices[~has_candidates_mask]
                if len(skipped_indices) > 0:
                    print(
                        f"  ❌ Skipping ball move for {len(skipped_indices)} envs (no valid objects found)."
                    )

                # Filter everything to valid subset
                target_env_ids = target_env_ids[has_candidates_mask]
                move_ball_indices = move_ball_indices[has_candidates_mask]
                batch_active_indices = batch_active_indices[
                    has_candidates_mask]
                batch_heights = batch_heights[has_candidates_mask]
                candidate_mask = candidate_mask[has_candidates_mask]
                batch_size = len(target_env_ids)

            if batch_size > 0:
                # 3. Select One Object Per Env
                # Add epsilon to mask to create weights (0.0 becomes 1e-6, 1.0 becomes 1.000001)
                weights = candidate_mask.float() + 1e-6
                # Multinomial samples indices where weights are high
                selected_local_indices = torch.multinomial(weights,
                                                           1).squeeze(-1)

                # Map local index (0..19) back to global object ID
                selected_global_indices = torch.gather(
                    batch_active_indices, 1,
                    selected_local_indices.unsqueeze(1)).squeeze(-1)

                # Get heights of selected objects
                selected_heights = torch.gather(
                    batch_heights, 1,
                    selected_local_indices.unsqueeze(1)).squeeze(-1)

                # 4. Update Goal & Object Poses
                target_goal_pos = goal_default_state[move_ball_indices, :3]
                target_env_origins = self.scene.env_origins[target_env_ids]

                # [FIX START] Use z_ratios instead of hardcoded 2.0
                # Get ratios for the selected objects
                selected_ratios = self.vpt_z_offset_ratios[
                    selected_global_indices]

                # Calculate new Z: Origin + (Height * Ratio)
                new_obj_z = target_env_origins[:, 2] + (selected_heights *
                                                        selected_ratios)
                # [FIX END]

                # Calculate new Z for Goal: Object Top + Radius + Buffer
                # Goal is always on top, so we use full height regardless of pivot
                # Top of object = Origin + Height (if base pivot) or Origin + Height/2 (if center pivot)?
                # Actually: Top of object = Object_Z_Pos + (Height * (1 - Ratio))
                # Easier way: Just use the Physical Top
                obj_physical_top = target_env_origins[:, 2] + selected_heights
                new_goal_z = obj_physical_top + (self.goal_radius + 0.01)

                # Apply updates to default state tensors
                # Update Object X, Y, Z
                vpt_obj_default_state[move_ball_indices,
                                      selected_global_indices,
                                      0] = target_goal_pos[:, 0]
                vpt_obj_default_state[move_ball_indices,
                                      selected_global_indices,
                                      1] = target_goal_pos[:, 1]
                # vpt_obj_default_state[move_ball_indices, selected_global_indices, 2] = new_obj_z

                # Update Goal Z
                goal_default_state[move_ball_indices, 2] = new_goal_z

                # Update tracking dict (CPU sync required only for this debug dict)
                # We do this efficiently
                move_ball_cpu = move_ball_indices.cpu().numpy()
                selected_global_cpu = selected_global_indices.cpu().numpy()
                for i, env_idx in enumerate(move_ball_cpu):
                    moved_vpt_for_ball[env_idx] = selected_global_cpu[i]

                # 5. Conflict Resolution (Batch)
                # Check if any *other* active objects are too close to the goal

                # Get positions of ALL active objects: [batch, active_objs, 2]
                # We fetch from default_state because we just updated the selected one there!
                # We need to expand target_env_ids to index dim 0
                all_active_pos = vpt_obj_default_state[
                    move_ball_indices.unsqueeze(1), batch_active_indices, :2]

                # Calculate distance from Goal (XY): [batch, active_objs]
                dists = torch.norm(all_active_pos -
                                   target_goal_pos[:, :2].unsqueeze(1),
                                   dim=2)

                # Create selection mask to ignore the object we just moved
                selection_one_hot = torch.nn.functional.one_hot(
                    selected_local_indices,
                    num_classes=self.active_vpt_objs).bool()

                # Conflict if: Distance < 1.5 AND Not the selected object
                conflict_mask = (dists < 1.5) & (~selection_one_hot)

                if conflict_mask.any():
                    # Resolve conflicts by scattering new random positions
                    num_conflicts = conflict_mask.sum()

                    # Generate random offsets
                    new_x = sample_uniform(-safe_x_range_obstacles,
                                           safe_x_range_obstacles,
                                           (num_conflicts, ), device)
                    new_y = sample_uniform(-safe_x_range_obstacles,
                                           safe_x_range_obstacles,
                                           (num_conflicts, ), device)

                    # Identify which (env, obj) pairs are conflicts
                    # We need to map these back to indices in vpt_obj_default_state

                    # Expand move_ball_indices to match shape [batch, active_objs]
                    expanded_env_indices = move_ball_indices.unsqueeze(
                        1).expand_as(conflict_mask)
                    # active indices are already [batch, active_objs]

                    # Filter indices using the mask
                    conflict_env_idxs = expanded_env_indices[conflict_mask]
                    conflict_obj_idxs = batch_active_indices[conflict_mask]

                    # Get origins for these specific conflict envs
                    conflict_origins = self.scene.env_origins[
                        env_ids[conflict_env_idxs]]

                    # Apply new positions
                    vpt_obj_default_state[conflict_env_idxs, conflict_obj_idxs,
                                          0] = conflict_origins[:, 0] + new_x
                    vpt_obj_default_state[conflict_env_idxs, conflict_obj_idxs,
                                          1] = conflict_origins[:, 1] + new_y

                    print(
                        f"  ⚠️ Resolved {num_conflicts} conflicts in ball movement batch."
                    )
        return moved_vpt_for_ball, [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state
        ]

    def vpt_occlusion_movement(self, env_ids, valid_indices,
                               visibility_categories, moved_vpt_for_ball,
                               states, device):
        """
        Move VPT objects to occlude the goal based on visibility categories.
        
        Parameters
        ----------
        env_ids : Sequence[int]
            List of environment IDs.
        valid_indices : Sequence[int]
            Indices of environments that are valid for occlusion processing.
        visibility_categories : List[str]
            List of visibility categories for each environment.
        moved_vpt_for_ball : Dict[int, int | None]
            Dictionary tracking which VPT object was moved for the ball in each environment.
        states : List[torch.Tensor]
            List of state tensors: [goal_state, camera_state, agent_state, vpt_object_state].
        device : torch.device
            Device to perform computations on.
        
        Returns
        -------
        List[torch.Tensor]
            Updated state tensors after occlusion processing.
        """

        if device is None:
            device = self._agent.device

        valid_env_ids = env_ids[valid_indices]

        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        camera_positions = camera_obj_default_state[valid_indices, :3]

        # For occluded cases, place one of the top 3 tallest objects between camera and goal
        for local_idx, env_idx in enumerate(valid_indices):
            if visibility_categories[env_idx] != "occluded":
                continue

            print(
                f"Triggered VPT displacement for env {env_ids[env_idx].item()} due to occlusion requirement"
            )

            env_id = valid_env_ids[local_idx]
            env_id_item = env_id.item()
            camera_pos = camera_positions[local_idx]
            goal_pos = goal_default_state[env_idx, :3]

            active_indices = self.active_vpt_indices[env_id_item]
            moved_idx = moved_vpt_for_ball[env_idx.item()]

            # If env had an object moved for ball, exclude it from occlusion placement
            if moved_idx is not None:
                available_local_indices = [
                    i for i in range(self.active_vpt_objs)
                    if active_indices[i].item() != moved_idx
                ]
                random_local_idx = random.choice(available_local_indices)
                random_obj_idx = active_indices[random_local_idx].item()
            else:
                random_local_idx = random.randint(0, self.active_vpt_objs - 1)
                random_obj_idx = active_indices[random_local_idx].item()

            # Direction vector from camera to goal
            direction_cam_to_goal = goal_pos[:2] - camera_pos[:2]
            distance_cam_to_goal = torch.norm(direction_cam_to_goal)

            # TODO: Add a surface to cam/target distance checker, min dist = 1.5
            if distance_cam_to_goal > 1e-6:
                # Normalize direction
                direction_cam_to_goal = direction_cam_to_goal / distance_cam_to_goal

                # Place object at random point between 30-70% along the line from camera to goal
                t = random.uniform(0.2, 0.8)
                new_pos = camera_pos[:2] + direction_cam_to_goal * (
                    distance_cam_to_goal * t)

                random_offset = sample_uniform(-0.4, 0.4, (2, ), device=device)

                vpt_obj_default_state[env_idx, random_obj_idx,
                                      0] = new_pos[0] + random_offset[0]
                vpt_obj_default_state[env_idx, random_obj_idx,
                                      1] = new_pos[1] + random_offset[1]
                # h = self.all_vpt_dims[env_id, random_obj_idx, 2]
                # ratio = self.vpt_z_offset_ratios[random_obj_idx]
                # vpt_obj_default_state[env_idx, random_obj_idx, 2] = env_origin[2] + (h * ratio)

        return [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state
        ]

    def vpt_in_out_fov_movement(self, env_ids, valid_indices,
                                visibility_categories, in_view_displaced,
                                outside_fov_displaced, moved_vpt_for_ball,
                                states, device=None):

        if device is None:
            device = self._agent.device

        valid_env_ids = env_ids[valid_indices]

        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        camera_positions = camera_obj_default_state[valid_indices, :3]

        for local_idx, env_idx in enumerate(valid_indices):
            if env_idx not in in_view_displaced and env_idx not in outside_fov_displaced:
                continue

            print(
                f"Triggered VPT displacement for env {env_ids[env_idx].item()} due to {visibility_categories[env_idx]} requirement"
            )

            env_id = valid_env_ids[local_idx]
            env_id_item = env_id.item()
            camera_pos = camera_positions[local_idx]
            goal_pos = goal_default_state[env_idx, :3]

            active_indices = self.active_vpt_indices[env_id_item]
            moved_idx = moved_vpt_for_ball[env_idx.item()]
            if moved_idx is not None:
                available_local_indices = [
                    i for i in range(self.active_vpt_objs)
                    if active_indices[i].item() != moved_idx
                ]
                random_local_idx = random.choice(available_local_indices)
                random_obj_idx = active_indices[random_local_idx].item()
            else:
                random_local_idx = random.randint(0, self.active_vpt_objs - 1)
                random_obj_idx = active_indices[random_local_idx].item()

            # Direction vector from camera to goal
            direction_cam_to_goal = goal_pos[:2] - camera_pos[:2]
            distance_cam_to_goal = torch.norm(direction_cam_to_goal)

            if distance_cam_to_goal > 1e-6:
                # Normalize direction
                direction_cam_to_goal = direction_cam_to_goal / distance_cam_to_goal

                # Place object at random point between 10-40% along the line from camera to goal
                t = random.uniform(0.2, 0.8)  # Interpolation factor
                new_pos = camera_pos[:2] + direction_cam_to_goal * (
                    distance_cam_to_goal * t)

                random_offset = sample_uniform(-0.4, 0.4, (2, ), device=device)

                vpt_obj_default_state[env_idx, random_obj_idx,
                                      0] = new_pos[0] + random_offset[0]
                vpt_obj_default_state[env_idx, random_obj_idx,
                                      1] = new_pos[1] + random_offset[1]
                # h = self.all_vpt_dims[env_id, random_obj_idx, 2]
                # ratio = self.vpt_z_offset_ratios[random_obj_idx]
                # vpt_obj_default_state[env_idx, random_obj_idx, 2] = env_origin[2] + (h * ratio)

        return [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state
        ]

    def outside_fov_camera_movement(self, valid_env_ids, valid_indices,
                                    visibility_categories, states, device):

        if device is None:
            device = self._agent.device

        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        camera_positions = camera_obj_default_state[valid_indices, :3]

        # Identify indices relative to the batch (final_valid_indices)
        current_categories = [
            visibility_categories[i] for i in valid_indices.cpu().tolist()
        ]
        outside_fov_mask = torch.tensor(
            [c == "outside_fov" for c in current_categories],
            device=device,
            dtype=torch.bool)

        # 2. Handle "Outside FOV" Logic (Modify Default State Tensor)
        if outside_fov_mask.any():
            # Get subset of indices
            outside_fov_local_idxs = torch.where(outside_fov_mask)[0]
            outside_fov_global_idxs = valid_indices[outside_fov_mask]

            camera_pos_batch = camera_obj_default_state[
                outside_fov_global_idxs, :3]
            goal_pos_batch = goal_default_state[outside_fov_global_idxs, :3]

            # Calculate Look-Away Rotation
            direction_to_goal = goal_pos_batch[:, :2] - camera_pos_batch[:, :2]
            yaw = torch.atan2(direction_to_goal[:, 1],
                              direction_to_goal[:, 0]) - math.radians(90)

            yaw_offset_magnitude = sample_uniform(
                math.radians(60),
                math.pi, (len(outside_fov_global_idxs), ),
                device=device)

            # Generate random signs (-1 or 1) to determine left or right side
            signs = torch.randint(0,
                                  2, (len(outside_fov_global_idxs), ),
                                  device=device).float() * 2 - 1

            # Apply offset to the base yaw
            yaw_away = yaw + (yaw_offset_magnitude * signs)
            roll = torch.full((len(outside_fov_global_idxs), ),
                              -math.radians(self.agent_camera_pitch),
                              device=device)
            zero = torch.zeros_like(roll)
            quaternion_away = quat_from_euler_xyz(roll, zero, yaw_away)

            # Update the Master State Tensor
            camera_obj_default_state[outside_fov_global_idxs,
                                     3:7] = quaternion_away

        # 3. Single Batch Write to Sim
        self._camera_obj.write_root_pose_to_sim(
            camera_obj_default_state[valid_indices, :7], valid_env_ids)

        # 4. Update Occlusion Camera (Sensor)
        camera_positions = camera_obj_default_state[valid_indices, :3]
        camera_orientations = camera_obj_default_state[valid_indices, 3:7]

        # Apply the 90-degree rotation offset for the sensor
        theta_left = math.pi / 2
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0,
             math.sin(half_theta_left)],
            device=device)

        rotated_orientations = math_utils.quat_mul(
            camera_orientations,
            left_90_quat.unsqueeze(0).expand(len(valid_env_ids), -1))

        # Set poses for the sensor
        self._occlusion_camera.set_world_poses(
            positions=camera_positions,
            orientations=rotated_orientations,
            env_ids=valid_env_ids.tolist(),
            convention="world")

        # 5. Step Simulation
        for _ in range(1):
            self.sim.step()
            self._occlusion_camera.update(self.sim.cfg.dt)

    def occlusion_validation_check(self, env_ids, valid_indices,
                             visibility_categories, envs_need_spawn_retry,
                             env_dict, states, device):

        if device is None:
            device = self._agent.device

        valid_env_ids = env_ids[valid_indices]

        goal_default_state = states[0]
        camera_obj_default_state = states[1]
        agent_default_state = states[2]
        vpt_obj_default_state = states[3]

        camera_positions = camera_obj_default_state[valid_indices, :3]

        # 1. First, do occlusion raycast validation for all envs
        occlusion_valid_mask = torch.ones(len(valid_indices),
                                          dtype=torch.bool,
                                          device=device)
        for local_idx, env_idx in enumerate(valid_indices):
            env_id = valid_env_ids[local_idx]
            env_id_item = env_id.item()
            visibility_category = visibility_categories[env_idx]
            camera_pos = camera_positions[local_idx]
            goal_pos = goal_default_state[env_idx, :3]

            if visibility_category in ["in_view", "occluded", "outside_fov"]:
                is_occluded = self._check_occlusion_raycast(
                    camera_pos, goal_pos, env_id)
                expected_occluded = (visibility_category == "occluded"
                                     or visibility_category == "outside_fov")
                occlusion_valid = (is_occluded == expected_occluded)
                occlusion_valid_mask[local_idx] = occlusion_valid
                # TODO: Fix this mess
                if not occlusion_valid:
                    if self.verbose >= 1:
                        print(
                            f"    ❌ Env {env_id_item}: Occlusion raycast validation FAILED"
                        )
                        print(
                            f"       Expected: {'occluded' if expected_occluded else 'visible'}, Got: {'occluded' if is_occluded else 'visible'}"
                        )
                    envs_need_spawn_retry[env_idx] = True
                else:
                    if self.verbose >= 2:
                        print(
                            f"    ✅ Env {env_id_item}: Occlusion raycast validation PASSED"
                        )
                    # TODO: Fix this mess 2
                    if env_id_item not in env_dict:
                        env_dict[env_id_item] = np.round(
                            time.time() - time.time(), 3)

        return occlusion_valid_mask, envs_need_spawn_retry, env_dict, [
            goal_default_state, camera_obj_default_state, agent_default_state,
            vpt_obj_default_state
        ]

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

            MIN_GEOMETRIC_VALID_POINTS = 60
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
                            randomize_objects: bool = True,
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
        safe_x_range = self.center_to_boundary - 4.0
        safe_x_range_obstacles = self.center_to_boundary - 3.5

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
            len(outside_fov_indices))[:len(outside_fov_indices) // 4]
        outside_fov_displaced = torch.tensor(outside_fov_indices,
                                             device=device)[random_indices]

        env_dict = {
            self.slot_to_env_id[env]: {
                "attempts": 0,
                "completion_status": "incomplete",
                "writing_spawn_pose_time": {},
                "moving_ball_time": {},
                "vpt_occlusion_time": {},
                "vpt_in_out_fov_time": {},
                "vpt_object_placement_time": {},
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
                        "vpt_in_out_fov_time", "vpt_object_placement_time",
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

            vpt_occlusion_start_time = time.time()

            states = self.vpt_occlusion_movement(
                env_ids=env_ids,
                valid_indices=valid_indices,
                visibility_categories=visibility_categories,
                moved_vpt_for_ball=moved_vpt_for_ball,
                states=[
                    goal_default_state, camera_obj_default_state,
                    agent_default_state, vpt_obj_default_state
                ],
                device=device,
            )

            goal_default_state = states[0]
            camera_obj_default_state = states[1]
            agent_default_state = states[2]
            vpt_obj_default_state = states[3]

            vpt_occlusion_end_time = time.time()

            # Update dict
            for idx in range(num_envs):
                if env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] == "incomplete":
                    if envs_need_spawn_retry[idx].item() is False:
                        env_dict[self.slot_to_env_id[idx]]["vpt_occlusion_time"][
                            spawn_attempt] = vpt_occlusion_end_time - vpt_occlusion_start_time
                    if envs_need_spawn_retry[idx].item() is True:
                        env_dict[self.slot_to_env_id[idx]][
                            "vpt_occlusion_time"][spawn_attempt] = "F"

            vpt_in_out_fov_start_time = time.time()

            states = self.vpt_in_out_fov_movement(
                env_ids=env_ids,
                valid_indices=valid_indices,
                visibility_categories=visibility_categories,
                in_view_displaced=in_view_displaced,
                outside_fov_displaced=outside_fov_displaced,
                moved_vpt_for_ball=moved_vpt_for_ball,
                states=[
                    goal_default_state, camera_obj_default_state,
                    agent_default_state, vpt_obj_default_state
                ],
            )

            goal_default_state = states[0]
            camera_obj_default_state = states[1]
            agent_default_state = states[2]
            vpt_obj_default_state = states[3]

            self.write_pose_to_sim(env_ids=valid_env_ids,
                                   indices=valid_indices,
                                   vpt_obj_default_state=vpt_obj_default_state)

            vpt_in_out_fov_end_time = time.time()

            # print("AFTER VPT DISPLACEMENT LOOP")
            # print(f"Z of active VPT objects: {self._vpt_objects.data.object_pos_w[valid_env_ids][:,:,2]}")

            # Update dict
            for idx in range(num_envs):
                if env_dict[self.slot_to_env_id[idx]][
                        "completion_status"] == "incomplete":
                    if envs_need_spawn_retry[idx].item() is False:
                        env_dict[
                            self.slot_to_env_id[idx]]["vpt_in_out_fov_time"][
                                spawn_attempt] = vpt_in_out_fov_end_time - vpt_in_out_fov_start_time
                        env_dict[self.slot_to_env_id[idx]][
                            "vpt_object_placement_time"][
                                spawn_attempt] = vpt_in_out_fov_end_time - vpt_occlusion_start_time
                    if envs_need_spawn_retry[idx].item() is True:
                        env_dict[self.slot_to_env_id[idx]][
                            "vpt_in_out_fov_time"][spawn_attempt] = "F"
                        env_dict[self.slot_to_env_id[idx]][
                            "vpt_object_placement_time"][spawn_attempt] = "F"

            # TODO: Check if this works, use new variables
            goal_new_pos = self._goal.data.root_pos_w[valid_env_ids]
            camera_new_pos = self._camera_obj.data.root_pos_w[valid_env_ids]
            agent_new_pos = self._agent.data.root_pos_w[valid_env_ids]
            vpt_new_pos = self._vpt_objects.data.object_pos_w[valid_env_ids]

            for local_idx, env_idx in enumerate(valid_indices):
                z_valid = ((goal_new_pos[local_idx, 2] >= 0.0)
                           and (goal_new_pos[local_idx, 2] <= 1.0)
                           and (camera_new_pos[local_idx, 2] >= 0.0)
                           and (camera_new_pos[local_idx, 2] <= 1.0)
                           and (agent_new_pos[local_idx, 2] >= 0.0)
                           and (agent_new_pos[local_idx, 2] <= 1.0) and
                           torch.all((vpt_new_pos[local_idx, :, 2] >= 0.0)
                                     & (vpt_new_pos[local_idx, :, 2] <= 0.1)).item())

                if not z_valid:
                    # Print z of all active vpt objects
                    print(f"Z of VPT objects: {vpt_new_pos[local_idx, :, 2]}")
                    if (goal_new_pos[local_idx, 2] < 0.0) or (goal_new_pos[local_idx, 2] > 1.0):
                        print(
                            f"    ❌ Env {env_ids[env_idx].item()}: Goal Z position out of bounds: {goal_new_pos[local_idx, 2].item()}"
                        )
                    if (camera_new_pos[local_idx, 2] < 0.0) or (camera_new_pos[local_idx, 2] > 1.0):
                        print(
                            f"    ❌ Env {env_ids[env_idx].item()}: Camera Z position out of bounds: {camera_new_pos[local_idx, 2].item()}"
                        )
                    if (agent_new_pos[local_idx, 2] < 0.0) or (agent_new_pos[local_idx, 2] > 1.0):
                        print(
                            f"    ❌ Env {env_ids[env_idx].item()}: Agent Z position out of bounds: {agent_new_pos[local_idx, 2].item()}"
                        )
                    if torch.any((vpt_new_pos[local_idx, :, 2] < 0.0) | (vpt_new_pos[local_idx, :, 2] > 0.1)):
                        # Get the specific values that are bad for better debugging
                        bad_indices = torch.where((vpt_new_pos[local_idx, :, 2] < 0.0) | (vpt_new_pos[local_idx, :, 2] > 0.1))[0]
                        bad_values = vpt_new_pos[local_idx, bad_indices, 2].cpu().numpy()
                        print(
                            f"    ❌ Env {env_ids[env_idx].item()}: VPT Object Z positions out of bounds at indices {bad_indices.cpu().numpy()}: {bad_values}"
                        )
                    envs_need_spawn_retry[env_idx] = True

            # TODO: Goal VPT Surface Distance Check, dist >= 0.01

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
                env_ids=final_valid_env_ids,
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
        self._agent.write_root_pose_to_sim(agent_default_state[:, :7], env_ids)

        circle_validation_start_time = time.time()
        all_valid_points = self.generate_valid_circle_points(env_ids=env_ids,
                                                             angle_step=2.0,
                                                             max_attempts=100)
        circle_validation_end_time = time.time()

        # Update dict
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

        if self.valid_viewpoint_poses is None:
            self.valid_viewpoint_poses = [None] * self.num_envs

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            valid_points_2d = all_valid_points[i]

            if valid_points_2d.shape[0] > 0:
                agent_z = self._agent.data.default_root_state[env_id, 2]
                valid_points_3d = torch.zeros((valid_points_2d.shape[0], 3),
                                              device=device)
                valid_points_3d[:, :2] = valid_points_2d
                valid_points_3d[:, 2] = agent_z
                self.valid_viewpoint_poses[
                    env_id_item] = valid_points_3d if valid_points_3d.shape[
                        0] >= self.images_per_env else torch.zeros(
                            (0, 3), device=device)
            else:
                self.valid_viewpoint_poses[env_id_item] = torch.zeros(
                    (0, 3), device=device)

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
                    "vpt_in_out_fov_time", "vpt_object_placement_time",
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
                             ["vpt_object_placement_time"].values())[-1] +
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
                              check_agent_fov: bool = False,
                              min_required_points: int = None) -> torch.Tensor:
        """Batch validate multiple points across multiple environments in parallel (Fully Vectorized)."""

        device = points.device
        num_points = points.shape[0]

        if min_required_points is None:
            min_required_points = self.images_per_env

        # Start with all true
        valid_mask = torch.ones(num_points, dtype=torch.bool, device=device)

        # --- 1. Boundary Check (Fast) ---
        # Use advanced indexing to get origins for all points at once
        env_origins = self.scene.env_origins[env_ids, :2]
        boundary_limit = self.center_to_boundary

        min_bounds = env_origins - boundary_limit
        max_bounds = env_origins + boundary_limit

        in_bounds = torch.all((points >= min_bounds) & (points <= max_bounds),
                              dim=1)
        valid_mask &= in_bounds

        if not valid_mask.any():
            return valid_mask

        # --- 2. VPT Distance Check (Vectorized) ---

        # Retrieve ALL object positions: [num_points, total_objs, 3]
        # We expand env_ids to match the batch dimension
        all_vpt_pos = self._vpt_objects.data.object_pos_w[env_ids, :, :2]

        # FIX: Handle List[Tensor] using torch.stack
        if isinstance(self.active_vpt_indices, list):
            if len(self.active_vpt_indices) > 0 and isinstance(
                    self.active_vpt_indices[0], torch.Tensor):
                # Case A: It's a list of Tensors -> Use torch.stack
                active_indices_tensor = torch.stack(
                    self.active_vpt_indices).to(dtype=torch.long,
                                                device=device)
            else:
                # Case B: It's a list of Ints/Lists -> Use torch.tensor
                active_indices_tensor = torch.tensor(self.active_vpt_indices,
                                                     device=device,
                                                     dtype=torch.long)
        else:
            # Case C: It's already a Tensor
            active_indices_tensor = self.active_vpt_indices

        # Now you can use tensor-based advanced indexing
        # active_indices_tensor shape: [num_envs, active_objs]
        # env_ids shape: [num_points]
        # Result shape: [num_points, active_objs]
        batch_active_indices = active_indices_tensor[env_ids]

        # Gather the active positions.
        # dim=1 tells gather to pick along the object dimension
        # We need to expand indices to match the coordinate dim (x,y) -> [num_points, active_objs, 2]
        batch_active_indices_expanded = batch_active_indices.unsqueeze(
            -1).expand(-1, -1, 2)
        active_vpt_positions = torch.gather(all_vpt_pos, 1,
                                            batch_active_indices_expanded)

        # Calculate Distances: |point - vpt_obj|
        # points: [num_points, 2] -> [num_points, 1, 2]
        # active_vpt_positions: [num_points, active_objs, 2]
        distances_to_vpt = torch.norm(points.unsqueeze(1) -
                                      active_vpt_positions,
                                      dim=2)

        # Check minimum distance across all active objects for each point
        min_vpt_distances = distances_to_vpt.min(dim=1)[0]
        valid_mask &= (min_vpt_distances >= min_obstacle_distance)

        if not valid_mask.any():
            return valid_mask

        # --- 3. Camera Checks (Vectorized) ---

        camera_positions = self._camera_obj.data.root_pos_w[env_ids, :2]

        # Camera -> Point Distance
        camera_point_distances = torch.norm(points - camera_positions, dim=1)
        valid_mask &= (camera_point_distances >= min_camera_target_distance)

        if not valid_mask.any():
            return valid_mask

        # Camera -> VPT Distance
        # We reuse 'active_vpt_positions' from Step 2!
        # camera_positions: [num_points, 2] -> [num_points, 1, 2]
        camera_vpt_distances = torch.norm(camera_positions.unsqueeze(1) -
                                          active_vpt_positions,
                                          dim=2)
        min_cam_vpt_dist = camera_vpt_distances.min(dim=1)[0]

        valid_mask &= (min_cam_vpt_dist >= min_camera_obstacle_distance)

        if not valid_mask.any():
            return valid_mask

        # --- 4. Agent FOV Check (Optional) ---
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
                sample_indices = torch.randperm(
                    num_points_for_env, device=device)[:MAX_SAMPLES_PER_ENV]
                env_to_points[env_id.item()] = env_points[sample_indices]
                env_to_indices[
                    env_id.item()] = env_points_indices[sample_indices]
                if self.verbose >= 2:
                    print(
                        f"  🎲 Env {env_id.item()}: Sampled {MAX_SAMPLES_PER_ENV}/{num_points_for_env} points for FOV validation"
                    )
            else:
                env_to_points[env_id.item()] = env_points
                env_to_indices[env_id.item()] = env_points_indices
                if self.verbose >= 2:
                    print(
                        f"  🎲 Env {env_id.item()}: Using all {num_points_for_env} points for FOV validation"
                    )

            env_completed[env_id.item()] = False
            env_valid_counts[env_id.item()] = 0

        # Find maximum number of points any environment has (after sampling)
        max_points_per_env = max(len(pts) for pts in env_to_points.values())

        fov_valid = torch.zeros(num_points, dtype=torch.bool, device=device)

        # Process one point index at a time across all environments
        for point_idx in range(max_points_per_env):
            if self.verbose >= 2 and point_idx % 10 == 0:
                print(
                    f"    🔄 FOV validation progress: {point_idx+1}/{max_points_per_env} points"
                )

            # Collect environments and points for this iteration (skip completed envs)
            batch_env_ids = []
            batch_points = []
            batch_global_indices = []

            for env_id in unique_env_ids:
                env_id_item = env_id.item()

                # Skip this env if it already has enough valid points
                if min_required_points is not None and env_completed[
                        env_id_item]:
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
                if min_required_points is not None and all(
                        env_completed.values()):
                    if self.verbose >= 1:
                        print(
                            f"    ✅ All environments have {min_required_points}+ valid points, stopping FOV check early"
                        )
                    break
                continue

            batch_env_ids = torch.stack(batch_env_ids)
            batch_points = torch.stack(batch_points)
            batch_global_indices = torch.stack(batch_global_indices)

            # Teleport all agents to their respective points
            temp_agent_pos = torch.zeros((len(batch_env_ids), 3),
                                         device=device)
            temp_agent_pos[:, :2] = batch_points
            temp_agent_pos[:, 2] = self._agent.data.default_root_state[
                batch_env_ids, 2]

            camera_pos_3d = self._camera_obj.data.root_pos_w[batch_env_ids]
            goal_pos_3d = self._goal.data.root_pos_w[batch_env_ids]
            midpoints_3d = (camera_pos_3d[:, :2] + goal_pos_3d[:, :2]) / 2.0

            directions = midpoints_3d - batch_points
            yaws = torch.atan2(directions[:, 1], directions[:, 0])

            temp_agent_quat = torch.zeros((len(batch_env_ids), 4),
                                          device=device)
            temp_agent_quat[:, 0] = torch.cos(yaws / 2)
            temp_agent_quat[:, 3] = torch.sin(yaws / 2)

            temp_poses = torch.cat([temp_agent_pos, temp_agent_quat], dim=1)
            self._agent.write_root_com_pose_to_sim(temp_poses, batch_env_ids)

            # Update simulation
            for _ in range(1):
                self.sim.step()
                self._tiled_camera.update(self.sim.cfg.dt)
                self._agent.update(self.sim.cfg.dt)
                self._camera_obj.update(self.sim.cfg.dt)
                self._goal.update(self.sim.cfg.dt)
                self._vpt_objects.update(self.sim.cfg.dt)

            # Check visibility for all agents in this batch in parallel
            batch_goal_visible, batch_camera_visible = self.check_batch_object_visibility(
                batch_env_ids)
            batch_fov_valid = batch_goal_visible & batch_camera_visible

            # Assign results back to global fov_valid tensor
            fov_valid[batch_global_indices] = batch_fov_valid

            # Update valid counts and check for completion
            if min_required_points is not None:
                for i, env_id in enumerate(batch_env_ids):
                    env_id_item = env_id.item()
                    if batch_fov_valid[i]:
                        env_valid_counts[env_id_item] += 1
                        if env_valid_counts[
                                env_id_item] >= min_required_points and not env_completed[
                                    env_id_item]:
                            env_completed[env_id_item] = True
                            if self.verbose >= 1:
                                print(
                                    f"    🎯 Env {env_id_item}: Reached {min_required_points} valid points, stopping FOV checks for this env"
                                )

        # Restore original agent positions
        restore_poses = torch.cat([current_agent_pos, current_agent_quat],
                                  dim=1)
        self._agent.write_root_com_pose_to_sim(restore_poses, env_ids)

        valid_mask &= fov_valid

        return valid_mask

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
        Refined Randomization (No Bounding Box for Position):
        1. Identifies object type.
        2. Calculates Scale and Z-Position analytically based on known base dims.
        3. Updates Physics/Visuals.
        """
        world = World.instance()
        if world.is_playing():
            world.pause()

        stage = get_current_stage()
        # We still need bbox_cache if you want to store the ACCURATE final visual bounds
        # into self.all_vpt_dims, but we won't use it for placement.
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                       [UsdGeom.Tokens.default_])

        if not hasattr(
                self,
                "all_vpt_dims") or self.all_vpt_dims.shape[0] != self.num_envs:
            self.all_vpt_dims = torch.zeros(
                (self.num_envs, self.cfg.num_vpt_objs, 3), device=self.device)
            self.vpt_obj_default_state = torch.zeros(
                (self.num_envs, self.num_objs, 3), device=self.device)

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

                # 1. Parse Indices
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
                    base_dim = self.vpt_base_dims[obj_idx].cpu().numpy(
                    )  # [x, y, z]
                    final_scale_vec = Gf.Vec3d(1, 1, 1)
                    final_z_pos = 0.0

                    # --- LOGIC BRANCHING ---

                    if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                        filename = spawn_cfg.usd_path.split("/")[-1].split(
                            ".")[0]

                        if filename.endswith(('X', 'L', 'T')):
                            if filename.endswith(("X")):
                                s_xz = random.uniform(0.75, 2.5)
                                s_y = random.uniform(0.75, 2.5)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_xz, base_dim[1] * s_y,
                                    base_dim[2] * s_xz)
                                z_offset_multiplier = 0.0
                                current_z_scale = final_scale_vec[2]
                                final_z_pos = current_z_scale * z_offset_multiplier
                            if filename.endswith(("L")):
                                s_factor = random.uniform(0.75, 2.5)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_factor,
                                    base_dim[1] * s_factor,
                                    base_dim[2] * s_factor)
                                z_offset_multiplier = 0.0
                                current_z_scale = final_scale_vec[2]
                                final_z_pos = current_z_scale * z_offset_multiplier
                        else:
                            # Generic USD: Center Pivoted
                            s_xy = random.uniform(0.75, 2.5)
                            s_z = random.uniform(0.75, 2.5)
                            final_scale_vec = Gf.Vec3d(base_dim[0] * s_xy,
                                                       base_dim[1] * s_xy,
                                                       base_dim[2] * s_z)
                            z_offset_multiplier = 0.5
                            final_z_pos = final_scale_vec[
                                2] * z_offset_multiplier

                    elif isinstance(spawn_cfg, sim_utils.CuboidCfg):
                        # Primitives: Center Pivoted
                        s_x = random.uniform(0.75, 2.5)
                        s_y = random.uniform(0.75, 2.5)
                        s_z = random.uniform(0.75, 2.5)
                        final_scale_vec = Gf.Vec3d(s_x, s_y, s_z)

                        z_offset_multiplier = 0.5
                        total_height = base_dim[2] * s_z
                        final_z_pos = total_height * z_offset_multiplier

                    elif isinstance(
                            spawn_cfg,
                        (sim_utils.CylinderCfg, sim_utils.ConeCfg)):
                        # Primitives: Center Pivoted
                        s_r = random.uniform(0.75, 1.0)
                        s_h = random.uniform(0.75, 2.5)
                        final_scale_vec = Gf.Vec3d(s_r, s_r, s_h)

                        z_offset_multiplier = 0.5
                        total_height = base_dim[2] * s_h
                        final_z_pos = total_height * z_offset_multiplier

                    # -----------------------

                    # Apply Scale
                    scale_op.Set(final_scale_vec)

                    # Apply Translation
                    current_trans = translate_op.Get()
                    translate_op.Set(Gf.Vec3d(current_trans[0], current_trans[1], final_z_pos))

                    self.vpt_obj_default_state[env_idx, obj_idx, 2] = final_z_pos

                    # --- IMPROVED UPDATE ---
                    # Don't re-compute BBox from USD (Slow & potentially misses root scale).
                    # Just multiply base dims by the scale you just calculated.
                    
                    # final_scale_vec is Gf.Vec3d, access elements with []
                    new_w = base_dim[0] * final_scale_vec[0]
                    new_l = base_dim[1] * final_scale_vec[1]
                    new_h = base_dim[2] * final_scale_vec[2]

                    self.all_vpt_dims[env_idx, obj_idx, 0] = new_w
                    self.all_vpt_dims[env_idx, obj_idx, 1] = new_l
                    self.all_vpt_dims[env_idx, obj_idx, 2] = new_h

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

    def randomize_global_light(self, prim_path="/World/Light"):
        """
        Randomizes intensity and rotates the sun position.
        
        NOTE: We do NOT change 'inputs:angle' to 45-70, because that controls 
        shadow softness. We change the X-Rotation (elevation) instead.
        """
        stage = get_current_stage()  # Assuming you have a wrapper for this
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            print(f"⚠️ Warning: Light prim not found at {prim_path}")
            return

        # --- 1. INTENSITY ---
        # Randomize brightness linearly
        # Default is 1.0, but suns usually need 1000+ in USD to look bright
        if prim_path == "/World/Light_A":
            rand_intensity = random.uniform(500.0, 2000.0)
        else:
            rand_intensity = random.uniform(1000.0, 3000.0)
        prim.GetAttribute("inputs:intensity").Set(rand_intensity)

        # --- 2. ROTATION (The "Angle" you want) ---
        # You want the sun to be between 45 and 70 degrees elevation.
        # In USD/Isaac Sim, we rotate the prim to move the sun.

        # Elevation (Height in sky): 45 to 70 degrees
        # We use negative X because the light defaults to pointing down -Z
        rand_elevation = random.uniform(45, 70)

        # Azimuth (Compass direction): 0 to 360 degrees
        # This ensures the shadows fall in different directions
        rand_azimuth = random.uniform(0, 360)

        # Apply the rotation using UsdGeom Xformable
        xform = UsdGeom.Xformable(prim)

        # helper to find or create the rotation operation
        rotate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op = op
                break

        if rotate_op is None:
            rotate_op = xform.AddRotateXYZOp()

        # Set the rotation:
        # X = -elevation (tilt up/down)
        # Y = 0
        # Z = azimuth (spin around)
        rotate_op.Set(Gf.Vec3d(-rand_elevation, 0, rand_azimuth))

        # --- 3. SHADOW SOFTNESS (Optional) ---
        # This is the "angle" from the docs.
        # Keep this small (approx 0.5 to 2.0) for distinct shadows.
        # If this is 70, shadows will disappear.
        prim.GetAttribute("inputs:angle").Set(0.7)

    def randomize_sphere_lights(self,
                                light_names: list = ["Light_A"],
                                z_heights: list = [None]):
        """
        Randomizes position, z-height, intensity, and color.
        Per environment, it randomly selects a subset of lights to be active 
        (from 0 to len(light_names)) and turns the rest off.
        """
        stage = get_current_stage()

        # Safety margin (e.g., 100% of the full width as per your code)
        limit = self.center_to_boundary * 0.9

        with Sdf.ChangeBlock():  # Batch updates for performance
            for i in range(self.num_envs):

                # 1. Determine how many and which lights are ON for this specific env
                # range is inclusive for start, exclusive for stop?
                # random.sample needs a count.
                # We want a random number between 1 and len(light_names) inclusive.
                num_active = random.randint(1, len(light_names))
                active_light_names = set(random.sample(light_names,
                                                       num_active))

                for light_name, z_req in zip(light_names, z_heights):
                    # Construct the path
                    prim_path = f"/World/envs/env_{i}/{light_name}"
                    prim = stage.GetPrimAtPath(prim_path)

                    if not prim.IsValid():
                        continue

                    # Check if this light was selected to be ON
                    if light_name in active_light_names:
                        # --- LIGHT IS ON ---

                        # 2. Determine Z Height (User Range)
                        if z_req is None:
                            final_z = random.uniform(5.0, 10.0)
                        else:
                            final_z = float(z_req)

                        # 3. Generate Random Local Positions
                        rand_x = random.uniform(-limit, limit)
                        rand_y = random.uniform(-limit, limit)

                        # 4. Apply Translation
                        xform = UsdGeom.Xformable(prim)
                        translate_op = None
                        for op in xform.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                translate_op = op
                                break

                        if translate_op is None:
                            translate_op = xform.AddTranslateOp()

                        translate_op.Set(
                            Gf.Vec3d(float(rand_x), float(rand_y), final_z))

                        # 5. Randomize Intensity (User Range)
                        rand_intensity = random.uniform(250_000.0, 750_000.0)
                        prim.GetAttribute("inputs:intensity").Set(
                            rand_intensity)

                        # 6. Randomize Color (User Range)
                        r = random.uniform(0.70, 1.0)
                        g = random.uniform(0.70, 1.0)
                        b = random.uniform(0.70, 0.95)

                        # Ensure G and B don't exceed R significantly to maintain warmth
                        g = min(g, r)
                        b = min(b, g)

                        prim.GetAttribute("inputs:color").Set(Gf.Vec3d(
                            r, g, b))

                    else:
                        # --- LIGHT IS OFF ---
                        # Set intensity to 0 to effectively disable it
                        prim.GetAttribute("inputs:intensity").Set(0.0)

    def get_color(self):
        """
        Generate a random pastel color that is not similar to red, green, blue, or pink.
        Returns a sim_utils.PreviewSurfaceCfg with the chosen color.
        """
        # Define forbidden colors and a threshold for "similarity"
        forbidden_colors = [
            np.array([1.0, 0.0, 0.0]),  # red
            np.array([0.0, 1.0, 0.0]),  # green
            np.array([0.0, 0.0, 1.0]),  # blue
            np.array([0.8, 0.0, 0.0]),  # pinkish-red
            np.array([1.0, 0.75, 0.8]),  # pink
            np.array([1.0, 0.2, 0.6]),  # pink
            np.array([0.9, 0.1, 0.5]),  # pink
            np.array([0.95, 0.3, 0.6]),  # pink
            np.array([0.0, 0.0, 0.0])  # black
        ]
        threshold = 0.2  # Euclidean distance threshold for "similarity"

        valid = False
        while not valid:
            # Generate a pastel color by mixing with white, but limit the mix to avoid being too white
            base = np.array([random.uniform(0.0, 0.9) for _ in range(3)])
            # Check similarity to forbidden colors
            too_close = any(
                np.linalg.norm(base - fc) < threshold
                for fc in forbidden_colors)
            # Avoid colors that are too close to pure red/green/blue/pink
            if not too_close:
                valid = True
        return base[0], base[1], base[2]

    def check_usd_pivot_alignment(self, prim_path_expr: str):
        """
        Checks if a USD object is Base-Pivoted or Center-Pivoted.
        """
        stage = get_current_stage()
        # Find the first matching prim to test
        if isinstance(prim_path_expr, list):
            prim_path_expr = prim_path_expr[0]

        found_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
        if not found_paths:
            print(f"❌ No prims found for {prim_path_expr}")
            return

        prim_path = found_paths[0]
        prim = stage.GetPrimAtPath(prim_path)

        # Compute Bounding Box
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                       [UsdGeom.Tokens.default_])
        # bound is Gf.Range3d
        bound = bbox_cache.ComputeLocalBound(prim).GetRange()

        z_min = bound.GetMin()[2]
        z_max = bound.GetMax()[2]
        height = z_max - z_min
        center_z = (z_min + z_max) / 2.0

        print(f"--- PIVOT CHECK for {prim_path} ---")
        print(f"📉 Z Min (Bottom): {z_min:.4f}")
        print(f"📈 Z Max (Top):    {z_max:.4f}")
        print(f"📏 Total Height:   {height:.4f}")

        # Diagnosis Logic
        # If Z Min is near 0.0, the Origin is at the Bottom (Base-Pivoted)
        if abs(z_min) < 0.05 * height:
            print(f"👉 DIAGNOSIS: [BASE PIVOTED]")
            print(f"   The origin (0,0,0) is at the FEET of the object.")
            print(
                f"   Placement Logic: Z = Floor_Z + 0.0 (Do NOT add half-height)"
            )

        # If Z Min is roughly -Height/2, the Origin is at the Center (Center-Pivoted)
        elif abs(z_min + (height / 2)) < 0.05 * height:
            print(f"👉 DIAGNOSIS: [CENTER PIVOTED]")
            print(f"   The origin (0,0,0) is at the BELLY of the object.")
            print(f"   Placement Logic: Z = Floor_Z + (Height / 2)")

        else:
            print(f"👉 DIAGNOSIS: [WEIRD OFFSET]")
            print(f"   The origin is neither at the bottom nor center.")
            print(f"   Offset needed: {-z_min:.4f}")

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
    
    def _get_dists_to_active_vpt_surface(
            self,
            env_ids,
            reference_points,
            device=None
        ):
        """
        Calculates distance from a global reference point to the surface of all ACTIVE VPT objects.
        """
        if device is None:
            device = self._agent.device
        
        if isinstance(reference_points, list):
            reference_points = torch.stack(reference_points, dim=0).to(device)
        
        vpt_obj_states = self._get_active_vpt_positions(env_ids)
        
        vpt_obj_dims = self._get_active_vpt_dims(env_ids)
        
        vpt_obj_pos = vpt_obj_states[:3]
        rel_pos = reference_points.unsqueeze(1) - vpt_obj_pos
        
        vpt_obj_quat = vpt_obj_states[3:7]
        
        quat_conjugate = vpt_obj_quat.clone()
        quat_conjugate[:, :3] *= -1
        
        xyz = quat_conjugate[:, :3]
        w = quat_conjugate[:, 3].unsqueeze(-1)
        
        t = 2 * torch.cross(xyz, rel_pos, dim=-1)
        
        point_local = rel_pos * w * t + torch.cross(xyz, t, dim=-1)
        
        half_dims = vpt_obj_dims / 2.0
        
        d = torch.abs(point_local) - half_dims.unsqueeze(0)
        
        dists = torch.norm(torch.clamp(d, min=0.0), dim=-1)
        
        return dists
    
    def randomize_mat_material(self, prim_paths: list):
        for prim in prim_paths:
            rand_material = random.choice(self.mat_material_paths)
            sim_utils.bind_visual_material(prim, rand_material)
    
    def randomize_vpt_material(self, prim_paths: list):
        for prim in prim_paths:
            rand_material = random.choice(self.vpt_material_paths)
            sim_utils.bind_visual_material(prim, rand_material)
    
    def get_all_mdls_from_nucleus_path(self, path: str) -> list[str]:
        """
        Recursively finds all .mdl files in a specific Nucleus directory 
        using the omni.client API.
        """
        mdl_files = []
        
        # List contents of the directory
        result, entries = omni.client.list(path)
        
        # If the path doesn't exist or isn't accessible, return empty list
        if result != omni.client.Result.OK:
            print(f"[Warning] Could not access path: {path} (Error: {result})")
            return []

        for entry in entries:
            # Nucleus always uses forward slashes
            full_path = f"{path}/{entry.relative_path}"
            
            if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                # It's a directory, recurse deeper
                mdl_files.extend(self.get_all_mdls_from_nucleus_path(full_path))
            else:
                # It's a file, check extension
                if full_path.endswith(".mdl"):
                    mdl_files.append(full_path)
                    
        return mdl_files

    def get_mat_material_paths(self) -> list[str]:
        """
        Returns a curated list of standard NVIDIA MDL material paths 
        for the Visual Perspective Taking (VPT) task.
        """
        # base = f"{NVIDIA_NUCLEUS_DIR}/Materials/2023_1/vMaterials_2/"
        base = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base"

        material_paths = [
            f"{base}/Architecture/Ceiling_Tiles.mdl",
            f"{base}/Architecture/Roof_Tiles.mdl",
            f"{base}/Architecture/Shingles_01.mdl",
            f"{base}/Carpet/Carpet_Diamond_Olive.mdl",
            f"{base}/Carpet/Carpet_Diamond_Yellow.mdl",
            f"{base}/Carpet/Carpet_Pattern_Leaf_Squares_Tan.mdl",
            f"{base}/Carpet/Carpet_Pattern_Squares_Multi.mdl",
            f"{base}/Carpet/Carpet_Forest.mdl",
            f"{base}/Carpet/Carpet_Cream.mdl",
            f"{base}/Carpet/Carpet_Beige.mdl",
            f"{base}/Carpet/Carpet_Berber_Multi.mdl",
            f"{base}/Masonry/Adobe_Brick.mdl",
            f"{base}/Masonry/Brick_Pavers.mdl",
            f"{base}/Masonry/Brick_Wall_Brown.mdl",
            f"{base}/Masonry/Brick_Wall_Red.mdl",
            f"{base}/Masonry/Concrete_Block.mdl",
            f"{base}/Masonry/Concrete_Rough.mdl",
            f"{base}/Masonry/Concrete_Formed.mdl",
            f"{base}/Metals/CorrugatedMetal.mdl",
            f"{base}/Metals/Brushed_Antique_Copper.mdl",
            f"{base}/Natural/Asphalt.mdl",
            f"{base}/Natural/Dirt.mdl",
            f"{base}/Natural/Grass_Cut.mdl",
            f"{base}/Natural/Grass_Countryside.mdl",
            f"{base}/Natural/Grass_Winter.mdl",
            f"{base}/Natural/Mulch_Brown.mdl",
            f"{base}/Natural/Soil_Rocky.mdl",
            f"{base}/Natural/Sand.mdl",
            f"{base}/Natural/Leaves.mdl",
            f"{base}/Plastics/Veneer_OU_Walnut.mdl",
            f"{base}/Plastics/Veneer_UX_Walnut_Cherry.mdl",
            f"{base}/Stone/Adobe_Octagon_Dots.mdl",
            f"{base}/Stone/Fieldstone.mdl",
            f"{base}/Stone/Pea_Gravel.mdl",
            f"{base}/Stone/Gravel_River_Rock.mdl",
            f"{base}/Stone/Gravel.mdl",
            f"{base}/Stone/Stone_Wall.mdl",
            f"{base}/Stone/Retaining_Block.mdl",
            f"{base}/Stone/Terracotta.mdl",
            f"{base}/Wood/Ash.mdl",
            f"{base}/Wood/Ash_Planks.mdl",
            f"{base}/Wood/Bamboo.mdl",
            f"{base}/Wood/Bamboo_Planks.mdl",
            f"{base}/Wood/Birch.mdl",
            f"{base}/Wood/Oak.mdl",
            f"{base}/Wood/Oak_Planks.mdl",
            f"{base}/Wood/Cherry.mdl",
            f"{base}/Wood/Cherry_Planks.mdl",
            f"{base}/Wood/Mahogany.mdl",
            f"{base}/Wood/Parquet_Floor.mdl",
            f"{base}/Wood/Plywood.mdl",
            f"{base}/Wood/Timber.mdl",
            f"{base}/Wood/Walnut.mdl",
            f"{base}/Wood/Walnut_Planks.mdl",
        ]
        
        return material_paths
    
    def get_vpt_material_paths(self) -> list[str]:
        """
        Returns a curated list of standard NVIDIA MDL material paths 
        for the Visual Perspective Taking (VPT) task.
        """
        # base = f"{NVIDIA_NUCLEUS_DIR}/Materials/2023_1/vMaterials_2/"
        base = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base"

        material_paths = [
            f"{base}/Architecture/Ceiling_Tiles.mdl",
            f"{base}/Architecture/Roof_Tiles.mdl",
            f"{base}/Carpet/Carpet_Diamond_Olive.mdl",
            f"{base}/Carpet/Carpet_Diamond_Yellow.mdl",
            f"{base}/Carpet/Carpet_Pattern_Leaf_Squares_Tan.mdl",
            f"{base}/Carpet/Carpet_Pattern_Squares_Multi.mdl",
            f"{base}/Carpet/Carpet_Forest.mdl",
            f"{base}/Carpet/Carpet_Cream.mdl",
            f"{base}/Carpet/Carpet_Beige.mdl",
            f"{base}/Carpet/Carpet_Berber_Multi.mdl",
            f"{base}/Masonry/Adobe_Brick.mdl",
            f"{base}/Masonry/Brick_Pavers.mdl",
            f"{base}/Masonry/Brick_Wall_Brown.mdl",
            f"{base}/Masonry/Brick_Wall_Red.mdl",
            f"{base}/Masonry/Concrete_Block.mdl",
            f"{base}/Masonry/Concrete_Rough.mdl",
            f"{base}/Masonry/Concrete_Formed.mdl",
            f"{base}/Metals/CorrugatedMetal.mdl",
            f"{base}/Metals/Brushed_Antique_Copper.mdl",
            f"{base}/Natural/Asphalt.mdl",
            f"{base}/Natural/Dirt.mdl",
            f"{base}/Natural/Grass_Cut.mdl",
            f"{base}/Natural/Grass_Countryside.mdl",
            f"{base}/Natural/Grass_Winter.mdl",
            f"{base}/Natural/Mulch_Brown.mdl",
            f"{base}/Natural/Soil_Rocky.mdl",
            f"{base}/Natural/Sand.mdl",
            f"{base}/Natural/Leaves.mdl",
            f"{base}/Plastics/Veneer_OU_Walnut.mdl",
            f"{base}/Plastics/Veneer_UX_Walnut_Cherry.mdl",
            f"{base}/Stone/Adobe_Octagon_Dots.mdl",
            f"{base}/Stone/Pea_Gravel.mdl",
            f"{base}/Stone/Gravel_River_Rock.mdl",
            f"{base}/Stone/Terracotta.mdl",
            f"{base}/Wood/Ash_Planks.mdl",
            f"{base}/Wood/Bamboo.mdl",
            f"{base}/Wood/Bamboo_Planks.mdl",
            f"{base}/Wood/Oak_Planks.mdl",
            f"{base}/Wood/Cherry_Planks.mdl",
            f"{base}/Wood/Mahogany.mdl",
            f"{base}/Wood/Parquet_Floor.mdl",
            f"{base}/Wood/Plywood.mdl",
            f"{base}/Wood/Walnut_Planks.mdl",
        ]
        
        return material_paths

    def get_mat_material_configs(self) -> list[sim_utils.MdlFileCfg]:
        """
        Retrieves material paths, selects a subset, and wraps them in MdlFileCfg objects.
        Returns: A list of configured MdlFileCfg objects.
        """
        # 1. Get the raw paths
        mat_materials_paths = self.get_mat_material_paths()

        # 2. Limit to 50 items
        num_mat_materials = min(len(mat_materials_paths), 100)
        mat_materials_paths = random.sample(mat_materials_paths, num_mat_materials)

        print(f"[INFO] Final count: {len(mat_materials_paths)} materials selected.")
        print(mat_materials_paths)

        # 3. Wrap in MdlFileCfg
        mat_mdl_configs_list = []
        for mat_path in mat_materials_paths:
            material = sim_utils.MdlFileCfg(
                mdl_path=mat_path,
                project_uvw=True,              
                texture_scale=(1000.0, 1000.0),   
            )
            mat_mdl_configs_list.append(material)
            
        return mat_mdl_configs_list
    
    def get_vpt_material_configs(self) -> list[sim_utils.MdlFileCfg]:
        """
        Retrieves material paths, selects a subset, and wraps them in MdlFileCfg objects.
        Returns: A list of configured MdlFileCfg objects.
        """
        # 1. Get the raw paths
        vpt_materials_paths = self.get_vpt_material_paths()

        # 2. Limit to 50 items
        num_vpt_materials = min(len(vpt_materials_paths), 100)
        vpt_materials_paths = random.sample(vpt_materials_paths, num_vpt_materials)

        print(f"[INFO] Final count: {len(vpt_materials_paths)} materials selected.")
        print(vpt_materials_paths)

        # 3. Wrap in MdlFileCfg
        vpt_mdl_configs_list = []
        for vpt_path in vpt_materials_paths:
            material = sim_utils.MdlFileCfg(
                mdl_path=vpt_path,
                project_uvw=True,              
                texture_scale=(2.0, 2.0),   
            )
            vpt_mdl_configs_list.append(material)
            
        return vpt_mdl_configs_list
