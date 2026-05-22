from __future__ import annotations

import math
import heapq
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
        # Right-half sweep: 0°, 15°, 30°, ..., 165° relative to agent heading.
        self.camera_sweep_angles_deg = list(range(0, 181, 15))  # 13 angles: 0..180
        self.camera_move_target_frames_per_env = int(os.getenv(
            "CAMERA_MOVE_TARGET_FRAMES_PER_ENV", "12"))
        self.images_per_env = self.camera_move_target_frames_per_env
        self.camera_move_balance_saved_frames = os.getenv(
            "CAMERA_MOVE_BALANCE_SAVED_FRAMES", "1").lower() not in {"0", "false", "no"}
        self.camera_move_min_pairs_per_env = int(os.getenv(
            "CAMERA_MOVE_MIN_PAIRS_PER_ENV", "1"))
        self.camera_move_geometric_occlusion_precheck = os.getenv(
            "CAMERA_MOVE_GEOMETRIC_OCCLUSION_PRECHECK", "1").lower() not in {"0", "false", "no"}
        self.camera_move_min_agent_viewpoints = int(os.getenv(
            "CAMERA_MOVE_MIN_AGENT_VIEWPOINTS", "1"))
        self.camera_move_max_agent_viewpoints = int(os.getenv(
            "CAMERA_MOVE_MAX_AGENT_VIEWPOINTS", "10"))
        self.camera_move_min_geometric_agent_candidates = int(os.getenv(
            "CAMERA_MOVE_MIN_GEOMETRIC_AGENT_CANDIDATES", "1"))
        self.camera_move_debug_obb = os.getenv(
            "CAMERA_MOVE_DEBUG_OBB", "0").lower() in {"1", "true", "yes"}
        self.camera_move_generation_stats = {
            "batch_sweep_calls": 0,
            "candidate_angles_after_geometry": 0,
            "candidate_angles_rendered": 0,
            "candidate_angles_precheck_skipped": 0,
            "predicted_visible_candidates": 0,
            "predicted_occluded_candidates": 0,
            "accepted_balanced_envs": 0,
            "rejected_no_final_balanced_pair": 0,
        }

        # Agent camera geometry (from VPTEnvCfg: focal_length=24, horizontal_aperture=34)
        # half_hfov ≈ 35.3°; subtract buffer so camera CENTER must be well within frame
        _focal = 24.0
        _aperture = 34.0
        self._agent_half_hfov_deg = math.degrees(math.atan(_aperture / 2.0 / _focal))  # ~35.3°
        self._agent_fov_buffer_deg = 7.0  # camera center must be >= this inside the FOV edge

        # Cache for image tensors captured during batch sweep, keyed by slot id.
        # Populated by _build_fixed_sweep_trajectory_batch; consumed and cleared by _save_slot_from_cache.
        self._cached_frames: dict[int, list] = {}
        self.min_viewpoint_distance = 0.1
        self.save_camera_pov = True
        self.goal_pixel_threshold = 500
        self.goal_pixel_threshold_occlusion = 500
        self.camera_pixel_threshold = 800
        self.verbose = 2

        # --- Collection Counters & State ---
        self.valid_viewpoint_poses = [None] * self.num_envs
        self.selected_viewpoints_for_collection = [None] * self.num_envs
        self.camera_move_trajectories = [None] * self.num_envs
        self.camera_circle_radii = [None] * self.num_envs
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
        self.max_attempts_per_slot = 1000  # 20 resets * 50 inner
        self.next_env_folder_idx = 0
        self.env_visibility_labels = {}
        self.env_visibility_reasons = {}
        self._reset_called = False
        self.times = {}

        # In __init__
        self._preallocate_visibility_labels()

        # File paths
        self.GPU_ID = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.NODE_ID = os.getenv("NODE_ID", os.getenv("SLURM_ARRAY_TASK_ID", "0"))
        base = os.getenv("BASE_PATH", "/oscar/scratch/arock3/VPT1_DATA/camera/v18_cam_optim")
        self.base_path = f"{base}/data/data_node{self.NODE_ID}_gpu{self.GPU_ID}"

        print("*" * 50)
        print(f"🚀 Initializing VPTEnv on Node {self.NODE_ID} GPU {self.GPU_ID}...")
        print("*" * 50)

        self.visibility_labels_json_path = os.path.join(self.base_path, "visibility_labels.json")

        # --- Precomputed Math ---
        self.theta = math.pi / 12
        self.half_theta = self.theta / 2
        
        # Quaternion rotations (shape: 1, 4)
        c, s = math.cos(self.half_theta), math.sin(self.half_theta)
        self.rot_q_left = torch.tensor([[c, 0., 0., s]], device=self.device)
        self.rot_q_right = torch.tensor([[c, 0., 0., -s]], device=self.device)

        # --- A* Planner Defaults (3-action: forward/left/right) ---
        self.camera_yaw_correction_deg = 90.0
        self.camera_yaw_correction_rad = math.radians(self.camera_yaw_correction_deg)
        self.heading_bin_rad = self.theta  # pi/12 == 15 degrees
        self.num_heading_bins = int(round((2.0 * math.pi) / self.heading_bin_rad))
        self.forward_step_m = 2.0 * self.cfg.sim.dt  # Mirrors move_agent forward displacement.
        self.grid_step_m = self.forward_step_m
        self.astar_max_expansions = 50000
        self.planner_inflation_m = 0.15

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
        """Pre-allocate logical mixed labels for camera-move collection."""
        if hasattr(self, 'visibility_label_pool') and self.visibility_label_pool:
            return
        total = self.total_envs_to_sim
        self.visibility_label_pool = ["mixed_camera_move"] * total

        if self.verbose >= 1:
            print(f"📋 Pre-allocated {total} mixed camera-move labels:")
            print(f"   - sweep angles: {self.camera_sweep_angles_deg}")
            print(f"   - target saved frames per env: {self.camera_move_target_frames_per_env}")
            print(f"   - require both labels: {self.camera_move_balance_saved_frames}")
            print(f"   - min Yes/No pairs per env: {self.camera_move_min_pairs_per_env}")
            print(f"   - geometric occlusion precheck: {self.camera_move_geometric_occlusion_precheck}")
            print(f"   - min fixed-agent viewpoints: {self.camera_move_min_agent_viewpoints}")
            print(f"   - max fixed-agent viewpoints to score: {self.camera_move_max_agent_viewpoints}")

    def _assign_next_visibility_label(self, folder_idx: int) -> str:
        """Assign the next logical mixed label from the pre-allocated pool."""
        if not self.visibility_label_pool:
            raise RuntimeError("Visibility label pool exhausted!")

        category = self.visibility_label_pool.pop(0)

        if category != "mixed_camera_move":
            raise RuntimeError(f"Unexpected camera-move label category: {category}")

        self.env_visibility_labels[folder_idx] = "Mixed"
        self.env_visibility_reasons[folder_idx] = "mixed_camera_move"

        return category

    def _setup_scene(self):
        # --- Static Elements ---
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(1000, 1000)))

        # --- Rigid Objects ---
        self._agent = RigidObject(self.cfg.agent)
        self._goal = RigidObject(self.cfg.goal_ball)
        self._mat = RigidObject(self.cfg.mat)
        self._camera_obj = RigidObject(self.cfg.camera_obj)
        self._vpt_objects = RigidObjectCollection(self.cfg.vpt_objects)
        
        # Walls
        self._boundary_top = RigidObject(self.cfg.top_wall)
        self._boundary_bottom = RigidObject(self.cfg.bottom_wall)
        self._boundary_left = RigidObject(self.cfg.left_wall)
        self._boundary_right = RigidObject(self.cfg.right_wall)

        # Register Objects
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

        # --- Sensors ---
        self._rgb_tiled_camera = TiledCamera(self.cfg.rgb_tiled_camera)
        self._occlusion_camera = TiledCamera(self.cfg.occlusion_camera)

        self.scene.sensors.update({
            "rgb_tiled_camera": self._rgb_tiled_camera,
            "occlusion_camera": self._occlusion_camera,
        })

        # --- Scene Setup ---
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # --- Lighting ---
        light_cfg = sim_utils.SphereLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/envs/env_0/Light_A", light_cfg)

        # --- Materials ---
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

    # TODO: CONSIDER MERGING WITH _check_occlusion
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

    def move_agent(self, actions: torch.Tensor, env_ids: Sequence[int] | None = None):
        """
        Moves the agent kinematically based on discrete actions.

        This method handles both physics-based movement (translation/rotation) 
        and environment resets. It updates the simulation state directly 
        (teleportation) rather than applying forces.

        Parameters
        ----------
        actions : torch.Tensor
            A tensor of action indices for each environment.
            0: Move Forward
            1: Move Backward
            2: Turn Left
            3: Turn Right
            5: Soft Reset (Non-RL)
            6: Hard Reset (RL)
        env_ids : Sequence[int] | None, optional
            Indices of environments to update. If None, updates all.

        Notes
        -----
        - Performance: Direct velocity writing is removed for optimization.
        - Orientation: Uses raw current orientation without upright normalization.
        """
        # --- 1. Setup & Input Normalization ---
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES
        
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # --- 2. Handle Resets ---
        # Action 5: Soft Reset, Action 6: Hard Reset
        reset_mask_5 = (actions == 5)
        reset_mask_6 = (actions == 6)
        physics_mask = ~(reset_mask_5 | reset_mask_6)

        if reset_mask_5.any():
            self._reset_idx(env_ids[reset_mask_5], rl_reset=False)

        if reset_mask_6.any():
            print("+" * 50) # Visual separator for hard resets
            self._reset_idx(env_ids[reset_mask_6], rl_reset=True)

        # Exit if only resets occurred
        if not physics_mask.any():
            return

        # --- 3. Prepare Physics Update ---
        phys_ids = env_ids[physics_mask]
        phys_actions = actions[physics_mask]
        dt = self.cfg.sim.dt

        # Clone current state to calculate proposed state
        current_pos = self._agent.data.root_pos_w[phys_ids]
        current_quat = self._agent.data.root_quat_w[phys_ids]
        
        tentative_pos = current_pos.clone()
        new_quat = current_quat.clone()

        # --- 4. Apply Rotation (Actions 2 & 3) ---
        mask_left = (phys_actions == 2)
        mask_right = (phys_actions == 3)

        if mask_left.any():
            new_quat[mask_left] = math_utils.quat_mul(
                current_quat[mask_left], 
                self.rot_q_left.expand(mask_left.sum(), -1)
            )

        if mask_right.any():
            new_quat[mask_right] = math_utils.quat_mul(
                current_quat[mask_right], 
                self.rot_q_right.expand(mask_right.sum(), -1)
            )

        # --- 5. Apply Translation (Actions 0 & 1) ---
        mask_fwd = (phys_actions == 0)
        mask_bwd = (phys_actions == 1)
        moving_mask = mask_fwd | mask_bwd

        if moving_mask.any():
            # Create local movement vectors: +x for fwd, -x for bwd
            n_moving = moving_mask.sum()
            local_move = torch.zeros((n_moving, 3), device=self.device)
            
            moving_actions = phys_actions[moving_mask]
            local_move[moving_actions == 0, 0] = 1.0
            local_move[moving_actions == 1, 0] = -1.0

            # Transform local vector to world frame using CURRENT orientation
            # (Movement is relative to where agent was facing before this step)
            world_vel = math_utils.quat_apply(current_quat[moving_mask], local_move)
            
            # Apply displacement (Speed * dt * 2.0 multiplier)
            tentative_pos[moving_mask] += (world_vel * 2.0 * dt)

        # Enforce fixed height constraint
        tentative_pos[:, 2] = self._agent.data.default_root_state[phys_ids, 2]

        # --- 6. Collision Checking ---
        if moving_mask.any():
            # Check collisions at the PROPOSED state
            is_collision = self._check_collisions_vectorized(phys_ids, tentative_pos, new_quat)
            
            # Revert state for agents that collided
            collision_mask = is_collision & moving_mask
            if collision_mask.any():
                revert_indices = torch.where(collision_mask)[0]
                # Revert to original state
                tentative_pos[revert_indices] = current_pos[revert_indices]
                new_quat[revert_indices] = current_quat[revert_indices]

        # --- 7. Commit State ---
        # Combine position and quaternion for bulk write
        new_pose = torch.cat([tentative_pos, new_quat], dim=1)
        self._agent.write_root_com_pose_to_sim(new_pose, phys_ids)
        # Zero out velocity so residual angular/linear velocity from
        # physics doesn't undo the pose we just wrote (rotation or
        # translation). Without this, sim.step() applies the old
        # angular velocity and partially reverts yaw changes.
        self._agent.write_root_com_velocity_to_sim(
            torch.zeros(len(phys_ids), 6, device=self.device), phys_ids)
        self._agent.reset()

    def _update_camera_poses(self, env_ids):
        """Helper to handle the camera/occlusion update logic."""
        camera_obj_pos = self._camera_obj.data.root_pos_w[env_ids].clone()
        camera_obj_quat = self._camera_obj.data.root_quat_w[env_ids].clone()

        # Rotate 90 degrees left
        half_theta = (math.pi / 2) / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta), 0.0, 0.0,
             math.sin(half_theta)],
            device=self.device)

        rotated_orientations = math_utils.quat_mul(
            camera_obj_quat, left_90_quat.expand(len(env_ids), -1))

        self._occlusion_camera.set_world_poses(
            positions=camera_obj_pos,
            orientations=rotated_orientations,
            env_ids=env_ids.tolist(),
            convention="world")

    def _normalize_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _quat_wxyz_to_yaw(self, quat: torch.Tensor) -> float:
        # Isaac Lab quaternions used here are [w, x, y, z].
        w, x, y, z = [float(v) for v in quat]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _get_agent_yaw(self, env_id: int) -> float:
        return self._quat_wxyz_to_yaw(self._agent.data.root_quat_w[env_id])

    def _get_camera_corrected_yaw(self, env_id: int) -> float:
        yaw = self._quat_wxyz_to_yaw(self._camera_obj.data.root_quat_w[env_id])
        return self._normalize_angle(yaw + self.camera_yaw_correction_rad)

    def _heading_bin_to_yaw(self, heading_bin: int) -> float:
        return self._normalize_angle(heading_bin * self.heading_bin_rad)

    def _yaw_to_heading_bin(self, yaw: float) -> int:
        idx = int(round(self._normalize_angle(yaw) / self.heading_bin_rad)) % self.num_heading_bins
        return idx

    def _yaw_to_quat_wxyz(self, yaw: float) -> torch.Tensor:
        half = yaw / 2.0
        return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], device=self.device, dtype=torch.float32)

    def _world_to_grid(self, xy: torch.Tensor, origin_xy: torch.Tensor, min_xy: torch.Tensor) -> tuple[int, int]:
        rel = (xy[:2] - origin_xy[:2]) - min_xy
        gx = int(round(float(rel[0] / self.grid_step_m)))
        gy = int(round(float(rel[1] / self.grid_step_m)))
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int, origin_xy: torch.Tensor, min_xy: torch.Tensor) -> torch.Tensor:
        x = float(origin_xy[0] + min_xy[0] + gx * self.grid_step_m)
        y = float(origin_xy[1] + min_xy[1] + gy * self.grid_step_m)
        return torch.tensor([x, y], device=self.device, dtype=torch.float32)

    def build_nav_grid(self, env_id: int, inflation_radius: float | None = None) -> dict:
        if inflation_radius is None:
            inflation_radius = self.planner_inflation_m

        if hasattr(self, "obb_corners_cache"):
            env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
            vpt_pose = self._vpt_objects.data.object_state_w[:, :, :7]
            self.update_obb_cache(env_ids, vpt_pose)

        origin_xy = self.scene.env_origins[env_id, :2]
        bound = float(self.center_to_boundary)
        min_xy = torch.tensor([-bound, -bound], device=self.device, dtype=torch.float32)
        max_xy = torch.tensor([bound, bound], device=self.device, dtype=torch.float32)

        size = int(math.floor((2.0 * bound) / self.grid_step_m)) + 1
        occ = torch.zeros((size, size), dtype=torch.bool, device=self.device)

        if hasattr(self, "obb_corners_cache"):
            corners_xy = self.obb_corners_cache[env_id, :, :, :2]
            mins = corners_xy.min(dim=1).values
            maxs = corners_xy.max(dim=1).values
            inflate = inflation_radius
            for i in range(mins.shape[0]):
                obj_min = mins[i] - inflate
                obj_max = maxs[i] + inflate
                gx0, gy0 = self._world_to_grid(obj_min, origin_xy, min_xy)
                gx1, gy1 = self._world_to_grid(obj_max, origin_xy, min_xy)
                lo_x = min(gx0, gx1)
                hi_x = max(gx0, gx1)
                lo_y = min(gy0, gy1)
                hi_y = max(gy0, gy1)
                gx0 = max(0, min(size - 1, lo_x))
                gx1 = max(0, min(size - 1, hi_x))
                gy0 = max(0, min(size - 1, lo_y))
                gy1 = max(0, min(size - 1, hi_y))
                occ[gx0:gx1 + 1, gy0:gy1 + 1] = True

        wall_cells = max(1, int(math.ceil(inflation_radius / self.grid_step_m)))
        occ[:wall_cells, :] = True
        occ[-wall_cells:, :] = True
        occ[:, :wall_cells] = True
        occ[:, -wall_cells:] = True

        return {
            "occupancy": occ,
            "origin_xy": origin_xy,
            "min_xy": min_xy,
            "max_xy": max_xy,
            "grid_size": size,
            "grid_step_m": self.grid_step_m,
            "heading_bin_deg": math.degrees(self.heading_bin_rad),
        }

    def _is_goal_reached_3act(
        self,
        env_id: int,
        pos_tol_m: float,
        yaw_tol_deg: float,
        camera_xy: torch.Tensor | None = None,
    ) -> tuple[bool, float, float]:
        agent_xy = self._agent.data.root_pos_w[env_id, :2]
        if camera_xy is None:
            camera_xy = self._camera_obj.data.root_pos_w[env_id, :2]
        pos_err = float(torch.norm(agent_xy - camera_xy).item())

        agent_yaw = self._get_agent_yaw(env_id)
        target_yaw = self._get_camera_corrected_yaw(env_id)
        yaw_err = abs(math.degrees(self._normalize_angle(target_yaw - agent_yaw)))
        reached = (pos_err <= pos_tol_m) and (yaw_err <= yaw_tol_deg)
        return reached, pos_err, yaw_err

    def plan_to_camera_actions_3act(
        self,
        env_id: int,
        pos_tol_m: float = 0.2,
        yaw_tol_deg: float = 10.0,
        max_steps: int = 512,
    ) -> dict:
        nav = self.build_nav_grid(env_id)
        occ = nav["occupancy"]
        size = nav["grid_size"]
        origin_xy = nav["origin_xy"]
        min_xy = nav["min_xy"]

        start_xy = self._agent.data.root_pos_w[env_id, :2]
        goal_xy = self._camera_obj.data.root_pos_w[env_id, :2]
        start_heading = self._yaw_to_heading_bin(self._get_agent_yaw(env_id))

        sx, sy = self._world_to_grid(start_xy, origin_xy, min_xy)
        gx, gy = self._world_to_grid(goal_xy, origin_xy, min_xy)
        sx = max(0, min(size - 1, sx))
        sy = max(0, min(size - 1, sy))
        gx = max(0, min(size - 1, gx))
        gy = max(0, min(size - 1, gy))

        if occ[sx, sy]:
            occ[sx, sy] = False
        if occ[gx, gy]:
            occ[gx, gy] = False

        start = (sx, sy, start_heading)
        open_heap = []
        heapq.heappush(open_heap, (0.0, 0, start))
        came_from = {}
        g_cost = {start: 0.0}
        visited = set()

        def h(x: int, y: int) -> float:
            return math.hypot(gx - x, gy - y)

        found = None
        expansions = 0
        while open_heap and expansions < self.astar_max_expansions:
            _, _, node = heapq.heappop(open_heap)
            if node in visited:
                continue
            visited.add(node)
            x, y, hd = node
            expansions += 1
            if math.hypot(gx - x, gy - y) <= 1.0:
                found = node
                break

            left = (x, y, (hd + 1) % self.num_heading_bins)
            right = (x, y, (hd - 1) % self.num_heading_bins)
            for nxt, action in ((left, 2), (right, 3)):
                tentative = g_cost[node] + 1.0
                if tentative < g_cost.get(nxt, float("inf")):
                    g_cost[nxt] = tentative
                    came_from[nxt] = (node, action)
                    f = tentative + h(nxt[0], nxt[1])
                    heapq.heappush(open_heap, (f, expansions, nxt))

            yaw = self._heading_bin_to_yaw(hd)
            dx = int(round(math.cos(yaw)))
            dy = int(round(math.sin(yaw)))
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and not occ[nx, ny]:
                fwd = (nx, ny, hd)
                tentative = g_cost[node] + 1.0
                if tentative < g_cost.get(fwd, float("inf")):
                    g_cost[fwd] = tentative
                    came_from[fwd] = (node, 0)
                    f = tentative + h(nx, ny)
                    heapq.heappush(open_heap, (f, expansions, fwd))

        if found is None:
            return {
                "success": False,
                "reason": "no_path",
                "actions": [],
                "metrics": {
                    "expanded_nodes": expansions,
                    "grid_step_m": self.grid_step_m,
                    "heading_bin_deg": math.degrees(self.heading_bin_rad),
                    "camera_yaw_correction_deg": self.camera_yaw_correction_deg,
                },
            }

        actions = []
        cur = found
        while cur != start:
            prev, action = came_from[cur]
            actions.append(int(action))
            cur = prev
        actions.reverse()

        align_target = self._get_camera_corrected_yaw(env_id)
        end_heading = found[2]
        end_yaw = self._heading_bin_to_yaw(end_heading)
        delta = self._normalize_angle(align_target - end_yaw)
        steps = int(round(abs(delta) / self.heading_bin_rad))
        turn_action = 2 if delta > 0 else 3
        actions.extend([turn_action] * steps)
        actions = actions[:max_steps]

        waypoints = []
        node = start
        for a in actions:
            if a == 2:
                node = (node[0], node[1], (node[2] + 1) % self.num_heading_bins)
            elif a == 3:
                node = (node[0], node[1], (node[2] - 1) % self.num_heading_bins)
            elif a == 0:
                yaw = self._heading_bin_to_yaw(node[2])
                node = (node[0] + int(round(math.cos(yaw))), node[1] + int(round(math.sin(yaw))), node[2])
            wp = self._grid_to_world(node[0], node[1], origin_xy, min_xy)
            waypoints.append([float(wp[0]), float(wp[1])])

        return {
            "success": True,
            "reason": "ok",
            "actions": actions,
            "waypoints": waypoints,
            "metrics": {
                "expanded_nodes": expansions,
                "path_len": len(actions),
                "grid_step_m": self.grid_step_m,
                "heading_bin_deg": math.degrees(self.heading_bin_rad),
                "camera_yaw_correction_deg": self.camera_yaw_correction_deg,
            },
        }

    def execute_action_sequence(
        self,
        env_ids: torch.Tensor,
        action_sequences: list[list[int]],
        pos_tol_m: float = 0.2,
        yaw_tol_deg: float = 10.0,
    ) -> dict:
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.view(-1)
        done = [False] * len(env_ids)
        steps_used = [0] * len(env_ids)

        max_len = max((len(seq) for seq in action_sequences), default=0)
        for t in range(max_len):
            actions = torch.full((len(env_ids),), 5, device=self.device, dtype=torch.long)
            for i, env_id in enumerate(env_ids.tolist()):
                if done[i]:
                    continue
                seq = action_sequences[i]
                if t < len(seq):
                    actions[i] = int(seq[t])
                else:
                    done[i] = True
                    continue
                steps_used[i] += 1
                reached, _, _ = self._is_goal_reached_3act(env_id, pos_tol_m, yaw_tol_deg)
                if reached:
                    done[i] = True
            self.move_agent(actions, env_ids)
            self.sim.step(render=False)
            if all(done):
                break

        final_pos_err = []
        final_yaw_err = []
        success = []
        for env_id in env_ids.tolist():
            reached, pos_err, yaw_err = self._is_goal_reached_3act(env_id, pos_tol_m, yaw_tol_deg)
            success.append(reached)
            final_pos_err.append(pos_err)
            final_yaw_err.append(yaw_err)

        return {
            "success": success,
            "steps_used": steps_used,
            "final_pos_err": final_pos_err,
            "final_yaw_err": final_yaw_err,
        }

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
        terminated = torch.zeros(self.num_envs,
                                 dtype=torch.bool,
                                 device=self.device)
        terminated = terminated | goal_reached
        time_outs = (self.episode_length_buf >= self.max_episode_length)
        return terminated, time_outs

    def _validate_env_state(self, env_id: torch.Tensor, folder_idx: int, min_viewpoints: int) -> tuple[bool, str]:
        """
        Validates a single environment's state after a reset attempt.

        Ensures sufficient viewpoints were generated and verifies that the 
        physical occlusion state (via raycast) matches the expected visibility label.

        Parameters
        ----------
        env_id : torch.Tensor
            The environment index (scalar tensor).
        folder_idx : int
            The folder/case index associated with this environment.
        min_viewpoints : int
            Minimum number of valid viewpoints required for success.

        Returns
        -------
        tuple[bool, str]
            (True, "") if state is valid, otherwise (False, failure_reason).
        """
        env_idx = env_id.item()

        trajectory = self.camera_move_trajectories[env_idx] if self.camera_move_trajectories else None
        if trajectory is None:
            return False, "missing camera trajectory"

        if len(trajectory) < min_viewpoints:
            return False, f"insufficient camera frames: {len(trajectory)}/{min_viewpoints}"

        if self.camera_move_balance_saved_frames:
            yes = sum(1 for item in trajectory if item.get("label") == "Yes")
            no = sum(1 for item in trajectory if item.get("label") == "No")
            if yes < self.camera_move_min_pairs_per_env:
                return False, f"insufficient Yes frames: {yes}/{self.camera_move_min_pairs_per_env}"
            if no < self.camera_move_min_pairs_per_env:
                return False, f"insufficient No frames: {no}/{self.camera_move_min_pairs_per_env}"

        radius = self.camera_circle_radii[env_idx] if self.camera_circle_radii else None
        if radius is None:
            return False, "missing camera circle radius"

        goal_xy = self._goal.data.root_pos_w[env_idx, :2]
        for item in trajectory:
            dist = torch.norm(item["position"] - goal_xy).item()
            if abs(dist - radius) > 1e-3:
                return False, f"camera radius drift: {dist:.6f} vs {radius:.6f}"

        if not self._validate_fixed_agent_view(env_idx):
            return False, "fixed agent does not see both goal and camera"

        return True, ""

    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)
        return obs, rewards, terminated, truncated, info

    def render(self):

        frame = self.obs[0].permute(1, 2, 0).cpu().numpy()
        #draw the action from 1st env on the frame
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
        cv2.putText(frame, action_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        #save frame as image
        cv2.imwrite(f"frame_{self.step_count}_{action_str}.png", frame)

        return frame

    def _check_occlusion(self, 
                         camera_pos: torch.Tensor, 
                         goal_pos: torch.Tensor, 
                         env_id: int | torch.Tensor, 
                         camera=None) -> bool:
        """
        Determines if the goal is occluded using semantic segmentation.

        Checks if the visible goal pixels (red) fall below the threshold.
        Also performs a 'lazy synchronization' of the camera sensor if it 
        has drifted from the underlying rigid body.

        Parameters
        ----------
        camera_pos : torch.Tensor
            Position of the main camera object (unused in logic, kept for API).
        goal_pos : torch.Tensor
            Position of the goal object (unused in logic, kept for API).
        env_id : int | torch.Tensor
            The environment index to check.
        camera : Any, optional
            The camera sensor to use. Defaults to self._occlusion_camera.

        Returns
        -------
        bool
            True if goal is occluded (pixel count < threshold), False otherwise.
        """
        if camera is None:
            camera = self._occlusion_camera

        # --- 1. Drift Check & Synchronization ---
        # Ensure sensor (camera) is aligned with the rigid body (camera_obj)
        target_pos = self._camera_obj.data.root_pos_w[env_id]
        sensor_pos = camera.data.pos_w[env_id]
        
        if torch.norm(target_pos - sensor_pos).item() > 0.01:
            # Construct 90-degree Z-axis rotation quaternion (Left Turn)
            # q = [cos(pi/4), 0, 0, sin(pi/4)]
            angle = math.pi / 4  # half of 90 degrees
            rot_correction = torch.tensor(
                [math.cos(angle), 0.0, 0.0, math.sin(angle)], 
                device=target_pos.device
            )

            # Apply rotation to current body orientation
            body_quat = self._camera_obj.data.root_quat_w[env_id]
            new_orient = math_utils.quat_mul(
                body_quat.unsqueeze(0), 
                rot_correction.unsqueeze(0)
            ).squeeze(0)

            # Force update pose and step simulation to render
            # Handle env_id as list for set_world_poses
            idx_list = [env_id.item()] if hasattr(env_id, "item") else [env_id]
            
            camera.set_world_poses(
                positions=target_pos.unsqueeze(0),
                orientations=new_orient.unsqueeze(0),
                env_ids=idx_list,
                convention="world"
            )
            self.sim.step()

        # --- 2. Semantic Visibility Check ---
        # Goal is Red: (R >= 0.95, G <= 0.05, B <= 0.05)
        sem_img = camera.data.output["semantic_segmentation"][env_id]
        r, g, b = sem_img[..., 0], sem_img[..., 1], sem_img[..., 2]

        red_mask = (r >= 0.95) & (g <= 0.05) & (b <= 0.05)
        visible_pixels = red_mask.sum().item()

        return visible_pixels < self.goal_pixel_threshold_occlusion

    def _save_visibility_labels(self):
        """
        Saves visibility labels and statistics to a JSON file.

        The output JSON structure includes:
        - environments: A mapping of folder indices to their label and reason.
        - statistics: Aggregated counts of visibility states (Yes/No) and reasons.

        Side Effects
        ------------
        - Creates the directory for `self.visibility_labels_json_path` if it doesn't exist.
        - Overwrites the JSON file at `self.visibility_labels_json_path`.
        """
        os.makedirs(os.path.dirname(self.visibility_labels_json_path), exist_ok=True)

        # --- 1. Compile Environment Details ---
        # Map: { "0": {"label": "Yes", "reason": "in_view"}, ... }
        env_details = {
            str(idx): {
                "label": label,
                "reason": self.env_visibility_reasons.get(idx, "unknown")
            }
            for idx, label in self.env_visibility_labels.items()
        }

        # --- 2. Calculate Statistics ---
        # Initialize defaults to ensure keys exist even if counts are 0
        reason_counts = {"in_view": 0, "occluded": 0, "mixed_camera_move": 0, "unknown": 0}
        
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
                "mixed_count": sum(1 for v in self.env_visibility_labels.values() if v == "Mixed"),
                "by_reason": reason_counts,
                "next_env_folder_idx": self.next_env_folder_idx
            }
        }

        # --- 3. Write to File ---
        with open(self.visibility_labels_json_path, 'w') as f:
            json.dump(labels_data, f, indent=2)

    def _check_target_in_img(self,
                             file_name: str,
                             cam_pov: np.ndarray,
                             return_red_count: bool = False) -> bool | tuple[bool, int]:
        """
        Checks if the target (red object) is visible in the provided image.

        Saves the image to disk and checks for visibility using:
        1. Color Thresholding (Red > 95%)
        2. Geometric detection (Hough Circles)

        Parameters
        ----------
        file_name : str
            Path to save the camera point-of-view image.
        cam_pov : np.ndarray
            The image array (RGB, uint8 expected).
        return_red_count : bool, optional
            If True, returns the pixel count along with the boolean result.

        Returns
        -------
        bool | tuple[bool, int]
            True if target is detected, or (True, red_pixel_count).
        """
        # TODO: Consider merging this logic with `_check_occlusion` to unify visibility checks.

        # --- 1. Save Image ---
        # Convert RGB to BGR for OpenCV saving
        cv2.imwrite(file_name, cv2.cvtColor(cam_pov, cv2.COLOR_RGB2BGR))

        # --- 2. Color Thresholding ---
        # Optimization: Analyze input array directly instead of reading back from disk.
        # Thresholds: Red > 242 (~0.95), Green/Blue < 13 (~0.05)
        r, g, b = cam_pov[..., 0], cam_pov[..., 1], cam_pov[..., 2]
        
        red_mask = (r >= 242) & (g <= 13) & (b <= 13)
        red_count = red_mask.sum().item()
        
        target_visible = red_count >= self.goal_pixel_threshold_occlusion

        # --- 3. Geometric Check (Hough Circles) ---
        # gray = cv2.cvtColor(cam_pov, cv2.COLOR_RGB2GRAY)
        # circles = cv2.HoughCircles(
        #     gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10, 
        #     param1=100, param2=12, minRadius=5, maxRadius=200
        # )
        
        # has_circles = circles is not None and len(circles[0]) > 0
        # is_detected = target_visible or has_circles
        is_detected = target_visible

        if return_red_count:
            return is_detected, red_count
        return is_detected

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

        camera_trajectory = []
        if (self.camera_move_trajectories is not None
                and env_id_item < len(self.camera_move_trajectories)
                and self.camera_move_trajectories[env_id_item] is not None):
            for frame_idx, item in enumerate(self.camera_move_trajectories[env_id_item]):
                camera_trajectory.append({
                    "frame_idx": frame_idx,
                    "angle_deg": int(item["angle_deg"]),
                    "position": item["position"].cpu().numpy().tolist(),
                    "yaw": float(item["yaw"]),
                    "label": item["label"],
                    "reason": item["reason"],
                    "red_count": int(item["red_count"]),
                })

        camera_circle_radius = None
        if (self.camera_circle_radii is not None
                and env_id_item < len(self.camera_circle_radii)
                and self.camera_circle_radii[env_id_item] is not None):
            camera_circle_radius = float(self.camera_circle_radii[env_id_item])

        # Build configuration dictionary
        config = {
            "metadata": {
                "env_id": env_id_item,
                "folder_idx": folder_idx,
                "visibility_label": label,
                "visibility_reason": reason,
                "cfg_version": "1.0",
                "optimized_env_copy": True,
            },
            "environment_settings": {
                "boundary_limits": list(self.cfg.boundary_limits),
                "agent_height": float(self.cfg.agent_height),
                "agent_camera_pitch": float(self.cfg.agent_camera_pitch),
                "action_scale": float(self.cfg.action_scale),
                "num_vpt_objs": int(self.cfg.num_vpt_objs),
                "planner": {
                    "camera_yaw_correction_deg": float(self.camera_yaw_correction_deg),
                    "grid_step_m": float(self.grid_step_m),
                    "heading_bin_deg": float(math.degrees(self.heading_bin_rad)),
                },
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
            },
            "camera_move_collection": {
                "mode": "fixed_agent_right_half_sweep",
                "sweep_angles_deg": self.camera_sweep_angles_deg,
                "images_per_env": self.images_per_env,
                "target_frames_per_env": int(self.camera_move_target_frames_per_env),
                "balance_saved_frames": bool(self.camera_move_balance_saved_frames),
                "min_pairs_per_env": int(self.camera_move_min_pairs_per_env),
                "geometric_occlusion_precheck": bool(
                    self.camera_move_geometric_occlusion_precheck),
                "min_agent_viewpoints": int(self.camera_move_min_agent_viewpoints),
                "max_agent_viewpoints": int(self.camera_move_max_agent_viewpoints),
                "min_geometric_agent_candidates": int(
                    self.camera_move_min_geometric_agent_candidates),
                "generation_stats_snapshot": dict(self.camera_move_generation_stats),
                "fixed_radius": camera_circle_radius,
                "fixed_agent_pose": {
                    "position": agent_pos,
                    "orientation": agent_quat,
                },
                "fixed_goal_pose": {
                    "position": goal_pos,
                    "orientation": goal_quat,
                },
                "trajectory": camera_trajectory,
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

    def _get_batch_active_indices(self, env_ids: int | list | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper to standardize environment IDs and retrieve their active object indices.

        Parameters
        ----------
        env_ids : int | list | torch.Tensor
            The environment indices to query.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (env_ids_flat, batch_indices)
            - env_ids_flat: 1D tensor of environment IDs.
            - batch_indices: 2D tensor of active object indices [batch_size, num_active].
        """
        # 1. Standardize env_ids to 1D Tensor
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        env_ids = env_ids.view(-1)

        # 2. Retrieve Active Indices Matrix
        if isinstance(self.active_vpt_indices, list):
            # Check first element to handle potentially empty lists gracefully
            if self.active_vpt_indices and isinstance(self.active_vpt_indices[0], torch.Tensor):
                full_indices = torch.stack(self.active_vpt_indices).to(self.device)
            else:
                full_indices = torch.tensor(self.active_vpt_indices, device=self.device, dtype=torch.long)
        else:
            full_indices = self.active_vpt_indices.to(self.device)

        return env_ids, full_indices[env_ids]

    def _select_active_vpt_indices(self, env_ids: torch.Tensor) -> None:
        """
        Randomly selects a subset of active VPT object indices for the specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor
            The environment indices to update.
        """
        env_ids = env_ids.view(-1) if torch.is_tensor(env_ids) else torch.tensor(env_ids)
        
        for env_id in env_ids:
            # Random permutation to select unique objects
            active = torch.randperm(self.num_objs, device=self.device)[:self.active_vpt_objs]
            self.active_vpt_indices[env_id.item()] = active

    def _store_inactive_vpt_objects(self, env_ids: torch.Tensor, vpt_obj_default_state: torch.Tensor) -> torch.Tensor:
        """
        Moves inactive VPT objects to a hidden storage position to prevent rendering/collision.
        """
        # Assuming this returns global env IDs and the active indices for them
        ids, active_indices = self._get_batch_active_indices(env_ids) 
        
        num_batch = len(ids)

        # 1. Create Inactive Mask (True = Inactive/Hidden)
        # Start with all True (Hidden)
        inactive_mask = torch.ones((num_batch, self.num_objs), dtype=torch.bool, device=self.device)
        
        # Scatter False (Active) into the mask
        # We use src=False. scatter_ expects src to be a tensor or scalar. 
        # Ensure active_indices is LongTensor and shaped [batch, num_active_indices]
        inactive_mask.scatter_(1, active_indices, False)

        # 2. Prepare Indices for Vectorized Update
        
        # [FIX]: Use torch.arange(num_batch) for local row indices, NOT ids (which are global)
        local_row_indices = torch.arange(num_batch, device=self.device).view(-1, 1).expand(-1, self.num_objs)
        
        col_indices = torch.arange(self.num_objs, device=self.device).expand(num_batch, -1)

        # Select only the inactive entries using the mask
        target_rows = local_row_indices[inactive_mask]
        target_cols = col_indices[inactive_mask]

        # 3. Apply Storage Position and Zero Velocity
        if len(target_rows) > 0:
            vpt_obj_default_state[target_rows, target_cols, 0] = self.storage_position[0]
            vpt_obj_default_state[target_rows, target_cols, 1] = self.storage_position[1]
            vpt_obj_default_state[target_rows, target_cols, 2] = self.storage_position[2]
            
            # Zero out velocities (indices 7 to 13)
            vpt_obj_default_state[target_rows, target_cols, 7:13] = 0.0

        return vpt_obj_default_state

    def _get_active_vpt_dims(self, env_ids: int | torch.Tensor) -> torch.Tensor:
        """
        Retrieves dimensions for the active objects in the given environments.

        Parameters
        ----------
        env_ids : int | torch.Tensor
            The environment indices.

        Returns
        -------
        torch.Tensor
            Dimensions tensor of shape [batch_size, active_objs, 3].
        """
        ids, batch_indices = self._get_batch_active_indices(env_ids)
        
        # Expand env_ids for advanced indexing: [batch, active_objs]
        env_ids_expanded = ids.view(-1, 1).expand_as(batch_indices)
        
        return self.all_vpt_dims[env_ids_expanded, batch_indices, :]

    def _get_active_vpt_positions(self, 
                                  env_ids: int | torch.Tensor, 
                                  base_pivoted: bool = False, 
                                  return_full_pose: bool = False) -> torch.Tensor:
        """
        Retrieves world positions (and optionally orientation) of active objects.

        Parameters
        ----------
        env_ids : int | torch.Tensor
            The environment indices.
        base_pivoted : bool, optional
            If True, adjusts Z-coordinate from COM to object base (floor level).
        return_full_pose : bool, optional
            If True, returns [x, y, z, qx, qy, qz, qw] (7 dim). 
            If False, returns [x, y, z] (3 dim).

        Returns
        -------
        torch.Tensor
            Tensor of positions or full poses [batch_size, active_objs, 3 or 7].
        """
        ids, batch_indices = self._get_batch_active_indices(env_ids)
        
        # Expand env_ids: [batch, active_objs]
        env_ids_expanded = ids.view(-1, 1).expand_as(batch_indices)

        # 1. Fetch Positions (Clone to protect sim state)
        active_pos = self._vpt_objects.data.object_pos_w[env_ids_expanded, batch_indices].clone()

        # 2. Apply Base Pivot (Adjust Z)
        if base_pivoted:
            heights = self.all_vpt_dims[env_ids_expanded, batch_indices, 2]
            ratios = self.vpt_z_offset_ratios[env_ids_expanded, batch_indices]
            
            # Subtract (height * ratio) to move from COM to floor
            active_pos[:, :, 2] -= (heights * ratios)

        # 3. Return Logic
        if return_full_pose:
            active_quat = self._vpt_objects.data.object_quat_w[env_ids_expanded, batch_indices].clone()
            return torch.cat([active_pos, active_quat], dim=-1)

        return active_pos

    def _select_viewpoints_for_collection(self, env_id: int) -> bool:
        """Select the prevalidated camera trajectory for a single environment slot.
        
        Args:
            env_id: Environment slot index (0-7)
            
        Returns:
            True if selection successful, False otherwise
        """
        if (self.camera_move_trajectories is None
                or env_id >= len(self.camera_move_trajectories)
                or self.camera_move_trajectories[env_id] is None
                or len(self.camera_move_trajectories[env_id]) == 0):
            return False

        trajectory = self.camera_move_trajectories[env_id]
        self.selected_viewpoints_for_collection[env_id] = torch.stack(
            [item["position"] for item in trajectory])
        return True

    def _reset_idx(self, env_ids: Sequence[int] | None, rl_reset: bool = True) -> None:
        """
        Resets specified environments.

        Modes:
        - rl_reset=True: Fast physics reset and randomization (Training).
        - rl_reset=False: Full pipeline with validation, data collection, and slot replenishment.

        Parameters
        ----------
        env_ids : Sequence[int] | None
            Indices of environments to reset. None defaults to all.
        rl_reset : bool
            If True, skips data collection/validation logic.
        """
        # --- 1. Standardization & Setup ---
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES
        
        # Ensure env_ids is a Tensor for internal ops, list for iteration
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        active_slots_list = env_ids.tolist()

        # Initialize slot tracking mechanism if this is the first run
        self._ensure_slot_initialization()

        # --- 2. Scene Randomization ---
        self._cache_base_dims()
        self._randomize_scene_props(active_slots_list)

        # Randomly select 25% of envs to place goal on top of an object
        num_targets = max(1, int(0.25 * len(env_ids)))
        subset_idx = torch.randperm(len(env_ids), device=self.device)[:num_targets]
        self.envs_to_move_ball = env_ids[subset_idx]

        # --- 3. Physics Reset ---
        # Map slots to their current folder/visibility state
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

        # ==========================================================
        #  DATA COLLECTION PIPELINE (rl_reset=False only)
        # ==========================================================
        
        valid_slots, exceeded_slots = self._validate_slots(active_slots_list)

        # --- 4. Collection ---
        for slot_idx in valid_slots:
            folder_idx = self.slot_folder_indices[slot_idx]

            if self._select_viewpoints_for_collection(slot_idx):
                if slot_idx in self._cached_frames:
                    # Fast path: images captured during batch sweep — just write to disk
                    self._save_slot_from_cache(slot_idx, folder_idx)
                else:
                    # Fallback: re-capture frames sequentially (no batch cache available)
                    self._collect_images_for_slot(
                        torch.tensor([slot_idx], device=self.device),
                        folder_idx)
                self.completed_envs.add(self.slot_to_env_id[slot_idx])

        # --- 5. Replenishment ---
        # Replace slots that succeeded OR failed too many times
        self._replenish_slots(valid_slots + exceeded_slots)
        self._reset_called = True
    
    def _ensure_slot_initialization(self) -> None:
        """Lazy initializes slot states and visibility labels on the first run."""
        if hasattr(self, "slot_folder_indices"):
            return

        if self.verbose >= 1:
            print("🔒 Initializing slot states...")

        # Initialize tracking lists
        self.slot_folder_indices = [self.next_env_folder_idx + i for i in range(self.num_envs)]
        self.slot_attempt_counts = [0] * self.num_envs
        
        # Assign initial visibility
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
            Slots that passed validation.
        exceeded_slots : list[int]
            Slots that failed too many times and should be replaced.
        """
        valid_slots = []
        exceeded_slots = []
        min_viewpoints = self.camera_move_target_frames_per_env

        for slot_idx in active_slots:
            env_id = self.slot_to_env_id[slot_idx]
            
            # Skip validation if already marked complete
            if env_id in self.completed_envs:
                continue

            # Validate viewpoint count and occlusion logic
            folder_idx = self.slot_folder_indices[slot_idx]
            is_valid, reason = self._validate_env_state(
                torch.tensor([slot_idx], device=self.device), 
                folder_idx, 
                min_viewpoints
            )

            if is_valid:
                valid_slots.append(slot_idx)
            else:
                self.slot_attempt_counts[slot_idx] += 1
                
                # Check for max attempt failure
                if self.slot_attempt_counts[slot_idx] >= self.max_attempts_per_slot:
                    exceeded_slots.append(slot_idx)
                    if self.verbose >= 1:
                        print(f"  ⚠️ Slot {slot_idx} | Env {env_id} EXCEEDED attempts.")
                elif self.verbose >= 2:
                    print(f"  ❌ Slot {slot_idx} | Env {env_id}: {reason}")
        
        return valid_slots, exceeded_slots

    def _replenish_slots(self, slots_to_replace: list[int]) -> None:
        """
        Advances the environment ID for slots that are either completed or failed.
        
        Updates the slot mapping, folder index, and resets attempt counters.
        """
        if not slots_to_replace:
            return

        for slot_idx in slots_to_replace:
            # Stop if we hit the total simulation limit
            if self.next_env_id >= self.total_envs_to_sim:
                continue

            old_env = self.slot_to_env_id[slot_idx]
            new_env = self.next_env_id
            self.next_env_id += 1
            
            # Update state
            self.slot_to_env_id[slot_idx] = new_env
            self.slot_folder_indices[slot_idx] = self.next_env_folder_idx + new_env
            self.slot_attempt_counts[slot_idx] = 0
            
            # Assign new visibility label
            self.slot_visibility_categories[slot_idx] = self._assign_next_visibility_label(
                self.slot_folder_indices[slot_idx]
            )

            if self.verbose >= 1:
                print(f"  🔄 Slot {slot_idx}: Replaced {old_env} -> {new_env}")

        self._save_visibility_labels()

    def _randomize_scene_props(self, env_ids: list[int]) -> None:
        """
        Randomizes scales, materials, and lights for the specified environments.
        """
        if not env_ids:
            return

        # 1. Randomize VPT Objects (Scale & Material)
        vpt_paths = [
            f"/World/envs/env_{eid}/obs_{oid}" 
            for eid in env_ids 
            for oid in range(self.cfg.num_vpt_objs)
        ]
        
        if vpt_paths:
            self.randomize_shape_scale(prim_path_expr=vpt_paths, is_random=True)
            self.randomize_material(prim_paths=vpt_paths, material_type="vpt")

        # 2. Randomize Floor Material
        floor_paths = [f"/World/envs/env_{i}/mat" for i in env_ids]
        if floor_paths:
            self.randomize_material(prim_paths=floor_paths, material_type="mat")
        
        self.randomize_shape_color(prim_path_expr=[
                "/World/envs/env_.*/bottom_wall",
                "/World/envs/env_.*/right_wall",
                "/World/envs/env_.*/left_wall", "/World/envs/env_.*/top_wall"
            ])

        # 3. Randomize Lights
        light_paths = [f"/World/envs/env_{i}/Light_A" for i in env_ids]
        if light_paths:
            self.randomize_spherical_lights(prim_paths=light_paths)


    def initial_spawn_loop_old(self,
                           env_ids,
                           envs_need_spawn_retry,
                           safe_range: float,
                           states,
                           allow_clipping: bool = False,
                           device=None):
        """
        Procedurally spawns the agent, goal, camera, and VPT objects within the environment.

        This function handles geometric constraints ensuring the camera has a clear 
        view of the goal (initially) and that objects do not overlap if `allow_clipping` is False.

        Parameters
        ----------
        env_ids : torch.Tensor
            Indices of all environments being processed.
        envs_need_spawn_retry : torch.Tensor
            Boolean mask indicating which environments require a spawn attempt.
        safe_range : float
            The +/- range from the origin for placing objects.
        states : list[torch.Tensor]
            List containing mutable state tensors [goal, camera, agent, vpt_objs].
        allow_clipping : bool, optional
            If True, skips expensive overlap checks between objects (faster).
            If False, uses Shapely to ensure no object overlaps (slower, higher quality).
        device : torch.device, optional
            The device for tensor operations.

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor]]
            Updated (envs_need_spawn_retry, states).
        """
        import math
        import random
        import torch
        from shapely.geometry import Point, box
        from shapely import affinity

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

        safe_x_range = safe_range - 4.0
        safe_x_range_obstacles = float(safe_range - 3.0)
        env_origins = self.scene.env_origins[global_retry_env_ids]

        # --- 1. Initial Sampling (Goal, Camera, Agent) ---
        goal_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        camera_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        goal_perturb_offsets = sample_uniform(-2, 2, (batch_size, 2), device)

        # Apply initial positions relative to environment origins
        goal_default_state[retry_indices, 0] = env_origins[:, 0] + goal_offsets[:, 0]
        goal_default_state[retry_indices, 1] = env_origins[:, 1] + goal_offsets[:, 1]
        goal_default_state[retry_indices, 2] = env_origins[:, 2]

        camera_obj_default_state[retry_indices, 0] = env_origins[:, 0] + camera_offsets[:, 0]
        camera_obj_default_state[retry_indices, 1] = env_origins[:, 1] + camera_offsets[:, 1]

        # --- 2. Enforce Camera-Goal Distance Constraints ---
        # Resample until camera is 4.5m - 18.0m away from the goal
        max_dist_retries = 20
        for _ in range(max_dist_retries):
            cam_pos_subset = camera_obj_default_state[retry_indices, :2]
            goal_pos_subset = goal_default_state[retry_indices, :2]
            dists = torch.norm(cam_pos_subset - goal_pos_subset, dim=1)

            bad_mask = (dists < 3.5) | (dists > 15.0)
            if not bad_mask.any():
                break

            # Resample only the invalid entries
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

        # --- 3. Orientation & Final Setup ---
        # Point camera at goal
        direction_to_goal = goal_default_state[retry_indices, :2] - camera_obj_default_state[retry_indices, :2]
        yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0]) - math.radians(90)
        
        # Apply fixed pitch and calculated yaw
        roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
        zero = torch.zeros_like(yaw)
        quaternion = quat_from_euler_xyz(roll, zero, yaw)
        camera_obj_default_state[retry_indices, 3:7] = quaternion

        # Apply minor perturbation to goal and set agent position
        goal_default_state[retry_indices, 0] += goal_perturb_offsets[:, 0]
        goal_default_state[retry_indices, 1] += goal_perturb_offsets[:, 1]

        agent_default_state[retry_indices, 0] = env_origins[:, 0] + agent_offsets[:, 0]
        agent_default_state[retry_indices, 1] = env_origins[:, 1] + agent_offsets[:, 1]

        # --- 4. VPT Object Scattering Loop ---
        def create_rotated_rect(x, y, w, l, yaw_rad):
            """Helper to create a Shapely polygon for collision checks."""
            poly = box(-w / 2.0, -l / 2.0, w / 2.0, l / 2.0)
            poly = affinity.rotate(poly, yaw_rad, use_radians=True)
            poly = affinity.translate(poly, x, y)
            return poly

        MARGIN = 0.1
        MAX_ATTEMPTS = 50
        NUM_CANDIDATES = 20

        # Iterate over each environment in the retry batch
        for batch_idx, local_idx in enumerate(retry_indices):

            global_env_id = env_ids[local_idx]
            global_env_id_item = global_env_id.item() if torch.is_tensor(global_env_id) else global_env_id

            # Cache global coordinates for this env
            cam_global_x = camera_obj_default_state[local_idx, 0].item()
            cam_global_y = camera_obj_default_state[local_idx, 1].item()
            goal_global_x = goal_default_state[local_idx, 0].item()
            goal_global_y = goal_default_state[local_idx, 1].item()

            origin_x = env_origins[batch_idx, 0].item()
            origin_y = env_origins[batch_idx, 1].item()

            # Create Shapely Points for distance checks
            cam_local_p = Point(cam_global_x - origin_x, cam_global_y - origin_y)
            goal_local_p = Point(goal_global_x - origin_x, goal_global_y - origin_y)

            active_indices = self.active_vpt_indices[global_env_id_item]
            active_dims = self.all_vpt_dims[global_env_id, active_indices, :3]

            placed_polys = []
            placement_failed = False

            # Place each active object
            for k, obj_idx in enumerate(active_indices):
                obj_w = active_dims[k, 0].item()
                obj_l = active_dims[k, 1].item()
                
                # Collision dimensions
                coll_w = obj_w + MARGIN
                coll_l = obj_l + MARGIN

                found = False

                for _ in range(MAX_ATTEMPTS):

                    if allow_clipping:
                        # --- FAST PATH: Random Placement (Clipping Allowed) ---
                        rx = (random.random() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
                        ry = (random.random() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
                        r_yaw = random.random() * 2 * math.pi

                        # Quick distance check (Euclidean)
                        dx_cam = rx - (cam_global_x - origin_x)
                        dy_cam = ry - (cam_global_y - origin_y)
                        dist_cam = math.sqrt(dx_cam**2 + dy_cam**2)

                        dx_goal = rx - (goal_global_x - origin_x)
                        dy_goal = ry - (goal_global_y - origin_y)
                        dist_goal = math.sqrt(dx_goal**2 + dy_goal**2)

                        if dist_cam < 5.5 or dist_goal < (self.goal_radius + 2.0 + 0.05):
                            continue

                        # Apply State
                        cand_global_x = origin_x + rx
                        cand_global_y = origin_y + ry
                        vpt_obj_default_state[local_idx, obj_idx, 0] = cand_global_x
                        vpt_obj_default_state[local_idx, obj_idx, 1] = cand_global_y

                        r_yaw_tensor = torch.tensor(r_yaw, device=device)
                        quat = quat_from_euler_xyz(torch.tensor(0.0, device=device), 
                                                   torch.tensor(0.0, device=device), 
                                                   r_yaw_tensor)
                        vpt_obj_default_state[local_idx, obj_idx, 3:7] = quat

                        found = True
                        break

                    else:
                        # --- ROBUST PATH: No Clipping (Shapely + Heuristics) ---
                        best_candidate = None
                        max_isolation_dist = -1.0

                        # A. Generate candidates and pick the one maximizing isolation
                        candidates = []
                        for _ in range(NUM_CANDIDATES):
                            raw_rx = (random.random() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
                            raw_ry = (random.random() * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
                            r_yaw = random.random() * 2 * math.pi
                            candidates.append((float(raw_rx), float(raw_ry), float(r_yaw)))

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

                        # B. Validate the Best Candidate
                        rx, ry, r_yaw = best_candidate
                        collision_poly = create_rotated_rect(rx, ry, coll_w, coll_l, r_yaw)

                        # Distance Checks
                        if collision_poly.distance(cam_local_p) < 3.0:
                            continue
                        
                        if collision_poly.distance(goal_local_p) < (self.goal_radius + 0.1):
                            continue
                        
                        # Boundary Checks
                        minx, miny, maxx, maxy = collision_poly.bounds
                        if (minx < -self.center_to_boundary or miny < -self.center_to_boundary or 
                                maxx > self.center_to_boundary or maxy > self.center_to_boundary):
                            continue

                        # Overlap Checks
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

                            r_yaw_tensor = torch.tensor(r_yaw, device=device)
                            quat = quat_from_euler_xyz(torch.tensor(0.0, device=device), 
                                                       torch.tensor(0.0, device=device), 
                                                       r_yaw_tensor)
                            vpt_obj_default_state[local_idx, obj_idx, 3:7] = quat

                            found = True
                            break

                if not found:
                    placement_failed = True
                    break

            # Mark environment as successfully spawned if no failures occurred
            if not placement_failed:
                envs_need_spawn_retry[local_idx] = False

        # Move unused objects to storage
        vpt_obj_default_state[retry_indices] = self._store_inactive_vpt_objects(
            env_ids[retry_indices], vpt_obj_default_state[retry_indices])

        return envs_need_spawn_retry, [
            goal_default_state, camera_obj_default_state, agent_default_state, vpt_obj_default_state
        ]

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

        safe_x_range = safe_range - 4.0
        safe_x_range_obstacles = float(safe_range - 3.0)
        env_origins = self.scene.env_origins[global_retry_env_ids]

        # ==========================================================
        # 1. SCATTER VPT OBSTACLES (Vectorized)
        # ==========================================================
        num_vpt_objs = vpt_obj_default_state.shape[1]

        rxs = (torch.rand((batch_size, num_vpt_objs), device=device) * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
        rys = (torch.rand((batch_size, num_vpt_objs), device=device) * 2 * safe_x_range_obstacles) - safe_x_range_obstacles
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
        # 2. PLACE GOAL (Fix 7: OBB collision check via shotgun)
        # ==========================================================
        goal_default_state[retry_indices] = self.place_object_safely(
            env_ids=global_retry_env_ids,
            object_state=goal_default_state[retry_indices],
            vpt_state=vpt_obj_default_state[retry_indices],
            safe_range=float(safe_x_range),
            object_type='goal'
        )

        # ==========================================================
        # 3. PLACE CAMERA (Fix 7: OBB collision check via shotgun)
        # ==========================================================
        camera_obj_default_state[retry_indices] = self.place_object_safely(
            env_ids=global_retry_env_ids,
            object_state=camera_obj_default_state[retry_indices],
            vpt_state=vpt_obj_default_state[retry_indices],
            safe_range=float(safe_x_range),
            object_type='cam_obj'
        )

        # ==========================================================
        # 4. ENFORCE CAMERA-GOAL DISTANCE [3.5, 15.0]
        # ==========================================================
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

            # Re-roll only the failed cameras (still with OBB check)
            camera_obj_default_state[bad_local_indices] = self.place_object_safely(
                env_ids=bad_global_env_ids,
                object_state=camera_obj_default_state[bad_local_indices],
                vpt_state=vpt_obj_default_state[bad_local_indices],
                safe_range=float(safe_x_range),
                object_type='cam_obj'
            )

        # ==========================================================
        # 5. PERTURB GOAL (after camera is finalized so orientation
        #    is computed from the perturbed position)
        # ==========================================================
        goal_perturb_offsets = sample_uniform(-2, 2, (batch_size, 2), device)
        goal_default_state[retry_indices, 0] += goal_perturb_offsets[:, 0]
        goal_default_state[retry_indices, 1] += goal_perturb_offsets[:, 1]

        # ==========================================================
        # 6. PLACE AGENT (random, no orientation constraint yet)
        # ==========================================================
        agent_offsets = sample_uniform(-safe_x_range, safe_x_range, (batch_size, 2), device)
        agent_default_state[retry_indices, 0] = env_origins[:, 0] + agent_offsets[:, 0]
        agent_default_state[retry_indices, 1] = env_origins[:, 1] + agent_offsets[:, 1]

        # ==========================================================
        # 7. ORIENT CAMERA TOWARD GOAL WITH YAW JITTER (Fix 2)
        # ==========================================================
        direction_to_goal = (goal_default_state[retry_indices, :2] -
                            camera_obj_default_state[retry_indices, :2])
        exact_yaw = (torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
                    - math.radians(90))

        # Jitter within 80% of half-FOV (24 deg) — goal stays in frame
        # but is no longer always centered
        half_fov_rad = math.radians(30 * 0.8)
        yaw_jitter = sample_uniform(-half_fov_rad, half_fov_rad, (batch_size,), device)
        yaw = exact_yaw + yaw_jitter

        roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
        zero = torch.zeros_like(yaw)
        camera_obj_default_state[retry_indices, 3:7] = quat_from_euler_xyz(roll, zero, yaw)

        # ==========================================================
        # 8. MARK ALL AS SPAWNED
        # ==========================================================
        envs_need_spawn_retry[retry_indices] = False

        return envs_need_spawn_retry, [
            goal_default_state, camera_obj_default_state,
            agent_default_state, vpt_obj_default_state
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
                         env_ids: torch.Tensor,
                         valid_indices: torch.Tensor,
                         visibility_categories: list[str],
                         moved_vpt_for_ball: list[int | None],
                         states: list[torch.Tensor],
                         in_view_displaced: torch.Tensor | None = None,
                         outside_fov_displaced: torch.Tensor | None = None,
                         device: torch.device | None = None) -> list[torch.Tensor]:
        """
        Displaces random VPT objects to interpose between the camera and the goal.

        Purpose:
        - "occluded": Displaces objects to intentionally block the goal view.
        - "in_view" / "outside_fov": Displaces objects probabilistically to add scene complexity.

        Parameters
        ----------
        env_ids : torch.Tensor
            Global environment IDs.
        valid_indices : torch.Tensor
            Indices of the environments in the current batch to process.
        visibility_categories : list[str]
            Visibility labels ("occluded", "in_view", etc.) for each environment.
        moved_vpt_for_ball : list[int | None]
            Mapping of env_idx to the object ID currently supporting the ball (to avoid moving it).
        states : list[torch.Tensor]
            Mutable list of state tensors [goal, camera, agent, vpt].
        in_view_displaced : torch.Tensor | None, optional
            Subset of indices allowed to move objects in 'in_view' mode.
        outside_fov_displaced : torch.Tensor | None, optional
            Subset of indices allowed to move objects in 'outside_fov' mode.

        Returns
        -------
        list[torch.Tensor]
            The modified states list.
        """
        if device is None:
            device = self._agent.device

        # Unpack mutable state tensors
        goal_state = states[0]
        camera_state = states[1]
        vpt_state = states[3]

        # Optimization: Pre-compute set of indices allowed to move for non-occluded categories
        indices_to_move = set()
        if in_view_displaced is not None:
            indices_to_move.update(in_view_displaced.tolist())
        if outside_fov_displaced is not None:
            indices_to_move.update(outside_fov_displaced.tolist())

        # Iterate through valid environments
        for env_idx in valid_indices:
            env_idx_int = env_idx.item()
            category = visibility_categories[env_idx_int]

            # --- 1. Filter Logic ---
            should_move = (category == "occluded") or (env_idx_int in indices_to_move)
            if not should_move:
                continue

            # --- 2. Select Object to Move ---
            # Exclude the object currently holding the ball (if any) to prevent physics explosions
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

            # --- 3. Calculate New Position (Interposition) ---
            # We want to place the object on the line segment between Camera and Goal
            cam_pos = camera_state[env_idx, :3]
            goal_pos = goal_state[env_idx, :3]

            vec_cam_to_goal = goal_pos[:2] - cam_pos[:2]
            dist = torch.norm(vec_cam_to_goal)

            if dist > 1e-6:
                # Determine interpolation factor (t)
                # 'in_view' generally keeps objects more centered (0.3-0.7)
                t_min, t_max = (0.3, 0.7) if category == "in_view" else (0.2, 0.8)
                t = random.uniform(t_min, t_max)
                
                # Calculate position: Origin + Direction * Distance + Jitter
                # Jitter range: [-0.4, 0.4]
                jitter = (torch.rand(2, device=device) * 0.8) - 0.4
                new_pos = cam_pos[:2] + (vec_cam_to_goal * t) + jitter

                # Update state
                vpt_state[env_idx, target_obj_idx, 0] = new_pos[0]
                vpt_state[env_idx, target_obj_idx, 1] = new_pos[1]

        return states

    # TODO: Use self._update_camera_poses
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

        # 5. Update Sensor
        camera_positions = camera_obj_default_state[valid_indices, :3]
        camera_orientations = camera_obj_default_state[valid_indices, 3:7]

        # Apply 90-degree offset for sensor
        theta_left = math.pi / 2
        half_theta_left = theta_left / 2
        left_90_quat = torch.tensor(
            [math.cos(half_theta_left), 0.0, 0.0,
             math.sin(half_theta_left)],
            device=device)

        rotated_orientations = math_utils.quat_mul(
            camera_orientations,
            left_90_quat.unsqueeze(0).expand(len(valid_env_ids), -1))

        # Normalize sensor quats too
        rotated_orientations = torch.nn.functional.normalize(
            rotated_orientations, p=2, dim=-1)

        self._occlusion_camera.set_world_poses(
            positions=camera_positions,
            orientations=rotated_orientations,
            env_ids=valid_env_ids.tolist(),
            convention="world")

        # 6. Step Simulation
        for _ in range(1):
            self.sim.step()
            self._occlusion_camera.update(self.sim.cfg.dt)

    def check_z_bounds(self,
                       env_ids: torch.Tensor,
                       valid_indices: torch.Tensor,
                       states: list[torch.Tensor],
                       envs_need_spawn_retry: torch.Tensor,
                       tolerance: float = 5e-2) -> torch.Tensor:
        """
        Verifies that entity Z-heights are within acceptable bounds.

        Checks the Goal, Camera, Agent, and active VPT objects. If any entity 
        exceeds the vertical limits, the environment is marked for a spawn retry.

        Parameters
        ----------
        env_ids : torch.Tensor
            Global environment IDs.
        valid_indices : torch.Tensor
            Local indices corresponding to the current batch.
        states : list[torch.Tensor]
            List of state tensors [goal, camera, agent, vpt].
        envs_need_spawn_retry : torch.Tensor
            Current retry mask.
        tolerance : float, optional
            Allowed deviation from the strict Z limits.

        Returns
        -------
        torch.Tensor
            Updated retry mask where True indicates a failure.
        """
        updated_retry_mask = envs_need_spawn_retry.clone()
        
        # Unpack states [Batch, ...]
        goal_pos, camera_pos, agent_pos, vpt_pos = states

        # Iterate only through the valid indices for this batch
        for local_idx, env_idx in enumerate(valid_indices):
            env_id_val = env_ids[env_idx].item()
            failure_reasons = []

            # --- 1. Single Entity Checks ---
            # Goal: [-tol, 1.0 + tol]
            goal_z = goal_pos[local_idx, 2].item()
            if not (-tolerance <= goal_z <= 1.0 + tolerance):
                failure_reasons.append(f"Goal Z: {goal_z:.6f}")

            # Camera: [0.0, 1.0]
            cam_z = camera_pos[local_idx, 2].item()
            if not (0.0 <= cam_z <= 1.0):
                failure_reasons.append(f"Camera Z: {cam_z:.4f}")

            # Agent: [0.0, 1.0]
            agent_z = agent_pos[local_idx, 2].item()
            if not (0.0 <= agent_z <= 1.0):
                failure_reasons.append(f"Agent Z: {agent_z:.4f}")

            # --- 2. VPT Object Checks ---
            # Retrieve active objects and their height adjustments
            active_indices = self.active_vpt_indices[env_id_val]
            
            # Get raw Z and apply offsets to check "base" height
            raw_z = vpt_pos[local_idx, :, 2]
            offsets = self.vpt_z_offset_ratios[env_id_val, active_indices]
            adjusted_z = raw_z * offsets

            # VPT Limits: [-tol, 0.1 + tol]
            valid_obj_mask = (adjusted_z >= -tolerance) & (adjusted_z <= 0.1 + tolerance)

            if not torch.all(valid_obj_mask):
                failed_indices = torch.where(~valid_obj_mask)[0]
                for idx in failed_indices:
                    global_id = active_indices[idx].item()
                    bad_z = adjusted_z[idx].item()
                    failure_reasons.append(f"VPT Obj {global_id} Z: {bad_z:.6f}")

            # --- 3. Update Mask & Report ---
            if failure_reasons:
                print(f"⚠️ Env {env_id_val} Z-Check Failed:")
                for reason in failure_reasons:
                    print(f"   - {reason}")
                updated_retry_mask[env_idx] = True

        return updated_retry_mask

    # TODO: Clean up later
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

                is_occluded = self._check_occlusion(
                    camera_pos, goal_pos, env_id)

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

    def geometric_occlusion_check(self, 
                                  env_ids: torch.Tensor, 
                                  valid_indices: torch.Tensor,
                                  occlusion_valid_mask: torch.Tensor, 
                                  envs_need_spawn_retry: torch.Tensor, 
                                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Validates that environments have enough geometrically valid viewpoints around the goal.

        Generates a circle of points around the goal (based on FOV and distance) and checks
        collision/boundary validity for each point. If an environment has fewer than 
        `MIN_GEOMETRIC_VALID_POINTS`, it is marked for a spawn retry.

        Parameters
        ----------
        env_ids : torch.Tensor
            Global environment IDs.
        valid_indices : torch.Tensor
            Local indices corresponding to the current batch.
        occlusion_valid_mask : torch.Tensor
            Boolean mask of environments that have already passed the occlusion check.
        envs_need_spawn_retry : torch.Tensor
            Current retry mask (modified in-place).
        device : torch.device
            Device for tensor operations.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (geometric_valid_mask, updated_envs_need_spawn_retry)
        """
        # Constants
        FOV_DEG = 30.0
        MIN_GEOMETRIC_VALID_POINTS = 40
        NUM_ANGLES = 180  # 360 / 2

        # Start with the mask of envs that passed previous checks
        geometric_valid_mask = occlusion_valid_mask.clone()
        passed_env_ids = env_ids[occlusion_valid_mask]

        if passed_env_ids.numel() == 0:
            return geometric_valid_mask, envs_need_spawn_retry

        num_envs = len(passed_env_ids)

        # --- 1. Calculate Viewing Circle Radius ---
        # Radius depends on distance to goal to maintain constant visual size
        cam_pos = self._camera_obj.data.root_pos_w[passed_env_ids, :2]
        goal_pos = self._goal.data.root_pos_w[passed_env_ids, :2]
        
        dist_to_goal = torch.norm(cam_pos - goal_pos, dim=1)
        half_fov = torch.tensor(math.radians(FOV_DEG) / 2, device=device)
        
        # r = (d / 2) / tan(fov/2) * 1.2 (safety margin)
        radii = ((dist_to_goal / 2) / torch.tan(half_fov)) * 1.2
        radii = radii.unsqueeze(1)  # [num_envs, 1]

        # --- 2. Generate Candidate Points (Vectorized) ---
        angles = torch.linspace(0, 2 * math.pi, NUM_ANGLES, device=device)
        angles_expanded = angles.unsqueeze(0).expand(num_envs, -1)  # [num_envs, num_angles]

        # Calculate X, Y for all points
        # shape: [num_envs, num_angles]
        circle_x = goal_pos[:, 0].unsqueeze(1) + radii * torch.cos(angles_expanded)
        circle_y = goal_pos[:, 1].unsqueeze(1) + radii * torch.sin(angles_expanded)

        # Flatten for batch validation
        total_points = num_envs * NUM_ANGLES
        flat_points = torch.stack([circle_x, circle_y], dim=2).reshape(total_points, 2)
        flat_env_ids = passed_env_ids.unsqueeze(1).expand(-1, NUM_ANGLES).reshape(total_points)

        # --- 3. Validate Points ---
        # Batch check against boundaries and obstacles
        is_valid_flat = self._is_point_valid_batch(
            points=flat_points,
            env_ids=flat_env_ids,
            check_agent_fov=False
        )
        
        # Reshape back to [num_envs, num_angles] count valid points per env
        valid_counts = is_valid_flat.reshape(num_envs, NUM_ANGLES).sum(dim=1)

        # --- 4. Update Status ---
        for i, global_env_id in enumerate(passed_env_ids):
            count = valid_counts[i].item()
            
            # Find local index in the original batch
            local_idx = (env_ids == global_env_id).nonzero(as_tuple=True)[0].item()
            batch_idx = valid_indices[local_idx]

            if count < MIN_GEOMETRIC_VALID_POINTS:
                # print(
                #     f"    ❌ Env {global_env_id.item()}: Geometric viewpoint check FAILED ({count}/{MIN_GEOMETRIC_VALID_POINTS} valid points)"
                # )
                envs_need_spawn_retry[batch_idx] = True
                geometric_valid_mask[local_idx] = False
            else:
                if self.verbose >= 2:
                    # print(
                    #     f"    ✅ Env {global_env_id.item()}: Geometric viewpoint check PASSED ({count}/{MIN_GEOMETRIC_VALID_POINTS} valid points)"
                    # )
                    pass

        return geometric_valid_mask, envs_need_spawn_retry

    def camera_pov_validation(self, 
                              env_ids: torch.Tensor, 
                              valid_indices: torch.Tensor,
                              geometric_valid_mask: torch.Tensor, 
                              visibility_categories: list[str],
                              envs_need_spawn_retry: torch.Tensor, 
                              folder_indices: list[int],
                              spawn_attempt: int) -> torch.Tensor:
        """
        Validates the camera view against the expected visibility category.

        Checks if the target (red goal) is visible in the camera image.
        - "in_view": Target MUST be visible.
        - "occluded" / "outside_fov": Target MUST NOT be visible.

        Parameters
        ----------
        env_ids : torch.Tensor
            Global environment IDs.
        valid_indices : torch.Tensor
            Local indices corresponding to the current batch.
        geometric_valid_mask : torch.Tensor
            Mask of environments that passed the previous geometric check.
        visibility_categories : list[str]
            Expected visibility labels for the current batch.
        envs_need_spawn_retry : torch.Tensor
            Current retry mask (modified in-place).
        folder_indices : list[int]
            Folder indices for debugging filenames.
        spawn_attempt : int
            Current spawn attempt counter for debugging filenames.

        Returns
        -------
        torch.Tensor
            Updated retry mask.
        """
        # Ensure debug directory exists once
        debug_folder = os.path.join(self.base_path, "debug_camera_pov")
        os.makedirs(debug_folder, exist_ok=True)

        for local_idx, env_idx in enumerate(valid_indices):
            # Skip if failed geometric check
            if not geometric_valid_mask[local_idx]:
                continue

            # Context Setup
            env_id = env_ids[local_idx]
            env_id_val = env_id.item()
            category = visibility_categories[env_idx]
            folder_idx = folder_indices[env_idx]

            # --- 1. Image Processing ---
            # Extract and normalize semantic image [H, W, 3]
            sem_img = self._occlusion_camera.data.output["semantic_segmentation"][env_id]
            cam_pov_img = sem_img[..., :3]
            
            # Handle float (0-1) vs byte (0-255) conversion safely
            if cam_pov_img.max() <= 1.0:
                cam_pov_np = (cam_pov_img.cpu().numpy() * 255.0).astype(np.uint8)
            else:
                cam_pov_np = cam_pov_img.cpu().numpy().astype(np.uint8)

            debug_filename = os.path.join(
                debug_folder, 
                f"env_{env_id_val}_folder_{folder_idx}_attempt_{spawn_attempt}.png"
            )

            # --- 2. Check Target Visibility ---
            target_visible, red_count = self._check_target_in_img(
                file_name=debug_filename,
                cam_pov=cam_pov_np,
                return_red_count=True
            )

            # --- 3. Validation Logic ---
            # "in_view" expects visibility=True. Others expect visibility=False.
            expected_visible = (category == "in_view")
            is_valid = (target_visible == expected_visible)

            # --- 4. Reporting & Retry ---
            if is_valid:
                if self.verbose >= 2:
                    print(f"    ✅ Env {env_id_val}: Camera valid ({category}) | Red: {red_count}")
            else:
                # Mark failure
                envs_need_spawn_retry[env_idx] = True
                
                if self.verbose >= 1:
                    status = "visible" if target_visible else "NOT visible"
                    print(f"    ❌ Env {env_id_val}: Camera check FAILED")
                    print(f"       Expected: {category}, Got: {status} (Red: {red_count})")
                    print(f"       Debug: {debug_filename}")

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

        # This variant stores mixed labels per scene, but the inherited spawn
        # pipeline expects a single camera-goal category. Use an in-view seed
        # pose, then build the mixed visible/occluded camera circle explicitly.
        scene_visibility_categories = ["in_view"] * num_envs

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
            i for i in range(num_envs) if scene_visibility_categories[i] == "in_view"
        ]
        rand_iv = torch.randperm(len(in_view_indices))[:len(in_view_indices) //
                                                       2]
        in_view_displaced = torch.tensor(
            in_view_indices,
            device=device)[rand_iv] if in_view_indices else torch.tensor(
                [], device=device)

        outside_fov_indices = [
            i for i in range(num_envs)
            if scene_visibility_categories[i] == "outside_fov"
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
                allow_clipping=False,
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
                visibility_categories=scene_visibility_categories,
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
                visibility_categories=scene_visibility_categories,
                states=[goal_state, camera_state, agent_state, vpt_state],
                device=device,
            )

            # timer.stop_timer("camera_posing_time", spawn_attempt, envs_need_spawn_retry)

            # --- E. Occlusion Validation (Raycast) ---
            # timer.start_timer("occlusion_raycast_time")

            occlusion_valid_mask, envs_need_spawn_retry, _, states = self.occlusion_validation_check(
                final_valid_env_ids=final_valid_env_ids,
                valid_indices=final_valid_indices,
                visibility_categories=scene_visibility_categories,
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
                    visibility_categories=scene_visibility_categories,
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
        agent_state = self.place_object_safely(
            env_ids=env_ids,
            object_state=agent_state,
            vpt_state=vpt_state,
            safe_range=self.center_to_boundary - 1.0,
            object_type='agent')

        if self.camera_move_debug_obb:
            self.debug_plot_analytic_obbs(vpt_state)

        # 2. Final Write
        # Push the finalized, safe states to the physics engine.
        self.write_pose_to_sim(env_ids=env_ids,
                               indices=torch.arange(len(env_ids),
                                                    device=device),
                               vpt_obj_default_state=vpt_state,
                               agent_default_state=agent_state)

        # --- H. Fixed-agent placement + camera circle generation (SKIPPED IF RL_RESET) ---
        if not rl_reset:
            # timer.start_timer("circle_validation_time")

            success_mask = ~envs_need_spawn_retry
            successful_env_ids = env_ids[success_mask]

            fixed_agent_points = []
            if len(successful_env_ids) > 0:
                fixed_agent_points = self.generate_valid_circle_points(
                    env_ids=successful_env_ids,
                    angle_step=2.0,
                    max_attempts=100)

            if self.valid_viewpoint_poses is None:
                self.valid_viewpoint_poses = [None] * self.num_envs

            # Clear failed envs and stale camera-move state.
            failed_env_ids = env_ids[envs_need_spawn_retry]
            for env_id in failed_env_ids:
                eid = env_id.item() if torch.is_tensor(env_id) else env_id
                self.valid_viewpoint_poses[eid] = torch.zeros((0, 3),
                                                              device=device)
                self.camera_move_trajectories[eid] = None
                self.camera_circle_radii[eid] = None

            # Phase A: place each agent sequentially (different pose per env), validate view.
            # Phase B: batch sweep all ready envs simultaneously (the expensive part).
            ready_eids: list[int] = []
            for i, env_id in enumerate(successful_env_ids):
                eid = env_id.item() if torch.is_tensor(env_id) else env_id
                points_2d = fixed_agent_points[i]

                if points_2d.shape[0] == 0:
                    self.valid_viewpoint_poses[eid] = torch.zeros((0, 3), device=device)
                    self.camera_move_trajectories[eid] = None
                    envs_need_spawn_retry[torch.where(env_ids == env_id)[0][0]] = True
                    continue

                agent_z = self._agent.data.default_root_state[env_id, 2]
                fixed_agent_point = torch.stack([points_2d[0, 0], points_2d[0, 1], agent_z])
                self._set_fixed_agent_pose_from_point(env_id, fixed_agent_point)

                if not self._validate_fixed_agent_view(env_id):
                    self.valid_viewpoint_poses[eid] = torch.zeros((0, 3), device=device)
                    self.camera_move_trajectories[eid] = None
                    envs_need_spawn_retry[torch.where(env_ids == env_id)[0][0]] = True
                    continue

                self.valid_viewpoint_poses[eid] = fixed_agent_point.unsqueeze(0)
                ready_eids.append(eid)

            # Phase B: batch camera sweep for all ready envs
            if ready_eids:
                success_map = self._build_fixed_sweep_trajectory_batch(ready_eids)
                for eid in ready_eids:
                    if not success_map[eid]:
                        self.valid_viewpoint_poses[eid] = torch.zeros((0, 3), device=device)
                        idx = torch.where(env_ids == torch.tensor(eid, device=device))[0]
                        if idx.numel():
                            envs_need_spawn_retry[idx[0]] = True
                    else:
                        # Restore camera to first-frame position for initial render
                        first_frame = self.camera_move_trajectories[eid][0]
                        self._set_camera_object_pose(
                            eid, first_frame["position"], self._goal.data.root_pos_w[eid])

            # timer.stop_timer("circle_validation_time", spawn_attempt, envs_need_spawn_retry)

            # Final update to catch any completion times
            timer.update_status(spawn_attempt, envs_need_spawn_retry)

        # timer.print_summary(spawn_attempt)

    def _compute_agent_look_at_midpoint_quat(self, agent_xy: torch.Tensor, env_id: int | torch.Tensor) -> torch.Tensor:
        """Return a no-jitter agent yaw quaternion that faces the camera/goal midpoint."""
        if torch.is_tensor(env_id):
            env_id = env_id.item()
        cam_pos_2d = self._camera_obj.data.root_pos_w[env_id, :2]
        goal_pos_2d = self._goal.data.root_pos_w[env_id, :2]
        midpoint = (cam_pos_2d + goal_pos_2d) / 2.0
        direction = midpoint - agent_xy
        yaw = torch.atan2(direction[1], direction[0])
        half_yaw = yaw / 2.0
        return torch.stack([
            torch.cos(half_yaw),
            torch.zeros((), device=self.device),
            torch.zeros((), device=self.device),
            torch.sin(half_yaw),
        ])

    def _set_fixed_agent_pose_from_point(self, env_id: int | torch.Tensor, point_3d: torch.Tensor) -> None:
        """Fix the agent at one validated viewpoint with no yaw jitter."""
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        env_tensor = torch.tensor([env_id_item], dtype=torch.long, device=self.device)
        quat = self._compute_agent_look_at_midpoint_quat(point_3d[:2], env_id_item)
        pose = torch.cat([point_3d[:3], quat], dim=0).unsqueeze(0)
        self._agent.write_root_com_pose_to_sim(pose, env_tensor)
        self._agent.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_tensor)
        for _ in range(5):
            self.sim.step()
            self._rgb_tiled_camera.update(self.sim.cfg.dt)
            self._agent.update(self.sim.cfg.dt)

    def _validate_fixed_agent_view(self, env_id: int | torch.Tensor) -> bool:
        """Require the fixed agent view to see both goal and camera object."""
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        self._rgb_tiled_camera.update(self.sim.cfg.dt)
        goal_visible, camera_visible = self.check_batch_object_visibility(
            torch.tensor([env_id_item], dtype=torch.long, device=self.device))
        return bool((goal_visible[0] & camera_visible[0]).item())

    def _compute_camera_look_at_goal_quat(self, camera_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
        """Return camera object quaternion whose corrected sensor yaw points at the goal."""
        direction_to_goal = goal_pos[:2] - camera_pos[:2]
        yaw = torch.atan2(direction_to_goal[1], direction_to_goal[0]) - math.radians(90.0)
        roll = torch.tensor(-math.radians(self.agent_camera_pitch), device=self.device)
        pitch = torch.zeros((), device=self.device)
        return quat_from_euler_xyz(roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0))[0]

    def _compute_camera_look_at_goal_quat_batch(
            self, camera_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
        """Vectorized look-at quat. camera_pos/goal_pos: (N, 3) → (N, 4)."""
        direction = goal_pos[:, :2] - camera_pos[:, :2]
        yaw = torch.atan2(direction[:, 1], direction[:, 0]) - math.radians(90.0)
        n = len(camera_pos)
        roll = torch.full((n,), -math.radians(self.agent_camera_pitch),
                          device=self.device, dtype=torch.float32)
        pitch = torch.zeros(n, device=self.device, dtype=torch.float32)
        return quat_from_euler_xyz(roll, pitch, yaw)  # (N, 4)

    def _set_camera_object_pose(self, env_id: int | torch.Tensor, camera_xy: torch.Tensor, goal_pos: torch.Tensor) -> float:
        """Move the camera object to `camera_xy`, point it at the goal, and update camera POV sensor."""
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        env_tensor = torch.tensor([env_id_item], dtype=torch.long, device=self.device)
        camera_pos = self._camera_obj.data.root_pos_w[env_id_item].clone()
        camera_pos[:2] = camera_xy
        quat = self._compute_camera_look_at_goal_quat(camera_pos, goal_pos)
        pose = torch.cat([camera_pos, quat], dim=0).unsqueeze(0)

        self._camera_obj.write_root_com_pose_to_sim(pose, env_tensor)
        self._camera_obj.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_tensor)

        left_90_quat = torch.tensor(
            [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
            device=self.device)
        sensor_quat = math_utils.quat_mul(quat.unsqueeze(0), left_90_quat.unsqueeze(0))
        sensor_quat = torch.nn.functional.normalize(sensor_quat, p=2, dim=-1)
        self._occlusion_camera.set_world_poses(
            positions=camera_pos.unsqueeze(0),
            orientations=sensor_quat,
            env_ids=[env_id_item],
            convention="world")

        for _ in range(5):
            self.sim.step()
            self._camera_obj.update(self.sim.cfg.dt)
            self._rgb_tiled_camera.update(self.sim.cfg.dt)
            if self.save_camera_pov:
                self._occlusion_camera.update(self.sim.cfg.dt)

        yaw = self._quat_wxyz_to_yaw(quat)
        return float(yaw)

    def _build_fixed_sweep_trajectory(self, env_id: int | torch.Tensor) -> bool:
        """
        Build camera trajectory by sweeping 0°, 15°, ..., 180° around the goal, measured
        from the goal→agent direction. Each valid position is captured and labeled Yes/No.

        Angle convention (all relative to the goal→agent line):
          - 0°   : camera collinear with agent and goal, on the agent's side — camera
                   looks toward the goal in the SAME direction the agent looks.
          - 90°  : camera on the agent's RIGHT side of the circle.
          - 180° : camera collinear, diametrically opposite the agent across the goal.
        Positive angle sweeps the agent's right half.

        Radius = min(initial_camera_to_goal dist, agent_to_goal dist) so the camera
        circle never places the camera behind the agent.

        The 90° left correction is applied via _set_camera_object_pose, which calls
        _compute_camera_look_at_goal_quat and sets the occlusion_camera sensor separately
        with a left_90_quat applied — same as the existing convention.

        # TODO: enforce global 50/50 in_view/occluded balance across the dataset.
        # For now this is a dry-run: collect all safe frames regardless of label split.
        """
        eid = env_id.item() if torch.is_tensor(env_id) else env_id
        env_tensor = torch.tensor([eid], dtype=torch.long, device=self.device)

        goal_pos = self._goal.data.root_pos_w[eid]
        goal_xy = goal_pos[:2]
        agent_xy = self._agent.data.root_pos_w[eid, :2]
        camera_xy_init = self._camera_obj.data.root_pos_w[eid, :2]
        camera_z = float(self._camera_obj.data.root_pos_w[eid, 2].item())

        cam_goal_dist = float(torch.norm(camera_xy_init - goal_xy).item())
        agent_goal_dist = float(torch.norm(agent_xy - goal_xy).item())
        radius = min(cam_goal_dist, agent_goal_dist)
        self.camera_circle_radii[eid] = radius

        agent_yaw = self._get_agent_yaw(eid)
        # Sweep is measured from the goal→agent direction: angle 0 puts the camera
        # on the agent's side (collinear), 180 diametrically opposite.
        base_angle = math.atan2(
            float(agent_xy[1]) - float(goal_xy[1]),
            float(agent_xy[0]) - float(goal_xy[0]))
        env_origin = self.scene.env_origins[eid, :2]
        bound = float(self.center_to_boundary)

        trajectory = []
        for angle_deg in self.camera_sweep_angles_deg:
            world_angle = base_angle + math.radians(angle_deg)
            cam_x = float(goal_xy[0]) + radius * math.cos(world_angle)
            cam_y = float(goal_xy[1]) + radius * math.sin(world_angle)
            camera_pos_2d = torch.tensor([cam_x, cam_y], device=self.device, dtype=torch.float32)

            # --- Bounds check ---
            if not (
                float(env_origin[0]) - bound <= cam_x <= float(env_origin[0]) + bound and
                float(env_origin[1]) - bound <= cam_y <= float(env_origin[1]) + bound
            ):
                if self.verbose >= 2:
                    print(f"    Sweep {angle_deg}°: out of bounds, skip")
                continue

            # --- Agent FOV check (geometric, no sim step) ---
            # Ensure camera CENTER is within agent's horizontal FOV minus buffer,
            # so the camera object doesn't clip the frame edge.
            dx = cam_x - float(agent_xy[0])
            dy = cam_y - float(agent_xy[1])
            angle_to_cam = math.atan2(dy, dx)
            rel_deg = math.degrees(
                math.atan2(math.sin(angle_to_cam - agent_yaw),
                           math.cos(angle_to_cam - agent_yaw))
            )
            max_allowed = self._agent_half_hfov_deg - self._agent_fov_buffer_deg
            if abs(rel_deg) > max_allowed:
                if self.verbose >= 2:
                    print(f"    Sweep {angle_deg}°: camera center outside agent FOV "
                          f"(rel={rel_deg:.1f}° > ±{max_allowed:.1f}°), skip")
                continue

            # --- OBB collision check (uses same logic as place_object_safely) ---
            camera_pos_3d = torch.tensor(
                [cam_x, cam_y, camera_z], device=self.device, dtype=torch.float32)
            quat = self._compute_camera_look_at_goal_quat(camera_pos_3d, goal_pos)
            is_collision = self._check_collisions_vectorized(
                env_tensor,
                camera_pos_3d.unsqueeze(0),
                quat.unsqueeze(0),
                object_type='cam_obj',
            )
            if bool(is_collision[0]):
                if self.verbose >= 2:
                    print(f"    Sweep {angle_deg}°: OBB collision, skip")
                continue

            # --- Place camera (applies 90° left sensor correction internally) ---
            yaw = self._set_camera_object_pose(eid, camera_pos_2d, goal_pos)

            # Require agent still sees both goal and camera object
            if not self._validate_fixed_agent_view(eid):
                if self.verbose >= 2:
                    print(f"    Sweep {angle_deg}°: agent view invalid, skip")
                continue

            visible, red_count = self._camera_goal_visible_from_current_pov(eid)
            trajectory.append({
                "angle_deg": angle_deg,
                "position": camera_pos_2d.clone(),
                "yaw": yaw,
                "label": "Yes" if visible else "No",
                "reason": "in_view" if visible else "occluded",
                "red_count": red_count,
            })
            if self.verbose >= 2:
                status = "✅ Yes" if visible else "❌ No"
                print(f"    Sweep {angle_deg}°: {status} (red_px={red_count})")

        trajectory, _ = self._select_balanced_camera_move_frames(trajectory)

        if not trajectory:
            self.camera_move_trajectories[eid] = None
            return False

        self.camera_move_trajectories[eid] = trajectory
        self.selected_viewpoints_for_collection[eid] = torch.stack(
            [item["position"] for item in trajectory])
        return True

    def _build_fixed_sweep_trajectory_batch(self, eids: list[int]) -> dict[int, bool]:
        """
        Batch version of _build_fixed_sweep_trajectory.

        Processes all eids simultaneously for each sweep angle:
          - vectorized bounds, FOV, and OBB checks
          - single batch of sim steps per angle (30) instead of per-env per-angle
          - reads and caches image tensors into self._cached_frames[eid]
            so _collect_images_for_slot can skip re-capturing

        Speedup vs sequential: ~num_envs× (e.g. 8× for num_envs=8).
        Returns {eid: success_bool}.
        """
        if not eids:
            return {}

        slots = torch.tensor(eids, dtype=torch.long, device=self.device)
        n = len(eids)

        goal_pos = self._goal.data.root_pos_w[slots].clone()           # (n, 3)
        goal_xy  = goal_pos[:, :2]                                      # (n, 2)
        agent_xy = self._agent.data.root_pos_w[slots, :2].clone()      # (n, 2)
        cam_xy0  = self._camera_obj.data.root_pos_w[slots, :2].clone() # (n, 2)
        cam_z    = self._camera_obj.data.root_pos_w[slots, 2].clone()  # (n,)

        radii = torch.minimum(
            torch.norm(cam_xy0 - goal_xy, dim=1),
            torch.norm(agent_xy - goal_xy, dim=1))  # (n,)
        for i, eid in enumerate(eids):
            self.camera_circle_radii[eid] = float(radii[i].item())

        agent_yaws = torch.tensor(
            [self._get_agent_yaw(eid) for eid in eids],
            dtype=torch.float32, device=self.device)  # (n,)

        # Sweep is measured from the goal→agent direction (per env): angle 0 puts the
        # camera on the agent's side (collinear), 180 diametrically opposite. Positive
        # angle sweeps the agent's right half.
        goal_to_agent = agent_xy - goal_xy                                # (n, 2)
        base_angles = torch.atan2(goal_to_agent[:, 1], goal_to_agent[:, 0])  # (n,)

        env_origins  = self.scene.env_origins[slots, :2].clone()  # (n, 2)
        bound        = float(self.center_to_boundary)
        max_fov_rad  = math.radians(self._agent_half_hfov_deg - self._agent_fov_buffer_deg)
        left_90      = torch.tensor(
            [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
            device=self.device, dtype=torch.float32)

        trajectories:   dict[int, list] = {eid: [] for eid in eids}
        cached_frames:  dict[int, list] = {eid: [] for eid in eids}
        self._bump_camera_move_stat("batch_sweep_calls")

        for angle_deg in self.camera_sweep_angles_deg:
            world_angle = base_angles + math.radians(angle_deg)         # (n,)
            cam_x = goal_xy[:, 0] + radii * torch.cos(world_angle)     # (n,)
            cam_y = goal_xy[:, 1] + radii * torch.sin(world_angle)     # (n,)

            # Bounds filter
            in_bounds = (
                (cam_x >= env_origins[:, 0] - bound) & (cam_x <= env_origins[:, 0] + bound) &
                (cam_y >= env_origins[:, 1] - bound) & (cam_y <= env_origins[:, 1] + bound))

            # FOV filter
            dx = cam_x - agent_xy[:, 0]
            dy = cam_y - agent_xy[:, 1]
            angle_to_cam = torch.atan2(dy, dx)
            rel = torch.atan2(
                torch.sin(angle_to_cam - agent_yaws),
                torch.cos(angle_to_cam - agent_yaws))
            in_fov = rel.abs() <= max_fov_rad

            filt_idx = (in_bounds & in_fov).nonzero(as_tuple=True)[0]  # indices into [0..n)
            if not filt_idx.numel():
                continue

            # OBB collision check (already vectorized across different envs/positions)
            f_slots   = slots[filt_idx]
            f_cam_pos = torch.stack([cam_x[filt_idx], cam_y[filt_idx], cam_z[filt_idx]], dim=1)
            f_goal    = goal_pos[filt_idx]
            f_quats   = self._compute_camera_look_at_goal_quat_batch(f_cam_pos, f_goal)
            no_col    = ~self._check_collisions_vectorized(
                f_slots, f_cam_pos, f_quats, object_type='cam_obj')

            place_idx = filt_idx[no_col]  # indices into [0..n)
            if not place_idx.numel():
                continue
            self._bump_camera_move_stat("candidate_angles_after_geometry", int(place_idx.numel()))

            # Batch place camera objects
            p_slots   = slots[place_idx]
            p_cam_pos = torch.stack([cam_x[place_idx], cam_y[place_idx], cam_z[place_idx]], dim=1)
            p_goal    = goal_pos[place_idx]
            p_quats   = self._compute_camera_look_at_goal_quat_batch(p_cam_pos, p_goal)

            if self.camera_move_geometric_occlusion_precheck and self.camera_move_balance_saved_frames:
                pred_occluded = self._segment_intersects_active_obbs_2d(
                    p_cam_pos[:, :2], p_goal[:, :2], p_slots)
                self._bump_camera_move_stat(
                    "predicted_occluded_candidates", int(pred_occluded.sum().item()))
                self._bump_camera_move_stat(
                    "predicted_visible_candidates", int((~pred_occluded).sum().item()))

            self._bump_camera_move_stat("candidate_angles_rendered", int(len(place_idx)))

            self._camera_obj.write_root_com_pose_to_sim(
                torch.cat([p_cam_pos, p_quats], dim=1), p_slots)
            self._camera_obj.write_root_com_velocity_to_sim(
                torch.zeros((len(place_idx), 6), device=self.device), p_slots)

            # Batch set occlusion sensors with 90° left correction
            sensor_quats = math_utils.quat_mul(
                p_quats, left_90.unsqueeze(0).expand(len(place_idx), -1))
            sensor_quats = torch.nn.functional.normalize(sensor_quats, p=2, dim=-1)
            self._occlusion_camera.set_world_poses(
                positions=p_cam_pos,
                orientations=sensor_quats,
                env_ids=p_slots.tolist(),
                convention="world")

            # Settle sim for all envs at once (30 steps = same as _collect_images_for_slot)
            for _ in range(30):
                self.sim.step()
                self._camera_obj.update(self.sim.cfg.dt)
                self._rgb_tiled_camera.update(self.sim.cfg.dt)
                if self.save_camera_pov:
                    self._occlusion_camera.update(self.sim.cfg.dt)

            # Agent view validation (batch pixel check)
            av_goal, av_cam = self.check_batch_object_visibility(p_slots)
            av_valid = av_goal & av_cam  # (p,)

            valid_j = torch.where(av_valid)[0]
            if not valid_j.numel():
                continue
            final_visible, red_counts = self._camera_goal_visible_from_current_pov_batch(
                p_slots[valid_j])

            # Per-slot: cache image tensors after batched final labels.
            for label_idx, j_tensor in enumerate(valid_j):
                j = int(j_tensor.item())
                pi  = int(place_idx[j].item())  # index into [0..n)
                eid = eids[pi]

                visible = bool(final_visible[label_idx].item())
                red_count = int(red_counts[label_idx].item())
                frame = {
                    "angle_deg": angle_deg,
                    "position":  torch.tensor(
                        [cam_x[pi].item(), cam_y[pi].item()],
                        device=self.device, dtype=torch.float32),
                    "yaw":       float(self._quat_wxyz_to_yaw(p_quats[j])),
                    "label":     "Yes" if visible else "No",
                    "reason":    "in_view" if visible else "occluded",
                    "red_count": red_count,
                }
                trajectories[eid].append(frame)
                cached_frames[eid].append({
                    **frame,
                    "_rgb":         self._rgb_tiled_camera.data.output["rgb"][eid].clone(),
                    "_semantic":    self._rgb_tiled_camera.data.output["semantic_segmentation"][eid].clone(),
                    "_cam_pov":     (self._occlusion_camera.data.output["semantic_segmentation"][eid].clone()
                                     if self.save_camera_pov else None),
                })
                if self.verbose >= 2:
                    print(f"    [batch] Env {eid} sweep {angle_deg}°: "
                          f"{'Yes' if visible else 'No'} (red_px={red_count})")

        # Store results
        success: dict[int, bool] = {}
        for eid in eids:
            traj, balanced_cached = self._select_balanced_camera_move_frames(
                trajectories[eid], cached_frames[eid])
            if traj:
                self.camera_move_trajectories[eid] = traj
                self.selected_viewpoints_for_collection[eid] = torch.stack(
                    [f["position"] for f in traj])
                self._cached_frames[eid] = balanced_cached
                self._bump_camera_move_stat("accepted_balanced_envs")
                success[eid] = True
            else:
                self.camera_move_trajectories[eid] = None
                self._bump_camera_move_stat("rejected_no_final_balanced_pair")
                success[eid] = False
        return success

    def _generate_camera_circle_candidates(
            self,
            env_ids: torch.Tensor,
            radius_by_env: torch.Tensor,
            angle_step_deg: float = 2.0) -> dict[int, torch.Tensor]:
        """Generate smooth, no-jitter camera XY candidates on a fixed circle around each goal."""
        num_angles = int(360.0 / angle_step_deg)
        angles = torch.linspace(0.0, 2.0 * math.pi, num_angles + 1, device=self.device)[:-1]
        candidates = {}
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
            goal_xy = self._goal.data.root_pos_w[env_id_item, :2]
            radius = radius_by_env[i]
            xy = torch.stack([
                goal_xy[0] + radius * torch.cos(angles),
                goal_xy[1] + radius * torch.sin(angles),
            ], dim=1)
            candidates[env_id_item] = xy
        return candidates

    def _camera_goal_visible_from_current_pov(self, env_id: int | torch.Tensor) -> tuple[bool, int]:
        """Pixel-check whether the red goal is visible from the current camera POV."""
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        sem_img = self._occlusion_camera.data.output["semantic_segmentation"][env_id_item][..., :3]
        if sem_img.max() <= 1.0:
            sem_np = (sem_img.cpu().numpy() * 255.0).astype(np.uint8)
        else:
            sem_np = sem_img.cpu().numpy().astype(np.uint8)
        r, g, b = sem_np[..., 0], sem_np[..., 1], sem_np[..., 2]
        red_mask = (r >= 242) & (g <= 13) & (b <= 13)
        red_count = int(red_mask.sum())
        return red_count >= self.goal_pixel_threshold_occlusion, red_count

    def _bump_camera_move_stat(self, key: str, value: int = 1) -> None:
        self.camera_move_generation_stats[key] = (
            self.camera_move_generation_stats.get(key, 0) + int(value))

    def _camera_goal_visible_from_current_pov_batch(
            self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched red-pixel check for the occlusion camera POV."""
        sem_img = self._occlusion_camera.data.output["semantic_segmentation"][env_ids][..., :3]
        if sem_img.max() <= 1.0:
            sem_img = sem_img * 255.0
        r, g, b = sem_img[..., 0], sem_img[..., 1], sem_img[..., 2]
        red_mask = (r >= 242) & (g <= 13) & (b <= 13)
        red_counts = red_mask.sum(dim=(1, 2)).to(torch.long)
        visible = red_counts >= self.goal_pixel_threshold_occlusion
        return visible, red_counts

    def _segment_intersects_active_obbs_2d(
            self,
            start_xy: torch.Tensor,
            end_xy: torch.Tensor,
            env_ids: torch.Tensor,
            eps: float = 1e-6) -> torch.Tensor:
        """Predict occlusion with vectorized 2D segment-vs-active-OBB intersections."""
        if start_xy.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)

        obs_corners = self.obb_corners_cache[env_ids][..., :2]
        if isinstance(self.active_vpt_indices, list):
            active_idx = torch.stack(self.active_vpt_indices).to(
                dtype=torch.long, device=self.device)[env_ids]
        else:
            active_idx = self.active_vpt_indices[env_ids].to(
                dtype=torch.long, device=self.device)

        gather_idx = active_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 2)
        obs_corners = torch.gather(obs_corners, 1, gather_idx)
        obs_xy = obs_corners[:, :, :4, :]
        edge_a = obs_xy
        edge_b = torch.roll(obs_xy, shifts=-1, dims=2)

        seg_a = start_xy[:, None, None, :]
        seg_b = end_xy[:, None, None, :]
        edge_a_exp = edge_a
        edge_b_exp = edge_b

        def cross_2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

        seg = seg_b - seg_a
        edge = edge_b_exp - edge_a_exp
        denom = cross_2d(seg, edge)
        rel = edge_a_exp - seg_a
        non_parallel = denom.abs() > eps
        t = cross_2d(rel, edge) / (denom + (~non_parallel).to(denom.dtype) * eps)
        u = cross_2d(rel, seg) / (denom + (~non_parallel).to(denom.dtype) * eps)
        intersects_edge = non_parallel & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

        # Also count starts/ends inside an obstacle footprint.
        axes = torch.stack([
            obs_xy[:, :, 1, :] - obs_xy[:, :, 0, :],
            obs_xy[:, :, 3, :] - obs_xy[:, :, 0, :],
        ], dim=2)
        axes = axes / (torch.norm(axes, dim=-1, keepdim=True) + eps)
        rel_start = start_xy[:, None, None, :] - obs_xy[:, :, None, 0, :]
        rel_end = end_xy[:, None, None, :] - obs_xy[:, :, None, 0, :]
        extents = torch.stack([
            torch.norm(obs_xy[:, :, 1, :] - obs_xy[:, :, 0, :], dim=-1),
            torch.norm(obs_xy[:, :, 3, :] - obs_xy[:, :, 0, :], dim=-1),
        ], dim=2)
        start_proj = (rel_start * axes).sum(dim=-1)
        end_proj = (rel_end * axes).sum(dim=-1)
        start_inside = (
            (start_proj >= -eps) &
            (start_proj <= extents + eps)
        ).all(dim=-1)
        end_inside = (
            (end_proj >= -eps) &
            (end_proj <= extents + eps)
        ).all(dim=-1)

        return intersects_edge.any(dim=(1, 2)) | start_inside.any(dim=1) | end_inside.any(dim=1)

    def _rank_fixed_agent_points_by_camera_geometry(
            self,
            env_id: int | torch.Tensor,
            points_2d: torch.Tensor) -> torch.Tensor:
        """Rank fixed-agent candidates by geometric camera-sweep feasibility."""
        if points_2d.numel() == 0:
            return points_2d

        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        points_2d = points_2d[:self.camera_move_max_agent_viewpoints]
        num_points = len(points_2d)
        if num_points <= 1:
            return points_2d

        device = points_2d.device
        goal_pos = self._goal.data.root_pos_w[env_id_item].clone()
        goal_xy = goal_pos[:2]
        camera_xy_init = self._camera_obj.data.root_pos_w[env_id_item, :2].clone()
        camera_z = self._camera_obj.data.root_pos_w[env_id_item, 2].clone()
        cam_goal_dist = torch.norm(camera_xy_init - goal_xy)
        agent_goal_dist = torch.norm(points_2d - goal_xy.unsqueeze(0), dim=1)
        radii = torch.minimum(
            cam_goal_dist.expand(num_points),
            agent_goal_dist)

        midpoint = (camera_xy_init + goal_xy) / 2.0
        agent_dirs = midpoint.unsqueeze(0) - points_2d
        agent_yaws = torch.atan2(agent_dirs[:, 1], agent_dirs[:, 0])
        base_angles = torch.atan2(
            points_2d[:, 1] - goal_xy[1],
            points_2d[:, 0] - goal_xy[0])
        sweep_angles = torch.tensor(
            self.camera_sweep_angles_deg, dtype=torch.float32, device=device)
        world_angles = base_angles[:, None] + torch.deg2rad(sweep_angles)[None, :]

        cam_x = goal_xy[0] + radii[:, None] * torch.cos(world_angles)
        cam_y = goal_xy[1] + radii[:, None] * torch.sin(world_angles)
        num_angles = len(self.camera_sweep_angles_deg)
        cam_xy = torch.stack([cam_x, cam_y], dim=-1)
        flat_cam_xy = cam_xy.reshape(-1, 2)

        env_ids = torch.full(
            (num_points * num_angles,),
            env_id_item,
            dtype=torch.long,
            device=device)
        env_origin = self.scene.env_origins[env_id_item, :2]
        bound = float(self.center_to_boundary)
        in_bounds = torch.all(
            (flat_cam_xy >= env_origin - bound) &
            (flat_cam_xy <= env_origin + bound),
            dim=1).reshape(num_points, num_angles)

        dx = cam_x - points_2d[:, 0:1]
        dy = cam_y - points_2d[:, 1:2]
        angle_to_cam = torch.atan2(dy, dx)
        rel = torch.atan2(
            torch.sin(angle_to_cam - agent_yaws[:, None]),
            torch.cos(angle_to_cam - agent_yaws[:, None]))
        max_fov_rad = math.radians(
            self._agent_half_hfov_deg - self._agent_fov_buffer_deg)
        in_fov = rel.abs() <= max_fov_rad

        flat_cam_pos = torch.cat([
            flat_cam_xy,
            camera_z.expand(num_points * num_angles, 1)
        ], dim=1)
        flat_goal_pos = goal_pos.unsqueeze(0).expand(num_points * num_angles, -1)
        flat_quats = self._compute_camera_look_at_goal_quat_batch(
            flat_cam_pos, flat_goal_pos)
        no_collision = ~self._check_collisions_vectorized(
            env_ids, flat_cam_pos, flat_quats, object_type='cam_obj')
        no_collision = no_collision.reshape(num_points, num_angles)

        valid = in_bounds & in_fov & no_collision
        pred_occluded = self._segment_intersects_active_obbs_2d(
            flat_cam_xy, flat_goal_pos[:, :2], env_ids).reshape(num_points, num_angles)
        pred_visible = ~pred_occluded

        valid_count = valid.sum(dim=1)
        valid_occluded = (valid & pred_occluded).sum(dim=1)
        valid_visible = (valid & pred_visible).sum(dim=1)
        balanced_pairs = torch.minimum(valid_occluded, valid_visible)
        eligible = balanced_pairs >= self.camera_move_min_pairs_per_env
        if not eligible.any():
            self._bump_camera_move_stat("agent_viewpoints_rejected_no_pred_balanced_pair", num_points)
            return points_2d[:0]

        score = (
            valid_count.to(torch.float32) * 1000.0 +
            balanced_pairs.to(torch.float32) * 100.0 +
            valid_occluded.to(torch.float32)
        )
        score = torch.where(
            eligible,
            score,
            torch.full_like(score, -1.0))
        order = torch.argsort(score, descending=True)
        order = order[eligible[order] & (valid_count[order] == valid_count[order[0]])]
        self._bump_camera_move_stat("agent_viewpoints_scored", num_points)
        self._bump_camera_move_stat("agent_sweep_valid_candidates", int(valid_count[order[0]].item()))
        self._bump_camera_move_stat("agent_sweep_pred_occluded", int(valid_occluded[order[0]].item()))
        return points_2d[order]

    def _select_balanced_camera_move_frames(
            self,
            trajectory: list[dict],
            cached_frames: list[dict] | None = None) -> tuple[list[dict], list[dict] | None]:
        """Keep target frames while requiring at least one Yes and one No."""
        if not self.camera_move_balance_saved_frames:
            return trajectory, cached_frames

        yes_idx = [i for i, item in enumerate(trajectory) if item.get("label") == "Yes"]
        no_idx = [i for i, item in enumerate(trajectory) if item.get("label") == "No"]
        if len(yes_idx) < self.camera_move_min_pairs_per_env:
            return [], [] if cached_frames is not None else None
        if len(no_idx) < self.camera_move_min_pairs_per_env:
            return [], [] if cached_frames is not None else None
        if len(trajectory) < self.camera_move_target_frames_per_env:
            return [], [] if cached_frames is not None else None

        target = self.camera_move_target_frames_per_env
        keep = list(range(target))

        def count_label(indices: list[int], label: str) -> int:
            return sum(1 for i in indices if trajectory[i].get("label") == label)

        def ensure_label(indices: list[int], label: str, donor_indices: list[int]) -> list[int] | None:
            while count_label(indices, label) < self.camera_move_min_pairs_per_env:
                donor = next((i for i in donor_indices if i not in indices), None)
                if donor is None:
                    return None
                replace_pos = next(
                    (pos for pos in range(len(indices) - 1, -1, -1)
                     if trajectory[indices[pos]].get("label") != label),
                    None)
                if replace_pos is None:
                    return None
                indices[replace_pos] = donor
            return indices

        keep = ensure_label(keep, "Yes", yes_idx)
        if keep is None:
            return [], [] if cached_frames is not None else None
        keep = ensure_label(keep, "No", no_idx)
        if keep is None:
            return [], [] if cached_frames is not None else None

        keep = set(keep)
        selected_trajectory = [
            item for i, item in enumerate(trajectory)
            if i in keep
        ]
        if cached_frames is None:
            return selected_trajectory, None
        selected_cached_frames = [
            item for i, item in enumerate(cached_frames)
            if i in keep
        ]
        return selected_trajectory, selected_cached_frames

    def _validate_camera_circle_candidates(
            self,
            env_id: int | torch.Tensor,
            candidates: torch.Tensor,
            radius: float,
            radius_tol: float = 1e-3) -> list[dict]:
        """Classify camera circle candidates as visible/occluded using final pixel checks."""
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        env_ids = torch.full((len(candidates),), env_id_item, dtype=torch.long, device=self.device)
        env_origins = self.scene.env_origins[env_ids, :2]
        in_bounds = torch.all(
            (candidates >= env_origins - self.center_to_boundary) &
            (candidates <= env_origins + self.center_to_boundary),
            dim=1)

        active_obs_pos = self._get_active_obstacle_positions(env_ids)
        dist_cam_obs = torch.norm(candidates.unsqueeze(1) - active_obs_pos, dim=2)
        goal_xy = self._goal.data.root_pos_w[env_id_item, :2]
        dist_goal = torch.norm(candidates - goal_xy.unsqueeze(0), dim=1)
        geometric_valid = (
            in_bounds &
            (dist_cam_obs.min(dim=1)[0] >= 0.4) &
            (torch.abs(dist_goal - radius) <= radius_tol)
        )

        valid = []
        for idx in torch.where(geometric_valid)[0].tolist():
            camera_xy = candidates[idx]
            yaw = self._set_camera_object_pose(env_id_item, camera_xy, self._goal.data.root_pos_w[env_id_item])
            if not self._validate_fixed_agent_view(env_id_item):
                continue
            visible, red_count = self._camera_goal_visible_from_current_pov(env_id_item)
            valid.append({
                "angle_index": idx,
                "position": camera_xy.clone(),
                "yaw": yaw,
                "label": "Yes" if visible else "No",
                "reason": "in_view" if visible else "occluded",
                "red_count": red_count,
            })

        return valid

    def _select_balanced_camera_trajectory(self, env_id: int | torch.Tensor, candidates: list[dict]) -> bool:
        """DEPRECATED: replaced by _build_fixed_sweep_trajectory. Not called."""
        raise NotImplementedError("Use _build_fixed_sweep_trajectory instead.")
        env_id_item = env_id.item() if torch.is_tensor(env_id) else env_id
        visible = [c for c in candidates if c["label"] == "Yes"]
        occluded = [c for c in candidates if c["label"] == "No"]
        if len(visible) < self.camera_circle_visible_frames or len(occluded) < self.camera_circle_occluded_frames:
            self.camera_move_trajectories[env_id_item] = None
            return False

        visible_step = max(1, len(visible) // self.camera_circle_visible_frames)
        occluded_step = max(1, len(occluded) // self.camera_circle_occluded_frames)
        selected = (
            visible[::visible_step][:self.camera_circle_visible_frames] +
            occluded[::occluded_step][:self.camera_circle_occluded_frames]
        )
        selected = sorted(selected, key=lambda item: item["angle_index"])
        self.camera_move_trajectories[env_id_item] = selected
        self.selected_viewpoints_for_collection[env_id_item] = torch.stack(
            [item["position"] for item in selected])
        return True

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
        self._agent.write_root_com_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids)

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
        # In _teleport_and_step, after:
        yaws = torch.atan2(dirs[:, 1], dirs[:, 0])
        yaw_jitter = (torch.rand(len(env_ids), device=points.device) - 0.5) * 2 * math.radians(15)
        yaws = yaws + yaw_jitter

        # Pose construction
        pos = torch.zeros((len(env_ids), 3), device=points.device)
        pos[:, :2] = points
        pos[:, 2] = self._agent.data.default_root_state[env_ids, 2]

        quat = torch.zeros((len(env_ids), 4), device=points.device)
        quat[:, 0] = torch.cos(yaws / 2)
        quat[:, 3] = torch.sin(yaws / 2)

        self._agent.write_root_com_pose_to_sim(torch.cat([pos, quat], dim=1),
                                               env_ids)
        self._agent.write_root_com_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids)

        # Step Physics
        self.sim.step()
        self._rgb_tiled_camera.update(self.sim.cfg.dt)
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

        MIN_REQUIRED_POINTS = self.camera_move_min_agent_viewpoints
        FOV_CHECK_TARGET_POINTS = max(
            self.camera_move_min_agent_viewpoints,
            self.camera_move_max_agent_viewpoints)

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
        # Scale radii by random factor [1.1, 1.5]
        scale_factor = (torch.rand(num_envs, device=device) * 0.4) + 1.0
        radii = radii * scale_factor
        radii = radii.unsqueeze(1)

        # Generate all points for all envs
        angles_expanded = angles.unsqueeze(0).expand(num_envs, -1)
        all_x = self._goal.data.root_pos_w[env_ids, 0].unsqueeze(
            1) + radii * torch.cos(angles_expanded)
        all_y = self._goal.data.root_pos_w[env_ids, 1].unsqueeze(
            1) + radii * torch.sin(angles_expanded)
        
        # Radial jitter: ±20% of radius
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

        MIN_CANDIDATES_FOR_FOV = max(
            MIN_REQUIRED_POINTS,
            self.camera_move_min_geometric_agent_candidates)

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
            FOV_CHECK_TARGET_POINTS  # Collect enough candidates to score geometry.
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
                ranked_points = self._rank_fixed_agent_points_by_camera_geometry(
                    env_id, valid_points_tensor)
                if ranked_points.shape[0] == 0:
                    all_valid_points.append(torch.zeros((0, 2), device=device))
                    if self.verbose >= 1:
                        print(
                            f"  Env {env_id_item}: ❌ 0/{len(valid_points_tensor)} FOV-valid agent points had predicted Yes+No camera sweep"
                        )
                    continue
                all_valid_points.append(ranked_points)
                if self.verbose >= 2:
                    print(
                        f"  Env {env_id_item}: ✅ {len(valid_points_tensor)}/{num_candidates} passed FOV; "
                        f"{len(ranked_points)} tied for max camera geometry ({fov_rejection_rate:.1f}% rejected)"
                    )
            else:
                all_valid_points.append(torch.zeros((0, 2), device=device))
                if self.verbose >= 1:
                    print(
                        f"  Env {env_id_item}: ❌ Only {len(valid_points_tensor)}/{MIN_REQUIRED_POINTS} points ({fov_rejection_rate:.1f}% FOV rejection)"
                    )

        return all_valid_points

    def _save_slot_from_cache(self, env_slot: int, folder_idx: int) -> None:
        """Save pre-captured frames from self._cached_frames without re-placing the camera."""
        frames = self._cached_frames.pop(env_slot, [])
        if not frames:
            return
        if self.verbose >= 1:
            print(f"    📸 Saving {len(frames)} cached frames | Slot {env_slot} -> Folder {folder_idx}")
        for frame in frames:
            self._save_camera_move_frame(
                env_slot=env_slot,
                folder_idx=folder_idx,
                rgb=frame["_rgb"],
                semantic=frame["_semantic"],
                cam_pov=frame.get("_cam_pov"),
                angle_deg=frame["angle_deg"],
                label=frame["label"])
        self._save_env_config_to_json(env_slot, folder_idx)
        self.selected_viewpoints_for_collection[env_slot] = None
        if self.verbose >= 1:
            print(f"    ✅ Saved Folder {folder_idx} (cached)")

    def _collect_images_for_slot(self, env_id: torch.Tensor, folder_idx: int) -> None:
        """
        Moves the camera object on a fixed circle while the agent/goal/obstacles stay fixed.

        Parameters
        ----------
        env_id : torch.Tensor
            The local slot index (0 to num_envs-1).
        folder_idx : int
            The global folder index for file naming.
        """
        env_slot = env_id.item() if torch.is_tensor(env_id) else env_id
        global_env_id = self.slot_to_env_id[env_slot]
        trajectory = self.camera_move_trajectories[env_slot]

        # --- Validation ---
        if trajectory is None:
            raise RuntimeError(f"No camera trajectory selected for Slot {env_slot} (Env {global_env_id})")

        if self.verbose >= 1:
            print(f"    📸 Collecting {self.images_per_env} camera-move images | Slot {env_slot} -> Folder {folder_idx}")

        goal_pos = self._goal.data.root_pos_w[env_slot].clone()

        # --- Collection Loop ---
        for frame in trajectory:
            angle_deg = frame["angle_deg"]
            label = frame["label"]

            # 1. Move only the camera object. Agent, goal, and obstacles remain fixed.
            self._set_camera_object_pose(env_slot, frame["position"], goal_pos)

            # 2. Update Sensors (Step Sim)
            for _ in range(30):
                self.sim.step()
                self._rgb_tiled_camera.update(self.sim.cfg.dt)
                if self.save_camera_pov:
                    self._occlusion_camera.update(self.sim.cfg.dt)

            # 3. Extract Data
            rgb = self._rgb_tiled_camera.data.output["rgb"][env_slot]
            semantic = self._rgb_tiled_camera.data.output["semantic_segmentation"][env_slot]
            cam_pov = self._occlusion_camera.data.output["semantic_segmentation"][env_slot] if self.save_camera_pov else None

            # 4. Save with angle suffix in filename (e.g. image_15d.png).
            self._save_camera_move_frame(
                env_slot=env_slot,
                folder_idx=folder_idx,
                rgb=rgb,
                semantic=semantic,
                cam_pov=cam_pov,
                angle_deg=angle_deg,
                label=label)

        # Cleanup
        self._save_env_config_to_json(env_slot, folder_idx)
        self.selected_viewpoints_for_collection[env_slot] = None

        if self.verbose >= 1:
            print(f"    ✅ Saved Folder {folder_idx}")

    def _save_camera_move_frame(
            self,
            env_slot: int,
            folder_idx: int,
            rgb: torch.Tensor,
            semantic: torch.Tensor,
            cam_pov: torch.Tensor | None,
            angle_deg: int,
            label: str) -> None:
        """Save one camera-move frame. Filename encodes sweep angle (e.g. image_15d.png)."""
        self._save_single_image(
            env_slot=env_slot,
            folder_idx=folder_idx,
            rgb_data=rgb,
            semantic_data=semantic,
            camera_pov_data=cam_pov,
            angle_deg=angle_deg,
            label_override=label)

    def _save_single_image(self,
                           env_slot: int,
                           folder_idx: int,
                           rgb_data: torch.Tensor,
                           semantic_data: torch.Tensor,
                           camera_pov_data: torch.Tensor | None = None,
                           angle_deg: int = 0,
                           label_override: str | None = None) -> None:
        """
        Save RGB/Semantic agent POV and semantic camera POV for one sweep frame.
        Filenames encode the sweep angle AND the Yes/No label: image_0d_Yes.png,
        image_15d_No.png, ... Per-frame label is also stored in the config JSON.
        All frames for an env go into the same Mixed directory regardless of label.

        Agent POV  -> RGB/ Semantic/
        Camera POV -> cam_Semantic/
        """
        # --- 1. Path Setup ---
        # Use "Mixed" as directory label — per-frame Yes/No is in the filename + config JSON.
        base_dir = f"{self.base_path}/{{}}/Mixed/env_{folder_idx}"
        rgb_dir = base_dir.format("RGB")
        semantic_dir = base_dir.format("Semantic")

        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(semantic_dir, exist_ok=True)
        if self.save_camera_pov:
            os.makedirs(base_dir.format("cam_Semantic"), exist_ok=True)

        label = label_override if label_override else "NA"
        fname = f"image_{angle_deg}d_{label}.png"

        # --- 2. Save RGB ---
        rgb_np = rgb_data.cpu().numpy()
        if rgb_np.dtype != np.uint8:
            rgb_np = (rgb_np * 255.0).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]
        cv2.imwrite(f"{rgb_dir}/{fname}", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # --- 3. Save Semantic (agent POV) ---
        if semantic_data is not None:
            sem_np = semantic_data.cpu().numpy()
            if sem_np.dtype != np.uint8:
                sem_np = (sem_np * 255.0).astype(np.uint8) if sem_np.max() <= 1.0 else sem_np.astype(np.uint8)
            if sem_np.shape[-1] == 4:
                sem_np = sem_np[..., :3]
            cv2.imwrite(f"{semantic_dir}/{fname}", cv2.cvtColor(sem_np, cv2.COLOR_RGB2BGR))

        # --- 4. Save Camera POV semantic at every sweep angle ---
        if self.save_camera_pov and camera_pov_data is not None:
            cam_sem_dir = base_dir.format("cam_Semantic")

            # Semantic seg from occlusion camera
            cam_pov_np = camera_pov_data.cpu().numpy()
            if cam_pov_np.dtype != np.uint8:
                cam_pov_np = (cam_pov_np * 255.0).astype(np.uint8) if cam_pov_np.max() <= 1.0 else cam_pov_np.astype(np.uint8)
            if cam_pov_np.shape[-1] == 4:
                cam_pov_np = cam_pov_np[..., :3]
            cv2.imwrite(f"{cam_sem_dir}/{fname}", cv2.cvtColor(cam_pov_np, cv2.COLOR_RGB2BGR))

    # TODO: UPDATE THIS TO MATCH NEW OBJECTS | Updated, just double check when you open isaacsim
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
                    scale = (1.0, 1.0, 1.0)
                
                # Extract filename to check for special scaling rules
                filename = spawn_cfg.usd_path.split("/")[-1].split(".")[0]

                # Special Case 1: Furniture (Table_A, Table_B, Bench)
                # These use standard 1.0 scaling
                if filename.endswith(('Table_A', 'Table_B', 'Bench')):
                    dims = torch.tensor(scale, device=self.device)
                
                # Special Case 2: Letters (X, L, T, I, A, H, Z)
                # These use a 0.25 multiplier on the Y axis
                elif filename.endswith(('X', 'L', 'T', 'I', 'A', 'H', 'Z')):
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

                            if filename.endswith(("L")):
                                s_factor = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_factor,
                                    base_dim[1] * s_factor,
                                    base_dim[2] * s_factor)
                            elif filename.endswith(("Table_B", "Bench")):
                                s_factor = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_factor,
                                    base_dim[1] * s_factor,
                                    base_dim[2] * s_factor)
                            elif filename.endswith(("Table_A")):
                                s_x = random.uniform(0.5, 3.0)
                                s_y = random.uniform(0.5, 3.0)
                                s_z = random.uniform(0.5, 3.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_x, base_dim[1] * s_y,
                                    base_dim[2] * s_z)
                            elif filename.endswith(("X", "A", "H", "I", "Z")):
                                s_xz = random.uniform(0.5, 3.0)
                                s_y = random.uniform(1.0, 5.0)
                                final_scale_vec = Gf.Vec3d(
                                    base_dim[0] * s_xz, base_dim[1] * s_y,
                                    base_dim[2] * s_xz)

                            # Calculation: Base is at 0, so Z=0
                            final_z_pos = 0.0
                        else:
                            z_offset_multiplier = 0.0
                            s_xy = random.uniform(0.5, 3.0)
                            s_z = random.uniform(0.5, 3.0)
                            final_scale_vec = Gf.Vec3d(base_dim[0] * s_xy,
                                                       base_dim[1] * s_xy,
                                                       base_dim[2] * s_z)
                            shape_name = -1

                    # 2. Cuboids
                    elif isinstance(spawn_cfg, sim_utils.MeshCuboidCfg):
                        shape_name = 2
                        z_offset_multiplier = 0.5
                        s_x = random.uniform(0.5, 3.0)
                        s_y = random.uniform(0.5, 3.0)
                        s_z = random.uniform(0.5, 3.0)
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
                        s_h = random.uniform(0.75, 3.0)
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
            rand_radius = random.uniform(0.75, 2.25)
            prim.GetAttribute("inputs:radius").Set(rand_radius)

            # 5. COLOR TEMPERATURE
            prim.GetAttribute("inputs:enableColorTemperature").Set(True)
            rand_temp = random.uniform(2000.0, 8000.0)
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
        threshold = 0.15

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
            tex_scale = (1000.0, 1000.0)
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
            local_corners.unsqueeze(0).unsqueeze(0),
            rot_mat.transpose(-1, -2)
        )
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
        agent_corners = self._get_object_corners(proposed_pos, proposed_quat, 
                                              object_type=object_type)

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

    def place_object_safely(self, env_ids, object_state, vpt_state,
                        safe_range, range_offsets=None, 
                        object_type='agent', custom_centers=None):
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
