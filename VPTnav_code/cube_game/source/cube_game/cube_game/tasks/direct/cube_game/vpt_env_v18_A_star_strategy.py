from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.api import World
from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform
from pxr import Gf, Sdf, Usd, UsdGeom

from .spawn_boundary import get_mat_material_paths, get_vpt_material_paths
from .vpt_env_cfg_v15_rl import VPTEnvCfg


class VPTEnvAStarStrategy(DirectRLEnv):
    """VPT-Strategy static scene generator.

    The environment samples one fixed observer, one fixed occluder, and ten
    camera/goal rail settings per accepted scene. Each accepted scene has exactly
    five camera-POV visible labels and five camera-POV not-visible labels.
    """

    cfg: VPTEnvCfg

    def __init__(self, cfg: VPTEnvCfg, render_mode: str | None = None, **kwargs):
        self.verbose = int(os.getenv("STRATEGY_VERBOSE", "1"))
        self.GPU_ID = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.NODE_ID = os.getenv("NODE_ID", os.getenv("SLURM_ARRAY_TASK_ID", "0"))
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.boundary_limits = self.cfg.boundary_limits
        self.agent_height = self.cfg.agent_height
        self.agent_camera_pitch = self.cfg.agent_camera_pitch
        self.goal_radius = self.cfg.goal_radius
        self.center_to_boundary = torch.abs(
            torch.tensor(self.boundary_limits, device=self.device).view(-1)[0]
        )

        self.num_objs = self.cfg.num_vpt_objs
        self.active_vpt_objs = self.cfg.objects_per_env
        self.storage_position = torch.tensor([250.0, 250.0, -100.0], device=self.device)

        self.images_per_scene = int(os.getenv("STRATEGY_IMAGES_PER_SCENE", "10"))
        self.required_yes = int(os.getenv("STRATEGY_REQUIRED_YES", "5"))
        self.required_no = int(os.getenv("STRATEGY_REQUIRED_NO", "5"))
        self.target_scenes = int(os.getenv("STRATEGY_TARGET", os.getenv("TOTAL_ENVS_TO_SIM", "2000")))
        self.max_scene_attempts = int(os.getenv("STRATEGY_MAX_ATTEMPTS", "250"))
        # TODO: Testing default only. Restore/tune to ~30 before production collection.
        self.settle_steps = int(os.getenv("SETTLE_STEPS", "30"))

        self.cam_red_thresh = int(os.getenv("STRATEGY_CAM_RED_THRESH", "200"))
        self.cam_no_red_max = int(os.getenv("STRATEGY_CAM_NO_RED_MAX", "0"))
        self.goal_pixel_threshold = int(os.getenv("STRATEGY_AGENT_GOAL_THRESH", "200"))
        self.camera_pixel_threshold = int(os.getenv("STRATEGY_AGENT_CAMERA_THRESH", "300"))
        self.agent_deadzone_deg = float(os.getenv("STRATEGY_AGENT_DEADZONE_DEG", "45.0"))
        self.agent_perp_deadzone_deg = float(
            os.getenv("STRATEGY_AGENT_PERP_DEADZONE_DEG", str(self.agent_deadzone_deg))
        )

        self.rail_extent = float(os.getenv("STRATEGY_RAIL_EXTENT", "6.0"))
        self.camera_goal_distance = float(os.getenv("STRATEGY_CAMERA_GOAL_DISTANCE", "4.0"))
        self.candidate_points = int(os.getenv("STRATEGY_CANDIDATE_POINTS", "80"))
        self.axis_candidates = int(os.getenv("STRATEGY_AXIS_CANDIDATES", "24"))
        self.min_valid_candidate_points = int(os.getenv("STRATEGY_MIN_VALID_CANDIDATES", "30"))
        self.min_camera_goal_clearance = float(os.getenv("STRATEGY_MIN_CAMERA_GOAL_CLEARANCE", "1.0"))
        self.max_camera_goal_distance = float(os.getenv("STRATEGY_MAX_CAMERA_GOAL_DISTANCE", "6.0"))
        self.min_selected_pair_distance = float(os.getenv("STRATEGY_MIN_SELECTED_PAIR_DISTANCE", "0.1"))
        self.min_agent_clearance = float(os.getenv("STRATEGY_MIN_AGENT_CLEARANCE", "2.0"))
        # Distance from scene center along the rail axis for structured observer candidates.
        # Larger sees more of the rail but shrinks objects; smaller increases pixels but may crop endpoints.
        self.agent_observer_distance = float(os.getenv("STRATEGY_AGENT_OBSERVER_DISTANCE", "10.0"))
        self.agent_observer_jitter = float(os.getenv("STRATEGY_AGENT_OBSERVER_JITTER", "1.0"))
        self.min_occluder_clearance = float(os.getenv("STRATEGY_MIN_OCCLUDER_CLEARANCE", "1.0"))
        self.occluder_index = int(os.getenv("STRATEGY_OCCLUDER_INDEX", "0"))
        self.distractor_clearance = float(os.getenv("STRATEGY_DISTRACTOR_CLEARANCE", "1.0"))
        self.occluder_scale_xy_min = float(os.getenv("STRATEGY_OCCLUDER_SCALE_XY_MIN", "1.5"))
        self.occluder_scale_xy_max = float(os.getenv("STRATEGY_OCCLUDER_SCALE_XY_MAX", "2.5"))
        self.occluder_scale_z_min = float(os.getenv("STRATEGY_OCCLUDER_SCALE_Z_MIN", "2.0"))
        self.occluder_scale_z_max = float(os.getenv("STRATEGY_OCCLUDER_SCALE_Z_MAX", "3.0"))
        self.distractor_scale_min = float(os.getenv("STRATEGY_DISTRACTOR_SCALE_MIN", "0.75"))
        self.distractor_scale_max = float(os.getenv("STRATEGY_DISTRACTOR_SCALE_MAX", "1.5"))

        self.camera_yaw_correction_rad = math.radians(90.0)

        self.scene_records: dict[str, Any] = {}
        self.scene_attempt_counts: dict[str, int] = {}
        self.next_scene_id = 0
        self._base_dims_cached = False

        base = os.getenv("BASE_PATH", "/oscar/scratch/arock3/VPT_STRATEGY/v18")
        self.base_path = f"{base}/data/data_node{self.NODE_ID}_gpu{self.GPU_ID}"
        self.strategy_labels_json_path = os.path.join(self.base_path, "strategy_labels.json")

        print("*" * 50, flush=True)
        print(f"Initializing VPTEnvAStarStrategy node={self.NODE_ID} gpu={self.GPU_ID}", flush=True)
        print(f"base_path={self.base_path}", flush=True)
        print(
            "[strategy:init] "
            f"target={self.target_scenes} images_per_scene={self.images_per_scene} "
            f"required_yes={self.required_yes} required_no={self.required_no} "
            f"candidate_points={self.candidate_points} "
            f"axis_candidates={self.axis_candidates} settle_steps={self.settle_steps} "
            f"verbose={self.verbose}",
            flush=True,
        )
        print(
            "[strategy:init] "
            f"cam_red_thresh={self.cam_red_thresh} cam_no_red_max={self.cam_no_red_max} "
            f"agent_goal_thresh={self.goal_pixel_threshold} "
            f"agent_camera_thresh={self.camera_pixel_threshold} "
            f"camera_goal_distance={self.camera_goal_distance}",
            flush=True,
        )
        print("*" * 50, flush=True)

    def close(self):
        super().close()

    def _log(self, msg: str, level: int = 1) -> None:
        """Print debug messages immediately when STRATEGY_VERBOSE is high enough."""
        if self.verbose >= level:
            print(msg, flush=True)

    def _min_med_max_str(self, values: torch.Tensor) -> str:
        """Return compact min/median/max debug stats for a tensor."""
        if values.numel() == 0:
            return "na/na/na"
        vals = values.detach().float().flatten()
        return (
            f"{int(vals.min().item())}/"
            f"{int(vals.median().item())}/"
            f"{int(vals.max().item())}"
        )

    def _setup_scene(self) -> None:
        """Create static actors, sensors, cloned envs, lights, and material pools."""
        t0 = time.perf_counter()
        self._log("[strategy:setup] start scene setup")
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

        light_cfg = sim_utils.SphereLightCfg(intensity=50_000.0, color=(0.75, 0.75, 0.75), radius=1.5)
        light_cfg.func("/World/envs/env_0/Light_A", light_cfg)

        self.mat_material_configs = self.get_material_configs("mat")
        self.vpt_material_configs = self.get_material_configs("vpt")
        self.mat_material_paths = []
        self.vpt_material_paths = []
        for idx, material in enumerate(self.mat_material_configs):
            path = f"/World/Looks/strategy_mat_material_{idx}"
            material.func(path, material)
            self.mat_material_paths.append(path)
        for idx, material in enumerate(self.vpt_material_configs):
            path = f"/World/Looks/strategy_vpt_material_{idx}"
            material.func(path, material)
            self.vpt_material_paths.append(path)
        self._log(f"[strategy:setup] done dt={time.perf_counter() - t0:.2f}s")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions for DirectRLEnv compatibility; strategy collection is reset-driven."""
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        """Treat reset actions as requests to generate more static scenes."""
        if not hasattr(self, "actions"):
            return
        reset_mask = (self.actions == 5) | (self.actions == 6)
        if reset_mask.any() and self.next_scene_id < self.target_scenes:
            env_ids = torch.where(reset_mask)[0].to(device=self.device, dtype=torch.long)
            self._reset_idx(env_ids)

    def _get_observations(self, mode=None) -> dict:
        """Return current agent RGB observation for API compatibility."""
        self._rgb_tiled_camera.update(self.sim.cfg.dt)
        rgb = self._rgb_tiled_camera.data.output["rgb"].permute(0, 3, 1, 2)[:, :3]
        return {"policy": rgb.clone()}

    def _get_rewards(self) -> torch.Tensor:
        """Return zero rewards; this env is not optimized by RL."""
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only timeout dones for API compatibility."""
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Generate one accepted strategy scene per requested env slot when possible."""
        t_reset = time.perf_counter()
        if env_ids is None:
            env_ids = self._agent._ALL_INDICES
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids = env_ids.view(-1)
        self._log(
            f"[strategy:reset] env_ids={env_ids.tolist()} "
            f"next_scene={self.next_scene_id}/{self.target_scenes}"
        )

        t0 = time.perf_counter()
        self._ensure_base_dims()
        self._log(f"[strategy:reset] base dims cached dt={time.perf_counter() - t0:.2f}s", level=2)
        t0 = time.perf_counter()
        self._randomize_strategy_scene_props(env_ids.tolist())
        self._log(f"[strategy:reset] textures/lights randomized dt={time.perf_counter() - t0:.2f}s", level=2)
        t0 = time.perf_counter()
        self._reset_raw_states(env_ids)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.step_dt)
        self._log(f"[strategy:reset] raw states reset dt={time.perf_counter() - t0:.2f}s", level=2)

        pending = env_ids.tolist()
        attempts = {env_id: 0 for env_id in pending}
        while pending and self.next_scene_id < self.target_scenes:
            t_attempt = time.perf_counter()
            for env_id in pending:
                attempts[env_id] += 1
            if self.verbose >= 1:
                self._log(
                    f"[strategy:batch] pending={len(pending)} "
                    f"attempt={min(attempts.values())} "
                    f"next_scene={self.next_scene_id}/{self.target_scenes}"
                )

            accepted, rejected = self._generate_strategy_scene_batch(pending)
            for payload in accepted:
                scene_id = self.next_scene_id
                self._commit_strategy_scene(scene_id, payload)
                self.scene_attempt_counts[str(scene_id)] = attempts[payload["env_id"]]
                self.next_scene_id += 1
                self._log(
                    f"[strategy:scene] scene={scene_id} env={payload['env_id']} "
                    f"SAVED attempts={attempts[payload['env_id']]} "
                    f"total_saved={len(self.scene_records)}/{self.target_scenes}"
                )
                if self.next_scene_id >= self.target_scenes:
                    break

            if accepted:
                self._save_strategy_labels()

            accepted_envs = {payload["env_id"] for payload in accepted}
            next_pending = []
            for env_id in pending:
                if env_id in accepted_envs:
                    continue
                if attempts[env_id] >= self.max_scene_attempts:
                    self._log(
                        f"[strategy:scene] env={env_id} FAILED "
                        f"attempts={self.max_scene_attempts} "
                        f"last={rejected.get(env_id, 'unknown')}"
                    )
                    continue
                next_pending.append(env_id)
            pending = next_pending
            self._log(
                f"[strategy:batch] done accepted={len(accepted)} "
                f"rejected={len(rejected)} remaining={len(pending)} "
                f"dt={time.perf_counter() - t_attempt:.2f}s"
            )

        self._log(f"[strategy:reset] done dt={time.perf_counter() - t_reset:.2f}s")

    def _generate_strategy_scene_batch(self, env_ids: list[int]) -> tuple[list[dict[str, Any]], dict[int, str]]:
        """Attempt one full ten-setting scene for many env slots in parallel."""
        if not env_ids:
            return [], {}

        scene_cfgs: dict[int, dict[str, Any]] = {}
        rejected: dict[int, str] = {}
        t_geom = time.perf_counter()
        base_cfgs = self._sample_strategy_base_batch(env_ids)
        for env_id, scene_cfg in base_cfgs.items():
            if len(scene_cfg["settings"]) < self.min_valid_candidate_points:
                rejected[env_id] = (
                    f"axis_candidate_bank_too_small n={len(scene_cfg['settings'])} "
                    f"min={self.min_valid_candidate_points}"
                )
                continue
            scene_cfgs[env_id] = scene_cfg

        agent_poses = self._sample_strategy_agent_pose_batch(scene_cfgs)
        for env_id in list(scene_cfgs.keys()):
            agent_pose = agent_poses.get(env_id)
            scene_cfg = scene_cfgs[env_id]
            if agent_pose is None:
                rejected[env_id] = "agent_pose_sample_failed"
                scene_cfgs.pop(env_id, None)
                continue
            scene_cfg["agent_pos"], scene_cfg["agent_quat"] = agent_pose
            scene_cfg["captures"] = []
            scene_cfg["labels"] = []
        self._log(
            f"[strategy:geom] envs={len(env_ids)} usable={len(scene_cfgs)} "
            f"rejected={len(rejected)} dt={time.perf_counter() - t_geom:.2f}s",
            level=2,
        )

        active = list(scene_cfgs.keys())
        if not active:
            return [], rejected

        max_candidates = max(len(scene_cfgs[env_id]["settings"]) for env_id in active)
        for setting_idx in range(max_candidates):
            t_setting = time.perf_counter()
            current = []
            for env_id in active:
                labels = scene_cfgs[env_id]["labels"]
                if labels.count("Yes") >= self.required_yes and labels.count("No") >= self.required_no:
                    continue
                if setting_idx >= len(scene_cfgs[env_id]["settings"]):
                    rejected[env_id] = (
                        f"candidate_exhausted Yes={labels.count('Yes')} "
                        f"No={labels.count('No')}"
                    )
                    continue
                current.append(env_id)
            if not current:
                break
            self._apply_strategy_setting_batch(
                current,
                [scene_cfgs[env_id]["settings"][setting_idx] for env_id in current],
                [scene_cfgs[env_id] for env_id in current],
            )
            self._orient_camera_to_goal_batch(current)
            t_settle = time.perf_counter()
            self._settle_and_update_cameras(self.settle_steps)
            settle_dt = time.perf_counter() - t_settle

            t_classify = time.perf_counter()
            pov_labels, pov_reasons, pov_red, pov_valid = self._classify_camera_pov_batch(current)
            agent_ok, agent_goal_px, agent_camera_px = self._validate_agent_view_batch(current)
            classify_dt = time.perf_counter() - t_classify

            still_active = []
            labels_this_setting = {
                "Yes": 0,
                "No": 0,
                "skip": 0,
                "complete": 0,
                "cam_deadzone": 0,
                "agent_view": 0,
                "quota_full": 0,
            }
            for local_i, env_id in enumerate(current):
                if not bool(pov_valid[local_i]):
                    red_count = int(pov_red[local_i].item())
                    scene_cfgs[env_id].setdefault("skipped_candidates", []).append(
                        f"setting_{setting_idx}:cam_deadzone_red={red_count}"
                    )
                    labels_this_setting["skip"] += 1
                    labels_this_setting["cam_deadzone"] += 1
                    still_active.append(env_id)
                    continue

                label = pov_labels[local_i]
                reason = pov_reasons[local_i]
                red_count = int(pov_red[local_i].item())
                goal_px = int(agent_goal_px[local_i].item())
                camera_px = int(agent_camera_px[local_i].item())
                if not bool(agent_ok[local_i]):
                    scene_cfgs[env_id].setdefault("skipped_candidates", []).append(
                        f"setting_{setting_idx}:agent_view "
                        f"goal_px={goal_px} camera_px={camera_px}"
                    )
                    labels_this_setting["skip"] += 1
                    labels_this_setting["agent_view"] += 1
                    still_active.append(env_id)
                    continue

                scene_cfg = scene_cfgs[env_id]
                if scene_cfg["labels"].count(label) >= (
                    self.required_yes if label == "Yes" else self.required_no
                ):
                    labels_this_setting["skip"] += 1
                    labels_this_setting["quota_full"] += 1
                    still_active.append(env_id)
                    continue
                if not self._selected_pair_distance_ok(
                    scene_cfg["captures"], scene_cfg["settings"][setting_idx]
                ):
                    scene_cfg.setdefault("skipped_candidates", []).append(
                        f"setting_{setting_idx}:selected_pair_too_close"
                    )
                    labels_this_setting["skip"] += 1
                    still_active.append(env_id)
                    continue

                scene_cfg["captures"].append({
                    "candidate_idx": int(scene_cfg["settings"][setting_idx].get("candidate_idx", setting_idx)),
                    "setting": scene_cfg["settings"][setting_idx],
                    "label": label,
                    "reason": reason,
                    "camera_pov_red_count": red_count,
                    "agent_goal_px": goal_px,
                    "agent_camera_px": camera_px,
                    "rgb": self._rgb_tiled_camera.data.output["rgb"][env_id].clone(),
                    "semantic": self._rgb_tiled_camera.data.output["semantic_segmentation"][env_id].clone(),
                    "cam_semantic": self._occlusion_camera.data.output["semantic_segmentation"][env_id].clone(),
                })
                scene_cfg["labels"].append(label)
                labels_this_setting[label] += 1
                if (
                    scene_cfg["labels"].count("Yes") >= self.required_yes
                    and scene_cfg["labels"].count("No") >= self.required_no
                ):
                    labels_this_setting["complete"] += 1
                else:
                    still_active.append(env_id)

            already_complete = []
            for env_id in active:
                if env_id in current:
                    continue
                labels = scene_cfgs[env_id]["labels"]
                if labels.count("Yes") >= self.required_yes and labels.count("No") >= self.required_no:
                    already_complete.append(env_id)
                elif env_id not in rejected:
                    still_active.append(env_id)
            active = still_active + already_complete
            best_env, best_yes, best_no = self._best_progress(scene_cfgs)
            self._log(
                f"[strategy:batch-candidate] idx={setting_idx} active={len(active)} "
                f"Yes={labels_this_setting['Yes']} No={labels_this_setting['No']} "
                f"skip={labels_this_setting['skip']} complete={labels_this_setting['complete']} "
                f"best_env={best_env} best={best_yes}Y/{best_no}N "
                f"deadzone={labels_this_setting['cam_deadzone']} "
                f"agent_view={labels_this_setting['agent_view']} "
                f"quota_full={labels_this_setting['quota_full']} "
                f"cam_red=min/med/max={self._min_med_max_str(pov_red)} "
                f"goal_px=min/med/max={self._min_med_max_str(agent_goal_px)} "
                f"cam_px=min/med/max={self._min_med_max_str(agent_camera_px)} "
                f"settle_dt={settle_dt:.2f}s classify_dt={classify_dt:.2f}s "
                f"dt={time.perf_counter() - t_setting:.2f}s",
                level=2,
            )
            if not current:
                break

        accepted: list[dict[str, Any]] = []
        for env_id, scene_cfg in scene_cfgs.items():
            labels = scene_cfgs[env_id]["labels"]
            if labels.count("Yes") == self.required_yes and labels.count("No") == self.required_no:
                accepted.append({"env_id": env_id, "scene_cfg": scene_cfgs[env_id]})
            else:
                rejected.setdefault(
                    env_id,
                    f"label_balance Yes={labels.count('Yes')} No={labels.count('No')}",
                )
        return accepted, rejected

    def _best_progress(self, scene_cfgs: dict[int, dict[str, Any]]) -> tuple[int, int, int]:
        """Return env id with maximum accepted Yes+No progress."""
        best_env = -1
        best_yes = 0
        best_no = 0
        best_total = -1
        for env_id, scene_cfg in scene_cfgs.items():
            labels = scene_cfg.get("labels", [])
            yes = labels.count("Yes")
            no = labels.count("No")
            total = yes + no
            if total > best_total:
                best_env = env_id
                best_yes = yes
                best_no = no
                best_total = total
        return best_env, best_yes, best_no

    def _selected_pair_distance_ok(
        self,
        captures: list[dict[str, Any]],
        setting: dict[str, Any],
    ) -> bool:
        """Greedy spacing guard copied from v18 viewpoint selection."""
        if not captures:
            return True
        candidate_midpoint = self._setting_midpoint_xy(setting)
        for capture in captures:
            if torch.norm(candidate_midpoint - self._setting_midpoint_xy(capture["setting"])).item() < (
                self.min_selected_pair_distance
            ):
                return False
        return True

    def _setting_midpoint_xy(self, setting: dict[str, Any]) -> torch.Tensor:
        """Return camera-goal pair midpoint in XY."""
        cam = torch.tensor(setting["camera_pos"][:2], dtype=torch.float32, device=self.device)
        goal = torch.tensor(setting["goal_pos"][:2], dtype=torch.float32, device=self.device)
        return (cam + goal) / 2.0

    def _commit_strategy_scene(self, scene_id: int, payload: dict[str, Any]) -> None:
        """Save a previously generated accepted scene payload to disk and labels."""
        env_id = int(payload["env_id"])
        scene_cfg = payload["scene_cfg"]
        labels = scene_cfg["labels"]
        saved_records = []
        for output_idx, capture in enumerate(scene_cfg["captures"]):
            meta = {k: v for k, v in capture.items() if k not in {"rgb", "semantic", "cam_semantic"}}
            saved_records.append(self._save_strategy_setting(
                env_id=env_id,
                scene_id=scene_id,
                setting_idx=output_idx,
                label=str(capture["label"]),
                meta=meta,
                rgb_data=capture["rgb"],
                semantic_data=capture["semantic"],
                cam_semantic_data=capture["cam_semantic"],
            ))

        self.scene_records[str(scene_id)] = {
            "scene_id": scene_id,
            "env_id": env_id,
            "labels": {"Yes": labels.count("Yes"), "No": labels.count("No")},
            "agent_pose": self._pose_to_list(scene_cfg["agent_pos"], scene_cfg["agent_quat"]),
            "occluder_pose": self._pose_to_list(scene_cfg["occluder_pos"], scene_cfg["occluder_quat"]),
            "occluder_scale": scene_cfg.get("occluder_scale", [1.0, 1.0, 1.0]),
            "rail_axis": scene_cfg["rail_axis"].detach().cpu().tolist(),
            "rail_normal": scene_cfg["rail_normal"].detach().cpu().tolist(),
            "candidate_bank_size": len(scene_cfg.get("settings", [])),
            "axis_valid_candidate_count": int(scene_cfg.get("axis_valid_candidate_count", 0)),
            "active_vpt_indices": self._strategy_active_indices(),
            "distractor_count": len(scene_cfg.get("distractor_poses", {})),
            "distractor_scales": scene_cfg.get("distractor_scales", {}),
            "skipped_candidates": scene_cfg.get("skipped_candidates", []),
            "settings": saved_records,
        }
        self._save_strategy_env_config(scene_id, env_id, scene_cfg, saved_records)

    def _generate_strategy_scene(self, env_id: int, scene_id: int) -> bool:
        """Try one complete VPT-Strategy scene and save it if it has a 5/5 split."""
        t_scene = time.perf_counter()
        t0 = time.perf_counter()
        scene_cfg = self._sample_strategy_base(env_id)
        self._log(f"[strategy:gen] scene={scene_id} sample_base dt={time.perf_counter() - t0:.2f}s", level=3)
        t0 = time.perf_counter()
        agent_pose = self._sample_strategy_agent_pose(env_id, scene_cfg)
        if agent_pose is None:
            return False
        self._log(f"[strategy:gen] scene={scene_id} sample_agent dt={time.perf_counter() - t0:.2f}s", level=3)
        scene_cfg["agent_pos"], scene_cfg["agent_quat"] = agent_pose

        t0 = time.perf_counter()
        settings = self._build_parallel_rail_settings(scene_cfg)
        self._log(f"[strategy:gen] scene={scene_id} build_settings n={len(settings)} dt={time.perf_counter() - t0:.2f}s", level=3)
        captures: list[dict[str, Any]] = []
        labels: list[str] = []

        for setting_idx, setting in enumerate(settings):
            t_setting = time.perf_counter()
            self._apply_strategy_setting(env_id, setting, scene_cfg)
            self._orient_camera_to_goal(env_id)
            t_settle = time.perf_counter()
            self._settle_and_update_cameras(self.settle_steps)
            settle_dt = time.perf_counter() - t_settle

            cam_label = self._classify_camera_pov(env_id)
            if cam_label is None:
                red_count = self._count_red_pixels(
                    self._occlusion_camera.data.output["semantic_segmentation"][env_id]
                )
                self._log(
                    f"[strategy:setting] scene={scene_id} idx={setting_idx} "
                    f"REJECT cam_deadzone red={red_count} settle_dt={settle_dt:.2f}s"
                )
                return False
            label, reason, red_count = cam_label

            agent_ok, goal_px, camera_px = self._validate_agent_view(env_id)
            if not agent_ok:
                self._log(
                    f"[strategy:setting] scene={scene_id} idx={setting_idx} "
                    f"REJECT agent_view label={label} cam_red={red_count} "
                    f"goal_px={goal_px} camera_px={camera_px} settle_dt={settle_dt:.2f}s"
                )
                return False

            captures.append({
                "setting_idx": setting_idx,
                "setting": setting,
                "label": label,
                "reason": reason,
                "camera_pov_red_count": red_count,
                "agent_goal_px": goal_px,
                "agent_camera_px": camera_px,
                "rgb": self._rgb_tiled_camera.data.output["rgb"][env_id].clone(),
                "semantic": self._rgb_tiled_camera.data.output["semantic_segmentation"][env_id].clone(),
                "cam_semantic": self._occlusion_camera.data.output["semantic_segmentation"][env_id].clone(),
            })
            labels.append(label)
            self._log(
                f"[strategy:setting] scene={scene_id} idx={setting_idx} "
                f"label={label} cam_red={red_count} goal_px={goal_px} "
                f"camera_px={camera_px} settle_dt={settle_dt:.2f}s "
                f"dt={time.perf_counter() - t_setting:.2f}s",
                level=2,
            )

        if labels.count("Yes") != self.required_yes or labels.count("No") != self.required_no:
            self._log(
                f"[strategy:gen] scene={scene_id} REJECT label_balance "
                f"Yes={labels.count('Yes')} No={labels.count('No')} "
                f"dt={time.perf_counter() - t_scene:.2f}s"
            )
            return False

        t0 = time.perf_counter()
        saved_records = []
        for capture in captures:
            meta = {k: v for k, v in capture.items() if k not in {"rgb", "semantic", "cam_semantic"}}
            saved_records.append(self._save_strategy_setting(
                env_id=env_id,
                scene_id=scene_id,
                setting_idx=int(capture["setting_idx"]),
                label=str(capture["label"]),
                meta=meta,
                rgb_data=capture["rgb"],
                semantic_data=capture["semantic"],
                cam_semantic_data=capture["cam_semantic"],
            ))
        self._log(f"[strategy:gen] scene={scene_id} save_images dt={time.perf_counter() - t0:.2f}s", level=2)

        self.scene_records[str(scene_id)] = {
            "scene_id": scene_id,
            "env_id": env_id,
            "labels": {"Yes": labels.count("Yes"), "No": labels.count("No")},
            "agent_pose": self._pose_to_list(scene_cfg["agent_pos"], scene_cfg["agent_quat"]),
            "occluder_pose": self._pose_to_list(scene_cfg["occluder_pos"], scene_cfg["occluder_quat"]),
            "rail_axis": scene_cfg["rail_axis"].detach().cpu().tolist(),
            "rail_normal": scene_cfg["rail_normal"].detach().cpu().tolist(),
            "settings": saved_records,
        }
        self._save_strategy_env_config(scene_id, env_id, scene_cfg, saved_records)
        self._log(
            f"[strategy:gen] scene={scene_id} ACCEPT "
            f"Yes={labels.count('Yes')} No={labels.count('No')} "
            f"dt={time.perf_counter() - t_scene:.2f}s"
        )
        return True

    def _sample_strategy_base_batch(self, env_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Sample scene bases for many envs with vectorized axis scoring."""
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        origins = self.scene.env_origins[ids].clone()
        safe = float(self.center_to_boundary.item()) - 4.0
        centers = origins[:, :2] + sample_uniform(
            -safe * 0.35, safe * 0.35, (len(env_ids), 2), self.device
        )
        axes, normals, valid_counts = self._select_strategy_axis_batch(origins, centers)

        occ_yaw = sample_uniform(-math.pi, math.pi, (len(env_ids),), self.device)
        zero = torch.zeros_like(occ_yaw)
        occ_quats = quat_from_euler_xyz(zero, zero, occ_yaw)
        scene_cfgs: dict[int, dict[str, Any]] = {}
        for row, env_id in enumerate(env_ids):
            occ_scale = self._sample_occluder_scale()
            occ_idx = self._strategy_active_indices()[0]
            occ_pos = torch.cat([centers[row], origins[row, 2].view(1)])
            scene_cfg = {
                "origin": origins[row],
                "center": centers[row],
                "rail_axis": axes[row],
                "rail_normal": normals[row],
                "axis_valid_candidate_count": int(valid_counts[row].item()),
                "occluder_pos": occ_pos,
                "occluder_quat": occ_quats[row],
                "occluder_scale": occ_scale,
            }
            scene_cfg["settings"] = self._build_parallel_rail_settings(scene_cfg)
            scene_cfgs[env_id] = scene_cfg
        return scene_cfgs

    def _select_strategy_axis_batch(
        self,
        origins: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose best rail axis per env in one tensor pass."""
        batch = origins.shape[0]
        thetas = sample_uniform(-math.pi, math.pi, (batch, self.axis_candidates), self.device)
        axes = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)
        normals = torch.stack([-torch.sin(thetas), torch.cos(thetas)], dim=-1)
        t_values = torch.linspace(-self.rail_extent, self.rail_extent, self.candidate_points, device=self.device)

        midpoints = centers[:, None, None, :] + t_values[None, None, :, None] * axes[:, :, None, :]
        cam_xy = midpoints - normals[:, :, None, :] * (self.camera_goal_distance / 2.0)
        goal_xy = midpoints + normals[:, :, None, :] * (self.camera_goal_distance / 2.0)
        safe_mask = self._safe_camera_goal_pair_mask_multi(origins, centers, cam_xy, goal_xy)
        counts = safe_mask.sum(dim=2)
        best_idx = counts.argmax(dim=1)
        rows = torch.arange(batch, device=self.device)
        return axes[rows, best_idx], normals[rows, best_idx], counts[rows, best_idx]

    def _safe_camera_goal_pair_mask_multi(
        self,
        origins: torch.Tensor,
        centers: torch.Tensor,
        cam_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized safety mask for shape [B, A, C, 2] candidate pairs."""
        limit = float(self.center_to_boundary.item()) - 1.0
        rel_cam = cam_xy - origins[:, None, None, :2]
        rel_goal = goal_xy - origins[:, None, None, :2]
        in_bounds = (
            (rel_cam.abs() <= limit).all(dim=-1)
            & (rel_goal.abs() <= limit).all(dim=-1)
        )
        d_cam_occ = torch.norm(cam_xy - centers[:, None, None, :], dim=-1)
        d_goal_occ = torch.norm(goal_xy - centers[:, None, None, :], dim=-1)
        d_cam_goal = torch.norm(cam_xy - goal_xy, dim=-1)
        clear = (
            (d_cam_occ >= self.min_occluder_clearance)
            & (d_goal_occ >= self.min_occluder_clearance)
            & (d_cam_goal >= self.min_camera_goal_clearance)
            & (d_cam_goal <= self.max_camera_goal_distance)
        )
        return in_bounds & clear

    def _sample_strategy_agent_pose_batch(
        self,
        scene_cfgs: dict[int, dict[str, Any]],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor] | None]:
        """Sample fixed observer poses biased to see the full camera/goal rail."""
        out: dict[int, tuple[torch.Tensor, torch.Tensor] | None] = {}
        random_samples = int(os.getenv("STRATEGY_AGENT_POSE_CANDIDATES", "128"))
        for env_id, scene_cfg in scene_cfgs.items():
            origin = scene_cfg["origin"]
            center = scene_cfg["center"]
            axis = scene_cfg["rail_axis"]
            normal = scene_cfg["rail_normal"]
            representative_cam = center - normal * (self.camera_goal_distance / 2.0)
            representative_goal = center + normal * (self.camera_goal_distance / 2.0)
            candidate_cam_xy = torch.tensor(
                [setting["camera_pos"][:2] for setting in scene_cfg.get("settings", [])],
                dtype=torch.float32,
                device=self.device,
            )
            candidate_goal_xy = torch.tensor(
                [setting["goal_pos"][:2] for setting in scene_cfg.get("settings", [])],
                dtype=torch.float32,
                device=self.device,
            )

            safe = float(self.center_to_boundary.item()) - 2.0
            side_distances = torch.tensor(
                [
                    self.agent_observer_distance,
                    -self.agent_observer_distance,
                    self.agent_observer_distance * 0.8,
                    -self.agent_observer_distance * 0.8,
                    self.agent_observer_distance * 1.2,
                    -self.agent_observer_distance * 1.2,
                ],
                device=self.device,
                dtype=torch.float32,
            )
            jitter = sample_uniform(
                -self.agent_observer_jitter,
                self.agent_observer_jitter,
                (len(side_distances), 2),
                self.device,
            )
            structured_xy = (
                center.unsqueeze(0)
                + side_distances.unsqueeze(1) * axis.unsqueeze(0)
                + jitter[:, 0:1] * axis.unsqueeze(0)
                + jitter[:, 1:2] * normal.unsqueeze(0)
            )
            random_offsets = sample_uniform(-safe, safe, (random_samples, 2), self.device)
            random_xy = origin[:2].unsqueeze(0) + random_offsets
            agent_xy = torch.cat([structured_xy, random_xy], dim=0)

            rel = agent_xy - origin[:2].unsqueeze(0)
            in_bounds = (rel.abs() <= safe).all(dim=1)
            valid = self._passes_strategy_agent_deadzone_batch(
                agent_xy, representative_cam, representative_goal
            )
            valid &= in_bounds

            d_cam = torch.norm(agent_xy - representative_cam.unsqueeze(0), dim=1)
            d_goal = torch.norm(agent_xy - representative_goal.unsqueeze(0), dim=1)
            d_occ = torch.norm(agent_xy - center.unsqueeze(0), dim=1)
            valid &= (torch.minimum(torch.minimum(d_cam, d_goal), d_occ) >= self.min_agent_clearance)

            if len(candidate_cam_xy) > 0:
                min_cam = torch.cdist(agent_xy.unsqueeze(0), candidate_cam_xy.unsqueeze(0)).squeeze(0).min(dim=1).values
                min_goal = torch.cdist(agent_xy.unsqueeze(0), candidate_goal_xy.unsqueeze(0)).squeeze(0).min(dim=1).values
                valid &= (torch.minimum(min_cam, min_goal) >= self.min_agent_clearance)

            valid_idx = torch.where(valid)[0]
            if len(valid_idx) == 0:
                out[env_id] = None
                continue

            chosen_xy = agent_xy[valid_idx[0]]
            target = self._agent_look_target(scene_cfg)
            yaw = torch.atan2(target[1] - chosen_xy[1], target[0] - chosen_xy[0])
            pos = torch.stack([chosen_xy[0], chosen_xy[1], origin[2] + self.agent_height])
            out[env_id] = (pos, self._yaw_to_quat(float(yaw.item())))
        return out

    def _agent_look_target(self, scene_cfg: dict[str, Any]) -> torch.Tensor:
        """Aim observer at the mean of candidate camera/goal positions."""
        settings = scene_cfg.get("settings", [])
        if not settings:
            return scene_cfg["center"]
        cam_xy = torch.tensor(
            [setting["camera_pos"][:2] for setting in settings],
            dtype=torch.float32,
            device=self.device,
        )
        goal_xy = torch.tensor(
            [setting["goal_pos"][:2] for setting in settings],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.cat([cam_xy, goal_xy], dim=0).mean(dim=0)

    def _passes_strategy_agent_deadzone_batch(
        self,
        agent_xy: torch.Tensor,
        camera_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Batch observer angular-deadzone check for one camera/goal axis.

        ``axis_angle`` is folded to 0..90 degrees:
        0 means parallel/anti-parallel to camera-goal line, 90 means perpendicular.
        Valid points must be outside both forbidden bands.
        """
        axis = goal_xy - camera_xy
        midpoint = (camera_xy + goal_xy) / 2.0
        rel = agent_xy - midpoint.unsqueeze(0)
        axis_norm = torch.norm(axis).clamp_min(1e-6)
        rel_norm = torch.norm(rel, dim=1).clamp_min(1e-6)
        cos_abs = torch.abs((rel * axis.unsqueeze(0)).sum(dim=1) / (rel_norm * axis_norm))
        axis_angle = torch.rad2deg(torch.acos(cos_abs.clamp(0.0, 1.0)))
        perp_angle = torch.abs(90.0 - axis_angle)
        return (axis_angle >= self.agent_deadzone_deg) & (perp_angle >= self.agent_perp_deadzone_deg)

    def _sample_strategy_base(self, env_id: int) -> dict[str, Any]:
        """Sample fixed occluder and rail geometry for one scene."""
        origin = self.scene.env_origins[env_id].clone()
        safe = float(self.center_to_boundary.item()) - 4.0
        center_offset = sample_uniform(-safe * 0.35, safe * 0.35, (2,), self.device)
        center = origin[:2] + center_offset

        occluder_pos = torch.stack([center[0], center[1], origin[2] + 0.5])
        occluder_quat = self._yaw_to_quat(float(sample_uniform(-math.pi, math.pi, (1,), self.device).item()))
        rail_axis, rail_normal, valid_count = self._select_strategy_axis(origin, center)

        return {
            "origin": origin,
            "center": center,
            "rail_axis": rail_axis,
            "rail_normal": rail_normal,
            "axis_valid_candidate_count": valid_count,
            "occluder_pos": occluder_pos,
            "occluder_quat": occluder_quat,
        }

    def _sample_strategy_agent_pose(
        self, env_id: int, scene_cfg: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample a fixed observer pose satisfying clearance and angular-deadzone rules."""
        origin = scene_cfg["origin"]
        center = scene_cfg["center"]
        normal = scene_cfg["rail_normal"]
        representative_cam = center - normal * (self.camera_goal_distance / 2.0)
        representative_goal = center + normal * (self.camera_goal_distance / 2.0)
        candidate_cam_xy = torch.tensor(
            [setting["camera_pos"][:2] for setting in scene_cfg.get("settings", [])],
            dtype=torch.float32,
            device=self.device,
        )
        candidate_goal_xy = torch.tensor(
            [setting["goal_pos"][:2] for setting in scene_cfg.get("settings", [])],
            dtype=torch.float32,
            device=self.device,
        )

        safe = float(self.center_to_boundary.item()) - 2.0
        for _ in range(100):
            offset = sample_uniform(-safe, safe, (2,), self.device)
            agent_xy = origin[:2] + offset
            if not self._passes_strategy_agent_deadzone(agent_xy, representative_cam, representative_goal):
                continue

            d_cam = torch.norm(agent_xy - representative_cam).item()
            d_goal = torch.norm(agent_xy - representative_goal).item()
            d_occ = torch.norm(agent_xy - center).item()
            if min(d_cam, d_goal, d_occ) < self.min_agent_clearance:
                continue
            if len(candidate_cam_xy) > 0:
                min_cam = torch.norm(candidate_cam_xy - agent_xy.unsqueeze(0), dim=1).min().item()
                min_goal = torch.norm(candidate_goal_xy - agent_xy.unsqueeze(0), dim=1).min().item()
                if min(min_cam, min_goal) < self.min_agent_clearance:
                    continue

            # NOTE/MODIFY: optional distance constraint for strategy observer.
            # dist_to_midpoint = torch.norm(agent_xy - center).item()
            # if not (4.0 <= dist_to_midpoint <= 12.0):
            #     continue

            look_at = center
            yaw = math.atan2(float(look_at[1] - agent_xy[1]), float(look_at[0] - agent_xy[0]))
            yaw += random.uniform(-math.radians(8.0), math.radians(8.0))
            pos = torch.stack([agent_xy[0], agent_xy[1], origin[2] + self.agent_height])
            return pos, self._yaw_to_quat(yaw)

        return None

    def _passes_strategy_agent_deadzone(
        self, agent_xy: torch.Tensor, camera_xy: torch.Tensor, goal_xy: torch.Tensor
    ) -> bool:
        """Reject observer locations near camera-goal axis or perpendicular axis."""
        axis = goal_xy - camera_xy
        rel = agent_xy - ((camera_xy + goal_xy) / 2.0)
        if torch.norm(axis).item() < 1e-6 or torch.norm(rel).item() < 1e-6:
            return False
        axis = axis / torch.norm(axis)
        rel = rel / torch.norm(rel)
        cos_abs = min(1.0, max(0.0, abs(float(torch.dot(axis, rel).item()))))
        axis_angle = math.degrees(math.acos(cos_abs))
        perp_angle = abs(90.0 - axis_angle)
        return axis_angle >= self.agent_deadzone_deg and perp_angle >= self.agent_perp_deadzone_deg

    def _build_parallel_rail_settings(self, scene_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Build a safe candidate bank of paired camera/goal poses along rails."""
        center = scene_cfg["center"]
        axis = scene_cfg["rail_axis"]
        normal = scene_cfg["rail_normal"]
        origin_z = float(scene_cfg["origin"][2].item())
        t_values = torch.linspace(-self.rail_extent, self.rail_extent, self.candidate_points, device=self.device)
        midpoints = center.unsqueeze(0) + t_values.unsqueeze(1) * axis.unsqueeze(0)
        cam_xy_all = midpoints - normal.unsqueeze(0) * (self.camera_goal_distance / 2.0)
        goal_xy_all = midpoints + normal.unsqueeze(0) * (self.camera_goal_distance / 2.0)
        safe_mask = self._safe_camera_goal_pair_mask(
            scene_cfg["origin"], center, cam_xy_all, goal_xy_all
        )
        line_dists = self._line_point_distance_tensor(cam_xy_all, goal_xy_all, center)
        settings = []
        for idx in torch.where(safe_mask)[0].detach().cpu().tolist():
            cam_xy = cam_xy_all[idx]
            goal_xy = goal_xy_all[idx]
            settings.append({
                "candidate_idx": idx,
                "camera_pos": [float(cam_xy[0]), float(cam_xy[1]), origin_z + 1.0],
                "goal_pos": [float(goal_xy[0]), float(goal_xy[1]), origin_z + self.goal_radius],
                "occluder_line_dist": float(line_dists[idx].item()),
            })
        return settings

    def _select_strategy_axis(self, origin: torch.Tensor, center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Choose the rail axis with the most safe candidate camera/goal pairs."""
        thetas = sample_uniform(-math.pi, math.pi, (self.axis_candidates,), self.device)
        best_axis = torch.tensor([1.0, 0.0], device=self.device)
        best_normal = torch.tensor([0.0, 1.0], device=self.device)
        best_count = -1
        for theta in thetas:
            axis = torch.stack([torch.cos(theta), torch.sin(theta)])
            normal = torch.stack([-torch.sin(theta), torch.cos(theta)])
            count = self._count_safe_axis_candidates(origin, center, axis, normal)
            if count > best_count:
                best_axis = axis
                best_normal = normal
                best_count = count
        return best_axis, best_normal, best_count

    def _count_safe_axis_candidates(
        self,
        origin: torch.Tensor,
        center: torch.Tensor,
        axis: torch.Tensor,
        normal: torch.Tensor,
    ) -> int:
        """Vectorized geometry precheck for one candidate rail axis."""
        t_values = torch.linspace(-self.rail_extent, self.rail_extent, self.candidate_points, device=self.device)
        midpoints = center.unsqueeze(0) + t_values.unsqueeze(1) * axis.unsqueeze(0)
        cam_xy = midpoints - normal.unsqueeze(0) * (self.camera_goal_distance / 2.0)
        goal_xy = midpoints + normal.unsqueeze(0) * (self.camera_goal_distance / 2.0)
        safe_mask = self._safe_camera_goal_pair_mask(origin, center, cam_xy, goal_xy)
        return int(safe_mask.sum().item())

    def _safe_camera_goal_pair_mask(
        self,
        origin: torch.Tensor,
        center: torch.Tensor,
        cam_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Return mask for in-bounds, non-clashing camera/goal pairs."""
        limit = float(self.center_to_boundary.item()) - 1.0
        rel_cam = cam_xy - origin[:2].unsqueeze(0)
        rel_goal = goal_xy - origin[:2].unsqueeze(0)
        in_bounds = (
            (rel_cam.abs() <= limit).all(dim=1)
            & (rel_goal.abs() <= limit).all(dim=1)
        )
        d_cam_occ = torch.norm(cam_xy - center.unsqueeze(0), dim=1)
        d_goal_occ = torch.norm(goal_xy - center.unsqueeze(0), dim=1)
        d_cam_goal = torch.norm(cam_xy - goal_xy, dim=1)
        clear = (
            (d_cam_occ >= self.min_occluder_clearance)
            & (d_goal_occ >= self.min_occluder_clearance)
            & (d_cam_goal >= self.min_camera_goal_clearance)
            & (d_cam_goal <= self.max_camera_goal_distance)
        )

        return in_bounds & clear

    def _is_safe_camera_goal_pair(
        self,
        origin: torch.Tensor,
        center: torch.Tensor,
        cam_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> bool:
        """Scalar wrapper around vectorized camera/goal safety checks."""
        mask = self._safe_camera_goal_pair_mask(
            origin,
            center,
            cam_xy.unsqueeze(0),
            goal_xy.unsqueeze(0),
        )
        return bool(mask.item())

    def _line_point_distance(self, start_xy: torch.Tensor, end_xy: torch.Tensor, point_xy: torch.Tensor) -> float:
        """Distance from point to finite 2D segment."""
        seg = end_xy - start_xy
        seg_len2 = torch.dot(seg, seg).clamp_min(1e-6)
        proj = (torch.dot(point_xy - start_xy, seg) / seg_len2).clamp(0.0, 1.0)
        closest = start_xy + proj * seg
        return float(torch.norm(closest - point_xy).item())

    def _line_point_distance_tensor(
        self,
        start_xy: torch.Tensor,
        end_xy: torch.Tensor,
        point_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized distance from point to finite 2D segments."""
        seg = end_xy - start_xy
        seg_len2 = (seg * seg).sum(dim=1).clamp_min(1e-6)
        proj = ((point_xy.unsqueeze(0) - start_xy) * seg).sum(dim=1) / seg_len2
        proj = proj.clamp(0.0, 1.0)
        closest = start_xy + proj.unsqueeze(1) * seg
        return torch.norm(closest - point_xy.unsqueeze(0), dim=1)

    def _apply_strategy_setting(self, env_id: int, setting: dict[str, Any], scene_cfg: dict[str, Any]) -> None:
        """Write all fixed and setting-specific poses to simulation."""
        ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
        zero = torch.zeros((1, 6), device=self.device)

        agent_pose = torch.cat([scene_cfg["agent_pos"], scene_cfg["agent_quat"]]).unsqueeze(0)
        goal_pos = torch.tensor(setting["goal_pos"], dtype=torch.float32, device=self.device)
        goal_pose = torch.cat([goal_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)]).unsqueeze(0)
        cam_pos = torch.tensor(setting["camera_pos"], dtype=torch.float32, device=self.device)
        cam_pose = torch.cat([cam_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)]).unsqueeze(0)

        self._agent.write_root_com_pose_to_sim(agent_pose, ids)
        self._agent.write_root_com_velocity_to_sim(zero, ids)
        self._goal.write_root_com_pose_to_sim(goal_pose, ids)
        self._goal.write_root_com_velocity_to_sim(zero, ids)
        self._camera_obj.write_root_com_pose_to_sim(cam_pose, ids)
        self._camera_obj.write_root_com_velocity_to_sim(zero, ids)

        vpt_state = torch.zeros((1, self.num_objs, 7), device=self.device)
        vpt_state[:, :, 0:3] = self.storage_position
        vpt_state[:, :, 3] = 1.0
        active_indices = self._strategy_active_indices()
        occ_idx = active_indices[0]
        vpt_state[0, occ_idx, :3] = scene_cfg["occluder_pos"]
        vpt_state[0, occ_idx, 3:7] = scene_cfg["occluder_quat"]
        scale_map = {occ_idx: scene_cfg.get("occluder_scale", [1.0, 1.0, 1.0])}
        distractor_poses = self._sample_strategy_distractors(scene_cfg, active_indices[1:])
        for obj_idx, pose in distractor_poses.items():
            vpt_state[0, obj_idx, :3] = pose[:3]
            vpt_state[0, obj_idx, 3:7] = pose[3:7]
            scale_map[obj_idx] = scene_cfg.get("distractor_scales", {}).get(obj_idx, [1.0, 1.0, 1.0])
        self._apply_strategy_object_scales([env_id], [scale_map])
        self._vpt_objects.write_object_pose_to_sim(vpt_state, ids)
        self._vpt_objects.write_object_velocity_to_sim(
            torch.zeros((1, self.num_objs, 6), device=self.device), ids
        )

    def _apply_strategy_setting_batch(
        self,
        env_ids: list[int],
        settings: list[dict[str, Any]],
        scene_cfgs: list[dict[str, Any]],
    ) -> None:
        """Write fixed and setting-specific poses for many env slots at once."""
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        n = len(env_ids)
        zero = torch.zeros((n, 6), device=self.device)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        agent_poses = torch.stack([
            torch.cat([scene_cfg["agent_pos"], scene_cfg["agent_quat"]])
            for scene_cfg in scene_cfgs
        ])
        goal_positions = torch.tensor(
            [setting["goal_pos"] for setting in settings],
            dtype=torch.float32,
            device=self.device,
        )
        camera_positions = torch.tensor(
            [setting["camera_pos"] for setting in settings],
            dtype=torch.float32,
            device=self.device,
        )
        goal_poses = torch.cat([goal_positions, identity.expand(n, -1)], dim=1)
        camera_poses = torch.cat([camera_positions, identity.expand(n, -1)], dim=1)

        self._agent.write_root_com_pose_to_sim(agent_poses, ids)
        self._agent.write_root_com_velocity_to_sim(zero, ids)
        self._goal.write_root_com_pose_to_sim(goal_poses, ids)
        self._goal.write_root_com_velocity_to_sim(zero, ids)
        self._camera_obj.write_root_com_pose_to_sim(camera_poses, ids)
        self._camera_obj.write_root_com_velocity_to_sim(zero, ids)

        vpt_state = torch.zeros((n, self.num_objs, 7), device=self.device)
        vpt_state[:, :, 0:3] = self.storage_position
        vpt_state[:, :, 3] = 1.0
        active_indices = self._strategy_active_indices()
        occ_idx = active_indices[0]
        scale_maps = []
        for row, scene_cfg in enumerate(scene_cfgs):
            vpt_state[row, occ_idx, :3] = scene_cfg["occluder_pos"]
            vpt_state[row, occ_idx, 3:7] = scene_cfg["occluder_quat"]
            scale_map = {occ_idx: scene_cfg.get("occluder_scale", [1.0, 1.0, 1.0])}
            if "distractor_poses" not in scene_cfg:
                scene_cfg["distractor_poses"] = self._sample_strategy_distractors(
                    scene_cfg, active_indices[1:]
                )
            for obj_idx, pose in scene_cfg["distractor_poses"].items():
                vpt_state[row, obj_idx, :3] = pose[:3]
                vpt_state[row, obj_idx, 3:7] = pose[3:7]
                scale_map[obj_idx] = scene_cfg.get("distractor_scales", {}).get(obj_idx, [1.0, 1.0, 1.0])
            scale_maps.append(scale_map)
        self._apply_strategy_object_scales(env_ids, scale_maps)
        self._vpt_objects.write_object_pose_to_sim(vpt_state, ids)
        self._vpt_objects.write_object_velocity_to_sim(
            torch.zeros((n, self.num_objs, 6), device=self.device), ids
        )

    def _strategy_active_indices(self) -> list[int]:
        """Return deterministic active VPT indices with occluder first."""
        occ_idx = max(0, min(self.occluder_index, self.num_objs - 1))
        indices = [occ_idx]
        for idx in range(self.num_objs):
            if idx != occ_idx:
                indices.append(idx)
            if len(indices) >= self.active_vpt_objs:
                break
        return indices

    def _sample_strategy_distractors(
        self,
        scene_cfg: dict[str, Any],
        obj_indices: list[int],
    ) -> dict[int, torch.Tensor]:
        """Scatter non-occluder active objects safely as VPT distractors."""
        poses: dict[int, torch.Tensor] = {}
        if not obj_indices:
            return poses
        origin = scene_cfg["origin"]
        center = scene_cfg["center"]
        safe = float(self.center_to_boundary.item()) - 2.0
        reserved = [center]
        settings = scene_cfg.get("settings", [])
        if settings:
            for setting in settings[::max(1, len(settings) // 10)]:
                reserved.append(torch.tensor(setting["camera_pos"][:2], dtype=torch.float32, device=self.device))
                reserved.append(torch.tensor(setting["goal_pos"][:2], dtype=torch.float32, device=self.device))

        for obj_idx in obj_indices:
            placed = None
            scale = [
                random.uniform(self.distractor_scale_min, self.distractor_scale_max),
                random.uniform(self.distractor_scale_min, self.distractor_scale_max),
                random.uniform(self.distractor_scale_min, self.distractor_scale_max),
            ]
            for _ in range(64):
                offset = sample_uniform(-safe, safe, (2,), self.device)
                xy = origin[:2] + offset
                if any(torch.norm(xy - p).item() < self.distractor_clearance for p in reserved):
                    continue
                yaw = sample_uniform(-math.pi, math.pi, (1,), self.device)[0]
                pos = torch.stack([xy[0], xy[1], origin[2]])
                quat = self._yaw_to_quat(float(yaw.item()))
                placed = torch.cat([pos, quat])
                reserved.append(xy)
                break
            if placed is not None:
                poses[obj_idx] = placed
                scene_cfg.setdefault("distractor_scales", {})[obj_idx] = scale
        return poses

    def _sample_occluder_scale(self) -> list[float]:
        """Sample a tall occluder scale to widen the strategy No band."""
        xy = random.uniform(self.occluder_scale_xy_min, self.occluder_scale_xy_max)
        z = random.uniform(self.occluder_scale_z_min, self.occluder_scale_z_max)
        return [xy, xy, z]

    def _strategy_local_z_offset(self, obj_idx: int, scale: list[float]) -> float:
        """Return v18-style local Z offset for object pivot/shape."""
        obj_cfg = list(self.cfg.vpt_objects.rigid_objects.values())[obj_idx]
        spawn_cfg = obj_cfg.spawn
        base_h = float(self.vpt_base_dims[obj_idx, 2].item())
        s_z = float(scale[2])

        if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
            # v18 USD assets are treated as base-pivoted: local z offset stays 0.
            return 0.0
        if isinstance(spawn_cfg, sim_utils.MeshCuboidCfg):
            return 0.5 * base_h * s_z
        if isinstance(spawn_cfg, sim_utils.MeshCylinderCfg):
            return 0.5 * base_h * s_z
        if isinstance(spawn_cfg, sim_utils.MeshConeCfg):
            # v18 cones use base pivot / zero local z.
            return 0.0
        return 0.0

    def _apply_strategy_object_scales(
        self,
        env_ids: list[int],
        scale_maps: list[dict[int, list[float]]],
    ) -> None:
        """Apply v18-style world-paused scale and local-Z grounding edits."""
        world = World.instance()
        was_playing = False
        if world is not None:
            try:
                was_playing = world.is_playing()
                if was_playing:
                    world.pause()
            except Exception:
                was_playing = False

        stage = get_current_stage()
        with Sdf.ChangeBlock():
            for env_id, scale_map in zip(env_ids, scale_maps):
                for obj_idx, scale in scale_map.items():
                    prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/obs_{obj_idx}")
                    if not prim.IsValid():
                        continue

                    xform = UsdGeom.Xformable(prim)
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

                    scale_op.Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
                    current = translate_op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
                    translate_op.Set(Gf.Vec3d(
                        float(current[0]),
                        float(current[1]),
                        self._strategy_local_z_offset(obj_idx, scale),
                    ))

        if world is not None and was_playing:
            try:
                world.play()
                self.sim.step(render=False)
            except Exception:
                pass

    def _orient_camera_to_goal(self, env_id: int) -> None:
        """Rotate camera object toward target, preserving v18 +90 degree sensor convention."""
        ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
        cam_pos = self._camera_obj.data.root_pos_w[env_id]
        goal_pos = self._goal.data.root_pos_w[env_id]
        direction = goal_pos[:2] - cam_pos[:2]
        yaw = math.atan2(float(direction[1]), float(direction[0])) - self.camera_yaw_correction_rad
        quat = quat_from_euler_xyz(
            torch.tensor(-math.radians(self.agent_camera_pitch), device=self.device),
            torch.tensor(0.0, device=self.device),
            torch.tensor(yaw, device=self.device),
        )
        pose = torch.cat([cam_pos, quat]).unsqueeze(0)
        self._camera_obj.write_root_com_pose_to_sim(pose, ids)
        self._camera_obj.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=self.device), ids)
        self._update_camera_poses(ids)

    def _orient_camera_to_goal_batch(self, env_ids: list[int]) -> None:
        """Rotate many camera objects toward their targets in one write."""
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        cam_pos = self._camera_obj.data.root_pos_w[ids]
        goal_pos = self._goal.data.root_pos_w[ids]
        direction = goal_pos[:, :2] - cam_pos[:, :2]
        yaw = torch.atan2(direction[:, 1], direction[:, 0]) - self.camera_yaw_correction_rad
        roll = torch.full_like(yaw, -math.radians(self.agent_camera_pitch))
        pitch = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        pose = torch.cat([cam_pos, quat], dim=1)
        self._camera_obj.write_root_com_pose_to_sim(pose, ids)
        self._camera_obj.write_root_com_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), ids
        )
        self._update_camera_poses(ids)

    def _settle_and_update_cameras(self, steps: int = 30) -> None:
        """Step simulation/render enough frames for stable lighting and sensors."""
        self.scene.write_data_to_sim()
        for _ in range(max(1, steps)):
            self.sim.step(render=True)
            self.scene.update(dt=self.step_dt)
            self._rgb_tiled_camera.update(self.sim.cfg.dt)
            self._occlusion_camera.update(self.sim.cfg.dt)

    def _classify_camera_pov(self, env_id: int) -> tuple[str, str, int] | None:
        """Classify camera semantic view with a deadzone between No and Yes."""
        sem = self._occlusion_camera.data.output["semantic_segmentation"][env_id]
        red_count = self._count_red_pixels(sem)
        if red_count > self.cam_red_thresh:
            return "Yes", "visible", red_count
        if red_count <= self.cam_no_red_max:
            return "No", "not_visible", red_count
        return None

    def _classify_camera_pov_batch(
        self,
        env_ids: list[int],
    ) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor]:
        """Classify camera semantic views for many envs at once."""
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        sem = self._occlusion_camera.data.output["semantic_segmentation"][ids]
        red_counts = self._count_red_pixels_tensor(sem)
        yes = red_counts > self.cam_red_thresh
        no = red_counts <= self.cam_no_red_max
        valid = yes | no
        labels = ["Yes" if bool(v) else "No" for v in yes.detach().cpu().tolist()]
        reasons = ["visible" if bool(v) else "not_visible" for v in yes.detach().cpu().tolist()]
        return labels, reasons, red_counts, valid

    def _validate_agent_view(self, env_id: int) -> tuple[bool, int, int]:
        """Require agent semantic view to see both target and camera object."""
        sem = self._rgb_tiled_camera.data.output["semantic_segmentation"][env_id]
        red_count = self._count_red_pixels(sem)
        green_count = self._count_green_pixels(sem)
        ok = red_count >= self.goal_pixel_threshold and green_count >= self.camera_pixel_threshold
        return ok, red_count, green_count

    def _validate_agent_view_batch(self, env_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Require agent semantic views to see both target and camera object."""
        ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        sem = self._rgb_tiled_camera.data.output["semantic_segmentation"][ids]
        red_counts = self._count_red_pixels_tensor(sem)
        green_counts = self._count_green_pixels_tensor(sem)
        ok = (red_counts >= self.goal_pixel_threshold) & (green_counts >= self.camera_pixel_threshold)
        return ok, red_counts, green_counts

    def _save_strategy_setting(
        self,
        env_id: int,
        scene_id: int,
        setting_idx: int,
        label: str,
        meta: dict[str, Any],
        rgb_data: torch.Tensor,
        semantic_data: torch.Tensor,
        cam_semantic_data: torch.Tensor,
    ) -> dict[str, Any]:
        """Save RGB, agent semantic, camera semantic, and return metadata record."""
        scene_name = f"scene_{scene_id}"
        file_stem = f"setting_{setting_idx:02d}"
        rgb_dir = Path(self.base_path) / "RGB" / label / scene_name
        sem_dir = Path(self.base_path) / "Semantic" / label / scene_name
        cam_dir = Path(self.base_path) / "cam" / label / scene_name
        rgb_dir.mkdir(parents=True, exist_ok=True)
        sem_dir.mkdir(parents=True, exist_ok=True)
        cam_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = rgb_dir / f"{file_stem}.png"
        sem_path = sem_dir / f"{file_stem}.png"
        cam_path = cam_dir / f"{file_stem}_cam_semantic.png"

        self._write_rgb_png(rgb_path, rgb_data)
        self._write_rgb_png(sem_path, semantic_data)
        self._write_rgb_png(cam_path, cam_semantic_data)

        record = {
            "env_id": env_id,
            "scene_id": scene_id,
            "setting_idx": setting_idx,
            "candidate_idx": int(meta.get("candidate_idx", setting_idx)),
            "label": label,
            "reason": meta["reason"],
            "camera_pov_red_count": int(meta["camera_pov_red_count"]),
            "agent_goal_px": int(meta["agent_goal_px"]),
            "agent_camera_px": int(meta["agent_camera_px"]),
            "camera_pos": meta["setting"]["camera_pos"],
            "goal_pos": meta["setting"]["goal_pos"],
            "occluder_line_dist": float(meta["setting"].get("occluder_line_dist", -1.0)),
            "paths": {
                "RGB": str(rgb_path),
                "Semantic": str(sem_path),
                "cam": str(cam_path),
            },
        }
        return record

    def _json_safe(self, value: Any) -> Any:
        """Convert tensors and numpy scalars into JSON-safe values."""
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return value

    def _pose_record(self, position: Any, orientation: Any) -> dict[str, Any]:
        """Return a replay-compatible pose record."""
        return {
            "position": self._json_safe(position),
            "orientation": self._json_safe(orientation),
        }

    def _strategy_object_config_records(self, scene_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Build replay metadata for active strategy VPT objects."""
        records: list[dict[str, Any]] = []
        active_indices = self._strategy_active_indices()
        occ_idx = active_indices[0]

        def add_record(obj_idx: int, pose: torch.Tensor, scale: list[float], role: str) -> None:
            bbox_dims = []
            if hasattr(self, "vpt_base_dims") and obj_idx < len(self.vpt_base_dims):
                bbox_dims = (self.vpt_base_dims[obj_idx] * torch.tensor(scale, device=self.device)).detach().cpu().tolist()
            records.append({
                "index": int(obj_idx),
                "obj_index": int(obj_idx),
                "role": role,
                "position": self._json_safe(pose[:3]),
                "orientation": self._json_safe(pose[3:7]),
                "applied_scale": self._json_safe(scale),
                "dimensions_bbox": bbox_dims,
                "z_offset_ratio": 0.0,
                "shape_id": -1.0,
            })

        occ_pose = torch.cat([scene_cfg["occluder_pos"], scene_cfg["occluder_quat"]])
        add_record(occ_idx, occ_pose, scene_cfg.get("occluder_scale", [1.0, 1.0, 1.0]), "occluder")

        for obj_idx, pose in scene_cfg.get("distractor_poses", {}).items():
            scale = scene_cfg.get("distractor_scales", {}).get(obj_idx, [1.0, 1.0, 1.0])
            add_record(int(obj_idx), pose, scale, "distractor")
        return records

    def _save_strategy_env_config(
        self,
        scene_id: int,
        env_id: int,
        scene_cfg: dict[str, Any],
        saved_records: list[dict[str, Any]],
    ) -> None:
        """Save a replayable config for a full VPT1 strategy scene."""
        if not saved_records:
            return
        first = saved_records[0]
        identity = [1.0, 0.0, 0.0, 0.0]
        labels = scene_cfg.get("labels", [])
        config = {
            "metadata": {
                "env_id": int(env_id),
                "folder_idx": int(scene_id),
                "scene_id": int(scene_id),
                "task": "VPT-v18-strategy",
                "visibility_label": first.get("label", "UNKNOWN"),
                "visibility_reason": first.get("reason", "strategy_first_setting"),
                "cfg_version": "strategy_v1",
            },
            "environment_settings": {
                "boundary_limits": list(self.cfg.boundary_limits),
                "agent_height": float(self.cfg.agent_height),
                "agent_camera_pitch": float(self.cfg.agent_camera_pitch),
                "action_scale": float(self.cfg.action_scale),
                "num_vpt_objs": int(self.cfg.num_vpt_objs),
                "images_per_scene": int(self.images_per_scene),
                "required_yes": int(self.required_yes),
                "required_no": int(self.required_no),
                "human_vpt1_accuracy_range": [76.0, 83.0],
                "vpt1_ft_expected_max_accuracy_note": "Prior VPT1 FT strategy-like checks did not cross 70.",
            },
            "goal_ball": self._pose_record(first["goal_pos"], identity),
            "camera_object": self._pose_record(first["camera_pos"], identity),
            "agent": self._pose_record(scene_cfg["agent_pos"], scene_cfg["agent_quat"]),
            "vpt_objects": {
                "total_count": int(self.num_objs),
                "active_count": int(len(self._strategy_active_indices())),
                "active_indices": self._strategy_active_indices(),
                "objects": self._strategy_object_config_records(scene_cfg),
            },
            "strategy_scene": {
                "labels": {"Yes": labels.count("Yes"), "No": labels.count("No")},
                "origin": self._json_safe(scene_cfg.get("origin")),
                "center": self._json_safe(scene_cfg.get("center")),
                "rail_axis": self._json_safe(scene_cfg.get("rail_axis")),
                "rail_normal": self._json_safe(scene_cfg.get("rail_normal")),
                "axis_valid_candidate_count": int(scene_cfg.get("axis_valid_candidate_count", 0)),
                "candidate_bank_size": len(scene_cfg.get("settings", [])),
                "thresholds": {
                    "cam_red_thresh": int(self.cam_red_thresh),
                    "cam_no_red_max": int(self.cam_no_red_max),
                    "agent_goal_thresh": int(self.goal_pixel_threshold),
                    "agent_camera_thresh": int(self.camera_pixel_threshold),
                    "agent_deadzone_deg": float(self.agent_deadzone_deg),
                    "agent_perp_deadzone_deg": float(self.agent_perp_deadzone_deg),
                },
                "rail": {
                    "rail_extent": float(self.rail_extent),
                    "camera_goal_distance": float(self.camera_goal_distance),
                    "min_selected_pair_distance": float(self.min_selected_pair_distance),
                },
                "settings": self._json_safe(saved_records),
                "skipped_candidates": self._json_safe(scene_cfg.get("skipped_candidates", [])),
            },
            "valid_viewpoints": {"count": 0, "positions": []},
            "collected_viewpoints": {
                "count": len(saved_records),
                "positions": [record["camera_pos"] for record in saved_records],
            },
        }

        config_dir = Path(self.base_path) / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"env_{scene_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _save_strategy_labels(self) -> None:
        """Write scene-level and setting-level labels to strategy_labels.json."""
        out = {
            "metadata": {
                "task": "VPT-v18-strategy",
                "base_path": self.base_path,
                "images_per_scene": self.images_per_scene,
                "required_yes": self.required_yes,
                "required_no": self.required_no,
                "cam_red_thresh": self.cam_red_thresh,
                "cam_no_red_max": self.cam_no_red_max,
                "agent_goal_thresh": self.goal_pixel_threshold,
                "agent_camera_thresh": self.camera_pixel_threshold,
                "agent_deadzone_deg": self.agent_deadzone_deg,
                "agent_perp_deadzone_deg": self.agent_perp_deadzone_deg,
                "rail_extent": self.rail_extent,
                "camera_goal_distance": self.camera_goal_distance,
                "max_camera_goal_distance": self.max_camera_goal_distance,
                "min_selected_pair_distance": self.min_selected_pair_distance,
                "occluder_scale_xy": [self.occluder_scale_xy_min, self.occluder_scale_xy_max],
                "occluder_scale_z": [self.occluder_scale_z_min, self.occluder_scale_z_max],
                "distractor_scale": [self.distractor_scale_min, self.distractor_scale_max],
                "settle_steps": self.settle_steps,
                "next_scene_id": self.next_scene_id,
            },
            "statistics": self._strategy_statistics(),
            "scene_attempt_counts": self.scene_attempt_counts,
            "scenes": self.scene_records,
        }
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        with open(self.strategy_labels_json_path, "w") as f:
            json.dump(out, f, indent=2)

    def _strategy_statistics(self) -> dict[str, Any]:
        """Compute aggregate scene and label counts for saved records."""
        label_counts = {"Yes": 0, "No": 0}
        for scene in self.scene_records.values():
            for setting in scene["settings"]:
                label_counts[setting["label"]] += 1
        return {
            "total_scenes": len(self.scene_records),
            "total_images": sum(label_counts.values()),
            "label_counts": label_counts,
        }

    def _reset_raw_states(self, env_ids: torch.Tensor) -> None:
        """Reset base rigid states and hide all VPT objects before strategy sampling."""
        env_ids = env_ids.view(-1)
        n = len(env_ids)
        zero = torch.zeros((n, 6), device=self.device)
        default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        origin = self.scene.env_origins[env_ids]

        agent_pose = torch.cat([origin + torch.tensor([0.0, 0.0, self.agent_height], device=self.device),
                                default_quat.expand(n, -1)], dim=1)
        goal_pose = torch.cat([origin + torch.tensor([2.0, 0.0, self.goal_radius], device=self.device),
                               default_quat.expand(n, -1)], dim=1)
        cam_pose = torch.cat([origin + torch.tensor([-2.0, 0.0, 1.0], device=self.device),
                              default_quat.expand(n, -1)], dim=1)

        self._agent.write_root_com_pose_to_sim(agent_pose, env_ids)
        self._agent.write_root_com_velocity_to_sim(zero, env_ids)
        self._goal.write_root_com_pose_to_sim(goal_pose, env_ids)
        self._goal.write_root_com_velocity_to_sim(zero, env_ids)
        self._camera_obj.write_root_com_pose_to_sim(cam_pose, env_ids)
        self._camera_obj.write_root_com_velocity_to_sim(zero, env_ids)

        vpt_pose = torch.zeros((n, self.num_objs, 7), device=self.device)
        vpt_pose[:, :, 0:3] = self.storage_position
        vpt_pose[:, :, 3] = 1.0
        self._vpt_objects.write_object_pose_to_sim(vpt_pose, env_ids)
        self._vpt_objects.write_object_velocity_to_sim(torch.zeros((n, self.num_objs, 6), device=self.device), env_ids)

    def _update_camera_poses(self, env_ids: torch.Tensor) -> None:
        """Set camera-POV sensor to camera object pose plus v18 +90 degree yaw offset."""
        env_ids = env_ids.view(-1)
        camera_pos = self._camera_obj.data.root_pos_w[env_ids].clone()
        camera_quat = self._camera_obj.data.root_quat_w[env_ids].clone()
        half_theta = (math.pi / 2.0) / 2.0
        left_90 = torch.tensor([math.cos(half_theta), 0.0, 0.0, math.sin(half_theta)], device=self.device)
        rotated = math_utils.quat_mul(camera_quat, left_90.expand(len(env_ids), -1))
        self._occlusion_camera.set_world_poses(
            positions=camera_pos,
            orientations=rotated,
            env_ids=env_ids.tolist(),
            convention="world",
        )

    def _ensure_base_dims(self) -> None:
        """Cache v18-style base VPT dimensions for scale-aware grounding."""
        if self._base_dims_cached:
            return
        dims = []
        for obj_cfg in self.cfg.vpt_objects.rigid_objects.values():
            spawn_cfg = obj_cfg.spawn
            dim = torch.tensor([1.0, 1.0, 1.0], device=self.device, dtype=torch.float32)
            if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
                scale = getattr(spawn_cfg, "scale", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)
                filename = spawn_cfg.usd_path.split("/")[-1].split(".")[0]
                if filename.endswith(("X", "L", "T", "I", "A", "H", "Z")):
                    dim = torch.tensor(
                        [1.0 * scale[0], 0.25 * scale[1], 1.0 * scale[2]],
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    dim = torch.tensor(scale, device=self.device, dtype=torch.float32)
            elif isinstance(spawn_cfg, sim_utils.MeshCuboidCfg):
                dim = torch.tensor(spawn_cfg.size, device=self.device, dtype=torch.float32)
            elif isinstance(spawn_cfg, (sim_utils.MeshConeCfg, sim_utils.MeshCylinderCfg)):
                dim = torch.tensor(
                    [2 * spawn_cfg.radius, 2 * spawn_cfg.radius, spawn_cfg.height],
                    device=self.device,
                    dtype=torch.float32,
                )
            elif hasattr(spawn_cfg, "size"):
                dim = torch.tensor(spawn_cfg.size, device=self.device, dtype=torch.float32)
            elif hasattr(spawn_cfg, "scale") and spawn_cfg.scale is not None:
                dim = torch.tensor(spawn_cfg.scale, device=self.device, dtype=torch.float32)
            dims.append(dim)
        self.vpt_base_dims = torch.stack(dims) if dims else torch.empty((0, 3), device=self.device)
        self._base_dims_cached = True

    def _randomize_strategy_scene_props(self, env_ids: list[int]) -> None:
        """Randomize v18-style textures and lighting without overwriting strategy scales."""
        if not env_ids:
            return

        active_indices = self._strategy_active_indices()
        vpt_paths = [
            f"/World/envs/env_{env_id}/obs_{obj_idx}"
            for env_id in env_ids
            for obj_idx in active_indices
        ]
        if vpt_paths:
            self.randomize_material(vpt_paths, "vpt")

        floor_paths = [f"/World/envs/env_{env_id}/mat" for env_id in env_ids]
        if floor_paths:
            self.randomize_material(floor_paths, "mat")

        self.randomize_shape_color(prim_path_expr=[
            "/World/envs/env_.*/bottom_wall", "/World/envs/env_.*/right_wall",
            "/World/envs/env_.*/left_wall", "/World/envs/env_.*/top_wall"
        ])

        light_paths = [f"/World/envs/env_{env_id}/Light_A" for env_id in env_ids]
        self.randomize_spherical_lights(light_paths)

    def get_material_configs(self, material_type: str) -> list[sim_utils.MdlFileCfg]:
        """Load a capped random material pool for mat or VPT obstacle surfaces."""
        if material_type == "mat":
            raw_paths = get_mat_material_paths()
            tex_scale = (1000.0, 1000.0)
        elif material_type == "vpt":
            raw_paths = get_vpt_material_paths()
            tex_scale = (2.0, 2.0)
        else:
            raise ValueError(f"Unknown material type: {material_type}")
        selected = random.sample(raw_paths, min(len(raw_paths), 100)) if raw_paths else []
        return [sim_utils.MdlFileCfg(mdl_path=p, project_uvw=True, texture_scale=tex_scale) for p in selected]

    def randomize_material(self, prim_paths: list[str], material_type: str) -> None:
        """Bind a random preloaded material to each prim path."""
        if material_type == "mat":
            material_pool = self.mat_material_paths
        elif material_type == "vpt":
            material_pool = self.vpt_material_paths
        else:
            self._log(f"[strategy:randomize] unknown material_type={material_type}", level=2)
            return
        if not material_pool:
            return
        for prim_path in prim_paths:
            sim_utils.bind_visual_material(prim_path, random.choice(material_pool))

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

    def _count_red_pixels(self, img: torch.Tensor) -> int:
        """Count exact/near red semantic pixels in uint8 or float semantic output."""
        arr = img.detach()
        if arr.shape[-1] == 4:
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        else:
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if arr.dtype == torch.uint8 or float(r.max().item()) > 1.5:
            mask = (r >= 242) & (g <= 13) & (b <= 13)
        else:
            mask = (r >= 0.95) & (g <= 0.05) & (b <= 0.05)
        return int(mask.sum().item())

    def _count_red_pixels_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        """Count red semantic pixels for a batch of images."""
        arr = imgs.detach()
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if arr.dtype == torch.uint8 or float(r.max().item()) > 1.5:
            mask = (r >= 242) & (g <= 13) & (b <= 13)
        else:
            mask = (r >= 0.95) & (g <= 0.05) & (b <= 0.05)
        return mask.flatten(start_dim=1).sum(dim=1)

    def _count_green_pixels(self, img: torch.Tensor) -> int:
        """Count exact/near green semantic pixels in uint8 or float semantic output."""
        arr = img.detach()
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if arr.dtype == torch.uint8 or float(g.max().item()) > 1.5:
            mask = (r <= 13) & (g >= 242) & (b <= 13)
        else:
            mask = (r <= 0.05) & (g >= 0.95) & (b <= 0.05)
        return int(mask.sum().item())

    def _count_green_pixels_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        """Count green semantic pixels for a batch of images."""
        arr = imgs.detach()
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if arr.dtype == torch.uint8 or float(g.max().item()) > 1.5:
            mask = (r <= 13) & (g >= 242) & (b <= 13)
        else:
            mask = (r <= 0.05) & (g >= 0.95) & (b <= 0.05)
        return mask.flatten(start_dim=1).sum(dim=1)

    def _write_rgb_png(self, path: Path, data: torch.Tensor) -> None:
        """Write RGB-like tensor data to PNG using OpenCV."""
        arr = data.detach().cpu().numpy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    def _yaw_to_quat(self, yaw: float) -> torch.Tensor:
        """Create a wxyz yaw-only quaternion."""
        half = yaw / 2.0
        return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], device=self.device, dtype=torch.float32)

    def _pose_to_list(self, pos: torch.Tensor, quat: torch.Tensor) -> list[float]:
        """Convert position/quaternion tensors to one JSON-safe list."""
        return [float(x) for x in torch.cat([pos, quat]).detach().cpu().tolist()]
