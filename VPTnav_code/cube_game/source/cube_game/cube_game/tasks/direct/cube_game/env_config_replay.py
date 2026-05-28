from __future__ import annotations

import json
import os
from typing import Any

import torch


def _pose_from_config(entity: dict[str, Any], origin: torch.Tensor, device: torch.device) -> torch.Tensor:
    if "position" in entity:
        pos = entity["position"]
        quat = entity.get("orientation", entity.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]))
    else:
        pos_local = entity["pos_local"]
        pos = (origin + torch.tensor(pos_local, dtype=torch.float32, device=device)).tolist()
        quat = entity.get("quat_wxyz", entity.get("orientation", [1.0, 0.0, 0.0, 0.0]))
    return torch.tensor([pos + quat], dtype=torch.float32, device=device)


def _object_position(obj_data: dict[str, Any], origin: torch.Tensor, device: torch.device) -> torch.Tensor:
    if "position" in obj_data:
        return torch.tensor(obj_data["position"], dtype=torch.float32, device=device)
    return origin + torch.tensor(obj_data["pos_local"], dtype=torch.float32, device=device)


def load_env_config_from_json(env: Any, config_path: str, target_env_id: int = 0) -> dict[str, Any]:
    """Replay a saved VPT/VPT2/depth/strategy config into one live env slot.

    Supports the original world-coordinate schema and the depth v2 local-coordinate
    schema. Visual material replay is intentionally left to env-specific loaders.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    device = env.device
    env_id = target_env_id.item() if torch.is_tensor(target_env_id) else int(target_env_id)
    env_ids = torch.tensor([env_id], dtype=torch.long, device=device)
    origin = env.scene.env_origins[env_id].to(device=device)

    if "entities" in config:
        entities = config["entities"]
        agent_entity = entities["agent"]
        goal_entity = entities["goal"]
        camera_entity = entities["camera"]
    else:
        agent_entity = config["agent"]
        goal_entity = config["goal_ball"]
        camera_entity = config["camera_object"]

    env._agent.write_root_com_pose_to_sim(_pose_from_config(agent_entity, origin, device), env_ids)
    env._agent.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=device), env_ids)
    env._goal.write_root_com_pose_to_sim(_pose_from_config(goal_entity, origin, device), env_ids)
    env._goal.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=device), env_ids)
    env._camera_obj.write_root_com_pose_to_sim(_pose_from_config(camera_entity, origin, device), env_ids)
    env._camera_obj.write_root_com_velocity_to_sim(torch.zeros((1, 6), device=device), env_ids)

    if hasattr(env, "_ref_objs") and "reference_objects" in config:
        for ref_data in config["reference_objects"].get("objects", []):
            ref_idx = int(ref_data["index"])
            if ref_idx >= len(env._ref_objs):
                continue
            pose = _pose_from_config(ref_data, origin, device)
            env._ref_objs[ref_idx].write_root_com_pose_to_sim(pose, env_ids)
            env._ref_objs[ref_idx].write_root_com_velocity_to_sim(torch.zeros((1, 6), device=device), env_ids)
        if hasattr(env, "active_ref_idx"):
            env.active_ref_idx[env_id] = config["reference_objects"].get("active_index")

    vpt_cfg = config.get("vpt_objects", {})
    active_indices = torch.tensor(vpt_cfg.get("active_indices", []), dtype=torch.long, device=device)
    if hasattr(env, "active_vpt_indices"):
        env.active_vpt_indices[env_id] = active_indices

    vpt_state = env._vpt_objects.data.default_object_state[env_id].clone()
    vpt_state[:, 0] = env.storage_position[0]
    vpt_state[:, 1] = env.storage_position[1]
    vpt_state[:, 2] = env.storage_position[2]
    vpt_state[:, 3:7] = 0.0
    vpt_state[:, 3] = 1.0
    if vpt_state.shape[1] > 7:
        vpt_state[:, 7:] = 0.0

    for obj_data in vpt_cfg.get("objects", []):
        obj_idx = int(obj_data.get("index", obj_data.get("obj_index")))
        vpt_state[obj_idx, :3] = _object_position(obj_data, origin, device)
        vpt_state[obj_idx, 3:7] = torch.tensor(
            obj_data.get("orientation", obj_data.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0])),
            dtype=torch.float32,
            device=device,
        )
        if vpt_state.shape[1] > 7:
            vpt_state[obj_idx, 7:] = 0.0

        if hasattr(env, "all_vpt_dims") and obj_data.get("dimensions_bbox"):
            env.all_vpt_dims[env_id, obj_idx] = torch.tensor(
                obj_data["dimensions_bbox"], dtype=torch.float32, device=device)
        if hasattr(env, "vpt_z_offset_ratios"):
            env.vpt_z_offset_ratios[env_id, obj_idx] = float(obj_data.get("z_offset_ratio", 0.0))
        if hasattr(env, "vpt_shapes"):
            env.vpt_shapes[env_id, obj_idx] = float(obj_data.get("shape_id", -1.0))

    env._vpt_objects.write_object_pose_to_sim(vpt_state[:, :7].unsqueeze(0), env_ids)
    env._vpt_objects.write_object_velocity_to_sim(
        torch.zeros((1, env.num_objs, 6), device=device), env_ids)

    metadata = config.get("metadata", {})
    folder_idx = int(metadata.get("folder_idx", env_id))
    if hasattr(env, "env_visibility_labels"):
        env.env_visibility_labels[folder_idx] = metadata.get("visibility_label", "UNKNOWN")
    if hasattr(env, "env_visibility_reasons"):
        env.env_visibility_reasons[folder_idx] = metadata.get("visibility_reason", "replay")

    if hasattr(env, "valid_viewpoint_poses"):
        viewpoints = config.get("valid_viewpoints", {}).get("positions", [])
        if not viewpoints and "viewpoints" in config:
            local = config["viewpoints"].get("valid_positions_local", [])
            viewpoints = (origin + torch.tensor(local, dtype=torch.float32, device=device)).tolist() if local else []
        env.valid_viewpoint_poses[env_id] = (
            torch.tensor(viewpoints, dtype=torch.float32, device=device)
            if viewpoints else torch.zeros((0, 3), dtype=torch.float32, device=device)
        )

    if hasattr(env, "selected_viewpoints_for_collection"):
        viewpoints = config.get("collected_viewpoints", {}).get("positions", [])
        if not viewpoints and "viewpoints" in config:
            local = config["viewpoints"].get("collected_positions_local", [])
            viewpoints = (origin + torch.tensor(local, dtype=torch.float32, device=device)).tolist() if local else []
        env.selected_viewpoints_for_collection[env_id] = (
            torch.tensor(viewpoints, dtype=torch.float32, device=device) if viewpoints else None
        )

    env.scene.write_data_to_sim()
    for _ in range(3):
        env.sim.step(render=False)
    env.scene.update(dt=env.step_dt)
    if hasattr(env, "update_obb_cache"):
        env.update_obb_cache(env_ids, vpt_state.unsqueeze(0))
    if hasattr(env, "_update_camera_poses"):
        env._update_camera_poses(env_ids)
    if hasattr(env, "_occlusion_camera"):
        env._occlusion_camera.update(env.sim.cfg.dt)
    if hasattr(env, "_rgb_tiled_camera"):
        env._rgb_tiled_camera.update(env.sim.cfg.dt)

    env.mode = "testing"
    env._reset_called = True
    return config
