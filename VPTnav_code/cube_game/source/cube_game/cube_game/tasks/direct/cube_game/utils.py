import torch

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