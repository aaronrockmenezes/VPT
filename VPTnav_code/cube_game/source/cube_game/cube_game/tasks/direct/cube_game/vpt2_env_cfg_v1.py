# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.lights import SphereLightCfg
from isaaclab.assets import RigidObjectCollectionCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass
import numpy as np

from .spawn_boundary import get_boundary_cfg, get_rigid_obj, get_mat_material_paths, get_obj_assignment
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# Materials
red_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
blue_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
green_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
yellow_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))
pink_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0))
cyan_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0))
boundary_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.2))


@configclass
class VPTEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 3000.0
    action_scale = 1.0  # [N]
    num_envs = 32
    dt = 0.1

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        device="cuda",
        physx=sim_utils.PhysxCfg(
            enable_ccd=False,  # Enable CCD globally
            enable_enhanced_determinism=False,  # Better determinism
            enable_stabilization=True,  # Enable stabilization  
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=2**22,
            gpu_found_lost_pairs_capacity=2**22,
            gpu_heap_capacity=2**28,
            gpu_temp_buffer_capacity=2**22
        ),
        render=sim_utils.RenderCfg(
            enable_dl_denoiser=True,
            enable_shadows=True,
            antialiasing_mode="DLAA",  # Set your desired mode here
            rendering_mode="quality"
        ))

    camera_obj = RigidObjectCfg(
        prim_path="/World/envs/env_.*/camera_obj",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/users/arock3/scratch/VPTnav_code/cube_game/assets/new_cam_latest.usd",
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=green_material,
            semantic_tags=[("class", "camera_obj")]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)))

    # Goal ball (green sphere)
    goal_radius = 0.25
    
    goal_ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_ball",
        spawn=sim_utils.MeshSphereCfg(
            radius=goal_radius,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=1,
                solver_velocity_iteration_count=1,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=red_material,
            semantic_tags=[("class", "target")]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),)
    
    reference_obj = RigidObjectCfg(
        prim_path="/World/envs/env_.*/reference_obj",
        spawn=sim_utils.MeshConeCfg(
            radius=0.2,
            height=0.5,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=1,
                solver_velocity_iteration_count=1,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=pink_material,
            semantic_tags=[("class", "reference_obj")]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),)

    agent_height = 1.0

    agent = RigidObjectCfg(
        prim_path="/World/envs/env_.*/agent",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=1,
                solver_velocity_iteration_count=1,
                max_linear_velocity=1.0,
                max_angular_velocity=0.0,
                disable_gravity=True
            ),
            visual_material=blue_material,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)))

    # extras
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -10.0, 10.0

    # [[x_min, y_min], [x_max, y_max]]
    boundary_limits = [[x_min, y_min], [x_max, y_max]]
    boundary_height = 30.0

    bottom_wall, top_wall, left_wall, right_wall = get_boundary_cfg(
        boundary_limits, boundary_height)

    mat = RigidObjectCfg(
        prim_path="/World/envs/env_.*/mat",
        spawn=sim_utils.CuboidCfg(
            size=(x_max - x_min, y_max - y_min, 0.001),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            visual_material=boundary_material),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)))
    
    # Agent camera tilt down
    agent_camera_pitch = 15

    # camera
    rgb_tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/agent/agent_camera_rgb",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0),
                                        rot=(math.cos(math.radians(agent_camera_pitch / 2)), 0.0, math.sin(math.radians(agent_camera_pitch / 2)), 0.0),
                                        convention="world"),
        data_types=["rgb", "semantic_segmentation","distance_to_camera"],  # Add depth!
        # data_types=["rgb"],  # Add depth!
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                         focus_distance=40.0,
                                         horizontal_aperture=34.0,
                                         vertical_aperture=34.0,
                                         clipping_range=(0.01, 30.0)),
        width=512//1,
        height=512//1,
        debug_vis=True,
        update_latest_camera_pose=True,
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={
            "class:obstacles": (0, 0, 255, 255),     #Blue
            "class:target": (255, 0, 0, 255),        #Red
            "class:camera_obj": (0, 255, 0, 255),    #Green
            "class:reference_obj": (255, 0, 255, 255), #Pink
        }
    )
    
    # Add second camera on camera_obj for occlusion checking
    occlusion_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/camera_obj/occlusion_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0),
                                        rot=(math.cos(math.radians(agent_camera_pitch / 2)), 0.0, math.sin(math.radians(agent_camera_pitch / 2)), 0.0),
                                        convention="world"),
        data_types=["semantic_segmentation"],  # Only depth
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                         focus_distance=40.0,
                                         horizontal_aperture=34.0,
                                         vertical_aperture=34.0,
                                         clipping_range=(0.1, 30.0)),
        width=512//1,
        height=512//1,
        debug_vis=True,
        update_latest_camera_pose=True,
        semantic_segmentation_mapping={
            "class:target": (255, 0, 0, 255),       #Red
            "class:obstacles": (0, 0, 255, 255),    #Blue
            "class:reference_obj": (255, 0, 255, 255), #Pink
        }
    )


    write_image_to_file = False

    num_vpt_objs = 32
    objects_per_env = 16

    storage_z = -100.0

    obj_assignment = get_obj_assignment(shapes=["cross", "L", "cuboid", "cylinder", "cone", "table_A", "table_B", "A", "H", "I", "Z", "Bench"], num_vpt_objs=num_vpt_objs)
    # obj_assignment = get_obj_assignment(shapes=["cross", "L", "cuboid", "cylinder", "cone"], num_vpt_objs=num_vpt_objs)
    # obj_assignment = get_obj_assignment(shapes=["L", "cross", "A", "I", "H", "Z"], num_vpt_objs=num_vpt_objs)
    # obj_assignment = get_obj_assignment(shapes=["table_A"], num_vpt_objs=num_vpt_objs)
    
    rigid_objects = []
    idx = 0
    for shape, count in obj_assignment.items():
        for _ in range(count):
            rigid_objects.append(get_rigid_obj(prim_path=f"/World/envs/env_.*/obs_{idx}", fix_shape=shape))
            idx += 1

    # import pdb; pdb.set_trace()
    rigid_objects_dict = {f"obs_{idx}": obj for idx, obj in enumerate(rigid_objects)}

    vpt_objects = RigidObjectCollectionCfg(
        rigid_objects=rigid_objects_dict)

    # spaces
    action_space = Discrete(
        n=4, start=0)  # 0=forward, 1=backward, 2=turn_left, 3=turn_right
    state_space = 0
    observation_space = Box(low=0,
                            high=255,
                            shape=(3, 512//1, 512//1))

    # change viewer settings
    viewer = ViewerCfg(eye=(10.0, 10.0, 30.0), lookat=(0.0, 0.0, 0.0), resolution=(512, 512))


    agent_speed = 1.5
    agent_turn_angle = math.pi / 24

    agent_success_distance = 0.4
    agent_success_angle = 0.4

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs,
                                                     env_spacing=30.0,
                                                     replicate_physics=False)
    
    config_file = None
