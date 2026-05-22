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

from .spawn_boundary import get_boundary_cfg, get_rigid_obj, get_mat_material_paths
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
    episode_length_s = 300.0
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
            usd_path="/home/arock3/cube_game/source/cube_game/cube_game/tasks/direct/cube_game/new_cam_2.usd",
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

    agent_height = 1.0

    agent = RigidObjectCfg(
        prim_path="/World/envs/env_.*/agent",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                max_linear_velocity=3.0,
                max_angular_velocity=0.0,
            ),
            visual_material=blue_material,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)))

    # extras
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -10.0, 10.0

    # [[x_min, y_min], [x_max, y_max]]
    boundary_limits = [[x_min, y_min], [x_max, y_max]]
    boundary_height = 5.0

    bottom_wall, top_wall, left_wall, right_wall = get_boundary_cfg(
        boundary_limits, boundary_height)

    mat = RigidObjectCfg(
        prim_path="/World/envs/env_.*/mat",
        spawn=sim_utils.CuboidCfg(
            size=(x_max - x_min, y_max - y_min, 0.001),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            visual_material=boundary_material),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)))
    
    # Agent camera tilt down
    agent_camera_pitch = 15

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/agent/agent_camera_semantic",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0),
                                        rot=(math.cos(math.radians(agent_camera_pitch / 2)), 0.0, math.sin(math.radians(agent_camera_pitch / 2)), 0.0),
                                        convention="world"),
        data_types=["semantic_segmentation"],  # Add depth!
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                         focus_distance=40.0,
                                         horizontal_aperture=34.0,
                                         vertical_aperture=34.0,
                                         clipping_range=(0.01, 30.0)),
        width=512,
        height=512,
        debug_vis=True,
        update_latest_camera_pose=True,
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={
            "class:obstacles": (0, 0, 255, 255),     #Blue
            "class:target": (255, 0, 0, 255),        #Red
            "class:camera_obj": (0, 255, 0, 255),    #Green
        }
    )

    rgb_tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/agent/agent_camera_rgb",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0),
                                        rot=(math.cos(math.radians(agent_camera_pitch / 2)), 0.0, math.sin(math.radians(agent_camera_pitch / 2)), 0.0),
                                        convention="world"),
        data_types=["rgb"],  # Add depth!
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                         focus_distance=40.0,
                                         horizontal_aperture=34.0,
                                         vertical_aperture=34.0,
                                         clipping_range=(0.01, 30.0)),
        width=512,
        height=512,
        debug_vis=True,
        update_latest_camera_pose=True,
    )

    distance_tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/agent/agent_camera_distance",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0),
                                        rot=(math.cos(math.radians(agent_camera_pitch / 2)), 0.0, math.sin(math.radians(agent_camera_pitch / 2)), 0.0),
                                        convention="world"),
        data_types=["distance_to_camera"],  # Add depth!
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                         focus_distance=40.0,
                                         horizontal_aperture=34.0,
                                         vertical_aperture=34.0,
                                         clipping_range=(0.01, 30.0)),
        width=512,
        height=512,
        debug_vis=True,
        update_latest_camera_pose=True
    )
    
    # Add second camera on camera_obj for occlusion checking
    occlusion_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/camera_obj/occlusion_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            # rot=(math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)),  # 90 degree rotation around Z-axis
            rot=(1, 0, 0 ,0),  # 90 degree rotation around Z-axis
            convention="world"),
        data_types=["semantic_segmentation"],  # Only depth
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                         focus_distance=40.0,
                                         horizontal_aperture=34.0,
                                         vertical_aperture=34.0,
                                         clipping_range=(0.1, 30.0)),
        width=512,
        height=512,
        debug_vis=True,
        update_latest_camera_pose=True,
        semantic_segmentation_mapping={
            "class:target": (255, 0, 0, 255),       #Red
            "class:obstacles": (0, 0, 255, 255),    #Blue
        }
    )
    
    # light_cfg_a = sim_utils.CylinderLightCfg(
    #     prim_type="CylinderLight",
    #     intensity=250_000.0,
    #     length=5.0,
    #     treat_as_line=True,
    #     color=(0.75, 0.75, 0.75)
    # )

    # light_cfg_b = SphereLightCfg(
    #     prim_type="SphereLight",
    #     intensity=0.0, 
    #     treat_as_point=True,
    #     color=(0.75, 0.75, 0.75)
    # )
    mat_materials_paths = get_mat_material_paths()
    num_mat_materials = np.min([len(mat_materials_paths), 100])
    mat_materials_paths = mat_materials_paths[:num_mat_materials]
    mat_mdl_configs_list = []
    for mat_path in mat_materials_paths:
        material = sim_utils.MdlFileCfg(
            mdl_path=mat_path,
            # project_uvw=True,
            project_uvw=False,
            texture_scale=(200.0, 200.0),
        )
        mat_mdl_configs_list.append(material)
    print("=|"*50)
    print(mat_materials_paths)
    print("=|"*50)

    write_image_to_file = False

    num_vpt_objs = 32
    objects_per_env = 16

    storage_z = -100.0

    # objects, objects_metadata = spawn_random_objects(num_objects=num_vpt_objs,
    #                                        boundary_corners=boundary_limits,
    #                                        prim_prefix="/World/envs/env_.*")
    obj_assignment = {
        "cross": num_vpt_objs // 5 + (num_vpt_objs % 5 > 0),
        "L": num_vpt_objs // 5 + (num_vpt_objs % 5 > 1),
        "cuboid":num_vpt_objs//5 + (num_vpt_objs % 5 > 2),
        "cone":num_vpt_objs//5 + (num_vpt_objs % 5 > 3),
        "cylinder":num_vpt_objs//5,
    }
    
    rigid_objects = []
    idx = 0
    for shape, count in obj_assignment.items():
        for _ in range(count):
            rigid_objects.append(get_rigid_obj(prim_path=f"/World/envs/env_.*/obs_{idx}", fix_shape=shape))
            idx += 1
    rigid_objects_dict = {f"obs_{idx}": obj for idx, obj in enumerate(rigid_objects)}

    vpt_objects = RigidObjectCollectionCfg(
        rigid_objects=rigid_objects_dict)

    # spaces
    action_space = Discrete(
        n=4, start=0)  # 0=forward, 1=backward, 2=turn_left, 3=turn_right
    state_space = 0
    observation_space = Box(low=0,
                            high=255,
                            shape=(3, tiled_camera.height, tiled_camera.width))

    # change viewer settings
    viewer = ViewerCfg(eye=(10.0, 10.0, 30.0), lookat=(0.0, 0.0, 0.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs,
                                                     env_spacing=30.0,
                                                     replicate_physics=False)
    
    config_file = None
