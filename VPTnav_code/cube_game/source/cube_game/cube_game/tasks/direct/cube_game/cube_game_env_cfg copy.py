# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


import isaacsim.core.utils.prims as prim_utils

from .spawn_boundary import get_boundary_cfg
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# Materials
red_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
blue_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
green_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
yellow_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))
pink_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0))
cyan_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0))
boundary_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2))

@configclass
class CubeGameEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 10.0
    action_scale = 1.0  # [N]
    num_envs = 4
    dt = 0.1


    # Simulation
    sim: SimulationCfg = SimulationCfg(
            dt=dt, 
            device="cuda",
            physx=sim_utils.PhysxCfg(
                enable_ccd=True,                 # Enable CCD globally
                enable_stabilization=True,       # Enable stabilization  
                enable_enhanced_determinism=True # Better determinism
            )
        )

    # Goal ball (green sphere)
    goal_ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_ball",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=green_material
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    
    # Wall (pink cuboid)
    wall = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wall",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 0.1, 0.05),      # Changed from (1.2, 0.05, 0.05) to (2.0, 0.1, 0.05)
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=pink_material,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )
    
    # Goal mat (yellow square) - static visual element
    goal_mat = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_mat",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.001),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            visual_material=yellow_material,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    )

    agent = RigidObjectCfg(
        prim_path="/World/envs/env_.*/agent",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_linear_velocity=3.0,
                max_angular_velocity=3.0,
            ),
            visual_material=blue_material,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05))
    )

    # extras
    boundary_limits = [[-3.0, -3.0], [3.0, 3.0]]    # [[x_min, y_min], [x_max, y_max]]

    bottom_wall, top_wall, left_wall, right_wall = get_boundary_cfg(boundary_limits)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/agent/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, 0.0, 0.2), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=40.0, horizontal_aperture=26.0, clipping_range=(0.1, 20.0)
        ),
        width=84,
        height=84,
    )
    write_image_to_file = False

    # spaces
    # action_space = 4  # 0=forward, 1=backward, 2=turn_left, 3=turn_right
    # state_space = 0
    # observation_space = tiled_camera.height * tiled_camera.width * 3
    action_space = Discrete(n=4, start=0)  # 0=forward, 1=backward, 2=turn_left, 3=turn_right
    state_space = 0
    observation_space = Box(low=0, high=255, shape=(3, tiled_camera.height, tiled_camera.width))

    # change viewer settings
    viewer = ViewerCfg(eye=(15.0, 10.0, 30.0), lookat=(15.0, 10.0, 0.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=10.0, replicate_physics=True)
