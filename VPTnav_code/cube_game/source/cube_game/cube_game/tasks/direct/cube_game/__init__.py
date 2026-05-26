# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register active Gym environments.
##


gym.register(
    id="Template-Cube-Game-Direct-v0",
    entry_point=f"{__name__}.cube_game_env:CubeGameEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_game_env_cfg:CubeGameEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v18-A-star",
    entry_point=f"{__name__}.vpt_env_v18_A_star:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v18",
    entry_point=f"{__name__}.vpt_env_v18:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v18-Depth",
    entry_point=f"{__name__}.vpt_env_v18_depth:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT2-v1",
    entry_point=f"{__name__}.vpt2_env_v1:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt2_env_cfg_v1:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT2-v2",
    entry_point=f"{__name__}.vpt2_env_v2:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt2_env_cfg_v2:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT2-v3",
    entry_point=f"{__name__}.vpt2_env_v3:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt2_env_cfg_v2:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT2-v4",
    entry_point=f"{__name__}.vpt2_env_v4:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt2_env_cfg_v2:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v18-strategy",
    entry_point=f"{__name__}.vpt_env_v18_A_star_strategy:VPTEnvAStarStrategy",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
    },
)

gym.register(
    id="VPT-v18-camera-move",
    entry_point=f"{__name__}.vpt_env_v18_camera:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
    },
)

gym.register(
    id="VPT-v18-camera-optim",
    entry_point=f"{__name__}.vpt_env_v18_cam_optim:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
    },
)
