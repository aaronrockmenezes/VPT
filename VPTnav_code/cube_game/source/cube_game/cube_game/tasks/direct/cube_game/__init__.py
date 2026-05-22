# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
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
    id="VPT-v0",
    entry_point=f"{__name__}.vpt_env_parallel_v5:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v5:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v5",
    entry_point=f"{__name__}.vpt_env_v5:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v5:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v6",
    entry_point=f"{__name__}.vpt_env_v6:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v6:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v6-timed",
    entry_point=f"{__name__}.vpt_env_v6_timed:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v6:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v6-fast",
    entry_point=f"{__name__}.vpt_env_v6_fast:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v6_fast:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v7",
    entry_point=f"{__name__}.vpt_env_v7:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v7:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v8",
    entry_point=f"{__name__}.vpt_env_v8:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v8:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v9",
    entry_point=f"{__name__}.vpt_env_v9:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v9:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v10",
    entry_point=f"{__name__}.vpt_env_v10:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v10:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v11",
    entry_point=f"{__name__}.vpt_env_v11:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v11:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v10-B",
    entry_point=f"{__name__}.vpt_env_v10_b:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v10:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v10-test",
    entry_point=f"{__name__}.vpt_env_v10_test_bench:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v10:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v12-test",
    entry_point=f"{__name__}.vpt_env_v12_test:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v12:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v13-test",
    entry_point=f"{__name__}.vpt_env_v13_test:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v13:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v14",
    entry_point=f"{__name__}.vpt_env_v14:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v14:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v15",
    entry_point=f"{__name__}.vpt_env_v15:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v16",
    entry_point=f"{__name__}.vpt_env_v16:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v16:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v17-v5",
    entry_point=f"{__name__}.vpt_env_v17_new_v5:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v17-RL",
    entry_point=f"{__name__}.vpt_env_v17_new:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v17-RL-D",
    entry_point=f"{__name__}.vpt_env_v17_new_depth:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)


gym.register(
    id="VPT-v17-RL-Alekh",
    entry_point=f"{__name__}.vpt_env_v17_alekh:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="VPT-v17-RL-D-Reload",
    entry_point=f"{__name__}.vpt_env_v17_new_v4_reload:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg_v15_rl:VPTEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_lstm_cfg_entry_point": f"{agents.__name__}:sb3_recurrent_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
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
        "env_cfg_entry_point": f"{__name__}.vpt2_env_v1:VPTEnvCfg",
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
        "env_cfg_entry_point": f"{__name__}.vpt2_env_v2:VPTEnvCfg",
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
        "env_cfg_entry_point": f"{__name__}.vpt2_env_v2:VPTEnvCfg",
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
        "env_cfg_entry_point": f"{__name__}.vpt2_env_v2:VPTEnvCfg",
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