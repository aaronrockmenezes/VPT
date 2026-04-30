# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="VPT-v0",
    entry_point=f"{__name__}.vpt_env:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg:VPTEnvCfg",
    },
)

gym.register(
    id="VPT-Depth-v0",
    entry_point=f"{__name__}.vpt_env_depth:VPTEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vpt_env_cfg:VPTEnvCfg",
    },
)
