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
    id="Isaac-Reach-UR3e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # 1. HARDCODED PATH (This fixes the "ModuleNotFound" error)
        # It forces Python to look in the main 'reach' folder
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg:UR3eReachEnvCfg",
        
        # 2. KEEP THE WORKING AGENT CONFIG
        # We reuse the UR10 brain config since we know it works for you
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10ReachPPORunnerCfg",
    },
    disable_env_checker=True,
)


