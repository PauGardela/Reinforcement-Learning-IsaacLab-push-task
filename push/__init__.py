# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-arm environments with end-effector pose tracking commands."""


import gymnasium as gym

# Importamos tu configuración de entorno (Asegúrate de que el archivo se llame ur3e_push_env.py)
from .push_env_cfg import UR3ePushEnvCfg

# --- IMPORTANTE: Configuración del Agente (PPO) ---
# Para que no te de otro error después, vamos a reutilizar la configuración de entrenamiento
# del Reach (agents.rsl_rl_cfg) o definimos una básica. 
# Si tienes una carpeta 'agents' dentro de 'push', úsala. Si no, usaremos la de Reach temporalmente:
from .ur3_config.ur_10.agents.rsl_rl_ppo_cfg import UR3ePushPPORunnerCfg
gym.register(
    id="Isaac-Push-UR3e-v0",  # ESTE es el nombre que usarás en la terminal
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": UR3ePushEnvCfg,
        # Reutilizamos la config de RL del Reach porque es muy parecida (mismo robot)
        "rsl_rl_cfg_entry_point": UR3ePushPPORunnerCfg, 
    },
    disable_env_checker=True,
)