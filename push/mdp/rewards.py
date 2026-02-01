# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def ee_target_pos_error(env, robot_name="robot", ee_body_name="wrist_3_link", target_name="target"):
    ee_idx = env.scene[robot_name].find_bodies(ee_body_name)[0][0]
    ee_pos = env.scene[robot_name].data.body_pos_w[:, ee_idx, :3]
    target_pos = env.scene[target_name].data.root_pos_w[:, :3]

    distance = torch.norm(ee_pos - target_pos, dim=-1)
    

    return torch.nan_to_num(torch.clamp(distance, max=5.0))

def ee_target_pos_tanh(env, std=0.1, robot_name="robot", ee_body_name="wrist_3_link", target_name="target"):
    """Recompensa fina (tanh) para cuando está muy cerca."""

    ee_idx = env.scene[robot_name].find_bodies(ee_body_name)[0][0]
    ee_pos = env.scene[robot_name].data.body_pos_w[:, ee_idx, :3]
    target_pos = env.scene[target_name].data.root_pos_w[:, :3]

    distance = torch.norm(ee_pos - target_pos, dim=-1)
   
    reward = 1.0 - torch.tanh(distance / std)

    return torch.nan_to_num(torch.clamp(reward, min=0.0, max=1.0))


def ee_orientation_down_reward(env, robot_name="robot", ee_body_name="wrist_3_link"):

    ee_idx = env.scene[robot_name].find_bodies(ee_body_name)[0][0]
    ee_quat = env.scene[robot_name].data.body_quat_w[:, ee_idx]
    
   
    ee_quat = torch.nn.functional.normalize(ee_quat, dim=-1)
    

    local_y = torch.zeros((env.num_envs, 3), device=env.device)
    local_y[:, 2] = 1.0  # eix z
    
    
    world_down = torch.zeros((env.num_envs, 3), device=env.device)
    world_down[:, 2] = -1.0
    

    current_y_dir = quat_apply(ee_quat, local_y)
    

    cos_sim = torch.sum(current_y_dir * world_down, dim=-1)
    
    
    sigma = 0.25 
    reward = torch.pow(torch.clamp(cos_sim, min=0.0), 3) 
    
    return torch.nan_to_num(reward)


def robot_table_collision_penalty(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot_contact")):
    """Penaliza si el robot toca cualquier cosa (mesa, suelo, etc.)."""

    contact_forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w
    
   
    force_mag = torch.norm(contact_forces, dim=-1) 
    

    max_force = torch.max(force_mag, dim=-1)[0]
    collision_detected = max_force > 5.0
    
    return collision_detected.float()


def cube_goal_distance(env, cube_name="cube", goal_name="goal"):
    cube_pos = env.scene[cube_name].data.root_pos_w[:, :3]
    goal_pos = env.scene[goal_name].data.root_pos_w[:, :3]
    
   
    dist = torch.norm(goal_pos - cube_pos + 1e-6, dim=-1)
    
    return torch.clamp(dist, max=2.0)

def cube_goal_tanh(env, std=0.2, cube_name="cube", goal_name="goal"):
    """Recompensa fina cuando el cubo está llegando a la meta."""
    cube_pos = env.scene[cube_name].data.root_pos_w[:, :3]
    goal_pos = env.scene[goal_name].data.root_pos_w[:, :3]
    
    dist = torch.norm(goal_pos - cube_pos, dim=-1)
    return 1.0 - torch.tanh(dist / std)

def cube_vertical_pressing_penalty(env, threshold: float = 15.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("cube_contact")):
   
    raw_force = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, 0, 2]
    
   
    force_z = torch.nan_to_num(raw_force, nan=0.0, posinf=0.0, neginf=0.0)
    
    pressure = torch.abs(force_z)
    penalty = torch.where(pressure > threshold, pressure, torch.zeros_like(pressure))
    
    
    return torch.clamp(penalty, max=10.0)


def cube_velocity_towards_goal(env, cube_name="cube", goal_name="goal"):
    """Recompensa que el cubo se mueva hacia la meta, pero satura rápido."""
   
    cube_vel = env.scene[cube_name].data.root_lin_vel_w[:, :3]
    cube_pos = env.scene[cube_name].data.root_pos_w[:, :3]
    goal_pos = env.scene[goal_name].data.root_pos_w[:, :3]
    
    
    to_goal_vec = goal_pos - cube_pos
    to_goal_dir = torch.nn.functional.normalize(to_goal_vec, dim=-1, eps=1e-6)
    
    
    vel_towards_goal = torch.sum(cube_vel * to_goal_dir, dim=-1)
    
   
    reward = torch.tanh(5.0 * vel_towards_goal)
    
   
    return torch.clamp(reward, min=0.0)



def maintain_contact_reward(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("cube_contact")):
    """Premia el contacto lateral, penalizando o ignorando si se presiona desde arriba."""
    
    contact_forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w
    
    
    force_xy = contact_forces[:, :, :2] 
    force_z = contact_forces[:, :, 2]
    
   
    mag_lateral = torch.norm(force_xy, dim=-1)
    
   
    
    cube_weight = 1.0  
    vertical_pressing = torch.abs(force_z - cube_weight) 
    

    is_lateral_push = (mag_lateral > 0.5) & (mag_lateral > vertical_pressing)
    

    reward = torch.any(is_lateral_push, dim=-1)
    
    return reward.float()


def relative_velocity_matching(env, cube_name="cube", robot_name="robot", ee_body_name="wrist_3_link"):
    cube_vel = env.scene[cube_name].data.root_lin_vel_w[:, :3]
    
    ee_idx = env.scene[robot_name].find_bodies(ee_body_name)[0][0]
    ee_vel = env.scene[robot_name].data.body_lin_vel_w[:, ee_idx, :3]
    

    vel_diff = torch.norm(cube_vel - ee_vel, dim=-1)
    
    reward = torch.exp(-vel_diff / 0.2) 
    return torch.nan_to_num(reward)


def ee_velocity_penalty(env: ManagerBasedRLEnv, robot_name: str = "robot", ee_body_name: str = "wrist_3_link") -> torch.Tensor:
    """Penaliza la velocidad excesiva del end-effector para evitar latigazos."""
   
    robot: Articulation = env.scene[robot_name]
    
    
    ee_body_index, _ = robot.find_bodies(ee_body_name)
    

    ee_vel = robot.data.body_vel_w[:, ee_body_index[0], :3]
    
    
    return torch.square(torch.norm(ee_vel, dim=-1))