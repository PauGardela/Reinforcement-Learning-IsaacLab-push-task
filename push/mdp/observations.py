import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_rotate_inverse
from isaaclab.managers import SceneEntityCfg
def update_target_relative_to_cube(env, env_ids, target_name, cube_name, goal_name, offset):

    cube_pos = env.scene[cube_name].data.root_pos_w[env_ids, :3]
    goal_pos = env.scene[goal_name].data.root_pos_w[env_ids, :3]


    vec_cube_to_goal = goal_pos - cube_pos
    

    dist = torch.norm(vec_cube_to_goal, dim=-1, keepdim=True)
    unit_vec = vec_cube_to_goal / (dist + 1e-6)

   
    
    distancia_detras = 0.1 
    altura_ajuste = 0.01   
    

    target_pos = cube_pos - (unit_vec * distancia_detras)
    
    
    target_pos[:, 2] += altura_ajuste 


    target_quat = env.scene[target_name].data.root_quat_w[env_ids].clone()
    new_pose = torch.cat([target_pos, target_quat], dim=-1)

    env.scene[target_name].write_root_pose_to_sim(new_pose, env_ids)
    env.scene[target_name].write_data_to_sim()


def reset_cube_with_exclusion(env, env_ids, asset_cfg, goal_name, min_dist=0.12):
   
    cube_asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    

    goal_pos = env.scene[goal_name].data.root_pos_w[env_ids, :2]
    
   
    table_pos = env.scene["table"].data.root_pos_w[env_ids, :2]
    

    current_x = table_pos[:, 0] + torch.empty(num_envs, device=env.device).uniform_(-0.3, 0.0)
    current_y = table_pos[:, 1] + torch.empty(num_envs, device=env.device).uniform_(-0.25, 0.25)


    new_poses = torch.zeros((num_envs, 7), device=env.device)

    new_velocities = torch.zeros((num_envs, 6), device=env.device)
    

    dist = torch.norm(torch.stack([current_x, current_y], dim=-1) - goal_pos, dim=-1)
    
  
    too_close = dist < min_dist
    current_x[too_close] = torch.where(goal_pos[too_close, 0] > 0.5, 0.36, 0.64)

    
    new_poses[:, 0] = current_x
    new_poses[:, 1] = current_y
    new_poses[:, 2] = 0.05  
    new_poses[:, 3] = 1.0   
    
   
    cube_asset.write_root_pose_to_sim(new_poses, env_ids)
    cube_asset.write_root_velocity_to_sim(new_velocities, env_ids) 



def rel_obj_to_obj(env, obj1_cfg, obj2_cfg):
    pos1 = env.scene[obj1_cfg.name].data.root_pos_w[:, :3]
    pos2 = env.scene[obj2_cfg.name].data.root_pos_w[:, :3]
    res = pos2 - pos1
    # Si el cubo desaparece, el robot solo verá un 5.0, no un infinito
    return torch.clamp(torch.nan_to_num(res), min=-5.0, max=5.0)

def rel_ee_to_target(env, robot_name="robot", ee_body_name="wrist_3_link", cube_name="cube", goal_name="goal", offset=torch.tensor([0.0,0.2,0.0])):
    ee_idx = env.scene[robot_name].find_bodies(ee_body_name)[0][0]
    ee_pos = env.scene[robot_name].data.body_pos_w[:, ee_idx, :3]
    
    cube_pos = env.scene[cube_name].data.root_pos_w[:, :3]
    goal_pos = env.scene[goal_name].data.root_pos_w[:, :3]
    offset = offset.to(goal_pos.device)
    target_pos = goal_pos - cube_pos + offset
    rel = target_pos - ee_pos

    return rel.view(rel.shape[0], -1)  # shape: (N_envs, 3)


def rel_obj_to_robot_base(env, object_cfg, robot_cfg) -> torch.Tensor:
    """Calcula la posición relativa del objeto respecto a la base del robot en el frame local del robot."""
    
   
    obj_data = env.scene[object_cfg.name].data
    obj_pos_w = obj_data.root_pos_w[:, :3] 

    robot_data = env.scene[robot_cfg.name].data
    root_pos_w = robot_data.root_pos_w[:, :3] 
    root_quat_w = robot_data.root_quat_w    


    rel_pos_w = obj_pos_w - root_pos_w

    rel_pos_local = quat_rotate_inverse(root_quat_w, rel_pos_w)

    return rel_pos_local

    # En mdp/observations.py

def ee_orientation_quat(env, robot_name="robot", ee_body_name="wrist_3_link"):
    """Devuelve el cuaternión de la muñeca en el frame del mundo."""
    ee_idx = env.scene[robot_name].find_bodies(ee_body_name)[0][0]
    return env.scene[robot_name].data.body_quat_w[:, ee_idx]

def cube_lin_vel(env, cube_name="cube"):
    """Velocidad lineal del cubo."""
    return env.scene[cube_name].data.root_lin_vel_w[:, :3]

def ee_lin_vel(env, robot_name="robot", ee_body_name="wrist_3_link"):
    """Velocidad lineal de la punta del brazo."""
    ee_idx = env.scene[robot_name].find_bodies(ee_body_name)[0][0]
    return env.scene[robot_name].data.body_lin_vel_w[:, ee_idx, :3]

def generated_cube_vel_obs(env, cube_name="cube"):
    """
    Extrae la velocidad lineal del cubo en el frame del mundo.
    
    Esta observación es vital para que la Policy entienda la relación 
    causa-efecto entre sus movimientos y el desplazamiento del objeto.
    """

    cube_vel = env.scene[cube_name].data.root_lin_vel_w[:, :3]
    

    clean_vel = torch.nan_to_num(cube_vel, nan=0.0, posinf=0.0, neginf=0.0)
    

    return torch.clamp(clean_vel, min=-3.0, max=3.0)