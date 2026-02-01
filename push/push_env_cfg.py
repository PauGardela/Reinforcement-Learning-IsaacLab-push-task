# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import math
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.envs.mdp as m_d
from isaaclab.assets import RigidObjectCfg

from isaaclab.sensors import ContactSensorCfg
import isaaclab_tasks.manager_based.manipulation.push.mdp as mdp






@configclass
class PushSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""


    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",


            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True, 
                disable_gravity=True,
        ),
        
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True
        ),
       
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )


    robot: ArticulationCfg = MISSING


    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


    cube = RigidObjectCfg(
       
        prim_path="{ENV_REGEX_NS}/cube", 
        
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05), 
            
          
            activate_contact_sensors=True, 
            

            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, 
                dynamic_friction=0.5
            ),
            

            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.025)), 
    )

    goal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False), 
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0, 0.025)),
    )
    target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetObject", 
            spawn=sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3.0, 0.0)),
        )
##
# MDP settings
##

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0, 
        history_length=6, 
        debug_vis=True  
    )

    cube_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            update_period=0.0,            
            history_length=6,
            debug_vis=True                 
        )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        

        ee_to_target = ObsTerm(func=mdp.rel_ee_to_target, noise=Unoise(n_min=-0.01, n_max=0.01))

        cube_to_goal = ObsTerm(
            func=mdp.rel_obj_to_obj, 
            params={
                "obj1_cfg": SceneEntityCfg("cube"),
                "obj2_cfg": SceneEntityCfg("goal")
            }
        )

        cube_to_robot = ObsTerm(
            func=mdp.rel_obj_to_robot_base,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "robot_cfg": SceneEntityCfg("robot")
            }
        )

        goal_to_robot = ObsTerm(
            func=mdp.rel_obj_to_robot_base,
            params={
                "object_cfg": SceneEntityCfg("goal"),
                "robot_cfg": SceneEntityCfg("robot")
            }
        )

        cube_vel = ObsTerm(func=mdp.generated_cube_vel_obs)
        ee_quat = ObsTerm(func=mdp.ee_orientation_quat)
        cube_velocity = ObsTerm(func=mdp.cube_lin_vel, params={"cube_name": "cube"})
        ee_velocity = ObsTerm(func=mdp.ee_lin_vel, params={"robot_name": "robot", "ee_body_name": "wrist_3_link"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    policy: PolicyCfg = PolicyCfg() 



@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


    reset_goal = EventTerm(
        func=m_d.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (0.1, 0.3), "z": (0, 0) }, 
            "velocity_range": {},  
            "asset_cfg": SceneEntityCfg("goal"),
        },
    )
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, -0.2), "y": (-0.2, 0)}, 
            "velocity_range": {}, 
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    update_target = EventTerm(
        func=mdp.update_target_relative_to_cube,
        mode="interval", interval_range_s= (0.0 , 0.0),  
        params={
            "target_name": "target",
            "cube_name": "cube",
            "goal_name": "goal",
            "offset": [0.0, 0.02, 0.0],
        }
    )
@configclass
class RewardsCfg:

    """Reward terms for the MDP."""
#FASE 1
####################################
    # ee_target_coarse = RewTerm(
    #     func=mdp.ee_target_pos_error,
    #     weight=-1,
    #     params={
    #         "robot_name": "robot",
    #         "ee_body_name": "wrist_3_link",
    #         "target_name": "target",
    #     },
    # )

    # ee_target_fine = RewTerm(
    #     func=mdp.ee_target_pos_tanh,
    #     weight=2,
    #     params={
    #         "robot_name": "robot",
    #         "ee_body_name": "wrist_3_link",
    #         "target_name": "target",
    #         "std": 0.05,
    #     },
    # )
   
    # align_ee_down = RewTerm(
    #     func=mdp.ee_orientation_down_reward,
    #     weight=0.5,
    #     params={"robot_name": "robot",
    #             "ee_body_name": "wrist_3_link",}
    # )
    # 
    # collision_penalty = RewTerm(
    #     func=mdp.robot_table_collision_penalty,
    #     weight=-2.0,
    #     params={
    #         
    #         "sensor_cfg": SceneEntityCfg("contact_forces") 
    #     },
    # )
    # 
    # action_rate = RewTerm(
    # func=mdp.action_rate_l2,
    # weight=-0.05, 
    # )
    # ee_vel_limit = RewTerm(
    #     func=mdp.ee_velocity_penalty,
    #     weight=-0.25, 
    #     params={
    #         "robot_name": "robot", 
    #         "ee_body_name": "wrist_3_link"
    #     },
    # )

    # FASE 2
    ##############################################################
    ee_target_coarse = RewTerm(
        func=mdp.ee_target_pos_error,
        weight=-1,
        params={
            "robot_name": "robot",
            "ee_body_name": "wrist_3_link",
            "target_name": "target",
        },
    )

    ee_target_fine = RewTerm(
        func=mdp.ee_target_pos_tanh,
        weight=0.25, 
        params={
            "robot_name": "robot",
            "ee_body_name": "wrist_3_link",
            "target_name": "target",
            "std": 0.2,
        },
    )

    align_ee_down = RewTerm(
        func=mdp.ee_orientation_down_reward,
        weight=0.5,
        params={"robot_name": "robot",
                "ee_body_name": "wrist_3_link",}
    )

    collision_penalty = RewTerm(
        func=mdp.robot_table_collision_penalty,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces") 
        },
    )

    action_rate = RewTerm(
    func=mdp.action_rate_l2,
    weight=-0.1, 
    )

    cube_to_goal_coarse = RewTerm(
        func=mdp.cube_goal_distance,
        weight=-5, 
        params={"cube_name": "cube", "goal_name": "goal"}
    )
    
    cube_to_goal_fine = RewTerm(
        func=mdp.cube_goal_tanh,
        weight=20,
        params={"cube_name": "cube", "goal_name": "goal", "std": 0.075}
    )

    maintain_contact = RewTerm(
        func=mdp.maintain_contact_reward,
        weight=5,
        params={
            "sensor_cfg": SceneEntityCfg("cube_contact")
        },
    )
    ee_vel_limit = RewTerm(
        func=mdp.ee_velocity_penalty,
        weight=-1.0, 
        params={
            "robot_name": "robot", 
            "ee_body_name": "wrist_3_link"
        },
    )
    
    cube_press_penalty = RewTerm(
        func=mdp.cube_vertical_pressing_penalty,
        weight=-2,
        params={
            "threshold": 30.0,
            "sensor_cfg": SceneEntityCfg("cube_contact") 
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_falling = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("cube")}, 
    )
    #TERMINACIONS PER A MALES FISIQUES
    bad_physics_robot = DoneTerm(
        func=mdp.bad_physics_termination,
        params={"threshold": 2000.0, "sensor_cfg": SceneEntityCfg("contact_forces")}
    )

    bad_physics_cube = DoneTerm(
        func=mdp.bad_physics_termination,
        params={"threshold": 300.0, "sensor_cfg": SceneEntityCfg("cube_contact")}
    )
# @configclass 
# class CurriculumCfg: #CurriculumCfg fa que a partir de 4500 passos aumenti la penalizacio de les accions
#     """Curriculum terms for the MDP."""

#     action_rate = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
#     )

#     joint_vel = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
#     )


##
# Environment configuration
##


@configclass
class PushEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: PushSceneCfg = PushSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    #commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    #curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 120.0

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
                "gamepad": Se3GamepadCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
            },
        )


# --- MANUAL ROBOT DEFINITION START ---
UR3e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur3e/ur3e.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # Matches UR10e config from your file
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 0.0,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        # BASE JOINTS
        "arm_base": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
            velocity_limit_sim=5.0,
            effort_limit_sim=50.0,
            stiffness=100.0,  
            damping=10.0,
        ),
        # WRIST JOINTS 
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            velocity_limit_sim=10.0,
            effort_limit_sim=12.0, 
            stiffness=25.0,   
            damping=2.5,
        ),
    },
)

@configclass
class UR3ePushEnvCfg(PushEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        #Robot Setup
        self.scene.robot = UR3e_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Actions
        self.actions.arm_action = m_d.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"], 
            scale=0.5, 
            use_default_offset=True,
        )

        # Body Names
        # self.commands.ee_pose.body_name = "wrist_3_link"
        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["wrist_3_link"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["wrist_3_link"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["wrist_3_link"]

        # Ranges
        # Default was X:(-0.5, 0.5), Z:(0.25, 0.75)
        # self.commands.ee_pose.ranges.pos_x = (0.15, 0.45)
        # self.commands.ee_pose.ranges.pos_y = (-0.4, 0.4)
        # self.commands.ee_pose.ranges.pos_z = (0.15, 0.6) 
        # self.commands.ee_pose.ranges.pitch = (0.0, 2*math.pi) # 0.0 represents end-effector pointing upwards

        # # Disable Penalties
        # self.rewards.action_rate.weight = 0.0
        # self.rewards.joint_vel.weight = 0.0