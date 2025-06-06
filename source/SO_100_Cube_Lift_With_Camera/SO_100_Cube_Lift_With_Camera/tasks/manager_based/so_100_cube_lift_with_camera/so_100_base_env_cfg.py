# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
import torch

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    # cube marker: will be populated by agent env cfg
    cube_marker: FrameTransformerCfg = MISSING
    # gripper camera: will be populated by agent env cfg
    gripper_camera: CameraCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    # gripper_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # actions = ObsTerm(func=mdp.last_action)
    
        #robot_state = torch.cat([joint_pos, joint_vel, actions], dim=-1)
        camera_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg":SceneEntityCfg("gripper_camera"),"data_type":"rgb"})
        #camera_depth = ObsTerm(func=mdp.image, params={"sensor_cfg":SceneEntityCfg("gripper_camera"),"data_type":"distance_to_image_plane"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")



# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""
#     # Reaching reward with lower weight
#     reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=5.0)

#     # Reward for grasping the object while it's still on the ground
#     grasping_on_ground = RewTerm(func=mdp.object_grasped_on_ground, weight=20.0)

#     # Lifting reward with higher weight
#     #lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.02}, weight=25.0)
#     lifting_object = RewTerm(func=mdp.lift_height_reward, params={"minimal_height": 0.02, "max_height": 0.15, "scale": 30.0}, weight=1.0)

#     # Action penalty to encourage smooth movements
#     action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

#     # Joint velocity penalty to prevent erratic movements
#     joint_vel = RewTerm(
#         func=mdp.joint_vel_l2,
#         weight=-1e-4,
#         params={"asset_cfg": SceneEntityCfg("robot")},
#     )
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # Reaching reward with lower weight
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=2)
    # Lifting reward with higher weight
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.02}, weight=25.0)
    # Action penalty to encourage smooth movements
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    # Joint velocity penalty to prevent erratic movements
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )
    success = DoneTerm(
        func=mdp.object_reached_goal,
        params={"threshold": 0.06, "robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")}
    )


# @configclass
# class CurriculumCfg:
# #     """Curriculum terms for the MDP."""

#     # Stage 1: Focus on reaching
#     # Start with higher reaching reward, then gradually decrease it
#     reaching_reward = CurrTerm(
#         func=mdp.modify_reward_weight, 
#         params={"term_name": "reaching_object", "weight": 1.0, "num_steps": 15000}
#     )

#     grasping_reward = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={"term_name": "grasping_on_ground", "weight": 2.0, "num_steps": 30000}
#     )

#     # Stage 2: Transition to lifting
#     # Start with lower lifting reward, gradually increase to encourage lifting behavior
#     lifting_reward = CurrTerm(
#         func=mdp.modify_reward_weight, 
#         params={"term_name": "lifting_object", "weight": 20.0, "num_steps": 90000}
#     )

#     # Stage 4: Stabilize the policy
#     # Gradually increase action penalties to encourage smoother, more stable movements
#     action_rate = CurrTerm(
#         func=mdp.modify_reward_weight, 
#         params={"term_name": "action_rate", "weight": -5e-4, "num_steps": 30000}
#     )

#     joint_vel = CurrTerm(
#         func=mdp.modify_reward_weight, 
#         params={"term_name": "joint_vel", "weight": -5e-4, "num_steps": 30000}
#     )
@configclass
class CurriculumCfg:
#     """Curriculum terms for the MDP."""
    # Stage 1: Focus on reaching
    # Start with higher reaching reward, then gradually decrease it
    reaching_reward = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "reaching_object", "weight": 1.0, "num_steps": 6000}
    )
    # Stage 2: Transition to lifting
    # Start with lower lifting reward, gradually increase to encourage lifting behavior
    lifting_reward = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "lifting_object", "weight": 35.0, "num_steps": 8000}
    )
    # Stage 3: Stabilize the policy
    # Gradually increase action penalties to encourage smoother, more stable movements
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "action_rate", "weight": -5e-4, "num_steps": 12000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "joint_vel", "weight": -5e-4, "num_steps": 12000}
    )

##
# Environment configuration
##

@configclass
class SO100LiftCameraEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=32, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        
        # Configure camera for closer view during video recording
        self.viewer.eye = (1.0, 1.0, 0.8)
        self.viewer.lookat = (0.5, 0.0, 0.2)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0