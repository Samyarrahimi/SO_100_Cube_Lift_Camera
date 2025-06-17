# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, ContactSensor
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def lift_height_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_height: float = 0.2,
    scale: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Shaped reward based on how high the object is lifted."""
    object: RigidObject = env.scene[object_cfg.name]
    z = object.data.root_pos_w[:, 2]
    # Reward starts after minimal height, scaled between 0 and 1
    reward = (z - minimal_height) / (max_height - minimal_height)
    reward = torch.clamp(reward, min=0.0, max=1.0)
    return reward * scale


def object_grasped_on_ground(
    env: ManagerBasedRLEnv,
    gripper_threshold: float = 0.3,          # Gripper is mostly closed
    height_threshold: float = 0.03,          # Still on ground
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward agent for grasping the object while it's still on the ground."""
    object = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Object height
    z = object.data.root_pos_w[:, 2]

    # Gripper joint state
    gripper_index = robot.data.joint_names.index("Gripper")
    gripper_state = robot.data.joint_pos[:, gripper_index]

    # Check: object is still on ground AND gripper is mostly closed
    grasped = (z < height_threshold) & (gripper_state < gripper_threshold)

    return grasped.float()


def gripper_reward(
    env: ManagerBasedRLEnv,
    target_gripper_pos: float = 0.0,  # Fully closed for binary actions
    gripper_threshold: float = 0.15,  # Based on cube size: 0.015m cube + margin = ~0.15 gripper position
    contact_threshold: float = 0.1,
    gripper_width_threshold: float = 0.15,  # Width of gripper opening when open
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Comprehensive gripper reward that handles both closing and successful grasping.
    
    Note: When grasping a cube, the gripper position will be > 0 because the cube
    physically prevents the gripper from closing to position 0.0.
    
    Threshold calculation:
    - Cube size: 0.015m (1.5cm) per side (DexCube with 0.3 scale)
    - Gripper position 0.0 = fully closed (no gap)
    - Gripper position 0.5 = fully open
    - When grasping 0.015m cube, gripper position ≈ 0.15 (30% of open position)
    - Threshold set to 0.15 to account for cube blocking + some margin
    
    Returns:
        - Closing reward when gripper is closing and cube is between jaws
        - Full reward when gripper is closed (or grasping cube), contact detected, and cube is between jaws
    """
    robot = env.scene[robot_cfg.name]
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    object = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get gripper joint state
    gripper_index = robot.data.joint_names.index("Gripper")
    gripper_state = robot.data.joint_pos[:, gripper_index]
    
    # Check if cube is between gripper jaws
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    ee_rot_w = ee_frame.data.target_rot_w[..., 0, :]  # (num_envs, 4)
    cube_pos_w = object.data.root_pos_w  # (num_envs, 3)
    
    # Calculate relative position of cube in end-effector frame
    cube_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_rot_w, cube_pos_w)
    
    # Check if cube is between gripper jaws
    cube_in_gripper_y = torch.abs(cube_pos_ee[:, 1]) < gripper_width_threshold / 2  # Within gripper width
    cube_in_gripper_x = torch.abs(cube_pos_ee[:, 0]) < 0.05  # Close to gripper center in X
    cube_in_gripper_z = torch.abs(cube_pos_ee[:, 2]) < 0.05  # Close to gripper center in Z
    
    # Cube is properly positioned between gripper jaws
    cube_between_jaws = cube_in_gripper_y & cube_in_gripper_x & cube_in_gripper_z
    
    # Check contact is detected
    contact_forces = contact_sensor.data.net_forces_w
    batch_size = contact_forces.shape[0]
    contact_forces_flat = contact_forces.view(batch_size, -1)
    contact_magnitude = torch.norm(contact_forces_flat, dim=-1)
    has_contact = contact_magnitude > contact_threshold
    
    # Check gripper states - account for cube blocking gripper closure
    # When grasping a cube, gripper_state will be > 0 due to physics
    gripper_closed_or_grasping = gripper_state < gripper_threshold
    
    # Reward for being close to target position (0.0) when not grasping
    # When grasping, we care more about contact than exact position
    gripper_error = torch.abs(gripper_state - target_gripper_pos)
    closing_reward = torch.exp(-gripper_error * 10.0)  # Exponential reward for being close to closed
    
    # Combined reward logic
    # 1. If gripper is closed/grasping + contact + cube between jaws = full reward (1.0)
    # 2. If gripper is closing + cube between jaws = partial reward (closing_reward)
    # 3. If gripper is closed but empty = NO REWARD (0.0)
    # 4. Otherwise = no reward (0.0)
    
    successful_grasp = gripper_closed_or_grasping & has_contact & cube_between_jaws
    closing_with_cube = cube_between_jaws & ~gripper_closed_or_grasping  # Closing but not yet grasping
    
    # Final reward: full reward for successful grasp, partial reward for closing with cube
    # NO reward for closed but empty gripper
    final_reward = successful_grasp.float() + (closing_reward * closing_with_cube.float())
    
    return final_reward


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Combine rewards multiplicatively
    return reach_reward * lift_reward