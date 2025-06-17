# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def gripper_joint_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Get the current gripper joint state."""
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Find gripper joint index
    gripper_index = robot.data.joint_names.index("Gripper")
    
    # Get gripper joint position and velocity
    gripper_pos = robot.data.joint_pos[:, gripper_index:gripper_index + 1]
    gripper_vel = robot.data.joint_vel[:, gripper_index:gripper_index + 1]
    
    # Combine position and velocity
    gripper_state = torch.cat([gripper_pos, gripper_vel], dim=-1)
    
    return gripper_state


def gripper_contact_forces(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact")
) -> torch.Tensor:
    """Get contact forces from the gripper contact sensor."""
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    
    # Get contact forces (3D vector) - use net_forces_w instead of contact_forces
    contact_forces = contact_sensor.data.net_forces_w
    
    # Get contact magnitude
    contact_magnitude = torch.norm(contact_forces, dim=-1, keepdim=True)
    
    # Combine forces and magnitude
    contact_info = torch.cat([contact_forces, contact_magnitude], dim=-1)
    
    return contact_info