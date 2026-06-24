from time import time
from warnings import WarningMessage
import numpy as np
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from humanoidverse.utils.spatial_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
# from humanoidverse.envs.env_utils.command_generator import CommandGenerator


class LeggedRobotLocomotion(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        # import ipdb; ipdb.set_trace()
        motion_cfg = config.robot.get("motion", {})
        if motion_cfg.get("hips_link", None):
            body_list = getattr(self.simulator, "_body_list", self.body_names)
            self.hips_dof_id = [body_list.index(link) - 1 for link in motion_cfg.hips_link] # Yuanhang: -1 for the base link (pelvis)

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges
        num_feet = self.feet_indices.shape[0]
        self.gait_air_time = torch.zeros(
            self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.gait_contact_time = torch.zeros_like(self.gait_air_time)
        self.gait_swing_peak_height = torch.zeros_like(self.gait_air_time)
        self.gait_last_air_time = torch.zeros_like(self.gait_air_time)
        self.gait_last_swing_peak_height = torch.zeros_like(self.gait_air_time)
        self.gait_first_contact = torch.zeros(
            self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.gait_contact_filt = torch.zeros_like(self.gait_first_contact)
        self.gait_last_contact = torch.zeros_like(self.gait_first_contact)
        self.gait_last_contact_filt = torch.zeros_like(self.gait_first_contact)

    def _setup_simulator_control(self):
        self.simulator.commands = self.commands

    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        super()._update_tasks_callback()

        # commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading), 
            self.command_ranges["ang_vel_yaw"][0], 
            self.command_ranges["ang_vel_yaw"][1]
        )
        self._update_gait_timing_buffers()

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        self.gait_air_time[env_ids] = 0.0
        self.gait_contact_time[env_ids] = 0.0
        self.gait_swing_peak_height[env_ids] = 0.0
        self.gait_last_air_time[env_ids] = 0.0
        self.gait_last_swing_peak_height[env_ids] = 0.0
        self.gait_first_contact[env_ids] = False
        self.gait_contact_filt[env_ids] = False
        self.gait_last_contact[env_ids] = False
        self.gait_last_contact_filt[env_ids] = False
        if not self.is_evaluating:
            self._resample_commands(env_ids)

    def _gait_reward_cfg(self, name, default):
        cfg = self.config.rewards.get("gait", {})
        return cfg.get(name, default)

    def _gait_moving_command_mask(self):
        min_speed = self._gait_reward_cfg("moving_command_min_speed", 0.1)
        return (self._gait_command_speed() > min_speed).float()

    def _gait_command_speed(self):
        return torch.norm(self.commands[:, :2], dim=1)

    def _gait_stand_command_mask(self):
        max_speed = self._gait_reward_cfg("stand_command_max_speed", 0.1)
        return (self._gait_command_speed() <= max_speed).float()

    def _gait_walk_command_mask(self):
        min_speed = self._gait_reward_cfg(
            "walk_command_min_speed",
            self._gait_reward_cfg("moving_command_min_speed", 0.1),
        )
        return (self._gait_command_speed() > min_speed).float()

    def _gait_straight_walk_command_mask(self):
        min_forward = self._gait_reward_cfg("straight_walk_min_forward_speed", 0.15)
        max_lateral = self._gait_reward_cfg("straight_walk_max_lateral_command", 1.0e6)
        max_yaw = self._gait_reward_cfg("straight_walk_max_yaw_command", 1.0e6)
        return (
            (self.commands[:, 0] > min_forward)
            & (torch.abs(self.commands[:, 1]) <= max_lateral)
            & (torch.abs(self.commands[:, 2]) <= max_yaw)
        ).float()

    def _gait_walk_reward_mask(self):
        if self._gait_reward_cfg("use_straight_walk_reward_mask", False):
            return self._gait_straight_walk_command_mask()
        return self._gait_walk_command_mask()

    def _update_gait_timing_buffers(self):
        threshold = self._gait_reward_cfg("contact_force_threshold", 1.0)
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > threshold
        contact_filt = torch.logical_or(contact, self.gait_last_contact)
        had_air_time = self.gait_air_time > 0.0
        first_contact = contact_filt & ~self.gait_last_contact_filt & had_air_time

        foot_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        next_air_time = self.gait_air_time + self.dt
        next_contact_time = self.gait_contact_time + self.dt
        swing_peak = torch.maximum(self.gait_swing_peak_height, foot_height)

        self.gait_first_contact = first_contact
        self.gait_contact_filt = contact_filt
        self.gait_last_air_time = torch.where(
            first_contact, next_air_time, torch.zeros_like(next_air_time)
        )
        self.gait_last_swing_peak_height = torch.where(
            first_contact, swing_peak, torch.zeros_like(swing_peak)
        )

        self.gait_air_time = torch.where(contact_filt, torch.zeros_like(next_air_time), next_air_time)
        self.gait_contact_time = torch.where(
            contact_filt, next_contact_time, torch.zeros_like(next_contact_time)
        )
        self.gait_swing_peak_height = torch.where(
            contact_filt, torch.zeros_like(swing_peak), swing_peak
        )
        self.gait_last_contact = contact
        self.gait_last_contact_filt = contact_filt

        moving_mask = self._gait_moving_command_mask().unsqueeze(1)
        first_contact_f = first_contact.float() * moving_mask
        first_contact_count = torch.sum(first_contact_f, dim=1)
        contact_count = torch.sum(contact_filt.float(), dim=1)
        self.log_dict["gait_first_contact_count"] = torch.mean(first_contact_count)
        self.log_dict["gait_contact_count"] = torch.mean(contact_count)
        self.log_dict["gait_last_air_time"] = torch.sum(
            self.gait_last_air_time * first_contact_f
        ) / torch.clamp(torch.sum(first_contact_f), min=1.0)
        self.log_dict["gait_last_swing_peak_height"] = torch.sum(
            self.gait_last_swing_peak_height * first_contact_f
        ) / torch.clamp(torch.sum(first_contact_f), min=1.0)
        self.log_dict["gait_stand_mask"] = torch.mean(self._gait_stand_command_mask())
        self.log_dict["gait_walk_mask"] = torch.mean(self._gait_walk_command_mask())
        self.log_dict["gait_straight_walk_mask"] = torch.mean(self._gait_straight_walk_command_mask())

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.commands.zero_()
        if command is not None:
            if isinstance(command, torch.Tensor):
                command_tensor = command.to(dtype=self.commands.dtype, device=self.device).flatten()
            else:
                command_tensor = torch.as_tensor(list(command), dtype=self.commands.dtype, device=self.device).flatten()
            if command_tensor.numel() < 3:
                raise ValueError("Evaluation command must contain at least [lin_vel_x, lin_vel_y, yaw].")
            self.commands[:, :3] = command_tensor[:3]
            if command_tensor.numel() >= 4:
                self.commands[:, 3] = command_tensor[3]
        self.simulator.commands = self.commands
    ########################### TRACKING REWARDS ###########################

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.config.rewards.reward_tracking_sigma.ang_vel)

    def _target_heading_error(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        return wrap_to_pi(self.commands[:, 3] - heading)

    def _reward_tracking_heading(self):
        heading_error = torch.square(self._target_heading_error())
        return torch.exp(-heading_error / self.config.rewards.reward_tracking_sigma.ang_vel)

    def _reward_penalty_heading_error(self):
        command_speed = torch.norm(self.commands[:, :2], dim=1)
        return torch.square(self._target_heading_error()) * (command_speed > 0.1).float()

    def _reward_penalty_lateral_vel(self):
        return torch.square(self.base_lin_vel[:, 1])

    ########################### PENALTY REWARDS ###########################

    def _reward_tracking_lin_vel_x(self):
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.config.rewards.reward_tracking_sigma.lin_vel)

    def _reward_penalty_low_forward_speed(self):
        command_x = torch.clamp(self.commands[:, 0], min=0.0)
        speed_deficit = torch.clamp(command_x - self.base_lin_vel[:, 0], min=0.0)
        return torch.square(speed_deficit) * (command_x > 0.1).float()

    def _reward_penalty_backward_vel(self):
        return torch.square(torch.clamp(-self.base_lin_vel[:, 0], min=0.0))

    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_penalty_ang_vel_xy_torso(self):
        # Penalize xy axes base angular velocity

        torso_ang_vel = quat_rotate_inverse(self.simulator._rigid_body_rot[:, self.torso_index], self.simulator._rigid_body_ang_vel[:, self.torso_index])
        return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)
    

    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)

    ########################### FEET REWARDS ###########################

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_air_time_target(self):
        target = self._gait_reward_cfg("feet_air_time_target", 0.45)
        clipped_air_time = torch.clamp(self.gait_last_air_time, max=target)
        reward = torch.sum(clipped_air_time * self.gait_first_contact.float(), dim=1)
        return reward * self._gait_moving_command_mask()

    def _reward_feet_single_stance_time(self):
        target = self._gait_reward_cfg("feet_single_stance_time_target", 0.25)
        contact_count = torch.sum(self.gait_contact_filt.int(), dim=1)
        single_stance = (contact_count == 1).unsqueeze(1)
        in_mode_time = torch.where(
            self.gait_contact_filt, self.gait_contact_time, self.gait_air_time
        )
        reward = torch.min(torch.where(single_stance, in_mode_time, torch.zeros_like(in_mode_time)), dim=1).values
        return torch.clamp(reward, max=target) * self._gait_moving_command_mask()

    def _reward_penalty_feet_air_time_short(self):
        min_air_time = self._gait_reward_cfg("feet_air_time_min", 0.30)
        shortfall = torch.clamp(min_air_time - self.gait_last_air_time, min=0.0)
        penalty = torch.sum(torch.square(shortfall) * self.gait_first_contact.float(), dim=1)
        return penalty * self._gait_moving_command_mask()

    def _reward_penalty_feet_swing_height(self):
        target_height = self._gait_reward_cfg("feet_swing_height_target", 0.12)
        shortfall = torch.clamp(target_height - self.gait_last_swing_peak_height, min=0.0)
        penalty = torch.sum(torch.square(shortfall) * self.gait_first_contact.float(), dim=1)
        return penalty * self._gait_moving_command_mask()

    def _reward_stand_double_support(self):
        min_contacts = self._gait_reward_cfg("stand_min_contact_count", self.feet_indices.shape[0])
        contact_count = torch.sum(self.gait_contact_filt.float(), dim=1)
        return (contact_count >= min_contacts).float() * self._gait_stand_command_mask()

    def _reward_penalty_stand_base_motion(self):
        yaw_weight = self._gait_reward_cfg("stand_yaw_motion_weight", 0.25)
        lin_motion = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        yaw_motion = torch.square(self.base_ang_vel[:, 2])
        return (lin_motion + yaw_weight * yaw_motion) * self._gait_stand_command_mask()

    def _reward_penalty_walk_knee_flexion(self):
        knee_names = list(
            self._gait_reward_cfg(
                "walk_knee_dof_names",
                ["left_knee_joint", "right_knee_joint"],
            )
        )
        missing = [name for name in knee_names if name not in self.dof_names]
        if missing:
            raise ValueError(f"Unknown DOF names in rewards.gait.walk_knee_dof_names: {missing}")
        knee_indices = torch.as_tensor(
            [self.dof_names.index(name) for name in knee_names],
            dtype=torch.long,
            device=self.device,
        )
        soft_limit = self._gait_reward_cfg("walk_knee_flexion_soft_limit", 1.05)
        excess = torch.clip(self.simulator.dof_pos[:, knee_indices] - soft_limit, min=0.0)
        return torch.sum(torch.square(excess), dim=1) * self._gait_walk_reward_mask()
    
    def _reward_penalty_in_the_air(self):
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward



    def _reward_penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)


    def _reward_penalty_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_base_height(self):
        # Penalize base height away from target

        base_height = self.simulator.robot_root_states[:, 2]
        return torch.square(base_height - self.config.rewards.desired_base_height)

    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        return torch.sum(torch.square(hip_pos), dim=1)

    def _reward_feet_heading_alignment(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])


        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))
        
        return heading_diff_left + heading_diff_right
    
    def _reward_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_penalty_feet_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    

    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 
    
    def _reward_penalty_close_feet_xy(self):
        # returns 1 if two feet are too close
        left_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
        feet_distance_xy = torch.norm(left_foot_xy - right_foot_xy, dim=1)
        return (feet_distance_xy < self.config.rewards.close_feet_threshold) * 1.0
    

    def _reward_penalty_close_knees_xy(self):
        # returns 1 if two knees are too close
        left_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[0], :2]
        right_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[1], :2]
        self.knee_distance_xy = torch.norm(left_knee_xy - right_knee_xy, dim=1)
        return (self.knee_distance_xy < self.config.rewards.close_knees_threshold)* 1.0
    

    def _reward_upperbody_joint_angle_freeze(self):
        # returns keep the upper body joint angles close to the default
        assert self.config.robot.has_upper_body_dof
        deviation = torch.abs(self.simulator.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:,self.upper_dof_indices])
        return torch.sum(deviation, dim=1)
    
    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return self.commands[:, :2]
    
    def _get_obs_command_ang_vel(self):
        return self.commands[:, 2:3]
