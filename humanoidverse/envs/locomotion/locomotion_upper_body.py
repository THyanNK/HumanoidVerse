import torch

from humanoidverse.utils.torch_utils import quat_rotate_inverse
from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion


class LeggedRobotLocomotionUpperBody(LeggedRobotLocomotion):
    """Locomotion task with light upper-body motion shaping for full H1."""

    def __init__(self, config, device):
        super().__init__(config, device)
        if not self.config.robot.has_upper_body_dof:
            raise ValueError("LeggedRobotLocomotionUpperBody requires a robot config with upper body DOFs.")
        self._setup_upper_body_reward_indices()

    def _setup_upper_body_reward_indices(self):
        self.name_to_dof_index = {name: index for index, name in enumerate(self.dof_names)}

        def idx(name):
            return self.name_to_dof_index[name]

        self.upper_body_reward_indices = torch.as_tensor(
            self.upper_dof_indices, dtype=torch.long, device=self.device
        )
        self.torso_dof_index = idx("torso_joint")
        self.left_shoulder_pitch_index = idx("left_shoulder_pitch_joint")
        self.left_shoulder_roll_index = idx("left_shoulder_roll_joint")
        self.left_shoulder_yaw_index = idx("left_shoulder_yaw_joint")
        self.left_elbow_index = idx("left_elbow_joint")
        self.right_shoulder_pitch_index = idx("right_shoulder_pitch_joint")
        self.right_shoulder_roll_index = idx("right_shoulder_roll_joint")
        self.right_shoulder_yaw_index = idx("right_shoulder_yaw_joint")
        self.right_elbow_index = idx("right_elbow_joint")
        self.left_hip_pitch_index = idx("left_hip_pitch_joint")
        self.right_hip_pitch_index = idx("right_hip_pitch_joint")

        self.shoulder_pitch_indices = torch.as_tensor(
            [self.left_shoulder_pitch_index, self.right_shoulder_pitch_index],
            dtype=torch.long,
            device=self.device,
        )
        self.elbow_indices = torch.as_tensor(
            [self.left_elbow_index, self.right_elbow_index],
            dtype=torch.long,
            device=self.device,
        )
        self.arm_lateral_indices = torch.as_tensor(
            [
                self.left_shoulder_roll_index,
                self.left_shoulder_yaw_index,
                self.right_shoulder_roll_index,
                self.right_shoulder_yaw_index,
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.arm_endpoint_indices = torch.as_tensor(
            [
                self.simulator.find_rigid_body_indice("left_elbow_link"),
                self.simulator.find_rigid_body_indice("right_elbow_link"),
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.arm_posture_indices = torch.as_tensor(
            [
                self.left_shoulder_roll_index,
                self.left_shoulder_yaw_index,
                self.left_elbow_index,
                self.right_shoulder_roll_index,
                self.right_shoulder_yaw_index,
                self.right_elbow_index,
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.locked_upper_body_action_indices = self._dof_indices_from_reward_cfg(
            "locked_action_dof_names"
        )
        self.locked_upper_body_reward_indices = self._dof_indices_from_reward_cfg(
            "locked_reward_dof_names"
        )
        self._setup_upper_body_random_actions()

    def _upper_body_reward_cfg(self, name, default):
        cfg = self.config.rewards.get("upper_body", {})
        return cfg.get(name, default)

    def _dof_indices_from_reward_cfg(self, cfg_name):
        names = list(self._upper_body_reward_cfg(cfg_name, []))
        missing = [name for name in names if name not in self.name_to_dof_index]
        if missing:
            raise ValueError(f"Unknown DOF names in rewards.upper_body.{cfg_name}: {missing}")
        return torch.as_tensor(
            [self.name_to_dof_index[name] for name in names],
            dtype=torch.long,
            device=self.device,
        )

    def _setup_upper_body_random_actions(self):
        self.randomize_upper_body_actions = bool(
            self.config.domain_rand.get("randomize_upper_body_actions", False)
        )
        default_names = list(self.config.robot.get("upper_dof_names", []))
        if not default_names:
            default_names = [
                self.dof_names[index]
                for index in self.upper_body_reward_indices.detach().cpu().tolist()
            ]
        names = list(
            self.config.domain_rand.get(
                "upper_body_random_action_dof_names",
                default_names,
            )
        )

        missing = [name for name in names if name not in self.name_to_dof_index]
        if missing:
            raise ValueError(f"Unknown DOF names in domain_rand.upper_body_random_action_dof_names: {missing}")

        lower_body_names = [
            name
            for name in names
            if name not in set(default_names)
        ]
        if lower_body_names:
            raise ValueError(
                "domain_rand.upper_body_random_action_dof_names must only contain upper-body DOFs; "
                f"got lower-body DOFs: {lower_body_names}"
            )

        self.upper_body_random_action_indices = torch.as_tensor(
            [self.name_to_dof_index[name] for name in names],
            dtype=torch.long,
            device=self.device,
        )
        self.upper_body_random_action_buf = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.upper_body_random_action_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.upper_body_random_action_active = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

    def _upper_body_random_action_enabled(self):
        if not self.randomize_upper_body_actions:
            return False
        if self.upper_body_random_action_indices.numel() == 0:
            return False
        if getattr(self, "is_evaluating", False) and not self.config.domain_rand.get(
            "upper_body_random_action_apply_during_eval",
            True,
        ):
            return False
        return True

    def _sample_upper_body_random_action_steps(self, cfg_name, default, num_envs, min_steps=1):
        interval_s = self.config.domain_rand.get(cfg_name, default)
        if isinstance(interval_s, (int, float)):
            min_s = float(interval_s)
            max_s = float(interval_s)
        else:
            min_s = float(interval_s[0])
            max_s = float(interval_s[1])
        min_steps = max(min_steps, int(round(min_s / self.dt)))
        max_steps = max(min_steps, int(round(max_s / self.dt)))
        if max_steps == 0:
            return torch.zeros(num_envs, dtype=torch.long, device=self.device)
        return torch.randint(
            min_steps,
            max_steps + 1,
            (num_envs,),
            dtype=torch.long,
            device=self.device,
        )

    def _sample_upper_body_random_action_interval_steps(self, num_envs):
        return self._sample_upper_body_random_action_steps(
            "upper_body_random_action_resample_s",
            [0.2, 0.5],
            num_envs,
        )

    def _sample_upper_body_random_action_active_steps(self, num_envs):
        return self._sample_upper_body_random_action_steps(
            "upper_body_random_action_active_s",
            [0.35, 0.70],
            num_envs,
        )

    def _sample_upper_body_random_action_recovery_steps(self, num_envs):
        return self._sample_upper_body_random_action_steps(
            "upper_body_random_action_recovery_s",
            [1.0, 2.0],
            num_envs,
        )

    def _sample_upper_body_random_action_start_delay_steps(self, num_envs):
        if "upper_body_random_action_start_delay_s" not in self.config.domain_rand:
            return torch.zeros(num_envs, dtype=torch.long, device=self.device)
        return self._sample_upper_body_random_action_steps(
            "upper_body_random_action_start_delay_s",
            [0.0, 0.0],
            num_envs,
            min_steps=0,
        )

    def _upper_body_random_action_amplitude(self):
        if getattr(self, "is_evaluating", False):
            return float(
                self.config.domain_rand.get(
                    "upper_body_random_action_eval_amp",
                    self.config.domain_rand.get("upper_body_random_action_amp", 0.0),
                )
            )
        return float(self.config.domain_rand.get("upper_body_random_action_amp", 0.0))

    def _sample_upper_body_random_action_values(self, env_ids):
        amp = self._upper_body_random_action_amplitude()
        if amp <= 0.0:
            return

        random_actions = (
            torch.rand(
                len(env_ids),
                self.upper_body_random_action_indices.numel(),
                dtype=torch.float,
                device=self.device,
            )
            * 2.0
            - 1.0
        ) * amp
        self.upper_body_random_action_buf[
            env_ids[:, None],
            self.upper_body_random_action_indices,
        ] = random_actions

    def _sample_upper_body_random_actions(self, env_ids):
        if len(env_ids) == 0:
            return

        self.upper_body_random_action_buf[env_ids] = 0.0
        self._sample_upper_body_random_action_values(env_ids)
        self.upper_body_random_action_counter[env_ids] = (
            self._sample_upper_body_random_action_interval_steps(len(env_ids))
        )

    def _update_upper_body_random_action_pulses(self, env_ids):
        if len(env_ids) == 0:
            return

        next_active = torch.logical_not(self.upper_body_random_action_active[env_ids])
        active_env_ids = env_ids[next_active]
        recovery_env_ids = env_ids[~next_active]

        if len(active_env_ids) > 0:
            self.upper_body_random_action_active[active_env_ids] = True
            self.upper_body_random_action_buf[active_env_ids] = 0.0
            self._sample_upper_body_random_action_values(active_env_ids)
            self.upper_body_random_action_counter[active_env_ids] = (
                self._sample_upper_body_random_action_active_steps(len(active_env_ids))
            )

        if len(recovery_env_ids) > 0:
            self.upper_body_random_action_active[recovery_env_ids] = False
            self.upper_body_random_action_buf[recovery_env_ids] = 0.0
            self.upper_body_random_action_counter[recovery_env_ids] = (
                self._sample_upper_body_random_action_recovery_steps(len(recovery_env_ids))
            )

    def _update_upper_body_random_actions(self):
        refresh_env_ids = (self.upper_body_random_action_counter <= 0).nonzero(
            as_tuple=False
        ).flatten()
        mode = self.config.domain_rand.get("upper_body_random_action_mode", "continuous")
        if mode == "continuous":
            self._sample_upper_body_random_actions(refresh_env_ids)
        elif mode == "pulse":
            self._update_upper_body_random_action_pulses(refresh_env_ids)
        else:
            raise ValueError(f"Unknown domain_rand.upper_body_random_action_mode: {mode}")

    def _apply_upper_body_random_actions(self):
        if not self._upper_body_random_action_enabled():
            return

        self._update_upper_body_random_actions()

        indices = self.upper_body_random_action_indices
        self.actions_after_delay[:, indices] += self.upper_body_random_action_buf[:, indices]
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions_after_delay[:, indices] = torch.clip(
            self.actions_after_delay[:, indices],
            -clip_action_limit,
            clip_action_limit,
        )
        self.upper_body_random_action_counter -= 1

        random_action_abs = torch.abs(self.upper_body_random_action_buf[:, indices])
        self.log_dict["upper_body_random_action_abs"] = torch.mean(random_action_abs)
        self.log_dict["upper_body_random_action_max"] = torch.max(random_action_abs)
        self.log_dict["upper_body_random_action_active"] = torch.mean(
            self.upper_body_random_action_active.float()
        )

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if self.locked_upper_body_action_indices.numel() > 0:
            self.actions[:, self.locked_upper_body_action_indices] = 0.0
            self.actions_after_delay[:, self.locked_upper_body_action_indices] = 0.0
        self._apply_upper_body_random_actions()

    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        if hasattr(self, "upper_body_random_action_buf"):
            self.upper_body_random_action_buf[env_ids] = 0.0
            self.upper_body_random_action_active[env_ids] = False
            self.upper_body_random_action_counter[env_ids] = (
                self._sample_upper_body_random_action_start_delay_steps(len(env_ids))
            )

    def _centered_dof_pos(self, index):
        return self.simulator.dof_pos[:, index] - self.default_dof_pos[:, index]

    def _centered_dof_pos_for_indices(self, indices):
        return self.simulator.dof_pos[:, indices] - self.default_dof_pos[:, indices]

    def _command_speed(self):
        return torch.norm(self.commands[:, :2], dim=1)

    def _moving_command_mask(self):
        min_speed = self._upper_body_reward_cfg("arm_leg_phase_min_command_speed", 0.15)
        return (self._command_speed() > min_speed).float()

    def _straight_walk_arm_swing_mask(self):
        min_forward = self._upper_body_reward_cfg("arm_swing_min_forward_speed", 0.15)
        max_lateral = self._upper_body_reward_cfg("arm_swing_max_lateral_command", 1.0e6)
        max_yaw = self._upper_body_reward_cfg("arm_swing_max_yaw_command", 1.0e6)
        return (
            (self.commands[:, 0] > min_forward)
            & (torch.abs(self.commands[:, 1]) <= max_lateral)
            & (torch.abs(self.commands[:, 2]) <= max_yaw)
        ).float()

    def _arm_swing_mask(self):
        if self._upper_body_reward_cfg("use_straight_walk_arm_swing_mask", False):
            return self._straight_walk_arm_swing_mask()
        return self._moving_command_mask()

    def _arm_swing_speed_multiplier(self):
        base = self._upper_body_reward_cfg("arm_swing_speed_gain_base", 1.0)
        slope = self._upper_body_reward_cfg("arm_swing_speed_gain_slope", 0.0)
        ref_speed = max(self._upper_body_reward_cfg("arm_swing_speed_gain_ref", 1.0), 1.0e-6)
        min_multiplier = self._upper_body_reward_cfg("arm_swing_speed_gain_min", 0.0)
        max_multiplier = self._upper_body_reward_cfg("arm_swing_speed_gain_max", 10.0)

        forward_command = torch.clamp(self.commands[:, 0], min=0.0)
        multiplier = base + slope * (forward_command / ref_speed)
        return torch.clamp(multiplier, min_multiplier, max_multiplier)

    def _teacher_swing_phase(self):
        frequency = self._upper_body_reward_cfg("teacher_swing_frequency", 0.9)
        phase_offset = self._upper_body_reward_cfg("teacher_swing_phase_offset", 0.0)
        time = self.episode_length_buf.to(dtype=torch.float32) * self.dt
        return 2.0 * torch.pi * frequency * time + phase_offset

    def _teacher_swing_omega(self):
        frequency = self._upper_body_reward_cfg("teacher_swing_frequency", 0.9)
        return 2.0 * torch.pi * frequency

    def _teacher_shoulder_pitch_targets(self):
        amplitude = self._upper_body_reward_cfg("teacher_shoulder_pitch_amplitude", 0.35)
        swing = amplitude * torch.sin(self._teacher_swing_phase())
        return torch.stack((swing, -swing), dim=1)

    def _teacher_shoulder_pitch_vel_targets(self):
        amplitude = self._upper_body_reward_cfg("teacher_shoulder_pitch_amplitude", 0.35)
        swing_vel = amplitude * self._teacher_swing_omega() * torch.cos(self._teacher_swing_phase())
        return torch.stack((swing_vel, -swing_vel), dim=1)

    def _teacher_endpoint_sagittal_delta_target(self):
        amplitude = self._upper_body_reward_cfg("teacher_endpoint_sagittal_delta", 0.16)
        return amplitude * torch.sin(self._teacher_swing_phase())

    def _teacher_endpoint_sagittal_delta_vel_target(self):
        amplitude = self._upper_body_reward_cfg("teacher_endpoint_sagittal_delta", 0.16)
        return amplitude * self._teacher_swing_omega() * torch.cos(self._teacher_swing_phase())

    def _body_vectors_in_base(self, vectors):
        num_bodies = vectors.shape[1]
        base_quat = self.base_quat.unsqueeze(1).expand(-1, num_bodies, -1)
        return quat_rotate_inverse(
            base_quat.reshape(-1, 4),
            vectors.reshape(-1, 3),
        ).reshape(self.num_envs, num_bodies, 3)

    def _arm_endpoint_pos_in_base(self):
        root_pos = self.simulator.robot_root_states[:, :3].unsqueeze(1)
        endpoint_pos = self.simulator._rigid_body_pos[:, self.arm_endpoint_indices]
        return self._body_vectors_in_base(endpoint_pos - root_pos)

    def _arm_endpoint_vel_in_base(self):
        root_vel = self.simulator.robot_root_states[:, 7:10].unsqueeze(1)
        endpoint_vel = self.simulator._rigid_body_vel[:, self.arm_endpoint_indices]
        return self._body_vectors_in_base(endpoint_vel - root_vel)

    def _arm_endpoint_lateral_targets(self):
        targets = self._upper_body_reward_cfg("arm_endpoint_lateral_targets", [0.22, -0.22])
        return torch.as_tensor(targets, dtype=torch.float, device=self.device).unsqueeze(0)

    def _reward_upperbody_locked_dof_pos(self):
        if self.locked_upper_body_reward_indices.numel() == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        deadband = self._upper_body_reward_cfg("locked_dof_deadband", 0.03)
        deviation = torch.abs(
            self.simulator.dof_pos[:, self.locked_upper_body_reward_indices]
            - self.default_dof_pos[:, self.locked_upper_body_reward_indices]
        )
        return torch.sum(torch.square(torch.clip(deviation - deadband, min=0.0)), dim=1)

    def _reward_upperbody_action_rate(self):
        action_delta = self.actions[:, self.upper_body_reward_indices] - self.last_actions[:, self.upper_body_reward_indices]
        return torch.sum(torch.square(action_delta), dim=1)

    def _reward_upperbody_dof_acc(self):
        upper_acc = (
            self.last_dof_vel[:, self.upper_body_reward_indices]
            - self.simulator.dof_vel[:, self.upper_body_reward_indices]
        ) / self.dt
        return torch.sum(torch.square(upper_acc), dim=1)

    def _reward_upperbody_torso_deviation(self):
        deadband = self._upper_body_reward_cfg("torso_deviation_deadband", 0.12)
        torso_deviation = torch.abs(self._centered_dof_pos(self.torso_dof_index))
        return torch.square(torch.clip(torso_deviation - deadband, min=0.0))

    def _reward_upperbody_arm_posture(self):
        deadband = self._upper_body_reward_cfg("arm_posture_deadband", 0.35)
        arm_deviation = torch.abs(
            self.simulator.dof_pos[:, self.arm_posture_indices]
            - self.default_dof_pos[:, self.arm_posture_indices]
        )
        return torch.sum(torch.square(torch.clip(arm_deviation - deadband, min=0.0)), dim=1)

    def _reward_upperbody_arm_symmetry(self):
        left_pitch = self._centered_dof_pos(self.left_shoulder_pitch_index)
        right_pitch = self._centered_dof_pos(self.right_shoulder_pitch_index)
        left_roll = self._centered_dof_pos(self.left_shoulder_roll_index)
        right_roll = self._centered_dof_pos(self.right_shoulder_roll_index)
        left_yaw = self._centered_dof_pos(self.left_shoulder_yaw_index)
        right_yaw = self._centered_dof_pos(self.right_shoulder_yaw_index)

        pitch_opposition = torch.square(left_pitch + right_pitch)
        lateral_balance = torch.square(left_roll + right_roll) + torch.square(left_yaw + right_yaw)
        return pitch_opposition + 0.25 * lateral_balance

    def _reward_upperbody_arm_lateral_deviation(self):
        deadband = self._upper_body_reward_cfg("arm_lateral_deadband", 0.08)
        lateral_deviation = torch.abs(
            self._centered_dof_pos_for_indices(self.arm_lateral_indices)
        )
        return torch.sum(torch.square(torch.clip(lateral_deviation - deadband, min=0.0)), dim=1)

    def _reward_upperbody_arm_endpoint_lateral_pos(self):
        deadband = self._upper_body_reward_cfg("arm_endpoint_lateral_deadband", 0.035)
        endpoint_y = self._arm_endpoint_pos_in_base()[:, :, 1]
        lateral_error = torch.abs(endpoint_y - self._arm_endpoint_lateral_targets())
        return torch.sum(torch.square(torch.clip(lateral_error - deadband, min=0.0)), dim=1)

    def _reward_upperbody_arm_endpoint_lateral_vel(self):
        deadband = self._upper_body_reward_cfg("arm_endpoint_lateral_vel_deadband", 0.08)
        endpoint_y_vel = torch.abs(self._arm_endpoint_vel_in_base()[:, :, 1])
        return torch.sum(torch.square(torch.clip(endpoint_y_vel - deadband, min=0.0)), dim=1)

    def _reward_upperbody_arm_endpoint_sagittal_phase(self):
        gain = self._upper_body_reward_cfg("arm_endpoint_sagittal_phase_gain", 0.10)
        endpoint_x = self._arm_endpoint_pos_in_base()[:, :, 0]
        endpoint_x_delta = endpoint_x[:, 0] - endpoint_x[:, 1]
        hip_phase = (
            self._centered_dof_pos(self.right_hip_pitch_index)
            - self._centered_dof_pos(self.left_hip_pitch_index)
        )
        target_delta = gain * self._arm_swing_speed_multiplier() * hip_phase
        return torch.square(endpoint_x_delta - target_delta) * self._arm_swing_mask()

    def _reward_upperbody_arm_endpoint_sagittal_balance(self):
        target = self._upper_body_reward_cfg("arm_endpoint_sagittal_balance_target", 0.02)
        deadband = self._upper_body_reward_cfg("arm_endpoint_sagittal_balance_deadband", 0.035)
        endpoint_x = self._arm_endpoint_pos_in_base()[:, :, 0]
        endpoint_x_mean = torch.mean(endpoint_x, dim=1)
        balance_error = torch.abs(endpoint_x_mean - target)
        return torch.square(torch.clip(balance_error - deadband, min=0.0)) * self._arm_swing_mask()

    def _reward_upperbody_arm_endpoint_sagittal_amplitude(self):
        min_delta = self._upper_body_reward_cfg("arm_endpoint_sagittal_min_delta", 0.08)
        deadband = self._upper_body_reward_cfg("arm_endpoint_sagittal_amplitude_deadband", 0.01)
        endpoint_x = self._arm_endpoint_pos_in_base()[:, :, 0]
        endpoint_x_delta = torch.abs(endpoint_x[:, 0] - endpoint_x[:, 1])
        target_delta = min_delta * self._arm_swing_speed_multiplier()
        amplitude_shortfall = torch.clip(target_delta - endpoint_x_delta - deadband, min=0.0)
        return torch.square(amplitude_shortfall) * self._arm_swing_mask()

    def _reward_upperbody_arm_endpoint_sagittal_vel_phase(self):
        gain = self._upper_body_reward_cfg("arm_endpoint_sagittal_vel_phase_gain", 0.06)
        same_direction_weight = self._upper_body_reward_cfg("arm_endpoint_sagittal_same_direction_weight", 0.25)

        endpoint_x_vel = self._arm_endpoint_vel_in_base()[:, :, 0]
        endpoint_x_vel_delta = endpoint_x_vel[:, 0] - endpoint_x_vel[:, 1]
        hip_vel_phase = (
            self.simulator.dof_vel[:, self.right_hip_pitch_index]
            - self.simulator.dof_vel[:, self.left_hip_pitch_index]
        )
        target_delta = gain * self._arm_swing_speed_multiplier() * hip_vel_phase
        phase_error = torch.square(endpoint_x_vel_delta - target_delta)
        same_direction_error = torch.square(endpoint_x_vel[:, 0] + endpoint_x_vel[:, 1])
        return (phase_error + same_direction_weight * same_direction_error) * self._arm_swing_mask()

    def _reward_upperbody_teacher_arm_swing_pitch(self):
        shoulder_pitch = self._centered_dof_pos_for_indices(self.shoulder_pitch_indices)
        target = self._teacher_shoulder_pitch_targets()
        error = shoulder_pitch - target
        self.log_dict["teacher_shoulder_pitch_abs"] = torch.mean(torch.abs(shoulder_pitch))
        self.log_dict["teacher_shoulder_pitch_target_abs"] = torch.mean(torch.abs(target))
        self.log_dict["teacher_shoulder_pitch_error_abs"] = torch.mean(torch.abs(error))
        self.log_dict["teacher_elbow_abs"] = torch.mean(torch.abs(self.simulator.dof_pos[:, self.elbow_indices]))
        return torch.sum(torch.square(error), dim=1)

    def _reward_upperbody_teacher_arm_swing_velocity(self):
        shoulder_pitch_vel = self.simulator.dof_vel[:, self.shoulder_pitch_indices]
        target = self._teacher_shoulder_pitch_vel_targets()
        return torch.sum(torch.square(shoulder_pitch_vel - target), dim=1)

    def _reward_upperbody_teacher_endpoint_sagittal(self):
        endpoint_x = self._arm_endpoint_pos_in_base()[:, :, 0]
        endpoint_x_delta = endpoint_x[:, 0] - endpoint_x[:, 1]
        return torch.square(endpoint_x_delta - self._teacher_endpoint_sagittal_delta_target())

    def _reward_upperbody_teacher_endpoint_sagittal_velocity(self):
        endpoint_x_vel = self._arm_endpoint_vel_in_base()[:, :, 0]
        endpoint_x_vel_delta = endpoint_x_vel[:, 0] - endpoint_x_vel[:, 1]
        return torch.square(
            endpoint_x_vel_delta - self._teacher_endpoint_sagittal_delta_vel_target()
        )

    def _reward_upperbody_arm_leg_phase(self):
        gain = self._upper_body_reward_cfg("arm_leg_phase_gain", 0.5)

        left_arm_pitch = self._centered_dof_pos(self.left_shoulder_pitch_index)
        right_arm_pitch = self._centered_dof_pos(self.right_shoulder_pitch_index)
        left_hip_pitch = self._centered_dof_pos(self.left_hip_pitch_index)
        right_hip_pitch = self._centered_dof_pos(self.right_hip_pitch_index)

        speed_multiplier = self._arm_swing_speed_multiplier()
        target_left_arm = gain * speed_multiplier * right_hip_pitch
        target_right_arm = gain * speed_multiplier * left_hip_pitch
        phase_error = torch.square(left_arm_pitch - target_left_arm) + torch.square(right_arm_pitch - target_right_arm)

        return phase_error * self._arm_swing_mask()

    def _reward_upperbody_arm_leg_velocity_phase(self):
        gain = self._upper_body_reward_cfg(
            "arm_leg_velocity_phase_gain",
            self._upper_body_reward_cfg("arm_leg_phase_gain", 0.5),
        )

        left_arm_vel = self.simulator.dof_vel[:, self.left_shoulder_pitch_index]
        right_arm_vel = self.simulator.dof_vel[:, self.right_shoulder_pitch_index]
        left_hip_vel = self.simulator.dof_vel[:, self.left_hip_pitch_index]
        right_hip_vel = self.simulator.dof_vel[:, self.right_hip_pitch_index]

        speed_multiplier = self._arm_swing_speed_multiplier()
        phase_error = (
            torch.square(left_arm_vel - gain * speed_multiplier * right_hip_vel)
            + torch.square(right_arm_vel - gain * speed_multiplier * left_hip_vel)
        )
        return phase_error * self._arm_swing_mask()

    def _reward_upperbody_arm_velocity_opposition(self):
        left_arm_vel = self.simulator.dof_vel[:, self.left_shoulder_pitch_index]
        right_arm_vel = self.simulator.dof_vel[:, self.right_shoulder_pitch_index]
        return torch.square(left_arm_vel + right_arm_vel) * self._arm_swing_mask()

    def _reward_upperbody_shoulder_pitch_limit(self):
        soft_limit = self._upper_body_reward_cfg("shoulder_pitch_soft_limit", 0.65)
        shoulder_pitch = torch.abs(
            self._centered_dof_pos_for_indices(self.shoulder_pitch_indices)
        )
        excess = torch.clip(shoulder_pitch - soft_limit, min=0.0)
        return torch.sum(torch.square(excess), dim=1)

    def _reward_upperbody_elbow_posture(self):
        target = self._upper_body_reward_cfg("elbow_flexion_target", 0.20)
        deadband = self._upper_body_reward_cfg("elbow_flexion_deadband", 0.15)
        elbow_pos = self.simulator.dof_pos[:, self.elbow_indices]
        deviation = torch.abs(elbow_pos - target)
        return torch.sum(torch.square(torch.clip(deviation - deadband, min=0.0)), dim=1)

    def _reward_upperbody_elbow_swing_coupling(self):
        base_target = self._upper_body_reward_cfg("elbow_flexion_target", 0.20)
        coupling_gain = self._upper_body_reward_cfg("elbow_swing_coupling_gain", 0.0)
        max_target = self._upper_body_reward_cfg("elbow_swing_coupling_max_target", 0.45)
        deadband = self._upper_body_reward_cfg("elbow_swing_coupling_deadband", 0.12)

        shoulder_pitch = torch.abs(
            self._centered_dof_pos_for_indices(self.shoulder_pitch_indices)
        )
        target = torch.clamp(base_target + coupling_gain * shoulder_pitch, max=max_target)
        elbow_pos = self.simulator.dof_pos[:, self.elbow_indices]
        deviation = torch.abs(elbow_pos - target)
        return (
            torch.sum(torch.square(torch.clip(deviation - deadband, min=0.0)), dim=1)
            * self._arm_swing_mask()
        )

    def _reward_upperbody_stationary_arm_posture(self):
        max_speed = self._upper_body_reward_cfg("stationary_arm_max_command_speed", 0.15)
        stationary_mask = (self._command_speed() <= max_speed).float()

        shoulder_pitch = self._centered_dof_pos_for_indices(self.shoulder_pitch_indices)
        elbow_deviation = (
            self.simulator.dof_pos[:, self.elbow_indices]
            - self.default_dof_pos[:, self.elbow_indices]
        )
        posture_error = (
            torch.sum(torch.square(shoulder_pitch), dim=1)
            + 0.5 * torch.sum(torch.square(elbow_deviation), dim=1)
        )
        return posture_error * stationary_mask

    def _reward_upperbody_torques(self):
        return torch.sum(torch.square(self.torques[:, self.upper_body_reward_indices]), dim=1)
