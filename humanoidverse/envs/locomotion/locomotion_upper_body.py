import torch

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

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if self.locked_upper_body_action_indices.numel() == 0:
            return
        self.actions[:, self.locked_upper_body_action_indices] = 0.0
        self.actions_after_delay[:, self.locked_upper_body_action_indices] = 0.0

    def _centered_dof_pos(self, index):
        return self.simulator.dof_pos[:, index] - self.default_dof_pos[:, index]

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
        lateral_balance = torch.square(left_roll - right_roll) + torch.square(left_yaw - right_yaw)
        return pitch_opposition + 0.25 * lateral_balance

    def _reward_upperbody_arm_leg_phase(self):
        gain = self._upper_body_reward_cfg("arm_leg_phase_gain", 0.5)
        min_speed = self._upper_body_reward_cfg("arm_leg_phase_min_command_speed", 0.15)

        left_arm_pitch = self._centered_dof_pos(self.left_shoulder_pitch_index)
        right_arm_pitch = self._centered_dof_pos(self.right_shoulder_pitch_index)
        left_hip_pitch = self._centered_dof_pos(self.left_hip_pitch_index)
        right_hip_pitch = self._centered_dof_pos(self.right_hip_pitch_index)

        target_left_arm = gain * right_hip_pitch
        target_right_arm = gain * left_hip_pitch
        phase_error = torch.square(left_arm_pitch - target_left_arm) + torch.square(right_arm_pitch - target_right_arm)

        command_speed = torch.norm(self.commands[:, :2], dim=1)
        return phase_error * (command_speed > min_speed).float()
