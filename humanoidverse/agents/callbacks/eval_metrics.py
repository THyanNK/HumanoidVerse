import csv
import json
import math
from pathlib import Path

import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger

from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback


class EvalMetricsCallback(RL_EvalCallback):
    """Finite-horizon evaluation metrics for poster-ready locomotion summaries."""

    def __init__(self, training_loop, **config):
        super().__init__(config, training_loop)
        self.env = self.training_loop.env
        self.max_steps = int(self.config.get("max_steps", 1000))
        self.output_dir = Path(self.config.get("output_dir", "") or self._default_output_dir())
        self.write_timeseries = bool(self.config.get("write_timeseries", True))
        self.recovery_speed_fraction = float(self.config.get("recovery_speed_fraction", 0.80))
        self.recovery_tilt_threshold = float(self.config.get("recovery_tilt_threshold", 0.35))
        self.contact_force_threshold = float(self.config.get("contact_force_threshold", 1.0))

    def _default_output_dir(self):
        try:
            return HydraConfig.get().runtime.output_dir
        except Exception:
            return "."

    def on_pre_evaluate_policy(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_envs = self.env.num_envs
        device = self.env.device
        self.fall_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.ever_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.first_failure_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        self.prev_disturbance_active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.waiting_for_recovery = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.pulse_end_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        self.recovery_times = []
        self.ended_pulses = 0
        self.recovered_pulses = 0
        self.sums = {
            "reward": 0.0,
            "tracking_error_x": 0.0,
            "abs_lateral_vel": 0.0,
            "tilt": 0.0,
            "max_tilt": 0.0,
            "contact_count": 0.0,
            "upper_action_abs": 0.0,
            "disturbance_active_frac": 0.0,
            "arm_swing_amp": 0.0,
            "arm_opposition_error": 0.0,
            "endpoint_sagittal_delta": 0.0,
        }
        self.samples = 0
        self.timeseries = []
        logger.info(
            f"EvalMetricsCallback enabled: max_steps={self.max_steps}, "
            f"output_dir={self.output_dir}"
        )

    def _mean_item(self, tensor):
        return float(torch.mean(tensor.detach()).cpu().item())

    def _active_disturbance(self):
        active = getattr(self.env, "upper_body_random_action_active", None)
        if active is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
        return active.bool()

    def _random_action_abs(self):
        buf = getattr(self.env, "upper_body_random_action_buf", None)
        indices = getattr(self.env, "upper_body_random_action_indices", None)
        if buf is None or indices is None or indices.numel() == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.env.device)
        return torch.mean(torch.abs(buf[:, indices]), dim=1)

    def _centered_dof(self, index):
        return self.env.simulator.dof_pos[:, index] - self.env.default_dof_pos[:, index]

    def _arm_metrics(self):
        zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.env.device)
        left = getattr(self.env, "left_shoulder_pitch_index", None)
        right = getattr(self.env, "right_shoulder_pitch_index", None)
        if left is None or right is None:
            return zeros, zeros, zeros
        left_pitch = self._centered_dof(left)
        right_pitch = self._centered_dof(right)
        arm_swing_amp = 0.5 * (torch.abs(left_pitch) + torch.abs(right_pitch))
        arm_opposition_error = torch.abs(left_pitch + right_pitch)

        if hasattr(self.env, "_arm_endpoint_pos_in_base"):
            endpoint_x = self.env._arm_endpoint_pos_in_base()[:, :, 0]
            endpoint_delta = torch.abs(endpoint_x[:, 0] - endpoint_x[:, 1])
        else:
            endpoint_delta = zeros
        return arm_swing_amp, arm_opposition_error, endpoint_delta

    def _update_recovery_metrics(self, step, disturbance_active, done):
        pulse_ended = self.prev_disturbance_active & ~disturbance_active
        self.ended_pulses += int(torch.sum(pulse_ended).detach().cpu().item())
        self.waiting_for_recovery[pulse_ended] = True
        self.pulse_end_step[pulse_ended] = step

        self.waiting_for_recovery[done] = False
        self.pulse_end_step[done] = -1

        command_x = torch.clamp(self.env.commands[:, 0], min=0.0)
        target_speed = self.recovery_speed_fraction * command_x
        speed_recovered = self.env.base_lin_vel[:, 0] >= target_speed
        tilt = torch.linalg.norm(self.env.projected_gravity[:, :2], dim=1)
        posture_recovered = tilt <= self.recovery_tilt_threshold
        recovered = self.waiting_for_recovery & speed_recovered & posture_recovered
        if torch.any(recovered):
            recovery_steps = step - self.pulse_end_step[recovered]
            self.recovery_times.extend((recovery_steps.float() * self.env.dt).detach().cpu().tolist())
            self.recovered_pulses += int(torch.sum(recovered).detach().cpu().item())
            self.waiting_for_recovery[recovered] = False
            self.pulse_end_step[recovered] = -1

        self.prev_disturbance_active = disturbance_active.clone()

    def on_post_eval_env_step(self, actor_state):
        step = int(actor_state.get("step", 0)) + 1
        done = actor_state["dones"].bool()
        time_outs = actor_state.get("extras", {}).get("time_outs", None)
        if time_outs is None:
            time_outs = torch.zeros_like(done)
        failed = done & ~time_outs.bool()
        self.fall_count += failed.long()
        first_fail = failed & ~self.ever_failed
        self.first_failure_step[first_fail] = step
        self.ever_failed |= failed

        command_x = self.env.commands[:, 0]
        tracking_error_x = torch.abs(command_x - self.env.base_lin_vel[:, 0])
        abs_lateral_vel = torch.abs(self.env.base_lin_vel[:, 1])
        tilt = torch.linalg.norm(self.env.projected_gravity[:, :2], dim=1)
        contact = self.env.simulator.contact_forces[:, self.env.feet_indices, 2] > self.contact_force_threshold
        contact_count = torch.sum(contact.float(), dim=1)
        disturbance_active = self._active_disturbance()
        random_action_abs = self._random_action_abs()
        arm_swing_amp, arm_opposition_error, endpoint_delta = self._arm_metrics()

        self.sums["reward"] += self._mean_item(actor_state["rewards"])
        self.sums["tracking_error_x"] += self._mean_item(tracking_error_x)
        self.sums["abs_lateral_vel"] += self._mean_item(abs_lateral_vel)
        self.sums["tilt"] += self._mean_item(tilt)
        self.sums["max_tilt"] = max(self.sums["max_tilt"], float(torch.max(tilt).detach().cpu().item()))
        self.sums["contact_count"] += self._mean_item(contact_count)
        self.sums["upper_action_abs"] += self._mean_item(random_action_abs)
        self.sums["disturbance_active_frac"] += self._mean_item(disturbance_active.float())
        self.sums["arm_swing_amp"] += self._mean_item(arm_swing_amp)
        self.sums["arm_opposition_error"] += self._mean_item(arm_opposition_error)
        self.sums["endpoint_sagittal_delta"] += self._mean_item(endpoint_delta)
        self.samples += 1

        self._update_recovery_metrics(step, disturbance_active, failed)

        if self.write_timeseries:
            self.timeseries.append(
                {
                    "time_s": step * self.env.dt,
                    "command_x": self._mean_item(command_x),
                    "base_vel_x": self._mean_item(self.env.base_lin_vel[:, 0]),
                    "tracking_error_x": self._mean_item(tracking_error_x),
                    "abs_lateral_vel": self._mean_item(abs_lateral_vel),
                    "tilt": self._mean_item(tilt),
                    "contact_count": self._mean_item(contact_count),
                    "disturbance_active_frac": self._mean_item(disturbance_active.float()),
                    "upper_action_abs": self._mean_item(random_action_abs),
                    "done_frac": self._mean_item(done.float()),
                    "fall_frac": self._mean_item(failed.float()),
                    "arm_swing_amp": self._mean_item(arm_swing_amp),
                    "arm_opposition_error": self._mean_item(arm_opposition_error),
                    "endpoint_sagittal_delta": self._mean_item(endpoint_delta),
                }
            )

        if step >= self.max_steps:
            actor_state["stop"] = True
        return actor_state

    def _tilt_degrees(self, tilt_value):
        clipped = min(max(float(tilt_value), 0.0), 1.0)
        return math.degrees(math.asin(clipped))

    def _summary(self):
        denom = max(self.samples, 1)
        horizon_s = self.samples * self.env.dt
        first_failure = self.first_failure_step.clone()
        survival_steps = torch.where(
            first_failure >= 0,
            first_failure,
            torch.full_like(first_failure, self.samples),
        )
        mean_tilt = self.sums["tilt"] / denom
        max_tilt = self.sums["max_tilt"]
        return {
            "num_envs": int(self.num_envs),
            "eval_steps": int(self.samples),
            "horizon_s": float(horizon_s),
            "survival_rate": float((~self.ever_failed).float().mean().cpu().item()),
            "fall_rate": float(self.ever_failed.float().mean().cpu().item()),
            "falls_total": int(torch.sum(self.fall_count).detach().cpu().item()),
            "falls_per_env_per_min": float(
                torch.sum(self.fall_count).detach().cpu().item()
                / max(self.num_envs * horizon_s / 60.0, 1.0e-6)
            ),
            "mean_survival_time_s": float(torch.mean(survival_steps.float()).cpu().item() * self.env.dt),
            "mean_reward_per_step": self.sums["reward"] / denom,
            "mean_tracking_error_x_mps": self.sums["tracking_error_x"] / denom,
            "mean_abs_lateral_vel_mps": self.sums["abs_lateral_vel"] / denom,
            "mean_tilt_proxy": mean_tilt,
            "mean_tilt_deg": self._tilt_degrees(mean_tilt),
            "max_tilt_proxy": max_tilt,
            "max_tilt_deg": self._tilt_degrees(max_tilt),
            "mean_contact_count": self.sums["contact_count"] / denom,
            "disturbance_active_frac": self.sums["disturbance_active_frac"] / denom,
            "mean_upper_action_abs": self.sums["upper_action_abs"] / denom,
            "recovery_success_rate": (
                float(self.recovered_pulses) / self.ended_pulses
                if self.ended_pulses > 0
                else None
            ),
            "recovered_pulses": int(self.recovered_pulses),
            "ended_pulses": int(self.ended_pulses),
            "mean_recovery_time_s": (
                float(sum(self.recovery_times) / len(self.recovery_times))
                if self.recovery_times
                else None
            ),
            "mean_arm_swing_amp_rad": self.sums["arm_swing_amp"] / denom,
            "mean_arm_opposition_error_rad": self.sums["arm_opposition_error"] / denom,
            "mean_endpoint_sagittal_delta_m": self.sums["endpoint_sagittal_delta"] / denom,
        }

    def _write_timeseries(self):
        if not self.write_timeseries or not self.timeseries:
            return
        path = self.output_dir / "eval_metrics_timeseries.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.timeseries[0].keys()))
            writer.writeheader()
            writer.writerows(self.timeseries)
        logger.info(f"Wrote eval metric time series to {path}")

    def on_post_evaluate_policy(self):
        summary = self._summary()
        summary_path = self.output_dir / "eval_metrics_summary.json"
        with summary_path.open("w") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
        self._write_timeseries()
        logger.info(f"Wrote eval metric summary to {summary_path}")
        logger.info(f"Eval metric summary: {json.dumps(summary, sort_keys=True)}")
