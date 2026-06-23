# Evaluation Results for Poster/Paper Writing Agent

## Context

This document summarizes the quantitative evaluation runs stored under `logs_eval/`.
The goal is to support the final poster on:

**Stable Arm-Swing Humanoid Locomotion under Upper-Body Perturbations**

The evaluation compares:

- **Stage8**: final policy trained/evaluated with upper-body random action perturbations.
- **Stage7**: arm-swing policy before the Stage8 perturbation-robustness setting; used as the main baseline.

We do **not** recommend using the 10DoF lower-body baseline as a quantitative robustness baseline, because it does not expose the same 19DoF upper-body action space. It is better used only as a motivation/baseline-development reference.

## Evaluation Protocol

All reported runs use:

- `num_envs = 32`
- `MAX_STEPS = 1000`
- Simulation horizon: `20.0 s`
- Commanded forward velocity: `[0.6, 0.0, 0.0]` m/s
- Headless evaluation on the server
- Upper-body pulse perturbations applied to torso, shoulders, and elbows
- Pulse active fraction: approximately 22-23% of the rollout

Source result files:

- Stage8 amp0: `logs_eval/logs_eval/metrics_amp0/20260623_030243/eval_metrics_summary.json`
- Stage8 amp1: `logs_eval/logs_eval/stage8_amp1/20260623_032932/eval_metrics_summary.json`
- Stage8 amp2: `logs_eval/logs_eval/stage8_amp2/20260623_033536/eval_metrics_summary.json`
- Stage7 amp2: `logs_eval/logs_eval/stage7_amp2/20260623_033732/eval_metrics_summary.json`

Checkpoints:

- Stage8: `logs/20260622_143537-pro-stage8/model_5000.pt`
- Stage7: `logs/20260622_122823-pro-stage7/model_3200.pt`

## Main Quantitative Table

| Policy | Eval perturbation amp | Survival rate ↑ | Fall rate ↓ | Mean survival time ↑ | Falls total ↓ | Tracking error x ↓ | Mean lateral velocity ↓ | Mean tilt ↓ | Max tilt ↓ | Recovery success ↑ | Mean reward / step ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage8 | 0.0 | 100.0% | 0.0% | 20.00 s | 0 | 0.0228 m/s | 0.0508 m/s | 3.30 deg | 4.67 deg | 100.0% | 0.1346 |
| Stage8 | 1.0 | 100.0% | 0.0% | 20.00 s | 0 | 0.0225 m/s | 0.0542 m/s | 3.53 deg | 7.81 deg | 100.0% | 0.1323 |
| Stage8 | 2.0 | 100.0% | 0.0% | 20.00 s | 0 | 0.0245 m/s | 0.0564 m/s | 3.40 deg | 9.12 deg | 100.0% | 0.1239 |
| Stage7 baseline | 2.0 | 0.0% | 100.0% | 5.48 s | 95 | 0.1401 m/s | 0.1335 m/s | 6.26 deg | 89.93 deg | 89.5% | 0.0463 |

## Arm-Swing / Upper-Body Metrics

| Policy | Amp | Mean upper action abs | Arm swing amp | Arm opposition error ↓ | Endpoint sagittal delta |
|---|---:|---:|---:|---:|---:|
| Stage8 | 0.0 | 0.0000 | 0.2317 rad | 0.4239 rad | 0.1035 m |
| Stage8 | 1.0 | 0.1095 | 0.2141 rad | 0.3249 rad | 0.1098 m |
| Stage8 | 2.0 | 0.2168 | 0.2536 rad | 0.3952 rad | 0.1165 m |
| Stage7 baseline | 2.0 | 0.2079 | 0.2638 rad | 0.2997 rad | 0.1459 m |

Interpretation:

- Stage8 preserves 100% survival even as the perturbation amplitude increases to 2.0.
- Stage8 tracking error remains low, changing only from 0.0228 m/s to 0.0245 m/s from amp0 to amp2.
- Stage8 max tilt increases under stronger perturbation, but remains bounded below 10 degrees.
- Stage7 has visibly active arm motion, but under the same amp2 disturbance it fails in every environment within the 20 s evaluation window.
- Stage7's recovery success metric alone is misleading because the policy can briefly satisfy the recovery criterion after some pulses, while still suffering repeated falls/resets overall. For the baseline comparison, survival rate, fall rate, mean survival time, and max tilt should be emphasized.

## Metric Definitions

**Survival rate**  
Fraction of environments that never triggered a non-timeout reset during the evaluation horizon. Higher is better.

**Fall rate**  
Fraction of environments that experienced at least one non-timeout reset. Lower is better.

**Falls total**  
Total number of non-timeout resets across all 32 environments. This can exceed the number of environments because an environment can reset and fail again.

**Mean survival time**  
Mean time until first failure. If an environment never fails, its survival time is counted as the full 20 s horizon.

**Tracking error x**  
Mean absolute error between commanded forward velocity and measured forward base velocity. Lower means better command tracking.

**Mean lateral velocity**  
Mean absolute lateral base velocity. Lower means the robot walks more straightly and drifts less sideways.

**Tilt proxy / tilt degrees**  
The evaluation computes `norm(projected_gravity_xy)` as a posture tilt proxy and converts it to degrees with `asin(proxy)`. Lower values indicate a more upright base. Max tilt is especially useful for perturbation robustness.

**Recovery success rate**  
For each upper-body perturbation pulse, the callback checks whether the robot returns to at least 80% of commanded forward speed while staying under the tilt threshold. This is useful for Stage8, but should not be used alone when the policy has many falls.

**Arm swing amplitude**  
Mean absolute shoulder-pitch displacement, averaged over left and right shoulders. Higher indicates more visible shoulder-pitch swing.

**Arm opposition error**  
Absolute value of `left_shoulder_pitch + right_shoulder_pitch`. Lower indicates stronger anti-phase left-right arm coordination.

**Endpoint sagittal delta**  
Absolute forward/backward separation between left and right elbow endpoints in the base frame. Higher indicates more visible sagittal arm swing.

## Suggested Poster Claims

Use conservative wording:

1. **Robustness under upper-body perturbation**  
   Stage8 maintained 100% survival across 32 parallel environments for 20 s under upper-body action pulse amplitudes up to 2.0.

2. **Tracking is preserved under disturbance**  
   Forward velocity tracking error remained below 0.025 m/s at amp2, compared with 0.0228 m/s without perturbation.

3. **Perturbation-induced tilt remains bounded**  
   Max tilt increased with perturbation amplitude, but remained below 10 degrees for Stage8 at amp2.

4. **Stage8 outperforms Stage7 under the same perturbation protocol**  
   Under amp2, Stage7 failed in all 32 environments, with mean survival time 5.48 s and max tilt near 90 degrees, while Stage8 survived the full 20 s horizon with no falls.

5. **Arm-swing is retained in the robust policy**  
   Stage8 still shows nonzero shoulder-pitch swing and elbow endpoint sagittal displacement under perturbation, indicating that the robust behavior is not simply produced by removing upper-body motion.

## Suggested Poster Table Caption

**Quantitative robustness evaluation.** We evaluate 32 parallel H1 humanoids for 20 s at a commanded forward velocity of 0.6 m/s. Upper-body pulse perturbations are injected into torso, shoulder, and elbow actions. Stage8 maintains full survival and low velocity-tracking error up to perturbation amplitude 2.0, while the Stage7 arm-swing baseline fails under the same amp2 perturbation.

## Caveats for Paper/Poster Agent

- These are short-horizon evaluations intended for a course poster, not a full benchmark.
- The evaluation is performed in simulation only.
- The amp0 run still uses the perturbation scheduling machinery, but the perturbation amplitude is zero.
- Recovery time is reported as 0.0 s for Stage8 because the recovery condition is already satisfied at pulse end in these rollouts. Prefer reporting recovery success rate and survival metrics.
- Do not claim sim-to-real performance.
- Do not compare Stage8 quantitatively against the 10DoF lower-body baseline for upper-body perturbation robustness.
