# Poster Content Plan

Working title: 上半身扰动下人形机器人摆臂行走策略的稳定性

English subtitle: Stable Arm-Swing Humanoid Locomotion under Upper-Body Perturbations

Status: academic-prose content draft for review before HTML poster integration.

## 0. Narrative Principle

This poster should read like a compact robotics paper, not a checklist. Most cards should contain one short academic paragraph plus a figure, equation, or table. Lists should be used sparingly, mainly for the contribution statement and the final takeaway.

The central claim is that stable full-body humanoid walking under upper-body motion is best framed as a staged robustness problem. We first preserve a reliable lower-body locomotion policy in the full 19DoF H1 action space, then introduce command-routed upper-body coordination, and finally evaluate the resulting policy under injected torso, shoulder, and elbow action disturbances.

## 1. Header

### Title

上半身扰动下人形机器人摆臂行走策略的稳定性

### Subtitle

Stable Arm-Swing Humanoid Locomotion under Upper-Body Perturbations

### Author Line

队员：请替换为组员姓名 · Shanghai Innovation Institute · Decision ML Big Homework · HumanoidVerse / Genesis / PPO

### One-Sentence Result

通过 19DoF 全身控制、命令条件化的 reward routing 与上半身 action pulse 评估，我们让 H1 人形机器人从上半身锁定行走过渡到协调摆臂，并在随机上半身扰动后继续保持稳定前进。

### Key Numbers

| Quantity               | Value                                                  |
| ---------------------- | ------------------------------------------------------ |
| Controllable joints    | 19DoF H1 full-body action space                        |
| Main stages visualized | Stage1, Stage6, Stage7, Stage8                         |
| Evaluation command     | 0.6 m/s forward walking                                |
| Perturbation protocol  | Held random action offsets on torso, shoulders, elbows |

When server metrics arrive, replace the last two cells with exact values such as evaluation horizon, number of environments, and perturbation amplitudes.

## 2. Research Question

解锁上半身后，人形机器人行走不再是单纯的下肢 tracking 问题。torso、shoulder 和 elbow 的动作会改变角动量分布、接触相位和姿态稳定性；如果直接把摆臂奖励叠加到 locomotion reward 上，策略可能学到上肢乱摆、躯干代偿或前向速度退化。本文将课程目标重新表述为一个稳定性问题：如何在保持前向 locomotion 的同时释放上肢自由度，并验证策略在上半身随机扰动后的恢复能力。

Suggested visual: a compact three-state schematic showing locked upper body, coordinated arm swing, and perturbed upper body during walking.

## 3. Contributions

This is one of the few places where bullets are appropriate because conference posters often use a compact contribution card.

- We migrate the task from a lower-body-only baseline to a 19DoF full-body H1 policy, enabling direct control of torso, shoulder, and elbow joints.
- We design a staged reward-routing curriculum that protects stable locomotion before strengthening shoulder-pitch coordination and endpoint swing amplitude.
- We introduce an upper-body action-pulse evaluation protocol that perturbs only torso, shoulder, and elbow actions while measuring whether the gait recovers.
- We combine temporal video evidence with finite-horizon evaluation metrics, allowing the final poster to support both qualitative and quantitative claims.

## 4. Task Mapping

The original assignment asks for humanoid locomotion with unlocked upper-body motion in simulation. Our implementation maps this requirement into a full-body stability benchmark: Stage1 verifies that the 19DoF model can preserve stable walking with the upper body constrained; Stage6 introduces command-conditioned reward routing; Stage7 produces visible arm swing; Stage8 tests robustness by injecting upper-body action offsets during evaluation. This creates a coherent progression from course deliverable to paper-style experimental narrative.

| Course requirement   | Paper-style interpretation                       | Evidence source                       |
| -------------------- | ------------------------------------------------ | ------------------------------------- |
| 前进一定距离         | Forward command tracking under stable locomotion | Stage1 / Stage7 videos                |
| 上半身解锁并同步运动 | Gait-synchronized shoulder-pitch swing           | Stage7 temporal strip                 |
| 仿真实验             | Genesis rollout videos and finite-horizon eval   | Video folder and metrics              |
| 鲁棒性加分           | Recovery under upper-body action pulses          | Stage8 disturbance videos and metrics |

## 5. Main Figure

### Figure 1. Dynamic Evidence for Robust Arm-Swing Locomotion

The main figure should be a three-row temporal strip rather than a grid of unrelated screenshots. Row A shows a locked-upper-body walking baseline from `stage1_1000_只有走路.mp4`, establishing that stable forward locomotion is preserved before arm swing is introduced. Row B uses `stage7_3200_走路摆臂.mp4` to show visible shoulder-pitch motion synchronized with the gait. Row C uses `扰动2.mp4` as the primary robustness evidence, because its middle frames show clearer upper-body deviation and recovery than `扰动1.mp4`.

Each row should contain four or five cropped frames sampled at increasing timestamps. The visual annotations should be minimal and paper-like: a gray label for the locked baseline, amber arcs for arm-swing motion, a red arrow for the active disturbance pulse, and a green arrow for post-pulse recovery. All frames must be cropped to remove OS bars, dock icons, terminal overlays, and OBS recursion. The checkerboard floor should remain visible enough to make forward progression legible.

Proposed caption:

Temporal strips show the transition from stable locked-upper-body walking to gait-synchronized arm swing and finally to recovery under injected upper-body action pulses. The qualitative claim is temporal rather than static: the robot continues walking while upper-body behavior changes from constrained, to coordinated, to externally perturbed.

## 6. Method Figure

### Figure 2. Staged Reward Routing for Full-Body Locomotion

The method figure should be a horizontal pipeline with five blocks: full-body H1 model, stable locomotion base, command-based reward routing, arm-swing shaping, and upper-body pulse evaluation. This figure should visually communicate that the work is not a single reward trick, but a staged training and evaluation protocol designed to reduce interference between stability and upper-body expressiveness.

The central equation can be shown as:

```text
R = R_track + R_gait + R_phase + R_smooth - C_upper
```

Suggested caption:

The policy is trained in a 19DoF H1 action space. Stable walking is first preserved through tracking and heading rewards; command-based masks then route gait-specific terms to standing, walking, or straight-walk states; arm-swing rewards are strengthened only after locomotion remains stable; finally, held random action offsets are injected into upper-body joints to evaluate recovery.

## 7. Method Text

We train the H1 full-body controller with PPO in Genesis. The policy receives proprioceptive observations and outputs joint action targets for the full 19DoF model. Stage1 protects the locomotion foundation by keeping upper-body joints constrained while the robot learns stable forward walking. Stage6 borrows the reward-routing idea from gait-conditioned locomotion: command-derived masks determine when standing, walking, and straight-walk rewards are active. Stage7 increases shoulder-pitch and endpoint-sagittal swing objectives to make arm motion visible without sacrificing gait stability. Stage8 does not introduce a new behavior objective; instead, it evaluates the learned policy under upper-body action pulses applied to torso, shoulder, and elbow joints.

This paragraph can appear directly under Figure 2. It should be short enough that the figure remains dominant.

## 8. Quantitative Robustness Table

### Table 1. Finite-Horizon Perturbation Evaluation

After the server runs finish, this table should become the main quantitative result. It should compare the same policy under different upper-body action-pulse amplitudes, preferably `amp=0.0`, `amp=1.0`, and `amp=2.0`. If the strong perturbation degrades performance, that is still useful as long as the wording calls it a stress test rather than a perfect robustness claim.

| Eval amp | Survival ↑ | Tracking error ↓ | Max tilt ↓ | Recovery ↑ | Recovery time ↓ |
| -------: | ----------: | ----------------: | ----------: | ----------: | ---------------: |
|      0.0 |         TBD |               TBD |         TBD |         N/A |              N/A |
|      1.0 |         TBD |               TBD |         TBD |         TBD |              TBD |
|      2.0 |         TBD |               TBD |         TBD |         TBD |              TBD |

Metric explanation for poster text:

Survival is the fraction of evaluation environments that avoid a non-timeout reset during the finite horizon. Tracking error is the mean absolute difference between commanded and measured forward velocity. Max tilt is computed from the projected-gravity tilt proxy and reported in degrees. Recovery success measures whether the policy returns to both a forward-speed threshold and a posture threshold after a pulse ends. Recovery time is the elapsed time from pulse termination to recovery.

Suggested caption:

Finite-horizon evaluation under increasing upper-body action-pulse amplitudes. A robust controller should maintain high survival, low velocity-tracking error, bounded tilt, and short recovery time after the disturbance window.

## 9. Stage Story Table

### Table 2. From Baseline Walking to Perturbation Recovery

This table replaces a long ablation paragraph. It should make the experimental progression easy to scan while still sounding like a research narrative.

| Stage          | Upper-body treatment            | Experimental role                                             | Evidence                     |
| -------------- | ------------------------------- | ------------------------------------------------------------- | ---------------------------- |
| 10DoF baseline | Upper body unavailable or fixed | Lower-body locomotion reference                               | Baseline videos              |
| 19DoF Stage1   | Upper body action-locked        | Verify stable full-body model setup                           | `stage1_1000_只有走路.mp4` |
| Stage6         | Shoulder pitch lightly routed   | Add command-conditioned gait rewards without breaking walking | Training/eval logs           |
| Stage7         | Visible arm swing               | Demonstrate gait-synchronized upper-body motion               | `stage7_3200_走路摆臂.mp4` |
| Stage8         | Random upper-body action pulses | Stress-test recovery under torso/arm disturbances             | `扰动1/2.mp4` and metrics  |

Suggested caption:

The experimental progression isolates why each stage is needed: stable walking must first be preserved, upper-body motion must then be introduced through routed rewards, and robustness must finally be tested by perturbing the upper-body action channels.

## 10. Key Findings

This is the second appropriate bullet-list area. It should be short and placed near the conclusion, not scattered across the whole poster.

- Stable arm swing emerges more reliably when locomotion is protected before upper-body rewards are strengthened.
- Command-routed rewards make shoulder-pitch coordination easier to interpret because arm-swing objectives are active mainly during straight walking.
- Upper-body robustness should be evaluated through recovery after disturbance pulses, not only through visual smoothness.
- The final behavior supports the course goal while also forming a paper-style story: staged curriculum, qualitative evidence, and finite-horizon metrics.

## 11. Limitations

Current evidence is simulation-only and relies on Genesis rollout videos plus finite-horizon evaluation. The naturalness of the upper-body motion is still guided by hand-designed phase and amplitude rewards rather than motion priors or human demonstrations. The quantitative evaluation focuses on survival, tracking, tilt, and recovery; a stronger future study would add angular-momentum decomposition, energy cost, longer horizons, and cleaner camera recordings.

This section should be small. If poster space is tight, fold it into the conclusion footer.

## 12. Final Conclusion

Stable upper-body motion is not obtained by simply unlocking more joints. In our experiments, the reliable path is to first preserve locomotion in the full-body action space, then introduce command-routed arm coordination, and finally evaluate whether the gait survives explicit upper-body perturbations. The resulting policy demonstrates stable forward walking, visible arm swing, and recovery behavior under torso/shoulder/elbow action pulses.

## 13. Figure and Asset Requirements

The final poster needs four core visual assets. The first is the main temporal strip figure built from `stage1_1000_只有走路.mp4`, `stage7_3200_走路摆臂.mp4`, and `扰动2.mp4`. The second is a clean method pipeline drawn directly in HTML/CSS. The third is the finite-horizon robustness table filled from `eval_metrics_summary.json`. The fourth is the stage story table above. Optional additions include a command-versus-velocity plot from `eval_metrics_timeseries.csv` or a compact arm-swing phase plot, but these should only be included if they are visually clean and do not compete with the main temporal figure.

Video frames should be uniformly cropped, brightened, and annotated. Raw full-screen recordings, OS UI, and recursive OBS frames should not appear in the final poster.

## 14. Metric Ingestion

When server results return, fill Table 1 from these fields in `eval_metrics_summary.json`: `survival_rate`, `mean_tracking_error_x_mps`, `max_tilt_deg`, `recovery_success_rate`, `mean_recovery_time_s`, `mean_arm_swing_amp_rad`, and `mean_endpoint_sagittal_delta_m`. Use percentages for survival and recovery, m/s for velocity tracking error, degrees for tilt, seconds for recovery time, radians for shoulder swing amplitude, and meters for endpoint sagittal delta. If `recovery_success_rate` is null for `amp=0.0`, display N/A because no perturbation pulse was evaluated.

## 15. References Footer

Use short, unobtrusive citations in the footer rather than a large bibliography block. The most relevant references are Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion, Humanoid-Gym, DeepMimic, AMP, and the HumanoidVerse/Genesis/PPO codebase used in this project.

Footer text:

Videos and metrics are generated from Genesis evaluations of the trained H1 policy. Quantitative values should be filled from finite-horizon evaluation runs before final submission.

## 16. Review Questions

The next review should decide whether the Chinese title should remain primary or whether the English title should lead for a stronger conference-poster feel. We also need to decide whether the center figure should stay as three temporal rows or reserve one row for a velocity-tracking plot after the server metrics arrive. The final robustness wording should be chosen after checking whether `amp=2.0` remains stable or should be described as a stress-test regime.
