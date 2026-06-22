# 上肢自然摆臂 Reward 设计

本文总结当前仓库中 H1 全身 19DoF 上肢自然摆臂相关的阶段化 reward 设计，并说明不同 stage 之间的 reward 区分。

主要入口是 `train_genesis_pro_staged.sh`，它现在支持 `--stage 1` 到 `--stage 5`。主线配置位于 `humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage*.yaml`。另外还有 cadence 分支、stage3 实验分支、单阶段配置和 arm-swing teacher 配置，本文在后面单独列出。

## 设计目标

目标是在 H1 全身 19DoF locomotion 中，让机器人先学会稳定直线前进，再逐步引入肩部自然摆臂。这里的“自然”不是单纯让手臂自由运动，而是让肩 pitch 产生和腿部步态相关的前后反相运动，同时压住 torso 扭动、肩 roll/yaw 横向甩臂、肘部怪异姿态和上肢高频抖动。

## Reward 实现位置

- locomotion 基础项：`humanoidverse/envs/locomotion/locomotion.py`
- 上肢摆臂项：`humanoidverse/envs/locomotion/locomotion_upper_body.py`
- 主线 stage 配置：`humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage1.yaml` 到 `stage5.yaml`
- cadence 分支配置：`reward_h1_locomotion_upper_body_cadence_stage1.yaml` 到 `cadence_stage3.yaml`
- stage3 实验分支：`reward_h1_locomotion_upper_body_stage3_exp01_straight_gated.yaml` 到 `exp03_torso_robust.yaml`
- 单阶段配置：`reward_h1_locomotion_upper_body.yaml`
- teacher 配置：`reward_h1_arm_swing_teacher.yaml`

## Reward 项分类

### 基础 locomotion 与直线行走

- `tracking_lin_vel`：跟踪 x/y 线速度命令。
- `tracking_lin_vel_x`：额外强调前向 x 速度跟踪。
- `tracking_ang_vel`：跟踪 yaw 角速度命令。
- `tracking_heading`：跟踪目标 heading。
- `penalty_low_forward_speed`：命令前进时惩罚前向速度不足。
- `penalty_backward_vel`：惩罚后退速度。
- `penalty_heading_error`：移动时惩罚 heading 偏差。
- `penalty_lateral_vel`：惩罚 base 横向速度，减少走偏。
- `feet_air_time`：鼓励步态中脚离地时间。
- `penalty_slippage`、`feet_heading_alignment`：抑制脚滑和脚朝向偏离。

### 上肢稳定与安全

- `upperbody_locked_dof_pos`：按 stage 把指定上肢/torso DOF 拉回默认位置。
- `upperbody_action_rate`：惩罚上肢 action 快速变化。
- `upperbody_dof_acc`：惩罚上肢关节加速度。
- `upperbody_torques`：惩罚上肢力矩。
- `upperbody_torso_deviation`：限制 torso 偏离默认角度。
- `upperbody_arm_posture`：限制肩 roll/yaw 和肘关节偏离默认姿态。
- `upperbody_shoulder_pitch_limit`：肩 pitch 软限位。
- `upperbody_elbow_posture`：让 elbow 保持轻微弯曲或接近目标弯曲角。
- `upperbody_stationary_arm_posture`：命令速度很小时让上肢回中。
- `upperbody_arm_symmetry`：左右肩 pitch 反相，roll/yaw 镜像平衡。
- `upperbody_arm_lateral_deviation`：直接压住 shoulder roll/yaw 横向偏移。
- `upperbody_arm_endpoint_lateral_pos`、`upperbody_arm_endpoint_lateral_vel`：约束 `elbow_link` 在 base frame 下的横向位置和横向速度。

### 上肢摆臂相位与 teacher

- `upperbody_arm_endpoint_sagittal_phase`：约束左右 `elbow_link` 前后 x 位置差与左右髋 pitch 相位匹配。
- `upperbody_arm_endpoint_sagittal_balance`：约束双臂前后平均位置，避免整体前伸或后缩。
- `upperbody_arm_endpoint_sagittal_vel_phase`：约束左右 `elbow_link` 前后速度差与髋部速度相位匹配，并抑制双臂同向运动。
- `upperbody_arm_endpoint_sagittal_amplitude`：不再追具体髋部相位，只要求左右手臂末端前后差达到最小摆幅。
- `upperbody_arm_velocity_opposition`：左右肩 pitch 速度反相。
- `upperbody_arm_leg_phase`：肩 pitch 跟对侧髋 pitch 建立角度相位关系。
- `upperbody_arm_leg_velocity_phase`：肩 pitch 速度跟对侧髋 pitch 速度建立相位关系。
- `upperbody_teacher_arm_swing_pitch`：肩 pitch 直接追随 sinusoidal teacher target。
- `upperbody_teacher_arm_swing_velocity`：肩 pitch 速度直接追随 teacher velocity target。
- `upperbody_teacher_endpoint_sagittal`、`upperbody_teacher_endpoint_sagittal_velocity`：teacher 配置中使用的末端前后差及速度目标。

## 主线五阶段区分

| Stage | 配置 | 动作锁定 | 核心 reward 目标 | 和上一阶段的主要区别 |
| --- | --- | --- | --- | --- |
| Stage 1 | `reward_h1_locomotion_upper_body_stage1.yaml` | 锁 torso + 双臂全部 9 个上肢 DOF | 只保稳定 locomotion 和直线行走，上肢不参与摆臂 | 引入全身 19DoF 机器人，但通过 `upperbody_locked_dof_pos=-4.0` 冻结上肢 |
| Stage 2 | `stage2.yaml` | 锁 torso、肩 roll/yaw、elbow，只放开左右 shoulder pitch | shoulder pitch 小幅前后预热，同时保直线行走 | 加入 endpoint 前后相位、横向末端约束、左右肩速度反相和弱肩-髋相位 |
| Stage 3 | `stage3.yaml` | 只锁 torso，双臂放开 | 完整上肢自然摆臂 | 放开 roll/yaw/elbow，但用横向、对称、肘部、站立回中和 endpoint balance 约束防止乱摆 |
| Stage 4 | `stage4.yaml` | 继承 stage3，只锁 torso | 直接塑造 shoulder pitch 摆臂幅度 | 关闭 stage3 的间接髋-臂相位项，改用 `upperbody_teacher_arm_swing_pitch` 和 `upperbody_arm_endpoint_sagittal_amplitude` |
| Stage 5 | `stage5.yaml` | 继承 stage4，只锁 torso | 更小、更保守的 shoulder pitch 摆臂 | 把 teacher pitch 从 `-16.0` 降到 `-8.0`，最小末端前后差从 `0.10` 降到 `0.04`，更适合收敛后微调 |

### 主线 locomotion/直线行走 scales

这些项保证所有 stage 的第一目标仍是走稳、走直，而不是为了摆臂牺牲 locomotion。

| Reward | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tracking_lin_vel` | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| `tracking_lin_vel_x` | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| `tracking_ang_vel` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| `tracking_heading` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `feet_air_time` | 0.3 | 0.4 | 0.5 | 0.5 | 0.5 |
| `penalty_low_forward_speed` | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 |
| `penalty_backward_vel` | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| `penalty_heading_error` | -2.0 | -2.0 | -1.5 | -1.5 | -1.5 |
| `penalty_lateral_vel` | -1.0 | -1.0 | -0.8 | -0.8 | -0.8 |
| `penalty_slippage` | -0.5 | -0.6 | -0.6 | -0.6 | -0.6 |
| `feet_heading_alignment` | -0.1 | -0.1 | -0.15 | -0.15 | -0.15 |

### 主线上肢稳定 scales

| Reward | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `upperbody_locked_dof_pos` | -4.0 | -3.0 | 0.0 | 0.0 | 0.0 |
| `upperbody_action_rate` | -0.02 | -0.02 | -0.02 | -0.004 | -0.004 |
| `upperbody_dof_acc` | -5e-8 | -5e-8 | -5e-8 | -8e-9 | -8e-9 |
| `upperbody_torques` | -5e-6 | -5e-6 | -5e-6 | -8e-7 | -8e-7 |
| `upperbody_torso_deviation` | -1.5 | -1.5 | -1.2 | -1.2 | -1.2 |
| `upperbody_arm_posture` | -0.3 | -0.3 | 0.0 | 0.0 | 0.0 |
| `upperbody_shoulder_pitch_limit` | -0.4 | -0.7 | -0.8 | -0.8 | -0.8 |
| `upperbody_elbow_posture` | 0.0 | 0.0 | -0.2 | 0.0 | 0.0 |
| `upperbody_stationary_arm_posture` | 0.0 | -0.4 | -0.6 | 0.0 | 0.0 |
| `upperbody_arm_symmetry` | 0.0 | -0.05 | -0.25 | -0.35 | -0.35 |
| `upperbody_arm_lateral_deviation` | 0.0 | -0.8 | -1.2 | -1.2 | -1.2 |
| `upperbody_arm_endpoint_lateral_pos` | 0.0 | -4.0 | -8.0 | 0.0 | 0.0 |
| `upperbody_arm_endpoint_lateral_vel` | 0.0 | -0.1 | -0.25 | 0.0 | 0.0 |

### 主线摆臂 scales

| Reward | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `upperbody_arm_endpoint_sagittal_phase` | 0.0 | -0.8 | -0.8 | 0.0 | 0.0 |
| `upperbody_arm_endpoint_sagittal_balance` | 0.0 | 0.0 | -4.0 | 0.0 | 0.0 |
| `upperbody_arm_endpoint_sagittal_vel_phase` | 0.0 | -0.02 | -0.02 | 0.0 | 0.0 |
| `upperbody_arm_endpoint_sagittal_amplitude` | 0.0 | 0.0 | 0.0 | -35.0 | -10.0 |
| `upperbody_arm_velocity_opposition` | 0.0 | -0.02 | -0.02 | 0.0 | 0.0 |
| `upperbody_arm_leg_phase` | 0.0 | -0.04 | -0.12 | 0.0 | 0.0 |
| `upperbody_arm_leg_velocity_phase` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `upperbody_teacher_arm_swing_pitch` | 0.0 | 0.0 | 0.0 | -16.0 | -8.0 |
| `upperbody_teacher_arm_swing_velocity` | 0.0 | 0.0 | 0.0 | -0.04 | 0.0 |

Stage4/5 的关键变化是：它们继承 stage3 的 locomotion 和安全约束，但把 stage3 的间接 phase reward 关掉，转向“肩 pitch 直接 teacher + 末端前后最小幅度”。这更直接地让肩 pitch 动起来，但也更像 teacher shaping，不再主要依赖腿部相位自然涌现。

注意：stage4/5 继承了 stage3 的 `reward_penalty_reward_names`，新加的 `upperbody_teacher_arm_swing_pitch`、`upperbody_teacher_arm_swing_velocity`、`upperbody_arm_endpoint_sagittal_amplitude` 没有被加入该列表。因此在打开 `rewards.reward_penalty_curriculum=True` 时，这些新 teacher/amplitude 项不会像 stage3 的部分 penalty 项一样被 curriculum 缩放，而是按配置 scale 直接生效。

## 各 stage 设计逻辑

### Stage 1：稳定走路，上肢冻结

入口：

```sh
bash train_genesis_pro_staged.sh --stage 1
```

Stage1 的任务是让 19DoF H1 先在全身模型下学会稳定 locomotion。它通过 `locked_action_dof_names` 和 `locked_reward_dof_names` 锁住 torso、左右 shoulder pitch/roll/yaw 和 elbow。摆臂相位项全部为 0，reward 重点是前进速度、heading、横向速度惩罚、脚部 air time 和上肢冻结误差。

### Stage 2：只放开 shoulder pitch

入口：

```sh
bash train_genesis_pro_staged.sh --stage 2 checkpoint=/path/to/stage1/model_x.pt algo.config.load_optimizer=False
```

Stage2 仍锁 torso、shoulder roll/yaw 和 elbow，只允许左右 shoulder pitch 前后小幅运动。它新增横向 endpoint 约束和弱相位 shaping：`upperbody_arm_endpoint_sagittal_phase=-0.8`、`upperbody_arm_leg_phase=-0.04`、`upperbody_arm_velocity_opposition=-0.02`。这个阶段不是追求明显摆臂，而是让 shoulder pitch 知道“可以前后反相动”。

### Stage 3：完整上肢自然摆臂

入口：

```sh
bash train_genesis_pro_staged.sh --stage 3 checkpoint=/path/to/stage2/model_x.pt algo.config.load_optimizer=False
```

Stage3 只锁 torso，双臂全部放开。它把 `upperbody_locked_dof_pos` 关到 0，同时用更强的横向约束和末端约束防止乱甩：`upperbody_arm_lateral_deviation=-1.2`、`upperbody_arm_endpoint_lateral_pos=-8.0`、`upperbody_arm_endpoint_lateral_vel=-0.25`。自然摆臂主要来自 `upperbody_arm_endpoint_sagittal_phase`、`upperbody_arm_endpoint_sagittal_balance`、`upperbody_arm_endpoint_sagittal_vel_phase` 和 `upperbody_arm_leg_phase`。

### Stage 4：肩 pitch 直接塑形

入口：

```sh
bash train_genesis_pro_staged.sh --stage 4 checkpoint=/path/to/stage3/model_x.pt algo.config.load_optimizer=False
```

Stage4 的目的是解决 stage3 如果手臂几乎不动或相位 reward 不够直接的问题。它放松平滑/能耗惩罚，把 `upperbody_action_rate` 从 `-0.02` 降到 `-0.004`，同时关闭 endpoint lateral、endpoint phase、velocity opposition 和 arm-leg phase 等间接项。真正驱动摆臂的是 `upperbody_teacher_arm_swing_pitch=-16.0`、`upperbody_teacher_arm_swing_velocity=-0.04` 和 `upperbody_arm_endpoint_sagittal_amplitude=-35.0`。

### Stage 5：小幅 shoulder swing 微调

入口：

```sh
bash train_genesis_pro_staged.sh --stage 5 checkpoint=/path/to/stage4/model_x.pt algo.config.load_optimizer=False
```

Stage5 继承 stage4，但显著减小摆臂强度：`teacher_shoulder_pitch_amplitude` 从 `0.30` 降到 `0.08`，`arm_endpoint_sagittal_min_delta` 从 `0.10` 降到 `0.04`，`upperbody_teacher_arm_swing_pitch` 从 `-16.0` 降到 `-8.0`，并关闭 teacher velocity 项。它适合在已经能动起来之后，把摆臂幅度收小，减少夸张或机械感。

## Cadence 分支 stage 区分

cadence 分支入口是 `experiments/upper_body_cadence_control/run_cadence_control_staged.sh`。它复用主线 stage1-stage3 的上肢锁定策略，但额外加入脚步节奏 reward，目标是让下肢步态更规律，从而让上肢摆臂不被不稳定步频牵着抖。

| Cadence stage | 继承 | 主要新增 reward | 上肢区别 |
| --- | --- | --- | --- |
| Cadence Stage 1 | 主线 Stage 1 | `feet_air_time_target=1.0`、`feet_single_stance_time=0.7`、`penalty_feet_air_time_short=-3.0`、`penalty_feet_swing_height=-4.0` | 上肢仍冻结 |
| Cadence Stage 2 | 主线 Stage 2 | air time/单支撑/摆脚高度项继续加强 | 比主线 Stage2 更弱的摆臂相位：`upperbody_arm_endpoint_sagittal_phase=-0.45`、`arm_leg_phase=-0.025` |
| Cadence Stage 3 | Stage3 exp02 | `feet_air_time_target=1.2`、`feet_single_stance_time=0.9`、`penalty_feet_swing_height=-5.0` | 使用 straight-walk gate、速度缩放和 elbow coupling，但摆臂速度目标更慢 |

cadence 分支的本质区别是：它先约束脚步节奏，再塑造上肢。主线更关注“直线走稳 + 上肢逐步解锁”；cadence 分支更关注“步态周期稳定 + 慢速摆臂”。

## Stage3 实验分支区分

| 实验配置 | 继承 | 主要 reward 改动 | 目的 |
| --- | --- | --- | --- |
| `stage3_exp01_straight_gated` | 主线 Stage 3 | `upperbody_arm_endpoint_sagittal_phase=-4.5`、`vel_phase=-0.12`、`arm_leg_phase=-0.14`，并开启 `use_straight_walk_arm_swing_mask=True` | 只在主要前进命令下强塑摆臂，避免转向/横移时强行套前进摆臂 |
| `stage3_exp02_speed_elbow` | exp01 | `upperbody_elbow_swing_coupling=-0.22`，摆臂目标随前向速度缩放 | 速度越快摆幅越大，同时肘部随肩摆轻微弯曲 |
| `stage3_exp03_torso_robust` | exp02 | torso 不再 action-lock，`upperbody_torso_deviation=-2.0`，`torso_deviation_deadband=0.08` | 测试 torso 可动但不让 torso 成为主要行走/摆臂作弊通道 |

这三个分支都不是主线必经 stage，而是围绕 stage3 的替代实验。若只追求最稳路线，优先按主线 stage1 到 stage5 训练；若 stage3 摆臂不自然，再用这些分支做对比。

## 单阶段与 Teacher 配置

`reward_h1_locomotion_upper_body.yaml` 是单阶段完整上肢 reward。它同时包含横向约束、endpoint 前后相位、endpoint balance、肩-髋相位和站立回中。优点是入口简单，缺点是从一开始就把 locomotion 和上肢摆臂混在一起，训练不稳定时不容易判断问题来自腿还是手臂。

`reward_h1_arm_swing_teacher.yaml` 不是 locomotion 主线 stage，而是训练/生成摆臂 teacher 的配置。它锁住腿、torso、肩 roll/yaw 和 elbow，主要让 shoulder pitch 追随 sinusoidal teacher：`upperbody_teacher_arm_swing_pitch=-18.0`、`upperbody_teacher_arm_swing_velocity=-0.15`、`upperbody_teacher_endpoint_sagittal=-0.8`。它适合给 `fuse_arm_teacher_with_walk.py` 这类 checkpoint 融合流程提供上肢 action head 参考。

## 论文依据

1. He et al., 2025, *Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion*：强调按步态/阶段逐步引入约束，先稳定 locomotion，再塑造更复杂的协调动作。
   https://arxiv.org/abs/2505.20619

2. Peng et al., 2018, *DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills*：自然动作最好来自 reference motion imitation；如果后续能拿到人类或机器人参考步态，可以把手写相位 reward 替换或补充为 imitation reward。
   https://arxiv.org/abs/1804.02717

3. Peng et al., 2021, *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*：当任务 reward 难以完整描述“自然”时，可以用动作先验/判别器提供 style reward，特别适合上肢自然性。
   https://arxiv.org/abs/2104.02180

4. Humanoid-Gym, 2024, *Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer*：实际 humanoid locomotion 需要同时考虑速度跟踪、姿态、能耗、动作平滑、关节/力矩约束和 sim-to-real 鲁棒性。
   https://arxiv.org/abs/2404.05695

## 调参判断

- 走不直：优先看 `tracking_heading`、`penalty_heading_error`、`penalty_lateral_vel`，不要先加强摆臂 reward。
- 手臂完全不动：stage2/3 可小步增加 `upperbody_arm_leg_phase` 或 endpoint sagittal phase；stage4 可使用 teacher/amplitude 直接塑形。
- 手臂左右甩：加强 `upperbody_arm_lateral_deviation`、`upperbody_arm_endpoint_lateral_pos`、`upperbody_arm_endpoint_lateral_vel`，并确认 shoulder roll/yaw 没有成为主要动作通道。
- 手臂前后摆太夸张：使用 stage5 的小幅 teacher，或降低 `teacher_shoulder_pitch_amplitude`、`arm_endpoint_sagittal_min_delta`。
- 肘部僵硬：用 exp02/cadence3 的 `upperbody_elbow_swing_coupling`，但不要过强，否则 elbow 会替 shoulder pitch 完成摆臂。
- 走路变差：降低上肢相位/teacher 项，先让 locomotion reward 恢复，再逐步加回摆臂项。