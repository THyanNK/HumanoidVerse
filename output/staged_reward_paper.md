# 从下肢行走到摆臂抗扰：H1 人形机器人多阶段 Reward 设计小论文

## 摘要

人形机器人行走策略如果只使用下肢自由度，往往可以较快学到稳定前进，却难以表现出自然的上肢协调运动；如果直接解锁上半身并叠加摆臂 reward，又容易破坏原有 locomotion，出现身体侧偏、上肢乱摆或摆臂完全不起作用的问题。本文基于 HumanoidVerse 与 Genesis 训练框架，设计了一套多阶段 reward curriculum：第一阶段在 19DoF 全身 H1 模型中锁定上肢，先恢复稳定下肢行走；第二阶段借鉴 gait-conditioned reinforcement learning 中的 reward routing 思想，在直线行走条件下释放肩部 pitch 并引导手臂与腿部反相摆动；第三阶段在摆臂策略基础上注入上肢随机 action pulse，使机器人学习在 torso、shoulder、elbow 受到随机动作扰动后继续保持直行和平衡。实验结果表明，最终策略在扰动幅度达到 2.0 时仍能在 32 个并行环境、20 秒评估窗口内保持 100% 存活，同时保留非零肩部摆幅和手臂末端前后位移。本文还总结了原始 `humanoidverse/` 中 stage2-stage5 的失败设计，说明“全局叠加更强摆臂 reward”并不能自动产生自然摆臂，关键在于先保护 locomotion，再按步态上下文分时激活上肢协调目标。

**关键词：** 人形机器人；强化学习；reward shaping；摆臂；gait-conditioned reward routing；抗扰动行走

## 1. 引言

本项目最初面对的问题并不是简单地“让机器人动起来”，而是让一个原本只能依靠下肢稳定行走的 H1 humanoid，在不破坏基础行走能力的前提下学会上肢自然摆臂，并进一步具备抵抗上肢随机干扰的能力。原始下肢 baseline 使用 10DoF H1 配置，`actions_dim=10` 且 `has_upper_body_dof=False`，上肢并不属于策略动作空间；而完整 H1 配置为 19DoF，包含 torso、双肩 pitch/roll/yaw 和双肘动作。直接把策略从 10DoF 改成 19DoF 不可行，因为动作维度、URDF/DOF 列表、观测维度和 checkpoint 网络结构都发生了变化。因此，我们采用了更保守的路线：在完整 19DoF H1 上重新训练，但先把上肢动作锁住，让策略在“可控但不参与”的上半身条件下恢复稳定行走。

这一路线的核心难点在于 reward 目标之间存在冲突。稳定直线行走要求速度 tracking、heading tracking、低侧向速度、低姿态倾斜和平滑动作；自然摆臂则要求肩部产生周期性、反相、随步态变化的前后摆动。如果摆臂 reward 在站立、转向、侧移和不稳定步态中都全局生效，它会向策略提供与当前 locomotion 目标不一致的梯度。早期失败实验说明，简单提高摆臂项权重或使用固定正弦 teacher，并不能稳定地把“会走路”转化成“会边走边摆臂”。

为了解决这个问题，我们将最终方法整理成发布版 `publish_code/` 中的三个有效阶段。需要特别说明的是，发布版只保留最终有效实验，编号与原始实验并不完全一致：`publish_code` 的 stage1、stage2、stage3 分别对应原始 `humanoidverse/` 中的 stage1、stage7、stage8；原始 `humanoidverse/` 的 stage2-stage5 是失败探索，不应与发布版 stage2 混淆。

| 发布版阶段 | 原始实验阶段 | 目标 | 主要配置 |
| --- | --- | --- | --- |
| stage1 | stage1 | 19DoF 模型中锁定上肢，训练稳定下肢行走 | `reward_h1_locomotion_upper_body_stage1.yaml` |
| stage2 | stage7 | 引入命令条件化摆臂 reward，释放双肩 pitch | `reward_h1_locomotion_upper_body_stage2.yaml` |
| stage3 | stage8 | 加入上肢随机动作扰动，训练扰动后恢复 | `reward_h1_locomotion_upper_body_stage3.yaml` 与 `upper_body_random_action.yaml` |

## 2. 方法总览

我们的 reward curriculum 可以写成如下形式：

```text
r = r_locomotion
  + r_upper_safety
  + m_stand * r_stand
  + m_walk * r_walk
  + m_straight * r_arm_swing
  + r_disturbance_robustness
```

其中 `r_locomotion` 是各阶段共享的速度、heading、姿态和足端稳定目标；`r_upper_safety` 用于限制上肢异常姿态、高频动作和躯干偏移；`m_stand`、`m_walk`、`m_straight` 是由速度命令构造的 mask，用于在不同步态上下文中激活不同 reward。这个形式体现了本文的核心思想：不是把所有 reward 同时打开，而是让 reward 在合适的状态下生效。

训练脚本 `publish_code/train_genesis_pro_staged.sh` 将三个阶段显式暴露给用户。stage1 使用 `NO_domain_rand`；stage2 从 stage1 checkpoint 继续训练；stage3 默认启用 `upper_body_random_action`，并从 stage2 checkpoint 继续训练。完整串联训练由 `publish_code/train_genesis_pro_all_stages.sh` 完成，保证每个阶段都从上一个阶段的最新 checkpoint 继承策略，而不是从零开始重新搜索。

## 3. Stage 1：先在全身模型中学会稳定下肢行走

Stage 1 的作用是建立一个可靠的 19DoF locomotion 基础。虽然机器人已经使用完整 H1 模型，动作空间也包含上半身，但 reward 和动作处理会把 torso、左右 shoulder pitch/roll/yaw、左右 elbow 全部锁在默认姿态附近。这样做有两个目的：第一，避免上肢自由度在训练早期成为策略逃避 locomotion 的通道；第二，让后续阶段能够复用同一个 19DoF 网络结构和同一套机器人模型，不再受 10DoF checkpoint 的结构限制。

Stage 1 的 reward 由两类项组成。第一类是基础行走项，包括 `tracking_lin_vel`、`tracking_lin_vel_x`、`tracking_ang_vel`、`tracking_heading`、`feet_air_time`、`penalty_low_forward_speed`、`penalty_backward_vel`、`penalty_heading_error`、`penalty_lateral_vel` 和 `penalty_slippage`。这些项共同要求机器人按命令前进、减少侧向漂移、保持朝向并避免脚底滑移。第二类是上肢约束项，包括 `upperbody_locked_dof_pos`、`upperbody_action_rate`、`upperbody_dof_acc`、`upperbody_torso_deviation`、`upperbody_arm_posture`、`upperbody_shoulder_pitch_limit` 和 `upperbody_torques`。这些项不追求摆臂，而是明确要求上肢不要破坏身体稳定。

这一阶段的关键不是让机器人“看起来像下肢模型”，而是在完整模型中保留下肢行走能力。换言之，stage1 是后续摆臂与抗扰的地基：如果这一阶段还不能直线稳定行走，后续任何摆臂 reward 都只会放大不稳定性。

## 4. Stage 2：借鉴 Gait-Conditioned Reward Routing 的自然摆臂设计

发布版 stage2 是本文最重要的设计阶段。它对应原始实验中的有效 stage7，而不是失败的原 stage2。该阶段从 stage1 继承稳定 locomotion，不再全局解锁所有上肢 DOF，而是主要释放左右 shoulder pitch，继续锁住 torso、shoulder roll/yaw 和 elbow。这样可以让手臂首先学习前后方向的自然摆动，而不是用横向甩臂、肘部乱动或躯干扭转来获取 reward。

Stage 2 的设计重点来自 `refer_paper/Gait-Conditioned Reinforcement Learning with Multi-Phase/Overview.tex` 中的三个思想。

第一，论文提出 gait-conditioned reward routing：不同 gait 下只激活对应 reward，减少 standing、walking、running 和 transition 之间的目标冲突。论文中使用 gait ID 与 gait mask，而我们的任务没有完整复现论文的多 gait observation 结构，因此采用轻量的 command-based mask。具体来说，`stand_mask` 在命令速度很小时激活站立稳定项，`walk_mask` 在行走命令下激活步态项，`straight_walk_mask` 在前向速度足够、侧向和 yaw 命令较小时激活摆臂项。这样，摆臂 reward 主要服务于直线前进行走，不会在站立或复杂转向时强迫手臂摆动。

第二，论文强调 multi-phase curriculum：先学习简单稳定行为，再引入更复杂协调目标。我们把这个思想迁移为“stage1 先保住稳定行走，stage2 再引入上肢协调”。早期失败实验试图在还没有充分隔离上下肢目标时直接强化摆臂，结果 reward 互相干扰；有效 stage2 则从 stage1 出发，保留稳定速度 tracking 和 heading 目标，只增加 gated walking 与 arm-swing 项。

第三，论文强调 reference-free 的 human-inspired reward，而不是依赖 MoCap 或固定轨迹。论文中的人体启发包括 straight-knee support、foot clearance、double support 和 arm-leg coordination。我们没有实现论文中的完整 centroidal angular momentum reward，也没有使用 MoCap imitation，而是用可从当前仿真状态稳定计算的近似项来表达同样的物理直觉：肩部与对侧髋部形成弱耦合，左右肩 pitch 速度反相，手臂末端在身体前后方向保持可见摆幅。

在代码上，stage2 的核心 reward 包括以下几类：

| 类别 | Reward 项 | 作用 |
| --- | --- | --- |
| 步态路由 | `stand_double_support`、`penalty_stand_base_motion` | 低速或站立时要求双脚支撑、身体少动 |
| 行走步态 | `feet_air_time_target`、`feet_single_stance_time`、`penalty_feet_air_time_short`、`penalty_feet_swing_height`、`penalty_walk_knee_flexion` | 让下肢维持可解释的行走节奏和摆脚高度 |
| 上肢安全 | `upperbody_action_rate`、`upperbody_shoulder_pitch_limit`、`upperbody_stationary_arm_posture` | 防止肩部高频抖动、过大摆动和低速乱动 |
| 摆臂协调 | `upperbody_arm_leg_phase`、`upperbody_arm_velocity_opposition`、`upperbody_arm_endpoint_sagittal_amplitude` | 只在直线行走时鼓励肩部与腿部反相、左右手臂反向、末端前后摆幅可见 |

其中 `upperbody_arm_leg_phase` 用对侧 hip pitch 作为 shoulder pitch 的相位参考，鼓励左臂随右腿、右臂随左腿产生协调摆动；`upperbody_arm_velocity_opposition` 惩罚左右肩 pitch 速度同向，避免双臂一起前后摆；`upperbody_arm_endpoint_sagittal_amplitude` 不追踪固定曲线，只要求左右 elbow link 在 base frame 的前后距离达到最低可见幅度。相比固定正弦 teacher，这种设计更接近论文的 reference-free 思路：reward 描述协调关系，而不是强迫关节追一条预设轨迹。

因此，stage2 的本质不是“加大摆臂权重”，而是“在不破坏直线行走的条件下，把摆臂目标路由到正确的步态上下文”。这也是它与原始失败 stage2-stage5 最大的区别。

## 5. Stage 3：上肢随机动作扰动与恢复

Stage 3 在 stage2 的摆臂策略上继续训练，但加入上肢随机动作扰动。扰动的实现位置在 `publish_code/humanoidverse/envs/locomotion/locomotion_upper_body.py`：环境先执行正常的 action delay 处理，然后只对上肢 DOF 的 `actions_after_delay` 加上随机 action offset，并进行 action clip。下肢 action 不被直接扰动，因此策略必须通过正常的全身控制与下肢补偿来恢复平衡。

扰动配置位于 `publish_code/humanoidverse/config/domain_rand/upper_body_random_action.yaml`。它选择 torso、左右 shoulder pitch/roll/yaw 和左右 elbow 作为扰动 DOF，其中 torso scale 为 0.35，肩肘为 1.0，避免扰动完全由 torso 主导。扰动模式使用 pulse：先等待 1.0-1.5 秒，再随机激活 0.35-0.70 秒，随后留出 1.20-2.20 秒恢复窗口。训练幅度为 `upper_body_random_action_amp=1.0`，评估默认幅度为 `upper_body_random_action_eval_amp=2.0`。这种 pulse 设计比连续噪声更适合观察“被打乱后恢复”的行为，因为它把冲击窗口和恢复窗口明确分开。

Stage 3 的 reward 继承 stage2 的摆臂目标，同时提高稳定直行和姿态约束的重要性：`tracking_lin_vel_x` 提高到 2.4，`tracking_heading` 提高到 1.2，并加强 `penalty_orientation`、`penalty_heading_error`、`penalty_lateral_vel` 和 `penalty_low_forward_speed`。这说明 stage3 的目标不是产生更大的摆臂，而是在上肢被随机扰动后仍保持速度、heading 和身体姿态。

## 6. 评估结果

有限时长评估使用 32 个并行环境、1000 step、20 秒 horizon，命令前向速度为 0.6 m/s。最终原始 stage8，也就是发布版 stage3，在上肢扰动幅度从 0 增加到 2.0 时都保持 100% 存活；而原始 stage7，也就是发布版 stage2，在同样 amp2 扰动下全部环境都发生失败。

| 策略 | 发布版对应 | 扰动幅度 | 存活率 | 平均存活时间 | x 速度误差 | 平均侧向速度 | 最大倾斜 | 总跌倒次数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 stage8 | stage3 | 0.0 | 100.0% | 20.00 s | 0.0228 m/s | 0.0508 m/s | 4.67 deg | 0 |
| 原 stage8 | stage3 | 1.0 | 100.0% | 20.00 s | 0.0225 m/s | 0.0542 m/s | 7.81 deg | 0 |
| 原 stage8 | stage3 | 2.0 | 100.0% | 20.00 s | 0.0245 m/s | 0.0564 m/s | 9.12 deg | 0 |
| 原 stage7 | stage2 | 2.0 | 0.0% | 5.48 s | 0.1401 m/s | 0.1335 m/s | 89.93 deg | 95 |

上肢指标也说明 stage3 的鲁棒性不是靠完全停止摆臂获得的。在 amp2 下，原 stage8 的 shoulder-pitch 平均摆幅为 0.2536 rad，elbow endpoint 前后位移为 0.1165 m，仍然存在可见摆臂；同时它的最大倾斜被限制在 10 度以内。相比之下，原 stage7 在 amp2 下虽然也有摆臂，但不能稳定承受扰动，说明“会摆臂”和“扰动后能恢复”是两个不同目标，需要额外的扰动训练阶段。

## 7. 失败设计反思：为什么原 stage2-stage5 没有学会摆臂

原始 `humanoidverse/` 中 stage2-stage5 是重要的失败探索。它们帮助我们排除了几类直觉上合理、实际效果不稳定的 reward 设计。

原 stage2 只释放 shoulder pitch，并加入 endpoint sagittal phase、arm-leg phase、arm velocity opposition 和横向 endpoint 约束。问题在于它仍然缺少清晰的 gait routing：摆臂项和姿态/横向限制混在一起，肩部虽然获得了自由度，但有效摆臂信号较弱，容易被平滑、限位和回中约束盖住。

原 stage3 进一步释放双臂，只锁 torso，并强化 lateral deviation、endpoint lateral position、endpoint balance、elbow posture 和 arm-leg phase。这个设计的目标更完整，但 reward 也更复杂：一边要求双臂前后摆，一边强烈限制横向位置、末端平衡、肘部姿态和站立回中。结果是策略收到的上肢梯度过于混杂，既不容易形成明显肩 pitch 摆臂，也可能通过僵硬上肢来避免惩罚。

原 stage4 尝试改用更直接的 shoulder pitch teacher 和 endpoint sagittal amplitude，降低 action-rate、dof-acc、torque 等平滑惩罚，关闭部分间接 phase reward。这个方向解决了“信号不直接”的问题，却引入了另一个问题：固定正弦 teacher 未必与实际足端接触相位和当前速度命令同步。它更像是在让肩部追一条外部曲线，而不是让手臂服务于当前 gait。

原 stage5 又把 teacher 幅度和 endpoint 最小摆幅调小，试图让策略更容易收敛。但幅度变小后，摆臂信号更容易被稳定项和平滑项淹没；同时它仍然没有解决 reward 全局生效、缺少步态上下文的问题。

这些失败说明，摆臂不是单个 reward 权重能解决的问题。有效的设计需要满足三个条件：第一，必须保留已有的稳定 locomotion；第二，摆臂 reward 只应在直线行走等合适上下文中生效；第三，摆臂目标应描述手臂与腿部、左右手臂之间的协调关系，而不是强制追踪与当前步态脱节的固定曲线。

## 8. 结论

本文将 H1 humanoid 从“只能动下肢稳定行走”扩展到“能够摆臂行走并抵抗上肢随机扰动”，关键并不是一次性设计一个复杂 reward，而是把学习目标拆成可继承的多阶段 curriculum。Stage 1 在完整 19DoF 模型中锁定上肢，建立稳定直线行走基础；Stage 2 借鉴 gait-conditioned reward routing，用 command-based masks 分时激活站立、行走和直线摆臂 reward，使肩部摆臂不再与站立和转向目标冲突；Stage 3 在动作层对上肢施加 pulse 扰动，并通过更强的速度、heading、姿态和侧向稳定约束训练恢复能力。

最终结果表明，发布版 stage3 在强上肢扰动下仍能保持 100% 存活和低速度 tracking error，同时保留可见上肢摆动。失败实验则进一步说明，直接解锁上肢、全局叠加 phase/teacher/amplitude reward，往往会导致 reward interference。对于没有 MoCap reference 的 humanoid 上肢摆臂任务，更稳妥的路线是：先让机器人走稳，再按步态上下文路由 reward，最后用有恢复窗口的随机上肢扰动训练鲁棒性。

## 参考资料

1. Tianhu Peng, Lingfan Bao, Chengxu Zhou. *Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion*. `refer_paper/Gait-Conditioned Reinforcement Learning with Multi-Phase/Overview.tex`.
2. `publish_code/README.md` 与 `publish_code/train_genesis_pro_staged.sh`：最终三阶段训练与评估入口。
3. `publish_code/humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage1.yaml`、`stage2.yaml`、`stage3.yaml`：发布版有效 reward 配置。
4. `publish_code/humanoidverse/config/domain_rand/upper_body_random_action.yaml`：上肢随机动作扰动配置。
5. `humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage2.yaml` 到 `stage5.yaml`：原始失败实验 reward 配置。
6. `output/eval_results_for_paper_agent.md`：有限时长扰动评估指标汇总。
