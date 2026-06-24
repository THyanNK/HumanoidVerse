# 从 Gait-Conditioned RL 论文中学习到的 Reward 设计思路

本文档总结 `refer/Gait-Conditioned Reinforcement Learning with Multi-Phase.pdf` 中对本项目有价值的设计思想，并说明这些思想如何迁移到当前 HumanoidVerse H1 上肢自然摆臂任务和 `stage6` reward 设计中。文档目标是为后续课程小论文和 poster 海报提供可直接复用的材料。

## 1. 论文核心问题

论文关注的问题是：如何让一个 humanoid policy 在不依赖 MoCap 参考动作的情况下，同时学习 standing、walking、running 以及 gait transition。

传统做法常见两类：

1. 使用 MoCap / AMP / imitation reward，让策略模仿已有参考动作。
2. 手写一组通用 locomotion reward，让策略自己探索稳定步态。

论文指出，多 gait 或多行为学习中最容易失败的地方不是单个 reward 不够强，而是不同 gait 的目标互相冲突。例如：

- standing 需要静止、双脚支撑、base motion 小；
- walking 需要前进速度、周期性脚步、摆腿高度；
- running 需要更短接触时间、push-off 和更强动态性。

如果这些 reward 在所有时刻同时启用，policy 会收到矛盾梯度，导致训练不稳定。论文的核心贡献就是用 gait-conditioned reward routing 解决这种 reward interference。

## 2. 最值得学习的思想：Reward Routing

论文的 reward routing 可以概括为：

```text
total reward = shared task rewards + active gait-specific rewards
```

其中 gait-specific reward 由 gait mask 控制。当前 gait 是 walk，就只启用 walking 相关项；当前 gait 是 stand，就只启用 standing 相关项；transition 也有自己的 reward 子集。

这点比单纯调 reward 权重更重要。论文消融实验显示：

| 设置 | 平均 episode length | return |
| --- | ---: | ---: |
| Curriculum + Routing | 972.6 | 89.3 |
| No Curriculum | 31.26 | 2.45 |
| No Reward Routing | 11.59 | 0.00 |

这说明：在多目标 locomotion 中，reward 是否被正确“分时激活”，比堆更多 reward 项更关键。

## 3. 第二个思想：先稳定基础步态，再引入复杂协调

论文使用 multi-phase curriculum，而不是一开始就训练所有 gait。它的训练顺序大致是：

1. Phase 1：只训练 walking，建立稳定周期步态。
2. Phase 2：加入 standing 和 walk-to-stand transition。
3. Phase 3：加入 running 和 run-to-walk transition。

迁移到我们的 H1 任务时，这个思想对应为：

1. 先保留能正常走路的 `stage1`。
2. 不继承失败的 `stage2-stage5` reward 设计。
3. 在 `stage6` 中从 `stage1` 重新出发，只加入轻量 gated reward。
4. 先保证 locomotion 不被破坏，再逐步增强手臂自然摆动。

这也是当前 `stage6` 的设计原则。

## 4. 第三个思想：Human-Inspired Reward 不等于固定轨迹模仿

论文强调 reference-free，即不要求策略追踪 MoCap 轨迹，而是把人类步态中的生物力学规律写成 reward：

- straight-knee support：支撑腿不要过度蹲屈；
- arm-leg coordination：手臂与腿部反相摆动；
- foot clearance：摆动脚要有足够离地高度；
- double support for standing：站立时保持双脚稳定接触；
- smooth transition：走到停、跑到走不要突变。

这对我们很重要，因为当前项目并没有可直接使用的人类或 H1 MoCap 参考动作。相比固定 sinusoidal teacher，论文给出的启发是：

```text
优先奖励“协调关系”和“状态条件”，而不是强迫关节追某条固定曲线。
```

例如，对手臂摆动而言，可以先奖励：

- 左右 shoulder pitch 反相；
- 手臂摆动只在直线 walking 时启用；
- shoulder pitch 与对侧 hip pitch 形成弱耦合；
- standing 或低速时关闭摆臂目标，让手臂回中。

这比直接让 shoulder pitch 追固定频率 sine wave 更不容易破坏走路。

## 5. 对当前 H1 Stage6 的迁移设计

当前 `stage6` 已按上述思路实现为一个从 `stage1` 重新出发的 reward routing 版本。

相关文件：

- `humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage6.yaml`
- `humanoidverse/config/exp/locomotion_pro_stage6.yaml`
- `humanoidverse/envs/locomotion/locomotion.py`
- `train_genesis_pro_staged.sh`

### 5.1 继承关系

`stage6` 的 reward 配置只继承：

```yaml
defaults:
  - /rewards/loco/reward_h1_locomotion_upper_body_stage1
  - _self_
```

这意味着它不使用 stage2、stage3、stage4、stage5 中那些失败的上肢 reward 设计。它保留 stage1 能走路的基础 reward，包括：

- `tracking_lin_vel`
- `tracking_lin_vel_x`
- `tracking_heading`
- `penalty_heading_error`
- `penalty_lateral_vel`
- `penalty_low_forward_speed`
- `penalty_backward_vel`

### 5.2 Stage6 的 gait masks

`stage6` 中增加了以下 mask：

| mask | 条件 | 作用 |
| --- | --- | --- |
| `stand_mask` | command speed <= 0.10 | 启用站立稳定 reward |
| `walk_mask` | command speed > 0.20 | 启用行走 reward |
| `straight_walk_mask` | forward command > 0.22 且 lateral/yaw command 小 | 只在直线行走时启用摆臂和直膝 reward |

这种设计是论文 gait mask 的简化版。论文使用 gait ID；我们当前没有改 observation 结构，所以用 command-based mask 实现轻量 routing。

### 5.3 Stage6 的 reward 分组

#### Shared locomotion reward

这些 reward 从 stage1 继承，始终服务于稳定直线行走：

- 速度跟踪；
- heading 跟踪；
- 横向速度惩罚；
- 后退惩罚；
- foot orientation / slippage / base height / action smoothness。

#### Walking-specific reward

只在 walking 或 straight walking 时启用：

- `feet_air_time_target`
- `feet_single_stance_time`
- `penalty_feet_air_time_short`
- `penalty_feet_swing_height`
- `penalty_walk_knee_flexion`
- `upperbody_arm_leg_phase`
- `upperbody_arm_velocity_opposition`

这些项对应论文中的 walking contact pattern、foot clearance、straight-knee support 和 arm-leg coordination。

#### Standing-specific reward

只在低速或站立命令时启用：

- `stand_double_support`
- `penalty_stand_base_motion`
- `upperbody_stationary_arm_posture`

这些项对应论文中的 standing double support、base stability 和 stillness。

## 6. 当前日志能说明什么

从目前 `stage6` 训练截图看，训练没有明显异常：

- episode length 保持在 1001，说明没有早摔；
- termination reward 为 0，说明稳定性良好；
- tracking lin vel 和 tracking heading 都很高，说明继承 stage1 的行走能力没有被破坏；
- `gait_stand_mask` 约 0.24-0.25，`gait_walk_mask` 约 0.75，说明 routing 确实在分时生效；
- `gait_straight_walk_mask` 约 0.72-0.73，说明大多数 walking 样本属于直线行走；
- penalty scale 从约 0.59 增长到约 0.86 后，策略仍未崩，说明 reward 设计比较稳。

需要注意的是，上肢摆臂项目前仍然较弱：

- `upperbody_arm_leg_phase` 的 reward magnitude 很小；
- `upperbody_arm_velocity_opposition` 也较小；
- 因此从 log 看，当前 `stage6` 更像是“稳定直线走路 + 轻量摆臂引导”，还不能证明已经学出明显自然摆臂。

这可以作为小论文中的中期实验观察：reward routing 先保证 locomotion 稳定，再逐步增强 arm coordination。

## 7. 小论文可以怎么写

### 题目建议

可以考虑以下题目：

1. 基于 Gait-Conditioned Reward Routing 的 Humanoid 上肢自然摆臂 Reward 设计
2. 面向 H1 Humanoid 直线行走的分阶段上肢摆臂强化学习奖励设计
3. 从稳定行走到自然摆臂：一种命令条件化的 Humanoid Reward Routing 方法

### 摘要思路

摘要可以按以下逻辑组织：

1. 背景：humanoid locomotion 不仅要走稳，还需要自然的全身协调。
2. 问题：直接叠加 locomotion reward 和 arm-swing reward 容易互相干扰，导致走路退化或手臂不自然。
3. 方法：借鉴 gait-conditioned reward routing，基于 command speed 构造 stand/walk/straight-walk masks，从稳定 stage1 继承并分时启用 walking 与 standing reward。
4. 实验：在 H1 19DoF 模型上训练 stage6，观察 episode length、heading tracking、straight-walk mask、termination 等指标。
5. 结论：routing 保持了 stage1 的稳定行走能力，并为后续增强 arm-leg coordination 提供了更稳的 reward 框架。

### 方法章节结构

建议写成四小节：

#### 1. Baseline: Stable Lower-Body Locomotion

说明 stage1 是当前能稳定直线行走的基础，核心 reward 是 velocity tracking、heading tracking、lateral velocity penalty 和 upper-body locking。

#### 2. Command-Based Gait Routing

定义：

```text
m_stand = 1[||v_cmd|| <= v_stand]
m_walk = 1[||v_cmd|| > v_walk]
m_straight = 1[v_x_cmd > v_min and |v_y_cmd| < eps_y and |w_z_cmd| < eps_yaw]
```

然后说明不同 reward 乘以不同 mask。

#### 3. Human-Inspired Reward Terms

把 stage6 reward 分为：

- walking foot clearance；
- walking single stance；
- straight-knee support；
- arm-leg anti-phase coordination；
- standing double support；
- standing base stillness。

#### 4. Training Observations

展示训练 log 中的指标：

- episode length；
- mean reward；
- tracking lin vel；
- tracking heading；
- gait mask ratio；
- termination；
- upper-body coordination reward。

## 8. Poster 可以怎么排版

建议 poster 做三栏：

### 左栏：Problem

标题：Why arm-swing reward is hard?

内容：

- 稳定行走和自然摆臂目标不同；
- 站立、行走、转向下 reward 目标冲突；
- 直接叠加 reward 会破坏 locomotion；
- 前几个 stage 的失败说明“只加权重”不够。

配图建议：

- H1 机器人图；
- 一张 stage1 直线行走截图；
- 一张 reward conflict 示意图。

### 中栏：Method

标题：Command-Based Reward Routing

核心图：

```text
command velocity
       |
       v
stand / walk / straight-walk mask
       |
       v
activate corresponding reward terms
```

表格：

| Mode | Active reward |
| --- | --- |
| Stand | double support, base stillness, stationary arms |
| Walk | velocity tracking, foot clearance, single stance |
| Straight walk | arm-leg phase, shoulder opposition, straight knee |

### 右栏：Results and Takeaways

展示当前 stage6 log 指标：

- episode length = 1001；
- termination = 0；
- straight walk mask 约 0.72-0.73；
- tracking heading 约 0.998；
- penalty scale 增大后仍稳定。

结论 bullet：

- routing 保留了 stage1 的稳定行走能力；
- gait masks 正常工作；
- arm-swing reward 仍偏弱，后续需要逐步增强；
- 相比固定 teacher，routing 更适合作为稳定自然摆臂的底座。

## 9. 不要过度宣称的内容

写小论文和 poster 时需要注意边界：

1. 当前 stage6 不是完整复现论文。
   论文使用 gait ID、LSTM、完整多 gait routing；我们当前是 command-based mask 的轻量实现。

2. 当前 stage6 还没有实现完整 angular momentum reward。
   论文使用 centroidal angular momentum；当前 Genesis 适配层没有统一暴露 link mass/inertia，因此我们只实现了 arm-leg phase 和 velocity opposition 的近似协调。

3. 当前日志只能证明稳定性和 routing 生效，不能单独证明自然摆臂已经成功。
   是否自然还需要 eval 视频或定量上肢摆幅指标。

4. 不应说 stage6 已经优于所有 stage。
   更准确的说法是：stage6 在不继承失败 stage 的情况下，提供了一个更稳的 reward routing 框架。

## 10. 后续可以补的实验

为了让小论文和 poster 更完整，建议后续补以下实验：

1. Stage1 vs Stage6 对比
   - episode length；
   - tracking lin vel；
   - tracking heading；
   - lateral velocity；
   - upper-body shoulder pitch amplitude。

2. Stage6 ablation
   - 去掉 `stand/walk mask`；
   - 去掉 `upperbody_arm_leg_phase`；
   - 去掉 `penalty_walk_knee_flexion`；
   - 对比是否影响稳定性和手臂摆动。

3. Eval video qualitative comparison
   - stage1：上肢冻结但走稳；
   - stage6：肩部轻量摆动，仍保持走稳；
   - 如果后续增强 reward，再展示更明显的 arm-leg coordination。

4. Arm swing quantitative metrics
   - left/right shoulder pitch amplitude；
   - shoulder pitch opposition error；
   - shoulder pitch 与对侧 hip pitch 的相关性；
   - torso yaw / base angular velocity 是否降低。

## 11. 可直接放进小论文的贡献表述

可以写成：

本文借鉴 gait-conditioned reward routing 思想，针对 H1 humanoid 上肢自然摆臂任务设计了一种 command-based reward routing 方法。该方法不直接引入复杂 MoCap imitation，而是从稳定行走 baseline 出发，根据速度命令构造 stand、walk 和 straight-walk masks，分时激活 standing stability、walking foot clearance、straight-knee support 和 arm-leg coordination rewards。初步训练日志表明，该设计在 penalty curriculum 增强过程中仍保持满长 episode 和较高 heading tracking，说明 routing 机制能够在不破坏基础 locomotion 的前提下，为后续自然摆臂学习提供稳定基础。

## 12. 可直接放进 Poster 的一句话结论

```text
Instead of adding stronger arm-swing rewards globally, we route rewards by gait context:
stand still when standing, walk straight when walking, and coordinate arms only during straight walking.
```

中文版本：

```text
我们不是把摆臂 reward 全局加大，而是按步态上下文分时启用 reward：
站立时稳住身体，行走时保持直线，只有直线行走时才引导手臂与腿部协调摆动。
```

