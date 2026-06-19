# 上肢自然摆臂 Reward 设计

目标：在 H1 全身 19DoF locomotion 中，让机器人保持稳定前进的同时，肩部产生与腿部步态协调的自然摆臂，而不是简单冻结上肢或让手臂无约束乱摆。

## 论文依据

1. He et al., 2025, *Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion*：强调按步态/阶段逐步引入约束，先稳定 locomotion，再塑造更复杂的协调动作。
   https://arxiv.org/abs/2505.20619

2. Peng et al., 2018, *DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills*：自然动作最好来自 reference motion imitation；如果后续能拿到人类或机器人参考步态，可以把手写相位 reward 替换或补充为 imitation reward。
   https://arxiv.org/abs/1804.02717

3. Peng et al., 2021, *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*：当任务 reward 难以完整描述“自然”时，可以用动作先验/判别器提供 style reward，特别适合上肢自然性。
   https://arxiv.org/abs/2104.02180

4. Humanoid-Gym, 2024, *Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer*：实际 humanoid locomotion 需要同时考虑速度跟踪、姿态、能耗、动作平滑、关节/力矩约束和 sim-to-real 鲁棒性。
   https://arxiv.org/abs/2404.05695

## 已实现的 reward 结构

当前实现仍沿用原 locomotion 奖励作为主目标：线速度/角速度/heading 跟踪、低前进速度惩罚、后退/侧向速度惩罚、脚部 air time、脚滑、脚朝向、终止、关节/力矩/动作变化惩罚。

新增上肢项位于 `humanoidverse/envs/locomotion/locomotion_upper_body.py`，全部按“误差项 + 负 scale”的方式实现：

- `upperbody_locked_dof_pos`：按 stage 锁定 torso 或部分手臂关节，保证学习过程不被上肢自由度拖垮。
- `upperbody_torso_deviation`：torso 尽量不扭动，避免靠腰部乱摆抵消腿部误差。
- `upperbody_arm_posture`：限制 shoulder roll/yaw 和 elbow 进入奇怪姿态。
- `upperbody_shoulder_pitch_limit`：允许肩 pitch 前后摆，但限制最大摆幅。
- `upperbody_elbow_posture`：stage3/单阶段中让 elbow 保持轻微自然弯曲。
- `upperbody_stationary_arm_posture`：命令速度很小时，上肢回到默认姿态，避免站立时手臂乱晃。
- `upperbody_arm_symmetry`：左右肩 pitch 近似反相，roll/yaw 按左右镜像符号保持平衡。
- `upperbody_arm_lateral_deviation`：压制 shoulder roll/yaw 的横向关节偏移。
- `upperbody_arm_endpoint_lateral_pos`：在 base frame 中约束左右 `elbow_link` 的横向 y 位置，直接限制手臂末端左右扫动。
- `upperbody_arm_endpoint_lateral_vel`：在 base frame 中惩罚左右 `elbow_link` 的横向 y 速度，抑制左右甩臂的动态振荡。
- `upperbody_arm_velocity_opposition`：左右肩 pitch 速度反相，抑制同向甩臂。
- `upperbody_arm_leg_phase`：肩 pitch 跟对侧髋 pitch 建立角度相位关系。
- `upperbody_arm_leg_velocity_phase`：肩 pitch 速度跟对侧髋 pitch 速度建立相位关系，让摆臂不是静态摆姿势。
- `upperbody_action_rate`、`upperbody_dof_acc`、`upperbody_torques`：约束动作平滑、关节加速度和上肢能耗。

## 推荐三阶段训练

### Stage 1：先学稳定走路，上肢冻结

入口：

```sh
bash train_genesis_pro_staged.sh --stage 1
```

配置：`humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage1.yaml`

策略：锁住 torso 和双臂全部自由度，只训练全身模型在 19DoF 配置下稳定 locomotion。这个阶段不追求摆臂。

### Stage 2：只放开肩 pitch

入口：

```sh
bash train_genesis_pro_staged.sh --stage 2 checkpoint=/path/to/stage1/model_x.pt algo.config.load_optimizer=False
```

配置：`humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage2.yaml`

策略：torso、shoulder roll/yaw、elbow 仍锁住，只允许左右 shoulder pitch 小幅运动。这里加入肩 pitch 幅度、站立回中、左右反向速度等约束，但还不强行要求臂腿相位。

### Stage 3：加入完整自然摆臂

入口：

```sh
bash train_genesis_pro_staged.sh --stage 3 checkpoint=/path/to/stage2/model_x.pt algo.config.load_optimizer=False
```

配置：`humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage3.yaml`

策略：只锁 torso，放开双臂。加入肩-髋角度相位、肩-髋速度相位、肘部轻微弯曲、肩 pitch 软限位、站立回中和上肢能耗约束。

## 单阶段训练

入口：

```sh
bash train_genesis_pro.sh
```

配置：`humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body.yaml`

单阶段配置已经包含完整上肢 reward，但如果训练不稳定，优先使用三阶段路线。

## 为什么会出现左右摆臂

H1 的 `shoulder_pitch_joint` 是前后摆动通道；`shoulder_roll_joint` 和 `shoulder_yaw_joint` 更容易造成横向摆动。之前 stage3 只锁 torso，roll/yaw 放开，同时 `arm_posture_deadband` 比较宽；另外左右 roll/yaw 的镜像对称项用了同号约束，和 H1 左右肩 roll/yaw 的镜像限位不匹配。现在已经改成 roll/yaw 反号镜像，并加入 `upperbody_arm_lateral_deviation` 抑制横向关节偏移。进一步加入末端 link 约束：`upperbody_arm_endpoint_lateral_pos` 和 `upperbody_arm_endpoint_lateral_vel` 会直接限制左右 `elbow_link` 在机器人 base frame 中的横向位置和速度，因此比单纯限制 roll/yaw 角度更能压住左右摆臂。

## 调参建议

- 手臂几乎不动：适当增大 `arm_leg_phase_gain` 到 0.45，或把 `upperbody_arm_leg_phase` 从 `-0.12` 调到 `-0.18`；同时确认 stage3 没有锁住肩关节。
- 手臂乱甩：减小 `arm_leg_phase_gain`，增大 `upperbody_shoulder_pitch_limit`、`upperbody_action_rate`、`upperbody_dof_acc` 或 `upperbody_stationary_arm_posture` 的惩罚强度。
- 走路变差：降低 `upperbody_arm_leg_phase` 和 `upperbody_arm_leg_velocity_phase`，先让 locomotion reward 恢复，再逐步加回摆臂项。
- 转向时手臂不自然：当前相位主要绑定前进步态；如果后续训练强转向，可增加 yaw command gate 或为转向单独设计 torso/arm 协调项。
- 想进一步自然：加入 mocap/reference motion，用 DeepMimic 式 pose/velocity/end-effector reward，或 AMP 风格判别器 reward，比手写相位规则更可靠。
