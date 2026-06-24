# HumanoidVerse H1 staged locomotion

这份发布代码只保留三个有效阶段。原来的 `stage1`、`stage7`、`stage8` 已整理为现在的 `stage1`、`stage2`、`stage3`：

- `stage1`: 只训练下肢行走，上肢动作和奖励都尽量锁在默认姿态。
- `stage2`: 在 stage1 基础上加入上肢自然摆臂，主要释放双肩 pitch。
- `stage3`: 在 stage2 基础上加入上肢随机动作扰动，训练扰动后的稳定行走和恢复。

## 训练脚本

先进入发布目录：

```bash
cd publish_code
```

脚本默认使用当前环境里的 `python`。如果需要指定环境：

```bash
export PYTHON_BIN=/path/to/python
```

一键按顺序训练三个阶段：

```bash
bash train_genesis_pro_all_stages.sh
```

常用参数：

```bash
bash train_genesis_pro_all_stages.sh algo.config.num_learning_iterations=1000
NUM_ENVS=2048 bash train_genesis_pro_all_stages.sh
bash train_genesis_pro_all_stages.sh --visual
```

从中间阶段继续训练：

```bash
START_STAGE=2 STAGE1_CHECKPOINT=logs/xxx_pro_stage1/model_1000.pt bash train_genesis_pro_all_stages.sh
START_STAGE=3 STAGE2_CHECKPOINT=logs/xxx_pro_stage2/model_1000.pt bash train_genesis_pro_all_stages.sh
```

单独训练某个阶段：

```bash
bash train_genesis_pro_staged.sh --stage 1
bash train_genesis_pro_staged.sh --stage 2 checkpoint=logs/xxx_pro_stage1/model_1000.pt algo.config.load_optimizer=False
bash train_genesis_pro_staged.sh --stage 3 checkpoint=logs/xxx_pro_stage2/model_1000.pt algo.config.load_optimizer=False
```

## 测试和评估脚本

打开 viewer 观察策略：

```bash
bash eval_staged.sh logs/xxx_pro_stage3/model_1000.pt
```

评估 stage3 的上肢扰动恢复效果时，可以指定扰动幅度：

```bash
bash eval_staged.sh logs/xxx_pro_stage3/model_1000.pt --upper-rand-amp 2.0
```

导出有限步长的指标统计：

```bash
MAX_STEPS=1000 bash eval_metrics_staged.sh logs/xxx_pro_stage3/model_1000.pt
MAX_STEPS=1000 bash eval_metrics_staged.sh logs/xxx_pro_stage3/model_1000.pt --upper-rand-amp 2.0
```

`eval_metrics_staged.sh` 会启用 `EvalMetricsCallback`，输出 `eval_metrics_summary.json` 和 `eval_metrics_timeseries.csv` 到对应的 `logs_eval/` 目录。

## Reward 设计

### Stage 1: 下肢行走

配置文件：

- `humanoidverse/config/exp/locomotion_pro_stage1.yaml`
- `humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage1.yaml`

作用：保留 H1 全身模型，但用 `upperbody_locked_dof_pos`、`upperbody_action_rate`、`upperbody_dof_acc`、`upperbody_torso_deviation`、`upperbody_arm_posture` 等惩罚项限制上肢，让策略先学会稳定的下肢前进、转向和抗滑移。

### Stage 2: 上肢摆臂

配置文件：

- `humanoidverse/config/exp/locomotion_pro_stage2.yaml`
- `humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage2.yaml`

作用：继承 stage1，释放左右肩 pitch，保留躯干、肩 roll/yaw 和肘部约束。核心 reward 包括：

- `upperbody_arm_leg_phase`: 鼓励手臂和腿部形成自然反相摆动。
- `upperbody_arm_velocity_opposition`: 鼓励左右手臂速度方向相反。
- `upperbody_arm_endpoint_sagittal_amplitude`: 鼓励手臂在前后方向有可见摆幅。
- `stand_double_support`、`penalty_feet_air_time_short`、`penalty_feet_swing_height`: 用步态路由区分站立和直线行走，避免摆臂破坏基本步态。

### Stage 3: 上肢随机扰动

配置文件：

- `humanoidverse/config/exp/locomotion_pro_stage3.yaml`
- `humanoidverse/config/rewards/loco/reward_h1_locomotion_upper_body_stage3.yaml`
- `humanoidverse/config/domain_rand/upper_body_random_action.yaml`

作用：继承 stage2 的摆臂目标，并在训练时对上肢 DOF 注入随机动作脉冲。下肢动作不被直接扰动。reward 额外提高 `tracking_lin_vel_x`、`tracking_heading`，并加强 `penalty_orientation`、`penalty_heading_error`、`penalty_lateral_vel`、`penalty_low_forward_speed`，让策略在上肢被扰动后仍能保持直行和平衡。
