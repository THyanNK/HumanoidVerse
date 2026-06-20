# Upper-Body Arm-Swing Experiments

This folder contains one-click launch scripts for three incremental Stage3 upper-body
reward experiments. Each script runs the existing Stage1 and Stage2 curriculum first,
then trains a custom Stage3 reward configuration.

## What changed

The reward implementation adds three optional mechanisms:

- Straight-walk arm-swing gating: strong phase rewards are applied only when the
  command is mostly forward walking. Turning and lateral commands still use posture,
  lateral endpoint, and smoothness rewards.
- Speed-scaled swing targets: shoulder and endpoint phase targets can grow with the
  commanded forward speed instead of using a fixed gain.
- Elbow swing coupling: elbow flexion can weakly follow shoulder swing magnitude,
  so the arm is less like a rigid shoulder pendulum.

Existing Stage1/Stage2/Stage3 configs keep their previous behavior unless the new
keys are enabled in one of these experiment configs.

## Experiments

1. `run_exp01_straight_gated.sh`
   - Keeps the current torso-locked Stage3 structure.
   - Adds straight-walk gating and slightly stronger forward arm-leg phase shaping.
   - Goal: reduce unnatural arm behavior during turning or lateral commands.

2. `run_exp02_speed_elbow.sh`
   - Builds on Exp01.
   - Adds speed-scaled swing gains and dynamic elbow coupling.
   - Goal: get larger swing at faster commands while keeping slow walking restrained.

3. `run_exp03_torso_robust.sh`
   - Builds on Exp02.
   - Lightly unlocks torso in Stage3 and adds mild push/friction/PD/delay randomization.
   - Goal: test whether a small amount of torso freedom improves robustness and naturalness.
   - This is the riskiest run; use Exp01/Exp02 as the safer baselines.

## Usage

Run from the repository root:

```sh
bash experiments/upper_body_arm_swing/run_exp01_straight_gated.sh
bash experiments/upper_body_arm_swing/run_exp02_speed_elbow.sh
bash experiments/upper_body_arm_swing/run_exp03_torso_robust.sh
```

Useful environment overrides:

```sh
NUM_ENVS=4096 \
STAGE1_ITERS=5000 \
STAGE2_ITERS=2500 \
STAGE3_ITERS=2500 \
bash experiments/upper_body_arm_swing/run_exp02_speed_elbow.sh
```

Resume from an existing Stage2 checkpoint:

```sh
START_STAGE=3 \
STAGE2_CHECKPOINT=/path/to/model_2500.pt \
bash experiments/upper_body_arm_swing/run_exp02_speed_elbow.sh
```

Pass extra Hydra overrides after the script name; they are appended after the script
defaults and can override iteration counts, logging names, or other config values.
