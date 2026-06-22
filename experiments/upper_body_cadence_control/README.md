# Upper-Body Cadence-Control Experiment

This experiment tests whether slower, cleaner lower-body cadence makes the
upper-body arm swing less twitchy.

The reward design follows three common locomotion shaping ideas:

- reward longer foot air time, capped at a target duration;
- reward stable single-stance intervals for biped walking;
- penalize short air-time contacts and insufficient swing-foot height.

Stage plan:

- Stage 1: locked upper body, slower command range, cadence-shaped gait.
- Stage 2: shoulder pitch unlocked, weak arm-leg shaping, cadence retained.
- Stage 3: full arms unlocked, slower arm-swing targets, cadence retained.

Run from the repository root:

```bash
bash experiments/upper_body_cadence_control/run_cadence_control_staged.sh
```

Useful overrides:

```bash
NUM_ENVS=4096 STAGE1_ITERS=5000 STAGE2_ITERS=2500 STAGE3_ITERS=2500 \
bash experiments/upper_body_cadence_control/run_cadence_control_staged.sh
```

Resume directly from an existing Stage 2 checkpoint:

```bash
START_STAGE=3 STAGE2_CHECKPOINT=/path/to/model_2500.pt \
bash experiments/upper_body_cadence_control/run_cadence_control_staged.sh
```
