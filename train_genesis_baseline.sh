#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"

if [[ ! -f "./humanoidverse/train_agent.py" ]]; then
  echo "HumanoidVerse repo not found: $HV_DIR" >&2
  echo "Set HUMANOIDVERSE_REPO if the repository moved." >&2
  exit 1
fi

# PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/cyh/.global_envs/humanoidverse/bin/python"
PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python"

PROJECT_NAME=HumanoidLocomotion
EXPERIMENT_NAME=H110dof_loco_Genesis
NUM_ENVS=4096
HEADLESS=True

CMD=(
  "$PYTHON_BIN" humanoidverse/train_agent.py
  +simulator=genesis
  +exp=locomotion
  +domain_rand=NO_domain_rand
  +rewards=loco/reward_h1_locomotion
  +robot=h1/h1_10dof
  +terrain=terrain_locomotion_plane
  +obs=loco/leggedloco_obs_singlestep_withlinvel
  num_envs="$NUM_ENVS"
  project_name="$PROJECT_NAME"
  experiment_name="$EXPERIMENT_NAME"
  headless="$HEADLESS"
  rewards.reward_penalty_curriculum=True
  rewards.reward_initial_penalty_scale=0.5
  "$@"
)

exec "${CMD[@]}"
