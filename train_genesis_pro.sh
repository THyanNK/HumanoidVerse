#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

# PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/cyh/.global_envs/humanoidverse/bin/python"
# PYTHON_BIN="/home/agilex/czt/HumanoidVerse/hgen/bin/python"
PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python"

HEADLESS="${HEADLESS:-True}"
if [[ "${1:-}" == "--visual" ]]; then
  HEADLESS=False
  export XLOCALEDIR=/usr/share/X11/locale
  shift
fi

exec "$PYTHON_BIN" humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp=locomotion_pro \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_h1_locomotion_upper_body \
  +robot=h1/h1 \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs="${NUM_ENVS:-4096}" \
  project_name="${PROJECT_NAME:-HumanoidLocomotion}" \
  experiment_name="${EXPERIMENT_NAME:-H1Pro_upperbody_loco_Genesis}" \
  headless="$HEADLESS" \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_initial_penalty_scale=0.5 \
  "$@"
