#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${STAGE:-1}"
HEADLESS="${HEADLESS:-True}"
DOMAIN_RAND_CONFIG="${DOMAIN_RAND_CONFIG:-}"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash train_genesis_pro_staged.sh --stage 1 [HYDRA_OVERRIDES...]
  bash train_genesis_pro_staged.sh --stage 2 checkpoint=logs/.../model_*.pt algo.config.load_optimizer=False
  bash train_genesis_pro_staged.sh --stage 3 checkpoint=logs/.../model_*.pt algo.config.load_optimizer=False

Stages:
  1  lower-body walking with upper body locked
  2  stage 1 + natural shoulder-pitch arm swing
  3  stage 2 + random upper-body action disturbance

Environment overrides:
  PYTHON_BIN              Python executable. Default: python
  NUM_ENVS                Number of parallel envs. Default: 4096
  RUN_TIMESTAMP           Log timestamp. Default: current time
  EXPERIMENT_DIR          Override output directory
  EXPERIMENT_NAME         Override experiment name
  DOMAIN_RAND_CONFIG      Override Hydra domain_rand config
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --stage=*)
      STAGE="${1#--stage=}"
      shift
      ;;
    --visual)
      HEADLESS=False
      export XLOCALEDIR=/usr/share/X11/locale
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

DEFAULT_DOMAIN_RAND_CONFIG="NO_domain_rand"

case "$STAGE" in
  1|stage1|lower|lowerbody|locked)
    STAGE_ID="stage1"
    EXP_CONFIG="locomotion_pro_stage1"
    REWARD_CONFIG="loco/reward_h1_locomotion_upper_body_stage1"
    DEFAULT_EXPERIMENT_NAME="H1Pro_stage1_lower_body_Genesis"
    ;;
  2|stage2|armswing|arm_swing)
    STAGE_ID="stage2"
    EXP_CONFIG="locomotion_pro_stage2"
    REWARD_CONFIG="loco/reward_h1_locomotion_upper_body_stage2"
    DEFAULT_EXPERIMENT_NAME="H1Pro_stage2_arm_swing_Genesis"
    ;;
  3|stage3|randomupper|upperrand|disturbance)
    STAGE_ID="stage3"
    EXP_CONFIG="locomotion_pro_stage3"
    REWARD_CONFIG="loco/reward_h1_locomotion_upper_body_stage3"
    DEFAULT_EXPERIMENT_NAME="H1Pro_stage3_random_upper_actions_Genesis"
    DEFAULT_DOMAIN_RAND_CONFIG="upper_body_random_action"
    ;;
  *)
    echo "Unknown STAGE '$STAGE'. Use 1, 2, or 3." >&2
    exit 1
    ;;
esac

DOMAIN_RAND_CONFIG="${DOMAIN_RAND_CONFIG:-$DEFAULT_DOMAIN_RAND_CONFIG}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_EXPERIMENT_DIR="logs/${RUN_TIMESTAMP}_pro_${STAGE_ID}"

exec "$PYTHON_BIN" humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp="$EXP_CONFIG" \
  +domain_rand="$DOMAIN_RAND_CONFIG" \
  +rewards="$REWARD_CONFIG" \
  +robot=h1/h1 \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs="${NUM_ENVS:-4096}" \
  project_name="${PROJECT_NAME:-HumanoidLocomotion}" \
  experiment_name="${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}" \
  experiment_dir="${EXPERIMENT_DIR:-$DEFAULT_EXPERIMENT_DIR}" \
  headless="$HEADLESS" \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_initial_penalty_scale="${REWARD_INITIAL_PENALTY_SCALE:-0.5}" \
  "${EXTRA_ARGS[@]}"
