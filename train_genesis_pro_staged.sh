#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

PYTHON_BIN="${PYTHON_BIN:-/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python}"
STAGE="${STAGE:-1}"
HEADLESS="${HEADLESS:-True}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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

case "$STAGE" in
  1|stage1|locked)
    STAGE_ID="stage1"
    EXP_CONFIG="locomotion_pro_stage1"
    REWARD_CONFIG="loco/reward_h1_locomotion_upper_body_stage1"
    DEFAULT_EXPERIMENT_NAME="H1Pro_stage1_locked_upper_Genesis"
    ;;
  2|stage2|shoulder)
    STAGE_ID="stage2"
    EXP_CONFIG="locomotion_pro_stage2"
    REWARD_CONFIG="loco/reward_h1_locomotion_upper_body_stage2"
    DEFAULT_EXPERIMENT_NAME="H1Pro_stage2_shoulder_pitch_Genesis"
    ;;
  3|stage3|armswing)
    STAGE_ID="stage3"
    EXP_CONFIG="locomotion_pro_stage3"
    REWARD_CONFIG="loco/reward_h1_locomotion_upper_body_stage3"
    DEFAULT_EXPERIMENT_NAME="H1Pro_stage3_arm_swing_Genesis"
    ;;
  4|stage4|amplitude|armamp)
    STAGE_ID="stage4"
    EXP_CONFIG="locomotion_pro_stage4"
    REWARD_CONFIG="loco/reward_h1_locomotion_upper_body_stage4"
    DEFAULT_EXPERIMENT_NAME="H1Pro_stage4_arm_amplitude_Genesis"
    ;;
  5|stage5|smallshoulder|smallarm)
    STAGE_ID="stage5"
    EXP_CONFIG="locomotion_pro_stage5"
    REWARD_CONFIG="loco/reward_h1_locomotion_upper_body_stage5"
    DEFAULT_EXPERIMENT_NAME="H1Pro_stage5_small_shoulder_swing_Genesis"
    ;;
  *)
    echo "Unknown STAGE '$STAGE'. Use 1, 2, 3, 4, or 5." >&2
    exit 1
    ;;
esac

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_EXPERIMENT_DIR="logs/${RUN_TIMESTAMP}-pro-${STAGE_ID}"

exec "$PYTHON_BIN" humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp="$EXP_CONFIG" \
  +domain_rand=NO_domain_rand \
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
