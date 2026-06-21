#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

PYTHON_BIN="${PYTHON_BIN:-/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python}"
HEADLESS="${HEADLESS:-True}"
TEACHER_ITERS="${TEACHER_ITERS:-3000}"
EXPERIMENT_DIR_VALUE="${EXPERIMENT_DIR:-logs/\${timestamp}-arm-swing-teacher}"
EXPERIMENT_NAME_VALUE="${EXPERIMENT_NAME:-H1ArmSwingTeacher_Genesis}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --visual)
      HEADLESS=False
      export XLOCALEDIR="${XLOCALEDIR:-/usr/share/X11/locale}"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

INIT_ARGS=()
if [[ -n "${TEACHER_INIT_CHECKPOINT:-}" ]]; then
  INIT_ARGS+=(checkpoint="$TEACHER_INIT_CHECKPOINT" algo.config.load_optimizer=False)
fi

exec "$PYTHON_BIN" humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp=arm_swing_teacher \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_h1_arm_swing_teacher \
  +robot=h1/h1 \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs="${NUM_ENVS:-4096}" \
  project_name="${PROJECT_NAME:-HumanoidLocomotion}" \
  experiment_name="$EXPERIMENT_NAME_VALUE" \
  experiment_dir="$EXPERIMENT_DIR_VALUE" \
  headless="$HEADLESS" \
  algo.config.num_learning_iterations="$TEACHER_ITERS" \
  "${INIT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"