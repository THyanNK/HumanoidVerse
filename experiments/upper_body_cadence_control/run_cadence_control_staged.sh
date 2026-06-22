#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HV_DIR="${HUMANOIDVERSE_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python}"
PROJECT_NAME="${PROJECT_NAME:-HumanoidLocomotion}"
NUM_ENVS="${NUM_ENVS:-4096}"
HEADLESS="${HEADLESS:-True}"
START_STAGE="${START_STAGE:-1}"
STAGE1_ITERS="${STAGE1_ITERS:-5000}"
STAGE2_ITERS="${STAGE2_ITERS:-2500}"
STAGE3_ITERS="${STAGE3_ITERS:-2500}"
REWARD_INITIAL_PENALTY_SCALE="${REWARD_INITIAL_PENALTY_SCALE:-0.5}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PASSTHROUGH_ARGS=()

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --visual)
        HEADLESS=False
        export XLOCALEDIR="${XLOCALEDIR:-/usr/share/X11/locale}"
        shift
        ;;
      --start-stage)
        START_STAGE="$2"
        shift 2
        ;;
      --start-stage=*)
        START_STAGE="${1#--start-stage=}"
        shift
        ;;
      *)
        PASSTHROUGH_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

latest_checkpoint() {
  local run_dir="$1"
  local best_ckpt=""
  local best_iter=-1

  shopt -s nullglob
  local ckpts=("${run_dir}"/model_*.pt)
  shopt -u nullglob

  for ckpt in "${ckpts[@]}"; do
    local name="${ckpt##*/}"
    local iter="${name#model_}"
    iter="${iter%.pt}"
    if [[ "$iter" =~ ^[0-9]+$ ]] && (( iter > best_iter )); then
      best_iter="$iter"
      best_ckpt="$ckpt"
    fi
  done

  if [[ -z "$best_ckpt" ]]; then
    echo "No model_*.pt checkpoint found in '${run_dir}'." >&2
    return 1
  fi

  printf '%s\n' "$best_ckpt"
}

run_stage() {
  local stage="$1"
  local exp_config="$2"
  local reward_config="$3"
  local experiment_dir="$4"
  local experiment_name="$5"
  local iterations="$6"
  local max_forward_speed="$7"
  local eval_forward_speed="$8"
  local checkpoint="${9:-}"

  echo ""
  echo "==== ${experiment_name}: stage ${stage} ===="
  echo "experiment_dir=${experiment_dir}"
  echo "reward_config=${reward_config}"
  echo "lin_vel_x=[0.15,${max_forward_speed}]"
  mkdir -p "$(dirname "$experiment_dir")"

  local checkpoint_args=()
  if [[ -n "$checkpoint" ]]; then
    echo "checkpoint=${checkpoint}"
    checkpoint_args=(
      "checkpoint=${checkpoint}"
      "algo.config.load_optimizer=False"
    )
  fi

  "$PYTHON_BIN" humanoidverse/train_agent.py \
    +simulator=genesis \
    +exp="$exp_config" \
    +domain_rand=NO_domain_rand \
    +rewards="$reward_config" \
    +robot=h1/h1 \
    +terrain=terrain_locomotion_plane \
    +obs=loco/leggedloco_obs_singlestep_withlinvel \
    num_envs="$NUM_ENVS" \
    project_name="$PROJECT_NAME" \
    experiment_name="$experiment_name" \
    experiment_dir="$experiment_dir" \
    headless="$HEADLESS" \
    algo.config.num_learning_iterations="$iterations" \
    algo.config.eval_command="[${eval_forward_speed},0.0,0.0]" \
    env.config.locomotion_command_ranges.lin_vel_x="[0.15,${max_forward_speed}]" \
    rewards.reward_penalty_curriculum=True \
    rewards.reward_initial_penalty_scale="$REWARD_INITIAL_PENALTY_SCALE" \
    "${checkpoint_args[@]}" \
    "${PASSTHROUGH_ARGS[@]}"
}

parse_args "$@"
cd "$HV_DIR"

case "$START_STAGE" in
  1|2|3) ;;
  *)
    echo "START_STAGE must be 1, 2, or 3; got '${START_STAGE}'." >&2
    exit 1
    ;;
esac

RUN_ROOT="${RUN_ROOT:-logs/${RUN_TIMESTAMP}_ub_cadence_control}"
STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}_stage1}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}_stage2}"
STAGE3_DIR="${STAGE3_DIR:-${RUN_ROOT}_stage3}"

if (( START_STAGE <= 1 )); then
  run_stage \
    1 \
    "locomotion_pro_stage1" \
    "loco/reward_h1_locomotion_upper_body_cadence_stage1" \
    "$STAGE1_DIR" \
    "ub_cadence_stage1_locked" \
    "$STAGE1_ITERS" \
    "0.55" \
    "0.40"
else
  echo "Skipping stage 1 because START_STAGE=${START_STAGE}."
fi

if (( START_STAGE <= 2 )); then
  STAGE1_CKPT="${STAGE1_CHECKPOINT:-$(latest_checkpoint "$STAGE1_DIR")}"
  run_stage \
    2 \
    "locomotion_pro_stage2" \
    "loco/reward_h1_locomotion_upper_body_cadence_stage2" \
    "$STAGE2_DIR" \
    "ub_cadence_stage2_shoulder" \
    "$STAGE2_ITERS" \
    "0.65" \
    "0.50" \
    "$STAGE1_CKPT"
else
  echo "Skipping stage 2 because START_STAGE=${START_STAGE}."
fi

STAGE2_CKPT="${STAGE2_CHECKPOINT:-$(latest_checkpoint "$STAGE2_DIR")}"
run_stage \
  3 \
  "locomotion_pro_stage3" \
  "loco/reward_h1_locomotion_upper_body_cadence_stage3" \
  "$STAGE3_DIR" \
  "ub_cadence_stage3_slow_arm" \
  "$STAGE3_ITERS" \
  "0.70" \
  "0.50" \
  "$STAGE2_CKPT"

echo ""
echo "Finished upper-body cadence-control experiment."
echo "stage1_dir=${STAGE1_DIR}"
echo "stage2_dir=${STAGE2_DIR}"
echo "stage3_dir=${STAGE3_DIR}"
