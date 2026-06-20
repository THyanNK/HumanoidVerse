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

parse_common_args() {
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

run_existing_stage() {
  local stage="$1"
  local experiment_dir="$2"
  local experiment_name="$3"
  local iterations="$4"
  shift 4

  echo ""
  echo "==== ${experiment_name}: stage ${stage} ===="
  echo "experiment_dir=${experiment_dir}"
  mkdir -p "$(dirname "$experiment_dir")"

  EXPERIMENT_DIR="$experiment_dir" \
  EXPERIMENT_NAME="$experiment_name" \
  PYTHON_BIN="$PYTHON_BIN" \
  NUM_ENVS="$NUM_ENVS" \
  HEADLESS="$HEADLESS" \
  REWARD_INITIAL_PENALTY_SCALE="$REWARD_INITIAL_PENALTY_SCALE" \
  bash train_genesis_pro_staged.sh \
    --stage "$stage" \
    algo.config.num_learning_iterations="$iterations" \
    "${PASSTHROUGH_ARGS[@]}" \
    "$@"
}

run_custom_stage3() {
  local experiment_dir="$1"
  local experiment_name="$2"
  local reward_config="$3"
  local domain_rand_config="$4"
  local checkpoint="$5"
  shift 5

  echo ""
  echo "==== ${experiment_name}: custom stage 3 ===="
  echo "experiment_dir=${experiment_dir}"
  echo "reward_config=${reward_config}"
  echo "domain_rand=${domain_rand_config}"
  echo "checkpoint=${checkpoint}"
  mkdir -p "$(dirname "$experiment_dir")"

  "$PYTHON_BIN" humanoidverse/train_agent.py \
    +simulator=genesis \
    +exp=locomotion_pro_stage3 \
    +domain_rand="$domain_rand_config" \
    +rewards="$reward_config" \
    +robot=h1/h1 \
    +terrain=terrain_locomotion_plane \
    +obs=loco/leggedloco_obs_singlestep_withlinvel \
    num_envs="$NUM_ENVS" \
    project_name="$PROJECT_NAME" \
    experiment_name="$experiment_name" \
    experiment_dir="$experiment_dir" \
    headless="$HEADLESS" \
    checkpoint="$checkpoint" \
    algo.config.load_optimizer=False \
    algo.config.num_learning_iterations="$STAGE3_ITERS" \
    rewards.reward_penalty_curriculum=True \
    rewards.reward_initial_penalty_scale="$REWARD_INITIAL_PENALTY_SCALE" \
    "$@" \
    "${PASSTHROUGH_ARGS[@]}"
}

run_upper_body_experiment() {
  local exp_id="$1"
  local reward_config="$2"
  local domain_rand_config="$3"
  shift 3
  local stage3_extra_args=("$@")

  cd "$HV_DIR"

  case "$START_STAGE" in
    1|2|3) ;;
    *)
      echo "START_STAGE must be 1, 2, or 3; got '${START_STAGE}'." >&2
      exit 1
      ;;
  esac

  local run_root="${RUN_ROOT:-logs/${RUN_TIMESTAMP}_${exp_id}}"
  local stage1_dir="${STAGE1_DIR:-${run_root}_stage1}"
  local stage2_dir="${STAGE2_DIR:-${run_root}_stage2}"
  local stage3_dir="${STAGE3_DIR:-${run_root}_stage3}"

  if (( START_STAGE <= 1 )); then
    run_existing_stage 1 "$stage1_dir" "${exp_id}_stage1_locked" "$STAGE1_ITERS"
  else
    echo "Skipping stage 1 because START_STAGE=${START_STAGE}."
  fi

  local stage1_ckpt=""
  if (( START_STAGE <= 2 )); then
    stage1_ckpt="${STAGE1_CHECKPOINT:-$(latest_checkpoint "$stage1_dir")}"
    echo "Stage 1 checkpoint for stage 2: ${stage1_ckpt}"
    run_existing_stage 2 "$stage2_dir" "${exp_id}_stage2_shoulder" "$STAGE2_ITERS" \
      checkpoint="$stage1_ckpt" \
      algo.config.load_optimizer=False
  else
    echo "Skipping stage 2 because START_STAGE=${START_STAGE}."
  fi

  local stage2_ckpt="${STAGE2_CHECKPOINT:-$(latest_checkpoint "$stage2_dir")}"
  echo "Stage 2 checkpoint for custom stage 3: ${stage2_ckpt}"
  run_custom_stage3 "$stage3_dir" "${exp_id}_stage3" "$reward_config" "$domain_rand_config" "$stage2_ckpt" \
    "${stage3_extra_args[@]}"

  echo ""
  echo "Finished ${exp_id}."
  echo "stage1_dir=${stage1_dir}"
  echo "stage2_dir=${stage2_dir}"
  echo "stage3_dir=${stage3_dir}"
}
