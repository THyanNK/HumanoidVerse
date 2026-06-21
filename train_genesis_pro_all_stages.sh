#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

PYTHON_BIN="${PYTHON_BIN:-/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python}"
HEADLESS="${HEADLESS:-True}"
START_STAGE="${START_STAGE:-1}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-}"
COMMON_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash train_genesis_pro_all_stages.sh [--visual] [--timestamp YYYYMMDD_HHMMSS] [--start-stage 1|2|3|4] [HYDRA_OVERRIDES...]

Examples:
  bash train_genesis_pro_all_stages.sh algo.config.num_learning_iterations=1000
  START_STAGE=2 bash train_genesis_pro_all_stages.sh algo.config.num_learning_iterations=1000
  STAGE3_CHECKPOINT=logs/20260622_pro_stage3/model_1600.pt bash train_genesis_pro_all_stages.sh --start-stage 4
  bash train_genesis_pro_all_stages.sh --timestamp 20260620_120000

Environment overrides:
  PYTHON_BIN              Python executable used by HumanoidVerse.
  NUM_ENVS                Forwarded through train_genesis_pro_staged.sh.
  RUN_TIMESTAMP           Shared timestamp for logs/<timestamp>_pro_stage{1,2,3,4}.
  START_STAGE             1 runs all; 2 uses latest stage1 ckpt; 3 uses latest stage2 ckpt; 4 uses latest stage3 ckpt.
  STAGE1_CHECKPOINT       Override checkpoint used to start stage2.
  STAGE2_CHECKPOINT       Override checkpoint used to start stage3.
  STAGE3_CHECKPOINT       Override checkpoint used to start stage4.
  STAGE1_DIR/STAGE2_DIR/STAGE3_DIR/STAGE4_DIR
                          Override experiment directories.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --timestamp)
      RUN_TIMESTAMP="$2"
      shift 2
      ;;
    --timestamp=*)
      RUN_TIMESTAMP="${1#--timestamp=}"
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
    --visual)
      HEADLESS=False
      export XLOCALEDIR=/usr/share/X11/locale
      shift
      ;;
    *)
      COMMON_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$START_STAGE" in
  1|2|3|4) ;;
  *)
    echo "START_STAGE must be 1, 2, 3, or 4; got '$START_STAGE'." >&2
    exit 1
    ;;
esac

if [[ -z "$RUN_TIMESTAMP" ]]; then
  RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
fi

STAGE1_DIR="${STAGE1_DIR:-logs/${RUN_TIMESTAMP}_pro_stage1}"
STAGE2_DIR="${STAGE2_DIR:-logs/${RUN_TIMESTAMP}_pro_stage2}"
STAGE3_DIR="${STAGE3_DIR:-logs/${RUN_TIMESTAMP}_pro_stage3}"
STAGE4_DIR="${STAGE4_DIR:-logs/${RUN_TIMESTAMP}_pro_stage4}"

find_latest_stage_dir() {
  local stage_id="$1"
  local latest=""
  shopt -s nullglob
  local candidates=(logs/*"${stage_id}"*)
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    return 1
  fi
  latest="$(printf '%s\n' "${candidates[@]}" | sort | tail -n 1)"
  [[ -n "$latest" ]] || return 1
  printf '%s\n' "$latest"
}

resolve_stage_dir() {
  local expected_dir="$1"
  local stage_id="$2"

  if [[ -d "$expected_dir" ]]; then
    printf '%s\n' "$expected_dir"
    return 0
  fi

  find_latest_stage_dir "$stage_id"
}

latest_checkpoint() {
  local run_dir="$1"
  local best_ckpt=""
  local best_iter=-1

  shopt -s nullglob
  local ckpts=("$run_dir"/model_*.pt)
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
    echo "No model_*.pt checkpoint found in '$run_dir'." >&2
    return 1
  fi

  printf '%s\n' "$best_ckpt"
}

run_stage() {
  local stage="$1"
  local experiment_dir="$2"
  shift 2

  mkdir -p "$(dirname "$experiment_dir")"
  echo ""
  echo "==== Running stage ${stage} ===="
  echo "experiment_dir=${experiment_dir}"

  EXPERIMENT_DIR="$experiment_dir" \
  PYTHON_BIN="$PYTHON_BIN" \
  HEADLESS="$HEADLESS" \
  bash train_genesis_pro_staged.sh \
    --stage "$stage" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

stage1_run_dir=""
if (( START_STAGE <= 1 )); then
  run_stage 1 "$STAGE1_DIR"
  stage1_run_dir="$STAGE1_DIR"
else
  echo "Skipping stage 1 because START_STAGE=${START_STAGE}."
fi

stage2_run_dir=""
if (( START_STAGE <= 2 )); then
  if [[ -n "$stage1_run_dir" ]]; then
    stage1_ckpt="${STAGE1_CHECKPOINT:-$(latest_checkpoint "$stage1_run_dir")}"
  else
    stage1_run_dir="$(resolve_stage_dir "$STAGE1_DIR" stage1)"
    stage1_ckpt="${STAGE1_CHECKPOINT:-$(latest_checkpoint "$stage1_run_dir")}"
  fi
  echo "Stage 1 checkpoint for stage 2: ${stage1_ckpt}"
  run_stage 2 "$STAGE2_DIR" \
    checkpoint="$stage1_ckpt" \
    algo.config.load_optimizer=False
  stage2_run_dir="$STAGE2_DIR"
else
  echo "Skipping stage 2 because START_STAGE=${START_STAGE}."
fi

stage3_run_dir=""
if (( START_STAGE <= 3 )); then
  if [[ -n "${STAGE2_CHECKPOINT:-}" ]]; then
    stage2_ckpt="$STAGE2_CHECKPOINT"
  else
    if [[ -z "$stage2_run_dir" ]]; then
      stage2_run_dir="$(resolve_stage_dir "$STAGE2_DIR" stage2)"
    fi
    stage2_ckpt="$(latest_checkpoint "$stage2_run_dir")"
  fi
  echo "Stage 2 checkpoint for stage 3: ${stage2_ckpt}"
  run_stage 3 "$STAGE3_DIR" \
    checkpoint="$stage2_ckpt" \
    algo.config.load_optimizer=False
  stage3_run_dir="$STAGE3_DIR"
else
  echo "Skipping stage 3 because START_STAGE=${START_STAGE}."
fi

if [[ -n "${STAGE3_CHECKPOINT:-}" ]]; then
  stage3_ckpt="$STAGE3_CHECKPOINT"
else
  if [[ -z "$stage3_run_dir" ]]; then
    stage3_run_dir="$(resolve_stage_dir "$STAGE3_DIR" stage3)"
  fi
  stage3_ckpt="$(latest_checkpoint "$stage3_run_dir")"
fi
echo "Stage 3 checkpoint for stage 4: ${stage3_ckpt}"

run_stage 4 "$STAGE4_DIR" \
  checkpoint="$stage3_ckpt" \
  algo.config.load_optimizer=False

echo ""
echo "All stages finished."
echo "stage1_dir=$(resolve_stage_dir "$STAGE1_DIR" stage1 2>/dev/null || true)"
echo "stage2_dir=$(resolve_stage_dir "$STAGE2_DIR" stage2 2>/dev/null || true)"
echo "stage3_dir=$(resolve_stage_dir "$STAGE3_DIR" stage3 2>/dev/null || true)"
echo "stage4_dir=$(resolve_stage_dir "$STAGE4_DIR" stage4)"
