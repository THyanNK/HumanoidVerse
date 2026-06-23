#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint.pt> [extra eval args...]" >&2
  echo "Example: $0 logs/.../model_3200.pt --upper-rand-amp 2.0" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="$1"
shift

MAX_STEPS="${MAX_STEPS:-1000}"
EVAL_COMMAND="${EVAL_COMMAND:-[0.6,0.0,0.0]}"
EVAL_NAME="${EVAL_NAME:-H1Pro_staged_eval_metrics_Genesis}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --upper-rand-amp)
      EXTRA_ARGS+=("++domain_rand.upper_body_random_action_eval_amp=$2")
      shift 2
      ;;
    --upper-rand-amp=*)
      EXTRA_ARGS+=("++domain_rand.upper_body_random_action_eval_amp=${1#--upper-rand-amp=}")
      shift
      ;;
    domain_rand.*)
      EXTRA_ARGS+=("++$1")
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

export XLOCALEDIR="${XLOCALEDIR:-/usr/share/X11/locale}"

exec "$PYTHON_BIN" humanoidverse/eval_agent.py \
  +checkpoint="$CHECKPOINT" \
  eval_name="$EVAL_NAME" \
  ++export_onnx=False \
  ++export_policy=False \
  "++algo.config.eval_command=${EVAL_COMMAND}" \
  ++algo.config.eval_callbacks.eval_metrics._target_=humanoidverse.agents.callbacks.eval_metrics.EvalMetricsCallback \
  "++algo.config.eval_callbacks.eval_metrics.max_steps=${MAX_STEPS}" \
  ++algo.config.eval_callbacks.eval_metrics.write_timeseries=True \
  "${EXTRA_ARGS[@]}"
