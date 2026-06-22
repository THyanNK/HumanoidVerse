#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint.pt> [extra eval args...]" >&2
  echo "Example: $0 logs/HumanoidLocomotion/run/model_10000.pt" >&2
  echo "Example: $0 logs/.../model_10000.pt --upper-rand-amp 1.2" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/home/agilex/czt/HumanoidVerse/hgen/bin/python}"
CHECKPOINT="$1"
shift

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
      # base_eval.yaml does not declare domain_rand, so eval-time overrides need ++.
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
  eval_name="${EVAL_NAME:-H1Pro_staged_eval_Genesis}" \
  ++export_onnx=False \
  ++export_policy=False \
  "++algo.config.eval_command=${EVAL_COMMAND:-[0.6,0.0,0.0]}" \
  "${EXTRA_ARGS[@]}"
