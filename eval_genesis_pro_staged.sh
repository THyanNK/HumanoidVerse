#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint.pt> [extra eval args...]" >&2
  echo "Example: $0 logs/HumanoidLocomotion/run/model_10000.pt" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python}"
CHECKPOINT="$1"
shift

export XLOCALEDIR="${XLOCALEDIR:-/usr/share/X11/locale}"

exec "$PYTHON_BIN" humanoidverse/eval_agent.py \
  +checkpoint="$CHECKPOINT" \
  eval_name="${EVAL_NAME:-H1Pro_staged_eval_Genesis}" \
  "++algo.config.eval_command=${EVAL_COMMAND:-[0.6,0.0,0.0]}" \
  "$@"