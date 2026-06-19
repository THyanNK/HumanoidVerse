#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

if [[ ! -f humanoidverse/eval_agent.py ]]; then
  echo "HumanoidVerse repo not found: $HV_DIR" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint.pt|epoch> [extra eval args...]" >&2
  echo "Example: $0 logs/20260619_123456-pro-stage1/model_10000.pt" >&2
  echo "Example: CHECKPOINT_DIR=logs/20260619_123456-pro-stage1 $0 10000" >&2
  exit 1
fi

target="$1"
shift

# PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/cyh/.global_envs/humanoidverse/bin/python"
PYTHON_BIN="${PYTHON_BIN:-/home/agilex/czt/HumanoidVerse/hgen/bin/python}"
# PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python"

if [[ "$target" == *.pt || -f "$target" ]]; then
  CHECKPOINT="$target"
else
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-logs/HumanoidLocomotion/h1_upper_body}"
  CHECKPOINT="$CHECKPOINT_DIR/model_${target}.pt"
fi

export XLOCALEDIR="${XLOCALEDIR:-/usr/share/X11/locale}"

exec "$PYTHON_BIN" humanoidverse/eval_agent.py \
  +checkpoint="$CHECKPOINT" \
  eval_name="${EVAL_NAME:-H1Pro_upperbody_loco_Genesis}" \
  "++algo.config.eval_command=${EVAL_COMMAND:-[0.6,0.0,0.0]}" \
  "$@"
