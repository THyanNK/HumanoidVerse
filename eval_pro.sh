#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

if [[ ! -f humanoidverse/eval_agent.py ]]; then
  echo "HumanoidVerse repo not found: $HV_DIR" >&2
  exit 1
fi

# PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/cyh/.global_envs/humanoidverse/bin/python"
# PYTHON_BIN="/home/agilex/czt/HumanoidVerse/hgen/bin/python"
PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python"

CHECKPOINT="${CHECKPOINT:-${1:-}}"
if [[ $# -gt 0 && "$1" == "$CHECKPOINT" ]]; then
  shift
fi
if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: bash eval_pro.sh <checkpoint> [hydra_overrides...]" >&2
  exit 1
fi

exec "$PYTHON_BIN" humanoidverse/eval_agent.py \
  +checkpoint="$CHECKPOINT" \
  eval_name=H1Pro_upperbody_loco_Genesis \
  "$@"
