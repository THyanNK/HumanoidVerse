#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

if [[ ! -f humanoidverse/eval_agent.py ]]; then
  echo "HumanoidVerse repo not found: $HV_DIR" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <epoch> [extra eval args...]"
  echo "Example: $0 1000"
  exit 1
fi

epoch="$1"
shift

# PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/cyh/.global_envs/humanoidverse/bin/python"
PYTHON_BIN="/home/agilex/czt/HumanoidVerse/hgen/bin/python"
# PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python"

export XLOCALEDIR=/usr/share/X11/locale

exec "$PYTHON_BIN" humanoidverse/eval_agent.py \
  +checkpoint="logs/HumanoidLocomotion/h1_upper_body/model_${epoch}.pt" \
  eval_name=H1Pro_upperbody_loco_Genesis \
  "$@"