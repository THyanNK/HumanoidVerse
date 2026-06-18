#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"

if [[ ! -f "./humanoidverse/eval_agent.py" ]]; then
  echo "HumanoidVerse repo not found: $HV_DIR" >&2
  echo "Set HUMANOIDVERSE_REPO if the repository moved." >&2
  exit 1
fi

# PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/cyh/.global_envs/humanoidverse/bin/python"
# PYTHON_BIN="/home/agilex/czt/HumanoidVerse/hgen/bin/python"
PYTHON_BIN="/inspire/qb-ilm/project/robot-reasoning/public/zhetao/HumanoidVerse/hgen/bin/python"
export XLOCALEDIR=/usr/share/X11/locale

CMD=(
  "$PYTHON_BIN" humanoidverse/eval_agent.py 
  +checkpoint=logs/HumanoidLocomotion/20260617_160240-H110dof_loco_Genesis-locomotion-h1_10dof/model_10000.pt
  # +headless=True
  "$@"
)

exec "${CMD[@]}"
