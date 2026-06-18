#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

if [[ ! -f "humanoidverse/eval_agent.py" ]]; then
  echo "HumanoidVerse repo not found: $HV_DIR" >&2
  echo "Set HUMANOIDVERSE_REPO if the repository moved." >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$HV_DIR/hgen/bin/python" ]]; then
    PYTHON_BIN="$HV_DIR/hgen/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
fi

if [[ "$PYTHON_BIN" == "~/"* ]]; then
  PYTHON_BIN="$HOME/${PYTHON_BIN#~/}"
elif [[ "$PYTHON_BIN" != /* && "$PYTHON_BIN" == */* ]]; then
  PYTHON_BIN="$HV_DIR/$PYTHON_BIN"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN:-<empty>}" >&2
  echo "Set PYTHON_BIN to an absolute interpreter path, for example:" >&2
  echo "  PYTHON_BIN=\"\$HOME/czt/HumanoidVerse/hgen/bin/python\" bash eval_pro.sh logs/HumanoidLocomotion/<run>/model_10000.pt" >&2
  exit 1
fi

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "$CHECKPOINT" && $# -gt 0 && "$1" != +* && "$1" != --* ]]; then
  CHECKPOINT="$1"
  shift
fi

if [[ -z "$CHECKPOINT" ]]; then
  echo "Missing checkpoint for pro evaluation." >&2
  echo "Usage: bash eval_pro.sh logs/HumanoidLocomotion/<H1Pro_run>/model_10000.pt [hydra_overrides...]" >&2
  echo "Or set CHECKPOINT=/absolute/or/repo-relative/model.pt." >&2
  exit 1
fi

export XLOCALEDIR=/usr/share/X11/locale

CMD=(
  "$PYTHON_BIN" humanoidverse/eval_agent.py 
  +checkpoint="$CHECKPOINT"
  eval_name=H1Pro_upperbody_loco_Genesis
  "$@"
)

exec "${CMD[@]}"
