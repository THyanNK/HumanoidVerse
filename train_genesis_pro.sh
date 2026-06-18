#!/usr/bin/env bash
set -euo pipefail

HV_DIR="${HUMANOIDVERSE_REPO:-$(pwd)}"
cd "$HV_DIR"

if [[ ! -f "humanoidverse/train_agent.py" ]]; then
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
  echo "  PYTHON_BIN=\"\$HOME/czt/HumanoidVerse/hgen/bin/python\" bash train_genesis_pro.sh" >&2
  exit 1
fi

PROJECT_NAME=HumanoidLocomotion
EXPERIMENT_NAME=H1Pro_upperbody_loco_Genesis
NUM_ENVS="${NUM_ENVS:-4096}"
HEADLESS=True
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --visual)
      HEADLESS=False
      export XLOCALEDIR=/usr/share/X11/locale
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

CMD=(
  "$PYTHON_BIN" humanoidverse/train_agent.py
  +simulator=genesis
  +exp=locomotion_pro
  +domain_rand=NO_domain_rand
  +rewards=loco/reward_h1_locomotion_upper_body
  +robot=h1/h1
  +terrain=terrain_locomotion_plane
  +obs=loco/leggedloco_obs_singlestep_withlinvel
  num_envs="$NUM_ENVS"
  project_name="$PROJECT_NAME"
  experiment_name="$EXPERIMENT_NAME"
  headless="$HEADLESS"
  rewards.reward_penalty_curriculum=True
  rewards.reward_initial_penalty_scale=0.5
  "${EXTRA_ARGS[@]}"
)

exec "${CMD[@]}"
