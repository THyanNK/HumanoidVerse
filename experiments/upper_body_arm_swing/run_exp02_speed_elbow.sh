#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

parse_common_args "$@"

run_upper_body_experiment \
  "ub_exp02_speed_elbow" \
  "loco/reward_h1_locomotion_upper_body_stage3_exp02_speed_elbow" \
  "NO_domain_rand"
