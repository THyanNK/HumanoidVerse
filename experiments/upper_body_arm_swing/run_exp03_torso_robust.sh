#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

parse_common_args "$@"

run_upper_body_experiment \
  "ub_exp03_torso_robust" \
  "loco/reward_h1_locomotion_upper_body_stage3_exp03_torso_robust" \
  "NO_domain_rand" \
  domain_rand.push_robots=True \
  domain_rand.max_push_vel_xy=0.35 \
  domain_rand.randomize_friction=True \
  "domain_rand.friction_range=[0.7,1.1]" \
  domain_rand.randomize_pd_gain=True \
  "domain_rand.kp_range=[0.9,1.1]" \
  "domain_rand.kd_range=[0.9,1.1]" \
  domain_rand.randomize_ctrl_delay=True \
  "domain_rand.ctrl_delay_step_range=[0,1]"
