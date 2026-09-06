#!/usr/bin/env bash
set -euo pipefail

# Run the three correction ablations with the same 200-sequence split.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NUM_SEQUENCES="${NUM_SEQUENCES:-200}"
export SETTING="${SETTING:-normal}"

for mode in direct hyp_replan ours; do
  MODE="${mode}" bash "${SCRIPT_DIR}/run_calvin_joint_belief_ablation.sh"
done
