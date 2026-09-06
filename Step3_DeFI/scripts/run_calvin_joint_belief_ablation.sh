#!/usr/bin/env bash
set -euo pipefail

# Controlled CALVIN ablations for joint-belief suffix correction.
#
# Shared controls across all modes:
#   - same Step3 DeFI checkpoint
#   - same eval sequence split
#   - same CALVIN env path
#   - no Qwen reflection
#   - no retry; each mode is evaluated as a single first-pass rollout
#
# Modes:
#   base        : DeFI backbone only; no suffix correction.
#   direct      : direct delta-action suffix correction baseline.
#   hyp_replan  : expected/posterior hypothesis model, keep/replan only.
#   ours        : expected/posterior hypothesis + innovation-conditioned suffix patch.
#   all         : run all four modes sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python}"
MODE="${MODE:-ours}"
SETTING="${SETTING:-normal}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
EP_LEN="${EP_LEN:-360}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:-/mnt/workspace/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm}"
ACTION_MODEL_FOLDER="${ACTION_MODEL_FOLDER:-/mnt/workspace/manipulation/DeFi/outputs/collect_step3_action_chunk_phi_1000_sanitize/train_folder}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ACTION_MODEL_FOLDER}/saved_models/step3_defi.pt}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/mnt/workspace/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32}"
T5_MODEL_PATH="${T5_MODEL_PATH:-/mnt/workspace/manipulation/DeFi/ckpts/t5_base}"
LANGUAGE_GOAL_PATH="${LANGUAGE_GOAL_PATH:-/mnt/workspace/manipulation/DeFi/ckpts/ViT-B-32.pt}"
CALVIN_ABC_DIR="${CALVIN_ABC_DIR:-/mnt/workspace/calvin/task_ABC_D}"
EVAL_SEQUENCES_PATH="${EVAL_SEQUENCES_PATH:-/mnt/workspace/manipulation/DeFi/outputs/splits/outcome_router_clean_1k_train1000.json}"

DELTA_ACTION_REPAIR_CKPT="${DELTA_ACTION_REPAIR_CKPT:-/mnt/workspace/manipulation/DeFi/outputs/delta_action_repair_v5_balanced/train_run/delta_action_repair_mlp.pt}"
CHUNK_EFFECT_CKPT="${CHUNK_EFFECT_CKPT:-/mnt/workspace/manipulation/DeFi/outputs/phi_chunk_effect_v4/train_run_s1024_balanced_seq934/chunk_effect_phi_mlp.pt}"
JOINT_BELIEF_TRANSITION_CKPT="${JOINT_BELIEF_TRANSITION_CKPT:-/mnt/workspace/manipulation/DeFi/outputs/factored_belief_action_expected_transition_v1/train_run/factored_belief_action_transition.pt}"
JOINT_PAIR_ACTION_GENERATOR_CKPT="${JOINT_PAIR_ACTION_GENERATOR_CKPT:-/mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt}"

PATCH_MIN_PROB="${PATCH_MIN_PROB:-0.55}"
PATCH_MARGIN="${PATCH_MARGIN:-0.10}"
REPLAN_MIN_PROB="${REPLAN_MIN_PROB:-0.85}"
PATCH_MIX="${PATCH_MIX:-0.02}"

# Direct suffix baseline uses the older chunk-effect trigger path. Keep it
# conservative by default; otherwise the historical defaults trigger suffix on
# almost every step and destroy the control baseline.
DIRECT_SUFFIX_SUCCESS_BELOW="${DIRECT_SUFFIX_SUCCESS_BELOW:-0.35}"
DIRECT_SUFFIX_RESIDUAL_ABOVE="${DIRECT_SUFFIX_RESIDUAL_ABOVE:-1000000000}"
DIRECT_SUFFIX_COMPAT_BELOW="${DIRECT_SUFFIX_COMPAT_BELOW:--1.0}"
DIRECT_CHUNK_KEEP_THRESHOLD="${DIRECT_CHUNK_KEEP_THRESHOLD:-0.35}"
DIRECT_CHUNK_REPLAN_THRESHOLD="${DIRECT_CHUNK_REPLAN_THRESHOLD:-0.75}"
DIRECT_CHUNK_SUCCESS_THRESHOLD="${DIRECT_CHUNK_SUCCESS_THRESHOLD:-0.5}"

if [[ "${SETTING}" != "normal" ]]; then
  echo "SETTING=${SETTING} is not implemented in this runner yet. Use SETTING=normal." >&2
  exit 2
fi

unset EGL_VISIBLE_DEVICES
unset EGL_VISIBLE_DEVICE
export EGL_VISIBLE_DEVICE="${EGL_VISIBLE_DEVICE:-0}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/tmp/nvidia470199/10_nvidia_470199.json}"
export LD_LIBRARY_PATH="/tmp/nvidia470199:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/tmp/nvidia_470199:/usr/local/nvidia/lib:/root/CoppeliaSim:${LD_LIBRARY_PATH:-}"
export DEFI_T5_SANITIZE="${DEFI_T5_SANITIZE:-1}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${STEP3_DIR}:${PYTHONPATH:-}"

COMMON_ARGS=(
  --video_model_path "${VIDEO_MODEL_PATH}"
  --action_model_folder "${ACTION_MODEL_FOLDER}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --clip_model_path "${CLIP_MODEL_PATH}"
  --t5_model_path "${T5_MODEL_PATH}"
  --language_goal_path "${LANGUAGE_GOAL_PATH}"
  --calvin_abc_dir "${CALVIN_ABC_DIR}"
  --eval_sequences_path "${EVAL_SEQUENCES_PATH}"
  --num_sequences "${NUM_SEQUENCES}"
  --ep_len "${EP_LEN}"
  --no_qwen_reflection
  --disable_retry_after_first_pass
)

run_mode() {
  local mode="$1"
  echo "[ABLATION] mode=${mode} setting=${SETTING} num_sequences=${NUM_SEQUENCES}" >&2
  cd "${STEP3_DIR}"
  case "${mode}" in
    base)
      exec_or_run "${PYTHON_BIN}" policy_evaluation/calvin_evaluate_with_memory_reflection.py "${COMMON_ARGS[@]}"
      ;;
    direct)
      exec_or_run "${PYTHON_BIN}" policy_evaluation/calvin_evaluate_with_memory_reflection.py \
        "${COMMON_ARGS[@]}" \
        --dynamic_coupling_first_pass \
        --dynamic_coupling_online \
        --dynamic_coupling_generate_action_intent \
        --dynamic_coupling_persistent_hypothesis \
        --dynamic_coupling_disable_replan \
        --dynamic_coupling_chunk_effect_ckpt "${CHUNK_EFFECT_CKPT}" \
        --dynamic_coupling_delta_action_repair_ckpt "${DELTA_ACTION_REPAIR_CKPT}" \
        --dynamic_coupling_delta_action_repair_mix "${PATCH_MIX}" \
        --dynamic_coupling_suffix_trigger_success_below "${DIRECT_SUFFIX_SUCCESS_BELOW}" \
        --dynamic_coupling_suffix_trigger_residual_above "${DIRECT_SUFFIX_RESIDUAL_ABOVE}" \
        --dynamic_coupling_suffix_trigger_compat_below "${DIRECT_SUFFIX_COMPAT_BELOW}" \
        --dynamic_coupling_chunk_keep_threshold "${DIRECT_CHUNK_KEEP_THRESHOLD}" \
        --dynamic_coupling_chunk_replan_threshold "${DIRECT_CHUNK_REPLAN_THRESHOLD}" \
        --dynamic_coupling_chunk_success_threshold "${DIRECT_CHUNK_SUCCESS_THRESHOLD}"
      ;;
    hyp_replan)
      exec_or_run "${PYTHON_BIN}" policy_evaluation/calvin_evaluate_with_memory_reflection.py \
        "${COMMON_ARGS[@]}" \
        --dynamic_coupling_first_pass \
        --dynamic_coupling_online \
        --dynamic_coupling_generate_action_intent \
        --dynamic_coupling_persistent_hypothesis \
        --joint_belief_transition \
        --joint_belief_clean_controller \
        --joint_belief_transition_ckpt "${JOINT_BELIEF_TRANSITION_CKPT}" \
        --joint_pair_action_generator_ckpt "${JOINT_PAIR_ACTION_GENERATOR_CKPT}" \
        --joint_belief_clean_min_confidence "${REPLAN_MIN_PROB}" \
        --joint_belief_clean_patch_min_prob 2.0 \
        --joint_belief_clean_patch_margin 2.0 \
        --joint_belief_clean_replan_min_prob "${REPLAN_MIN_PROB}"
      ;;
    ours)
      exec_or_run "${PYTHON_BIN}" policy_evaluation/calvin_evaluate_with_memory_reflection.py \
        "${COMMON_ARGS[@]}" \
        --dynamic_coupling_first_pass \
        --dynamic_coupling_online \
        --dynamic_coupling_generate_action_intent \
        --dynamic_coupling_persistent_hypothesis \
        --joint_belief_transition \
        --joint_belief_clean_controller \
        --dynamic_coupling_disable_replan \
        --joint_belief_transition_ckpt "${JOINT_BELIEF_TRANSITION_CKPT}" \
        --joint_pair_action_generator_ckpt "${JOINT_PAIR_ACTION_GENERATOR_CKPT}" \
        --joint_belief_clean_patch_min_prob "${PATCH_MIN_PROB}" \
        --joint_belief_clean_patch_margin "${PATCH_MARGIN}" \
        --joint_belief_clean_replan_min_prob "${REPLAN_MIN_PROB}" \
        --dynamic_coupling_delta_action_repair_mix "${PATCH_MIX}"
      ;;
    *)
      echo "Unknown MODE=${mode}. Use base, direct, hyp_replan, ours, or all." >&2
      exit 2
      ;;
  esac
}

exec_or_run() {
  if [[ "${MODE}" == "all" ]]; then
    "$@"
  else
    exec "$@"
  fi
}

if [[ "${MODE}" == "all" ]]; then
  for mode in base direct hyp_replan ours; do
    run_mode "${mode}"
  done
else
  run_mode "${MODE}"
fi
