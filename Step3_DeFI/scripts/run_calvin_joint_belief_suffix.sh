#!/usr/bin/env bash
set -euo pipefail

# Conservative no-retry CALVIN rollout for the joint-belief suffix controller.
# By default z_a is kept inside the joint hypothesis and is NOT injected into
# the policy decoder. Do not add --dynamic_coupling_apply_action_intent_to_policy
# unless you explicitly want to test decoder-side action-intent conditioning.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${STEP3_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python}"
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
JOINT_BELIEF_TRANSITION_CKPT="${JOINT_BELIEF_TRANSITION_CKPT:-/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_transition.pt}"
JOINT_PAIR_ACTION_GENERATOR_CKPT="${JOINT_PAIR_ACTION_GENERATOR_CKPT:-/mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt}"

# Conservative suffix defaults from the Sep 3 no-retry run:
#  - patch rarely
#  - patch only with clear patch-vs-keep margin
#  - apply a very small suffix delta
JOINT_BELIEF_CLEAN_PATCH_MIN_PROB="${JOINT_BELIEF_CLEAN_PATCH_MIN_PROB:-0.55}"
JOINT_BELIEF_CLEAN_PATCH_MARGIN="${JOINT_BELIEF_CLEAN_PATCH_MARGIN:-0.10}"
JOINT_BELIEF_CLEAN_REPLAN_MIN_PROB="${JOINT_BELIEF_CLEAN_REPLAN_MIN_PROB:-0.85}"
DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX="${DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX:-0.02}"

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

cd "${STEP3_DIR}"

exec "${PYTHON_BIN}" \
  policy_evaluation/calvin_evaluate_with_memory_reflection.py \
  --video_model_path "${VIDEO_MODEL_PATH}" \
  --action_model_folder "${ACTION_MODEL_FOLDER}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --clip_model_path "${CLIP_MODEL_PATH}" \
  --t5_model_path "${T5_MODEL_PATH}" \
  --language_goal_path "${LANGUAGE_GOAL_PATH}" \
  --calvin_abc_dir "${CALVIN_ABC_DIR}" \
  --eval_sequences_path "${EVAL_SEQUENCES_PATH}" \
  --num_sequences "${NUM_SEQUENCES}" \
  --ep_len "${EP_LEN}" \
  --no_qwen_reflection \
  --disable_retry_after_first_pass \
  --dynamic_coupling_first_pass \
  --dynamic_coupling_online \
  --dynamic_coupling_generate_action_intent \
  --dynamic_coupling_persistent_hypothesis \
  --joint_belief_transition \
  --joint_belief_clean_controller \
  --dynamic_coupling_disable_replan \
  --joint_belief_transition_ckpt "${JOINT_BELIEF_TRANSITION_CKPT}" \
  --joint_pair_action_generator_ckpt "${JOINT_PAIR_ACTION_GENERATOR_CKPT}" \
  --joint_belief_clean_patch_min_prob "${JOINT_BELIEF_CLEAN_PATCH_MIN_PROB}" \
  --joint_belief_clean_patch_margin "${JOINT_BELIEF_CLEAN_PATCH_MARGIN}" \
  --joint_belief_clean_replan_min_prob "${JOINT_BELIEF_CLEAN_REPLAN_MIN_PROB}" \
  --dynamic_coupling_delta_action_repair_mix "${DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX}" \
  "$@"
