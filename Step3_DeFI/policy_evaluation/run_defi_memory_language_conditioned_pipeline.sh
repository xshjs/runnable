#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export NVIDIA_DRIVER_CAPABILITIES=all
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
if [[ "${UNSET_DISPLAY:-0}" == "1" ]]; then
  unset DISPLAY
fi
REQUESTED_EGL_VISIBLE_DEVICES="${EGL_VISIBLE_DEVICES:-0}"
if [[ "${UNSET_EGL_VISIBLE_DEVICES:-0}" == "1" ]]; then
  unset EGL_VISIBLE_DEVICES
  unset EGL_VISIBLE_DEVICE
else
  export EGL_VISIBLE_DEVICES="$REQUESTED_EGL_VISIBLE_DEVICES"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

unset LIBGL_ALWAYS_SOFTWARE
unset MESA_LOADER_DRIVER_OVERRIDE

export __GLX_VENDOR_LIBRARY_NAME=nvidia
if [[ "${PREPEND_SYSTEM_GL_LIBS:-0}" == "1" ]]; then
  export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

ROOT_DIR="${ROOT_DIR:-/mnt/data/manipulation/DeFi}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/mnt/data/manipulation/llm_models/Qwen3-8B}"
QWEN_LORA_PATH="${QWEN_LORA_PATH:-}"
QWEN_PYTHON_BIN="${QWEN_PYTHON_BIN:-}"
VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:-$ROOT_DIR/ckpts/svd/checkpoint-100000}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/ckpts/step3_defi.pt}"
ACTION_MODEL_FOLDER="${ACTION_MODEL_FOLDER:-}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-$ROOT_DIR/ckpts/openai_clip_vit_base_patch32}"
T5_MODEL_PATH="${T5_MODEL_PATH:-$ROOT_DIR/ckpts/t5_base}"
LANGUAGE_GOAL_PATH="${LANGUAGE_GOAL_PATH:-ViT-B/32}"
CALVIN_DIR="${CALVIN_DIR:-$ROOT_DIR/ckpts/task_ABC_D}"

EVAL_SEQUENCES_PATH="${EVAL_SEQUENCES_PATH:-$ROOT_DIR/outputs/splits/outcome_router_clean_1k_heldout110.json}"
USE_CALVIN_DEFAULT_SEQUENCES="${USE_CALVIN_DEFAULT_SEQUENCES:-0}"
NUM_SEQUENCES="${NUM_SEQUENCES:-110}"
EP_LEN="${EP_LEN:-}"
DEVICE="${DEVICE:-cuda:0}"
MEMORY_KEY_PATH="${MEMORY_KEY_PATH:-$ROOT_DIR/outputs/defi_memory_pipeline_clean110/memory/key_memory.jsonl}"
MEMORY_NPZ_PATH="${MEMORY_NPZ_PATH:-}"
CALIBRATOR_CKPT="${CALIBRATOR_CKPT:-}"
FUTURE_ADAPTER_CKPT="${FUTURE_ADAPTER_CKPT:-}"
PIPELINE_ROOT="${PIPELINE_ROOT:-$ROOT_DIR/outputs/defi_memory_language_conditioned}"
DISABLE_REFLECTION="${DISABLE_REFLECTION:-0}"

export PYTHONPATH="$ROOT_DIR/Step3_DeFI:$ROOT_DIR/calvin:$ROOT_DIR/calvin/calvin_env:${PYTHONPATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-$ROOT_DIR/scripts/nvidia_egl_vendor.json}"

echo "===== inside pipeline env ====="
echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHONPATH=$PYTHONPATH"
echo "NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "EGL_VISIBLE_DEVICES=${EGL_VISIBLE_DEVICES:-<unset>}"
echo "__EGL_VENDOR_LIBRARY_FILENAMES=$__EGL_VENDOR_LIBRARY_FILENAMES"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

cd "$ROOT_DIR"

mkdir -p "$PIPELINE_ROOT"

ACTION_MODEL_FOLDER_ARGS=()
if [[ -n "$ACTION_MODEL_FOLDER" ]]; then
  mkdir -p "$ACTION_MODEL_FOLDER/saved_models"
  ln -sf "$CHECKPOINT_PATH" "$ACTION_MODEL_FOLDER/saved_models/$(basename "$CHECKPOINT_PATH")"
  ACTION_MODEL_FOLDER_ARGS=(--action-model-folder "$ACTION_MODEL_FOLDER")
fi

EP_LEN_ARGS=()
if [[ -n "$EP_LEN" ]]; then
  EP_LEN_ARGS=(--ep-len "$EP_LEN")
fi

QWEN_LORA_ARGS=()
if [[ -n "$QWEN_LORA_PATH" ]]; then
  QWEN_LORA_ARGS=(--qwen-lora-path "$QWEN_LORA_PATH")
fi

QWEN_PYTHON_ARGS=()
if [[ -n "$QWEN_PYTHON_BIN" ]]; then
  QWEN_PYTHON_ARGS=(--qwen-python-bin "$QWEN_PYTHON_BIN")
fi

DISABLE_REFLECTION_ARGS=()
if [[ "$DISABLE_REFLECTION" == "1" ]]; then
  DISABLE_REFLECTION_ARGS=(--disable-reflection)
fi

CALVIN_DEFAULT_SEQUENCE_ARGS=()
if [[ "$USE_CALVIN_DEFAULT_SEQUENCES" == "1" ]]; then
  CALVIN_DEFAULT_SEQUENCE_ARGS=(--use-calvin-default-sequences)
fi

FUTURE_ADAPTER_ARGS=()
if [[ -n "$FUTURE_ADAPTER_CKPT" ]]; then
  FUTURE_ADAPTER_ARGS=(--future-adapter-ckpt "$FUTURE_ADAPTER_CKPT")
  if [[ -z "$MEMORY_NPZ_PATH" || -z "$CALIBRATOR_CKPT" ]]; then
    echo "FUTURE_ADAPTER_CKPT requires MEMORY_NPZ_PATH and CALIBRATOR_CKPT" >&2
    exit 2
  fi
  FUTURE_ADAPTER_ARGS+=(--memory-npz "$MEMORY_NPZ_PATH" --calibrator-ckpt "$CALIBRATOR_CKPT")
fi

"$PYTHON_BIN" Step3_DeFI/policy_evaluation/evaluate_defi_memory_language_conditioned.py \
  --video-model-path "$VIDEO_MODEL_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  "${ACTION_MODEL_FOLDER_ARGS[@]}" \
  --clip-model-path "$CLIP_MODEL_PATH" \
  --t5-model-path "$T5_MODEL_PATH" \
  --language-goal-path "$LANGUAGE_GOAL_PATH" \
  --calvin-abc-dir "$CALVIN_DIR" \
  --eval-sequences-path "$EVAL_SEQUENCES_PATH" \
  --output-dir "$PIPELINE_ROOT" \
  --memory-jsonl "$MEMORY_KEY_PATH" \
  --qwen-model-path "$QWEN_MODEL_PATH" \
  "${QWEN_LORA_ARGS[@]}" \
  "${QWEN_PYTHON_ARGS[@]}" \
  "${DISABLE_REFLECTION_ARGS[@]}" \
  "${CALVIN_DEFAULT_SEQUENCE_ARGS[@]}" \
  "${FUTURE_ADAPTER_ARGS[@]}" \
  --num-sequences "$NUM_SEQUENCES" \
  "${EP_LEN_ARGS[@]}" \
  --device "$DEVICE"

echo "done: $PIPELINE_ROOT/defi_memory_language_conditioned/summary.json"
