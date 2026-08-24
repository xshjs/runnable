#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

ROOT_DATA_DIR="${ROOT_DATA_DIR:-}"
VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:-}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-}"
T5_MODEL_PATH="${T5_MODEL_PATH:-}"
LANGUAGE_GOAL_PATH="${LANGUAGE_GOAL_PATH:-}"
TOKEN_CKPT_PATH="${TOKEN_CKPT_PATH:-}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-28}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${ROOT_DATA_DIR}" || -z "${VIDEO_MODEL_PATH}" || -z "${TEXT_ENCODER_PATH}" || -z "${T5_MODEL_PATH}" || -z "${LANGUAGE_GOAL_PATH}" ]]; then
  cat <<'EOF'
Missing required environment variables.

Required:
  ROOT_DATA_DIR
  VIDEO_MODEL_PATH
  TEXT_ENCODER_PATH
  T5_MODEL_PATH
  LANGUAGE_GOAL_PATH

Optional:
  TOKEN_CKPT_PATH
  NUM_GPUS=1
  BATCH_SIZE=28
  MAX_EPOCHS=12
  NUM_WORKERS=12
  PYTHON_BIN=python

Example:
  ROOT_DATA_DIR=/path/to/real_robot_dataset \
  VIDEO_MODEL_PATH=/path/to/step1_gfdm \
  TEXT_ENCODER_PATH=/path/to/openai_clip_vit_base_patch32 \
  T5_MODEL_PATH=/path/to/t5_base \
  LANGUAGE_GOAL_PATH=/path/to/ViT-B-32.pt \
  NUM_GPUS=4 \
  BATCH_SIZE=28 \
  MAX_EPOCHS=12 \
  PYTHON_BIN=/path/to/venv/bin/python \
  bash Step3_DeFI/scripts/train_calvin_real_robot.sh
EOF
  exit 1
fi

TOKEN_ARGS=()
if [[ -n "${TOKEN_CKPT_PATH}" ]]; then
  TOKEN_ARGS+=(--token_ckpt_path "${TOKEN_CKPT_PATH}")
fi

LAUNCH_ARGS=()
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  LAUNCH_ARGS=( -m accelerate.commands.launch --num_processes "${NUM_GPUS}" )
fi

"${PYTHON_BIN}" "${LAUNCH_ARGS[@]}" "${ROOT_DIR}/scripts/train_calvin.py" \
  --root_data_dir "${ROOT_DATA_DIR}" \
  --video_model_path "${VIDEO_MODEL_PATH}" \
  --text_encoder_path "${TEXT_ENCODER_PATH}" \
  --t5_model_path "${T5_MODEL_PATH}" \
  --language_goal_path "${LANGUAGE_GOAL_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  "${TOKEN_ARGS[@]}"
