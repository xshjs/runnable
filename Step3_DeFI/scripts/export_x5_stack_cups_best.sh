#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/workspace/manipulation/DeFi/Step3_DeFI"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python}"
CKPT="${CKPT:-/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/best_val.pt}"
ROOT_DATA_DIR="${ROOT_DATA_DIR:-/mnt/workspace/manipulation/datasets/defi_x5_left_stack_cups_joint_action}"
VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:-/mnt/data/xiyin/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-/mnt/data/xiyin/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32}"
T5_MODEL_PATH="${T5_MODEL_PATH:-/mnt/data/xiyin/manipulation/DeFi/ckpts/t5_base}"
LANGUAGE_GOAL_PATH="${LANGUAGE_GOAL_PATH:-/mnt/data/xiyin/manipulation/DeFi/ckpts/ViT-B-32.pt}"
URDF="${URDF:-/tmp/arx_x5_sdk_src/arx_x5_sdk-0.1.7/arx_x5_sdk/urdf/x5_2025.urdf}"
SPLIT="${SPLIT:-validation}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/x5_exports_best_val_joint_action}"

cd "$ROOT_DIR"

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" \
"$PYTHON_BIN" scripts/export_x5_action_from_ckpt.py \
  --config_name VPP_Calvinabc_train_joint_action \
  --action_format joint_absolute \
  --ckpt "$CKPT" \
  --root_data_dir "$ROOT_DATA_DIR" \
  --video_model_path "$VIDEO_MODEL_PATH" \
  --text_encoder_path "$TEXT_ENCODER_PATH" \
  --t5_model_path "$T5_MODEL_PATH" \
  --language_goal_path "$LANGUAGE_GOAL_PATH" \
  --split "$SPLIT" \
  --sample_index "$SAMPLE_INDEX" \
  --export_mode all \
  --urdf "$URDF" \
  --output_dir "$OUTPUT_DIR"
