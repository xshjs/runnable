#!/bin/bash

<<COMMENT
Command:
sh scripts/eval_svd.sh \
    /mnt/data/xiyin/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm \
    /mnt/data/xiyin/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32 \
    /mnt/workspace/manipulation/datasets/defi_x5_left_stack_cups_fk_ee_offset5/validation \
    0 \
    30 \
    2>&1 | tee eval_svd.log
COMMENT

# 把脚本上一级目录（项目根）设为PYTHONPATH
export PYTHONPATH=$(dirname $(dirname $(realpath $0)))
echo "PYTHONPATH is $PYTHONPATH"

# 接收命令行参数
Video_Model_Path=$1
Clip_Model_Path=$2
Val_Dataset_Dir=$3
Val_Idx=${4:-"2+10+8+14"}
Num_Inference_Steps=${5:-30}  # 默认值30

echo "Starting SVD evaluation..."
echo "Video Model Path: $Video_Model_Path"
echo "Clip Model Path: $Clip_Model_Path"
echo "Validation Dataset Dir: $Val_Dataset_Dir"
echo "Validation Index: $Val_Idx"
echo "Num Inference Steps: $Num_Inference_Steps"

# 启动评估
python scripts/eval_svd.py \
    --eval \
    --config video_conf/val_calvin_svd.yaml \
    --video_model_path "$Video_Model_Path" \
    --clip_model_path "$Clip_Model_Path" \
    --val_dataset_dir "$Val_Dataset_Dir" \
    --val_idx "$Val_Idx" \
    --num_inference_steps $Num_Inference_Steps
