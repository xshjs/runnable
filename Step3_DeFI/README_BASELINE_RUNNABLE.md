# DeFi Baseline Runnable Notes

This note is for the pure DeFi baseline path only:

- training: `Step3_DeFI/policy_conf/VPP_Calvinabc_train.yaml`
- evaluation: `Step3_DeFI/policy_evaluation/calvin_evaluate.py`

It is intended to make the repo runnable by collaborators without editing local absolute paths in source files.

## 1. Dataset layout expected by training/eval

The datamodule expects:

```text
<ROOT_DATA_DIR>/
  training/
    ep_start_end_ids.npy
    statistics.yaml                  # optional but recommended
    auto_lang_ann.npy                # or training/lang_clip_resnet50/auto_lang_ann.npy
    episode_0000000.npz              # naming pattern can vary, but must be consistent
    episode_0000001.npz
    ...
  validation/
    ep_start_end_ids.npy
    statistics.yaml                  # optional
    auto_lang_ann.npy                # or validation/lang_clip_resnet50/auto_lang_ann.npy
    episode_0000000.npz
    episode_0000001.npz
    ...
```

Each episode file should at minimum provide keys used by `ExtendedDiskDataset`:

- `rgb_static`
- `rgb_gripper`
- `robot_obs`
- `scene_obs`
- `rel_actions`

Optional keys that are consumed if present:

- `stage`
- `trans_action_indicies`
- `rot_grip_action_indicies`
- `ignore_collisions`
- `gripper_pose`
- `rlbench_target_index`

`auto_lang_ann.npy` should contain the usual CALVIN-style structure:

- `info.indx`
- `language.emb`
- `language.ann`
- optional `language.task`

## 2. Important behavior change

`HulcDataModule` no longer interactively downloads the tiny debug dataset by default.

If your dataset path is wrong, training now fails fast with a clear `FileNotFoundError`.

If you intentionally want the tiny debug dataset, set:

```bash
export DEFI_ALLOW_DEBUG_DATASET_DOWNLOAD=1
```

## 3. Pure baseline evaluation

The evaluation entrypoint now supports parameterized horizon instead of hardcoding 5-step or 10-step in source.

### 5-step, 1000 sequences

```bash
cd /path/to/DeFi/Step3_DeFI

PYTHONPATH=/path/to/DeFi/Step3_DeFI \
python policy_evaluation/calvin_evaluate.py \
  --video_model_path /path/to/step1_gfdm \
  --action_model_folder /path/to/action_model_folder \
  --clip_model_path /path/to/openai_clip_vit_base_patch32 \
  --t5_model_path /path/to/t5_base \
  --language_goal_path /path/to/ViT-B-32.pt \
  --calvin_abc_dir /path/to/task_ABC_D \
  --eval_sequences_path /path/to/5step_sequences.json \
  --num_sequences 1000 \
  --eval_horizon 5
```

### 10-step, 100 sequences from a custom JSON

```bash
cd /path/to/DeFi/Step3_DeFI

PYTHONPATH=/path/to/DeFi/Step3_DeFI \
python policy_evaluation/calvin_evaluate.py \
  --video_model_path /path/to/step1_gfdm \
  --action_model_folder /path/to/action_model_folder \
  --clip_model_path /path/to/openai_clip_vit_base_patch32 \
  --t5_model_path /path/to/t5_base \
  --language_goal_path /path/to/ViT-B-32.pt \
  --calvin_abc_dir /path/to/task_ABC_D \
  --default_sequences_path /path/to/custom_sequences.json \
  --default_sequence_len 10 \
  --num_sequences 100 \
  --use_default_sequences \
  --eval_horizon 10
```

## 4. Training command skeleton

For collaborators training on real robot data, the important thing is to pass dataset and model paths through config/CLI instead of modifying source.

Actual training entrypoint in this repo:

- `Step3_DeFI/scripts/train_calvin.py`
- `Step3_DeFI/scripts/train_calvin_real_robot.sh`

Single-GPU example:

```bash
cd /path/to/DeFi/Step3_DeFI

PYTHONPATH=/path/to/DeFi/Step3_DeFI \
python scripts/train_calvin.py \
  --root_data_dir /path/to/real_robot_dataset \
  --video_model_path /path/to/step1_gfdm \
  --text_encoder_path /path/to/openai_clip_vit_base_patch32 \
  --t5_model_path /path/to/t5_base \
  --language_goal_path /path/to/ViT-B-32.pt \
  --batch_size 28 \
  --max_epochs 12
```

Multi-GPU example:

```bash
cd /path/to/DeFi/Step3_DeFI

PYTHONPATH=/path/to/DeFi/Step3_DeFI \
accelerate launch --num_processes 4 \
  scripts/train_calvin.py \
  --root_data_dir /path/to/real_robot_dataset \
  --video_model_path /path/to/step1_gfdm \
  --text_encoder_path /path/to/openai_clip_vit_base_patch32 \
  --t5_model_path /path/to/t5_base \
  --language_goal_path /path/to/ViT-B-32.pt \
  --batch_size 28 \
  --max_epochs 12
```

Wrapper script example:

```bash
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
```

If your team uses a different training launcher, keep the same contract:

- dataset root passed from config/CLI
- no source edits for local paths
- no interactive fallback download

## 5. What should not stay hardcoded in a shared branch

Before pushing a collaborative branch, avoid keeping these in source:

- `/mnt/workspace/...`
- `/mnt/data/...`
- personal eval JSON paths
- personal output directories

Those should live in:

- CLI args
- environment variables
- small wrapper shell scripts committed under `scripts/`
