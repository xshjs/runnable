# Runnable Upload Manifest

This file lists the minimal code and checkpoint recommendations for the real-robot DeFi runnable upload.

## Recommended ckpts

- `pen_tape_cutter_tray` primary:
  - `/mnt/data/xiyin/manipulation/DeFi/outputs/calvin_train/2026-08-24_20-08-09/saved_models/epoch_002.pt`
- `pen_tape_cutter_tray` secondary:
  - `/mnt/data/xiyin/manipulation/DeFi/outputs/calvin_train/2026-08-25_19-44-39/saved_models/best_val.pt`
- `stack_cups` primary:
  - `/mnt/data/xiyin/manipulation/DeFi/outputs/calvin_train/2026-08-25_09-47-55/saved_models/epoch_004.pt`
- `stack_cups` secondary:
  - `/mnt/data/xiyin/manipulation/DeFi/outputs/calvin_train/2026-08-25_09-47-55/saved_models/epoch_007.pt`

## Upload this code set

These files are the core changes for real-robot dataset conversion, training, validation, checkpoint saving, and compatibility fixes:

- `Step3_DeFI/scripts/convert_lerobot_v30_to_defi_real_robot.py`
- `Step3_DeFI/scripts/compute_x5_ee_from_lerobot.py`
- `Step3_DeFI/scripts/train_calvin.py`
- `Step3_DeFI/scripts/train_calvin_real_robot.sh`
- `Step3_DeFI/policy_conf/VPP_Calvinabc_train.yaml`
- `Step3_DeFI/policy_models/VPP_policy.py`
- `Step3_DeFI/policy_models/datasets/disk_dataset.py`
- `Step3_DeFI/policy_models/edm_diffusion/gc_sampling.py`
- `Step3_DeFI/policy_models/edm_diffusion/score_wrappers.py`
- `Step3_DeFI/policy_models/m_former_univla/blocks.py`
- `Step3_DeFI/policy_models/module/Video_Former.py`
- `Step3_DeFI/policy_models/utils/utils.py`
- `Step3_DeFI/README_BASELINE_RUNNABLE.md`

## What these changes do

- Real-robot LeRobot v3 to DeFi conversion for X5 data.
- FK-based EE delta action generation with `future_state_offset=5`.
- Validation metrics at epoch end:
  - `val_loss`
  - `val_first_mae`
  - `val_chunk_mae`
  - `val_xyz_mae`
  - `val_rot_mae`
  - `val_gripper_mae`
- Save checkpoints:
  - every `save_every` steps
  - every epoch as `epoch_xxx.pt`
  - best validation checkpoint as `best_val.pt`
- Compatibility fixes for missing optional deps and CPU/CUDA hardcoding issues.

## Current training config behavior

- `save_every: 2000`
- `val_num_batches: 20`
- `act_seq_len: 50`
- `multistep: 50`

## Note on `action_dim_weights`

`policy_models/edm_diffusion/score_wrappers.py` and `policy_models/VPP_policy.py` include optional action-dimension weighting support.

Old checkpoints may show:

- `Missing keys (in model but not in checkpoint): model.action_dim_weights`

That is expected for checkpoints trained before this parameter existed.

## Suggested upload policy

Do not upload the entire current working tree blindly.

Prefer uploading only the files listed in `Upload this code set`, because the repo currently also contains many unrelated local changes and experimental files.
