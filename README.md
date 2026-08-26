# Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining
### [[Paper]](https://openreview.net/pdf?id=DdrsHWobR1) [[HuggingFace]](https://huggingface.co/zbzzbz/DeFI) 

> **Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining**            
> [Wenyao Zhang*](https://zhangwenyao1.github.io/), [Bozhou Zhang*](https://zbozhou.github.io/), [Zekun Qi](https://qizekun.github.io/), [Wenjun Zeng](https://scholar.google.com/citations?user=_cUfvYQAAAAJ&hl=zh-CN), [Xin Jin](https://scholar.google.com/citations?user=byaSC-kAAAAJ&hl=zh-CN), [Li Zhang](https://lzrobots.github.io)  
> **ICLR 2026**

## X5 Real-Robot Runnable

This repository now includes a runnable X5 real-robot path built around `Step3_DeFI`.

If you only care about running the current X5 checkpoints on real data, start here instead of reading the full paper pipeline first.

### Shared dataset folders

- `pen`: `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403`
- `cups`: `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133`

### Recommended checkpoints

- `pen` first-choice:
  - `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/epoch_002.pt`
- `pen` second-choice:
  - `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/best_val.pt`
- `cups` first-choice:
  - `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/epoch_004.pt`
- `cups` second-choice:
  - `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/epoch_007.pt`

As of August 26, 2026, the following export paths have already been verified to run successfully:

- `pen` `epoch_002.pt`
- `cups` `epoch_004.pt`

### What the model outputs

The Step3 DeFi action head predicts normalized 7D relative actions:

- dim `0:3`: normalized EE `delta_xyz`
- dim `3:6`: normalized EE `delta_euler_xyz`
- dim `6`: gripper signal

This is not a raw X5 command. For real-robot rollout, export it into:

- `ee.npy` if the execution side consumes end-effector delta pose
- `joint.npy` if the execution side consumes joint deltas

For current X5 integration, prefer `ee.npy` first.

### Real-robot export commands

`pen` first-choice, export `raw + ee + joint`:

```bash
cd /mnt/workspace/manipulation/DeFi/Step3_DeFI

python scripts/export_x5_action_from_ckpt.py \
  --ckpt /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/epoch_002.pt \
  --root_data_dir /mnt/workspace/manipulation/datasets/defi_x5_left_pen_tape_cutter_tray_fk_ee_offset5 \
  --video_model_path /mnt/data/xiyin/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm \
  --text_encoder_path /mnt/data/xiyin/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32 \
  --t5_model_path /mnt/data/xiyin/manipulation/DeFi/ckpts/t5_base \
  --language_goal_path /mnt/data/xiyin/manipulation/DeFi/ckpts/ViT-B-32.pt \
  --split validation \
  --sample_index 0 \
  --export_mode all \
  --raw_dataset_root /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403 \
  --episode_index 0 \
  --frame_index 0 \
  --urdf /tmp/arx_x5_sdk_src/arx_x5_sdk-0.1.7/arx_x5_sdk/urdf/x5_2025.urdf \
  --binary_gripper
```

`cups` first-choice, export `raw + ee + joint`:

```bash
cd /mnt/workspace/manipulation/DeFi/Step3_DeFI

python scripts/export_x5_action_from_ckpt.py \
  --ckpt /mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/epoch_004.pt \
  --root_data_dir /mnt/workspace/manipulation/datasets/defi_x5_left_stack_cups_fk_ee_offset5 \
  --video_model_path /mnt/data/xiyin/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm \
  --text_encoder_path /mnt/data/xiyin/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32 \
  --t5_model_path /mnt/data/xiyin/manipulation/DeFi/ckpts/t5_base \
  --language_goal_path /mnt/data/xiyin/manipulation/DeFi/ckpts/ViT-B-32.pt \
  --split validation \
  --sample_index 0 \
  --export_mode all \
  --raw_dataset_root /mnt/data/shared/hxw/x5_left_stack_cups_0824_1133 \
  --episode_index 0 \
  --frame_index 0 \
  --urdf /tmp/arx_x5_sdk_src/arx_x5_sdk-0.1.7/arx_x5_sdk/urdf/x5_2025.urdf \
  --binary_gripper
```

Generated files are written under each shared folder:

- `x5_exports/validation_sample_00000_raw.npy`
- `x5_exports/validation_sample_00000_ee.npy`
- `x5_exports/validation_sample_00000_joint.npy`
- `x5_exports/validation_sample_00000_summary.json`

### Training on X5 datasets

The current X5 real-robot training entrypoint is:

- `Step3_DeFI/scripts/train_calvin_real_robot.sh`

The converted DeFi-format datasets currently used are:

- `pen`: `/mnt/workspace/manipulation/datasets/defi_x5_left_pen_tape_cutter_tray_fk_ee_offset5`
- `cups`: `/mnt/workspace/manipulation/datasets/defi_x5_left_stack_cups_fk_ee_offset5`

Current training setup used for the recommended checkpoints:

- action chunk: `50`
- action definition: future state offset `5`
- checkpoint saving:
  - every epoch
  - every `2000` steps
  - best validation checkpoint as `best_val.pt`

For detailed Step3 runnable notes, see:

- `Step3_DeFI/README_BASELINE_RUNNABLE.md`
- `Step3_DeFI/RUNNABLE_UPLOAD_MANIFEST.md`

## Abstract
Vision-language-action (VLA) models have shown great potential in building generalist robots, but still face a dilemma–misalignment of 2D image forecasting and 3D action prediction. Besides, such a vision-action entangled training manner limits model learning from large-scale, action-free web video data. To address these issues, we propose DeFI, a novel framework that Decouples visual Forward and Inverse dynamics pretraining to exploit respective data sources, wherein video generation and action prediction are disentangled. We introduce the General Forward Dynamics Model (GFDM), pretrained on diverse human and robot videos for future prediction, and the General Inverse Dynamics Model (GIDM), trained via self-supervised learning to infer latent actions from unlabeled video transitions. These models are then integrated into a unified architecture for end-to-end fine-tuning on downstream tasks. In this manner, GFDM and GIDM first shine separately and then cooperate for mutual benefit. Extensive experiments on CALVIN ABC-D and SimplerEnv demonstrate state-of-the-art performance, with DeFI achieving an average task length of 4.51 for CALVIN, 51.2% success rate on SimplerEnvFractal benchmark and 81.3% success rate in real-world deployment, significantly outperforming prior methods.

## News
- 2026-03, we release the original version of DeFI, which includes pre-training and evaluation on the CALVIN benchmark.

## Pipeline
<div align="center">
  <img src="asset/defi.PNG"/>
</div><br/>

## TODO list
- [x] release the original version on the CALVIN benchmark
- [ ] add more benchmarks
- [ ] integrated into lerobot and starvla format

## Environment
```
conda create -n defi python==3.10
conda activate defi

pip install setuptools==57.5.0
git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin
sh install.sh

cd DeFI_PATH
pip install -r requirements.txt

pip uninstall -y torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

## The required checkpoints and data

<details>
<summary><b> Checkpoints </b></summary>

- [stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
- [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- ["ViT-B-32.pt" (ViT-B/32) in the CLIP](https://github.com/openai/CLIP)
- [t5-base](https://huggingface.co/google-t5/t5-base)

Download these weights and place them in the "ckpts" folder.

</details>

<details>
<summary><b> Data </b></summary>

### Stage 1
For the pre-training of GFDM, we follow the data processing procedure and data format of [VPP](https://github.com/roboterax/video-prediction-policy?tab=readme-ov-file#-stage-1-training-video-model).

Example:
```
data/opensource_robotdata/bridge/
├── annotation/
│   ├── train/
│   └── val/
│       ├── 2.json      ← annotation file
│       └── ... (more json files)
└── videos/
│   ├── train/
│   └── val/
│       ├── 2/
│       │   └── rgb.mp4      ← video file
│       └── ... (more video directories)
└── latent_videos/
    ├── train/
    └── val/
        ├── 2/
        │   └── 0.pt      ← latent video file
        └── ... (more latent video directories)


```
- During training, the JSON files (annotation files) are used, which further reference the PT files (latent video files).
- During evaluation, the JSON files (annotation files) are used, which further reference the MP4 files (video files).

### Stage 2
For the pre-training of GIDM, we follow the data processing procedure and data format of [UniVLA](https://github.com/OpenDriveLab/UniVLA?tab=readme-ov-file#zero-data-preparation).

Example:
```
data/oxe/
├── DOWNLOAD_DIR/
│   ├── fractal20220817_data/
│   ├── language_table/
│   └── ...
└── CONVERSION_DIR/
    ├── fractal20220817_data/
    ├── language_table/
    └── ...
```

### Stage 3
Download the Calvin ABC-D dataset from follow [Calvin](https://github.com/mees/calvin?tab=readme-ov-file#computer--quick-start).

</details>

## Train and eval
```
### Stage 1 Scripts ###

cd Step1_GFDM

# Encode dataset videos with VAE for preprocessing
scripts/prepare_data_latent.sh

# Train FDM
scripts/train_svd.sh

# Evaluate FDM
scripts/eval_svd.sh

### Stage 2 Scripts ###

cd Step2_GIDM

# Train IDM
train.sh

### Stage 3 Scripts ###

cd Step3_DeFI

# Train DeFI
scripts/train_calvin.sh

# Evaluate DeFI
scripts/rollout_calvin.sh

```

## BibTeX
```bibtex
@article{zhang2026disentangled,
  title={Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining},
  author={Zhang, Wenyao and Zhang, Bozhou and Qi, Zekun and Zeng, Wenjun and Jin, Xin and Zhang, Li},
  journal={arXiv preprint arXiv:2604.16391},
  year={2026}
}
```

## Acknowledgements
- [VPP](https://github.com/roboterax/video-prediction-policy)
- [UniVLA](https://github.com/OpenDriveLab/UniVLA)
- [X-VLA](https://github.com/2toinf/X-VLA)
