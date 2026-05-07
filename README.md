# Multi-Frame Keypoint Detection with Temporal Decoder

Official repository for multi-frame surgical tool keypoint detection with temporal context modeling.

This project extends a heatmap-based surgical tool keypoint localization pipeline by adding a temporal decoder for frame-wise feature aggregation and representation tracking across surgical video frames.

The pipeline supports pose map generation, depth map generation, action-wise dataset reconstruction, multi-frame model training, temporal decoder training, and inference.

---

## Overview

This repository focuses on surgical tool keypoint localization from multi-frame video sequences.

Given a sequence of surgical video frames, the model predicts keypoint heatmaps for sparse functional landmarks such as tool tips and anchor points. The updated framework introduces a temporal decoder to aggregate visual features across frames and improve frame-to-frame consistency.

Main components include:

- Multi-frame surgical video input
- Heatmap-based keypoint localization
- Depth map generation using Depth Anything V2
- Action-wise dataset reconstruction
- Temporal decoder for feature aggregation and representation tracking
- Optional segmentation or mask-guided structural learning
- Training and inference scripts for KPT and ACT datasets

---

## Requirements

```text
configargparse
logging
json
numpy
torch
torchvision
albumentations
opencv-python
tensorboardX
Depth-Anything-V2
```

Install common dependencies with:

```bash
pip install numpy torch torchvision albumentations opencv-python tensorboardX configargparse
```

Depth Anything V2 should be installed or cloned separately following its official setup instructions.

---

## Code Structure

```text
surg_act_keypoint/
├── README.md
├── mfcnet-tracker/
│   ├── configs/                 # Configuration files
│   ├── models/                  # Model definitions
│   │   ├── unet.py
│   │   ├── mfcnet.py
│   │   └── temporal_decoder.py  # Temporal decoder for multi-frame feature aggregation
│   ├── scripts/                 # Preprocessing, training, and inference scripts
│   │   ├── pose_map_generate.py
│   │   ├── reconstruction_by_action.py
│   │   ├── train_multiframe_detection.py
│   │   ├── train_multiframe_decoder.py
│   │   └── infer_new.py
│   └── src/
│       ├── dataloader.py
│       ├── engine.py
│       └── ...
├── Depth-Anything-V2/
│   └── depth_generate.py        # Depth map generation
└── utils/
    ├── log_utils.py
    └── ...
```

---

## Dataset

This repository supports surgical datasets including **KPT** and **ACT**.

Because the raw datasets have different folder structures, preprocessing is required before training. The preprocessing pipeline reorganizes the data by action category and generates the required RGB frames, pose maps, depth maps, and optional segmentation masks.

---

## Expected Dataset Structure

### KPT Dataset

After preprocessing, the KPT dataset should follow this structure:

```text
reconstruction_by_action/
├── clip/
│   ├── case_001_video_part_001_segment_3/
│   │   ├── video_001/
│   │   │   ├── depth_maps/
│   │   │   ├── images/
│   │   │   ├── pose_map/
│   │   │   └── visible_frames.json
│   │   ├── video_002/
│   │   ├── video_003/
│   │   └── video_004/
│   └── ...
├── grasp/
├── dissect/
└── cut/
```

### ACT Dataset

After preprocessing, the ACT dataset should follow this structure:

```text
training_act_data_by_action/
├── clip/
│   ├── tissue_1_2_clipping_first_clip_left_tub_.../
│   │   ├── depth_maps/
│   │   ├── images/
│   │   ├── pose_map/
│   │   └── sam_results/
├── grasp/
└── cut/
```

---

## Preprocessing Pipeline

### 1. Generate Pose Maps and Extract RGB Frames

```bash
cd mfcnet-tracker

python3 scripts/pose_map_generate.py \
  --dataset KPT
```

For ACT:

```bash
python3 scripts/pose_map_generate.py \
  --dataset ACT
```

---

### 2. Generate Depth Maps

```bash
cd Depth-Anything-V2

python3 depth_generate.py \
  --dataset KPT
```

For ACT:

```bash
python3 depth_generate.py \
  --dataset ACT
```

---

### 3. Reconstruct Dataset by Action

```bash
cd mfcnet-tracker

python3 scripts/reconstruction_by_action.py \
  --dataset KPT
```

For ACT:

```bash
python3 scripts/reconstruction_by_action.py \
  --dataset ACT
```

The reconstructed dataset will be organized by action category:

```text
clip/
grasp/
dissect/
cut/
```

---

## Model

The framework follows a multi-frame heatmap prediction design.

The input is a sequence of surgical video frames. Frame-level visual features are extracted and passed to a temporal decoder. The decoder aggregates temporal context across frames and predicts keypoint heatmaps for the target frame.

The temporal decoder is used to:

- Track surgical tool representations across frames
- Aggregate temporal context from neighboring frames
- Improve frame-to-frame consistency
- Refine heatmap-based keypoint predictions

---

## Training

### Multi-Frame Keypoint Detection

```bash
cd mfcnet-tracker

torchrun --nproc-per-node=2 scripts/train_multiframe_detection.py \
  --data_dir <RECONSTRUCTION_BY_ACTION_PATH> \
  --dataset KPT \
  --action dissect \
  --prediction_task keypoint_segmentation \
  --num_input_frames 8 \
  --expt_savedir <CHECKPOINT_DIR> \
  --expt_name multiframe_segmentation_expt_kpt_dissect_final \
  --batch_size 8 \
  --num_workers 16 \
  --num_classes 4 \
  --lr 1e-4 \
  --num_epochs 20
```

---

### Multi-Frame Training with Temporal Decoder

Use the decoder training script when temporal feature aggregation is enabled:

```bash
cd mfcnet-tracker

torchrun --nproc-per-node=2 scripts/train_multiframe_decoder.py \
  --data_dir <RECONSTRUCTION_BY_ACTION_PATH> \
  --dataset KPT \
  --action dissect \
  --prediction_task keypoint_segmentation \
  --num_input_frames 8 \
  --use_temporal_decoder True \
  --expt_savedir <CHECKPOINT_DIR> \
  --expt_name multiframe_decoder_kpt_dissect_final \
  --batch_size 8 \
  --num_workers 16 \
  --num_classes 4 \
  --lr 1e-4 \
  --num_epochs 20
```

For other actions, replace:

```text
--action dissect
```

with:

```text
--action clip
--action grasp
--action cut
```

---

## Experiment Outputs

All experiment outputs are saved under the specified checkpoint directory:

```text
checkpoint/
├── multiframe_segmentation_expt_kpt_clip_keypoint_final/
│   ├── ckpts/
│   ├── logs/
│   └── outputs/
├── multiframe_decoder_kpt_dissect_final/
│   ├── ckpts/
│   ├── logs/
│   └── outputs/
└── ...
```

The output folders contain:

```text
ckpts/      # Saved model checkpoints
logs/       # Training logs
outputs/    # Inference or visualization outputs
```

---

## Inference

Run inference with a trained checkpoint:

```bash
cd mfcnet-tracker

python scripts/infer_new.py \
  --data_dir <RECONSTRUCTION_BY_ACTION_PATH> \
  --dataset KPT \
  --action dissect \
  --prediction_task keypoint_segmentation \
  --expt_savedir <RESULTS_DIR> \
  --expt_name <RESULTS_NAME> \
  --num_workers 12 \
  --num_classes 4 \
  --seed 42 \
  --model_type <MODEL_TYPE> \
  --use_temporal_decoder True \
  --input_height 480 \
  --input_width 640 \
  --load_wts_model <MODEL_TRAINED_WEIGHTS_PATH>
```

Example:

```bash
python scripts/infer_new.py \
  --data_dir /path/to/reconstruction_by_action \
  --dataset KPT \
  --action dissect \
  --prediction_task keypoint_segmentation \
  --expt_savedir /path/to/results \
  --expt_name decoder_dissect_inference \
  --num_workers 12 \
  --num_classes 4 \
  --seed 42 \
  --model_type mfcnet_decoder \
  --use_temporal_decoder True \
  --input_height 480 \
  --input_width 640 \
  --load_wts_model /path/to/checkpoint/model_010.pth
```

---

## Checkpoints

Model checkpoints are not included in this repository because of file size limitations.

Please place trained model weights under:

```text
checkpoint/<experiment_name>/ckpts/
```

Example:

```text
checkpoint/
└── multiframe_decoder_kpt_dissect_final/
    └── ckpts/
        └── model_010.pth
```

Large model weights should be managed using Git LFS or stored externally.

---

## Notes

- Dataset files are not included due to size and privacy restrictions.
- Large model checkpoints should not be committed directly to GitHub.
- Depth Anything V2 checkpoints should be downloaded separately.
- The temporal decoder can be enabled for experiments requiring temporal feature aggregation.
- The repository is designed for action-wise surgical video keypoint detection and temporal modeling.

---

## Project Summary

This project implements a multi-frame surgical tool keypoint detection pipeline with temporal decoder-based representation tracking.

It combines heatmap-based keypoint localization, depth map generation, action-wise preprocessing, and temporal context modeling to improve surgical video understanding.

The main goal is to localize sparse functional landmarks, such as tool tips and anchor points, from surgical video sequences while using temporal information to improve prediction consistency.
