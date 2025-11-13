from tokenize import group
from torch.utils.data import Dataset
import sys
sys.path.append('../utils/')
from utils.dataloader_utils import load_image, load_mask, load_depthmap
from collections import defaultdict
from natsort import natsorted 
from pathlib import Path
import cv2
import torch
import numpy as np
import json
import re

def get_video_root(f):
    ''' Given a file path, return the root directory of the video sequence. 
    /..case/video/images/0001.png -> /..case/video/
    input: file_name: including the full path.png
    output: the parent directory of the images file ''' 
    parts = f.parts  # split path into parts
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[:idx])
    else:
        return f.parent
    
def read_visible_frame(video_root):
    visible_frame_path = video_root / "visible_frames.json"
    visible_frames = []
    if visible_frame_path.exists():
        with open(visible_frame_path, 'r') as f:
            data = json.load(f)
            visible_frames = data.get("visible_frames", [])
    return visible_frames

def int_frame(frame_path):
    ''' Extract integer frame number from a file path.
    /..case/video/images/frame_001.png -> 1 '''
    s = frame_path.stem
    parts = s.split('_')
    last = parts[-1]
    return int(last)

class KPT(Dataset):
    """
    KeyPoint/Segmentation dataset with:
      - Target frame filtering: if action == "dissect", only frames listed in
        `visible_frames.json` are used as targets (t / idx).
      - Sliding window inputs: builds [t, t-1, t-2, ...] within the SAME video.
        When the window reaches the first frame of the video, it keeps padding
        with that first frame (no modulo wrap-around; no cross-video sampling).
    """
    def __init__(self, file_names, transform, mode, prediction_task,
                 num_input_frames, add_depth_inputs=False, action=None):
        # ---- Basic config ----
        # Ensure all inputs are Path objects
        self.file_names = [f if isinstance(f, Path) else Path(f) for f in file_names]
        self.transform = transform
        self.mode = mode
        self.prediction_task = prediction_task
        self.num_input_frames = num_input_frames
        self.add_depth_inputs = add_depth_inputs
        self.action = action

        # ---- Group by video root and sort each group ----
        groups = defaultdict(list)
        for f in self.file_names:
            groups[get_video_root(f)].append(f)
        for k in groups.keys():
            groups[k] = natsorted(groups[k])

        # ---- Flatten groups into parallel lists: all_frames / video_roots ----
        self.all_frames = []
        self.video_roots = []
        for vr in natsorted(list(groups.keys())):
            frames = groups[vr]
            self.all_frames.extend(frames)
            self.video_roots.extend([vr] * len(frames))

        # ---- For each video, record its starting global index in all_frames ----
        # Used to clamp the sliding window so it never crosses videos.
        self.start_idx_by_root = {}
        pos = 0
        for vr in natsorted(groups.keys()):
            self.start_idx_by_root[vr] = pos
            pos += len(groups[vr])

        # ---- Load visible-frame lists for "dissect" (frame numbers from file stem) ----
        self.visible_dict = {}
        self.skip_videos = set()  # <--- 新增：记录需要跳过的 video 根目录
        if self.action == "dissect":
            for vr in natsorted(groups.keys()):
                vis_list = read_visible_frame(vr)  # 可能不存在、为空或有内容
                if vis_list is not None and len(vis_list) == 0:
                    # JSON 存在但为空：跳过整个视频
                    print(f"[INFO] visible_frames.json empty -> skip video: {vr}")
                    self.skip_videos.add(vr)
                    continue
                # 正常收集（可能是空列表但我们上面已 continue，不会到这里）
                vis = set(vis_list) if vis_list else set()
                if vis:
                    self.visible_dict[vr] = vis

        # ---- Build the list of target indices (only these are sampled by DataLoader) ----
        # If action == "dissect": only visible frames become targets;
        # otherwise: all frames are valid targets.
        self.target_indices = []
        offset = 0
        for vr in natsorted(groups.keys()):
            # 新增：若该视频被标记为跳过，直接 continue
            if self.action == "dissect" and vr in self.skip_videos:
                offset += len(groups[vr])
                continue

            frames = groups[vr]  # already natsorted
            start = self.start_idx_by_root[vr]
            if self.action == "dissect":
                vis = self.visible_dict.get(vr, set())
                for j, fp in enumerate(frames):
                    # fp.stem must be an integer frame id, e.g., "0001" -> 1
                    if int_frame(fp) in vis:
                        self.target_indices.append(start + j)
            else:
                self.target_indices.extend(range(start, start + len(frames)))
            offset += len(frames)

        # Dataset length for DataLoader (counts only target frames)
        self.N = len(self.target_indices)
        # Total number of frames (for internal indexing only)
        self.N_all = len(self.all_frames)
        print(f"[DEBUG] Total frames={len(self.all_frames)}, target frames={self.N}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Map loader index -> global frame index of the target frame t
        gidx = self.target_indices[idx]
        img_t = self.all_frames[gidx]
        tgt_root = self.video_roots[gidx]

        # Load supervision (mask) on the target frame t
        mask = load_mask(img_t, self.prediction_task)

        # Build a non-cross-video window: [t, t-1, t-2, ...].
        # If the window goes past the first frame of this video,
        # clamp to that first frame (front padding).
        start = self.start_idx_by_root[tgt_root]  # global index of this video's first frame
        input_imgs, input_depth = [], []

        # If you want "strict history only" (exclude t itself), change range to:
        # for i in range(1, self.num_input_frames + 1):
        for i in range(self.num_input_frames):
            cur = gidx - i
            if cur < start:
                cur = start  # front padding at the video's first frame
            fp = self.all_frames[cur]
            input_imgs.append(load_image(fp))
            if self.add_depth_inputs:
                input_depth.append(load_depthmap(fp))

        sample = {'input': input_imgs, 'mask': mask}
        if self.add_depth_inputs:
            sample['input_depth'] = input_depth
        sample = self.transform(sample)
        return sample
