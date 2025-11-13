from tokenize import group
from torch.utils.data import Dataset
import sys
sys.path.append('../utils/')
from utils.dataloader_utils import load_image, load_mask, load_depthmap
from collections import defaultdict
from natsort import natsorted 
import cv2
import torch
import numpy as np
from pathlib import Path


def get_act_video_root(f):
    ''' Given a file path, return the root directory of the video sequence. 
    /case/images/0001.png -> /..case'''
    parts = f.parts
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[:idx])
    else:
        return f.parent



class ACT(Dataset):
    def __init__(self, file_names, transform, mode, prediction_task, num_input_frames, add_depth_inputs=False): 
        self.file_names = file_names
        self.transform = transform
        self.mode = mode 
        self.prediction_task = prediction_task
        self.num_input_frames = num_input_frames
        #self.num_frames_per_video = num_frames_per_video
        self.add_depth_inputs = add_depth_inputs

        # Group frames by video sequence
        groups = defaultdict(list)
        for f in file_names:
            groups[get_act_video_root(f)].append(f)
        for k in groups.keys():
            groups[k] = natsorted(groups[k])
        
        self.all_frames = []
        self.video_roots = []
        self.frame_idx_in_video = {}
        for vr in natsorted(list(groups.keys())):
            frames = groups[vr]
            for local_idx, f in enumerate(frames):
                self.frame_idx_in_video[f] = local_idx
                self.all_frames.append(f)
                self.video_roots.append(vr)

        self.N = len(self.all_frames)

    def __len__(self): 
        return self.N
    
    def __getitem__(self, idx):
        img_t = self.all_frames[idx]
        tgt_root = self.video_roots[idx]

        # ==mask//heatmap reading==
        if self.prediction_task == 'keypoint_heatmap':
            frame_idx_in_video = self.frame_idx_in_video[img_t]
            sam_results_dir = tgt_root / 'sam_results'
            heatmap_path = sam_results_dir / f"frame_{frame_idx_in_video+1:03d}.npy"
            if heatmap_path.exists():
                heatmap = np.load(str(heatmap_path)).astype(np.float32)  # H x W x C
            else:
                print(f"[INFO] Cannot find heatmap for {img_t}")
                h, w = img_t.shape[:2]
                heatmap = np.zeros((h, w, 1), dtype=np.float32)
            mask = heatmap
        else:
            mask = load_mask(img_t, self.prediction_task)

        input_imgs = []
        input_depth = []

        last_valid_idx = idx

        # [t, t-1, t-2, ...] 
        for i in range(self.num_input_frames):
            cur = (idx - i) % self.N  
            # （last_valid_idx）
            if self.video_roots[cur] != tgt_root:
                cur = last_valid_idx
            else:
                last_valid_idx = cur

            fp = self.all_frames[cur]
            input_imgs.append(load_image(fp))
            if self.add_depth_inputs:
                input_depth.append(load_depthmap(fp))

        sample = {'input': input_imgs, 'mask': mask}
        if self.add_depth_inputs:
            sample['input_depth'] = input_depth

        sample = self.transform(sample)
        return sample        
