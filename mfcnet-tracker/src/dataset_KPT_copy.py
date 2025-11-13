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
    

class KPT_test(Dataset):
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
            groups[get_video_root(f)].append(f)
        for k in groups.keys():
            groups[k] = natsorted(groups[k])

        # Build frame groups
        self.all_frames = []
        self.video_roots = []
        for vr in natsorted(list(groups.keys())):
            frames = groups[vr]
            self.all_frames.extend(frames)
            self.video_roots.extend([vr] * len(frames))

        self.N = len(self.all_frames)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img_t = self.all_frames[idx]
        tgt_root = self.video_roots[idx]

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