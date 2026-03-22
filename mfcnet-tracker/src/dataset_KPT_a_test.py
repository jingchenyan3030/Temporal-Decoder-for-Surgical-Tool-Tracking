from torch.utils.data import Dataset
import sys
sys.path.append('../utils/')
from utils.dataloader_utils import load_image, load_mask, load_depthmap
from collections import defaultdict
from natsort import natsorted
from pathlib import Path
import torch
import numpy as np
import json


'''
'mask': heatmap (train ground truth)
'sam': sam_mask (spatial prior)
'''

def get_video_root(f):
    parts = f.parts  
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[:idx])
    else:
        return f.parent
    
import torch

def sanitize_points(points, visibility, H, W, *, oob_policy="invalidate"):
    pts = points.clone()
    vis = visibility.clone().float()

    finite = torch.isfinite(pts).all(dim=1)
    vis[~finite] = 0.0
    pts[~finite] = 0.0

    oob = (vis > 0.5) & (
        (pts[:, 0] < 0) | (pts[:, 0] > W - 1) |
        (pts[:, 1] < 0) | (pts[:, 1] > H - 1)
    )
    if oob.any():
        if oob_policy == "invalidate":
            vis[oob] = 0.0
            pts[oob] = 0.0
        else:  # "clamp"
            pts[:, 0] = pts[:, 0].clamp(0.0, float(W - 1))
            pts[:, 1] = pts[:, 1].clamp(0.0, float(H - 1))

    vis = (vis > 0.5).float()
    return pts, vis


class KPT_test_a(Dataset):

    def __init__(
        self,
        file_names,
        transform,
        mode,
        prediction_task,
        num_input_frames,
        add_depth_inputs=False,
        target_pos_from_start=4,
    ):

        self.file_names = file_names
        self.transform = transform
        self.mode = mode
        self.prediction_task = prediction_task
        self.num_input_frames = num_input_frames
        self.add_depth_inputs = add_depth_inputs

    
        self.K = num_input_frames
        assert target_pos_from_start is not None, "target_pos_from_start must be set"
        self.k = target_pos_from_start
  

        self.post = self.k - 1
        self.pre = self.K - self.k

     
        groups = defaultdict(list)
        for f in file_names:
            groups[get_video_root(f)].append(f)

        for k_ in groups.keys():
            groups[k_] = natsorted(groups[k_])


        max_len = self.K 
        for vr in groups.keys():
            frames = groups[vr]
            if len(frames) < max_len:
                frames = frames + [frames[-1]] * (max_len - len(frames))
            groups[vr] = frames        

   
        self.all_frames = []
        self.video_roots = []
        self.frame_video_start = []  
        self.frame_local_idx = []    
        self.frame_video_len = []   

        start = 0
        for vr in natsorted(list(groups.keys())):
            frames = groups[vr]
            L = len(frames)
            for j, f in enumerate(frames):
                self.all_frames.append(f)
                self.video_roots.append(vr)
                self.frame_video_start.append(start)
                self.frame_local_idx.append(j)
                self.frame_video_len.append(L)
            start += L

        self.centers = list(range(len(self.all_frames)))
        self.N = len(self.centers)




    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        center = self.centers[idx]

        video_start = self.frame_video_start[center]
        local_j = self.frame_local_idx[center]   
        L = self.frame_video_len[center]         

     
        input_indices = []
        for offset in range(-self.post, self.pre + 1):  
            local_i = local_j + offset
            if local_i < 0:
                local_i = 0
            if local_i >= L:
                local_i = L - 1
            global_i = video_start + local_i
            input_indices.append(global_i)

        #=== task specific loading
        img_t = self.all_frames[center]
        tgt_root = self.video_roots[center]

        sam = None
        # === task specific loading (ACT-style, clean) ===
        if self.prediction_task == 'keypoint_heatmap':
            # 1. heatmap(gt)
            sam_dir = tgt_root / 'sam_results'
            heatmap_path = sam_dir / (img_t.stem + ".npy")
            if not heatmap_path.exists():
                raise FileNotFoundError(f"Missing heatmap: {heatmap_path}")
            mask = np.load(str(heatmap_path)).astype(np.float32)
            # 2. sam_mask (spatial prior) neglect for now, can be added back if needed
            '''
            sam_path = sam_dir / (img_t.stem + "_mask.npy")
            if not sam_path.exists():
                raise FileNotFoundError(f"Missing SAM mask: {sam_path}")
            sam = np.load(str(sam_path)).astype(np.float32)
            sam = (sam > 0).astype(np.float32)
            '''
            '''
            # 3. weight map for sam_mask
            sam_weight = None
            sam_weight_path = sam_dir / (img_t.stem + "_weight.npy")
            if sam_weight_path.exists():
                sam_weight = np.load(str(sam_weight_path)).astype(np.float32)
            else:
                sam_weight = np.ones(mask.shape, dtype=np.float32)
            '''
      
        # debug
        if self.prediction_task == 'keypoint_heatmap':
            heatmap = np.load(str(heatmap_path))
            if heatmap.ndim == 3:
                C = heatmap.shape[0] if heatmap.shape[0] < heatmap.shape[-1] else heatmap.shape[-1]
            else:
                C = 1
            if C != 2:
                print(f"[BAD HEATMAP] {heatmap_path}, shape={heatmap.shape}")
        # debug end

        input_imgs = []
        input_depth = []

        for g in input_indices:
            fp = self.all_frames[g]
            input_imgs.append(load_image(fp))
            if self.add_depth_inputs:
                input_depth.append(load_depthmap(fp))

        sample = {
            'input': input_imgs,
            'mask': mask,
        }
        '''
        if self.prediction_task == 'keypoint_heatmap':
            sample['sam_weight'] = sam_weight
        '''
        if self.add_depth_inputs:
            sample['input_depth'] = input_depth
        sample = self.transform(sample)
        if self.prediction_task == 'keypoint_heatmap':
            full_mask = sample['mask']
            sample['anchor_map'] = full_mask[:,1:2,:,:]
            sample['mask'] = full_mask[:,0:1,:,:]
        sample['center_path'] = str(self.all_frames[center])
        return sample
