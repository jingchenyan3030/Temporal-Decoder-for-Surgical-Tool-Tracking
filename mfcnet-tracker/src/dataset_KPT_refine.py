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


class KPT_refine(Dataset):

    def __init__(
        self,
        file_names,
        transform,
        mode,
        prediction_task,
        num_input_frames,
        add_depth_inputs=False,
        target_pos_from_start=4,
        use_mask=True
    ):

        self.file_names = file_names
        self.transform = transform
        self.mode = mode
        self.prediction_task = prediction_task
        assert prediction_task == 'keypoint_heatmap'
        self.num_input_frames = num_input_frames
        self.add_depth_inputs = add_depth_inputs
        self.use_mask = use_mask
    
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
            sam_dir = tgt_root / 'sam_results'
            heatmap_path = sam_dir / (img_t.stem + ".npy")
            if not heatmap_path.exists():
                raise FileNotFoundError(f"Missing heatmap: {heatmap_path}")
            mask = load_mask(img_t, self.prediction_task).astype(np.float32)


            # 4. visibility for sam_mask
            detr_dir = tgt_root / 'points_detr'
            json_path = detr_dir / (img_t.stem + ".json")
            if not json_path.exists():
                raise FileNotFoundError(f"Missing DETR points: {json_path}")
            with open(json_path, 'r') as f:
                plist = json.load(f)
            if not isinstance(plist, list):
                raise ValueError(f"DETR points file corrupted: {json_path}")   
            tip_vis = 0
            anchor_vis = 0
            tip_flags = []
            for p in plist:
                lab = p.get("label","")
                vis = 1 if bool(p.get("visibility",True)) else 0

                if lab == "tool_tip":
                    tip_flags.append(vis)
                elif lab == "tool_anchor":
                    anchor_vis = vis
            tip_vis = 1 if any(tip_flags) else 0        
      
        else:
            mask = load_mask(img_t, self.prediction_task)
        #=== end task specific loading

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
        input_coarse = []
        input_sam = []
        input_depth = []

        for g in input_indices:
            fp = self.all_frames[g]
            vr = self.video_roots[g]

            input_imgs.append(load_image(fp))

            coarse_path = vr / "heatmap_coarse" / (fp.stem + ".npy")
            coarse = np.load(str(coarse_path)).astype(np.float32)
            if coarse.ndim == 2:
                raise ValueError(f"Coarse heatmap must have 2 channels at {coarse_path}")
            if coarse.ndim != 3:
                raise ValueError(f"Unexpected coarse shape {coarse.shape} at {coarse_path}")
            if coarse.shape[0] == 2 and coarse.shape[1] != 2:
                coarse = np.transpose(coarse, (1, 2, 0))
            if coarse.shape[2] != 2:
                raise ValueError(f"Coarse last dim must be 2. Got {coarse.shape} at {coarse_path}")
            input_coarse.append(coarse)

            if self.use_mask:
                sam_path = vr / "mask_coarse" / (fp.stem + ".npy")
                sam = np.load(str(sam_path)).astype(np.float32)
                if sam.ndim == 2:
                    sam = sam[..., None]
                elif sam.ndim == 3 and sam.shape[0] == 1:
                    sam = np.transpose(sam, (1, 2, 0))
                elif sam.ndim == 3 and sam.shape[2] == 1:
                    pass
                else:
                    raise ValueError(f"Unexpected sam shape {sam.shape} at {sam_path}")
            else:
                H, W = coarse.shape[:2]
                sam = np.zeros((H, W, 1), dtype=np.float32)

            input_sam.append(sam)

            if self.add_depth_inputs:
                input_depth.append(load_depthmap(fp))

        sample = {
                'input': input_imgs,
                'coarse': input_coarse,
                'sam': input_sam,
                'mask': mask,
                'heatmap_valid': np.array([tip_vis, anchor_vis], dtype=np.float32),
            }
        if self.prediction_task == 'keypoint_heatmap':
            sample['heatmap_valid'] = np.array([tip_vis, anchor_vis], dtype=np.float32)
       
        if self.add_depth_inputs:
            sample['input_depth'] = input_depth

        sample = self.transform(sample)
        sample['center_path'] = str(self.all_frames[center])
        sample['video_id'] = str(tgt_root)  
        sample['frame_id'] = int(local_j)  
        return sample
