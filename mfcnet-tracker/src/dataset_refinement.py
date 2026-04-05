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

sys.path.append('../utils/')
from utils.dataloader_utils import load_image

def get_video_root(img_path):
    parts = img_path.parts
    idx = parts.index("images")
    return Path(*parts[:idx])

class KPT_refine(Dataset):
    def __init__(
            self,
            file_names,
            transform,
            mode,
            prediction_task,
            add_depth_inputs=False,
            use_mask=True
    ):
        self.file_names = natsorted(file_names)
        self.transform = transform
        self.mode = mode
        self.prediction_task = prediction_task
        assert prediction_task == 'keypoint_heatmap'
        self.add_depth_inputs = add_depth_inputs
        self.use_mask = use_mask

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_t = self.file_names[idx]
        tgt_root = get_video_root(img_t)
        local_j = idx

        if self.prediction_task != 'keypoint_heatmap':
            raise ValueError(f"Unsupported prediction_task: {self.prediction_task}")

        coarse_dir = tgt_root / "heatmap_coarse"
        mask_dir = tgt_root / "mask_coarse"
        gt_dir = tgt_root / "sam_results"

        coarse_path = coarse_dir / (img_t.stem + ".npy")

        coarse = np.load(str(coarse_path)).astype(np.float32)
        if coarse.ndim == 2:
            raise ValueError(f"Coarse heatmap must have 2 channels at {coarse_path}")
        if coarse.ndim != 3:
            raise ValueError(f"Unexpected coarse shape {coarse.shape} at {coarse_path}")

        if coarse.shape[0] == 2 and coarse.shape[1] != 2:
            coarse = np.transpose(coarse, (1, 2, 0))   # (H,W,2)

        if coarse.shape[2] != 2:
            raise ValueError(f"Coarse last dim must be 2. Got {coarse.shape} at {coarse_path}")
        
        gt = load_mask(img_t, self.prediction_task).astype(np.float32)   # (H,W,2)

        if self.use_mask:
            sam_path = mask_dir / (img_t.stem + ".npy")
            sam = np.load(str(sam_path)).astype(np.float32)

            if sam.ndim == 2:
                sam = sam[..., None]   # (H,W) -> (H,W,1)
            elif sam.ndim == 3 and sam.shape[0] == 1:
                sam = np.transpose(sam, (1, 2, 0))   # (1,H,W) -> (H,W,1)
            elif sam.ndim == 3 and sam.shape[2] == 1:
                pass   # already (H,W,1)
            else:
                raise ValueError(f"Unexpected sam shape {sam.shape} at {sam_path}")
        else:
            H, W = coarse.shape[:2]
            sam = np.zeros((H, W, 1), dtype=np.float32)

        detr_dir = tgt_root / 'points_detr'
        json_path = detr_dir / (img_t.stem + ".json")

        tip_vis = 1
        anchor_vis = 1
        if json_path.exists():
            with open(json_path, 'r') as f:
                plist = json.load(f)

            tip_flags = []
            anchor_vis = 0
            for p in plist:
                lab = p.get("label", "")
                vis = 1 if bool(p.get("visibility", True)) else 0
                if lab == "tool_tip":
                    tip_flags.append(vis)
                elif lab == "tool_anchor":
                    anchor_vis = vis
            tip_vis = 1 if any(tip_flags) else 0

        image = load_image(img_t)

        sample = {
            'input': image,
            'coarse': coarse,
            'mask': gt,
            'sam': sam,
            'heatmap_valid': np.array([tip_vis, anchor_vis], dtype=np.float32),
        }

        if self.add_depth_inputs:
            sample['input_depth'] = load_depthmap(img_t)

        if self.transform is not None:
            sample = self.transform(sample)

        sample['center_path'] = str(img_t)
        sample['video_id'] = str(tgt_root)
        sample['frame_id'] = img_t.stem
        return sample


                

    