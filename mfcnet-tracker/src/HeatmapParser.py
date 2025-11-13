import torch
import cv2
import numpy as np

class HeatmapParser:
    def __init__(self, channel_configs, nms_kernel = 7, nms_padding = 1):
        # channel_configs is dict: {"anchor":{"topk":5, "max_keep": 2, threshold":0.1}, {"tip":{"topk":5, "max_keep": 2, threshold":0.1}
        # contact:{"max_keep": 4, threshold":0.3}}
        self.channel_configs = channel_configs
        self.pool = torch.nn.MaxPool2d(nms_kernel, stride=1, padding=nms_padding)

    def nms(self, heatmap):
        # heatmap: (B, C, H, W)
        heatmap_max = self.pool(heatmap)
        keep = (heatmap_max == heatmap).float()
        return heatmap * keep
    
    def top_k(self, heatmap, k):
        # heatmap: (B, C, H, W)
        heatmap = self.nms(heatmap)
        B, C, H, W = heatmap.shape
        heatmap_reshaped = heatmap.view(B, C, -1)
        topk_scores, topk_indices = torch.topk(heatmap_reshaped, k)
        topk_ys = (topk_indices // W).float()
        topk_xs = (topk_indices % W).float()
        return topk_xs, topk_ys, topk_scores
    
    def adjust(self, x, y, heatmap):
        x_adj, y_adj = [], []
        hmap = heatmap.cpu().numpy()
        for xi, yi in zip(x.cpu().numpy(), y.cpu().numpy()):
            xi_f, yi_f = float(xi), float(yi)
            xi_int, yi_int = int(round(xi_f)), int(round(yi_f))
            
            if hmap[min(yi_int+1, hmap.shape[0]-1), xi_int] > hmap[max(yi_int-1, 0), xi_int]:
                yi_f += 0.25
            else:
                yi_f -= 0.25
            yi_int = int(round(yi_f))
            
            if hmap[yi_int, min(xi_int+1, hmap.shape[1]-1)] > hmap[yi_int, max(xi_int-1, 0)]:
                xi_f += 0.25
            else:
                xi_f -= 0.25
            x_adj.append(xi_f + 0.5)
            y_adj.append(yi_f + 0.5)
        return np.array(x_adj), np.array(y_adj)

    
    def parse(self, det):
        """
        det: torch.Tensor, shape [B, C, H, W]
        return: dict, 每个 channel 一个 keypoints 数组 (N, 3) → [x, y, val]
        """
        B, C, H, W = det.shape
        results = {cfg["name"]: [] for cfg in self.channel_configs.values()}

        for c, cfg in self.channel_configs.items():
            channel_det = det[:, c, :, :]
            name = cfg["name"]
            x, y, val = self.top_k(channel_det.unsqueeze(1), cfg["topk"])
            for b in range(B):
                vals = val[b,0]
                mask = vals > cfg["threshold"]
                x_b, y_b, val_b = x[b,0][mask], y[b,0][mask], val[b,0][mask]
                if len(x_b) > 0:
                    x_ref, y_ref = self.adjust(x_b, y_b, channel_det[b])
                    coords = np.stack([x_ref, y_ref, val_b.cpu().numpy()], axis=-1)
                    coords = coords[np.argsort(-coords[:, 2])][:cfg["max_keep"]]
                else:
                    coords = np.zeros((0, 3))
                results[name].append(coords)

        return results


        
