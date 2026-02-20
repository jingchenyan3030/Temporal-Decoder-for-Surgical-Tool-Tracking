import torch
import torch.nn.functional as F
from torch import nn

class PostProcessPointsFixed(nn.Module):
    def __init__(self, H=256, W=320, vis_thresh=0.5, cls_thresh=0.0, no_object_id=2):
        super().__init__()
        self.H = H
        self.W = W
        self.vis_thresh = vis_thresh
        self.cls_thresh = cls_thresh
        self.no_object_id = no_object_id

    @torch.no_grad()
    def forward(self, outputs):
        logits = outputs["pred_logits"]          # (B,Q,3)
        pts01  = outputs["pred_points"]          # (B,Q,2) in [0,1]
        vislog = outputs["pred_visibility"]      # (B,Q) logits

        prob = logits.softmax(-1)                # (B,Q,3)
        visp = vislog.sigmoid()                  # (B,Q)

        B, Q, _ = prob.shape
        results = []

        for b in range(B):
            items = []
            for q in range(Q):
                v = float(visp[b, q].item())
                if v < self.vis_thresh:
                    continue

                # only pick foreground classes (tip/anchor), exclude no_object explicitly
                scores_fg, labels_fg = prob[b, q, :-1].max(dim=-1)  # ( ) scalar
                cls_id = int(labels_fg.item())      # 0 or 1
                cls_score = float(scores_fg.item())
                if cls_score < self.cls_thresh:
                    continue

                x = float(pts01[b, q, 0].item()) * (self.W - 1)
                y = float(pts01[b, q, 1].item()) * (self.H - 1)

                label = "tool_tip" if cls_id == 0 else "tool_anchor"
                items.append({
                    "label": label,
                    "x": x,
                    "y": y,
                    "visibility": True,
                    "score": cls_score,
                    "vis_prob": v,
                    "query": q,
                })

            results.append(items)

        return results