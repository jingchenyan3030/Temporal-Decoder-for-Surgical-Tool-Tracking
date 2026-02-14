# loss_detr_keypoint.py
import torch
import torch.nn.functional as F

'''
def aux_seg_loss(outputs, target_mask):
    seg_logits = outputs["aux_seg_logits"]  # (B, C, H, W)

    # target_mask: (B,H,W) long
    if target_mask.dim() == 4 and target_mask.size(1) == 1:
        target_mask = target_mask[:, 0]
    target_mask = target_mask.long()

    return F.cross_entropy(seg_logits, target_mask)
'''

import torch
import torch.nn.functional as F

def compute_detr_point_loss(outputs, targets, indices,
                            w_cls=1.0, w_point=1.0, w_vis=1.0):


    zero = (
    outputs["pred_logits"].sum()
    + outputs["pred_points"].sum()
    + outputs["pred_visibility"].sum()
) * 0.0


    loss_cls   = zero
    loss_point = zero
    loss_vis   = zero
    num = 0  # count valid matched batches (len(src_idx) > 0)

    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            continue

        pred_logits = outputs["pred_logits"][b, src_idx]               # (n, C)
        pred_pts    = outputs["pred_points"][b, src_idx]               # (n, 2)
        pred_vis    = outputs["pred_visibility"][b, src_idx].view(-1)  # (n,)

        tgt = targets[b]
        if "visibility" in tgt:
            keep = (tgt["visibility"] > 0.5)
            gt_labels_all = tgt["labels"][keep]
            gt_pts_all    = tgt["points"][keep]
            vis_all       = tgt["visibility"][keep].float()  # 这里基本全是1
        else:
            gt_labels_all = tgt["labels"]
            gt_pts_all    = tgt["points"]
            vis_all       = torch.ones_like(gt_labels_all, dtype=torch.float32)

        gt_labels = gt_labels_all[tgt_idx].long()
        gt_pts    = gt_pts_all[tgt_idx]
        vis       = vis_all[tgt_idx]


        # class (mask by vis, but don't skip; denom clamp avoids 0-div)
        ce = F.cross_entropy(pred_logits, gt_labels, reduction="none")  # (n,)
        denom = vis.sum().clamp(min=1.0)
        loss_cls = loss_cls + (ce * vis).sum() / denom

        # point (L1, mask by vis)
        l1 = F.l1_loss(pred_pts, gt_pts, reduction="none").sum(dim=1)   # (n,)
        denom = vis.sum().clamp(min=1.0)
        loss_point = loss_point + (l1 * vis).sum() / denom

        # visibility (always computed)
        loss_vis = loss_vis + F.binary_cross_entropy_with_logits(pred_vis, vis)

        num += 1

    if num > 0:
        loss_cls   = loss_cls / num
        loss_point = loss_point / num
        loss_vis   = loss_vis / num


    total = w_cls * loss_cls + w_point * loss_point + w_vis * loss_vis

    return {
        "loss_total": total,
        "loss_cls": loss_cls,
        "loss_point": loss_point,
        "loss_vis": loss_vis,
    }
