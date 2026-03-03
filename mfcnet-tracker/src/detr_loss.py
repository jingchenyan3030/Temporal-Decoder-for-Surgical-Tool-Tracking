# Chenyan
import torch
import torch.nn.functional as F

def compute_detr_point_loss(
    outputs,
    targets,
    indices,
    w_cls=1.0,
    w_point=1.0,
    w_vis=1.0,
    alpha=0.02,
    no_object=2,
):
    pred_logits    = outputs["pred_logits"]        # (B,Q,3)
    pred_points    = outputs["pred_points"]        # (B,Q,2) in [0,1]
    pred_vislogits = outputs["pred_visibility"]    # (B,Q) logits

    B, Q, C = pred_logits.shape
    device = pred_logits.device

    zero = (pred_logits.sum() + pred_points.sum() + pred_vislogits.sum()) * 0.0

    loss_cls_sum = zero
    loss_point_sum = zero
    loss_vis_matched_sum = zero
    loss_vis_unmatched_sum = zero

    n_matched_total = 0
    n_unmatched_total = 0

    for b in range(B):
        src_idx, tgt_idx = indices[b]

        # robust n_match
        if torch.is_tensor(src_idx):
            n_match = int(src_idx.numel())
        else:
            n_match = int(len(src_idx))

        # -------- build the SAME keep as matcher used --------
        tgt = targets[b]
        if "valid" in tgt:
            keep = tgt["valid"].to(torch.bool)
        else:
            keep = (tgt["labels"] >= 0)

        if "visibility" in tgt:
            keep = keep & (tgt["visibility"] > 0.5)

        # these are the arrays matcher indexed into
        gt_labels_kept = tgt["labels"][keep]            # (Ng,)
        gt_points_kept = tgt["points"][keep]            # (Ng,2)
        gt_vis_kept    = tgt["visibility"][keep].float() if "visibility" in tgt else None

        # ---- (A) classification for all queries ----
        target_classes = torch.full((Q,), no_object, dtype=torch.long, device=device)

        if n_match > 0:
            # IMPORTANT: use kept arrays, then index by tgt_idx
            gt_labels = gt_labels_kept[tgt_idx].long()      # (n_match,) in {0,1}
            target_classes[src_idx] = gt_labels

        loss_cls_sum = loss_cls_sum + F.cross_entropy(pred_logits[b], target_classes)

        # ---- (B) point regression only for matched ----
        if n_match > 0:
            gt_pts   = gt_points_kept[tgt_idx]              # (n,2) in [0,1]
            pred_pts = pred_points[b, src_idx]              # (n,2) in [0,1]
            l1 = F.l1_loss(pred_pts, gt_pts, reduction="none").sum(dim=1)  # (n,)
            loss_point_sum = loss_point_sum + l1.sum()
            n_matched_total += n_match

        # ---- (C) visibility: matched + unmatched (strict as your figure) ----
        if n_match > 0:
            if gt_vis_kept is not None:
                vis_tgt_m = gt_vis_kept[tgt_idx]            # (n,) usually all 1
            else:
                vis_tgt_m = torch.ones((n_match,), device=device)

            vis_pred_m = pred_vislogits[b, src_idx].view(-1)
            loss_vis_matched_sum = loss_vis_matched_sum + F.binary_cross_entropy_with_logits(
                vis_pred_m, vis_tgt_m, reduction="sum"
            )

        unmatched_mask = torch.ones((Q,), dtype=torch.bool, device=device)
        if n_match > 0:
            unmatched_mask[src_idx] = False

        vis_pred_u = pred_vislogits[b, unmatched_mask].view(-1)  # (#u,)
        if vis_pred_u.numel() > 0:
            vis_tgt_u = torch.zeros_like(vis_pred_u)
            loss_vis_unmatched_sum = loss_vis_unmatched_sum + F.binary_cross_entropy_with_logits(
                vis_pred_u, vis_tgt_u, reduction="sum"
            )
            n_unmatched_total += int(vis_pred_u.numel())

    # ---- normalize ----
    loss_cls = loss_cls_sum / max(B, 1)

    if n_matched_total > 0:
        loss_point = loss_point_sum / float(n_matched_total)
        loss_vis_matched = loss_vis_matched_sum / float(n_matched_total)
    else:
        loss_point = zero
        loss_vis_matched = zero

    if n_unmatched_total > 0:
        loss_vis_unmatched = loss_vis_unmatched_sum / float(n_unmatched_total)
    else:
        loss_vis_unmatched = zero

    loss_vis = loss_vis_matched + alpha * loss_vis_unmatched
    total = w_cls * loss_cls + w_point * loss_point + w_vis * loss_vis

    return {
        "loss_total": total,
        "loss_cls": loss_cls,
        "loss_point": loss_point,
        "loss_vis": loss_vis,
        "loss_vis_matched": loss_vis_matched,
        "loss_vis_unmatched": loss_vis_unmatched,
        "n_matched": torch.tensor(float(n_matched_total), device=device),
        "n_unmatched": torch.tensor(float(n_unmatched_total), device=device),
    }


import torch
import torch.nn.functional as F

def compute_tracking_loss(
    pred_points_t, gt_points_t, vis_t,
    prev_pred_points=None, prev_vis=None,
    prev_gt_points=None,              # optional, only for match_gt_vel
    alpha=0.01, eps=1e-6,
    vel_mode="pred_smooth"            # "pred_smooth" or "match_gt_vel"
):
    """
    pred_points_t: (B,Q,2)
    gt_points_t:   (B,Q,2)
    vis_t:         (B,Q) {0,1} or bool
    prev_pred_points: (B,Q,2) from t-1 (recommend detach before passing)
    prev_vis:         (B,Q)
    prev_gt_points:   (B,Q,2) from t-1 (needed only if vel_mode=="match_gt_vel")
    """

    # --- L_vis (single-frame) ---
    m = (vis_t > 0.5).float()  # (B,Q)
    per_q = F.smooth_l1_loss(pred_points_t, gt_points_t, reduction="none").sum(dim=-1)  # (B,Q)
    denom = m.sum()
    L_vis_t = (per_q * m).sum() / (denom + eps)

    # --- L_vel (pair-masked) ---
    L_vel_t = pred_points_t.sum() * 0.0
    if prev_pred_points is not None and prev_vis is not None:
        mp = ((vis_t > 0.5) & (prev_vis > 0.5)).float()  # (B,Q)
        denom_p = mp.sum()

        if denom_p.item() > 0:
            if vel_mode == "pred_smooth":
                dp = pred_points_t - prev_pred_points  # (B,Q,2)
                vel = torch.abs(dp).sum(dim=-1)        # (B,Q)

            elif vel_mode == "match_gt_vel":
                assert prev_gt_points is not None, "prev_gt_points required for match_gt_vel"
                pred_vel = pred_points_t - prev_pred_points
                gt_vel   = gt_points_t - prev_gt_points
                vel = torch.abs(pred_vel - gt_vel).sum(dim=-1)

            else:
                raise ValueError(f"Unknown vel_mode: {vel_mode}")

            L_vel_t = (vel * mp).sum() / (denom_p + eps)

    L_t = L_vis_t + alpha * L_vel_t
    return L_t, L_vis_t, L_vel_t