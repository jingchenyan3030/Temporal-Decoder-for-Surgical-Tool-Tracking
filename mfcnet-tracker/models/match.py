# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_point: This is the relative weight of the L1 error of the point coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0 , "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 "pred_points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "points": Tensor of dim [num_target_boxes, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # New Adding visibility prediction
        tgt_ids_list = []
        tgt_pts_list = []
        sizes = []

        for v in targets:
            if "valid" in v:
                keep = v["valid"]
            else:
                keep = (v["labels"] >= 0)

            if "visibility" in v:
                keep = keep & (v["visibility"] > 0.5)

            tgt_ids_list.append(v["labels"][keep])
            tgt_pts_list.append(v["points"][keep])
            sizes.append(int(keep.sum().item()))

        # End
        if sum (sizes) == 0:
            # If no visible points in batch, return empty matches
            return [(torch.empty(0, dtype=torch.int64),
             torch.empty(0, dtype=torch.int64)) for _ in range(bs)]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat(tgt_ids_list)
        tgt_points = torch.cat(tgt_pts_list)

        if not torch.isfinite(outputs["pred_logits"]).all():
            raise ValueError("Pred has NaN/Inf in pred_logits")
        if not torch.isfinite(tgt_points).all():
            raise ValueError("GT has NaN/Inf in tgt_points")
        if not torch.isfinite(out_points).all():
            raise ValueError("Pred has NaN/Inf in out_points")
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between points
        cost_point = torch.cdist(out_points, tgt_points, p=1)

        # Final cost matrix
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()
        if not torch.isfinite(C).all():
            raise ValueError("Cost matrix C has NaN/Inf")

        Cs = list(C.split(sizes, -1))
        indices = []
        for b, c in enumerate(Cs):
            if sizes[b] == 0:
                indices.append((torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64)))
            else:
                i, j = linear_sum_assignment(c[b])
                indices.append((torch.as_tensor(i, dtype=torch.int64),
                                torch.as_tensor(j, dtype=torch.int64)))

        return indices



def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_point=args.set_cost_point)