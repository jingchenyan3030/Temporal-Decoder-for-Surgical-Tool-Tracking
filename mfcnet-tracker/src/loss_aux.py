import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 

def get_loss(outputs, targets, loss_fns, loss_wts, args,
             sam_weight=None, heatmap_valid=None,
             aux_mask_target=None, aux_mask_weight=0.05):
    loss_dict = {} 
    total_loss = 0.0
    if isinstance(outputs, dict):
        heatmap_outputs = outputs['heatmap']
        aux_mask_outputs = outputs.get('aux_mask', None)
    else:
        heatmap_outputs = outputs
        aux_mask_outputs = None
    # main loss
    for loss_fn, loss_wt in zip(loss_fns, loss_wts):
        if loss_fn == 'mse':
            mse_loss = LossMSE()
            loss = mse_loss(heatmap_outputs, targets, sam=sam_weight, heatmap_valid=heatmap_valid)
        elif loss_fn == 'nll':
            loss = LossNLL(class_weights=args.class_weights, num_classes=args.num_classes)(heatmap_outputs, targets)
        elif loss_fn == 'soft_jaccard':
            loss = LossSoftJaccard(num_classes=args.num_classes)(heatmap_outputs, targets)
        else: 
            raise ValueError(f'Loss function {loss_fn} not implemented')
        total_loss += loss_wt * loss
        loss_dict['loss_' + loss_fn] = loss.item()
    aux_loss = torch.tensor(0.0, device= heatmap_outputs.device)
    if aux_mask_outputs is not None and aux_mask_target is not None:
        if aux_mask_target.dim() == 3:
            aux_mask_target = aux_mask_target.unsqueeze(1)   # [B,1,H,W]

        aux_mask_target = aux_mask_target.float().to(aux_mask_outputs.device)

        # ignore all-zero masks
        valid_mask = (aux_mask_target.flatten(1).sum(dim=1) > 0)   # [B]

        if valid_mask.any():
            pred_valid = aux_mask_outputs[valid_mask]      # [B,1,H,W]
            target_valid = aux_mask_target[valid_mask]     # [B,1,H,W]

            aux_loss = F.binary_cross_entropy_with_logits(
                pred_valid,
                target_valid
            )

            total_loss += aux_mask_weight * aux_loss

    loss_dict['loss_aux_mask'] = aux_loss.item()
    loss_dict['loss_total'] = total_loss.item()

    return total_loss, loss_dict


import torch
import torch.nn as nn
class LossMSE:
    def __init__(self, beta=1.0, eps=1e-6):
        self.beta = beta
        self.eps = eps

    def __call__(self, outputs, targets, sam=None, heatmap_valid=None):
        diff = (outputs - targets) ** 2   # [B, C, H, W]

        # spatial weighting (optional)
        if sam is not None:
            if sam.dim() == 3:
                sam = sam.unsqueeze(1)   # [B,1,H,W]
            sam = sam.float().to(outputs.device)
            sam = (sam > 0).float()
            weight = 1 + self.beta * sam
            diff = diff * weight

        # [B, C]
        per_channel_loss = diff.mean(dim=(2, 3))

        # no visibility mask
        if heatmap_valid is None:
            return per_channel_loss.mean()

        # [B, C]
        heatmap_valid = heatmap_valid.float().to(outputs.device)

        loss = (per_channel_loss * heatmap_valid).sum() / (heatmap_valid.sum() + self.eps)
        return loss
    
class LossNLL: 
    def __init__(self, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32))
            nll_weight = nll_weight.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.num_classes = num_classes
    
    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        return loss

class LossSoftJaccard:
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.eps = 1e-15  # Small constant to avoid division by zero

    def __call__(self, outputs, targets):
        loss = 0.0  # Initialize total loss
        for cls in range(1, self.num_classes):  # Exclude background class
            # Create binary masks for the current class
            jaccard_target = (targets == cls).float()
            jaccard_output = outputs[:, cls].exp()  # Assuming outputs are logits, use exp to get probabilities
            # Compute intersection and union
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum() - intersection
            # Compute Jaccard loss for the current class and accumulate
            jaccard_loss = -torch.log((intersection + self.eps) / (union + self.eps))
            loss += jaccard_loss
        # Return the average loss over all classes
        return loss / self.num_classes

class LossWassersteinDistance(nn.Module):  # Inherit from nn.Module to use register_buffer
    def __init__(self, num_classes, image_size, normalize=True):
        super(LossWassersteinDistance, self).__init__()
        self.num_classes = num_classes
        self.eps = 1e-15  # Small constant to avoid division by zero
        self.normalize = normalize
        self.image_size = image_size  # Fixed image size (height, width)

        # Compute the cost matrix based on the fixed image size and register it as a buffer
        cost_matrix = self.compute_cost_matrix(*image_size)
        self.register_buffer('cost_matrix', cost_matrix)  # Register the cost matrix as a buffer

    def compute_cost_matrix(self, height, width):
        """
        Computes the cost matrix, which is the pairwise distance between each pixel location.
        This matrix represents the cost of transporting mass between pixels.
        """
        # Create a grid of coordinates for each pixel in the mask
        x = torch.arange(width).float()
        y = torch.arange(height).float()
        X, Y = torch.meshgrid(x, y)

        # Flatten the grid and calculate pairwise Euclidean distances
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        dist_matrix = torch.cdist(coords, coords, p=2)  # p=2 for Euclidean distance
        return dist_matrix

    def forward(self, outputs, targets):
        """
        Compute the Wasserstein distance loss between predicted and target masks.
        """
        loss = 0.0  # Initialize total loss

        batch_size, _, height, width = outputs.size()

        # Use the stored cost matrix buffer
        cost_matrix = self.cost_matrix

        for cls in range(self.num_classes):
            # Extract soft predictions (probabilities) and create binary target masks for the current class
            target_mask = (targets == cls).float().view(batch_size, -1)  # Flatten the target mask
            pred_mask = outputs[:, cls].exp().view(batch_size, -1)  # Flatten the predicted mask (assume logits passed)

            # Normalize the masks to sum to 1 (valid probability distributions)
            if self.normalize:
                target_mask = target_mask / (target_mask.sum(dim=1, keepdim=True) + self.eps)
                pred_mask = pred_mask / (pred_mask.sum(dim=1, keepdim=True) + self.eps)

            # Compute Wasserstein distance (using the cost matrix and mass transport)
            wasserstein_distance = torch.sum(cost_matrix * (target_mask - pred_mask).abs(), dim=[1, 2])

            # Accumulate the Wasserstein distance for all classes
            loss += wasserstein_distance.mean()  # Average over batch

        return loss / self.num_classes  # Return average loss over classes
