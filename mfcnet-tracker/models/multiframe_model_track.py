# written by Chenyan
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .ternausnet import TernausNet11, TernausNet16
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from segmentation_models_pytorch import Segformer
from hrnet import HighResolutionNet
from detr_decoder import DETRDecoder, DETRDecoderLayer

import torch
import torch.nn as nn


# position encoding module

class PositionEmbeddingSine(nn.Module):
    """
    Standard 2D sine-cos positional encoding (DETR-style).
    Output: (B, C, H, W) where C = 2*num_pos_feats*2? (actually 2*num_pos_feats*2 if both x/y)
    Here C = 2 * num_pos_feats * 2? No: it becomes 2*num_pos_feats*2?? Let's be explicit below.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x):
        # x: (B, C, H, W)
        B, _, H, W = x.shape
        device = x.device

        mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)  # no padding mask
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B,H,W,F)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        # (B,H,W,2F) + (B,H,W,2F) -> (B,H,W,4F) ??? actually each becomes F, so concat -> 2F
        pos = torch.cat((pos_y, pos_x), dim=3)  # (B,H,W,2*num_pos_feats)
        pos = pos.permute(0, 3, 1, 2).contiguous()  # (B, 2*num_pos_feats, H, W)
        return pos






class MultiFrameNetBase(nn.Module):
    def __init__(self, num_classes, num_frames, has_base_perframe_model_trained=False, with_optflow=False, with_depth=False):
        super(MultiFrameNetBase, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.with_optflow = with_optflow
        self.with_depth = with_depth

        # self.in_channels = self.num_frames * self.num_classes
        if has_base_perframe_model_trained:
            self.in_channels = self.num_frames * self.num_classes
        else:
            self.in_channels = 1 * self.num_frames * self.num_classes

        if self.with_optflow:
            self.in_channels += 2 * (self.num_frames - 1)

        if self.with_depth:
            self.in_channels += 1 * self.num_frames

    def forward(self, x):
        raise NotImplementedError("This is a base class. Use MultiFrameNetBasic or MultiFrameNetLarge.")

# class MultiFrameNetBasic(MultiFrameNetBase):
#     def __init__(self, num_classes, num_frames, has_base_perframe_model_trained=False, with_optflow=False, with_depth=False):
#         super(MultiFrameNetBasic, self).__init__(num_classes, num_frames, has_base_perframe_model_trained, with_optflow, with_depth)

#         self.multiframe_net = nn.Sequential(
#             nn.Conv2d(self.in_channels, self.num_frames * self.num_classes, kernel_size=11, stride=1, padding=5, bias=False),
#             nn.BatchNorm2d(self.num_frames * self.num_classes), 
#             nn.ReLU(),
#             nn.Conv2d(self.num_frames * self.num_classes, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False),
#         )

#     def forward(self, x):
#         return self.multiframe_net(x)

class MultiFrameNetBasic(MultiFrameNetBase):
    def __init__(self, num_classes, num_frames, has_base_perframe_model_trained=False, with_optflow=False, with_depth=False):
        super(MultiFrameNetBasic, self).__init__(num_classes, num_frames, has_base_perframe_model_trained, with_optflow, with_depth)
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.with_optflow = with_optflow
        self.with_depth = with_depth
        
        self.multiframe_net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_frames * self.num_classes, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes), 
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_frames * self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes),
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_frames * self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes),
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # self.multiframe_net = nn.Sequential(
        #     nn.Conv2d(self.in_channels, self.num_frames * self.num_classes, kernel_size=11, stride=1, padding=5, bias=False),
        #     nn.BatchNorm2d(self.num_frames * self.num_classes), 
        #     nn.ReLU(),
        #     nn.Conv2d(self.num_frames * self.num_classes, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        # )

        # Register the mesh grid as a buffer
        self.register_buffer('grid', self._create_mesh_grid())

    def forward(self, x):
        if self.with_optflow:
            x = self.warp_segmentation_and_depth(x)
        return self.multiframe_net(x)

    def warp_segmentation_and_depth(self, x):
        """
        Warp segmentation maps and depth maps according to the optical flow.

        Args:
            x (torch.Tensor): Input tensor of shape (B, NK + 2K - 2 + K, H, W), where the first NK channels are segmentation maps,
                             the next 2K-2 channels are optical flow maps, and the last K channels are depth maps.

        Returns:
            torch.Tensor: Warped segmentation and depth maps of shape (B, NK + K, H, W).
        """
        N = self.num_classes
        K = self.num_frames
        
        # Split the input into segmentation maps, optical flow maps, and depth maps
        segmentation = x[:, :N*K, :, :]
        flow = x[:, N*K:N*K + 2*K - 2, :, :]
        depth = x[:, N*K + 2*K - 2:, :, :] if self.with_depth else None

        # Warp each segmentation map and depth map using the corresponding optical flow
        warped_segmentations = []
        warped_depths = []
        for i in range(1,K):
            flow_i = flow[:, 2*(i-1):2*i, :, :]
            for j in range(N):
                segmentation_ij = segmentation[:, i*N + j:i*N + j + 1, :, :]
                warped_segmentation_ij = self._warp_single_map(segmentation_ij, flow_i)
                warped_segmentations.append(warped_segmentation_ij)

            if self.with_depth:
                depth_i = depth[:, i:i+1, :, :]
                warped_depth_i = self._warp_single_map(depth_i, flow_i)
                warped_depths.append(warped_depth_i)
        
        # Append the first frame segmentation map without warping
        warped_segmentations.insert(0, segmentation[:, 0:N, :, :])
        
        # Append the first frame depth map without warping
        if self.with_depth:
            depth_0 = depth[:, 0:1, :, :]
            warped_depths.insert(0, depth_0)

        # Concatenate warped segmentation maps along the channel dimension
        warped_segmentation = torch.cat(warped_segmentations, dim=1)

        if self.with_depth:
            # Concatenate warped depth maps along the channel dimension
            warped_depth = torch.cat(warped_depths, dim=1)
            return torch.cat((warped_segmentation, warped_depth), dim=1)
        else:
            return warped_segmentation

    def _warp_single_map(self, map, flow):
        """
        Warp a map (segmentation or depth) according to the RAFT optical flow output.

        Args:
            map (torch.Tensor): Map of shape (B, 1, H, W).
            flow (torch.Tensor): Optical flow map of shape (B, 2, H, W), where flow[:, 0, :, :] is the x-component
                                 and flow[:, 1, :, :] is the y-component of the flow.

        Returns:
            torch.Tensor: Warped map of shape (B, 1, H, W).
        """
        _, _, H, W = map.size()

        # Use the precomputed mesh grid (normalized to [-1, 1] for grid_sample)
        grid = self.grid[:, :, :H, :W]  # Shape: (1, 2, H, W)

        # Add the flow to the grid using broadcasting instead of repeating the grid
        flow_x = flow[:, 0, :, :] / ((W - 1) / 2.0)
        flow_y = flow[:, 1, :, :] / ((H - 1) / 2.0)
        flow = torch.stack((flow_x, flow_y), dim=1)
        new_grid = grid + flow  # Shape: (B, 2, H, W)

        # Permute grid to (B, H, W, 2) for grid_sample
        new_grid = new_grid.permute(0, 2, 3, 1)

        # Warp the map using the computed flow grid
        warped_map = F.grid_sample(map, new_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_map

    def _create_mesh_grid(self):
        """
        Create a mesh grid for the image dimensions, normalized to [-1, 1] for use in grid_sample.

        Returns:
            torch.Tensor: Mesh grid of shape (1, 2, H, W).
        """
        H, W = 256, 320  # Default size, will be cropped/resized as needed
        y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid_y = 2.0 * y / (H - 1) - 1.0
        grid_x = 2.0 * x / (W - 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # Shape: (2, H, W)
        grid = grid.unsqueeze(0)  # Shape: (1, 2, H, W)
        return grid

class MultiFrameNetLarge(MultiFrameNetBase):
    def __init__(self, num_classes, num_frames, has_base_perframe_model_trained=False, with_optflow=False, with_depth=False):
        super(MultiFrameNetLarge, self).__init__(num_classes, num_frames, has_base_perframe_model_trained, with_optflow, with_depth)

        self.feature_net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_frames * self.num_classes, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes),
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_frames * self.num_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes),
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_frames * self.num_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes),
            nn.ReLU(),
        )

        self.head = nn.Conv2d(self.num_frames * self.num_classes, self.num_classes, kernel_size=1, bias=False)


    def forward(self, x, return_feat=False, return_seg=False):
        if self.with_optflow:
            x = self.warp_segmentation_and_depth(x)

        feat = self.feature_net(x)

        seg = None
        if return_seg:
            seg = self.head(feat)

        if return_feat and return_seg:
            return feat, seg
        if return_seg:
            return seg
        if return_feat:
            return feat

        return self.head(feat)

class TernausNetMultiBasic(nn.Module):
    def __init__(self, num_classes, num_frames, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(TernausNetMultiBasic, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        if loadpath is not None:
            self.base_model = TernausNet16(num_classes=self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = True
        else:
            self.base_model = TernausNet16(num_classes=1*self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained,
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)
    
    def forward(self, x, optflow=None, depth=None):
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        feat = self.multiframe_net(y_output, return_feat=True)
        out  = self.multiframe_net.head(feat)   
        return out, feat  # feat: (B, C_feat, H, W), C_feat = N_f * N_c: num_frames * num_classes = 8 * 2 = 16

class TernausNetMultiLarge(nn.Module):
    def __init__(self, num_classes, num_frames, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(TernausNetMultiLarge, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        if loadpath is not None:
            self.base_model = TernausNet16(num_classes=self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = True
        else:
            self.base_model = TernausNet16(num_classes=1*self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained,
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)
    
    def forward(self, x, optflow=None, depth=None):
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


# DETR-based multi-frame segmentation models
class TernusNetMultiDETRBasic(nn.Module):
    def __init__(self, num_classes, num_frames, d_model=256, num_queries=3, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(TernusNetMultiDETRBasic, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.tgt_embed = nn.Embedding(num_queries, d_model)
        if loadpath is not None:
            self.base_model = TernausNet16(num_classes=self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = True
        else:
            self.base_model = TernausNet16(num_classes=1*self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained,
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)
        # proj
        C_feat = self.num_frames * self.num_classes
        self.proj = nn.Conv2d(C_feat, d_model, kernel_size=1)
        # DETR decoder
        self.query_embed = nn.Embedding(num_queries, d_model)
        decoder_layer = DETRDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.decoder = DETRDecoder(
            decoder_layer=decoder_layer,
            num_layers=6,
            norm=nn.LayerNorm(d_model),
            return_intermediate=False
        )
  
        # point regression head
        self.point_head = nn.Linear(d_model, 2)  # x, y
        self.num_obj_classes = 2  # anchor, tip
        self.class_head = nn.Linear(d_model, self.num_obj_classes + 1)
        self.visibility_head = nn.Linear(d_model,1)  # visible or not
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
        for p in self.multiframe_net.head.parameters():
            p.requires_grad_(False)

    def forward(self, x, optflow=None, depth=None, prev_hs = None):
        y_output = []
        for x_img in x:
            y_img = self.base_model(x_img)
            y_output.append(y_img)

        if optflow is not None:
            y_output.extend(optflow)
        if depth is not None:
            y_output.extend(depth)

        y_output = torch.cat(y_output, dim=1)  # (B, C_feat, H, W)
        # 2. multi-frame fusion feature

        feat = self.multiframe_net(y_output, return_feat=True, return_seg=False) # feat: (B, C_feat, H, W)
        # 3. proj to d_model
        feat = self.proj(feat)                 # (B, d_model, H, W)
        # 4. flatten to DETR memory
        B, C, H, W = feat.shape
        memory = feat.flatten(2).permute(2, 0, 1)  # (HW, B, d_model)
        # 5. queries
        pos = self.pos_embed(feat)                         # (B, d_model, H, W)
        pos = pos.flatten(2).permute(2, 0, 1)              # (HW, B, d_model)
        
        # 6. use prev_hs to initialize the decoder queries for temporal consistency\
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        if prev_hs is None:
            # first frame use learnable init or zeros
            if hasattr(self, "tgt_embed"):
                tgt = self.tgt_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (Q,B,D)
            else:
                tgt = torch.zeros_like(query_pos)  # (Q,B,D)
        else:
            tgt = prev_hs.permute(1, 0, 2).contiguous()  # (Q, B, d_model)

        # 7. DETR decoder
        hs = self.decoder(tgt, memory, query_pos=query_pos, pos=pos)  # (Q, B, d_model)
        hs = hs.permute(1, 0, 2).contiguous()                            # (B, Q, d_model)

        # resize
        H_in, W_in = x[0].shape[-2], x[0].shape[-1]
        pred_points = self.point_head(hs).sigmoid()
        pred_visibility = self.visibility_head(hs).squeeze(-1)  # raw logits (B,Q)


        out = {
            "pred_logits": self.class_head(hs),  # (B, Q, 3)
            "pred_points": pred_points,  # (B, Q, 2)
            "pred_visibility": pred_visibility,  # (B, Q)
        }
        return out, hs  # hs: (B, Q, d_model)




class DeepLabMultiBasic(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(DeepLabMultiBasic, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = models.segmentation.deeplabv3_resnet101(pretrained=self.pretrained, progress=True)
        if loadpath is not None: 
            self.base_model.classifier = DeepLabHead(2048, self.num_classes)
            has_base_preframe_model_trained = True
        else:
            self.base_model.classifier = DeepLabHead(2048, 1*self.num_classes)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetBasic(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)

    def forward(self, x, optflow=None, depth=None): 
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)['out']
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output

class DeepLabMultiLarge(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(DeepLabMultiLarge, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = models.segmentation.deeplabv3_resnet101(pretrained=self.pretrained, progress=True)
        if loadpath is not None: 
            self.base_model.classifier = DeepLabHead(2048, self.num_classes)
            has_base_preframe_model_trained = True
        else:
            self.base_model.classifier = DeepLabHead(2048, 1*self.num_classes)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)

    def forward(self, x, optflow=None, depth=None): 
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)['out']
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class SegFormerMultiBasic(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(SegFormerMultiBasic, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = Segformer(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=self.num_classes, activation='logsoftmax')
        if loadpath is not None: 
            has_base_preframe_model_trained = True
        else:
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetBasic(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)

    def forward(self, x, optflow=None, depth=None): 
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class SegFormerMultiLarge(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(SegFormerMultiLarge, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = Segformer(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=self.num_classes, activation='logsoftmax')
        if loadpath is not None: 
            has_base_preframe_model_trained = True
        else:
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)

    def forward(self, x, optflow=None, depth=None): 
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class HRNetMultiBasic(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(HRNetMultiBasic, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = HighResolutionNet(num_classes=self.num_classes)
        if loadpath is not None:
            has_base_preframe_model_trained = True
        else:
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetBasic(self.num_classes, self.num_frames, has_base_preframe_model_trained,
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)
    
    def forward(self, x, optflow=None, depth=None):
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class HRNetMultiLarge(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(HRNetMultiLarge, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = HighResolutionNet(num_classes=self.num_classes)
        if loadpath is not None:
            has_base_preframe_model_trained = True
        else:
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained,
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)
    
    def forward(self, x, optflow=None, depth=None):
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class FCNMultiBasic(nn.Module): 
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(FCNMultiBasic, self).__init__()
        self.num_classes = num_classes 
        self.num_frames = num_frames 
        self.pretrained = pretrained 
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = models.segmentation.fcn_resnet101(pretrained=self.pretrained, progress=True) 
        if loadpath is not None: 
            self.base_model.classifier = FCNHead(2048, self.num_classes)
            has_base_preframe_model_trained = True
        else:
            self.base_model.classifier = FCNHead(2048, 1*self.num_classes)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetBasic(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs) 
    
    def forward(self, x, optflow=None, depth=None):
        y_output = [] 
        for x_img in x: 
            y_img = self.base_model(x_img)['out'] 
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output

class FCNMultiLarge(nn.Module): 
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(FCNMultiLarge, self).__init__()
        self.num_classes = num_classes 
        self.num_frames = num_frames 
        self.pretrained = pretrained 
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = models.segmentation.fcn_resnet101(pretrained=self.pretrained, progress=True) 
        if loadpath is not None: 
            self.base_model.classifier = FCNHead(2048, self.num_classes)
            has_base_preframe_model_trained = True
        else:
            self.base_model.classifier = FCNHead(2048, 1*self.num_classes)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs) 
    
    def forward(self, x, optflow=None, depth=None):
        y_output = [] 
        for x_img in x: 
            y_img = self.base_model(x_img)['out'] 
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output
