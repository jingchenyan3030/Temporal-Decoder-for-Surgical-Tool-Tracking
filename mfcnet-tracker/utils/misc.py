# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
'''
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if float(torchvision.__version__[:3]) < 0.5:
    import math
    from torchvision.ops.misc import _NewEmptyTensorOp
    def _check_size_scale_factor(dim, size, scale_factor):
        # type: (int, Optional[List[int]], Optional[float]) -> None
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if not (scale_factor is not None and len(scale_factor) != dim):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )
    def _output_size(dim, input, size, scale_factor):
        # type: (int, Tensor, Optional[List[int]], Optional[float]) -> List[int]
        assert dim == 2
        _check_size_scale_factor(dim, size, scale_factor)
        if size is not None:
            return size
        # if dim is not 2 or scale_factor is iterable use _ntuple instead of concat
        assert scale_factor is not None and isinstance(scale_factor, (int, float))
        scale_factors = [scale_factor, scale_factor]
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]
elif float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size
'''


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
