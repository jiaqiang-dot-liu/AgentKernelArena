# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
    return x / rms


class RMSNorm_(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(RMSNorm_, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x, fn=module_fn):
        return fn(x, self.eps)


batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    # RMSNorm; `features` is fixed by get_init_inputs, vary batch/spatial.
    for b, d1, d2 in [(16, 256, 256), (32, 128, 128), (64, 256, 128), (112, 128, 128)]:
        yield [torch.rand(b, features, d1, d2)]


def get_init_inputs():
    return [features]
