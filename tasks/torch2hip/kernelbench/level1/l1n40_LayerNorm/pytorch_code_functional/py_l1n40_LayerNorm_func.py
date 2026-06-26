# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              normalized_shape, eps: float) -> torch.Tensor:
    return F.layer_norm(x, normalized_shape, weight, bias, eps)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x, fn=module_fn):
        return fn(x, self.ln.weight, self.ln.bias, self.ln.normalized_shape, self.ln.eps)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    # LayerNorm normalized_shape=(features,dim1,dim2) is fixed; vary batch only.
    for b in [4, 8, 16, 32]:
        yield [torch.rand(b, features, dim1, dim2)]


def get_init_inputs():
    return [(features, dim1, dim2)]
