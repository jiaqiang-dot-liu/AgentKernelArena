# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x, weight, bias):
    x = F.linear(x, weight, bias)
    x = torch.softmax(x, dim=1)
    return x


class Matmul_Dropout_Softmax(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, fn=module_fn):
        return fn(x, self.matmul.weight, self.matmul.bias)


batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    # in_features fixed, escalate batch (dropout disabled in eval()).
    for b in [64, 128, 256, 512, 1024]:
        yield [torch.rand(b, in_features)]


def get_init_inputs():
    return [in_features, out_features, dropout_p]
