# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(predictions, targets)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)


batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    # No init -> both batch and num_classes are free; escalate both (gpumode CE varies C).
    for b, c in [(8192, 1024), (16384, 2048), (32768, 4096), (16384, 8192), (8192, 16384)]:
        yield [torch.rand(b, c), torch.randint(0, c, (b,))]


def get_init_inputs():
    return []
