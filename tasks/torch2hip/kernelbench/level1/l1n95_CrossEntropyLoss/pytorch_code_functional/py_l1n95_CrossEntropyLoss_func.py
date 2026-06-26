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
    # CrossEntropy; num_classes fixed, vary batch.
    for b in [4096, 8192, 16384, 32768]:
        yield [torch.rand(b, num_classes), torch.randint(0, num_classes, (b,))]


def get_init_inputs():
    return []
