# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL starter for this torch2flydsl task.

Implement the target entry points below using FlyDSL. ``model.py`` contains the
PyTorch reference/specification and ``test_kernel_harness.py`` contains the
correctness and performance checks. These stubs intentionally do not call the
reference implementation, so an unimplemented task cannot pass validation.
"""

def flydsl_layernorm2d_with_add(*args, **kwargs):
    raise NotImplementedError("Implement flydsl_layernorm2d_with_add using FlyDSL for the layernorm2d_with_add_kernel task.")


def build_layernorm2d_with_add_module(*args, **kwargs):
    raise NotImplementedError("Implement build_layernorm2d_with_add_module using FlyDSL for the layernorm2d_with_add_kernel task.")
