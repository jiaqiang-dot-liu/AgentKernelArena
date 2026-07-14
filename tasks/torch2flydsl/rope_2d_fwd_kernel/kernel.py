# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL starter for this torch2flydsl task.

Implement the target entry points below using FlyDSL. ``model.py`` contains the
PyTorch reference/specification and ``test_kernel_harness.py`` contains the
correctness and performance checks. These stubs intentionally do not call the
reference implementation, so an unimplemented task cannot pass validation.
"""

def flydsl_rope_2d_fwd(*args, **kwargs):
    raise NotImplementedError("Implement flydsl_rope_2d_fwd using FlyDSL for the rope_2d_fwd_kernel task.")


def build_rope_2d_fwd_module(*args, **kwargs):
    raise NotImplementedError("Implement build_rope_2d_fwd_module using FlyDSL for the rope_2d_fwd_kernel task.")
