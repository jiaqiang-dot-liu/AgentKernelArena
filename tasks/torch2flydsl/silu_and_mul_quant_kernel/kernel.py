# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL starter for this torch2flydsl task.

Implement the target entry points below using FlyDSL. ``model.py`` contains the
PyTorch reference/specification and ``test_kernel_harness.py`` contains the
correctness and performance checks. These stubs intentionally do not call the
reference implementation, so an unimplemented task cannot pass validation.
"""

def flydsl_silu_and_mul_quant(*args, **kwargs):
    raise NotImplementedError("Implement flydsl_silu_and_mul_quant using FlyDSL for the silu_and_mul_quant_kernel task.")


def build_silu_and_mul_quant_module(*args, **kwargs):
    raise NotImplementedError("Implement build_silu_and_mul_quant_module using FlyDSL for the silu_and_mul_quant_kernel task.")
