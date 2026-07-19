"""FlyDSL port of `rmsnorm` (rewritten from ../rmsnorm.py).

This is the candidate slot the forge-rewrite agent fills in. It is a skeleton that
fixes only the factory SYMBOL the measurement driver imports; it is NOT a working
implementation yet, so correctness fails on purpose until real FlyDSL is written.

Contract (match how ../rewrite_driver.py calls it):
    build_rmsnorm_module(M, N, dtype_str) -> launch_fn
    launch_fn(x, weight, output, M)
Implement with FlyDSL only (import flydsl...). Do NOT call Triton/torch/HIP.
"""


def build_rmsnorm_module(*args, **kwargs):
    # TODO: build and return a FlyDSL launch callable (@flyc.kernel + @flyc.jit).
    def launch_fn(*a, **k):
        raise NotImplementedError(
            "FlyDSL rmsnorm kernel not implemented yet — port rmsnorm.py to FlyDSL here."
        )

    return launch_fn
