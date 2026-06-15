# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only equivalence gate for generated torch2hip tasks.

For each task it loads the reference module and the functional model exactly as
eval_tools/correctness_check.py would (same get_init_inputs / state-dict alignment
semantics), then runs every get_inputs() case on CPU and asserts that

    module(*inputs)  ==  functional(*inputs)        # functional uses fn=module_fn

within rtol=1e-4, atol=1e-5. Because the functional model's default `fn` is the
pure-PyTorch module_fn, a pass here proves the functional refactor is numerically
faithful to the reference before any GPU/HIP run.

Usage:
    python tools/validate_functional_cpu.py tasks/torch2hip/kernelbench

Requires a working torch install. (On the Windows box torch may be broken; run this
on the ROCm/MI300X box or any machine with CPU torch.)
"""
import argparse
import copy
import importlib.util
import re
import sys
from pathlib import Path

import torch
import yaml


def _load_attr(py_path: Path, attr: str):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr)


def _instantiate(py_path: Path, class_name: str):
    cls = _load_attr(py_path, class_name)
    init_params = _load_attr(py_path, "get_init_inputs")()
    if len(init_params) == 0:
        return cls()
    if len(init_params) == 2 and isinstance(init_params[0], list) and isinstance(init_params[1], dict):
        return cls() if len(init_params[1]) == 0 else cls(**init_params[1])
    return cls(*init_params)


def _canonical_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _candidate_module_keys(func_key, module_keys):
    if func_key in module_keys:
        return [func_key]
    func_ck = _canonical_key(func_key)
    exact = [k for k in module_keys if _canonical_key(k) == func_ck]
    if exact:
        return exact
    suffix = [k for k in module_keys if _canonical_key(k).endswith(func_ck)]
    if len(suffix) == 1:
        return suffix
    bi_suffix = [k for k in module_keys if func_ck.endswith(_canonical_key(k))]
    if len(bi_suffix) == 1:
        return bi_suffix
    return []


def _align_state_dict(module_obj, func_obj):
    module_sd = module_obj.state_dict()
    func_sd = func_obj.state_dict()
    aligned = dict(func_sd)
    used = set()
    module_keys = list(module_sd.keys())
    for fk, fv in func_sd.items():
        cands = [k for k in _candidate_module_keys(fk, module_keys) if k not in used]
        if not cands:
            continue
        mk = cands[0]
        if module_sd[mk].shape != fv.shape:
            continue
        aligned[fk] = module_sd[mk].detach().clone()
        used.add(mk)
    func_obj.load_state_dict(aligned, strict=False)


def _compare(a, b, rtol=1e-4, atol=1e-5) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_compare(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_compare(x, y, rtol, atol) for x, y in zip(a, b))
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.allclose(a, b, rtol=rtol, atol=atol)
    return a == b


def _normalize_inputs(result):
    if hasattr(result, "__next__") or hasattr(result, "send"):
        return result
    def _gen():
        yield result
    return _gen()


def validate_task(task_dir: Path) -> bool:
    cfg = yaml.safe_load((task_dir / "config.yaml").read_text(encoding="utf-8"))
    hip_name = Path(cfg["target_file_path"]).name
    kernel_name = hip_name.split(".hip")[0].split("_", 2)[-1]

    modu_path = task_dir / cfg["source_file_path"][0]
    func_rel = cfg["correctness_command"][0].split("--py_func_file", 1)[1].split()[0]
    func_path = task_dir / func_rel

    modu = _instantiate(modu_path, kernel_name).eval()
    func = _instantiate(func_path, kernel_name).eval()
    _align_state_dict(modu, func)

    get_inputs = _load_attr(modu_path, "get_inputs")
    torch.manual_seed(0)
    cases = list(_normalize_inputs(get_inputs()))
    for idx, inputs in enumerate(cases):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        inputs = list(inputs)
        with torch.no_grad():
            r_modu = modu(*copy.deepcopy(inputs))
            r_func = func(*copy.deepcopy(inputs))  # func uses default fn=module_fn
        if not _compare(r_modu, r_func):
            print(f"  [MISMATCH] {kernel_name} case {idx}")
            return False
    print(f"  [OK] {kernel_name} ({len(cases)} case(s))")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="dir containing generated tasks (globs **/config.yaml)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    configs = sorted(root.glob("**/config.yaml"))
    if not configs:
        print(f"no config.yaml found under {root}")
        sys.exit(1)

    passed, failed = 0, 0
    for cfg in configs:
        task_dir = cfg.parent
        print(f"{task_dir.name}:")
        try:
            ok = validate_task(task_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {e}")
            ok = False
        passed += ok
        failed += not ok

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
