#!/usr/bin/env python3
"""Propagate the canonical performance-benchmark helpers into every task copy.

Tasks are executed standalone inside per-run workspaces (a copy of the task
folder), so each task ships its own copy of the perf helper rather than importing
a shared module. To keep all those copies in sync from a single source of truth,
edit the canonical files under tools/perf/ and then run this script:

    python tools/sync_perf_helpers.py            # apply
    python tools/sync_perf_helpers.py --check     # verify in sync (CI-friendly, non-zero on drift)

Two helper families:
  1. tools/perf/performance_utils_pytest.py  -> copied verbatim to every
     */rocmbench/**/performance_utils_pytest.py
  2. tools/perf/vllm_cuda_graph_block.py (the two helper functions) -> injected
     between the AKA-GENERATED markers in every
     triton2triton/vllm/*/scripts/task_runner.py
"""
import argparse
import glob
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PERF = ROOT / "tools" / "perf"

MARK_START = "# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers — edit tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>"
MARK_END = "# <<< AKA-GENERATED <<<"

FUNC_ANCHOR = "def _measure_cuda_event_fallback(fn, repetition):"


def _vllm_block() -> str:
    """The canonical helper functions only (strip the canonical file's docstring)."""
    text = (PERF / "vllm_cuda_graph_block.py").read_text()
    return text[text.index(FUNC_ANCHOR):]


def _rocmbench_targets():
    return sorted(glob.glob(str(ROOT / "tasks/*/rocmbench/**/performance_utils_pytest.py"), recursive=True))


def _vllm_targets():
    return sorted(glob.glob(str(ROOT / "tasks/triton2triton/vllm/*/scripts/task_runner.py")))


def _new_vllm_text(current: str, block: str):
    """Return the file text with the marked region replaced by `block`, or None on error."""
    if MARK_START not in current or MARK_END not in current:
        return None
    pre = current[: current.index(MARK_START)]
    post = current[current.index(MARK_END) + len(MARK_END):]
    return pre + MARK_START + "\n" + block + MARK_END + post


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift without writing (exit 1 if any)")
    args = ap.parse_args()

    canon_roc = (PERF / "performance_utils_pytest.py").read_text()
    block = _vllm_block()
    drift = []
    wrote = 0

    for t in _rocmbench_targets():
        p = pathlib.Path(t)
        if p.read_text() != canon_roc:
            drift.append(t)
            if not args.check:
                p.write_text(canon_roc)
                wrote += 1

    for t in _vllm_targets():
        p = pathlib.Path(t)
        cur = p.read_text()
        new = _new_vllm_text(cur, block)
        if new is None:
            print(f"ERROR: missing AKA-GENERATED markers in {t}", file=sys.stderr)
            return 2
        if new != cur:
            drift.append(t)
            if not args.check:
                p.write_text(new)
                wrote += 1

    if args.check:
        if drift:
            print(f"OUT OF SYNC ({len(drift)} files). Run: python tools/sync_perf_helpers.py")
            for t in drift:
                print(f"  {t}")
            return 1
        print("perf helpers in sync")
        return 0

    print(f"synced {wrote} file(s) "
          f"({len(_rocmbench_targets())} rocmbench + {len(_vllm_targets())} vllm checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
