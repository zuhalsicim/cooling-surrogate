"""Wall-clock cost per case: solver vs surrogate (float32 and int8), on one CPU thread each."""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from . import physics
from .model import ModelMeta, predict


def _median_seconds(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


@contextmanager
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def time_model(net: nn.Module, meta: ModelMeta, inputs: np.ndarray, repeats: int) -> dict:
    """Median seconds for one request (one case) and per case when all `inputs` go in one call."""
    predict(net, meta, inputs[:1])  # warm-up
    single = _median_seconds(lambda: predict(net, meta, inputs[:1]), repeats)
    batch = _median_seconds(lambda: predict(net, meta, inputs), 5) / len(inputs)
    return {"single_case_s_median": single, "batched_s_per_case": batch}


def measure(models: dict[str, tuple[nn.Module, ModelMeta]], inputs: np.ndarray, cfg: dict) -> dict:
    with one_thread():
        n_solver = int(cfg["solver_cases"])
        solver_single = [_median_seconds(lambda i=i: physics.solve(inputs[i : i + 1]), 1) for i in range(n_solver)]
        start = time.perf_counter()
        physics.solve(inputs)
        solver_batch = (time.perf_counter() - start) / len(inputs)
        out = {
            "threads": 1,
            "batch_cases": len(inputs),
            "solver": {
                "single_case_s_median": float(np.median(solver_single)),
                "single_case_s_max": float(np.max(solver_single)),
                "single_case_runs": n_solver,
                "batched_s_per_case": solver_batch,
            },
            "models": {},
        }
        for name, (net, meta) in models.items():
            t = time_model(net, meta, inputs, int(cfg["repeats"]))
            t["speedup_single_vs_solver"] = out["solver"]["single_case_s_median"] / t["single_case_s_median"]
            t["speedup_batched_vs_solver"] = solver_batch / t["batched_s_per_case"]
            out["models"][name] = t
        return out
