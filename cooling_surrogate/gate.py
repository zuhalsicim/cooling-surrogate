"""The evaluation gate: may this model version become `production`?

A version passes only if, on the held-out in-distribution test set,
  * the upper end of the 95 % bootstrap interval of every error metric is within its threshold, and
  * for every physics check, the upper end of the 95 % Wilson interval of the violation rate is
    within its threshold,
and its lineage holds: the test set's file hashes match the manifest and the model was trained on
that same dataset version. Gating on the upper interval end means a small or noisy test set cannot
wave a model through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch import nn

from . import checks, data, metrics
from .model import ModelMeta, predict

ERROR_CRITERIA = {
    # metric name in metrics.summarise -> policy key
    "t_cool_rel_error_p95": "max_t_cool_rel_error_p95",
    "field_rmse_k_mean": "max_field_rmse_k_mean",
}


def run(model: nn.Module, meta: ModelMeta, dataset_dir: Path, policy: dict, n_boot: int, seed: int) -> dict:
    manifest = data.verify(dataset_dir)  # raises if any file was changed
    test = data.load_split(dataset_dir, "test")
    pred = predict(model, meta, test.inputs)
    summary = metrics.summarise(metrics.errors(pred, test), n_boot, seed)

    criteria = []
    for metric, key in ERROR_CRITERIA.items():
        est = summary[metric]
        criteria.append(
            {
                "name": metric,
                "estimate": est["value"],
                "upper95": est["high"],
                "threshold": policy[key],
                "passed": est["high"] <= policy[key],
            }
        )
    for name, violated in checks.run_all(pred, test.inputs).items():
        rate = metrics.wilson(int(np.sum(violated)), len(violated))
        criteria.append(
            {
                "name": f"physics:{name}",
                "estimate": rate["value"],
                "upper95": rate["high"],
                "failures": int(np.sum(violated)),
                "threshold": policy["max_violation_rate"],
                "passed": rate["high"] <= policy["max_violation_rate"],
            }
        )
    same_data = meta.dataset_version == manifest["version"]
    criteria.append(
        {
            "name": "lineage:trained_on_gate_dataset",
            "estimate": meta.dataset_version,
            "threshold": manifest["version"],
            "passed": same_data,
        }
    )
    failed = [c["name"] for c in criteria if not c["passed"]]
    return {
        "dataset_version": manifest["version"],
        "test_cases": summary["n"],
        "policy": policy,
        "criteria": criteria,
        "decision": "block" if failed else "promote",
        "failed": failed,
    }
