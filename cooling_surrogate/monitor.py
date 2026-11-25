"""Monitoring: flag inputs outside the training box, and measure live error from the request log.

The service calls `out_of_box` on every request. Offline, `report` reads the service's request log
and answers three questions:
  * how much traffic falls outside what the model was trained on, and along which parameter;
  * how far off the surrogate is on the traffic it did answer (a sample is re-run with the solver);
  * how far off it would have been on the traffic it did not answer (the service computes the
    surrogate's answer for flagged requests too, logs it, and returns the solver's).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import metrics, physics


def out_of_box(params: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[list[str]]:
    """For each case, the names of the parameters outside [lo, hi]."""
    params = np.atleast_2d(params)
    outside = (params < lo) | (params > hi)
    return [[physics.PARAM_NAMES[j] for j in np.flatnonzero(row)] for row in outside]


def read_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def report(log_path: Path, sample_size: int, seed: int, n_boot: int, alerts_policy: dict) -> dict:
    records = read_log(log_path)
    n = len(records)
    served = Counter(r["source"] for r in records)
    reasons = Counter(r["fallback_reason"] for r in records if r["fallback_reason"])
    by_feature = Counter(f for r in records for f in r["out_of_box"])
    ood_rate = metrics.wilson(sum(bool(r["out_of_box"]) for r in records), n)

    # Shadow evaluation: re-run the solver on a sample of the answers the surrogate gave.
    answered = [r for r in records if r["source"] == "surrogate"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(answered), size=min(sample_size, len(answered)), replace=False)
    sample = [answered[i] for i in sorted(pick)]
    shadow: dict = {"sampled": len(sample)}
    if sample:
        params = np.array([[r["inputs"][k] for k in physics.PARAM_NAMES] for r in sample])
        truth = physics.solve(params).t_cool
        err = np.abs(np.array([r["t_cool_s"] for r in sample]) - truth) / truth
        shadow["t_cool_rel_error_mean"] = metrics.bootstrap(err, "mean", n_boot, seed)
        shadow["t_cool_rel_error_p95"] = metrics.bootstrap(err, "p95", n_boot, seed)

    # Fallbacks: the solver answered; the logged surrogate value shows the error that was avoided.
    avoided: dict = {}
    for reason in sorted(reasons):
        rows = [r for r in records if r["fallback_reason"] == reason]
        err = np.array([abs(r["surrogate_t_cool_s"] - r["t_cool_s"]) / r["t_cool_s"] for r in rows])
        avoided[reason] = {
            "n": len(rows),
            "t_cool_rel_error_mean": metrics.bootstrap(err, "mean", n_boot, seed),
            "t_cool_rel_error_p95": metrics.bootstrap(err, "p95", n_boot, seed),
        }

    alerts = []
    if ood_rate["low"] > alerts_policy["max_out_of_box_rate"]:
        alerts.append(
            f"out-of-box traffic {100 * ood_rate['value']:.1f} % (95 % CI from {100 * ood_rate['low']:.1f} %) "
            f"is above {100 * alerts_policy['max_out_of_box_rate']:.0f} %; most often outside: "
            + ", ".join(f"{k} ({v})" for k, v in by_feature.most_common(3))
        )
    if sample and shadow["t_cool_rel_error_p95"]["low"] > alerts_policy["max_shadow_t_cool_rel_error_p95"]:
        alerts.append("live cooling-time error on answered traffic is above the gate threshold")

    latency = {src: float(np.median([r["latency_ms"] for r in records if r["source"] == src])) for src in served}
    return {
        "requests": n,
        "served_by": dict(served),
        "fallback_reasons": dict(reasons),
        "out_of_box_rate": ood_rate,
        "out_of_box_by_feature": dict(by_feature.most_common()),
        "median_latency_ms": latency,
        "shadow_on_answered": shadow,
        "surrogate_error_avoided_by_fallback": avoided,
        "alerts": alerts,
    }
