"""Per-case errors against the solver, and the intervals reported around them."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import physics
from .data import Split
from .model import Prediction


def errors(pred: Prediction, truth: Split) -> dict[str, np.ndarray]:
    """One value per case."""
    case = physics.Case.from_params(truth.inputs)
    diff = pred.field - truth.field
    return {
        "t_cool_rel_error": np.abs(pred.t_cool - truth.t_cool) / truth.t_cool,
        "field_rmse_k": np.sqrt(np.mean(diff**2, axis=(1, 2))),
        "field_max_error_k": np.abs(diff).max(axis=(1, 2)),
        "heat_error_frac": np.abs(pred.heat - truth.heat).max(axis=1) / physics.removable_heat(case),
    }


def p95(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.percentile(values, 95, axis=axis)


def bootstrap(values: np.ndarray, stat: str, n_boot: int, seed: int, level: float = 0.95) -> dict:
    """Point estimate and percentile-bootstrap interval of the mean or the 95th percentile."""
    fn = {"mean": np.mean, "p95": p95}[stat]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = fn(values[idx], axis=1)
    alpha = (1.0 - level) / 2.0
    return {
        "value": float(fn(values)),
        "low": float(np.quantile(boots, alpha)),
        "high": float(np.quantile(boots, 1.0 - alpha)),
    }


def wilson(failures: int, n: int, level: float = 0.95) -> dict:
    """Wilson score interval for a rate (here: the share of cases that violate a check)."""
    z = norm.ppf(1.0 - (1.0 - level) / 2.0)
    p = failures / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"value": p, "low": float(max(0.0, centre - half)), "high": float(min(1.0, centre + half)), "n": n}


def summarise(errs: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    return {
        "n": len(errs["t_cool_rel_error"]),
        "t_cool_rel_error_mean": bootstrap(errs["t_cool_rel_error"], "mean", n_boot, seed),
        "t_cool_rel_error_p95": bootstrap(errs["t_cool_rel_error"], "p95", n_boot, seed),
        "field_rmse_k_mean": bootstrap(errs["field_rmse_k"], "mean", n_boot, seed),
        "field_max_error_k_p95": bootstrap(errs["field_max_error_k"], "p95", n_boot, seed),
        "heat_error_frac_mean": bootstrap(errs["heat_error_frac"], "mean", n_boot, seed),
    }
