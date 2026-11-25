"""Physics checks on a prediction. Each returns a boolean per case: True means the case violates it.

The same checks run in the gate (over a test set) and in the service (on every answer, before it is
returned). They need no solver run, only the inputs and the predicted outputs.
"""

from __future__ import annotations

import numpy as np

from . import physics
from .model import Prediction

TOL_K = 1.0  # temperature slack for bounds and monotonicity
TOL_ENERGY = 0.02  # energy balance slack, as a fraction of the heat the wall can give off
TOL_EJECT_K = 2.0  # mid-plane at the predicted cooling time vs the ejection temperature


def within_bounds(pred: Prediction, case: physics.Case) -> np.ndarray:
    """Never colder than the mould or hotter than the melt."""
    low = pred.field.min(axis=(1, 2)) < case.t_mould - TOL_K
    high = pred.field.max(axis=(1, 2)) > case.t_melt + TOL_K
    return low | high


def cools_monotonically(pred: Prediction, case: physics.Case) -> np.ndarray:
    """No point warms up between two output times (the wall only loses heat)."""
    return np.diff(pred.field, axis=1).max(axis=(1, 2)) > TOL_K


def coolest_at_face(pred: Prediction, case: physics.Case) -> np.ndarray:
    """At every time the temperature falls from the mid-plane towards the mould face."""
    return np.diff(pred.field, axis=2).max(axis=(1, 2)) > TOL_K


def energy_balance(pred: Prediction, case: physics.Case) -> np.ndarray:
    """Heat that left through the face equals the drop in stored enthalpy of the predicted profile."""
    start = case.half * physics.enthalpy(case.t_melt, case.latent)
    stored = physics.stored_heat(pred.field, case, physics.N_CELLS)
    residual = np.abs(start[:, None] - stored - pred.heat) / physics.removable_heat(case)[:, None]
    return residual.max(axis=1) > TOL_ENERGY


def ejects_at_cooling_time(pred: Prediction, case: physics.Case) -> np.ndarray:
    """The predicted profile at the predicted cooling time has its mid-plane at the ejection temperature."""
    return np.abs(pred.field[:, -1, 0] - physics.T_EJECT) > TOL_EJECT_K


CHECKS = {
    "within_bounds": within_bounds,
    "cools_monotonically": cools_monotonically,
    "coolest_at_face": coolest_at_face,
    "energy_balance": energy_balance,
    "ejects_at_cooling_time": ejects_at_cooling_time,
}


def run_all(pred: Prediction, params: np.ndarray) -> dict[str, np.ndarray]:
    case = physics.Case.from_params(params)
    return {name: check(pred, case) for name, check in CHECKS.items()}
