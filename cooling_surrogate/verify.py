"""Solver verification: how far is the solver from the exact answer, and how does that error shrink?

1. Without latent heat the exact series solution exists. Refine the grid and the time step together
   (dFo proportional to dx, so Crank-Nicolson's time error and the space error both scale as dx^2)
   and measure the error in cooling time and profile; the observed order should approach 2.
2. With latent heat there is no exact solution. Compare the dataset setting with a run on a 4x finer
   grid and 4x smaller time steps (self-convergence).
3. Energy: heat that left through the face must equal the drop in stored enthalpy.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import data, physics


def _exact_field(params: np.ndarray, t_cool: np.ndarray, n_cells: int) -> np.ndarray:
    case = physics.Case.from_params(params)
    xi = np.linspace(0.0, 1.0, n_cells + 1)
    out = np.empty((len(params), physics.FRACTIONS.size, n_cells + 1))
    for i in range(len(params)):
        fo = physics.FRACTIONS * t_cool[i] * case.diffusivity[i] / case.half[i] ** 2
        theta = physics.analytic_theta(xi, fo, float(case.biot[i]))
        out[i] = case.t_mould[i] + theta * (case.t_melt[i] - case.t_mould[i])
    return out


def _energy_residual(params: np.ndarray, sol: physics.Solution, n_cells: int) -> np.ndarray:
    case = physics.Case.from_params(params)
    start = case.half * physics.enthalpy(case.t_melt, case.latent)
    stored = physics.stored_heat(sol.field, case, n_cells)
    return np.abs(start[:, None] - stored - sol.heat).max(axis=1) / physics.removable_heat(case)


def run(n_cases: int = 40, seed: int = 11, log=print) -> dict:
    lo, hi = data.box_arrays(data.TRAIN_BOX)
    rng = np.random.default_rng(seed)
    params = lo + (hi - lo) * rng.random((n_cases, lo.size))
    linear = params.copy()
    linear[:, physics.PARAM_NAMES.index("latent_heat_kj_kg")] = 0.0
    exact_t = physics.analytic_cooling_time(linear)

    grid = []
    for n in (8, 16, 32, 64, 128):
        scale = n / physics.N_CELLS
        dfo = physics.DFO / scale
        steps = round(physics.STEPS_TO_EJECT * scale)
        sol = physics.solve(linear, n_cells=n, dfo=dfo, steps_to_eject=steps)
        t_err = np.abs(sol.t_cool - exact_t) / exact_t
        f_err = np.abs(sol.field - _exact_field(linear, sol.t_cool, n))
        grid.append(
            {
                "n_cells": n,
                "dfo": dfo,
                "steps_to_eject": steps,
                "t_cool_rel_error_max": float(t_err.max()),
                "field_error_k_max": float(f_err.max()),
                "energy_residual_max": float(_energy_residual(linear, sol, n).max()),
            }
        )
        log(f"  N = {n:3d}: cooling time error {100 * t_err.max():.4f} %, profile error {f_err.max():.3f} K")
    for coarse, fine in itertools.pairwise(grid):
        fine["order_t_cool"] = float(np.log2(coarse["t_cool_rel_error_max"] / fine["t_cool_rel_error_max"]))
        fine["order_field"] = float(np.log2(coarse["field_error_k_max"] / fine["field_error_k_max"]))

    prod = physics.solve(params)
    ref = physics.solve(
        params, n_cells=4 * physics.N_CELLS, dfo=physics.DFO / 4, steps_to_eject=4 * physics.STEPS_TO_EJECT
    )
    t_diff = np.abs(prod.t_cool - ref.t_cool) / ref.t_cool
    f_diff = np.abs(prod.field - ref.field[:, :, ::4])
    latent = {
        "reference": {"n_cells": 4 * physics.N_CELLS, "dfo": physics.DFO / 4},
        "t_cool_rel_diff_max": float(t_diff.max()),
        "field_diff_k_max": float(f_diff.max()),
        "field_diff_k_max_by_fraction": f_diff.max(axis=(0, 2)).round(4).tolist(),
        "energy_residual_max": float(_energy_residual(params, prod, physics.N_CELLS).max()),
        "midplane_at_t_cool_minus_eject_k_max": float(np.abs(prod.field[:, -1, 0] - physics.T_EJECT).max()),
    }
    log(
        f"  latent heat, dataset setting vs 4x finer: cooling time {100 * t_diff.max():.4f} %, "
        f"profile {f_diff.max():.3f} K, energy residual {latent['energy_residual_max']:.1e}"
    )
    return {
        "cases": n_cases,
        "dataset_setting": {"n_cells": physics.N_CELLS, "dfo": physics.DFO, "steps_to_eject": physics.STEPS_TO_EJECT},
        "against_exact_without_latent_heat": grid,
        "with_latent_heat_against_finer_run": latent,
    }
