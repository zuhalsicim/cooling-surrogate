import numpy as np
import pytest

from cooling_surrogate import checks, physics
from cooling_surrogate.model import Prediction

# melt, mould, thickness mm, diffusivity mm^2/s, htc, latent kJ/kg
CASES = np.array(
    [
        [230.0, 40.0, 2.0, 0.10, 2500.0, 0.0],
        [200.0, 60.0, 1.0, 0.12, 4000.0, 0.0],
        [260.0, 20.0, 3.0, 0.08, 1500.0, 0.0],
    ]
)


def test_cooling_time_matches_the_exact_series():
    exact = physics.analytic_cooling_time(CASES)
    solved = physics.solve(CASES).t_cool
    assert np.all(np.abs(solved - exact) / exact < 5e-4)


def test_profiles_match_the_exact_series():
    sol = physics.solve(CASES)
    case = physics.Case.from_params(CASES)
    xi = np.linspace(0.0, 1.0, physics.N_CELLS + 1)
    for i in range(len(CASES)):
        fo = physics.FRACTIONS * sol.t_cool[i] * case.diffusivity[i] / case.half[i] ** 2
        theta = physics.analytic_theta(xi, fo, float(case.biot[i]))
        exact = case.t_mould[i] + theta * (case.t_melt[i] - case.t_mould[i])
        assert np.abs(sol.field[i] - exact).max() < 1.0  # kelvin


def test_error_shrinks_at_second_order():
    exact = physics.analytic_cooling_time(CASES[:1])
    errors = []
    for n in (16, 32, 64):
        scale = n / physics.N_CELLS
        t = physics.cooling_time(physics.Case.from_params(CASES[:1]), n_cells=n, dfo=physics.DFO / scale)
        errors.append(abs(t[0] - exact[0]) / exact[0])
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all((orders > 1.6) & (orders < 2.4)), orders


def test_energy_is_conserved_with_latent_heat():
    params = CASES.copy()
    params[:, 5] = [80.0, 40.0, 10.0]
    sol = physics.solve(params)
    case = physics.Case.from_params(params)
    start = case.half * physics.enthalpy(case.t_melt, case.latent)
    stored = physics.stored_heat(sol.field, case, physics.N_CELLS)
    residual = np.abs(start[:, None] - stored - sol.heat) / physics.removable_heat(case)[:, None]
    assert residual.max() < 1e-6


def test_latent_heat_delays_ejection():
    with_latent = CASES.copy()
    with_latent[:, 5] = 60.0
    assert np.all(physics.solve(with_latent).t_cool > physics.solve(CASES).t_cool)


def test_cases_in_a_batch_do_not_influence_each_other():
    # Only the Picard stopping rule is shared (it waits for the slowest case), hence the small slack.
    params = CASES.copy()
    params[:, 5] = [80.0, 0.0, 30.0]
    batch = physics.solve(params)
    for i in range(len(params)):
        single = physics.solve(params[i : i + 1])
        np.testing.assert_allclose(single.t_cool[0], batch.t_cool[i], rtol=1e-9)
        np.testing.assert_allclose(single.field[0], batch.field[i], atol=1e-5)


def test_solver_output_passes_every_physics_check():
    params = CASES.copy()
    params[:, 5] = [80.0, 0.0, 30.0]
    sol = physics.solve(params)
    pred = Prediction(t_cool=sol.t_cool, field=sol.field, heat=sol.heat)
    for name, violated in checks.run_all(pred, params).items():
        assert not violated.any(), name


@pytest.mark.parametrize(
    "row",
    [
        [230.0, 100.0, 2.0, 0.1, 2500.0, 0.0],  # mould at the ejection temperature
        [90.0, 40.0, 2.0, 0.1, 2500.0, 0.0],  # melt below it
        [230.0, 40.0, -1.0, 0.1, 2500.0, 0.0],
    ],
)
def test_rejects_cases_without_a_cooling_time(row):
    with pytest.raises(ValueError):
        physics.Case.from_params(np.array([row]))
