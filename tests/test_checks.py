import numpy as np
import pytest

from cooling_surrogate import checks, physics
from cooling_surrogate.model import Prediction

PARAMS = np.array([[230.0, 40.0, 2.0, 0.10, 2500.0, 50.0]])


@pytest.fixture(scope="module")
def solved() -> Prediction:
    sol = physics.solve(PARAMS)
    return Prediction(t_cool=sol.t_cool, field=sol.field, heat=sol.heat)


def _broken(pred: Prediction, **changes) -> Prediction:
    fields = {"t_cool": pred.t_cool.copy(), "field": pred.field.copy(), "heat": pred.heat.copy(), **changes}
    return Prediction(**fields)


def _violated(pred: Prediction) -> set[str]:
    return {name for name, bad in checks.run_all(pred, PARAMS).items() if bad[0]}


def test_the_solver_passes(solved):
    assert _violated(solved) == set()


def test_colder_than_the_mould(solved):
    field = solved.field.copy()
    field[0, -1, -1] = 40.0 - 5.0
    assert "within_bounds" in _violated(_broken(solved, field=field))


def test_warming_up_between_two_times(solved):
    field = solved.field.copy()
    field[0, 6, 10] = field[0, 5, 10] + 5.0
    assert "cools_monotonically" in _violated(_broken(solved, field=field))


def test_warmer_at_the_face_than_inside(solved):
    field = solved.field.copy()
    field[0, 3] = field[0, 3, ::-1]
    assert "coolest_at_face" in _violated(_broken(solved, field=field))


def test_heat_that_does_not_add_up(solved):
    assert _violated(_broken(solved, heat=solved.heat * 1.1)) == {"energy_balance"}


def test_profile_and_cooling_time_that_disagree(solved):
    field = solved.field.copy()
    field[0, -1] += 5.0
    assert "ejects_at_cooling_time" in _violated(_broken(solved, field=field))
