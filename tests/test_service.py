import numpy as np
import pytest
from fastapi.testclient import TestClient

from cooling_surrogate import checks, monitor, physics, service

IN_BOX = {
    "melt_temp_c": 230.0,
    "mould_temp_c": 40.0,
    "thickness_mm": 2.0,
    "diffusivity_mm2_s": 0.1,
    "htc_w_m2k": 2500.0,
    "latent_heat_kj_kg": 40.0,
}


@pytest.fixture
def client(bundle, tmp_path):
    return TestClient(service.create_app(bundle, request_log=tmp_path / "requests.jsonl"))


def _no_violations(pred, params):
    return {name: np.zeros(len(params), dtype=bool) for name in checks.CHECKS}


def test_health_and_model_info(client):
    assert client.get("/healthz").json() == {"status": "ok", "model_version": "7"}
    info = client.get("/v1/model").json()
    assert info["training_box"]["thickness_mm"] == [1.0, 3.0]


def test_in_box_request_is_answered_by_the_surrogate(client, monkeypatch):
    monkeypatch.setattr(checks, "run_all", _no_violations)
    body = client.post("/v1/cooling-time", json=IN_BOX).json()
    assert body["source"] == "surrogate"
    assert body["fallback_reason"] is None
    assert len(body["profile"]["temperatures_c"]) == physics.FRACTIONS.size
    assert len(body["profile"]["positions_mm"]) == physics.N_CELLS + 1
    assert body["profile"]["positions_mm"][-1] == pytest.approx(1.0)


def test_out_of_box_request_falls_back_to_the_solver(client):
    body = client.post("/v1/cooling-time", json={**IN_BOX, "thickness_mm": 4.5}).json()
    assert body["source"] == "solver"
    assert body["fallback_reason"] == "out_of_box"
    assert body["out_of_box"] == ["thickness_mm"]
    exact = physics.solve(np.array([[IN_BOX[k] if k != "thickness_mm" else 4.5 for k in physics.PARAM_NAMES]]))
    assert body["cooling_time_s"] == pytest.approx(exact.t_cool[0])


def test_failed_physics_check_falls_back_to_the_solver(client, monkeypatch):
    def energy_fails(pred, params):
        out = _no_violations(pred, params)
        out["energy_balance"][:] = True
        return out

    monkeypatch.setattr(checks, "run_all", energy_fails)
    body = client.post("/v1/cooling-time", json=IN_BOX).json()
    assert body["source"] == "solver"
    assert body["fallback_reason"] == "physics_check"
    assert body["failed_checks"] == ["energy_balance"]


@pytest.mark.parametrize(
    "change",
    [{"mould_temp_c": 100.0}, {"melt_temp_c": 95.0}, {"thickness_mm": 0.0}, {"latent_heat_kj_kg": -1.0}],
)
def test_unphysical_input_is_rejected(client, change):
    assert client.post("/v1/cooling-time", json={**IN_BOX, **change}).status_code == 422


def test_requests_are_logged_and_the_monitor_reads_them(client, monkeypatch, tmp_path):
    monkeypatch.setattr(checks, "run_all", _no_violations)
    for thickness in (1.5, 2.0, 2.5, 4.0):
        client.post("/v1/cooling-time", json={**IN_BOX, "thickness_mm": thickness})
    assert client.get("/metrics").json() == {
        "requests": 4,
        "source_surrogate": 3,
        "source_solver": 1,
        "fallback_out_of_box": 1,
    }
    policy = {"max_out_of_box_rate": 0.05, "max_shadow_t_cool_rel_error_p95": 0.02}
    rep = monitor.report(tmp_path / "requests.jsonl", sample_size=2, seed=0, n_boot=100, alerts_policy=policy)
    assert rep["requests"] == 4
    assert rep["out_of_box_by_feature"] == {"thickness_mm": 1}
    assert rep["shadow_on_answered"]["sampled"] == 2
    assert rep["surrogate_error_avoided_by_fallback"]["out_of_box"]["n"] == 1
