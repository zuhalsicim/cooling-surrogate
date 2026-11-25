import dataclasses
import shutil

import numpy as np
import pytest

from cooling_surrogate import gate, model

LOOSE = {"max_t_cool_rel_error_p95": 1.0, "max_field_rmse_k_mean": 100.0, "max_violation_rate": 1.0}
STRICT = {"max_t_cool_rel_error_p95": 1e-6, "max_field_rmse_k_mean": 1e-6, "max_violation_rate": 0.0}


def _run(tiny_model, dataset, policy, meta_changes=None):
    net, meta = model.load(tiny_model)
    if meta_changes:
        meta = dataclasses.replace(meta, **meta_changes)
    return gate.run(net, meta, dataset, policy, n_boot=200, seed=0)


def test_a_model_within_the_policy_is_promoted(tiny_model, tiny_dataset):
    report = _run(tiny_model, tiny_dataset, LOOSE)
    assert report["decision"] == "promote"
    assert report["failed"] == []


def test_a_model_outside_the_policy_is_blocked_with_reasons(tiny_model, tiny_dataset):
    report = _run(tiny_model, tiny_dataset, STRICT)
    assert report["decision"] == "block"
    assert {"t_cool_rel_error_p95", "field_rmse_k_mean"} <= set(report["failed"])


def test_the_gate_uses_the_upper_end_of_the_interval(tiny_model, tiny_dataset):
    report = _run(tiny_model, tiny_dataset, LOOSE)
    crit = next(c for c in report["criteria"] if c["name"] == "t_cool_rel_error_p95")
    policy = {**LOOSE, "max_t_cool_rel_error_p95": (crit["estimate"] + crit["upper95"]) / 2}
    assert crit["estimate"] < policy["max_t_cool_rel_error_p95"] < crit["upper95"]
    assert "t_cool_rel_error_p95" in _run(tiny_model, tiny_dataset, policy)["failed"]


def test_a_model_trained_on_other_data_is_blocked(tiny_model, tiny_dataset):
    report = _run(tiny_model, tiny_dataset, LOOSE, {"dataset_version": "ds-000000000000"})
    assert report["failed"] == ["lineage:trained_on_gate_dataset"]


def test_a_changed_test_set_stops_the_gate(tiny_model, tiny_dataset, tmp_path):
    copy = tmp_path / tiny_dataset.name
    shutil.copytree(tiny_dataset, copy)
    x = np.load(copy / "test" / "inputs.npy")
    x[0, 0] += 1.0
    np.save(copy / "test" / "inputs.npy", x)
    with pytest.raises(ValueError):
        _run(tiny_model, copy, LOOSE)
