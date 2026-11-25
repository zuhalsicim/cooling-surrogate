from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cooling_surrogate import data, train

TINY_DATA = {"seed": 1, "n_train": 64, "n_val": 16, "n_test": 24, "n_ood_per_shift": 4}
TINY_MODEL = {
    "name": "tiny",
    "note": "test model",
    "hidden": 32,
    "depth": 2,
    "epochs": 20,
    "batch_size": 32,
    "lr": 0.003,
    "seed": 0,
    "train_fraction": 1.0,
    "loss_weights": {"t_cool": 10.0, "field": 1.0, "heat": 1.0},
}


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    return data.generate(TINY_DATA, tmp_path_factory.mktemp("data"), log=lambda *_: None)


@pytest.fixture(scope="session")
def tiny_model(tiny_dataset, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("model") / "tiny"
    train.train(tiny_dataset, TINY_MODEL, out, log=lambda *_: None)
    return out


@pytest.fixture
def bundle(tiny_model, tmp_path) -> Path:
    dest = tmp_path / "bundle"
    shutil.copytree(tiny_model, dest / "model")
    (dest / "bundle.json").write_text(json.dumps({"model_version": "7", "dataset_version": "test"}))
    return dest
