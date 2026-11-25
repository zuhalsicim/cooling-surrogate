import shutil

import numpy as np
import pytest

from cooling_surrogate import data, physics
from tests.conftest import TINY_DATA


def test_manifest_matches_the_files(tiny_dataset):
    manifest = data.verify(tiny_dataset)
    assert tiny_dataset.name == manifest["version"]
    assert manifest["splits"]["train"] == TINY_DATA["n_train"]
    assert set(manifest["splits"]) == {"train", "val", "test"} | {f"ood_{g}" for g in data.OOD_SHIFTS}


def test_a_changed_file_is_caught(tiny_dataset, tmp_path):
    copy = tmp_path / tiny_dataset.name
    shutil.copytree(tiny_dataset, copy)
    t = np.load(copy / "test" / "t_cool.npy")
    t[0] *= 1.001
    np.save(copy / "test" / "t_cool.npy", t)
    with pytest.raises(ValueError, match="does not match the manifest"):
        data.verify(copy)


def test_same_recipe_reuses_the_dataset(tiny_dataset):
    messages = []
    again = data.generate(TINY_DATA, tiny_dataset.parent, log=messages.append)
    assert again == tiny_dataset
    assert "already built" in messages[0]


def test_ood_splits_leave_the_box_in_exactly_one_parameter(tiny_dataset):
    lo, hi = data.box_arrays(data.TRAIN_BOX)
    for group, (name, _) in data.OOD_SHIFTS.items():
        x = data.load_split(tiny_dataset, f"ood_{group}").inputs
        outside = (x < lo) | (x > hi)
        assert np.all(outside.sum(axis=1) == 1)
        assert np.all(outside[:, physics.PARAM_NAMES.index(name)])


def test_in_distribution_splits_stay_inside_the_box(tiny_dataset):
    lo, hi = data.box_arrays(data.TRAIN_BOX)
    for split in ("train", "val", "test"):
        x = data.load_split(tiny_dataset, split).inputs
        assert np.all((x >= lo) & (x <= hi))
