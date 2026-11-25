import pytest

from cooling_surrogate.registry import Registry


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "state")


def _register(registry, model_dir, name):
    run_id = registry.log_run(name, params={"name": name}, metrics={"m": 1.0}, tags={}, model_dir=model_dir)
    return registry.register(run_id, model_dir, tags={"name": name, "dataset_version": "ds-x"})


def test_versions_aliases_and_lineage(registry, tiny_model, tmp_path):
    v1 = _register(registry, tiny_model, "first")
    v2 = _register(registry, tiny_model, "second")
    assert (v1, v2) == ("1", "2")
    assert registry.production_version() is None

    registry.promote(v1)
    assert registry.production_version() == "1"
    registry.promote(v2)
    assert registry.production_version() == "2"

    tags = registry.tags(v2)
    assert tags["dataset_version"] == "ds-x"
    assert len(tags["weights_sha256"]) == 64
    assert [v["aliases"] for v in registry.versions()] == [[], ["production"]]


def test_download_checks_the_weights(registry, tiny_model, tmp_path):
    version = _register(registry, tiny_model, "only")
    folder = registry.download(version, tmp_path / "dl")
    assert (folder / "weights.pt").exists()
    registry.set_tags(version, {"weights_sha256": "0" * 64})
    with pytest.raises(ValueError, match="SHA-256"):
        registry.download(version, tmp_path / "dl2")
