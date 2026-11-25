import numpy as np

from cooling_surrogate import data, model


def test_targets_convert_back_to_physical_units(tiny_dataset, tiny_model):
    _, meta = model.load(tiny_model)
    split = data.load_split(tiny_dataset, "val")
    back = model.to_physical(model.make_targets(split, meta), split.inputs, meta)
    np.testing.assert_allclose(back.t_cool, split.t_cool, rtol=1e-10)
    np.testing.assert_allclose(back.field, split.field, atol=1e-9)
    np.testing.assert_allclose(back.heat, split.heat, rtol=1e-9)


def test_int8_model_survives_save_and_load(tiny_dataset, tiny_model, tmp_path):
    net, meta = model.load(tiny_model)
    meta.quantized = True
    q = model.quantize_int8(net)
    model.save(q, meta, tmp_path / "int8")
    loaded, loaded_meta = model.load(tmp_path / "int8")
    assert loaded_meta.quantized
    x = data.load_split(tiny_dataset, "test").inputs
    a, b = model.predict(q, meta, x), model.predict(loaded, loaded_meta, x)
    np.testing.assert_array_equal(a.t_cool, b.t_cool)
    # and the int8 copy stays close to the float model it came from
    ref = model.predict(net, meta, x)
    assert np.abs(a.t_cool / ref.t_cool - 1).max() < 0.05
