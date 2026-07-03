import dataclasses

from nn.nn_model_spec import LayerSpec, NNModelSpec


def test_to_dict_is_plain_nested_dict():
    spec = NNModelSpec.default()
    d = spec.to_dict()

    assert isinstance(d, dict)
    # no dataclass instances survive — everything is plain dict/list/scalar
    assert not dataclasses.is_dataclass(d)
    assert "device" in d and "seed" in d  # asdict keeps these (hash drops them, dict does not)

    assert isinstance(d["layers"], list) and len(d["layers"]) >= 1
    for layer in d["layers"]:
        assert isinstance(layer, dict)
        assert set(("kind", "units", "params")) <= set(layer)
        assert isinstance(layer["kind"], str)
        assert isinstance(layer["params"], dict)

    assert isinstance(d["targets"], list) and len(d["targets"]) >= 1
    for tgt in d["targets"]:
        assert isinstance(tgt, dict)
        assert "name" in tgt and "horizons" in tgt
        assert isinstance(tgt["horizons"], list)

    assert isinstance(d["grouping"], dict)


def test_to_yaml_from_yaml_roundtrip_preserves_hash(tmp_path):
    original = NNModelSpec.default()
    path = tmp_path / "spec.yaml"

    original.to_yaml(str(path))
    assert path.exists()

    reloaded = NNModelSpec.from_yaml(str(path))
    assert reloaded.spec_hash == original.spec_hash


def test_roundtrip_preserves_mixed_layers_and_timeframes(tmp_path):
    original = dataclasses.replace(
        NNModelSpec.default(),
        timeframes=[15, 60],
        layers=[
            LayerSpec("conv1d", 16, {"kernel_size": 5}),
            LayerSpec("lstm", 32, {"num_layers": 2}),
        ],
    )
    path = tmp_path / "spec.yaml"
    original.to_yaml(str(path))

    reloaded = NNModelSpec.from_yaml(str(path))

    assert [l.kind for l in reloaded.layers] == ["conv1d", "lstm"]
    assert [l.units for l in reloaded.layers] == [16, 32]
    assert reloaded.layers[0].params == {"kernel_size": 5}
    assert reloaded.layers[1].params == {"num_layers": 2}
    assert reloaded.timeframes == [15, 60]
    assert reloaded.spec_hash == original.spec_hash


def test_to_yaml_creates_parent_dirs(tmp_path):
    original = NNModelSpec.default()
    path = tmp_path / "a" / "b" / "spec.yaml"  # parents a/, a/b/ do not exist yet

    original.to_yaml(str(path))

    assert path.exists()
    reloaded = NNModelSpec.from_yaml(str(path))
    assert reloaded.spec_hash == original.spec_hash
