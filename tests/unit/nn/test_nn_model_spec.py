"""Unit tests for nn/nn_model_spec.py — NNModelSpec, LayerSpec, GroupingSpec, TargetSpec.

TDD: tests written BEFORE implementation (RED → GREEN).
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from nn.nn_model_spec import (
    GroupingSpec,
    LayerSpec,
    NNModelSpec,
    TargetSpec,
)


# ---------------------------------------------------------------------------
# spec_hash determinism
# ---------------------------------------------------------------------------


class TestSpecHash:
    """spec_hash must be a stable content-addressed fingerprint."""

    def _minimal_spec(self, **overrides):
        base = dict(
            name="test",
            timeframes=[15],
            indicators=["rsi_14", "logret"],
            history_points=8,
            layers=[LayerSpec(kind="dense", units=32)],
            targets=[
                TargetSpec(
                    name="dir15",
                    kind="direction",
                    horizons=[1],
                    label_tf=15,
                    label_m=1.0,
                    label_x=0.3,
                )
            ],
        )
        base.update(overrides)
        return NNModelSpec(**base)

    def test_same_fields_identical_hash(self):
        s1 = self._minimal_spec()
        s2 = self._minimal_spec()
        assert s1.spec_hash == s2.spec_hash

    def test_hash_is_64_char_hex(self):
        s = self._minimal_spec()
        h = s.spec_hash
        assert len(h) == 64
        int(h, 16)  # raises ValueError if not hex

    def test_reordering_params_dict_identical_hash(self):
        """Dict key order in LayerSpec.params must not affect hash."""
        s1 = self._minimal_spec(
            layers=[LayerSpec(kind="conv1d", units=64, params={"kernel_size": 3, "stride": 1})]
        )
        s2 = self._minimal_spec(
            layers=[LayerSpec(kind="conv1d", units=64, params={"stride": 1, "kernel_size": 3})]
        )
        assert s1.spec_hash == s2.spec_hash

    def test_changing_device_does_not_change_hash(self):
        s1 = self._minimal_spec(device="auto")
        s2 = self._minimal_spec(device="cpu")
        assert s1.spec_hash == s2.spec_hash

    def test_changing_seed_does_not_change_hash(self):
        s1 = self._minimal_spec(seed=0)
        s2 = self._minimal_spec(seed=42)
        assert s1.spec_hash == s2.spec_hash

    def test_changing_layers_changes_hash(self):
        s1 = self._minimal_spec(layers=[LayerSpec(kind="dense", units=32)])
        s2 = self._minimal_spec(layers=[LayerSpec(kind="dense", units=64)])
        assert s1.spec_hash != s2.spec_hash

    def test_changing_indicators_changes_hash(self):
        s1 = self._minimal_spec(indicators=["rsi_14"])
        s2 = self._minimal_spec(indicators=["rsi_14", "logret"])
        assert s1.spec_hash != s2.spec_hash

    def test_changing_targets_changes_hash(self):
        s1 = self._minimal_spec(
            targets=[TargetSpec(name="dir15", kind="direction", horizons=[1], label_tf=15, label_m=1.0, label_x=0.3)]
        )
        s2 = self._minimal_spec(
            targets=[TargetSpec(name="dir60", kind="direction", horizons=[1], label_tf=60, label_m=1.0, label_x=0.3)]
        )
        assert s1.spec_hash != s2.spec_hash

    def test_changing_timeframes_changes_hash(self):
        s1 = self._minimal_spec(timeframes=[15])
        s2 = self._minimal_spec(timeframes=[15, 60])
        assert s1.spec_hash != s2.spec_hash

    def test_layer_order_matters(self):
        """Layers are ordered; swapping two layers changes hash."""
        s1 = self._minimal_spec(
            layers=[
                LayerSpec(kind="dense", units=64),
                LayerSpec(kind="dense", units=32),
            ]
        )
        s2 = self._minimal_spec(
            layers=[
                LayerSpec(kind="dense", units=32),
                LayerSpec(kind="dense", units=64),
            ]
        )
        assert s1.spec_hash != s2.spec_hash


# ---------------------------------------------------------------------------
# from_yaml
# ---------------------------------------------------------------------------


class TestFromYaml:
    """NNModelSpec.from_yaml must build nested specs and coerce types."""

    def _write_yaml(self, data: dict, tmp_path: Path) -> str:
        p = tmp_path / "spec.yaml"
        p.write_text(yaml.dump(data))
        return str(p)

    def test_basic_round_trip(self, tmp_path):
        data = {
            "name": "round_trip",
            "timeframes": [15, 60],
            "indicators": ["rsi_14", "logret"],
            "history_points": 16,
            "layers": [{"kind": "dense", "units": 64}],
            "targets": [
                {
                    "name": "dir15",
                    "kind": "direction",
                    "horizons": [1],
                    "label_tf": 15,
                    "label_m": 1.0,
                    "label_x": 0.3,
                }
            ],
        }
        path = self._write_yaml(data, tmp_path)
        spec = NNModelSpec.from_yaml(path)
        assert spec.name == "round_trip"
        assert spec.timeframes == [15, 60]
        assert spec.history_points == 16
        assert isinstance(spec.grouping, GroupingSpec)
        assert isinstance(spec.layers[0], LayerSpec)
        assert isinstance(spec.targets[0], TargetSpec)

    def test_timeframes_coerced_to_list_int(self, tmp_path):
        """YAML may emit timeframes as plain ints; must be list[int]."""
        data = {
            "name": "coerce_test",
            "timeframes": [15, 60],
            "indicators": ["rsi_14"],
            "layers": [{"kind": "dense", "units": 32}],
            "targets": [{"name": "dir15", "kind": "direction", "horizons": [1], "label_tf": 15, "label_m": 1.0, "label_x": 0.3}],
        }
        path = self._write_yaml(data, tmp_path)
        spec = NNModelSpec.from_yaml(path)
        assert all(isinstance(tf, int) for tf in spec.timeframes)

    def test_horizons_coerced_to_list_int(self, tmp_path):
        data = {
            "name": "horizon_test",
            "timeframes": [15],
            "indicators": ["rsi_14"],
            "layers": [{"kind": "dense", "units": 32}],
            "targets": [{"name": "dir15", "kind": "direction", "horizons": [1, 2, 3], "label_tf": 15, "label_m": 1.0, "label_x": 0.3}],
        }
        path = self._write_yaml(data, tmp_path)
        spec = NNModelSpec.from_yaml(path)
        assert all(isinstance(h, int) for h in spec.targets[0].horizons)

    def test_absent_keys_use_dataclass_defaults(self, tmp_path):
        """Minimal YAML: absent optional fields take dataclass defaults."""
        data = {
            "name": "minimal",
            "timeframes": [15],
            "indicators": ["rsi_14"],
            "layers": [{"kind": "dense", "units": 32}],
            "targets": [{"name": "dir15", "kind": "direction", "label_tf": 15, "label_m": 1.0, "label_x": 0.3}],
        }
        path = self._write_yaml(data, tmp_path)
        spec = NNModelSpec.from_yaml(path)
        assert spec.activation == "relu"
        assert spec.dropout == 0.0
        assert spec.optimizer == "adam"
        assert spec.learning_rate == pytest.approx(1e-3)
        assert spec.batch_size == 32
        assert spec.epochs == 100
        assert spec.validation_split == pytest.approx(0.2)
        assert spec.val_strategy == "time_holdout"
        assert spec.early_stopping_patience == 10
        assert spec.class_weight == "balanced"
        assert spec.device == "auto"
        assert spec.seed == 0
        assert spec.loss_fn == "auto"
        assert spec.weight_decay == 0.0
        assert spec.grouping.mode == "single"

    def test_grouping_by_indicator_parsed(self, tmp_path):
        data = {
            "name": "grp_test",
            "timeframes": [60],
            "indicators": ["rsi_14"],
            "layers": [{"kind": "dense", "units": 32}],
            "targets": [{"name": "dir15", "kind": "direction", "label_tf": 15, "label_m": 1.0, "label_x": 0.3}],
            "grouping": {
                "mode": "by_indicator",
                "column": "60_vol_regime",
                "classes": [0, 1, 2],
            },
        }
        path = self._write_yaml(data, tmp_path)
        spec = NNModelSpec.from_yaml(path)
        assert spec.grouping.mode == "by_indicator"
        assert spec.grouping.column == "60_vol_regime"
        assert spec.grouping.classes == [0, 1, 2]

    def test_layer_params_preserved(self, tmp_path):
        data = {
            "name": "params_test",
            "timeframes": [15],
            "indicators": ["rsi_14"],
            "layers": [{"kind": "conv1d", "units": 64, "params": {"kernel_size": 3}}],
            "targets": [{"name": "dir15", "kind": "direction", "label_tf": 15, "label_m": 1.0, "label_x": 0.3}],
        }
        path = self._write_yaml(data, tmp_path)
        spec = NNModelSpec.from_yaml(path)
        assert spec.layers[0].params == {"kernel_size": 3}

    def test_from_yaml_default_path_loads_configs(self):
        """from_yaml() with default path reads configs/nn_spec.yaml."""
        # Only run if the file exists (it should after Task 02 implementation)
        config_path = "configs/nn_spec.yaml"
        if not Path(config_path).exists():
            pytest.skip("configs/nn_spec.yaml not yet created")
        spec = NNModelSpec.from_yaml(config_path)
        assert isinstance(spec, NNModelSpec)
        assert len(spec.targets) >= 1


# ---------------------------------------------------------------------------
# NNModelSpec.default()
# ---------------------------------------------------------------------------


class TestDefaultClassmethod:
    """NNModelSpec.default() must return a minimal but valid spec."""

    def test_returns_nn_model_spec(self):
        spec = NNModelSpec.default()
        assert isinstance(spec, NNModelSpec)

    def test_has_valid_name(self):
        spec = NNModelSpec.default()
        assert isinstance(spec.name, str) and spec.name

    def test_timeframes_contains_15(self):
        spec = NNModelSpec.default()
        assert 15 in spec.timeframes

    def test_has_at_least_one_indicator(self):
        spec = NNModelSpec.default()
        assert len(spec.indicators) >= 1

    def test_has_at_least_one_layer(self):
        spec = NNModelSpec.default()
        assert len(spec.layers) >= 1
        assert all(isinstance(l, LayerSpec) for l in spec.layers)

    def test_has_at_least_one_target(self):
        spec = NNModelSpec.default()
        assert len(spec.targets) >= 1
        assert all(isinstance(t, TargetSpec) for t in spec.targets)

    def test_has_direction_target_with_label_tf_15(self):
        spec = NNModelSpec.default()
        direction_targets = [t for t in spec.targets if t.kind == "direction"]
        assert len(direction_targets) >= 1
        assert direction_targets[0].label_tf == 15

    def test_spec_hash_is_stable(self):
        s1 = NNModelSpec.default()
        s2 = NNModelSpec.default()
        assert s1.spec_hash == s2.spec_hash

    def test_history_points_positive(self):
        spec = NNModelSpec.default()
        assert spec.history_points > 0

    def test_grouping_is_grouping_spec(self):
        spec = NNModelSpec.default()
        assert isinstance(spec.grouping, GroupingSpec)


# ---------------------------------------------------------------------------
# TargetSpec.out_columns()
# ---------------------------------------------------------------------------


class TestTargetSpecOutColumns:
    """out_columns() must return the correct timeframe-agnostic column names."""

    def test_direction_single_horizon(self):
        t = TargetSpec(name="dir15", kind="direction", horizons=[1])
        cols = t.out_columns()
        assert cols == [
            "nn_res_dir15_prob_up",
            "nn_res_dir15_prob_neutral",
            "nn_res_dir15_prob_down",
        ]

    def test_direction_multi_horizon(self):
        t = TargetSpec(name="dir15", kind="direction", horizons=[1, 3])
        cols = t.out_columns()
        assert cols == [
            "nn_res_dir15_h1_prob_up",
            "nn_res_dir15_h1_prob_neutral",
            "nn_res_dir15_h1_prob_down",
            "nn_res_dir15_h3_prob_up",
            "nn_res_dir15_h3_prob_neutral",
            "nn_res_dir15_h3_prob_down",
        ]

    def test_label_single_horizon(self):
        t = TargetSpec(name="lbl15", kind="label", horizons=[1])
        cols = t.out_columns()
        assert cols == ["nn_res_lbl15_prob"]

    def test_label_multi_horizon(self):
        t = TargetSpec(name="lbl15", kind="label", horizons=[1, 5])
        cols = t.out_columns()
        assert cols == [
            "nn_res_lbl15_h1_prob",
            "nn_res_lbl15_h5_prob",
        ]

    def test_regression_single_horizon(self):
        t = TargetSpec(name="ret1", kind="regression", horizons=[1])
        cols = t.out_columns()
        assert cols == ["nn_res_ret1"]

    def test_regression_multi_horizon(self):
        t = TargetSpec(name="ret", kind="regression", horizons=[1, 3, 5])
        cols = t.out_columns()
        assert cols == [
            "nn_res_ret_h1",
            "nn_res_ret_h3",
            "nn_res_ret_h5",
        ]

    def test_direction_horizon_1_no_h_suffix(self):
        """Single horizon [1] must NOT add _h1 suffix."""
        t = TargetSpec(name="dir15", kind="direction", horizons=[1])
        cols = t.out_columns()
        for col in cols:
            assert "_h1" not in col

    def test_regression_horizon_1_no_h_suffix(self):
        """Single horizon [1] must NOT add _h1 suffix."""
        t = TargetSpec(name="ret", kind="regression", horizons=[1])
        cols = t.out_columns()
        assert cols == ["nn_res_ret"]
        assert "_h1" not in cols[0]

    def test_out_columns_is_list(self):
        t = TargetSpec(name="dir15", kind="direction", horizons=[1])
        assert isinstance(t.out_columns(), list)
