import pytest
from nn.nn_model_spec import NNModelSpec
from nn.nn_dataset import _profit_long_col, _profit_short_col

SPEC_DIR = "configs/nn_specs/profit_strict"

CASES = [
    ("ps5_n1_long",  "long",  1, "5_pslong_n1_m1_x0p3_l15_y0p2"),
    ("ps5_n1_short", "short", 1, "5_psshort_n1_m1_x0p3_l15_y0p2"),
    ("ps5_n2_long",  "long",  2, "5_pslong_n2_m1_x0p3_l15_y0p2"),
    ("ps5_n2_short", "short", 2, "5_psshort_n2_m1_x0p3_l15_y0p2"),
]

@pytest.mark.parametrize("stem,side,horizon,column", CASES)
def test_ps5_spec_resolves_to_tf5_column(stem, side, horizon, column):
    spec = NNModelSpec.from_yaml(f"{SPEC_DIR}/{stem}.yaml")
    assert spec.name == f"nnfo_{stem}"
    tgt = spec.targets[0]
    assert tgt.strict is True
    assert tgt.label_tf == 5
    assert tgt.side == side
    assert tgt.horizons[0] == horizon
    resolver = _profit_long_col if side == "long" else _profit_short_col
    assert resolver(tgt, horizon) == column
