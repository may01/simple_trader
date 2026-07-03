import numpy as np
from indicators import DataAttributes
from nn.nn_dataset import NNDataset
from nn.nn_model import NNModel
from nn.nn_model_spec import LayerSpec
from tests.unit.nn.test_nn_dataset import make_wide_df, small_spec


def test_ragged_dataset_sizes_model_and_trains(tmp_path):
    # rsi_14 exists at 15 but NOT at 60 → ragged:
    #   tf15 → [15_logret, 15_rsi_14], tf60 → [60_logret]  (3 features total)
    # rows=300 so tf=60 has enough closed candles for history_points=4 (else all rows drop).
    df = make_wide_df(rows=300)
    spec = small_spec(
        timeframes=[15, 60],
        indicators=["logret", "rsi_14"],
        layers=[LayerSpec(kind="dense", units=8)],
        epochs=1,
        device="cpu",
    )
    ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
    assert ds.manifest["feature_cols"]["15"] == ["15_logret", "15_rsi_14"]
    assert ds.manifest["feature_cols"]["60"] == ["60_logret"]

    model = NNModel(spec)
    model.train(ds)
    assert model._n_features == 3                       # 2 + 1, NOT 2*2
    assert model.input_size == 3 * spec.history_points

    x = np.random.randn(model.input_size).astype("float32")
    out = model.run(x)
    assert out.shape[0] == model.output_size
