#!/usr/bin/env python3
"""One-shot: append the 4 TF5 profit_strict label columns to df_with_indicators.pkl.

Avoids a full data re-prep. Run once per dataset via NN_TRAIN_ENV:

    NN_TRAIN_ENV=configs/nn_train_dataset_2y.env docker compose run --rm nn-train \
        python3 scripts/nn_ps5_materialize.py
    NN_TRAIN_ENV=configs/oos2m_dataset.env      docker compose run --rm nn-train \
        python3 scripts/nn_ps5_materialize.py
"""
import pandas as pd

from helpers import wide_df_path
from indicators.labels import add_profit_strict_labels

_PARAMS = dict(m=1, x=0.3, l=15, y=0.2)  # mirror TF15; atr_period/ma_length defaults
_EXPECTED = [
    "5_pslong_n1_m1_x0p3_l15_y0p2", "5_psshort_n1_m1_x0p3_l15_y0p2",
    "5_pslong_n2_m1_x0p3_l15_y0p2", "5_psshort_n2_m1_x0p3_l15_y0p2",
]


def add_ps5_labels(wide_df: pd.DataFrame) -> list[str]:
    """Append the 4 TF5 strict columns in place; idempotent. Returns their names."""
    if not set(_EXPECTED).issubset(wide_df.columns):
        for n in (1, 2):
            add_profit_strict_labels(wide_df, 5, n=n, **_PARAMS)
    return list(_EXPECTED)


def main() -> None:
    path = wide_df_path()
    df = pd.read_pickle(path)
    before = set(df.columns)
    add_ps5_labels(df)
    if set(df.columns) != before:
        df.to_pickle(path)
        print(f"[ps5] wrote {len(set(df.columns) - before)} columns -> {path}", flush=True)
    else:
        print(f"[ps5] columns already present -> {path} (no-op)", flush=True)


if __name__ == "__main__":
    main()
