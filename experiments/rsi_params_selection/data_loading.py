"""Load per-TF closed-row frames (diffs + labels) from 2y parts and oos2m.

2y parts carry NO label columns — labels are computed per part in memory via
indicators.labels (vectorized). Per-part boundary tails yield NaN labels and
are dropped downstream (spec §5.2 accepted loss). oos2m has labels baked.
"""
import glob
import os

import pandas as pd

from . import config


def add_labels_for_tfs(df: pd.DataFrame, tfs, config_path: str = config.CONFIG_PATH) -> None:
    """Append profit-label columns for the given tfs, in place.

    Same loop as training/data_preparer._compute_profit_labels, restricted
    to tfs we score — parts are 43200x716, full label pass is wasted work.
    """
    from config_loader import load_labels_config
    from indicators.labels import add_profit_labels, add_profit_strict_labels

    for spec in load_labels_config(config_path):
        for tf in spec.tfs:
            if tf not in tfs:
                continue
            if spec.type == "profit":
                add_profit_labels(
                    df, tf, spec.n, spec.m, spec.x,
                    atr_period=spec.atr_period, ma_length=spec.ma_length,
                )
            else:  # profit_strict — validated by load_labels_config
                add_profit_strict_labels(
                    df, tf, spec.n, spec.m, spec.x, spec.l, spec.y,
                    atr_period=spec.atr_period, ma_length=spec.ma_length,
                )


def extract_closed(df: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Closed-candle rows for one TF, renamed to short keys.

    Columns: diff8/diff12/diff24 + the 8 label keys of config.label_cols(tf).
    """
    cols = {f"diff{w}": f"{tf}_rsi_ma{w}_diff" for w in config.WINDOWS}
    cols.update(config.label_cols(tf))
    mask = df[f"{tf}_is_closed"].astype(bool).to_numpy()
    out = pd.DataFrame(
        {short: df[wide].to_numpy()[mask] for short, wide in cols.items()},
        index=df.index[mask],
    )
    return out


def _part_paths(limit: int | None = None) -> list[str]:
    paths = sorted(glob.glob(
        os.path.join(config.TWOY_DIR, "df_with_indicators.part_*.pkl")))
    if not paths:
        raise FileNotFoundError(f"no 2y parts under {config.TWOY_DIR}")
    return paths[:limit] if limit else paths


def load_2y(tfs=tuple(config.TFS), limit: int | None = None) -> dict[int, pd.DataFrame]:
    """One concat'd closed-row frame per TF across 2y parts (labels computed)."""
    frames: dict[int, list] = {tf: [] for tf in tfs}
    for path in _part_paths(limit):
        df = pd.read_pickle(path)
        add_labels_for_tfs(df, list(tfs))
        for tf in tfs:
            frames[tf].append(extract_closed(df, tf))
        del df
    return {tf: pd.concat(parts) for tf, parts in frames.items()}


def load_oos(tfs=tuple(config.TFS)) -> dict[int, pd.DataFrame]:
    """Per-TF closed-row frames from oos2m (labels already baked)."""
    df = pd.read_pickle(config.OOS_PATH)
    return {tf: extract_closed(df, tf) for tf in tfs}
