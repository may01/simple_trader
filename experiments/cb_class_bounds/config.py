"""Constants + paths for the cb_class_bounds experiment.

Spec: external/docs/superpowers/experiment/cb_class_bounds.md
Plan: external/docs/superpowers/plans/2026-10-25-cb-class-bounds.md
Measurement-only harness — never imported by production code.
"""
import os

VOL = os.environ.get("CBC_VOL", "/trader_data_long/train")
DATASETS = {
    "2y": {"dir": f"{VOL}/2y_link_usdt", "parts": True},
    "oos2m": {"dir": f"{VOL}/oos2m_link_usdt", "parts": False},
}
AZ_FRAME = f"{VOL}/2y_az_link_usdt/df_with_indicators.pkl"       # frozen-json training frame (gate) + replay minute frame
CB_2Y = f"{VOL}/2y_link_usdt/df_with_candle_bounds.pkl"
NC_2Y = f"{VOL}/2y_link_usdt/df_with_candle_bounds_nc.pkl"
SIDECAR = "cb_class_bounds"

TFS = (15, 60, 240)
SIDES = ("high", "low")
HIGHER = {15: (60, 240), 60: (240,), 240: ()}
FAMILIES = ("mc", "zc", "cz", "ec")
CLASS_VALUES = {"mc": tuple(range(-3, 4)), "zc": tuple(range(0, 5)), "cz": (-1, 0, 1), "ec": (-1, 0, 1)}

# Raw wide-frame columns pulled per tf at its closed rows (kernel.closed_frame inputs + class inputs).
RAW_OWN = ("open", "high", "low", "close", "rsi_14", "rsi_ma8", "macd_12_26_9", "macd_hist_12_26_9", "ema_25", "cci_14")
REF_OWN = ("move_class", "move_class_sym0", "zone_class", "ema_25_slope_class", "ema_25_diff_prc")   # optional, reference only
# Forming values of a higher tf read at the lower tf's closing minute.
RAW_HIGHER = ("rsi_ma8", "cci_14", "ema_25", "is_closed")
REF_HIGHER = ("move_class", "ema_25_slope_class")

# Chronological 2y split (locked decision 8).
TRAIN_END = "2024-06-30 23:59:59+00:00"
MIN_N_GRID = (100, 300, 1000)
ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)
SHUFFLE_SEEDS = (0, 1, 2)
BLOCK = "1D"


def sidecar_dir(dataset: str) -> str:
    return os.path.join(DATASETS[dataset]["dir"], SIDECAR)


def closed_path(dataset: str, tf: int) -> str:
    return os.path.join(sidecar_dir(dataset), f"closed_tf{tf}.pkl")
