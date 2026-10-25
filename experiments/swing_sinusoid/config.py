"""Constants + paths for the swing-sinusoid experiment.

Spec: external/docs/superpowers/experiment/swing_sinusoid.md
Plan: external/docs/superpowers/plans/2026-10-24-swing-sinusoid.md
Measurement-only harness — never imported by production code.
"""
import os

# Container paths (docker-compose.experiment.yml mounts the long volume here).
VOL = os.environ.get("SWING_VOL", "/trader_data_long/train")
DATASETS = {
    "2y": {"dir": f"{VOL}/2y_link_usdt", "parts": True},
    "oos2m": {"dir": f"{VOL}/oos2m_link_usdt", "parts": False},
}
SIDECAR = "swing_sinusoid"  # sub-dir per dataset for every artifact we write
CONFIG_PATH = "configs/indicators_config.yaml"

TFS = [15, 60, 240]

# Series fitted for swings (all closed-row, per tf). Keys are the short names
# used everywhere downstream; values describe how they are derived.
SERIES = [
    "rsi_ma6", "rsi_ma8", "cci_ma6", "cci_ma8",
    "close_m_ema25", "close_diff_prc", "close_diff_prc_rm_6",
]
RSI_MA_LENGTHS = (6, 8)
CCI_MA_LENGTHS = (6, 8)

# Raw wide-frame columns pulled per tf (before derivation).
RAW_COLS = ["close", "rsi_14", "cci_14", "ema_25", "close_diff_prc", "close_diff_prc_rm_6"]

# Per-TF label params baked into column names (configs/indicators_config.yaml labels:).
_X = {15: "0p3", 60: "0p2", 240: "0p1"}
_Y = {15: "0p2", 60: "0p1", 240: "0p1"}

# Chronological 2y split (locked decision 7).
TRAIN_END = "2024-06-30 23:59:59+00:00"

# Forward horizons in candles (locked decision 2).
HORIZONS = (1, 2, 3, 5)

# Fit windows swept (candles).
WINDOWS = (12, 16, 24, 32, 48)


def label_cols(tf: int) -> dict[str, str]:
    """Map short label key -> wide-frame column name for one TF."""
    out = {}
    for n in (1, 2):
        out[f"plain_n{n}_long"] = f"{tf}_plong_n{n}_m1_x{_X[tf]}"
        out[f"plain_n{n}_short"] = f"{tf}_pshort_n{n}_m1_x{_X[tf]}"
        out[f"strict_n{n}_long"] = f"{tf}_pslong_n{n}_m1_x{_X[tf]}_l15_y{_Y[tf]}"
        out[f"strict_n{n}_short"] = f"{tf}_psshort_n{n}_m1_x{_X[tf]}_l15_y{_Y[tf]}"
    return out


def sidecar_dir(dataset: str) -> str:
    return os.path.join(DATASETS[dataset]["dir"], SIDECAR)
