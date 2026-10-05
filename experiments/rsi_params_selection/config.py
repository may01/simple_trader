"""Grid constants + dataset paths for the RSI parameters selection experiment.

Spec: external/docs/superpowers/specs/2026-08-06-rsi-parameters-selection-design.md
Measurement-only harness — never imported by production code.
"""

VOL = "/media/om/Alexandria/simple_trader/simple_trader_vol_long/train"
TWOY_DIR = f"{VOL}/2y_link_usdt"
OOS_PATH = f"{VOL}/oos2m_link_usdt/df_with_indicators.pkl"
CONFIG_PATH = "configs/indicators_config.yaml"

TFS = [15, 60, 240]
WINDOWS = [8, 12, 24]
TECHNIQUES = ["quantile", "sym0", "zscore"]
CLASS_COUNTS = [5, 7]

QUANTILE_CUTS = {5: [10, 30, 70, 90], 7: [2, 10, 30, 70, 90, 98]}
SYM0_K = {5: [0.3, 1.0], 7: [0.3, 1.0, 2.0]}
ZSCORE_K = {5: [0.5, 1.0], 7: [0.5, 1.0, 2.0]}

# Per-TF label params baked into column names (configs/indicators_config.yaml labels:)
_X = {15: "0p3", 60: "0p2", 240: "0p1"}
_Y = {15: "0p2", 60: "0p1", 240: "0p1"}


def label_cols(tf: int) -> dict[str, str]:
    """Map short label key -> wide-frame column name for one TF."""
    out = {}
    for n in (1, 2):
        out[f"plain_n{n}_long"] = f"{tf}_plong_n{n}_m1_x{_X[tf]}"
        out[f"plain_n{n}_short"] = f"{tf}_pshort_n{n}_m1_x{_X[tf]}"
        out[f"strict_n{n}_long"] = f"{tf}_pslong_n{n}_m1_x{_X[tf]}_l15_y{_Y[tf]}"
        out[f"strict_n{n}_short"] = f"{tf}_psshort_n{n}_m1_x{_X[tf]}_l15_y{_Y[tf]}"
    return out


# (kind, n) pairs used by metrics for long-vs-short separation
LABEL_PAIRS = [("plain", 1), ("plain", 2), ("strict", 1), ("strict", 2)]
