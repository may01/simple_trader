"""indicators.registry — name → field-factory mapping loaded by Indicators.

Registry keys equal the field's derived name (which carries its parameters),
so the produced column is always ``{tf}_{key}``. Every factory forwards
``cfg.params`` from indicators_config.yaml — an empty params dict yields the
defaults encoded in each field class.
"""

from __future__ import annotations

from .library.momentum import *  # noqa: F401,F403
from .library.trend import *  # noqa: F401,F403
from .library.oscillators import *  # noqa: F401,F403
from .library.volatility import *  # noqa: F401,F403
from .library.volume import *  # noqa: F401,F403
from .library.price_derivatives import *  # noqa: F401,F403
from .library.classification import *  # noqa: F401,F403
from .library.targets import *  # noqa: F401,F403
from .library.trend_flags import *  # noqa: F401,F403
from .library.nn_features import *  # noqa: F401,F403

_FIELD_REGISTRY: dict[str, object] = {
    # Momentum
    "rsi_14": lambda cfg: RSI14Field(**cfg.params),
    "rsi_ma8": lambda cfg: RSI_MAField(**{"length": 8, **cfg.params}),
    "rsi_ma12": lambda cfg: RSI_MAField(**{"length": 12, **cfg.params}),
    "rsi_ma24": lambda cfg: RSI_MAField(**{"length": 24, **cfg.params}),
    "rsi_ma8_diff": lambda cfg: RSI_MA_DiffField(**{"source_ma": "rsi_ma8", **cfg.params}),
    "rsi_ma12_diff": lambda cfg: RSI_MA_DiffField(**{"source_ma": "rsi_ma12", **cfg.params}),
    "rsi_ma24_diff": lambda cfg: RSI_MA_DiffField(**{"source_ma": "rsi_ma24", **cfg.params}),
    # Trend
    "ema_7": lambda cfg: EMAField(**{"length": 7, **cfg.params}),
    "ema_14": lambda cfg: EMAField(**{"length": 14, **cfg.params}),
    "ema_25": lambda cfg: EMAField(**{"length": 25, **cfg.params}),
    "ema_50": lambda cfg: EMAField(**{"length": 50, **cfg.params}),
    "ema_100": lambda cfg: EMAField(**{"length": 100, **cfg.params}),
    "macd_12_26_9": lambda cfg: MACDField(**cfg.params),
    "macd_signal_12_26_9": lambda cfg: MACDSignalField(**cfg.params),
    "macd_hist_12_26_9": lambda cfg: MACDHistField(**cfg.params),
    "macd_5_13_9": lambda cfg: MACDFastField(**cfg.params),
    "macd_signal_5_13_9": lambda cfg: MACDFastSignalField(**cfg.params),
    "adx_14": lambda cfg: ADXField(**cfg.params),
    # Oscillators
    "sar_002_02": lambda cfg: SARField(**cfg.params),
    "cci_14": lambda cfg: CCI14Field(**cfg.params),
    "cci_14_ma_5": lambda cfg: CCI_MAField(**cfg.params),
    "cci_diff": lambda cfg: CCIDiffField(**cfg.params),
    # Volatility
    "atr_14": lambda cfg: ATR14Field(**cfg.params),
    "natr_14": lambda cfg: NATR14Field(**cfg.params),
    "atr_14_ma_5": lambda cfg: ATR_MAField(**cfg.params),
    "natr_14_ma_5": lambda cfg: NATR_MAField(**cfg.params),
    "bb_upper_20_2": lambda cfg: BollingerUpperField(**cfg.params),
    "bb_middle_20_2": lambda cfg: BollingerMiddleField(**cfg.params),
    "bb_lower_20_2": lambda cfg: BollingerLowerField(**cfg.params),
    "bb_upper_10_15": lambda cfg: BollingerFastUpperField(**cfg.params),
    "bb_lower_10_15": lambda cfg: BollingerFastLowerField(**cfg.params),
    "bb_upper_20_3": lambda cfg: BollingerWideUpperField(**cfg.params),
    "bb_lower_20_3": lambda cfg: BollingerWideLowerField(**cfg.params),
    # Volume
    "vol_ma_20": lambda cfg: VolMAField(**cfg.params),
    "vol_buy_ma_20": lambda cfg: VolBuyMAField(**cfg.params),
    "vol_sell_ma_20": lambda cfg: VolSellMAField(**cfg.params),
    # Price derivatives
    "close_diff_prc": lambda cfg: CloseDiffPrcField(**cfg.params),
    "close_diff_prc_rm_6": lambda cfg: CloseDiffPrcRMField(**cfg.params),
    "close_diff_prc_rm_6_mean_above": lambda cfg: CloseDiffPrcRMMeanAboveField(**cfg.params),
    "close_diff_prc_rm_6_mean_below": lambda cfg: CloseDiffPrcRMMeanBelowField(**cfg.params),
    "high_diff_prc": lambda cfg: HighDiffPrcField(**cfg.params),
    "high_diff_prc_rm_6": lambda cfg: HighDiffPrcRMField(**cfg.params),
    "high_diff_prc_rm_6_mean_above": lambda cfg: HighDiffPrcRMMeanAboveField(**cfg.params),
    "high_diff_prc_rm_6_mean_below": lambda cfg: HighDiffPrcRMMeanBelowField(**cfg.params),
    "low_diff_prc": lambda cfg: LowDiffPrcField(**cfg.params),
    "low_diff_prc_rm_6": lambda cfg: LowDiffPrcRMField(**cfg.params),
    "low_diff_prc_rm_6_mean_above": lambda cfg: LowDiffPrcRMMeanAboveField(**cfg.params),
    "low_diff_prc_rm_6_mean_below": lambda cfg: LowDiffPrcRMMeanBelowField(**cfg.params),
    "close_diff_prc_rm_6_std_above": lambda cfg: CloseDiffPrcRMStdAboveField(**cfg.params),
    "close_diff_prc_rm_6_std_below": lambda cfg: CloseDiffPrcRMStdBelowField(**cfg.params),
    "high_diff_prc_rm_6_std_above": lambda cfg: HighDiffPrcRMStdAboveField(**cfg.params),
    "high_diff_prc_rm_6_std_below": lambda cfg: HighDiffPrcRMStdBelowField(**cfg.params),
    "low_diff_prc_rm_6_std_above": lambda cfg: LowDiffPrcRMStdAboveField(**cfg.params),
    "low_diff_prc_rm_6_std_below": lambda cfg: LowDiffPrcRMStdBelowField(**cfg.params),
    # Classification (stats-driven; no numeric params)
    "move_class": lambda cfg: MoveClassField(),
    "zone_class": lambda cfg: ZoneClassField(),
    "over_low": lambda cfg: OverLowField(),
    "over_high": lambda cfg: OverHighField(),
    # Targets (stats-driven; no numeric params)
    "tgt_long": lambda cfg: TgtLongField(),
    "sl_long": lambda cfg: SLLongField(),
    "tgt_short": lambda cfg: TgtShortField(),
    "sl_short": lambda cfg: SLShortField(),
    "ZB": lambda cfg: ZBField(),
    "ZS": lambda cfg: ZSField(),
    # Trend flags
    "trend_up_50": lambda cfg: TrendUpField(**cfg.params),
    "trend_down_50": lambda cfg: TrendDownField(**cfg.params),
    # NN features
    "nn_rsi_ma8_norm_mean_20": lambda cfg: NNRSINormField(**cfg.params),
    "nn_close_diff_atr_14_ma_5": lambda cfg: NNCloseDiffATRField(**cfg.params),
}
