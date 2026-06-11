"""indicators.registry — name → field-factory mapping loaded by Indicators."""

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

# ===========================================================================
# Factory registry — maps field names → factory callables
# ===========================================================================

_FIELD_REGISTRY: dict[str, object] = {
    # Momentum
    "rsi_14": lambda cfg: RSI14Field(),
    "rsi_ma8": lambda cfg: RSI_MAField("rsi_14", 8, "rsi_ma8"),
    "rsi_ma12": lambda cfg: RSI_MAField("rsi_14", 12, "rsi_ma12"),
    "rsi_ma24": lambda cfg: RSI_MAField("rsi_14", 24, "rsi_ma24"),
    "rsi_ma8_diff": lambda cfg: RSI_MA_DiffField("rsi_ma8", "rsi_ma8_diff"),
    "rsi_ma12_diff": lambda cfg: RSI_MA_DiffField("rsi_ma12", "rsi_ma12_diff"),
    "rsi_ma24_diff": lambda cfg: RSI_MA_DiffField("rsi_ma24", "rsi_ma24_diff"),
    # Trend
    "ema_7": lambda cfg: EMAField(7, "ema_7"),
    "ema_14": lambda cfg: EMAField(14, "ema_14"),
    "ema_25": lambda cfg: EMAField(25, "ema_25"),
    "ema_50": lambda cfg: EMAField(50, "ema_50"),
    "ema_100": lambda cfg: EMAField(100, "ema_100"),
    "macd": lambda cfg: MACDField(),
    "macd_signal": lambda cfg: MACDSignalField(),
    "macd_hist": lambda cfg: MACDHistField(),
    "macd_fast": lambda cfg: MACDFastField(),
    "macd_fast_signal": lambda cfg: MACDFastSignalField(),
    # Oscillators
    "sar": lambda cfg: SARField(),
    "cci_14": lambda cfg: CCI14Field(),
    "cci_ma": lambda cfg: CCI_MAField(),
    # Volatility
    "atr_14": lambda cfg: ATR14Field(),
    "natr_14": lambda cfg: NATR14Field(),
    "atr_ma": lambda cfg: ATR_MAField(),
    "bb_upper": lambda cfg: BollingerUpperField(),
    "bb_middle": lambda cfg: BollingerMiddleField(),
    "bb_lower": lambda cfg: BollingerLowerField(),
    "bb_fast_upper": lambda cfg: BollingerFastUpperField(),
    "bb_fast_lower": lambda cfg: BollingerFastLowerField(),
    "bb_wide_upper": lambda cfg: BollingerWideUpperField(),
    "bb_wide_lower": lambda cfg: BollingerWideLowerField(),
    # Volume
    "vol_ma": lambda cfg: VolMAField(),
    "vol_buy_ma": lambda cfg: VolBuyMAField(),
    "vol_sell_ma": lambda cfg: VolSellMAField(),
    # Price derivatives
    "close_diff_prc": lambda cfg: CloseDiffPrcField(),
    "close_diff_prc_rm": lambda cfg: CloseDiffPrcRMField(),
    "close_diff_prc_rm_mean_above": lambda cfg: CloseDiffPrcRMMeanAboveField(),
    "close_diff_prc_rm_mean_below": lambda cfg: CloseDiffPrcRMMeanBelowField(),
    "high_diff_prc": lambda cfg: HighDiffPrcField(),
    "high_diff_prc_rm": lambda cfg: HighDiffPrcRMField(),
    "high_diff_prc_rm_mean_above": lambda cfg: HighDiffPrcRMMeanAboveField(),
    "high_diff_prc_rm_mean_below": lambda cfg: HighDiffPrcRMMeanBelowField(),
    "low_diff_prc": lambda cfg: LowDiffPrcField(),
    "low_diff_prc_rm": lambda cfg: LowDiffPrcRMField(),
    "low_diff_prc_rm_mean_above": lambda cfg: LowDiffPrcRMMeanAboveField(),
    "low_diff_prc_rm_mean_below": lambda cfg: LowDiffPrcRMMeanBelowField(),
    # Classification
    "move_class": lambda cfg: MoveClassField(),
    "zone_class": lambda cfg: ZoneClassField(),
    "over_low": lambda cfg: OverLowField(),
    "over_high": lambda cfg: OverHighField(),
    # Targets
    "tgt_long": lambda cfg: TgtLongField(),
    "sl_long": lambda cfg: SLLongField(),
    "tgt_short": lambda cfg: TgtShortField(),
    "sl_short": lambda cfg: SLShortField(),
    "ZB": lambda cfg: ZBField(),
    "ZS": lambda cfg: ZSField(),
    # Trend flags
    "trend_up": lambda cfg: TrendUpField(),
    "trend_down": lambda cfg: TrendDownField(),
    # NN features
    "nn_rsi_ma8_norm_mean": lambda cfg: NNRSINormField(),
    "nn_close_diff_atr_ma": lambda cfg: NNCloseDiffATRField(),
}
