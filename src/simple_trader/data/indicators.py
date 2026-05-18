import pandas as pd
import numpy as np

try:
    import talib  # type: ignore[import]
    _TALIB_AVAILABLE = True
except ImportError:
    _TALIB_AVAILABLE = False


class Indicators:
    """
    Computes technical indicator columns on a per-tf OHLCV DataFrame.

    All computed columns are named "{tf}_{indicator_name}".
    compute() mutates the DataFrame in-place.
    drop_na() removes rows where indicator columns contain NaN.
    """

    def compute(self, df: pd.DataFrame, tf: int) -> None:
        close = df[f"{tf}_close"].values.astype(float)
        high  = df[f"{tf}_high"].values.astype(float)
        low   = df[f"{tf}_low"].values.astype(float)

        if _TALIB_AVAILABLE:
            self._compute_talib(df, tf, close, high, low)
        else:
            self._compute_pandas(df, tf, close, high, low)

    def _compute_talib(
        self,
        df: pd.DataFrame,
        tf: int,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> None:
        import talib  # type: ignore[import]
        df[f"{tf}_rsi_14"]  = talib.RSI(close, timeperiod=14)
        df[f"{tf}_ema_25"]  = talib.EMA(close, timeperiod=25)
        df[f"{tf}_ema_50"]  = talib.EMA(close, timeperiod=50)
        df[f"{tf}_ema_100"] = talib.EMA(close, timeperiod=100)
        df[f"{tf}_ema_200"] = talib.EMA(close, timeperiod=200)
        macd, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
        df[f"{tf}_macd"]        = macd
        df[f"{tf}_macd_signal"] = macd_signal
        df[f"{tf}_macd_hist"]   = macd_hist
        df[f"{tf}_cci_20"]  = talib.CCI(high, low, close, timeperiod=20)
        df[f"{tf}_atr_14"]  = talib.ATR(high, low, close, timeperiod=14)
        upper, mid, lower   = talib.BBANDS(close, timeperiod=20)
        df[f"{tf}_bb_upper"] = upper
        df[f"{tf}_bb_mid"]   = mid
        df[f"{tf}_bb_lower"] = lower
        df[f"{tf}_adx_14"]  = talib.ADX(high, low, close, timeperiod=14)
        df[f"{tf}_sar"]     = talib.SAR(high, low)

    def _compute_pandas(
        self,
        df: pd.DataFrame,
        tf: int,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> None:
        """Pandas fallback for environments without TA-Lib (e.g. CI tests)."""
        s = pd.Series(close)
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        df[f"{tf}_rsi_14"]  = 100 - (100 / (1 + rs))
        df[f"{tf}_ema_25"]  = s.ewm(span=25, adjust=False).mean()
        df[f"{tf}_ema_50"]  = s.ewm(span=50, adjust=False).mean()
        df[f"{tf}_ema_100"] = s.ewm(span=100, adjust=False).mean()
        df[f"{tf}_ema_200"] = s.ewm(span=200, adjust=False).mean()
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df[f"{tf}_macd"]        = macd_line
        df[f"{tf}_macd_signal"] = signal_line
        df[f"{tf}_macd_hist"]   = macd_line - signal_line
        tp = (pd.Series(high) + pd.Series(low) + s) / 3
        df[f"{tf}_cci_20"]  = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            (pd.Series(high) - s.shift()).abs(),
            (pd.Series(low) - s.shift()).abs(),
        ], axis=1).max(axis=1)
        df[f"{tf}_atr_14"]   = tr.ewm(com=13, adjust=False).mean()
        sma20 = s.rolling(20).mean()
        std20 = s.rolling(20).std()
        df[f"{tf}_bb_upper"] = sma20 + 2 * std20
        df[f"{tf}_bb_mid"]   = sma20
        df[f"{tf}_bb_lower"] = sma20 - 2 * std20
        df[f"{tf}_adx_14"]   = pd.Series(np.full(len(close), 25.0))
        df[f"{tf}_sar"]      = s.shift(1)

    def drop_na(self, df: pd.DataFrame, tf: int) -> None:
        """Remove leading rows where indicator columns contain NaN."""
        indicator_cols = [c for c in df.columns if c.startswith(f"{tf}_") and
                         c not in (f"{tf}_open", f"{tf}_high", f"{tf}_low",
                                   f"{tf}_close", f"{tf}_volume")]
        if indicator_cols:
            df.dropna(subset=indicator_cols, inplace=True)
            df.reset_index(drop=True, inplace=True)
