"""indicators.library.price_derivatives — close/high/low percentage-diff family.

Four layers per source column (close, high, low):
  {src}_diff_prc            — 1-step percentage change
  {src}_diff_prc_rm_{w}     — rolling mean over w
  {src}_diff_prc_rm_{w}_mean_above / _mean_below
                            — rolling mean of positive/negative values over w
  {src}_diff_prc_rm_{w}_std_above / _std_below
                            — rolling mean ± rolling std of the diff over w
"""

from __future__ import annotations

import pandas as pd

from ..framework import IndicatorField


class _DiffPrcBase(IndicatorField):
    """1-step percentage change of a price column."""

    group = "price_derivatives"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    _source: str  # "close" | "high" | "low"

    def __init__(self) -> None:
        self.params: dict = {}
        self.name = f"{self._source}_diff_prc"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_{self._source}"]
        return (series - series.shift(1)) / series.shift(1) * 100


class _DiffPrcRMBase(IndicatorField):
    """Rolling mean of a diff_prc column over a window."""

    group = "price_derivatives"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    _source: str

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.params = {"window": window}
        self.name = f"{self._source}_diff_prc_rm_{window}"
        self.dependencies = [f"{self._source}_diff_prc"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_{self._source}_diff_prc"].rolling(self.window).mean()


class _DiffPrcRMMeanSideBase(IndicatorField):
    """Rolling mean of one-signed values of a diff_prc_rm column."""

    group = "price_derivatives"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    _source: str
    _above: bool

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.params = {"window": window}
        side = "above" if self._above else "below"
        self.name = f"{self._source}_diff_prc_rm_{window}_mean_{side}"
        self.dependencies = [f"{self._source}_diff_prc_rm_{window}"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_{self._source}_diff_prc_rm_{self.window}"]
        if self._above:
            return series.rolling(self.window).apply(
                lambda x: x[x > 0].mean() if (x > 0).any() else 0.0, raw=True
            )
        return series.rolling(self.window).apply(
            lambda x: x[x < 0].mean() if (x < 0).any() else 0.0, raw=True
        )


class CloseDiffPrcField(_DiffPrcBase):
    _source = "close"


class CloseDiffPrcRMField(_DiffPrcRMBase):
    _source = "close"


class CloseDiffPrcRMMeanAboveField(_DiffPrcRMMeanSideBase):
    _source = "close"
    _above = True


class CloseDiffPrcRMMeanBelowField(_DiffPrcRMMeanSideBase):
    _source = "close"
    _above = False


class HighDiffPrcField(_DiffPrcBase):
    _source = "high"


class HighDiffPrcRMField(_DiffPrcRMBase):
    _source = "high"


class HighDiffPrcRMMeanAboveField(_DiffPrcRMMeanSideBase):
    _source = "high"
    _above = True


class HighDiffPrcRMMeanBelowField(_DiffPrcRMMeanSideBase):
    _source = "high"
    _above = False


class LowDiffPrcField(_DiffPrcBase):
    _source = "low"


class LowDiffPrcRMField(_DiffPrcRMBase):
    _source = "low"


class LowDiffPrcRMMeanAboveField(_DiffPrcRMMeanSideBase):
    _source = "low"
    _above = True


class LowDiffPrcRMMeanBelowField(_DiffPrcRMMeanSideBase):
    _source = "low"
    _above = False


class _DiffPrcRMStdSideBase(IndicatorField):
    """Rolling mean ± rolling std band of a diff_prc column."""

    group = "price_derivatives"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    _source: str
    _above: bool

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.params = {"window": window}
        side = "above" if self._above else "below"
        self.name = f"{self._source}_diff_prc_rm_{window}_std_{side}"
        self.dependencies = [
            f"{self._source}_diff_prc",
            f"{self._source}_diff_prc_rm_{window}",
        ]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        diff = df[f"{tf}_{self._source}_diff_prc"]
        rm = df[f"{tf}_{self._source}_diff_prc_rm_{self.window}"]
        std = diff.rolling(self.window).std()
        return rm + std if self._above else rm - std


class CloseDiffPrcRMStdAboveField(_DiffPrcRMStdSideBase):
    _source = "close"
    _above = True


class CloseDiffPrcRMStdBelowField(_DiffPrcRMStdSideBase):
    _source = "close"
    _above = False


class HighDiffPrcRMStdAboveField(_DiffPrcRMStdSideBase):
    _source = "high"
    _above = True


class HighDiffPrcRMStdBelowField(_DiffPrcRMStdSideBase):
    _source = "high"
    _above = False


class LowDiffPrcRMStdAboveField(_DiffPrcRMStdSideBase):
    _source = "low"
    _above = True


class LowDiffPrcRMStdBelowField(_DiffPrcRMStdSideBase):
    _source = "low"
    _above = False
