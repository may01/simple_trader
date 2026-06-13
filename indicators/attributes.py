"""indicators.attributes — DataAttributes (NN normalisation stats persistence)."""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

class DataAttributes:
    """Compute and persist dataset statistics for classification and normalisation.

    Two types of stats are managed:
    - ``rsi_classification.json``: RSI mean/std per TF, used by classification fields.
    - ``diff_stats.pkl``: Price diff mean/std per TF, used by target fields.
    - ``column_stats``: per-column mean/std for NN feature normalisation (in-memory).
    """

    _STAT_TFS: list[int] = [15, 60, 240, 1440]

    def __init__(self) -> None:
        self.column_stats: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public compute entry-point
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> None:
        """Compute and save all stats files if absent. Idempotent.

        Calls ``_compute_rsi_classification`` if ``rsi_classification.json``
        is absent, ``_compute_diff_stats`` if ``diff_stats.pkl`` is absent,
        and ``_compute_indicator_stats`` if ``indicator_stats.json`` is absent.

        Args:
            df: Wide DataFrame with ``{tf}_rsi_ma8`` and ``{tf}_is_closed``
                columns.
        """
        from helpers import stats_folder  # local import: reads env at call time
        base = stats_folder()
        os.makedirs(base, exist_ok=True)

        rsi_path = base + "rsi_classification.json"
        if not os.path.exists(rsi_path):
            self._compute_rsi_classification(df)

        diff_path = base + "diff_stats.pkl"
        if not os.path.exists(diff_path):
            self._compute_diff_stats(df)

        indicator_path = base + "indicator_stats.json"
        if not os.path.exists(indicator_path):
            self._compute_indicator_stats(df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rsi_classification(self, df: pd.DataFrame) -> None:
        """Compute per-TF rsi_ma8 level and diff stats for closed-candle rows.

        Each entry carries ``mean``/``std`` of ``{tf}_rsi_ma8`` (zone_class
        thresholds) and ``diff_mean``/``diff_std`` of ``{tf}_rsi_ma8_diff``
        (move_class thresholds). TFs with fewer than 2 valid rows are skipped
        entirely — the file never contains NaN; classification fields fall
        back to the nearest available TF.

        Saves to ``stats_folder() + 'rsi_classification.json'``.
        """
        from helpers import stats_folder
        base = stats_folder()
        os.makedirs(base, exist_ok=True)

        result: dict = {}
        for tf in self._STAT_TFS:
            closed_col = f"{tf}_is_closed"
            rsi_col = f"{tf}_rsi_ma8"
            diff_col = f"{tf}_rsi_ma8_diff"
            if closed_col not in df.columns or rsi_col not in df.columns:
                continue
            closed = df[df[closed_col] == True]  # noqa: E712
            levels = closed[rsi_col].dropna()
            diffs = (
                closed[diff_col].dropna()
                if diff_col in df.columns
                else levels.diff().dropna()
            )
            if len(levels) < 2 or len(diffs) < 2:
                continue
            result[str(tf)] = {
                "mean": float(levels.mean()),
                "std": float(levels.std()),
                "diff_mean": float(diffs.mean()),
                "diff_std": float(diffs.std()),
            }

        out_path = base + "rsi_classification.json"
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as fh:
            json.dump(result, fh)
        os.rename(tmp_path, out_path)

    def _compute_diff_stats(self, df: pd.DataFrame) -> None:
        """Compute price differential stats per TF.

        For each TF in ``_STAT_TFS``, computes mean and std of the percentage
        close change over closed-candle rows.  Saves to
        ``stats_folder() + 'diff_stats.pkl'``.
        """
        from helpers import stats_folder
        base = stats_folder()
        os.makedirs(base, exist_ok=True)

        result: dict = {}
        for tf in self._STAT_TFS:
            closed_col = f"{tf}_is_closed"
            close_col = f"{tf}_close"
            if closed_col not in df.columns or close_col not in df.columns:
                continue
            closed_close = df[df[closed_col] == True][close_col].dropna()  # noqa: E712
            pct_changes = closed_close.pct_change().dropna()
            result[str(tf)] = {
                "mean_diff": float(pct_changes.mean()),
                "std_diff": float(pct_changes.std()),
            }

        out_path = base + "diff_stats.pkl"
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as fh:
            pickle.dump(result, fh)
        os.rename(tmp_path, out_path)

    def _compute_indicator_stats(self, df: pd.DataFrame) -> None:
        """Compute per-TF distance stats over closed-candle rows.

        For each TF in ``_STAT_TFS``:
        - ``rsi_14`` / ``cci_14``:  mean/std of the raw indicator values
        - ``rsi_14_minus_rsi_ma8``: mean/std of ``{tf}_rsi_14 − {tf}_rsi_ma8``
        - ``cci_diff``:             mean/std of ``{tf}_cci_diff``
        - ``vol_minus_vol_ma_20``:  mean/std of ``{tf}_volume − {tf}_vol_ma_20``
        - ``diff_prc_std_{src}``:   mean/std of
          ``{tf}_{src}_diff_prc − {tf}_{src}_diff_prc_rm_6``
          for src in close/high/low

        Groups whose source columns are missing for a TF are omitted.
        Saves to ``stats_folder() + 'indicator_stats.json'``.
        """
        from helpers import stats_folder
        base = stats_folder()
        os.makedirs(base, exist_ok=True)

        result: dict = {}
        for tf in self._STAT_TFS:
            closed_col = f"{tf}_is_closed"
            if closed_col not in df.columns:
                continue
            closed = df[df[closed_col] == True]  # noqa: E712
            tf_stats: dict = {}

            specs = {
                "rsi_14": (f"{tf}_rsi_14", None),
                "cci_14": (f"{tf}_cci_14", None),
                "rsi_14_minus_rsi_ma8": (f"{tf}_rsi_14", f"{tf}_rsi_ma8"),
                "cci_diff": (f"{tf}_cci_diff", None),
                "vol_minus_vol_ma_20": (f"{tf}_volume", f"{tf}_vol_ma_20"),
            }
            for src in ["close", "high", "low"]:
                specs[f"diff_prc_std_{src}"] = (
                    f"{tf}_{src}_diff_prc",
                    f"{tf}_{src}_diff_prc_rm_6",
                )
            for key, (col, minus_col) in specs.items():
                if col not in df.columns:
                    continue
                if minus_col is not None:
                    if minus_col not in df.columns:
                        continue
                    series = (closed[col] - closed[minus_col]).dropna()
                else:
                    series = closed[col].dropna()
                tf_stats[key] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                }

            result[str(tf)] = tf_stats

        out_path = base + "indicator_stats.json"
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as fh:
            json.dump(result, fh)
        os.rename(tmp_path, out_path)

    # ------------------------------------------------------------------
    # Class-level loaders
    # ------------------------------------------------------------------

    @classmethod
    def load_rsi_classification(cls) -> dict:
        """Load ``rsi_classification.json`` from ``stats_folder()``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        from helpers import stats_folder
        path = stats_folder() + "rsi_classification.json"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"rsi_classification.json not found at: {path}"
            )
        with open(path, "r") as fh:
            return json.load(fh)

    @classmethod
    def load_diff_stats(cls) -> dict:
        """Load ``diff_stats.pkl`` from ``stats_folder()``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        from helpers import stats_folder
        path = stats_folder() + "diff_stats.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"diff_stats.pkl not found at: {path}"
            )
        with open(path, "rb") as fh:
            return pickle.load(fh)

    @classmethod
    def load_indicator_stats(cls) -> dict:
        """Load ``indicator_stats.json`` from ``stats_folder()``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        from helpers import stats_folder
        path = stats_folder() + "indicator_stats.json"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"indicator_stats.json not found at: {path}"
            )
        with open(path, "r") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # NN column stats
    # ------------------------------------------------------------------

    def compute_nn_stats(self, df: pd.DataFrame, feature_cols: list) -> None:
        """Compute mean/std for each feature column using only closed-candle rows.

        The TF is parsed from the column name prefix (e.g. ``"15_nn_rsi_ma8"``
        → tf=15).  Only rows where ``{tf}_is_closed == True`` are used.

        Args:
            df:           Wide DataFrame.
            feature_cols: List of column names to compute stats for.
        """
        for col in feature_cols:
            # Parse TF from column prefix: "15_something" → tf=15
            parts = col.split("_", 1)
            try:
                tf = int(parts[0])
            except (ValueError, IndexError):
                # Cannot parse TF — fall back to all rows
                series = df[col].dropna()
            else:
                closed_col = f"{tf}_is_closed"
                if closed_col in df.columns:
                    series = df[df[closed_col] == True][col].dropna()  # noqa: E712
                else:
                    series = df[col].dropna()

            self.column_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
            }

    def get_stats(self, col: str) -> tuple:
        """Return ``(mean, std)`` for *col*.

        Raises:
            KeyError: If *col* is not in ``column_stats``.
        """
        if col not in self.column_stats:
            raise KeyError(f"Column '{col}' not found in column_stats")
        entry = self.column_stats[col]
        return float(entry["mean"]), float(entry["std"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle self to *path* using an atomic write (write .tmp then rename).

        Args:
            path: Destination file path.
        """
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh)
        os.rename(tmp, path)

    @classmethod
    def load(cls, path: str) -> "DataAttributes":
        """Unpickle and return a :class:`DataAttributes` instance from *path*.

        Args:
            path: Source file path.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)


