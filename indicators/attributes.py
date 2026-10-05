"""indicators.attributes — DataAttributes (NN normalisation stats persistence)."""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

class DataAttributes:
    """Compute and persist dataset statistics for classification and normalisation.

    Stats managed (consumed by classification + target indicator fields):
    - ``rsi_classification.json``: RSI mean/std per TF, used by classification fields.
    - ``diff_stats.pkl``: Price diff mean/std per TF, used by target fields.
    - ``indicator_stats.json``: per-TF distance stats for classification fields.

    NN feature normalisation is NOT handled here — it lives in the dataset
    manifest (``NNDataset`` computes train-split winsorised stats and bundles
    them into the checkpoint). See DECISIONS-LOG D13.
    """

    _STAT_TFS: list[int] = [15, 60, 240, 1440]

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

        # side-stats companion depends on rsi_classification.json (cuts); compute after it
        side_path = base + "rsi_side_stats.json"
        if not os.path.exists(side_path) and os.path.exists(rsi_path):
            self._compute_rsi_side_stats(df)

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
            diff_std = float(diffs.std())
            zone_cuts = [float(v) for v in np.percentile(levels, [10, 30, 70, 90])]
            move_cuts = [m * diff_std for m in (-1.0, -0.3, 0.3, 1.0)]
            result[str(tf)] = {
                "mean": float(levels.mean()),
                "std": float(levels.std()),
                "diff_mean": float(diffs.mean()),
                "diff_std": diff_std,
                "zone_cuts": zone_cuts,
                "move_cuts": move_cuts,
            }

        out_path = base + "rsi_classification.json"
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as fh:
            json.dump(result, fh)
        os.rename(tmp_path, out_path)

    def _compute_rsi_side_stats(self, df: pd.DataFrame) -> None:
        """Per-TF, per-new-class long/short win-rate + lift (viewer tooltip source).

        For ``zone_class_q`` (rsi_ma8 vs ``zone_cuts``, classes 0..4) and
        ``move_class_sym0`` (rsi_ma8_diff vs ``move_cuts``, classes -2..2), tally
        over closed-candle rows the next-bar directional win-rate:
        ``long_winrate = P(fwd>0 | class)``, ``short_winrate = P(fwd<0 | class)``,
        lifts vs the TF base rate. Forward return is close-to-close over one closed
        bar of that TF (side-agnostic). Reads the cuts from the just-written
        ``rsi_classification.json``; empty classes carry zeros (never NaN).

        Saves to ``stats_folder() + 'rsi_side_stats.json'``.
        """
        from helpers import stats_folder
        base = stats_folder()
        os.makedirs(base, exist_ok=True)
        with open(base + "rsi_classification.json") as fh:
            rsi_stats = json.load(fh)

        fields = {
            "zone_class_q": ("_rsi_ma8", "zone_cuts", 0),
            "move_class_sym0": ("_rsi_ma8_diff", "move_cuts", -2),
        }

        result: dict = {}
        for tf in self._STAT_TFS:
            if str(tf) not in rsi_stats:
                continue
            closed_col = f"{tf}_is_closed"
            close_col = f"{tf}_close"
            if closed_col not in df.columns or close_col not in df.columns:
                continue
            closed = df[df[closed_col] == True]  # noqa: E712
            close = closed[close_col].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                fwd = np.log(np.concatenate([close[1:] / close[:-1], [np.nan]]))
            entry = rsi_stats[str(tf)]
            tf_out: dict = {}
            for fname, (suffix, cut_key, cbase) in fields.items():
                feat_col = f"{tf}{suffix}"
                if feat_col not in closed.columns:
                    continue
                feat = closed[feat_col].to_numpy(dtype=float)
                cuts = entry[cut_key]
                valid = ~np.isnan(fwd) & ~np.isnan(feat)
                y = fwd[valid]
                idx = np.digitize(feat[valid], cuts)  # 0..4
                base_long = float((y > 0).mean()) if len(y) else 0.0
                base_short = float((y < 0).mean()) if len(y) else 0.0
                cls_out: dict = {}
                for k in range(5):
                    sel = idx == k
                    n = int(sel.sum())
                    if n:
                        lw = float((y[sel] > 0).mean())
                        sw = float((y[sel] < 0).mean())
                    else:
                        lw = sw = 0.0
                    cls_out[str(cbase + k)] = {
                        "long_winrate": round(lw, 6),
                        "short_winrate": round(sw, 6),
                        "long_lift": round(lw - base_long, 6),
                        "short_lift": round(sw - base_short, 6),
                        "base_long": round(base_long, 6),
                        "n": n,
                    }
                tf_out[fname] = cls_out
            result[str(tf)] = tf_out

        out_path = base + "rsi_side_stats.json"
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


