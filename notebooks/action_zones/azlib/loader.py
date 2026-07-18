"""azlib/loader.py — Task 1: read-only wide-df access + profit label attachment.

Two responsibilities:

1. ``load_wide_df`` — reads the pair's ``df_with_indicators.pkl`` via the
   existing, env-driven ``helpers.wide_df_path()`` and returns it unchanged.
   Read-only by convention: callers must never write the returned (or a
   later labeled) frame back to that path. Labels added via ``add_labels``
   below are look-ahead and must stay in-memory only for the lifetime of a
   notebook/script run — persisting them would leak future information into
   ``df_with_indicators.pkl``, which every other consumer (live trading,
   indicator features, other NN pipelines) treats as feature data.

2. ``add_labels`` / ``label_col`` — a thin, dataclass-driven wrapper around
   the existing ``indicators.labels.add_profit_labels`` and
   ``add_profit_strict_labels`` functions (imported, not reimplemented).
   ``LabelParams`` bundles the knobs those two functions already take
   positionally; ``label_col`` reproduces the exact column-naming convention
   those functions use (via the shared ``indicators.labels._fmt`` helper) so
   callers can look up a label column without re-deriving the string format.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from helpers import wide_df_path
from indicators.labels import _fmt, add_profit_labels, add_profit_strict_labels


def load_wide_df() -> pd.DataFrame:
    """Return the pair's wide df, read straight off disk. Read-only.

    Path comes from ``helpers.wide_df_path()`` (env-driven: ROOT_FOLDER /
    DATA_ROOT / DATA_SET_NAME / PAIR). The returned frame is exactly what
    ``pd.read_pickle`` produces — no copying, filtering, or mutation here.
    """
    return pd.read_pickle(wide_df_path())


@dataclass
class LabelParams:
    """Knobs for one (tf, direction-pair) set of profit labels.

    Mirrors the positional args of ``indicators.labels.add_profit_labels``
    and ``add_profit_strict_labels`` (the strict variant additionally uses
    ``l``/``y``; the non-strict variant ignores them). See
    ``azlib/config.py`` for this experiment's default values and a
    per-field explanation of what each knob means.
    """

    tf: int
    n: int
    m: float
    x: float
    l: int
    y: float
    atr_period: int = 14
    ma_length: int = 5


def add_labels(wide_df: pd.DataFrame, p: LabelParams) -> None:
    """Attach strict + non-strict profit labels for ``p.tf``, in place.

    Adds four columns to ``wide_df``:
      - ``{tf}_pslong_{suffix}`` / ``{tf}_psshort_{suffix}`` (strict;
        suffix includes ``l``/``y``)
      - ``{tf}_plong_{suffix}`` / ``{tf}_pshort_{suffix}`` (non-strict)
    via the existing ``indicators.labels`` functions — this function does
    no label math of its own. Both functions require the precomputed
    ``{tf}_atr_{p.atr_period}_ma_{p.ma_length}`` column to already exist on
    ``wide_df``; if it's missing they raise ``KeyError`` with a message
    naming the missing column, which propagates unchanged from here.
    """
    add_profit_strict_labels(
        wide_df, p.tf, p.n, p.m, p.x, p.l, p.y, p.atr_period, p.ma_length,
    )
    add_profit_labels(
        wide_df, p.tf, p.n, p.m, p.x, p.atr_period, p.ma_length,
    )


def label_col(p: LabelParams, direction: str, strict: bool) -> str:
    """Return the exact column name ``add_labels(wide_df, p)`` writes for
    (``direction``, ``strict``).

    ``direction`` must be ``"long"`` or ``"short"``. Numeric suffix
    formatting reuses ``indicators.labels._fmt`` (import, not
    reimplementation) so this can never drift from what the label
    functions actually name their columns: integer-valued numbers render
    verbatim (``2.0`` -> ``"2"``), other floats render with ``.`` -> ``p``
    (``1.5`` -> ``"1p5"``).
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    if strict:
        prefix = "pslong" if direction == "long" else "psshort"
        suffix = (
            f"n{_fmt(p.n)}_m{_fmt(p.m)}_x{_fmt(p.x)}_l{_fmt(p.l)}_y{_fmt(p.y)}"
        )
    else:
        prefix = "plong" if direction == "long" else "pshort"
        suffix = f"n{_fmt(p.n)}_m{_fmt(p.m)}_x{_fmt(p.x)}"

    return f"{p.tf}_{prefix}_{suffix}"
