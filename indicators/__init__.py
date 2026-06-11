"""indicators — public interface of the indicator framework.

External users import from this package only:

    from indicators import Indicators, build_indicator_input, DataAttributes

Layout:
    framework.py   — IndicatorField ABC, Indicators orchestrator, build_indicator_input
    attributes.py  — DataAttributes (NN normalisation stats)
    registry.py    — _FIELD_REGISTRY (name → field factory)
    library/       — IndicatorField implementations, one file per group
"""

from config_loader import load_indicators_config  # noqa: F401  (re-export, legacy import path)

from .attributes import DataAttributes  # noqa: F401
from .framework import (  # noqa: F401
    IndicatorField,
    Indicators,
    _PlaceholderField,
    _closed_pos_cache,
    _warned_resources,
    build_indicator_input,
)
from .registry import *  # noqa: F401,F403  — _FIELD_REGISTRY + all field classes
from .registry import _FIELD_REGISTRY  # noqa: F401  (explicit: underscore names skip *)
