# structure/signal/__init__.py
"""
Signal Generation, Validation, Filtering & Scoring
==================================================

This module turns raw market structure breaks into high-quality, context-aware trading signals:

- **Raw signals**: initial BOS/CHOCH detections
- **Validation**: retest dynamics, follow-through, immediate failure
- **Filtering**: regime, compression, retest respect
- **Scoring**: quality [0,1] based on momentum, zones, sessions, liquidity

All functions are pure NumPy and vectorized for performance.
"""

from .generator import generate_raw_signals
from .validator import validate_signals
from .filter import filter_signals
from .quality import score_signals
from .config import (
    SignalGeneratorConfig,
    SignalValidatorConfig,
    SignalFilterConfig,
    SignalQualityConfig,
)

__all__ = [
    # Signal lifecycle
    "generate_raw_signals",
    "validate_signals",
    "filter_signals",
    "score_signals",

    # Configs
    "SignalGeneratorConfig",
    "SignalValidatorConfig",
    "SignalFilterConfig",
    "SignalQualityConfig",
]