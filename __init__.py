# structure/failures/__init__.py
"""Structure break detection with market regime awareness."""

from .failures.config import StructureBreakConfig
from .failures.detector import detect_structure_breaks
from .swings import detect_swing_points, detect_market_structure
from .trends import detect_trend_state
from .failures.regime import detect_market_regime
from .failures.market_hours import tag_sessions, add_liquidity_awareness
from .failures.signal_filter import RegimeSignalFilter

__all__ = [
    'StructureBreakConfig',
    'detect_structure_breaks',
    'detect_swing_points',
    'detect_market_structure',
    'detect_trend_state',
    'detect_market_regime',
    'tag_sessions',
    'add_liquidity_awareness',
    'RegimeSignalFilter',
]