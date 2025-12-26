from .swings import detect_swing_points
from .structure import detect_market_structure
from .trend import detect_trend_state
from .breaks import detect_structure_break_signals
from .config import SwingConfig, TrendConfig, BreakConfig

__all__ = [
    # Core detection functions
    "detect_swing_points",
    "detect_market_structure",
    "detect_trend_state",
    "detect_structure_break_signals",

    # Configuration classes
    "SwingConfig",
    "TrendConfig",
    "BreakConfig",
]