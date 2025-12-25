# __init__.py
"""Structure break detection with full market context."""

from .failures.config import StructureBreakConfig
from .failures.detector import (
    detect_structure_breaks,
    detect_structure_breaks_with_mtf,
    filter_signals_by_context
)
from .swings import detect_swing_points, detect_market_structure
from .trends import detect_trend_state
from .failures.regime import detect_market_regime
from .failures.market_hours import tag_sessions, add_liquidity_awareness
from .failures.zones import detect_support_resistance_zones

# Optional imports
try:
    from .mtf_detector import detect_mtf_structure_breaks, get_mtf_summary
    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False

__all__ = [
    'StructureBreakConfig',
    'detect_structure_breaks',
    'detect_structure_breaks_with_mtf',
    'filter_signals_by_context',
    'detect_swing_points',
    'detect_market_structure',
    'detect_trend_state',
    'detect_market_regime',
    'tag_sessions',
    'add_liquidity_awareness',
    'detect_support_resistance_zones',
]

# Add MTF functions if available
if MTF_AVAILABLE:
    __all__.extend(['detect_mtf_structure_breaks', 'get_mtf_summary'])