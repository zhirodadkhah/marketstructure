# structure/failures/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class StructureBreakConfig:
    """Immutable configuration for structure break detection."""
    # Existing structure break config
    min_break_body_ratio: float = 0.6
    min_break_atr_mult: float = 0.5
    pullback_min_bars: int = 2
    pullback_max_bars: int = 20
    momentum_continuation_bars: int = 5
    max_active_levels: int = 50
    buffer_multiplier: float = 0.5
    momentum_threshold: float = 1.5
    wick_body_ratio: float = 2.0
    upper_wick_ratio: float = 0.3
    follow_through_bars: int = 3
    follow_through_close_ratio: float = 0.6

    # ➕ ATR CONFIG (for consistency)
    atr_period: int = 14

    # ➕ REGIME DETECTION CONFIG
    enable_regime_detection: bool = True
    regime_swing_window: int = 100
    regime_atr_slope_window: int = 20
    regime_consistency_window: int = 50
    regime_swing_density_high: float = 0.02
    regime_swing_density_moderate: float = 0.01
    regime_swing_density_low: float = 0.005
    regime_efficiency_high: float = 0.5
    regime_efficiency_low: float = 0.3
    regime_consistency_high: float = 0.6
    regime_atr_slope_threshold: float = 0.001

    # ➕ SESSION CONFIG
    enable_session_tagging: bool = True
    session_timezone: str = 'UTC'
    session_asia_start: int = 0
    session_asia_end: int = 7
    session_london_start: int = 7
    session_london_end: int = 16
    session_ny_start: int = 12
    session_ny_end: int = 21
    session_london_ny_overlap_start: int = 12
    session_london_ny_overlap_end: int = 16