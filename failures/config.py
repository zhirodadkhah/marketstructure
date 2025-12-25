# config.py
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class StructureBreakConfig:
    """Immutable configuration for structure break detection."""

    # --- EXISTING STRUCTURE BREAK CONFIG ---
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

    # --- ATR CONFIG ---
    atr_period: int = 14

    # --- GROUP 1: MARKET REGIME CONFIG ---
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

    # --- GROUP 1: SESSION CONFIG ---
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

    # --- GROUP 2: ZONE DETECTION CONFIG ---
    zone_detection_enabled: bool = True
    zone_lookback_bars: int = 200
    zone_proximity_atr_mult: float = 1.5
    min_zone_strength: int = 2
    zone_buffer_multiplier: float = 0.5
    zone_active_window: int = 50
    max_zones: int = 100

    # --- GROUP 3: MTF CONFIG ---
    mtf_enabled: bool = False
    mtf_periods: List[str] = field(default_factory=lambda: ['1H'])
    mtf_min_confluence_score: float = 0.6