from dataclasses import dataclass


@dataclass(frozen=True)
class StructureBreakConfig:
    """Immutable configuration for structure break detection."""
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

