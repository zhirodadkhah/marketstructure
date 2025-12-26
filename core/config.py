# core/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SwingConfig:
    half_window: int = 2

@dataclass(frozen=True)
class TrendConfig:
    invalidation_buffer: float = 0.0
    include_metrics: bool = False

@dataclass(frozen=True)
class BreakConfig:
    min_break_atr_mult: float = 0.5
    buffer_multiplier: float = 0.5
    follow_through_bars: int = 3
    follow_through_close_ratio: float = 0.6
    pullback_min_bars: int = 3
    pullback_max_bars: int = 50
    max_pullback_velocity: float = 0.8
    