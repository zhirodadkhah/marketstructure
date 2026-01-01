from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SignalGeneratorConfig:
    min_break_atr_mult: float = 0.5
    buffer_multiplier: float = 0.5

@dataclass(frozen=True)
class SignalValidatorConfig:
    follow_through_bars: int = 3
    follow_through_close_ratio: float = 0.6
    pullback_min_bars: int = 3
    pullback_max_bars: int = 50
    max_pullback_velocity: float = 0.8
    min_retest_respect_bars: int = 5  # ✅ ADD THIS

@dataclass(frozen=True)
class SignalFilterConfig:
    min_regime_score: float = 0.6
    min_zone_confluence: float = 0.5
    avoid_range_compression: bool = True
    avoid_fast_retests: bool = False
    min_retest_respect_score: Optional[float] = 0.6  # ➕ NEW

@dataclass(frozen=True)
class SignalQualityConfig:
    session_weights: dict = None
    regime_boost: float = 0.2
    zone_boost: float = 0.15

    def __post_init__(self):
        if self.session_weights is None:
            object.__setattr__(self, 'session_weights', {
                'overlap': 1.2, 'ny': 1.1, 'london': 1.0,
                'asia': 0.8, 'low_liquidity': 0.5
            })