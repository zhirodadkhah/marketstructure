from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SignalGeneratorConfig:
    min_break_atr_mult: float = 0.5
    buffer_multiplier: float = 0.5


# structure/signal/config.py
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SignalValidatorConfig:
    """
    Configuration for signal validation and quality scoring.

    Combines retest validation, follow-through confirmation, and quality thresholds.
    """

    # FOLLOW-THROUGH VALIDATION (CRITICAL - was missing from previous version!)
    follow_through_bars: int = 3
    follow_through_close_ratio: float = 0.6

    # RETEST VALIDATION
    pullback_min_bars: int = 3
    pullback_max_bars: int = 50
    max_pullback_velocity: float = 0.8
    min_retest_respect_bars: int = 5

    # MULTI-ATTEMPT FAILURE
    max_retest_attempts: int = 3

    # QUALITY THRESHOLDS (FOR FILTERING)
    min_momentum_strength: float = 0.3
    min_zone_quality: float = 0.5

    # IMMEDIATE FAILURE (BARS 1-3)
    immediate_failure_bars: int = 3

    def __post_init__(self):
        """Validate configuration."""
        # Follow-through validation
        if self.follow_through_bars < 1:
            raise ValueError("follow_through_bars must be ≥ 1")
        if not 0.0 <= self.follow_through_close_ratio <= 1.0:
            raise ValueError("follow_through_close_ratio must be between 0 and 1")

        # Retest validation
        if self.pullback_min_bars < 1:
            raise ValueError("pullback_min_bars must be ≥ 1")
        if self.pullback_max_bars <= self.pullback_min_bars:
            raise ValueError("pullback_max_bars must be > pullback_min_bars")
        if self.max_pullback_velocity <= 0:
            raise ValueError("max_pullback_velocity must be > 0")
        if self.min_retest_respect_bars < 1:
            raise ValueError("min_retest_respect_bars must be ≥ 1")
        if self.max_retest_attempts < 1:
            raise ValueError("max_retest_attempts must be ≥ 1")

        # Quality thresholds
        if not 0 <= self.min_momentum_strength <= 1:
            raise ValueError("min_momentum_strength must be between 0 and 1")
        if not 0 <= self.min_zone_quality <= 1:
            raise ValueError("min_zone_quality must be between 0 and 1")

        # Immediate failure
        if self.immediate_failure_bars < 1:
            raise ValueError("immediate_failure_bars must be ≥ 1")

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