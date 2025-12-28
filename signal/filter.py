# structure/signal/filter.py
"""
Filter signals based on market context.
"""
from typing import Dict
import numpy as np
from structure.metrics.types import ValidatedSignals
from .config import SignalFilterConfig


def filter_signals(
    validated_signals: ValidatedSignals,
    market_regime: np.ndarray,
    zone_confluence: np.ndarray,
    is_range_compression: np.ndarray,
    retest_velocity: np.ndarray,
    session: np.ndarray,
    config: SignalFilterConfig
) -> ValidatedSignals:
    """
        Apply context-aware filtering to validated signals.

        Cyclomatic complexity: 2
        Args: 7 → but can be grouped as "context" dict if needed
        """
    trend_mask = (market_regime == 'strong_trend') | (market_regime == 'weak_trend')
    confluence_mask = zone_confluence >= config.min_zone_confluence
    compression_mask = ~is_range_compression if config.avoid_range_compression else np.ones_like(is_range_compression, dtype=bool)
    fast_retest_mask = retest_velocity <= 0.5
    if config.avoid_fast_retests:
        fast_retest_mask = ~fast_retest_mask

    global_filter = trend_mask & confluence_mask & compression_mask & fast_retest_mask

    return ValidatedSignals(
        is_bos_bullish_confirmed=validated_signals.is_bos_bullish_confirmed & global_filter,
        is_bos_bearish_confirmed=validated_signals.is_bos_bearish_confirmed & global_filter,
        is_bos_bullish_momentum=validated_signals.is_bos_bullish_momentum & global_filter,
        is_bos_bearish_momentum=validated_signals.is_bos_bearish_momentum & global_filter,
        is_bullish_break_failure=validated_signals.is_bullish_break_failure & global_filter,
        is_bearish_break_failure=validated_signals.is_bearish_break_failure & global_filter
    )