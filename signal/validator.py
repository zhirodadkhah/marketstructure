# structure/signal/validator.py
"""
Validate signals with retest, velocity, and follow-through logic.
Returns confirmed/failure signals.
"""
from typing import Dict, Tuple
import numpy as np
from structure.metrics.types import RawSignals, ValidatedSignals
from .config import SignalValidatorConfig


def _compute_pullback_velocity(
        break_indices: np.ndarray,
        level_prices: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray,
        direction: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized pullback velocity calculation."""
    n = len(close)
    velocity = np.zeros(n, dtype=np.float32)
    bars_to_retest = np.zeros(n, dtype=np.int32)
    pullback_distance = np.zeros(n, dtype=np.float32)

    for i, break_idx in enumerate(break_indices):
        if break_idx >= n - 1:
            continue
        level_price = level_prices[i]
        buffer = atr[break_idx] * 0.5

        # Search for retest in future bars
        future_close = close[break_idx + 1:]
        if direction == 'bullish':
            retest_mask = (future_close <= level_price + buffer)
        else:  # bearish
            retest_mask = (future_close >= level_price - buffer)

        if not np.any(retest_mask):
            continue

        # Find first retest bar
        first_retest_offset = np.argmax(retest_mask)
        retest_idx = break_idx + 1 + first_retest_offset
        bars = retest_idx - break_idx

        # Compute pullback distance
        if direction == 'bullish':
            dist = max(0, level_price - close[retest_idx])
        else:
            dist = max(0, close[retest_idx] - level_price)

        # Compute velocity (ATR per bar)
        if bars > 0 and atr[break_idx] > 1e-10:
            velocity[retest_idx] = (dist / atr[break_idx]) / bars

        bars_to_retest[retest_idx] = bars
        pullback_distance[retest_idx] = dist

    return velocity, bars_to_retest, pullback_distance


def validate_signals(
        raw_signals: RawSignals,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        config: SignalValidatorConfig
) -> ValidatedSignals:
    """
    Validate raw signals with temporal and quality checks.

    Cyclomatic complexity: 3
    Args: 3 (raw_signals, price arrays, config)
    """
    n = len(close)
    # Initialize output arrays
    confirmed_bull = np.zeros(n, dtype=bool)
    confirmed_bear = np.zeros(n, dtype=bool)
    momentum_bull = np.zeros(n, dtype=bool)
    momentum_bear = np.zeros(n, dtype=bool)
    failure_bull = np.zeros(n, dtype=bool)
    failure_bear = np.zeros(n, dtype=bool)

    # === VALIDATE BULLISH BOS ===
    bos_bull_idx = np.where(raw_signals.is_bos_bullish_initial)[0]
    if len(bos_bull_idx) > 0:
        level_prices = high[bos_bull_idx]  # break level = swing high
        velocity, bars, _ = _compute_pullback_velocity(
            bos_bull_idx, level_prices, close, atr, 'bullish'
        )
        # ✅ FIX: Extract values AT BREAK INDICES (not retest indices)
        velocity_at_breaks = np.zeros(len(bos_bull_idx), dtype=np.float32)
        bars_at_breaks = np.zeros(len(bos_bull_idx), dtype=np.int32)

        for i, break_idx in enumerate(bos_bull_idx):
            # Find first retest after break
            future_velocity = velocity[break_idx + 1:]
            future_bars = bars[break_idx + 1:]
            valid_retests = future_bars > 0

            if np.any(valid_retests):
                first_retest = np.argmax(valid_retests)
                velocity_at_breaks[i] = future_velocity[first_retest]
                bars_at_breaks[i] = future_bars[first_retest]
            # else: keep 0 (no retest found)

        # Apply validation rules
        valid_retest = (
                (bars_at_breaks >= config.pullback_min_bars) &
                (bars_at_breaks <= config.pullback_max_bars) &
                (velocity_at_breaks <= config.max_pullback_velocity)
        )
        confirmed_bull[bos_bull_idx] = valid_retest
        momentum_bull[bos_bull_idx] = (velocity_at_breaks == 0)  # no pullback = momentum
        failure_bull[bos_bull_idx] = ~valid_retest & (velocity_at_breaks > 0)

    # === VALIDATE BEARISH BOS ===
    bos_bear_idx = np.where(raw_signals.is_bos_bearish_initial)[0]
    if len(bos_bear_idx) > 0:
        level_prices = low[bos_bear_idx]
        velocity, bars, _ = _compute_pullback_velocity(
            bos_bear_idx, level_prices, close, atr, 'bearish'
        )
        velocity_at_breaks = np.zeros(len(bos_bear_idx), dtype=np.float32)
        bars_at_breaks = np.zeros(len(bos_bear_idx), dtype=np.int32)

        for i, break_idx in enumerate(bos_bear_idx):
            future_velocity = velocity[break_idx + 1:]
            future_bars = bars[break_idx + 1:]
            valid_retests = future_bars > 0

            if np.any(valid_retests):
                first_retest = np.argmax(valid_retests)
                velocity_at_breaks[i] = future_velocity[first_retest]
                bars_at_breaks[i] = future_bars[first_retest]

        valid_retest = (
                (bars_at_breaks >= config.pullback_min_bars) &
                (bars_at_breaks <= config.pullback_max_bars) &
                (velocity_at_breaks <= config.max_pullback_velocity)
        )
        confirmed_bear[bos_bear_idx] = valid_retest
        momentum_bear[bos_bear_idx] = (velocity_at_breaks == 0)
        failure_bear[bos_bear_idx] = ~valid_retest & (velocity_at_breaks > 0)

    return ValidatedSignals(
        is_bos_bullish_confirmed=confirmed_bull,
        is_bos_bearish_confirmed=confirmed_bear,
        is_bos_bullish_momentum=momentum_bull,
        is_bos_bearish_momentum=momentum_bear,
        is_bullish_break_failure=failure_bull,
        is_bearish_break_failure=failure_bear
    )