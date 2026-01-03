# structure/signal/validator.py
"""
Validate raw break signals using retest dynamics, follow-through confirmation, and temporal rules.

This module implements practical price-action validation:
- Confirmed signals require a *respectful retest* within bounds
- Validated signals must show *follow-through* confirmation
- Fast retests and failed retests are flagged
- Momentum breaks (no retest) are tracked separately
- Immediate failures are detected within N bars

**SIGNAL COVERAGE:**
- ✅ **BOS (Break of Structure)**: Full validation (retest + follow-through + failures)
- ⚠️ **CHOCH (Change of Character)**: *Immediate failure only* (by design — reversal signals use different rules)

All logic is array-based with O(n) time complexity and minimal memory allocation.
"""

from __future__ import annotations
from typing import Tuple, NamedTuple, Optional
import numpy as np

from structure.metrics.types import (
    RawSignals,
    ValidatedSignals,
    RetestMetrics
)
from .config import SignalValidatorConfig

# ==============================================================================
# SECTION: Constants
# ==============================================================================

_BUFFER_MULTIPLIER: float = 0.5
_MIN_ATR_VALUE: float = 1e-10


# ==============================================================================
# SECTION: Configuration Validation
# ==============================================================================

def _validate_config(config: SignalValidatorConfig) -> None:
    """
    Validate SignalValidatorConfig parameters before use.

    Raises
    ------
    ValueError
        If any config parameter is out of bounds.
    """
    if config.follow_through_bars < 1:
        raise ValueError("follow_through_bars must be ≥ 1")
    if not 0.0 <= config.follow_through_close_ratio <= 1.0:
        raise ValueError("follow_through_close_ratio must be between 0 and 1")
    if config.pullback_min_bars < 1:
        raise ValueError("pullback_min_bars must be ≥ 1")
    if config.pullback_max_bars <= config.pullback_min_bars:
        raise ValueError("pullback_max_bars must be > pullback_min_bars")
    if config.max_pullback_velocity <= 0:
        raise ValueError("max_pullback_velocity must be > 0")
    if config.min_retest_respect_bars < 1:
        raise ValueError("min_retest_respect_bars must be ≥ 1")
    if config.max_retest_attempts < 1:
        raise ValueError("max_retest_attempts must be ≥ 1")
    if config.immediate_failure_bars < 1:
        raise ValueError("immediate_failure_bars must be ≥ 1")


# ==============================================================================
# SECTION: Helper Functions
# ==============================================================================

def _create_retest_condition_mask(
        future_close: np.ndarray,
        level_price: float,
        buffer: float,
        direction: str
) -> np.ndarray:
    """
    Create retest condition mask based on direction.

    Precondition:
    - `future_close` is 1D array of floats
    - `direction` in {"bullish", "bearish"}

    Postcondition:
    - Returns boolean array same length as `future_close`
    """
    if direction == "bullish":
        return future_close <= (level_price + buffer)
    elif direction == "bearish":
        return future_close >= (level_price - buffer)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'bullish' or 'bearish'.")


def _compute_first_retest_metrics(
        break_idx: int,
        level_price: float,
        future_bars: np.ndarray,
        future_close: np.ndarray,
        atr_value: float,
        buffer: float,
        direction: str,
        config: SignalValidatorConfig,
        close: np.ndarray
) -> Tuple[float, int, float, int]:
    """
    Compute metrics for the first valid retest.

    Precondition:
    - `future_bars`, `future_close` aligned slices from `break_idx + 1`
    - `atr_value >= 0`, `config` validated

    Postcondition:
    - Returns (velocity, bars_elapsed, pullback_distance, total_attempts)
    - If no valid retest, `bars_elapsed = 0`
    """
    if len(future_close) == 0:
        return 0.0, 0, 0.0, 0

    retest_positions = np.where(_create_retest_condition_mask(
        future_close, level_price, buffer, direction
    ))[0]
    total_attempts = len(retest_positions)

    if total_attempts == 0:
        return 0.0, 0, 0.0, 0

    for pos in retest_positions:
        actual_bar = future_bars[pos]
        bars_elapsed = actual_bar - break_idx
        if (bars_elapsed >= config.pullback_min_bars and
                bars_elapsed >= config.min_retest_respect_bars):
            close_at_retest = close[actual_bar]
            if direction == "bullish":
                pullback_distance = max(0.0, level_price - close_at_retest)
            else:
                pullback_distance = max(0.0, close_at_retest - level_price)

            atr_safe = max(atr_value, _MIN_ATR_VALUE)
            velocity = (pullback_distance / atr_safe) / bars_elapsed if bars_elapsed > 0 else 0.0
            return velocity, bars_elapsed, pullback_distance, total_attempts

    return 0.0, 0, 0.0, total_attempts


def compute_pullback_metrics(
        breakout_indices: np.ndarray,
        level_prices: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray,
        direction: str,
        config: SignalValidatorConfig
) -> RetestMetrics:
    """
    Compute retest metrics for all breakouts of given direction.

    Postcondition:
    - `retest_attempts[i]` = total attempts at breakout bar `i`
    - Retest metrics stored at retest bar index
    """
    n = len(close)
    velocity = np.zeros(n, dtype=np.float32)
    bars_to_retest = np.zeros(n, dtype=np.int32)
    pullback_distance = np.zeros(n, dtype=np.float32)
    retest_attempts = np.zeros(n, dtype=np.int32)

    for i, break_idx in enumerate(breakout_indices):
        if break_idx >= n - 1:
            continue
        level_price = level_prices[i]
        atr_value = atr[break_idx]
        buffer = atr_value * _BUFFER_MULTIPLIER
        search_start = break_idx + 1
        search_end = min(break_idx + config.pullback_max_bars + 1, n)
        if search_start >= search_end:
            continue

        future_bars = np.arange(search_start, search_end)
        future_close = close[future_bars]

        vel, bars, dist, attempts = _compute_first_retest_metrics(
            break_idx, level_price, future_bars, future_close,
            atr_value, buffer, direction, config, close
        )

        retest_attempts[break_idx] = attempts
        if bars > 0:
            retest_bar = break_idx + bars
            if retest_bar < n:
                velocity[retest_bar] = vel
                bars_to_retest[retest_bar] = bars
                pullback_distance[retest_bar] = dist

    is_fast_retest = velocity > config.max_pullback_velocity
    is_slow_retest = (bars_to_retest > 0) & (bars_to_retest > config.pullback_max_bars)

    break_levels = np.full(n, np.nan, dtype=np.float32)
    if len(breakout_indices) > 0:
        sorted_idx = np.argsort(breakout_indices)
        sorted_breakouts = breakout_indices[sorted_idx]
        sorted_levels = level_prices[sorted_idx]
        insertion = np.searchsorted(sorted_breakouts, np.arange(n), side='right') - 1
        valid = insertion >= 0
        break_levels[valid] = sorted_levels[insertion[valid]]

    return RetestMetrics(
        retest_velocity=velocity,
        bars_to_retest=bars_to_retest,
        pullback_distance=pullback_distance,
        is_fast_retest=is_fast_retest,
        is_slow_retest=is_slow_retest,
        retest_attempts=retest_attempts,
        retest_close=close,
        retest_indices=np.where(bars_to_retest > 0)[0],
        break_levels=break_levels,
        direction=direction
    )


def _extract_metrics_at_breakouts(
        breakout_indices: np.ndarray,
        retest_metrics: RetestMetrics,
        max_search_bars: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract retest metrics at breakout points."""
    k = len(breakout_indices)
    velocity_at_break = np.zeros(k, dtype=np.float32)
    bars_at_break = np.zeros(k, dtype=np.int32)
    attempts_at_break = retest_metrics.retest_attempts[breakout_indices]

    for i, break_idx in enumerate(breakout_indices):
        if attempts_at_break[i] == 0:
            continue
        search_end = min(break_idx + max_search_bars + 1, len(retest_metrics.bars_to_retest))
        if break_idx + 1 >= search_end:
            continue
        future_bars = retest_metrics.bars_to_retest[break_idx + 1:search_end]
        retest_mask = future_bars > 0
        if np.any(retest_mask):
            first_pos = np.argmax(retest_mask)
            bars_at_break[i] = future_bars[first_pos]
            velocity_at_break[i] = retest_metrics.retest_velocity[break_idx + 1 + first_pos]

    return velocity_at_break, bars_at_break, attempts_at_break


def _classify_breakouts(
        velocity_at_break: np.ndarray,
        bars_at_break: np.ndarray,
        attempts_at_break: np.ndarray,
        config: SignalValidatorConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify breakouts into confirmed, momentum, or failed."""
    n = len(velocity_at_break)
    confirmed = np.zeros(n, dtype=bool)
    momentum = np.zeros(n, dtype=bool)
    failed = np.zeros(n, dtype=bool)

    for i in range(n):
        attempts = attempts_at_break[i]
        if attempts == 0:
            momentum[i] = True
            continue

        bars = bars_at_break[i]
        vel = velocity_at_break[i]

        valid_bars = (config.pullback_min_bars <= bars <= config.pullback_max_bars and
                      bars >= config.min_retest_respect_bars)
        valid_velocity = vel <= config.max_pullback_velocity
        valid_attempts = attempts <= config.max_retest_attempts

        if valid_bars and valid_velocity and valid_attempts:
            confirmed[i] = True
        else:
            failed[i] = True

    return confirmed, momentum, failed


def _validate_follow_through(
        signal_indices: np.ndarray,
        level_prices: np.ndarray,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        direction: str,
        config: SignalValidatorConfig
) -> np.ndarray:
    """Validate follow-through for signal confirmation."""
    n = len(close)
    valid = np.zeros(len(signal_indices), dtype=bool)

    for idx_in_signals, signal_idx in enumerate(signal_indices):
        if signal_idx >= n - config.follow_through_bars:
            continue
        level_price = level_prices[idx_in_signals]
        buffer = atr[signal_idx] * _BUFFER_MULTIPLIER

        ft_start = signal_idx + 1
        ft_end = min(signal_idx + config.follow_through_bars + 1, n)
        if ft_start >= ft_end:
            continue

        qualifying = 0
        required = max(1, int(np.ceil(config.follow_through_close_ratio * config.follow_through_bars)))

        for offset in range(ft_end - ft_start):
            i = ft_start + offset
            bar_range = max(high[i] - low[i], atr[i], _MIN_ATR_VALUE)
            if direction == "bullish":
                if close[i] > level_price + buffer:
                    close_loc = (close[i] - low[i]) / bar_range
                    if close_loc >= config.follow_through_close_ratio:
                        qualifying += 1
            else:
                if close[i] < level_price - buffer:
                    close_loc = (high[i] - close[i]) / bar_range
                    if close_loc >= config.follow_through_close_ratio:
                        qualifying += 1

        valid[idx_in_signals] = qualifying >= required

    return valid


def _mark_immediate_failures(
        raw_signals: RawSignals,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        config: SignalValidatorConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mark immediate failures within `immediate_failure_bars`."""
    n = len(close)
    immediate_fail_bull = np.zeros(n, dtype=bool)
    immediate_fail_bear = np.zeros(n, dtype=bool)
    failed_choch_bull = np.zeros(n, dtype=bool)
    failed_choch_bear = np.zeros(n, dtype=bool)

    def _check_immediate_failure(idx: int, direction: str, level_price: float, buffer: float) -> bool:
        for offset in range(1, min(config.immediate_failure_bars + 1, n - idx)):
            check_idx = idx + offset
            if direction == "bullish":
                if close[check_idx] < level_price - buffer:
                    return True
            else:
                if close[check_idx] > level_price + buffer:
                    return True
        return False

    for idx in np.where(raw_signals.is_bos_bullish_initial)[0]:
        if idx < n - config.immediate_failure_bars:
            level_price = high[idx]
            buffer = atr[idx] * _BUFFER_MULTIPLIER
            if _check_immediate_failure(idx, "bullish", level_price, buffer):
                immediate_fail_bull[idx] = True

    for idx in np.where(raw_signals.is_bos_bearish_initial)[0]:
        if idx < n - config.immediate_failure_bars:
            level_price = low[idx]
            buffer = atr[idx] * _BUFFER_MULTIPLIER
            if _check_immediate_failure(idx, "bearish", level_price, buffer):
                immediate_fail_bear[idx] = True

    for idx in np.where(raw_signals.is_choch_bullish)[0]:
        if idx < n - config.immediate_failure_bars:
            level_price = low[idx]
            buffer = atr[idx] * _BUFFER_MULTIPLIER
            if _check_immediate_failure(idx, "bearish", level_price, buffer):
                failed_choch_bull[idx] = True

    for idx in np.where(raw_signals.is_choch_bearish)[0]:
        if idx < n - config.immediate_failure_bars:
            level_price = high[idx]
            buffer = atr[idx] * _BUFFER_MULTIPLIER
            if _check_immediate_failure(idx, "bullish", level_price, buffer):
                failed_choch_bear[idx] = True

    return immediate_fail_bull, immediate_fail_bear, failed_choch_bull, failed_choch_bear


# ==============================================================================
# SECTION: Direction Processing Function (MOVED OUTSIDE validate_signals)
# ==============================================================================

def _process_direction(
        is_bos_mask: np.ndarray,
        price_levels: np.ndarray,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        direction: str,
        config: SignalValidatorConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process BOS signals for one direction.

    Parameters
    ----------
    is_bos_mask : np.ndarray[bool]
        BOS signal mask (full length)
    price_levels : np.ndarray[float32]
        Full price array (high for bullish, low for bearish)
    close, high, low, atr : np.ndarray[float32]
        Price arrays for follow-through validation
    direction : str
        "bullish" or "bearish"

    Returns
    -------
    Tuple of full-length masks (confirmed, momentum, failure)
    """
    n = len(is_bos_mask)
    indices = np.where(is_bos_mask)[0]

    # Initialize full-length arrays
    confirmed_full = np.zeros(n, dtype=bool)
    momentum_full = np.zeros(n, dtype=bool)
    failed_full = np.zeros(n, dtype=bool)

    if len(indices) == 0:
        return confirmed_full, momentum_full, failed_full

    breakout_levels = price_levels[indices]

    metrics = compute_pullback_metrics(indices, breakout_levels, close, atr, direction, config)
    vel, bars, att = _extract_metrics_at_breakouts(indices, metrics, config.pullback_max_bars)
    confirmed, momentum, failed = _classify_breakouts(vel, bars, att, config)

    # Apply follow-through to BOTH confirmed AND momentum
    if np.any(confirmed | momentum):
        all_follow_indices = indices[confirmed | momentum]
        ft_valid = _validate_follow_through(
            all_follow_indices,
            breakout_levels[confirmed | momentum],
            close, high, low, atr, direction, config
        )

        # Track original combined mask before FT masking
        original_combined = confirmed | momentum

        # Create FT mask aligned with indices array
        ft_mask = np.zeros(len(indices), dtype=bool)
        ft_mask[confirmed | momentum] = ft_valid

        # Apply FT filter
        confirmed = confirmed & ft_mask
        momentum = momentum & ft_mask

        # Fail those that were candidates but failed FT
        failed = failed | (original_combined & ~ft_mask)

    # Map results back to full-length arrays
    confirmed_full[indices] = confirmed
    momentum_full[indices] = momentum
    failed_full[indices] = failed

    return confirmed_full, momentum_full, failed_full


# ==============================================================================
# SECTION: Main Validation Function
# ==============================================================================

def validate_signals(
        raw_signals: RawSignals,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        config: SignalValidatorConfig
) -> ValidatedSignals:
    """
    Validate raw signals with full retest, follow-through, and failure logic.

    Precondition:
    - All arrays same length
    - Config validated

    Postcondition:
    - Returns complete ValidatedSignals with all mask types
    - BOS: full validation (retest + follow-through)
    - CHOCH: immediate failure only (by design)
    """
    _validate_config(config)

    n = len(close)
    if not all(len(arr) == n for arr in [high, low, atr]):
        raise ValueError("All price arrays must have same length.")
    if len(raw_signals.is_bos_bullish_initial) != n:
        raise ValueError("Raw signals length mismatch.")

    # Process BOS signals for both directions
    confirmed_bull, momentum_bull, failure_bull = _process_direction(
        raw_signals.is_bos_bullish_initial, high, close, high, low, atr, "bullish", config
    )

    confirmed_bear, momentum_bear, failure_bear = _process_direction(
        raw_signals.is_bos_bearish_initial, low, close, high, low, atr, "bearish", config
    )

    # Get immediate failures (already full-length arrays)
    (immediate_fail_bull, immediate_fail_bear,
     failed_choch_bull, failed_choch_bear) = _mark_immediate_failures(
        raw_signals, close, high, low, atr, config
    )

    # Ensure mutual exclusivity: immediate failures override other failure classifications
    # If a breakout is an immediate failure, it shouldn't also be a regular break failure
    failure_bull = failure_bull & ~immediate_fail_bull
    failure_bear = failure_bear & ~immediate_fail_bear

    return ValidatedSignals(
        is_bos_bullish_confirmed=confirmed_bull,
        is_bos_bearish_confirmed=confirmed_bear,
        is_bos_bullish_momentum=momentum_bull,
        is_bos_bearish_momentum=momentum_bear,
        is_bullish_break_failure=failure_bull,
        is_bearish_break_failure=failure_bear,
        is_bullish_immediate_failure=immediate_fail_bull,
        is_bearish_immediate_failure=immediate_fail_bear,
        is_failed_choch_bullish=failed_choch_bull,
        is_failed_choch_bearish=failed_choch_bear
    )