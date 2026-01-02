# structure/signal/validator.py
"""
Validate raw break signals using retest dynamics, follow-through confirmation, and temporal rules.

This module implements practical price-action validation:
- Confirmed signals require a *respectful retest* within bounds
- Validated signals must show *follow-through* confirmation
- Fast retests and failed retests are flagged
- Momentum breaks (no retest) are tracked separately
- Quality scores incorporate momentum, zones, and retest respect

All logic is array-based with O(n) time complexity and minimal memory allocation.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
from structure.metrics.types import (
    RawSignals, ValidatedSignals, RetestMetrics, SignalQuality
)
from .config import SignalValidatorConfig

# ==============================================================================
# SECTION: Constants & Type Definitions
# ==============================================================================

_BUFFER_MULTIPLIER = 0.5  # ATR multiplier for retest buffer zone
_MIN_ATR_VALUE = 1e-10  # Minimum ATR to avoid division by zero


# ==============================================================================
# SECTION: Core Retest Analysis
# ==============================================================================

def _create_retest_condition_mask(
        future_close: np.ndarray,
        level_price: float,
        buffer: float,
        direction: str
) -> np.ndarray:
    """
    Create boolean mask for retest condition based on direction.

    For bullish: price moves DOWN to breakout level (close <= level_price + buffer)
    For bearish: price moves UP to breakout level (close >= level_price - buffer)
    """
    if direction == "bullish":
        return future_close <= (level_price + buffer)
    elif direction == "bearish":
        return future_close >= (level_price - buffer)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'bullish' or 'bearish'.")


def _compute_retest_metrics_at_bar(
        break_idx: int,
        level_price: float,
        future_bars: np.ndarray,
        future_close: np.ndarray,
        atr_value: float,
        buffer: float,
        direction: str,
        min_bars: int,
        max_bars: int,
        close: np.ndarray
) -> Tuple[float, int, float, int]:
    """
    Compute retest metrics for the FIRST retest occurrence.
    """
    retest_mask = _create_retest_condition_mask(
        future_close, level_price, buffer, direction
    )

    if not np.any(retest_mask):
        return 0.0, 0, 0.0, 0

    # Get positions of all retests
    retest_positions = np.where(retest_mask)[0]
    total_attempts = len(retest_positions)

    # Find first retest that meets min_bars requirement
    for pos in retest_positions:
        actual_bar = future_bars[pos]
        bars_elapsed = actual_bar - break_idx

        if bars_elapsed >= min_bars:
            # Calculate metrics
            close_at_retest = close[actual_bar]
            if direction == "bullish":
                pullback_distance = max(0.0, level_price - close_at_retest)
            else:
                pullback_distance = max(0.0, close_at_retest - level_price)

            atr_safe = max(atr_value, _MIN_ATR_VALUE)
            velocity = (pullback_distance / atr_safe) / bars_elapsed if bars_elapsed > 0 else 0.0

            return velocity, bars_elapsed, pullback_distance, total_attempts

    # No retest meets min_bars, but return attempts count
    return 0.0, 0, 0.0, total_attempts

# ==============================================================================
# SECTION: Pullback Velocity & Retest Metrics (Group 5)
# ==============================================================================

def compute_pullback_metrics(
    break_indices: np.ndarray,
    level_prices: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    direction: str,
    config: SignalValidatorConfig,
    return_full_metrics: bool = False  # ← ADD THIS
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], RetestMetrics]:
    """
    Compute pullback metrics for multiple breakouts.
    """
    n = len(close)
    velocity = np.zeros(n, dtype=np.float32)
    bars_to_retest = np.zeros(n, dtype=np.int32)
    pullback_distance = np.zeros(n, dtype=np.float32)
    retest_attempts = np.zeros(n, dtype=np.int32)

    for i, break_idx in enumerate(break_indices):
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

        vel, bars, dist, attempts = _compute_retest_metrics_at_bar(
            break_idx=break_idx,
            level_price=level_price,
            future_bars=future_bars,
            future_close=future_close,
            atr_value=atr_value,
            buffer=buffer,
            direction=direction,
            min_bars=config.pullback_min_bars,
            max_bars=config.pullback_max_bars,
            close=close
        )

        # Store metrics at retest bar
        if bars > 0:
            retest_bar = break_idx + bars
            if retest_bar < n:
                velocity[retest_bar] = vel
                bars_to_retest[retest_bar] = bars
                pullback_distance[retest_bar] = dist
                retest_attempts[retest_bar] = attempts

        # Store attempts at breakout bar for classification
        retest_attempts[break_idx] = attempts  # ← This is critical!

    if return_full_metrics:
        return RetestMetrics(
            retest_velocity=velocity,
            bars_to_retest=bars_to_retest,
            pullback_distance=pullback_distance,
            is_fast_retest=np.zeros(n, dtype=bool),  # Add proper logic
            is_slow_retest=np.zeros(n, dtype=bool),
            retest_attempts=retest_attempts,
            retest_close=np.zeros(n, dtype=float),
            retest_indices=np.full(n, -1, dtype=int),
            break_levels=np.zeros(n, dtype=float),
            direction=direction
        )
    else:
        return velocity, bars_to_retest, pullback_distance, retest_attempts

# ==============================================================================
# SECTION: Signal Validation Core Logic
# ==============================================================================

def _extract_breakout_metrics(
        break_indices: np.ndarray,
        velocity: np.ndarray,
        bars: np.ndarray,
        attempts: np.ndarray,
        max_search_bars: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract velocity, bars, and attempts at each breakout point.
    """
    k = len(break_indices)
    velocity_at_break = np.zeros(k, dtype=np.float32)
    bars_at_break = np.zeros(k, dtype=np.int32)
    attempts_at_break = np.zeros(k, dtype=np.int32)

    for i, break_idx in enumerate(break_indices):
        # Get attempts from breakout bar
        attempts_at_break[i] = attempts[break_idx]

        # If no attempts, skip searching for retest
        if attempts_at_break[i] == 0:
            continue

        # Search for retest metrics starting from breakout bar
        search_start = break_idx + 1
        search_end = min(break_idx + max_search_bars + 1, len(velocity))
        if search_start >= search_end:
            continue

        # Find first retest
        future_bars = bars[search_start:search_end]
        future_velocity = velocity[search_start:search_end]

        retest_mask = future_bars > 0
        if np.any(retest_mask):
            first_retest_pos = np.argmax(retest_mask)
            bars_at_break[i] = future_bars[first_retest_pos]
            velocity_at_break[i] = future_velocity[first_retest_pos]

    return velocity_at_break, bars_at_break, attempts_at_break


def _classify_breakout_signals(
        velocity_at_break: np.ndarray,
        bars_at_break: np.ndarray,
        attempts_at_break: np.ndarray,
        config: SignalValidatorConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify breakouts based on retest metrics.
    """
    n = len(velocity_at_break)

    # Initialize masks
    confirmed = np.zeros(n, dtype=bool)
    momentum = np.zeros(n, dtype=bool)
    failure = np.zeros(n, dtype=bool)

    for i in range(n):
        vel = velocity_at_break[i]
        bars = bars_at_break[i]
        att = attempts_at_break[i]

        # Momentum: no retest attempts
        if att == 0:
            momentum[i] = True
            continue

        # Valid retest criteria
        valid_bars = (config.pullback_min_bars <= bars <= config.pullback_max_bars)
        valid_velocity = (vel <= config.max_pullback_velocity)
        valid_attempts = (att <= config.max_retest_attempts)

        # Confirmed: retest exists AND all criteria met
        if valid_bars and valid_velocity and valid_attempts:
            confirmed[i] = True
        else:
            # Failure: retest exists but invalid
            failure[i] = True

    return confirmed, momentum, failure


def _mark_immediate_failures(
        raw_signals: RawSignals,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        config: SignalValidatorConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mark immediate failures for BOS and CHOCH signals.
    """
    n = len(close)
    immediate_fail_bull = np.zeros(n, dtype=bool)
    immediate_fail_bear = np.zeros(n, dtype=bool)
    failed_choch_bull = np.zeros(n, dtype=bool)
    failed_choch_bear = np.zeros(n, dtype=bool)

    def _check_immediate_failure(idx: int, direction: str, level_price: float, buffer: float) -> bool:
        """Check if immediate failure occurred in first N bars."""
        for offset in range(1, min(config.immediate_failure_bars + 1, n - idx)):
            check_idx = idx + offset
            if direction == "bullish":
                if close[check_idx] < level_price - buffer:
                    return True
            else:
                if close[check_idx] > level_price + buffer:
                    return True
        return False

    # BOS Bullish - break of swing high, so level = high[idx]
    bos_bull_idx = np.where(raw_signals.is_bos_bullish_initial)[0]
    for idx in bos_bull_idx:
        if idx >= n - 1:
            continue
        level_price = high[idx]  # breakout level = swing high
        buffer = atr[idx] * _BUFFER_MULTIPLIER
        if _check_immediate_failure(idx, "bullish", level_price, buffer):
            immediate_fail_bull[idx] = True

    # BOS Bearish - break of swing low, so level = low[idx]
    bos_bear_idx = np.where(raw_signals.is_bos_bearish_initial)[0]
    for idx in bos_bear_idx:
        if idx >= n - 1:
            continue
        level_price = low[idx]  # breakout level = swing low
        buffer = atr[idx] * _BUFFER_MULTIPLIER
        if _check_immediate_failure(idx, "bearish", level_price, buffer):
            immediate_fail_bear[idx] = True

    # CHOCH Bullish - break of swing low (reversal)
    choch_bull_idx = np.where(raw_signals.is_choch_bullish)[0]
    for idx in choch_bull_idx:
        if idx >= n - 1:
            continue
        level_price = low[idx]  # CHOCH breaks below swing low
        buffer = atr[idx] * _BUFFER_MULTIPLIER
        # Failure if price goes back ABOVE the broken low
        if _check_immediate_failure(idx, "bearish", level_price, buffer):
            failed_choch_bull[idx] = True

    # CHOCH Bearish - break of swing high (reversal)
    choch_bear_idx = np.where(raw_signals.is_choch_bearish)[0]
    for idx in choch_bear_idx:
        if idx >= n - 1:
            continue
        level_price = high[idx]  # CHOCH breaks above swing high
        buffer = atr[idx] * _BUFFER_MULTIPLIER
        # Failure if price goes back BELOW the broken high
        if _check_immediate_failure(idx, "bullish", level_price, buffer):
            failed_choch_bear[idx] = True

    return immediate_fail_bull, immediate_fail_bear, failed_choch_bull, failed_choch_bear


def _validate_follow_through(
        confirmed_signals: np.ndarray,
        level_prices: np.ndarray,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        direction: str,
        config: SignalValidatorConfig
) -> np.ndarray:
    """
    Validate follow-through confirmation after retest.
    """
    n = len(confirmed_signals)
    valid_follow_through = np.zeros(n, dtype=bool)

    if not np.any(confirmed_signals):
        return valid_follow_through

    confirmed_indices = np.where(confirmed_signals)[0]

    for i, signal_idx in enumerate(confirmed_indices):
        if signal_idx >= n - config.follow_through_bars:
            continue

        # Handle level_prices indexing safely
        if i < len(level_prices):
            level_price = level_prices[i]
        else:
            continue

        atr_value = atr[signal_idx]
        buffer = atr_value * _BUFFER_MULTIPLIER

        # Follow-through window: signal_idx + 1 to signal_idx + follow_through_bars
        ft_start = signal_idx + 1
        ft_end = min(signal_idx + config.follow_through_bars + 1, n)

        if ft_start >= ft_end:
            continue

        ft_close = close[ft_start:ft_end]
        ft_high = high[ft_start:ft_end]
        ft_low = low[ft_start:ft_end]

        qualifying_closes = 0
        required_closes = int(np.ceil(config.follow_through_close_ratio * config.follow_through_bars))
        min_required = max(1, required_closes)  # At least 1 qualifying close

        for j in range(len(ft_close)):
            current_close = ft_close[j]
            current_high = ft_high[j]
            current_low = ft_low[j]

            if direction == "bullish":
                # Check if close is above breakout level
                if current_close > (level_price + buffer):
                    # Check close location (close near high of bar)
                    bar_range = max(current_high - current_low, atr_value)
                    if bar_range > 0:
                        close_location = (current_close - current_low) / bar_range
                        if close_location >= config.follow_through_close_ratio:
                            qualifying_closes += 1
            else:  # bearish
                if current_close < (level_price - buffer):
                    bar_range = max(current_high - current_low, atr_value)
                    if bar_range > 0:
                        close_location = (current_high - current_close) / bar_range
                        if close_location >= config.follow_through_close_ratio:
                            qualifying_closes += 1

        # Validate follow-through
        if qualifying_closes >= min_required:
            valid_follow_through[signal_idx] = True

    return valid_follow_through


def validate_signals(
        raw_signals: RawSignals,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        config: SignalValidatorConfig
) -> ValidatedSignals:
    """
    Validate raw break signals using retest dynamics and follow-through confirmation.
    """
    n = len(close)
    input_arrays = [high, low, atr]
    if not all(len(arr) == n for arr in input_arrays):
        raise ValueError("All price/ATR arrays must have same length as close")

    if len(raw_signals.is_bos_bullish_initial) != n:
        raise ValueError("Raw signals array must match price array length")

    confirmed_bull = np.zeros(n, dtype=bool)
    confirmed_bear = np.zeros(n, dtype=bool)
    momentum_bull = np.zeros(n, dtype=bool)
    momentum_bear = np.zeros(n, dtype=bool)
    failure_bull = np.zeros(n, dtype=bool)
    failure_bear = np.zeros(n, dtype=bool)

    # Process bullish breakouts
    bull_break_idx = np.where(raw_signals.is_bos_bullish_initial)[0]
    if len(bull_break_idx) > 0:
        bull_metrics = compute_pullback_metrics(
            bull_break_idx, high[bull_break_idx], close, atr,
            "bullish", config, return_full_metrics=False
        )

        velocity_at_break, bars_at_break, attempts_at_break = _extract_breakout_metrics(
            bull_break_idx, bull_metrics[0], bull_metrics[1],
            bull_metrics[3], config.pullback_max_bars
        )

        bull_retest_confirmed, bull_momentum, bull_failure = _classify_breakout_signals(
            velocity_at_break, bars_at_break, attempts_at_break, config
        )

        bull_levels = high[bull_break_idx]
        bull_follow_through = _validate_follow_through(
            bull_retest_confirmed, bull_levels, close, high, low, atr, "bullish", config
        )

        confirmed_bull[bull_break_idx] = bull_retest_confirmed & bull_follow_through
        momentum_bull[bull_break_idx] = bull_momentum
        failure_bull[bull_break_idx] = bull_failure

    # Process bearish breakouts
    bear_break_idx = np.where(raw_signals.is_bos_bearish_initial)[0]
    if len(bear_break_idx) > 0:
        bear_metrics = compute_pullback_metrics(
            bear_break_idx, low[bear_break_idx], close, atr,
            "bearish", config, return_full_metrics=False
        )

        velocity_at_break, bars_at_break, attempts_at_break = _extract_breakout_metrics(
            bear_break_idx, bear_metrics[0], bear_metrics[1],
            bear_metrics[3], config.pullback_max_bars
        )

        bear_retest_confirmed, bear_momentum, bear_failure = _classify_breakout_signals(
            velocity_at_break, bars_at_break, attempts_at_break, config
        )

        bear_levels = low[bear_break_idx]
        bear_follow_through = _validate_follow_through(
            bear_retest_confirmed, bear_levels, close, high, low, atr, "bearish", config
        )

        confirmed_bear[bear_break_idx] = bear_retest_confirmed & bear_follow_through
        momentum_bear[bear_break_idx] = bear_momentum
        failure_bear[bear_break_idx] = bear_failure

    (immediate_fail_bull, immediate_fail_bear,
     failed_choch_bull, failed_choch_bear) = _mark_immediate_failures(
        raw_signals, close, high, low, atr, config
    )

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


# ==============================================================================
# SECTION: Extended Metrics Access
# ==============================================================================

def get_full_retest_metrics(
        raw_signals: RawSignals,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        config: SignalValidatorConfig
) -> Tuple[Optional[RetestMetrics], Optional[RetestMetrics]]:
    """
    Get complete retest metrics for both bullish and bearish breakouts.
    """
    bullish_metrics = None
    bearish_metrics = None

    bull_break_idx = np.where(raw_signals.is_bos_bullish_initial)[0]
    if len(bull_break_idx) > 0:
        bullish_metrics = compute_pullback_metrics(
            bull_break_idx, high[bull_break_idx], close, atr,
            "bullish", config, return_full_metrics=True
        )

    bear_break_idx = np.where(raw_signals.is_bos_bearish_initial)[0]
    if len(bear_break_idx) > 0:
        bearish_metrics = compute_pullback_metrics(
            bear_break_idx, low[bear_break_idx], close, atr,
            "bearish", config, return_full_metrics=True
        )

    return bullish_metrics, bearish_metrics


# ==============================================================================
# SECTION: Quality Scoring & Enhanced Validation
# ==============================================================================

def _compute_choch_quality(
        signal_mask: np.ndarray,
        momentum: np.ndarray,
        structure: Dict[str, np.ndarray],
        retest_velocity: np.ndarray,
        bars_to_retest: np.ndarray,
        config: SignalValidatorConfig
) -> np.ndarray:
    """
    Compute CHOCH signal quality (different weights than BOS).
    """
    n = len(signal_mask)
    quality = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if signal_mask[i]:
            score = 0.0

            # 1. Structure contribution (40% for CHOCH)
            if structure.get('is_impulse', np.zeros(n, dtype=bool))[i]:
                score += 0.4 * 0.8
            elif structure.get('is_correction', np.zeros(n, dtype=bool))[i]:
                score += 0.4 * 0.5
            else:
                score += 0.4 * 0.3

            # 2. Momentum contribution (30%)
            mom_score = min(1.0, abs(momentum[i]) * 2.0)
            if abs(momentum[i]) >= config.min_momentum_strength:
                score += 0.3 * mom_score

            # 3. Retest velocity (30%)
            if bars_to_retest[i] > 0:
                vel = abs(retest_velocity[i])
                if vel <= config.max_pullback_velocity:
                    vel_score = 1.0 - min(1.0, abs(vel - 0.2) / 0.2)
                    score += 0.3 * vel_score

            quality[i] = min(1.0, score)

    return quality


def _compute_signal_quality(
        signal_mask: np.ndarray,
        momentum: np.ndarray,
        zone_score: np.ndarray,
        retest_velocity: np.ndarray,
        bars_to_retest: np.ndarray,
        config: SignalValidatorConfig
) -> np.ndarray:
    """
    Compute signal quality score [0, 1] using momentum, zones, and retest metrics.
    """
    n = len(signal_mask)
    quality = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if signal_mask[i]:
            score = 0.0

            # 1. Momentum contribution (30%)
            mom_score = min(1.0, abs(momentum[i]) * 2.0)
            if abs(momentum[i]) < config.min_momentum_strength:
                continue
            score += 0.3 * mom_score

            # 2. Zone quality contribution (40%)
            zone_score_val = min(1.0, zone_score[i])
            if zone_score_val < config.min_zone_quality:
                continue
            score += 0.4 * zone_score_val

            # 3. Retest velocity contribution (30%)
            if bars_to_retest[i] > 0:
                vel = abs(retest_velocity[i])
                if vel <= config.max_pullback_velocity:
                    vel_score = 1.0 - min(1.0, abs(vel - 0.3) / 0.3)
                    score += 0.3 * vel_score
                else:
                    continue

            quality[i] = min(1.0, score)

    return quality


def validate_signals_with_quality(
        raw_signals: RawSignals,
        momentum: np.ndarray,
        structure: Dict[str, np.ndarray],
        breaks: Dict[str, np.ndarray],
        zones: Tuple,
        config: SignalValidatorConfig
) -> Tuple[ValidatedSignals, SignalQuality]:
    """
    Enhanced validation with momentum, structure, zones, and retest scoring.
    """
    n = len(momentum)
    close = breaks.get('close', np.zeros(n, dtype=np.float32))
    high = breaks.get('high', np.zeros(n, dtype=np.float32))
    low = breaks.get('low', np.zeros(n, dtype=np.float32))
    atr = breaks.get('atr', np.ones(n, dtype=np.float32))
    zone_score = zones[8] if len(zones) > 8 else np.zeros(n, dtype=np.float32)

    validated = validate_signals(
        raw_signals=raw_signals,
        close=close,
        high=high,
        low=low,
        atr=atr,
        config=config
    )

    bullish_metrics, bearish_metrics = get_full_retest_metrics(
        raw_signals, close, high, low, atr, config
    )

    retest_velocity = np.zeros(n, dtype=np.float32)
    bars_to_retest = np.zeros(n, dtype=np.int32)

    if bullish_metrics is not None:
        retest_velocity = np.maximum(retest_velocity, bullish_metrics.retest_velocity)
        bars_to_retest = np.maximum(bars_to_retest, bullish_metrics.bars_to_retest.astype(np.int32))
    if bearish_metrics is not None:
        retest_velocity = np.maximum(retest_velocity, bearish_metrics.retest_velocity)
        bars_to_retest = np.maximum(bars_to_retest, bearish_metrics.bars_to_retest.astype(np.int32))

    bos_bullish_quality = _compute_signal_quality(
        validated.is_bos_bullish_confirmed,
        momentum,
        zone_score,
        retest_velocity,
        bars_to_retest,
        config
    )

    bos_bearish_quality = _compute_signal_quality(
        validated.is_bos_bearish_confirmed,
        -momentum,
        zone_score,
        retest_velocity,
        bars_to_retest,
        config
    )

    choch_bullish_quality = _compute_choch_quality(
        validated.is_failed_choch_bullish,
        momentum,
        structure,
        retest_velocity,
        bars_to_retest,
        config
    )

    choch_bearish_quality = _compute_choch_quality(
        validated.is_failed_choch_bearish,
        -momentum,
        structure,
        retest_velocity,
        bars_to_retest,
        config
    )

    quality = SignalQuality(
        bos_bullish_quality=bos_bullish_quality,
        bos_bearish_quality=bos_bearish_quality,
        choch_bullish_quality=choch_bullish_quality,
        choch_bearish_quality=choch_bearish_quality
    )

    return validated, quality