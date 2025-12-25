from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from structure.failures.config import StructureBreakConfig
from structure.failures.entity import BreakLevel, ResultBuilder, LevelState

def _track_swing_sequences(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    swings = {'hh': [], 'hl': [], 'lh': [], 'll': []}
    for i in range(len(df)):
        if df.iloc[i]['is_swing_high']:
            if df.iloc[i]['is_higher_high']:
                swings['hh'].append({'idx': i, 'price': df.iloc[i]['high']})
            elif df.iloc[i]['is_lower_high']:
                swings['lh'].append({'idx': i, 'price': df.iloc[i]['high']})
        if df.iloc[i]['is_swing_low']:
            if df.iloc[i]['is_higher_low']:
                swings['hl'].append({'idx': i, 'price': df.iloc[i]['low']})
            elif df.iloc[i]['is_lower_low']:
                swings['ll'].append({'idx': i, 'price': df.iloc[i]['low']})
    return swings

def _is_bullish_reversal_candle(
    open_p: float, high_p: float, low_p: float, close_p: float,
    level_price: float, buffer: float, wick_body_ratio: float
) -> bool:
    if close_p <= open_p or (high_p - low_p) < 1e-10:
        return False
    body = close_p - open_p
    lower_wick = min(open_p, close_p) - low_p
    return lower_wick >= wick_body_ratio * body and low_p <= level_price + buffer

def _is_bearish_reversal_candle(
    open_p: float, high_p: float, low_p: float, close_p: float,
    level_price: float, buffer: float, wick_body_ratio: float, upper_wick_ratio: float
) -> bool:
    if close_p >= open_p or (high_p - low_p) < 1e-10:
        return False
    body = open_p - close_p
    upper_wick = high_p - max(open_p, close_p)
    return (upper_wick >= wick_body_ratio * body and
            (min(open_p, close_p) - low_p) <= upper_wick_ratio * (high_p - low_p) and
            high_p >= level_price - buffer)

def _compute_pullback_velocity(
    level: BreakLevel,
    current_idx: int,
    price_at_retest: float,
    atr: float
) -> Tuple[float, int, float]:
    bars_since_break = current_idx - level.break_idx
    if level.direction == 'bullish':
        pullback_distance = max(0, level.price - price_at_retest)
    else:
        pullback_distance = max(0, price_at_retest - level.price)
    if bars_since_break > 0 and atr > 1e-10:
        velocity_atr = (pullback_distance / atr) / bars_since_break
    else:
        velocity_atr = 0.0
    return velocity_atr, bars_since_break, pullback_distance

def _validate_temporal_conditions(
    level: BreakLevel,
    bars_since_break: int,
    velocity_atr: float,
    config: StructureBreakConfig
) -> Tuple[bool, str]:
    if bars_since_break < config.pullback_min_bars:
        return False, f"Too early: {bars_since_break} < {config.pullback_min_bars} bars"
    if bars_since_break > config.pullback_max_bars:
        return False, f"Too late: {bars_since_break} > {config.pullback_max_bars} bars"
    if velocity_atr > config.max_pullback_velocity:
        return False, f"Too fast: {velocity_atr:.2f} > {config.max_pullback_velocity} ATR/bar"
    return True, "Valid retest"

def _calculate_retest_quality(
    level: BreakLevel,
    bars_since_break: int,
    velocity_atr: float,
    config: StructureBreakConfig,
    reversal_candle: bool = False
) -> float:
    score = 0.5
    # Time factor
    if bars_since_break >= config.min_retest_respect_bars:
        time_score = min(1.0, bars_since_break / (config.pullback_max_bars / 2))
        score += time_score * 0.2
    else:
        penalty = 1.0 - (bars_since_break / config.min_retest_respect_bars)
        score -= penalty * 0.3
    # Velocity factor
    ideal_velocity = config.max_pullback_velocity * 0.5
    velocity_ratio = velocity_atr / ideal_velocity if ideal_velocity > 0 else 1.0
    if velocity_ratio <= 1.0:
        velocity_score = 1.0 - velocity_ratio * 0.5
        score += velocity_score * 0.2
    else:
        excess = velocity_ratio - 1.0
        penalty = min(0.3, excess * 0.5)
        score -= penalty
    # Reversal candle bonus
    if reversal_candle:
        score += 0.1
    # Moved away bonus
    if level.moved_away_distance >= level.atr_at_break:
        atr_multiple = level.moved_away_distance / level.atr_at_break
        if atr_multiple >= 1.0:
            score += min(0.15, (atr_multiple - 1.0) * 0.1)
    # Zone confluence bonus
    if hasattr(level, 'is_confluence_zone') and level.is_confluence_zone:
        if hasattr(level, 'zone_strength'):
            zone_bonus = min(0.1, level.zone_strength * 0.02)
            score += zone_bonus
    return max(0.0, min(1.0, score))

def _handle_bullish_retest(
    level: BreakLevel,
    high: float,
    low: float,
    close: float,
    open_price: float,
    prev_high: Optional[float],
    current_idx: int,
    config: StructureBreakConfig,
    builder: ResultBuilder,
    atr: float
) -> None:
    if level.moved_away_distance < level.atr_at_break:
        return
    in_zone = (low <= level.price + level.buffer and close >= level.price - level.buffer)
    if not in_zone:
        if level.retest_active:
            level.retest_active = False
        return
    came_from_above = prev_high is not None and prev_high > level.price + level.buffer
    velocity_atr, bars_since_break, pullback_distance = _compute_pullback_velocity(
        level, current_idx, low, atr
    )
    is_valid, reason = _validate_temporal_conditions(level, bars_since_break, velocity_atr, config)
    if not is_valid:
        if level.retest_active:
            level.retest_active = False
        return
    reversal_candle = _is_bullish_reversal_candle(
        open_price, high, low, close, level.price, level.buffer, config.wick_body_ratio
    )
    retest_quality = _calculate_retest_quality(level, bars_since_break, velocity_atr, config, reversal_candle)
    level.retest_velocity = velocity_atr
    level.bars_to_retest = bars_since_break
    level.pullback_distance = pullback_distance
    level.retest_quality_score = retest_quality
    level.is_fast_retest = bars_since_break < config.min_retest_respect_bars
    level.is_slow_retest = bars_since_break >= config.min_retest_respect_bars
    if not level.retest_active and came_from_above:
        level.retest_attempts += 1
        if level.retest_attempts > 3:
            builder.set_signal('is_bullish_break_failure', current_idx)
            level.state = LevelState.FAILED_RETEST
            return
        level.retest_start_bar = current_idx
        level.retest_active = True
    elif level.retest_active:
        if reversal_candle:
            level.retest_end_bar = current_idx
            level.retest_duration = current_idx - level.retest_start_bar if level.retest_start_bar else 0
            builder.set_signal('is_bos_bullish_confirmed', level.break_idx)
            level.state = LevelState.CONFIRMED
        elif close < level.price - level.buffer:
            builder.set_signal('is_bullish_break_failure', current_idx)
            level.state = LevelState.FAILED_RETEST
            level.retest_active = False

def _handle_bearish_retest(
    level: BreakLevel,
    high: float,
    low: float,
    close: float,
    open_price: float,
    prev_low: Optional[float],
    current_idx: int,
    config: StructureBreakConfig,
    builder: ResultBuilder,
    atr: float
) -> None:
    if level.moved_away_distance < level.atr_at_break:
        return
    in_zone = (high >= level.price - level.buffer and close <= level.price + level.buffer)
    if not in_zone:
        if level.retest_active:
            level.retest_active = False
        return
    came_from_below = prev_low is not None and prev_low < level.price - level.buffer
    velocity_atr, bars_since_break, pullback_distance = _compute_pullback_velocity(
        level, current_idx, high, atr
    )
    is_valid, reason = _validate_temporal_conditions(level, bars_since_break, velocity_atr, config)
    if not is_valid:
        if level.retest_active:
            level.retest_active = False
        return
    reversal_candle = _is_bearish_reversal_candle(
        open_price, high, low, close, level.price, level.buffer,
        config.wick_body_ratio, config.upper_wick_ratio
    )
    retest_quality = _calculate_retest_quality(level, bars_since_break, velocity_atr, config, reversal_candle)
    level.retest_velocity = velocity_atr
    level.bars_to_retest = bars_since_break
    level.pullback_distance = pullback_distance
    level.retest_quality_score = retest_quality
    level.is_fast_retest = bars_since_break < config.min_retest_respect_bars
    level.is_slow_retest = bars_since_break >= config.min_retest_respect_bars
    if not level.retest_active and came_from_below:
        level.retest_attempts += 1
        if level.retest_attempts > 3:
            builder.set_signal('is_bearish_break_failure', current_idx)
            level.state = LevelState.FAILED_RETEST
            return
        level.retest_start_bar = current_idx
        level.retest_active = True
    elif level.retest_active:
        if reversal_candle:
            level.retest_end_bar = current_idx
            level.retest_duration = current_idx - level.retest_start_bar if level.retest_start_bar else 0
            builder.set_signal('is_bos_bearish_confirmed', level.break_idx)
            level.state = LevelState.CONFIRMED
        elif close > level.price + level.buffer:
            builder.set_signal('is_bearish_break_failure', current_idx)
            level.state = LevelState.FAILED_RETEST
            level.retest_active = False

RETEST_HANDLERS: Dict[Tuple[str, str], Callable] = {
    ('bos', 'bullish'): _handle_bullish_retest,
    ('bos', 'bearish'): _handle_bearish_retest,
    ('choch', 'bullish'): _handle_bullish_retest,
    ('choch', 'bearish'): _handle_bearish_retest,
}