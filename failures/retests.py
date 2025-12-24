from typing import Dict, List, Any, Optional, Tuple, Callable

import pandas as pd
import numpy as np
from structure.failures.config import StructureBreakConfig
from structure.failures.entity import BreakLevel, ResultBuilder, LevelState


def _compute_metrics(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Compute derived price metrics from OHLC data for structure break detection.

    Derived metrics include candle body characteristics, wick analysis,
    and volatility measures.

    :param df: Input DataFrame containing 'open', 'high', 'low', 'close' columns.
    :param atr_period: Period for Average True Range calculation (default: 14)
    :return: A copy of `df` with added columns:
        - Basic candle metrics:
            * 'body': close - open
            * 'candle_range': high - low
            * 'body_ratio': |body| / candle_range (0.0 if range near zero)
            * 'is_bullish_body': close > open
            * 'is_bearish_body': close < open
        - Candle wick analysis:
            * 'close_location': Relative position of close within candle range
                              (0 = bottom, 1 = top, 0.5 = neutral default)
            * 'upper_wick': Absolute size of upper wick
            * 'lower_wick': Absolute size of lower wick
            * 'upper_wick_ratio': Upper wick as proportion of candle range
            * 'lower_wick_ratio': Lower wick as proportion of candle range
        - Volatility:
            * 'atr': {atr_period}-period Average True Range, backfilled and floored at 0.001
    :note: Does not modify the original DataFrame.
    """
    df = df.copy()

    # Pre-compute frequently used values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values

    # Basic metrics
    body = close_arr - open_arr
    candle_range = high_arr - low_arr

    # Avoid repeated np.where calls
    safe_range_mask = candle_range > 1e-10
    safe_range = np.where(safe_range_mask, candle_range, 1.0)  # Avoid division by zero

    # Body ratio
    body_ratio = np.abs(body) / safe_range
    body_ratio = np.where(safe_range_mask, body_ratio, 0.0)

    # Close location
    close_location = np.where(
        safe_range_mask,
        (close_arr - low_arr) / safe_range,
        0.5
    )

    # Wick calculations
    max_oc = np.maximum(open_arr, close_arr)
    min_oc = np.minimum(open_arr, close_arr)

    upper_wick = high_arr - max_oc
    lower_wick = min_oc - low_arr

    upper_wick_ratio = upper_wick / safe_range
    lower_wick_ratio = lower_wick / safe_range
    upper_wick_ratio = np.where(safe_range_mask, upper_wick_ratio, 0.0)
    lower_wick_ratio = np.where(safe_range_mask, lower_wick_ratio, 0.0)

    # Assign all at once
    df = df.assign(
        body=body,
        candle_range=candle_range,
        is_bullish_body=close_arr > open_arr,
        is_bearish_body=close_arr < open_arr,
        body_ratio=body_ratio,
        close_location=close_location,
        upper_wick=upper_wick,
        lower_wick=lower_wick,
        upper_wick_ratio=upper_wick_ratio,
        lower_wick_ratio=lower_wick_ratio
    )

    # ATR calculation (unchanged, as rolling is the bottleneck)
    tr0 = df['high'] - df['low']
    tr1 = np.abs(df['high'] - df['close'].shift(1))
    tr2 = np.abs(df['low'] - df['close'].shift(1))
    tr = np.maximum.reduce([tr0, tr1, tr2])

    df['atr'] = tr.rolling(window=atr_period, min_periods=1).mean()
    df['atr'] = df['atr'].fillna(method='bfill').fillna(0.001)

    return df

def _track_swing_sequences(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract chronological sequences of confirmed swing points by type.

    :param df: DataFrame with boolean columns:
        'is_swing_high', 'is_swing_low',
        'is_higher_high', 'is_lower_high',
        'is_higher_low', 'is_lower_low'
    :return: Dictionary with keys 'hh', 'hl', 'lh', 'll'.
        Each value is a list of dicts: {'idx': int, 'price': float},
        ordered by occurrence.
    :note: Only processes rows where swing flags are True.
        Assumes input comes from `detect_swing_points` and `detect_market_structure`.
    """
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
    """
    Detect a bullish reversal candle (e.g., hammer) at a support level.

    :param open_p: Candle open price.
    :param high_p: Candle high price.
    :param low_p: Candle low price.
    :param close_p: Candle close price.
    :param level_price: Price of the support level being tested.
    :param buffer: Tolerance zone around `level_price` (typically 0.5 × ATR).
    :param wick_body_ratio: Minimum ratio of lower wick to body (e.g., 2.0).
    :return: True if candle shows strong bullish rejection at support.
    :note: Requires:
        - Bullish body (close > open)
        - Lower wick ≥ wick_body_ratio × body size
        - Low within [level_price - buffer, level_price + buffer]
    """
    if close_p <= open_p or (high_p - low_p) < 1e-10:
        return False
    body = close_p - open_p
    lower_wick = min(open_p, close_p) - low_p
    return lower_wick >= wick_body_ratio * body and low_p <= level_price + buffer


def _is_bearish_reversal_candle(
    open_p: float, high_p: float, low_p: float, close_p: float,
    level_price: float, buffer: float, wick_body_ratio: float, upper_wick_ratio: float
) -> bool:
    """
    Detect a bearish reversal candle (e.g., shooting star) at a resistance level.

    :param open_p: Candle open price.
    :param high_p: Candle high price.
    :param low_p: Candle low price.
    :param close_p: Candle close price.
    :param level_price: Price of the resistance level being tested.
    :param buffer: Tolerance zone around `level_price`.
    :param wick_body_ratio: Minimum ratio of upper wick to body (e.g., 2.0).
    :param upper_wick_ratio: Maximum allowed lower wick as fraction of total range (e.g., 0.3).
    :return: True if candle shows strong bearish rejection at resistance.
    :note: Requires:
        - Bearish body (close < open)
        - Upper wick ≥ wick_body_ratio × body
        - Lower wick ≤ upper_wick_ratio × (high - low)
        - High within [level_price - buffer, level_price + buffer]
    """
    if close_p >= open_p or (high_p - low_p) < 1e-10:
        return False
    body = open_p - close_p
    upper_wick = high_p - max(open_p, close_p)
    return (upper_wick >= wick_body_ratio * body and
            (min(open_p, close_p) - low_p) <= upper_wick_ratio * (high_p - low_p) and
            high_p >= level_price - buffer)


def _handle_bullish_retest(
    level: BreakLevel,
    high: float, low: float, close: float, open_price: float,
    prev_high: Optional[float],
    current_idx: int,
    config: StructureBreakConfig,
    builder: ResultBuilder
) -> None:
    """
    Manage retest logic for a bullish break (broken resistance now acting as support).

    :param level: Current break level state.
    :param high: Current bar high.
    :param low: Current bar low.
    :param close: Current bar close.
    :param open_price: Current bar open.
    :param prev_high: Previous bar high (for directional validation).
    :param current_idx: Current bar index (position-based).
    :param config: Configuration parameters.
    :param builder: Signal emitter.
    :note: Only processes valid retests that:
        - Moved away by ≥1 ATR after break
        - Return from above the level
        - Limit to 3 retest attempts
    Emits 'is_bos_bullish_confirmed' or 'is_bullish_break_failure' as appropriate.
    """
    if level.moved_away_distance == 0.0:
        movement = level.max_post_break_high - level.price
        level.moved_away_distance = movement if movement > 0 else 0.0

    if level.moved_away_distance < level.atr_at_break:
        return

    in_zone = (low <= level.price + level.buffer and close >= level.price - level.buffer)
    if not in_zone:
        if level.retest_active:
            level.retest_active = False
        return

    came_from_above = prev_high is not None and prev_high > level.price + level.buffer

    if not level.retest_active and came_from_above:
        level.retest_attempts += 1
        if level.retest_attempts > 3:
            builder.set_signal('is_bullish_break_failure', current_idx)
            level.state = LevelState.FAILED_RETEST
            return

        level.retest_active = True
        level.retest_start_idx = current_idx
    elif level.retest_active:
        if _is_bullish_reversal_candle(
            open_price, high, low, close, level.price, level.buffer, config.wick_body_ratio
        ):
            builder.set_signal('is_bos_bullish_confirmed', level.break_idx)
            level.state = LevelState.CONFIRMED
        elif close < level.price - level.buffer:
            builder.set_signal('is_bullish_break_failure', current_idx)
            level.state = LevelState.FAILED_RETEST
            level.retest_active = False


def _handle_bearish_retest(
    level: BreakLevel,
    high: float, low: float, close: float, open_price: float,
    prev_low: Optional[float],
    current_idx: int,
    config: StructureBreakConfig,
    builder: ResultBuilder
) -> None:
    """
    Manage retest logic for a bearish break (broken support now acting as resistance).

    :param level: Current break level state.
    :param high: Current bar high.
    :param low: Current bar low.
    :param close: Current bar close.
    :param open_price: Current bar open.
    :param prev_low: Previous bar low (for directional validation).
    :param current_idx: Current bar index (position-based).
    :param config: Configuration parameters.
    :param builder: Signal emitter.
    :note: Only processes valid retests that:
        - Moved away by ≥1 ATR after break
        - Return from below the level
        - Limit to 3 retest attempts
    Emits 'is_bos_bearish_confirmed' or 'is_bearish_break_failure' as appropriate.
    """
    if level.moved_away_distance == 0.0:
        movement = level.price - level.min_post_break_low
        level.moved_away_distance = movement if movement > 0 else 0.0

    if level.moved_away_distance < level.atr_at_break:
        return

    in_zone = (high >= level.price - level.buffer and close <= level.price + level.buffer)
    if not in_zone:
        if level.retest_active:
            level.retest_active = False
        return

    came_from_below = prev_low is not None and prev_low < level.price - level.buffer

    if not level.retest_active and came_from_below:
        level.retest_attempts += 1
        if level.retest_attempts > 3:
            builder.set_signal('is_bearish_break_failure', current_idx)
            level.state = LevelState.FAILED_RETEST
            return

        level.retest_active = True
        level.retest_start_idx = current_idx
    elif level.retest_active:
        if _is_bearish_reversal_candle(
            open_price, high, low, close, level.price, level.buffer,
            config.wick_body_ratio, config.upper_wick_ratio
        ):
            builder.set_signal('is_bos_bearish_confirmed', level.break_idx)
            level.state = LevelState.CONFIRMED
        elif close > level.price + level.buffer:
            builder.set_signal('is_bearish_break_failure', current_idx)
            level.state = LevelState.FAILED_RETEST
            level.retest_active = False


"""
Mapping from (role, direction) to retest handler functions.

:note: Both BOS and CHOCH levels use the same retest logic,
    as the price action response at a broken level is identical
    regardless of the level's original role.
"""
RETEST_HANDLERS: Dict[Tuple[str, str], Callable] = {
    ('bos', 'bullish'): _handle_bullish_retest,
    ('bos', 'bearish'): _handle_bearish_retest,
    ('choch', 'bullish'): _handle_bullish_retest,
    ('choch', 'bearish'): _handle_bearish_retest,
}
