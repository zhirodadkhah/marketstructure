import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# ----------------------------
# 1. Core Data Structures
# ----------------------------

class BreakLevel:
    """Stateful representation of a broken structural level."""

    def __init__(
            self,
            swing_idx: int,
            price: float,
            direction: str,  # 'bullish' or 'bearish'
            role: str,  # 'bos' or 'choch'
            break_idx: int,
            atr_at_break: float
    ):
        self.swing_idx = swing_idx
        self.price = price
        self.direction = direction
        self.role = role
        self.break_idx = break_idx
        self.atr_at_break = atr_at_break
        self.max_post_break_high = -np.inf
        self.min_post_break_low = np.inf
        self.retest_started = False
        self.retest_bar_idx: Optional[int] = None
        self.state = 'broken'  # 'broken', 'confirmed', 'failed', 'momentum'


# ----------------------------
# 2. Helper Functions
# ----------------------------

def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized computation of body, range, ATR."""
    df = df.copy()
    df['body'] = df['close'] - df['open']
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio'] = np.where(
        df['candle_range'] > 1e-10,
        np.abs(df['body']) / df['candle_range'],
        0.0
    )

    tr = np.maximum(
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift(1)),
        np.abs(df['low'] - df['close'].shift(1))
    )
    df['atr'] = tr.rolling(window=14, min_periods=1).mean()
    df['atr'] = df['atr'].fillna(method='bfill').fillna(0.001)
    return df


def _track_swing_sequences(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Vectorized-ish swing tracking using boolean masks."""
    swings = {
        'hh': [],
        'hl': [],
        'lh': [],
        'll': []
    }

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


def _get_current_targets(
        swings: Dict[str, List],
        trend_state: str,
        i: int
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Get current BOS/CHOCH targets based on latest swings."""
    return {
        'bos_bull': swings['hh'][-1] if swings['hh'] and trend_state == 'uptrend' else None,
        'bos_bear': swings['ll'][-1] if swings['ll'] and trend_state == 'downtrend' else None,
        'choch_bear': swings['hl'][-1] if swings['hl'] and trend_state == 'uptrend' else None,
        'choch_bull': swings['lh'][-1] if swings['lh'] and trend_state == 'downtrend' else None
    }


def _is_bullish_reversal_candle(
        open_p: float, high_p: float, low_p: float, close_p: float,
        level_price: float, buffer: float, body_ratio: float
) -> bool:
    """Detect hammer or bullish engulfing at support."""
    if close_p <= open_p:
        return False
    body = close_p - open_p
    range_val = high_p - low_p
    if range_val < 1e-10:
        return False

    # Hammer
    lower_wick = min(open_p, close_p) - low_p
    if lower_wick >= 2 * body and (high_p - max(open_p, close_p)) <= 0.3 * range_val:
        return low_p <= level_price + buffer

    return False


def _is_bearish_reversal_candle(
        open_p: float, high_p: float, low_p: float, close_p: float,
        level_price: float, buffer: float, body_ratio: float
) -> bool:
    """Detect shooting star or bearish engulfing at resistance."""
    if close_p >= open_p:
        return False
    body = open_p - close_p
    range_val = high_p - low_p
    if range_val < 1e-10:
        return False

    # Shooting star
    upper_wick = high_p - max(open_p, close_p)
    if upper_wick >= 2 * body and (min(open_p, close_p) - low_p) <= 0.3 * range_val:
        return high_p >= level_price - buffer

    return False


# ----------------------------
# 3. Main Function
# ----------------------------

def detect_structure_breaks(
        df: pd.DataFrame,
        min_break_body_ratio: float = 0.6,
        min_break_atr_mult: float = 0.5,
        pullback_min_bars: int = 2,
        pullback_max_bars: int = 20,
        momentum_continuation_bars: int = 5,
        max_active_levels: int = 50
) -> pd.DataFrame:
    """
    Vectorized-friendly, modular structure break detection.
    Maintains index integrity and uses state objects for clarity.
    """
    # Input validation
    required_cols = {
        'open', 'high', 'low', 'close', 'swing_type', 'trend_state',
        'is_swing_high', 'is_swing_low',
        'is_higher_high', 'is_lower_high', 'is_higher_low', 'is_lower_low'
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing required columns: {missing}")

    if df.empty:
        result = df.copy()
        for col in [
            'is_bos_bullish_initial', 'is_bos_bearish_initial',
            'is_bos_bullish_confirmed', 'is_bos_bearish_confirmed',
            'is_bos_bullish_momentum', 'is_bos_bearish_momentum',
            'is_choch_bullish', 'is_choch_bearish',
            'is_failed_choch_bullish', 'is_failed_choch_bearish',
            'is_bullish_break_failure', 'is_bearish_break_failure',
            'is_bullish_immediate_failure', 'is_bearish_immediate_failure'
        ]:
            result[col] = False
        return result

    # Ensure we work with position-based logic but preserve original index
    df_calc = df.reset_index(drop=True)  # work with RangeIndex internally
    original_index = df.index.copy()

    # Compute metrics
    df_calc = _compute_metrics(df_calc)

    # Initialize result
    result = df_calc.copy()
    for col in [
        'is_bos_bullish_initial', 'is_bos_bearish_initial',
        'is_bos_bullish_confirmed', 'is_bos_bearish_confirmed',
        'is_bos_bullish_momentum', 'is_bos_bearish_momentum',
        'is_choch_bullish', 'is_choch_bearish',
        'is_failed_choch_bullish', 'is_failed_choch_bearish',
        'is_bullish_break_failure', 'is_bearish_break_failure',
        'is_bullish_immediate_failure', 'is_bearish_immediate_failure'
    ]:
        result[col] = False

    # Track swings
    swings = _track_swing_sequences(df_calc)

    # Active levels: key = (role, swing_idx), value = BreakLevel
    active_levels: Dict[Tuple[str, int], BreakLevel] = {}

    n = len(df_calc)
    for i in range(n):
        current_row = df_calc.iloc[i]
        trend = current_row['trend_state']
        close = current_row['close']
        high = current_row['high']
        low = current_row['low']
        open_price = current_row['open']
        atr_val = current_row['atr']
        body_ratio_val = current_row['body_ratio']

        # Get targets
        targets = _get_current_targets(swings, trend, i)
        min_move = atr_val * min_break_atr_mult

        # --- Break Detection ---
        def _register_break(role: str, direction: str, target: Dict[str, Any]):
            key = (role, target['idx'])
            if key not in active_levels:
                active_levels[key] = BreakLevel(
                    swing_idx=target['idx'],
                    price=target['price'],
                    direction=direction,
                    role='bos' if 'bos' in role else 'choch',
                    break_idx=i,
                    atr_at_break=atr_val
                )
                # Set initial move-away
                if direction == 'bullish':
                    active_levels[key].max_post_break_high = high
                else:
                    active_levels[key].min_post_break_low = low

                # Emit signal
                if role == 'bos_bull':
                    result.at[i, 'is_bos_bullish_initial'] = True
                elif role == 'bos_bear':
                    result.at[i, 'is_bos_bearish_initial'] = True
                elif role == 'choch_bear':
                    result.at[i, 'is_choch_bearish'] = True
                elif role == 'choch_bull':
                    result.at[i, 'is_choch_bullish'] = True

        # Check all break conditions
        if (targets['bos_bull'] is not None and
                close > targets['bos_bull']['price'] + min_move and
                body_ratio_val >= min_break_body_ratio and
                current_row['body'] > 0):
            _register_break('bos_bull', 'bullish', targets['bos_bull'])

        if (targets['bos_bear'] is not None and
                close < targets['bos_bear']['price'] - min_move and
                body_ratio_val >= min_break_body_ratio and
                current_row['body'] < 0):
            _register_break('bos_bear', 'bearish', targets['bos_bear'])

        if (targets['choch_bear'] is not None and
                close < targets['choch_bear']['price'] - min_move and
                current_row['body'] < 0):
            _register_break('choch_bear', 'bearish', targets['choch_bear'])

        if (targets['choch_bull'] is not None and
                close > targets['choch_bull']['price'] + min_move and
                current_row['body'] > 0):
            _register_break('choch_bull', 'bullish', targets['choch_bull'])

        # --- Process Active Levels ---
        keys_to_remove = []
        for key, level in active_levels.items():
            if i < level.break_idx:
                continue

            # Update move-away extremes
            if level.direction == 'bullish':
                level.max_post_break_high = max(level.max_post_break_high, high)
            else:
                level.min_post_break_low = min(level.min_post_break_low, low)

            buffer = level.atr_at_break * 0.5
            bars_since_break = i - level.break_idx

            # Immediate failure (within 3 bars)
            if level.state == 'broken' and bars_since_break <= 3:
                if level.direction == 'bullish' and close < level.price - buffer:
                    col = 'is_failed_choch_bearish' if level.role == 'choch' else 'is_bullish_immediate_failure'
                    result.at[i, col] = True
                    level.state = 'failed_immediate'
                    keys_to_remove.append(key)
                elif level.direction == 'bearish' and close > level.price + buffer:
                    col = 'is_failed_choch_bullish' if level.role == 'choch' else 'is_bearish_immediate_failure'
                    result.at[i, col] = True
                    level.state = 'failed_immediate'
                    keys_to_remove.append(key)

            # Momentum continuation
            if (level.state == 'broken' and level.role == 'bos' and
                    bars_since_break == momentum_continuation_bars):
                moved_far = False
                if level.direction == 'bullish':
                    moved_far = (level.max_post_break_high - level.price) >= 1.5 * level.atr_at_break
                else:
                    moved_far = (level.price - level.min_post_break_low) >= 1.5 * level.atr_at_break
                if moved_far:
                    col = 'is_bos_bullish_momentum' if level.direction == 'bullish' else 'is_bos_bearish_momentum'
                    result.at[level.break_idx, col] = True
                    level.state = 'momentum_confirmed'
                    keys_to_remove.append(key)

            # Pullback detection (directional)
            if (level.state == 'broken' and
                    pullback_min_bars <= bars_since_break <= pullback_max_bars):

                moved_away = False
                came_from_correct_side = False
                in_zone = False

                if level.direction == 'bullish':
                    moved_away = (level.max_post_break_high - level.price) >= level.atr_at_break
                    came_from_correct_side = (i > 0 and df_calc.iloc[i - 1]['high'] > level.price + buffer)
                    in_zone = (low <= level.price + buffer and close >= level.price - buffer)
                else:
                    moved_away = (level.price - level.min_post_break_low) >= level.atr_at_break
                    came_from_correct_side = (i > 0 and df_calc.iloc[i - 1]['low'] < level.price - buffer)
                    in_zone = (high >= level.price - buffer and close <= level.price + buffer)

                if moved_away and came_from_correct_side and in_zone:
                    level.retest_started = True
                    level.retest_bar_idx = i

            # Retest outcome
            if level.retest_started and level.retest_bar_idx == i:
                if level.direction == 'bullish':
                    if _is_bullish_reversal_candle(
                            open_price, high, low, close, level.price, buffer, body_ratio_val
                    ):
                        result.at[level.break_idx, 'is_bos_bullish_confirmed'] = True
                        level.state = 'confirmed'
                        keys_to_remove.append(key)
                    elif close < level.price - buffer:
                        result.at[i, 'is_bullish_break_failure'] = True
                        level.state = 'failed_retest'
                        keys_to_remove.append(key)
                else:
                    if _is_bearish_reversal_candle(
                            open_price, high, low, close, level.price, buffer, body_ratio_val
                    ):
                        result.at[level.break_idx, 'is_bos_bearish_confirmed'] = True
                        level.state = 'confirmed'
                        keys_to_remove.append(key)
                    elif close > level.price + buffer:
                        result.at[i, 'is_bearish_break_failure'] = True
                        level.state = 'failed_retest'
                        keys_to_remove.append(key)

        # Cleanup
        for key in keys_to_remove:
            active_levels.pop(key, None)
        if len(active_levels) > max_active_levels:
            oldest = min(active_levels.keys(), key=lambda k: active_levels[k].break_idx)
            active_levels.pop(oldest, None)

    # Restore original index
    result.index = original_index
    return result