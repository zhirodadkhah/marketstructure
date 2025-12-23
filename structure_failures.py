import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set


# =============================================================================
# 1. Core Data Structures
# =============================================================================

class BreakLevel:
    """Stateful representation of a broken structural level."""
    __slots__ = (
        'swing_idx', 'price', 'direction', 'role', 'break_idx', 'atr_at_break',
        'is_gap_break', 'max_post_break_high', 'min_post_break_low',
        'retest_active', 'retest_start_idx', 'state'
    )

    def __init__(
            self,
            swing_idx: int,
            price: float,
            direction: str,
            role: str,
            break_idx: int,
            atr_at_break: float,
            is_gap_break: bool = False
    ):
        self.swing_idx = swing_idx
        self.price = price
        self.direction = direction
        self.role = role
        self.break_idx = break_idx
        self.atr_at_break = atr_at_break
        self.is_gap_break = is_gap_break
        self.max_post_break_high = -np.inf
        self.min_post_break_low = np.inf
        self.retest_active = False
        self.retest_start_idx: Optional[int] = None
        self.state = 'broken'


# =============================================================================
# 2. Helper Functions (Pure & Stateless)
# =============================================================================

def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
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


def _is_bullish_reversal_candle(open_p: float, high_p: float, low_p: float, close_p: float, level_price: float,
                                buffer: float) -> bool:
    if close_p <= open_p or (high_p - low_p) < 1e-10:
        return False
    body = close_p - open_p
    lower_wick = min(open_p, close_p) - low_p
    return lower_wick >= 2 * body and (high_p - max(open_p, close_p)) <= 0.3 * (
                high_p - low_p) and low_p <= level_price + buffer


def _is_bearish_reversal_candle(open_p: float, high_p: float, low_p: float, close_p: float, level_price: float,
                                buffer: float) -> bool:
    if close_p >= open_p or (high_p - low_p) < 1e-10:
        return False
    body = open_p - close_p
    upper_wick = high_p - max(open_p, close_p)
    return upper_wick >= 2 * body and (min(open_p, close_p) - low_p) <= 0.3 * (
                high_p - low_p) and high_p >= level_price - buffer


def _validate_input(df: pd.DataFrame) -> None:
    required_cols = {
        'open', 'high', 'low', 'close', 'swing_type', 'trend_state',
        'is_swing_high', 'is_swing_low',
        'is_higher_high', 'is_lower_high', 'is_higher_low', 'is_lower_low'
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing required columns: {missing}")


def _initialize_result(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    signal_cols = [
        'is_bos_bullish_initial', 'is_bos_bearish_initial',
        'is_bos_bullish_confirmed', 'is_bos_bearish_confirmed',
        'is_bos_bullish_momentum', 'is_bos_bearish_momentum',
        'is_choch_bullish', 'is_choch_bearish',
        'is_failed_choch_bullish', 'is_failed_choch_bearish',
        'is_bullish_break_failure', 'is_bearish_break_failure',
        'is_bullish_immediate_failure', 'is_bearish_immediate_failure'
    ]
    for col in signal_cols:
        result[col] = False
    return result


# =============================================================================
# 3. State Manager Class
# =============================================================================

class StructureBreakStateManager:
    """Manages active break levels and their state transitions."""

    def __init__(self, max_active_levels: int = 50):
        self.active_levels: Dict[Tuple[str, int], BreakLevel] = {}
        self.max_active_levels = max_active_levels
        self.keys_to_remove: Set[Tuple[str, int]] = set()

    def register_break(
            self,
            key: Tuple[str, int],
            swing_idx: int,
            price: float,
            direction: str,
            role: str,
            break_idx: int,
            atr_at_break: float,
            is_gap_break: bool,
            current_high: float,
            current_low: float
    ) -> None:
        if key not in self.active_levels:
            level = BreakLevel(swing_idx, price, direction, role, break_idx, atr_at_break, is_gap_break)
            if direction == 'bullish':
                level.max_post_break_high = current_high
            else:
                level.min_post_break_low = current_low
            self.active_levels[key] = level

    def update_level_extremes(self, level: BreakLevel, high: float, low: float) -> None:
        if level.direction == 'bullish':
            level.max_post_break_high = max(level.max_post_break_high, high)
        else:
            level.min_post_break_low = min(level.min_post_break_low, low)

    def handle_immediate_failure(
            self,
            level: BreakLevel,
            close: float,
            buffer: float,
            current_idx: int,
            result: pd.DataFrame
    ) -> None:
        if level.direction == 'bullish' and close < level.price - buffer:
            col = 'is_failed_choch_bearish' if level.role == 'choch' else 'is_bullish_immediate_failure'
            result.at[current_idx, col] = True
            level.state = 'failed_immediate'
            self.keys_to_remove.add((level.role, level.swing_idx) if 'bos' in level.role else
                                    ('choch_bear' if level.direction == 'bearish' else 'choch_bull', level.swing_idx))
        elif level.direction == 'bearish' and close > level.price + buffer:
            col = 'is_failed_choch_bullish' if level.role == 'choch' else 'is_bearish_immediate_failure'
            result.at[current_idx, col] = True
            level.state = 'failed_immediate'
            self.keys_to_remove.add((level.role, level.swing_idx) if 'bos' in level.role else
                                    ('choch_bull' if level.direction == 'bullish' else 'choch_bear', level.swing_idx))

    def handle_momentum_continuation(
            self,
            level: BreakLevel,
            atr_break: float,
            current_idx: int,
            break_idx: int,
            result: pd.DataFrame
    ) -> None:
        moved_far = False
        if level.direction == 'bullish':
            moved_far = (level.max_post_break_high - level.price) >= 1.5 * atr_break
        else:
            moved_far = (level.price - level.min_post_break_low) >= 1.5 * atr_break
        if moved_far:
            col = 'is_bos_bullish_momentum' if level.direction == 'bullish' else 'is_bos_bearish_momentum'
            result.at[break_idx, col] = True
            level.state = 'momentum_confirmed'
            self.keys_to_remove.add(('bos_bull' if level.direction == 'bullish' else 'bos_bear', level.swing_idx))

    def process_retest_zone(
            self,
            level: BreakLevel,
            high: float,
            low: float,
            close: float,
            open_price: float,
            buffer: float,
            current_idx: int,
            result: pd.DataFrame
    ) -> None:
        in_zone = False
        if level.direction == 'bullish':
            in_zone = (low <= level.price + buffer and close >= level.price - buffer)
        else:
            in_zone = (high >= level.price - buffer and close <= level.price + buffer)

        if not level.retest_active and in_zone:
            level.retest_active = True
            level.retest_start_idx = current_idx
        elif level.retest_active:
            if in_zone:
                if level.direction == 'bullish':
                    if _is_bullish_reversal_candle(open_price, high, low, close, level.price, buffer):
                        result.at[level.break_idx, 'is_bos_bullish_confirmed'] = True
                        level.state = 'confirmed'
                        self.keys_to_remove.add(('bos_bull', level.swing_idx))
                    elif close < level.price - buffer:
                        result.at[current_idx, 'is_bullish_break_failure'] = True
                        level.state = 'failed_retest'
                        level.retest_active = False
                        self.keys_to_remove.add(('bos_bull', level.swing_idx))
                else:
                    if _is_bearish_reversal_candle(open_price, high, low, close, level.price, buffer):
                        result.at[level.break_idx, 'is_bos_bearish_confirmed'] = True
                        level.state = 'confirmed'
                        self.keys_to_remove.add(('bos_bear', level.swing_idx))
                    elif close > level.price + buffer:
                        result.at[current_idx, 'is_bearish_break_failure'] = True
                        level.state = 'failed_retest'
                        level.retest_active = False
                        self.keys_to_remove.add(('bos_bear', level.swing_idx))
            else:
                level.retest_active = False

    def cleanup(self) -> None:
        for key in self.keys_to_remove:
            self.active_levels.pop(key, None)
        self.keys_to_remove.clear()

        if len(self.active_levels) > self.max_active_levels:
            oldest_key = min(self.active_levels.keys(), key=lambda k: self.active_levels[k].break_idx)
            self.active_levels.pop(oldest_key, None)


# =============================================================================
# 4. Break Detection Functions
# =============================================================================

def _detect_gap_breaks(
        i: int,
        df_calc: pd.DataFrame,
        targets: Dict[str, Optional[Dict[str, Any]]],
        min_move: float
) -> Dict[str, bool]:
    if i == 0:
        return {k: False for k in ['bos_bull', 'bos_bear', 'choch_bear', 'choch_bull']}

    open_price = df_calc.iloc[i]['open']
    return {
        'bos_bull': bool(targets['bos_bull'] and open_price > targets['bos_bull']['price'] + min_move),
        'bos_bear': bool(targets['bos_bear'] and open_price < targets['bos_bear']['price'] - min_move),
        'choch_bear': bool(targets['choch_bear'] and open_price < targets['choch_bear']['price'] - min_move),
        'choch_bull': bool(targets['choch_bull'] and open_price > targets['choch_bull']['price'] + min_move)
    }


def _emit_break_signals(
        result: pd.DataFrame,
        idx: int,
        role: str
) -> None:
    signal_map = {
        'bos_bull': 'is_bos_bullish_initial',
        'bos_bear': 'is_bos_bearish_initial',
        'choch_bear': 'is_choch_bearish',
        'choch_bull': 'is_choch_bullish'
    }
    if role in signal_map:
        result.at[idx, signal_map[role]] = True


# =============================================================================
# 5. Main Function (Now Clean & Concise)
# =============================================================================

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
    Detect structure breaks with institutional-grade price action logic.
    Fully modular, testable, and maintainable.
    """
    _validate_input(df)

    if df.empty:
        return _initialize_result(df)

    # Setup
    df_calc = df.reset_index(drop=True)
    original_index = df.index.copy()
    df_calc = _compute_metrics(df_calc)
    result = _initialize_result(df_calc)
    swings = _track_swing_sequences(df_calc)
    state_manager = StructureBreakStateManager(max_active_levels)

    # Main loop
    for i in range(len(df_calc)):
        row = df_calc.iloc[i]
        trend = row['trend_state']
        close = row['close']
        high = row['high']
        low = row['low']
        open_price = row['open']
        atr_val = row['atr']
        body_ratio_val = row['body_ratio']
        min_move = atr_val * min_break_atr_mult

        # Get targets
        targets = {
            'bos_bull': swings['hh'][-1] if swings['hh'] and trend == 'uptrend' else None,
            'bos_bear': swings['ll'][-1] if swings['ll'] and trend == 'downtrend' else None,
            'choch_bear': swings['hl'][-1] if swings['hl'] and trend == 'uptrend' else None,
            'choch_bull': swings['lh'][-1] if swings['lh'] and trend == 'downtrend' else None
        }

        # Detect breaks
        gap_breaks = _detect_gap_breaks(i, df_calc, targets, min_move)
        body_cond = row['body']

        for role, target in targets.items():
            if target is None:
                continue

            is_gap = gap_breaks[role]
            direction = 'bullish' if role in ['bos_bull', 'choch_bull'] else 'bearish'
            price = target['price']

            # Check break condition
            break_cond = False
            if direction == 'bullish':
                break_cond = close > price + min_move and body_ratio_val >= min_break_body_ratio and body_cond > 0
            else:
                break_cond = close < price - min_move and body_ratio_val >= min_break_body_ratio and body_cond < 0

            if break_cond:
                key = (role, target['idx'])
                state_manager.register_break(
                    key, target['idx'], price, direction,
                    'bos' if 'bos' in role else 'choch', i, atr_val, is_gap, high, low
                )
                _emit_break_signals(result, i, role)

        # Process active levels
        for level in state_manager.active_levels.values():
            if i < level.break_idx:
                continue

            state_manager.update_level_extremes(level, high, low)
            buffer = level.atr_at_break * 0.5
            bars_since_break = i - level.break_idx

            # Immediate failure
            if level.state == 'broken' and bars_since_break <= 3:
                state_manager.handle_immediate_failure(level, close, buffer, i, result)

            # Momentum continuation
            elif (level.state == 'broken' and level.role == 'bos' and
                  bars_since_break == momentum_continuation_bars):
                state_manager.handle_momentum_continuation(level, level.atr_at_break, i, level.break_idx, result)

            # Retest processing
            elif (level.state == 'broken' and
                  pullback_min_bars <= bars_since_break <= pullback_max_bars):
                state_manager.process_retest_zone(level, high, low, close, open_price, buffer, i, result)

        state_manager.cleanup()

    result.index = original_index
    return result