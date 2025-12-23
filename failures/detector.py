from enum import IntEnum
from typing import Optional, Dict, List, Any, Tuple, Set
import numpy as np
import pandas as pd
from structure.failures.config import StructureBreakConfig
from structure.failures.entity import BreakLevel, ResultBuilder, LevelState, SIGNAL_COLS
from structure.failures.retests import RETEST_HANDLERS, _compute_metrics, _track_swing_sequences


def _process_bar(
    i: int,
    close: float,
    high: float,
    low: float,
    open_price: float,
    atr_val: float,
    body_ratio: float,
    is_bullish_body: bool,
    is_bearish_body: bool,
    trend: str,
    prev_high: Optional[float],
    prev_low: Optional[float],
    swings: Dict[str, List[Dict[str, Any]]],
    active_levels: Dict[Tuple[str, int], BreakLevel],
    builder: ResultBuilder,
    config: StructureBreakConfig
) -> None:
    """
    Process a single bar for structure break detection.
    
    :note: This function is performance-critical. All inputs are scalars or precomputed.
    """
    min_move = atr_val * config.min_break_atr_mult

    # Get targets
    targets = {
        'bos_bull': swings['hh'][-1] if swings['hh'] and trend == 'uptrend' else None,
        'bos_bear': swings['ll'][-1] if swings['ll'] and trend == 'downtrend' else None,
        'choch_bear': swings['hl'][-1] if swings['hl'] and trend == 'uptrend' else None,
        'choch_bull': swings['lh'][-1] if swings['lh'] and trend == 'downtrend' else None
    }

    # Detect gap breaks
    is_gap = {k: False for k in targets}
    if i > 0:
        if targets['bos_bull'] and open_price > targets['bos_bull']['price'] + min_move:
            is_gap['bos_bull'] = True
        if targets['bos_bear'] and open_price < targets['bos_bear']['price'] - min_move:
            is_gap['bos_bear'] = True
        if targets['choch_bear'] and open_price < targets['choch_bear']['price'] - min_move:
            is_gap['choch_bear'] = True
        if targets['choch_bull'] and open_price > targets['choch_bull']['price'] + min_move:
            is_gap['choch_bull'] = True

    # Register new breaks
    for role, target in targets.items():
        if target is None:
            continue
        direction = 'bullish' if role in ('bos_bull', 'choch_bull') else 'bearish'
        price = target['price']
        
        break_cond = False
        if direction == 'bullish':
            break_cond = (close > price + min_move and 
                         body_ratio >= config.min_break_body_ratio and 
                         is_bullish_body)
        else:
            break_cond = (close < price - min_move and 
                         body_ratio >= config.min_break_body_ratio and 
                         is_bearish_body)
        
        if break_cond:
            key = (role, target['idx'])
            if key not in active_levels:
                level = BreakLevel(
                    swing_idx=target['idx'],
                    price=price,
                    direction=direction,
                    role='bos' if 'bos' in role else 'choch',
                    break_idx=i,
                    atr_at_break=atr_val,
                    is_gap_break=is_gap[role],
                    config=config
                )
                if not is_gap[role]:
                    if direction == 'bullish':
                        level.max_post_break_high = high
                    else:
                        level.min_post_break_low = low
                active_levels[key] = level
                
                signal_map = {
                    'bos_bull': 'is_bos_bullish_initial',
                    'bos_bear': 'is_bos_bearish_initial',
                    'choch_bear': 'is_choch_bearish',
                    'choch_bull': 'is_choch_bullish'
                }
                builder.set_signal(signal_map[role], i)

    # Process active levels
    keys_to_remove: Set[Tuple[str, int]] = set()
    for key, level in active_levels.items():
        if i < level.break_idx:
            continue

        # Update extremes and movement
        if level.direction == 'bullish':
            level.max_post_break_high = max(level.max_post_break_high, high)
            level.moved_away_distance = level.max_post_break_high - level.price
        else:
            level.min_post_break_low = min(level.min_post_break_low, low)
            level.moved_away_distance = level.price - level.min_post_break_low

        bars_since_break = i - level.break_idx

        # Immediate failure
        if level.state == LevelState.BROKEN and bars_since_break <= 3:
            if level.direction == 'bullish' and close < level.price - level.buffer:
                col = 'is_failed_choch_bearish' if level.role == 'choch' else 'is_bullish_immediate_failure'
                builder.set_signal(col, i)
                level.state = LevelState.FAILED_IMMEDIATE
                keys_to_remove.add(key)
            elif level.direction == 'bearish' and close > level.price + level.buffer:
                col = 'is_failed_choch_bullish' if level.role == 'choch' else 'is_bearish_immediate_failure'
                builder.set_signal(col, i)
                level.state = LevelState.FAILED_IMMEDIATE
                keys_to_remove.add(key)

        # Momentum continuation
        elif (level.state == LevelState.BROKEN and level.role == 'bos' and
              bars_since_break == config.momentum_continuation_bars):
            never_pulled_back = True
            if level.direction == 'bullish':
                # Note: This requires access to df_calc — so we keep momentum check in main loop
                pass  # Handled in main function
            else:
                pass  # Handled in main function

        # Retest processing
        elif (level.state == LevelState.BROKEN and
              config.pullback_min_bars <= bars_since_break <= config.pullback_max_bars):
            handler = RETEST_HANDLERS.get((level.role, level.direction))
            if handler:
                if level.direction == 'bullish':
                    handler(level, high, low, close, open_price, prev_high, i, config, builder)
                else:
                    handler(level, high, low, close, open_price, prev_low, i, config, builder)
                if level.state in (LevelState.CONFIRMED, LevelState.FAILED_RETEST):
                    keys_to_remove.add(key)

    # Cleanup
    for key in keys_to_remove:
        active_levels.pop(key, None)
    if len(active_levels) > config.max_active_levels:
        oldest_key = min(active_levels.keys(), key=lambda k: active_levels[k].break_idx)
        active_levels.pop(oldest_key, None)


def detect_structure_breaks(
    df: pd.DataFrame,
    config: Optional[StructureBreakConfig] = None
) -> pd.DataFrame:
    """
    Detect institutional-grade structure breaks (BOS, CHOCH) and failures using pure price action.

    :param df: DataFrame with OHLC, swing points, market structure, and trend state.
    :param config: Optional configuration (uses defaults if None).
    :return: DataFrame with boolean signal columns for all structure events.
    """
    if config is None:
        config = StructureBreakConfig()

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
        for col in SIGNAL_COLS:
            result[col] = False
        return result

    # Setup
    df_calc = df.reset_index(drop=True)
    original_index = df.index.copy()
    df_calc = _compute_metrics(df_calc)
    builder = ResultBuilder(len(df_calc))
    swings = _track_swing_sequences(df_calc)
    active_levels: Dict[Tuple[str, int], BreakLevel] = {}
    n = len(df_calc)

    # Main loop
    for i in range(n):
        row = df_calc.iloc[i]
        trend = row['trend_state']
        close = row['close']
        high = row['high']
        low = row['low']
        open_price = row['open']
        atr_val = row['atr']
        body_ratio = row['body_ratio']
        is_bullish_body = row['is_bullish_body']
        is_bearish_body = row['is_bearish_body']
        
        prev_high = df_calc.iloc[i - 1]['high'] if i > 0 else None
        prev_low = df_calc.iloc[i - 1]['low'] if i > 0 else None

        # Process breaks and retests
        _process_bar(
            i, close, high, low, open_price, atr_val, body_ratio,
            is_bullish_body, is_bearish_body, trend,
            prev_high, prev_low, swings, active_levels, builder, config
        )

        # Handle momentum (requires DataFrame slice — keep here)
        for key, level in active_levels.items():
            if (level.state == LevelState.BROKEN and level.role == 'bos' and
                i - level.break_idx == config.momentum_continuation_bars):
                
                never_pulled_back = True
                if level.direction == 'bullish':
                    post_break_lows = df_calc.iloc[level.break_idx:i + 1]['low'].values
                    never_pulled_back = np.all(post_break_lows > level.price - level.buffer)
                else:
                    post_break_highs = df_calc.iloc[level.break_idx:i + 1]['high'].values
                    never_pulled_back = np.all(post_break_highs < level.price + level.buffer)

                moved_far = level.moved_away_distance >= config.momentum_threshold * level.atr_at_break
                if moved_far and never_pulled_back:
                    col = 'is_bos_bullish_momentum' if level.direction == 'bullish' else 'is_bos_bearish_momentum'
                    builder.set_signal(col, level.break_idx)
                    level.state = LevelState.MOMENTUM

    result = builder.build(df)
    result.index = original_index
    return result