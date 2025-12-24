# detector.py
from typing import Optional, Dict, Tuple, List, DefaultDict
import numpy as np
import pandas as pd
from collections import defaultdict
from structure.failures.bar import BarProcessor
from structure.failures.config import StructureBreakConfig
from structure.failures.entity import BreakLevel, ResultBuilder, LevelState, SIGNAL_COLS
from structure.failures.retests import _track_swing_sequences
from structure.failures.calcs import _compute_metrics


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

    # ➕ PRE-EXTRACT ALL COLUMNS AS NUMPY ARRAYS (NO .iloc in loop)
    n = len(df_calc)
    close_arr = df_calc['close'].values
    high_arr = df_calc['high'].values
    low_arr = df_calc['low'].values
    open_arr = df_calc['open'].values
    atr_arr = df_calc['atr'].values
    body_ratio_arr = df_calc['body_ratio'].values
    is_bullish_body_arr = df_calc['is_bullish_body'].values
    is_bearish_body_arr = df_calc['is_bearish_body'].values
    close_location_arr = df_calc['close_location'].values
    upper_wick_ratio_arr = df_calc['upper_wick_ratio'].values
    lower_wick_ratio_arr = df_calc['lower_wick_ratio'].values
    trend_state_arr = df_calc['trend_state'].values

    # ➕ MOMENTUM SCHEDULING QUEUE
    momentum_queue: DefaultDict[int, List[BreakLevel]] = defaultdict(list)

    active_levels: Dict[Tuple[str, int], BreakLevel] = {}

    # Main loop — now fully vector-access optimized
    for i in range(n):
        # Fetch scalar values
        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        open_price = open_arr[i]
        atr_val = atr_arr[i]
        body_ratio = body_ratio_arr[i]
        is_bullish_body = is_bullish_body_arr[i]
        is_bearish_body = is_bearish_body_arr[i]
        close_location = close_location_arr[i]
        upper_wick_ratio = upper_wick_ratio_arr[i]
        lower_wick_ratio = lower_wick_ratio_arr[i]
        trend = trend_state_arr[i]

        prev_high = high_arr[i - 1] if i > 0 else None
        prev_low = low_arr[i - 1] if i > 0 else None

        # Process breaks and retests
        BarProcessor.process_bar_vectorized(
            bar_index=i,
            close=close,
            high=high,
            low=low,
            open_price=open_price,
            atr_val=atr_val,
            body_ratio=body_ratio,
            is_bullish_body=is_bullish_body,
            is_bearish_body=is_bearish_body,
            trend=trend,
            prev_high=prev_high,
            prev_low=prev_low,
            swings=swings,
            active_levels=active_levels,
            builder=builder,
            config=config,
            close_location=close_location,
            upper_wick_ratio=upper_wick_ratio,
            lower_wick_ratio=lower_wick_ratio,
            momentum_queue=momentum_queue  # ➕ pass queue to schedule momentum
        )

        # ➕ Process scheduled momentum checks
        if i in momentum_queue:
            for level in momentum_queue[i]:
                if level.state == LevelState.CONFIRMED and level.role == 'bos':
                    start_idx = level.break_idx
                    end_idx = i + 1
                    if level.direction == 'bullish':
                        never_pulled = np.all(low_arr[start_idx:end_idx] > level.price - level.buffer)
                    else:
                        never_pulled = np.all(high_arr[start_idx:end_idx] < level.price + level.buffer)
                    moved_far = level.moved_away_distance >= config.momentum_threshold * level.atr_at_break
                    if moved_far and never_pulled:
                        col = 'is_bos_bullish_momentum' if level.direction == 'bullish' else 'is_bos_bearish_momentum'
                        builder.set_signal(col, level.break_idx)
                        level.state = LevelState.MOMENTUM

    result = builder.build(df)
    result.index = original_index
    return result