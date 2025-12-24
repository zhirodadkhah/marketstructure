from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from structure.failures.bar import BarProcessor
from structure.failures.config import StructureBreakConfig
from structure.failures.entity import BreakLevel, ResultBuilder, LevelState, SIGNAL_COLS
from structure.failures.retests import _compute_metrics, _track_swing_sequences


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
    df_calc = _compute_metrics(df_calc)  # now includes close_location, wick ratios
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
        close_location = row['close_location']
        upper_wick_ratio = row['upper_wick_ratio']
        lower_wick_ratio = row['lower_wick_ratio']

        prev_high = df_calc.iloc[i - 1]['high'] if i > 0 else None
        prev_low = df_calc.iloc[i - 1]['low'] if i > 0 else None

        # Process breaks and retests using BarProcessor
        BarProcessor.process_bar(
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
            lower_wick_ratio=lower_wick_ratio
        )

        # Handle momentum (requires DataFrame slice — keep in main loop)
        for key, level in active_levels.items():
            if (level.state == LevelState.CONFIRMED and level.role == 'bos' and
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