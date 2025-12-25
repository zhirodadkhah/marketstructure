from typing import Optional, Dict, DefaultDict, Tuple, List
import pandas as pd
import numpy as np
from collections import defaultdict

from structure import detect_support_resistance_zones
from structure.failures.bar import BarProcessor
from structure.failures.config import StructureBreakConfig
from structure.failures.entity import BreakLevel, ResultBuilder, LevelState, SIGNAL_COLS
from structure.failures.retests import _track_swing_sequences
from structure.failures.calcs import _compute_metrics

# ➕ New imports for regime detection
from structure.failures.regime import detect_market_regime
from structure.failures.market_hours import tag_sessions, add_liquidity_awareness


def detect_structure_breaks(
        df: pd.DataFrame,
        config: Optional[StructureBreakConfig] = None
) -> pd.DataFrame:
    """
    Detect structure breaks with market regime and context awareness.

    Args:
        df: DataFrame with OHLC, swing points, market structure, and trend state.
        config: Optional configuration (uses defaults if None).

    Returns:
        DataFrame with boolean signal columns and context columns.
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

    # Use ATR period from config
    df_calc = _compute_metrics(df_calc, atr_period=config.atr_period)

    builder = ResultBuilder(len(df_calc))
    swings = _track_swing_sequences(df_calc)

    # ➕ MARKET REGIME DETECTION
    if config.enable_regime_detection:
        try:
            regime_config = {
                'swing_density_high': config.regime_swing_density_high,
                'swing_density_moderate': config.regime_swing_density_moderate,
                'swing_density_low': config.regime_swing_density_low,
                'efficiency_high': config.regime_efficiency_high,
                'efficiency_low': config.regime_efficiency_low,
                'consistency_high': config.regime_consistency_high,
                'atr_slope_threshold': config.regime_atr_slope_threshold,
            }
            df_calc['market_regime'] = detect_market_regime(
                df_calc,
                swing_window=config.regime_swing_window,
                atr_slope_window=config.regime_atr_slope_window,
                consistency_window=config.regime_consistency_window,
                config=regime_config
            )
        except Exception as e:
            df_calc['market_regime'] = pd.Series('neutral', index=df_calc.index, dtype="category")

    # ➕ SESSION TAGGING
    if config.enable_session_tagging and isinstance(original_index, pd.DatetimeIndex):
        try:
            session_config = {
                'asia_start': config.session_asia_start,
                'asia_end': config.session_asia_end,
                'london_start': config.session_london_start,
                'london_end': config.session_london_end,
                'ny_start': config.session_ny_start,
                'ny_end': config.session_ny_end,
                'london_ny_overlap_start': config.session_london_ny_overlap_start,
                'london_ny_overlap_end': config.session_london_ny_overlap_end,
                'timezone': config.session_timezone
            }
            df_calc['session'] = tag_sessions(df_calc, config=session_config)
            df_calc = add_liquidity_awareness(df_calc, session_col='session')
        except Exception as e:
            df_calc['session'] = pd.Series('unknown', index=df_calc.index)
            df_calc['liquidity_score'] = 0.5
            df_calc['is_high_liquidity'] = False

    if config.zone_detection_enabled:
        try:
            df_calc = detect_support_resistance_zones(df_calc, config)
        except Exception as e:
            # Fallback: neutral zone info
            df_calc['zone_id'] = -1
            df_calc['zone_price'] = np.nan
            df_calc['zone_strength'] = 0
            df_calc['zone_type'] = 'none'
            df_calc['is_confluence_zone'] = False
            df_calc['retest_quality'] = 0.0
            df_calc['retest_count'] = 0
            df_calc['is_double_test'] = False
            df_calc['is_triple_test'] = False
            df_calc['signal_zone_score'] = 0.0

    # ➕ PRE-EXTRACT ALL COLUMNS AS NUMPY ARRAYS
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

    # Main loop
    for i in range(n):
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
            momentum_queue=momentum_queue
        )

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

    # Build result
    result = builder.build(df_calc)
    result.index = original_index

    return result