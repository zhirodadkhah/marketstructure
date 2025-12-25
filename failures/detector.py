from typing import Optional, Dict, DefaultDict, Tuple, List
import pandas as pd
import numpy as np
from collections import defaultdict

from structure.failures.bar import BarProcessor
from structure.failures.config import StructureBreakConfig
from structure.failures.entity import BreakLevel, ResultBuilder, LevelState, SIGNAL_COLS
from structure.failures.retests import _track_swing_sequences
from structure.failures.calcs import _compute_metrics

# Group 1: Market Regime
from structure.failures.regime import detect_market_regime
from structure.failures.market_hours import tag_sessions, add_liquidity_awareness

# Group 2: Zone Detection
from structure.failures.zones import detect_support_resistance_zones

# Group 3: MTF Context (optional)
try:
    from structure.failures.mtf_detector import detect_mtf_structure_breaks

    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False


class RetestContext:
    """Track retest metrics for Group 5."""
    __slots__ = ('velocity', 'bars', 'quality', 'is_fast', 'is_slow', 'respect_score')

    def __init__(self):
        self.velocity = 0.0
        self.bars = 0
        self.quality = 0.0
        self.is_fast = False
        self.is_slow = False
        self.respect_score = 0.0


def _validate_inputs(df: pd.DataFrame, required_cols: set) -> None:
    """Validate input DataFrame."""
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if df.empty:
        return

    # Check for NaN in required columns
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values")


def _setup_base_metrics(df: pd.DataFrame, config: StructureBreakConfig) -> pd.DataFrame:
    """Compute base metrics with Group 4 advanced metrics."""
    # Prepare config for Group 4 metrics
    momentum_config = {
        'momentum_period': config.momentum_period,
        'acceleration_period': config.acceleration_period,
        'normalize_by_atr': config.normalize_momentum_by_atr,
        'smooth_momentum': config.smooth_momentum,
        'smoothing_period': config.momentum_smoothing_period,
    } if config.enable_advanced_metrics else None

    range_config = {
        'range_window': config.range_window,
        'expansion_threshold': config.range_expansion_threshold,
        'compression_threshold': config.range_compression_threshold,
        'squeeze_threshold': config.squeeze_threshold,
        'volatility_regime_window': config.volatility_regime_window,
    } if config.enable_advanced_metrics else None

    # Compute metrics with Group 4 if enabled
    if config.enable_advanced_metrics:
        return _compute_metrics(df, atr_period=config.atr_period,
                                momentum_config=momentum_config, range_config=range_config)
    else:
        return _compute_metrics(df, atr_period=config.atr_period)


def _apply_group1_context(df: pd.DataFrame, config: StructureBreakConfig,
                          original_index: pd.Index) -> pd.DataFrame:
    """Apply Group 1: Market Regime & Session Context."""
    result = df.copy()

    # 1. Market Regime Detection
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
            result['market_regime'] = detect_market_regime(
                result,
                swing_window=config.regime_swing_window,
                atr_slope_window=config.regime_atr_slope_window,
                consistency_window=config.regime_consistency_window,
                config=regime_config
            )
        except Exception as e:
            result['market_regime'] = pd.Series('neutral', index=result.index, dtype="category")

    # 2. Session Tagging (if datetime index)
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
            result['session'] = tag_sessions(result, config=session_config)
            result = add_liquidity_awareness(result, session_col='session')
        except Exception as e:
            result['session'] = pd.Series('unknown', index=result.index)
            result['liquidity_score'] = 0.5
            result['is_high_liquidity'] = False

    return result


def _apply_group2_context(df: pd.DataFrame, config: StructureBreakConfig) -> pd.DataFrame:
    """Apply Group 2: Zone Detection."""
    if not config.zone_detection_enabled:
        return df

    try:
        return detect_support_resistance_zones(df, config)
    except Exception as e:
        # Fallback: neutral zone info
        df['zone_id'] = -1
        df['zone_price'] = np.nan
        df['zone_strength'] = 0
        df['zone_type'] = 'none'
        df['is_confluence_zone'] = False
        df['retest_quality'] = 0.0  # Note: This is ZONE retest quality, not Group 5
        df['retest_count'] = 0
        df['is_double_test'] = False
        df['is_triple_test'] = False
        df['signal_zone_score'] = 0.0
        return df


def _extract_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract all required arrays for vectorized processing."""
    arrays = {}
    n = len(df)

    # Core price arrays
    core_cols = ['close', 'high', 'low', 'open', 'atr', 'body_ratio',
                 'is_bullish_body', 'is_bearish_body', 'close_location',
                 'upper_wick_ratio', 'lower_wick_ratio', 'trend_state']

    for col in core_cols:
        if col in df.columns:
            if df[col].dtype == bool:
                arrays[col] = df[col].values.astype(bool)
            elif df[col].dtype == object:
                # Convert categorical/string to int codes for efficiency
                if hasattr(df[col], 'cat'):
                    arrays[col] = df[col].cat.codes.values.astype(np.int8)
                else:
                    # Create numeric representation
                    unique_vals = df[col].unique()
                    val_to_code = {val: i for i, val in enumerate(unique_vals)}
                    arrays[col] = np.array([val_to_code.get(v, -1) for v in df[col]], dtype=np.int8)
            else:
                arrays[col] = df[col].values.astype(np.float32)
        else:
            # Provide defaults
            if col in ['close', 'high', 'low', 'open', 'atr']:
                arrays[col] = np.zeros(n, dtype=np.float32)
            elif col in ['is_bullish_body', 'is_bearish_body']:
                arrays[col] = np.zeros(n, dtype=bool)
            elif col == 'trend_state':
                arrays[col] = np.zeros(n, dtype=np.int8)  # 0=neutral, 1=uptrend, -1=downtrend

    # Group 2: Zone arrays
    zone_cols = ['zone_strength', 'retest_quality', 'is_confluence_zone',
                 'retest_count', 'signal_zone_score']

    for col in zone_cols:
        if col in df.columns:
            if df[col].dtype == bool:
                arrays[col] = df[col].values.astype(bool)
            else:
                arrays[col] = df[col].values.astype(np.float32)
        else:
            if col == 'is_confluence_zone':
                arrays[col] = np.zeros(n, dtype=bool)
            else:
                arrays[col] = np.zeros(n, dtype=np.float32)

    # Group 4: Advanced metrics arrays
    advanced_cols = ['momentum_g4', 'acceleration_g4', 'normalized_momentum_g4',
                     'normalized_acceleration_g4', 'is_range_expansion',
                     'is_range_compression', 'is_squeeze', 'range_expansion_quality']

    for col in advanced_cols:
        if col in df.columns:
            if df[col].dtype == bool:
                arrays[col] = df[col].values.astype(bool)
            else:
                arrays[col] = df[col].values.astype(np.float32)
        else:
            if 'is_' in col:
                arrays[col] = np.zeros(n, dtype=bool)
            else:
                arrays[col] = np.zeros(n, dtype=np.float32)

    return arrays


def _process_bar_with_all_context(
        i: int,
        arrays: Dict[str, np.ndarray],
        swings: Dict[str, List[Dict[str, Any]]],
        active_levels: Dict[Tuple[str, int], BreakLevel],
        builder: ResultBuilder,
        momentum_queue: DefaultDict[int, List[BreakLevel]],
        config: StructureBreakConfig,
        retest_contexts: List[RetestContext]
) -> None:
    """Process a single bar with all context groups applied."""
    # Extract values from arrays
    close = arrays['close'][i]
    high = arrays['high'][i]
    low = arrays['low'][i]
    open_price = arrays['open'][i]
    atr_val = arrays['atr'][i]
    body_ratio = arrays['body_ratio'][i]
    is_bullish_body = arrays['is_bullish_body'][i]
    is_bearish_body = arrays['is_bearish_body'][i]
    close_location = arrays['close_location'][i]
    upper_wick_ratio = arrays['upper_wick_ratio'][i]
    lower_wick_ratio = arrays['lower_wick_ratio'][i]

    # Decode trend state
    trend_code = arrays['trend_state'][i]
    trend = {0: 'neutral', 1: 'uptrend', -1: 'downtrend'}.get(trend_code, 'neutral')

    # Group 2: Zone context
    zone_strength = arrays.get('zone_strength', np.zeros(len(arrays['close'])))[i]
    zone_retest_quality = arrays.get('retest_quality', np.zeros(len(arrays['close'])))[i]
    is_confluence_zone = arrays.get('is_confluence_zone', np.zeros(len(arrays['close']), dtype=bool))[i]
    retest_count = arrays.get('retest_count', np.zeros(len(arrays['close'])))[i]
    signal_zone_score = arrays.get('signal_zone_score', np.zeros(len(arrays['close'])))[i]

    # Group 4: Advanced metrics
    momentum_g4 = arrays.get('momentum_g4', np.zeros(len(arrays['close'])))[i]
    acceleration_g4 = arrays.get('acceleration_g4', np.zeros(len(arrays['close'])))[i]
    is_range_expansion = arrays.get('is_range_expansion', np.zeros(len(arrays['close']), dtype=bool))[i]
    is_range_compression = arrays.get('is_range_compression', np.zeros(len(arrays['close']), dtype=bool))[i]
    is_squeeze = arrays.get('is_squeeze', np.zeros(len(arrays['close']), dtype=bool))[i]
    range_expansion_quality = arrays.get('range_expansion_quality', np.zeros(len(arrays['close'])))[i]

    prev_high = arrays['high'][i - 1] if i > 0 else None
    prev_low = arrays['low'][i - 1] if i > 0 else None

    # Prepare zone info for bar processor
    zone_info = {
        'zone_strength': zone_strength,
        'retest_quality': zone_retest_quality,
        'is_confluence_zone': is_confluence_zone,
        'retest_count': retest_count,
        'signal_zone_score': signal_zone_score,
        # Group 4 metrics
        'momentum_g4': momentum_g4,
        'acceleration_g4': acceleration_g4,
        'is_range_expansion': is_range_expansion,
        'is_range_compression': is_range_compression,
        'is_squeeze': is_squeeze,
        'range_expansion_quality': range_expansion_quality,
    }

    # Process the bar
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
        momentum_queue=momentum_queue,
        zone_info=zone_info
    )

    # Track retest context for Group 5
    _track_retest_context(i, active_levels, retest_contexts, config)


def _track_retest_context(
        i: int,
        active_levels: Dict[Tuple[str, int], BreakLevel],
        retest_contexts: List[RetestContext],
        config: StructureBreakConfig
) -> None:
    """Track retest context for the current bar (Group 5)."""
    # Find active retests at this bar
    for level in active_levels.values():
        if (level.retest_active or
                (level.state == LevelState.CONFIRMED and level.bars_to_retest > 0)):

            # Update retest context for this bar
            rc = retest_contexts[i]
            rc.velocity = level.retest_velocity
            rc.bars = level.bars_to_retest
            rc.quality = level.retest_quality_score
            rc.is_fast = level.is_fast_retest
            rc.is_slow = level.is_slow_retest

            # Calculate respect score
            respect_score = level.retest_quality_score
            if level.is_slow_retest:
                respect_score *= config.slow_retest_boost
            elif level.is_fast_retest:
                respect_score *= config.fast_retest_penalty

            rc.respect_score = max(0.0, min(1.0, respect_score))
            break  # Only track the most recent retest


def _process_momentum_queue(
        i: int,
        momentum_queue: DefaultDict[int, List[BreakLevel]],
        arrays: Dict[str, np.ndarray],
        builder: ResultBuilder,
        config: StructureBreakConfig
) -> None:
    """Process scheduled momentum checks."""
    if i not in momentum_queue:
        return

    close_arr = arrays['close']
    high_arr = arrays['high']
    low_arr = arrays['low']

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


def _add_group5_columns(result: pd.DataFrame, retest_contexts: List[RetestContext]) -> pd.DataFrame:
    """Add Group 5 retest metrics to result DataFrame."""
    n = len(result)

    # Convert retest contexts to arrays
    retest_velocity = np.array([rc.velocity for rc in retest_contexts], dtype=np.float32)
    bars_to_retest = np.array([rc.bars for rc in retest_contexts], dtype=np.int32)
    retest_quality = np.array([rc.quality for rc in retest_contexts], dtype=np.float32)
    is_fast_retest = np.array([rc.is_fast for rc in retest_contexts], dtype=bool)
    is_slow_retest = np.array([rc.is_slow for rc in retest_contexts], dtype=bool)
    retest_respect_score = np.array([rc.respect_score for rc in retest_contexts], dtype=np.float32)

    # Add columns
    result['retest_velocity_g5'] = retest_velocity
    result['bars_to_retest_g5'] = bars_to_retest
    result['retest_quality_g5'] = retest_quality
    result['is_fast_retest_g5'] = is_fast_retest
    result['is_slow_retest_g5'] = is_slow_retest
    result['retest_respect_score_g5'] = retest_respect_score

    return result


def detect_structure_breaks(
        df: pd.DataFrame,
        config: Optional[StructureBreakConfig] = None,
        use_mtf: bool = False
) -> pd.DataFrame:
    """
    Detect structure breaks with all context groups.

    Args:
        df: DataFrame with OHLC, swing points, market structure, and trend state.
        config: Optional configuration (uses defaults if None).
        use_mtf: Whether to use MTF detection (Group 3). If True and MTF available,
                uses detect_mtf_structure_breaks instead.

    Returns:
        DataFrame with boolean signal columns and all context columns.
    """
    if config is None:
        config = StructureBreakConfig()

    # Check if we should use MTF detection
    if use_mtf and MTF_AVAILABLE:
        return detect_mtf_structure_breaks(df, config)

    # Validate inputs
    required_cols = {
        'open', 'high', 'low', 'close', 'swing_type', 'trend_state',
        'is_swing_high', 'is_swing_low',
        'is_higher_high', 'is_lower_high', 'is_higher_low', 'is_lower_low'
    }
    _validate_inputs(df, required_cols)

    if df.empty:
        result = df.copy()
        for col in SIGNAL_COLS:
            result[col] = False
        return result

    # Setup
    df_calc = df.reset_index(drop=True)
    original_index = df.index.copy()

    # Step 1: Compute base metrics (includes Group 4)
    df_calc = _setup_base_metrics(df_calc, config)

    # Step 2: Apply Group 1 context (Market Regime & Session)
    df_calc = _apply_group1_context(df_calc, config, original_index)

    # Step 3: Apply Group 2 context (Zone Detection)
    df_calc = _apply_group2_context(df_calc, config)

    # Setup for main processing
    builder = ResultBuilder(len(df_calc))
    swings = _track_swing_sequences(df_calc)

    # Extract all arrays for vectorized processing
    arrays = _extract_arrays(df_calc)
    n = len(df_calc)

    # Initialize data structures
    momentum_queue: DefaultDict[int, List[BreakLevel]] = defaultdict(list)
    active_levels: Dict[Tuple[str, int], BreakLevel] = {}

    # Initialize Group 5 retest contexts
    retest_contexts = [RetestContext() for _ in range(n)]

    # Main processing loop
    for i in range(n):
        # Process bar with all context
        _process_bar_with_all_context(
            i, arrays, swings, active_levels, builder,
            momentum_queue, config, retest_contexts
        )

        # Process momentum queue
        _process_momentum_queue(i, momentum_queue, arrays, builder, config)

    # Build result
    result = builder.build(df_calc)
    result.index = original_index

    # Add Group 5 columns
    result = _add_group5_columns(result, retest_contexts)

    return result


# Helper function for backward compatibility
def detect_structure_breaks_with_mtf(
        df: pd.DataFrame,
        config: Optional[StructureBreakConfig] = None
) -> pd.DataFrame:
    """
    Detect structure breaks with MTF context (Group 3).

    This is a convenience wrapper that always uses MTF if available.
    """
    if MTF_AVAILABLE:
        return detect_mtf_structure_breaks(df, config)
    else:
        print("Warning: MTF detection not available. Falling back to standard detection.")
        return detect_structure_breaks(df, config, use_mtf=False)


def filter_signals_by_context(
        df: pd.DataFrame,
        config: Optional[StructureBreakConfig] = None,
        min_confluence_score: float = 0.6,
        min_retest_quality: float = 0.5,
        min_respect_score: float = 0.6,
        require_confluence_zone: bool = True,
        avoid_compression: bool = True,
        avoid_fast_retests: bool = False
) -> pd.DataFrame:
    """
    Filter signals based on context from all groups.

    Args:
        df: DataFrame with signals and context columns
        config: Configuration (for threshold defaults)
        min_confluence_score: Minimum zone confluence score
        min_retest_quality: Minimum retest quality (Group 5)
        min_respect_score: Minimum retest respect score (Group 5)
        require_confluence_zone: Require confluence zones
        avoid_compression: Avoid range compression zones
        avoid_fast_retests: Avoid fast retests

    Returns:
        Filtered DataFrame
    """
    if config is None:
        config = StructureBreakConfig()

    result = df.copy()

    # Get signal columns
    signal_cols = [col for col in SIGNAL_COLS if col in result.columns]

    # Create filter mask
    filter_mask = pd.Series(False, index=result.index)

    # Group 2: Zone filtering
    if require_confluence_zone and 'is_confluence_zone' in result.columns:
        filter_mask |= ~result['is_confluence_zone']

    if 'signal_zone_score' in result.columns:
        filter_mask |= result['signal_zone_score'] < min_confluence_score

    # Group 4: Range dynamics filtering
    if avoid_compression and 'is_range_compression' in result.columns:
        filter_mask |= result['is_range_compression']

    # Group 5: Retest quality filtering
    if 'retest_quality_g5' in result.columns:
        filter_mask |= result['retest_quality_g5'] < min_retest_quality

    if 'retest_respect_score_g5' in result.columns:
        filter_mask |= result['retest_respect_score_g5'] < min_respect_score

    if avoid_fast_retests and 'is_fast_retest_g5' in result.columns:
        filter_mask |= result['is_fast_retest_g5']

    # Apply filter to all signals
    for col in signal_cols:
        result.loc[filter_mask, col] = False

    # Add filter reason column
    result['filter_reason'] = ''
    if require_confluence_zone and 'is_confluence_zone' in result.columns:
        result.loc[~result['is_confluence_zone'], 'filter_reason'] += 'no_confluence;'

    if avoid_compression and 'is_range_compression' in result.columns:
        result.loc[result['is_range_compression'], 'filter_reason'] += 'compression;'

    if avoid_fast_retests and 'is_fast_retest_g5' in result.columns:
        result.loc[result['is_fast_retest_g5'], 'filter_reason'] += 'fast_retest;'

    return result