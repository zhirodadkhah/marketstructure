# structure/failures/calcs.py
from typing import Dict, Optional
import pandas as pd
import numpy as np


def _compute_momentum_metrics(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compute momentum, acceleration, and normalized versions.

    Args:
        df: DataFrame with 'close' and 'atr' columns
        config: Optional configuration dict

    Returns:
        DataFrame with momentum columns added
    """
    df = df.copy()

    # Default configuration
    default_config = {
        'momentum_period': 1,  # Bar-over-bar momentum
        'acceleration_period': 1,  # Change in momentum
        'normalize_by_atr': True,
        'smooth_momentum': False,
        'smoothing_period': 3,
    }

    cfg = {**default_config, **(config or {})}

    close = df['close'].values.astype(np.float32)
    atr = df['atr'].values.astype(np.float32)
    n = len(df)

    # 1. Simple Momentum (velocity) - bar-over-bar close delta
    momentum = np.zeros(n, dtype=np.float32)
    if cfg['momentum_period'] == 1:
        momentum[1:] = close[1:] - close[:-1]
    else:
        # For longer period momentum (e.g., 5-bar momentum)
        for i in range(cfg['momentum_period'], n):
            momentum[i] = close[i] - close[i - cfg['momentum_period']]

    # 2. Acceleration - change in momentum
    acceleration = np.zeros(n, dtype=np.float32)
    if cfg['acceleration_period'] == 1:
        acceleration[2:] = momentum[2:] - momentum[1:-1]
    else:
        for i in range(cfg['acceleration_period'] + 1, n):
            acceleration[i] = momentum[i] - momentum[i - cfg['acceleration_period']]

    # 3. Optional smoothing (simple moving average)
    if cfg['smooth_momentum'] and cfg['smoothing_period'] > 1:
        window = cfg['smoothing_period']
        smoothed_momentum = np.zeros(n, dtype=np.float32)
        for i in range(n):
            start = max(0, i - window + 1)
            smoothed_momentum[i] = momentum[start:i + 1].mean()
        momentum = smoothed_momentum

    # 4. Normalized by ATR (volatility-adjusted)
    normalized_momentum = np.zeros(n, dtype=np.float32)
    normalized_acceleration = np.zeros(n, dtype=np.float32)

    if cfg['normalize_by_atr']:
        safe_atr = np.maximum(atr, 1e-10)  # Avoid division by zero
        normalized_momentum = momentum / safe_atr
        normalized_acceleration = acceleration / safe_atr
    else:
        normalized_momentum = momentum
        normalized_acceleration = acceleration

    # 5. Momentum direction and strength
    momentum_direction = np.sign(momentum).astype(np.int8)
    momentum_strength = np.abs(normalized_momentum)

    # Add columns
    df['momentum'] = momentum
    df['acceleration'] = acceleration
    df['normalized_momentum'] = normalized_momentum
    df['normalized_acceleration'] = normalized_acceleration
    df['momentum_direction'] = momentum_direction
    df['momentum_strength'] = momentum_strength

    # 6. Momentum divergence flags (simplified)
    df['momentum_divergence_bullish'] = False
    df['momentum_divergence_bearish'] = False

    # Simple divergence detection: price makes new high/low but momentum doesn't
    if n > 10:
        # Look for price highs with lower momentum
        for i in range(5, n):
            if df.iloc[i]['is_swing_high'] and i > 5:
                # Check last few swing highs
                prev_highs = []
                for j in range(i - 1, max(0, i - 20), -1):
                    if df.iloc[j]['is_swing_high']:
                        prev_highs.append(j)
                        if len(prev_highs) >= 2:
                            break

                if len(prev_highs) >= 2:
                    # Bearish divergence: higher price, lower momentum
                    price_rising = close[i] > close[prev_highs[0]]
                    momentum_falling = normalized_momentum[i] < normalized_momentum[prev_highs[0]]
                    if price_rising and momentum_falling:
                        df.at[df.index[i], 'momentum_divergence_bearish'] = True

        # Check for bullish divergence at swing lows
        for i in range(5, n):
            if df.iloc[i]['is_swing_low'] and i > 5:
                # Check last few swing lows
                prev_lows = []
                for j in range(i - 1, max(0, i - 20), -1):
                    if df.iloc[j]['is_swing_low']:
                        prev_lows.append(j)
                        if len(prev_lows) >= 2:
                            break

                if len(prev_lows) >= 2:
                    # Bullish divergence: lower price, higher momentum
                    price_falling = close[i] < close[prev_lows[0]]
                    momentum_rising = normalized_momentum[i] > normalized_momentum[prev_lows[0]]
                    if price_falling and momentum_rising:
                        df.at[df.index[i], 'momentum_divergence_bullish'] = True

    return df


def _compute_range_dynamics(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Detect range expansion/compression and volatility regimes.

    Args:
        df: DataFrame with 'high', 'low', and 'atr' columns
        config: Optional configuration dict

    Returns:
        DataFrame with range dynamics columns
    """
    df = df.copy()

    # Default configuration
    default_config = {
        'range_window': 20,
        'expansion_threshold': 1.5,  # > 1.5x median range = expansion
        'compression_threshold': 0.7,  # < 0.7x median range = compression
        'squeeze_threshold': 0.5,  # < 0.5x median range = squeeze
        'volatility_regime_window': 50,
    }

    cfg = {**default_config, **(config or {})}

    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)
    atr = df['atr'].values.astype(np.float32)
    n = len(df)

    # 1. True range (or simple candle range)
    candle_range = high - low

    # 2. Rolling median range (more robust than mean)
    rolling_range = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - cfg['range_window'] + 1)
        rolling_range[i] = np.median(candle_range[start:i + 1])

    # 3. Avoid division by zero and set reasonable defaults
    safe_rolling_range = np.maximum(rolling_range, 1e-10)
    range_ratio = candle_range / safe_rolling_range

    # 4. Range expansion/compression flags
    is_range_expansion = range_ratio > cfg['expansion_threshold']
    is_range_compression = range_ratio < cfg['compression_threshold']
    is_squeeze = range_ratio < cfg['squeeze_threshold']

    # 5. Range acceleration (change in range size)
    range_acceleration = np.zeros(n, dtype=np.float32)
    range_acceleration[1:] = candle_range[1:] - candle_range[:-1]
    normalized_range_acceleration = np.zeros(n, dtype=np.float32)

    safe_atr = np.maximum(atr, 1e-10)
    normalized_range_acceleration = range_acceleration / safe_atr

    # 6. Volatility regime based on ATR percentile
    atr_percentile = np.zeros(n, dtype=np.float32)
    vol_window = min(cfg['volatility_regime_window'], n)

    for i in range(n):
        start = max(0, i - vol_window + 1)
        window_atr = atr[start:i + 1]
        atr_percentile[i] = np.sum(window_atr <= atr[i]) / len(window_atr)

    # Classify volatility regime
    volatility_regime = np.full(n, 'normal', dtype=object)
    volatility_regime[atr_percentile > 0.66] = 'high'
    volatility_regime[atr_percentile < 0.33] = 'low'

    # 7. Range-close relationship
    close = df['close'].values.astype(np.float32) if 'close' in df.columns else (high + low) / 2
    close_location = np.zeros(n, dtype=np.float32)
    safe_range = np.maximum(candle_range, 1e-10)
    close_location = (close - low) / safe_range  # 0 = at low, 1 = at high

    # 8. Inside/outside bar detection
    is_inside_bar = np.zeros(n, dtype=bool)
    is_outside_bar = np.zeros(n, dtype=bool)

    for i in range(1, n):
        # Inside bar: high <= prev high AND low >= prev low
        is_inside_bar[i] = (high[i] <= high[i - 1]) and (low[i] >= low[i - 1])
        # Outside bar: high > prev high AND low < prev low
        is_outside_bar[i] = (high[i] > high[i - 1]) and (low[i] < low[i - 1])

    # Add all columns
    df['candle_range'] = candle_range
    df['range_ratio'] = range_ratio
    df['is_range_expansion'] = is_range_expansion
    df['is_range_compression'] = is_range_compression
    df['is_squeeze'] = is_squeeze
    df['range_acceleration'] = range_acceleration
    df['normalized_range_acceleration'] = normalized_range_acceleration
    df['atr_percentile'] = atr_percentile
    df['volatility_regime'] = volatility_regime
    df['close_location'] = close_location
    df['is_inside_bar'] = is_inside_bar
    df['is_outside_bar'] = is_outside_bar

    # 9. Range expansion quality score
    range_quality = np.zeros(n, dtype=np.float32)
    for i in range(n):
        score = 0.0

        # Base score from range ratio
        if is_range_expansion[i]:
            score += min(1.0, (range_ratio[i] - 1.0) / 2.0)  # 0-0.5 score

        # Boost if expansion continues
        if i > 0 and is_range_expansion[i] and is_range_expansion[i - 1]:
            score *= 1.2

        # Boost if with momentum in same direction
        if 'normalized_momentum' in df.columns:
            momentum = df['normalized_momentum'].iloc[i]
            if abs(momentum) > 0.5:
                score *= 1.1

        # Boost if close is near high/low (directional conviction)
        if close_location[i] > 0.7 or close_location[i] < 0.3:
            score *= 1.1

        range_quality[i] = min(1.0, score)

    df['range_expansion_quality'] = range_quality

    return df


def _compute_fractal_efficiency_extended(close: pd.Series) -> Dict[str, pd.Series]:
    """
    Enhanced fractal efficiency with additional metrics.

    Args:
        close: Close price series

    Returns:
        Dictionary with fractal efficiency metrics
    """
    efficiency_period = 10
    if len(close) < efficiency_period:
        empty = pd.Series(0.0, index=close.index, dtype=np.float32)
        return {
            'fractal_efficiency': empty,
            'is_efficient': empty.astype(bool),
            'fractal_slope': empty,
            'fractal_consistency': empty,
        }

    # Original fractal efficiency
    net_change = (close - close.shift(efficiency_period)).abs()
    path_length = close.diff().abs().rolling(window=efficiency_period, min_periods=1).sum()
    epsilon = 1e-10
    fractal_efficiency = net_change / (path_length + epsilon)
    fractal_efficiency = fractal_efficiency.clip(0.0, 1.0).fillna(0.0).astype(np.float32)
    is_efficient = fractal_efficiency >= 0.6

    # Fractal slope (rate of change of efficiency)
    fractal_slope = fractal_efficiency.diff().fillna(0.0).astype(np.float32)

    # Fractal consistency (rolling standard deviation, low = consistent)
    fractal_consistency = fractal_efficiency.rolling(window=20, min_periods=1).std().fillna(0.0).astype(np.float32)

    return {
        'fractal_efficiency': fractal_efficiency,
        'is_efficient': is_efficient,
        'fractal_slope': fractal_slope,
        'fractal_consistency': fractal_consistency,
    }


# Update the main _compute_metrics function to include Group 4 metrics
def _compute_metrics(df: pd.DataFrame, atr_period: int = 14, momentum_config: Optional[Dict] = None,
                     range_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compute all derived price metrics including Group 4 advanced behavior.

    Args:
        df: Input DataFrame containing 'open', 'high', 'low', 'close' columns.
        atr_period: Period for Average True Range calculation
        momentum_config: Optional configuration for momentum metrics
        range_config: Optional configuration for range dynamics

    Returns:
        DataFrame with all metrics added
    """
    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing required columns: {missing}")

    if df.empty:
        result = df.copy()
        # Add all expected columns with empty values
        result = result.assign(
            body=0.0,
            candle_range=0.0,
            body_ratio=0.0,
            close_location=0.5,
            upper_wick=0.0,
            lower_wick=0.0,
            upper_wick_ratio=0.0,
            lower_wick_ratio=0.0,
            is_bullish_body=False,
            is_bearish_body=False,
            atr=0.001,
            momentum_ema=0.0,
            momentum_direction=0,
            momentum_strength=0.0,
            volatility_regime=pd.Categorical([],
                                             categories=['low', 'normal', 'high']),
            vol_percentile=0.0,
            fractal_efficiency=0.0,
            is_efficient=False,
            # ➕ GROUP 4 METRICS
            momentum=0.0,
            acceleration=0.0,
            normalized_momentum=0.0,
            normalized_acceleration=0.0,
            momentum_direction_g4=0,
            momentum_strength_g4=0.0,
            momentum_divergence_bullish=False,
            momentum_divergence_bearish=False,
            candle_range_g4=0.0,
            range_ratio=1.0,
            is_range_expansion=False,
            is_range_compression=False,
            is_squeeze=False,
            range_acceleration=0.0,
            normalized_range_acceleration=0.0,
            atr_percentile=0.5,
            volatility_regime_g4='normal',
            close_location_g4=0.5,
            is_inside_bar=False,
            is_outside_bar=False,
            range_expansion_quality=0.0,
            fractal_slope=0.0,
            fractal_consistency=0.0,
        )
        return result

    # Extract arrays for performance
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values

    # 1. Compute basic candle features (existing)
    candle_features = _compute_candle_features(open_arr, high_arr, low_arr, close_arr)

    # 2. Compute ATR (existing)
    atr = _compute_atr(df['high'], df['low'], df['close'], atr_period)

    # 3. Compute momentum features (existing)
    momentum_features = _compute_momentum(df['close'])

    # 4. Compute volatility regime (existing)
    vol_features = _compute_volatility_regime(atr)

    # 5. Compute fractal efficiency (existing + extended)
    eff_features = _compute_fractal_efficiency_extended(df['close'])

    # 6. ➕ GROUP 4: Compute advanced momentum metrics
    # Create temp DataFrame with basic metrics for momentum calculation
    temp_df = pd.DataFrame({
        'close': close_arr,
        'atr': atr.values,
        'is_swing_high': df['is_swing_high'].values if 'is_swing_high' in df.columns else np.zeros(len(df), dtype=bool),
        'is_swing_low': df['is_swing_low'].values if 'is_swing_low' in df.columns else np.zeros(len(df), dtype=bool),
    }, index=df.index)

    momentum_metrics_df = _compute_momentum_metrics(temp_df, momentum_config)

    # 7. ➕ GROUP 4: Compute range dynamics
    range_df = pd.DataFrame({
        'high': high_arr,
        'low': low_arr,
        'close': close_arr,
        'atr': atr.values,
    }, index=df.index)

    range_metrics_df = _compute_range_dynamics(range_df, range_config)

    # Combine all results
    result = df.copy()

    # Add existing features
    result = result.assign(
        **candle_features,
        atr=atr,
        **momentum_features,
        **vol_features,
    )

    # Add fractal efficiency metrics (including extended ones)
    result['fractal_efficiency'] = eff_features['fractal_efficiency']
    result['is_efficient'] = eff_features['is_efficient']
    result['fractal_slope'] = eff_features['fractal_slope']
    result['fractal_consistency'] = eff_features['fractal_consistency']

    # ➕ Add Group 4 momentum metrics
    momentum_cols = ['momentum', 'acceleration', 'normalized_momentum', 'normalized_acceleration',
                     'momentum_direction', 'momentum_strength', 'momentum_divergence_bullish',
                     'momentum_divergence_bearish']
    for col in momentum_cols:
        if col in momentum_metrics_df.columns:
            result[col + '_g4'] = momentum_metrics_df[col]

    # ➕ Add Group 4 range dynamics metrics
    range_cols = ['candle_range', 'range_ratio', 'is_range_expansion', 'is_range_compression',
                  'is_squeeze', 'range_acceleration', 'normalized_range_acceleration',
                  'atr_percentile', 'volatility_regime', 'close_location', 'is_inside_bar',
                  'is_outside_bar', 'range_expansion_quality']
    for col in range_cols:
        if col in range_metrics_df.columns:
            # Rename to avoid conflicts with existing columns
            new_name = col if col not in result.columns else col + '_g4'
            result[new_name] = range_metrics_df[col]

    return result


# Keep the convenience wrapper for backward compatibility
def compute_all_metrics(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Compute all metrics for structure break detection.

    Args:
        df: Input DataFrame with OHLC data
        atr_period: Period for ATR calculation

    Returns:
        DataFrame with all metrics
    """
    return _compute_metrics(df, atr_period)