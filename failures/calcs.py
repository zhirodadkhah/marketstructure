from typing import Dict
import pandas as pd
import numpy as np


def _fast_rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Compute fast rolling percentile rank without O(n²) complexity.
    :param series: Input numeric series.
    :param window: Rolling window size (must be > 0).
    :return: Series with values in [0.0, 1.0] representing percentile rank.
        NaN values are preserved in the output.
    :note: For each bar i, computes rank of series[i] within series[i-window+1:i+1].
    Uses optimized indexing to avoid redundant NaN checks.
    Handles edge cases:
        - window = 1 → always returns 1.0 for non-NaN values (correct)
        - all-equal values → returns 1.0
        - NaN in current or window → returns NaN
    Complexity: O(n · k) where k = average number of valid values per window.
    """
    if len(series) == 0:
        return pd.Series([], dtype=np.float32, index=series.index)

    if window <= 0:
        raise ValueError(f"Window must be > 0, got {window}")

    # ➕ EARLY EXIT: window = 1 → always 1.0 for valid values
    if window == 1:
        result = np.where(~np.isnan(series.values), 1.0, np.nan)
        return pd.Series(result, index=series.index, dtype=np.float32)

    values = series.values
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float32)

    # ➕ PRE-FILTER VALID INDICES ONCE
    valid_mask = ~np.isnan(values)
    valid_indices = np.where(valid_mask)[0]
    valid_values = values[valid_mask]

    if len(valid_indices) == 0:
        return pd.Series(result, index=series.index, dtype=np.float32)

    # ➕ PROCESS ONLY VALID INDICES
    for idx in valid_indices:
        start = max(0, idx - window + 1)

        # Find range of valid indices within [start, idx]
        window_start_pos = np.searchsorted(valid_indices, start, side='left')
        window_end_pos = np.searchsorted(valid_indices, idx, side='right')

        window_valid_count = window_end_pos - window_start_pos

        if window_valid_count == 0:
            result[idx] = np.nan
        else:
            current_val = values[idx]
            window_vals = valid_values[window_start_pos:window_end_pos]
            rank_val = np.sum(window_vals <= current_val)
            result[idx] = rank_val / window_valid_count

    return pd.Series(result, index=series.index, dtype=np.float32)

def _compute_candle_features(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute candlestick-derived metrics from OHLC price arrays.
    :param open_arr: Array of open prices.
    :param high_arr: Array of high prices.
    :param low_arr: Array of low prices.
    :param close_arr: Array of close prices.
    :return: Dictionary of NumPy arrays for:
        - 'body': close - open
        - 'candle_range': high - low
        - 'body_ratio': |body| / range (0.0 if range near zero)
        - 'is_bullish_body': close > open
        - 'is_bearish_body': close < open
        - 'close_location': (close - low) / range (0.5 if flat)
        - 'upper_wick', 'lower_wick': absolute wick sizes
        - 'upper_wick_ratio', 'lower_wick_ratio': wicks as % of range
    :note: All outputs are same length as inputs. Division-by-zero is safely guarded.
          Arrays are cast to float32 for memory efficiency.
    """
    # Cast to float32 early for memory savings
    open_arr = open_arr.astype(np.float32)
    high_arr = high_arr.astype(np.float32)
    low_arr = low_arr.astype(np.float32)
    close_arr = close_arr.astype(np.float32)

    body = close_arr - open_arr
    candle_range = high_arr - low_arr
    safe_range_mask = candle_range > 1e-10
    safe_range = np.where(safe_range_mask, candle_range, 1.0)

    body_ratio = np.abs(body) / safe_range
    body_ratio = np.where(safe_range_mask, body_ratio, 0.0)

    close_location = np.where(safe_range_mask, (close_arr - low_arr) / safe_range, 0.5)

    max_oc = np.maximum(open_arr, close_arr)
    min_oc = np.minimum(open_arr, close_arr)
    upper_wick = high_arr - max_oc
    lower_wick = min_oc - low_arr

    upper_wick_ratio = np.where(safe_range_mask, upper_wick / safe_range, 0.0)
    lower_wick_ratio = np.where(safe_range_mask, lower_wick / safe_range, 0.0)

    is_bullish_body = close_arr > open_arr
    is_bearish_body = close_arr < open_arr

    return {
        'body': body,
        'candle_range': candle_range,
        'body_ratio': body_ratio,
        'close_location': close_location,
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
        'upper_wick_ratio': upper_wick_ratio,
        'lower_wick_ratio': lower_wick_ratio,
        'is_bullish_body': is_bullish_body,
        'is_bearish_body': is_bearish_body,
    }


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int
) -> pd.Series:
    """
    Compute Average True Range (ATR) using standard Wilder's method.
    :param high: High prices.
    :param low: Low prices.
    :param close: Close prices.
    :param period: ATR lookback window.
    :return: ATR series, backfilled and floored at 0.001.
    :note: Uses simple moving average (not EMA) for consistency with original.
    """
    if not all(isinstance(s, pd.Series) for s in [high, low, close]):
        raise TypeError("ATR inputs must be pandas Series")
    if period < 1:
        raise ValueError(f"ATR period must be ≥ 1, got {period}")
    if not (len(high) == len(low) == len(close)):
        raise ValueError("High, low, close must have equal length")

    tr0 = high - low
    tr1 = (high - close.shift(1)).abs()
    tr2 = (low - close.shift(1)).abs()
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    atr = atr.bfill().fillna(0.001)
    return atr.astype(np.float32)


def _compute_momentum(close: pd.Series) -> Dict[str, pd.Series]:
    """
    Compute directional momentum features from price series.
    :param close: Close price series.
    :return: Dictionary with:
        - 'momentum_ema': EMA-smoothed % rate of change (14-bar)
        - 'momentum_direction': +1 (up), -1 (down), 0
        - 'momentum_strength': rolling percentile rank of |momentum| (0.0–1.0)
    :note: Strength uses 50-bar lookback for adaptive scaling.
          Uses fast rolling percentile to avoid O(n²) performance trap.
          Handles NaNs safely.
    """
    momentum_period = 14
    # ➕ Explicitly disable fill_method to avoid FutureWarning
    roc = close.pct_change(periods=momentum_period, fill_method=None) * 100
    momentum_ema = roc.ewm(span=momentum_period, adjust=False).mean()

    # ➕ Safe sign conversion: NaN stays NaN (as float), then cast with care
    momentum_sign = np.sign(momentum_ema.values)
    # Convert to int8, but keep NaN as NaN in a float32 series first
    momentum_direction = pd.Series(
        momentum_sign, index=close.index, dtype=np.float32
    ).fillna(0.0).astype(np.int8)

    momentum_strength = _fast_rolling_percentile(momentum_ema.abs(), window=50)
    return {
        'momentum_ema': momentum_ema.astype(np.float32),
        'momentum_direction': momentum_direction,
        'momentum_strength': momentum_strength,
    }


def _compute_volatility_regime(atr: pd.Series) -> Dict[str, pd.Series]:
    """
    Classify market into volatility regimes based on ATR percentile.
    :param atr: ATR series.
    :return: Dictionary with:
        - 'volatility_regime': category ('low', 'normal', 'high')
        - 'vol_percentile': ATR rolling percentile (0.0–1.0)
    :note: Uses 50-bar lookback. Thresholds: low <33%, high >66%.
          Handles zero-ATR case (all equal prices) by forcing 'low'.
    """
    vol_lookback = 50

    # Handle zero-ATR case
    atol = 1e-10
    if np.all(atr.values < atol):
        vol_percentile = pd.Series(0.0, index=atr.index, dtype=np.float32)
    else:
        vol_percentile = _fast_rolling_percentile(atr, window=vol_lookback)

    # Predefine categories
    vol_regime = pd.Categorical(
        ['normal'] * len(atr),
        categories=['low', 'normal', 'high'],
        ordered=False
    )
    vol_regime = pd.Series(vol_regime, index=atr.index)

    vol_regime[vol_percentile < 0.33] = 'low'
    vol_regime[vol_percentile > 0.66] = 'high'

    return {
        'volatility_regime': vol_regime,
        'vol_percentile': vol_percentile,
    }


def _compute_fractal_efficiency(close: pd.Series) -> Dict[str, pd.Series]:
    """
    Compute Kaufman's Fractal Efficiency Ratio (trend quality vs. noise).
    :param close: Close price series.
    :return: Dictionary with:
        - 'fractal_efficiency': 0.0 (chop) to 1.0 (perfect trend)
        - 'is_efficient': boolean (≥0.6 threshold)
    :note: Efficiency = |net change| / total path over 10 bars.
          Safe division with epsilon to prevent NaNs.
    """
    efficiency_period = 10
    if len(close) < efficiency_period:
        empty = pd.Series(0.0, index=close.index, dtype=np.float32)
        return {
            'fractal_efficiency': empty,
            'is_efficient': empty.astype(bool),
        }
    net_change = (close - close.shift(efficiency_period)).abs()
    path_length = close.diff().abs().rolling(window=efficiency_period, min_periods=1).sum()
    epsilon = 1e-10
    fractal_efficiency = net_change / (path_length + epsilon)
    fractal_efficiency = fractal_efficiency.clip(0.0, 1.0).fillna(0.0).astype(np.float32)
    is_efficient = fractal_efficiency >= 0.6
    return {
        'fractal_efficiency': fractal_efficiency,
        'is_efficient': is_efficient,
    }


def _compute_metrics(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Compute derived price metrics from OHLC data for structure break detection.
    :param df: Input DataFrame containing 'open', 'high', 'low', 'close' columns.
    :param atr_period: Period for Average True Range calculation (default: 14)
    :return: A copy of `df` with added columns:
        - Candle features: body, wicks, ratios, directional flags
        - Volatility: 'atr'
        - Momentum: 'momentum_ema', 'momentum_direction', 'momentum_strength'
        - Volatility regime: 'volatility_regime', 'vol_percentile'
        - Fractal efficiency: 'fractal_efficiency', 'is_efficient'
    :note: Does not modify the original DataFrame. All computations are vectorized.
          Critical performance fix: replaces slow pd.rank(pct=True) with fast alternative.
    """
    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Missing required columns: {missing}")

    if df.empty:
        result = df.copy()
        # ➕ Add all expected columns with empty values
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
            is_efficient=False
        )
        return result

    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values

    candle_features = _compute_candle_features(open_arr, high_arr, low_arr, close_arr)
    atr = _compute_atr(df['high'], df['low'], df['close'], atr_period)
    momentum_features = _compute_momentum(df['close'])
    vol_features = _compute_volatility_regime(atr)
    eff_features = _compute_fractal_efficiency(df['close'])

    result = df.copy()
    result = result.assign(
        **candle_features,
        atr=atr,
        **momentum_features,
        **vol_features,
        **eff_features
    )
    return result


# Add to structure/failures/calcs.py (at the end of the file)

def compute_all_metrics(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Compute all metrics for structure break detection.

    This is a convenience wrapper that ensures all required metrics
    are computed for the detector.

    Args:
        df: Input DataFrame with OHLC data
        atr_period: Period for ATR calculation

    Returns:
        DataFrame with all metrics
    """
    return _compute_metrics(df, atr_period)