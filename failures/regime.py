# structure/failures/regime.py
"""Market regime classification based on price action context."""
from typing import Optional, Dict
import pandas as pd
import numpy as np


def _compute_swing_density(swing_mask: pd.Series, window: int = 100) -> np.ndarray:
    """Vectorized swing density calculation."""
    swing_float = swing_mask.astype(np.float32).values
    n = len(swing_float)

    if n < window:
        density = np.zeros(n, dtype=np.float32)
        for i in range(n):
            density[i] = swing_float[:i + 1].mean()
    else:
        density = np.zeros(n, dtype=np.float32)
        for i in range(window - 1):
            density[i] = swing_float[:i + 1].mean()

        cumsum = np.cumsum(swing_float)
        density[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0], cumsum[:-window]])) / window

    return density


def _compute_atr_slope(atr: pd.Series, window: int = 20) -> np.ndarray:
    """Fast rolling linear regression slope using precomputed weights."""
    values = atr.values.astype(np.float32)
    n = len(values)
    slopes = np.zeros(n, dtype=np.float32)

    if n < window:
        return slopes

    x = np.arange(window, dtype=np.float32)
    x_mean = x.mean()
    x -= x_mean
    x_var = np.dot(x, x)

    for i in range(window - 1):
        if i == 0:
            slopes[i] = 0.0
        else:
            y = values[:i + 1]
            y_mean = y.mean()
            y_centered = y - y_mean
            x_local = np.arange(i + 1, dtype=np.float32) - (i + 1) / 2
            x_var_local = np.dot(x_local, x_local)
            if x_var_local > 1e-10:
                slopes[i] = np.dot(x_local, y_centered) / x_var_local
            else:
                slopes[i] = 0.0

    for i in range(window - 1, n):
        y = values[i - window + 1:i + 1]
        y_mean = y.mean()
        y_centered = y - y_mean
        if x_var > 1e-10:
            slopes[i] = np.dot(x, y_centered) / x_var
        else:
            slopes[i] = 0.0

    return slopes


def _compute_structure_consistency(
        df: pd.DataFrame,
        window: int = 50
) -> np.ndarray:
    """Fast structure consistency calculation using vectorized operations."""
    n = len(df)
    consistency = np.zeros(n, dtype=np.float32)

    has_higher = 'is_higher_low' in df.columns and 'is_higher_high' in df.columns
    has_lower = 'is_lower_low' in df.columns and 'is_lower_high' in df.columns

    if not (has_higher and has_lower):
        return consistency

    direction = np.zeros(n, dtype=np.int8)
    higher_mask = df['is_higher_low'].values | df['is_higher_high'].values
    lower_mask = df['is_lower_low'].values | df['is_lower_high'].values

    direction[higher_mask] = 1
    direction[lower_mask] = -1

    valid_mask = direction != 0
    if not valid_mask.any():
        return consistency

    valid_indices = np.where(valid_mask)[0]
    valid_directions = direction[valid_indices]

    for i in range(n):
        window_end = i + 1
        window_start = max(0, window_end - window)

        mask_in_window = (valid_indices >= window_start) & (valid_indices < window_end)

        if not mask_in_window.any():
            consistency[i] = 0.0
            continue

        window_directions = valid_directions[mask_in_window]

        if len(window_directions) > 0:
            dominant = 1 if np.sum(window_directions) > 0 else -1
            consistent_count = np.sum(window_directions == dominant)
            consistency[i] = consistent_count / len(window_directions)
        else:
            consistency[i] = 0.0

    return consistency


def detect_market_regime(
        df: pd.DataFrame,
        swing_window: int = 100,
        atr_slope_window: int = 20,
        consistency_window: int = 50,
        config: Optional[Dict] = None
) -> pd.Series:
    """
    Classify market regime based on pure price action metrics.

    Returns categorical Series with values:
        'strong_trend' | 'weak_trend' | 'ranging' | 'chop' | 'neutral'

    Args:
        df: DataFrame with required columns
        swing_window: Bars for swing density calculation
        atr_slope_window: Bars for ATR slope calculation
        consistency_window: Bars for structure consistency
        config: Optional configuration dict with thresholds
    """
    required = {'swing_type', 'atr', 'fractal_efficiency'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for regime detection: {missing}")

    default_config = {
        'swing_density_high': 0.02,
        'swing_density_moderate': 0.01,
        'swing_density_low': 0.005,
        'efficiency_high': 0.5,
        'efficiency_low': 0.3,
        'consistency_high': 0.6,
        'atr_slope_threshold': 0.001,
    }

    cfg = {**default_config, **(config or {})}

    n = len(df)

    swing_mask = df['swing_type'].notna()
    swing_density = _compute_swing_density(swing_mask, swing_window)
    atr_slope = _compute_atr_slope(df['atr'], atr_slope_window)
    consistency = _compute_structure_consistency(df, consistency_window)
    efficiency = df['fractal_efficiency'].values

    high_density = swing_density > cfg['swing_density_high']
    moderate_density = swing_density > cfg['swing_density_moderate']
    low_density = swing_density > cfg['swing_density_low']

    high_efficiency = efficiency > cfg['efficiency_high']
    low_efficiency = efficiency < cfg['efficiency_low']

    high_consistency = consistency > cfg['consistency_high']

    rising_vol = atr_slope > cfg['atr_slope_threshold']
    falling_vol = atr_slope < -cfg['atr_slope_threshold']
    stable_vol = ~(rising_vol | falling_vol)

    regime = np.full(n, 'chop', dtype=object)

    strong_trend_mask = (
            high_efficiency &
            high_consistency &
            high_density &
            (rising_vol | falling_vol)
    )
    regime[strong_trend_mask] = 'strong_trend'

    weak_trend_mask = (
            (high_efficiency | high_consistency) &
            moderate_density &
            ~strong_trend_mask
    )
    regime[weak_trend_mask] = 'weak_trend'

    ranging_mask = (
            low_efficiency &
            low_density &
            stable_vol &
            ~(strong_trend_mask | weak_trend_mask)
    )
    regime[ranging_mask] = 'ranging'

    return pd.Series(regime, index=df.index, dtype="category")