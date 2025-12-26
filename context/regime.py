# structure/context/regime.py
"""
Pure NumPy market regime classifier.
Classifies price action into: strong_trend | weak_trend | ranging | chop | neutral
"""
from typing import Dict, Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from structure.metrics.types import Prices, RegimeMask
from .config import RegimeConfig
from structure.metrics.atr import compute_atr
from structure.metrics.efficiency import compute_fractal_efficiency
from structure.core.swings import detect_swing_points
from structure.core.structure import detect_market_structure


def detect_market_regime(
        high: Prices,
        low: Prices,
        close: Prices,
        config: RegimeConfig,
        swings: Optional[Dict[str, np.ndarray]] = None,
        structure: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, RegimeMask]:
    """
    Classify market regime using pure price-action metrics.

    Parameters
    ----------
    high, low, close : Prices
        1D float32 arrays of OHLC prices.
    config : RegimeConfig
        Configuration thresholds and windows.
    swings : dict, optional
        Precomputed swing masks to avoid recomputation.
    structure : dict, optional
        Precomputed structure labels to avoid recomputation.

    Returns
    -------
    dict[str, RegimeMask] with mutually exclusive masks:
        - 'is_strong_trend'
        - 'is_weak_trend'
        - 'is_ranging'
        - 'is_chop'
        - 'is_neutral'

    Pre-conditions
    --------------
    - All inputs are 1D numeric arrays of same length.
    - Config thresholds satisfy: 0 < efficiency_low < efficiency_high < 1
    - All windows ≥ 1

    Post-conditions
    ---------------
    - Output masks are mutually exclusive and exhaustive.
    - Early bars (insufficient data) are labeled 'is_neutral'.
    - All operations are truly vectorized (O(n) time per window).

    Notes
    -----
    Regime logic is based on 4 pillars:
    1. Volatility regime (ATR slope)
    2. Price efficiency (Kaufman's fractal efficiency)
    3. Swing density
    4. Structure consistency
    """
    # === PRECONDITIONS ===
    n = len(close)
    if not all(arr.shape == (n,) and arr.ndim == 1 for arr in (high, low, close)):
        raise ValueError("All inputs must be same-length 1D arrays")
    if not all(np.issubdtype(arr.dtype, np.number) for arr in (high, low, close)):
        raise TypeError("Inputs must be numeric")

    # Validate config
    _validate_config(config)

    # Compute minimum bars needed (INCLUDE ATR period!)
    min_bars = max(
        config.atr_period,  # ← ADDED
        config.volatility_window,
        config.regime_swing_window,
        config.regime_consistency_window,
        config.regime_atr_slope_window
    )

    # === 1. VOLATILITY REGIME ===
    atr = compute_atr(high, low, close, period=config.atr_period)
    is_high_vol, is_low_vol, is_stable_vol = _compute_volatility_regime(atr, config)

    # === 2. PRICE EFFICIENCY ===
    # Using fixed 10-bar efficiency period per Kaufman's original specification
    # for short-term regime classification. This provides optimal balance
    # between responsiveness and noise reduction.
    efficiency = compute_fractal_efficiency(close, period=10)
    is_high_eff = efficiency >= config.regime_efficiency_high
    is_low_eff = efficiency <= config.regime_efficiency_low

    # === 3. SWING DENSITY ===
    if swings is None:
        swings = detect_swing_points(high, low, half_window=2)
    swing_mask = (swings['is_swing_high'] | swings['is_swing_low']).astype(np.float32)
    swing_density = _compute_rolling_mean_robust(swing_mask, config.regime_swing_window)
    is_high_density = swing_density > config.regime_swing_density_high
    is_moderate_density = swing_density > config.regime_swing_density_moderate
    is_low_density = swing_density <= config.regime_swing_density_low

    # === 4. STRUCTURE CONSISTENCY (TRULY VECTORIZED) ===
    if structure is None:
        structure = detect_market_structure(high, low, **swings)
    direction = _compute_structure_direction_safe(structure)
    consistency = _compute_structure_consistency_fast(direction, config.regime_consistency_window)
    is_high_consistency = consistency >= config.regime_consistency_high

    # === 5. ATR SLOPE (TRULY VECTORIZED) ===
    atr_slope = _compute_atr_slope_truly_vectorized(atr, config.regime_atr_slope_window)
    rising_vol = atr_slope > config.regime_atr_slope_threshold
    falling_vol = atr_slope < -config.regime_atr_slope_threshold

    # === 6. REGIME LOGIC (VECTORIZED) ===
    strong = (
            is_high_eff &
            is_high_consistency &
            is_high_density &
            (rising_vol | falling_vol)
    )
    weak = (
            (is_high_eff | is_high_consistency) &
            is_moderate_density &
            ~strong
    )
    ranging = (
            is_low_eff &
            is_low_density &
            is_stable_vol &
            ~(strong | weak)
    )
    chop = ~(strong | weak | ranging)

    # === EARLY BARS HANDLING (VECTORIZED) ===
    early_mask = np.arange(n) < min_bars
    neutral = early_mask
    strong = strong & ~early_mask
    weak = weak & ~early_mask
    ranging = ranging & ~early_mask
    chop = chop & ~early_mask

    # === POSTCONDITION: ENFORCE MUTUAL EXCLUSIVITY ===
    result = {
        'is_strong_trend': strong,
        'is_weak_trend': weak,
        'is_ranging': ranging,
        'is_chop': chop,
        'is_neutral': neutral
    }
    return _enforce_mutual_exclusivity(result)


# === HELPER FUNCTIONS (FULLY VECTORIZED) ===

def _validate_config(config: RegimeConfig) -> None:
    """Validate RegimeConfig parameters."""
    if not (0 < config.regime_efficiency_low < config.regime_efficiency_high < 1):
        raise ValueError("Regime efficiency thresholds must satisfy: 0 < low < high < 1")
    windows = [
        config.atr_period,
        config.volatility_window,
        config.regime_swing_window,
        config.regime_consistency_window,
        config.regime_atr_slope_window
    ]
    if any(w < 1 for w in windows):
        raise ValueError("All window sizes must be ≥ 1")


def _compute_volatility_regime(atr: np.ndarray, config: RegimeConfig) -> tuple:
    """Robust volatility regime computation."""
    n = len(atr)
    if n < config.volatility_window:
        return np.zeros(n, bool), np.zeros(n, bool), np.ones(n, bool)

    windows = sliding_window_view(atr, window_shape=config.volatility_window)
    median_atr = np.nanpercentile(windows, 50, axis=1)

    # Handle all-NaN windows
    all_nan_windows = np.all(np.isnan(windows), axis=1)
    if np.any(all_nan_windows):
        overall_median = np.nanmedian(atr) if np.any(~np.isnan(atr)) else 1.0
        median_atr[all_nan_windows] = overall_median

    full_median = np.full(n, np.nan)
    full_median[config.volatility_window - 1:] = median_atr

    if len(median_atr) > 0:
        first_valid = median_atr[0]
        if np.isnan(first_valid):
            valid_idx = np.where(~np.isnan(median_atr))[0]
            if len(valid_idx) > 0:
                first_valid = median_atr[valid_idx[0]]
            else:
                first_valid = 1.0
        full_median[:config.volatility_window - 1] = first_valid
    else:
        full_median[:] = 1.0

    full_median = np.where(np.isnan(full_median), 1.0, full_median)
    ratio = atr / np.maximum(full_median, np.finfo(np.float32).eps)
    ratio = np.where(np.isnan(ratio), 1.0, ratio)

    high_vol = ratio > config.regime_threshold
    low_vol = ratio < (1.0 / config.regime_threshold)
    stable = ~(high_vol | low_vol)
    return high_vol, low_vol, stable


def _compute_rolling_mean_robust(arr: np.ndarray, window: int) -> np.ndarray:
    """Robust rolling mean with no NaNs."""
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float32)

    arr_filled = np.where(np.isnan(arr), 0.0, arr)

    if n < window:
        cumsum = np.cumsum(arr_filled)
        counts = np.arange(1, n + 1)
        return cumsum / counts

    windows = sliding_window_view(arr_filled, window_shape=window)
    means = np.mean(windows, axis=1)

    result = np.full(n, 0.0, dtype=np.float32)
    result[window - 1:] = means
    if len(means) > 0:
        result[:window - 1] = means[0]
    return result


def _compute_structure_direction_safe(structure: dict) -> np.ndarray:
    """Convert structure labels to directional array with safety checks."""
    n = len(structure['is_higher_high'])
    direction = np.zeros(n, dtype=np.int8)

    bullish = structure['is_higher_high'] | structure['is_higher_low']
    bearish = structure['is_lower_high'] | structure['is_lower_low']

    # Resolve conflicts by setting to neutral (0)
    conflict = bullish & bearish
    if np.any(conflict):
        bullish = bullish & ~conflict
        bearish = bearish & ~conflict

    direction[bullish] = 1
    direction[bearish] = -1
    return direction


def _compute_structure_consistency_fast(direction: np.ndarray, window: int) -> np.ndarray:
    """Fast approximate consistency using cumulative sums (truly O(n))."""
    n = len(direction)
    if n < window:
        return np.zeros(n, dtype=np.float32)

    # Convert to directional counts
    bullish = (direction == 1).astype(np.float32)
    bearish = (direction == -1).astype(np.float32)

    # Cumulative sums for rolling window
    cum_bullish = np.cumsum(bullish)
    cum_bearish = np.cumsum(bearish)

    # Rolling counts using cumulative sums
    roll_bullish = cum_bullish[window - 1:] - np.concatenate([[0], cum_bullish[:-window]])
    roll_bearish = cum_bearish[window - 1:] - np.concatenate([[0], cum_bearish[:-window]])

    # Total valid directions and dominant direction count
    total_valid = roll_bullish + roll_bearish
    max_direction = np.maximum(roll_bullish, roll_bearish)

    # Consistency = dominant / total (0 if no valid directions)
    consistency = np.where(total_valid > 0, max_direction / total_valid, 0.0)

    # Pad to full length
    result = np.zeros(n, dtype=np.float32)
    result[window - 1:] = consistency
    return result


def _compute_atr_slope_truly_vectorized(atr: np.ndarray, window: int) -> np.ndarray:
    """Truly vectorized ATR slope using linear regression."""
    n = len(atr)
    if n < window:
        return np.zeros(n, dtype=np.float32)

    windows = sliding_window_view(atr, window_shape=window)

    # Precompute x statistics
    x = np.arange(window, dtype=np.float32)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = np.dot(x_centered, x_centered)

    # Vectorized computation
    y_means = np.mean(windows, axis=1, keepdims=True)
    y_centered = windows - y_means
    numerator = np.sum(y_centered * x_centered, axis=1)

    if x_var > 1e-10:
        slopes = numerator / x_var
    else:
        slopes = np.zeros_like(numerator)

    # Pad to full length
    result = np.zeros(n, dtype=np.float32)
    result[window - 1:] = slopes
    return result


def _enforce_mutual_exclusivity(masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Ensure masks are mutually exclusive and exhaustive."""
    n = len(next(iter(masks.values())))

    # Priority order: strong → weak → ranging → chop → neutral
    priority_keys = [
        'is_strong_trend',
        'is_weak_trend',
        'is_ranging',
        'is_chop',
        'is_neutral'
    ]

    labels = np.full(n, -1, dtype=np.int8)
    for i, key in enumerate(priority_keys):
        mask = masks[key]
        labels[mask & (labels == -1)] = i

    # Handle any unlabeled bars (shouldn't happen, but safe)
    labels[labels == -1] = 4  # neutral

    result = {}
    for i, key in enumerate(priority_keys):
        result[key] = (labels == i)

    return result