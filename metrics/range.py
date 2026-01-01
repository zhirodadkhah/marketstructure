# structure/metrics/range.py
"""
Range dynamics including expansion quality, inside/outside bars, and volatility regime.
All metrics are pure NumPy, O(n), and designed for price-action signal scoring.
"""
from typing import TypedDict
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .types import Prices, ATRArray, FloatArray, BoolArray


class RangeMetrics(TypedDict):
    raw_range: FloatArray
    range_ratio: FloatArray
    normalized_range: FloatArray
    range_percentile: FloatArray
    range_expansion: BoolArray
    range_compression: BoolArray
    range_squeeze: BoolArray
    volatility_regime: FloatArray
    # ➕ GROUP 4: NEW METRICS
    is_inside_bar: BoolArray
    is_outside_bar: BoolArray
    range_expansion_quality: FloatArray


def compute_range_dynamics(
        high: Prices,
        low: Prices,
        close: Prices,
        atr: ATRArray,
        normalized_momentum: FloatArray,
        range_window: int = 20,
        expansion_threshold: float = 1.5,
        compression_threshold: float = 0.7,
        squeeze_threshold: float = 0.5,
        volatility_regime_window: int = 50
) -> RangeMetrics:
    """
    Compute extended range dynamics including inside/outside bars and quality scoring.

    Parameters
    ----------
    high, low, close, atr, normalized_momentum : Prices
        Input price and metric arrays (all same length).
    range_window : int, default=20
        Window for rolling range percentile.
    expansion_threshold : float, default=1.5
        Normalized range multiplier for expansion detection.
    compression_threshold : float, default=0.7
        Normalized range multiplier for compression.
    squeeze_threshold : float, default=0.5
        Normalized range multiplier for squeeze.
    volatility_regime_window : int, default=50
        Window for volatility smoothing.

    Returns
    -------
    RangeMetrics
        Typed dictionary containing all range dynamics metrics.

    Raises
    ------
    ValueError
        If window sizes < 1 or array shapes mismatch.
    TypeError
        If inputs are non-numeric.

    Notes
    -----
    - `range_expansion_quality` is a composite score (0–1) based on:
        * Range ratio strength
        * Momentum direction alignment
        * Close location conviction
        * Expansion continuation
    - Inside/outside bars use strict 1-bar lookback logic
    - All outputs are float32/bool arrays of same length as input
    """
    # === INPUT VALIDATION ===
    n = len(close)
    inputs = [high, low, close, atr, normalized_momentum]
    if not all(arr.shape == (n,) and arr.ndim == 1 for arr in inputs):
        raise ValueError("All arrays must be 1D and same length")
    if not all(np.issubdtype(arr.dtype, np.number) for arr in inputs):
        raise TypeError("All inputs must be numeric arrays")
    if min(range_window, volatility_regime_window) < 1:
        raise ValueError("Windows must be ≥ 1")

    # === PREPARE ARRAYS ===
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    atr = np.asarray(atr, dtype=np.float32)
    normalized_momentum = np.asarray(normalized_momentum, dtype=np.float32)

    raw_range = high - low

    # === RANGE RATIO ===
    range_ratio = np.full(n, np.nan, dtype=np.float32)
    if n > 1:
        range_ratio[1:] = raw_range[1:] / np.maximum(raw_range[:-1], 1e-10)

    # === NORMALIZED RANGE ===
    safe_atr = np.where(atr > 1e-10, atr, 1e-10)
    norm_range = raw_range / safe_atr

    # === RANGE PERCENTILE ===
    range_percentile = np.full(n, np.nan, dtype=np.float32)
    if n >= range_window:
        windows = sliding_window_view(norm_range, window_shape=range_window)
        roll_min = np.nanmin(windows, axis=1)
        roll_max = np.nanmax(windows, axis=1)
        eps = 1e-10
        pct = (norm_range[range_window - 1:] - roll_min) / np.maximum(roll_max - roll_min, eps)
        range_percentile[range_window - 1:] = pct.astype(np.float32)

    # === VOLATILITY REGIME ===
    vol_regime = np.full(n, np.nan, dtype=np.float32)
    if n >= volatility_regime_window:
        kernel = np.ones(volatility_regime_window, dtype=np.float32) / volatility_regime_window
        vol_regime[volatility_regime_window - 1:] = np.convolve(norm_range, kernel, mode='valid')

    # === FLAGS ===
    expansion = norm_range > expansion_threshold
    compression = norm_range < compression_threshold
    squeeze = norm_range < squeeze_threshold

    # === INSIDE/OUTSIDE BARS (VECTORIZED) ===
    is_inside_bar = np.zeros(n, dtype=bool)
    is_outside_bar = np.zeros(n, dtype=bool)
    if n > 1:
        is_inside_bar[1:] = (high[1:] <= high[:-1]) & (low[1:] >= low[:-1])
        is_outside_bar[1:] = (high[1:] > high[:-1]) & (low[1:] < low[:-1])

    # === RANGE EXPANSION QUALITY SCORE ===
    quality = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if not expansion[i]:
            continue

        score = 0.0

        # Base score from range ratio (0–0.5)
        if range_ratio[i] > 1.0:
            score += min(0.5, (range_ratio[i] - 1.0) / 2.0)

        # Boost if expansion continues (previous bar also expanded)
        if i > 0 and expansion[i - 1]:
            score *= 1.2

        # Boost if momentum aligns with direction
        momentum = normalized_momentum[i]
        if abs(momentum) > 0.5:
            score *= 1.1

        # Boost if close shows directional conviction
        close_loc = (close[i] - low[i]) / (raw_range[i] + 1e-10)
        if close_loc > 0.7 or close_loc < 0.3:
            score *= 1.1

        quality[i] = min(1.0, score)

    return RangeMetrics(
        raw_range=raw_range,
        range_ratio=range_ratio,
        normalized_range=norm_range,
        range_percentile=range_percentile,
        range_expansion=expansion,
        range_compression=compression,
        range_squeeze=squeeze,
        volatility_regime=vol_regime,
        is_inside_bar=is_inside_bar,
        is_outside_bar=is_outside_bar,
        range_expansion_quality=quality
    )