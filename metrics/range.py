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

def compute_range_metrics(
    high: Prices,
    low: Prices,
    close: Prices,
    atr: ATRArray,
    range_window: int = 20,
    expansion_threshold: float = 1.5,
    compression_threshold: float = 0.7,
    squeeze_threshold: float = 0.5,
    volatility_regime_window: int = 50
) -> RangeMetrics:
    """
    Compute range dynamics metrics.
    :param high: High prices
    :param low: Low prices
    :param close: Close prices
    :param atr: ATR values
    :param range_window: Window for rolling range percentile
    :param expansion_threshold: Normalized range > this → expansion
    :param compression_threshold: Normalized range < this → compression
    :param squeeze_threshold: Normalized range < this → squeeze
    :param volatility_regime_window: Window for volatility smoothing
    :return: Typed dict of range metrics
    :raise ValueError: if window < 1 or arrays mismatched
    """
    # VALIDATE DTYPES — INCLUDE atr
    if not all(
        np.issubdtype(arr.dtype, np.number)
        for arr in (high, low, close, atr)
    ):
        raise TypeError("All input arrays must have numeric dtypes.")

    # VALIDATE SHAPES
    if not (high.shape == low.shape == close.shape):
        raise ValueError("Input arrays must have the same shape.")
    if high.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")

    # ➕ VALIDATE DTYPES
    if not all(np.issubdtype(arr.dtype, np.number) for arr in (high, low, close)):
        raise TypeError("All input arrays must have numeric dtypes.")

    n = len(close)
    if not all(len(x) == n for x in (high, low, atr)):
        raise ValueError("All arrays must have same length")
    if range_window < 1 or volatility_regime_window < 1:
        raise ValueError("Windows must be ≥1")

    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    atr = np.asarray(atr, dtype=np.float32)

    raw_range = high - low
    range_ratio = np.full(n, np.nan, dtype=np.float32)
    range_ratio[1:] = raw_range[1:] / np.maximum(raw_range[:-1], 1e-10)

    safe_atr = np.where(atr > 1e-10, atr, 1e-10)
    norm_range = raw_range / safe_atr

    # ✅ Vectorized rolling min/max
    range_percentile = np.full(n, np.nan, dtype=np.float32)
    if n >= range_window:
        windows = sliding_window_view(norm_range, window_shape=range_window)
        roll_min = np.min(windows, axis=1)
        roll_max = np.max(windows, axis=1)
        eps = 1e-10
        pct = (norm_range[range_window - 1:] - roll_min) / np.maximum(roll_max - roll_min, eps)
        range_percentile[range_window - 1:] = pct

    expansion = norm_range > expansion_threshold
    compression = norm_range < compression_threshold
    squeeze = norm_range < squeeze_threshold

    vol_regime = np.full(n, np.nan, dtype=np.float32)
    if n >= volatility_regime_window:
        kernel = np.ones(volatility_regime_window) / volatility_regime_window
        vol_regime[volatility_regime_window - 1:] = np.convolve(norm_range, kernel, mode='valid')

    return {
        'raw_range': raw_range,
        'range_ratio': range_ratio,
        'normalized_range': norm_range,
        'range_percentile': range_percentile,
        'range_expansion': expansion,
        'range_compression': compression,
        'range_squeeze': squeeze,
        'volatility_regime': vol_regime,
    }