"""
Fractal efficiency and directional consistency metrics.
Measures how "efficiently" price moves over a window.
"""

from typing import Dict
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .types import Prices, FloatArray


def compute_fractal_efficiency(close: Prices, period: int = 10) -> FloatArray:
    """
    Compute Kaufman’s Fractal Efficiency Ratio.

    Efficiency = |Net Displacement| / Total Path Length.
    Ranges from 0 (zigzag) to 1 (perfect straight line).

    Args:
        close: 1D array of closing prices (numeric).
        period: Lookback window for efficiency. Default is 10.

    Returns:
        FloatArray: Efficiency values in [0, 1], NaN where insufficient data.

    Raises:
        ValueError: If `period < 1` or input is not 1D.
        TypeError: If input is not numeric.

    Notes:
        - Handles NaN/inf by setting efficiency = 0.0.
        - 0/0 case (no movement) is treated as perfect efficiency (=1.0).
        - Output is float32.
    """
    # === PRECONDITIONS ===
    if close.ndim != 1:
        raise ValueError("close must be 1-dimensional")
    if not np.issubdtype(close.dtype, np.number):
        raise TypeError("close must be numeric")
    if period < 1:
        raise ValueError("period must be ≥ 1")

    n = len(close)
    if n < period:
        return np.full(n, np.nan, dtype=np.float32)

    close = close.astype(np.float32, copy=False)
    eff = np.full(n, np.nan, dtype=np.float32)

    # === TRIVIAL CASE: PERIOD = 1 ===
    if period == 1:
        eff[1:] = 1.0
        return eff

    # === NET DISPLACEMENT (|P_t - P_{t-period+1}|) ===
    net = np.abs(close[period - 1:] - close[:-(period - 1)])

    # === TOTAL PATH LENGTH (Σ|ΔP| over window) ===
    diffs = np.abs(np.diff(close))  # shape: (n - 1,)
    if period == 2:
        path = diffs
    else:
        windows = sliding_window_view(diffs, window_shape=period - 1)
        path = np.sum(windows, axis=1)  # shape: (n - period + 1,)

    # === EFFICIENCY WITH NUMERICAL SAFEGUARDS ===
    eps = np.finfo(np.float32).eps
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        efficiency = net / np.maximum(path, eps)

    # === SPECIAL CASE HANDLING ===
    # Unreliable (inf/NaN) → 0.0
    efficiency = np.where(np.isinf(efficiency) | np.isnan(efficiency), 0.0, efficiency)
    # No move at all → perfect efficiency
    efficiency = np.where((net == 0) & (path == 0), 1.0, efficiency)

    # === FINAL OUTPUT ===
    eff[period - 1:] = np.clip(efficiency, 0.0, 1.0)
    return eff


def compute_consistency_score(close: Prices, window: int = 20) -> FloatArray:
    """
    Compute Directional Consistency Score.

    Consistency = |Net Move| / Gross Move = |ΣΔP| / Σ|ΔP|.
    Equivalent to efficiency but emphasizes directional alignment.

    Args:
        close: 1D array of closing prices (numeric).
        window: Lookback window. Default is 20.

    Returns:
        FloatArray: Consistency scores in [0, 1], NaN if insufficient data.

    Raises:
        ValueError: If `window < 1` or input is not 1D.
        TypeError: If input is not numeric.

    Notes:
        - Identical mathematical form to fractal efficiency.
        - Used to detect persistent directional bias.
    """
    # === PRECONDITIONS ===
    if close.ndim != 1:
        raise ValueError("close must be 1-dimensional")
    if not np.issubdtype(close.dtype, np.number):
        raise TypeError("close must be numeric")
    if window < 1:
        raise ValueError("window must be ≥ 1")

    n = len(close)
    if n < window:
        return np.full(n, np.nan, dtype=np.float32)

    close = close.astype(np.float32, copy=False)
    score = np.full(n, np.nan, dtype=np.float32)

    # === TRIVIAL CASE ===
    if window == 1:
        score[1:] = 1.0
        return score

    # === NET AND GROSS MOVES ===
    net = np.abs(close[window - 1:] - close[:-(window - 1)])
    diffs = np.abs(np.diff(close))
    if window == 2:
        gross = diffs
    else:
        windows = sliding_window_view(diffs, window_shape=window - 1)
        gross = np.sum(windows, axis=1)

    # === COMPUTE CONSISTENCY ===
    eps = np.finfo(np.float32).eps
    with np.errstate(divide='ignore', invalid='ignore'):
        consistency = net / np.maximum(gross, eps)
    consistency = np.where((net == 0) & (gross == 0), 1.0, consistency)

    score[window - 1:] = np.clip(consistency, 0.0, 1.0)
    return score


def compute_fractal_efficiency_extended(close: Prices) -> Dict[str, np.ndarray]:
    """
    Compute extended fractal efficiency and consistency metrics at standard windows.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'efficiency_10': Fractal efficiency with period=10
            - 'efficiency_20': Fractal efficiency with period=20
            - 'consistency_20': Directional consistency with window=20

    Notes:
        - All outputs are float32 arrays.
        - NaN-padded for warm-up periods.
    """
    return {
        'efficiency_10': compute_fractal_efficiency(close, 10),
        'efficiency_20': compute_fractal_efficiency(close, 20),
        'consistency_20': compute_consistency_score(close, 20),
    }