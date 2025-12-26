import numpy as np
from .types import Prices, ATRArray, FloatArray


def compute_true_range(high: Prices, low: Prices, close: Prices) -> FloatArray:
    """
    Compute the True Range (TR) for each price bar.

    True Range is the greatest of:
      - Current high minus current low
      - Absolute value of current high minus previous close
      - Absolute value of current low minus previous close

    Args:
        high: 1D array of high prices.
        low: 1D array of low prices.
        close: 1D array of closing prices.

    Returns:
        FloatArray: True range values as float32 (same length as inputs).

    Raises:
        TypeError: If inputs are not convertible to numeric numpy arrays.
        ValueError: If inputs are not 1D or have mismatched shapes.

    Notes:
        - First value uses only `high[0] - low[0]` (no prior close).
        - Output dtype is always float32.
    """
    # === INPUT VALIDATION ===
    try:
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
    except (ValueError, TypeError) as e:
        raise TypeError("Inputs must be convertible to numpy arrays") from e

    if not all(np.issubdtype(arr.dtype, np.number) for arr in (high, low, close)):
        raise TypeError("All input arrays must have numeric dtypes.")
    if not (high.shape == low.shape == close.shape):
        raise ValueError("Input arrays must have the same shape.")
    if high.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")

    # === EDGE CASE: EMPTY INPUT ===
    n = len(close)
    if n == 0:
        return np.array([], dtype=np.float32)

    # === TRUE RANGE COMPUTATION ===
    tr = np.empty(n, dtype=np.float32)
    tr[0] = high[0] - low[0]
    if n > 1:
        prev_close = close[:-1]
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - prev_close),
                np.abs(low[1:] - prev_close)
            )
        )
    return tr


def compute_atr(high: Prices, low: Prices, close: Prices, period: int = 14) -> ATRArray:
    """
    Compute Wilder's Average True Range (ATR) using EMA-like smoothing.

    Args:
        high: 1D array of high prices.
        low: 1D array of low prices.
        close: 1D array of closing prices.
        period: Smoothing window (≥1). Default is 14.

    Returns:
        ATRArray: ATR values as float32, with first `period - 1` entries as NaN.

    Raises:
        ValueError: If `period < 1`.

    Notes:
        - Uses simple moving average for initial value.
        - Smoothing follows Wilder’s formula:
          ATR[i] = (TR[i] + (period - 1) * ATR[i-1]) / period
        - Returns NaN-padded output for warm-up period.
    """
    if period < 1:
        raise ValueError("period must be ≥1")

    tr = compute_true_range(high, low, close)
    n = len(tr)
    atr = np.full(n, np.nan, dtype=np.float32)

    # === INSUFFICIENT DATA ===
    if n < period:
        return atr

    # === INITIAL VALUE (SMA) ===
    atr[period - 1] = np.mean(tr[:period])

    # === RECURSIVE SMOOTHING ===
    for i in range(period, n):
        atr[i] = (tr[i] + (period - 1) * atr[i - 1]) / period

    return atr