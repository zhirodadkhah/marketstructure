# structure/context/mtf.py
"""
Multi-timeframe (HTF) resampling without recursion or DataFrame.
"""
from typing import Dict
import numpy as np
from structure.metrics.types import Prices, Timestamps
from .config import MTFConfig


def resample_to_htf(
    timestamps: Timestamps,
    open_: Prices,
    high: Prices,
    low: Prices,
    close: Prices,
    config: MTFConfig
) -> Dict[str, np.ndarray]:
    """
    Resample OHLC to higher timeframe (e.g., 1H → 1D).

    Parameters
    ----------
    timestamps : Timestamps (datetime64[s])
    open_, high, low, close : Prices
    config : MTFConfig

    Returns
    -------
    dict with HTF arrays:
        - 'htf_timestamps'
        - 'htf_open', 'htf_high', 'htf_low', 'htf_close'

    Pre-conditions
    --------------
    - All arrays same length.
    - timestamps sorted ascending.

    Notes
    -----
    - Uses simple bar grouping by index (not time alignment).
    - For true time-based resampling, use pd.DataFrame externally.
    """
    # === PRECONDITIONS ===
    n = len(close)
    if not all(arr.shape == (n,) for arr in (open_, high, low, close)):
        raise ValueError("All price arrays same length")
    if timestamps.shape != (n,):
        raise ValueError("Timestamps same length as prices")
    if config.htf_bar_size < 1:
        raise ValueError("htf_bar_size must be ≥ 1")

    # === GROUP BARS ===
    num_bars = (n // config.htf_bar_size) * config.htf_bar_size
    if num_bars == 0:
        return {
            'htf_timestamps': np.array([], dtype=timestamps.dtype),
            'htf_open': np.array([], dtype=np.float32),
            'htf_high': np.array([], dtype=np.float32),
            'htf_low': np.array([], dtype=np.float32),
            'htf_close': np.array([], dtype=np.float32)
        }

    trimmed = slice(0, num_bars)
    o = open_[trimmed].reshape(-1, config.htf_bar_size)
    h = high[trimmed].reshape(-1, config.htf_bar_size)
    l = low[trimmed].reshape(-1, config.htf_bar_size)
    c = close[trimmed].reshape(-1, config.htf_bar_size)
    t = timestamps[trimmed].reshape(-1, config.htf_bar_size)

    # === AGGREGATE ===
    htf_open = o[:, 0]
    htf_high = np.max(h, axis=1)
    htf_low = np.min(l, axis=1)
    htf_close = c[:, -1]
    htf_timestamps = t[:, -1]  # or t[:, 0] for "backward"

    return {
        'htf_timestamps': htf_timestamps,
        'htf_open': htf_open.astype(np.float32),
        'htf_high': htf_high.astype(np.float32),
        'htf_low': htf_low.astype(np.float32),
        'htf_close': htf_close.astype(np.float32)
    }