# structure/context/zones.py
"""
Support/Resistance zone detection via swing clustering.
"""
from typing import Tuple, Optional
import numpy as np
from structure.metrics.atr import compute_atr
from structure.core.swings import detect_swing_points
from structure.metrics.types import Prices, ZoneArray
from .config import ZoneConfig


def detect_sr_zones(
    high: Prices,
    low: Prices,
    close: Prices,
    config: ZoneConfig
) -> Tuple[ZoneArray, ZoneArray]:
    """
    Detect support and resistance zones via swing clustering.

    Parameters
    ----------
    high, low, close : Prices
    config : ZoneConfig

    Returns
    -------
    (support_levels, resistance_levels) : Tuple[ZoneArray, ZoneArray]
        Sorted arrays of S/R levels (float32).

    Pre-conditions
    --------------
    - Inputs are valid price arrays.

    Post-conditions
    ---------------
    - Levels are sorted ascending.
    - No duplicate levels within radius.

    Notes
    -----
    - Uses ATR to normalize clustering radius.
    - Only considers recent `recent_bars` swings.
    """
    # === PRECONDITIONS ===
    n = len(close)
    if not all(arr.shape == (n,) for arr in (high, low, close)):
        raise ValueError("All inputs same length")

    if n < 5:
        return (np.array([], dtype=np.float32), np.array([], dtype=np.float32))

    # === GET SWINGS ===
    swings = detect_swing_points(high, low, half_window=2)
    is_sh = swings['is_swing_high']
    is_sl = swings['is_swing_low']

    # === USE RECENT BARS ONLY ===
    start_idx = max(0, n - config.recent_bars)
    sh_prices = high[start_idx:][is_sh[start_idx:]]
    sl_prices = low[start_idx:][is_sl[start_idx:]]

    if len(sh_prices) == 0 and len(sl_prices) == 0:
        return (np.array([], dtype=np.float32), np.array([], dtype=np.float32))

    # === COMPUTE ATR FOR NORMALIZATION ===
    atr = compute_atr(high, low, close, period=14)
    recent_atr = np.nanmean(atr[-20:]) if np.any(~np.isnan(atr[-20:])) else 1.0
    radius = config.clustering_radius * recent_atr

    # === CLUSTER RESISTANCE (SWING HIGHS) ===
    res_levels = _cluster_levels(sh_prices, radius, config.min_cluster_size)
    # === CLUSTER SUPPORT (SWING LOWS) ===
    sup_levels = _cluster_levels(sl_prices, radius, config.min_cluster_size)

    return (sup_levels, res_levels)


def _cluster_levels(
    prices: np.ndarray,
    radius: float,
    min_cluster_size: int
) -> ZoneArray:
    """Simple density-based clustering."""
    if len(prices) == 0:
        return np.array([], dtype=np.float32)

    # Sort prices
    sorted_prices = np.sort(prices)
    levels = []

    i = 0
    while i < len(sorted_prices):
        # Find all points within radius
        cluster = [sorted_prices[i]]
        j = i + 1
        while j < len(sorted_prices) and (sorted_prices[j] - sorted_prices[i]) <= radius:
            cluster.append(sorted_prices[j])
            j += 1

        # Accept if large enough
        if len(cluster) >= min_cluster_size:
            levels.append(np.mean(cluster))  # or median

        i = j

    return np.array(levels, dtype=np.float32)