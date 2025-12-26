# core/structure.py
"""
Market structure labeling (HH, HL, LH, LL) from swing points.
Assumes alternating swings (enforced by detect_swing_points).
"""
from typing import Dict
import numpy as np
from structure.metrics.types import Prices, SwingsMask, StructureLabel


def detect_market_structure(
    high: Prices,
    low: Prices,
    is_swing_high: SwingsMask,
    is_swing_low: SwingsMask
) -> Dict[str, StructureLabel]:
    """
    Label higher highs, lower highs, higher lows, lower lows.

    Parameters
    ----------
    high, low : Prices
        Price arrays.
    is_swing_high, is_swing_low : SwingsMask
        Boolean masks from `detect_swing_points`.

    Returns
    -------
    dict of StructureLabel (bool arrays)

    Pre-conditions
    --------------
    - Swings must be alternating (no consecutive same type).
    - Masks must be mutually exclusive.
    - All inputs same length.

    Post-conditions
    ---------------
    - Exactly one structure label may be True per swing.
    - Non-swing bars are all False.

    Raises
    ------
    ValueError if inputs invalid.
    """
    # === PRECONDITIONS ===
    n = len(high)
    if not all(arr.shape == (n,) for arr in (high, low, is_swing_high, is_swing_low)):
        raise ValueError("Precondition violated: all inputs must be same-length 1D arrays")
    if not all(arr.dtype == bool for arr in (is_swing_high, is_swing_low)):
        raise ValueError("Precondition violated: swing masks must be boolean")
    if np.any(is_swing_high & is_swing_low):
        raise ValueError("Precondition violated: swing masks must be mutually exclusive")

    # === INITIALIZE OUTPUT ===
    is_higher_high = np.zeros(n, dtype=bool)
    is_lower_high = np.zeros(n, dtype=bool)
    is_higher_low = np.zeros(n, dtype=bool)
    is_lower_low = np.zeros(n, dtype=bool)

    last_high = np.nan
    last_low = np.nan

    # === LABEL SWINGS IN ORDER ===
    for i in range(n):
        if is_swing_high[i]:
            price = high[i]
            if not np.isnan(last_high):
                if price > last_high:
                    is_higher_high[i] = True
                elif price < last_high:
                    is_lower_high[i] = True
            last_high = price

        if is_swing_low[i]:
            price = low[i]
            if not np.isnan(last_low):
                if price > last_low:
                    is_higher_low[i] = True
                elif price < last_low:
                    is_lower_low[i] = True
            last_low = price

    # === POSTCONDITIONS ===
    n_swings = np.sum(is_swing_high | is_swing_low)
    n_labels = (
        np.sum(is_higher_high) + np.sum(is_lower_high) +
        np.sum(is_higher_low) + np.sum(is_lower_low)
    )
    assert n_labels <= n_swings, "Postcondition failed: more labels than swings"

    return {
        'is_higher_high': is_higher_high,
        'is_lower_high': is_lower_high,
        'is_higher_low': is_higher_low,
        'is_lower_low': is_lower_low
    }