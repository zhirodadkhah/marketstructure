# core/test_swings.py
"""
Pure NumPy swing detection with enforced alternation.
Implements vectorized extremum detection and conflict resolution.
"""
from typing import Dict
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from structure.metrics.types import Prices, SwingsMask


def detect_swing_points(
        high: Prices,
        low: Prices,
        half_window: int = 2
) -> Dict[str, SwingsMask]:
    """
    Detect alternating swing highs and lows from OHLC price arrays.

    Parameters
    ----------
    high : Prices
        1D array of high prices (float32).
    low : Prices
        1D array of low prices (float32).
    half_window : int, default=2
        Number of bars to left/right for validation.

    Returns
    -------
    dict with keys:
        - 'is_swing_high': bool array
        - 'is_swing_low': bool array

    Pre-conditions
    --------------
    - `high` and `low` must be 1D, same length, numeric.
    - `half_window` ≥ 1.

    Post-conditions
    ---------------
    - No index is both a swing high and swing low.
    - Swings strictly alternate (high → low → high...).
    - Edge bars (first/last `half_window`) are never swings.

    Raises
    ------
    ValueError
        If inputs violate shape or half_window constraints.
    TypeError
        If inputs are non-numeric.

    Notes
    -----
    - Uses SciPy's `maximum_filter1d` for O(n) extremum detection.
    - Plateaus are resolved by extremity score relative to neighbors.
    - Final alternation is enforced via linear sweep (O(k), k = swing count).
    """
    # === PRECONDITIONS: VALIDATE INPUTS ===
    if half_window < 1:
        raise ValueError("Precondition violated: half_window must be ≥ 1")
    if high.shape != low.shape or high.ndim != 1:
        raise ValueError("Precondition violated: inputs must be 1D arrays of same length")
    if not (np.issubdtype(high.dtype, np.number) and np.issubdtype(low.dtype, np.number)):
        raise TypeError("Precondition violated: inputs must be numeric")

    n = len(high)
    if n < 2 * half_window + 1:
        empty = np.zeros(n, dtype=bool)
        return {'is_swing_high': empty, 'is_swing_low': empty}

    # === CONVERT TO FLOAT32 ===
    high = high.astype(np.float32, copy=False)
    low = low.astype(np.float32, copy=False)

    # === SWING CANDIDATE DETECTION (VECTORIZED) ===
    window = 2 * half_window + 1
    high_max = maximum_filter1d(high, size=window, mode='constant')
    low_min = minimum_filter1d(low, size=window, mode='constant')

    sh = (high == high_max)
    sl = (low == low_min)

    # Invalidate edges
    sh[:half_window] = False
    sh[-half_window:] = False
    sl[:half_window] = False
    sl[-half_window:] = False

    # === ENFORCE STRICT EXTREMUM (NO PLATEAUS) ===
    sh &= (high > np.concatenate([[high[0] - 1], high[:-1]]))  # left neighbor
    sh &= (high > np.concatenate([high[1:], [high[-1] - 1]]))  # right neighbor
    sl &= (low < np.concatenate([[low[0] + 1], low[:-1]]))
    sl &= (low < np.concatenate([low[1:], [low[-1] + 1]]))

    # === RESOLVE CONFLICTS (BOTH HIGH & LOW AT SAME BAR) ===
    conflict = sh & sl
    if np.any(conflict):
        # Compute extremity relative to immediate neighbors
        left_high = np.concatenate([[high[0]], high[:-1]])
        right_high = np.concatenate([high[1:], [high[-1]]])
        high_ext = high - np.minimum(left_high, right_high)

        left_low = np.concatenate([[low[0]], low[:-1]])
        right_low = np.concatenate([low[1:], [low[-1]]])
        low_ext = np.maximum(left_low, right_low) - low

        prefer_high = high_ext > low_ext
        sh = sh.copy()
        sl = sl.copy()
        sh[conflict] = prefer_high[conflict]
        sl[conflict] = ~prefer_high[conflict]

    # === ENFORCE ALTERNATION (LINEAR SWEEP) ===
    swing_idx = np.where(sh | sl)[0]
    if len(swing_idx) == 0:
        return {'is_swing_high': sh, 'is_swing_low': sl}

    swing_types = np.where(sh[swing_idx], 1, -1)  # 1=high, -1=low
    swing_prices = np.where(sh[swing_idx], high[swing_idx], low[swing_idx])

    final_high = np.zeros(n, dtype=bool)
    final_low = np.zeros(n, dtype=bool)

    last_type = swing_types[0]
    last_price = swing_prices[0]
    last_idx = swing_idx[0]

    if last_type == 1:
        final_high[last_idx] = True
    else:
        final_low[last_idx] = True

    for i in range(1, len(swing_idx)):
        curr_type = swing_types[i]
        curr_price = swing_prices[i]
        curr_idx = swing_idx[i]

        if curr_type == last_type:
            # Same type: keep more extreme
            if (curr_type == 1 and curr_price > last_price) or \
                    (curr_type == -1 and curr_price < last_price):
                if last_type == 1:
                    final_high[last_idx] = False
                else:
                    final_low[last_idx] = False
                if curr_type == 1:
                    final_high[curr_idx] = True
                else:
                    final_low[curr_idx] = True
                last_price = curr_price
                last_idx = curr_idx
        else:
            # Alternate: keep both
            if curr_type == 1:
                final_high[curr_idx] = True
            else:
                final_low[curr_idx] = True
            last_type = curr_type
            last_price = curr_price
            last_idx = curr_idx

    # === POSTCONDITIONS: VALIDATE OUTPUT ===
    assert final_high.shape == (n,), "Postcondition failed: output shape mismatch"
    assert final_low.shape == (n,), "Postcondition failed: output shape mismatch"
    assert not np.any(final_high & final_low), "Postcondition failed: non-alternating swings"

    return {
        'is_swing_high': final_high,
        'is_swing_low': final_low
    }