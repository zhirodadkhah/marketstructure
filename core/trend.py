# core/trend.py
"""
Trend state detection from market structure labels.
Uses only close prices for invalidation.
"""
from typing import Dict
import numpy as np
from structure.metrics.types import Prices, StructureLabel, TrendStateArray


def detect_trend_state(
    close: Prices,
    is_higher_high: StructureLabel,
    is_lower_high: StructureLabel,
    is_higher_low: StructureLabel,
    is_lower_low: StructureLabel,
    invalidation_buffer: float = 0.0,
    include_metrics: bool = False
) -> Dict[str, np.ndarray]:
    """
    Detect dynamic trend state from structure events.

    Parameters
    ----------
    close : Prices
        Close prices for invalidation.
    is_* : StructureLabel
        Boolean masks from `detect_market_structure`.
    invalidation_buffer : float
        Fractional buffer for trend invalidation.
    include_metrics : bool
        Whether to return strength/since_index.

    Returns
    -------
    dict with:
        - 'trend_state': TrendStateArray (int8: -1,0,1)
        - optional 'trend_strength', 'trend_since_index'

    Pre-conditions
    --------------
    - All inputs same length.
    - invalidation_buffer ≥ 0.

    Post-conditions
    ---------------
    - Trend state is neutral until confirmed by structure sequence.
    - State is invalidated when close breaks reference level.

    Raises
    ------
    ValueError if invalid buffer or shape mismatch.
    """
    # === PRECONDITIONS ===
    if invalidation_buffer < 0:
        raise ValueError("Precondition violated: invalidation_buffer must be ≥ 0")

    n = len(close)
    if not all(arr.shape == (n,) for arr in [
        is_higher_high, is_lower_high, is_higher_low, is_lower_low
    ]):
        raise ValueError("Precondition violated: all inputs must have same length")

    # === INITIALIZE STATE ===
    trend_state = np.zeros(n, dtype=np.int8)  # 0 = neutral
    strength = np.zeros(n, dtype=np.float32) if include_metrics else None
    since_index = np.full(n, -1, dtype=np.int32) if include_metrics else None

    current_state = 0  # 0=neutral, 1=uptrend, -1=downtrend
    current_ref = np.nan
    current_strength = 0.0
    current_since = -1

    last_hl_idx = -1  # index of last Higher Low
    last_lh_idx = -1  # index of last Lower High

    # === MAIN LOOP: CONFIRM & INVALIDATE TRENDS ===
    for i in range(n):
        # === TREND CONFIRMATION ===
        if is_higher_high[i] and last_hl_idx != -1:
            current_state = 1
            current_ref = close[last_hl_idx]
            current_since = i
            if include_metrics:
                base = abs(current_ref) or 1.0
                current_strength = min(abs(close[i] - current_ref) / base * 100.0, 10.0)

        elif is_lower_low[i] and last_lh_idx != -1:
            current_state = -1
            current_ref = close[last_lh_idx]
            current_since = i
            if include_metrics:
                base = abs(current_ref) or 1.0
                current_strength = min(abs(close[i] - current_ref) / base * 100.0, 10.0)

        # === TRACK LAST HL/LH ===
        if is_higher_low[i]:
            last_hl_idx = i
        elif is_lower_high[i]:
            last_lh_idx = i

        # === TREND INVALIDATION ===
        if current_state != 0 and not np.isnan(current_ref):
            buffer_abs = abs(current_ref) * invalidation_buffer if current_ref != 0 else 0.0
            if current_state == 1 and close[i] < (current_ref - buffer_abs):
                current_state = 0
                current_ref = np.nan
                current_strength = 0.0
                current_since = -1
            elif current_state == -1 and close[i] > (current_ref + buffer_abs):
                current_state = 0
                current_ref = np.nan
                current_strength = 0.0
                current_since = -1

        trend_state[i] = current_state
        if include_metrics:
            strength[i] = current_strength
            since_index[i] = current_since

    # === POSTCONDITIONS ===
    assert trend_state.shape == (n,), "Postcondition failed: trend_state shape mismatch"
    if include_metrics:
        assert strength.shape == (n,), "Postcondition failed: strength shape mismatch"
        assert since_index.shape == (n,), "Postcondition failed: since_index shape mismatch"

    result = {'trend_state': trend_state}
    if include_metrics:
        result['trend_strength'] = strength
        result['trend_since_index'] = since_index
    return result