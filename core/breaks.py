# core/breaks.py
"""
Pure signal detection for initial structure breaks.
Does NOT handle retests, confirmation, or failures — those require stateful logic.
"""
from typing import Dict
import numpy as np
from structure.metrics.types import Prices, SwingsMask, StructureLabel, TrendStateArray, SignalMask


def detect_structure_break_signals(
    high: Prices,
    low: Prices,
    close: Prices,
    open_: Prices,
    atr: Prices,
    is_swing_high: SwingsMask,
    is_swing_low: SwingsMask,
    is_higher_high: StructureLabel,
    is_lower_low: StructureLabel,
    trend_state: TrendStateArray,
    min_break_atr_mult: float = 0.5,
    buffer_multiplier: float = 0.5
) -> Dict[str, SignalMask]:
    """
    Detect initial break signals (BOS/CHOCH) using pure array logic.

    Parameters
    ----------
    high, low, close, open_, atr : Prices
        Price and ATR arrays.
    is_* : masks from core modules.
    trend_state : TrendStateArray
        From `detect_trend_state`.
    min_break_atr_mult : float
        Minimum move = atr * this value.
    buffer_multiplier : float
        Unused in this function (kept for API consistency).

    Returns
    -------
    dict of SignalMask for initial break types.

    Pre-conditions
    --------------
    - All inputs same length.
    - trend_state ∈ {-1, 0, 1}

    Post-conditions
    ---------------
    - All output masks are boolean, same length as input.
    - Signals only fire when trend and price action align.

    Notes
    -----
    - This function only emits **initial break signals**.
    - Full signal lifecycle (confirmation, failure) requires stateful tracking
      in `detector.py`.
    - Assumes `trend_state`: 1=uptrend, -1=downtrend, 0=neutral.
    """
    # === PRECONDITIONS ===
    n = len(close)
    inputs = [high, low, close, open_, atr, trend_state]
    inputs += [is_swing_high, is_swing_low, is_higher_high, is_lower_low]
    if not all(arr.shape == (n,) for arr in inputs):
        raise ValueError("Precondition violated: all inputs must be same-length 1D arrays")
    if not np.all(np.isin(trend_state, [-1, 0, 1])):
        raise ValueError("Precondition violated: trend_state must be in {-1,0,1}")

    # === SIGNAL THRESHOLDS ===
    min_move = atr * min_break_atr_mult

    # === FORWARD-FILL LAST HH/LL ===
    last_hh_price = np.full(n, np.nan, dtype=np.float32)
    last_ll_price = np.full(n, np.nan, dtype=np.float32)

    hh_price = np.where(is_higher_high, high, np.nan)
    ll_price = np.where(is_lower_low, low, np.nan)

    last_valid = np.nan
    for i in range(n):
        if not np.isnan(hh_price[i]):
            last_valid = hh_price[i]
        last_hh_price[i] = last_valid

    last_valid = np.nan
    for i in range(n):
        if not np.isnan(ll_price[i]):
            last_valid = ll_price[i]
        last_ll_price[i] = last_valid

    # === SIGNAL LOGIC ===
    is_bos_bullish_initial = (
        (trend_state == 1) &
        (close > last_hh_price + min_move) &
        ~np.isnan(last_hh_price)
    )
    is_bos_bearish_initial = (
        (trend_state == -1) &
        (close < last_ll_price - min_move) &
        ~np.isnan(last_ll_price)
    )
    is_choch_bearish = (
        (trend_state == 1) &
        (close < last_hh_price - min_move) &
        ~np.isnan(last_hh_price)
    )
    is_choch_bullish = (
        (trend_state == -1) &
        (close > last_ll_price + min_move) &
        ~np.isnan(last_ll_price)
    )

    # === POSTCONDITIONS ===
    signals = {
        'is_bos_bullish_initial': is_bos_bullish_initial,
        'is_bos_bearish_initial': is_bos_bearish_initial,
        'is_choch_bullish': is_choch_bullish,
        'is_choch_bearish': is_choch_bearish,
    }
    for mask in signals.values():
        assert mask.shape == (n,), "Postcondition failed: signal shape mismatch"
        assert mask.dtype == bool, "Postcondition failed: signal not boolean"

    return signals