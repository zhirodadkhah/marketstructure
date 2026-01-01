# structure/signal/filter.py
"""
Context-aware signal filtering based on regime, range, and retest quality.
Pure vectorized logic with clear, testable conditions.
"""
from typing import Dict, Optional
import numpy as np
from structure.metrics.types import ValidatedSignals
from .config import SignalFilterConfig


def filter_signals(
        validated_signals: ValidatedSignals,
        market_regime: np.ndarray,
        is_range_compression: np.ndarray,
        retest_respect_score: Optional[np.ndarray],
        config: SignalFilterConfig
) -> ValidatedSignals:
    """
    Filter signals using market context and retest quality.

    Parameters
    ----------
    validated_signals : ValidatedSignals
        Input signals to filter.
    market_regime : np.ndarray[str]
        Market regime labels.
    is_range_compression : np.ndarray[bool]
        True if bar is in range compression.
    retest_respect_score : np.ndarray[float32], optional
        Retest respect score [0,1]. If None, skips retest filter.
    config : SignalFilterConfig
        Filtering thresholds and flags.

    Returns
    -------
    ValidatedSignals
        Filtered signal masks.

    Filtering Logic
    ---------------
    A signal is **rejected** if ANY of:
    1. `market_regime == 'chop'` AND signal is BOS
    2. `is_range_compression == True` (if `avoid_range_compression`)
    3. `retest_respect_score < min_retest_respect_score` (if provided)

    Notes
    -----
    - All logic is vectorized (no loops)
    - CHOCH signals are **not filtered** by regime (only BOS)
    - Time Complexity: O(n)
    """
    n = len(market_regime)

    # === 1. Regime filter: reject BOS in chop ===
    is_chop = (market_regime == 'chop')
    chop_reject = is_chop  # Applies to BOS only

    # === 2. Range compression filter ===
    compression_reject = is_range_compression if config.avoid_range_compression else np.zeros(n, dtype=bool)

    # === 3. Retest respect filter ===
    retest_reject = np.zeros(n, dtype=bool)
    if retest_respect_score is not None and config.min_retest_respect_score is not None:
        retest_reject = retest_respect_score < config.min_retest_respect_score

    # === COMBINE REJECTION MASKS ===
    global_reject = chop_reject | compression_reject | retest_reject

    # === APPLY TO BOS SIGNALS ONLY (not CHOCH or failures) ===
    # Only confirmed BOS signals are filtered by regime/compression
    bos_signals = [
        'is_bos_bullish_confirmed',
        'is_bos_bearish_confirmed',
        'is_bos_bullish_momentum',
        'is_bos_bearish_momentum'
    ]

    filtered_dict = {}
    for field in validated_signals.__dataclass_fields__:
        original_mask = getattr(validated_signals, field)
        if field in bos_signals:
            filtered_mask = original_mask & (~global_reject)
        else:
            # Keep failures, CHOCH, etc. unchanged
            filtered_mask = original_mask
        filtered_dict[field] = filtered_mask

    return ValidatedSignals(**filtered_dict)