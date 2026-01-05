# structure/signal/filter.py
"""
Signal filtering based on market context: regime, session, range dynamics.
Removes low-probability signals using price-action principles.

FILTERING LOGIC:
- Reject BOS in chop/ranging markets (no trend = no BOS validity)
- Reject signals during range compression (low volatility = poor follow-through)
- Reject signals that don't respect retest zones (poor retest quality)
- Accept signals in trending markets with proper context

DESIGN PRINCIPLES:
- Defensive programming: All inputs validated
- Performance: Vectorized operations, O(n) time complexity
- Configurable: All thresholds in SignalFilterConfig
- Clear semantics: Explicit filtering rules with documentation
"""

from __future__ import annotations
from typing import Set
import numpy as np

from structure.metrics.types import ValidatedSignals, FilteredSignals, SignalQuality
from .config import SignalFilterConfig

# ==============================================================================
# SECTION: Constants
# ==============================================================================

_VALID_REGIMES: Set[str] = {'strong_trend', 'weak_trend', 'ranging', 'chop', 'neutral'}
_FILTERING_REGIMES: Set[str] = {'chop', 'ranging', 'neutral'}  # Regimes where BOS should be filtered


# ==============================================================================
# SECTION: Input Validation
# ==============================================================================

def _validate_inputs(
        market_regime: np.ndarray,
        range_compression: np.ndarray,
        retest_respect_score: np.ndarray,
        config: SignalFilterConfig
) -> None:
    """
    Validate all inputs meet expected constraints.

    Raises
    ------
    ValueError
        If inputs violate length, range, or regime constraints.
    TypeError
        If array dtypes are incorrect.
    """
    n = len(market_regime)

    # Array length validation
    if not all(len(arr) == n for arr in [range_compression, retest_respect_score]):
        raise ValueError(
            f"Array length mismatch: market_regime={n}, "
            f"range_compression={len(range_compression)}, "
            f"retest_respect_score={len(retest_respect_score)}"
        )

    # Dtype validation
    if not np.issubdtype(range_compression.dtype, np.floating):
        raise TypeError(f"range_compression must be float array, got {range_compression.dtype}")
    if not np.issubdtype(retest_respect_score.dtype, np.floating):
        raise TypeError(f"retest_respect_score must be float array, got {retest_respect_score.dtype}")

    # Regime validation
    if market_regime.size > 0:
        non_empty_mask = market_regime != ''
        if np.any(non_empty_mask):
            non_empty_regimes = market_regime[non_empty_mask]
            unique_regimes = set(np.unique(non_empty_regimes))
            invalid_regimes = unique_regimes - _VALID_REGIMES
            if invalid_regimes:
                raise ValueError(
                    f"Invalid market_regime values: {invalid_regimes}. "
                    f"Valid values: {_VALID_REGIMES}"
                )

    # Score range validation
    for name, arr in [("range_compression", range_compression),
                      ("retest_respect_score", retest_respect_score)]:
        if arr.size > 0:
            if np.any((arr < 0) | (arr > 1)):
                raise ValueError(f"{name} must be in range [0, 1]")


# ==============================================================================
# SECTION: Core Filtering Logic
# ==============================================================================

def _filter_bos_by_regime(
        signal_mask: np.ndarray,
        market_regime: np.ndarray,
        allow_weak_trend: bool = True
) -> np.ndarray:
    """
    Filter BOS signals by market regime.

    Precondition:
    - `signal_mask` and `market_regime` same length
    - `market_regime` contains only valid regime values

    Postcondition:
    - Returns mask with signals removed in chop/ranging/neutral regimes
    - Strong trend always allowed, weak trend conditionally allowed

    Parameters
    ----------
    signal_mask : np.ndarray[bool]
        Original signal mask
    market_regime : np.ndarray[str]
        Market regime per bar
    allow_weak_trend : bool
        Whether to allow signals in weak_trend regime

    Returns
    -------
    np.ndarray[bool]
        Filtered signal mask
    """
    if not np.any(signal_mask):
        return signal_mask.copy()

    # Always allow strong_trend
    allowed_regimes = ['strong_trend']
    if allow_weak_trend:
        allowed_regimes.append('weak_trend')

    regime_allowed = np.isin(market_regime, allowed_regimes)
    return signal_mask & regime_allowed


def _filter_by_range_compression(
        signal_mask: np.ndarray,
        range_compression: np.ndarray,
        max_compression_threshold: float
) -> np.ndarray:
    """
    Filter signals during range compression (low volatility periods).

    Precondition:
    - `range_compression` values in [0, 1] where 1.0 = max compression
    - `max_compression_threshold` in [0, 1]

    Postcondition:
    - Returns mask with signals removed where compression > threshold

    Parameters
    ----------
    signal_mask : np.ndarray[bool]
        Signal mask to filter
    range_compression : np.ndarray[float32]
        Range compression score [0, 1]
    max_compression_threshold : float
        Maximum allowed compression (signals filtered above this)

    Returns
    -------
    np.ndarray[bool]
        Filtered signal mask
    """
    if not np.any(signal_mask):
        return signal_mask.copy()

    compression_allowed = range_compression <= max_compression_threshold
    return signal_mask & compression_allowed


def _filter_by_retest_respect(
        signal_mask: np.ndarray,
        retest_respect_score: np.ndarray,
        min_retest_respect_threshold: float
) -> np.ndarray:
    """
    Filter signals by retest respect quality.

    Precondition:
    - `retest_respect_score` values in [0, 1]
    - `min_retest_respect_threshold` in [0, 1]

    Postcondition:
    - Returns mask with signals removed where retest respect < threshold

    Parameters
    ----------
    signal_mask : np.ndarray[bool]
        Signal mask to filter
    retest_respect_score : np.ndarray[float32]
        Retest respect score [0, 1]
    min_retest_respect_threshold : float
        Minimum required retest respect

    Returns
    -------
    np.ndarray[bool]
        Filtered signal mask
    """
    if not np.any(signal_mask):
        return signal_mask.copy()

    respect_sufficient = retest_respect_score >= min_retest_respect_threshold
    return signal_mask & respect_sufficient


# ==============================================================================
# SECTION: Signal-Specific Filtering
# ==============================================================================

def _apply_bos_filtering(
        signal_mask: np.ndarray,
        market_regime: np.ndarray,
        range_compression: np.ndarray,
        retest_respect_score: np.ndarray,
        config: SignalFilterConfig
) -> np.ndarray:
    """
    Apply comprehensive filtering to BOS signals.

    BOS signals require trending markets and good context.
    """
    filtered = signal_mask.copy()

    # Filter by regime (BOS needs trend)
    filtered = _filter_bos_by_regime(
        filtered, market_regime, allow_weak_trend=config.allow_weak_trend_bos
    )

    # Filter by range compression
    filtered = _filter_by_range_compression(
        filtered, range_compression, config.max_range_compression
    )

    # Filter by retest respect (for confirmed signals)
    filtered = _filter_by_retest_respect(
        filtered, retest_respect_score, config.min_retest_respect_filter
    )

    return filtered


def _apply_choch_filtering(
        signal_mask: np.ndarray,
        market_regime: np.ndarray,
        range_compression: np.ndarray,
        retest_respect_score: np.ndarray,
        config: SignalFilterConfig
) -> np.ndarray:
    """
    Apply comprehensive filtering to CHOCH signals.

    CHOCH signals can work in ranging markets but still need good context.
    """
    filtered = signal_mask.copy()

    # CHOCH can work in more regimes, but avoid extreme chop
    choch_allowed_regimes = ['strong_trend', 'weak_trend', 'ranging', 'neutral']
    regime_allowed = np.isin(market_regime, choch_allowed_regimes)
    filtered = filtered & regime_allowed

    # Still filter extreme range compression
    filtered = _filter_by_range_compression(
        filtered, range_compression, config.max_range_compression_choch
    )

    # CHOCH may have different retest respect requirements
    filtered = _filter_by_retest_respect(
        filtered, retest_respect_score, config.min_retest_respect_filter_choch
    )

    return filtered


def _apply_momentum_filtering(
        signal_mask: np.ndarray,
        market_regime: np.ndarray,
        range_compression: np.ndarray,
        config: SignalFilterConfig
) -> np.ndarray:
    """
    Apply filtering to momentum signals (no retest).

    Momentum signals need strong trending markets and low compression.
    """
    filtered = signal_mask.copy()

    # Momentum needs strong trend only (no weak trend)
    strong_trend_only = market_regime == 'strong_trend'
    filtered = filtered & strong_trend_only

    # Strict range compression filtering
    filtered = _filter_by_range_compression(
        filtered, range_compression, config.max_range_compression_momentum
    )

    # No retest respect filtering (momentum has no retest)
    return filtered


# ==============================================================================
# SECTION: Public Interface
# ==============================================================================

def filter_signals(
        validated_signals: ValidatedSignals,
        signal_quality: SignalQuality,
        market_regime: np.ndarray,
        range_compression: np.ndarray,
        retest_respect_score: np.ndarray,
        config: SignalFilterConfig
) -> FilteredSignals:
    """
    Filter validated signals based on market context and quality metrics.

    Parameters
    ----------
    validated_signals : ValidatedSignals
        Validated signal masks from signal validation step
    signal_quality : SignalQuality
        Quality scores for all signal types
    market_regime : np.ndarray[str]
        Market regime classification per bar
        Valid: {'strong_trend', 'weak_trend', 'ranging', 'chop', 'neutral'}
    range_compression : np.ndarray[float32]
        Normalized range compression score [0.0, 1.0] per bar
        0.0 = no compression, 1.0 = maximum compression
    retest_respect_score : np.ndarray[float32]
        Retest respect quality score [0.0, 1.0] per bar
    config : SignalFilterConfig
        Filtering thresholds and rules

    Returns
    -------
    FilteredSignals
        Filtered signal masks with low-probability signals removed

    Raises
    ------
    ValueError
        If inputs violate validation constraints
    TypeError
        If input dtypes are incorrect

    Notes
    -----
    - BOS signals filtered in chop/ranging markets (need trend)
    - CHOCH signals filtered only in extreme chop (can work in ranging)
    - Momentum signals require strong trend only
    - All signals filtered by range compression and retest respect
    - Time complexity: O(n), Memory: O(n)

    Examples
    --------
    >>> config = SignalFilterConfig(
    ...     max_range_compression=0.7,
    ...     min_retest_respect_filter=0.5,
    ...     allow_weak_trend_bos=True
    ... )
    >>> filtered = filter_signals(
    ...     validated_signals,
    ...     signal_quality,
    ...     market_regime,
    ...     range_compression,
    ...     retest_respect_score,
    ...     config
    ... )
    """
    # === INPUT VALIDATION ===
    _validate_inputs(market_regime, range_compression, retest_respect_score, config)

    # === BOS CONFIRMED SIGNALS ===
    bos_bullish_filtered = _apply_bos_filtering(
        validated_signals.is_bos_bullish_confirmed,
        market_regime,
        range_compression,
        retest_respect_score,
        config
    )
    bos_bearish_filtered = _apply_bos_filtering(
        validated_signals.is_bos_bearish_confirmed,
        market_regime,
        range_compression,
        retest_respect_score,
        config
    )

    # === BOS MOMENTUM SIGNALS ===
    bos_bullish_momentum_filtered = _apply_momentum_filtering(
        validated_signals.is_bos_bullish_momentum,
        market_regime,
        range_compression,
        config
    )
    bos_bearish_momentum_filtered = _apply_momentum_filtering(
        validated_signals.is_bos_bearish_momentum,
        market_regime,
        range_compression,
        config
    )

    # === CHOCH SIGNALS ===
    # Use raw CHOCH signals from quality scoring (non-failed CHOCH)
    choch_bullish_filtered = _apply_choch_filtering(
        signal_quality.choch_bullish_quality > 0,  # Non-zero quality = valid CHOCH
        market_regime,
        range_compression,
        retest_respect_score,
        config
    )
    choch_bearish_filtered = _apply_choch_filtering(
        signal_quality.choch_bearish_quality > 0,
        market_regime,
        range_compression,
        retest_respect_score,
        config
    )

    # === FAILED SIGNALS (typically not filtered, but pass through) ===
    # Failed signals are usually not traded, so we don't filter them further
    failed_bullish_filtered = validated_signals.is_bullish_break_failure.copy()
    failed_bearish_filtered = validated_signals.is_bearish_break_failure.copy()
    failed_choch_bullish_filtered = validated_signals.is_failed_choch_bullish.copy()
    failed_choch_bearish_filtered = validated_signals.is_failed_choch_bearish.copy()

    return FilteredSignals(
        is_bos_bullish_confirmed=bos_bullish_filtered,
        is_bos_bearish_confirmed=bos_bearish_filtered,
        is_bos_bullish_momentum=bos_bullish_momentum_filtered,
        is_bos_bearish_momentum=bos_bearish_momentum_filtered,
        is_choch_bullish=choch_bullish_filtered,
        is_choch_bearish=choch_bearish_filtered,
        is_bullish_break_failure=failed_bullish_filtered,
        is_bearish_break_failure=failed_bearish_filtered,
        is_failed_choch_bullish=failed_choch_bullish_filtered,
        is_failed_choch_bearish=failed_choch_bearish_filtered
    )