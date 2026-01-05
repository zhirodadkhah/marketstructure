# structure/signal/quality.py
"""
Signal quality scoring using market context: regime, session, liquidity, zones.
Pure NumPy implementation with full type safety and O(n) performance.

QUALITY SCORING LOGIC:
- Base score = 0.5 (confirmed BOS), 0.4 (momentum), 0.45 (CHOCH), 0.1 (failed)
- Confirmed BOS: boosted by trending regime
- CHOCH: boosted by ranging/neutral regime (reversal context)
- All signals boosted by zone confluence, liquidity, and session
QUALITY SCORING LOGIC:
- 0.0: No signal
- 0.1: Failed signals (minimal quality)
- 0.4: Momentum BOS base quality
- 0.45: CHOCH base quality
- 0.5: Confirmed BOS base quality
- Up to 1.0: With context boosts (regime, zones, liquidity, session)

DESIGN PRINCIPLES:
- Defensive programming: All inputs validated, all edge cases handled
- Performance: Vectorized operations, O(n) time complexity
- Clarity: Explicit error messages, clean separation of concerns
"""

from __future__ import annotations
from typing import Dict, Set
import numpy as np

from structure.metrics.types import ValidatedSignals, SignalQuality, RawSignals
from .config import SignalQualityConfig

# ==============================================================================
# SECTION: Constants
# ==============================================================================

_VALID_REGIMES: Set[str] = {'strong_trend', 'weak_trend', 'ranging', 'chop', 'neutral'}

# Base quality scores by signal type
_BASE_BOS_CONFIRMED: float = 0.5  # Validated BOS with retest + follow-through
_BASE_BOS_MOMENTUM: float = 0.4  # BOS without retest (momentum break)
_BASE_CHOCH: float = 0.45  # CHOCH reversal signals (non-failed)
_BASE_FAILED: float = 0.1  # Failed signals (minimal quality)

_MIN_QUALITY: float = 0.0
_MAX_QUALITY: float = 1.0


# ==============================================================================
# SECTION: Input Validation
# ==============================================================================

def _validate_inputs(
        market_regime: np.ndarray,
        session: np.ndarray,
        liquidity_score: np.ndarray,
        zone_confluence: np.ndarray,
        config: SignalQualityConfig
) -> None:
    """
    Validate all inputs meet expected constraints.

    Raises
    ------
    ValueError
        If any input violates constraints.
    TypeError
        If array dtypes are incorrect.

    Notes
    -----
    - Empty arrays are valid (no data to validate)
    - Empty strings are filtered out before validation
    - All validation errors include debugging information
    """
    n = len(market_regime)

    # === ARRAY LENGTH VALIDATION ===
    if not all(len(arr) == n for arr in [session, liquidity_score, zone_confluence]):
        raise ValueError(
            f"Array length mismatch: market_regime={n}, "
            f"session={len(session)}, liquidity={len(liquidity_score)}, "
            f"zone={len(zone_confluence)}. All arrays must have same length."
        )

    # === DTYPE VALIDATION ===
    if not np.issubdtype(liquidity_score.dtype, np.floating):
        raise TypeError(
            f"liquidity_score must be float array, got {liquidity_score.dtype}. "
            f"Array shape: {liquidity_score.shape}"
        )
    if not np.issubdtype(zone_confluence.dtype, np.floating):
        raise TypeError(
            f"zone_confluence must be float array, got {zone_confluence.dtype}. "
            f"Array shape: {zone_confluence.shape}"
        )

    # === REGIME VALIDATION ===
    # Filter out empty strings before validation
    if market_regime.size > 0:
        non_empty_mask = market_regime != ''
        if np.any(non_empty_mask):
            non_empty_regimes = market_regime[non_empty_mask]
            unique_regimes = set(np.unique(non_empty_regimes))
            invalid_regimes = unique_regimes - _VALID_REGIMES
            if invalid_regimes:
                raise ValueError(
                    f"Invalid market_regime values: {invalid_regimes}. "
                    f"Valid values: {_VALID_REGIMES}. "
                    f"Array shape: {market_regime.shape}, dtype: {market_regime.dtype}"
                )

    # === SESSION VALIDATION ===
    if not config.session_weights:
        raise ValueError("config.session_weights cannot be empty")

    valid_sessions = set(config.session_weights.keys())
    if session.size > 0:
        # Handle empty strings safely
        non_empty_mask = session != ''
        if np.any(non_empty_mask):
            non_empty_sessions = session[non_empty_mask]
            unique_sessions = set(np.unique(non_empty_sessions))
            invalid_sessions = unique_sessions - valid_sessions
            if invalid_sessions:
                raise ValueError(
                    f"Invalid session values: {invalid_sessions}. "
                    f"Valid sessions: {valid_sessions}. "
                    f"Array shape: {session.shape}, dtype: {session.dtype}"
                )

    # === SCORE RANGE VALIDATION ===
    for name, arr in [("liquidity_score", liquidity_score),
                      ("zone_confluence", zone_confluence)]:
        if arr.size > 0:  # Only validate non-empty arrays
            if np.any(arr < 0) or np.any(arr > 1):
                raise ValueError(
                    f"{name} must be in range [0, 1], "
                    f"got min={arr.min():.3f}, max={arr.max():.3f}. "
                    f"Array shape: {arr.shape}"
                )


# ==============================================================================
# SECTION: Core Scoring Engine
# ==============================================================================

def _compute_session_boost(
        session: np.ndarray,
        config: SignalQualityConfig
) -> np.ndarray:
    """
    Compute session boost array based on config weights.

    Returns
    -------
    np.ndarray[float32]
        Boost values where boost = (weight - 1.0) * scale
    """
    n = len(session)
    boost = np.zeros(n, dtype=np.float32)

    for sess_name, sess_weight in config.session_weights.items():
        sess_mask = (session == sess_name)
        # Boost = (weight - 1.0) * scale
        # Example: weight=1.2, scale=0.1 → boost=0.02
        boost[sess_mask] = (sess_weight - 1.0) * config.session_boost_scale

    return boost


def _compute_signal_score(
        signal_mask: np.ndarray,
        base_score: float,
        market_regime: np.ndarray,
        session_boost: np.ndarray,
        liquidity_score: np.ndarray,
        zone_confluence: np.ndarray,
        config: SignalQualityConfig,
        regime_preference: str  # 'trending' or 'reversal'
) -> np.ndarray:
    """
    Compute quality score for a signal type with context boosts.

    Parameters
    ----------
    signal_mask : np.ndarray[bool]
        Boolean mask indicating where signals exist
    base_score : float
        Starting quality score for this signal type
    regime_preference : str
        'trending': Prefer trending regimes (for BOS signals)
        'reversal': Prefer ranging/neutral regimes (for CHOCH signals)

    Returns
    -------
    np.ndarray[float32]
        Quality scores clamped to [0.1, 1.0]

    Notes
    -----
    - Scores where no signal exists are 0.0
    - All boosts are additive to base score
    - Final score clamped to prevent overflow/underflow
    """
    n = len(signal_mask)
    if not np.any(signal_mask):
        return np.zeros(n, dtype=np.float32)

    # Initialize scores (0.0 where no signal)
    scores = np.zeros(n, dtype=np.float32)
    active_idx = np.where(signal_mask)[0]

    # Apply base score to active signals
    scores[active_idx] = base_score

    # === REGIME BOOST ===
    if regime_preference == 'trending':
        # BOS: Boost in trending regimes
        regime_mask = np.isin(market_regime, ['strong_trend', 'weak_trend'])
        scores[active_idx] += regime_mask[active_idx].astype(np.float32) * config.regime_boost
    else:  # 'reversal'
        # CHOCH: Boost in reversal-friendly regimes
        regime_mask = np.isin(market_regime, ['ranging', 'neutral', 'chop'])
        scores[active_idx] += regime_mask[active_idx].astype(np.float32) * config.regime_boost

    # === CONTEXT BOOSTS ===
    scores[active_idx] += zone_confluence[active_idx] * config.zone_boost
    scores[active_idx] += liquidity_score[active_idx] * config.liquidity_boost
    scores[active_idx] += session_boost[active_idx]

    # === CLAMP AND RETURN ===
    return np.clip(scores, _MIN_QUALITY, _MAX_QUALITY)


def _extract_choch_signals(
        raw_signals: RawSignals,
        validated_signals: ValidatedSignals
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract valid CHOCH signals (non-failed).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (valid_choch_bullish, valid_choch_bearish) masks

    Notes
    -----
    - Valid CHOCH = Raw CHOCH AND NOT Failed CHOCH
    - Handles missing raw CHOCH data gracefully
    - Ensures output arrays have correct length
    """
    n = len(validated_signals.is_bos_bullish_confirmed)

    # Get raw CHOCH signals with defensive checks
    if (hasattr(raw_signals, 'is_choch_bullish') and
            raw_signals.is_choch_bullish is not None and
            len(raw_signals.is_choch_bullish) == n):
        raw_choch_bull = raw_signals.is_choch_bullish
    else:
        # Fallback: assume no raw CHOCH signals
        raw_choch_bull = np.zeros(n, dtype=bool)

    if (hasattr(raw_signals, 'is_choch_bearish') and
            raw_signals.is_choch_bearish is not None and
            len(raw_signals.is_choch_bearish) == n):
        raw_choch_bear = raw_signals.is_choch_bearish
    else:
        # Fallback: assume no raw CHOCH signals
        raw_choch_bear = np.zeros(n, dtype=bool)

    # Remove failed CHOCH signals
    valid_choch_bull = raw_choch_bull & ~validated_signals.is_failed_choch_bullish
    valid_choch_bear = raw_choch_bear & ~validated_signals.is_failed_choch_bearish

    return valid_choch_bull, valid_choch_bear


def _create_failed_quality(
        fail_mask: np.ndarray,
        n: int
) -> np.ndarray:
    """
    Create quality array with minimal score for failed signals.

    Parameters
    ----------
    fail_mask : np.ndarray[bool]
        Boolean mask of failed signals
    n : int
        Total number of bars

    Returns
    -------
    np.ndarray[float32]
        Quality array where failed signals = _BASE_FAILED, others = 0.0
    """
    quality = np.zeros(n, dtype=np.float32)
    quality[fail_mask] = _BASE_FAILED
    return quality


# ==============================================================================
# SECTION: Public Interface
# ==============================================================================

def score_signals(
        raw_signals: RawSignals,
        validated_signals: ValidatedSignals,
        market_regime: np.ndarray,
        session: np.ndarray,
        liquidity_score: np.ndarray,
        zone_confluence: np.ndarray,
        config: SignalQualityConfig
) -> SignalQuality:
    """
    Compute context-aware quality scores [0.1, 1.0] for all signal types.

    Parameters
    ----------
    raw_signals : RawSignals
        Original detected signals (contains raw CHOCH signals)
    validated_signals : ValidatedSignals
        Validated signals with confirmation/failure status
    market_regime : np.ndarray[str]
        Market regime classification per bar.
        Valid: {'strong_trend', 'weak_trend', 'ranging', 'chop', 'neutral'}
    session : np.ndarray[str]
        Trading session per bar (e.g., 'ny', 'london', 'asia')
        Must match keys in `config.session_weights`
    liquidity_score : np.ndarray[float32]
        Normalized liquidity score [0.0, 1.0] per bar
    zone_confluence : np.ndarray[float32]
        Zone confluence score [0.0, 1.0] per bar
    config : SignalQualityConfig
        Boost weights and session mappings

    Returns
    -------
    SignalQuality
        Quality scores with the following semantics:
        - 0.0: No signal present at that bar
        - 0.1: Failed signal (break failure or immediate failure)
        - 0.4-0.5: Valid signal base quality (momentum or confirmed BOS, or CHOCH)
        - 0.45-1.0: Valid CHOCH signal with context boosts
        - All scores clamped to [0.0, 1.0] range

    Raises
    ------
    ValueError
        If inputs violate validation rules.
    TypeError
        If input dtypes are incorrect.

    Notes
    -----
    - BOS (trend-following) signals prefer trending regimes
    - CHOCH (reversal) signals prefer ranging/neutral regimes
    - Failed signals get minimal quality (0.1) for filtering
    - All scores clamped to [0.1, 1.0] range
    - Time complexity: O(n), Memory: O(n)

    Examples
    --------
    >>> config = SignalQualityConfig(
    ...     session_weights={'ny': 1.2, 'london': 1.1, 'asia': 0.9},
    ...     regime_boost=0.2,
    ...     zone_boost=0.15,
    ...     liquidity_boost=0.1,
    ...     session_boost_scale=0.1
    ... )
    >>> quality = score_signals(
    ...     raw_signals,
    ...     validated_signals,
    ...     market_regime,
    ...     session,
    ...     liquidity_score,
    ...     zone_confluence,
    ...     config
    ... )
    >>> quality.bos_bullish_confirmed_quality.shape
    (n_bars,)
    """
    # === VALIDATE INPUTS ===
    _validate_inputs(market_regime, session, liquidity_score, zone_confluence, config)

    n = len(market_regime)

    # Precompute session boost once (used by all signal types)
    session_boost = _compute_session_boost(session, config)

    # === SCORE BOS SIGNALS (TREND-FOLLOWING) ===
    bos_bullish_confirmed_quality = _compute_signal_score(
        signal_mask=validated_signals.is_bos_bullish_confirmed,
        base_score=_BASE_BOS_CONFIRMED,
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=config,
        regime_preference='trending'
    )

    bos_bearish_confirmed_quality = _compute_signal_score(
        signal_mask=validated_signals.is_bos_bearish_confirmed,
        base_score=_BASE_BOS_CONFIRMED,
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=config,
        regime_preference='trending'
    )

    bos_bullish_momentum_quality = _compute_signal_score(
        signal_mask=validated_signals.is_bos_bullish_momentum,
        base_score=_BASE_BOS_MOMENTUM,
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=config,
        regime_preference='trending'
    )

    bos_bearish_momentum_quality = _compute_signal_score(
        signal_mask=validated_signals.is_bos_bearish_momentum,
        base_score=_BASE_BOS_MOMENTUM,
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=config,
        regime_preference='trending'
    )

    # === SCORE CHOCH SIGNALS (REVERSAL) ===
    valid_choch_bull, valid_choch_bear = _extract_choch_signals(raw_signals, validated_signals)

    choch_bullish_quality = _compute_signal_score(
        signal_mask=valid_choch_bull,
        base_score=_BASE_CHOCH,
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=config,
        regime_preference='reversal'
    )

    choch_bearish_quality = _compute_signal_score(
        signal_mask=valid_choch_bear,
        base_score=_BASE_CHOCH,
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=config,
        regime_preference='reversal'
    )

    # === FAILED SIGNALS GET MINIMAL QUALITY ===
    failed_bullish_quality = _create_failed_quality(
        validated_signals.is_bullish_break_failure |
        validated_signals.is_bullish_immediate_failure,
        n
    )

    failed_bearish_quality = _create_failed_quality(
        validated_signals.is_bearish_break_failure |
        validated_signals.is_bearish_immediate_failure,
        n
    )

    failed_choch_bullish_quality = _create_failed_quality(
        validated_signals.is_failed_choch_bullish,
        n
    )

    failed_choch_bearish_quality = _create_failed_quality(
        validated_signals.is_failed_choch_bearish,
        n
    )

    return SignalQuality(
        bos_bullish_confirmed_quality=bos_bullish_confirmed_quality,
        bos_bearish_confirmed_quality=bos_bearish_confirmed_quality,
        bos_bullish_momentum_quality=bos_bullish_momentum_quality,
        bos_bearish_momentum_quality=bos_bearish_momentum_quality,
        choch_bullish_quality=choch_bullish_quality,
        choch_bearish_quality=choch_bearish_quality,
        failed_bullish_quality=failed_bullish_quality,
        failed_bearish_quality=failed_bearish_quality,
        failed_choch_bullish_quality=failed_choch_bullish_quality,
        failed_choch_bearish_quality=failed_choch_bearish_quality
    )