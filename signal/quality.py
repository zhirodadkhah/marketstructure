# structure/signal/quality.py
"""
Signal quality scoring using market context: regime, session, liquidity, zones.
Pure NumPy implementation with full type safety and O(n) performance.
"""
from typing import Dict
import numpy as np
from structure.metrics.types import ValidatedSignals, SignalQuality
from .config import SignalQualityConfig


def score_signals(
        validated_signals: ValidatedSignals,
        market_regime: np.ndarray,
        session: np.ndarray,
        liquidity_score: np.ndarray,
        zone_confluence: np.ndarray,
        config: SignalQualityConfig
) -> SignalQuality:
    """
    Compute context-aware quality scores for validated signals.

    Parameters
    ----------
    validated_signals : ValidatedSignals
        Confirmed/momentum/failure signal masks.
    market_regime : np.ndarray[str]
        Regime labels: 'strong_trend', 'weak_trend', 'ranging', 'chop', 'neutral'
    session : np.ndarray[str]
        Session tags: 'overlap', 'ny', 'london', 'asia', 'low_liquidity', etc.
    liquidity_score : np.ndarray[float32]
        Normalized liquidity score [0,1]
    zone_confluence : np.ndarray[float32]
        Zone confluence score [0,1]
    config : SignalQualityConfig
        Boosts, weights, and session mappings.

    Returns
    -------
    SignalQuality
        Quality scores for each signal type.

    Notes
    -----
    - Base score = 0.5
    - Regime boost: +config.regime_boost if trend state
    - Zone boost: +zone_confluence * config.zone_boost
    - Liquidity boost: +liquidity_score * 0.1
    - Session boost: configurable per session
    - Final score clipped to [0,1]
    - Time Complexity: O(n)
    - Space Complexity: O(n)
    """
    n = len(market_regime)
    base_score = 0.5

    # Regime boost mask: trending markets only
    is_trending = np.isin(market_regime, ['strong_trend', 'weak_trend'])

    # Session boost array
    session_boost = np.zeros(n, dtype=np.float32)
    for sess, weight in config.session_weights.items():
        mask = (session == sess)
        session_boost[mask] = (weight - 1.0) * 0.1  # Scale to max ±0.1

    def _compute_score(signal_mask: np.ndarray) -> np.ndarray:
        score = np.full(n, 0.0, dtype=np.float32)
        active_mask = signal_mask & (signal_mask != 0)  # defensive
        if not np.any(active_mask):
            return score

        score[active_mask] = base_score
        score[active_mask] += is_trending[active_mask].astype(np.float32) * config.regime_boost
        score[active_mask] += zone_confluence[active_mask] * config.zone_boost
        score[active_mask] += liquidity_score[active_mask] * 0.1
        score[active_mask] += session_boost[active_mask]
        return np.clip(score, 0.0, 1.0)

    return SignalQuality(
        bos_bullish_quality=_compute_score(validated_signals.is_bos_bullish_confirmed),
        bos_bearish_quality=_compute_score(validated_signals.is_bos_bearish_confirmed),
        # CHOCH signals not implemented yet → return zeros
        choch_bullish_quality=np.zeros(n, dtype=np.float32),
        choch_bearish_quality=np.zeros(n, dtype=np.float32)
    )