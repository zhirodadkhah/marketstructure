# structure/signal/quality.py
from structure.metrics.types import ValidatedSignals, SignalQuality
from .config import SignalQualityConfig
import numpy as np

def score_signals(
    validated_signals: ValidatedSignals,
    market_regime: np.ndarray,
    zone_confluence: np.ndarray,
    liquidity_score: np.ndarray,
    session: np.ndarray,
    config: SignalQualityConfig
) -> SignalQuality:
    """
        Compute quality scores for each signal type.
        Cyclomatic complexity: 2 | Args: 6
        """
    n = len(market_regime)
    base_score = 0.5

    def _compute_score(signal_mask: np.ndarray) -> np.ndarray:
        score = np.full(n, 0.0, dtype=np.float32)
        if not np.any(signal_mask):
            return score
        score[signal_mask] = base_score
        is_trending = np.isin(market_regime, ['strong_trend', 'weak_trend'])
        score += is_trending.astype(np.float32) * config.regime_boost
        score += zone_confluence * config.zone_boost
        score += liquidity_score * 0.1
        session_boost = np.zeros(n, dtype=np.float32)
        for sess, weight in config.session_weights.items():
            session_boost[session == sess] = weight - 1.0
        score += session_boost * 0.1
        return np.clip(score, 0.0, 1.0)

    return SignalQuality(
        bos_bullish_quality=_compute_score(validated_signals.is_bos_bullish_confirmed),
        bos_bearish_quality=_compute_score(validated_signals.is_bos_bearish_confirmed),
        choch_bullish_quality=_compute_score(np.zeros(n, bool)),
        choch_bearish_quality=_compute_score(np.zeros(n, bool))
    )