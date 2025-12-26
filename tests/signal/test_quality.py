import pytest
import numpy as np
from structure.signal.quality import score_signals
from structure.metrics.types import ValidatedSignals
from structure.signal.config import SignalQualityConfig


def test_quality_positive():
    n = 4
    signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([True, False, True, False]),
        is_bos_bearish_confirmed=np.array([False, True, False, True]),
        is_bos_bullish_momentum=np.zeros(n, bool),
        is_bos_bearish_momentum=np.zeros(n, bool),
        is_bullish_break_failure=np.zeros(n, bool),
        is_bearish_break_failure=np.zeros(n, bool)
    )
    result = score_signals(
        signals,
        market_regime=np.array(['strong_trend'] * n),
        zone_confluence=np.ones(n) * 0.8,
        liquidity_score=np.ones(n) * 0.7,
        session=np.array(['ny'] * n),
        config=SignalQualityConfig()
    )
    assert result.bos_bullish_quality[0] > 0.5
    assert 0 <= result.bos_bullish_quality[0] <= 1.0


def test_quality_negative_no_signals():
    n = 3
    signals = ValidatedSignals(**{k: np.zeros(n, bool) for k in [
        'is_bos_bullish_confirmed', 'is_bos_bearish_confirmed',
        'is_bos_bullish_momentum', 'is_bos_bearish_momentum',
        'is_bullish_break_failure', 'is_bearish_break_failure'
    ]})
    result = score_signals(signals, np.array(['ranging']*n), np.zeros(n), np.zeros(n), np.array(['asia']*n), SignalQualityConfig())
    assert np.all(result.bos_bullish_quality == 0.0)


def test_quality_edge_empty():
    signals = ValidatedSignals(**{k: np.array([], bool) for k in [
        'is_bos_bullish_confirmed', 'is_bos_bearish_confirmed',
        'is_bos_bullish_momentum', 'is_bos_bearish_momentum',
        'is_bullish_break_failure', 'is_bearish_break_failure'
    ]})
    result = score_signals(signals, np.array([]), np.array([]), np.array([]), np.array([]), SignalQualityConfig())
    assert len(result.bos_bullish_quality) == 0