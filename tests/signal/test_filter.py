import pytest
import numpy as np
from structure.signal.filter import filter_signals
from structure.metrics.types import ValidatedSignals
from structure.signal.config import SignalFilterConfig

FIELDS = [
    'is_bos_bullish_confirmed', 'is_bos_bearish_confirmed',
    'is_bos_bullish_momentum', 'is_bos_bearish_momentum',
    'is_bullish_break_failure', 'is_bearish_break_failure'
]

def test_filter_positive_all_pass():
    n = 5
    signals = ValidatedSignals(
        **{k: np.ones(n, bool) for k in [
            'is_bos_bullish_confirmed', 'is_bos_bearish_confirmed',
            'is_bos_bullish_momentum', 'is_bos_bearish_momentum',
            'is_bullish_break_failure', 'is_bearish_break_failure'
        ]}
    )
    config = SignalFilterConfig(min_zone_confluence=0.5, avoid_range_compression=True)
    result = filter_signals(
        signals,
        market_regime=np.array(['strong_trend'] * n),
        zone_confluence=np.full(n, 0.9),
        is_range_compression=np.zeros(n, bool),
        retest_velocity=np.zeros(n),
        session=np.array(['ny'] * n),
        config=config
    )
    assert result['is_bos_bullish_confirmed'].all()


def test_filter_negative_low_zone_confluence():
    n = 3
    signals = ValidatedSignals(**{k: np.ones(n, bool) for k in FIELDS})
    config = SignalFilterConfig(min_zone_confluence=0.8)
    result = filter_signals(
        signals,
        market_regime=np.array(['strong_trend'] * n),
        zone_confluence=np.array([0.9, 0.6, 0.9]),
        is_range_compression=np.zeros(n, bool),
        retest_velocity=np.zeros(n),
        session=np.array(['ny'] * n),
        config=config
    )
    assert not result['is_bos_bullish_confirmed'][1]


def test_filter_edge_empty():
    signals = ValidatedSignals(**{k: np.array([], bool) for k in FIELDS})
    config = SignalFilterConfig()
    result = filter_signals(
        signals,
        np.array([]), np.array([]), np.array([], bool), np.array([]), np.array([]), config
    )
    assert len(result['is_bos_bullish_confirmed']) == 0