# test_regime.py
import pytest
import numpy as np
from structure.context.regime import detect_market_regime
from structure.context.config import RegimeConfig


# === POSITIVE TEST ===
def test_regime_positive():
    n = 200
    close = np.cumsum(np.random.randn(n)).astype(np.float32) + 100
    high = close + np.abs(np.random.randn(n)).astype(np.float32)
    low = close - np.abs(np.random.randn(n)).astype(np.float32)
    config = RegimeConfig(
        atr_period=14,
        volatility_window=20,
        regime_swing_window=10,
        regime_consistency_window=15,
        regime_atr_slope_window=10,
        regime_efficiency_low=0.2,
        regime_efficiency_high=0.7,
        regime_swing_density_low=0.05,
        regime_swing_density_moderate=0.1,
        regime_swing_density_high=0.2,
        regime_consistency_high=0.6,
        regime_atr_slope_threshold=0.05,
        regime_threshold=1.5
    )

    result = detect_market_regime(high, low, close, config)

    # All outputs are boolean masks of same length
    for mask in result.values():
        assert mask.dtype == bool
        assert mask.shape == (n,)

    # Mutually exclusive + exhaustive
    stacked = np.vstack(list(result.values()))
    assert np.all(stacked.sum(axis=0) == 1)


# === NEGATIVE TEST: MISMATCHED LENGTHS ===
def test_regime_negative_mismatched_lengths():
    high = np.array([1, 2, 3], dtype=np.float32)
    low = np.array([1, 2], dtype=np.float32)  # too short
    close = np.array([1, 2, 3], dtype=np.float32)
    config = RegimeConfig()

    with pytest.raises(ValueError, match="same-length 1D arrays"):
        detect_market_regime(high, low, close, config)


# === NEGATIVE TEST: INVALID CONFIG (efficiency threshold) ===
def test_regime_negative_invalid_config():
    n = 100
    arr = np.random.rand(n).astype(np.float32)
    high = low = close = arr
    # Break config rule: low >= high
    config = RegimeConfig(regime_efficiency_low=0.8, regime_efficiency_high=0.6)

    with pytest.raises(ValueError, match="0 < low < high < 1"):
        detect_market_regime(high, low, close, config)


# === EDGE CASE: EMPTY INPUT ===
def test_regime_edge_empty():
    high = low = close = np.array([], dtype=np.float32)
    config = RegimeConfig()

    result = detect_market_regime(high, low, close, config)

    for mask in result.values():
        assert mask.shape == (0,)


# === EDGE CASE: TOO FEW BARS (all neutral) ===
def test_regime_edge_too_short():
    n = 5  # less than any window
    high = low = close = np.ones(n, dtype=np.float32)
    config = RegimeConfig(
        atr_period=10,
        volatility_window=10,
        regime_swing_window=10,
        regime_consistency_window=10,
        regime_atr_slope_window=10
    )

    result = detect_market_regime(high, low, close, config)

    assert np.all(result['is_neutral'])
    for key in ['is_strong_trend', 'is_weak_trend', 'is_ranging', 'is_chop']:
        assert np.all(~result[key])