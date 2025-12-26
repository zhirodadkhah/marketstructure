# test_breaks.py
import pytest
import numpy as np
from structure.core.breaks import detect_structure_break_signals
from structure.core.config import BreakConfig

def test_detect_structure_break_signals_positive():
    n = 50
    high = np.linspace(100, 150, n, dtype=np.float32)
    low = high - 1
    close = (high + low) / 2
    open_ = close.copy()
    atr = np.full(n, 1.0, dtype=np.float32)

    # Create clear HH in uptrend
    hh = np.zeros(n, bool)
    hh[-5] = True
    ll = np.zeros(n, bool)
    sh = sl = np.zeros(n, bool)
    sh[-5] = True
    trend = np.ones(n, dtype=np.int8)  # uptrend

    signals = detect_structure_break_signals(
        high, low, close, open_, atr, sh, sl, hh, ll, trend, min_break_atr_mult=0.5
    )
    assert 'is_bos_bullish_initial' in signals
    for mask in signals.values():
        assert mask.dtype == bool
        assert mask.shape == (n,)

def test_detect_structure_break_signals_negative_invalid_trend():
    n = 5
    inputs = [np.ones(n, dtype=np.float32) for _ in range(9)]
    trend = np.array([0, 1, 2, -1, 0], dtype=np.int8)  # 2 is invalid
    with pytest.raises(ValueError, match="trend_state must be in"):
        detect_structure_break_signals(*inputs[:5], *inputs[5:], trend)


def test_detect_structure_break_signals_all_shape_checks():
    """Test shape validation for each input parameter."""
    n = 10
    # Create all correct arrays
    correct_arrays = {
        'high': np.random.randn(n).astype(np.float32),
        'low': np.random.randn(n).astype(np.float32),
        'close': np.random.randn(n).astype(np.float32),
        'open_': np.random.randn(n).astype(np.float32),
        'atr': np.random.randn(n).astype(np.float32),
        'is_swing_high': np.random.randn(n).astype(np.float32) > 0,
        'is_swing_low': np.random.randn(n).astype(np.float32) > 0,
        'is_higher_high': np.random.randn(n).astype(np.float32) > 0,
        'is_lower_low': np.random.randn(n).astype(np.float32) > 0,
        'trend_state': np.zeros(n, dtype=np.int8)
    }

    # Test each parameter with wrong length
    for param_name in correct_arrays.keys():
        # Create a copy of correct arrays
        test_args = correct_arrays.copy()
        # Replace one with wrong length
        test_args[param_name] = np.random.randn(n - 2).astype(
            correct_arrays[param_name].dtype
        )

        with pytest.raises(ValueError, match="same-length"):
            detect_structure_break_signals(
                test_args['high'], test_args['low'], test_args['close'],
                test_args['open_'], test_args['atr'],
                test_args['is_swing_high'], test_args['is_swing_low'],
                test_args['is_higher_high'], test_args['is_lower_low'],
                test_args['trend_state']
            )

def test_detect_structure_break_signals_edge_all_false():
    n = 20
    price = np.linspace(100, 120, n, dtype=np.float32)
    atr = np.ones(n, dtype=np.float32)
    false_mask = np.zeros(n, bool)
    neutral_trend = np.zeros(n, dtype=np.int8)
    signals = detect_structure_break_signals(
        price, price - 1, price, price, atr,
        false_mask, false_mask, false_mask, false_mask, neutral_trend
    )
    for mask in signals.values():
        assert not np.any(mask)

def test_detect_structure_break_signals_edge_nan_atr():
    n = 10
    price = np.ones(n, dtype=np.float32)
    atr = np.full(n, np.nan, dtype=np.float32)
    masks = [np.zeros(n, bool) for _ in range(4)]
    sh = sl = hh = ll = masks[0]
    trend = np.zeros(n, dtype=np.int8)
    signals = detect_structure_break_signals(
        price, price - 1, price, price, atr, sh, sl, hh, ll, trend
    )
    for mask in signals.values():
        assert mask.shape == (n,)
        assert mask.dtype == bool

def test_break_config():
    config = BreakConfig(
        min_break_atr_mult=0.3,
        buffer_multiplier=0.6,
        follow_through_bars=5
    )
    assert config.min_break_atr_mult == 0.3
    assert config.follow_through_bars == 5