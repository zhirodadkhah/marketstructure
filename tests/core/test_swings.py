# test_swings.py
import pytest
import numpy as np
from structure.core.swings import detect_swing_points
from structure.core.config import SwingConfig

def test_detect_swing_points_positive():
    high = np.array([5, 8, 6, 9, 7, 10, 8], dtype=np.float32)
    low = np.array([4, 5, 4, 6, 5, 7, 6], dtype=np.float32)
    result = detect_swing_points(high, low, half_window=1)
    assert 'is_swing_high' in result
    assert 'is_swing_low' in result
    assert result['is_swing_high'].dtype == bool
    assert result['is_swing_low'].dtype == bool
    assert not np.any(result['is_swing_high'] & result['is_swing_low'])

def test_detect_swing_points_negative_half_window_zero():
    high = low = np.ones(5, dtype=np.float32)
    with pytest.raises(ValueError, match="half_window must be ≥ 1"):
        detect_swing_points(high, low, half_window=0)

def test_detect_swing_points_negative_shape_mismatch():
    high = np.ones(5, dtype=np.float32)
    low = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="same length"):
        detect_swing_points(high, low)

def test_detect_swing_points_negative_non_numeric():
    high = np.array(['a', 'b', 'c'])
    low = np.ones(3, dtype=np.float32)
    with pytest.raises(TypeError, match="numeric"):
        detect_swing_points(high, low)

def test_detect_swing_points_edge_empty():
    empty = np.array([], dtype=np.float32)
    result = detect_swing_points(empty, empty, half_window=2)
    assert result['is_swing_high'].shape == (0,)
    assert result['is_swing_low'].shape == (0,)

def test_detect_swing_points_edge_too_short():
    high = np.array([1, 2], dtype=np.float32)
    low = np.array([0, 1], dtype=np.float32)
    result = detect_swing_points(high, low, half_window=2)
    assert not np.any(result['is_swing_high'])
    assert not np.any(result['is_swing_low'])

def test_detect_swing_points_edge_all_same_price():
    price = np.full(10, 100.0, dtype=np.float32)
    result = detect_swing_points(price, price, half_window=2)
    assert not np.any(result['is_swing_high'])
    assert not np.any(result['is_swing_low'])

def test_detect_swing_points_edge_extreme_values():
    high = np.array([1e10, 2e10, 1e10], dtype=np.float32)
    low = np.array([1e-10, 2e-10, 1e-10], dtype=np.float32)
    result = detect_swing_points(high, low, half_window=1)
    assert result['is_swing_high'].dtype == bool
    assert result['is_swing_low'].dtype == bool

def test_swing_config():
    config = SwingConfig(half_window=3)
    assert config.half_window == 3