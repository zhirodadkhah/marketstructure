# test_structure.py
import pytest
import numpy as np
from structure.core.structure import detect_market_structure

def test_detect_market_structure_positive():
    high = np.array([5, 8, 6, 9, 7, 10, 8], dtype=np.float32)
    low = np.array([4, 5, 4, 6, 5, 7, 6], dtype=np.float32)
    swings = {
        'is_swing_high': np.array([False, True, False, True, False, True, False]),
        'is_swing_low':  np.array([True, False, True, False, True, False, True])
    }
    result = detect_market_structure(high, low, **swings)
    keys = ['is_higher_high', 'is_lower_high', 'is_higher_low', 'is_lower_low']
    for k in keys:
        assert k in result
        assert result[k].dtype == bool
    # Only swings should be labeled
    swing_mask = swings['is_swing_high'] | swings['is_swing_low']
    for mask in result.values():
        assert np.all(mask[~swing_mask] == False)

def test_detect_market_structure_negative_mutually_exclusive():
    n = 5
    high = low = np.ones(n, dtype=np.float32)
    sh = np.array([True, False, False, False, False])
    sl = np.array([True, False, False, False, False])  # conflict at 0
    with pytest.raises(ValueError, match="mutually exclusive"):
        detect_market_structure(high, low, sh, sl)

def test_detect_market_structure_negative_shape_mismatch():
    high = np.ones(5, dtype=np.float32)
    low = np.ones(4, dtype=np.float32)
    sh = sl = np.zeros(5, dtype=bool)
    with pytest.raises(ValueError, match="same-length"):
        detect_market_structure(high, low, sh, sl)

def test_detect_market_structure_edge_no_swings():
    n = 10
    high = low = np.random.randn(n).astype(np.float32)
    sh = sl = np.zeros(n, dtype=bool)
    result = detect_market_structure(high, low, sh, sl)
    for mask in result.values():
        assert not np.any(mask)