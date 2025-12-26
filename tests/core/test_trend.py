# test_trend.py
import pytest
import numpy as np
from structure.core.trend import detect_trend_state
from structure.core.config import TrendConfig

def test_detect_trend_state_positive():
    n = 30
    close = np.linspace(100, 130, n, dtype=np.float32)
    hh = hl = lh = ll = np.zeros(n, dtype=bool)
    hl[5] = True
    hh[10] = True
    result = detect_trend_state(close, hh, lh, hl, ll)
    assert 'trend_state' in result
    assert result['trend_state'].dtype == np.int8
    assert np.all(np.isin(result['trend_state'], [-1, 0, 1]))

def test_detect_trend_state_positive_with_metrics():
    n = 20
    close = np.linspace(100, 120, n, dtype=np.float32)
    hh = hl = np.zeros(n, dtype=bool)
    hl[3] = True
    hh[7] = True
    result = detect_trend_state(close, hh, np.zeros(n, bool), hl, np.zeros(n, bool), include_metrics=True)
    assert 'trend_strength' in result
    assert 'trend_since_index' in result

def test_detect_trend_state_negative_invalid_buffer():
    n = 5
    close = np.ones(n, dtype=np.float32)
    masks = [np.zeros(n, bool) for _ in range(4)]
    with pytest.raises(ValueError, match="invalidation_buffer must be ≥ 0"):
        detect_trend_state(close, *masks, invalidation_buffer=-0.1)

def test_detect_trend_state_negative_shape_mismatch():
    close = np.ones(5, dtype=np.float32)
    hh = np.zeros(4, dtype=bool)
    with pytest.raises(ValueError, match="same length"):
        detect_trend_state(close, hh, hh, hh, hh)

def test_detect_trend_state_edge_all_neutral():
    n = 20
    close = np.random.randn(n).astype(np.float32)
    masks = [np.zeros(n, bool) for _ in range(4)]
    result = detect_trend_state(close, *masks)
    assert np.all(result['trend_state'] == 0)

def test_detect_trend_state_edge_nan_close():
    close = np.array([1.0, 2.0, np.nan, 3.0], dtype=np.float32)
    masks = [np.zeros(4, bool) for _ in range(4)]
    result = detect_trend_state(close, *masks)
    assert result['trend_state'].shape == (4,)

def test_trend_config():
    config = TrendConfig(invalidation_buffer=0.1, include_metrics=True)
    assert config.invalidation_buffer == 0.1
    assert config.include_metrics is True