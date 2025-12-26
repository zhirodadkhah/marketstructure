# test_zones.py
import pytest
import numpy as np
from structure.context.zones import detect_sr_zones
from structure.context.config import ZoneConfig

def test_zones_positive():
    n = 200
    close = np.cumsum(np.random.randn(n)) + 100
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    config = ZoneConfig(recent_bars=100, min_cluster_size=2, clustering_radius=2.0)
    sup, res = detect_sr_zones(high, low, close, config)
    assert sup.dtype == np.float32
    assert res.dtype == np.float32
    assert np.all(np.diff(sup) >= 0)  # sorted
    assert np.all(np.diff(res) >= 0)

def test_zones_negative_mismatched_lengths():
    high = np.array([1, 2, 3], dtype=np.float32)
    low = close = np.array([1, 2], dtype=np.float32)  # mismatch
    config = ZoneConfig()
    with pytest.raises(ValueError, match="same length"):
        detect_sr_zones(high, low, close, config)

def test_zones_edge_too_few_bars():
    high = low = close = np.array([1.0, 2.0], dtype=np.float32)
    config = ZoneConfig()
    sup, res = detect_sr_zones(high, low, close, config)
    assert len(sup) == 0
    assert len(res) == 0