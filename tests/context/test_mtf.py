# test_mtf.py
import pytest
import numpy as np
from structure.context.mtf import resample_to_htf
from structure.context.config import MTFConfig

def _create_price_data(n):
    np.random.seed(42)
    base = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = base + np.abs(np.random.randn(n) * 0.2)
    low = base - np.abs(np.random.randn(n) * 0.2)
    close = (high + low) / 2
    open_ = close + np.random.randn(n) * 0.05
    return open_.astype(np.float32), high.astype(np.float32), low.astype(np.float32), close.astype(np.float32)

def _create_timestamps(n, interval_min=60):
    base = np.datetime64('2024-01-01T00:00:00')
    return base + np.arange(n, dtype='timedelta64[s]') * (interval_min * 60)

# --- Positive ---
def test_mtf_positive():
    n = 48
    ts = _create_timestamps(n)
    o, h, l, c = _create_price_data(n)
    config = MTFConfig(htf_bar_size=24)
    result = resample_to_htf(ts, o, h, l, c, config)
    assert len(result['htf_close']) == 2
    assert result['htf_high'][0] == pytest.approx(np.max(h[:24]))
    assert result['htf_low'][0] == pytest.approx(np.min(l[:24]))

# --- Negative ---
def test_mtf_invalid_bar_size():
    ts = _create_timestamps(10)
    o, h, l, c = _create_price_data(10)
    config = MTFConfig(htf_bar_size=0)
    with pytest.raises(ValueError, match="htf_bar_size must be ≥ 1"):
        resample_to_htf(ts, o, h, l, c, config)

def test_mtf_shape_mismatch():
    ts = _create_timestamps(10)
    o, h, l, c = _create_price_data(10)
    o_bad = np.random.randn(8).astype(np.float32)
    config = MTFConfig(htf_bar_size=3)
    with pytest.raises(ValueError, match="same length"):
        resample_to_htf(ts, o_bad, h, l, c, config)

# --- Edge ---
def test_mtf_empty():
    ts = np.array([], dtype='datetime64[s]')
    empty = np.array([], dtype=np.float32)
    config = MTFConfig(htf_bar_size=5)
    result = resample_to_htf(ts, empty, empty, empty, empty, config)
    assert len(result['htf_close']) == 0

def test_mtf_too_short():
    ts = _create_timestamps(2)
    o, h, l, c = _create_price_data(2)
    config = MTFConfig(htf_bar_size=3)
    result = resample_to_htf(ts, o, h, l, c, config)
    assert len(result['htf_close']) == 0

def test_mtf_exact_multiple():
    n = 30
    ts = _create_timestamps(n)
    o, h, l, c = _create_price_data(n)
    config = MTFConfig(htf_bar_size=3)
    result = resample_to_htf(ts, o, h, l, c, config)
    assert len(result['htf_close']) == 10
    assert result['htf_close'][0] == pytest.approx(c[2])