# tests/metrics/test_efficiency.py
"""
Comprehensive test suite for efficiency metrics.
Includes positive, negative, and boundary tests.
"""
import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from structure.metrics.efficiency import (
    compute_fractal_efficiency,
    compute_consistency_score,
    compute_fractal_efficiency_extended
)


# ==============================================================================
# Tests for compute_fractal_efficiency
# ==============================================================================

def test_compute_fractal_efficiency_perfect():
    """Test perfect efficiency (straight line)."""
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=3)

    # With period=3:
    # At index 2 (first valid): close[2]=102, close[0]=100
    # Net: |102-100| = 2
    # Path: |101-100| + |102-101| = 1 + 1 = 2
    # Efficiency: 2/2 = 1.0
    assert result[2] == pytest.approx(1.0, abs=1e-6)
    assert result[3] == pytest.approx(1.0, abs=1e-6)
    assert result[4] == pytest.approx(1.0, abs=1e-6)


def test_compute_fractal_efficiency_inefficient():
    """Test inefficient movement (zigzag)."""
    close = np.array([100.0, 101.0, 100.0, 101.0, 100.0], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=5)

    # With period=5:
    # At index 4: close[4]=100, close[0]=100
    # Net: |100-100| = 0
    # Path: |101-100| + |100-101| + |101-100| + |100-101| = 1+1+1+1 = 4
    # Efficiency: 0/4 = 0.0
    assert result[4] == pytest.approx(0.0, abs=1e-6)


def test_compute_fractal_efficiency_partial():
    """Test partial efficiency."""
    close = np.array([100.0, 102.0, 101.0, 103.0, 102.0], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=4)

    # For period=4 at index 3: close[3]=103, close[0]=100
    # Net: |103-100| = 3
    # Path: |102-100| + |101-102| + |103-101| = 2 + 1 + 2 = 5
    # Efficiency: 3/5 = 0.6
    assert result[3] == pytest.approx(0.6, abs=1e-6)

    # At index 4: close[4]=102, close[1]=102
    # Net: |102-102| = 0
    # Path: |101-102| + |103-101| + |102-103| = 1 + 2 + 1 = 4
    # Efficiency: 0/4 = 0.0
    assert result[4] == pytest.approx(0.0, abs=1e-6)


def test_compute_fractal_efficiency_period_1():
    """Test period=1 edge case."""
    close = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=1)

    # Period 1: always 1.0 after first element
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(1.0, abs=1e-6)
    assert result[2] == pytest.approx(1.0, abs=1e-6)


def test_compute_fractal_efficiency_all_same():
    """Test all same prices."""
    close = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=3)

    # Net=0, Path=0 -> Efficiency=1.0 (special case)
    assert result[3] == pytest.approx(1.0, abs=1e-6)


def test_compute_fractal_efficiency_empty():
    """Test empty input."""
    close = np.array([], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=3)

    assert len(result) == 0
    assert result.dtype == np.float32


def test_compute_fractal_efficiency_small():
    """Test input smaller than period."""
    close = np.array([100.0, 101.0], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=3)

    assert len(result) == 2
    assert np.all(np.isnan(result))


def test_compute_fractal_efficiency_nan():
    """Test with NaN values."""
    close = np.array([100.0, np.nan, 102.0, 103.0], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=2)

    # Implementation handles NaN by setting to 0.0
    assert len(result) == 4
    # Check last value: |103-102|/|103-102| = 1.0
    assert result[3] == pytest.approx(1.0, abs=1e-6)


def test_compute_fractal_efficiency_invalid_period():
    """Test invalid period values."""
    close = np.array([100.0, 101.0], dtype=np.float32)

    with pytest.raises(ValueError, match="period must be ≥ 1"):
        compute_fractal_efficiency(close, period=0)

    with pytest.raises(ValueError, match="period must be ≥ 1"):
        compute_fractal_efficiency(close, period=-1)


def test_compute_fractal_efficiency_negative():
    """Test all negative prices."""
    close = np.array([-100.0, -101.0, -102.0], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=2)

    # Should work with negatives
    assert result[1:] == pytest.approx(1.0, abs=1e-6)


def test_compute_fractal_efficiency_zero_path():
    """Test with zero path length (all same values)."""
    close = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    result = compute_fractal_efficiency(close, period=3)

    # Implementation handles net=0, path=0 as efficiency=1.0
    assert result[3] == pytest.approx(1.0, abs=1e-6)


# ==============================================================================
# Tests for compute_consistency_score
# ==============================================================================

def test_compute_consistency_score_basic():
    """Test basic consistency score calculation."""
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float32)
    result = compute_consistency_score(close, window=3)

    # Should return an array
    assert isinstance(result, np.ndarray)
    assert len(result) == len(close)
    assert result.dtype == np.float32

    # With window=3 on straight line, should be 1.0
    assert result[4] == pytest.approx(1.0, abs=1e-6)


def test_compute_consistency_score_identical_to_efficiency():
    """Test that consistency score matches fractal efficiency for same window."""
    close = np.random.uniform(100.0, 110.0, 50).astype(np.float32)

    window = 10
    consistency = compute_consistency_score(close, window=window)
    efficiency = compute_fractal_efficiency(close, period=window)

    # They should be mathematically identical
    assert np.allclose(consistency, efficiency, equal_nan=True, rtol=1e-6)


def test_compute_consistency_score_edge_cases():
    """Test edge cases for consistency score."""
    # Empty input
    close_empty = np.array([], dtype=np.float32)
    result_empty = compute_consistency_score(close_empty, window=3)
    assert len(result_empty) == 0

    # Single element
    close_single = np.array([100.0], dtype=np.float32)
    result_single = compute_consistency_score(close_single, window=3)
    assert len(result_single) == 1
    assert np.isnan(result_single[0])

    # All same values
    close_same = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    result_same = compute_consistency_score(close_same, window=2)
    assert result_same[2] == pytest.approx(1.0, abs=1e-6)


# ==============================================================================
# Tests for compute_fractal_efficiency_extended
# ==============================================================================

def test_compute_fractal_efficiency_extended_basic():
    """Test basic extended function."""
    close = np.linspace(100.0, 110.0, 50, dtype=np.float32)
    result = compute_fractal_efficiency_extended(close)

    # Check all expected keys are present
    expected_keys = {
        'efficiency_10', 'efficiency_20', 'consistency_20',
        'fractal_slope', 'fractal_consistency'
    }
    assert set(result.keys()) == expected_keys

    # Check all arrays have correct length
    for key, arr in result.items():
        assert len(arr) == len(close)
        assert arr.dtype == np.float32

    # Check specific metrics have expected properties
    eff_10 = result['efficiency_10']
    eff_20 = result['efficiency_20']

    # Efficiency values should be between 0 and 1 (ignoring NaN)
    eff_10_valid = eff_10[~np.isnan(eff_10)]
    eff_20_valid = eff_20[~np.isnan(eff_20)]

    if len(eff_10_valid) > 0:
        assert np.all(eff_10_valid >= 0.0) and np.all(eff_10_valid <= 1.0)
    if len(eff_20_valid) > 0:
        assert np.all(eff_20_valid >= 0.0) and np.all(eff_20_valid <= 1.0)


def test_compute_fractal_efficiency_extended_empty():
    """Test extended function with empty input."""
    close = np.array([], dtype=np.float32)
    result = compute_fractal_efficiency_extended(close)

    # Should return all keys with empty arrays
    expected_keys = {
        'efficiency_10', 'efficiency_20', 'consistency_20',
        'fractal_slope', 'fractal_consistency'
    }
    assert set(result.keys()) == expected_keys

    for key, arr in result.items():
        assert len(arr) == 0
        assert arr.dtype == np.float32


def test_compute_fractal_efficiency_extended_small():
    """Test extended function with small input."""
    close = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    result = compute_fractal_efficiency_extended(close)

    # Should return all keys with arrays of length 3
    for key, arr in result.items():
        assert len(arr) == 3
        assert arr.dtype == np.float32

        # For small arrays, many values will be NaN
        # That's expected for windows larger than array length


def test_compute_fractal_efficiency_extended_fractal_slope():
    """Test fractal slope calculation."""
    # Create a simple series
    close = np.array([100.0, 101.0, 102.0, 101.0, 102.0], dtype=np.float32)
    result = compute_fractal_efficiency_extended(close)

    eff_10 = result['efficiency_10']
    slope = result['fractal_slope']

    # Slope should be difference of efficiency_10
    # Check non-NaN values
    for i in range(1, len(close)):
        if not np.isnan(eff_10[i]) and not np.isnan(eff_10[i - 1]) and not np.isnan(slope[i]):
            expected = eff_10[i] - eff_10[i - 1]
            assert slope[i] == pytest.approx(expected, abs=1e-6)


def test_compute_fractal_efficiency_extended_fractal_consistency():
    """Test fractal consistency calculation."""
    # Create a longer series for rolling std
    np.random.seed(42)
    close = np.random.uniform(100.0, 110.0, 100).astype(np.float32)
    result = compute_fractal_efficiency_extended(close)

    eff_10 = result['efficiency_10']
    fractal_consistency = result['fractal_consistency']

    # Check that fractal_consistency is rolling std of efficiency_10
    window = 20
    for i in range(window - 1, len(close)):
        if not np.isnan(fractal_consistency[i]):
            window_vals = eff_10[i - window + 1:i + 1]
            # Use nanstd to match the implementation
            expected_std = np.nanstd(window_vals)
            # Allow small tolerance due to floating point differences
            assert fractal_consistency[i] == pytest.approx(expected_std, abs=1e-5)


def test_compute_fractal_efficiency_extended_consistency_matches():
    """Test that consistency_20 matches efficiency_20."""
    np.random.seed(42)
    close = np.random.uniform(100.0, 110.0, 50).astype(np.float32)
    result = compute_fractal_efficiency_extended(close)

    eff_20 = result['efficiency_20']
    consistency_20 = result['consistency_20']

    # They should be identical (when both are not NaN)
    mask = ~np.isnan(eff_20) & ~np.isnan(consistency_20)
    if np.any(mask):
        assert np.allclose(eff_20[mask], consistency_20[mask])


# ==============================================================================
# Property-based tests
# ==============================================================================

def test_efficiency_bounds():
    """Property test: Efficiency always between 0 and 1."""
    np.random.seed(42)
    for _ in range(10):
        n = np.random.randint(10, 100)
        close = np.random.uniform(90, 110, n).astype(np.float32)

        eff_10 = compute_fractal_efficiency(close, 10)
        eff_20 = compute_fractal_efficiency(close, 20)

        # Check bounds (ignore NaN values)
        for eff in [eff_10, eff_20]:
            valid = eff[~np.isnan(eff)]
            if len(valid) > 0:
                assert np.all(valid >= 0.0)
                assert np.all(valid <= 1.0)


def test_extended_output_shapes():
    """Property test: All extended outputs have same shape."""
    np.random.seed(42)
    for n in [5, 20, 50, 100]:
        close = np.random.uniform(100, 110, n).astype(np.float32)
        result = compute_fractal_efficiency_extended(close)

        # All arrays should have same length as input
        for arr in result.values():
            assert len(arr) == n


def test_slope_consistency():
    """Property test: Slope is difference of efficiency."""
    np.random.seed(42)
    close = np.random.uniform(100, 110, 30).astype(np.float32)
    result = compute_fractal_efficiency_extended(close)

    eff_10 = result['efficiency_10']
    slope = result['fractal_slope']

    # For indices where both are not NaN, slope should be difference
    for i in range(1, len(close)):
        if not np.isnan(eff_10[i]) and not np.isnan(eff_10[i - 1]) and not np.isnan(slope[i]):
            expected = eff_10[i] - eff_10[i - 1]
            assert slope[i] == pytest.approx(expected, abs=1e-6)


# ==============================================================================
# Error handling tests
# ==============================================================================

def test_efficiency_wrong_dimensions():
    """Test error for wrong input dimensions."""
    close_2d = np.array([[100.0, 101.0], [102.0, 103.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="close must be 1-dimensional"):
        compute_fractal_efficiency(close_2d, period=3)


def test_efficiency_non_numeric():
    """Test error for non-numeric input."""
    close_str = np.array(['100', '101', '102'])

    with pytest.raises(TypeError, match="close must be numeric"):
        compute_fractal_efficiency(close_str, period=2)


# ==============================================================================
# Integration tests
# ==============================================================================

def test_integration_with_realistic_data():
    """Integration test with realistic price data."""
    # Simulate some realistic price movement
    np.random.seed(42)
    trend = np.linspace(100, 120, 100)
    noise = np.random.normal(0, 1, 100)
    close = (trend + noise).astype(np.float32)

    # Test all functions
    eff_10 = compute_fractal_efficiency(close, 10)
    eff_20 = compute_fractal_efficiency(close, 20)
    extended = compute_fractal_efficiency_extended(close)

    # Check basic properties
    assert len(eff_10) == 100
    assert len(eff_20) == 100
    assert len(extended) == 5  # 5 metrics

    # Check extended contains all metrics
    assert 'efficiency_10' in extended
    assert 'efficiency_20' in extended
    assert 'fractal_slope' in extended
    assert 'fractal_consistency' in extended

    # Check efficiency_10 matches direct calculation
    assert np.allclose(eff_10, extended['efficiency_10'], equal_nan=True)
    assert np.allclose(eff_20, extended['efficiency_20'], equal_nan=True)


def test_performance_large_dataset():
    """Performance test with large dataset."""
    n = 10000
    close = np.random.uniform(100, 200, n).astype(np.float32)

    import time
    start = time.time()
    result = compute_fractal_efficiency_extended(close)
    elapsed = time.time() - start

    # Should complete quickly
    assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s for {n} elements"

    # All arrays should be present
    for key in ['efficiency_10', 'efficiency_20', 'consistency_20',
                'fractal_slope', 'fractal_consistency']:
        assert key in result
        assert len(result[key]) == n


# ==============================================================================
# Helper tests
# ==============================================================================

def test_sliding_window_view_compatibility():
    """Test that sliding_window_view works as expected."""
    # This is a sanity check for the numpy function we depend on
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    windows = sliding_window_view(data, window_shape=3)

    assert windows.shape == (3, 3)  # 3 windows of size 3
    assert np.array_equal(windows[0], [1, 2, 3])
    assert np.array_equal(windows[1], [2, 3, 4])
    assert np.array_equal(windows[2], [3, 4, 5])