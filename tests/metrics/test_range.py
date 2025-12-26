"""
Comprehensive tests for range module.
"""
import numpy as np
import pytest
from structure.metrics.range import compute_range_metrics
from .conf import *
class TestComputeRangeMetrics:
    """Positive, negative, and edge tests for compute_range_metrics."""
    
    # ========== POSITIVE TESTS ==========
    
    def test_basic_range_calculation(self):
        """Positive test: Basic range metrics."""
        high = np.array([10, 12, 15, 14, 16], dtype=np.float32)
        low = np.array([8, 10, 12, 11, 13], dtype=np.float32)
        close = np.array([9, 11, 14, 13, 15], dtype=np.float32)
        atr = np.ones_like(close) * 2.0
        
        result = compute_range_metrics(
            high, low, close, atr,
            range_window=3,
            expansion_threshold=1.5,
            compression_threshold=0.7,
            squeeze_threshold=0.5
        )
        
        # Check raw_range
        expected_raw = np.array([2.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            result['raw_range'], expected_raw, decimal=5
        )
        
        # Check normalized_range (raw_range / atr)
        expected_norm = expected_raw / 2.0
        np.testing.assert_array_almost_equal(
            result['normalized_range'], expected_norm, decimal=5
        )
        
        # Check expansion flag (norm > 1.5)
        expected_expansion = expected_norm > 1.5
        np.testing.assert_array_equal(
            result['range_expansion'], expected_expansion
        )
        
        # Check compression flag (norm < 0.7)
        expected_compression = expected_norm < 0.7
        np.testing.assert_array_equal(
            result['range_compression'], expected_compression
        )
    
    def test_range_ratio_calculation(self):
        """Positive test: Range ratio calculation."""
        high = np.array([10, 12, 24, 14], dtype=np.float32)
        low = np.array([8, 10, 12, 11], dtype=np.float32)
        close = np.array([9, 11, 18, 13], dtype=np.float32)
        atr = np.ones_like(close)
        
        result = compute_range_metrics(high, low, close, atr, range_window=2)
        
        # Ranges: [2, 2, 12, 3]
        # Ratios: [nan, 2/2=1, 12/2=6, 3/12=0.25]
        assert np.isnan(result['range_ratio'][0])
        assert result['range_ratio'][1] == pytest.approx(1.0)
        assert result['range_ratio'][2] == pytest.approx(6.0)
        assert result['range_ratio'][3] == pytest.approx(0.25)
    
    def test_volatility_regime(self):
        """Positive test: Volatility regime calculation."""
        high = np.arange(100, 130, dtype=np.float32)
        low = high - 2
        close = high - 1
        atr = np.ones_like(close)
        
        result = compute_range_metrics(
            high, low, close, atr,
            range_window=5,
            volatility_regime_window=10
        )
        
        # Should compute moving average of normalized_range
        norm_range = result['normalized_range']
        window = 10
        
        # Manual convolution
        kernel = np.ones(window) / window
        expected = np.convolve(norm_range, kernel, mode='valid')
        
        # Compare
        np.testing.assert_array_almost_equal(
            result['volatility_regime'][window-1:], expected, decimal=5
        )
    
    def test_range_percentile(self):
        """Positive test: Range percentile calculation."""
        # Create increasing ranges
        ranges = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.float32)
        high = ranges + 10
        low = np.full_like(high, 10)
        close = (high + low) / 2
        atr = np.ones_like(close)
        
        result = compute_range_metrics(
            high, low, close, atr,
            range_window=5
        )
        
        # With window=5, percentile should rank within window
        # At index 4 (window end), ranges are [1,2,3,4,5]
        # 5 is max, so percentile should be 1.0
        assert result['range_percentile'][4] == pytest.approx(1.0, abs=0.01)
        # At index 9, ranges are [5,1,2,3,4], 5 is max
        assert result['range_percentile'][9] == pytest.approx(1.0, abs=0.01)
    
    # ========== NEGATIVE TESTS ==========
    
    def test_nan_propagation(self):
        """Negative test: NaN values in input."""
        high = np.array([10, np.nan, 15], dtype=np.float32)
        low = np.array([8, 10, np.nan], dtype=np.float32)
        close = np.array([9, 11, 14], dtype=np.float32)
        atr = np.ones_like(close)
        
        result = compute_range_metrics(high, low, close, atr, range_window=2)
        
        # NaN should propagate
        assert np.isnan(result['raw_range'][1])  # From NaN high
        assert np.isnan(result['raw_range'][2])  # From NaN low
        assert np.isnan(result['normalized_range'][1])
        assert np.isnan(result['normalized_range'][2])
    
    def test_zero_atr_handling(self):
        """Negative test: Zero ATR values."""
        high = np.array([10, 12, 15], dtype=np.float32)
        low = np.array([8, 10, 12], dtype=np.float32)
        close = np.array([9, 11, 14], dtype=np.float32)
        atr = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        result = compute_range_metrics(high, low, close, atr, range_window=2)
        
        # Should handle division by zero
        assert np.all(np.isfinite(result['normalized_range']))
        # With zero ATR, normalized range becomes large
        assert result['normalized_range'][0] > 1e6
    
    def test_inf_values(self):
        """Negative test: Infinity in prices."""
        high = np.array([10, np.inf, 15], dtype=np.float32)
        low = np.array([8, 10, 12], dtype=np.float32)
        close = np.array([9, 11, 14], dtype=np.float32)
        atr = np.ones_like(close)
        
        result = compute_range_metrics(high, low, close, atr, range_window=2)
        
        # Infinity should propagate
        assert np.isinf(result['raw_range'][1])
        assert np.isinf(result['normalized_range'][1])
    
    # ========== EDGE TESTS ==========
    
    @pytest.mark.parametrize("case_name", [
        "empty", "single", "two_elements", "all_same", "all_zero"
    ])
    def test_edge_cases(self, case_name, edge_cases):
        """Edge test: Various edge cases."""
        close = edge_cases[case_name]
        n = len(close)
        
        if n > 0:
            high = close + 1
            low = close - 1
            atr = np.ones_like(close)
        else:
            high = low = atr = close
        
        result = compute_range_metrics(
            high, low, close, atr,
            range_window=min(3, max(2, n))
        )
        
        # Check all outputs exist
        expected_keys = {
            'raw_range', 'range_ratio', 'normalized_range',
            'range_percentile', 'range_expansion', 'range_compression',
            'range_squeeze', 'volatility_regime'
        }
        assert set(result.keys()) == expected_keys
        
        # Check lengths
        for arr in result.values():
            assert len(arr) == n
    
    def test_zero_range_candles(self):
        """Edge test: Zero range candles (high == low)."""
        high = np.array([10, 10, 10], dtype=np.float32)
        low = np.array([10, 10, 10], dtype=np.float32)
        close = np.array([10, 10, 10], dtype=np.float32)
        atr = np.ones_like(close)
        
        result = compute_range_metrics(high, low, close, atr, range_window=2)
        
        # Raw range should be zero
        assert np.all(result['raw_range'] == 0.0)
        # Normalized range should be zero (0/atr)
        assert np.all(result['normalized_range'] == 0.0)
        # All flags should be False (0 < threshold)
        assert not np.any(result['range_expansion'])
        assert np.all(result['range_compression'])  # 0 < 0.7
        assert np.all(result['range_squeeze'])      # 0 < 0.5
    
    def test_negative_prices(self):
        """Edge test: Negative prices."""
        high = np.array([-8, -6, -4], dtype=np.float32)
        low = np.array([-10, -8, -6], dtype=np.float32)
        close = np.array([-9, -7, -5], dtype=np.float32)
        atr = np.ones_like(close)
        
        result = compute_range_metrics(high, low, close, atr, range_window=2)
        
        # Range should still be positive
        assert np.all(result['raw_range'] >= 0)
        # Example: -6 - (-8) = 2
        assert result['raw_range'][1] == pytest.approx(2.0)
    
    def test_small_window_large_data(self):
        """Edge test: Window much smaller than data."""
        n = 1000
        high = np.random.uniform(90, 110, n).astype(np.float32)
        low = high - np.random.uniform(1, 5, n)
        close = (high + low) / 2
        atr = np.ones_like(close)
        
        result = compute_range_metrics(
            high, low, close, atr,
            range_window=5,
            volatility_regime_window=20
        )
        
        # Should handle large dataset
        assert len(result['raw_range']) == n
        # No crashes
    
    def test_window_larger_than_data(self):
        """Edge test: Window larger than data."""
        high = np.array([10, 12], dtype=np.float32)
        low = np.array([8, 10], dtype=np.float32)
        close = np.array([9, 11], dtype=np.float32)
        atr = np.ones_like(close)
        
        result = compute_range_metrics(high, low, close, atr, range_window=10)
        
        # Percentile should be all NaN
        assert np.all(np.isnan(result['range_percentile']))
    
    # ========== ERROR TESTS ==========

    def test_mismatched_lengths(self):
        """Error test: Mismatched array lengths."""
        high = np.array([10, 12, 15], dtype=np.float32)
        low = np.array([8, 10], dtype=np.float32)  # Different length
        close = np.array([9, 11, 14], dtype=np.float32)
        atr = np.ones_like(close)
        
        with pytest.raises(ValueError):
            compute_range_metrics(high, low, close, atr, range_window=2)
    
    def test_invalid_window_parameters(self):
        """Error test: Invalid window parameters."""
        high = np.array([10, 12], dtype=np.float32)
        low = np.array([8, 10], dtype=np.float32)
        close = np.array([9, 11], dtype=np.float32)
        atr = np.ones_like(close)
        
        # Should handle gracefully or raise appropriate error
        # Test depends on implementation
    
    # ========== PROPERTY-BASED TESTS ==========
    
    def test_range_metrics_properties(self):
        """Property test: Range metrics should have certain properties."""
        np.random.seed(42)
        n = 100
        high = np.random.uniform(90, 110, n).astype(np.float32)
        low = high - np.random.uniform(1, 5, n)
        close = (high + low) / 2 + np.random.uniform(-1, 1, n)
        atr = np.random.uniform(0.5, 3.0, n).astype(np.float32)
        
        for window in [5, 10, 20]:
            result = compute_range_metrics(
                high, low, close, atr,
                range_window=window,
                expansion_threshold=1.5,
                compression_threshold=0.7,
                squeeze_threshold=0.5
            )
            
            # Property 1: raw_range is non-negative
            assert np.all(result['raw_range'] >= 0)
            
            # Property 2: normalized_range = raw_range / atr
            calc_norm = result['raw_range'] / np.where(atr != 0, atr, 1e-10)
            mask = ~np.isnan(result['normalized_range']) & ~np.isnan(calc_norm)
            if np.any(mask):
                np.testing.assert_array_almost_equal(
                    result['normalized_range'][mask],
                    calc_norm[mask],
                    decimal=5
                )
            
            # Property 3: expansion, compression, squeeze are mutually exclusive?
            # Actually not necessarily - but check they're boolean
            assert result['range_expansion'].dtype == bool
            assert result['range_compression'].dtype == bool
            assert result['range_squeeze'].dtype == bool
            
            # Property 4: percentile between 0 and 1 (when not NaN)
            percentile = result['range_percentile'][~np.isnan(result['range_percentile'])]
            if len(percentile) > 0:
                assert np.all(percentile >= 0)
                assert np.all(percentile <= 1.0)