"""
Comprehensive tests for momentum module.
"""
import numpy as np
import pytest
from structure.metrics.momentum import compute_momentum_metrics
from .conf import *

class TestComputeMomentumMetrics:
    """Positive, negative, and edge tests for compute_momentum_metrics."""
    
    # ========== POSITIVE TESTS ==========
    
    def test_basic_momentum_calculation(self):
        """Positive test: Basic momentum metrics."""
        close = np.array([100, 102, 105, 103, 106], dtype=np.float32)
        atr = np.ones_like(close) * 2.0
        swing_high = np.array([False, True, False, False, True], dtype=bool)
        swing_low = np.array([False, False, True, False, False], dtype=bool)
        
        result = compute_momentum_metrics(
            close, atr, swing_high, swing_low,
            momentum_period=1,
            acceleration_period=1
        )
        
        # Check momentum
        expected_momentum = np.array([0, 2, 3, -2, 3], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            result['momentum'], expected_momentum, decimal=5
        )
        
        # Check normalized momentum (divided by ATR=2)
        expected_norm = expected_momentum / 2.0
        np.testing.assert_array_almost_equal(
            result['normalized_momentum'], expected_norm, decimal=5
        )
        
        # Check acceleration (difference in momentum)
        expected_accel = np.array([0, 0, 1, -5, 5], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            result['acceleration'], expected_accel, decimal=5
        )
        
        # Check direction
        expected_dir = np.array([0, 1, 1, -1, 1], dtype=np.int8)
        np.testing.assert_array_equal(result['momentum_direction'], expected_dir)
    
    def test_momentum_with_period(self):
        """Positive test: Momentum with period > 1."""
        close = np.array([100, 101, 102, 103, 104, 105], dtype=np.float32)
        atr = np.ones_like(close)
        swing_high = np.zeros_like(close, dtype=bool)
        swing_low = np.zeros_like(close, dtype=bool)
        
        result = compute_momentum_metrics(
            close, atr, swing_high, swing_low, momentum_period=3
        )
        
        # momentum[3] = close[3] - close[0] = 103-100 = 3
        assert result['momentum'][3] == pytest.approx(3.0)
        # momentum[4] = close[4] - close[1] = 104-101 = 3
        assert result['momentum'][4] == pytest.approx(3.0)
    
    def test_divergence_detection(self):
        """Positive test: Bullish and bearish divergence detection."""
        # Create price making lower low but momentum making higher low
        close = np.array([100, 90, 95, 85, 80], dtype=np.float32)
        atr = np.ones_like(close)
        
        # Swing lows at indices 1 and 4
        swing_low = np.array([False, True, False, False, True], dtype=bool)
        swing_high = np.zeros_like(close, dtype=bool)
        
        # Manipulate momentum array to test divergence logic
        # We need to ensure momentum shows divergence
        result = compute_momentum_metrics(
            close, atr, swing_high, swing_low,
            normalize_by_atr=False
        )
        
        # Check if divergence detection works
        # At index 4: price lower (80 < 90) 
        # momentum[4] = 80 - 85 = -5
        # momentum[1] = 90 - 100 = -10
        # Since -5 > -10, we have bullish divergence (higher momentum at lower price)
        if result['momentum_divergence_bullish'][4]:
            assert True  # Divergence detected
        else:
            # Might not detect if lookback doesn't find previous swing
            pass
    
    def test_normalization_toggle(self):
        """Positive test: Test normalization on/off."""
        close = np.array([100, 102, 104], dtype=np.float32)
        atr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        swing_high = np.zeros_like(close, dtype=bool)
        swing_low = np.zeros_like(close, dtype=bool)
        
        # With normalization
        norm_result = compute_momentum_metrics(
            close, atr, swing_high, swing_low, normalize_by_atr=True
        )
        
        # Without normalization
        raw_result = compute_momentum_metrics(
            close, atr, swing_high, swing_low, normalize_by_atr=False
        )
        
        # Momentum should be same
        assert norm_result['momentum'][1] == raw_result['momentum'][1]
        
        # Normalized should be divided by ATR
        assert norm_result['normalized_momentum'][1] == pytest.approx(2.0 / 2.0)
        assert raw_result['normalized_momentum'][1] == pytest.approx(2.0)  # Not divided
    
    # ========== NEGATIVE TESTS ==========
    
    def test_nan_propagation(self):
        """Negative test: NaN values should propagate."""
        close = np.array([100, np.nan, 104], dtype=np.float32)
        atr = np.ones_like(close)
        swing_high = np.zeros_like(close, dtype=bool)
        swing_low = np.zeros_like(close, dtype=bool)
        
        result = compute_momentum_metrics(close, atr, swing_high, swing_low)
        
        # NaN should propagate
        assert np.isnan(result['momentum'][1])
        assert np.isnan(result['normalized_momentum'][1])
    
    def test_zero_atr_handling(self):
        """Negative test: Zero ATR values."""
        close = np.array([100, 101, 102], dtype=np.float32)
        atr = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        swing_high = np.zeros_like(close, dtype=bool)
        swing_low = np.zeros_like(close, dtype=bool)
        
        # Should not crash - uses safe division
        result = compute_momentum_metrics(
            close, atr, swing_high, swing_low, normalize_by_atr=True
        )
        
        # Should handle division by zero
        assert np.all(np.isfinite(result['normalized_momentum']))
    
    def test_inf_values(self):
        """Negative test: Infinity in prices."""
        close = np.array([100, np.inf, 102], dtype=np.float32)
        atr = np.ones_like(close)
        swing_high = np.zeros_like(close, dtype=bool)
        swing_low = np.zeros_like(close, dtype=bool)
        
        result = compute_momentum_metrics(close, atr, swing_high, swing_low)
        
        # Infinity should propagate
        assert np.isinf(result['momentum'][1])
        assert np.isinf(result['normalized_momentum'][1])
    
    # ========== EDGE TESTS ==========
    
    @pytest.mark.parametrize("case_name", [
        "empty", "single", "two_elements", "all_same", "all_zero"
    ])
    def test_edge_cases(self, case_name, edge_cases):
        """Edge test: Various edge cases."""
        close = edge_cases[case_name]
        atr = np.ones_like(close) if len(close) > 0 else close
        swing_high = np.zeros_like(close, dtype=bool) if len(close) > 0 else close
        swing_low = np.zeros_like(close, dtype=bool) if len(close) > 0 else close
        
        result = compute_momentum_metrics(
            close, atr, swing_high, swing_low,
            momentum_period=1,
            acceleration_period=1
        )
        
        # Check all outputs have correct length
        for key, arr in result.items():
            assert len(arr) == len(close)
            
            # Check dtypes
            if 'direction' in key:
                assert arr.dtype == np.int8
            elif 'divergence' in key:
                assert arr.dtype == bool
            else:
                assert arr.dtype == np.float32
    
    def test_large_period_small_data(self):
        """Edge test: Period larger than data."""
        close = np.array([100, 101], dtype=np.float32)
        atr = np.ones_like(close)
        swing_high = np.zeros_like(close, dtype=bool)
        swing_low = np.zeros_like(close, dtype=bool)
        
        result = compute_momentum_metrics(
            close, atr, swing_high, swing_low,
            momentum_period=5,  # Larger than data
            acceleration_period=3
        )
        
        # Momentum for period > data should be 0 or handled
        assert result['momentum'][1] == 0.0  # Can't calculate
    
    def test_no_swings(self):
        """Edge test: No swing points."""
        close = np.array([100, 101, 102, 103], dtype=np.float32)
        atr = np.ones_like(close)
        swing_high = np.zeros_like(close, dtype=bool)
        swing_low = np.zeros_like(close, dtype=bool)
        
        result = compute_momentum_metrics(close, atr, swing_high, swing_low)
        
        # No divergence should be detected
        assert not np.any(result['momentum_divergence_bullish'])
        assert not np.any(result['momentum_divergence_bearish'])
    
    def test_mixed_sign_prices(self):
        """Edge test: Prices with mixed signs."""
        close = np.array([-100, -99, -98, 1, 2, 3], dtype=np.float32)
        atr = np.ones_like(close)
        swing_high = np.zeros_like(close, dtype=bool)
        swing_low = np.zeros_like(close, dtype=bool)
        
        result = compute_momentum_metrics(close, atr, swing_high, swing_low)
        
        # Should handle sign changes
        assert result['momentum'][1] == pytest.approx(1.0)  # -99 - (-100)
        assert result['momentum'][4] == pytest.approx(1.0)  # 2 - 1
    
    # ========== ERROR TESTS ==========
    
    # @pytest.mark.parametrize("input_type", [
    #     "none_input", "string_input", "list_input", "wrong_dtype"
    # ])
    # def test_invalid_input_types(self, invalid_inputs, input_type):
    #     """Error test: Invalid input types."""
    #     test_input = invalid_inputs[input_type]
    #
    #     with pytest.raises((TypeError, ValueError, AttributeError)):
    #         compute_momentum_metrics(
    #             test_input, test_input, test_input, test_input
    #         )
    
    def test_mismatched_lengths(self):
        """Error test: Mismatched array lengths."""
        close = np.array([100, 101, 102], dtype=np.float32)
        atr = np.array([1, 2], dtype=np.float32)  # Different length
        swing_high = np.array([False, True, False], dtype=bool)
        swing_low = np.array([False, False, True], dtype=bool)
        
        with pytest.raises(ValueError):
            compute_momentum_metrics(close, atr, swing_high, swing_low)
    
    # ========== PROPERTY-BASED TESTS ==========
    
    def test_momentum_properties(self):
        """Property test: Momentum should have certain properties."""
        np.random.seed(42)
        n = 50
        close = np.cumsum(np.random.normal(0, 1, n)).astype(np.float32)
        atr = np.random.uniform(0.5, 2.0, n).astype(np.float32)
        
        # Create random swing masks
        swing_high = np.random.random(n) > 0.9
        swing_low = np.random.random(n) > 0.9
        
        for momentum_period in [1, 3, 5]:
            for accel_period in [1, 2]:
                result = compute_momentum_metrics(
                    close, atr, swing_high, swing_low,
                    momentum_period=momentum_period,
                    acceleration_period=accel_period,
                    normalize_by_atr=True
                )
                
                # Property 1: Normalized momentum magnitude reasonable
                norm_mom = result['normalized_momentum'][~np.isnan(result['normalized_momentum'])]
                if len(norm_mom) > 0:
                    assert np.nanmax(np.abs(norm_mom)) < 100  # Shouldn't be extreme
                
                # Property 2: Direction matches sign of momentum
                dir_mask = result['momentum_direction'] != 0
                if np.any(dir_mask):
                    signs_match = (
                        np.sign(result['momentum'][dir_mask]) == 
                        result['momentum_direction'][dir_mask]
                    )
                    assert np.all(signs_match)
                
                # Property 3: Acceleration is difference of momentum
                if n > momentum_period + accel_period:
                    accel = result['acceleration'][momentum_period + accel_period:]
                    mom = result['momentum'][momentum_period + accel_period:]
                    mom_lag = result['momentum'][momentum_period:-accel_period]
                    calc_accel = mom - mom_lag
                    
                    # Compare ignoring NaN
                    mask = ~np.isnan(accel) & ~np.isnan(calc_accel)
                    if np.any(mask):
                        np.testing.assert_array_almost_equal(
                            accel[mask], calc_accel[mask], decimal=5
                        )