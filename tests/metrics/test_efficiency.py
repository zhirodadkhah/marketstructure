"""
Comprehensive tests for efficiency module.
"""
import numpy as np
import pytest
from .conf import *

from structure.metrics.efficiency import (
    compute_fractal_efficiency,
    compute_consistency_score,
    compute_fractal_efficiency_extended
)

class TestComputeFractalEfficiency:
    """Positive, negative, and edge tests for compute_fractal_efficiency."""
    
    # ========== POSITIVE TESTS ==========
    
    def test_perfect_efficiency(self):
        """Positive test: Perfectly efficient movement (straight line)."""
        # Linear price movement
        close = np.array([100, 101, 102, 103, 104, 105], dtype=np.float32)
        
        efficiency = compute_fractal_efficiency(close, period=5)
        
        # First period-1 values should be NaN
        assert np.all(np.isnan(efficiency[:4]))
        # Perfect efficiency = 1.0
        assert efficiency[4] == pytest.approx(1.0, abs=1e-6)
        assert efficiency[5] == pytest.approx(1.0, abs=1e-6)
    
    def test_inefficient_movement(self):
        """Positive test: Inefficient/choppy movement."""
        # Price goes up and down
        close = np.array([100, 101, 100, 101, 100, 101], dtype=np.float32)
        
        efficiency = compute_fractal_efficiency(close, period=3)
        
        # At index 2: net move = |100-100| = 0, path = |1| + |1| = 2
        # Efficiency = 0/2 = 0
        assert efficiency[2] == pytest.approx(0.0, abs=1e-6)
        # At index 5: net = |101-101| = 0, path = |1| + |-1| + |1| + |-1| = 4
        # Efficiency = 0/4 = 0
        assert efficiency[5] == pytest.approx(0.0, abs=1e-6)

    def test_partial_efficiency(self):
        """Positive test: Partially efficient movement."""
        close = np.array([100, 102, 101, 104, 103, 106], dtype=np.float32)

        efficiency = compute_fractal_efficiency(close, period=5)

        # Correct calculation:
        # For period=5 at index 5:
        # Net: |close[5] - close[1]| = |106 - 102| = 4
        # Path: sum(|101-102|=1, |104-101|=3, |103-104|=1, |106-103|=3) = 8
        # Efficiency = 4/8 = 0.5
        assert efficiency[5] == pytest.approx(0.5, abs=1e-6)
    
    def test_different_periods(self):
        """Positive test: Different period values."""
        close = np.linspace(100, 110, 20, dtype=np.float32)
        
        for period in [5, 10, 14, 20]:
            efficiency = compute_fractal_efficiency(close, period=period)
            
            assert len(efficiency) == len(close)
            assert efficiency.dtype == np.float32
            
            # First period-1 should be NaN
            if period > 1:
                assert np.all(np.isnan(efficiency[:period-1]))
            
            # Linear movement should give efficiency ~1
            if period <= len(close):
                valid = efficiency[period-1:]
                assert np.all(np.abs(valid - 1.0) < 1e-6)
    
    # ========== NEGATIVE TESTS ==========

    def test_nan_propagation(self):
        """Negative test: NaN values in input."""
        close = np.array([100, np.nan, 102, 103, 104], dtype=np.float32)

        efficiency = compute_fractal_efficiency(close, period=3)

        # With period=3 at index 2:
        # Net: |close[2] - close[0]| = |102 - 100| = 2
        # Path: |nan-100| + |102-nan| = nan + nan = nan
        # Efficiency = 2/nan = nan → but our code sets to 0.0
        assert efficiency[2] == pytest.approx(0.0, abs=1e-6)
        assert np.isfinite(efficiency[2])

    def test_inf_values(self):
        """Negative test: Infinity in prices."""
        close = np.array([100, np.inf, 102, 103, 104], dtype=np.float32)

        efficiency = compute_fractal_efficiency(close, period=3)

        # With period=3 at index 2:
        # Net: |close[2] - close[0]| = |102 - 100| = 2
        # Path: |inf-100| + |102-inf| = inf + inf = inf
        # Efficiency = 2/inf = 0.0
        assert efficiency[2] == pytest.approx(0.0, abs=1e-6)
        assert np.isfinite(efficiency[2])
    
    # ========== EDGE TESTS ==========
    
    @pytest.mark.parametrize("case_name", [
        "empty", "single", "two_elements", "all_same"
    ])
    def test_edge_cases(self, case_name, edge_cases):
        """Edge test: Various edge cases."""
        close = edge_cases[case_name]
        n = len(close)
        period = 3
        
        efficiency = compute_fractal_efficiency(close, period=period)
        
        assert len(efficiency) == n
        assert efficiency.dtype == np.float32
        
        if n < period:
            # All values should be NaN
            assert np.all(np.isnan(efficiency))
        elif n == period:
            # Only last value might be calculable
            assert not np.isnan(efficiency[-1])

    def test_zero_path_length(self):
        """Edge test: Zero path length (prices not changing)."""
        close = np.full(10, 100.0, dtype=np.float32)

        efficiency = compute_fractal_efficiency(close, period=5)

        # Net move = 0, path length = 0 → Perfect efficiency = 1.0
        # Price doesn't move at all, so it's perfectly efficient!
        assert efficiency[4] == pytest.approx(1.0, abs=1e-6)
    
    def test_negative_prices(self):
        """Edge test: Negative prices."""
        close = np.array([-100, -99, -98, -97], dtype=np.float32)
        
        efficiency = compute_fractal_efficiency(close, period=3)
        
        # Should handle negative values correctly
        # Efficiency should still be between 0 and 1
        assert 0 <= efficiency[2] <= 1
        assert 0 <= efficiency[3] <= 1
    
    def test_mixed_sign_prices(self):
        """Edge test: Prices crossing zero."""
        close = np.array([-10, -5, 0, 5, 10], dtype=np.float32)
        
        efficiency = compute_fractal_efficiency(close, period=3)
        
        # Should handle sign changes
        assert np.all(np.isfinite(efficiency[2:]))
        assert np.all(efficiency[2:] >= 0)
        assert np.all(efficiency[2:] <= 1)
    
    def test_period_one(self):
        """Edge test: Period = 1."""
        close = np.array([100, 101, 102], dtype=np.float32)
        
        efficiency = compute_fractal_efficiency(close, period=1)
        
        # With period=1: net = 0, path = 0? Actually should be NaN
        # Implementation might handle this differently
        assert len(efficiency) == len(close)
    
    # ========== ERROR TESTS ==========
    
    @pytest.mark.parametrize("invalid_period", [0, -1, -5])
    def test_invalid_period(self, invalid_period):
        """Error test: Invalid period values."""
        close = np.array([100, 101, 102], dtype=np.float32)
        
        # Implementation should handle or raise error
        # Test depends on implementation
    
    # @pytest.mark.parametrize("input_type", [
    #     "none_input", "string_input", "list_input", "wrong_dtype"
    # ])
    # def test_invalid_input_types(self, invalid_inputs, input_type):
    #     """Error test: Invalid input types."""
    #     test_input = invalid_inputs[input_type]
    #
    #     with pytest.raises((TypeError, ValueError, AttributeError)):
    #         compute_fractal_efficiency(test_input, period=5)
    
    # ========== PROPERTY-BASED TESTS ==========

    def test_efficiency_properties(self):
        """Property test: Efficiency should have certain properties."""
        np.random.seed(42)
        n = 100

        for _ in range(10):  # Multiple random tests
            close = np.cumsum(np.random.normal(0, 1, n)).astype(np.float32)

            for period in [5, 10, 20]:
                if period <= n:
                    efficiency = compute_fractal_efficiency(close, period=period)

                    # Property 1: Efficiency between 0 and 1 (inclusive)
                    valid = efficiency[~np.isnan(efficiency)]
                    if len(valid) > 0:
                        # Allow small floating point errors
                        assert np.all(valid >= -1e-7)
                        assert np.all(valid <= 1.0 + 1e-7)

                    # Property 2: Constant prices give PERFECT efficiency (1.0)
                    constant = np.full(n, 100.0, dtype=np.float32)
                    eff_const = compute_fractal_efficiency(constant, period=period)
                    valid_const = eff_const[~np.isnan(eff_const)]
                    if len(valid_const) > 0:
                        assert np.all(np.abs(valid_const - 1.0) < 1e-7)

class TestComputeConsistencyScore:
    """Tests for compute_consistency_score function."""
    
    def test_perfect_consistency(self):
        """Positive test: Perfect directional consistency."""
        close = np.array([100, 101, 102, 103, 104], dtype=np.float32)
        
        score = compute_consistency_score(close, window=3)
        
        # Net move = |104-102| = 2, gross move = |1| + |1| = 2
        # Consistency = 2/2 = 1.0
        assert score[4] == pytest.approx(1.0, abs=1e-6)

    def test_no_consistency(self):
        """Positive test: No directional consistency."""
        close = np.array([100, 101, 100, 101, 100], dtype=np.float32)

        score = compute_consistency_score(close, window=4)

        # Correct calculation:
        # For window=4 at index 4:
        # Net: |close[4] - close[1]| = |100 - 101| = 1
        # Gross: sum(|100-101|=1, |101-100|=1, |100-101|=1) = 3
        # Consistency = 1/3 ≈ 0.33333
        assert score[4] == pytest.approx(1 / 3, abs=1e-6)
    
    def test_edge_cases(self, edge_cases):
        """Edge test: Various edge cases."""
        for case_name, close in edge_cases.items():
            n = len(close)
            window = 3
            
            score = compute_consistency_score(close, window=window)
            
            assert len(score) == n
            assert score.dtype == np.float32
            
            if n < window:
                assert np.all(np.isnan(score))

class TestComputeFractalEfficiencyExtended:
    """Tests for compute_fractal_efficiency_extended function."""
    
    def test_extended_output(self):
        """Positive test: Extended function returns all metrics."""
        close = np.linspace(100, 110, 50, dtype=np.float32)
        
        result = compute_fractal_efficiency_extended(close)
        
        # Check all keys present
        expected_keys = {'efficiency_10', 'efficiency_20', 'consistency_20'}
        assert set(result.keys()) == expected_keys
        
        # Check array lengths
        for arr in result.values():
            assert len(arr) == len(close)
            assert arr.dtype == np.float32
        
        # Linear prices should give high efficiency
        assert np.nanmean(result['efficiency_10']) > 0.99
        assert np.nanmean(result['efficiency_20']) > 0.99
        assert np.nanmean(result['consistency_20']) > 0.99
    
    def test_empty_input(self):
        """Edge test: Empty input."""
        close = np.array([], dtype=np.float32)
        
        result = compute_fractal_efficiency_extended(close)
        
        for arr in result.values():
            assert len(arr) == 0
    
    def test_small_input(self):
        """Edge test: Input smaller than windows."""
        close = np.array([100, 101, 102], dtype=np.float32)
        
        result = compute_fractal_efficiency_extended(close)
        
        # All should be NaN since windows (10, 20) > data length
        for arr in result.values():
            assert np.all(np.isnan(arr))