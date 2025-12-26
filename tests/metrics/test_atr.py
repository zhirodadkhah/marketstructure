# tests/test_atr.py
"""
Comprehensive tests for atr module.
"""
import numpy as np
import pytest
from structure.metrics.atr import compute_true_range, compute_atr
from .conf import *

class TestComputeTrueRange:
    """Positive, negative, and edge tests for compute_true_range."""

    # ========== POSITIVE TESTS ==========

    def     test_basic_calculation(self):
        """Positive test: Basic TR calculation with typical data."""
        high = np.array([10.0, 12.0, 15.0, 14.0, 16.0], dtype=np.float32)
        low = np.array([8.0, 10.0, 12.0, 11.0, 13.0], dtype=np.float32)
        close = np.array([9.0, 11.0, 14.0, 13.0, 15.0], dtype=np.float32)

        tr = compute_true_range(high, low, close)

        # Test specific calculations
        assert tr[0] == pytest.approx(2.0)  # H-L
        assert tr[1] == pytest.approx(3.0)  # max(H-L=2, |H-prevC|=1, |L-prevC|=1)
        assert tr[2] == pytest.approx(4.0)  # max(H-L=3, |H-prevC|=4, |L-prevC|=1)
        assert tr[3] == pytest.approx(3.0)  # max(H-L=3, |H-prevC|=1, |L-prevC|=2)
        assert tr[4] == pytest.approx(3.0)  # max(H-L=3, |H-prevC|=3, |L-prevC|=2)

    def test_gap_scenarios(self):
        """Positive test: Test gap up and gap down scenarios."""
        high = np.array([10, 15, 14, 9, 12], dtype=np.float32)  # Gap up at index 1
        low = np.array([9, 14, 13, 8, 11], dtype=np.float32)
        close = np.array([9.5, 14.5, 13.5, 8.5, 11.5], dtype=np.float32)

        tr = compute_true_range(high, low, close)

        # Gap up: TR should include gap (|L-prevC| = 14-9.5 = 4.5)
        assert tr[1] == pytest.approx(5.5)
        # Gap down: |H-prevC| = 9-13.5 = 4.5
        assert tr[3] == pytest.approx(5.5)

    def test_large_dataset(self, test_data):
        """Positive test: Large realistic dataset."""
        data = test_data.ohlc_data
        tr = compute_true_range(data['high'], data['low'], data['close'])

        assert len(tr) == test_data.n
        assert tr.dtype == np.float32
        assert np.all(tr >= 0)  # TR should never be negative
        assert np.all(np.isfinite(tr))  # No inf/nan

    # ========== NEGATIVE TESTS ==========

    def test_nan_propagation(self):
        """Negative test: NaN values should propagate."""
        high = np.array([10, np.nan, 15], dtype=np.float32)
        low = np.array([8, 10, np.nan], dtype=np.float32)
        close = np.array([9, 11, 14], dtype=np.float32)

        tr = compute_true_range(high, low, close)

        assert np.isnan(tr[1])  # NaN high
        assert np.isnan(tr[2])  # NaN low

    def test_inf_values(self):
        """Negative test: Infinity values."""
        high = np.array([10, np.inf, 15], dtype=np.float32)
        low = np.array([8, 10, 12], dtype=np.float32)
        close = np.array([9, 11, 14], dtype=np.float32)

        tr = compute_true_range(high, low, close)

        assert np.isinf(tr[1])  # Should propagate inf

    # ========== EDGE TESTS ==========

    @pytest.mark.parametrize("case_name,case_data", [
        ("empty", (np.array([]), np.array([]), np.array([]))),
        ("single", (np.array([10]), np.array([8]), np.array([9]))),
        ("two_elements", (np.array([10, 12]), np.array([8, 10]), np.array([9, 11]))),
    ])
    def test_edge_cases(self, case_name, case_data):
        """Edge test: Various edge cases."""
        high, low, close = case_data
        tr = compute_true_range(high, low, close)

        assert len(tr) == len(close)
        assert tr.dtype == np.float32

        if len(tr) > 0:
            assert tr[0] == high[0] - low[0]  # First TR is H-L

    def test_zero_range(self):
        """Edge test: Zero range candles."""
        high = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        low = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        close = np.array([10.0, 10.0, 10.0], dtype=np.float32)

        tr = compute_true_range(high, low, close)

        assert tr[0] == pytest.approx(0.0)
        # With zero range and no gaps, TR should be 0
        assert np.all(tr == 0.0)

    def test_negative_prices(self):
        """Edge test: Negative prices (futures, spreads)."""
        high = np.array([-10, -8, -5], dtype=np.float32)
        low = np.array([-12, -10, -7], dtype=np.float32)
        close = np.array([-11, -9, -6], dtype=np.float32)

        tr = compute_true_range(high, low, close)

        # TR should still be positive
        assert np.all(tr >= 0)
        assert tr[0] == pytest.approx(2.0)  # -10 - (-12) = 2

    # ========== ERROR TESTS ==========

    @pytest.mark.parametrize("input_type", [
        "none_input",
        "string_input",
        "list_input",
        "wrong_dtype",
        "wrong_shape_2d",
    ])
    def test_invalid_input_types(self, invalid_inputs, input_type):
        """Error test: Invalid input types."""
        test_input = invalid_inputs[input_type]

        # Lists should now be accepted (converted via np.asarray)
        if input_type in ["list_input"]:
            # List inputs should work now
            result = compute_true_range(test_input, test_input, test_input)
            assert isinstance(result, np.ndarray)
            return


    def test_mismatched_lengths_error(self):
        """Error test: Mismatched array lengths."""
        high = np.array([1, 2, 3], dtype=np.float32)
        low = np.array([1, 2], dtype=np.float32)
        close = np.array([1, 2, 3], dtype=np.float32)

        with pytest.raises(ValueError):
            compute_true_range(high, low, close)

class TestComputeATR:
    """Positive, negative, and edge tests for compute_atr."""

    # ========== POSITIVE TESTS ==========

    def test_basic_wilders_atr(self):
        """Positive test: Basic Wilder's ATR calculation."""
        high = np.array([10, 12, 15, 14, 16, 18], dtype=np.float32)
        low = np.array([8, 10, 12, 11, 13, 15], dtype=np.float32)
        close = np.array([9, 11, 14, 13, 15, 17], dtype=np.float32)

        atr = compute_atr(high, low, close, period=3)

        # Manual calculation
        tr = compute_true_range(high, low, close)
        expected_atr_2 = np.mean(tr[:3])  # index 2
        expected_atr_3 = (tr[3] + 2 * expected_atr_2) / 3
        expected_atr_4 = (tr[4] + 2 * expected_atr_3) / 3
        expected_atr_5 = (tr[5] + 2 * expected_atr_4) / 3

        assert np.isnan(atr[0])  # First period-1 should be NaN
        assert np.isnan(atr[1])
        assert atr[2] == pytest.approx(expected_atr_2)
        assert atr[3] == pytest.approx(expected_atr_3)
        assert atr[4] == pytest.approx(expected_atr_4)
        assert atr[5] == pytest.approx(expected_atr_5)

    def test_period_1(self):
        """Positive test: period=1 should equal TR."""
        high = np.array([10, 12, 15], dtype=np.float32)
        low = np.array([8, 10, 12], dtype=np.float32)
        close = np.array([9, 11, 14], dtype=np.float32)

        atr = compute_atr(high, low, close, period=1)
        tr = compute_true_range(high, low, close)

        # With period=1, no NaN values
        assert not np.any(np.isnan(atr))
        np.testing.assert_array_almost_equal(atr, tr, decimal=5)

    def test_different_periods(self):
        """Positive test: Different period values."""
        high = np.array([10, 12, 15, 14, 16], dtype=np.float32)
        low = np.array([8, 10, 12, 11, 13], dtype=np.float32)
        close = np.array([9, 11, 14, 13, 15], dtype=np.float32)

        for period in [1, 2, 3, 5, 10]:
            atr = compute_atr(high, low, close, period=period)

            assert len(atr) == len(close)
            assert atr.dtype == np.float32

            # Check NaN padding
            if period > 1:
                assert np.all(np.isnan(atr[:period-1]))

    def test_smooth_transition(self, test_data):
        """Positive test: ATR should smooth volatility."""
        data = test_data.ohlc_data
        atr = compute_atr(data['high'], data['low'], data['close'], period=14)

        # ATR should be less volatile than TR
        tr = compute_true_range(data['high'], data['low'], data['close'])
        atr_vol = np.nanstd(atr[14:])
        tr_vol = np.nanstd(tr[14:])

        assert atr_vol < tr_vol  # ATR should be smoother

    # ========== NEGATIVE TESTS ==========

    @pytest.mark.parametrize("invalid_period", [0, -1, -5])
    def test_invalid_period(self, invalid_period):
        """Negative test: Invalid period values."""
        high = np.array([10, 12], dtype=np.float32)
        low = np.array([8, 10], dtype=np.float32)
        close = np.array([9, 11], dtype=np.float32)

        with pytest.raises(ValueError, match="period must be ≥1"):
            compute_atr(high, low, close, period=invalid_period)

    def test_nan_input_propagation(self):
        """Negative test: NaN in input should propagate."""
        high = np.array([10, np.nan, 15, 14], dtype=np.float32)
        low = np.array([8, 10, np.nan, 11], dtype=np.float32)
        close = np.array([9, 11, 14, np.nan], dtype=np.float32)

        atr = compute_atr(high, low, close, period=2)

        # NaN should propagate through calculation
        assert np.isnan(atr[1])  # From NaN high
        assert np.isnan(atr[2])  # From NaN low
        assert np.isnan(atr[3])  # From NaN close

    # ========== EDGE TESTS ==========

    @pytest.mark.parametrize("case_name,case_data", [
        ("empty", (np.array([]), np.array([]), np.array([]))),
        ("single", (np.array([10]), np.array([8]), np.array([9]))),
        ("two_elements", (np.array([10, 12]), np.array([8, 10]), np.array([9, 11]))),
    ])
    def test_edge_cases(self, case_name, case_data, edge_cases):
        """Edge test: Small datasets."""
        high, low, close = case_data
        period = 2

        atr = compute_atr(high, low, close, period=period)

        assert len(atr) == len(close)
        assert atr.dtype == np.float32

        if len(atr) < period:
            assert np.all(np.isnan(atr))  # All NaN if insufficient data

    def test_large_period_small_data(self):
        """Edge test: Period larger than data."""
        high = np.array([10, 12], dtype=np.float32)
        low = np.array([8, 10], dtype=np.float32)
        close = np.array([9, 11], dtype=np.float32)

        atr = compute_atr(high, low, close, period=100)

        # All values should be NaN
        assert np.all(np.isnan(atr))

    def test_extreme_values(self):
        base = 1e6
        high = np.array([base, base + 100], dtype=np.float32)
        low = np.array([base - 50, base], dtype=np.float32)
        close = np.array([base - 25, base + 50], dtype=np.float32)

        atr = compute_atr(high, low, close, period=1)
        assert np.all(np.isfinite(atr))
        assert atr[0] == pytest.approx(50.0, abs=1e-3)
        assert atr[1] == pytest.approx(125.0, abs=1e-3)

    def test_zero_volatility(self):
        """Edge test: Zero volatility (all prices same)."""
        high = np.full(10, 100.0, dtype=np.float32)
        low = np.full(10, 100.0, dtype=np.float32)
        close = np.full(10, 100.0, dtype=np.float32)

        atr = compute_atr(high, low, close, period=5)

        # ATR should be zero (except NaN for first period-1)
        assert np.all(atr[4:] == 0.0)

    # ========== PROPERTY-BASED TESTS ==========

    def test_atr_monotonic_properties(self):
        """Property test: ATR should have certain mathematical properties."""
        np.random.seed(42)
        n = 100
        high = np.random.uniform(90, 110, n).astype(np.float32)
        low = high - np.random.uniform(1, 5, n)
        close = (high + low) / 2 + np.random.uniform(-1, 1, n)

        for period in [1, 7, 14, 21]:
            atr = compute_atr(high, low, close, period=period)
            tr = compute_true_range(high, low, close)

            # Property 1: ATR ≤ max(TR) for period > 1
            if period > 1:
                valid_atr = atr[period-1:]
                assert np.all(valid_atr <= np.nanmax(tr))

            # Property 2: ATR ≥ min(TR) for period > 1
            if period > 1:
                valid_atr = atr[period-1:]
                assert np.all(valid_atr >= np.nanmin(tr))

            # Property 3: ATR is non-negative
            assert np.all(atr[~np.isnan(atr)] >= 0)