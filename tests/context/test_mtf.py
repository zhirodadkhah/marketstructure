# tests/context/test_mtf.py
"""
Comprehensive test suite for MTF context and confluence scoring.
"""
import numpy as np
import pytest
from structure.context.mtf import (
    calculate_mtf_confluence_score,
    resample_and_align_context,
    interpolate_htf_to_ltf
)
from structure.context.config import MTFConfig


# ==============================================================================
# Tests for calculate_mtf_confluence_score
# ==============================================================================

def test_calculate_mtf_confluence_score_perfect_alignment():
    """Test perfect confluence (trend and regime both aligned)."""
    n = 10
    ltf_trend = np.ones(n, dtype=np.int8)  # All uptrend
    htf_trend = np.ones(n, dtype=np.int8)  # All uptrend

    # String regime labels
    ltf_regime = np.array(['strong_trend'] * n, dtype='<U20')
    htf_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig()
    score = calculate_mtf_confluence_score(
        ltf_trend, ltf_regime, htf_trend, htf_regime, config
    )

    # Perfect alignment should give score of 1.0
    assert score.shape == (n,)
    assert score.dtype == np.float32
    assert np.all(score == pytest.approx(1.0, abs=1e-6))


def test_calculate_mtf_confluence_score_trend_divergence():
    """Test trend divergence (regime aligned but trend opposite)."""
    n = 10
    ltf_trend = np.ones(n, dtype=np.int8)  # Uptrend
    htf_trend = -np.ones(n, dtype=np.int8)  # Downtrend

    ltf_regime = np.array(['strong_trend'] * n, dtype='<U20')
    htf_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig()
    score = calculate_mtf_confluence_score(
        ltf_trend, ltf_regime, htf_trend, htf_regime, config
    )

    # Trend not aligned, but regime aligned
    # Score = (0.0 + 1.0) / 2 = 0.5
    assert np.all(score == pytest.approx(0.5, abs=1e-6))


def test_calculate_mtf_confluence_score_regime_divergence():
    """Test regime divergence (trend aligned but regime different)."""
    n = 10
    ltf_trend = np.ones(n, dtype=np.int8)  # Uptrend
    htf_trend = np.ones(n, dtype=np.int8)  # Uptrend

    ltf_regime = np.array(['strong_trend'] * n, dtype='<U20')
    htf_regime = np.array(['chop'] * n, dtype='<U20')  # Different regime

    config = MTFConfig()
    score = calculate_mtf_confluence_score(
        ltf_trend, ltf_regime, htf_trend, htf_regime, config
    )

    # Trend aligned, regime not aligned
    # Score = (1.0 + 0.0) / 2 = 0.5
    assert np.all(score == pytest.approx(0.5, abs=1e-6))


def test_calculate_mtf_confluence_score_neutral_trend():
    """Test with neutral trend (should not count as aligned)."""
    n = 10
    ltf_trend = np.zeros(n, dtype=np.int8)  # Neutral
    htf_trend = np.zeros(n, dtype=np.int8)  # Neutral

    ltf_regime = np.array(['strong_trend'] * n, dtype='<U20')
    htf_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig()
    score = calculate_mtf_confluence_score(
        ltf_trend, ltf_regime, htf_trend, htf_regime, config
    )

    # Both neutral - trend alignment = 0.0 (neutral ignored)
    # Regime aligned = 1.0
    # Score = (0.0 + 1.0) / 2 = 0.5
    assert np.all(score == pytest.approx(0.5, abs=1e-6))


def test_calculate_mtf_confluence_score_int_regime():
    """Test with integer regime encoding."""
    n = 10
    ltf_trend = np.ones(n, dtype=np.int8)
    htf_trend = np.ones(n, dtype=np.int8)

    # Integer regime: 1=strong_trend, 2=weak_trend, 0=neutral, -1=chop
    ltf_regime = np.ones(n, dtype=np.int8)  # strong_trend
    htf_regime = np.full(n, 2, dtype=np.int8)  # weak_trend

    config = MTFConfig()
    score = calculate_mtf_confluence_score(
        ltf_trend, ltf_regime, htf_trend, htf_regime, config
    )

    # Both trending (regime > 0), so regime aligned
    # Trend aligned
    # Score = (1.0 + 1.0) / 2 = 1.0
    assert np.all(score == pytest.approx(1.0, abs=1e-6))


def test_calculate_mtf_confluence_score_empty_input():
    """Test with empty arrays."""
    ltf_trend = np.array([], dtype=np.int8)
    ltf_regime = np.array([], dtype='<U20')
    htf_trend = np.array([], dtype=np.int8)
    htf_regime = np.array([], dtype='<U20')

    config = MTFConfig()
    score = calculate_mtf_confluence_score(
        ltf_trend, ltf_regime, htf_trend, htf_regime, config
    )

    assert len(score) == 0
    assert score.dtype == np.float32


def test_calculate_mtf_confluence_score_mixed_regime():
    """Test with mixed trending/non-trending regimes."""
    n = 4
    ltf_trend = np.array([1, 1, -1, -1], dtype=np.int8)
    htf_trend = np.array([1, -1, 1, -1], dtype=np.int8)

    ltf_regime = np.array(['strong_trend', 'chop', 'weak_trend', 'chop'], dtype='<U20')
    htf_regime = np.array(['strong_trend', 'strong_trend', 'chop', 'chop'], dtype='<U20')

    config = MTFConfig()
    score = calculate_mtf_confluence_score(
        ltf_trend, ltf_regime, htf_trend, htf_regime, config
    )

    # Expected scores:
    # 0: trend aligned (1==1), regime aligned (strong_trend==strong_trend) -> 1.0
    # 1: trend not aligned (1!=-1), regime not aligned (chop!=strong_trend) -> 0.0
    # 2: trend not aligned (-1!=1), regime not aligned (weak_trend!=chop) -> 0.0
    # 3: trend aligned (-1==-1), regime aligned (chop==chop) -> 1.0
    expected = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert np.allclose(score, expected, atol=1e-6)


# ==============================================================================
# Tests for resample_and_align_context
# ==============================================================================

def test_resample_and_align_context_basic():
    """Basic resampling test."""
    n = 24  # Exactly 1 HTF bar with size=24
    timestamps = np.arange(n, dtype=np.float64) * 3600  # 1-hour intervals
    open_ = np.ones(n, dtype=np.float32) * 100.0
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + 0.5
    trend_state = np.ones(n, dtype=np.int8)
    market_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig(htf_bar_size=24)

    result = resample_and_align_context(
        timestamps, open_, high, low, close,
        trend_state, market_regime, config
    )

    htf_timestamps, htf_open, htf_high, htf_low, htf_close, htf_trend, htf_regime = result

    # Should have 1 HTF bar
    assert len(htf_timestamps) == 1
    assert len(htf_open) == 1
    assert len(htf_trend) == 1
    assert len(htf_regime) == 1

    # Check OHLC values
    assert htf_open[0] == pytest.approx(100.0)
    assert htf_high[0] == pytest.approx(101.0)
    assert htf_low[0] == pytest.approx(99.0)
    assert htf_close[0] == pytest.approx(100.5)

    # Check timestamp (should be last timestamp in window)
    assert htf_timestamps[0] == timestamps[-1]

    # Check trend and regime
    assert htf_trend[0] == 1
    assert htf_regime[0] == 'strong_trend'


def test_resample_and_align_context_multiple_bars():
    """Test resampling with multiple HTF bars."""
    n = 48  # 2 HTF bars with size=24
    timestamps = np.arange(n, dtype=np.float64) * 3600
    open_ = np.ones(n, dtype=np.float32) * 100.0
    high = open_ + np.arange(n) * 0.1  # Increasing highs
    low = open_ - 1.0
    close = open_ + 0.5
    trend_state = np.ones(n, dtype=np.int8)
    market_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig(htf_bar_size=24)

    result = resample_and_align_context(
        timestamps, open_, high, low, close,
        trend_state, market_regime, config
    )

    htf_timestamps, htf_open, htf_high, htf_low, htf_close, htf_trend, htf_regime = result

    # Should have 2 HTF bars
    assert len(htf_timestamps) == 2
    assert len(htf_open) == 2

    # Check OHLC values for first HTF bar
    assert htf_open[0] == pytest.approx(100.0)
    assert htf_high[0] == pytest.approx(100.0 + 23 * 0.1)  # Max of first 24 highs
    assert htf_close[0] == pytest.approx(100.5)

    # Check timestamp alignment
    assert htf_timestamps[0] == timestamps[23]  # Last of first 24
    assert htf_timestamps[1] == timestamps[47]  # Last of second 24


def test_resample_and_align_context_partial_window():
    """Test when data doesn't fill complete HTF bars."""
    n = 30  # 1 full HTF bar + 6 extra bars
    timestamps = np.arange(n, dtype=np.float64) * 3600
    open_ = np.ones(n, dtype=np.float32) * 100.0
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + 0.5
    trend_state = np.ones(n, dtype=np.int8)
    market_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig(htf_bar_size=24)

    result = resample_and_align_context(
        timestamps, open_, high, low, close,
        trend_state, market_regime, config
    )

    htf_timestamps, htf_open, htf_high, htf_low, htf_close, htf_trend, htf_regime = result

    # Should have 1 HTF bar (30 // 24 = 1)
    assert len(htf_timestamps) == 1
    # Trimmed data should be 24 bars (30 // 24 * 24 = 24)
    # Check timestamp is the last of the 24 used bars
    assert htf_timestamps[0] == timestamps[23]


def test_resample_and_align_context_small_data():
    """Test with data smaller than HTF bar size."""
    n = 12  # Smaller than HTF bar size of 24
    timestamps = np.arange(n, dtype=np.float64) * 3600
    open_ = np.ones(n, dtype=np.float32) * 100.0
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + 0.5
    trend_state = np.ones(n, dtype=np.int8)
    market_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig(htf_bar_size=24)

    result = resample_and_align_context(
        timestamps, open_, high, low, close,
        trend_state, market_regime, config
    )

    htf_timestamps, htf_open, htf_high, htf_low, htf_close, htf_trend, htf_regime = result

    # Should return empty arrays
    assert len(htf_timestamps) == 0
    assert len(htf_open) == 0
    assert len(htf_trend) == 0
    assert len(htf_regime) == 0


def test_resample_and_align_context_invalid_bar_size():
    """Test with invalid HTF bar size."""
    n = 10
    timestamps = np.arange(n, dtype=np.float64) * 3600
    open_ = np.ones(n, dtype=np.float32) * 100.0
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + 0.5
    trend_state = np.ones(n, dtype=np.int8)
    market_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig(htf_bar_size=0)  # Invalid

    with pytest.raises(ValueError, match="htf_bar_size must be ≥ 1"):
        resample_and_align_context(
            timestamps, open_, high, low, close,
            trend_state, market_regime, config
        )


def test_resample_and_align_context_changing_regime():
    """Test with regime changes within HTF window."""
    n = 24
    timestamps = np.arange(n, dtype=np.float64) * 3600
    open_ = np.ones(n, dtype=np.float32) * 100.0
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + 0.5

    # First half uptrend, second half chop
    trend_state = np.concatenate([np.ones(12, dtype=np.int8), -np.ones(12, dtype=np.int8)])
    market_regime = np.concatenate([
        np.array(['strong_trend'] * 12, dtype='<U20'),
        np.array(['chop'] * 12, dtype='<U20')
    ])

    config = MTFConfig(htf_bar_size=24)

    result = resample_and_align_context(
        timestamps, open_, high, low, close,
        trend_state, market_regime, config
    )

    htf_timestamps, htf_open, htf_high, htf_low, htf_close, htf_trend, htf_regime = result

    # Should have 1 HTF bar
    assert len(htf_timestamps) == 1

    # Should use last value for trend and regime
    assert htf_trend[0] == -1  # Last trend is downtrend
    assert htf_regime[0] == 'chop'  # Last regime is chop


# ==============================================================================
# Tests for interpolate_htf_to_ltf
# ==============================================================================

def test_interpolate_htf_to_ltf_basic():
    """Basic forward-fill test."""
    # LTF timestamps: 1, 2, 3, 4, 5
    # HTF timestamps: 2, 4 (aligned at even timestamps)
    ltf_timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    htf_timestamps = np.array([2.0, 4.0], dtype=np.float64)
    htf_trend = np.array([1, -1], dtype=np.int8)  # Uptrend, then downtrend
    htf_regime = np.array(['strong_trend', 'chop'], dtype='<U20')

    ltf_trend, ltf_regime = interpolate_htf_to_ltf(
        ltf_timestamps, htf_timestamps, htf_trend, htf_regime
    )

    # Expected:
    # Timestamp 1: no HTF bar yet, should get neutral
    # Timestamp 2: gets first HTF bar (uptrend, strong_trend)
    # Timestamp 3: forward-fills from timestamp 2
    # Timestamp 4: gets second HTF bar (downtrend, chop)
    # Timestamp 5: forward-fills from timestamp 4

    expected_trend = np.array([0, 1, 1, -1, -1], dtype=np.int8)
    expected_regime = np.array(['neutral', 'strong_trend', 'strong_trend', 'chop', 'chop'], dtype='<U20')

    assert np.array_equal(ltf_trend, expected_trend)
    assert np.array_equal(ltf_regime, expected_regime)


def test_interpolate_htf_to_ltf_empty_input():
    """Test with empty arrays."""
    ltf_timestamps = np.array([], dtype=np.float64)
    htf_timestamps = np.array([], dtype=np.float64)
    htf_trend = np.array([], dtype=np.int8)
    htf_regime = np.array([], dtype='<U20')

    ltf_trend, ltf_regime = interpolate_htf_to_ltf(
        ltf_timestamps, htf_timestamps, htf_trend, htf_regime
    )

    assert len(ltf_trend) == 0
    assert len(ltf_regime) == 0
    assert ltf_trend.dtype == np.int8
    assert ltf_regime.dtype == htf_regime.dtype


def test_interpolate_htf_to_ltf_htf_before_ltf():
    """Test when first HTF timestamp is after first LTF timestamp."""
    # LTF starts at 1, HTF starts at 3
    ltf_timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    htf_timestamps = np.array([3.0, 5.0], dtype=np.float64)
    htf_trend = np.array([1, -1], dtype=np.int8)
    htf_regime = np.array(['strong_trend', 'chop'], dtype='<U20')

    ltf_trend, ltf_regime = interpolate_htf_to_ltf(
        ltf_timestamps, htf_timestamps, htf_trend, htf_regime
    )

    # First two LTF timestamps have no HTF bar, should be neutral
    expected_trend = np.array([0, 0, 1, 1, -1], dtype=np.int8)
    expected_regime = np.array(['neutral', 'neutral', 'strong_trend', 'strong_trend', 'chop'], dtype='<U20')

    assert np.array_equal(ltf_trend, expected_trend)
    assert np.array_equal(ltf_regime, expected_regime)


def test_interpolate_htf_to_ltf_single_htf():
    """Test with single HTF bar covering all LTF."""
    ltf_timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    htf_timestamps = np.array([5.0], dtype=np.float64)  # Single bar at end
    htf_trend = np.array([1], dtype=np.int8)
    htf_regime = np.array(['strong_trend'], dtype='<U20')

    ltf_trend, ltf_regime = interpolate_htf_to_ltf(
        ltf_timestamps, htf_timestamps, htf_trend, htf_regime
    )

    # All LTF timestamps should get the same HTF values
    expected_trend = np.array([0, 0, 0, 0, 1], dtype=np.int8)
    expected_regime = np.array(['neutral', 'neutral', 'neutral', 'neutral', 'strong_trend'], dtype='<U20')

    assert np.array_equal(ltf_trend, expected_trend)
    assert np.array_equal(ltf_regime, expected_regime)


def test_interpolate_htf_to_ltf_htf_after_all_ltf():
    """Test when all HTF timestamps are after LTF timestamps."""
    ltf_timestamps = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    htf_timestamps = np.array([4.0, 5.0], dtype=np.float64)  # All after LTF
    htf_trend = np.array([1, -1], dtype=np.int8)
    htf_regime = np.array(['strong_trend', 'chop'], dtype='<U20')

    ltf_trend, ltf_regime = interpolate_htf_to_ltf(
        ltf_timestamps, htf_timestamps, htf_trend, htf_regime
    )

    # All LTF timestamps should be neutral (no HTF bar <= LTF timestamp)
    expected_trend = np.array([0, 0, 0], dtype=np.int8)
    expected_regime = np.array(['neutral', 'neutral', 'neutral'], dtype='<U20')

    assert np.array_equal(ltf_trend, expected_trend)
    assert np.array_equal(ltf_regime, expected_regime)


def test_interpolate_htf_to_ltf_int_regime():
    """Test with integer regime encoding."""
    ltf_timestamps = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    htf_timestamps = np.array([2.0, 4.0], dtype=np.float64)
    htf_trend = np.array([1, -1], dtype=np.int8)
    htf_regime = np.array([1, -1], dtype=np.int8)  # 1=strong_trend, -1=chop

    ltf_trend, ltf_regime = interpolate_htf_to_ltf(
        ltf_timestamps, htf_timestamps, htf_trend, htf_regime
    )

    # Check that regime dtype is preserved
    assert ltf_regime.dtype == np.int8

    # Expected values
    expected_trend = np.array([0, 1, 1, -1, -1], dtype=np.int8)
    expected_regime = np.array([0, 1, 1, -1, -1], dtype=np.int8)  # neutral=0

    assert np.array_equal(ltf_trend, expected_trend)
    assert np.array_equal(ltf_regime, expected_regime)


# ==============================================================================
# Integration tests
# ==============================================================================

def test_mtf_pipeline_integration():
    """Test complete MTF pipeline."""
    # Create synthetic data
    n = 100
    timestamps = np.arange(n, dtype=np.float64) * 3600  # Hourly data

    # Prices with some trend
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1).astype(np.float32)
    high = close + np.random.rand(n) * 0.5
    low = close - np.random.rand(n) * 0.5
    open_ = np.roll(close, 1)
    open_[0] = 100.0

    # Trend and regime (simplified)
    trend_state = np.ones(n, dtype=np.int8)  # All uptrend
    market_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig(htf_bar_size=24)  # Daily bars from hourly

    # Step 1: Resample to HTF
    htf_context = resample_and_align_context(
        timestamps, open_, high, low, close,
        trend_state, market_regime, config
    )

    htf_timestamps, htf_open, htf_high, htf_low, htf_close, htf_trend, htf_regime = htf_context

    # Should have some HTF bars
    num_htf_expected = n // config.htf_bar_size
    assert len(htf_timestamps) == num_htf_expected

    # Step 2: Interpolate back to LTF
    ltf_aligned_trend, ltf_aligned_regime = interpolate_htf_to_ltf(
        ltf_timestamps=timestamps,
        htf_timestamps=htf_timestamps,
        htf_trend=htf_trend,
        htf_regime=htf_regime
    )

    # Should have same length as LTF
    assert len(ltf_aligned_trend) == n
    assert len(ltf_aligned_regime) == n

    # Step 3: Calculate confluence score
    confluence_score = calculate_mtf_confluence_score(
        ltf_trend=trend_state,
        ltf_regime=market_regime,
        htf_trend=ltf_aligned_trend,
        htf_regime=ltf_aligned_regime,
        config=config
    )

    # Score should be between 0 and 1
    assert len(confluence_score) == n
    assert np.all(confluence_score >= 0.0)
    assert np.all(confluence_score <= 1.0)


def test_mtf_confluence_with_changing_regimes():
    """Test MTF confluence with regime changes."""
    n = 50
    timestamps = np.arange(n, dtype=np.float64) * 3600

    # Create LTF data with regime changes
    ltf_trend = np.array([1] * 25 + [-1] * 25, dtype=np.int8)
    ltf_regime = np.array(['strong_trend'] * 20 + ['chop'] * 30, dtype='<U20')

    # Create HTF data (simulated)
    htf_trend = np.array([1] * 25 + [-1] * 25, dtype=np.int8)
    htf_regime = np.array(['strong_trend'] * 40 + ['chop'] * 10, dtype='<U20')

    config = MTFConfig()

    score = calculate_mtf_confluence_score(
        ltf_trend, ltf_regime, htf_trend, htf_regime, config
    )

    # Check specific cases
    assert len(score) == n

    # First 20: both trending, trend aligned -> score = 1.0
    assert np.all(score[:20] == pytest.approx(1.0, abs=1e-6))

    # Next 5: LTF chop, HTF strong_trend -> regime not aligned
    # Trend aligned (uptrend) -> score = (1.0 + 0.0) / 2 = 0.5
    assert np.all(score[20:25] == pytest.approx(0.5, abs=1e-6))

    # Next 15: LTF chop, HTF strong_trend, trend aligned (downtrend)
    # Score = (1.0 + 0.0) / 2 = 0.5
    # NOT 0.0! The test expectation was wrong.
    assert np.all(score[25:40] == pytest.approx(0.5, abs=1e-6))

    # Last 10: both chop, trend aligned (downtrend) -> score = 1.0
    assert np.all(score[40:] == pytest.approx(1.0, abs=1e-6))

# ==============================================================================
# Performance tests
# ==============================================================================

def test_performance_large_dataset():
    """Performance test with large dataset."""
    n = 10000
    timestamps = np.arange(n, dtype=np.float64) * 3600
    open_ = np.ones(n, dtype=np.float32) * 100.0
    high = open_ + 1.0
    low = open_ - 1.0
    close = open_ + 0.5
    trend_state = np.ones(n, dtype=np.int8)
    market_regime = np.array(['strong_trend'] * n, dtype='<U20')

    config = MTFConfig(htf_bar_size=24)

    import time
    start = time.time()

    # Resample
    htf_context = resample_and_align_context(
        timestamps, open_, high, low, close,
        trend_state, market_regime, config
    )

    # Interpolate
    ltf_aligned_trend, ltf_aligned_regime = interpolate_htf_to_ltf(
        timestamps, htf_context[0], htf_context[5], htf_context[6]
    )

    # Calculate confluence
    confluence_score = calculate_mtf_confluence_score(
        trend_state, market_regime, ltf_aligned_trend, ltf_aligned_regime, config
    )

    elapsed = time.time() - start

    # Should complete quickly
    assert elapsed < 1.0, f"MTF pipeline too slow: {elapsed:.3f}s for {n} elements"

    # Check results
    assert len(confluence_score) == n
    assert np.all(confluence_score >= 0.0)
    assert np.all(confluence_score <= 1.0)


# ==============================================================================
# Error handling tests
# ==============================================================================

def test_mismatched_array_lengths():
    """Test error handling for mismatched array lengths."""
    ltf_trend = np.ones(10, dtype=np.int8)
    ltf_regime = np.array(['strong_trend'] * 10, dtype='<U20')
    htf_trend = np.ones(5, dtype=np.int8)  # Different length
    htf_regime = np.array(['strong_trend'] * 5, dtype='<U20')

    config = MTFConfig()

    # This should raise an error due to broadcasting
    with pytest.raises(ValueError):
        calculate_mtf_confluence_score(
            ltf_trend, ltf_regime, htf_trend, htf_regime, config
        )


def test_invalid_regime_dtype():
    """Test with invalid regime data type."""
    n = 10
    ltf_trend = np.ones(n, dtype=np.int8)
    htf_trend = np.ones(n, dtype=np.int8)

    # Invalid dtype (complex numbers)
    ltf_regime = np.array([1 + 2j] * n, dtype=np.complex128)
    htf_regime = np.array([1 + 2j] * n, dtype=np.complex128)

    config = MTFConfig()

    # Should fail when trying to check regime dtype
    with pytest.raises(Exception):
        calculate_mtf_confluence_score(
            ltf_trend, ltf_regime, htf_trend, htf_regime, config
        )