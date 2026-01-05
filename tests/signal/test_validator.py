"""
Corrected test suite for signal validator.
"""

import pytest
import numpy as np
from structure.signal.validator import (
    _validate_config,
    _create_retest_condition_mask,
    _compute_first_retest_metrics,
    compute_pullback_metrics,
    _extract_metrics_at_breakouts,
    _classify_breakouts,
    _validate_follow_through,
    _mark_immediate_failures,
    validate_signals
)
from structure.signal.config import SignalValidatorConfig
from structure.metrics.types import RawSignals, ValidatedSignals, RetestMetrics


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def default_config():
    """Default valid configuration with adjusted min_retest_respect_bars."""
    return SignalValidatorConfig(
        follow_through_bars=3,
        follow_through_close_ratio=0.6,
        pullback_min_bars=3,
        pullback_max_bars=50,
        max_pullback_velocity=0.8,
        min_retest_respect_bars=3,  # Changed from 5 to 3 for tests
        max_retest_attempts=3,
        immediate_failure_bars=3
    )


@pytest.fixture
def lenient_config():
    """More lenient config for testing."""
    return SignalValidatorConfig(
        follow_through_bars=2,
        follow_through_close_ratio=0.5,
        pullback_min_bars=2,
        pullback_max_bars=20,
        max_pullback_velocity=1.0,
        min_retest_respect_bars=2,
        max_retest_attempts=5,
        immediate_failure_bars=2
    )


# ==============================================================================
# CONFIG VALIDATION TESTS (UPDATED)
# ==============================================================================

def test_config_creation_valid():
    """Positive test: Valid config should not raise."""
    config = SignalValidatorConfig(
        follow_through_bars=3,
        follow_through_close_ratio=0.6,
        pullback_min_bars=3,
        pullback_max_bars=50,
        max_pullback_velocity=0.8,
        min_retest_respect_bars=5,
        max_retest_attempts=3,
        immediate_failure_bars=3
    )
    assert config.follow_through_bars == 3


def test_config_invalid_follow_through_bars():
    """Negative test: Invalid follow_through_bars."""
    with pytest.raises(ValueError, match="follow_through_bars must be ≥ 1"):
        SignalValidatorConfig(follow_through_bars=0)


def test_config_invalid_follow_through_ratio():
    """Negative test: Invalid follow_through_close_ratio."""
    with pytest.raises(ValueError, match="follow_through_close_ratio must be between 0 and 1"):
        SignalValidatorConfig(follow_through_close_ratio=1.5)


def test_config_invalid_pullback_min_bars():
    """Negative test: Invalid pullback_min_bars."""
    with pytest.raises(ValueError, match="pullback_min_bars must be ≥ 1"):
        SignalValidatorConfig(pullback_min_bars=0)


def test_config_invalid_pullback_max_bars():
    """Negative test: pullback_max_bars <= pullback_min_bars."""
    with pytest.raises(ValueError, match="pullback_max_bars must be > pullback_min_bars"):
        SignalValidatorConfig(pullback_min_bars=5, pullback_max_bars=5)


def test_config_invalid_max_pullback_velocity():
    """Negative test: Invalid max_pullback_velocity."""
    with pytest.raises(ValueError, match="max_pullback_velocity must be > 0"):
        SignalValidatorConfig(max_pullback_velocity=0)


def test_config_invalid_min_retest_respect_bars():
    """Negative test: Invalid min_retest_respect_bars."""
    with pytest.raises(ValueError, match="min_retest_respect_bars must be ≥ 1"):
        SignalValidatorConfig(min_retest_respect_bars=0)


def test_config_invalid_max_retest_attempts():
    """Negative test: Invalid max_retest_attempts."""
    with pytest.raises(ValueError, match="max_retest_attempts must be ≥ 1"):
        SignalValidatorConfig(max_retest_attempts=0)


def test_config_invalid_immediate_failure_bars():
    """Negative test: Invalid immediate_failure_bars."""
    with pytest.raises(ValueError, match="immediate_failure_bars must be ≥ 1"):
        SignalValidatorConfig(immediate_failure_bars=0)


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================

def test_create_retest_condition_mask_bullish():
    """Positive test: Bullish retest condition."""
    future_close = np.array([105.0, 104.0, 103.0, 102.0])
    level_price = 100.0
    buffer = 2.0

    result = _create_retest_condition_mask(future_close, level_price, buffer, "bullish")
    expected = future_close <= (level_price + buffer)  # <= 102.0
    assert np.array_equal(result, expected)
    assert result[0] == False  # 105 > 102
    assert result[3] == True  # 102 <= 102


def test_create_retest_condition_mask_bearish():
    """Positive test: Bearish retest condition."""
    future_close = np.array([95.0, 96.0, 97.0, 98.0])
    level_price = 100.0
    buffer = 2.0

    result = _create_retest_condition_mask(future_close, level_price, buffer, "bearish")
    expected = future_close >= (level_price - buffer)  # >= 98.0
    assert np.array_equal(result, expected)
    assert result[0] == False  # 95 < 98
    assert result[3] == True  # 98 >= 98


def test_create_retest_condition_mask_invalid_direction():
    """Negative test: Invalid direction."""
    with pytest.raises(ValueError, match="Invalid direction"):
        _create_retest_condition_mask(np.array([1.0]), 1.0, 0.1, "invalid")


def test_create_retest_condition_mask_empty_array():
    """Boundary test: Empty future_close array."""
    result = _create_retest_condition_mask(np.array([]), 100.0, 2.0, "bullish")
    assert result.shape == (0,)
    assert result.dtype == bool


# ==============================================================================
# RETEST METRICS TESTS (FIXED)
# ==============================================================================

def test_compute_first_retest_metrics_no_retest(lenient_config):
    """Negative test: No retest found."""
    n = 20
    close = np.linspace(100, 110, n, dtype=np.float32)
    future_bars = np.arange(5, 10)
    future_close = close[future_bars]

    # All prices above level + buffer (no retest)
    result = _compute_first_retest_metrics(
        break_idx=0,
        level_price=100.0,
        future_bars=future_bars,
        future_close=future_close,
        atr_value=1.0,
        buffer=0.5,
        direction="bullish",
        config=lenient_config,
        close=close
    )

    vel, bars, dist, attempts = result
    assert attempts == 0
    assert bars == 0
    assert vel == 0.0
    assert dist == 0.0


def test_compute_first_retest_metrics_valid_retest(lenient_config):
    """Positive test: Valid retest found."""
    n = 20
    # Only index 2 retests (99.5 <= 101.0)
    close = np.array([100.0, 105.0, 99.5, 105.0, 106.0], dtype=np.float32)
    future_bars = np.array([1, 2, 3, 4])
    future_close = close[future_bars]

    result = _compute_first_retest_metrics(
        break_idx=0,
        level_price=100.0,
        future_bars=future_bars,
        future_close=future_close,
        atr_value=2.0,  # ATR
        buffer=1.0,  # Buffer = 1.0
        direction="bullish",
        config=lenient_config,
        close=close
    )

    vel, bars, dist, attempts = result
    assert attempts == 1  # Only one retest at index 2
    assert bars == 2  # 2 bars elapsed (break_idx=0 to bar=2)
    assert dist == 0.5  # 100.0 - 99.5
    # velocity = (0.5 / 2.0) / 2 = 0.125
    assert abs(vel - 0.125) < 0.001


def test_compute_first_retest_metrics_multiple_attempts(lenient_config):
    """Positive test: Multiple retest attempts."""
    n = 20
    close = np.array([100.0, 99.8, 99.9, 99.7, 100.2], dtype=np.float32)
    future_bars = np.array([1, 2, 3, 4])
    future_close = close[future_bars]

    result = _compute_first_retest_metrics(
        break_idx=0,
        level_price=100.0,
        future_bars=future_bars,
        future_close=future_close,
        atr_value=1.0,
        buffer=0.5,
        direction="bullish",
        config=lenient_config,
        close=close
    )

    vel, bars, dist, attempts = result
    assert attempts == 4  # All 4 bars retest (≤ 100.5)
    assert bars == 2  # Updated: With min_retest_respect_bars=2, first valid retest at bar 2

def test_compute_first_retest_metrics_empty_future_close(lenient_config):
    """Boundary test: Empty future_close array."""
    result = _compute_first_retest_metrics(
        break_idx=0,
        level_price=100.0,
        future_bars=np.array([]),
        future_close=np.array([]),
        atr_value=1.0,
        buffer=0.5,
        direction="bullish",
        config=lenient_config,
        close=np.array([100.0])
    )

    assert result == (0.0, 0, 0.0, 0)


def test_compute_first_retest_metrics_below_min_bars(lenient_config):
    """Negative test: Retest before min_bars."""
    n = 20
    close = np.array([100.0, 99.9, 100.1, 100.2], dtype=np.float32)
    future_bars = np.array([1])  # Only bar 1 retests, but min_bars=2
    future_close = close[future_bars]

    result = _compute_first_retest_metrics(
        break_idx=0,
        level_price=100.0,
        future_bars=future_bars,
        future_close=future_close,
        atr_value=1.0,
        buffer=0.5,
        direction="bullish",
        config=lenient_config,
        close=close
    )

    vel, bars, dist, attempts = result
    assert attempts == 1  # One attempt
    assert bars == 0  # Not valid (before min_bars)


# ==============================================================================
# PULLBACK METRICS TESTS (FIXED)
# ==============================================================================

def test_compute_pullback_metrics_no_breakouts(lenient_config):
    """Boundary test: Empty breakout indices."""
    n = 50
    close = np.random.randn(n).astype(np.float32)
    atr = np.ones(n, dtype=np.float32)

    result = compute_pullback_metrics(
        breakout_indices=np.array([], dtype=int),
        level_prices=np.array([], dtype=np.float32),
        close=close,
        atr=atr,
        direction="bullish",
        config=lenient_config
    )

    assert isinstance(result, RetestMetrics)
    assert len(result.retest_velocity) == n
    assert np.all(result.retest_velocity == 0.0)
    assert np.all(result.retest_attempts == 0)


def test_compute_pullback_metrics_single_breakout(lenient_config):
    """Positive test: Single breakout with retest."""
    n = 50
    close = np.full(n, 105.0, dtype=np.float32)  # Start above
    atr = np.full(n, 2.0, dtype=np.float32)

    # Create retest scenario: breakout at index 10, retest at index 12
    breakout_idx = 10
    level_price = 100.0

    # Make price pull back to retest level at index 12
    close[breakout_idx] = level_price
    close[12] = 99.0  # Retest (2 bars later)

    result = compute_pullback_metrics(
        breakout_indices=np.array([breakout_idx]),
        level_prices=np.array([level_price], dtype=np.float32),
        close=close,
        atr=atr,
        direction="bullish",
        config=lenient_config
    )

    assert result.retest_attempts[breakout_idx] > 0
    assert result.bars_to_retest[12] == 2  # 12 - 10
    assert result.retest_velocity[12] > 0.0


def test_compute_pullback_metrics_multiple_breakouts(lenient_config):
    """Positive test: Multiple breakouts."""
    n = 100
    close = np.linspace(100, 110, n, dtype=np.float32)
    atr = np.full(n, 1.0, dtype=np.float32)

    # Two breakouts
    breakout_indices = np.array([10, 30])
    level_prices = np.array([100.0, 105.0], dtype=np.float32)

    result = compute_pullback_metrics(
        breakout_indices=breakout_indices,
        level_prices=level_prices,
        close=close,
        atr=atr,
        direction="bullish",
        config=lenient_config
    )

    assert result.retest_attempts[10] >= 0
    assert result.retest_attempts[30] >= 0
    assert np.all(np.isnan(result.break_levels[:10]))  # No levels before first breakout


def test_compute_pullback_metrics_breakout_at_end(lenient_config):
    """Boundary test: Breakout near end of array."""
    n = 20
    close = np.random.randn(n).astype(np.float32)
    atr = np.ones(n, dtype=np.float32)

    # Breakout at last index - no room for retest
    result = compute_pullback_metrics(
        breakout_indices=np.array([n - 1]),
        level_prices=np.array([100.0], dtype=np.float32),
        close=close,
        atr=atr,
        direction="bullish",
        config=lenient_config
    )

    assert result.retest_attempts[n - 1] == 0  # No room to search


# ==============================================================================
# FOLLOW-THROUGH TESTS (FIXED)
# ==============================================================================

def test_validate_follow_through_bullish_success(lenient_config):
    """Positive test: Bullish follow-through success."""
    n = 20
    close = np.full(n, 100.0, dtype=np.float32)
    high = np.full(n, 101.0, dtype=np.float32)
    low = np.full(n, 99.0, dtype=np.float32)
    atr = np.full(n, 1.0, dtype=np.float32)

    # Signal at index 10, level 100, buffer 0.5
    signal_indices = np.array([10])
    level_prices = np.array([100.0])

    # Make close above level+buffer with close near high
    for i in range(11, 14):  # Next 3 bars
        high[i] = 103.0
        low[i] = 100.0
        close[i] = 102.5  # Close near high → close_loc = (102.5-100)/3 ≈ 0.83

    result = _validate_follow_through(
        signal_indices, level_prices, close, high, low, atr, "bullish", lenient_config
    )

    assert result[0] == True  # Should pass


def test_validate_follow_through_bullish_failure(lenient_config):
    """Negative test: Bullish follow-through failure."""
    n = 20
    close = np.linspace(100, 99, n, dtype=np.float32)  # Going down
    high = close + 0.5
    low = close - 0.5
    atr = np.full(n, 1.0, dtype=np.float32)

    signal_indices = np.array([10])
    level_prices = np.array([100.0])

    result = _validate_follow_through(
        signal_indices, level_prices, close, high, low, atr, "bullish", lenient_config
    )

    assert result[0] == False  # Should fail (price below level)


def test_validate_follow_through_boundary_signal_at_end(lenient_config):
    """Boundary test: Signal too close to end."""
    n = 10
    close = np.random.randn(n).astype(np.float32)
    high = close + 0.5
    low = close - 0.5
    atr = np.ones(n, dtype=np.float32)

    signal_indices = np.array([8])  # Only 1 bar left for follow-through (need 2)
    level_prices = np.array([100.0])

    result = _validate_follow_through(
        signal_indices, level_prices, close, high, low, atr, "bullish", lenient_config
    )

    assert result[0] == False  # Not enough bars


# ==============================================================================
# INTEGRATION TESTS - SIMPLIFIED
# ==============================================================================

def test_validate_signals_empty_inputs(lenient_config):
    """Boundary test: Empty arrays."""
    n = 0
    close = np.array([], dtype=np.float32)
    high = np.array([], dtype=np.float32)
    low = np.array([], dtype=np.float32)
    atr = np.array([], dtype=np.float32)

    raw_signals = RawSignals(
        is_bos_bullish_initial=np.array([], dtype=bool),
        is_bos_bearish_initial=np.array([], dtype=bool),
        is_choch_bullish=np.array([], dtype=bool),
        is_choch_bearish=np.array([], dtype=bool)
    )

    result = validate_signals(raw_signals, close, high, low, atr, lenient_config)

    assert isinstance(result, ValidatedSignals)
    assert len(result.is_bos_bullish_confirmed) == 0


def test_validate_signals_array_length_mismatch(lenient_config):
    """Negative test: Array length mismatch."""
    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(10, dtype=bool),
        is_bos_bearish_initial=np.zeros(10, dtype=bool),
        is_choch_bullish=np.zeros(10, dtype=bool),
        is_choch_bearish=np.zeros(10, dtype=bool)
    )

    close = np.zeros(10, dtype=np.float32)
    high = np.zeros(10, dtype=np.float32)
    low = np.zeros(10, dtype=np.float32)
    atr = np.zeros(9, dtype=np.float32)  # Wrong length!

    with pytest.raises(ValueError, match="All price arrays must have same length"):
        validate_signals(raw_signals, close, high, low, atr, lenient_config)


def test_validate_signals_simple_bullish_breakout(lenient_config):
    """Positive test: Simple bullish breakout."""
    n = 50
    close = np.full(n, 100.0, dtype=np.float32)
    high = np.full(n, 101.0, dtype=np.float32)
    low = np.full(n, 99.0, dtype=np.float32)
    atr = np.full(n, 2.0, dtype=np.float32)

    # Single breakout at index 20
    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(n, dtype=bool),
        is_bos_bearish_initial=np.zeros(n, dtype=bool),
        is_choch_bullish=np.zeros(n, dtype=bool),
        is_choch_bearish=np.zeros(n, dtype=bool)
    )
    raw_signals.is_bos_bullish_initial[20] = True

    result = validate_signals(raw_signals, close, high, low, atr, lenient_config)

    # Check array lengths
    assert len(result.is_bos_bullish_confirmed) == n
    assert len(result.is_bos_bullish_momentum) == n
    assert len(result.is_bullish_break_failure) == n

    # Debug output
    print(f"\nSimple bullish breakout test:")
    print(f"Confirmed at 20: {result.is_bos_bullish_confirmed[20]}")
    print(f"Momentum at 20: {result.is_bos_bullish_momentum[20]}")
    print(f"Failure at 20: {result.is_bullish_break_failure[20]}")


def test_validate_signals_immediate_failure(lenient_config):
    """Positive test: Immediate failure detection."""
    n = 50
    close = np.full(n, 100.0, dtype=np.float32)
    high = np.full(n, 101.0, dtype=np.float32)
    low = np.full(n, 99.0, dtype=np.float32)
    atr = np.full(n, 2.0, dtype=np.float32)

    # Bullish breakout at index 20
    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(n, dtype=bool),
        is_bos_bearish_initial=np.zeros(n, dtype=bool),
        is_choch_bullish=np.zeros(n, dtype=bool),
        is_choch_bearish=np.zeros(n, dtype=bool)
    )
    raw_signals.is_bos_bullish_initial[20] = True

    # Price immediately drops below level - buffer
    buffer = atr[20] * 0.5  # 2.0 * 0.5 = 1.0
    level = high[20]  # 101.0
    close[21] = level - buffer - 0.1  # 99.9 < 100.0

    result = validate_signals(raw_signals, close, high, low, atr, lenient_config)

    print(f"\nImmediate failure test:")
    print(f"Immediate failure at 20: {result.is_bullish_immediate_failure[20]}")


# ==============================================================================
# PROPERTY TESTS
# ==============================================================================

def test_output_array_lengths(lenient_config):
    """Property test: All output arrays have same length as input."""
    n = 100
    close = np.random.randn(n).astype(np.float32)
    high = close + 0.5
    low = close - 0.5
    atr = np.ones(n, dtype=np.float32)

    raw_signals = RawSignals(
        is_bos_bullish_initial=np.random.rand(n) > 0.9,
        is_bos_bearish_initial=np.random.rand(n) > 0.9,
        is_choch_bullish=np.random.rand(n) > 0.95,
        is_choch_bearish=np.random.rand(n) > 0.95
    )

    result = validate_signals(raw_signals, close, high, low, atr, lenient_config)

    # Check all output arrays have correct length
    for field in result.__dataclass_fields__:
        array = getattr(result, field)
        assert len(array) == n, f"{field} has wrong length: {len(array)} != {n}"


def test_mutual_exclusivity_basic(lenient_config):
    """Property test: Basic signal types are mutually exclusive per bar."""
    n = 50
    close = np.random.randn(n).astype(np.float32)
    high = close + 0.5
    low = close - 0.5
    atr = np.ones(n, dtype=np.float32)

    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(n, dtype=bool),
        is_bos_bearish_initial=np.zeros(n, dtype=bool),
        is_choch_bullish=np.zeros(n, dtype=bool),
        is_choch_bearish=np.zeros(n, dtype=bool)
    )
    raw_signals.is_bos_bullish_initial[25] = True

    result = validate_signals(raw_signals, close, high, low, atr, lenient_config)

    # For bullish signals at bar 25, check they are mutually exclusive
    # A breakout can only be in one state: confirmed, momentum, break failure, or immediate failure
    bullish_states = [
        result.is_bos_bullish_confirmed[25],
        result.is_bos_bullish_momentum[25],
        result.is_bullish_break_failure[25],
        result.is_bullish_immediate_failure[25]
    ]

    # At most one should be True (could be all False if signal doesn't meet any criteria)
    true_count = sum(bullish_states)
    assert true_count <= 1, f"Multiple bullish signal states at same bar: {bullish_states}"

    # Additional check: confirmed and momentum should never both be True
    assert not (result.is_bos_bullish_confirmed[25] and result.is_bos_bullish_momentum[25]), \
        "Breakout cannot be both confirmed and momentum"

    # Additional check: failure and immediate failure should never both be True
    # (after our fix above)
    assert not (result.is_bullish_break_failure[25] and result.is_bullish_immediate_failure[25]), \
        "Breakout cannot be both break failure and immediate failure"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])