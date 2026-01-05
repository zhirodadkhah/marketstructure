# tests/signal/test_filter.py
"""
Comprehensive test suite for signal filtering module.

COVERAGE:
- Positive tests: Valid inputs, correct filtering behavior
- Negative tests: Invalid inputs, error handling
- Boundary tests: Edge cases, empty arrays, extreme values
- Property tests: Consistency, invariants, edge cases
"""

import numpy as np
import pytest

from structure.metrics.types import (
    ValidatedSignals,
    FilteredSignals,
    SignalQuality,
    RawSignals
)
from structure.signal.config import SignalFilterConfig
from structure.signal.filter import (
    filter_signals,
    _validate_inputs,
    _filter_bos_by_regime,
    _filter_by_range_compression,
    _filter_by_retest_respect,
    _VALID_REGIMES
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def default_config():
    """Default filtering configuration."""
    return SignalFilterConfig()


@pytest.fixture
def strict_config():
    """Strict filtering configuration."""
    return SignalFilterConfig(
        max_range_compression=0.3,
        max_range_compression_choch=0.5,
        max_range_compression_momentum=0.2,
        min_retest_respect_filter=0.8,
        min_retest_respect_filter_choch=0.6,
        allow_weak_trend_bos=False
    )


@pytest.fixture
def lenient_config():
    """Lenient filtering configuration."""
    return SignalFilterConfig(
        max_range_compression=0.9,
        max_range_compression_choch=1.0,
        max_range_compression_momentum=0.8,
        min_retest_respect_filter=0.2,
        min_retest_respect_filter_choch=0.1,
        allow_weak_trend_bos=True
    )


@pytest.fixture
def sample_data():
    """Create sample data for tests."""
    n = 20
    market_regime = np.array(
        ['strong_trend'] * 5 +
        ['weak_trend'] * 5 +
        ['ranging'] * 5 +
        ['chop'] * 5,
        dtype='U20'
    )

    range_compression = np.array(
        [0.1, 0.3, 0.5, 0.7, 0.9] * 4,  # Various compression levels
        dtype=np.float32
    )

    retest_respect_score = np.array(
        [0.9, 0.7, 0.5, 0.3, 0.1] * 4,  # Various respect scores
        dtype=np.float32
    )

    return market_regime, range_compression, retest_respect_score


@pytest.fixture
def sample_signals():
    """Create sample signals for tests."""
    n = 20
    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([True] * 5 + [False] * 15, dtype=bool),
        is_bos_bearish_confirmed=np.array([False] * 10 + [True] * 5 + [False] * 5, dtype=bool),
        is_bos_bullish_momentum=np.array([False] * 8 + [True] * 4 + [False] * 8, dtype=bool),
        is_bos_bearish_momentum=np.array([False] * 12 + [True] * 3 + [False] * 5, dtype=bool),
        is_bullish_break_failure=np.array([False] * 15 + [True] + [False] * 4, dtype=bool),
        is_bearish_break_failure=np.array([False] * 18 + [True] + [False], dtype=bool),
        is_bullish_immediate_failure=np.array([False] * 16 + [True] + [False] * 3, dtype=bool),
        is_bearish_immediate_failure=np.array([False] * 19 + [True], dtype=bool),
        is_failed_choch_bullish=np.array([False] * 17 + [True] + [False] * 2, dtype=bool),
        is_failed_choch_bearish=np.array([False] * 14 + [True] + [False] * 5, dtype=bool)
    )

    # Signal quality with CHOCH signals - COMPLETE STRUCTURE
    signal_quality = SignalQuality(
        bos_bullish_confirmed_quality=np.full(n, 0.5, dtype=np.float32),
        bos_bearish_confirmed_quality=np.full(n, 0.5, dtype=np.float32),
        bos_bullish_momentum_quality=np.full(n, 0.4, dtype=np.float32),
        bos_bearish_momentum_quality=np.full(n, 0.4, dtype=np.float32),
        choch_bullish_quality=np.array([0.0] * 10 + [0.6] * 5 + [0.0] * 5, dtype=np.float32),
        choch_bearish_quality=np.array([0.0] * 5 + [0.6] * 5 + [0.0] * 10, dtype=np.float32),
        failed_bullish_quality=np.full(n, 0.1, dtype=np.float32),
        failed_bearish_quality=np.full(n, 0.1, dtype=np.float32),
        failed_choch_bullish_quality=np.full(n, 0.1, dtype=np.float32),
        failed_choch_bearish_quality=np.full(n, 0.1, dtype=np.float32)
    )

    return validated_signals, signal_quality


# ==============================================================================
# POSITIVE TESTS: Valid Inputs
# ==============================================================================

def test_validate_inputs_valid(sample_data, default_config):
    """Positive test: Valid inputs should not raise."""
    market_regime, range_compression, retest_respect_score = sample_data
    # Should not raise
    _validate_inputs(market_regime, range_compression, retest_respect_score, default_config)


def test_validate_inputs_empty_arrays(default_config):
    """Positive test: Empty arrays are valid."""
    empty_str = np.array([], dtype='U20')
    empty_float = np.array([], dtype=np.float32)

    # Should not raise
    _validate_inputs(empty_str, empty_float, empty_float, default_config)


def test_validate_inputs_mixed_regimes(default_config):
    """Positive test: All valid regime values."""
    market_regime = np.array(list(_VALID_REGIMES), dtype='U20')
    range_compression = np.ones(len(_VALID_REGIMES), dtype=np.float32) * 0.5
    retest_respect_score = np.ones(len(_VALID_REGIMES), dtype=np.float32) * 0.5

    # Should not raise
    _validate_inputs(market_regime, range_compression, retest_respect_score, default_config)


def test_filter_signals_valid_inputs(sample_data, sample_signals, default_config):
    """Positive test: Filter signals with valid inputs."""
    market_regime, range_compression, retest_respect_score = sample_data
    validated_signals, signal_quality = sample_signals

    filtered = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=default_config
    )

    # Should return FilteredSignals object
    assert isinstance(filtered, FilteredSignals)

    # All arrays should have correct length
    n = len(market_regime)
    for field in filtered.__dict__.values():
        assert len(field) == n


def test_filter_bos_by_regime_strong_trend():
    """Positive test: BOS allowed in strong trend."""
    n = 10
    signal_mask = np.ones(n, dtype=bool)
    market_regime = np.array(['strong_trend'] * n, dtype='U20')

    filtered = _filter_bos_by_regime(signal_mask, market_regime)

    # All signals should remain in strong trend
    assert np.all(filtered == signal_mask)


def test_filter_bos_by_regime_weak_trend_allowed():
    """Positive test: BOS allowed in weak trend when configured."""
    n = 10
    signal_mask = np.ones(n, dtype=bool)
    market_regime = np.array(['weak_trend'] * n, dtype='U20')

    filtered = _filter_bos_by_regime(signal_mask, market_regime, allow_weak_trend=True)

    # All signals should remain when weak trend is allowed
    assert np.all(filtered == signal_mask)


def test_filter_bos_by_regime_weak_trend_not_allowed():
    """Positive test: BOS filtered in weak trend when not configured."""
    n = 10
    signal_mask = np.ones(n, dtype=bool)
    market_regime = np.array(['weak_trend'] * n, dtype='U20')

    filtered = _filter_bos_by_regime(signal_mask, market_regime, allow_weak_trend=False)

    # No signals should remain when weak trend is not allowed
    assert not np.any(filtered)


def test_filter_bos_by_regime_ranging_filtered():
    """Positive test: BOS filtered in ranging regime."""
    n = 10
    signal_mask = np.ones(n, dtype=bool)
    market_regime = np.array(['ranging'] * n, dtype='U20')

    filtered = _filter_bos_by_regime(signal_mask, market_regime)

    # No signals should remain in ranging
    assert not np.any(filtered)


def test_filter_by_range_compression_low_compression():
    """Positive test: Signals allowed with low compression."""
    n = 10
    signal_mask = np.ones(n, dtype=bool)
    range_compression = np.full(n, 0.2, dtype=np.float32)
    threshold = 0.5

    filtered = _filter_by_range_compression(signal_mask, range_compression, threshold)

    # All signals should remain (0.2 < 0.5)
    assert np.all(filtered == signal_mask)


def test_filter_by_range_compression_high_compression():
    """Positive test: Signals filtered with high compression."""
    n = 10
    signal_mask = np.ones(n, dtype=bool)
    range_compression = np.full(n, 0.8, dtype=np.float32)
    threshold = 0.5

    filtered = _filter_by_range_compression(signal_mask, range_compression, threshold)

    # No signals should remain (0.8 > 0.5)
    assert not np.any(filtered)


def test_filter_by_retest_respect_high_respect():
    """Positive test: Signals allowed with high retest respect."""
    n = 10
    signal_mask = np.ones(n, dtype=bool)
    retest_respect_score = np.full(n, 0.8, dtype=np.float32)
    threshold = 0.5

    filtered = _filter_by_retest_respect(signal_mask, retest_respect_score, threshold)

    # All signals should remain (0.8 >= 0.5)
    assert np.all(filtered == signal_mask)


def test_filter_by_retest_respect_low_respect():
    """Positive test: Signals filtered with low retest respect."""
    n = 10
    signal_mask = np.ones(n, dtype=bool)
    retest_respect_score = np.full(n, 0.3, dtype=np.float32)
    threshold = 0.5

    filtered = _filter_by_retest_respect(signal_mask, retest_respect_score, threshold)

    # No signals should remain (0.3 < 0.5)
    assert not np.any(filtered)


# ==============================================================================
# NEGATIVE TESTS: Invalid Inputs & Error Handling
# ==============================================================================

def test_validate_inputs_length_mismatch(default_config):
    """Negative test: Array length mismatch."""
    market_regime = np.array(['strong_trend'] * 10, dtype='U20')
    range_compression = np.ones(5, dtype=np.float32)  # Wrong length
    retest_respect_score = np.ones(10, dtype=np.float32)

    with pytest.raises(ValueError, match="Array length mismatch"):
        _validate_inputs(market_regime, range_compression, retest_respect_score, default_config)


def test_validate_inputs_invalid_regime(default_config):
    """Negative test: Invalid regime values."""
    market_regime = np.array(['invalid_regime', 'strong_trend'], dtype='U20')
    range_compression = np.ones(2, dtype=np.float32)
    retest_respect_score = np.ones(2, dtype=np.float32)

    with pytest.raises(ValueError, match="Invalid market_regime values"):
        _validate_inputs(market_regime, range_compression, retest_respect_score, default_config)


def test_validate_inputs_wrong_dtype(default_config):
    """Negative test: Wrong array dtype."""
    market_regime = np.array(['strong_trend'], dtype='U20')
    range_compression = np.array([1], dtype=np.int32)  # Not float
    retest_respect_score = np.ones(1, dtype=np.float32)

    with pytest.raises(TypeError, match="must be float array"):
        _validate_inputs(market_regime, range_compression, retest_respect_score, default_config)


def test_validate_inputs_range_compression_out_of_range(default_config):
    """Negative test: Range compression out of range [0, 1]."""
    market_regime = np.array(['strong_trend'], dtype='U20')
    range_compression = np.array([1.5], dtype=np.float32)  # > 1.0
    retest_respect_score = np.ones(1, dtype=np.float32)

    with pytest.raises(ValueError, match="must be in range"):
        _validate_inputs(market_regime, range_compression, retest_respect_score, default_config)


def test_validate_inputs_retest_respect_out_of_range(default_config):
    """Negative test: Retest respect out of range [0, 1]."""
    market_regime = np.array(['strong_trend'], dtype='U20')
    range_compression = np.ones(1, dtype=np.float32)
    retest_respect_score = np.array([-0.1], dtype=np.float32)  # < 0.0

    with pytest.raises(ValueError, match="must be in range"):
        _validate_inputs(market_regime, range_compression, retest_respect_score, default_config)


def test_filter_signals_invalid_config(sample_data, sample_signals):
    """Negative test: Invalid config parameters."""
    market_regime, range_compression, retest_respect_score = sample_data
    validated_signals, signal_quality = sample_signals

    # Config with invalid range compression
    with pytest.raises(ValueError, match="must be in range"):
        invalid_config = SignalFilterConfig(max_range_compression=1.5)

    # Config with invalid retest respect
    with pytest.raises(ValueError, match="must be in range"):
        invalid_config = SignalFilterConfig(min_retest_respect_filter=-0.1)


# ==============================================================================
# BOUNDARY TESTS: Edge Cases
# ==============================================================================

def test_filter_bos_by_regime_empty_signal():
    """Boundary test: Empty signal mask."""
    n = 10
    signal_mask = np.zeros(n, dtype=bool)  # All False
    market_regime = np.array(['strong_trend'] * n, dtype='U20')

    filtered = _filter_bos_by_regime(signal_mask, market_regime)

    # Should return empty mask (all False)
    assert not np.any(filtered)
    assert np.all(filtered == signal_mask)


def test_filter_by_range_compression_threshold_exact():
    """Boundary test: Compression exactly at threshold."""
    n = 4
    signal_mask = np.ones(n, dtype=bool)
    range_compression = np.full(n, 0.5, dtype=np.float32)
    threshold = 0.5  # Exactly equal

    filtered = _filter_by_range_compression(signal_mask, range_compression, threshold)

    # Signals should be allowed (compression <= threshold)
    assert np.all(filtered == signal_mask)


def test_filter_by_range_compression_threshold_zero():
    """Boundary test: Threshold zero."""
    n = 4
    signal_mask = np.ones(n, dtype=bool)
    range_compression = np.array([0.0, 0.1, 0.0, 0.1], dtype=np.float32)
    threshold = 0.0

    filtered = _filter_by_range_compression(signal_mask, range_compression, threshold)

    # Only compression = 0.0 should pass
    expected = np.array([True, False, True, False], dtype=bool)
    assert np.array_equal(filtered, expected)


def test_filter_by_range_compression_threshold_one():
    """Boundary test: Threshold one."""
    n = 4
    signal_mask = np.ones(n, dtype=bool)
    range_compression = np.array([0.9, 1.0, 0.5, 1.0], dtype=np.float32)
    threshold = 1.0

    filtered = _filter_by_range_compression(signal_mask, range_compression, threshold)

    # All should pass (compression <= 1.0)
    assert np.all(filtered == signal_mask)


def test_filter_by_retest_respect_threshold_exact():
    """Boundary test: Retest respect exactly at threshold."""
    n = 4
    signal_mask = np.ones(n, dtype=bool)
    retest_respect_score = np.full(n, 0.5, dtype=np.float32)
    threshold = 0.5  # Exactly equal

    filtered = _filter_by_retest_respect(signal_mask, retest_respect_score, threshold)

    # Signals should be allowed (respect >= threshold)
    assert np.all(filtered == signal_mask)


def test_filter_by_retest_respect_threshold_zero():
    """Boundary test: Threshold zero."""
    n = 4
    signal_mask = np.ones(n, dtype=bool)
    retest_respect_score = np.array([0.0, 0.1, 0.0, 0.1], dtype=np.float32)
    threshold = 0.0

    filtered = _filter_by_retest_respect(signal_mask, retest_respect_score, threshold)

    # All should pass (respect >= 0.0)
    assert np.all(filtered == signal_mask)


def test_filter_by_retest_respect_threshold_one():
    """Boundary test: Threshold one."""
    n = 4
    signal_mask = np.ones(n, dtype=bool)
    retest_respect_score = np.array([0.9, 1.0, 0.5, 1.0], dtype=np.float32)
    threshold = 1.0

    filtered = _filter_by_retest_respect(signal_mask, retest_respect_score, threshold)

    # Only respect = 1.0 should pass
    expected = np.array([False, True, False, True], dtype=bool)
    assert np.array_equal(filtered, expected)


def test_filter_signals_single_bar(sample_signals):
    """Boundary test: Single bar arrays."""
    market_regime = np.array(['strong_trend'], dtype='U20')
    range_compression = np.array([0.5], dtype=np.float32)
    retest_respect_score = np.array([0.8], dtype=np.float32)
    validated_signals, signal_quality = sample_signals

    # Create single bar versions
    validated_single = ValidatedSignals(
        **{k: v[:1] for k, v in validated_signals.__dict__.items()}
    )
    quality_single = SignalQuality(
        **{k: v[:1] for k, v in signal_quality.__dict__.items()}
    )

    config = SignalFilterConfig()

    filtered = filter_signals(
        validated_signals=validated_single,
        signal_quality=quality_single,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=config
    )

    # Should return FilteredSignals with single element arrays
    assert isinstance(filtered, FilteredSignals)
    for field in filtered.__dict__.values():
        assert len(field) == 1
        assert field.dtype == bool


def test_filter_signals_empty_arrays():
    """Boundary test: All empty arrays."""
    empty_str = np.array([], dtype='U20')
    empty_float = np.array([], dtype=np.float32)
    empty_bool = np.array([], dtype=bool)

    empty_validated = ValidatedSignals(
        is_bos_bullish_confirmed=empty_bool,
        is_bos_bearish_confirmed=empty_bool,
        is_bos_bullish_momentum=empty_bool,
        is_bos_bearish_momentum=empty_bool,
        is_bullish_break_failure=empty_bool,
        is_bearish_break_failure=empty_bool,
        is_bullish_immediate_failure=empty_bool,
        is_bearish_immediate_failure=empty_bool,
        is_failed_choch_bullish=empty_bool,
        is_failed_choch_bearish=empty_bool
    )

    empty_quality = SignalQuality(
        bos_bullish_confirmed_quality=empty_float,
        bos_bearish_confirmed_quality=empty_float,
        bos_bullish_momentum_quality=empty_float,
        bos_bearish_momentum_quality=empty_float,
        choch_bullish_quality=empty_float,
        choch_bearish_quality=empty_float,
        failed_bullish_quality=empty_float,
        failed_bearish_quality=empty_float,
        failed_choch_bullish_quality=empty_float,
        failed_choch_bearish_quality=empty_float
    )

    config = SignalFilterConfig()

    filtered = filter_signals(
        validated_signals=empty_validated,
        signal_quality=empty_quality,
        market_regime=empty_str,
        range_compression=empty_float,
        retest_respect_score=empty_float,
        config=config
    )

    # Should return FilteredSignals with empty arrays
    assert isinstance(filtered, FilteredSignals)
    for field in filtered.__dict__.values():
        assert len(field) == 0
        assert field.dtype == bool


# ==============================================================================
# PROPERTY TESTS: Invariants & Consistency
# ==============================================================================

def test_filtering_never_adds_signals(sample_data, sample_signals, default_config):
    """Property test: Filtering never adds new signals."""
    market_regime, range_compression, retest_respect_score = sample_data
    validated_signals, signal_quality = sample_signals

    filtered = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=default_config
    )

    # Filtered signals should always be subset of original signals
    for field_name in validated_signals.__dict__.keys():
        if hasattr(filtered, field_name):
            original = getattr(validated_signals, field_name)
            filtered_arr = getattr(filtered, field_name)
            # filtered ⊆ original
            assert np.all((filtered_arr & ~original) == False)


def test_failed_signals_not_filtered(sample_data, sample_signals, default_config):
    """Property test: Failed signals pass through unchanged."""
    market_regime, range_compression, retest_respect_score = sample_data
    validated_signals, signal_quality = sample_signals

    filtered = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=default_config
    )

    # Failed signals should be identical
    assert np.array_equal(
        filtered.is_bullish_break_failure,
        validated_signals.is_bullish_break_failure
    )
    assert np.array_equal(
        filtered.is_bearish_break_failure,
        validated_signals.is_bearish_break_failure
    )
    assert np.array_equal(
        filtered.is_failed_choch_bullish,
        validated_signals.is_failed_choch_bullish
    )
    assert np.array_equal(
        filtered.is_failed_choch_bearish,
        validated_signals.is_failed_choch_bearish
    )


def test_choch_filtering_uses_quality(sample_data, sample_signals, default_config):
    """Property test: CHOCH filtering uses quality scores."""
    market_regime, range_compression, retest_respect_score = sample_data
    validated_signals, signal_quality = sample_signals

    filtered = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=default_config
    )

    # CHOCH signals should only exist where quality > 0
    assert np.all((filtered.is_choch_bullish & (signal_quality.choch_bullish_quality == 0)) == False)
    assert np.all((filtered.is_choch_bearish & (signal_quality.choch_bearish_quality == 0)) == False)


def test_strict_config_filters_more(sample_data, sample_signals, default_config, strict_config):
    """Property test: Strict config filters more signals than default."""
    market_regime, range_compression, retest_respect_score = sample_data
    validated_signals, signal_quality = sample_signals

    filtered_default = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=default_config
    )

    filtered_strict = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=strict_config
    )

    # Strict config should filter more (or equal) signals
    for field_name in filtered_default.__dict__.keys():
        default_arr = getattr(filtered_default, field_name)
        strict_arr = getattr(filtered_strict, field_name)
        # strict ⊆ default
        assert np.all((strict_arr & ~default_arr) == False)


def test_lenient_config_filters_less(sample_data, sample_signals, default_config, lenient_config):
    """Property test: Lenient config filters fewer signals than default."""
    market_regime, range_compression, retest_respect_score = sample_data
    validated_signals, signal_quality = sample_signals

    filtered_default = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=default_config
    )

    filtered_lenient = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=lenient_config
    )

    # Lenient config should filter less (or equal) signals
    for field_name in filtered_default.__dict__.keys():
        default_arr = getattr(filtered_default, field_name)
        lenient_arr = getattr(filtered_lenient, field_name)
        # default ⊆ lenient
        assert np.all((default_arr & ~lenient_arr) == False)


def test_all_filtered_arrays_same_length(sample_data, sample_signals, default_config):
    """Property test: All filtered arrays have same length as input."""
    market_regime, range_compression, retest_respect_score = sample_data
    validated_signals, signal_quality = sample_signals

    filtered = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=default_config
    )

    n = len(market_regime)
    for field in filtered.__dict__.values():
        assert len(field) == n


# ==============================================================================
# INTEGRATION TESTS: Realistic Scenarios
# ==============================================================================

def test_realistic_trading_scenario():
    """Integration test: Realistic trading scenario."""
    n = 100

    # Create realistic market regime sequence
    market_regime = np.array(
        ['strong_trend'] * 20 +
        ['weak_trend'] * 15 +
        ['ranging'] * 30 +
        ['chop'] * 20 +
        ['neutral'] * 15,
        dtype='U20'
    )

    # Range compression (higher during chop/ranging)
    range_compression = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.4 + 0.5
    range_compression = np.clip(range_compression, 0, 1).astype(np.float32)

    # Retest respect (higher during trends)
    retest_respect_score = np.cos(np.linspace(0, 3 * np.pi, n)) * 0.3 + 0.6
    retest_respect_score = np.clip(retest_respect_score, 0, 1).astype(np.float32)

    # Validated signals
    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.random.rand(n) > 0.9,  # 10% signals
        is_bos_bearish_confirmed=np.random.rand(n) > 0.9,
        is_bos_bullish_momentum=np.random.rand(n) > 0.92,  # 8% signals
        is_bos_bearish_momentum=np.random.rand(n) > 0.92,
        is_bullish_break_failure=np.random.rand(n) > 0.98,  # 2% failures
        is_bearish_break_failure=np.random.rand(n) > 0.98,
        is_bullish_immediate_failure=np.zeros(n, dtype=bool),
        is_bearish_immediate_failure=np.zeros(n, dtype=bool),
        is_failed_choch_bullish=np.random.rand(n) > 0.98,  # 2% failed CHOCH
        is_failed_choch_bearish=np.random.rand(n) > 0.98
    )

    # Signal quality (simulate scoring) - COMPLETE STRUCTURE
    signal_quality = SignalQuality(
        bos_bullish_confirmed_quality=np.clip(np.random.randn(n) * 0.2 + 0.5, 0, 1).astype(np.float32),
        bos_bearish_confirmed_quality=np.clip(np.random.randn(n) * 0.2 + 0.5, 0, 1).astype(np.float32),
        bos_bullish_momentum_quality=np.clip(np.random.randn(n) * 0.2 + 0.4, 0, 1).astype(np.float32),
        bos_bearish_momentum_quality=np.clip(np.random.randn(n) * 0.2 + 0.4, 0, 1).astype(np.float32),
        choch_bullish_quality=np.clip(np.random.randn(n) * 0.2 + 0.45, 0, 1).astype(np.float32),
        choch_bearish_quality=np.clip(np.random.randn(n) * 0.2 + 0.45, 0, 1).astype(np.float32),
        failed_bullish_quality=np.full(n, 0.1, dtype=np.float32),
        failed_bearish_quality=np.full(n, 0.1, dtype=np.float32),
        failed_choch_bullish_quality=np.full(n, 0.1, dtype=np.float32),
        failed_choch_bearish_quality=np.full(n, 0.1, dtype=np.float32)
    )

    config = SignalFilterConfig()

    # Should not raise
    filtered = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=config
    )

    # Verify basic properties
    assert isinstance(filtered, FilteredSignals)
    assert len(filtered.is_bos_bullish_confirmed) == n


# ==============================================================================
# PERFORMANCE TESTS: Large Arrays
# ==============================================================================

def test_performance_large_arrays():
    """Performance test: Handle large arrays efficiently."""
    n = 10000  # 10k bars

    market_regime = np.array(['strong_trend'] * (n // 2) + ['ranging'] * (n // 2), dtype='U20')
    range_compression = np.random.rand(n).astype(np.float32)
    retest_respect_score = np.random.rand(n).astype(np.float32)

    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.random.rand(n) > 0.95,
        is_bos_bearish_confirmed=np.random.rand(n) > 0.95,
        is_bos_bullish_momentum=np.random.rand(n) > 0.96,
        is_bos_bearish_momentum=np.random.rand(n) > 0.96,
        is_bullish_break_failure=np.random.rand(n) > 0.99,
        is_bearish_break_failure=np.random.rand(n) > 0.99,
        is_bullish_immediate_failure=np.zeros(n, dtype=bool),
        is_bearish_immediate_failure=np.zeros(n, dtype=bool),
        is_failed_choch_bullish=np.random.rand(n) > 0.99,
        is_failed_choch_bearish=np.random.rand(n) > 0.99
    )

    # COMPLETE SignalQuality structure
    signal_quality = SignalQuality(
        bos_bullish_confirmed_quality=np.full(n, 0.5, dtype=np.float32),
        bos_bearish_confirmed_quality=np.full(n, 0.5, dtype=np.float32),
        bos_bullish_momentum_quality=np.full(n, 0.4, dtype=np.float32),
        bos_bearish_momentum_quality=np.full(n, 0.4, dtype=np.float32),
        choch_bullish_quality=np.full(n, 0.45, dtype=np.float32),
        choch_bearish_quality=np.full(n, 0.45, dtype=np.float32),
        failed_bullish_quality=np.full(n, 0.1, dtype=np.float32),
        failed_bearish_quality=np.full(n, 0.1, dtype=np.float32),
        failed_choch_bullish_quality=np.full(n, 0.1, dtype=np.float32),
        failed_choch_bearish_quality=np.full(n, 0.1, dtype=np.float32)
    )

    config = SignalFilterConfig()

    # Time the execution
    import time
    start = time.time()

    filtered = filter_signals(
        validated_signals=validated_signals,
        signal_quality=signal_quality,
        market_regime=market_regime,
        range_compression=range_compression,
        retest_respect_score=retest_respect_score,
        config=config
    )

    elapsed = time.time() - start

    # Should complete within reasonable time (sub-second for 10k)
    assert elapsed < 1.0, f"Filtering 10k bars took {elapsed:.2f}s, expected < 1.0s"

    # Verify results
    assert isinstance(filtered, FilteredSignals)
    assert len(filtered.is_bos_bullish_confirmed) == n