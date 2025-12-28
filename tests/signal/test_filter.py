# test_filter.py
import pytest
import numpy as np
from structure.signal.filter import filter_signals
from structure.metrics.types import ValidatedSignals
from structure.signal.config import SignalFilterConfig


def test_filter_negative_low_zone_confluence():
    """❌ Negative: zone confluence below threshold → signal filtered."""
    n = 2
    signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([True, True]),
        is_bos_bearish_confirmed=np.array([False, False]),
        is_bos_bullish_momentum=np.array([False, False]),
        is_bos_bearish_momentum=np.array([False, False]),
        is_bullish_break_failure=np.array([False, False]),
        is_bearish_break_failure=np.array([False, False])
    )
    config = SignalFilterConfig(min_zone_confluence=0.7)  # require high confluence
    result = filter_signals(
        validated_signals=signals,
        market_regime=np.array(['strong_trend', 'strong_trend']),
        zone_confluence=np.array([0.8, 0.6]),  # second < 0.7
        is_range_compression=np.array([False, False]),
        retest_velocity=np.array([0.1, 0.1]),
        session=np.array(['ny', 'ny']),
        config=config
    )
    assert result.is_bos_bullish_confirmed[0]   # passes
    assert not result.is_bos_bullish_confirmed[1]  # filtered due to low confluence


def test_filter_edge_empty_inputs():
    """⚠️ Edge: zero-length inputs → return empty arrays, no crash."""
    signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([], dtype=bool),
        is_bos_bearish_confirmed=np.array([], dtype=bool),
        is_bos_bullish_momentum=np.array([], dtype=bool),
        is_bos_bearish_momentum=np.array([], dtype=bool),
        is_bullish_break_failure=np.array([], dtype=bool),
        is_bearish_break_failure=np.array([], dtype=bool)
    )
    config = SignalFilterConfig()
    result = filter_signals(
        validated_signals=signals,
        market_regime=np.array([]),
        zone_confluence=np.array([]),
        is_range_compression=np.array([], dtype=bool),
        retest_velocity=np.array([]),
        session=np.array([]),
        config=config
    )
    assert len(result.is_bos_bullish_confirmed) == 0
    assert isinstance(result, ValidatedSignals)


# tests/signal/test_filter.py
import pytest
import numpy as np
from structure.signal.filter import filter_signals
from structure.metrics.types import ValidatedSignals
from structure.signal.config import SignalFilterConfig


def test_filter_positive_all_conditions_met():
    """✅ Positive: signals in strong trend, high confluence, no compression → pass."""
    n = 3
    signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([True, True, False]),
        is_bos_bearish_confirmed=np.array([False, False, True]),
        is_bos_bullish_momentum=np.array([False, False, False]),
        is_bos_bearish_momentum=np.array([False, False, False]),
        is_bullish_break_failure=np.array([False, False, False]),
        is_bearish_break_failure=np.array([False, False, False])
    )
    config = SignalFilterConfig(
        min_regime_score=0.6,
        min_zone_confluence=0.5,
        avoid_range_compression=True,
        avoid_fast_retests=False  # With False, fast retests (velocity > 0.5) are filtered
    )

    # Make sure retest_velocity <= 0.5 for signals that should pass
    result = filter_signals(
        validated_signals=signals,
        market_regime=np.array(['strong_trend', 'weak_trend', 'ranging']),
        zone_confluence=np.array([0.9, 0.8, 0.4]),
        is_range_compression=np.array([False, False, True]),
        retest_velocity=np.array([0.2, 0.4, 0.3]),  # Changed: 0.4 instead of 0.6
        session=np.array(['ny', 'london', 'asia']),
        config=config
    )

    # First two bullish signals should pass (trending + high confluence + no compression + velocity <= 0.5)
    assert result.is_bos_bullish_confirmed[0]
    assert result.is_bos_bullish_confirmed[1]

    # Third bullish signal doesn't exist (False), so result should be False
    assert not result.is_bos_bullish_confirmed[2]

    # Bearish signal at index 2 is in ranging market, so should be filtered
    assert not result.is_bos_bearish_confirmed[2]


def test_filter_with_avoid_fast_retests():
    """Test filter with avoid_fast_retests=True."""
    n = 3
    signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([True, True, True]),
        is_bos_bearish_confirmed=np.array([False, False, False]),
        is_bos_bullish_momentum=np.array([False, False, False]),
        is_bos_bearish_momentum=np.array([False, False, False]),
        is_bullish_break_failure=np.array([False, False, False]),
        is_bearish_break_failure=np.array([False, False, False])
    )

    # Test BOTH cases: avoid_fast_retests=False and avoid_fast_retests=True
    for avoid_fast_retests in [False, True]:
        config = SignalFilterConfig(
            min_regime_score=0.6,
            min_zone_confluence=0.5,
            avoid_range_compression=True,
            avoid_fast_retests=avoid_fast_retests
        )

        # Signals with different retest velocities
        retest_velocity = np.array([0.3, 0.6, 0.8])  # slow, fast, very fast

        result = filter_signals(
            validated_signals=signals,
            market_regime=np.array(['strong_trend', 'strong_trend', 'strong_trend']),
            zone_confluence=np.array([0.7, 0.7, 0.7]),
            is_range_compression=np.array([False, False, False]),
            retest_velocity=retest_velocity,
            session=np.array(['ny', 'ny', 'ny']),
            config=config
        )

        if not avoid_fast_retests:
            # With avoid_fast_retests=False, only slow retests pass (velocity <= 0.5)
            assert result.is_bos_bullish_confirmed[0]  # velocity=0.3 <= 0.5
            assert not result.is_bos_bullish_confirmed[1]  # velocity=0.6 > 0.5
            assert not result.is_bos_bullish_confirmed[2]  # velocity=0.8 > 0.5
        else:
            # With avoid_fast_retests=True, fast retests are filtered out
            # fast_retest_mask becomes: ~(retest_velocity <= 0.5) = (retest_velocity > 0.5)
            # So we KEEP signals with velocity > 0.5
            assert not result.is_bos_bullish_confirmed[0]  # velocity=0.3 <= 0.5 (filtered)
            assert result.is_bos_bullish_confirmed[1]  # velocity=0.6 > 0.5 (kept)
            assert result.is_bos_bullish_confirmed[2]  # velocity=0.8 > 0.5 (kept)


def test_filter_negative_low_zone_confluence():
    """❌ Negative: zone confluence below threshold → signals filtered."""
    n = 3
    signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([True, True, True]),
        is_bos_bearish_confirmed=np.array([False, False, False]),
        is_bos_bullish_momentum=np.array([False, False, False]),
        is_bos_bearish_momentum=np.array([False, False, False]),
        is_bullish_break_failure=np.array([False, False, False]),
        is_bearish_break_failure=np.array([False, False, False])
    )
    config = SignalFilterConfig(
        min_regime_score=0.6,
        min_zone_confluence=0.7,  # Higher threshold
        avoid_range_compression=True,
        avoid_fast_retests=False
    )

    result = filter_signals(
        validated_signals=signals,
        market_regime=np.array(['strong_trend', 'strong_trend', 'strong_trend']),
        zone_confluence=np.array([0.8, 0.6, 0.9]),  # Middle one below threshold
        is_range_compression=np.array([False, False, False]),
        retest_velocity=np.array([0.2, 0.2, 0.2]),
        session=np.array(['ny', 'ny', 'ny']),
        config=config
    )

    # Only signals with confluence >= 0.7 should pass
    assert result.is_bos_bullish_confirmed[0]  # 0.8 >= 0.7
    assert not result.is_bos_bullish_confirmed[1]  # 0.6 < 0.7
    assert result.is_bos_bullish_confirmed[2]  # 0.9 >= 0.7


def test_filter_edge_empty_inputs():
    """⚠️ Edge: empty arrays."""
    signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([], dtype=bool),
        is_bos_bearish_confirmed=np.array([], dtype=bool),
        is_bos_bullish_momentum=np.array([], dtype=bool),
        is_bos_bearish_momentum=np.array([], dtype=bool),
        is_bullish_break_failure=np.array([], dtype=bool),
        is_bearish_break_failure=np.array([], dtype=bool)
    )
    config = SignalFilterConfig()

    result = filter_signals(
        validated_signals=signals,
        market_regime=np.array([]),
        zone_confluence=np.array([]),
        is_range_compression=np.array([], dtype=bool),
        retest_velocity=np.array([]),
        session=np.array([]),
        config=config
    )

    # Should return empty arrays
    assert len(result.is_bos_bullish_confirmed) == 0
    assert len(result.is_bos_bearish_confirmed) == 0


def test_filter_range_compression():
    """Test range compression filter."""
    n = 3
    signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([True, True, True]),
        is_bos_bearish_confirmed=np.array([False, False, False]),
        is_bos_bullish_momentum=np.array([False, False, False]),
        is_bos_bearish_momentum=np.array([False, False, False]),
        is_bullish_break_failure=np.array([False, False, False]),
        is_bearish_break_failure=np.array([False, False, False])
    )

    # Test with avoid_range_compression=True (default)
    config = SignalFilterConfig(avoid_range_compression=True)

    result = filter_signals(
        validated_signals=signals,
        market_regime=np.array(['strong_trend', 'strong_trend', 'strong_trend']),
        zone_confluence=np.array([0.8, 0.8, 0.8]),
        is_range_compression=np.array([False, True, False]),  # Middle one has compression
        retest_velocity=np.array([0.2, 0.2, 0.2]),
        session=np.array(['ny', 'ny', 'ny']),
        config=config
    )

    # With avoid_range_compression=True, compression signals are filtered
    assert result.is_bos_bullish_confirmed[0]  # No compression
    assert not result.is_bos_bullish_confirmed[1]  # Has compression
    assert result.is_bos_bullish_confirmed[2]  # No compression

    # Test with avoid_range_compression=False
    config2 = SignalFilterConfig(avoid_range_compression=False)

    result2 = filter_signals(
        validated_signals=signals,
        market_regime=np.array(['strong_trend', 'strong_trend', 'strong_trend']),
        zone_confluence=np.array([0.8, 0.8, 0.8]),
        is_range_compression=np.array([False, True, False]),
        retest_velocity=np.array([0.2, 0.2, 0.2]),
        session=np.array(['ny', 'ny', 'ny']),
        config=config2
    )

    # With avoid_range_compression=False, all pass regardless of compression
    assert result2.is_bos_bullish_confirmed[0]
    assert result2.is_bos_bullish_confirmed[1]  # Now passes even with compression
    assert result2.is_bos_bullish_confirmed[2]


def test_filter_market_regime():
    """Test market regime filter."""
    n = 4
    signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([True, True, True, True]),
        is_bos_bearish_confirmed=np.array([False, False, False, False]),
        is_bos_bullish_momentum=np.array([False, False, False, False]),
        is_bos_bearish_momentum=np.array([False, False, False, False]),
        is_bullish_break_failure=np.array([False, False, False, False]),
        is_bearish_break_failure=np.array([False, False, False, False])
    )
    config = SignalFilterConfig()

    result = filter_signals(
        validated_signals=signals,
        market_regime=np.array(['strong_trend', 'weak_trend', 'ranging', 'consolidation']),
        zone_confluence=np.array([0.8, 0.8, 0.8, 0.8]),
        is_range_compression=np.array([False, False, False, False]),
        retest_velocity=np.array([0.2, 0.2, 0.2, 0.2]),
        session=np.array(['ny', 'ny', 'ny', 'ny']),
        config=config
    )

    # Only trending regimes pass
    assert result.is_bos_bullish_confirmed[0]  # strong_trend
    assert result.is_bos_bullish_confirmed[1]  # weak_trend
    assert not result.is_bos_bullish_confirmed[2]  # ranging
    assert not result.is_bos_bullish_confirmed[3]  # consolidation