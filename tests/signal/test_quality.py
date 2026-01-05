# tests/signal/test_quality.py
"""
Comprehensive test suite for signal quality scoring.
Tests positive, negative, and boundary cases for all functions.
"""

import pytest
import numpy as np
from typing import Dict
from structure.signal.quality import (
    score_signals,
    _validate_inputs,
    _compute_session_boost,
    _compute_signal_score,
    _extract_choch_signals,
    _MIN_QUALITY,
    _MAX_QUALITY,
    _BASE_BOS_CONFIRMED,
    _BASE_BOS_MOMENTUM,
    _BASE_CHOCH,
    _BASE_FAILED
)
from structure.signal.config import SignalQualityConfig
from structure.metrics.types import RawSignals, ValidatedSignals, SignalQuality


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def base_config():
    """Default valid configuration."""
    return SignalQualityConfig(
        session_weights={
            'ny': 1.2,
            'london': 1.1,
            'asia': 0.9,
            'overlap': 1.3,
            'low_liquidity': 0.5
        },
        regime_boost=0.2,
        zone_boost=0.15,
        liquidity_boost=0.1,
        session_boost_scale=0.1
    )


@pytest.fixture
def minimal_config():
    """Minimal valid configuration."""
    return SignalQualityConfig(
        session_weights={'ny': 1.0},
        regime_boost=0.0,
        zone_boost=0.0,
        liquidity_boost=0.0,
        session_boost_scale=0.0
    )


@pytest.fixture
def sample_arrays():
    """Create sample arrays for testing."""
    n = 100
    market_regime = np.array(['strong_trend'] * 30 + ['ranging'] * 30 + ['chop'] * 40, dtype='U20')
    session = np.array(['ny'] * 40 + ['london'] * 30 + ['asia'] * 30, dtype='U20')
    liquidity_score = np.linspace(0.1, 0.9, n, dtype=np.float32)
    zone_confluence = np.linspace(0.2, 0.8, n, dtype=np.float32)

    return market_regime, session, liquidity_score, zone_confluence


@pytest.fixture
def sample_signals():
    """Create sample signal masks for testing."""
    n = 100

    # Raw signals (contains raw CHOCH)
    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(n, dtype=bool),
        is_bos_bearish_initial=np.zeros(n, dtype=bool),
        is_choch_bullish=np.zeros(n, dtype=bool),
        is_choch_bearish=np.zeros(n, dtype=bool)
    )

    # Validated signals
    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.zeros(n, dtype=bool),
        is_bos_bearish_confirmed=np.zeros(n, dtype=bool),
        is_bos_bullish_momentum=np.zeros(n, dtype=bool),
        is_bos_bearish_momentum=np.zeros(n, dtype=bool),
        is_bullish_break_failure=np.zeros(n, dtype=bool),
        is_bearish_break_failure=np.zeros(n, dtype=bool),
        is_bullish_immediate_failure=np.zeros(n, dtype=bool),
        is_bearish_immediate_failure=np.zeros(n, dtype=bool),
        is_failed_choch_bullish=np.zeros(n, dtype=bool),
        is_failed_choch_bearish=np.zeros(n, dtype=bool)
    )

    return raw_signals, validated_signals


# ==============================================================================
# POSITIVE TESTS: _validate_inputs
# ==============================================================================

def test_validate_inputs_valid(base_config, sample_arrays):
    """Positive test: Valid inputs should not raise."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    # Should not raise
    _validate_inputs(market_regime, session, liquidity_score, zone_confluence, base_config)


def test_validate_inputs_empty_arrays(base_config):
    """Positive test: Empty arrays are valid."""
    empty_str = np.array([], dtype='U20')
    empty_float = np.array([], dtype=np.float32)

    # Should not raise
    _validate_inputs(empty_str, empty_str, empty_float, empty_float, base_config)


def test_validate_inputs_mixed_regimes(base_config):
    """Positive test: All valid regime values."""
    market_regime = np.array(['strong_trend', 'weak_trend', 'ranging', 'chop', 'neutral'], dtype='U20')
    session = np.array(['ny'] * 5, dtype='U20')
    liquidity = np.ones(5, dtype=np.float32) * 0.5
    zone = np.ones(5, dtype=np.float32) * 0.5

    # Should not raise
    _validate_inputs(market_regime, session, liquidity, zone, base_config)


# ==============================================================================
# NEGATIVE TESTS: _validate_inputs
# ==============================================================================

def test_validate_inputs_length_mismatch(base_config, sample_arrays):
    """Negative test: Array length mismatch."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays

    with pytest.raises(ValueError, match="Array length mismatch"):
        _validate_inputs(market_regime[:50], session, liquidity_score, zone_confluence, base_config)


def test_validate_inputs_invalid_regime(base_config, sample_arrays):
    """Negative test: Invalid regime value."""
    _, session, liquidity_score, zone_confluence = sample_arrays
    market_regime = np.array(['invalid_regime'] * 100, dtype='U20')

    with pytest.raises(ValueError, match="Invalid market_regime values"):
        _validate_inputs(market_regime, session, liquidity_score, zone_confluence, base_config)


def test_validate_inputs_invalid_session(base_config, sample_arrays):
    """Negative test: Invalid session value."""
    market_regime, _, liquidity_score, zone_confluence = sample_arrays
    session = np.array(['invalid_session'] * 100, dtype='U20')

    with pytest.raises(ValueError, match="Invalid session values"):
        _validate_inputs(market_regime, session, liquidity_score, zone_confluence, base_config)


def test_validate_inputs_liquidity_out_of_range(base_config, sample_arrays):
    """Negative test: Liquidity score out of range."""
    market_regime, session, _, zone_confluence = sample_arrays
    liquidity_score = np.full(100, 1.5, dtype=np.float32)  # > 1.0

    with pytest.raises(ValueError, match="liquidity_score must be in range"):
        _validate_inputs(market_regime, session, liquidity_score, zone_confluence, base_config)


def test_validate_inputs_zone_out_of_range(base_config, sample_arrays):
    """Negative test: Zone confluence out of range."""
    market_regime, session, liquidity_score, _ = sample_arrays
    zone_confluence = np.full(100, -0.1, dtype=np.float32)  # < 0.0

    with pytest.raises(ValueError, match="zone_confluence must be in range"):
        _validate_inputs(market_regime, session, liquidity_score, zone_confluence, base_config)


def test_validate_inputs_wrong_dtype(base_config, sample_arrays):
    """Negative test: Wrong dtype for numeric arrays."""
    market_regime, session, _, _ = sample_arrays
    liquidity_score = np.ones(100, dtype=np.int32)  # Wrong dtype
    zone_confluence = np.ones(100, dtype=np.float32)

    with pytest.raises(TypeError, match="liquidity_score must be float array"):
        _validate_inputs(market_regime, session, liquidity_score, zone_confluence, base_config)


# ==============================================================================
# POSITIVE TESTS: _compute_session_boost
# ==============================================================================

def test_compute_session_boost_positive_weight(base_config):
    """Positive test: Session weight > 1.0 gives positive boost."""
    session = np.array(['ny', 'london', 'asia'], dtype='U20')
    boost = _compute_session_boost(session, base_config)

    # ny weight = 1.2, scale = 0.1 → boost = (1.2-1)*0.1 = 0.02
    # london weight = 1.1 → boost = (1.1-1)*0.1 = 0.01
    # asia weight = 0.9 → boost = (0.9-1)*0.1 = -0.01

    assert boost[0] == pytest.approx(0.02)
    assert boost[1] == pytest.approx(0.01)
    assert boost[2] == pytest.approx(-0.01)


def test_compute_session_boost_neutral_weight(minimal_config):
    """Positive test: Session weight = 1.0 gives zero boost."""
    session = np.array(['ny'], dtype='U20')
    boost = _compute_session_boost(session, minimal_config)

    assert boost[0] == 0.0


def test_compute_session_boost_mixed_session(base_config):
    """Positive test: Mixed session values."""
    session = np.array(['ny', 'asia', 'ny', 'london', 'asia'], dtype='U20')
    boost = _compute_session_boost(session, base_config)

    # Check correct mapping
    assert boost[0] == pytest.approx(0.02)  # ny
    assert boost[1] == pytest.approx(-0.01)  # asia
    assert boost[2] == pytest.approx(0.02)  # ny
    assert boost[3] == pytest.approx(0.01)  # london
    assert boost[4] == pytest.approx(-0.01)  # asia


# ==============================================================================
# BOUNDARY TESTS: _compute_session_boost
# ==============================================================================

def test_compute_session_boost_empty_array(base_config):
    """Boundary test: Empty session array."""
    session = np.array([], dtype='U20')
    boost = _compute_session_boost(session, base_config)

    assert len(boost) == 0
    assert boost.dtype == np.float32


def test_compute_session_boost_unknown_session_gets_zero(base_config):
    """Boundary test: Unknown session values get zero boost."""
    session = np.array(['unknown_session'], dtype='U20')
    boost = _compute_session_boost(session, base_config)

    assert boost[0] == 0.0  # Unknown session → not in weights → zero boost


# ==============================================================================
# POSITIVE TESTS: _compute_signal_score
# ==============================================================================

def test_compute_signal_score_no_signals(base_config, sample_arrays):
    """Positive test: Empty signal mask returns all zeros."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    signal_mask = np.zeros(100, dtype=bool)
    session_boost = _compute_session_boost(session, base_config)

    scores = _compute_signal_score(
        signal_mask=signal_mask,
        base_score=0.5,
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config,
        regime_preference='trending'
    )

    assert np.all(scores == 0.0)
    assert len(scores) == 100


def test_compute_signal_score_trending_preference(base_config):
    """Positive test: Trending preference boosts trending regimes."""
    n = 10
    market_regime = np.array(['strong_trend'] * 5 + ['ranging'] * 5, dtype='U20')
    session = np.array(['ny'] * n, dtype='U20')
    liquidity_score = np.full(n, 0.5, dtype=np.float32)
    zone_confluence = np.full(n, 0.5, dtype=np.float32)
    session_boost = _compute_session_boost(session, base_config)

    # Signal in first 3 bars (all trending)
    signal_mask = np.array([True, True, True, False, False] + [False] * 5, dtype=bool)

    scores = _compute_signal_score(
        signal_mask=signal_mask,
        base_score=_BASE_BOS_CONFIRMED,  # 0.5
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config,
        regime_preference='trending'
    )

    # First 3 should have regime boost (0.2), others zero
    assert scores[0] > 0.5  # Has regime boost
    assert scores[0] < 1.0  # Within max
    assert np.all(scores[3:] == 0.0)  # No signal


def test_compute_signal_score_reversal_preference(base_config):
    """Positive test: Reversal preference boosts ranging regimes."""
    n = 10
    market_regime = np.array(['strong_trend'] * 5 + ['ranging'] * 5, dtype='U20')
    session = np.array(['ny'] * n, dtype='U20')
    liquidity_score = np.full(n, 0.5, dtype=np.float32)
    zone_confluence = np.full(n, 0.5, dtype=np.float32)
    session_boost = _compute_session_boost(session, base_config)

    # Signal in last 3 bars (all ranging)
    signal_mask = np.array([False] * 7 + [True, True, True], dtype=bool)

    scores = _compute_signal_score(
        signal_mask=signal_mask,
        base_score=_BASE_CHOCH,  # 0.45
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config,
        regime_preference='reversal'
    )

    # Last 3 should have regime boost (0.2), others zero
    assert scores[7] > 0.45  # Has regime boost
    assert scores[7] < 1.0  # Within max
    assert np.all(scores[:7] == 0.0)  # No signal


def test_compute_signal_score_clamping(base_config):
    """Positive test: Scores are clamped to [MIN_QUALITY, MAX_QUALITY]."""
    n = 3
    market_regime = np.array(['strong_trend'] * n, dtype='U20')
    session = np.array(['ny'] * n, dtype='U20')

    # High values that would exceed 1.0 without clamping
    liquidity_score = np.full(n, 1.0, dtype=np.float32)
    zone_confluence = np.full(n, 1.0, dtype=np.float32)
    session_boost = _compute_session_boost(session, base_config)

    signal_mask = np.array([True, True, True], dtype=bool)

    scores = _compute_signal_score(
        signal_mask=signal_mask,
        base_score=0.5,
        market_regime=market_regime,
        session_boost=session_boost,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config,
        regime_preference='trending'
    )

    # All scores should be clamped to 1.0 max
    assert np.all(scores <= 1.0)
    assert np.all(scores >= _MIN_QUALITY)


# ==============================================================================
# POSITIVE TESTS: _extract_choch_signals
# ==============================================================================

def test_extract_choch_signals_no_failures():
    """Positive test: No failed CHOCH → all raw CHOCH are valid."""
    n = 10
    # Create alternating pattern of correct length (10 elements)
    choch_bullish_pattern = np.array([True, False, True, False, True, False, True, False, True, False], dtype=bool)
    choch_bearish_pattern = np.array([False, True, False, True, False, True, False, True, False, True], dtype=bool)

    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(n, dtype=bool),
        is_bos_bearish_initial=np.zeros(n, dtype=bool),
        is_choch_bullish=choch_bullish_pattern,
        is_choch_bearish=choch_bearish_pattern
    )

    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.zeros(n, dtype=bool),
        is_bos_bearish_confirmed=np.zeros(n, dtype=bool),
        is_bos_bullish_momentum=np.zeros(n, dtype=bool),
        is_bos_bearish_momentum=np.zeros(n, dtype=bool),
        is_bullish_break_failure=np.zeros(n, dtype=bool),
        is_bearish_break_failure=np.zeros(n, dtype=bool),
        is_bullish_immediate_failure=np.zeros(n, dtype=bool),
        is_bearish_immediate_failure=np.zeros(n, dtype=bool),
        is_failed_choch_bullish=np.zeros(n, dtype=bool),
        is_failed_choch_bearish=np.zeros(n, dtype=bool)
    )

    choch_bull, choch_bear = _extract_choch_signals(raw_signals, validated_signals)

    # All raw CHOCH should be valid (no failures)
    assert np.array_equal(choch_bull, raw_signals.is_choch_bullish)
    assert np.array_equal(choch_bear, raw_signals.is_choch_bearish)

def test_extract_choch_signals_with_failures():
    """Positive test: Some failed CHOCH → only non-failed are valid."""
    n = 6
    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(n, dtype=bool),
        is_bos_bearish_initial=np.zeros(n, dtype=bool),
        is_choch_bullish=np.array([True, True, True, False, False, False], dtype=bool),
        is_choch_bearish=np.array([False, False, False, True, True, True], dtype=bool)
    )

    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.zeros(n, dtype=bool),
        is_bos_bearish_confirmed=np.zeros(n, dtype=bool),
        is_bos_bullish_momentum=np.zeros(n, dtype=bool),
        is_bos_bearish_momentum=np.zeros(n, dtype=bool),
        is_bullish_break_failure=np.zeros(n, dtype=bool),
        is_bearish_break_failure=np.zeros(n, dtype=bool),
        is_bullish_immediate_failure=np.zeros(n, dtype=bool),
        is_bearish_immediate_failure=np.zeros(n, dtype=bool),
        is_failed_choch_bullish=np.array([False, True, False, False, False, False], dtype=bool),
        is_failed_choch_bearish=np.array([False, False, False, False, True, False], dtype=bool)
    )

    choch_bull, choch_bear = _extract_choch_signals(raw_signals, validated_signals)

    # CHOCH bullish: positions 0,2 are valid (1 is failed)
    expected_bull = np.array([True, False, True, False, False, False], dtype=bool)
    assert np.array_equal(choch_bull, expected_bull)

    # CHOCH bearish: positions 3,5 are valid (4 is failed)
    expected_bear = np.array([False, False, False, True, False, True], dtype=bool)
    assert np.array_equal(choch_bear, expected_bear)


# ==============================================================================
# COMPREHENSIVE POSITIVE TESTS: score_signals
# ==============================================================================

def test_score_signals_empty_signals(base_config, sample_arrays, sample_signals):
    """Positive test: All empty signal masks."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    raw_signals, validated_signals = sample_signals

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config
    )

    # All quality arrays should be zeros (no signals)
    assert isinstance(quality, SignalQuality)
    assert np.all(quality.bos_bullish_confirmed_quality == 0.0)
    assert np.all(quality.choch_bullish_quality == 0.0)
    assert np.all(quality.failed_bullish_quality == 0.0)


def test_score_signals_bos_confirmed_signals(base_config, sample_arrays, sample_signals):
    """Positive test: BOS confirmed signals get correct scoring."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    raw_signals, validated_signals = sample_signals

    # Add some BOS confirmed signals
    validated_signals.is_bos_bullish_confirmed[10:15] = True  # Trending regime
    validated_signals.is_bos_bearish_confirmed[60:65] = True  # Ranging regime

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config
    )

    # Bullish confirmed in trending regime should have regime boost
    assert np.all(quality.bos_bullish_confirmed_quality[10:15] > _BASE_BOS_CONFIRMED)
    assert np.all(quality.bos_bullish_confirmed_quality[10:15] <= _MAX_QUALITY)

    # Bearish confirmed in ranging regime should have less/no regime boost (trending preference)
    # Base score still applies
    assert np.all(quality.bos_bearish_confirmed_quality[60:65] >= _BASE_BOS_CONFIRMED)

    # No signals elsewhere
    assert np.all(quality.bos_bullish_confirmed_quality[:10] == 0.0)
    assert np.all(quality.bos_bullish_confirmed_quality[15:] == 0.0)


def test_score_signals_choch_signals(base_config, sample_arrays, sample_signals):
    """Positive test: CHOCH signals get reversal scoring."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    raw_signals, validated_signals = sample_signals

    # Add raw CHOCH signals
    raw_signals.is_choch_bullish[40:45] = True  # Ranging regime (good for reversal)
    raw_signals.is_choch_bearish[20:25] = True  # Trending regime (bad for reversal)

    # Mark some as failed
    validated_signals.is_failed_choch_bullish[42] = True  # This one fails
    validated_signals.is_failed_choch_bearish[22] = True  # This one fails

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config
    )

    # CHOCH bullish in ranging regime should have regime boost (except failed one)
    assert quality.choch_bullish_quality[40] > _BASE_CHOCH  # Good regime
    assert quality.choch_bullish_quality[42] == 0.0  # Failed CHOCH

    # CHOCH bearish in trending regime should have less/no regime boost
    assert quality.choch_bearish_quality[20] >= _BASE_CHOCH  # Base score
    assert quality.choch_bearish_quality[22] == 0.0  # Failed CHOCH

    # Failed CHOCH should get minimal quality in failed arrays
    assert quality.failed_choch_bullish_quality[42] == _BASE_FAILED
    assert quality.failed_choch_bearish_quality[22] == _BASE_FAILED


def test_score_signals_failed_signals(base_config, sample_arrays, sample_signals):
    """Positive test: Failed signals get minimal quality."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    raw_signals, validated_signals = sample_signals

    # Add failed signals
    validated_signals.is_bullish_break_failure[30] = True
    validated_signals.is_bullish_immediate_failure[31] = True
    validated_signals.is_bearish_break_failure[70] = True

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config
    )

    # Failed signals should get minimal quality
    assert quality.failed_bullish_quality[30] == _BASE_FAILED
    assert quality.failed_bullish_quality[31] == _BASE_FAILED
    assert quality.failed_bearish_quality[70] == _BASE_FAILED

    # Non-failed positions should be zero
    assert quality.failed_bullish_quality[0] == 0.0
    assert quality.failed_bearish_quality[0] == 0.0


def test_score_signals_all_signal_types(base_config):
    """Positive test: Comprehensive test with all signal types."""
    n = 20
    market_regime = np.array(['strong_trend'] * 10 + ['ranging'] * 10, dtype='U20')
    session = np.array(['ny'] * n, dtype='U20')
    liquidity_score = np.full(n, 0.7, dtype=np.float32)
    zone_confluence = np.full(n, 0.6, dtype=np.float32)

    # Raw signals
    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(n, dtype=bool),
        is_bos_bearish_initial=np.zeros(n, dtype=bool),
        is_choch_bullish=np.array([False] * 15 + [True] * 5, dtype=bool),  # In ranging
        is_choch_bearish=np.array([True] * 5 + [False] * 15, dtype=bool)  # In trending
    )

    # Validated signals
    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.array([True] * 3 + [False] * 17, dtype=bool),  # In trending
        is_bos_bearish_confirmed=np.array([False] * 12 + [True] * 3 + [False] * 5, dtype=bool),  # In ranging
        is_bos_bullish_momentum=np.array([False] * 5 + [True] * 2 + [False] * 13, dtype=bool),  # In trending
        is_bos_bearish_momentum=np.array([False] * 14 + [True] * 2 + [False] * 4, dtype=bool),  # In ranging
        is_bullish_break_failure=np.array([False] * 8 + [True] + [False] * 11, dtype=bool),
        is_bearish_break_failure=np.array([False] * 16 + [True] + [False] * 3, dtype=bool),
        is_bullish_immediate_failure=np.array([False] * 9 + [True] + [False] * 10, dtype=bool),
        is_bearish_immediate_failure=np.array([False] * 17 + [True] + [False] * 2, dtype=bool),
        is_failed_choch_bullish=np.array([False] * 18 + [True] + [False], dtype=bool),  # One failed CHOCH bull
        is_failed_choch_bearish=np.array([False] * 4 + [True] + [False] * 15, dtype=bool)  # One failed CHOCH bear
    )

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config
    )

    # Verify all signal types have appropriate scores
    assert isinstance(quality, SignalQuality)

    # BOS confirmed in trending should have highest scores
    assert np.all(quality.bos_bullish_confirmed_quality[:3] > _BASE_BOS_CONFIRMED)

    # CHOCH in favorable regimes should have good scores (except failed ones)
    # Position 18 has failed_choch_bullish = True, so it should be 0.0
    choch_bullish_scores = quality.choch_bullish_quality[15:]  # Positions 15-19
    # Check non-failed positions (15, 16, 17, 19)
    non_failed_indices = [0, 1, 2, 4]  # Relative to the slice
    assert np.all(choch_bullish_scores[non_failed_indices] >= _BASE_CHOCH)
    # Check failed position (18, which is index 3 in the slice)
    assert choch_bullish_scores[3] == 0.0


# ==============================================================================
# NEGATIVE TESTS: score_signals
# ==============================================================================

def test_score_signals_invalid_config():
    """Negative test: Invalid config parameter."""
    # Test that invalid config cannot be created
    with pytest.raises(ValueError, match="regime_boost must be in"):
        invalid_config = SignalQualityConfig(
            session_weights={'ny': 1.0},
            regime_boost=0.5,  # Too high (> 0.3)
            zone_boost=0.15,
            liquidity_boost=0.1,
            session_boost_scale=0.1
        )
        # This line will never execute if the above raises
        score_signals(
            raw_signals=np.array([]),
            validated_signals=ValidatedSignals(*[np.array([]) for _ in range(10)]),
            market_regime=np.array([]),
            session=np.array([]),
            liquidity_score=np.array([]),
            zone_confluence=np.array([]),
            config=invalid_config
        )

def test_score_signals_empty_session_weights(sample_arrays, sample_signals):
    """Negative test: Empty session weights."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    raw_signals, validated_signals = sample_signals

    invalid_config = SignalQualityConfig(session_weights={})  # Empty

    with pytest.raises(ValueError, match="session_weights cannot be empty"):
        score_signals(
            raw_signals=raw_signals,
            validated_signals=validated_signals,
            market_regime=market_regime,
            session=session,
            liquidity_score=liquidity_score,
            zone_confluence=zone_confluence,
            config=invalid_config
        )


# ==============================================================================
# BOUNDARY TESTS
# ==============================================================================
def test_score_signals_minimal_config(sample_arrays, sample_signals):
    """Boundary test: Minimal config with all zeros."""
    market_regime, original_session, liquidity_score, zone_confluence = sample_arrays
    raw_signals, validated_signals = sample_signals

    # Create session array with only 'ny' to match minimal config
    n = len(market_regime)
    session = np.array(['ny'] * n, dtype='U20')

    # Add one signal of each type
    validated_signals.is_bos_bullish_confirmed[0] = True
    validated_signals.is_bos_bullish_momentum[1] = True
    raw_signals.is_choch_bullish[2] = True
    validated_signals.is_bullish_break_failure[3] = True

    minimal_config = SignalQualityConfig(
        session_weights={'ny': 1.0},
        regime_boost=0.0,
        zone_boost=0.0,
        liquidity_boost=0.0,
        session_boost_scale=0.0
    )

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=minimal_config
    )

    # All scores should be exactly base scores (no boosts)
    assert quality.bos_bullish_confirmed_quality[0] == _BASE_BOS_CONFIRMED
    assert quality.bos_bullish_momentum_quality[1] == _BASE_BOS_MOMENTUM
    assert quality.choch_bullish_quality[2] == _BASE_CHOCH

    # Failed signals get minimal quality (0.1), not 0.0
    assert quality.failed_bullish_quality[3] == 0.1  # Changed from 0.0 to 0.1

    # All other positions should be zero (except our signal positions)
    # Check first 10 positions for simplicity
    for i in range(10):
        if i in [0, 1, 2, 3]:  # Skip signal positions
            continue
        assert quality.bos_bullish_confirmed_quality[i] == 0.0
        assert quality.bos_bullish_momentum_quality[i] == 0.0
        assert quality.choch_bullish_quality[i] == 0.0
        assert quality.failed_bullish_quality[i] == 0.0

def test_score_signals_maximum_boost():
    """Boundary test: Maximum boost values produce max quality."""
    n = 5
    market_regime = np.array(['strong_trend'] * n, dtype='U20')  # Trending for BOS
    session = np.array(['overlap'] * n, dtype='U20')  # Highest weight in config
    liquidity_score = np.ones(n, dtype=np.float32)
    zone_confluence = np.ones(n, dtype=np.float32)

    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(n, dtype=bool),
        is_bos_bearish_initial=np.zeros(n, dtype=bool),
        is_choch_bullish=np.zeros(n, dtype=bool),
        is_choch_bearish=np.zeros(n, dtype=bool)
    )

    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.ones(n, dtype=bool),  # All signals
        is_bos_bearish_confirmed=np.zeros(n, dtype=bool),
        is_bos_bullish_momentum=np.zeros(n, dtype=bool),
        is_bos_bearish_momentum=np.zeros(n, dtype=bool),
        is_bullish_break_failure=np.zeros(n, dtype=bool),
        is_bearish_break_failure=np.zeros(n, dtype=bool),
        is_bullish_immediate_failure=np.zeros(n, dtype=bool),
        is_bearish_immediate_failure=np.zeros(n, dtype=bool),
        is_failed_choch_bullish=np.zeros(n, dtype=bool),
        is_failed_choch_bearish=np.zeros(n, dtype=bool)
    )

    max_config = SignalQualityConfig(
        session_weights={'overlap': 2.0},  # Max reasonable weight
        regime_boost=0.3,  # Max
        zone_boost=0.3,  # Max
        liquidity_boost=0.2,  # Max
        session_boost_scale=0.2  # Max
    )

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=max_config
    )

    # All scores should be maxed at 1.0
    assert np.all(quality.bos_bullish_confirmed_quality == 1.0)


def test_score_signals_single_bar():
    """Boundary test: Single bar arrays."""
    market_regime = np.array(['strong_trend'], dtype='U20')
    session = np.array(['ny'], dtype='U20')
    liquidity_score = np.array([0.5], dtype=np.float32)
    zone_confluence = np.array([0.5], dtype=np.float32)

    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(1, dtype=bool),
        is_bos_bearish_initial=np.zeros(1, dtype=bool),
        is_choch_bullish=np.ones(1, dtype=bool),
        is_choch_bearish=np.zeros(1, dtype=bool)
    )

    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.ones(1, dtype=bool),
        is_bos_bearish_confirmed=np.zeros(1, dtype=bool),
        is_bos_bullish_momentum=np.zeros(1, dtype=bool),
        is_bos_bearish_momentum=np.zeros(1, dtype=bool),
        is_bullish_break_failure=np.zeros(1, dtype=bool),
        is_bearish_break_failure=np.zeros(1, dtype=bool),
        is_bullish_immediate_failure=np.zeros(1, dtype=bool),
        is_bearish_immediate_failure=np.zeros(1, dtype=bool),
        is_failed_choch_bullish=np.zeros(1, dtype=bool),
        is_failed_choch_bearish=np.zeros(1, dtype=bool)
    )

    config = SignalQualityConfig(
        session_weights={'ny': 1.0},
        regime_boost=0.0,
        zone_boost=0.0,
        liquidity_boost=0.0,
        session_boost_scale=0.0
    )

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=config
    )

    assert quality.bos_bullish_confirmed_quality[0] == _BASE_BOS_CONFIRMED
    assert quality.choch_bullish_quality[0] == _BASE_CHOCH


# ==============================================================================
# PROPERTY TESTS
# ==============================================================================

def test_quality_scores_range(sample_arrays, sample_signals, base_config):
    """Property test: All quality scores are in range [0, 1]."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    raw_signals, validated_signals = sample_signals

    # Add various signals
    validated_signals.is_bos_bullish_confirmed[10:20] = True
    validated_signals.is_bos_bullish_momentum[30:40] = True
    raw_signals.is_choch_bullish[50:60] = True
    validated_signals.is_bullish_break_failure[70] = True

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config
    )

    # Check all fields are in valid range
    for field_name, field_value in quality.__dict__.items():
        assert np.all((field_value >= 0) & (field_value <= 1.0)), \
            f"Field {field_name} has values outside [0, 1]"


def test_quality_array_lengths(sample_arrays, sample_signals, base_config):
    """Property test: All quality arrays have same length as input."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    raw_signals, validated_signals = sample_signals

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config
    )

    n = len(market_regime)
    for field_name, field_value in quality.__dict__.items():
        assert len(field_value) == n, f"Field {field_name} has wrong length"


def test_failed_signals_minimal_quality(sample_arrays, sample_signals, base_config):
    """Property test: Failed signals always get minimal quality."""
    market_regime, session, liquidity_score, zone_confluence = sample_arrays
    raw_signals, validated_signals = sample_signals

    # Add failed signals in various contexts
    validated_signals.is_bullish_break_failure[10] = True
    validated_signals.is_bearish_break_failure[20] = True
    validated_signals.is_failed_choch_bullish[30] = True

    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=base_config
    )

    # Failed signals should always get minimal quality
    assert quality.failed_bullish_quality[10] == _BASE_FAILED
    assert quality.failed_bearish_quality[20] == _BASE_FAILED
    assert quality.failed_choch_bullish_quality[30] == _BASE_FAILED

    # Non-failed positions should be zero
    assert quality.failed_bullish_quality[0] == 0.0
    assert quality.failed_bearish_quality[0] == 0.0


# ==============================================================================
# INTEGRATION TEST: Realistic Scenario
# ==============================================================================

def test_realistic_trading_scenario():
    """Integration test: Realistic trading scenario with mixed signals."""
    n = 200  # 200 bars

    # Create realistic market regime sequence
    market_regime = np.array(
        ['strong_trend'] * 50 +
        ['weak_trend'] * 30 +
        ['ranging'] * 40 +
        ['chop'] * 30 +
        ['neutral'] * 50,
        dtype='U20'
    )

    # Create session pattern (NY, London, Asia overlap)
    session = np.array(
        ['asia'] * 20 +
        ['overlap'] * 10 +
        ['london'] * 30 +
        ['overlap'] * 10 +
        ['ny'] * 40 +
        ['overlap'] * 10 +
        ['london'] * 20 +
        ['asia'] * 60,
        dtype='U20'
    )

    # Realistic liquidity pattern (higher during overlaps)
    liquidity_score = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.3 + 0.5
    liquidity_score = np.clip(liquidity_score, 0, 1).astype(np.float32)

    # Zone confluence (random with some structure)
    zone_confluence = np.random.rand(n).astype(np.float32) * 0.8 + 0.2

    # Raw signals - more during trends and ranges
    raw_signals = RawSignals(
        is_bos_bullish_initial=np.zeros(n, dtype=bool),
        is_bos_bearish_initial=np.zeros(n, dtype=bool),
        is_choch_bullish=np.random.rand(n) > 0.95,  # 5% CHOCH bullish
        is_choch_bearish=np.random.rand(n) > 0.95  # 5% CHOCH bearish
    )

    # Validated signals - fewer than raw (validation filters)
    validated_signals = ValidatedSignals(
        is_bos_bullish_confirmed=np.random.rand(n) > 0.98,  # 2% confirmed BOS
        is_bos_bearish_confirmed=np.random.rand(n) > 0.98,  # 2% confirmed BOS
        is_bos_bullish_momentum=np.random.rand(n) > 0.97,  # 3% momentum BOS
        is_bos_bearish_momentum=np.random.rand(n) > 0.97,  # 3% momentum BOS
        is_bullish_break_failure=np.random.rand(n) > 0.99,  # 1% failures
        is_bearish_break_failure=np.random.rand(n) > 0.99,  # 1% failures
        is_bullish_immediate_failure=np.random.rand(n) > 0.99,  # 1% immediate failures
        is_bearish_immediate_failure=np.random.rand(n) > 0.99,  # 1% immediate failures
        is_failed_choch_bullish=np.random.rand(n) > 0.99,  # 1% failed CHOCH
        is_failed_choch_bearish=np.random.rand(n) > 0.99  # 1% failed CHOCH
    )

    config = SignalQualityConfig()

    # Should not raise
    quality = score_signals(
        raw_signals=raw_signals,
        validated_signals=validated_signals,
        market_regime=market_regime,
        session=session,
        liquidity_score=liquidity_score,
        zone_confluence=zone_confluence,
        config=config
    )

    # Verify outputs
    assert isinstance(quality, SignalQuality)

    # Check all quality arrays exist and have correct length
    for field_name, field_value in quality.__dict__.items():
        assert field_value is not None
        assert len(field_value) == n
        assert np.all((field_value >= 0) & (field_value <= 1.0))

    # Verify some signals got scored
    assert np.any(quality.bos_bullish_confirmed_quality > 0)
    assert np.any(quality.choch_bullish_quality > 0)
    assert np.any(quality.failed_bullish_quality > 0)

    print(f"\nRealistic scenario stats:")
    print(f"BOS confirmed quality range: [{np.min(quality.bos_bullish_confirmed_quality):.3f}, "
          f"{np.max(quality.bos_bullish_confirmed_quality):.3f}]")
    print(f"CHOCH quality range: [{np.min(quality.choch_bullish_quality):.3f}, "
          f"{np.max(quality.choch_bullish_quality):.3f}]")
    print(f"Failed signals: {np.sum(quality.failed_bullish_quality > 0)}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])