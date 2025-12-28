# tests/signal/test_generator.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from structure.signal.generator import generate_raw_signals
from structure.signal.config import SignalGeneratorConfig


def test_generator_positive():
    """Positive: Generator should call detector and return complete RawSignals."""
    n = 10
    with patch('structure.signal.generator.detect_structure_break_signals') as mock:
        # Mock should return ALL fields that RawSignals expects
        mock.return_value = {
            'is_bos_bullish_initial': np.zeros(n, bool),
            'is_bos_bearish_initial': np.zeros(n, bool),
            'is_choch_bullish': np.zeros(n, bool),  # Added this
            'is_choch_bearish': np.zeros(n, bool)  # Added this
        }
        config = SignalGeneratorConfig()

        result = generate_raw_signals(
            high=np.arange(n, dtype=np.float32),
            low=np.arange(n, dtype=np.float32) - 1,
            close=np.arange(n, dtype=np.float32) - 0.5,
            open_=np.arange(n, dtype=np.float32) - 0.6,
            atr=np.ones(n, np.float32),
            is_swing_high=np.zeros(n, bool),
            is_swing_low=np.zeros(n, bool),
            is_higher_high=np.zeros(n, bool),
            is_lower_low=np.zeros(n, bool),
            trend_state=np.ones(n, dtype=np.int8),
            config=config
        )

        # Verify detector was called with correct parameters
        mock.assert_called_once()

        # Check that config values were passed
        call_kwargs = mock.call_args[1]
        assert call_kwargs['min_break_atr_mult'] == 0.5
        assert call_kwargs['buffer_multiplier'] == 0.5

        # Verify result has all expected fields
        assert hasattr(result, 'is_bos_bullish_initial')
        assert hasattr(result, 'is_bos_bearish_initial')
        assert hasattr(result, 'is_choch_bullish')
        assert hasattr(result, 'is_choch_bearish')

        # All signals should be False in our mock
        assert not result.is_bos_bullish_initial.any()
        assert not result.is_bos_bearish_initial.any()
        assert not result.is_choch_bullish.any()
        assert not result.is_choch_bearish.any()


def test_generator_negative_mismatched_shapes():
    """Negative: Mismatched array shapes should raise error."""
    config = SignalGeneratorConfig()

    with patch('structure.signal.generator.detect_structure_break_signals') as mock:
        # Mock should not be called if shapes mismatch
        mock.side_effect = Exception("Should not be called")

        with pytest.raises(Exception):
            generate_raw_signals(
                high=np.array([100, 101], dtype=np.float32),
                low=np.array([99], dtype=np.float32),  # Wrong length
                close=np.array([99.5, 100.5], dtype=np.float32),
                open_=np.array([99, 100], dtype=np.float32),
                atr=np.array([1.0, 1.0], dtype=np.float32),
                is_swing_high=np.array([False, False], dtype=bool),
                is_swing_low=np.array([False, False], dtype=bool),
                is_higher_high=np.array([False, False], dtype=bool),
                is_lower_low=np.array([False, False], dtype=bool),
                trend_state=np.array([0, 0]),
                config=config
            )


def test_generator_edge_empty():
    """Edge: Empty arrays should work."""
    config = SignalGeneratorConfig()

    with patch('structure.signal.generator.detect_structure_break_signals') as mock:
        mock.return_value = {
            'is_bos_bullish_initial': np.array([], bool),
            'is_bos_bearish_initial': np.array([], bool),
            'is_choch_bullish': np.array([], bool),  # Added this
            'is_choch_bearish': np.array([], bool)  # Added this
        }

        result = generate_raw_signals(
            high=np.array([], np.float32),
            low=np.array([], np.float32),
            close=np.array([], np.float32),
            open_=np.array([], np.float32),
            atr=np.array([], np.float32),
            is_swing_high=np.array([], bool),
            is_swing_low=np.array([], bool),
            is_higher_high=np.array([], bool),
            is_lower_low=np.array([], bool),
            trend_state=np.array([], dtype=np.int8),
            config=config
        )

        # Should handle empty arrays without crashing
        assert len(result.is_bos_bullish_initial) == 0
        assert len(result.is_bos_bearish_initial) == 0
        assert len(result.is_choch_bullish) == 0
        assert len(result.is_choch_bearish) == 0

        # Verify detector was called
        mock.assert_called_once()


def test_generator_with_realistic_signals():
    """Test with realistic signal patterns."""
    n = 20
    config = SignalGeneratorConfig(min_break_atr_mult=0.8, buffer_multiplier=0.3)

    with patch('structure.signal.generator.detect_structure_break_signals') as mock:
        # Create realistic signal pattern
        bos_bullish = np.zeros(n, bool)
        bos_bullish[5] = True
        bos_bullish[12] = True

        bos_bearish = np.zeros(n, bool)
        bos_bearish[8] = True
        bos_bearish[15] = True

        choch_bullish = np.zeros(n, bool)
        choch_bullish[3] = True

        choch_bearish = np.zeros(n, bool)
        choch_bearish[10] = True

        mock.return_value = {
            'is_bos_bullish_initial': bos_bullish,
            'is_bos_bearish_initial': bos_bearish,
            'is_choch_bullish': choch_bullish,
            'is_choch_bearish': choch_bearish
        }

        # Create price data with some structure
        prices = np.linspace(100, 120, n)
        result = generate_raw_signals(
            high=prices + 2,
            low=prices - 2,
            close=prices,
            open_=prices - 0.5,
            atr=np.ones(n) * 1.5,
            is_swing_high=np.random.rand(n) > 0.8,
            is_swing_low=np.random.rand(n) > 0.8,
            is_higher_high=prices > np.roll(prices, 1),
            is_lower_low=prices < np.roll(prices, 1),
            trend_state=np.where(prices > np.mean(prices), 1, -1),
            config=config
        )

        # Verify signals were preserved
        assert result.is_bos_bullish_initial[5] == True
        assert result.is_bos_bullish_initial[12] == True
        assert result.is_bos_bearish_initial[8] == True
        assert result.is_choch_bullish[3] == True
        assert result.is_choch_bearish[10] == True

        # Verify config was passed correctly
        call_kwargs = mock.call_args[1]
        assert call_kwargs['min_break_atr_mult'] == 0.8
        assert call_kwargs['buffer_multiplier'] == 0.3


def test_generator_zero_atr():
    """Test with zero ATR values."""
    n = 5
    config = SignalGeneratorConfig()

    with patch('structure.signal.generator.detect_structure_break_signals') as mock:
        mock.return_value = {
            'is_bos_bullish_initial': np.array([True, False, True, False, False], bool),
            'is_bos_bearish_initial': np.array([False, True, False, False, True], bool),
            'is_choch_bullish': np.zeros(n, bool),
            'is_choch_bearish': np.zeros(n, bool)
        }

        result = generate_raw_signals(
            high=np.array([100, 101, 102, 103, 104], dtype=np.float32),
            low=np.array([99, 100, 101, 102, 103], dtype=np.float32),
            close=np.array([99.5, 100.5, 101.5, 102.5, 103.5], dtype=np.float32),
            open_=np.array([99, 100, 101, 102, 103], dtype=np.float32),
            atr=np.array([0, 0, 0, 0, 0], dtype=np.float32),  # Zero ATR
            is_swing_high=np.zeros(n, bool),
            is_swing_low=np.zeros(n, bool),
            is_higher_high=np.zeros(n, bool),
            is_lower_low=np.zeros(n, bool),
            trend_state=np.ones(n, dtype=np.int8),
            config=config
        )

        # Should still return signals even with zero ATR
        assert result.is_bos_bullish_initial[0] == True
        assert result.is_bos_bearish_initial[1] == True