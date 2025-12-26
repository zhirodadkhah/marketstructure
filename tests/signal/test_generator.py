import pytest
import numpy as np
from unittest.mock import patch
from structure.signal.generator import generate_raw_signals
from structure.signal.config import SignalGeneratorConfig


def test_generator_positive_calls_detector():
    n = 10
    with patch('structure.signal.generator.detect_structure_break_signals') as mock:
        mock.return_value = {
            'is_bos_bullish_initial': np.zeros(n, bool),
            'is_bos_bearish_initial': np.zeros(n, bool)
        }
        config = SignalGeneratorConfig()
        result = generate_raw_signals(
            high=np.arange(n, dtype=np.float32) + 100,
            low=np.arange(n, dtype=np.float32) + 95,
            close=np.arange(n, dtype=np.float32) + 97,
            open_=np.arange(n, dtype=np.float32) + 96,
            atr=np.ones(n, np.float32),
            is_swing_high=np.zeros(n, bool),
            is_swing_low=np.zeros(n, bool),
            is_higher_high=np.zeros(n, bool),
            is_lower_low=np.zeros(n, bool),
            trend_state=np.ones(n, dtype=np.int8),
            config=config
        )
        mock.assert_called_once()
        assert 'is_bos_bullish_initial' in result


def test_generator_negative_mismatched_shapes():
    config = SignalGeneratorConfig()
    with pytest.raises(Exception):  # likely from detect_structure_break_signals
        generate_raw_signals(
            high=np.array([1, 2, 3], np.float32),
            low=np.array([1, 2], np.float32),  # mismatch
            close=np.array([1, 2, 3], np.float32),
            open_=np.array([1, 2, 3], np.float32),
            atr=np.ones(3, np.float32),
            is_swing_high=np.zeros(3, bool),
            is_swing_low=np.zeros(3, bool),
            is_higher_high=np.zeros(3, bool),
            is_lower_low=np.zeros(3, bool),
            trend_state=np.ones(3, dtype=np.int8),
            config=config
        )


def test_generator_edge_single_candle():
    config = SignalGeneratorConfig()
    with patch('structure.signal.generator.detect_structure_break_signals') as mock:
        mock.return_value = {'is_bos_bullish_initial': np.array([False]), 'is_bos_bearish_initial': np.array([False])}
        result = generate_raw_signals(
            high=np.array([100.]), low=np.array([99.]), close=np.array([99.5]), open_=np.array([99.]),
            atr=np.array([1.]), is_swing_high=np.array([False]), is_swing_low=np.array([False]),
            is_higher_high=np.array([False]), is_lower_low=np.array([False]),
            trend_state=np.array([0], dtype=np.int8),
            config=config
        )
        assert len(result['is_bos_bullish_initial']) == 1