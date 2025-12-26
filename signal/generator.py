# structure/signal/test_generator.py
"""
Generate raw break signals from core price-action logic.
"""
from typing import Dict
import numpy as np
from structure.metrics.types import RawSignals
from .config import SignalGeneratorConfig
from structure.core.breaks import detect_structure_break_signals


def generate_raw_signals(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_: np.ndarray,
        atr: np.ndarray,
        is_swing_high: np.ndarray,
        is_swing_low: np.ndarray,
        is_higher_high: np.ndarray,
        is_lower_low: np.ndarray,
        trend_state: np.ndarray,
        config: SignalGeneratorConfig
) -> RawSignals:
    """
    Generate raw BOS/CHOCH signals.

    Cyclomatic complexity: 1
    Args: ≤ 11 (but grouped logically; all required for signal gen)
    """
    signals = detect_structure_break_signals(
        high=high,
        low=low,
        close=close,
        open_=open_,
        atr=atr,
        is_swing_high=is_swing_high,
        is_swing_low=is_swing_low,
        is_higher_high=is_higher_high,
        is_lower_low=is_lower_low,
        trend_state=trend_state,
        min_break_atr_mult=config.min_break_atr_mult,
        buffer_multiplier=config.buffer_multiplier
    )
    return RawSignals(**signals)  # type-safe dict