
from enum import IntEnum
from typing import Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from structure.failures.config import StructureBreakConfig

SIGNAL_COLS = [
    'is_bos_bullish_initial', 'is_bos_bearish_initial',
    'is_bos_bullish_confirmed', 'is_bos_bearish_confirmed',
    'is_bos_bullish_momentum', 'is_bos_bearish_momentum',
    'is_choch_bullish', 'is_choch_bearish',
    'is_failed_choch_bullish', 'is_failed_choch_bearish',
    'is_bullish_break_failure', 'is_bearish_break_failure',
    'is_bullish_immediate_failure', 'is_bearish_immediate_failure'
]


class ResultBuilder:
    __slots__ = ('length', 'signals')

    def __init__(self, length: int):
        self.length = length
        self.signals = {col: np.zeros(length, dtype=bool) for col in SIGNAL_COLS}

    def set_signal(self, col: str, idx: int) -> None:
        if 0 <= idx < self.length:
            self.signals[col][idx] = True

    def build(self, original_df: pd.DataFrame) -> pd.DataFrame:
        result = original_df.copy()
        for col, arr in self.signals.items():
            result[col] = arr
        return result
    
@dataclass
class LevelState(IntEnum):
    BROKEN = 0
    CONFIRMED = 1
    FAILED_IMMEDIATE = 2
    FAILED_RETEST = 3
    MOMENTUM = 4


class BreakLevel:
    __slots__ = (
        'swing_idx', 'price', 'direction', 'role', 'break_idx', 'atr_at_break',
        'is_gap_break', 'max_post_break_high', 'min_post_break_low',
        'retest_active', 'retest_start_idx', 'state', 'buffer',
        'moved_away_distance', 'retest_attempts'
    )

    def __init__(
            self,
            swing_idx: int,
            price: float,
            direction: str,
            role: str,
            break_idx: int,
            atr_at_break: float,
            is_gap_break: bool,
            config: StructureBreakConfig
    ):
        self.swing_idx = swing_idx
        self.price = price
        self.direction = direction
        self.role = role
        self.break_idx = break_idx
        self.atr_at_break = atr_at_break
        self.is_gap_break = is_gap_break
        self.buffer = atr_at_break * config.buffer_multiplier
        self.retest_active = False
        self.retest_start_idx: Optional[int] = None
        self.state = LevelState.BROKEN
        self.retest_attempts = 0

        # Initialize extremes; gap breaks still track real price movement
        if is_gap_break:
            self.moved_away_distance = config.min_break_atr_mult * atr_at_break

        else:
            self.moved_away_distance = 0.0

        # Keep extremes as -inf/inf for both cases
        self.max_post_break_high = -np.inf
        self.min_post_break_low = np.inf


@dataclass
class BreakTarget:
    """Represents a potential break target."""
    role: str  # 'bos_bull', 'bos_bear', 'choch_bear', 'choch_bull'
    direction: str  # 'bullish' or 'bearish'
    price: float
    idx: int
    is_valid: bool = True