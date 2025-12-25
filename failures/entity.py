
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
    'is_bullish_immediate_failure', 'is_bearish_immediate_failure',
    'is_bos_bullish_failed_follow_through',
    'is_bos_bearish_failed_follow_through'
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
    FOLLOW_THROUGH_PENDING = 5  # waiting for confirmation


# In entity.py, update the BreakLevel class:

# In entity.py, replace the BreakLevel class with:
class BreakLevel:
    __slots__ = (
        'swing_idx', 'price', 'direction', 'role', 'break_idx', 'atr_at_break',
        'is_gap_break', 'max_post_break_high', 'min_post_break_low',
        'retest_active', 'retest_start_idx', 'state', 'buffer',
        'moved_away_distance', 'retest_attempts', 'follow_through_start',
        'follow_through_progress',
        # ➕ ZONE CONTEXT
        'zone_strength', 'retest_quality', 'is_confluence_zone',
        'retest_count', 'signal_zone_score',
        # ➕ GROUP 5: Temporal & Velocity slots
        'retest_velocity', 'bars_to_retest', 'pullback_distance',
        'retest_start_bar', 'retest_end_bar', 'retest_duration',
        'velocity_smoothed', 'is_fast_retest', 'is_slow_retest',
        'retest_quality_score'
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
            config: StructureBreakConfig,
            # ➕ ZONE CONTEXT
            zone_strength: int = 0,
            retest_quality: float = 0.0,
            is_confluence_zone: bool = False,
            retest_count: int = 0,
            signal_zone_score: float = 0.0
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
        self.follow_through_start = break_idx if not is_gap_break else None
        self.follow_through_progress = 0
        # ➕ ZONE CONTEXT
        self.zone_strength = zone_strength
        self.retest_quality = retest_quality
        self.is_confluence_zone = is_confluence_zone
        self.retest_count = retest_count
        self.signal_zone_score = signal_zone_score
        if is_gap_break:
            self.moved_away_distance = config.min_break_atr_mult * atr_at_break
        else:
            self.moved_away_distance = 0.0
        self.max_post_break_high = -np.inf
        self.min_post_break_low = np.inf
        # ➕ GROUP 5: Temporal & Velocity
        self.retest_velocity = 0.0
        self.bars_to_retest = 0
        self.pullback_distance = 0.0
        self.retest_start_bar = None
        self.retest_end_bar = None
        self.retest_duration = 0
        self.velocity_smoothed = 0.0
        self.is_fast_retest = False
        self.is_slow_retest = False
        self.retest_quality_score = 0.0

@dataclass
class BreakTarget:
    """Represents a potential break target."""
    role: str  # 'bos_bull', 'bos_bear', 'choch_bear', 'choch_bull'
    direction: str  # 'bullish' or 'bearish'
    price: float
    idx: int
    is_valid: bool = True