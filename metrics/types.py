from dataclasses import dataclass

import numpy as np
from typing import TypeAlias, TypedDict
from numpy.typing import NDArray

FloatLongArray: TypeAlias = NDArray[np.float64]
FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int32]
IntTinyArray: TypeAlias = NDArray[np.int8]



TrendStateArray = IntTinyArray

Timestamps = FloatLongArray

Prices      = FloatArray
ATRArray    = FloatArray       # VolatilityArray, ATR is volatility in price units
BreakBuffer = FloatArray    # DistanceArray, offset in price units
ZoneArray   = Prices          # PriceLevelArray, actual price levels (S/R)

SwingsMask      = BoolArray
StructureLabel  = BoolArray
SignalMask      = BoolArray
RegimeMask      = BoolArray
SessionMask     = BoolArray

@dataclass
class RetestMetrics:
    retest_velocity: FloatArray
    bars_to_retest: IntArray
    pullback_distance: FloatArray
    is_fast_retest: BoolArray
    is_slow_retest: BoolArray
    retest_attempts: IntArray
    # Extended fields for full metrics
    retest_close: FloatArray
    retest_indices: IntArray
    break_levels: FloatArray
    direction: str


@dataclass
class RawSignals:
    is_bos_bullish_initial: SignalMask
    is_bos_bearish_initial: SignalMask
    is_choch_bullish: SignalMask
    is_choch_bearish: SignalMask

# structure/metrics/types.py
@dataclass
class ValidatedSignals:
    is_bos_bullish_confirmed: SignalMask
    is_bos_bearish_confirmed: SignalMask
    is_bos_bullish_momentum: SignalMask
    is_bos_bearish_momentum: SignalMask
    is_bullish_break_failure: SignalMask
    is_bearish_break_failure: SignalMask
    # ➕ NEW: Immediate & CHOCH failures
    is_bullish_immediate_failure: SignalMask
    is_bearish_immediate_failure: SignalMask
    is_failed_choch_bullish: SignalMask
    is_failed_choch_bearish: SignalMask


@dataclass
class SignalQuality:
    """Quality scores [0.1, 1.0] for all signal types."""

    # BOS confirmed signals (trend continuation with retest)
    bos_bullish_confirmed_quality: FloatArray
    bos_bearish_confirmed_quality: FloatArray

    # BOS momentum signals (trend continuation without retest)
    bos_bullish_momentum_quality: FloatArray
    bos_bearish_momentum_quality: FloatArray

    # CHOCH reversal signals (valid reversals)
    choch_bullish_quality: FloatArray
    choch_bearish_quality: FloatArray

    # Failed signals (minimal quality for filtering)
    failed_bullish_quality: FloatArray  # Failed BOS bullish
    failed_bearish_quality: FloatArray  # Failed BOS bearish
    failed_choch_bullish_quality: FloatArray  # Failed CHOCH bullish
    failed_choch_bearish_quality: FloatArray  # Failed CHOCH bearish

    def __post_init__(self):
        """Validate all quality arrays."""
        arrays = [
            self.bos_bullish_confirmed_quality,
            self.bos_bearish_confirmed_quality,
            self.bos_bullish_momentum_quality,
            self.bos_bearish_momentum_quality,
            self.choch_bullish_quality,
            self.choch_bearish_quality,
            self.failed_bullish_quality,
            self.failed_bearish_quality,
            self.failed_choch_bullish_quality,
            self.failed_choch_bearish_quality
        ]

        # Check all arrays have same length
        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) != 1:
            field_names = list(self.__annotations__.keys())
            length_map = dict(zip(field_names, lengths))
            raise ValueError(
                f"All quality arrays must have same length. Got: {length_map}"
            )

        # Validate score ranges
        for name, arr in self.__dict__.items():
            if np.any((arr < 0) | (arr > 1)):
                min_val = np.min(arr)
                max_val = np.max(arr)
                raise ValueError(
                    f"{name} must be in range [0, 1], got [{min_val:.3f}, {max_val:.3f}]"
                )