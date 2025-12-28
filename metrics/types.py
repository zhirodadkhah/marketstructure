from dataclasses import dataclass

import numpy as np
from typing import TypeAlias, TypedDict
from numpy.typing import NDArray

LongFloatArray: TypeAlias = NDArray[np.float64]
FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int8]


Timestamps = LongFloatArray

TrendStateArray = IntArray

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
class RawSignals:
    is_bos_bullish_initial: SignalMask
    is_bos_bearish_initial: SignalMask
    is_choch_bullish: SignalMask
    is_choch_bearish: SignalMask


@dataclass
class ValidatedSignals:
    is_bos_bullish_confirmed: SignalMask
    is_bos_bearish_confirmed: SignalMask
    is_bos_bullish_momentum: SignalMask
    is_bos_bearish_momentum: SignalMask
    is_bullish_break_failure: SignalMask
    is_bearish_break_failure: SignalMask


@dataclass
class SignalQuality:
    bos_bullish_quality: FloatArray
    bos_bearish_quality: FloatArray
    choch_bullish_quality: FloatArray
    choch_bearish_quality: FloatArray