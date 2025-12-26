import numpy as np
from typing import TypeAlias
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int8]

Prices = FloatArray
ATRArray = FloatArray
SwingsMask = BoolArray
StructureLabel = BoolArray
TrendStateArray = IntArray
SignalMask = BoolArray
BreakBuffer = FloatArray