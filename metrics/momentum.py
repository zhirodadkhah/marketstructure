import numpy as np
from .types import Prices, ATRArray, SwingsMask


def compute_momentum_metrics(
    close: Prices,
    atr: ATRArray,
    swing_high_mask: SwingsMask,
    swing_low_mask: SwingsMask,
    momentum_period: int = 1,
    acceleration_period: int = 1,
    normalize_by_atr: bool = True,
    max_divergence_lookback: int = 20
) -> dict[str, np.ndarray]:
    """
    Compute momentum, acceleration, and divergence signals.

    Args:
        close: Closing prices (1D float32 array).
        atr: Average True Range for normalization.
        swing_high_mask: Boolean mask of swing highs.
        swing_low_mask: Boolean mask of swing lows.
        momentum_period: Lookback for momentum (default=1 → 1-bar delta).
        acceleration_period: Lookback for acceleration (default=1).
        normalize_by_atr: Whether to divide by ATR for scale invariance.
        max_divergence_lookback: Max bars to look back for divergence.

    Returns:
        dict: Contains:
            - 'momentum', 'acceleration': raw differences
            - 'normalized_momentum', 'normalized_acceleration'
            - 'momentum_direction': sign as int8 (-1, 0, +1)
            - 'momentum_divergence_bullish', 'momentum_divergence_bearish': bool masks

    Notes:
        - Divergence logic:
            - Bearish: higher high + lower momentum
            - Bullish: lower low + higher momentum
        - All outputs same length as input.
        - Uses safe ATR (clamped at 1e-10) to avoid division by zero.
    """
    n = len(close)
    close = close.astype(np.float32, copy=False)
    atr = atr.astype(np.float32, copy=False)

    # === MOMENTUM ===
    momentum = np.zeros(n, dtype=np.float32)
    if momentum_period == 1:
        momentum[1:] = close[1:] - close[:-1]
    elif momentum_period < n:
        momentum[momentum_period:] = close[momentum_period:] - close[:-momentum_period]

    # === ACCELERATION ===
    acceleration = np.zeros(n, dtype=np.float32)
    if acceleration_period == 1 and n > 2:
        acceleration[2:] = momentum[2:] - momentum[1:-1]
    elif acceleration_period < n - 1:
        acceleration[acceleration_period + 1:] = (
            momentum[acceleration_period + 1:] - momentum[1:-acceleration_period]
        )

    # === NORMALIZATION ===
    safe_atr = np.where(atr > 1e-10, atr, 1e-10)
    norm_mom = momentum / safe_atr if normalize_by_atr else momentum
    norm_acc = acceleration / safe_atr if normalize_by_atr else acceleration

    # === DIVERGENCE DETECTION (SPARSE) ===
    def _find_prev(mask: np.ndarray, idx: int, lookback: int) -> int | None:
        """Find most recent True in mask before idx within lookback."""
        start = max(0, idx - lookback)
        hits = np.where(mask[start:idx])[0]
        return int(hits[-1] + start) if len(hits) else None

    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)

    for i in range(5, n):
        # Bearish divergence: price ↑, momentum ↓
        if swing_high_mask[i]:
            p = _find_prev(swing_high_mask, i, max_divergence_lookback)
            if p is not None and close[i] > close[p] and norm_mom[i] < norm_mom[p]:
                bearish_div[i] = True
        # Bullish divergence: price ↓, momentum ↑
        if swing_low_mask[i]:
            p = _find_prev(swing_low_mask, i, max_divergence_lookback)
            if p is not None and close[i] < close[p] and norm_mom[i] > norm_mom[p]:
                bullish_div[i] = True

    return {
        'momentum': momentum,
        'acceleration': acceleration,
        'normalized_momentum': norm_mom,
        'normalized_acceleration': norm_acc,
        'momentum_direction': np.sign(momentum).astype(np.int8),
        'momentum_divergence_bullish': bullish_div,
        'momentum_divergence_bearish': bearish_div,
    }