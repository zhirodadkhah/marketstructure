# structure/context/mtf.py
"""
Multi-timeframe (MTF) context with confluence scoring.
Pure NumPy, no pandas in core logic.
"""
from typing import Dict, Tuple, Optional
import numpy as np
from structure.metrics.types import (
    Prices, Timestamps, TrendStateArray
)
from .config import MTFConfig


# In structure/context/mtf.py

def calculate_mtf_confluence_score(
        ltf_trend: TrendStateArray,
        ltf_regime: np.ndarray,
        htf_trend: TrendStateArray,
        htf_regime: np.ndarray,
        config: MTFConfig
) -> np.ndarray:
    """
    Compute MTF confluence score based on trend and regime alignment.
    Score = (trend_aligned + regime_aligned) / 2
    - 1.0 = perfect confluence, 0.0 = complete divergence
    """
    n = len(ltf_trend)
    if n == 0:
        return np.array([], dtype=np.float32)

    # === VALIDATE ARRAY LENGTHS ===
    if len(htf_trend) != n:
        raise ValueError(f"Trend arrays must have same length: LTF={n}, HTF={len(htf_trend)}")
    if len(htf_regime) != n:
        raise ValueError(f"Regime arrays must have same length: LTF={n}, HTF={len(htf_regime)}")

    # === VALIDATE REGIME DTYPES (CRITICAL FIX) ===
    def _validate_regime_dtype(arr: np.ndarray, name: str) -> None:
        """Ensure regime arrays have valid dtypes (int, float, or string-like)."""
        kind = arr.dtype.kind
        if kind not in ('i', 'u', 'f', 'U', 'O'):  # integer, unsigned, float, unicode, object
            raise TypeError(
                f"{name} dtype {arr.dtype} (kind='{kind}') not supported. "
                f"Must be int, float, or string-like (U/O). Complex dtypes are invalid."
            )

    _validate_regime_dtype(ltf_regime, "LTF regime")
    _validate_regime_dtype(htf_regime, "HTF regime")

    # === TREND ALIGNMENT ===
    trend_aligned = (
            (ltf_trend == htf_trend) &
            (ltf_trend != 0)
    ).astype(np.float32)

    # === REGIME ALIGNMENT ===
    def _is_trending(regime: np.ndarray) -> np.ndarray:
        if regime.dtype.kind in ('U', 'O'):
            return np.isin(regime, ['strong_trend', 'weak_trend'])
        else:
            # Only valid for numeric dtypes (validated above)
            return regime > 0

    ltf_trending = _is_trending(ltf_regime)
    htf_trending = _is_trending(htf_regime)
    regime_aligned = (ltf_trending == htf_trending).astype(np.float32)

    return ((trend_aligned + regime_aligned) / 2.0).astype(np.float32)

def resample_and_align_context(
        timestamps: Timestamps,
        open_: Prices,
        high: Prices,
        low: Prices,
        close: Prices,
        trend_state: TrendStateArray,
        market_regime: np.ndarray,
        config: MTFConfig
) -> Tuple[
    Timestamps, Prices, Prices, Prices, Prices,
    TrendStateArray, np.ndarray
]:
    """Resample OHLC + context to HTF."""
    n = len(close)
    if config.htf_bar_size < 1:
        raise ValueError("htf_bar_size must be ≥ 1")

    num_htf_bars = n // config.htf_bar_size
    if num_htf_bars == 0:
        empty_ts = np.array([], dtype=timestamps.dtype)
        empty_p = np.array([], dtype=np.float32)
        empty_t = np.array([], dtype=np.int8)
        empty_r = np.full(0, 'neutral', dtype=market_regime.dtype)
        return (empty_ts, empty_p, empty_p, empty_p, empty_p, empty_t, empty_r)

    trimmed = slice(0, num_htf_bars * config.htf_bar_size)
    o = open_[trimmed].reshape(-1, config.htf_bar_size)
    h = high[trimmed].reshape(-1, config.htf_bar_size)
    l = low[trimmed].reshape(-1, config.htf_bar_size)
    c = close[trimmed].reshape(-1, config.htf_bar_size)
    ts = timestamps[trimmed].reshape(-1, config.htf_bar_size)
    t = trend_state[trimmed].reshape(-1, config.htf_bar_size)
    r = market_regime[trimmed].reshape(-1, config.htf_bar_size)

    htf_open = o[:, 0]
    htf_high = np.max(h, axis=1)
    htf_low = np.min(l, axis=1)
    htf_close = c[:, -1]
    htf_timestamps = ts[:, -1]
    htf_trend = t[:, -1].astype(np.int8)
    htf_regime = r[:, -1]

    return (
        htf_timestamps, htf_open, htf_high, htf_low, htf_close,
        htf_trend, htf_regime
    )


# structure/context/mtf.py
def interpolate_htf_to_ltf(
        ltf_timestamps: Timestamps,
        htf_timestamps: Timestamps,
        htf_trend: TrendStateArray,
        htf_regime: np.ndarray
) -> Tuple[TrendStateArray, np.ndarray]:
    """
    Forward-fill HTF context to LTF resolution.
    """
    n_ltf = len(ltf_timestamps)
    if n_ltf == 0 or len(htf_timestamps) == 0:
        return (
            np.zeros(n_ltf, dtype=np.int8),
            np.full(n_ltf, 'neutral', dtype=htf_regime.dtype)
        )

    ltf_trend = np.zeros(n_ltf, dtype=np.int8)

    # Handle string vs numeric regime
    if htf_regime.dtype.kind in ('U', 'O'):
        ltf_regime = np.full(n_ltf, 'neutral', dtype=htf_regime.dtype)
    else:
        ltf_regime = np.zeros(n_ltf, dtype=htf_regime.dtype)

    htf_idx = 0
    for i, ts in enumerate(ltf_timestamps):
        # Find latest HTF timestamp <= current LTF timestamp
        while htf_idx < len(htf_timestamps) - 1 and htf_timestamps[htf_idx + 1] <= ts:
            htf_idx += 1
        if htf_timestamps[htf_idx] <= ts:  # Only assign if HTF bar exists
            ltf_trend[i] = htf_trend[htf_idx]
            ltf_regime[i] = htf_regime[htf_idx]

    return ltf_trend, ltf_regime

# BACKWARD COMPATIBILITY (for existing tests)
def resample_to_htf(timestamps, open_, high, low, close, config):
    """
    Legacy alias for backward compatibility.
    Only returns OHLC — no context.
    """
    result = resample_and_align_context(
        timestamps, open_, high, low, close,
        trend_state=np.zeros(len(close), dtype=np.int8),
        market_regime=np.full(len(close), 'neutral', dtype=object),
        config=config
    )
    return {
        'htf_timestamps': result[0],
        'htf_open': result[1],
        'htf_high': result[2],
        'htf_low': result[3],
        'htf_close': result[4]
    }