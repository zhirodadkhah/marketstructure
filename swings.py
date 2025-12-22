from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from typing import Tuple, List

# ----------------------------
# Type Aliases (for clarity)
# ----------------------------
SwingResult = pd.DataFrame


def _validate_inputs(df: pd.DataFrame, half_window: int) -> None:
    """Validate input DataFrame and half_window parameter.

    Args:
        df: Input DataFrame to validate.
        half_window: Lookback/lookforward window size.

    Raises:
        ValueError: If `half_window` is not a positive integer.
        KeyError: If 'high' or 'Low' columns are missing from `df`.
    """
    if not isinstance(half_window, int) or half_window < 1:
        raise ValueError(f"half_window must be a positive integer, got {half_window}")

    required_cols = {'high', 'low'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}. DataFrame must contain 'high' and 'low'.")


def _detect_swing_candidates(
        high: np.ndarray,
        low: np.ndarray,
        half_window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect initial swing high and low candidates

    Args:
        high: Array of high prices.
        low: Array of low prices.
        half_window: Number of bars to look left/right for validation.

    Returns:
        A tuple of two boolean arrays:
            - First: swing high candidates
            - Second: swing low candidates

    Note:
        Boundary regions (first and last `half_window` bars) are explicitly
        set to False to avoid edge artifacts.
    """
    window_size = 2 * half_window + 1

    high_max = maximum_filter1d(high, size=window_size, mode='constant')
    low_min = minimum_filter1d(low, size=window_size, mode='constant')

    sh_candidates = (high == high_max)
    sl_candidates = (low == low_min)

    # Invalidate edges
    sh_candidates[:half_window] = False
    sh_candidates[-half_window:] = False
    sl_candidates[:half_window] = False
    sl_candidates[-half_window:] = False

    return sh_candidates, sl_candidates


def _enforce_strict_extrema(
        high: np.ndarray,
        low: np.ndarray,
        sh_candidates: np.ndarray,
        sl_candidates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Enforce strict local extrema by requiring strict inequality with neighbors.

    Removes candidates that are part of flat plateaus (e.g., equal highs/lows).

    Args:
        high: Array of high prices.
        low: Array of low prices.
        sh_candidates: Boolean array of swing high candidates.
        sl_candidates: Boolean array of swing low candidates.

    Returns:
        Updated swing high and low candidate arrays with plateaus removed.

    Note:
        Uses np.roll (which wraps), so first and last bars are manually set to False.
    """
    # Compare with immediate neighbors (allows wrap — we'll fix edges)
    sh_strict = (high > np.roll(high, 1)) & (high > np.roll(high, -1))
    sl_strict = (low < np.roll(low, 1)) & (low < np.roll(low, -1))

    sh_candidates &= sh_strict
    sl_candidates &= sl_strict

    # Fix boundary artifacts from np.roll (first/last bars)
    sh_candidates[0] = sh_candidates[-1] = False
    sl_candidates[0] = sl_candidates[-1] = False

    return sh_candidates, sl_candidates


def _resolve_conflicts(
        high: np.ndarray,
        low: np.ndarray,
        sh_candidates: np.ndarray,
        sl_candidates: np.ndarray,
        half_window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve points that are both swing high and swing low candidates.

    For ambiguous points, selects the type (high or low) with greater
    relative extremity compared to neighbors at ±`half_window`.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        sh_candidates: Boolean array of swing high candidates.
        sl_candidates: Boolean array of swing low candidates.
        half_window: Window size used for extremity comparison.

    Returns:
        Updated candidate arrays where no point is both a swing high and low.
    """
    conflict = sh_candidates & sl_candidates
    if not np.any(conflict):
        return sh_candidates, sl_candidates

    # Extremity relative to ±half_window neighbors
    left_high = np.roll(high, half_window)
    right_high = np.roll(high, -half_window)
    high_extremity = high - np.minimum(left_high, right_high)

    left_low = np.roll(low, half_window)
    right_low = np.roll(low, -half_window)
    low_extremity = np.maximum(left_low, right_low) - low

    prefer_high = high_extremity > low_extremity

    sh_candidates = sh_candidates.copy()
    sl_candidates = sl_candidates.copy()

    sh_candidates[conflict] = prefer_high[conflict]
    sl_candidates[conflict] = ~prefer_high[conflict]

    return sh_candidates, sl_candidates


def _enforce_alternation(
        high: np.ndarray,
        low: np.ndarray,
        sh_candidates: np.ndarray,
        sl_candidates: np.ndarray
) -> Tuple[List[int], List[str]]:
    """Enforce alternating swing sequence (high → low → high...).

    Merges consecutive same-type swings by keeping the most extreme,
    and ensures swings alternate between high and low.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        sh_candidates: Final swing high candidate mask.
        sl_candidates: Final swing low candidate mask.

    Returns:
        A tuple of two lists:
            - First: list of swing indices (int)
            - Second: list of swing types ('high' or 'low')
    """
    swings: List[Tuple[int, str, float]] = []

    # Add swing highs
    for idx in np.where(sh_candidates)[0]:
        swings.append((int(idx), 'high', float(high[idx])))

    # Add swing lows
    for idx in np.where(sl_candidates)[0]:
        swings.append((int(idx), 'low', float(low[idx])))

    if not swings:
        return [], []

    # Sort by index (time order)
    swings.sort(key=lambda x: x[0])

    # Enforce alternation and extremity
    final = [swings[0]]
    for i in range(1, len(swings)):
        curr_idx, curr_type, curr_price = swings[i]
        last_idx, last_type, last_price = final[-1]

        if curr_type == last_type:
            # Same type: keep more extreme
            if (curr_type == 'high' and curr_price > last_price) or \
                    (curr_type == 'low' and curr_price < last_price):
                final[-1] = swings[i]
        else:
            final.append(swings[i])

    indices = [s[0] for s in final]
    types = [s[1] for s in final]
    return indices, types


def _build_output(
        df: pd.DataFrame,
        swing_indices: List[int],
        swing_types: List[str]
) -> SwingResult:
    """
    Args:
        df: Original input DataFrame.
        swing_indices: List of integer indices where swings occur.
        swing_types: List of swing types ('high' or 'low').

    Returns:
        DataFrame with same index as `df` and three new columns:
            - 'swing_type'
            - 'is_swing_high'
            - 'is_swing_low'
    """
    index = df.index
    swing_type = pd.Series(pd.NA, index=index, dtype="object")
    is_swing_high = pd.Series(False, index=index, dtype=bool)
    is_swing_low = pd.Series(False, index=index, dtype=bool)

    for idx, typ in zip(swing_indices, swing_types):
        swing_type.iloc[idx] = typ
        if typ == 'high':
            is_swing_high.iloc[idx] = True
        else:
            is_swing_low.iloc[idx] = True

    return pd.DataFrame({
        'swing_type': swing_type,
        'is_swing_high': is_swing_high,
        'is_swing_low': is_swing_low
    })


def detect_swing_points(
        df: pd.DataFrame,
        half_window: int = 2
) -> SwingResult:
    """Detect alternating Swing Highs and Swing Lows in OHLC price data.

    Args:
        df: Input DataFrame containing at least 'high' and 'low' columns.
            The index may be of any type (e.g., pd.DatetimeIndex).
        half_window: Number of bars to look left and right to validate a swing.
            Must be a positive integer. Default is 2, meaning a 5-bar window
            (2 left + current + 2 right).

    Returns:
        A DataFrame with the same index as `df`, containing:
            - 'swing_type': pd.Series with values 'high', 'low', or pd.NA
            - 'is_swing_high': pd.Series[bool] indicating swing highs
            - 'is_swing_low': pd.Series[bool] indicating swing lows

    Raises:
        ValueError: If `half_window` is not a positive integer.
        KeyError: If the input DataFrame is missing 'high' or 'low' columns.
        """
    _validate_inputs(df, half_window)

    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)

    sh_candidates, sl_candidates = _detect_swing_candidates(high, low, half_window)
    sh_candidates, sl_candidates = _enforce_strict_extrema(high, low, sh_candidates, sl_candidates)
    sh_candidates, sl_candidates = _resolve_conflicts(high, low, sh_candidates, sl_candidates, half_window)
    swing_indices, swing_types = _enforce_alternation(high, low, sh_candidates, sl_candidates)

    return _build_output(df, swing_indices, swing_types)