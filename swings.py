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


def _validate_market_structure_input(df: pd.DataFrame) -> None:
    """Validate presence and numeric type of required columns."""
    required_cols = {'swing_type', 'high', 'low'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}.")

    if not (np.issubdtype(df['high'].dtype, np.number) and
            np.issubdtype(df['low'].dtype, np.number)):
        raise ValueError("'high' and 'low' columns must be numeric.")


def _initialize_structure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add and initialize market structure label columns."""
    result = df.copy()
    labels = ['is_higher_high', 'is_lower_high', 'is_higher_low', 'is_lower_low']
    result[labels] = False
    return result


def _extract_swing_arrays(
    df: pd.DataFrame,
    swing_mask: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract swing indices, types, and prices as NumPy arrays."""
    swing_df = df[swing_mask]
    indices = swing_df.index.values
    types = swing_df['swing_type'].values
    prices = np.where(
        types == 'high',
        df.loc[swing_df.index, 'high'].values.astype(float),
        df.loc[swing_df.index, 'low'].values.astype(float)
    )
    return indices, types, prices


def _validate_swing_alternation(swing_types: np.ndarray) -> None:
    """Ensure swing types strictly alternate (no consecutive same types)."""
    if len(swing_types) > 1:
        if np.any(swing_types[:-1] == swing_types[1:]):
            raise ValueError(
                "Swing sequence is not alternating. "
                "Ensure input is generated by `detect_swing_points()`."
            )


def _label_structure_swings(
    result: pd.DataFrame,
    swing_indices: np.ndarray,
    swing_types: np.ndarray,
    swing_prices: np.ndarray
) -> None:
    """Label HH, LH, HL, LL by comparing each swing to the last of its type."""
    last_high = np.nan
    last_low = np.nan

    for idx, typ, price in zip(swing_indices, swing_types, swing_prices):
        if np.isnan(price):
            continue

        if typ == 'high':
            if not np.isnan(last_high):
                if price > last_high:
                    result.at[idx, 'is_higher_high'] = True
                elif price < last_high:
                    result.at[idx, 'is_lower_high'] = True
            last_high = price
        elif typ == 'low':
            if not np.isnan(last_low):
                if price > last_low:
                    result.at[idx, 'is_higher_low'] = True
                elif price < last_low:
                    result.at[idx, 'is_lower_low'] = True
            last_low = price

def detect_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect higher highs, lower highs, higher lows, and lower lows from swing points.

    Args:
        df: Input DataFrame produced by `detect_swing_points()`, containing:
            - 'swing_type': values are 'high', 'low', or pd.NA
            - 'high': numeric series of high prices
            - 'low': numeric series of low prices

    Returns:
        A DataFrame with the same index and columns as input, augmented with:
            - 'is_higher_high': bool, True if current swing high > prior swing high
            - 'is_lower_high': bool, True if current swing high < prior swing high
            - 'is_higher_low': bool, True if current swing low > prior swing low
            - 'is_lower_low': bool, True if current swing low < prior swing low

    Raises:
        KeyError: If required columns ('swing_type', 'high', 'low') are missing.
        ValueError: If swing sequence is not strictly alternating,
                    or if 'high'/'low' are non-numeric.
    """
    _validate_market_structure_input(df)

    swing_mask = df['swing_type'].notna()
    if not swing_mask.any():
        return _initialize_structure_columns(df)

    swing_indices, swing_types, swing_prices = _extract_swing_arrays(df, swing_mask)
    _validate_swing_alternation(swing_types)

    result = _initialize_structure_columns(df)
    _label_structure_swings(result, swing_indices, swing_types, swing_prices)
    return result