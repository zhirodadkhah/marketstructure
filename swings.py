import pandas as pd
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d


def detect_swing_points(df: pd.DataFrame, half_window: int = 2) -> pd.DataFrame:
    """
    Detect Swing Highs and Swing Lows in OHLC data.

    Algorithm:
    1. Find all local maxima/minima (swing candidates)
    2. Filter to ensure alternation (high-low-high-low...)
    3. Optionally, keep only the most extreme when same-type occurs

    :param df : pd.DataFrame, Must contain 'High' and 'Low' columns.
    :param half_window : int. Lookback/lookforward window (number of candles on each side to compare).
        Must be >= 1.

    :return pd.DataFrame with boolean columns:
        - 'is_swing_high'
        - 'is_swing_low'
    """

    if half_window < 1:
        raise ValueError("Lookback 'n' must be >= 1")

    high = df['high'].values
    low = df['low'].values
    length = len(df)

    # Step 1: Find swing candidates using vectorized operations
    window_size = 2 * half_window + 1

    # Find local maxima (swing high candidates)
    high_max = maximum_filter1d(high, size=window_size, mode='constant')
    is_swing_high_candidate = (high == high_max)

    # Find local minima (swing low candidates)
    low_min = minimum_filter1d(low, size=window_size, mode='constant')
    is_swing_low_candidate = (low == low_min)

    # Exclude boundaries
    is_swing_high_candidate[:half_window] = False
    is_swing_high_candidate[-half_window:] = False
    is_swing_low_candidate[:half_window] = False
    is_swing_low_candidate[-half_window:] = False

    # Ensure strict inequality (optional but recommended)
    is_swing_high_candidate &= (high > np.roll(high, 1)) & (high > np.roll(high, -1))
    is_swing_low_candidate &= (low < np.roll(low, 1)) & (low < np.roll(low, -1))

    # Handle edge cases for roll operation
    is_swing_high_candidate[0] = False
    is_swing_high_candidate[-1] = False
    is_swing_low_candidate[0] = False
    is_swing_low_candidate[-1] = False

    # Step 2: Resolve conflicts (points that are both high and low)
    # Keep only one type based on relative extremity
    conflict_mask = is_swing_high_candidate & is_swing_low_candidate

    if conflict_mask.any():
        # For conflicting points, choose based on which is more extreme
        high_range = high - low
        high_extremity = high - np.minimum(np.roll(high, -half_window), np.roll(high, half_window))
        low_extremity = np.maximum(np.roll(low, -half_window), np.roll(low, half_window)) - low

        # Prefer the type with greater relative extremity
        prefer_high = high_extremity > low_extremity

        is_swing_high_candidate[conflict_mask] = prefer_high[conflict_mask]
        is_swing_low_candidate[conflict_mask] = ~prefer_high[conflict_mask]

    # Step 3: Build arrays of swing indices and types
    swing_indices = []
    swing_types = []

    # Get indices where swings occur
    high_indices = np.where(is_swing_high_candidate)[0]
    low_indices = np.where(is_swing_low_candidate)[0]

    # Combine and sort by index
    all_swings = []
    for idx in high_indices:
        all_swings.append((idx, 'high', high[idx]))
    for idx in low_indices:
        all_swings.append((idx, 'low', low[idx]))

    # Sort by index
    all_swings.sort(key=lambda x: x[0])

    # Step 4: Enforce alternation (single pass)
    if all_swings:
        final_swings = [all_swings[0]]

        for i in range(1, len(all_swings)):
            current_idx, current_type, current_price = all_swings[i]
            last_idx, last_type, last_price = final_swings[-1]

            if current_type == last_type:
                # Same type - keep the more extreme one
                if (current_type == 'high' and current_price > last_price) or \
                        (current_type == 'low' and current_price < last_price):
                    # Replace with more extreme swing
                    final_swings[-1] = all_swings[i]
                # else: keep the previous one
            else:
                # Different type - check if reasonable distance
                # (Optional: add minimum distance filter here)
                final_swings.append(all_swings[i])

        # Convert to final arrays
        swing_indices = [s[0] for s in final_swings]
        swing_types = [s[1] for s in final_swings]

    # Step 5: Create output DataFrame
    swing_type_result = pd.Series(pd.NA, index=df.index, dtype='object')
    is_swing_high_result = pd.Series(False, index=df.index, dtype=bool)
    is_swing_low_result = pd.Series(False, index=df.index, dtype=bool)

    for idx, typ in zip(swing_indices, swing_types):
        swing_type_result.iloc[idx] = typ
        if typ == 'high':
            is_swing_high_result.iloc[idx] = True
        else:
            is_swing_low_result.iloc[idx] = True

    return pd.DataFrame({
        'swing_type': swing_type_result,
        'is_swing_high': is_swing_high_result,
        'is_swing_low': is_swing_low_result
    })