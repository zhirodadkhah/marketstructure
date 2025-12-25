# zones.py
from typing import Tuple, Dict
import pandas as pd
import numpy as np
from collections import defaultdict


def detect_support_resistance_zones(
        df: pd.DataFrame,
        config
) -> pd.DataFrame:
    """Optimized vectorized S/R zone detection."""
    required = {'is_swing_high', 'is_swing_low', 'high', 'low', 'atr', 'close'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise KeyError(f"Missing columns: {missing}")

    if df.empty or len(df) < 2:
        return _empty_zone_df(df)

    # Pre-extract arrays for speed
    is_swing_high_arr = df['is_swing_high'].values
    is_swing_low_arr = df['is_swing_low'].values
    high_arr = df['high'].values.astype(np.float32)
    low_arr = df['low'].values.astype(np.float32)
    atr_arr = df['atr'].values.astype(np.float32)
    close_arr = df['close'].values.astype(np.float32)

    # Extract swing points
    swing_mask = is_swing_high_arr | is_swing_low_arr
    if not swing_mask.any():
        return _empty_zone_df(df)

    swing_idx = np.where(swing_mask)[0]
    swing_is_high = is_swing_high_arr[swing_idx]

    # Get swing prices
    swing_price = np.where(
        swing_is_high,
        high_arr[swing_idx],
        low_arr[swing_idx]
    ).astype(np.float32)

    # Cluster swings into zones
    zone_id, zone_price, zone_strength, zone_type = _vectorized_zone_clustering(
        indices=swing_idx,
        prices=swing_price,
        is_high=swing_is_high,
        atrs=atr_arr[swing_idx],
        lookback=config.zone_lookback_bars,
        proximity_mult=config.zone_proximity_atr_mult
    )

    # Map back to full DataFrame
    return _create_zone_dataframe(
        df=df,
        swing_idx=swing_idx,
        swing_is_high=swing_is_high,
        zone_id=zone_id,
        zone_price=zone_price,
        zone_strength=zone_strength,
        zone_type=zone_type,
        config=config
    )


def _vectorized_zone_clustering(
        indices: np.ndarray,
        prices: np.ndarray,
        is_high: np.ndarray,
        atrs: np.ndarray,
        lookback: int,
        proximity_mult: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast zone clustering with type tracking."""
    n = len(indices)
    zone_id = np.full(n, -1, dtype=np.int32)
    zone_price = np.full(n, np.nan, dtype=np.float32)
    zone_strength = np.zeros(n, dtype=np.int32)
    zone_type_arr = np.full(n, 'none', dtype=object)

    zones = []  # (price, last_idx, strength, support_count, resistance_count, id)
    next_zone_id = 0

    for i in range(n):
        current_idx = indices[i]
        current_price = prices[i]
        current_is_high = is_high[i]
        proximity = atrs[i] * proximity_mult

        # Find nearest compatible zone within lookback
        best_z_idx = -1
        min_dist = proximity

        for j in range(len(zones) - 1, -1, -1):
            z_price, z_last_idx, z_strength, z_support, z_resist, z_id = zones[j]

            # Time filter
            if current_idx - z_last_idx > lookback:
                break

            # Distance check
            dist = abs(current_price - z_price)
            if dist <= proximity and dist < min_dist:
                best_z_idx = j
                min_dist = dist

        if best_z_idx >= 0:
            # Merge into existing zone
            (z_price, z_last_idx, z_strength,
             z_support, z_resist, z_id) = zones[best_z_idx]

            # Update zone
            new_price = (z_price * z_strength + current_price) / (z_strength + 1)
            new_strength = z_strength + 1
            new_support = z_support + (0 if current_is_high else 1)
            new_resist = z_resist + (1 if current_is_high else 0)

            zones[best_z_idx] = (new_price, current_idx, new_strength,
                                 new_support, new_resist, z_id)

            # Update output arrays
            zone_id[i] = z_id
            zone_price[i] = new_price
            zone_strength[i] = new_strength

            # Determine zone type
            if new_support > 0 and new_resist > 0:
                zone_type = 'both'
            elif new_support > 0:
                zone_type = 'support'
            else:
                zone_type = 'resistance'
            zone_type_arr[i] = zone_type

        else:
            # Create new zone
            z_id = next_zone_id
            next_zone_id += 1

            support_count = 0 if current_is_high else 1
            resist_count = 1 if current_is_high else 0
            zone_type = 'resistance' if current_is_high else 'support'

            zones.append((current_price, current_idx, 1,
                          support_count, resist_count, z_id))

            zone_id[i] = z_id
            zone_price[i] = current_price
            zone_strength[i] = 1
            zone_type_arr[i] = zone_type

    return zone_id, zone_price, zone_strength, zone_type_arr


def _create_zone_dataframe(
        df: pd.DataFrame,
        swing_idx: np.ndarray,
        swing_is_high: np.ndarray,
        zone_id: np.ndarray,
        zone_price: np.ndarray,
        zone_strength: np.ndarray,
        zone_type: np.ndarray,
        config
) -> pd.DataFrame:
    """Create final DataFrame with all zone features."""
    result = df.copy()
    n = len(result)

    # Initialize columns
    result['zone_id'] = -1
    result['zone_price'] = np.nan
    result['zone_strength'] = 0
    result['zone_type'] = 'none'

    # Assign to swing bars
    result['zone_id'].values[swing_idx] = zone_id
    result['zone_price'].values[swing_idx] = zone_price
    result['zone_strength'].values[swing_idx] = zone_strength
    result['zone_type'].values[swing_idx] = zone_type

    # Confluence flag
    result['is_confluence_zone'] = (result['zone_strength'] >= config.min_zone_strength)

    # Calculate retest quality efficiently
    result['retest_quality'] = _calculate_retest_quality(
        result, config
    )

    # Multi-touch detection
    result = _calculate_multi_touch(result)

    # Signal score
    result['signal_zone_score'] = _calculate_zone_score(result)

    return result


def _calculate_retest_quality(
        df: pd.DataFrame,
        config
) -> np.ndarray:
    """Efficient retest quality calculation."""
    n = len(df)
    retest_quality = np.zeros(n, dtype=np.float32)

    # Get unique zones
    valid_zones = df[df['zone_id'] >= 0]
    if valid_zones.empty:
        return retest_quality

    unique_zones = valid_zones.groupby('zone_id').agg({
        'zone_price': 'first',
        'zone_strength': 'first'
    }).reset_index()

    # Vectorized distance calculation
    close_prices = df['close'].values.astype(np.float32)
    atr_buf = df['atr'].values * config.zone_buffer_multiplier
    zone_prices = unique_zones['zone_price'].values.astype(np.float32)
    zone_strengths = unique_zones['zone_strength'].values.astype(np.float32)

    # For each bar, check all zones (O(n×z) but z is small)
    for i in range(n):
        distances = np.abs(close_prices[i] - zone_prices)
        within_buffer = distances <= atr_buf[i]

        if within_buffer.any():
            # Get strongest zone within buffer
            valid_strengths = zone_strengths[within_buffer]
            max_strength = np.max(valid_strengths)
            retest_quality[i] = min(1.0, max_strength / 5.0)

    return retest_quality


def _calculate_multi_touch(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate multi-touch retest counts."""
    result = df.copy()

    # Track touches per zone
    zone_touches = defaultdict(int)
    retest_counts = np.zeros(len(result), dtype=np.int32)
    is_double_test = np.zeros(len(result), dtype=bool)
    is_triple_test = np.zeros(len(result), dtype=bool)

    for i in range(len(result)):
        z_id = result['zone_id'].iloc[i]
        if z_id >= 0:
            zone_touches[z_id] += 1
            touch_count = zone_touches[z_id]
            retest_counts[i] = touch_count

            if touch_count == 2:
                is_double_test[i] = True
            elif touch_count >= 3:
                is_triple_test[i] = True

    result['retest_count'] = retest_counts
    result['is_double_test'] = is_double_test
    result['is_triple_test'] = is_triple_test

    return result


def _calculate_zone_score(df: pd.DataFrame) -> np.ndarray:
    """Calculate composite zone-based signal score."""
    score = (
            df['retest_quality'].values * 0.4 +
            (df['is_confluence_zone'].values * np.minimum(df['zone_strength'].values * 0.1, 0.4)) +
            (df['is_double_test'].values * 0.1) +
            (df['is_triple_test'].values * 0.2)
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _empty_zone_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with empty zone columns."""
    df = df.copy()
    df['zone_id'] = -1
    df['zone_price'] = np.nan
    df['zone_strength'] = 0
    df['zone_type'] = 'none'
    df['is_confluence_zone'] = False
    df['retest_quality'] = 0.0
    df['retest_count'] = 0
    df['is_double_test'] = False
    df['is_triple_test'] = False
    df['signal_zone_score'] = 0.0
    return df