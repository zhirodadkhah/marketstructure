# structure/context/zones.py
"""
Support/Resistance zone detection with confluence scoring and multi-touch tracking.
Pure NumPy implementation with enhanced quality metrics and tightness-aware clustering.
"""
from typing import Tuple, Optional
import numpy as np
from structure.metrics.atr import compute_atr
from structure.core.swings import detect_swing_points
from structure.metrics.types import Prices, ZoneArray, BoolArray, IntArray, FloatArray
from .config import ZoneConfig


# ==============================================================================
# SECTION: Enhanced Zone Clustering with Tightness Quality
# ==============================================================================

def _cluster_levels(
        prices: np.ndarray,
        atr_radius: float,
        min_cluster_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Density-based clustering with quality scores based on cluster tightness.

    Parameters
    ----------
    prices : np.ndarray[float32]
        Unsorted array of swing prices (highs or lows).
    atr_radius : float
        Clustering radius in price units (ATR-scaled).
    min_cluster_size : int
        Minimum points to form a valid zone.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - cluster_centers: Zone price levels
        - cluster_strength: Number of swings per zone
        - cluster_tightness: Mean normalized distance-to-center [0,1] (1 = tightest)
        - member_mask: Boolean mask of clustered points

    Notes
    -----
    - Time Complexity: O(m log m) for m = swing count
    - Space Complexity: O(m)
    - Tightness = 1 - (avg_distance / max_radius)
    """
    if len(prices) == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            np.array([], dtype=bool)
        )

    # Sort prices for sequential scanning
    sorted_idx = np.argsort(prices)
    sorted_prices = prices[sorted_idx]

    cluster_centers = []
    cluster_strengths = []
    cluster_tightnesses = []
    member_flags = np.zeros(len(prices), dtype=bool)

    i = 0
    while i < len(sorted_prices):
        start_price = sorted_prices[i]
        j = i

        # Expand cluster within radius
        while j < len(sorted_prices) and (sorted_prices[j] - start_price) <= atr_radius:
            j += 1

        cluster_size = j - i
        if cluster_size >= min_cluster_size:
            cluster_points = sorted_prices[i:j]
            center = np.mean(cluster_points)

            # Calculate tightness: 1.0 = perfect cluster, 0.0 = loose
            distances = np.abs(cluster_points - center)
            max_possible_dist = atr_radius
            avg_normalized_dist = np.mean(distances) / max_possible_dist if max_possible_dist > 0 else 0
            tightness = max(0.0, 1.0 - avg_normalized_dist)

            cluster_centers.append(center)
            cluster_strengths.append(cluster_size)
            cluster_tightnesses.append(tightness)
            member_flags[sorted_idx[i:j]] = True

        i = j

    return (
        np.array(cluster_centers, dtype=np.float32),
        np.array(cluster_strengths, dtype=np.int32),
        np.array(cluster_tightnesses, dtype=np.float32),
        member_flags
    )


# ==============================================================================
# SECTION: Multi-Touch Retest Tracking with Zone Selection
# ==============================================================================

def _compute_retest_touches(
        close: np.ndarray,
        zone_centers: np.ndarray,
        zone_strength: np.ndarray,
        zone_tightness: np.ndarray,
        atr: np.ndarray,
        zone_config: ZoneConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute retest touches, quality, and zone IDs with time-aware multi-touch logic.

    Parameters
    ----------
    close : np.ndarray[float32]
        Close prices.
    zone_centers : np.ndarray[float32]
        Zone price levels.
    zone_strength : np.ndarray[int32]
        Strength (count) of each zone.
    zone_tightness : np.ndarray[float32]
        Tightness quality [0,1] of each zone.
    atr : np.ndarray[float32]
        ATR for buffer calculation.
    zone_config : ZoneConfig
        Configuration with min_touch_bars, buffer_multiplier, etc.

    Returns
    -------
    Tuple of arrays (length = len(close)):
        - retest_quality: Combined quality [0,1]
        - retest_count: Cumulative touches per zone
        - is_double_test: True on 2nd valid touch
        - is_triple_test: True on 3rd+ valid touch
        - zone_id: ID of touched zone (-1 if none)

    Notes
    -----
    - Zone selection uses weighted score: strength × tightness / (distance + ε)
    - Minimum time gap enforced between touches to avoid false multi-touch
    """
    n = len(close)
    z = len(zone_centers)

    if z == 0:
        return (
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=bool),
            np.zeros(n, dtype=bool),
            np.full(n, -1, dtype=np.int32)
        )

    # Initialize outputs
    retest_quality = np.zeros(n, dtype=np.float32)
    retest_count = np.zeros(n, dtype=np.int32)
    is_double_test = np.zeros(n, dtype=bool)
    is_triple_test = np.zeros(n, dtype=bool)
    zone_id = np.full(n, -1, dtype=np.int32)

    # Track per-zone state
    zone_last_touch_idx = np.full(z, -1, dtype=np.int32)
    zone_touch_count = np.zeros(z, dtype=np.int32)

    buffer = atr * zone_config.buffer_multiplier
    min_touch_bars = getattr(zone_config, 'min_touch_bars', 3)

    for i in range(n):
        current_buffer = buffer[i]
        distances = np.abs(close[i] - zone_centers)
        within_buffer = distances <= current_buffer

        if not np.any(within_buffer):
            continue

        # Select best zone by weighted score
        valid_indices = np.where(within_buffer)[0]
        valid_distances = distances[valid_indices]
        valid_strengths = zone_strength[valid_indices]
        valid_tightnesses = zone_tightness[valid_indices]

        epsilon = current_buffer * 0.01  # Avoid division by zero
        zone_scores = (valid_strengths * valid_tightnesses) / (valid_distances + epsilon)
        best_idx = valid_indices[np.argmax(zone_scores)]

        # Enforce minimum time between touches
        last_touch = zone_last_touch_idx[best_idx]
        if last_touch >= 0 and (i - last_touch) < min_touch_bars:
            continue

        # Update zone tracking
        zone_last_touch_idx[best_idx] = i
        zone_touch_count[best_idx] += 1
        touch_num = zone_touch_count[best_idx]

        # Store results
        zone_id[i] = best_idx
        retest_count[i] = touch_num

        if touch_num == 2:
            is_double_test[i] = True
        elif touch_num >= 3:
            is_triple_test[i] = True

        # Compute quality: 70% strength + 30% tightness
        normalized_strength = min(1.0, float(valid_strengths[np.argmax(zone_scores)]) / 10.0)
        quality = 0.7 * normalized_strength + 0.3 * valid_tightnesses[np.argmax(zone_scores)]
        retest_quality[i] = np.clip(quality, 0.0, 1.0)

    return retest_quality, retest_count, is_double_test, is_triple_test, zone_id


# ==============================================================================
# SECTION: Main Zone Detection Function with Full Metrics
# ==============================================================================

def detect_sr_zones(
        high: Prices,
        low: Prices,
        close: Prices,
        config: ZoneConfig
) -> Tuple[
    ZoneArray,  # support_levels
    ZoneArray,  # resistance_levels
    FloatArray,  # retest_quality
    IntArray,  # retest_count
    BoolArray,  # is_double_test
    BoolArray,  # is_triple_test
    BoolArray,  # is_confluence_zone_support
    BoolArray,  # is_confluence_zone_resistance
    FloatArray,  # signal_zone_score
    IntArray,  # zone_id
    FloatArray,  # zone_strength_normalized
    FloatArray  # zone_tightness
]:
    """
    Detect S/R zones and compute comprehensive confluence metrics.

    Parameters
    ----------
    high, low, close : Prices
        OHLC price arrays.
    config : ZoneConfig
        Zone detection and scoring configuration.

    Returns
    -------
    Tuple of arrays:
        - support_levels: Sorted support zones
        - resistance_levels: Sorted resistance zones
        - retest_quality: [0,1] quality based on strength + tightness
        - retest_count: Touch count per bar
        - is_double_test: True on 2nd touch
        - is_triple_test: True on 3rd+ touch
        - is_confluence_zone_support: Boolean mask
        - is_confluence_zone_resistance: Boolean mask
        - signal_zone_score: Weighted composite score
        - zone_id: ID of touched zone (-1 if none)
        - zone_strength_normalized: [0,1] normalized strength
        - zone_tightness: Cluster tightness at touch point

    Notes
    -----
    - Time Complexity: O(n log n + n·z) where z = zone count (typically small)
    - Space Complexity: O(n + z)
    - Uses recent_bars to limit swing consideration
    - All output arrays same length as input (except zone lists)
    """
    n = len(close)
    if n < 5:
        empty_f = np.zeros(n, dtype=np.float32)
        empty_i = np.zeros(n, dtype=np.int32)
        empty_b = np.zeros(n, dtype=bool)
        return (
            np.array([], dtype=np.float32),  # support
            np.array([], dtype=np.float32),  # resistance
            empty_f,  # quality
            empty_i,  # count
            empty_b,  # double
            empty_b,  # triple
            empty_b,  # conf_support
            empty_b,  # conf_resistance
            empty_f,  # score
            np.full(n, -1, dtype=np.int32),  # zone_id
            empty_f,  # strength_norm
            empty_f  # tightness
        )

    # === GET SWINGS ===
    swings = detect_swing_points(high, low, half_window=config.swing_window)
    is_sh = swings['is_swing_high']
    is_sl = swings['is_swing_low']

    # === USE RECENT BARS ONLY ===
    start_idx = max(0, n - config.recent_bars)
    sh_mask = is_sh[start_idx:]
    sl_mask = is_sl[start_idx:]
    sh_prices = high[start_idx:][sh_mask]
    sl_prices = low[start_idx:][sl_mask]

    # === COMPUTE ATR FOR NORMALIZATION ===
    atr = compute_atr(high, low, close, period=config.atr_period)
    recent_atr = np.nanmean(atr[-20:]) if np.any(~np.isnan(atr[-20:])) else 1.0
    radius = config.clustering_radius * recent_atr

    # === CLUSTER ZONES ===
    sup_centers, sup_strength, sup_tightness, sup_mask = _cluster_levels(
        sl_prices, radius, config.min_cluster_size
    )
    res_centers, res_strength, res_tightness, res_mask = _cluster_levels(
        sh_prices, radius, config.min_cluster_size
    )

    # === COMBINE ALL ZONES ===
    all_centers = np.concatenate([sup_centers, res_centers])
    all_strength = np.concatenate([sup_strength, res_strength])
    all_tightness = np.concatenate([sup_tightness, res_tightness])

    # === COMPUTE RETEST METRICS ===
    (retest_quality, retest_count, is_double_test,
     is_triple_test, zone_id) = _compute_retest_touches(
        close, all_centers, all_strength, all_tightness, atr, config
    )

    # === CONFLUENCE FLAGS ===
    is_confluence_zone = np.zeros(n, dtype=bool)
    is_confluence_support = np.zeros(n, dtype=bool)
    is_confluence_resistance = np.zeros(n, dtype=bool)

    if len(all_centers) > 0:
        buffer = atr * config.buffer_multiplier

        # Vectorized confluence detection
        for i in range(n):
            distances = np.abs(close[i] - all_centers)
            within = distances <= buffer[i]

            if np.any(within):
                valid_strengths = all_strength[within]
                is_confluence_zone[i] = np.any(valid_strengths >= config.min_confluence_strength)

                # Support confluence
                if len(sup_centers) > 0:
                    sup_dist = np.abs(close[i] - sup_centers)
                    sup_within = sup_dist <= buffer[i]
                    if np.any(sup_within):
                        sup_str = sup_strength[sup_within]
                        is_confluence_support[i] = np.any(sup_str >= config.min_confluence_strength)

                # Resistance confluence
                if len(res_centers) > 0:
                    res_dist = np.abs(close[i] - res_centers)
                    res_within = res_dist <= buffer[i]
                    if np.any(res_within):
                        res_str = res_strength[res_within]
                        is_confluence_resistance[i] = np.any(res_str >= config.min_confluence_strength)

    # === NORMALIZED STRENGTH & TIGHTNESS ARRAYS ===
    zone_strength_normalized = np.zeros(n, dtype=np.float32)
    zone_tightness_arr = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if zone_id[i] >= 0:
            zone_strength_normalized[i] = min(1.0, float(all_strength[zone_id[i]]) / 10.0)
            zone_tightness_arr[i] = all_tightness[zone_id[i]]

    # === SIGNAL ZONE SCORE (COMPOSITE) ===
    confluence_boost = is_confluence_zone.astype(np.float32) * config.confluence_weight
    double_boost = is_double_test.astype(np.float32) * config.double_weight
    triple_boost = is_triple_test.astype(np.float32) * config.triple_weight

    signal_zone_score = np.clip(
        config.quality_weight * retest_quality +
        double_boost + triple_boost + confluence_boost,
        0.0, 1.0
    )

    return (
        sup_centers,
        res_centers,
        retest_quality,
        retest_count,
        is_double_test,
        is_triple_test,
        is_confluence_support,
        is_confluence_resistance,
        signal_zone_score,
        zone_id,
        zone_strength_normalized,
        zone_tightness_arr
    )