""" Objectively classify market regime based solely on swing sequence. """
import pandas as pd
import numpy as np

from typing import List, Dict, Any, Optional, Tuple


def _validate_trend_input(df: pd.DataFrame, buffer: float) -> None:
    """Validate input DataFrame and parameters."""
    if buffer < 0:
        raise ValueError("invalidation_buffer must be non-negative.")

    required = {
        'swing_type', 'high', 'low', 'close',
        'is_higher_high', 'is_lower_high',
        'is_higher_low', 'is_lower_low'
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}.")


def _initialize_trend_columns(df: pd.DataFrame, include_metrics: bool) -> pd.DataFrame:
    """Initialize output DataFrame with neutral trend state."""
    result = df.copy()
    result['trend_state'] = 'neutral'
    if include_metrics:
        result['trend_strength'] = 0.0
        result['trend_since'] = pd.NaT
    return result


def _extract_and_validate_swings(df: pd.DataFrame) -> pd.DataFrame:
    """Extract swing points and validate alternation."""
    swing_mask = df['swing_type'].notna()
    swing_df = df[swing_mask].sort_index()

    types = swing_df['swing_type'].values
    if len(types) > 1 and np.any(types[:-1] == types[1:]):
        raise ValueError(
            "Swing sequence is not alternating. "
            "Ensure input is from `detect_swing_points()`."
        )
    return swing_df


def _build_trend_events(
        swing_df: pd.DataFrame,
        include_metrics: bool
) -> List[Dict[str, Any]]:
    """Build list of trend confirmation events."""
    events = []
    last_hl_price = last_lh_price = None
    last_hl_idx = last_lh_idx = None

    for idx in swing_df.index:
        row = swing_df.loc[idx]
        if row['swing_type'] == 'low':
            if row['is_higher_low']:
                last_hl_price, last_hl_idx = row['low'], idx
            elif row['is_lower_low'] and last_lh_price is not None:
                event = {'idx': idx, 'state': 'downtrend', 'ref': last_hl_price}
                if include_metrics:
                    base = abs(last_hl_price) or 1.0
                    diff_pct = abs(row['low'] - last_hl_price) / base
                    event['strength'] = min(diff_pct * 100.0, 10.0)
                events.append(event)
        else:  # high
            if row['is_lower_high']:
                last_lh_price, last_lh_idx = row['high'], idx
            elif row['is_higher_high'] and last_hl_price is not None:
                event = {'idx': idx, 'state': 'uptrend', 'ref': last_hl_price}
                if include_metrics:
                    base = abs(last_hl_price) or 1.0
                    diff_pct = abs(row['high'] - last_hl_price) / base
                    event['strength'] = min(diff_pct * 100.0, 10.0)
                events.append(event)
    return events


def _propagate_trend_state(
        index: pd.Index,
        close_prices: np.ndarray,
        events: List[Dict[str, Any]],
        buffer: float
) -> Dict[str, List]:
    """Propagate trend state with invalidation; return column data."""
    states: List[str] = []
    strengths: List[float] = []
    start_indices: List[Optional[Any]] = []

    current_state = 'neutral'
    current_ref: Optional[float] = None
    current_strength = 0.0
    trend_start: Optional[Any] = None

    event_iter = iter(events)
    next_event = next(event_iter, None)

    # Pre-extract event fields for speed
    event_idx = next_event['idx'] if next_event else None

    for i, idx in enumerate(index):
        # Activate new trend
        if next_event is not None and idx == event_idx:
            current_state = next_event['state']
            current_ref = next_event['ref']
            current_strength = next_event.get('strength', 0.0)
            trend_start = idx
            next_event = next(event_iter, None)
            event_idx = next_event['idx'] if next_event else None

        # Invalidation check
        if current_state != 'neutral' and current_ref is not None:
            close = close_prices[i]
            buffer_abs = abs(current_ref) * buffer if current_ref != 0 else 0.0

            if current_state == 'uptrend':
                if close < (current_ref - buffer_abs):
                    current_state, current_ref, current_strength, trend_start = 'neutral', None, 0.0, None
            else:  # downtrend
                if close > (current_ref + buffer_abs):
                    current_state, current_ref, current_strength, trend_start = 'neutral', None, 0.0, None

        states.append(current_state)
        strengths.append(current_strength)
        start_indices.append(trend_start if current_state != 'neutral' else pd.NaT)

    return {
        'states': states,
        'strengths': strengths,
        'start_indices': start_indices
    }


def _assemble_result(
        df: pd.DataFrame,
        trend_data: Dict[str, List],
        include_metrics: bool
) -> pd.DataFrame:
    """Assemble final result DataFrame."""
    result = df.copy()
    result['trend_state'] = trend_data['states']
    if include_metrics:
        result['trend_strength'] = trend_data['strengths']
        result['trend_since'] = trend_data['start_indices']
    return result

def detect_trend_state(
        df: pd.DataFrame,
        invalidation_buffer: float = 0.0,
        include_metrics: bool = False
) -> pd.DataFrame:
    """
    Detect and maintain dynamic market trend state with optional strength metrics.

    This function identifies trending markets based on confirmed sequences of
    higher highs/lows (uptrend) or lower highs/lows (downtrend), and dynamically
    invalidates trends when price action breaks key structural reference levels.

    Trend confirmation logic:
        - **Uptrend**: Confirmed at a Higher High (HH) that follows a Higher Low (HL).
          The HL price becomes the reference support level.
        - **Downtrend**: Confirmed at a Lower Low (LL) that follows a Lower High (LH).
          The LH price becomes the reference resistance level.

    Trend invalidation logic:
        - **Uptrend invalidated** when the closing price falls below the reference HL level
          (optionally adjusted by `invalidation_buffer`).
        - **Downtrend invalidated** when the closing price rises above the reference LH level
          (optionally adjusted by `invalidation_buffer`).

    Args:
        df: Input DataFrame that must contain the following columns:
            - 'swing_type': values are 'high', 'low', or pd.NA
              (as produced by `detect_swing_points()`)
            - 'high', 'low', 'close': numeric price series
            - 'is_higher_high', 'is_lower_high', 'is_higher_low', 'is_lower_low':
              boolean structure labels (as produced by `detect_market_structure()`)

        invalidation_buffer: Fractional buffer (0.0 to 1.0) for invalidation tolerance.
            Applied as absolute offset: `buffer = |reference| * buffer`.
            Safe for negative prices. Default: 0.0 (exact break).

        include_metrics: If True, adds 'trend_strength' (0.0–10.0) and 'trend_since'.

    Returns:
        DataFrame with 'trend_state' and optional metric columns.

    Raises:
        KeyError: If required columns missing.
        ValueError: If buffer < 0 or swings not alternating.
    """
    _validate_trend_input(df, invalidation_buffer)

    if not df['swing_type'].notna().any():
        return _initialize_trend_columns(df, include_metrics)

    swing_df = _extract_and_validate_swings(df)
    events = _build_trend_events(swing_df, include_metrics)

    if not events:
        return _initialize_trend_columns(df, include_metrics)

    trend_data = _propagate_trend_state(
        df.index,
        df['close'].values,
        events,
        invalidation_buffer
    )

    return _assemble_result(df, trend_data, include_metrics)
