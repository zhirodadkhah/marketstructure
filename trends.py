""" Objectively classify market regime based solely on swing sequence. """
import pandas as pd

def detect_trend_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect market trend state based on confirmed market structure sequences.

    Requires input from `detect_market_structure()`, which must contain:
        - 'swing_type'
        - 'is_higher_high', 'is_lower_high'
        - 'is_higher_low', 'is_lower_low'

    Trend is defined as:
        - 'uptrend': after a Higher Low (HL) is followed by a Higher High (HH)
        - 'downtrend': after a Lower High (LH) is followed by a Lower Low (LL)
        - 'neutral': no confirmed structure or ambiguous state

    The trend state is assigned starting at the bar of the **second swing**
    in the confirming pair (e.g., HH bar for uptrend) and persists until
    a new structure is confirmed.

    Args:
        df: DataFrame with market structure labels.

    Returns:
        DataFrame with same index and columns as input, plus 'trend_state' column.

    Raises:
        KeyError: If required structure columns are missing.
    """
    required_cols = {
        'swing_type',
        'is_higher_high', 'is_lower_high',
        'is_higher_low', 'is_lower_low'
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for trend detection: {missing}")

    result = df.copy()
    result['trend_state'] = 'neutral'

    # Get swing points in order
    swing_mask = result['swing_type'].notna()
    if not swing_mask.any():
        return result

    swing_df = result[swing_mask].copy()
    swing_df = swing_df.sort_index()  # ensure chronological order

    # Track last relevant structure events
    last_hl_idx = None   # index of last Higher Low
    last_lh_idx = None   # index of last Lower High
    current_trend = 'neutral'

    # We'll record at which index a trend was confirmed
    trend_start_idx = None
    pending_trend = None  # 'uptrend' or 'downtrend' waiting to be applied

    for idx in swing_df.index:
        row = swing_df.loc[idx]

        if row['swing_type'] == 'low':
            if row['is_higher_low']:
                last_hl_idx = idx
            elif row['is_lower_low']:
                # Check: was there a prior LH before this LL?
                if last_lh_idx is not None and last_lh_idx < idx:
                    # Confirm downtrend starting at this LL
                    pending_trend = 'downtrend'
                    trend_start_idx = idx

        elif row['swing_type'] == 'high':
            if row['is_lower_high']:
                last_lh_idx = idx
            elif row['is_higher_high']:
                # Check: was there a prior HL before this HH?
                if last_hl_idx is not None and last_hl_idx < idx:
                    # Confirm uptrend starting at this HH
                    pending_trend = 'uptrend'
                    trend_start_idx = idx

        # Apply pending trend from its start index onward
        if pending_trend is not None:
            # Fill from trend_start_idx to current idx (inclusive)
            mask = (result.index >= trend_start_idx) & (result.index <= idx)
            result.loc[mask, 'trend_state'] = pending_trend
            current_trend = pending_trend
            pending_trend = None  # reset

    # Propagate final trend state to the end
    if current_trend != 'neutral' and trend_start_idx is not None:
        mask = result.index >= trend_start_idx
        result.loc[mask, 'trend_state'] = current_trend

    return result