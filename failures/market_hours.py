# structure/failures/market_hours.py
"""Market session and liquidity time detection."""
import pandas as pd
import numpy as np
from typing import Optional, Dict


def tag_sessions(
        df: pd.DataFrame,
        config: Optional[Dict] = None
) -> pd.Series:
    """
    Tag each bar with market session based on UTC hour.

    Returns categorical Series with values:
        'asia' | 'london' | 'ny' | 'overlap' | 'low_liquidity' | 'unknown'
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series('unknown', index=df.index, dtype="category")

    default_config = {
        'asia_start': 0,
        'asia_end': 7,
        'london_start': 7,
        'london_end': 16,
        'ny_start': 12,
        'ny_end': 21,
        'london_ny_overlap_start': 12,
        'london_ny_overlap_end': 16,
        'timezone': 'UTC'
    }

    cfg = {**default_config, **(config or {})}

    if df.index.tz is None:
        index_utc = df.index.tz_localize(cfg['timezone']).tz_convert('UTC')
    else:
        index_utc = df.index.tz_convert('UTC')

    hours = index_utc.hour.values
    n = len(hours)

    sessions = np.full(n, 'low_liquidity', dtype=object)

    asia_mask = (hours >= cfg['asia_start']) & (hours < cfg['asia_end'])
    sessions[asia_mask] = 'asia'

    london_mask = (hours >= cfg['london_start']) & (hours < cfg['london_end'])
    sessions[london_mask] = 'london'

    ny_mask = (hours >= cfg['ny_start']) & (hours < cfg['ny_end'])
    sessions[ny_mask] = 'ny'

    overlap_mask = (
            (hours >= cfg['london_ny_overlap_start']) &
            (hours < cfg['london_ny_overlap_end'])
    )
    sessions[overlap_mask] = 'overlap'

    return pd.Series(
        sessions,
        index=df.index,
        dtype="category"
    ).cat.set_categories(['asia', 'london', 'ny', 'overlap', 'low_liquidity', 'unknown'])


def add_liquidity_awareness(
        df: pd.DataFrame,
        volatility_col: str = 'atr',
        session_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Add liquidity awareness features to DataFrame.

    Args:
        df: Input DataFrame
        volatility_col: Column name for volatility measure
        session_col: Optional existing session column

    Returns:
        DataFrame with added liquidity features
    """
    result = df.copy()

    if session_col is None or session_col not in result.columns:
        result['session'] = tag_sessions(result)
        session_col = 'session'

    liquidity_score = np.zeros(len(result), dtype=np.float32)

    session_weights = {
        'overlap': 1.0,
        'ny': 0.8,
        'london': 0.7,
        'asia': 0.5,
        'low_liquidity': 0.3,
        'unknown': 0.5
    }

    sessions = result[session_col].astype(str).values
    for i, session in enumerate(sessions):
        liquidity_score[i] = session_weights.get(session, 0.5)

    if volatility_col in result.columns:
        volatility = result[volatility_col].values
        n = len(volatility)
        vol_percentile = np.zeros(n, dtype=np.float32)

        lookback = min(50, n)
        for i in range(n):
            start = max(0, i - lookback + 1)
            window = volatility[start:i + 1]
            if len(window) > 0:
                vol_percentile[i] = np.sum(window <= volatility[i]) / len(window)

        liquidity_score *= (1.0 - vol_percentile * 0.5)

    result['liquidity_score'] = liquidity_score
    result['is_high_liquidity'] = liquidity_score > 0.7

    return result