# structure/failures/mtf.py
"""Multi-timeframe (MTF) context for structure break detection."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from structure.failures.config import StructureBreakConfig


def resample_ohlc_to_htf(df_ltf: pd.DataFrame, htf_rule: str) -> pd.DataFrame:
    """
    Resample LTF OHLC to higher timeframe.

    Args:
        df_ltf: Low-timeframe DataFrame with OHLC
        htf_rule: Pandas resampling rule (e.g., '15T', '1H', '1D')

    Returns:
        Resampled HTF DataFrame
    """
    if df_ltf.empty:
        return df_ltf.copy()

    # Ensure datetime index
    if not isinstance(df_ltf.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for resampling")

    # Resample OHLC
    df_resampled = df_ltf.resample(htf_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    return df_resampled


def run_full_pipeline_on_htf(
        df_htf: pd.DataFrame,
        config: StructureBreakConfig
) -> pd.DataFrame:
    """
    Run full detection pipeline on a single HTF.

    Args:
        df_htf: Higher timeframe DataFrame
        config: Configuration object

    Returns:
        HTF DataFrame with all analysis columns
    """
    # Import here to avoid circular imports
    from structure.failures.swings import detect_swing_points, detect_market_structure
    from structure.failures.trends import detect_trend_state
    from structure.failures.detector import detect_structure_breaks

    # Validate required columns
    required = {'open', 'high', 'low', 'close'}
    if not required.issubset(df_htf.columns):
        raise ValueError(f"HTF DataFrame missing required columns: {required - set(df_htf.columns)}")

    # Run pipeline
    df_result = df_htf.copy()

    # 1. Swing detection
    df_result = detect_swing_points(df_result, half_window=2)

    # 2. Market structure
    df_result = detect_market_structure(df_result)

    # 3. Trend state
    df_result = detect_trend_state(df_result, invalidation_buffer=0.0, include_metrics=True)

    # 4. Structure breaks (without MTF to avoid recursion)
    config_no_mtf = StructureBreakConfig(
        **{k: v for k, v in config.__dict__.items() if not k.startswith('mtf_')},
        mtf_enabled=False  # Disable MTF for HTF analysis
    )
    df_result = detect_structure_breaks(df_result, config_no_mtf)

    return df_result


def align_htf_signals_to_ltf(
        df_htf: pd.DataFrame,
        ltf_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Forward-fill HTF signals to LTF index.

    Args:
        df_htf: Higher timeframe DataFrame with signals
        ltf_index: Low timeframe index to align to

    Returns:
        DataFrame aligned to LTF index
    """
    if df_htf.empty:
        return pd.DataFrame(index=ltf_index)

    # Select relevant columns
    signal_patterns = [
        'is_',  # All signals
        'trend_state',
        'market_regime',
        'session',
        'zone_strength',
        'is_confluence_zone',
        'retest_quality'
    ]

    cols_to_align = []
    for col in df_htf.columns:
        if any(pattern in col for pattern in signal_patterns):
            cols_to_align.append(col)

    if not cols_to_align:
        return pd.DataFrame(index=ltf_index)

    # Forward fill HTF data to LTF bars
    htf_relevant = df_htf[cols_to_align]

    # Reindex and forward fill
    aligned = htf_relevant.reindex(ltf_index, method='ffill')

    # Backfill initial values
    aligned = aligned.fillna(method='bfill')

    return aligned


def calculate_mtf_confluence_score(
        df_ltf: pd.DataFrame,
        htf_contexts: Dict[str, pd.DataFrame],
        config: StructureBreakConfig
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Calculate MTF confluence score based on LTF-HTF alignment.

    Args:
        df_ltf: Low timeframe DataFrame with signals
        htf_contexts: Dictionary of {htf_rule: aligned_dataframe}
        config: Configuration object

    Returns:
        Tuple of (updated DataFrame, confluence scores)
    """
    n = len(df_ltf)
    confluence_scores = np.ones(n, dtype=np.float32)  # Default 1.0

    if not htf_contexts:
        return df_ltf, confluence_scores

    # Track confluence per bar
    for i in range(n):
        total_htfs = len(htf_contexts)
        aligned_htfs = 0

        # Check each HTF timeframe
        for htf_rule, htf_df in htf_contexts.items():
            if i >= len(htf_df):
                continue

            # 1. Check trend alignment
            ltf_trend = df_ltf['trend_state'].iloc[i] if 'trend_state' in df_ltf.columns else 'neutral'
            htf_trend = htf_df['trend_state'].iloc[i] if 'trend_state' in htf_df.columns else 'neutral'

            if ltf_trend != 'neutral' and htf_trend == ltf_trend:
                aligned_htfs += 1

            # 2. Check market regime alignment
            if 'market_regime' in df_ltf.columns and 'market_regime' in htf_df.columns:
                ltf_regime = df_ltf['market_regime'].iloc[i]
                htf_regime = htf_df['market_regime'].iloc[i]

                # Both trending or both ranging/chop
                ltf_is_trending = 'trend' in str(ltf_regime)
                htf_is_trending = 'trend' in str(htf_regime)

                if ltf_is_trending == htf_is_trending:
                    aligned_htfs += 1

        # Calculate score for this bar
        if total_htfs > 0:
            confluence_scores[i] = aligned_htfs / (total_htfs * 2)  # *2 for trend + regime checks

    return df_ltf, confluence_scores