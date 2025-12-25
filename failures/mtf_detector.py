# structure/failures/mtf_detector.py
"""MTF-aware structure break detector."""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from structure.failures.config import StructureBreakConfig
from structure.failures.mtf import (
    resample_ohlc_to_htf,
    run_full_pipeline_on_htf,
    align_htf_signals_to_ltf,
    calculate_mtf_confluence_score
)
from structure.failures.detector import detect_structure_breaks


def detect_mtf_structure_breaks(
        df_ltf: pd.DataFrame,
        config: Optional[StructureBreakConfig] = None
) -> pd.DataFrame:
    """
    Detect structure breaks with MTF confluence context.

    Args:
        df_ltf: Low-timeframe DataFrame with DatetimeIndex and OHLC
        config: Configuration with MTF settings

    Returns:
        LTF DataFrame with:
        - All original signals
        - HTF context columns (e.g., 'htf_trend_1H')
        - MTF confluence score
    """
    if config is None:
        config = StructureBreakConfig()

    if df_ltf.empty:
        return df_ltf.copy()

    if not isinstance(df_ltf.index, pd.DatetimeIndex):
        raise ValueError("MTF detection requires DatetimeIndex")

    # Step 1: Run base LTF detection
    print("Running base LTF detection...")
    df_result = detect_structure_breaks(df_ltf, config)

    # Step 2: Check if MTF is enabled
    if not config.mtf_enabled or not config.mtf_periods:
        df_result['mtf_confluence_score'] = 1.0
        df_result['mtf_htf_count'] = 0
        return df_result

    print(f"Processing MTF timeframes: {config.mtf_periods}")

    # Step 3: Process each HTF period
    htf_contexts: Dict[str, pd.DataFrame] = {}
    htf_context_columns = []

    for htf_rule in config.mtf_periods:
        try:
            print(f"  Processing {htf_rule} timeframe...")

            # 1. Resample to HTF
            df_htf = resample_ohlc_to_htf(df_ltf, htf_rule)
            if df_htf.empty or len(df_htf) < 10:
                print(f"    Skipping {htf_rule}: insufficient data")
                continue

            # 2. Run full pipeline on HTF
            df_htf_analysis = run_full_pipeline_on_htf(df_htf, config)

            # 3. Align to LTF
            df_htf_aligned = align_htf_signals_to_ltf(df_htf_analysis, df_ltf.index)

            if df_htf_aligned.empty:
                print(f"    Skipping {htf_rule}: alignment failed")
                continue

            # 4. Rename columns with HTF suffix
            rename_map = {}
            for col in df_htf_aligned.columns:
                # Create safe suffix
                suffix = htf_rule.replace(' ', '_').replace('/', '_').replace('T', 'min')
                new_col = f"htf_{col}_{suffix}"
                rename_map[col] = new_col

            df_htf_aligned = df_htf_aligned.rename(columns=rename_map)

            # 5. Store context
            htf_contexts[htf_rule] = df_htf_aligned
            htf_context_columns.extend(df_htf_aligned.columns.tolist())

            # 6. Merge into result
            df_result = df_result.join(df_htf_aligned, how='left')

            print(f"    Added {len(df_htf_aligned.columns)} columns from {htf_rule}")

        except Exception as e:
            print(f"    Error processing {htf_rule}: {e}")
            continue

    # Step 4: Calculate MTF confluence score
    if htf_contexts:
        df_result, confluence_scores = calculate_mtf_confluence_score(
            df_result, htf_contexts, config
        )
        df_result['mtf_confluence_score'] = confluence_scores
        df_result['mtf_htf_count'] = len(htf_contexts)

        print(f"MTF analysis complete: {len(htf_contexts)} timeframes processed")
    else:
        df_result['mtf_confluence_score'] = 1.0
        df_result['mtf_htf_count'] = 0
        print("MTF analysis skipped: no valid HTF data")

    return df_result


def filter_signals_by_mtf_confluence(
        df: pd.DataFrame,
        min_confluence_score: float = 0.6,
        signal_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter signals based on MTF confluence score.

    Args:
        df: DataFrame with signals and MTF columns
        min_confluence_score: Minimum confluence score required
        signal_columns: List of signal columns to filter

    Returns:
        DataFrame with filtered signals
    """
    if 'mtf_confluence_score' not in df.columns:
        return df

    result = df.copy()

    if signal_columns is None:
        # Auto-detect signal columns
        signal_columns = [col for col in df.columns if col.startswith('is_')]

    # Filter signals below confluence threshold
    low_confluence = df['mtf_confluence_score'] < min_confluence_score

    for col in signal_columns:
        if col in result.columns:
            result.loc[low_confluence, col] = False

    return result


def get_mtf_summary(df: pd.DataFrame) -> Dict:
    """
    Generate MTF analysis summary.

    Args:
        df: DataFrame with MTF columns

    Returns:
        Dictionary with MTF summary statistics
    """
    summary = {
        'has_mtf_data': 'mtf_confluence_score' in df.columns,
        'total_bars': len(df)
    }

    if not summary['has_mtf_data']:
        return summary

    # Confluence score distribution
    summary['avg_confluence_score'] = float(df['mtf_confluence_score'].mean())
    summary['median_confluence_score'] = float(df['mtf_confluence_score'].median())

    # Score buckets
    score_ranges = {
        'low': (0.0, 0.3),
        'medium': (0.3, 0.7),
        'high': (0.7, 1.0)
    }

    for name, (low, high) in score_ranges.items():
        mask = (df['mtf_confluence_score'] >= low) & (df['mtf_confluence_score'] < high)
        count = mask.sum()
        summary[f'confluence_{name}'] = int(count)
        summary[f'confluence_{name}_pct'] = float(count / len(df) * 100)

    # HTF count if available
    if 'mtf_htf_count' in df.columns:
        summary['htf_count'] = int(df['mtf_htf_count'].iloc[0]) if not df.empty else 0

    # Signal-confluence correlation
    if 'is_bos_bullish_initial' in df.columns:
        bullish_confluence = df[df['is_bos_bullish_initial']]['mtf_confluence_score'].mean()
        summary['bullish_signals_avg_confluence'] = float(bullish_confluence) if not pd.isna(
            bullish_confluence) else 0.0

    if 'is_bos_bearish_initial' in df.columns:
        bearish_confluence = df[df['is_bos_bearish_initial']]['mtf_confluence_score'].mean()
        summary['bearish_signals_avg_confluence'] = float(bearish_confluence) if not pd.isna(
            bearish_confluence) else 0.0

    return summary