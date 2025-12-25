# structure/failures/signal_filter.py
"""Regime-based signal filtering and validation."""
from typing import Dict, List, Set, Optional
import pandas as pd
import numpy as np
from structure.failures.config import StructureBreakConfig


class RegimeSignalFilter:
    """Filter and validate structure break signals based on market regime."""

    def __init__(self, config: Optional[StructureBreakConfig] = None):
        self.config = config or StructureBreakConfig()

        self.regime_preferences = {
            'strong_trend': {
                'preferred': {
                    'bullish': ['is_bos_bullish_confirmed', 'is_bos_bullish_momentum'],
                    'bearish': ['is_bos_bearish_confirmed', 'is_bos_bearish_momentum'],
                },
                'avoid': ['is_choch_bullish', 'is_choch_bearish']
            },
            'weak_trend': {
                'preferred': {
                    'bullish': ['is_bos_bullish_confirmed'],
                    'bearish': ['is_bos_bearish_confirmed'],
                },
                'avoid': ['is_bos_bullish_momentum', 'is_bos_bearish_momentum']
            },
            'ranging': {
                'preferred': {
                    'bullish': ['is_bos_bullish_initial', 'is_choch_bullish'],
                    'bearish': ['is_bos_bearish_initial', 'is_choch_bearish'],
                },
                'avoid': ['is_bos_bullish_momentum', 'is_bos_bearish_momentum']
            },
            'chop': {
                'preferred': {
                    'bullish': ['is_failed_choch_bearish', 'is_bullish_break_failure'],
                    'bearish': ['is_failed_choch_bullish', 'is_bearish_break_failure'],
                },
                'avoid': ['is_bos_bullish_initial', 'is_bos_bearish_initial',
                          'is_bos_bullish_confirmed', 'is_bos_bearish_confirmed']
            },
            'neutral': {
                'preferred': {
                    'bullish': [],
                    'bearish': [],
                },
                'avoid': []
            }
        }

        self.session_quality = {
            'overlap': 1.2,
            'ny': 1.1,
            'london': 1.0,
            'asia': 0.8,
            'low_liquidity': 0.5,
            'unknown': 0.7
        }

    def calculate_signal_quality(
            self,
            df: pd.DataFrame,
            signal_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate quality scores for each signal based on regime and context."""
        if signal_columns is None:
            from structure.failures.entity import SIGNAL_COLS
            signal_columns = [col for col in SIGNAL_COLS if col in df.columns]

        result = df.copy()
        n = len(result)

        for col in signal_columns:
            quality_col = f"{col}_quality"
            result[quality_col] = 0.0

        if 'market_regime' not in result.columns:
            for col in signal_columns:
                quality_col = f"{col}_quality"
                result[quality_col] = 0.5
            return result

        for i in range(n):
            regime = result.iloc[i]['market_regime']
            session = result.iloc[i].get('session', 'unknown')
            liquidity = result.iloc[i].get('liquidity_score', 0.5)

            if regime in self.regime_preferences:
                prefs = self.regime_preferences[regime]
                avoid_set = set(prefs['avoid'])

                for col in signal_columns:
                    if result.iloc[i][col]:
                        is_bullish = 'bullish' in col
                        signal_type = 'bullish' if is_bullish else 'bearish'

                        if col in avoid_set:
                            base_quality = 0.2
                        elif col in prefs['preferred'].get(signal_type, []):
                            base_quality = 0.9
                        else:
                            base_quality = 0.5

                        session_mult = self.session_quality.get(str(session), 0.7)
                        base_quality *= session_mult

                        if liquidity > 0.7:
                            base_quality *= 1.1
                        elif liquidity < 0.3:
                            base_quality *= 0.8

                        final_quality = max(0.0, min(1.0, base_quality))

                        quality_col = f"{col}_quality"
                        result.at[result.index[i], quality_col] = final_quality

        return result

    def filter_by_regime(
            self,
            df: pd.DataFrame,
            min_quality: float = 0.6,
            filter_avoided: bool = True
    ) -> pd.DataFrame:
        """Filter signals based on regime context."""
        result = df.copy()

        quality_cols = [col for col in result.columns if col.endswith('_quality')]
        if not quality_cols:
            result = self.calculate_signal_quality(result)

        signal_columns = [col for col in result.columns
                          if col.startswith('is_') and not col.endswith('_quality')]

        for col in signal_columns:
            quality_col = f"{col}_quality"
            if quality_col in result.columns:
                low_quality_mask = result[quality_col] < min_quality
                result.loc[low_quality_mask, col] = False

                if filter_avoided:
                    for i in range(len(result)):
                        if result.iloc[i][col]:
                            regime = result.iloc[i]['market_regime']
                            if regime in self.regime_preferences:
                                avoid_set = self.regime_preferences[regime]['avoid']
                                if col in avoid_set:
                                    result.at[result.index[i], col] = False

        return result

    def get_regime_summary(self, df: pd.DataFrame) -> Dict:
        """Generate a summary of regime distribution and signal statistics."""
        if 'market_regime' not in df.columns:
            return {"error": "No regime data available"}

        summary = {}

        regime_counts = df['market_regime'].value_counts()
        summary['regime_distribution'] = regime_counts.to_dict()

        signal_columns = [col for col in df.columns
                          if col.startswith('is_') and not col.endswith('_quality')]

        regime_signals = {}
        for regime in df['market_regime'].unique():
            regime_mask = df['market_regime'] == regime
            regime_df = df[regime_mask]

            signal_counts = {}
            for col in signal_columns:
                if col in regime_df.columns:
                    count = regime_df[col].sum()
                    if count > 0:
                        signal_counts[col] = int(count)

            regime_signals[str(regime)] = {
                'count': int(regime_mask.sum()),
                'signals': signal_counts
            }

        summary['signals_by_regime'] = regime_signals

        if 'session' in df.columns:
            session_counts = df['session'].value_counts()
            summary['session_distribution'] = session_counts.to_dict()

        return summary