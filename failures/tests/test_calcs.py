# test_calcs.py
import pytest
import pandas as pd
import numpy as np
from structure.failures.calcs import _compute_metrics


def test_compute_metrics_positive():
    """Positive test: normal OHLC data with valid swings and structure."""
    df = pd.DataFrame({
        'open': [1.0, 1.1, 1.2, 1.3, 1.4],
        'high': [1.2, 1.3, 1.4, 1.5, 1.6],
        'low': [0.9, 1.0, 1.1, 1.2, 1.3],
        'close': [1.1, 1.2, 1.3, 1.4, 1.5]
    })
    result = _compute_metrics(df)

    # Check all expected columns are present
    expected_cols = {
        'body', 'candle_range', 'body_ratio', 'close_location',
        'upper_wick', 'lower_wick', 'upper_wick_ratio', 'lower_wick_ratio',
        'is_bullish_body', 'is_bearish_body', 'atr',
        'momentum_ema', 'momentum_direction', 'momentum_strength',
        'volatility_regime', 'vol_percentile',
        'fractal_efficiency', 'is_efficient'
    }
    assert expected_cols.issubset(result.columns)

    # Basic sanity checks
    assert result['is_bullish_body'].all()
    assert (result['body_ratio'] >= 0).all()
    assert (result['body_ratio'] <= 1).all()
    assert (result['vol_percentile'] >= 0).all()
    assert (result['vol_percentile'] <= 1).all()


def test_compute_metrics_missing_column():
    """Negative test: missing 'close' column."""
    df = pd.DataFrame({
        'open': [1.0, 1.1],
        'high': [1.2, 1.3],
        'low': [0.9, 1.0],
        # 'close' is missing
    })
    with pytest.raises(KeyError, match="Missing required columns"):
        _compute_metrics(df)


def test_compute_metrics_empty_dataframe():
    """Edge case: empty DataFrame."""
    df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
    result = _compute_metrics(df)
    assert len(result) == 0
    assert 'atr' in result.columns


def test_compute_metrics_zero_range_candles():
    """Edge case: candles with zero range (flat)."""
    df = pd.DataFrame({
        'open': [1.0, 1.0, 1.0],
        'high': [1.0, 1.0, 1.0],
        'low': [1.0, 1.0, 1.0],
        'close': [1.0, 1.0, 1.0]
    })
    result = _compute_metrics(df)

    # Should not crash; body_ratio = 0, close_location = 0.5
    assert (result['body_ratio'] == 0.0).all()
    assert (result['close_location'] == 0.5).all()
    assert (result['upper_wick'] == 0.0).all()
    assert (result['lower_wick'] == 0.0).all()


def test_compute_metrics_with_nans():
    """Edge case: DataFrame with NaNs in price series."""
    df = pd.DataFrame({
        'open': [1.0, np.nan, 1.2],
        'high': [1.1, np.nan, 1.3],
        'low': [0.9, np.nan, 1.1],
        'close': [1.05, np.nan, 1.25]
    })
    result = _compute_metrics(df)

    # ATR is backfilled, so no NaNs in vol_percentile or momentum_strength
    # But momentum_ema may be NaN for early rows
    assert pd.isna(result.loc[0, 'momentum_ema'])  # first row: NaN from pct_change
    assert not pd.isna(result.loc[2, 'vol_percentile'])  # backfilled → valid

def test_compute_metrics_single_row():
    """Edge case: single-row DataFrame."""
    df = pd.DataFrame({
        'open': [1.0],
        'high': [1.1],
        'low': [0.9],
        'close': [1.05]
    })
    result = _compute_metrics(df)
    assert len(result) == 1
    # ATR should be floored to 0.001 or computed safely
    assert result['atr'].iloc[0] > 0


def test_compute_metrics_all_equal_prices():
    """Edge case: all prices identical → fractal efficiency = 0."""
    df = pd.DataFrame({
        'open': [1.0] * 20,
        'high': [1.0] * 20,
        'low': [1.0] * 20,
        'close': [1.0] * 20
    })
    result = _compute_metrics(df)
    # Net change = 0 → efficiency = 0
    assert (result['fractal_efficiency'] == 0.0).all()
    # Volatility regime should be 'low'
    assert result['volatility_regime'].iloc[-1] == 'low'