import pandas as pd
import numpy as np


def detect_swing_points(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    Detect Swing Highs and Swing Lows in OHLC data.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'High' and 'Low' columns.
    n : int
        Lookback/lookforward window (number of candles on each side to compare).
        Must be >= 1.

    Returns:
    --------
    pd.DataFrame with boolean columns:
        - 'is_swing_high'
        - 'is_swing_low'
    """
    if n < 1:
        raise ValueError("Lookback period 'n' must be at least 1.")

    high = df['High']
    low = df['Low']

    # Rolling window: check if current high is the max in [i-n, i+n]
    # We use rolling with center=True to align window around each point
    window_size = 2 * n + 1

    # For swing high: current high == max over window AND not flat (to avoid plateaus)
    rolling_high = high.rolling(window=window_size, center=True).max()
    rolling_low = low.rolling(window=window_size, center=True).min()

    is_swing_high = (high == rolling_high)
    is_swing_low = (low == rolling_low)

    # Optional: Exclude flat regions (where multiple candles share same high/low)
    # To enforce strict inequality: current > all neighbors
    # We do this by checking neighbors directly
    is_swing_high = (
            (high > high.shift(1)) &
            (high > high.shift(-1))
    )
    is_swing_low = (
            (low < low.shift(1)) &
            (low < low.shift(-1))
    )

    # Now extend to 'n' candles on each side
    for i in range(2, n + 1):
        is_swing_high &= (high > high.shift(i)) & (high > high.shift(-i))
        is_swing_low &= (low < low.shift(i)) & (low < low.shift(-i))

    # Handle boundaries: first n and last n rows can't be swing points
    is_swing_high.iloc[:n] = False
    is_swing_high.iloc[-n:] = False
    is_swing_low.iloc[:n] = False
    is_swing_low.iloc[-n:] = False

    return pd.DataFrame({
        'is_swing_high': is_swing_high,
        'is_swing_low': is_swing_low
    })