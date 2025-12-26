"""
Shared test fixtures and configurations.
"""
import numpy as np
import pytest
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class TestData:
    """Container for test data."""
    n: int = 100
    seed: int = 42

    def __post_init__(self):
        np.random.seed(self.seed)

    @property
    def linear_prices(self) -> np.ndarray:
        """Perfectly efficient linear price movement."""
        return np.linspace(100, 120, self.n, dtype=np.float32)

    @property
    def random_prices(self) -> np.ndarray:
        """Random walk prices."""
        returns = np.random.normal(0.001, 0.02, self.n)
        prices = 100 * np.exp(np.cumsum(returns))
        return prices.astype(np.float32)

    @property
    def sine_prices(self) -> np.ndarray:
        """Sine wave pattern (choppy market)."""
        t = np.linspace(0, 4*np.pi, self.n)
        return (100 + 5 * np.sin(t)).astype(np.float32)

    @property
    def gap_prices(self) -> np.ndarray:
        """Prices with gaps (for TR testing)."""
        prices = np.ones(self.n, dtype=np.float32) * 100
        prices[10] = 110  # Up gap
        prices[30] = 90   # Down gap
        return prices

    @property
    def constant_prices(self) -> np.ndarray:
        """Constant prices (edge case)."""
        return np.full(self.n, 100.0, dtype=np.float32)

    @property
    def ohlc_data(self) -> Dict[str, np.ndarray]:
        """Realistic OHLC data."""
        close = self.random_prices
        high = close + np.abs(np.random.normal(0.5, 0.2, self.n))
        low = close - np.abs(np.random.normal(0.5, 0.2, self.n))
        open_price = np.roll(close, 1)
        open_price[0] = close[0] - np.random.normal(0, 0.5)

        return {
            'open': open_price.astype(np.float32),
            'high': high.astype(np.float32),
            'low': low.astype(np.float32),
            'close': close.astype(np.float32)
        }

    @property
    def swing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Realistic swing high/low masks."""
        swing_high = np.zeros(self.n, dtype=bool)
        swing_low = np.zeros(self.n, dtype=bool)

        # Create alternating pattern
        for i in range(10, self.n, 20):
            if i < self.n:
                swing_high[i] = True
            if i + 10 < self.n:
                swing_low[i + 10] = True

        # Add some divergences
        swing_high[50] = True
        swing_high[70] = True
        swing_low[60] = True
        swing_low[80] = True

        return swing_high, swing_low

    @property
    def atr_data(self) -> np.ndarray:
        """Realistic ATR data."""
        return np.random.uniform(1.0, 3.0, self.n).astype(np.float32)

# Global test data fixture
@pytest.fixture
def test_data() -> TestData:
    return TestData()

# Edge case fixtures
@pytest.fixture
def edge_cases() -> Dict[str, np.ndarray]:
    return {
        'empty': np.array([], dtype=np.float32),
        'single': np.array([100.0], dtype=np.float32),
        'two_elements': np.array([100.0, 101.0], dtype=np.float32),
        'all_nan': np.full(10, np.nan, dtype=np.float32),
        'all_zero': np.zeros(10, dtype=np.float32),
        'all_same': np.full(10, 100.0, dtype=np.float32),
        'extremely_large': np.array([1e10, 1e10 + 1], dtype=np.float32),
        'extremely_small': np.array([1e-10, 2e-10], dtype=np.float32),
        'mixed_sign': np.array([-100, 100, -50, 50], dtype=np.float32),
    }

@pytest.fixture
def invalid_inputs() -> Dict[str, any]:
    return {
        'none_input': None,
        'string_input': 'not an array',
        'list_input': [1, 2, 3],
        'wrong_dtype': np.array([1, 2, 3], dtype=np.int64),
        'wrong_shape_2d': np.array([[1, 2], [3, 4]], dtype=np.float32),
        'mismatched_lengths': (
            np.array([1, 2, 3], dtype=np.float32),
            np.array([1, 2], dtype=np.float32)
        ),
    }
