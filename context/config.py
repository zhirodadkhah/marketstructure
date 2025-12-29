# structure/context/config.py
"""
Structured configs for context layers.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Add atr_period to RegimeConfig
@dataclass(frozen=True)
class RegimeConfig:
    atr_period: int = 14
    volatility_window: int = 50
    regime_threshold: float = 0.7
    regime_swing_window: int = 100
    regime_atr_slope_window: int = 20
    regime_consistency_window: int = 50
    regime_swing_density_high: float = 0.02
    regime_swing_density_moderate: float = 0.01
    regime_swing_density_low: float = 0.005
    regime_efficiency_high: float = 0.5
    regime_efficiency_low: float = 0.3
    regime_consistency_high: float = 0.6
    regime_atr_slope_threshold: float = 0.001

@dataclass(frozen=True)
class SessionConfig:
    # Session start/end as seconds since midnight UTC
    asian_open: int = 0 * 3600
    asian_close: int = 8 * 3600
    london_open: int = 7 * 3600
    london_close: int = 16 * 3600
    ny_open: int = 12 * 3600
    ny_close: int = 21 * 3600
    timezone_offset: int = 0  # UTC offset in seconds


@dataclass(frozen=True)
class ZoneConfig:
    """Configuration for zone detection and confluence scoring."""

    # Core detection
    swing_window: int = 2
    recent_bars: int = 100
    clustering_radius: float = 1.0  # In ATR units
    min_cluster_size: int = 3
    buffer_multiplier: float = 0.5  # ATR multiplier for touch detection

    # Confluence thresholds
    min_confluence_strength: int = 4  # Zones with >= this strength are confluence

    # Multi-touch settings
    min_touch_bars: int = 3  # Minimum bars between touches to count as separate

    # ATR settings
    atr_period: int = 14

    # Score weights
    quality_weight: float = 0.4
    double_weight: float = 0.1
    triple_weight: float = 0.2
    confluence_weight: float = 0.3

    def __post_init__(self):
        """Validate configuration."""
        if self.swing_window < 1:
            raise ValueError("swing_window must be >= 1")
        if self.recent_bars < 10:
            raise ValueError("recent_bars must be >= 10")
        if self.clustering_radius <= 0:
            raise ValueError("clustering_radius must be > 0")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if self.min_confluence_strength < self.min_cluster_size:
            raise ValueError("min_confluence_strength must be >= min_cluster_size")
        if not (0.99 <= sum([self.quality_weight, self.double_weight,
                             self.triple_weight, self.confluence_weight]) <= 1.01):
            raise ValueError("Score weights must sum to approximately 1.0")


@dataclass(frozen=True)
class MTFConfig:
    htf_bar_size: int = 24  # e.g., 24 → 1D from 1H
    alignment_method: str = "backward"  # "backward", "forward", "center"