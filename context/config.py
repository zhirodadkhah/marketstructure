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
    clustering_radius: float = 1.5  # in ATR units
    min_cluster_size: int = 3
    recent_bars: int = 200  # only consider recent swings

@dataclass(frozen=True)
class MTFConfig:
    htf_bar_size: int = 24  # e.g., 24 → 1D from 1H
    alignment_method: str = "backward"  # "backward", "forward", "center"