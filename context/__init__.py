# structure/context/__init__.py
"""
Context layer: market-aware logic (regime, session, zones, MTF).
All functions are pure, typed, and accept only arrays + config.
"""

from .regime import detect_market_regime
from .session import detect_sessions
from .zones import detect_sr_zones
from .mtf import resample_to_htf

# Re-export config and types for convenience
from .config import RegimeConfig, SessionConfig, ZoneConfig, MTFConfig


__all__ = [
    # Functions
    'detect_market_regime',
    'detect_sessions',
    'detect_sr_zones',
    'resample_to_htf',

    # Configs
    'RegimeConfig',
    'SessionConfig',
    'ZoneConfig',
    'MTFConfig',
]