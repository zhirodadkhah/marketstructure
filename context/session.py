# structure/context/test_session.py
"""
Session tagging based on UTC timestamps.
No pandas, no DataFrame — pure NumPy datetime64 logic.
"""
from typing import Dict
import numpy as np
from structure.metrics.types import Timestamps, SessionMask
from .config import SessionConfig


def detect_sessions(
    timestamps: Timestamps,
    config: SessionConfig
) -> Dict[str, SessionMask]:
    """
    Tag bars with major trading sessions (Asian, London, NY).

    Parameters
    ----------
    timestamps : Timestamps
        UTC timestamps as datetime64[ns] or [s].
    config : SessionConfig
        Session window definitions.

    Returns
    -------
    dict of SessionMask:
        - 'is_asian_session'
        - 'is_london_session'
        - 'is_ny_session'

    Pre-conditions
    --------------
    - timestamps is 1D datetime64 array.
    - config times in seconds since midnight UTC.

    Post-conditions
    ---------------
    - Output masks same length as input.
    - Sessions may overlap (e.g., London/NY).

    Notes
    -----
    - Assumes daily session cycle (no DST handling).
    - Timezone offset applied to align to UTC.
    """
    # === PRECONDITIONS ===
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be 1D")
    if timestamps.dtype.kind != 'M':
        raise TypeError("timestamps must be datetime64")

    # Convert to seconds since epoch
    ts_seconds = timestamps.astype('datetime64[s]').astype(np.int64)
    # Seconds since midnight UTC
    seconds_since_midnight = (ts_seconds + config.timezone_offset) % 86400

    # === SESSION MASKS ===
    is_asian = (
        (seconds_since_midnight >= config.asian_open) &
        (seconds_since_midnight < config.asian_close)
    )
    is_london = (
        (seconds_since_midnight >= config.london_open) &
        (seconds_since_midnight < config.london_close)
    )
    is_ny = (
        (seconds_since_midnight >= config.ny_open) &
        (seconds_since_midnight < config.ny_close)
    )

    return {
        'is_asian_session': is_asian,
        'is_london_session': is_london,
        'is_ny_session': is_ny
    }