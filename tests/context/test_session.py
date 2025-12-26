# test_session.py
import pytest
import numpy as np
from structure.context.session import detect_sessions
from structure.context.config import SessionConfig

def test_session_positive():
    ts = np.array(['2025-12-26T03:00', '2025-12-26T10:00', '2025-12-26T18:00'], dtype='datetime64[s]')
    config = SessionConfig()
    result = detect_sessions(ts, config)
    assert result['is_asian_session'][0]
    assert result['is_london_session'][1]
    assert result['is_ny_session'][2]

def test_session_negative_wrong_dim():
    ts = np.array([['2025-12-26T12:00']], dtype='datetime64[s]')  # 2D
    config = SessionConfig()
    with pytest.raises(ValueError, match="1D"):
        detect_sessions(ts, config)

def test_session_edge_empty():
    ts = np.array([], dtype='datetime64[s]')
    config = SessionConfig()
    result = detect_sessions(ts, config)
    assert len(result['is_asian_session']) == 0