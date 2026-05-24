"""
N=8 down-break signal detection and session labelling.
No ATR filter. No session filter. Pure price structure.
"""

import numpy as np
import pandas as pd

N_BARS = 8

SESSIONS = {
    "asia":    (0,  8),
    "london":  (8,  13),
    "ny":      (16, 21),
}


def detect_signals(df):
    """
    Returns boolean Series: True where close < min of previous 8 closes.
    shift(1) guarantees no lookahead — bar i only sees closes up to i-1.
    """
    c       = df["Close"]
    c_min_n = c.shift(1).rolling(N_BARS).min()
    return (c < c_min_n).fillna(False)


def session_of(hour):
    for name, (s, e) in SESSIONS.items():
        if s <= hour < e:
            return name
    return "other"


def add_session(df):
    out = df.copy()
    out["hour_utc"] = df.index.hour
    out["session"]  = out["hour_utc"].map(session_of)
    return out
