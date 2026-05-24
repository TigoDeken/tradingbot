import numpy as np
import pandas as pd

SESSIONS = {
    "asia":    (0, 8),
    "london":  (8, 13),
    "overlap": (13, 16),
    "ny":      (16, 21),
}

_FWD_HORIZON = 10  # bars used for outcome computation


def _forward_max_high(high, n):
    """max(High[i+1], ..., High[i+n]) — outcome variable only, not a feature."""
    return pd.concat([high.shift(-(k + 1)) for k in range(n)], axis=1).max(axis=1)


def _forward_min_low(low, n):
    """min(Low[i+1], ..., Low[i+n]) — outcome variable only, not a feature."""
    return pd.concat([low.shift(-(k + 1)) for k in range(n)], axis=1).min(axis=1)


def _session_label(hour):
    for name, (s, e) in SESSIONS.items():
        if s <= hour < e:
            return name
    return "other"


def build_features(df):
    """
    Add bar-structure features and forward-outcome columns to df.

    Feature columns (use as X):
      hour_utc, dow, session
      bar_range, bar_direction, body_pct, close_position
      upper_wick_ratio, lower_wick_ratio, body_position

    Outcome columns (use as Y only — contain future data):
      fwd_close_1, fwd_close_5, fwd_close_10
      fwd_max_h_10, fwd_min_l_10
    """
    h, l, o, c = df["High"], df["Low"], df["Open"], df["Close"]

    bar_range = (h - l).replace(0, np.nan)
    body      = (c - o).abs()
    hi_oc     = pd.concat([o, c], axis=1).max(axis=1)
    lo_oc     = pd.concat([o, c], axis=1).min(axis=1)

    out = df.copy()

    # Time features (no lookahead: index is the bar's open time)
    out["hour_utc"] = df.index.hour
    out["dow"]      = df.index.dayofweek
    out["session"]  = out["hour_utc"].map(_session_label)

    # Bar structure features
    out["bar_range"]         = h - l
    out["bar_direction"]     = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(np.int8)
    out["body_pct"]          = body / bar_range
    out["close_position"]    = (c - l) / bar_range
    out["upper_wick_ratio"]  = (h - hi_oc) / bar_range
    out["lower_wick_ratio"]  = (lo_oc - l) / bar_range
    out["body_position"]     = ((o + c) / 2.0 - l) / bar_range

    # range_expansion uses pre_atr (already in df from range_builder)
    if "pre_atr" in df.columns:
        out["range_expansion"] = (h - l) / df["pre_atr"].replace(0, np.nan)

    # ── Outcome columns (future data — Y variables only) ─────────────────────
    out["fwd_close_1"]  = c.shift(-1)
    out["fwd_close_5"]  = c.shift(-5)
    out["fwd_close_10"] = c.shift(-_FWD_HORIZON)
    out["fwd_max_h_10"] = _forward_max_high(h, _FWD_HORIZON)
    out["fwd_min_l_10"] = _forward_min_low(l, _FWD_HORIZON)

    return out
