"""
Regime filter for M5 bars: volatility + structure.

Volatility: compares 100-bar rolling ATR to 20-session baseline per session.
  Thresholds are per-session p33/p67 from 150k bars GOLD# M5 (2024-2026).

Structure: classifies bar-by-bar directional commitment over a rolling window.
  clean_up:   high > prev_high, low >= prev_low, close > prev_high
  clean_down: low < prev_low,  high <= prev_high, close < prev_low
  dirty:      everything else (outside bar, inside bar, wick rejection)
"""

import numpy as np
import pandas as pd

SESSION_WINDOWS = {
    "asia":   (0,  3),
    "london": (7,  10),
    "ny":     (12, 15),
}

# ── Volatility constants ──────────────────────────────────────────────────────
LOOKBACK_BARS    = 100
BARS_PER_SESSION = 36      # 3h × 12 bars/h on M5
BASELINE_DAYS    = 20
BASELINE_BARS    = BASELINE_DAYS * BARS_PER_SESSION  # 720 session bars

# Per-session (p33, p67) — derived from empirical ratio distribution
SESSION_THRESHOLDS = {
    "asia":   (1.0347, 1.6435),
    "london": (0.7844, 1.0642),
    "ny":     (0.8561, 1.0689),
}

# ── Structure constants ───────────────────────────────────────────────────────
STRUCTURE_WINDOW = 12    # default rolling window (1 hour on M5)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    h      = df["High"]
    l      = df["Low"]
    c_prev = df["Close"].shift(1)
    return pd.concat([
        (h - l),
        (h - c_prev).abs(),
        (l - c_prev).abs(),
    ], axis=1).max(axis=1)


# ── Volatility ────────────────────────────────────────────────────────────────

def compute_vol_ratio(df: pd.DataFrame) -> pd.Series:
    """Raw ratio = current_vol / session_baseline. NaN outside sessions / warmup."""
    tr          = _true_range(df)
    hour        = df.index.hour
    current_vol = tr.shift(1).rolling(LOOKBACK_BARS, min_periods=50).mean()
    baseline    = pd.Series(np.nan, index=df.index, dtype=float)
    for name, (start, end) in SESSION_WINDOWS.items():
        mask    = (hour >= start) & (hour < end)
        sess_tr = tr[mask]
        sess_bl = (
            sess_tr
            .shift(BARS_PER_SESSION)
            .rolling(BASELINE_BARS, min_periods=BASELINE_BARS // 2)
            .mean()
        )
        baseline.loc[mask] = sess_bl
    return current_vol / baseline.replace(0, np.nan)


def classify_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Returns "low" / "mid" / "high" / NaN per bar.
    NaN outside tracked sessions or during warmup.
    Thresholds are per-session p33/p67 from empirical data.
    """
    ratio = compute_vol_ratio(df)
    hour  = df.index.hour

    vol_class = pd.Series(np.nan, index=df.index, dtype=object)
    for name, (start, end) in SESSION_WINDOWS.items():
        lo, hi = SESSION_THRESHOLDS[name]
        mask   = (hour >= start) & (hour < end)
        r      = ratio[mask]
        cls    = pd.Series("mid", index=r.index, dtype=object)
        cls[r <  lo]  = "low"
        cls[r >  hi]  = "high"
        cls[r.isna()] = np.nan
        vol_class.loc[mask] = cls

    return vol_class


# ── Structure ─────────────────────────────────────────────────────────────────

def classify_structure(df: pd.DataFrame,
                       window: int = STRUCTURE_WINDOW) -> pd.Series:
    """
    Returns cleanness score 0–100 per bar.

    Per-bar classification vs previous bar:
      clean_up:   high > prev_high, low >= prev_low, close > prev_high
      clean_down: low < prev_low,  high <= prev_high, close < prev_low
      dirty:      outside bar, inside bar, wick rejection

    Score = (clean_up + clean_down) / window * 100
      100 = every bar in the window was a clean directional bar
      0   = every bar was dirty

    NaN during warmup (first window bars). No lookahead.
    Direction is intentionally excluded — that is the entry signal's job.
    """
    H, L, C = df["High"], df["Low"], df["Close"]
    prev_H  = H.shift(1)
    prev_L  = L.shift(1)

    clean_up   = (H > prev_H) & (L >= prev_L) & (C > prev_H)
    clean_down = (L < prev_L) & (H <= prev_H) & (C < prev_L)

    is_clean    = clean_up | clean_down
    clean_count = is_clean.rolling(window, min_periods=window).sum()

    return (clean_count / window * 100).where(clean_count.notna())
