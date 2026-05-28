"""
Frozen spec signal logic — single source of truth for live and dashboard.

Spec:
  Entry: funding_z crosses below -0.5  AND  atr_ratio < 0.80  AND  oi_z > 1.0
  Exit:  funding_z rises above 0.0
  Stop:  10-day lowest low, min 5% below entry
"""

import numpy as np
import pandas as pd

# ── Frozen parameters (must match final_spec_study.py) ───────────────────────
FZ_THRESH    = -0.5
ATR_THRESH   = 0.80
OI_Z_THRESH  = 1.0
EXIT_Z       = 0.0
FZ_WIN       = 90
OI_Z_WIN     = 90
STOP_DAYS    = 10
MIN_STOP     = 0.05


def compute_signals(spot_df: pd.DataFrame,
                    oi_df:   pd.DataFrame,
                    perp_df: pd.DataFrame) -> dict:
    """
    Compute current signal state for one coin.
    Returns a dict with all signal values and entry/exit flags.
    spot_df:  daily spot bars with funding_z and atr_ratio already built
              (output of build_coin_df from zscore_entry_study)
    oi_df:    daily perp OI (time, oi)
    perp_df:  daily perp klines (time, perp_close)
    """
    if spot_df.empty or oi_df.empty or perp_df.empty:
        return _empty()

    # ── Merge to daily ───────────────────────────────────────────────────────
    s = spot_df.copy()
    s["date"] = pd.to_datetime(s["time"]).dt.normalize().dt.tz_localize(None)

    oi = oi_df.copy()
    oi["date"] = pd.to_datetime(oi["time"]).dt.normalize().dt.tz_localize(None)
    oi = oi.groupby("date")["oi"].last().reset_index()

    perp = perp_df.copy()
    perp["date"] = pd.to_datetime(perp["time"]).dt.normalize().dt.tz_localize(None)
    perp = perp.groupby("date")["perp_close"].last().reset_index()

    df = s.merge(oi, on="date", how="inner").merge(perp, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < FZ_WIN + 10:
        return _empty()

    # ── OI z-score ───────────────────────────────────────────────────────────
    roll = df["oi"].rolling(OI_Z_WIN)
    df["oi_z"] = (df["oi"] - roll.mean()) / roll.std()

    # ── Latest bar values ────────────────────────────────────────────────────
    last  = df.iloc[-1]
    prev  = df.iloc[-2] if len(df) >= 2 else last

    fz_now  = float(last["funding_z"])  if not np.isnan(last["funding_z"])  else np.nan
    fz_prev = float(prev["funding_z"])  if not np.isnan(prev["funding_z"])  else np.nan
    oiz     = float(last["oi_z"])       if not np.isnan(last["oi_z"])       else np.nan
    atr     = float(last["atr_ratio"])  if not np.isnan(last["atr_ratio"])  else np.nan
    close   = float(last["close"])

    # ── Stop level (10-day low, min 5%) ──────────────────────────────────────
    lo_window = df["low"].iloc[max(0, len(df)-STOP_DAYS):len(df)]
    stop_10d  = float(lo_window.min()) if len(lo_window) > 0 else close * (1 - MIN_STOP)
    stop_pct  = (close - stop_10d) / close
    if stop_pct < MIN_STOP:
        stop_10d = close * (1 - MIN_STOP)
        stop_pct = MIN_STOP

    # ── Signal flags ────────────────────────────────────────────────────────
    entry_signal = (
        not np.isnan(fz_now) and not np.isnan(fz_prev) and
        not np.isnan(oiz) and not np.isnan(atr) and
        fz_now  <  FZ_THRESH and
        fz_prev >= FZ_THRESH and
        atr     <  ATR_THRESH and
        oiz     >  OI_Z_THRESH
    )
    exit_signal = (
        not np.isnan(fz_now) and
        fz_now > EXIT_Z
    )

    # How "ready" is this coin? Useful for ranking in the scanner.
    # Score 0-3: +1 for each condition met
    conditions_met = sum([
        not np.isnan(oiz)  and oiz  > OI_Z_THRESH,
        not np.isnan(fz_now) and fz_now < 0,
        not np.isnan(atr)  and atr  < ATR_THRESH,
    ])

    return {
        "funding_z":    fz_now,
        "funding_z_prev": fz_prev,
        "oi_z":         oiz,
        "atr_ratio":    atr,
        "close":        close,
        "stop":         stop_10d,
        "stop_pct":     stop_pct * 100,
        "entry_signal": entry_signal,
        "exit_signal":  exit_signal,
        "conditions_met": conditions_met,
        "date":         str(last["date"])[:10],
    }


def _empty() -> dict:
    return {
        "funding_z": np.nan, "funding_z_prev": np.nan,
        "oi_z": np.nan, "atr_ratio": np.nan,
        "close": np.nan, "stop": np.nan, "stop_pct": np.nan,
        "entry_signal": False, "exit_signal": False,
        "conditions_met": 0, "date": None,
    }
