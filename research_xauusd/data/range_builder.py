import numpy as np
import pandas as pd


def wilder_atr(high, low, close, period=20):
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def build_ranges(df, N_values, atr_period=20, spread_price=0.15):
    """
    Augment df with N-bar range features.  All features are fully
    no-lookahead: rolling windows use shift(1) so bar i only sees
    bars [i-N, i-1].

    Columns added:
      pre_atr             — ATR(20) as of bar i-1
      spread_rt_atr       — round-trip spread normalised by pre_atr
      rh_{N}, rl_{N}      — range high/low of previous N bars
      rm_{N}              — range midpoint
      rw_{N}              — range width (price units)
      rw_atr_{N}          — range width in ATR units
      bp_{N}              — bar position within range (0=bottom, 1=top)
    """
    h, l, c = df["High"], df["Low"], df["Close"]

    atr_raw = wilder_atr(h, l, c, atr_period)
    pre_atr = atr_raw.shift(1)  # ATR before setup bar — no lookahead

    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out["pre_atr"] = pre_atr
    out["spread_rt_atr"] = (2.0 * spread_price) / pre_atr.replace(0, np.nan)

    for N in N_values:
        # shift(1) so bar i sees only bars [i-N, i-1]
        rh = h.shift(1).rolling(N).max()
        rl = l.shift(1).rolling(N).min()
        rw = (rh - rl).replace(0, np.nan)
        rm = (rh + rl) / 2.0
        rw_atr = rw / pre_atr.replace(0, np.nan)
        bp = (c - rl) / rw  # 0 = at range_low, 1 = at range_high

        out[f"rh_{N}"] = rh
        out[f"rl_{N}"] = rl
        out[f"rm_{N}"] = rm
        out[f"rw_{N}"] = rw
        out[f"rw_atr_{N}"] = rw_atr
        out[f"bp_{N}"] = bp

    # Lookahead assertion: every range column at row i only uses data up to i-1
    # (shift(1) guarantees this; this comment documents the invariant)
    return out
