"""
Regime features — computed from price only, no lookahead.

At signal bar i (close just printed):
  pre_atr          : ATR(14) through bar i — known at signal time
  atr_ratio        : pre_atr / SMA(ATR, 50)
  linreg_r2        : R² of OLS on last linreg_n closes [i-n+1 .. i]
  linreg_slope_atr : OLS slope (price/bar) normalised by pre_atr
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


LINREG_N    = 20
ATR_PERIOD  = 14
ATR_SMA_N   = 50


def _wilder_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _rolling_linreg(close_arr, n):
    """
    Vectorized OLS on rolling window of n closes.

    Window at bar i = close[i-n+1 .. i]  (current bar included, no lookahead).
    Returns r2_arr, slope_arr aligned to close index.
    """
    N = len(close_arr)
    r2_arr    = np.full(N, np.nan)
    slope_arr = np.full(N, np.nan)

    if N < n:
        return r2_arr, slope_arr

    x     = np.arange(n, dtype=float)
    x_c   = x - x.mean()          # centred x
    x_var = (x_c ** 2).sum()

    windows = sliding_window_view(close_arr, n)   # (N-n+1, n)
    y_mean  = windows.mean(axis=1)
    y_c     = windows - y_mean[:, None]

    slope   = (x_c * y_c).sum(axis=1) / x_var
    y_hat   = slope[:, None] * x_c[None, :] + y_mean[:, None]
    ss_res  = ((windows - y_hat) ** 2).sum(axis=1)
    ss_tot  = (y_c ** 2).sum(axis=1)

    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)
    r2 = np.clip(r2, 0.0, 1.0)

    # windows[k] = close[k : k+n]  →  bar i uses window k = i-n+1
    # r2_arr[i]  = r2[i-n+1]  for i in [n-1, N-1]
    r2_arr[n - 1:]    = r2
    slope_arr[n - 1:] = slope

    return r2_arr, slope_arr


def build_regime_features(df,
                          linreg_n   = LINREG_N,
                          atr_period = ATR_PERIOD,
                          atr_sma_n  = ATR_SMA_N):
    h, l, c = df["High"], df["Low"], df["Close"]

    atr       = _wilder_atr(h, l, c, atr_period)
    atr_sma   = atr.rolling(atr_sma_n).mean()
    atr_ratio = atr / atr_sma.replace(0, np.nan)

    r2_arr, slope_arr = _rolling_linreg(c.values.astype(float), linreg_n)

    slope_atr = np.where(atr.values > 0, slope_arr / atr.values, np.nan)

    out = df.copy()
    out["pre_atr"]          = atr.values
    out["atr_ratio"]        = atr_ratio.values
    out["linreg_r2"]        = r2_arr
    out["linreg_slope_atr"] = slope_atr

    return out
