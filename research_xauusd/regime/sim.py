"""
Vectorized MFE / MAE and win-rate simulation.

For every signal bar i:
  entry = open[i+1]
  LONG  = reversal bet  (price expected to go up from new low)
  SHORT = continuation  (price expected to keep falling)

MFE / MAE in ATR units at each horizon.
Win-rate grid for stop_sizes × rr_values × horizons.
Conservative tie-break: stop beats TP when both hit on the same bar.
"""

import numpy as np

HORIZONS   = {"30m": 6, "1h": 12, "3h": 36}
STOP_SIZES = [0.5, 1.0, 1.5]          # ATR multiples
RR_VALUES  = [1.0, 1.5, 2.0, 3.0]


def _fwd_cum_arrays(h_arr, l_arr, c_arr, max_k):
    """
    Build forward cumulative max-high, min-low, and final-close matrices.

    fwd_cum_max_h[i, k] = max(high[i+1 .. i+k+1])   (0-indexed k, k in 0..max_k-1)
    fwd_cum_min_l[i, k] = min(low [i+1 .. i+k+1])
    fwd_close    [i, k] = close[i+k+1]
    """
    N = len(h_arr)

    fwd_h = np.full((N, max_k), np.nan)
    fwd_l = np.full((N, max_k), np.nan)
    fwd_c = np.full((N, max_k), np.nan)

    for k in range(max_k):
        j = k + 1
        fwd_h[: N - j, k] = h_arr[j:]
        fwd_l[: N - j, k] = l_arr[j:]
        fwd_c[: N - j, k] = c_arr[j:]

    cum_max_h = np.maximum.accumulate(fwd_h, axis=1)
    cum_min_l = np.minimum.accumulate(fwd_l, axis=1)

    return cum_max_h, cum_min_l, fwd_c


def run_simulation(df, signal_mask):
    """
    Returns a list of result dicts — one row per (signal, direction, stop, rr, horizon).
    Caller concatenates into a DataFrame.

    Columns per row:
      bar_idx, entry, pre_atr, direction, stop_mult, rr, horizon,
      outcome (1=TP, -1=stop, 0=timeout),
      result_r, mfe_atr, mae_atr, edge_ratio
    """
    max_k = max(HORIZONS.values())

    h_arr   = df["High"].values.astype(float)
    l_arr   = df["Low"].values.astype(float)
    o_arr   = df["Open"].values.astype(float)
    c_arr   = df["Close"].values.astype(float)
    atr_arr = df["pre_atr"].values.astype(float)

    cum_max_h, cum_min_l, fwd_c = _fwd_cum_arrays(h_arr, l_arr, c_arr, max_k)

    # Signal indices — exclude last max_k+1 bars (not enough forward data)
    sig_idx = np.where(signal_mask.values)[0]
    sig_idx = sig_idx[sig_idx + max_k + 1 < len(df)]

    if len(sig_idx) == 0:
        return []

    entry_arr = o_arr[sig_idx + 1]
    atr_sig   = atr_arr[sig_idx]
    valid     = (atr_sig > 0) & ~np.isnan(entry_arr) & ~np.isnan(atr_sig)

    sig_idx   = sig_idx[valid]
    entry_arr = entry_arr[valid]
    atr_sig   = atr_sig[valid]

    rows = []

    for direction in ("long", "short"):
        for stop_mult in STOP_SIZES:
            stop_dist = stop_mult * atr_sig

            if direction == "long":
                stop_thr = entry_arr - stop_dist
                tp_base  = entry_arr + stop_dist   # base for rr scaling
            else:
                stop_thr = entry_arr + stop_dist
                tp_base  = entry_arr - stop_dist

            for rr in RR_VALUES:
                tp_dist = rr * stop_dist

                if direction == "long":
                    tp_thr     = entry_arr + tp_dist
                    tp_hit_all = cum_max_h[sig_idx, :]  >= tp_thr[:, None]
                    st_hit_all = cum_min_l[sig_idx, :] <= stop_thr[:, None]
                else:
                    tp_thr     = entry_arr - tp_dist
                    tp_hit_all = cum_min_l[sig_idx, :] <= tp_thr[:, None]
                    st_hit_all = cum_max_h[sig_idx, :] >= stop_thr[:, None]

                for hz_name, k in HORIZONS.items():
                    tp_k = tp_hit_all[:, :k]
                    st_k = st_hit_all[:, :k]

                    tp_any = tp_k.any(axis=1)
                    st_any = st_k.any(axis=1)

                    tp_bar = np.where(tp_any, np.argmax(tp_k, axis=1), k)
                    st_bar = np.where(st_any, np.argmax(st_k, axis=1), k)

                    won     = tp_any & (tp_bar < st_bar)
                    lost    = st_any & ~won
                    timeout = ~tp_any & ~st_any

                    last_c  = fwd_c[sig_idx, k - 1]
                    if direction == "long":
                        t_r = (last_c - entry_arr) / stop_dist
                    else:
                        t_r = (entry_arr - last_c) / stop_dist

                    result_r = np.where(won, rr, np.where(lost, -1.0, t_r))
                    outcome  = np.where(won, 1,  np.where(lost, -1,   0))

                    # MFE / MAE at this horizon
                    max_h_k = cum_max_h[sig_idx, k - 1]
                    min_l_k = cum_min_l[sig_idx, k - 1]

                    if direction == "long":
                        mfe = np.clip(max_h_k - entry_arr, 0, None) / atr_sig
                        mae = np.clip(entry_arr - min_l_k, 0, None) / atr_sig
                    else:
                        mfe = np.clip(entry_arr - min_l_k, 0, None) / atr_sig
                        mae = np.clip(max_h_k - entry_arr, 0, None) / atr_sig

                    edge_ratio = np.where(mae > 0, mfe / mae, np.nan)

                    for i in range(len(sig_idx)):
                        rows.append({
                            "bar_idx":   int(sig_idx[i]),
                            "entry":     float(entry_arr[i]),
                            "pre_atr":   float(atr_sig[i]),
                            "direction": direction,
                            "stop_mult": stop_mult,
                            "rr":        rr,
                            "horizon":   hz_name,
                            "outcome":   int(outcome[i]),
                            "result_r":  float(result_r[i]),
                            "mfe_atr":   float(mfe[i]),
                            "mae_atr":   float(mae[i]),
                            "edge_ratio":float(edge_ratio[i]) if not np.isnan(edge_ratio[i]) else None,
                        })

    return rows
