"""
Funding z-score direct entry study — iterated.

Core hypothesis:
  When the 90-day funding z-score drops below a threshold (crowd absent),
  enter long at next bar open. Hold for N days or until stop is hit.
  No price trigger. Z-score IS the entry. Price only defines risk (stop).

Iterations tested in sequence:
  1. Raw signal: hold 7/14/21d, z < -0.25, no filters
  2. Threshold sensitivity: z < -0.25 / -0.5 / -1.0
  3. Stop sensitivity: 5-day low / 10-day low / 1.5x ATR
  4. ATR compression filter added
  5. Regime filter (200d MA) added
  6. Best combo summary

Run: python -m algo.data.zscore_entry_study
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from pybit.unified_trading import HTTP

SESSION   = HTTP(testnet=False)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "swing_study")

FZ_WINDOW    = 90
MIN_RISK_PCT = 0.003   # skip if stop < 0.3% away (flat coin, no structure)
MAX_RISK_PCT = 0.30    # skip if stop > 30% away (too wide)


# ── Data loading ──────────────────────────────────────────────────────────────

def fetch_daily(symbol: str, n_bars: int = 730) -> pd.DataFrame:
    bars, end = [], None
    while len(bars) < n_bars:
        params = dict(category="spot", symbol=symbol, interval="D", limit=200)
        if end:
            params["end"] = end
        raw = SESSION.get_kline(**params)["result"]["list"]
        if not raw:
            break
        bars.extend(raw)
        end = int(raw[-1][0]) - 1
        time.sleep(0.1)
        if len(raw) < 200:
            break
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars, columns=["ts","open","high","low","close","volume","turnover"])
    df = df.astype({c: float for c in ["open","high","low","close","volume","turnover"]})
    df["ts"]   = df["ts"].astype(int)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True).iloc[-n_bars:].reset_index(drop=True)


def load_daily(symbol: str) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"{symbol}_daily.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache).reset_index(drop=True)
    df = fetch_daily(symbol)
    if not df.empty:
        df.to_parquet(cache)
    return df


def load_funding(symbol: str) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"{symbol}_funding.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    return pd.DataFrame()


def get_funded_symbols() -> list[str]:
    out = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith("_funding.parquet"):
            out.append(f.replace("_funding.parquet",""))
    return sorted(out)


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_coin_df(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df["date"] = df["time"].dt.normalize()

    # 200-day MA regime
    df["ma200"] = df["close"].rolling(200).mean()
    df["bull"]  = df["close"] > df["ma200"]

    # ATR compression (5/10)
    df["atr5"]  = compute_atr(df, 5)
    df["atr10"] = compute_atr(df, 10)
    df["atr_ratio"] = df["atr5"] / df["atr10"]

    # Funding z-score
    df["funding_z"] = np.nan
    if not funding.empty:
        funding = funding.copy()
        funding["date"] = funding["time"].dt.normalize()
        daily_f = funding.groupby("date")["funding_rate"].sum()
        roll    = daily_f.rolling(FZ_WINDOW)
        daily_z = (daily_f - roll.mean()) / roll.std()
        df["funding_z"] = df["date"].map(daily_z)

    return df


# ── Signal & simulation ───────────────────────────────────────────────────────

def stop_price(df: pd.DataFrame, idx: int, stop_type: str) -> float:
    """Compute stop price at bar idx (uses only data before idx)."""
    if stop_type == "low5":
        window = df["low"].iloc[max(0, idx-5):idx]
        return window.min() if len(window) > 0 else np.nan
    elif stop_type == "low10":
        window = df["low"].iloc[max(0, idx-10):idx]
        return window.min() if len(window) > 0 else np.nan
    elif stop_type == "atr":
        atr = df["atr10"].iloc[idx]
        entry = df["open"].iloc[idx + 1] if idx + 1 < len(df) else np.nan
        return entry - 1.5 * atr if not np.isnan(atr) else np.nan
    return np.nan


def simulate(df: pd.DataFrame,
             z_thresh: float,
             hold_days: int,
             stop_type: str,
             require_bull: bool = False,
             require_compressed: bool = False) -> pd.DataFrame:
    """
    Scan all bars. Enter long on z-score signal. Exit after hold_days or stop.
    Returns DataFrame of trades.
    """
    fz     = df["funding_z"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    bull   = df["bull"].values
    atr_r  = df["atr_ratio"].values
    n      = len(df)

    trades    = []
    in_trade  = False
    trade_end = 0

    for i in range(FZ_WINDOW + 10, n - hold_days - 1):
        # Only enter new trade if not in one on this coin
        if in_trade and i < trade_end:
            continue
        in_trade = False

        z_prev = fz[i - 1] if i > 0 else np.nan
        z_curr = fz[i]

        # Long signal: z crosses below threshold (new entry only)
        if np.isnan(z_curr) or np.isnan(z_prev):
            continue
        if not (z_curr < z_thresh and z_prev >= z_thresh):
            continue

        # Optional filters
        if require_bull and not bool(bull[i]):
            continue
        if require_compressed and not (not np.isnan(atr_r[i]) and atr_r[i] < 0.80):
            continue

        if i + 1 >= n:
            continue
        entry = opens[i + 1]
        stop  = stop_price(df, i, stop_type)

        if np.isnan(stop) or np.isnan(entry) or entry <= 0:
            continue
        risk = entry - stop
        if risk <= 0:
            continue
        risk_pct = risk / entry
        if risk_pct < MIN_RISK_PCT or risk_pct > MAX_RISK_PCT:
            continue

        # Simulate hold
        exit_price  = None
        exit_reason = "time"
        hold_actual = hold_days

        for j in range(i + 1, min(i + 1 + hold_days, n)):
            if lows[j] <= stop:
                exit_price  = stop
                exit_reason = "stop"
                hold_actual = j - (i + 1)
                break
        if exit_price is None:
            last_j     = min(i + hold_days, n - 1)
            exit_price = closes[last_j]
            hold_actual = last_j - (i + 1)

        ret_pct = (exit_price - entry) / entry * 100
        ret_r   = (exit_price - entry) / risk

        trades.append({
            "entry_bar":  i + 1,
            "entry":      entry,
            "stop":       stop,
            "exit":       exit_price,
            "risk_pct":   risk_pct * 100,
            "ret_pct":    ret_pct,
            "ret_r":      ret_r,
            "exit_reason": exit_reason,
            "hold":       hold_actual,
            "z_at_entry": z_curr,
            "bull":       bool(bull[i]),
        })

        in_trade  = True
        trade_end = i + 1 + hold_actual + 1

    return pd.DataFrame(trades)


# ── Reporting ─────────────────────────────────────────────────────────────────

def summarise(trades: pd.DataFrame, label: str, min_trades: int = 10) -> dict:
    if trades.empty or len(trades) < min_trades:
        print(f"    {label:55} n={len(trades):4d}  (insufficient data)")
        return {}
    wr      = (trades["ret_pct"] > 0).mean()
    avg_pct = trades["ret_pct"].mean()
    avg_r   = trades["ret_r"].mean()
    sharpe  = trades["ret_pct"].mean() / trades["ret_pct"].std() if trades["ret_pct"].std() > 0 else 0
    stops   = (trades["exit_reason"] == "stop").sum()
    times   = (trades["exit_reason"] == "time").sum()
    edge    = "EDGE" if avg_pct > 0.5 else ("slight" if avg_pct > 0 else "NO EDGE")
    print(f"    {label:55} n={len(trades):4d}  wr={wr*100:.0f}%  "
          f"avg={avg_pct:+.2f}%  R={avg_r:+.2f}  sharpe={sharpe:+.2f}  [{edge}]"
          f"  (stop={stops} time={times})")
    return {"n": len(trades), "wr": wr, "avg_pct": avg_pct, "avg_r": avg_r,
            "sharpe": sharpe, "label": label}


def section(title: str):
    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")


# ── Load all data ─────────────────────────────────────────────────────────────

def load_all() -> dict[str, pd.DataFrame]:
    symbols = get_funded_symbols()
    print(f"Loading daily OHLCV for {len(symbols)} coins (fetching missing)...")
    out = {}
    for sym in symbols:
        ohlcv   = load_daily(sym)
        funding = load_funding(sym)
        if ohlcv.empty or len(ohlcv) < FZ_WINDOW + 30:
            continue
        if funding.empty:
            continue
        df = build_coin_df(ohlcv, funding)
        if df["funding_z"].notna().sum() < 30:
            continue
        out[sym] = df
    print(f"Ready: {len(out)} coins\n")
    return out


# ── Run all iterations ────────────────────────────────────────────────────────

def run_combo(coins: dict, z_thresh: float, hold_days: int, stop_type: str,
              require_bull: bool = False, require_compressed: bool = False) -> pd.DataFrame:
    all_trades = []
    for sym, df in coins.items():
        t = simulate(df, z_thresh, hold_days, stop_type, require_bull, require_compressed)
        if not t.empty:
            t["symbol"] = sym
            all_trades.append(t)
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


if __name__ == "__main__":
    coins = load_all()

    # ─────────────────────────────────────────────────────────────────────────
    section("ITERATION 1 — RAW SIGNAL: z < -0.25, vary hold period, stop=5-day low")
    print("    Question: does the z-score entry have any edge at all? Which hold is best?")
    print()
    results_iter1 = {}
    for hold in [7, 14, 21]:
        t = run_combo(coins, z_thresh=-0.25, hold_days=hold, stop_type="low5")
        r = summarise(t, f"z<-0.25  hold={hold}d  stop=5d-low")
        if r:
            results_iter1[hold] = r

    # ─────────────────────────────────────────────────────────────────────────
    section("ITERATION 2 — THRESHOLD SENSITIVITY: best hold from iter1, vary z-threshold")
    best_hold_i1 = max(results_iter1, key=lambda k: results_iter1[k].get("avg_pct", -999)) if results_iter1 else 7
    print(f"    Using hold={best_hold_i1}d (best from iter1). Question: how selective should entry be?")
    print()
    results_iter2 = {}
    for z in [-0.25, -0.5, -1.0]:
        t = run_combo(coins, z_thresh=z, hold_days=best_hold_i1, stop_type="low5")
        r = summarise(t, f"z<{z:.2f}  hold={best_hold_i1}d  stop=5d-low")
        if r:
            results_iter2[z] = r

    # ─────────────────────────────────────────────────────────────────────────
    section("ITERATION 3 — STOP SENSITIVITY: best hold+threshold, vary stop placement")
    best_z_i2 = max(results_iter2, key=lambda k: results_iter2[k].get("avg_r", -999)) if results_iter2 else -0.25
    print(f"    Using z<{best_z_i2}, hold={best_hold_i1}d. Question: where does stop go?")
    print()
    results_iter3 = {}
    for stype in ["low5", "low10", "atr"]:
        t = run_combo(coins, z_thresh=best_z_i2, hold_days=best_hold_i1, stop_type=stype)
        r = summarise(t, f"z<{best_z_i2}  hold={best_hold_i1}d  stop={stype}")
        if r:
            results_iter3[stype] = r

    # ─────────────────────────────────────────────────────────────────────────
    section("ITERATION 4 — ATR COMPRESSION FILTER: proven in swing study")
    best_stop_i3 = max(results_iter3, key=lambda k: results_iter3[k].get("avg_r", -999)) if results_iter3 else "low5"
    print(f"    Using z<{best_z_i2}, hold={best_hold_i1}d, stop={best_stop_i3}.")
    print(f"    Question: does requiring quiet price (ATR compressed) improve results?")
    print()
    results_iter4 = {}
    for compressed in [False, True]:
        label = f"z<{best_z_i2}  hold={best_hold_i1}d  stop={best_stop_i3}  compression={'on' if compressed else 'off'}"
        t = run_combo(coins, z_thresh=best_z_i2, hold_days=best_hold_i1,
                      stop_type=best_stop_i3, require_compressed=compressed)
        r = summarise(t, label)
        if r:
            results_iter4[compressed] = r

    # ─────────────────────────────────────────────────────────────────────────
    section("ITERATION 5 — REGIME FILTER: 200-day MA, long only in bull")
    print(f"    Same best params. Question: does regime improve directional accuracy?")
    print()
    results_iter5 = {}
    best_comp = max(results_iter4, key=lambda k: results_iter4[k].get("avg_r", -999)) if results_iter4 else False
    for bull_only in [False, True]:
        label = f"z<{best_z_i2}  hold={best_hold_i1}d  bull={'required' if bull_only else 'any'}  comp={'on' if best_comp else 'off'}"
        t = run_combo(coins, z_thresh=best_z_i2, hold_days=best_hold_i1,
                      stop_type=best_stop_i3, require_bull=bull_only,
                      require_compressed=best_comp)
        r = summarise(t, label)
        if r:
            results_iter5[bull_only] = r

    # ─────────────────────────────────────────────────────────────────────────
    section("ITERATION 6 — FULL GRID: all combos ranked by avg return %")
    print("    Exhaustive search across all parameter combinations.")
    print()
    grid_results = []
    for z in [-0.25, -0.5, -1.0]:
        for hold in [7, 14, 21]:
            for stype in ["low5", "low10", "atr"]:
                for bull in [False, True]:
                    for comp in [False, True]:
                        t = run_combo(coins, z_thresh=z, hold_days=hold,
                                      stop_type=stype, require_bull=bull,
                                      require_compressed=comp)
                        if t.empty or len(t) < 20:
                            continue
                        grid_results.append({
                            "z_thresh": z, "hold": hold, "stop": stype,
                            "bull": bull, "comp": comp,
                            "n": len(t),
                            "wr": (t["ret_pct"] > 0).mean(),
                            "avg_pct": t["ret_pct"].mean(),
                            "avg_r": t["ret_r"].mean(),
                            "sharpe": t["ret_pct"].mean() / t["ret_pct"].std() if t["ret_pct"].std() > 0 else 0,
                        })

    if grid_results:
        grid = pd.DataFrame(grid_results).sort_values("avg_pct", ascending=False)
        print(f"    {'z':>6} {'hold':>5} {'stop':>6} {'bull':>5} {'comp':>5} "
              f"{'n':>5} {'wr%':>5} {'avg%':>6} {'avgR':>5} {'sharpe':>7}")
        print(f"    {'-'*70}")
        for _, row in grid.head(15).iterrows():
            edge = " <<" if row["avg_pct"] > 0.5 else ""
            print(f"    {row['z_thresh']:>6.2f} {int(row['hold']):>5} {row['stop']:>6} "
                  f"{'Y' if row['bull'] else 'N':>5} {'Y' if row['comp'] else 'N':>5} "
                  f"{int(row['n']):>5} {row['wr']*100:>4.0f}% "
                  f"{row['avg_pct']:>+5.2f}% {row['avg_r']:>+4.2f} "
                  f"{row['sharpe']:>+6.3f}{edge}")

        best = grid.iloc[0]
        section("BEST COMBINATION — DEEP DIVE")
        t_best = run_combo(coins,
                           z_thresh=best["z_thresh"], hold_days=int(best["hold"]),
                           stop_type=best["stop"], require_bull=bool(best["bull"]),
                           require_compressed=bool(best["comp"]))
        print(f"    z<{best['z_thresh']}  hold={int(best['hold'])}d  stop={best['stop']}  "
              f"bull={'required' if best['bull'] else 'any'}  "
              f"compression={'on' if best['comp'] else 'off'}")
        print()
        print(f"    Trades:          {len(t_best)}")
        print(f"    Win rate:        {(t_best['ret_pct']>0).mean()*100:.1f}%")
        print(f"    Avg return:      {t_best['ret_pct'].mean():+.2f}%")
        print(f"    Avg R:           {t_best['ret_r'].mean():+.2f}")
        print(f"    Median return:   {t_best['ret_pct'].median():+.2f}%")
        print(f"    Std dev:         {t_best['ret_pct'].std():.2f}%")
        print(f"    Sharpe:          {t_best['ret_pct'].mean()/t_best['ret_pct'].std():+.3f}")
        print(f"    Best trade:      {t_best['ret_pct'].max():+.1f}%")
        print(f"    Worst trade:     {t_best['ret_pct'].min():+.1f}%")
        print(f"    Stop rate:       {(t_best['exit_reason']=='stop').mean()*100:.1f}%")
        print(f"    Time exit rate:  {(t_best['exit_reason']=='time').mean()*100:.1f}%")
        print(f"    Avg hold:        {t_best['hold'].mean():.1f} days")
        print(f"    Avg risk/trade:  {t_best['risk_pct'].mean():.1f}%")

        # Per-coin breakdown
        print(f"\n    Per-coin results (best combo):")
        per_coin = t_best.groupby("symbol").agg(
            n=("ret_pct","count"),
            avg_pct=("ret_pct","mean"),
            avg_r=("ret_r","mean"),
            wr=("ret_pct", lambda x: (x>0).mean())
        ).sort_values("avg_pct", ascending=False)
        print(f"    {'Symbol':18} {'n':>4} {'wr%':>5} {'avg%':>7} {'avgR':>6}")
        for sym, row in per_coin.iterrows():
            flag = " <<" if row["avg_pct"] > 1.0 else (" >>" if row["avg_pct"] < -1.0 else "")
            print(f"    {sym:18} {int(row['n']):>4} {row['wr']*100:>4.0f}% "
                  f"{row['avg_pct']:>+6.2f}% {row['avg_r']:>+5.2f}{flag}")

        # Consistency check: first half vs second half of data
        t_best_sorted = t_best.sort_values("entry_bar")
        mid = len(t_best_sorted) // 2
        h1  = t_best_sorted.iloc[:mid]
        h2  = t_best_sorted.iloc[mid:]
        print(f"\n    Consistency (is edge stable over time?):")
        print(f"    First half:  n={len(h1)}  avg={h1['ret_pct'].mean():+.2f}%  wr={( h1['ret_pct']>0).mean()*100:.0f}%")
        print(f"    Second half: n={len(h2)}  avg={h2['ret_pct'].mean():+.2f}%  wr={(h2['ret_pct']>0).mean()*100:.0f}%")

        # Fee impact
        avg_risk = t_best["risk_pct"].mean() / 100
        fee_r    = 0.002 / avg_risk if avg_risk > 0 else 0
        print(f"\n    Fee impact (0.2% round trip):")
        print(f"    Avg risk per trade: {avg_risk*100:.1f}%")
        print(f"    Fee cost in R:      {fee_r:.3f}R per trade")
        print(f"    Net avg R:          {best['avg_r'] - fee_r:+.3f}R per trade")

    section("CONCLUSIONS")
    print(f"""
  What this study proves or disproves:
  - If avg_pct > 0 consistently across parameters: z-score entry has real edge
  - If best combo sharpe > 0.1: signal is exploitable
  - If edge only appears with many filters: likely curve-fitted, treat with suspicion
  - If edge appears across most parameter combos: it's robust
  - If nothing works: the z-score is a forward return predictor but not a tradeable entry
    """)

    print("Done.")
