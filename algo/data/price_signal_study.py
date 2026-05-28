"""
Price signal study: volatility, momentum, range compression, and relative strength
as predictors of 1H forward returns across volume tiers.

Methodology mirrors volume_tier_study.py — same quintile engine, Spearman IC,
permutation test, and half-split consistency check.

Loads existing parquet caches (no API calls needed).
BTC cache used for relative-strength calculation.

Signals tested:
  atr_pct           ATR(24) as % of price  — absolute volatility level
  atr_compression   ATR(24) / ATR(168)     — coiling: < 1 = compressed vs last week
  bb_width          Bollinger Band width (480-bar / 20d) as % of mid
  range_7d_pct      (7d high − 7d low) / close %  — tight range = sleeping
  roc_24h           24h price return %
  roc_168h          168h price return %
  pct_from_20d_high (close − 20d high) / 20d high %  — 0 = breakout zone
  pct_from_7d_low   (close − 7d low)  / 7d low  %
  rs_btc_24h        coin 24h return − BTC 24h return  — quiet outperformance

Analysis slices per signal per symbol:
  ALL               full dataset
  BULL only         above_200d == True
  BULL + CALM       BULL and funding_zscore < 0  (crowd absent)

Run: python -m algo.data.price_signal_study
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
OUTPUT    = os.path.join(CACHE_DIR, "price_signal_study_output.txt")
HORIZONS  = [4, 24, 168]
MIN_EDGE  = 0.40
N_PERMS   = 1000

SYMBOLS = {
    "ETHUSDT":  "$200M tier",
    "SOLUSDT":  "$50M tier",
    "ADAUSDT":  "$10M tier",
    "UNIUSDT":  "$1M tier",
    "ALGOUSDT": "$300K tier",
}

_out_file = None


def _print(*args, **kwargs):
    print(*args, **kwargs)
    if _out_file:
        print(*args, **kwargs, file=_out_file)


# ── Signal computation ────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    prev  = df["close"].astype(float).shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_price_signals(df: pd.DataFrame, btc_roc_24h: pd.Series = None) -> pd.DataFrame:
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    atr_24  = _atr(df, 24)
    atr_168 = _atr(df, 168)

    df["atr_pct"]         = atr_24 / close * 100
    df["atr_compression"] = atr_24 / atr_168

    ma_480 = close.rolling(480).mean()
    df["bb_width"] = close.rolling(480).std() * 4 / ma_480 * 100

    df["range_7d_pct"] = (high.rolling(168).max() - low.rolling(168).min()) / close * 100

    df["roc_24h"]  = close.pct_change(24)  * 100
    df["roc_168h"] = close.pct_change(168) * 100

    high_20d = high.rolling(480).max()
    low_7d   = low.rolling(168).min()
    df["pct_from_20d_high"] = (close - high_20d) / high_20d * 100
    df["pct_from_7d_low"]   = (close - low_7d)   / low_7d   * 100

    if btc_roc_24h is not None:
        df["rs_btc_24h"] = df["roc_24h"] - btc_roc_24h.reindex(df.index, method="ffill")

    return df


PRICE_SIGNALS = [
    ("atr_pct",          "ATR(24) as % of price"),
    ("atr_compression",  "ATR compression: ATR(24) / ATR(168)"),
    ("bb_width",         "Bollinger Band width (20d) as %"),
    ("range_7d_pct",     "7-day high-low range as % of price"),
    ("roc_24h",          "24h momentum %"),
    ("roc_168h",         "168h momentum %"),
    ("pct_from_20d_high","% below 20-day high  (0 = breakout zone)"),
    ("pct_from_7d_low",  "% above 7-day low"),
    ("rs_btc_24h",       "24h return minus BTC 24h return"),
]


# ── Stats ─────────────────────────────────────────────────────────────────────

def spearman_ic(signal: np.ndarray, fwd: np.ndarray):
    ic, pval = stats.spearmanr(signal, fwd)
    return float(ic), float(pval)


def permutation_test(signal: np.ndarray, fwd: np.ndarray, n: int = N_PERMS) -> float:
    real_ic, _ = spearman_ic(signal, fwd)
    rng = np.random.default_rng(42)
    count = sum(
        abs(spearman_ic(rng.permutation(signal), fwd)[0]) >= abs(real_ic)
        for _ in range(n)
    )
    return count / n


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse_signal(df: pd.DataFrame, signal: str, label: str):
    _print(f"\n  -- {label} --")

    slices = [
        ("ALL",          pd.Series(True, index=df.index)),
        ("BULL only",    df["above_200d"]),
        ("BULL + CALM",  df["above_200d"] & (df["funding_zscore"] < 0)),
    ]

    for slice_label, mask in slices:
        sub = df[mask].dropna(subset=[signal]).copy()
        if len(sub) < 200:
            _print(f"\n    {slice_label}  (skipped — <200 bars)")
            continue

        sub["q"] = pd.qcut(
            sub[signal].rank(method="first"), 5,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
        )

        _print(f"\n    {slice_label}  ({len(sub)} bars)")
        header = f"    {'':5}" + "".join(f"  {h:>4}h_ret    WR" for h in HORIZONS) + "      N"
        _print(header)

        q_rets = {}
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            grp = sub[sub["q"] == q]
            row = f"    {q:5}"
            q_rets[q] = {}
            for h in HORIZONS:
                col = f"fwd_{h}h"
                ret = grp[col].mean()
                wr  = (grp[col] > 0).mean() * 100
                q_rets[q][h] = ret
                row += f"  {ret:+.2f}%  {wr:.0f}%"
            row += f"  {len(grp):6}"
            _print(row)

        _print(f"\n    Q5-Q1 spread (edge threshold = {MIN_EDGE}%):")
        for h in HORIZONS:
            spread = q_rets["Q5"][h] - q_rets["Q1"][h]
            flag   = "  <<" if abs(spread) >= MIN_EDGE else ""
            _print(f"      {h:>4}h: {spread:+.3f}%{flag}")

        sig_vals = sub[signal].values
        _print(f"\n    Spearman IC:")
        for h in HORIZONS:
            fwd_vals = sub[f"fwd_{h}h"].values
            ic, pval = spearman_ic(sig_vals, fwd_vals)
            p_perm   = permutation_test(sig_vals, fwd_vals)
            sig_flag = "  ** significant" if p_perm < 0.05 else ""
            _print(f"      {h:>4}h: IC={ic:+.4f}  p(perm)={p_perm:.3f}{sig_flag}")

    # Consistency on the BULL+CALM slice (most relevant for entries)
    bull_calm = df[df["above_200d"] & (df["funding_zscore"] < 0)].dropna(subset=[signal]).copy()
    if len(bull_calm) >= 400:
        mid = len(bull_calm) // 2
        _print(f"\n    Consistency (Q5-Q1 spread, 24h fwd, BULL+CALM):")
        for name, half in [("First half ", bull_calm.iloc[:mid]), ("Second half", bull_calm.iloc[mid:])]:
            half = half.copy()
            half["q"] = pd.qcut(half[signal].rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"])
            q5 = half[half["q"] == "Q5"]["fwd_24h"].mean()
            q1 = half[half["q"] == "Q1"]["fwd_24h"].mean()
            yr0, yr1 = half.index[0].year, half.index[-1].year
            _print(f"      {name} ({yr0}-{yr1}): Q5-Q1 = {q5 - q1:+.3f}%")


def run_symbol(symbol: str, tier: str, btc_roc_24h: pd.Series):
    cache = os.path.join(CACHE_DIR, f"{symbol.lower()}_1h_vtier.parquet")
    if not os.path.exists(cache):
        _print(f"  No cache — skipping {symbol}")
        return

    _print(f"\n{'#'*70}")
    _print(f"  {symbol}  ({tier})")
    _print(f"{'#'*70}")

    df = pd.read_parquet(cache).copy()
    df = add_price_signals(df, btc_roc_24h)

    required = [s for s, _ in PRICE_SIGNALS if s != "rs_btc_24h"] + [f"fwd_{h}h" for h in HORIZONS]
    if "rs_btc_24h" in df.columns:
        required.append("rs_btc_24h")
    df = df.dropna(subset=required)

    bull_calm_n = int((df["above_200d"] & (df["funding_zscore"] < 0)).sum())
    _print(f"  Dataset: {len(df)} bars  |  "
           f"{df.index[0].date()} to {df.index[-1].date()}")
    _print(f"  BULL: {df['above_200d'].sum()}  |  BEAR: {(~df['above_200d']).sum()}  |  "
           f"BULL+CALM: {bull_calm_n}")

    signals_to_run = [(s, l) for s, l in PRICE_SIGNALS if s in df.columns]
    for col, label in signals_to_run:
        analyse_signal(df, col, label)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load BTC for relative strength
    btc_cache = os.path.join(CACHE_DIR, "btc_1h_study.parquet")
    btc_roc_24h = None
    if os.path.exists(btc_cache):
        btc_df = pd.read_parquet(btc_cache)
        btc_roc_24h = btc_df["close"].astype(float).pct_change(24) * 100
        btc_roc_24h.name = "btc_roc_24h"

    with open(OUTPUT, "w") as f:
        _out_file = f

        _print("Price Signal Study")
        _print(f"Symbols: {', '.join(SYMBOLS.keys())}")
        _print(f"Signals: {', '.join(s for s, _ in PRICE_SIGNALS)}")
        _print(f"Horizons: {HORIZONS}h  |  Min edge: {MIN_EDGE}%  |  Perms: {N_PERMS}")
        _print(f"Slices: ALL / BULL only / BULL+CALM (funding_zscore < 0)")

        for symbol, tier in SYMBOLS.items():
            run_symbol(symbol, tier, btc_roc_24h)

        _print("\n\nDone.")

    print(f"\nOutput saved to: {OUTPUT}")
