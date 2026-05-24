"""
Volatility ratio distribution analysis for GOLD# M5.

Computes the ratio = current_vol / session_baseline for each bar
in the three session windows, then shows the empirical distribution
so threshold choices are data-driven, not guesses.

Output:
  results/vol_thresholds.png  — histogram per session + candidate thresholds
  results/vol_thresholds.txt  — percentile table + split comparison

Usage: cd research_xauusd && python run_vol_thresholds.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.loader      import connect, disconnect, fetch_ohlcv
from regime_filter    import compute_vol_ratio, SESSION_WINDOWS, VOL_LOW_THRESH, VOL_HIGH_THRESH

SYMBOL    = "GOLD#"
TIMEFRAME = "M5"
N_BARS    = 150_000
OUT_DIR   = "results"


def _split_stats(ratio, lo, hi):
    n    = len(ratio)
    low  = (ratio < lo).sum()
    mid  = ((ratio >= lo) & (ratio <= hi)).sum()
    high = (ratio > hi).sum()
    return low / n, mid / n, high / n


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Connecting to MT5...")
    if not connect():
        sys.exit(1)

    try:
        print(f"Fetching {SYMBOL} {TIMEFRAME} ({N_BARS:,} bars)...")
        df, _ = fetch_ohlcv(SYMBOL, TIMEFRAME, N_BARS, fallbacks=["XAUUSD"])
        print(f"  Got {len(df):,} bars  {df.index[0].date()} – {df.index[-1].date()}")

        print("Computing vol ratio...")
        ratio_full = compute_vol_ratio(df)

        hour = df.index.hour
        in_any_session = pd.Series(False, index=df.index)
        for start, end in SESSION_WINDOWS.values():
            in_any_session |= (hour >= start) & (hour < end)

        ratio_all = ratio_full[in_any_session].dropna()
        print(f"  Valid session bars: {len(ratio_all):,}")

        # ── Percentile table ─────────────────────────────────────────────────
        pcts   = [5, 10, 20, 25, 33, 40, 50, 60, 67, 75, 80, 90, 95]
        values = {p: np.percentile(ratio_all, p) for p in pcts}

        p33 = values[33]
        p67 = values[67]
        p20 = values[20]
        p80 = values[80]

        lines = []
        lines.append(f"GOLD# M5 Vol Ratio — {df.index[0].date()} to {df.index[-1].date()}")
        lines.append(f"Session bars (post-warmup): {len(ratio_all):,}\n")

        lines.append("Percentile table (all sessions combined):")
        lines.append(f"  {'pct':>5}   {'ratio':>7}")
        for p in pcts:
            lines.append(f"  {p:>4}%   {values[p]:>7.4f}")

        lines.append("")
        lines.append("Threshold comparison:")
        lines.append(f"  {'option':<22}  {'low':>6}  {'mid':>6}  {'high':>6}  {'low_thr':>8}  {'high_thr':>9}")

        candidates = [
            ("current (0.80/1.20)",  VOL_LOW_THRESH, VOL_HIGH_THRESH),
            ("tertiles  (p33/p67)", p33, p67),
            ("outliers  (p20/p80)", p20, p80),
        ]
        for label, lo, hi in candidates:
            lw, md, hg = _split_stats(ratio_all, lo, hi)
            lines.append(f"  {label:<22}  {lw:>5.1%}  {md:>5.1%}  {hg:>5.1%}  {lo:>8.4f}  {hi:>9.4f}")

        # Per-session percentiles
        lines.append("")
        lines.append("Per-session p33 / p67:")
        for name, (start, end) in SESSION_WINDOWS.items():
            mask = (hour >= start) & (hour < end)
            r    = ratio_full[mask].dropna()
            if len(r) < 50:
                continue
            lo_s = np.percentile(r, 33)
            hi_s = np.percentile(r, 67)
            lw, md, hg = _split_stats(r, lo_s, hi_s)
            lines.append(
                f"  {name:<8}  p33={lo_s:.4f}  p67={hi_s:.4f}"
                f"  (n={len(r):,}  low={lw:.0%} mid={md:.0%} high={hg:.0%})"
            )

        txt_path = f"{OUT_DIR}/vol_thresholds.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
        print("\n" + "\n".join(lines))

        # ── Histogram ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"GOLD# M5  Vol Ratio per Session  (n={len(ratio_all):,} bars  "
            f"{df.index[0].date()} – {df.index[-1].date()})",
            fontsize=11,
        )
        fig.patch.set_facecolor("#0e1117")

        for ax, (name, (start, end)) in zip(axes, SESSION_WINDOWS.items()):
            mask = (hour >= start) & (hour < end)
            r    = ratio_full[mask].dropna()

            lo_s = np.percentile(r, 33)
            hi_s = np.percentile(r, 67)

            ax.set_facecolor("#0e1117")
            ax.hist(r, bins=80, color="#42a5f5", edgecolor="none", alpha=0.75)

            ax.axvline(VOL_LOW_THRESH,  color="#ef5350", lw=1.5, linestyle="--",
                       label=f"current low  {VOL_LOW_THRESH:.2f}")
            ax.axvline(VOL_HIGH_THRESH, color="#66bb6a", lw=1.5, linestyle="--",
                       label=f"current high {VOL_HIGH_THRESH:.2f}")
            ax.axvline(lo_s, color="#ff9800", lw=1.5, linestyle=":",
                       label=f"p33  {lo_s:.4f}")
            ax.axvline(hi_s, color="#ce93d8", lw=1.5, linestyle=":",
                       label=f"p67  {hi_s:.4f}")

            ax.set_title(f"{name.capitalize()}  {start:02d}–{end:02d}h UTC  "
                         f"n={len(r):,}", color="white", fontsize=10)
            ax.set_xlabel("vol ratio  (current ATR / session baseline)", color="#aaa")
            ax.set_ylabel("bars", color="#aaa")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

        plt.tight_layout()
        img_path = f"{OUT_DIR}/vol_thresholds.png"
        plt.savefig(img_path, dpi=130, facecolor=fig.get_facecolor())
        print(f"\nHistogram -> {img_path}")
        print(f"Table     -> {txt_path}")
        print(f"\nRecommended thresholds:  low < {p33:.4f}  |  high > {p67:.4f}  (p33/p67)")

    finally:
        disconnect()


if __name__ == "__main__":
    main()
