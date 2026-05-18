import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_pipeline import get_data
from constants import PIP
from swing_engine import build_swings, SWING_N, MIN_SWING_SIZE, MIN_SWING_INCREMENT

# Parameters — all configurable
MIN_TREND_SIZE   = 50    # pips — minimum total trend size to qualify
TREND_RANGE_RATIO = 0.40  # trend size must be this fraction of last-20-bar range


def classify_trend(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    swing_n: int = SWING_N,
    min_swing_increment: float = MIN_SWING_INCREMENT,
    min_trend_size: float = MIN_TREND_SIZE,
    trend_range_ratio: float = TREND_RANGE_RATIO,
    pip: float = PIP,
) -> pd.DataFrame:
    """
    Adds three columns to df and returns it:
      regime             — UPTREND | DOWNTREND | AMBIGUOUS
      trend_strength_score — mean pip increment across the 3-swing sequence (null when AMBIGUOUS)
      trend_start_date   — when the current regime run first began (NaT when AMBIGUOUS)

    A swing at bar j is considered confirmed only after bar j + swing_n so we
    never look ahead.
    """
    df = df.copy()

    # Precompute rolling 20-bar high-low range (vectorised)
    roll_range = (df["High"].rolling(20).max() - df["Low"].rolling(20).min()).values

    # Convert swings to numpy arrays for fast access
    sw = swings.sort_values("bar").reset_index(drop=True)
    sw_bars   = sw["bar"].values
    sw_types  = sw["type"].values          # 'high' | 'low'
    sw_prices = sw["price"].values

    min_incr  = min_swing_increment * pip
    min_size  = min_trend_size * pip

    regimes   = np.full(len(df), "AMBIGUOUS", dtype=object)
    strengths = np.full(len(df), np.nan)

    for i in range(len(df)):
        # Only use swings confirmed at or before bar i
        n = int(np.searchsorted(sw_bars, i - swing_n, side="right"))
        if n < 2:
            continue

        types  = sw_types[:n]
        prices = sw_prices[:n]

        last_type = types[-1]

        h_mask = types == "high"
        l_mask = types == "low"
        h = prices[h_mask][-3:]
        l = prices[l_mask][-3:]

        if len(h) < 3 or len(l) < 3:
            continue

        rng = roll_range[i]
        if rng <= 0:
            continue

        if last_type == "high":
            # --- Uptrend ---
            if (h[1] - h[0] >= min_incr and h[2] - h[1] >= min_incr and
                    l[1] - l[0] >= min_incr and l[2] - l[1] >= min_incr):
                trend_size = h[2] - l[0]   # most recent high − oldest low
                if trend_size >= min_size and trend_size / rng >= trend_range_ratio:
                    diffs = np.abs(np.concatenate([np.diff(h), np.diff(l)])) / pip
                    strengths[i] = float(diffs.mean())
                    regimes[i] = "UPTREND"

        elif last_type == "low":
            # --- Downtrend ---
            if (h[0] - h[1] >= min_incr and h[1] - h[2] >= min_incr and
                    l[0] - l[1] >= min_incr and l[1] - l[2] >= min_incr):
                trend_size = h[0] - l[2]   # oldest high − most recent low
                if trend_size >= min_size and trend_size / rng >= trend_range_ratio:
                    diffs = np.abs(np.concatenate([np.diff(h), np.diff(l)])) / pip
                    strengths[i] = float(diffs.mean())
                    regimes[i] = "DOWNTREND"

    # Hysteresis: once a trend is confirmed, maintain it through AMBIGUOUS bars.
    # Only flip when the *opposite* trend is explicitly confirmed.
    for i in range(1, len(regimes)):
        if regimes[i] == "AMBIGUOUS" and regimes[i - 1] in ("UPTREND", "DOWNTREND"):
            regimes[i] = regimes[i - 1]

    df["regime"] = regimes
    df["trend_strength_score"] = np.where(regimes == "AMBIGUOUS", np.nan, strengths)

    # trend_start_date: date when the current regime run began
    start_dates = [pd.NaT] * len(df)
    for i in range(len(df)):
        if regimes[i] == "AMBIGUOUS":
            continue
        if i == 0 or regimes[i] != regimes[i - 1]:
            start_dates[i] = df.index[i]
        else:
            start_dates[i] = start_dates[i - 1]
    df["trend_start_date"] = start_dates

    return df


def validate_trend(df: pd.DataFrame) -> None:
    counts = df["regime"].value_counts()
    total  = len(df)
    ambig_pct = counts.get("AMBIGUOUS", 0) / total * 100

    print("\n=== Regime counts ===")
    for regime, cnt in counts.items():
        print(f"  {regime:<12} {cnt:>5}  ({cnt/total*100:.1f}%)")

    if ambig_pct > 85:
        print(f"\n  *** WARNING: {ambig_pct:.1f}% AMBIGUOUS — parameters are too strict ***")

    print("\n=== Last 50 candles ===")
    tail = df[["Close", "regime", "trend_strength_score"]].tail(50)
    print(tail.to_string())


def plot_regimes(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(df.index, df["Close"], color="black", linewidth=0.7, label="Close")

    color_map = {"UPTREND": "green", "DOWNTREND": "red", "AMBIGUOUS": "grey"}
    regimes   = df["regime"].values
    dates     = df.index

    # Find block boundaries efficiently
    changes = np.where(
        np.concatenate([[True], regimes[1:] != regimes[:-1], [True]])
    )[0]

    for j in range(len(changes) - 1):
        s, e = changes[j], changes[j + 1] - 1
        ax.axvspan(dates[s], dates[min(e + 1, len(dates) - 1)],
                   alpha=0.18, color=color_map[regimes[s]], linewidth=0)

    patches = [
        mpatches.Patch(color="green",  alpha=0.5, label="UPTREND"),
        mpatches.Patch(color="red",    alpha=0.5, label="DOWNTREND"),
        mpatches.Patch(color="grey",   alpha=0.5, label="AMBIGUOUS"),
    ]
    ax.legend(handles=patches, loc="upper left")
    ax.set_title(
        f"EURUSD 4H — Trend Classification  |  "
        f"MIN_TREND_SIZE={MIN_TREND_SIZE}pip  TREND_RANGE_RATIO={TREND_RANGE_RATIO}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("trend_classification.png", dpi=150)
    print("\nPlot saved → trend_classification.png")
    plt.show()


if __name__ == "__main__":
    print("Fetching data...")
    df = get_data()
    print(f"Loaded {len(df)} bars")

    print("Detecting swings...")
    swings = build_swings(df, swing_n=SWING_N, min_swing_size=MIN_SWING_SIZE,
                          min_swing_increment=MIN_SWING_INCREMENT, pip=PIP)
    print(f"Swing points: {len(swings)}")

    print("Classifying trend regimes...")
    df = classify_trend(df, swings, swing_n=SWING_N, min_swing_increment=MIN_SWING_INCREMENT,
                        min_trend_size=MIN_TREND_SIZE, trend_range_ratio=TREND_RANGE_RATIO, pip=PIP)

    validate_trend(df)
    plot_regimes(df)
