import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_pipeline import get_data

# Parameters — all configurable, nothing hardcoded
SWING_N = 2
MIN_SWING_SIZE = 20       # pips — minimum candle range to qualify
MIN_SWING_INCREMENT = 10  # pips — minimum move between consecutive same-type swings
PIP = 0.0001              # EURUSD pip value


def _detect_raw(df: pd.DataFrame, n: int) -> tuple[list[int], list[int]]:
    """
    Find candidate swing high/low bar indices using strict comparison on the left
    and non-strict on the right so the first occurrence wins on equal values.
    """
    highs = df["High"].values
    lows = df["Low"].values
    high_idx, low_idx = [], []

    for i in range(n, len(df) - n):
        left_h, right_h = highs[i - n:i], highs[i + 1:i + n + 1]
        if np.all(highs[i] > left_h) and np.all(highs[i] >= right_h):
            high_idx.append(i)

        left_l, right_l = lows[i - n:i], lows[i + 1:i + n + 1]
        if np.all(lows[i] < left_l) and np.all(lows[i] <= right_l):
            low_idx.append(i)

    return high_idx, low_idx


def build_swings(
    df: pd.DataFrame,
    swing_n: int = SWING_N,
    min_swing_size: float = MIN_SWING_SIZE,
    min_swing_increment: float = MIN_SWING_INCREMENT,
    pip: float = PIP,
) -> pd.DataFrame:
    """
    Return a DataFrame of confirmed swing points: bar, date, price, type.

    Algorithm:
    1. Raw detection with first-occurrence tie-breaking.
    2. MIN_SWING_SIZE filter on candle range.
    3. Merge highs and lows into time order.
    4. Enforce alternation — consecutive same-type: keep the more extreme.
    5. MIN_SWING_INCREMENT filter — reject a new swing if it doesn't differ
       from the previous same-type swing by at least this many pips (absolute).
       Rejected swings are silently dropped; alternation re-enforces itself.
    """
    min_size_price = min_swing_size * pip
    min_incr_price = min_swing_increment * pip

    raw_high_idx, raw_low_idx = _detect_raw(df, swing_n)

    candle_ranges = (df["High"] - df["Low"]).values

    candidates: list[dict] = []
    for i in raw_high_idx:
        if candle_ranges[i] >= min_size_price:
            candidates.append({"bar": i, "date": df.index[i], "price": df["High"].iloc[i], "type": "high"})
    for i in raw_low_idx:
        if candle_ranges[i] >= min_size_price:
            candidates.append({"bar": i, "date": df.index[i], "price": df["Low"].iloc[i], "type": "low"})

    candidates.sort(key=lambda x: x["bar"])

    result: list[dict] = []
    for c in candidates:
        if not result:
            result.append(c)
            continue

        last = result[-1]

        if c["type"] == last["type"]:
            # Consecutive same type — keep the more extreme value
            if c["type"] == "high" and c["price"] > last["price"]:
                result[-1] = c
            elif c["type"] == "low" and c["price"] < last["price"]:
                result[-1] = c

        else:
            # New type — check MIN_SWING_INCREMENT against previous same-type swing
            prev_same = next((s for s in reversed(result) if s["type"] == c["type"]), None)
            if prev_same is None or abs(c["price"] - prev_same["price"]) >= min_incr_price:
                result.append(c)
            # else: reject — sequence remains valid without this point

    if not result:
        return pd.DataFrame(columns=["bar", "date", "price", "type"])

    return pd.DataFrame(result).reset_index(drop=True)


def validate_swings(
    swings: pd.DataFrame,
    pip: float = PIP,
    min_swing_increment: float = MIN_SWING_INCREMENT,
) -> None:
    print("\n=== Last 20 Swing Points ===")
    display = swings.tail(20).copy()
    display["price_pips"] = (display["price"] / pip).round(1)
    print(display[["bar", "date", "price", "type"]].to_string(index=False))

    print("\n=== Alternation Check ===")
    types = swings["type"].tolist()
    alternating = all(types[i] != types[i + 1] for i in range(len(types) - 1))
    consecutive_violations = [(i, types[i]) for i in range(len(types) - 1) if types[i] == types[i + 1]]
    print(f"  Strictly alternating: {alternating}")
    if consecutive_violations:
        print(f"  Violations at indices: {consecutive_violations[:5]}")

    print(f"\n=== Increment Check (min {min_swing_increment} pips) ===")
    min_incr_price = min_swing_increment * pip
    all_ok = True
    for t in ("high", "low"):
        prices = swings[swings["type"] == t]["price"].values
        if len(prices) < 2:
            print(f"  {t}s: fewer than 2 points, skipping")
            continue
        diffs = np.abs(np.diff(prices))
        ok = bool(np.all(diffs >= min_incr_price))
        if not ok:
            all_ok = False
        print(
            f"  {t}s: count={len(prices)}, "
            f"min_diff={diffs.min()/pip:.1f}pip, "
            f"max_diff={diffs.max()/pip:.1f}pip, "
            f"all >= {min_swing_increment}pip: {ok}"
        )
    print(f"  All increment checks passed: {all_ok}")


def plot_swings(df: pd.DataFrame, swings: pd.DataFrame, last_n: int = 200) -> None:
    df_plot = df.tail(last_n).copy()
    cutoff = df_plot.index[0]

    sw_plot = swings[swings["date"] >= cutoff].copy()
    highs = sw_plot[sw_plot["type"] == "high"]
    lows = sw_plot[sw_plot["type"] == "low"]

    fig, ax = plt.subplots(figsize=(18, 7))

    # High-low range bars
    for _, row in df_plot.iterrows():
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax.plot([row.name, row.name], [row["Low"], row["High"]], color=color, linewidth=0.6, alpha=0.6)
        ax.plot([row.name, row.name], [row["Open"], row["Close"]], color=color, linewidth=2.5)

    # Swing markers
    ax.scatter(highs["date"], highs["price"], marker="^", color="red", s=90, zorder=6, label="Swing High")
    ax.scatter(lows["date"], lows["price"], marker="v", color="green", s=90, zorder=6, label="Swing Low")

    # Connect swing points
    if not sw_plot.empty:
        sw_sorted = sw_plot.sort_values("date")
        ax.plot(sw_sorted["date"], sw_sorted["price"], color="royalblue", linewidth=1.0,
                linestyle="--", alpha=0.7, zorder=5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45, fontsize=7)
    ax.set_title(
        f"EURUSD 4H — Swing Detection  |  "
        f"SWING_N={SWING_N}  MIN_SWING_SIZE={MIN_SWING_SIZE}pip  MIN_SWING_INCREMENT={MIN_SWING_INCREMENT}pip"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("swing_detection.png", dpi=150)
    print("\nPlot saved → swing_detection.png")
    plt.show()


if __name__ == "__main__":
    print("Fetching EURUSD 4H data...")
    df = get_data()
    print(f"Loaded {len(df)} bars ({df.index[0]} → {df.index[-1]})")

    print("\nDetecting swings...")
    swings = build_swings(df, swing_n=SWING_N, min_swing_size=MIN_SWING_SIZE,
                          min_swing_increment=MIN_SWING_INCREMENT, pip=PIP)
    print(f"Total swing points detected: {len(swings)}")

    validate_swings(swings, pip=PIP, min_swing_increment=MIN_SWING_INCREMENT)
    plot_swings(df, swings, last_n=200)
