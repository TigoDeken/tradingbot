"""
Borrow availability check — which of our 78 coins can be shorted via spot margin?

Bybit spot margin allows shorting by borrowing the asset.
Not all small-cap coins have available borrow inventory.
This determines whether the short-side mirror strategy is viable.

Run: python -m algo.data.borrow_check
"""

import time
import pandas as pd
from pybit.unified_trading import HTTP
from algo.data.zscore_entry_study import load_all
from algo.data.perp_flow_study import load_perp_oi

SESSION = HTTP(testnet=False)


def check_borrow(token: str) -> dict:
    try:
        resp = SESSION.get_spot_margin_data(currency=token)
        data = resp.get("result", {}).get("vipMarginList", [])
        if not data:
            return {"available": False, "rate_daily": None, "max_leverage": None}
        row = data[0]
        borrow_avail = float(row.get("borrowable", 0)) > 0
        rate = float(row.get("hourlyBorrowRate", 0)) * 24 * 100  # convert to daily %
        lev  = float(row.get("maxLeverage", 0))
        return {"available": borrow_avail, "rate_daily": rate, "max_leverage": lev}
    except Exception as e:
        return {"available": False, "rate_daily": None, "max_leverage": None, "error": str(e)}


if __name__ == "__main__":
    print("Checking spot margin borrow availability for all coins...")
    coins = load_all()

    results = []
    for sym in sorted(coins.keys()):
        token = sym.replace("USDT", "")
        info  = check_borrow(token)
        info["symbol"] = sym
        info["token"]  = token
        results.append(info)
        status = "YES" if info["available"] else "no "
        rate_s = f"{info['rate_daily']:.4f}%/day" if info["rate_daily"] is not None else "n/a"
        lev_s  = f"{info['max_leverage']:.1f}x" if info["max_leverage"] else "n/a"
        print(f"  {sym:18} borrow={status}  rate={rate_s:>12}  max_lev={lev_s}")
        time.sleep(0.1)

    df = pd.DataFrame(results)

    available = df[df["available"] == True]
    not_avail = df[df["available"] == False]

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Total coins checked:     {len(df)}")
    print(f"  Shortable (borrow avail): {len(available)} ({len(available)/len(df)*100:.0f}%)")
    print(f"  NOT shortable:            {len(not_avail)} ({len(not_avail)/len(df)*100:.0f}%)")

    if not available.empty:
        print(f"\n  Shortable coins (sorted by borrow rate):")
        avail_sorted = available.sort_values("rate_daily")
        print(f"  {'Symbol':18} {'Rate/day':>10}  {'Max lev':>8}")
        for _, row in avail_sorted.iterrows():
            rate_s = f"{row['rate_daily']:.4f}%" if row['rate_daily'] is not None else "n/a"
            lev_s  = f"{row['max_leverage']:.1f}x" if row['max_leverage'] else "n/a"
            print(f"  {row['symbol']:18} {rate_s:>10}  {lev_s:>8}")

        if available["rate_daily"].notna().any():
            rates = available["rate_daily"].dropna()
            print(f"\n  Borrow rate stats (daily %):")
            print(f"    Min:    {rates.min():.4f}%")
            print(f"    Median: {rates.median():.4f}%")
            print(f"    Max:    {rates.max():.4f}%")
            print(f"    At median, 7-day hold costs: {rates.median()*7:.3f}%")

    if not not_avail.empty:
        print(f"\n  Not shortable: {', '.join(sorted(not_avail['symbol'].tolist()))}")

    # Assessment for short strategy
    print(f"\n{'='*60}")
    print(f"SHORT STRATEGY ASSESSMENT")
    print(f"{'='*60}")
    if len(available) >= 20:
        print(f"\n  {len(available)} coins shortable -> short side is VIABLE")
        print(f"  Enough universe to run mirror strategy (oi_z low + funding high).")
    elif len(available) >= 10:
        print(f"\n  {len(available)} coins shortable -> short side is MARGINAL")
        print(f"  Limited universe. Test with available coins first.")
    else:
        print(f"\n  Only {len(available)} coins shortable -> short side NOT VIABLE")
        print(f"  Borrow inventory too thin for small-cap universe.")

    print("\nDone.")
