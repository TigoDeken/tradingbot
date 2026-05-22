"""
Compare session_start=7 (current) vs session_start=3 (adds 04:00 UTC candle).
Runs all 5 symbols, shows combined portfolio metrics for both.
"""
import sys
sys.path.insert(0, ".")

import MetaTrader5 as mt5
import pandas as pd
from data_pipeline import fetch_ohlc
from trade_engine import run_trades, MIN_STOP_PIPS
from backtest_engine import compute_metrics, build_equity

SYMBOLS  = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD"]
BARS     = 9999
SPLIT    = 0.70
INITIAL  = 10_000.0

def get_pip_info(sym):
    info = mt5.symbol_info(sym)
    if info:
        pip = round(info.point * 10, 6)
        pv  = round(info.trade_tick_value * (pip / info.trade_tick_size), 4)
        return pip, pv
    return (0.01, 9.0) if "JPY" in sym else (0.0001, 10.0)

def run_all(session_start):
    all_trades = []
    sym_rows   = []
    for sym in SYMBOLS:
        df = fetch_ohlc(sym, bars=BARS)
        pip, pv = get_pip_info(sym)
        trades, _ = run_trades(df, pip=pip, pip_value=pv, close_strength=0.6,
                               session_start=session_start, session_end=20)
        all_trades.extend(trades)
        if trades:
            split_dt = trades[0].entry_date + (trades[-1].entry_date - trades[0].entry_date) * SPLIT
            oos = [t for t in trades if t.entry_date >= split_dt]
            m   = compute_metrics(trades, initial=INITIAL)
            mo  = compute_metrics(oos,    initial=INITIAL)
            sym_rows.append({
                "Symbol":    sym,
                "Trades":    m["total_trades"],
                "WR%":       m["win_rate_pct"],
                "EV(R)":     m["expectancy_net_r"],
                "MaxDD%":    m["max_drawdown_pct"],
                "TPM":       m["trades_per_month"],
                "OOS_T":     mo["total_trades"],
                "OOS_WR%":   mo["win_rate_pct"],
                "OOS_EV":    mo["expectancy_net_r"],
            })
    # Combined portfolio
    all_trades.sort(key=lambda t: t.entry_date)
    if all_trades:
        split_dt = all_trades[0].entry_date + (all_trades[-1].entry_date - all_trades[0].entry_date) * SPLIT
        oos = [t for t in all_trades if t.entry_date >= split_dt]
        m   = compute_metrics(all_trades, initial=INITIAL)
        mo  = compute_metrics(oos,        initial=INITIAL)
        eq  = build_equity(all_trades, initial=INITIAL)
        combined = {
            "Trades":  m["total_trades"],  "WR%":    m["win_rate_pct"],
            "EV(R)":   m["expectancy_net_r"], "MaxDD%": m["max_drawdown_pct"],
            "TPM":     m["trades_per_month"],
            "OOS_T":   mo["total_trades"], "OOS_WR%": mo["win_rate_pct"],
            "OOS_EV":  mo["expectancy_net_r"],
            "EndBal":  round(eq["balance"].iloc[-1], 2),
        }
    else:
        combined = {}
    return pd.DataFrame(sym_rows), combined

mt5.initialize()

print("Running session_start=7 (current)...", flush=True)
df7, c7 = run_all(session_start=7)
print("Running session_start=3 (adds 04:00 candle)...", flush=True)
df3, c3 = run_all(session_start=3)

mt5.shutdown()

HDR = f"  {'Symbol':<8} {'Trades':>6} {'WR%':>6} {'EV(R)':>7} {'MaxDD%':>7} {'TPM':>5}  |  {'OOS_T':>5} {'OOS_WR%':>8} {'OOS_EV':>7}"
SEP = "-" * 80
W   = "=" * 80

for label, df, c in [("SESSION 7-20 (current)", df7, c7), ("SESSION 3-20 (adds 04:00 bar)", df3, c3)]:
    print(f"\n{W}")
    print(f"  {label}")
    print(W)
    print(HDR)
    print(SEP)
    for _, r in df.iterrows():
        print(f"  {r['Symbol']:<8} {r['Trades']:>6} {r['WR%']:>6.1f} {r['EV(R)']:>7.4f} {r['MaxDD%']:>7.2f} {r['TPM']:>5.1f}  |  {r['OOS_T']:>5} {r['OOS_WR%']:>8.1f} {r['OOS_EV']:>7.4f}")
    print(SEP)
    print(f"  {'COMBINED':<8} {c['Trades']:>6} {c['WR%']:>6.1f} {c['EV(R)']:>7.4f} {c['MaxDD%']:>7.2f} {c['TPM']:>5.1f}  |  {c['OOS_T']:>5} {c['OOS_WR%']:>8.1f} {c['OOS_EV']:>7.4f}   EndBal=${c['EndBal']:,.0f}")

print(f"\n{W}")
print(f"  DELTA  (session 3-20 minus 7-20)")
print(W)
print(f"  {'Symbol':<8} {'dTrades':>8} {'dWR%':>6} {'dEV(R)':>8} {'dOOS_T':>7} {'dOOS_EV':>8}")
print(SEP)
for i in range(len(df7)):
    a, b = df3.iloc[i], df7.iloc[i]
    print(f"  {a['Symbol']:<8} {int(a['Trades']-b['Trades']):>+8} {a['WR%']-b['WR%']:>+6.1f} {a['EV(R)']-b['EV(R)']:>+8.4f} {int(a['OOS_T']-b['OOS_T']):>+7} {a['OOS_EV']-b['OOS_EV']:>+8.4f}")
print(SEP)
dt = int(c3['Trades']-c7['Trades']); doot = int(c3['OOS_T']-c7['OOS_T'])
print(f"  {'COMBINED':<8} {dt:>+8} {c3['WR%']-c7['WR%']:>+6.1f} {c3['EV(R)']-c7['EV(R)']:>+8.4f} {doot:>+7} {c3['OOS_EV']-c7['OOS_EV']:>+8.4f}   dEndBal=${c3['EndBal']-c7['EndBal']:+,.0f}")
print(W)
