"""
Session window test — runs backtest for each symbol × session combo.
Usage: python run_session_test.py
Requires MT5 to be running.
"""
import sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from data_pipeline   import get_data
from trade_engine    import run_trades
from backtest_engine import compute_metrics
import MetaTrader5 as mt5

SYMBOLS  = ["EURUSD", "GBPUSD", "USDCAD", "USDJPY", "AUDUSD"]
BARS     = 9999
SPLIT    = 0.70
BALANCE  = 10_000
SESSIONS = [
    (0,  24, "All hours"),
    (0,   9, "00–09 Asian"),
    (7,  17, "07–17 European"),
    (7,  20, "07–20 Eur+US"),
    (12, 22, "12–22 US"),
    (20, 24, "20–24 US late"),
    (21,  9, "21–09 Sydney/Tokyo"),
]

PARAMS = dict(
    tp_rr        = 1.0,
    risk_pct     = 0.005,
    min_stop_pips= 5,
    max_lot      = 10.0,
    max_open_lots= 0.0,
    fixed_stop_pips = 25.0,
    atr_period   = 14,
    atr_max_pips = 0.0,
    stop_atr_mult= 0.8,
)

def pip_info(sym):
    mt5.initialize()
    info = mt5.symbol_info(sym)
    if info:
        pip      = round(info.point * 10, 6)
        pip_val  = round(info.trade_tick_value * (pip / info.trade_tick_size), 4)
        return pip, pip_val
    return (0.01, 9.0) if "JPY" in sym else (0.0001, 10.0)

rows = []
for sym in SYMBOLS:
    print(f"\n{'='*50}\n{sym}", flush=True)
    try:
        pip, pip_val = pip_info(sym)
        df = get_data(symbol=sym, bars=BARS)
        split_dt = df.index[0] + (df.index[-1] - df.index[0]) * SPLIT
    except Exception as e:
        print(f"  SKIP — {e}", flush=True)
        continue

    for s_start, s_end, label in SESSIONS:
        try:
            trades, _ = run_trades(
                df, account_balance=BALANCE,
                pip=pip, pip_value=pip_val,
                session_start=s_start, session_end=s_end,
                **PARAMS
            )
            oos_t = [t for t in trades if t.entry_date >= split_dt]
            m = compute_metrics(oos_t, initial=BALANCE)
            ev  = round(m.get("expectancy_net_r", 0) or 0, 3)
            wr  = round(m.get("win_rate_pct",     0) or 0, 1)
            tpm = round(m.get("trades_per_month",  0) or 0, 1)
            dd  = round(m.get("max_drawdown_pct",  0) or 0, 1)
            n   = m.get("total_trades", 0)
            print(f"  {label:22s}  OOS n={n:3d}  EV={ev:+.3f}R  WR={wr:.0f}%  T/mo={tpm:.1f}  DD={dd:.1f}%", flush=True)
            rows.append(dict(Symbol=sym, Session=label, N=n, EV_R=ev, WR_pct=wr, T_month=tpm, MaxDD_pct=dd))
        except Exception as e:
            print(f"  {label:22s}  ERROR: {e}", flush=True)

print("\n\n" + "="*80)
print("RESULTS TABLE")
print("="*80)
df_res = pd.DataFrame(rows)
if not df_res.empty:
    df_res = df_res.sort_values(["Symbol", "EV_R"], ascending=[True, False])
    with pd.option_context("display.max_rows", 200, "display.width", 120, "display.float_format", "{:.3f}".format):
        print(df_res.to_string(index=False))

    print("\n\nBEST SESSION PER SYMBOL")
    print("-"*60)
    for sym in SYMBOLS:
        sub = df_res[df_res.Symbol == sym]
        if sub.empty:
            continue
        best = sub.loc[sub.EV_R.idxmax()]
        print(f"  {sym:8s}  {best.Session:22s}  EV={best.EV_R:+.3f}R  WR={best.WR_pct:.0f}%  T/mo={best.T_month:.1f}")
