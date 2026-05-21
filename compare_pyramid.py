"""Compare pyramid strategy with min_pyramid_bars = 2, 3, 4."""

from data_pipeline   import get_data
from trade_engine    import run_trades
from backtest_engine import compute_metrics, _split_date, _split_trades

df = get_data()
split_dt = _split_date(df)

BASE = dict(close_strength=0.6, max_open_lots=0.4, pyramid_min_profit_r=0.5)

configs = [2, 3, 4]
results = {}

for mpb in configs:
    trades = run_trades(df, min_pyramid_bars=mpb, **BASE)
    is_t, oos_t = _split_trades(trades, split_dt)
    results[mpb] = {
        "all":  compute_metrics(trades),
        "is":   compute_metrics(is_t),
        "oos":  compute_metrics(oos_t),
        "n":    len(trades),
        "n_is": len(is_t),
        "n_oos":len(oos_t),
    }
    print(f"min_pyramid_bars={mpb}: {len(trades)} trades")

header = f"{'Metric':<28} {'mpb=2':>10} {'mpb=3':>10} {'mpb=4':>10}"
sep    = "-" * len(header)

def row(label, key, section="all", fmt="{:.4f}"):
    vals = [results[m][section].get(key, 0) for m in configs]
    return f"{label:<28} " + "  ".join(fmt.format(v).rjust(8) for v in vals)

for label, section in [("FULL", "all"), ("IN-SAMPLE", "is"), ("OUT-OF-SAMPLE", "oos")]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(sep)
    print(header)
    print(sep)
    n_key = "n" if section == "all" else f"n_{section}"
    ns = [results[m][n_key] for m in configs]
    print(f"{'Trades':<28} " + "  ".join(str(n).rjust(8) for n in ns))
    print(row("Win rate %",          "win_rate_pct",      section, "{:.1f}"))
    print(row("Expectancy (net R)",  "expectancy_net_r",  section))
    print(row("Total net R",         "net_r",             section, "{:.1f}"))
    print(row("Max drawdown %",      "max_drawdown_pct",  section, "{:.2f}"))
    print(row("Trades/month",        "trades_per_month",  section, "{:.1f}"))
