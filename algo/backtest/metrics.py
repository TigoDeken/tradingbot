"""
backtest/metrics.py
Performance metrics computation.

Responsibilities:
- Compute core metrics from a list of completed Trade records:
    win_rate, expectancy, profit_factor, sharpe_ratio, max_drawdown_pct,
    total_trades, avg_rr, longest_win_streak, longest_loss_streak
- Build an equity curve from trade P&L sequence
- Support walk-forward validation splits (in-sample / out-of-sample)
"""

# TODO: implement compute_metrics() and build_equity_curve()
