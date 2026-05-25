# algo — Bybit Crypto Trading Bot

Algorithmic trading system for Bybit linear perpetuals (USDT-margined).
Targets mid-to-small cap crypto pairs using an N-bar close breakout strategy.

## Structure

```
algo/
├── config/              # JSON configuration files
│   ├── config.json      # Live trading parameters
│   └── backtest_config.json  # Backtest-only parameters
│
├── live/                # Live trading engine
│   ├── main.py          # Entry point — polling loop
│   ├── exchange.py      # Bybit REST adapter (data + orders)
│   ├── strategy.py      # Signal generation
│   ├── risk.py          # Position sizing + circuit breaker
│   ├── execution.py     # Order lifecycle management
│   └── logger.py        # Logging setup
│
├── backtest/            # Historical backtesting
│   ├── engine.py        # Bar-by-bar simulation loop
│   ├── data_loader.py   # Bybit historical data + local cache
│   ├── strategy.py      # Signal logic (mirrors live/strategy.py)
│   ├── metrics.py       # Performance metrics + equity curve
│   └── report.py        # Output: console, CSV, plots
│
├── dashboard/           # Streamlit monitoring UI
│   └── app.py           # Live + backtest + log viewer
│
├── utils/               # Shared utilities
│   ├── indicators.py    # ATR, N-bar min/max, bar structure
│   └── helpers.py       # round_to_step, load_json, etc.
│
├── data/cache/          # Cached Parquet files (gitignored)
└── logs/                # Runtime log files (gitignored)
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Paper mode (no real orders)
python -m algo.live.main

# Run dashboard
streamlit run algo/dashboard/app.py
```

## Environment variables (live mode)

```
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
```
