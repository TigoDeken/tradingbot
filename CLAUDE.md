# tradingbot — Project Context for Claude

## Philosophy
- No assumptions. Every component is hypothesis → test → prove → add.
- 100% accurate simulation: factor in everything that happens live before accepting results.
- Goal: positive EV, good Sharpe, high trade frequency. Realistic expectations.
- No general trading advice. Proof and facts only.

## What this is
Bybit linear perpetuals (USDT-margined) trading bot, built from scratch.
Working directory: `C:\Users\tigod\OneDrive\Documents\tradingbot`

## Current state
- Scaffold in place, API keys configured in `algo/.env`
- Historical data fetch verified clean: `algo/data/fetch_and_check.py`
- Ready to build data pipeline

## Regulatory constraint
- User is Dutch — cannot trade derivatives or perpetuals (ESMA rules)
- Spot only, with up to 10x margin leverage on Bybit
- category="spot" for all API calls, not "linear"
- Can still go long AND short via spot margin (borrow USDT to buy, borrow crypto to sell)

## Data findings
- Bybit spot BTCUSDT: clean data, no gaps, no bad candles
- Daily bars go back to July 2021 (~5 years)
- Intraday depth TBD (still checking)
- Public kline endpoint needs NO auth — always use `HTTP(testnet=False)` for data
- Testnet has fake/garbage prices — never use it for historical data
- BYBIT_TESTNET=false in .env

## Universe
- 447 USDT spot pairs on Bybit
- Target: $500K–$10M daily 24h volume (USD)
  - Floor $500K: below this spreads are too wide and exits are unreliable
  - Ceiling $10M: above this market becomes too efficient, signal edge shrinks
- This range is where signal inefficiency is real and liquidity is still workable
- Configured in config.json watchlist.min_volume_usd / max_volume_usd

## Build order (evolves as we test)
1. Data pipeline — fetch, cache, serve OHLCV  ← next
2. (TBD based on what data shows)

## Signal findings
Tested on 5 volume tiers (ETH $200M, SOL $50M, ADA $10M, UNI $1M, ALGO $300K), 1H bars, 4 years.
Signals tested: OI 24h change, OI 72h change, OI Z-Score (90d), Funding raw, Funding Z-Score, Rel. Volume.
Horizons: 4h, 24h, 168h forward return. Edge threshold: Q5-Q1 spread ≥ 0.4%.

**OI 24h change — retire as a gate**
- Fast and noisy: BTC median quintile run = 2h, mean 3.4h, max ~2 days
- Direction inconsistent across coins (negative SOL/ADA, positive UNI) — no universal edge
- Current dashboard gate (block Q4/Q5) is not statistically justified — drop it

**OI 72h change — better, use as soft indicator only**
- Stronger edges than 24h: SOL bear -1.22%, UNI bull -1.66% at 24h horizon
- Direction mostly negative (high OI growth = bearish) but consistency checks fail post-2024
- Do not use as a hard gate; track as a soft regime indicator

**Funding Z-Score (90d) — context signal, not a gate**
- Most consistent across time and regimes
- ALGO: first half +0.297%, second half +0.488% (positive and strengthening)
- UNI: first half +0.254%, second half +0.717% (consistent and strengthening)
- NOT used as a simple pass/fail filter — used to understand where we are in the cycle:
  - Q1/Q2 (calm) → crowd absent, price cheap → good conditions to look for entries
  - Q3/Q4 (warming) → move already building → ok to hold, late to enter
  - Q4/Q5 (excited) → crowd fully in → think about exits, not entries
- Rising z-score confirms a move is starting; Q5 is the exit warning zone

**Funding raw in BULL regime — large edge for smaller coins**
- UNI BULL 168h forward: +6.57% Q5-Q1 spread
- ALGO BULL 168h forward: +7.84% Q5-Q1 spread
- Effect is significant and consistent across both time halves
- High funding in bull market = strong positive predictor at weekly horizon

**Relative Volume — drop entirely**
- Never clears 0.4% threshold with statistical significance
- No consistent edge at any horizon or regime

**Volume tier insight**
- Signal strength increases as daily volume decreases
- ETH ($200M): nearly no actionable edge from these inputs
- ADA/UNI/ALGO tier: real, consistent edges — better hunting ground

**How signals are used**
- Funding Z-Score: context signal — calm = look for entries, excited = look for exits
- OI 168h: soft veto only — Q5 means trade is crowded, skip it
- 200d MA regime: hard filter — only trade longs in BULL
- Nothing is a simple pass/fail gate except the 200d MA

**Next step**
- Define what "waking up" looks like in price (the entry trigger)
- Build baseline backtest with no signal overlay first
- Layer signals on top only after baseline is proven

## Key decisions
- Single Bybit API key (public endpoints don't need auth for market data)
- Pybit is the Bybit Python SDK
- Nothing about strategy is decided yet
