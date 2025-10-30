README
======

CryptoForecast (Bybit + ML/TA)
------------------------------
Multi-timeframe crypto forecaster with console UX. The script fetches Bybit kline data, computes
technical indicators, and runs an ensemble of models (LightGBM + ElasticNet) to produce short-term
forecasts per timeframe and an overall decision.

Key features
 - Bybit Kline API (default category = linear)
 - Ensemble ML: LightGBM (price), LightGBM (return→price), ElasticNet (return→price)
 - Technical Indicators: RSI, MACD, EMA, Bollinger Bands, ATR, StochRSI, MFI, OBV
 - Colorized console output for numeric values (green=BUY, yellow=FLAT, red=SELL)
 - Uses closed candles only (no repaint)

Output order: [OVERALL], then 1w, 1d, 4h, 1h, 5m.

Installation
------------
Create a virtualenv and install required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -U scikit-learn lightgbm ta aiohttp numpy pandas colorama
```

If LightGBM fails to compile on Linux, install build tools first:

```bash
sudo apt-get update && sudo apt-get install -y build-essential
```

Files
-----
- `cryptoforecast.py`  — main script
- `logs/<scriptname>/<PAIR>/<YYYY-MM-DD_HH-MM-SS>.log` — per-run summary logs

Quick run examples
------------------
Single run (default BTCUSDT, linear category):

```bash
python3 cryptoforecast.py
```

Specify another symbol:

```bash
python3 cryptoforecast.py --symbol ETHUSDT
```

Run continuously (aligned to 5m by default):

```bash
python3 cryptoforecast.py --loop
```

Run with custom interval (seconds):

```bash
python3 cryptoforecast.py --loop --every 60
```

Other flags
 - `--category` : Bybit category (linear, inverse, spot)
 - `--compact`  : compact summary output
 - `--no-color` : disable ANSI colors
 - `--no-icons` : disable emojis/icons
 - `--strategy` : run a named strategy (see Strategies below)
 - `--strategy-category` : run all strategies in a category (Day Trading, Swing Trading, Long-Term)

Shell helper (optional)
-----------------------
`cryptoforecast.sh` can be a small wrapper:

```bash
#!/bin/bash
source .venv/bin/activate
python3 cryptoforecast.py "$@"

chmod +x cryptoforecast.sh
```

Strategies (categories & quick descriptions)
-------------------------------------------
The repository now exposes a strategy registry accessible via `--strategy` (run one) or
`--strategy-category` (run all of a category). Most strategies are initially provided as
lightweight stubs that either reuse the core forecasting logic (named `FirstOne`) or note
that additional data/APIs are required.

Day Trading
 - Scalping — many quick trades on 1–5m TFs; uses tight stops and fast execution (stub).
 - Breakout — intraday opening-range / breakout setups on short TFs (stub).
 - Range Trading — mean-reversion inside short-term ranges; buy at support / sell at resistance (stub).
 - AI/ML Algorithmic — high-frequency ML-driven signals (requires model training/data; stub).
 - Multi-Indicator “High Win Rate” — confluence of multiple indicators across timeframes (stub).
 - Price Action / ICT — liquidity & market-structure based entries (stub).
 - Arbitrage — cross-exchange or spot/futures price differences (placeholder — needs market data).

Swing Trading
 - Trend-Following — trade multi-day trends (4H/1D), enter on pullbacks (stub).
 - Breakout & Momentum Swing — multi-day breakouts from consolidations (stub).
 - Support/Resistance Range Swings — trade bounces on daily/4H ranges (stub).
 - Sentiment & News-Driven Swings — uses social/news data (requires APIs; stub).
 - Ichimoku Cloud Swing — Ichimoku-based entries on 4H/1D (stub).

Long-Term Investing
 - HODL (Buy-and-Hold) — long-term holding; no per-run action (note).
 - DCA — dollar-cost averaging strategy (simulation stub).
 - Staking & Yield — reward/compounding simulation (requires staking data; stub).
 - Diversified Portfolio — manage basket/rebalance simulation (stub).
 - Value Investing — fundamental, long-term picks (stub).

Usage examples (strategies)
--------------------------
Run the original strategy (renamed `FirstOne`, default):

```bash
python3 cryptoforecast.py --strategy FirstOne
```

Run all Day Trading strategies (all stubs):

```bash
python3 cryptoforecast.py --strategy-category "Day Trading"
```

Run a single named strategy in loop mode (example):

```bash
python3 cryptoforecast.py --loop --strategy Scalping --every 60
```

Notes & next steps
------------------
- Most added strategies are currently stubs. They either call `FirstOne` as a base or print a
  short note that external data/APIs (orderbooks, social feeds, cross-exchange prices) are required.
- If you want concrete implementations for any strategy (e.g., true scalping rules on 1–5m TFs,
  ORB breakout rules, a DCA scheduler, or sentiment integration), tell me which one(s) to
  prioritize and I will implement them in the code (or wire the necessary external APIs).
- This project isn't financial advice. Use proper risk management and test in paper mode.

Troubleshooting
---------------
- If you see import errors for sklearn, ensure you installed `scikit-learn` (not `sklearn`).
- If HTTP calls fail, check network, Bybit API limits, and increase `--every` when using `--loop`.
- If LightGBM fails to compile, install system build tools as noted above.

License & disclaimer
--------------------
Provided as-is for research and educational purposes. Not financial advice.
