
CryptoForecast (Bybit + ML/TA)

Multi-timeframe crypto forecaster with console UX. The project fetches Bybit kline data, computes
technical indicators, and runs an ensemble of models alongside heuristic strategies.

This README is a consolidated, up-to-date guide covering the recent refactor, the five-level
signal system, the strategy registry, and the SMTP-only watchlist notifier.

## High-level summary

- Core refactor: original forecasting logic extracted into `_core_forecast()` and `FirstOne()`.
- Strategies: a `STRATEGIES` registry groups Day Trading, Swing Trading and Long-Term strategies;
  the script computes shared multi-timeframe data once per run and executes all registered
  strategies by default.
- Rendering: prints the overall and timeframe blocks (1w, 1d, 4h, 1h, 5m) once, then appends a
  standardized per-strategy block (Strategy, Decision, Reason, Applies to).
- Progress: when run in a terminal the script shows a 0→100% TTY-only progress bar while it
  fetches data and computes models; library warnings (including LightGBM) are suppressed
  during this phase so the console remains clean until the final SUMMARY is printed.
- Interactive mode: when run with no CLI args from a TTY, the script prompts for runtime options.
- Signals: unified five-level signal system (STRONGBUY, BUY, FLAT, SELL, STRONGSELL).

## Five-level signal scale

- STRONGBUY — strongest bullish signal
- BUY       — bullish
- FLAT      — neutral / no-action
- SELL      — bearish
- STRONGSELL— strongest bearish signal

Signals derive from model deltas and indicator heuristics using conservative cutoffs. The
implementation uses a base threshold in basis points (`BUY_BPS` / `SELL_BPS`, default 15 bps)
and a larger `STRONG` multiplier (default x5) so STRONGBUY/STRONGSELL are harder to reach.

Timeframe signals map to multipliers (STRONGBUY=+2, BUY=+1, FLAT=0, SELL=-1, STRONGSELL=-2).
Aggregation (how the overall decision is computed):

- `vote = sum(weight[tf] * multiplier(sig_tf) for tf in available_timeframes)`
- `total_weight = sum(weights for available_timeframes)`
- STRONG overall requires a larger fraction of aligned weight: the code now uses 70% of
  the available weight (instead of 60%) to classify `STRONGBUY`/`STRONGSELL`.

This combination (larger base threshold, higher STRONG multiplier, and a 70% overall
confluence requirement) reduces how often STRONG* signals appear and favors clearer,
high-confidence alerts.

## Files

- `cryptoforecast.py` — main script (core forecast, strategy registry, renderer)
- `watchlist_notify.py` — watchlist helper that runs `_core_forecast` across symbols and
  sends email alerts when a strong overall signal appears (STRONGBUY or STRONGSELL)
- `cryptoforecast.sh` — optional wrapper
- `logs/` — per-run summary logs under `logs/cryptoforecast/<PAIR>/`

## Installation

Create a virtualenv and install required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -U scikit-learn lightgbm ta aiohttp numpy pandas colorama
```

If LightGBM fails to compile on Linux, install system build tools first:

```bash
sudo apt-get update && sudo apt-get install -y build-essential
```

## Quick run examples

Single run (default BTCUSDT):

```bash
python3 cryptoforecast.py
```

Run continuously (aligned to 5m by default):

```bash
python3 cryptoforecast.py --loop
```

Run in compact mode:

```bash
python3 cryptoforecast.py --compact
```

Interactive mode: run without CLI args from a TTY and follow prompts for symbol, loop, interval,
compact mode, colors and icons.

## Strategies (overview)

Strategies are grouped by category and include short "Applies to" hints describing intended
timeframes (Day Trading, Swing Trading, Long-Term). Many strategies are implemented as
heuristics using the shared multi-timeframe data; some remain placeholders where external
data (orderbooks, sentiment APIs, staking info) is required.

## Watchlist notifier (SMTP-only)

`watchlist_notify.py` monitors a list of symbols and emails you when a symbol's overall
decision reaches a strong signal (STRONGBUY or STRONGSELL).

Configuration (required):
- `EMAIL_USER` — SMTP username (for Gmail, your Gmail address)
- `EMAIL_PASS` — SMTP password; for Gmail use an App Password (recommended)
- `EMAIL_TO`   — recipient (defaults to `EMAIL_USER`)
- `WATCHLIST`  — optional comma-separated symbols (defaults to a 10-symbol list)

CLI options of interest:
- `--test-email` — send a test email and exit
- `--once`       — run checks once and exit
- `--symbols`    — override watchlist on the command line
- `--concurrency`— number of symbols to check in parallel (default 5 or `WATCHER_CONCURRENCY` env)

Usage examples

Test email (verify SMTP credentials):

```bash
export EMAIL_USER="marcosrioj@gmail.com"
export EMAIL_PASS="your_app_password"
python3 watchlist_notify.py --test-email --email-to marcosrioj@gmail.com
```

Run once (check all symbols immediately):

```bash
EMAIL_USER=you@gmail.com EMAIL_PASS=your_app_password python3 watchlist_notify.py --once
```

Run continuously (hourly checks) with concurrency 8:

```bash
export EMAIL_USER="marcosrioj@gmail.com"
export EMAIL_PASS="your_app_password"
python3 watchlist_notify.py --concurrency 8
```

Notes:
- The notifier triggers on both `STRONGBUY` and `STRONGSELL` by default.
- Increase `--concurrency` to check many symbols faster; lower it if you hit API rate limits or CPU
  constraints.
- The script suppresses library warnings and avoids printing full tracebacks; errors are reported
  compactly.

## Security & App Passwords (Gmail)

If you use Gmail SMTP, create an App Password and use it as `EMAIL_PASS`:

1. Enable 2-Step Verification: https://myaccount.google.com/security
2. Create an App Password (Mail) and copy the 16-character value.
3. Export env vars and test the notifier using `--test-email` as shown above.

Do not commit credentials to version control. Use environment variables, a secrets manager, or
process manager secrets.

## Logging

Each run writes a brief summary to `logs/cryptoforecast/<PAIR>/` with a timestamped filename for
later inspection.

## Troubleshooting

- SMTPAuthenticationError (535): usually wrong credentials or missing App Password for Gmail.
- If you need STARTTLS (port 587) instead of SSL (465) we can add a fallback or CLI flags.
- If you want OAuth2 instead of App Passwords, tell me and I will re-enable and document the flow.

## Next steps (optional)

- Add `--list-strategies` to enumerate registered strategies.
- Add STARTTLS/host/port flags or automatic 465/587 fallback.
- Add `.env` support (python-dotenv) and add `.env` to `.gitignore` for local testing.
- Implement concrete strategy logic requested (scalping, ORB breakout, DCA scheduler, sentiment
  integration).

---

If you want any of the optional items implemented, tell me which and I'll add it.

## What changed / high-level summary

- The original forecasting logic was extracted and exposed as `FirstOne()` and a shared
  core implementation `_core_forecast()` that computes multi-timeframe features once per run
  and is reused by all strategies.
- A `STRATEGIES` registry groups named strategies into categories (Day Trading, Swing Trading,
  Long-Term). The script now runs all registered strategies by default in a single run.
- Standardized rendering: the script prints the shared timeframe blocks (1w, 1d, 4h, 1h, 5m)
  once, then appends each strategy's summary block in a consistent format (Strategy, Decision,
  Reason, Applies to).
- Interactive mode: when run without CLI args from a TTY, the script prompts for runtime
  options (symbol, loop, every, category, compact, colors, icons) for faster experimentation.
- Signals: replaced the old BUY/FLAT/SELL mapping with a five-level signal system:
  STRONGBUY, BUY, FLAT, SELL, STRONGSELL. These are used across indicators, timeframe
  summaries, strategies, and aggregated voting.
- Watchlist notifier: `watchlist_notify.py` checks a list of symbols hourly and sends an email
  when any symbol's overall decision becomes `STRONGBUY`. This script now uses simple SMTP
  authentication (EMAIL_USER + EMAIL_PASS) by default.

## Five-level signal scale

The project uses the following signal labels consistently:

- STRONGBUY — strongest bullish signal
- BUY       — bullish
- FLAT      — neutral / no-action
- SELL      — bearish
- STRONGSELL— strongest bearish signal

Color mapping (console):
- STRONGBUY : darker green
- BUY       : bright green
- FLAT      : yellow
- SELL      : bright red
- STRONGSELL: darker red

How signals are derived:
- Model deltas (predicted vs last) use base thresholds (`BUY_BPS` / `SELL_BPS`). Values exceeding
- Use `python3 watchlist_notify.py --test-email` to validate email configuration.
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
The repository registers a set of strategies grouped by category. The script now runs all
registered strategies by default in a single run: it computes the shared multi-timeframe data
(1w, 1d, 4h, 1h, 5m) once, prints the main header and timeframe blocks, and then appends each
strategy's summary block below the timeframes. Some strategies implement heuristics with the

Note: each strategy's printed "Reason" now includes an "Applies to" hint indicating the trading
category and the main timeframe(s) that strategy targets (for quick human context).
available kline+indicator data while a few remain placeholders requiring external data (noted below).

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

Implemented strategy logic & usage examples
-----------------------------------------
The script now computes the shared multi-timeframe data (1w, 1d, 4h, 1h, 5m) once per run and
passes it to each strategy. Strategies then apply their own decision logic using that shared data.

Implemented (heuristic) strategies
 - Scalping (Day Trading): uses the 5m timeframe (EMA short vs mid, StochRSI and AI ensemble as filter).
 - Breakout (Day Trading): breakout detection on 5m with volume confirmation proxy (OBV slope).
 - RangeTrading (Day Trading): Bollinger Bands position on 1h/5m to enter mean-reversion trades.
 - AI_ML (Day Trading): ensemble ENS delta weighted across 1h/5m.
 - MultiIndicator (Day Trading): confluence on 1h/4h of EMA trend + RSI + Stoch.
 - PriceAction (Day Trading): wick/liquidity heuristics on 1h/4h (BB position + MACD).

Implemented Swing strategies (heuristic)
 - TrendFollowing: 1d/4h EMA trend + RSI pullback entries.
 - BreakoutMomentumSwing: daily breakout strength + OBV slope.
 - SupportResistanceSwing: daily BB position bounces.
 - IchimokuSwing: simplified Ichimoku proxy using EMA crossovers.
 - SentimentSwing: volume/OBV proxy for sentiment (note: uses volume as a proxy; real sentiment needs APIs).

Long-term helpers
 - HODL: uses weekly EMA trend to label HOLD and give a simple note.
 - DCA: recommends DCA buys when price is significantly below the long-term EMA.
 - Staking/Diversified/ValueInvesting/Arbitrage: placeholders — require external data (staking APYs, portfolio holdings, fundamentals, or cross-exchange orderbooks).

Usage examples
--------------
Run everything (default behavior — computes core once and runs all strategies):

```bash
python3 cryptoforecast.py
```

Run continuously (aligned to 5m by default):

```bash
python3 cryptoforecast.py --loop
```

Run in compact mode:

```bash
python3 cryptoforecast.py --compact
```

Watchlist notifier (new)
------------------------
There's a helper script `watchlist_notify.py` that checks a list of symbols hourly and sends
an email when any symbol's overall decision becomes `STRONGBUY`.

Setup:
 - Set `EMAIL_USER` and `EMAIL_PASS` in the environment. For Gmail use an App Password (recommended).
 - Optionally set `EMAIL_TO` for recipient (defaults to `EMAIL_USER`).
 - Optionally set `WATCHLIST` as a comma-separated list of symbols to monitor; otherwise the
   default 10-symbol list will be used (BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT, MATICUSDT,
   DOGEUSDT, LTCUSDT, LINKUSDT, XRPUSDT).

OAuth2 (recommended for Gmail)
------------------------------
Instead of using `EMAIL_PASS`, you can authenticate using OAuth2 (preferred). To use OAuth2 set
the following environment variables (obtain these from your Google Cloud project / OAuth consent):

- `OAUTH2_CLIENT_ID`
- `OAUTH2_CLIENT_SECRET`
- `OAUTH2_REFRESH_TOKEN`  (you must obtain a refresh token via the OAuth flow once)
- `OAUTH2_USER`           (email address to use/send from)

Then run the notifier with the `--use-oauth` flag (or the script will auto-detect a refresh token
and prefer OAuth):

```bash
OAUTH2_CLIENT_ID=... OAUTH2_CLIENT_SECRET=... OAUTH2_REFRESH_TOKEN=... OAUTH2_USER=you@gmail.com \
    python3 watchlist_notify.py --test-email --use-oauth
```

Notes:
- Getting a refresh token requires an OAuth2 consent flow (interactive) and `offline` access scope.
- The script exchanges the refresh token for an access token at runtime and uses XOAUTH2 with SMTP.
- OAuth2 avoids storing app passwords and is generally more secure. If you want, I can add a small
  helper to guide obtaining a refresh token (requires creating a Google Cloud OAuth client ID).

Run once (for testing):

```bash
EMAIL_USER=you@gmail.com EMAIL_PASS=your_app_password python3 watchlist_notify.py --once
```

Run as a daemon (hourly checks):

```bash
EMAIL_USER=you@gmail.com EMAIL_PASS=your_app_password python3 watchlist_notify.py
```

Advanced: use `--symbols` to pass a custom comma-separated list, or `WATCHLIST` env var to persist.

Security note: Do not commit `EMAIL_PASS` to version control. Prefer process managers or secret
stores to inject credentials. Gmail app passwords are recommended for SMTP.


Autotrade (Bybit futures)
-------------------------
`autotrade_bybit.py` is a helper that uses `cryptoforecast.FirstOne` to manage a single Bybit
USDT-perpetual position for a given symbol on a 5‑minute cadence (aligned to 00:00, 00:05,
00:10 ...). It's designed for cautious automation and defaults to dry-run so you can inspect
planned trades before enabling live execution.

Key behaviors and safety rules

- Dry-run by default: the script prints planned actions and writes a JSON audit for every
  attempted execution under `logs/autotrade/<SYMBOL>/YYYY-MM-DD_HH-MM-SS.json`. From a TTY
  you can optionally accept an immediate live execution for a single run.
- Price source: the script prefers Bybit's mark price (v5 market tickers) for entry and
  TP/SL calculations. It falls back to v5 ticker or v2 public tickers when mark price is
  unavailable.
- Position management: any existing position for the symbol is closed (market, reduce-only)
  before a new position is opened. This keeps the bot focused on a single directional
  position per-symbol.
- Leverage: the script sets configured leverage (default 10×) before opening a new position.
- Take Profit (TP): computed using the predicted 5m Δ% from `FirstOne`. The TP safety factor
  is set to 1.0 (i.e., the full 5m predicted Δ% is used when computing the TP). This reflects
  the requested conservative behavior of using the full short-term predicted move.
- Stop Loss (SL): a fixed adverse move (default 2.5%) is used to protect capital. You can
  edit `SL_FRAC` in the script to change this value.
- Minimum profit threshold (default 0.25%): the bot enforces a minimum target profit on
  margin before placing a trade. Because profits on futures are magnified by leverage, the
  required price move is adjusted by leverage:

  effective required price move = MIN_PROFIT_FRAC / LEVERAGE

  For example, with MIN_PROFIT_FRAC = 0.25% and LEVERAGE = 10×, the entry price must allow a
  ~0.025% move in price (0.25% / 10) to meet the margin profit target. If the computed TP
  doesn't meet this effective requirement the trade is skipped and an audit entry records
  the reason.

- Auditing: every planned, skipped or executed attempt writes a JSON record into
  `logs/autotrade/<SYMBOL>/` with fields including timestamp, symbol, signal, side, entry
  price, market price used, 5m Δ% used, TP, SL, qty, leverage, effective required price
  fraction, and the API response (for live runs). This is intended for post-mortem review
  and compliance.

Configuration (environment variables)

- `BYBIT_API_KEY` — Bybit API key (required for live trading)
- `BYBIT_API_SECRET` — Bybit API secret (required for live trading)
- `BYBIT_TESTNET` — set to `1` to use Bybit testnet (recommended for testing)
- `TRADE_USDT` — notional USDT amount per trade (default: `10`)
- `TRADE_LEVERAGE` — leverage to use (default: `10`)
- `SL_FRAC` — stop-loss fraction (default `0.025` = 2.5%)
- `MIN_PROFIT_FRAC` — minimum profit on margin required (default `0.0025` = 0.25%)

Quick usage examples

Dry-run single cycle (recommended for testing):

```bash
python3 autotrade_bybit.py --symbol BTC --once
```

Interactive dry-run -> live override (TTY):

```bash
python3 autotrade_bybit.py --symbol BTC
```

When run from a TTY the script prompts for symbol, notional, leverage and whether to
execute the planned trade live. If you accept live execution from the prompt the script
requires `BYBIT_API_KEY` and `BYBIT_API_SECRET` in the environment and will place signed
orders.

Non-interactive live run (testnet recommended):

```bash
BYBIT_TESTNET=1 BYBIT_API_KEY=... BYBIT_API_SECRET=... python3 autotrade_bybit.py --symbol BTC --live --once
```

Notes and recommended workflow

- Always start with dry-run `--once` to inspect planned TP/SL, the 5m Δ% used, and the
  generated audit JSON files.
- Use Bybit testnet (`BYBIT_TESTNET=1`) for integration testing with your API keys before
  enabling live mode.
- Adjust `TRADE_USDT`, `TRADE_LEVERAGE`, `SL_FRAC` and `MIN_PROFIT_FRAC` in the environment
  or directly in the script to match your risk profile.
- The bot intentionally focuses on USDT perpetual (linear) products and will refuse to
  trade non-USDT symbols.

If you'd like, I can add a small CLI help command that prints the TP/SL formulas and a
calculator to preview effective price-move thresholds for chosen leverage and MIN_PROFIT_FRAC.

Testnet dry-run (no real money):

```bash
export BYBIT_TESTNET=1
export BYBIT_API_KEY="<your_test_api_key>"
export BYBIT_API_SECRET="<your_test_api_secret>"
python3 autotrade_bybit.py --symbol BTCUSDT --dry-run --once
```

Live mode (only after extensive testing): remove `--dry-run` to enable real orders. The
script aligns to 5-minute boundaries automatically; to run once and exit use `--once`.

Interactive prompts
-------------------
When you launch `autotrade_bybit.py` from a terminal (TTY), the script will prompt you for
each configurable parameter one-by-one: symbol, dry-run vs live choice, run-once, trade
notional (`TRADE_USDT`), leverage, testnet toggle, API key, API secret (entered securely),
and Bybit category. Press Enter to accept the shown default for each prompt. This is the
recommended workflow for initial configuration and testing.

Important safety notes

- TP/SL semantics vary by Bybit account and market type. The script attempts to attach
  `takeProfit`/`stopLoss` to the order via the v5 API. If your account requires separate
  conditional orders for TP/SL, let me know and I will modify the script to create
  conditional OCO-style orders after opening the market position.
- Quantity calculation is a simple notional approximation: `qty = (TRADE_USDT * LEVERAGE) / price`.
  Adjust sizing logic if you prefer margin-based sizing or percent-of-balance sizing.
- Always test on testnet and with `--dry-run` before enabling live trading. The author is not
  responsible for trading losses.


If you want to run only a subset of strategies or reintroduce CLI selection, it's possible to
add a convenience flag (e.g. `--list-strategies` or `--strategy`) — tell me if you'd like that
added back in and I'll implement it.

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
