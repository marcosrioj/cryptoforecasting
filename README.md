
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
