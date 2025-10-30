#!/usr/bin/env python3
"""
autotrade_bybit.py

Runs `cryptoforecast.FirstOne` for a given symbol every 5 minutes (aligned to multiples of 5)
and manages a single futures position on Bybit:
 - If there is an existing open position for the symbol, close it immediately by market.
 - If FirstOne returns BUY/STRONGBUY -> open LONG (10x)
 - If FirstOne returns SELL/STRONGSELL -> open SHORT (10x)

Trade sizing and auth:
 - BYBIT_API_KEY and BYBIT_API_SECRET must be set in env to perform live trades.
 - Set BYBIT_TESTNET=1 to use the Bybit testnet endpoint instead of mainnet (recommended for testing).
 - Set TRADE_USDT to the notional USDT size you want to allocate per trade (default 10).
 - Use --dry-run to avoid sending any signed requests (local simulation only).

Take profit / stop loss:
 - We compute a weighted Δ% (predicted percent) across TFs using the weights in
   `cryptoforecast.WEIGHTS`. TP is set to entry +/- weighted_delta * 0.9 (i.e. 10% less)
   and stop loss is 25% adverse move from entry.

WARNING: This script places real trades if API keys are provided. Test with
--dry-run and/or BYBIT_TESTNET=1 first.

"""
import os
import asyncio
import time
import hmac, hashlib
import json
from datetime import datetime, timezone
import aiohttp
import argparse
import math
from typing import Optional
import sys
import getpass

import cryptoforecast as cf

# Configuration defaults (can be overridden by env or interactive prompts)
BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY")
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET")
BYBIT_TESTNET = os.environ.get("BYBIT_TESTNET") in ("1", "true", "True")
TRADE_USDT = float(os.environ.get("TRADE_USDT") or 10.0)
LEVERAGE = int(os.environ.get("TRADE_LEVERAGE") or 10)
CATEGORY = os.environ.get("BYBIT_CATEGORY") or "linear"

API_HOST = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"


def _ask(prompt: str, default=None) -> str:
    """Ask the user for a value showing a default. Returns the value or default on empty."""
    default_txt = f" (default: {default})" if default is not None else ""
    try:
        val = input(f"{prompt}{default_txt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        return default
    if val == "":
        return default
    return val


def _ask_bool(prompt: str, default: bool) -> bool:
    dchar = "Y/n" if default else "y/N"
    try:
        val = input(f"{prompt} [{dchar}] (default: {default}): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        return default
    if val == "":
        return default
    if val in ("y", "yes", "1", "true", "t"):
        return True
    if val in ("n", "no", "0", "false", "f"):
        return False
    return default

# Small helper to normalize symbol like in cryptoforecast
def normalize_symbol(s: str) -> str:
    if not s:
        return s
    s = s.strip().upper()
    if len(s) <= 5 and s.isalpha() and not s.endswith("USDT"):
        return s + "USDT"
    return s

# Helpers: Bybit v5 signing (simple HMAC SHA256)
# Bybit v5 signature pattern: signature = HMAC_SHA256(secret, timestamp + method + requestPath + (body or ""))
# We'll provide small wrappers for GET/POST signed requests.

async def bybit_signed_request(session: aiohttp.ClientSession, method: str, path: str, params=None, body=None, api_key=None, api_secret=None):
    api_key = api_key or BYBIT_API_KEY
    api_secret = api_secret or BYBIT_API_SECRET
    if api_key is None or api_secret is None:
        raise RuntimeError("BYBIT_API_KEY and BYBIT_API_SECRET must be set for signed requests")
    ts = str(int(time.time() * 1000))
    q = ""
    if params:
        # build query string in deterministic order
        q = "?" + "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    req_path = path + q
    body_str = json.dumps(body) if body else ""
    to_sign = ts + method.upper() + req_path + body_str
    signature = hmac.new(api_secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json"
    }
    url = API_HOST + req_path
    if method.upper() == "GET":
        async with session.get(url, headers=headers, timeout=30) as resp:
            text = await resp.text()
            return resp.status, text
    elif method.upper() == "POST":
        async with session.post(url, headers=headers, data=body_str or "", timeout=30) as resp:
            text = await resp.text()
            return resp.status, text
    else:
        raise RuntimeError("Unsupported HTTP method")

# Convenience wrappers
async def get_positions(session: aiohttp.ClientSession, symbol: str) -> dict:
    # v5 position list
    path = "/v5/position/list"
    params = {"category": CATEGORY, "symbol": symbol}
    status, txt = await bybit_signed_request(session, "GET", path, params=params)
    try:
        return json.loads(txt)
    except Exception:
        return {"ret_msg": txt}

async def set_leverage(session: aiohttp.ClientSession, symbol: str, buy_leverage: int):
    path = "/v5/position/set-leverage"
    body = {"category": CATEGORY, "symbol": symbol, "leverage": str(buy_leverage)}
    status, txt = await bybit_signed_request(session, "POST", path, body=body)
    try:
        return json.loads(txt)
    except Exception:
        return {"ret_msg": txt}

async def place_market_order(session: aiohttp.ClientSession, symbol: str, side: str, qty: float, reduce_only: bool = False, take_profit: Optional[float] = None, stop_loss: Optional[float] = None):
    # v5 order create
    path = "/v5/order/create"
    body = {
        "category": CATEGORY,
        "symbol": symbol,
        "side": side.upper(),  # BUY or SELL
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "ImmediateOrCancel",
        "reduceOnly": "true" if reduce_only else "false",
        "closeOnTrigger": "false",
    }
    if take_profit is not None:
        body["takeProfit"] = str(take_profit)
    if stop_loss is not None:
        body["stopLoss"] = str(stop_loss)
    status, txt = await bybit_signed_request(session, "POST", path, body=body)
    try:
        return json.loads(txt)
    except Exception:
        return {"ret_msg": txt}


def _divider(char="-"):
    return char * 60


def _format_pos(p: dict) -> str:
    # Defensive extraction of common Bybit v5 position fields
    sym = p.get("symbol") or p.get("instId") or "N/A"
    side = p.get("side") or p.get("positionSide") or "N/A"
    size = p.get("size") or p.get("qty") or 0
    entry = p.get("entryPrice") or p.get("avgEntryPrice") or p.get("price") or "N/A"
    tp = p.get("takeProfit") or p.get("tp") or "N/A"
    sl = p.get("stopLoss") or p.get("sl") or "N/A"
    liq = p.get("liqPrice") or p.get("liq_price") or "N/A"
    upl = p.get("unrealisedPnl") or p.get("unrealizedPnl") or p.get("floatProfit") or "N/A"
    lev = p.get("leverage") or p.get("leverageRate") or "N/A"
    return f"{sym} | side={side} size={size} entry={entry} TP={tp} SL={sl} LIQ={liq} UPL={upl} LEV={lev}"


async def fetch_positions_list(session: aiohttp.ClientSession, symbol: str):
    try:
        resp = await get_positions(session, symbol)
    except Exception as e:
        print(f"[autotrade] Failed to fetch positions: {e}")
        return []
    pos_list = []
    try:
        if isinstance(resp, dict) and resp.get("ret_code") == 0:
            pos_list = resp.get("result", {}).get("list", [])
        elif isinstance(resp, dict) and resp.get("result"):
            pos_list = resp.get("result", {}).get("list", [])
        elif isinstance(resp, dict) and resp.get("data"):
            # other shapes
            pos_list = resp.get("data", [])
    except Exception:
        pos_list = []
    return pos_list


def print_trade_summary(symbol: str, signal: str, side: str, qty: float, entry_price: float, tp_price: float, sl_price: float, executed_resp: dict, pos_list: list, dry_run: bool):
    print(_divider("="))
    print(f"AUTOTRADE SUMMARY for {symbol} — {datetime.now(timezone.utc).isoformat()}")
    print(_divider("="))
    print(f"Signal     : {signal}")
    print(f"Planned side: {side}  Qty: {qty:.6f}")
    print(f"Entry price : {entry_price}")
    print(f"Take Profit : {tp_price}")
    print(f"Stop Loss   : {sl_price}")
    print(f"Mode        : {'DRY-RUN' if dry_run else 'LIVE'}")
    print(_divider())
    print("Order result / response:")
    print(json.dumps(executed_resp or {}, indent=2))
    print(_divider())
    print("Active positions:")
    if not pos_list:
        print("  (no active positions returned)")
    else:
        for p in pos_list:
            print("  " + _format_pos(p))
    print(_divider("="))

# Helper to compute weighted avg dpct from cryptoforecast dpcts_tmp (percent values)
def weighted_dpct(dpcts: dict) -> float:
    total_w = 0
    acc = 0.0
    for tf, pct in dpcts.items():
        w = cf.WEIGHTS.get(tf, 1)
        acc += pct * w
        total_w += w
    return acc / max(1, total_w)

# Core trading logic per symbol
async def handle_symbol(symbol: str, dry_run: bool = True):
    symbol = normalize_symbol(symbol)
    print(f"[autotrade] Checking {symbol} at {datetime.now(timezone.utc).isoformat()} (dry_run={dry_run})")

    # Run shared core and get FirstOne decision
    core = await cf._core_forecast(symbol)
    # FirstOne returns a dict via render_strategy_summary; pass core to reuse
    try:
        firstone_res = await cf.FirstOne(symbol, core=core)
    except Exception as e:
        print(f"[autotrade] Error running FirstOne: {e}")
        return
    # firstone_res should be a dict with 'signal' key
    sig = firstone_res.get("signal") or firstone_res.get("sig")
    if not sig or sig == "FLAT":
        print(f"[autotrade] FirstOne signal is {sig}; no action taken.")
        return

    # compute weighted dpct from core
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    avg_dpct = weighted_dpct(dpcts_tmp)  # in percent
    # take profit delta is 90% of avg_dpct
    tp_pct = avg_dpct * 0.9

    # fetch current mid/last price from results: pick highest-weight available TF
    price = None
    for tf in sorted(cf.WEIGHTS.keys(), key=lambda k: -cf.WEIGHTS[k]):
        r = results.get(tf)
        if r:
            price = r.get("last")
            break
    if price is None:
        print("[autotrade] Could not determine market price from core; aborting")
        return

    side = None
    # Take profit: use 90% of the predicted Δ% (i.e. 10% less than predicted)
    # Stop loss: 2.5% adverse move from entry
    SL_FRAC = 0.025
    if sig in ("BUY", "STRONGBUY"):
        side = "Buy"
        tp_price = price * (1.0 + tp_pct / 100.0)
        # TP uses 90% of avg_dpct (tp_pct already set to avg_dpct*0.9 above)
        # Stop loss is 2.5% below entry
        sl_price = price * (1.0 - SL_FRAC)
    elif sig in ("SELL", "STRONGSELL"):
        side = "Sell"
        tp_price = price * (1.0 - tp_pct / 100.0)
        # Stop loss is 2.5% above entry for shorts
        sl_price = price * (1.0 + SL_FRAC)
    else:
        print(f"[autotrade] Unhandled signal {sig}; skipping")
        return

    # calculate quantity: simple approximation qty = (TRADE_USDT * LEVERAGE) / price
    qty = (TRADE_USDT * LEVERAGE) / price
    # round qty to a reasonable precision (6 decimal places)
    qty = float(f"{qty:.6f}")

    # Ensure we're operating on a USDT perpetual (linear) market
    if not symbol.upper().endswith("USDT") or CATEGORY != "linear":
        print(f"[autotrade] Aborting: autotrade is configured for USDT perpetuals (linear). symbol={symbol} CATEGORY={CATEGORY}")
        return

    # Sanity check: if TP would not be beyond current price (no room for profit), skip order
    if side == "Buy" and tp_price <= price:
        print(f"[autotrade] Skipping order: computed TP ({tp_price:.8f}) <= current price ({price:.8f}) — no upside")
        # show summary/context
        pos_list = []
        if BYBIT_API_KEY and BYBIT_API_SECRET:
            async with aiohttp.ClientSession() as sess:
                pos_list = await fetch_positions_list(sess, symbol)
        print_trade_summary(symbol, sig, side, 0.0, price, tp_price, sl_price, executed_resp=None, pos_list=pos_list, dry_run=dry_run)
        return
    if side == "Sell" and tp_price >= price:
        print(f"[autotrade] Skipping order: computed TP ({tp_price:.8f}) >= current price ({price:.8f}) — no downside")
        pos_list = []
        if BYBIT_API_KEY and BYBIT_API_SECRET:
            async with aiohttp.ClientSession() as sess:
                pos_list = await fetch_positions_list(sess, symbol)
        print_trade_summary(symbol, sig, side, 0.0, price, tp_price, sl_price, executed_resp=None, pos_list=pos_list, dry_run=dry_run)
        return

    # If dry-run, just print the actions
    if dry_run:
        print(f"[autotrade][DRY] Would set leverage={LEVERAGE} for {symbol}")
        print(f"[autotrade][DRY] Would close any existing position for {symbol} by market")
        print(f"[autotrade][DRY] Would open {side} {qty} {symbol} (notional {TRADE_USDT} USDT at x{LEVERAGE})")
        print(f"[autotrade][DRY] TP={tp_price:.8f} SL={sl_price:.8f} (tp_pct={tp_pct:.4f}%)")
        # Interactive override: offer to execute live immediately when running in a TTY
        if sys.stdin.isatty():
            proceed_live = _ask_bool("Dry-run is enabled — execute LIVE order now?", default=False)
            if proceed_live:
                if not (BYBIT_API_KEY and BYBIT_API_SECRET):
                    print("[autotrade] Cannot execute live order: BYBIT_API_KEY/BYBIT_API_SECRET not set")
                else:
                    print("[autotrade] Proceeding to execute LIVE order as requested")
                    dry_run = False
            else:
                # If user declines, just show summary and return
                pos_list = []
                if BYBIT_API_KEY and BYBIT_API_SECRET:
                    async with aiohttp.ClientSession() as sess:
                        pos_list = await fetch_positions_list(sess, symbol)
                print_trade_summary(symbol, sig, side, qty, price, tp_price, sl_price, executed_resp=None, pos_list=pos_list, dry_run=True)
                return
        else:
            # Non-interactive: show summary and return
            pos_list = []
            if BYBIT_API_KEY and BYBIT_API_SECRET:
                async with aiohttp.ClientSession() as sess:
                    pos_list = await fetch_positions_list(sess, symbol)
            print_trade_summary(symbol, sig, side, qty, price, tp_price, sl_price, executed_resp=None, pos_list=pos_list, dry_run=True)
            return

    # Live mode: perform API calls
    async with aiohttp.ClientSession() as sess:
        # 1) Check existing positions
        try:
            pos_resp = await get_positions(sess, symbol)
        except Exception as e:
            print(f"[autotrade] Error fetching positions: {e}")
            return
        # parse response: depending on v5 response structure
        pos_list = []
        try:
            if isinstance(pos_resp, dict) and pos_resp.get("ret_code") == 0:
                pos_list = pos_resp.get("result", {}).get("list", [])
            elif isinstance(pos_resp, dict) and pos_resp.get("result"):
                pos_list = pos_resp.get("result", {}).get("list", [])
        except Exception:
            pos_list = []

        # Close existing non-zero positions by placing opposite market with reduceOnly
        for p in pos_list:
            size = float(p.get("size") or 0)
            pos_side = p.get("side")  # Buy or Sell
            if size == 0:
                continue
            # To close: submit market order opposite side with reduceOnly true and qty=size
            close_side = "Sell" if pos_side == "Buy" else "Buy"
            print(f"[autotrade] Closing existing position: side={pos_side} size={size} (sending market {close_side})")
            close_res = await place_market_order(sess, symbol, close_side, qty=size, reduce_only=True)
            print(f"[autotrade] Close order response: {close_res}")

        # 2) set leverage
        lv = await set_leverage(sess, symbol, LEVERAGE)
        print(f"[autotrade] Set leverage response: {lv}")

        # 3) place market order to open position
        open_res = await place_market_order(sess, symbol, side, qty=qty, reduce_only=False, take_profit=tp_price, stop_loss=sl_price)
        print(f"[autotrade] Open order response: {open_res}")
        # after placing order, fetch current positions and print a UX summary (inside same session)
        pos_list_after = await fetch_positions_list(sess, symbol)
        print_trade_summary(symbol, sig, side, qty, price, tp_price, sl_price, executed_resp=open_res, pos_list=pos_list_after, dry_run=False)

# Scheduler: align to 5-minute boundaries and run forever (or once)
async def scheduler(symbol: str, dry_run: bool = True, once: bool = False):
    symbol = normalize_symbol(symbol)
    # Run immediately once on startup, then align to 5-minute boundaries
    try:
        await handle_symbol(symbol, dry_run=dry)
    except Exception as e:
        print(f"[autotrade] Error during initial handle_symbol: {e}")
    if once:
        return

    while True:
        now = time.time()
        # number of seconds until next multiple of 300
        wait = 300 - (int(now) % 300)
        print(f"[autotrade] Sleeping {wait}s until next 5-min boundary")
        await asyncio.sleep(wait)
        try:
            await handle_symbol(symbol, dry_run=dry)
        except Exception as e:
            print(f"[autotrade] Error during handle_symbol: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Autotrade Bybit futures using cryptoforecast.FirstOne")
    p.add_argument("--symbol", default="", help="Symbol to trade (e.g., BTCUSDT or BTC). Short tokens get USDT appended")
    p.add_argument("--live", action="store_true", default=False, help="Run in live mode (send signed API calls). Default: dry-run")
    p.add_argument("--once", action="store_true", help="Run one cycle then exit")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # If running interactively, prompt for each parameter showing defaults
    if sys.stdin.isatty():
        print("Interactive configuration for autotrade_bybit.py — press Enter to accept defaults")
        # Symbol
        symbol_input = _ask("Symbol to trade (e.g., BTCUSDT)", default=args.symbol or "")
        args.symbol = symbol_input
        # Dry-run (interactive): default is True unless --live flag used
        default_dry = not bool(args.live)
        dry_choice = _ask_bool("Dry-run (no signed API calls)?", default=default_dry)
        args.dry_run = dry_choice
        # Once
        args.once = _ask_bool("Run once and exit?", default=args.once)
        # TRADE_USDT
        try:
            tu = _ask("Trade notional in USDT (TRADE_USDT)", default=str(TRADE_USDT))
            TRADE_USDT = float(tu)
        except Exception:
            pass
        # LEVERAGE
        try:
            lv = _ask("Leverage to use (TRADE_LEVERAGE)", default=str(LEVERAGE))
            LEVERAGE = int(lv)
        except Exception:
            pass
        # Testnet
        BYBIT_TESTNET = _ask_bool("Use Bybit testnet?", default=BYBIT_TESTNET)
        # API key/secret (prompt if not set)
        if not BYBIT_API_KEY:
            k = _ask("BYBIT_API_KEY (leave empty to keep unset)", default="")
            if k:
                BYBIT_API_KEY = k
        # Secret: use getpass to avoid echo
        if not BYBIT_API_SECRET:
            try:
                ssec = getpass.getpass("BYBIT_API_SECRET (leave empty to keep unset): ")
            except Exception:
                ssec = _ask("BYBIT_API_SECRET", default="")
            if ssec:
                BYBIT_API_SECRET = ssec
        # Category
        CATEGORY = _ask("Bybit category (linear/inverse/spot)", default=CATEGORY)

    sym = normalize_symbol(args.symbol)
    dry = bool(getattr(args, "dry_run", True))
    if dry:
        print("WARNING: Running in dry-run mode (no signed API calls will be made). Use --live or set live during the prompt to enable live trading.")
    try:
        # Update API_HOST based on (possibly) updated BYBIT_TESTNET
        API_HOST = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"
        asyncio.run(scheduler(sym, dry_run=dry, once=args.once))
    except KeyboardInterrupt:
        print("Stopped by user")
