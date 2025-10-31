#!/usr/bin/env python3
"""
autotrade_bybit.py

Clean autotrader implementing requested features:
 - Use FirstOne 5m Δ% (full value) for TP
 - Use Bybit mark price (v5) as the market price source
 - Enforce minimum profit threshold (0.25%) before placing trades
 - Write JSON audit logs to logs/autotrade/<SYMBOL>/timestamp.json for every attempted execution
 - Keep dry-run default, interactive prompts, testnet support
"""
import os
import sys
import time
import json
import hmac
import hashlib
import argparse
import asyncio
from datetime import datetime, timezone
from typing import Optional, Tuple

import aiohttp

import cryptoforecast as cf

# -------- Config --------
BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY")
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET")
BYBIT_TESTNET = os.environ.get("BYBIT_TESTNET") in ("1", "true", "True")
TRADE_USDT = float(os.environ.get("TRADE_USDT") or 10.0)
LEVERAGE = int(os.environ.get("TRADE_LEVERAGE") or 10)
CATEGORY = "linear"  # only USDT perpetuals
SL_FRAC = 0.025       # 2.5% stop loss
TP_SAFETY = 1.0       # use full 5m dpct (user requested)
MIN_PROFIT_FRAC = 0.0025  # 0.25% minimum profit requirement

API_HOST = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"


def _ask(prompt: str, default: Optional[str] = None) -> str:
    default_txt = f" (default: {default})" if default is not None else ""
    try:
        val = input(f"{prompt}{default_txt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        return default
    return val or default


def _ask_bool(prompt: str, default: bool) -> bool:
    dchar = "Y/n" if default else "y/N"
    try:
        val = input(f"{prompt} [{dchar}] (default: {default}): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        return default
    if val == "":
        return default
    return val in ("y", "yes", "1", "true", "t")


def normalize_symbol(s: str) -> str:
    if not s:
        return s
    s = s.strip().upper()
    if len(s) <= 5 and s.isalpha() and not s.endswith("USDT"):
        return s + "USDT"
    return s


async def fetch_mark_price(session: aiohttp.ClientSession, symbol: str) -> Tuple[Optional[float], str]:
    """Fetch Bybit mark price (v5 preferred). Return (price, source).
    source: 'v5_mark', 'v5_ticker', 'v2', or 'none'."""
    try:
        # Try v5 tickers which may include markPrice or mark_price
        url_v5 = API_HOST + "/v5/market/tickers"
        async with session.get(url_v5, params={"category": CATEGORY, "symbol": symbol}, timeout=8) as resp:
            try:
                data = await resp.json()
            except Exception:
                data = None
        if isinstance(data, dict):
            res = data.get("result")
            if isinstance(res, list) and res:
                r0 = res[0]
                # prefer mark price fields if present
                mark = r0.get("markPrice") or r0.get("mark_price")
                if mark is not None:
                    return float(mark), "v5_mark"
                last = r0.get("lastPrice") or r0.get("last")
                if last is not None:
                    return float(last), "v5_ticker"
            if isinstance(res, dict):
                first = next(iter(res.values())) if res else None
                if isinstance(first, dict):
                    mark = first.get("markPrice") or first.get("mark_price")
                    if mark is not None:
                        return float(mark), "v5_mark"
                    last = first.get("lastPrice") or first.get("last")
                    if last is not None:
                        return float(last), "v5_ticker"

        # fallback to v2 tickers
        url_v2 = API_HOST + "/v2/public/tickers"
        async with session.get(url_v2, params={"symbol": symbol}, timeout=6) as resp:
            try:
                d2 = await resp.json()
            except Exception:
                d2 = None
        if isinstance(d2, dict):
            if d2.get("ret_code") == 0 and d2.get("result"):
                r0 = d2.get("result")[0]
                last = r0.get("last_price") or r0.get("last")
                if last is not None:
                    return float(last), "v2"
            if d2.get("data"):
                r0 = d2.get("data")[0]
                last = r0.get("last_price") or r0.get("last")
                if last is not None:
                    return float(last), "v2"
    except Exception:
        pass
    return None, "none"


def _sign(api_secret: str, ts: str, method: str, path_q: str, body: str) -> str:
    to_sign = ts + method.upper() + path_q + body
    return hmac.new(api_secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()


async def bybit_signed_request(session: aiohttp.ClientSession, method: str, path: str, params=None, body=None, api_key=None, api_secret=None):
    api_key = api_key or BYBIT_API_KEY
    api_secret = api_secret or BYBIT_API_SECRET
    if not api_key or not api_secret:
        raise RuntimeError("BYBIT_API_KEY and BYBIT_API_SECRET required for signed requests")
    ts = str(int(time.time() * 1000))
    q = ""
    if params:
        q = "?" + "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    path_q = path + q
    body_str = json.dumps(body) if body else ""
    sig = _sign(api_secret, ts, method, path_q, body_str)
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-SIGN": sig,
        "Content-Type": "application/json",
    }
    url = API_HOST + path_q
    if method.upper() == "GET":
        async with session.get(url, headers=headers, timeout=30) as resp:
            txt = await resp.text()
            return resp.status, txt
    else:
        async with session.post(url, headers=headers, data=body_str or "", timeout=30) as resp:
            txt = await resp.text()
            return resp.status, txt


async def get_positions(session: aiohttp.ClientSession, symbol: str) -> dict:
    path = "/v5/position/list"
    params = {"category": CATEGORY, "symbol": symbol}
    status, txt = await bybit_signed_request(session, "GET", path, params=params)
    try:
        return json.loads(txt)
    except Exception:
        return {"ret_msg": txt}


async def set_leverage(session: aiohttp.ClientSession, symbol: str, leverage: int) -> dict:
    path = "/v5/position/set-leverage"
    body = {"category": CATEGORY, "symbol": symbol, "leverage": str(leverage)}
    status, txt = await bybit_signed_request(session, "POST", path, body=body)
    try:
        return json.loads(txt)
    except Exception:
        return {"ret_msg": txt}


async def place_market_order(session: aiohttp.ClientSession, symbol: str, side: str, qty: float, reduce_only: bool = False, take_profit: Optional[float] = None, stop_loss: Optional[float] = None) -> dict:
    path = "/v5/order/create"
    body = {
        "category": CATEGORY,
        "symbol": symbol,
        "side": side.upper(),
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


def _divider(ch: str = "-") -> str:
    return ch * 60


def _format_pos(p: dict) -> str:
    sym = p.get("symbol") or p.get("instId") or "N/A"
    side = p.get("side") or p.get("positionSide") or "N/A"
    size = p.get("size") or p.get("qty") or 0
    entry = p.get("entryPrice") or p.get("avgEntryPrice") or p.get("price") or "N/A"
    tp = p.get("takeProfit") or p.get("tp") or "N/A"
    sl = p.get("stopLoss") or p.get("sl") or "N/A"
    liq = p.get("liqPrice") or p.get("liq_price") or "N/A"
    return f"{sym} | side={side} size={size} entry={entry} TP={tp} SL={sl} LIQ={liq}"


async def fetch_positions_list(session: aiohttp.ClientSession, symbol: str):
    try:
        resp = await get_positions(session, symbol)
    except Exception:
        return []
    try:
        if isinstance(resp, dict) and resp.get("ret_code") == 0:
            return resp.get("result", {}).get("list", [])
        if isinstance(resp, dict) and resp.get("result"):
            return resp.get("result", {}).get("list", [])
        if isinstance(resp, dict) and resp.get("data"):
            return resp.get("data", [])
    except Exception:
        return []
    return []


def write_audit_log(record: dict, symbol: str) -> None:
    base = os.path.join("logs", "autotrade", symbol.upper())
    os.makedirs(base, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    fname = os.path.join(base, f"{ts}.json")
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


def print_trade_summary(symbol: str, signal: str, side: str, qty: float, entry_price: float, tp_price: float, sl_price: float, executed_resp: dict, pos_list: list, dry_run: bool, market_price: Optional[float], dpct5m: Optional[float]):
    print(_divider("="))
    print(f"AUTOTRADE SUMMARY for {symbol} — {datetime.now(timezone.utc).isoformat()}")
    print(_divider("="))
    print(f"Signal      : {signal}")
    print(f"Planned side: {side}  Qty: {qty:.6f}")
    print(f"Market price: {market_price if market_price is not None else 'N/A'}")
    print(f"Entry price : {entry_price}")
    print(f"5m Δ% used  : {dpct5m:.4f}%" if dpct5m is not None else "5m Δ% used  : N/A")
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


async def handle_symbol(symbol: str, dry_run: bool = True):
    symbol = normalize_symbol(symbol)

    # fetch live mark price early
    market_price = None
    market_src = "none"
    async with aiohttp.ClientSession() as s:
        market_price, market_src = await fetch_mark_price(s, symbol)

    print(f"[autotrade] {symbol} check @ {datetime.now(timezone.utc).isoformat()} market_price={market_price} src={market_src} dry_run={dry_run}")

    # core + FirstOne
    core = await cf._core_forecast(symbol)
    try:
        first = await cf.FirstOne(symbol, core=core)
    except Exception as e:
        print(f"[autotrade] Error running FirstOne: {e}")
        return
    sig = first.get("signal") or first.get("sig")
    if not sig or sig == "FLAT":
        print(f"[autotrade] FirstOne signal is {sig}; no action.")
        return

    results, dpcts_tmp, duration, now_utc, summary_lines = core
    dpct5m = None
    if isinstance(dpcts_tmp, dict):
        dpct5m = dpcts_tmp.get("5m") or dpcts_tmp.get("5min") or dpcts_tmp.get("05m")
    if dpct5m is None:
        dpct5m = (sum((dpcts_tmp.get(tf, 0.0) * cf.WEIGHTS.get(tf, 1) for tf in dpcts_tmp.keys())) / max(1, sum(cf.WEIGHTS.get(tf, 1) for tf in dpcts_tmp.keys())))

    # decide entry price
    entry_price = market_price
    if entry_price is None:
        for tf in sorted(cf.WEIGHTS.keys(), key=lambda k: -cf.WEIGHTS[k]):
            r = results.get(tf)
            if r:
                entry_price = r.get("last")
                break
    if entry_price is None:
        print("[autotrade] Could not determine entry price; aborting")
        return

    # compute TP/SL using full 5m dpct (TP_SAFETY=1.0 by request)
    if sig in ("BUY", "STRONGBUY"):
        tp_price = entry_price * (1.0 + (dpct5m * TP_SAFETY) / 100.0)
        sl_price = entry_price * (1.0 - SL_FRAC)
        side = "Buy"
    else:
        tp_price = entry_price * (1.0 - (dpct5m * TP_SAFETY) / 100.0)
        sl_price = entry_price * (1.0 + SL_FRAC)
        side = "Sell"

    # enforce minimum profit threshold
    # take leverage into account: required price move (fraction) = MIN_PROFIT_FRAC / LEVERAGE
    effective_frac = MIN_PROFIT_FRAC / max(1, LEVERAGE)
    if side == "Buy":
        min_tp = entry_price * (1.0 + effective_frac)
        if tp_price < min_tp:
            print(f"[autotrade] Skipping: TP {tp_price:.8f} < min required {min_tp:.8f} (min profit {MIN_PROFIT_FRAC*100:.2f}% on margin -> required price move {effective_frac*100:.5f}% considering leverage={LEVERAGE})")
            # record audit
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "signal": sig,
                "side": side,
                "entry_price": entry_price,
                "market_price": market_price,
                "dpct5m": dpct5m,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "qty": 0.0,
                "dry_run": dry_run,
                "reason": "min_profit",
                "min_profit_frac": MIN_PROFIT_FRAC,
                "leverage": LEVERAGE,
                "effective_price_frac": effective_frac,
            }
            write_audit_log(record, symbol)
            return
    else:
        max_tp = entry_price * (1.0 - effective_frac)
        if tp_price > max_tp:
            print(f"[autotrade] Skipping: TP {tp_price:.8f} > max required {max_tp:.8f} for short (min profit {MIN_PROFIT_FRAC*100:.2f}% on margin -> required price move {effective_frac*100:.5f}% considering leverage={LEVERAGE})")
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "signal": sig,
                "side": side,
                "entry_price": entry_price,
                "market_price": market_price,
                "dpct5m": dpct5m,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "qty": 0.0,
                "dry_run": dry_run,
                "reason": "min_profit",
                "min_profit_frac": MIN_PROFIT_FRAC,
                "leverage": LEVERAGE,
                "effective_price_frac": effective_frac,
            }
            write_audit_log(record, symbol)
            return

    # compute qty
    qty = float(f"{(TRADE_USDT * LEVERAGE) / entry_price:.6f}")

    # ensure USDT perpetual
    if not symbol.upper().endswith("USDT") or CATEGORY != "linear":
        print(f"[autotrade] Aborting: only USDT perpetuals supported. symbol={symbol} category={CATEGORY}")
        return

    # dry-run path
    if dry_run:
        print(f"[autotrade][DRY] Plan: set lev={LEVERAGE}, open {side} qty={qty} {symbol} TP={tp_price:.8f} SL={sl_price:.8f}")
        # write audit log for planned trade
        write_audit_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal": sig,
            "side": side,
            "entry_price": entry_price,
            "market_price": market_price,
            "dpct5m": dpct5m,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "qty": qty,
            "dry_run": True,
            "result": None,
        }, symbol)
        # interactive immediate live option
        if sys.stdin.isatty():
            if _ask_bool("Dry-run: execute LIVE now?", default=False):
                if not (BYBIT_API_KEY and BYBIT_API_SECRET):
                    print("[autotrade] BYBIT keys not set; cannot execute live")
                    return
                dry_run = False
            else:
                return
        else:
            return

    # live execution
    async with aiohttp.ClientSession() as sess:
        # fetch and close existing positions
        try:
            pos_resp = await get_positions(sess, symbol)
        except Exception as e:
            print(f"[autotrade] Error fetching positions: {e}")
            return
        pos_list = []
        try:
            if isinstance(pos_resp, dict) and pos_resp.get("ret_code") == 0:
                pos_list = pos_resp.get("result", {}).get("list", [])
            elif isinstance(pos_resp, dict) and pos_resp.get("result"):
                pos_list = pos_resp.get("result", {}).get("list", [])
        except Exception:
            pos_list = []

        for p in pos_list:
            size = float(p.get("size") or 0)
            pos_side = p.get("side")
            if size == 0:
                continue
            close_side = "Sell" if pos_side == "Buy" else "Buy"
            print(f"[autotrade] Closing existing position: side={pos_side} size={size} -> market {close_side}")
            close_res = await place_market_order(sess, symbol, close_side, qty=size, reduce_only=True)
            print(f"[autotrade] Close response: {close_res}")

        # set leverage
        lv = await set_leverage(sess, symbol, LEVERAGE)
        print(f"[autotrade] Set leverage response: {lv}")

        # re-check mark price before opening
        live_now, src = await fetch_mark_price(sess, symbol)
        if live_now is not None:
            if side == "Buy" and live_now > entry_price:
                print(f"[autotrade] Skip open: live {live_now} > entry {entry_price} (moved up)")
                write_audit_log({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "signal": sig,
                    "side": side,
                    "entry_price": entry_price,
                    "market_price": live_now,
                    "dpct5m": dpct5m,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "qty": 0.0,
                    "dry_run": False,
                    "reason": "moved_against",
                }, symbol)
                return
            if side == "Sell" and live_now < entry_price:
                print(f"[autotrade] Skip open: live {live_now} < entry {entry_price} (moved down)")
                write_audit_log({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "signal": sig,
                    "side": side,
                    "entry_price": entry_price,
                    "market_price": live_now,
                    "dpct5m": dpct5m,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "qty": 0.0,
                    "dry_run": False,
                    "reason": "moved_against",
                }, symbol)
                return

        # place market order
        open_res = await place_market_order(sess, symbol, side, qty=qty, reduce_only=False, take_profit=tp_price, stop_loss=sl_price)
        print(f"[autotrade] Open response: {open_res}")
        pos_after = await fetch_positions_list(sess, symbol)
        print_trade_summary(symbol, sig, side, qty, entry_price, tp_price, sl_price, executed_resp=open_res, pos_list=pos_after, dry_run=False, market_price=live_now or market_price, dpct5m=dpct5m)
        # write audit
        write_audit_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal": sig,
            "side": side,
            "entry_price": entry_price,
            "market_price": live_now or market_price,
            "dpct5m": dpct5m,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "qty": qty,
            "dry_run": False,
            "result": open_res,
            "positions": pos_after,
        }, symbol)


def parse_args():
    p = argparse.ArgumentParser(description="Autotrade Bybit futures using cryptoforecast.FirstOne")
    p.add_argument("--symbol", default="", help="Symbol to trade (e.g., BTCUSDT). Short tokens get USDT appended")
    p.add_argument("--live", action="store_true", help="Run in live non-interactive mode (signed API calls)")
    p.add_argument("--once", action="store_true", help="Run once and exit")
    return p.parse_args()


async def scheduler(symbol: str, dry_run: bool = True, once: bool = False):
    await handle_symbol(symbol, dry_run=dry_run)
    if once:
        return
    while True:
        now = time.time()
        wait = 300 - (int(now) % 300)
        print(f"[autotrade] Sleeping {wait}s until next 5-min boundary")
        await asyncio.sleep(wait)
        await handle_symbol(symbol, dry_run=dry_run)


if __name__ == "__main__":
    args = parse_args()
    # interactive prompts when running in a TTY
    if sys.stdin.isatty():
        print("Interactive autotrade_bybit configuration — press Enter to accept defaults")
        s = _ask("Symbol to trade (e.g. BTCUSDT)", default=args.symbol or "")
        args.symbol = s
        dry_default = not args.live
        dry_choice = _ask_bool("Dry-run (no signed API calls)?", default=dry_default)
        args.dry_run = dry_choice
        args.once = _ask_bool("Run once and exit?", default=args.once)
        try:
            t = _ask("Trade notional in USDT (TRADE_USDT)", default=str(TRADE_USDT))
            TRADE_USDT = float(t)
        except Exception:
            pass
        try:
            lv = _ask("Leverage to use (TRADE_LEVERAGE)", default=str(LEVERAGE))
            LEVERAGE = int(lv)
        except Exception:
            pass
        BYBIT_TESTNET = _ask_bool("Use Bybit testnet?", default=BYBIT_TESTNET)
        if not BYBIT_API_KEY:
            ak = _ask("BYBIT_API_KEY (leave empty to keep unset)", default="")
            if ak:
                BYBIT_API_KEY = ak
        if not BYBIT_API_SECRET:
            try:
                import getpass
                sk = getpass.getpass("BYBIT_API_SECRET (leave empty to keep unset): ")
            except Exception:
                sk = _ask("BYBIT_API_SECRET", default="")
            if sk:
                BYBIT_API_SECRET = sk

    sym = normalize_symbol(args.symbol)
    dry = bool(getattr(args, "dry_run", True))
    if dry:
        print("WARNING: running in dry-run mode (no signed API calls). Use --live to disable")
    # update API_HOST if testnet flag changed interactively
    API_HOST = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"
    try:
        asyncio.run(scheduler(sym, dry_run=dry, once=args.once))
    except KeyboardInterrupt:
        print("Interrupted by user")
#!/usr/bin/env python3
