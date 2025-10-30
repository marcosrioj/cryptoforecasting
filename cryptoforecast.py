# cryptoforecast.py
#!/usr/bin/env python3
import asyncio, os, platform, warnings, aiohttp, time, argparse, shutil, sys
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from pathlib import Path

from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
from colorama import init as colorama_init, Fore, Style

warnings.filterwarnings("ignore")
colorama_init(autoreset=True)

# ===================== DEFAULT CONFIG =====================
SYMBOL   = "BTCUSDT"
CATEGORY = "linear"
TIMEFRAMES = {"1w": "W", "1d": "D", "4h": "240", "1h": "60", "5m": "5"}
ORDERED_TF = ["1w","1d","4h","1h","5m"]
LIMITS     = {"1w": 520, "1d": 1500, "4h": 1500, "1h": 1500, "5m": 1500}

# Per-timeframe minimum history
MIN_ROWS_TF = {"1w": 30, "1d": 200, "4h": 200, "1h": 200, "5m": 200}

BUY_BPS, SELL_BPS = 10, -10
WEIGHTS = {"1w": 5, "1d": 4, "4h": 3, "1h": 2, "5m": 1}
SEED = 42
W_PRICE, W_RET_LGB, W_RET_EN = 0.5, 0.3, 0.2
# ==========================================================

# ---- color setup ----
COLOR_ENABLED = True
def _set_color(enabled: bool):
    global COLOR_ENABLED, GREEN, RED, YELL, CYAN, WHITE, RESET
    COLOR_ENABLED = enabled
    if enabled:
        GREEN = Fore.GREEN + Style.BRIGHT
        RED   = Fore.RED   + Style.BRIGHT
        YELL  = Fore.YELLOW+ Style.BRIGHT
        CYAN  = Fore.CYAN  + Style.BRIGHT
        WHITE = Style.BRIGHT
        RESET = Style.RESET_ALL
    else:
        GREEN = RED = YELL = CYAN = WHITE = RESET = ""

_set_color(True)

# ---- icon setup ----
ICONS_ENABLED = True
ICON_OVERALL = "🧭"
ICON_1W = "🕒"
ICON_1D = "📅"
ICON_4H = "⏰"
ICON_1H = "🕐"
ICON_5M = "⚡"
TF_ICON = {"1w": ICON_1W, "1d": ICON_1D, "4h": ICON_4H, "1h": ICON_1H, "5m": ICON_5M}

def clear_console():
    os.system("cls" if platform.system().lower().startswith("win") else "clear")

def term_width(default=80):
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default

def divider(char="─"):
    return char * max(40, min(120, term_width()))

# ---------- LOG PATH ----------
def _script_base_name() -> str:
    return Path(__file__).resolve().with_suffix("").name

def _log_path(now_utc: datetime, symbol: str) -> Path:
    base = Path("logs") / _script_base_name() / symbol.upper()
    fname = now_utc.strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    return base / fname

def _write_summary_log(text: str, now_utc: datetime, symbol: str) -> None:
    path = _log_path(now_utc, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# --------- Time alignment ----------
def _week_start_utc(dt_utc: datetime) -> datetime:
    monday = dt_utc.date() - timedelta(days=dt_utc.weekday())
    return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)

def interval_ms(code: str) -> int:
    if code == "W": return 7 * 24 * 60 * 60 * 1000
    if code == "D": return 24 * 60 * 60 * 1000
    if code.isdigit(): return int(code) * 60 * 1000
    raise ValueError(f"Unsupported interval code: {code}")

def last_closed_open_time(code: str) -> datetime:
    now = datetime.now(timezone.utc)
    if code == "W": return _week_start_utc(now) - timedelta(weeks=1)
    if code == "D": return datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)
    im = interval_ms(code); now_ms = int(now.timestamp() * 1000)
    floored_ms = (now_ms // im) * im
    return datetime.fromtimestamp((floored_ms - im) / 1000, tz=timezone.utc)

def trim_to_last_closed(df: pd.DataFrame, code: str) -> pd.DataFrame:
    cutoff = pd.to_datetime(last_closed_open_time(code)).tz_convert(None)
    return df[df["ts"] <= cutoff].sort_values("ts").reset_index(drop=True)

# --------- helpers for formatting like raw last ---------
def decimals_in_str(num_str: str) -> int:
    if "." not in num_str:
        return 0
    return len(num_str.split(".")[1])

def format_like(last_str: str, value: float) -> str:
    d = decimals_in_str(last_str)
    if d == 0:
        # If last had no decimals, show integer
        return str(int(round(value)))
    return f"{value:.{d}f}"

# --------- Fetch ----------
async def fetch_kline(session, interval_code: str, limit: int, symbol: str) -> pd.DataFrame:
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": CATEGORY, "symbol": symbol, "interval": interval_code, "limit": limit}
    async with session.get(url, params=params, timeout=30) as resp:
        data = await resp.json()
        rows = data.get("result", {}).get("list", [])
        if not rows:
            return pd.DataFrame()
        # Build as strings first, keep original 'close' text
        df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"])
        df["close_str"] = df["close"].astype(str)
        # Now convert numerics
        df["ts"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
        for c in ["open","high","low","close","volume","turnover"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().sort_values("ts").reset_index(drop=True)
        return df[["ts","open","high","low","close","volume","turnover","close_str"]]

async def fetch_all_tf(symbol: str) -> Dict[str, pd.DataFrame]:
    async with aiohttp.ClientSession() as s:
        tasks = [fetch_kline(s, iv, LIMITS[tf], symbol) for tf, iv in TIMEFRAMES.items()]
        res = await asyncio.gather(*tasks, return_exceptions=True)
    return {tf: r for (tf,_), r in zip(TIMEFRAMES.items(), res)}

# --------- Features & models ----------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret_1"] = d["close"].pct_change()
    for w in [3,6,12,24,60]:
        ema = EMAIndicator(d["close"], window=w).ema_indicator()
        d[f"ema_{w}"] = ema
        d[f"close_ema_dist_{w}"] = (d["close"] - ema) / ema
    d["rsi_14"] = RSIIndicator(d["close"], window=14).rsi()
    stoch = StochRSIIndicator(d["close"], window=14, smooth1=3, smooth2=3)
    d["stoch_k"], d["stoch_d"] = stoch.stochrsi_k(), stoch.stochrsi_d()
    macd = MACD(d["close"], window_slow=26, window_fast=12, window_sign=9)
    d["macd"], d["macd_sig"], d["macd_diff"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    bb = BollingerBands(d["close"], window=20, window_dev=2)
    d["bb_h"] = bb.bollinger_hband(); d["bb_l"] = bb.bollinger_lband(); d["bb_m"] = bb.bollinger_mavg()
    d["bb_width"] = (d["bb_h"] - d["bb_l"]) / d["close"]
    d["atr_14"] = AverageTrueRange(d["high"], d["low"], d["close"], window=14).average_true_range()
    d["rv_12"] = d["ret_1"].rolling(12).std()
    d["mfi_14"] = MFIIndicator(d["high"], d["low"], d["close"], d["volume"], window=14).money_flow_index()
    d["obv"] = OnBalanceVolumeIndicator(d["close"], d["volume"]).on_balance_volume()
    d["y_next_close"] = d["close"].shift(-1)
    d["y_next_ret_pct"] = ((d["y_next_close"] - d["close"]) / d["close"]) * 100.0
    return d.dropna()

def ts_cv_metrics(X, y):
    X = X.select_dtypes(include=[np.number])
    tscv = TimeSeriesSplit(n_splits=3)
    mape_scores, dir_hits, dir_total = [], 0, 0
    for tr, te in tscv.split(X):
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, objective="mape", random_state=SEED)
        m.fit(X.iloc[tr], y.iloc[tr])
        yhat = m.predict(X.iloc[te])
        mape_scores.append(mean_absolute_percentage_error(y.iloc[te], yhat))
        proxy = X.iloc[te]["ema_3"] if "ema_3" in X.columns else X.iloc[te]["close"]
        dir_hits += (np.sign(yhat - proxy) == np.sign(y.iloc[te] - proxy)).sum()
        dir_total += len(y.iloc[te])
    return float(np.mean(mape_scores)), float(dir_hits / max(dir_total, 1))

def fit_predict_ensemble(df_feat, last_close):
    X = df_feat.drop(columns=["y_next_close","y_next_ret_pct","ts","start"], errors="ignore").select_dtypes(include=[np.number])
    y_close = df_feat["y_next_close"]; y_ret = df_feat["y_next_ret_pct"]
    cv_mape, dir_acc = ts_cv_metrics(X, y_close)

    mdl_a = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, objective="mape", random_state=SEED)
    mdl_a.fit(X, y_close); pred_a = float(mdl_a.predict(X.tail(1))[0])

    mdl_b = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, random_state=SEED)
    mdl_b.fit(X, y_ret);   pred_b_ret = float(mdl_b.predict(X.tail(1))[0])
    pred_b = last_close * (1.0 + pred_b_ret / 100.0)

    mdl_c = ElasticNet(alpha=0.001, l1_ratio=0.15, random_state=SEED, max_iter=2000)
    mdl_c.fit(X, y_ret);   pred_c_ret = float(mdl_c.predict(X.tail(1))[0])
    pred_c = last_close * (1.0 + pred_c_ret / 100.0)

    preds = np.array([pred_a, pred_b, pred_c])
    weights = np.array([W_PRICE, W_RET_LGB, W_RET_EN]); weights /= weights.sum()
    pred_ens = float(np.dot(weights, preds))
    ens_std = float(np.std(preds))
    preds_dict = {"A": pred_a, "B": pred_b, "C": pred_c, "ENS": pred_ens}
    return pred_ens, cv_mape, dir_acc, ens_std, preds_dict

def decide_signal(dpct):
    bps = dpct * 100
    if bps >= BUY_BPS: return "BUY"
    if bps <= SELL_BPS: return "SELL"
    return "FLAT"

def overlay_with_rsi_macd(sig, rsi, macd_diff, last_price):
    if 45 <= rsi <= 55 and abs(macd_diff)/max(last_price,1e-9) < 5e-4: return "FLAT"
    return sig

# ---------- color helpers ----------
def bias_color(bias: str) -> str:
    if not COLOR_ENABLED: return ""
    return GREEN if bias == "BUY" else RED if bias == "SELL" else YELL

def paint_val(val_str: str, bias: str) -> str:
    if not COLOR_ENABLED: return val_str
    return f"{bias_color(bias)}{val_str}{RESET}"

def color_num_delta(dpct: float) -> str:
    sig = decide_signal(dpct)
    return paint_val(f"{dpct:.2f}", sig)

def color_num_mape(mape_pct: float) -> str:
    if mape_pct < 5: bias="BUY"
    elif mape_pct <= 15: bias="FLAT"
    else: bias="SELL"
    return paint_val(f"{mape_pct:.2f}%", bias)

def color_num_diracc(dir_acc_pct: float) -> str:
    if dir_acc_pct >= 60: bias="BUY"
    elif dir_acc_pct >= 50: bias="FLAT"
    else: bias="SELL"
    return paint_val(f"{dir_acc_pct:.2f}%", bias)

def color_num_rsi(rsi: float) -> str:
    if rsi < 30: bias="BUY"
    elif rsi > 70: bias="SELL"
    else: bias="FLAT"
    return paint_val(f"{rsi:.2f}", bias)

def color_num_macd(macd_hist: float, last_price: float) -> str:
    near = abs(macd_hist) / max(last_price,1e-9) < 5e-4
    if near: bias="FLAT"
    elif macd_hist > 0: bias="BUY"
    else: bias="SELL"
    return paint_val(f"{macd_hist:.2f}", bias)

def color_num_conf(conf_pct: float) -> str:
    if conf_pct >= 85: bias="BUY"
    elif conf_pct >= 60: bias="FLAT"
    else: bias="SELL"
    return paint_val(f"{conf_pct:.2f}%", bias)

def color_num_ema_diff(pct: float) -> str:
    bias = "BUY" if pct > 0 else "SELL" if pct < 0 else "FLAT"
    if abs(pct) < 0.05: bias = "FLAT"
    return paint_val(f"{pct:.2f}%", bias)

def color_num_bb_pos(pct: float) -> str:
    if pct <= 20: bias="BUY"
    elif pct >= 80: bias="SELL"
    else: bias="FLAT"
    return paint_val(f"{pct:.2f}%", bias)

def color_num_bb_width(pct: float) -> str:
    return paint_val(f"{pct:.2f}%", "FLAT")

def color_num_atr_pct(pct: float) -> str:
    return paint_val(f"{pct:.2f}%", "FLAT")

def color_num_stoch(k: float) -> str:
    if k <= 20: bias="BUY"
    elif k >= 80: bias="SELL"
    else: bias="FLAT"
    return paint_val(f"{k:.2f}", bias)

def color_num_mfi(v: float) -> str:
    if v <= 20: bias="BUY"
    elif v >= 80: bias="SELL"
    else: bias="FLAT"
    return paint_val(f"{v:.2f}", bias)

def color_num_obv_slope(pct: float) -> str:
    if abs(pct) < 0.10: bias="FLAT"
    elif pct > 0: bias="BUY"
    else: bias="SELL"
    return paint_val(f"{pct:.2f}%", bias)

# ---------- pretty printing helpers ----------
def _icon(tf: str) -> str:
    if not ICONS_ENABLED: return ""
    if tf == "OVERALL": return ICON_OVERALL + " "
    return TF_ICON.get(tf, "") + " "

def _color_signal_word(sig: str) -> str:
    return f"{bias_color(sig)}{sig}{RESET}" if COLOR_ENABLED else sig

def print_header(symbol, category, now_str, thresholds, weights, duration):
    print(divider())
    print(("🚀 CRYPTOFORECAST SUMMARY" if ICONS_ENABLED else "CRYPTOFORECAST SUMMARY").center(max(40, min(80, term_width()))))
    print(divider())
    print(f"Symbol     : {symbol}")
    print(f"Category   : {category}")
    print(f"Generated  : {now_str}")
    print(f"Thresholds : BUY ≥ +{thresholds['buy_bps']} bps | SELL ≤ {thresholds['sell_bps']} bps")
    wtxt = ", ".join([f"{k}={v}" for k,v in weights.items()])
    print(f"Weights    : {wtxt}")
    print(f"Exec time  : {duration:.2f} s")
    print(divider())

def print_overall(signals_ordered: List[str], results: Dict[str,dict], vote: int, overall: str):
    print(("🧭 " if ICONS_ENABLED else "") + "[OVERALL]")
    parts = []
    for tf in signals_ordered:
        if tf in results:
            sig = results[tf]["sig"]
            parts.append(f"{tf}:{_color_signal_word(sig)}")
    print(f"Signals    : {', '.join(parts)}")
    print(f"Weights    : {WEIGHTS}  Vote: {_color_signal_word('BUY' if vote>0 else 'SELL' if vote<0 else 'FLAT')} ({vote})")
    print(f"Decision   : {_color_signal_word(overall)}")
    print(divider())

def print_timeframe_block(tf: str, r: dict):
    head = f"{_icon(tf)}[{tf}]  - Signal: {_color_signal_word(r['sig'])}"
    print(head)
    # Use original last_str and format pred with the same decimals
    last_str = r["last_str"]
    pred_fmt = format_like(last_str, r["pred"])
    dpct_val = (r["pred"] - r["last"]) / r["last"] * 100
    row1 = (
        f"last={last_str}  "
        f"pred={pred_fmt}  "
        f"Δ%={color_num_delta(dpct_val):>7}  "
        f"MAPE={color_num_mape(r['cv']*100):>7}  "
        f"dir_acc={color_num_diracc(r['acc']):>7}  "
        f"conf={color_num_conf(r['conf']):>7}"
    )
    print(row1)

    nums = r["nums"]
    ind_line = (
        "Indicators : "
        + ", ".join([
            f"RSI14={color_num_rsi(r['rsi'])}",
            f"MACD={color_num_macd(r['macd'], r['last'])}",
            f"EMAΔ%={color_num_ema_diff(nums['ema_diff_pct'])}",
            f"BBpos%={color_num_bb_pos(nums['bb_pos_pct'])}",
            f"BBwidth%={color_num_bb_width(nums['bb_width_pct'])}",
            f"ATR%={color_num_atr_pct(nums['atr_pct'])}",
            f"StochK={color_num_stoch(nums['stoch_k'])}",
            f"MFI={color_num_mfi(nums['mfi'])}",
            f"OBV%={color_num_obv_slope(nums['obv_slope_pct'])}",
        ])
    )
    print(ind_line)

    ai_d = r["ai_deltas"]; ai = r["ai"]
    ai_tokens = [f"{name}={color_num_delta(ai_d[name])}%" for name in ["A","B","C","ENS"]]
    agree_txt = f"{ai['agree']}/3"
    agree_bias = "BUY" if ai["agree"] == 3 else "FLAT" if ai["agree"] == 2 else "SELL"
    agree_colored = paint_val(agree_txt, agree_bias)
    overlay_txt = ai["overlay"]
    if COLOR_ENABLED:
        overlay_txt = f"{YELL}{overlay_txt}{RESET}"
    print(f"AI         : {', '.join(ai_tokens)}  agree={agree_colored}  overlay={overlay_txt}")
    print(divider())

def print_compact(symbol, overall, vote, order_tfs, results, dpcts):
    print(divider())
    title = (f"🧭 {symbol}: {overall} (vote={vote})" if ICONS_ENABLED else f"{symbol}: {overall} (vote={vote})")
    print(title)
    parts = []
    for tf in order_tfs:
        if tf in results:
            parts.append(f"{tf}={results[tf]['sig']}")
    print("TF Signals : " + ", ".join(parts))
    dpct_parts = []
    for tf in order_tfs:
        if tf in dpcts:
            dpct_val = dpcts[tf]
            dpct_parts.append(f"{tf}:{dpct_val:.2f}%")
    print("Δ%         : " + " | ".join(dpct_parts))
    print(divider())


def render_strategy_summary(symbol: str, strategy_name: str, core, strategy_decision: dict, compact: bool = False):
    """Render a consistent summary for any strategy using the shared core results.

    strategy_decision: dict with keys 'sig' (BUY/SELL/FLAT) and optional 'reason' and 'meta'
    """
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    clear_console()
    thresholds = {"buy_bps": BUY_BPS, "sell_bps": SELL_BPS}
    # show strategy name in the header category field
    print_header(symbol, f"{CATEGORY} | {strategy_name}", now_str, thresholds, WEIGHTS, duration)

    # compute underlying overall (from core results)
    vote = sum((WEIGHTS.get(tf,1) if r["sig"]=="BUY" else -WEIGHTS.get(tf,1) if r["sig"]=="SELL" else 0)
               for tf,r in results.items())
    underlying_overall = "BUY" if vote>0 else "SELL" if vote<0 else "FLAT"

    # Strategy decision
    strat_sig = strategy_decision.get("sig", "FLAT")
    reason = strategy_decision.get("reason", [])
    if isinstance(reason, list):
        reason_txt = "; ".join(reason)
    else:
        reason_txt = str(reason)

    # print underlying overall and then the strategy decision
    print(("🧭 " if ICONS_ENABLED else "") + f"[UNDERLYING OVERALL] {underlying_overall} (vote={vote})")
    print(divider())
    print(f"Strategy   : {strategy_name}")
    print(f"Decision   : {_color_signal_word(strat_sig)}  Reason: {reason_txt}")
    print(divider())

    if compact:
        print_compact(symbol, underlying_overall, vote, ORDERED_TF, results, dpcts_tmp)
    else:
        print_overall(ORDERED_TF, results, vote, underlying_overall)
        for tf in ORDERED_TF:
            r = results.get(tf)
            if r:
                print_timeframe_block(tf, r)

    # also write a summary log for the strategy decision
    log_lines = [f"STRATEGY: {strategy_name}", f"Decision: {strat_sig}", f"Reason: {reason_txt}", ""]
    log_lines.extend(summary_lines)
    _write_summary_log("\n".join(log_lines) + "\n", now_utc, symbol)
    return {"name": strategy_name, "signal": strat_sig, "reason": reason_txt}

# --------- Core run / Strategies ---------
async def _core_forecast(symbol: str):
    """Internal: perform the existing multi-timeframe forecast and return structured results.

    Returns a tuple: (results_dict, dpcts_tmp, duration, now_utc, summary_lines)
    """
    start_time = time.perf_counter()

    raw = await fetch_all_tf(symbol)
    results = {}
    dpcts_tmp = {}
    for tf, data in raw.items():
        if isinstance(data, Exception) or not isinstance(data, pd.DataFrame) or data.empty:
            continue
        code = TIMEFRAMES[tf]
        data = trim_to_last_closed(data, code)
        if len(data) < MIN_ROWS_TF.get(tf, 200):
            continue
        feat = make_features(data)
        if feat.empty: continue

        last = float(data["close"].iloc[-1])
        last_str = str(data["close_str"].iloc[-1])
        pred, cv, acc, ens_std, preds = fit_predict_ensemble(feat, last)
        rsi_v, macd_hist = float(feat["rsi_14"].iloc[-1]), float(feat["macd_diff"].iloc[-1])

        dpct = (pred - last) / last * 100
        dpcts_tmp[tf] = dpct
        base_sig = decide_signal(dpct)
        sig = overlay_with_rsi_macd(base_sig, rsi_v, macd_hist, last)
        conf = max(5.0, min(95.0, (1.0 - cv)*100.0) - (ens_std/last)*100.0)

        ema_s, ema_l = float(feat["ema_3"].iloc[-1]), float(feat["ema_24"].iloc[-1])
        bb_l, bb_m, bb_h = float(feat["bb_l"].iloc[-1]), float(feat["bb_m"].iloc[-1]), float(feat["bb_h"].iloc[-1])
        atr_v = float(feat["atr_14"].iloc[-1]); stoch_k = float(feat["stoch_k"].iloc[-1])
        mfi_v = float(feat["mfi_14"].iloc[-1])

        ema_diff_pct = (ema_s - ema_l) / max(ema_l, 1e-9) * 100.0
        bb_pos_pct   = (last - bb_l) / max((bb_h - bb_l), 1e-9) * 100.0
        bb_width_pct = (bb_h - bb_l) / max(last, 1e-9) * 100.0
        atr_pct      = atr_v / max(last, 1e-9) * 100.0
        stoch_k_val  = float(stoch_k)
        mfi_val      = float(mfi_v)
        obv_prev     = float(feat["obv"].iloc[-6]) if len(feat["obv"]) >= 6 else float(feat["obv"].iloc[0])
        obv_last     = float(feat["obv"].iloc[-1])
        denom        = abs(obv_prev) if abs(obv_prev) > 1e-9 else 1.0
        obv_slope_pct= (obv_last - obv_prev) / denom * 100.0

        ai_deltas = {
            "A":   (preds["A"]   - last) / last * 100.0,
            "B":   (preds["B"]   - last) / last * 100.0,
            "C":   (preds["C"]   - last) / last * 100.0,
            "ENS": (preds["ENS"] - last) / last * 100.0,
        }
        ai_A = decide_signal(ai_deltas["A"])
        ai_B = decide_signal(ai_deltas["B"])
        ai_C = decide_signal(ai_deltas["C"])
        ai_ens = decide_signal(ai_deltas["ENS"])
        ai_agree = sum(s in ("BUY","SELL") and s == ai_ens for s in [ai_A, ai_B, ai_C])
        overlay_applied = "Yes" if sig != base_sig else "No"

        results[tf] = {
            "last": last, "last_str": last_str, "pred": pred, "dpct": dpct,
            "cv": cv, "acc": acc*100.0, "conf": conf,
            "rsi": rsi_v, "macd": macd_hist, "sig": sig,
            "nums": {
                "ema_diff_pct": ema_diff_pct, "bb_pos_pct": bb_pos_pct, "bb_width_pct": bb_width_pct,
                "atr_pct": atr_pct, "stoch_k": stoch_k_val, "mfi": mfi_val, "obv_slope_pct": obv_slope_pct
            },
            "ai": {"A": ai_A, "B": ai_B, "C": ai_C, "ENS": ai_ens, "agree": ai_agree, "overlay": overlay_applied},
            "ai_deltas": ai_deltas
        }
    # vote & overall
    vote = sum((WEIGHTS.get(tf,1) if r["sig"]=="BUY" else -WEIGHTS.get(tf,1) if r["sig"]=="SELL" else 0)
               for tf,r in results.items())
    overall = "BUY" if vote>0 else "SELL" if vote<0 else "FLAT"

    duration = time.perf_counter() - start_time

    # ---------- SUMMARY INFO ----------
    now_utc = datetime.now(timezone.utc); now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # ---------- SUMMARY LOG (plain text) ----------
    summary_lines = [
        "SUMMARY",
        f"symbol: {SYMBOL}  category: {CATEGORY}  generated_at: {now_str}",
        f"thresholds_bps: buy>={BUY_BPS}, sell<={SELL_BPS}",
        f"weights: {WEIGHTS}",
        f"execution_time_seconds: {duration:.2f}",
        ""
    ]

    for tf in ORDERED_TF:
        r = results.get(tf)
        if not r:
            continue
        dpct_val = (r["pred"] - r["last"]) / r["last"] * 100
        pred_fmt = format_like(r["last_str"], r["pred"])
        summary_lines.append(
            f"[{tf}] last={r['last_str']}  pred={pred_fmt}  Δ%={dpct_val:.2f}  "
            f"MAPE={(r['cv']*100):.2f}%  dir_acc={(r['acc']):.2f}%  conf={r['conf']:.2f}%  "
            f"signal={r['sig']}"
        )

    summary_lines.append("")
    summary_lines.append(f"overall_decision: {overall}  (vote={vote})")

    _write_summary_log("\n".join(summary_lines) + "\n", now_utc, SYMBOL)

    return results, dpcts_tmp, duration, now_utc, summary_lines


async def FirstOne(symbol: str, *, compact=False):
    """The original strategy (renamed to FirstOne).

    This preserves the behavior of the original script. It runs the multi-timeframe
    ensemble forecast and prints the human-friendly summary.
    """
    results, dpcts_tmp, duration, now_utc, summary_lines = await _core_forecast(symbol)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    clear_console()
    thresholds = {"buy_bps": BUY_BPS, "sell_bps": SELL_BPS}
    print_header(symbol, CATEGORY, now_str, thresholds, WEIGHTS, duration)

    # compute vote/overall from results
    vote = sum((WEIGHTS.get(tf,1) if r["sig"]=="BUY" else -WEIGHTS.get(tf,1) if r["sig"]=="SELL" else 0)
               for tf,r in results.items())
    overall = "BUY" if vote>0 else "SELL" if vote<0 else "FLAT"

    if compact:
        print_compact(symbol, overall, vote, ORDERED_TF, results, dpcts_tmp)
    else:
        print_overall(ORDERED_TF, results, vote, overall)
        for tf in ORDERED_TF:
            r = results.get(tf)
            if r: print_timeframe_block(tf, r)

    return {"name": "FirstOne", "results": results, "summary_lines": summary_lines}


# Strategy registry: name -> dict(category, fn, description)
STRATEGIES = {}
def _register(name: str, category: str, fn, description: str = ""):
    STRATEGIES[name] = {"category": category, "fn": fn, "description": description}

# ---- Stubs for requested strategies (they may reuse FirstOne logic for now) ----
async def _scalping(symbol: str, *, core=None, compact=False):
    """Scalping: focus on 5m TF. Simple rule:
    - BUY when short EMA > mid EMA, stoch_k < 40 (recovering), and AI ensemble suggests BUY
    - SELL when short EMA < mid EMA and stoch_k > 80
    - otherwise FLAT
    """
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    tf = "5m"
    r = results.get(tf)
    sig = "FLAT"
    reason = []
    if r:
        nums = r["nums"]
        ema3 = nums.get("ema_diff_pct")  # actually percent diff 3 vs 24
        stoch_k = nums.get("stoch_k")
        ai_ens = r["ai"]["ENS"]
        if ema3 is not None and stoch_k is not None:
            if ema3 > 0 and stoch_k <= 40 and ai_ens == "BUY":
                sig = "BUY"; reason.append("EMA up + Stoch recovering + AI")
            elif ema3 < 0 and stoch_k >= 80:
                sig = "SELL"; reason.append("EMA down + Stoch overbought")
            else:
                sig = "FLAT"; reason.append("No clear scalping edge")
    # render summary
    return render_strategy_summary(symbol, "Scalping", core, {"sig": sig, "reason": reason}, compact=compact)

async def _breakout_day(symbol: str, *, core=None, compact=False):
    """Breakout day trading: look for breakout on 5m over recent range and volume confirmation."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    tf = "5m"
    r = results.get(tf)
    sig = "FLAT"
    reason = []
    if r:
        # We need raw data to check recent highs/volumes — try to access feature arrays via make_features not stored;
        # fallback to AI ensemble delta
        dpct = r.get("dpct", 0.0)
        last = r.get("last", 0.0)
        # breakout if dpct > 0.5% (arbitrary) and volume spike (use obv_slope_pct as proxy)
        vol_proxy = r["nums"].get("obv_slope_pct", 0.0)
        if dpct >= 0.5 and vol_proxy > 1.0:
            sig = "BUY"; reason.append("Price breakout + volume spike")
        elif dpct <= -0.5 and vol_proxy < -1.0:
            sig = "SELL"; reason.append("Breakdown + volume spike")
        else:
            reason.append("No confirmed breakout")
    return render_strategy_summary(symbol, "Breakout", core, {"sig": sig, "reason": reason}, compact=compact)

async def _range_trading(symbol: str, *, core=None, compact=False):
    """Range trading (mean reversion): use BB position on 1h/5m to buy near bottom, sell near top."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    # check 1h first, then 5m
    for tf in ("1h","5m"):
        r = results.get(tf)
        if not r: continue
        bb_pos = r["nums"].get("bb_pos_pct", 50.0)
        if bb_pos <= 20:
            reason = [f"BB pos {bb_pos:.1f}% on {tf}"]
            return render_strategy_summary(symbol, "RangeTrading", core, {"sig": "BUY", "reason": reason}, compact=compact)
        if bb_pos >= 80:
            reason = [f"BB pos {bb_pos:.1f}% on {tf}"]
            return render_strategy_summary(symbol, "RangeTrading", core, {"sig": "SELL", "reason": reason}, compact=compact)
    return render_strategy_summary(symbol, "RangeTrading", core, {"sig": "FLAT", "reason": ["No range entry conditions"]}, compact=compact)

async def _ai_ml(symbol: str, *, core=None, compact=False):
    """AI/ML strategy: rely on ensemble ENS delta on 1h and 5m — stronger weight to 1h for day trading."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    score = 0.0
    # weight 1h more
    for tf, w in (("1h", 0.7), ("5m", 0.3)):
        r = results.get(tf)
        if not r: continue
        ens_pct = r["ai_deltas"].get("ENS", 0.0)
        score += ens_pct * w
    sig = decide_signal(score)
    return render_strategy_summary(symbol, "AI_ML", core, {"sig": sig, "reason": [f"score={score:.3f}%"]}, compact=compact)

async def _multi_indicator(symbol: str, *, core=None, compact=False):
    """Multi-indicator high-win-rate style: require confluence across EMA trend, RSI, and Stoch on 1h or 4h."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    for tf in ("1h","4h"):
        r = results.get(tf)
        if not r: continue
        nums = r["nums"]
        ema_diff = nums.get("ema_diff_pct", 0.0)
        rsi = r.get("rsi", 50.0)
        stoch_k = nums.get("stoch_k", 50.0)
        if ema_diff > 0 and rsi < 60 and stoch_k < 50:
            return render_strategy_summary(symbol, "MultiIndicator", core, {"sig": "BUY", "reason": [f"confluence on {tf} (EMA+, RSI{rsi:.0f}, Stoch{stoch_k:.0f})"]}, compact=compact)
        if ema_diff < 0 and rsi > 40 and stoch_k > 50:
            return render_strategy_summary(symbol, "MultiIndicator", core, {"sig": "SELL", "reason": [f"confluence on {tf} (EMA-, RSI{rsi:.0f}, Stoch{stoch_k:.0f})"]}, compact=compact)
    return render_strategy_summary(symbol, "MultiIndicator", core, {"sig": "FLAT", "reason": ["No confluence"]}, compact=compact)

async def _price_action(symbol: str, *, core=None, compact=False):
    """Price action / liquidity-based: look for wick+rejection patterns on 1h/4h as proxy for ICT setups."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    for tf in ("1h","4h"):
        r = results.get(tf)
        if not r: continue
        nums = r["nums"]
        atr = nums.get("atr_pct", 0.0)
        last = r.get("last", 0.0)
        bb_pos = nums.get("bb_pos_pct", 50.0)
        # simple heuristic: if BB pos near top but recent MACD negative => liquidity grab => BUY
        if bb_pos >= 80 and r.get("macd", 0.0) < 0:
            return render_strategy_summary(symbol, "PriceAction", core, {"sig": "BUY", "reason": [f"liquidity grab on {tf}"]}, compact=compact)
        if bb_pos <= 20 and r.get("macd", 0.0) > 0:
            return render_strategy_summary(symbol, "PriceAction", core, {"sig": "SELL", "reason": [f"lower liquidity grab on {tf}"]}, compact=compact)
    return render_strategy_summary(symbol, "PriceAction", core, {"sig": "FLAT", "reason": ["No PA setups"]}, compact=compact)

async def _arbitrage(symbol: str, *, core=None, compact=False):
    # Arbitrage requires cross-exchange orderbook data; cannot implement from Bybit klines only.
    return render_strategy_summary(symbol, "Arbitrage", core, {"sig": "FLAT", "reason": ["not implemented: requires cross-exchange orderbook/spread data"]}, compact=compact)

async def _trend_following(symbol: str, *, core=None, compact=False):
    """Trend-following swing: look at 4h and 1d EMA trends; buy on pullbacks in uptrend."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    for tf in ("1d","4h"):
        r = results.get(tf)
        if not r: continue
        ema_diff = r["nums"].get("ema_diff_pct", 0.0)
        rsi = r.get("rsi", 50.0)
        if ema_diff > 0 and rsi < 55:
            return render_strategy_summary(symbol, "TrendFollowing", core, {"sig": "BUY", "reason": [f"EMA up, RSI {rsi:.0f} on {tf}"]}, compact=compact)
        if ema_diff < 0 and rsi > 45:
            return render_strategy_summary(symbol, "TrendFollowing", core, {"sig": "SELL", "reason": [f"EMA down, RSI {rsi:.0f} on {tf}"]}, compact=compact)
    return render_strategy_summary(symbol, "TrendFollowing", core, {"sig": "FLAT", "reason": ["No trend entries"]}, compact=compact)

async def _breakout_momentum_swing(symbol: str, *, core=None, compact=False):
    """Swing breakout: detect daily breakout above recent resistance (30-day range) with rising volume."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    r = results.get("1d")
    if not r:
        return render_strategy_summary(symbol, "BreakoutMomentumSwing", core, {"sig": "FLAT", "reason": ["no daily data"]}, compact=compact)
    # Use dpct as proxy for breakout strength
    dpct = r.get("dpct", 0.0)
    obv = r["nums"].get("obv_slope_pct", 0.0)
    if dpct >= 1.0 and obv > 1.0:
        return render_strategy_summary(symbol, "BreakoutMomentumSwing", core, {"sig": "BUY", "reason": [f"daily breakout dpct={dpct:.2f}%, OBV_slope={obv:.2f}%"]}, compact=compact)
    if dpct <= -1.0 and obv < -1.0:
        return render_strategy_summary(symbol, "BreakoutMomentumSwing", core, {"sig": "SELL", "reason": [f"daily breakdown dpct={dpct:.2f}%"]}, compact=compact)
    return render_strategy_summary(symbol, "BreakoutMomentumSwing", core, {"sig": "FLAT", "reason": ["no strong breakout"]}, compact=compact)

async def _support_resistance_swing(symbol: str, *, core=None, compact=False):
    """Support/Resistance swing: use 1d BB pos to trade bounces on daily ranges."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    r = results.get("1d")
    if not r:
        return render_strategy_summary(symbol, "SupportResistanceSwing", core, {"sig": "FLAT", "reason": ["no daily data"]}, compact=compact)
    bb_pos = r["nums"].get("bb_pos_pct", 50.0)
    if bb_pos <= 25:
        return render_strategy_summary(symbol, "SupportResistanceSwing", core, {"sig": "BUY", "reason": [f"BB pos {bb_pos:.1f}% on 1d"]}, compact=compact)
    if bb_pos >= 75:
        return render_strategy_summary(symbol, "SupportResistanceSwing", core, {"sig": "SELL", "reason": [f"BB pos {bb_pos:.1f}% on 1d"]}, compact=compact)
    return render_strategy_summary(symbol, "SupportResistanceSwing", core, {"sig": "FLAT", "reason": ["within range mid-band"]}, compact=compact)

async def _sentiment_swing(symbol: str, *, core=None, compact=False):
    # Sentiment requires external social/news APIs. We'll provide a simple volume-spike proxy:
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    r = results.get("1d") or results.get("4h")
    if not r:
        return render_strategy_summary(symbol, "SentimentSwing", core, {"sig": "FLAT", "reason": ["no TF data"]}, compact=compact)
    obv = r["nums"].get("obv_slope_pct", 0.0)
    if obv > 5.0:
        return render_strategy_summary(symbol, "SentimentSwing", core, {"sig": "BUY", "reason": [f"OBV_slope={obv:.1f}%"]}, compact=compact)
    if obv < -5.0:
        return render_strategy_summary(symbol, "SentimentSwing", core, {"sig": "SELL", "reason": [f"OBV_slope={obv:.1f}%"]}, compact=compact)
    return render_strategy_summary(symbol, "SentimentSwing", core, {"sig": "FLAT", "reason": ["no strong sentiment proxy"]}, compact=compact)

async def _ichimoku_swing(symbol: str, *, core=None, compact=False):
    """Simplified Ichimoku-style swing using EMA crossovers as proxy for Tenkan/Kijun."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    r = results.get("4h") or results.get("1h")
    if not r:
        return render_strategy_summary(symbol, "IchimokuSwing", core, {"sig": "FLAT", "reason": ["no TF data"]}, compact=compact)
    nums = r["nums"]
    # use ema_12 vs ema_24 proxy
    ema_diff = nums.get("ema_diff_pct", 0.0)
    if ema_diff > 0:
        return render_strategy_summary(symbol, "IchimokuSwing", core, {"sig": "BUY", "reason": ["ema12 > ema24 proxy"]}, compact=compact)
    if ema_diff < 0:
        return render_strategy_summary(symbol, "IchimokuSwing", core, {"sig": "SELL", "reason": ["ema12 < ema24"]}, compact=compact)
    return render_strategy_summary(symbol, "IchimokuSwing", core, {"sig": "FLAT", "reason": ["no clear signal"]}, compact=compact)

async def _hodl(symbol: str, *, core=None, compact=False):
    """HODL: recommend holding — check if long-term trend is favorable (1w EMA)."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    r = results.get("1w")
    if not r:
        return render_strategy_summary(symbol, "HODL", core, {"sig": "HOLD", "reason": ["No weekly data"]}, compact=compact)
    ema_diff = r["nums"].get("ema_diff_pct", 0.0)
    if ema_diff > 0:
        return render_strategy_summary(symbol, "HODL", core, {"sig": "HOLD", "reason": ["long-term up"]}, compact=compact)
    else:
        return render_strategy_summary(symbol, "HODL", core, {"sig": "HOLD", "reason": ["long-term down"]}, compact=compact)

async def _dca(symbol: str, *, core=None, compact=False):
    """DCA helper: recommend DCA buy when price below N-week moving average as a simple rule."""
    results, dpcts_tmp, duration, now_utc, summary_lines = core
    r = results.get("1w") or results.get("1d")
    if not r:
        return render_strategy_summary(symbol, "DCA", core, {"sig": "HOLD", "reason": ["no data"]}, compact=compact)
    # if weekly ema indicates price below long-term EMA24 -> suggest DCA buy
    ema_diff = r["nums"].get("ema_diff_pct", 0.0)
    if ema_diff < -2.0:
        return render_strategy_summary(symbol, "DCA", core, {"sig": "BUY", "reason": ["price significantly below long-term EMA"]}, compact=compact)
    return render_strategy_summary(symbol, "DCA", core, {"sig": "HOLD", "reason": ["No DCA action recommended"]}, compact=compact)

async def _staking(symbol: str, *, core=None, compact=False):
    return render_strategy_summary(symbol, "Staking", core, {"sig": "FLAT", "reason": ["Requires staking/APY info — not implemented"]}, compact=compact)

async def _diversified(symbol: str, *, core=None, compact=False):
    return render_strategy_summary(symbol, "Diversified", core, {"sig": "FLAT", "reason": ["Portfolio rebalance logic not implemented"]}, compact=compact)

async def _value_investing(symbol: str, *, core=None, compact=False):
    return render_strategy_summary(symbol, "ValueInvesting", core, {"sig": "FLAT", "reason": ["Requires fundamental data/research — not implemented"]}, compact=compact)

# Register strategies
_register("FirstOne", "Default", FirstOne, "Original forecasting strategy (default)")
_register("Scalping", "Day Trading", _scalping, "Scalping day-trading stub")
_register("Breakout", "Day Trading", _breakout_day, "Breakout day-trading stub")
_register("RangeTrading", "Day Trading", _range_trading, "Range trading stub")
_register("AI_ML", "Day Trading", _ai_ml, "AI/ML day-trading stub")
_register("MultiIndicator", "Day Trading", _multi_indicator, "Multi-indicator day-trading stub")
_register("PriceAction", "Day Trading", _price_action, "Price action / ICT stub")
_register("Arbitrage", "Day Trading", _arbitrage, "Arbitrage stub")

_register("TrendFollowing", "Swing Trading", _trend_following, "Trend-following swing stub")
_register("BreakoutMomentumSwing", "Swing Trading", _breakout_momentum_swing, "Breakout/momentum swing stub")
_register("SupportResistanceSwing", "Swing Trading", _support_resistance_swing, "Support/resistance swing stub")
_register("SentimentSwing", "Swing Trading", _sentiment_swing, "Sentiment/news swing stub")
_register("IchimokuSwing", "Swing Trading", _ichimoku_swing, "Ichimoku swing stub")

_register("HODL", "Long-Term", _hodl, "Buy-and-hold long-term stub")
_register("DCA", "Long-Term", _dca, "Dollar-cost-averaging stub")
_register("Staking", "Long-Term", _staking, "Staking/yield stub")
_register("Diversified", "Long-Term", _diversified, "Diversified portfolio stub")
_register("ValueInvesting", "Long-Term", _value_investing, "Value investing stub")


async def run_once(symbol: str, *, strategy_name: str = None, strategy_category: str = None, compact=False):
    """Run one or more strategies for the symbol.

    If strategy_name provided, runs that strategy. If strategy_category provided, runs all strategies
    that belong to that category. If neither provided, runs FirstOne (original behavior).
    """
    # determine which strategies to run
    to_run = []
    if strategy_name:
        s = STRATEGIES.get(strategy_name)
        if not s:
            print(f"Unknown strategy '{strategy_name}'. Available: {list(STRATEGIES.keys())}")
            return
        to_run.append((strategy_name, s["fn"]))
    elif strategy_category:
        for name, meta in STRATEGIES.items():
            if meta["category"].lower() == strategy_category.lower():
                to_run.append((name, meta["fn"]))
        if not to_run:
            print(f"No strategies found for category '{strategy_category}'. Available categories: "
                  f"{sorted(set(m["category"] for m in STRATEGIES.values()))}")
            return
    else:
        to_run.append(("FirstOne", STRATEGIES["FirstOne"]["fn"]))

    # run core forecast once and share results to all strategies
    core = await _core_forecast(symbol)
    for name, fn in to_run:
        print(divider())
        print(f"Running strategy: {name} (category={STRATEGIES[name]['category']})")
        try:
            await fn(symbol, core=core, compact=compact)
        except TypeError:
            # fallback if strategy signature older
            await fn(symbol)

# --------- Scheduler / CLI ---------
async def scheduler_loop(loop: bool, every: int, symbol: str, *, strategy_name: str = None, strategy_category: str = None, compact=False):
    await run_once(symbol, strategy_name=strategy_name, strategy_category=strategy_category, compact=compact)  # immediate first run
    if not loop:
        return
    while True:
        if every == 300:
            now = datetime.utcnow()
            wait = 300 - ((now.minute * 60 + now.second) % 300)
        else:
            wait = every
        await asyncio.sleep(wait)
        await run_once(symbol, strategy_name=strategy_name, strategy_category=strategy_category, compact=compact)

def parse_args():
    p = argparse.ArgumentParser(description="Bybit Multi-timeframe Forecast")
    p.add_argument("--symbol", default=SYMBOL, help="Symbol (e.g., BTCUSDT). Default: BTCUSDT")
    p.add_argument("--loop", action="store_true", help="Run continuously (default: single run)")
    p.add_argument("--every", type=int, default=300, help="Seconds between runs when --loop (default: 300)")
    p.add_argument("--category", default=CATEGORY, help="Bybit category: linear, inverse, spot (default: linear)")
    p.add_argument("--compact", action="store_true", help="Compact summary mode")
    p.add_argument("--strategy", default=None, help="Strategy name to run (e.g., FirstOne, Scalping).")
    p.add_argument("--strategy-category", default=None, help="Strategy category to run (Day Trading, Swing Trading, Long-Term).")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    p.add_argument("--no-icons", action="store_true", help="Disable icons/emojis")
    return p.parse_args()


def _ask(prompt: str, default=None, options: List[str]=None) -> str:
    """Ask the user for a value showing options and default. Returns a string or default when empty."""
    opt_txt = f" Options: {', '.join(options)}." if options else ""
    default_txt = f" (default: {default})" if default is not None else ""
    try:
        val = input(f"{prompt}{opt_txt}{default_txt}: ").strip()
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
    if val in ("y","yes","1","true","t"): return True
    if val in ("n","no","0","false","f"): return False
    return default


def interactive_fill_args(args: argparse.Namespace) -> argparse.Namespace:
    """If run interactively (no CLI args), prompt the user for parameters showing options and defaults.

    This only triggers when stdin is a TTY. For non-interactive use (cron), it leaves args unchanged.
    """
    if not sys.stdin.isatty():
        return args
    # Only prompt when the user ran the script without CLI args
    if len(sys.argv) > 1:
        return args

    print("Interactive mode — configure runtime parameters. Press Enter to accept the default shown.")
    # Symbol
    args.symbol = _ask("Symbol (e.g. BTCUSDT)", default=args.symbol).upper()

    # Loop (boolean)
    args.loop = _ask_bool("Run continuously (loop)?", default=args.loop)

    # Every (int)
    every_val = _ask("Seconds between runs when --loop (integer)", default=str(args.every))
    try:
        args.every = int(every_val)
    except Exception:
        args.every = args.every

    # Bybit category
    cat_opts = ["linear","inverse","spot"]
    args.category = _ask("Bybit category", default=args.category, options=cat_opts)

    # Compact
    args.compact = _ask_bool("Compact summary mode?", default=args.compact)

    # Colors / icons
    args.no_color = not _ask_bool("Enable ANSI colors?", default=not args.no_color)
    args.no_icons = not _ask_bool("Enable icons/emojis?", default=not args.no_icons)

    # Strategy selection
    try:
        available = sorted(STRATEGIES.keys())
    except Exception:
        available = []
    if available:
        print(f"Available strategies: {', '.join(available)}")
        s = _ask("Strategy name (leave blank to run default FirstOne)", default=args.strategy)
        args.strategy = s if s else None
        if not args.strategy:
            # ask category of strategies
            cats = sorted(set(meta["category"] for meta in STRATEGIES.values()))
            print(f"Available strategy categories: {', '.join(cats)}")
            sc = _ask("Strategy category to run (leave blank for none)", default=args.strategy_category, options=cats)
            args.strategy_category = sc if sc else None

    return args

if __name__ == "__main__":
    args = parse_args()
    args = interactive_fill_args(args)
    SYMBOL = args.symbol.upper()
    CATEGORY = args.category
    _set_color(not args.no_color)
    ICONS_ENABLED = not args.no_icons
    asyncio.run(scheduler_loop(loop=args.loop, every=args.every, symbol=SYMBOL,
                              strategy_name=args.strategy, strategy_category=args.strategy_category,
                              compact=args.compact))
