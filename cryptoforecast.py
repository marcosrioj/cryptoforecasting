# cryptoforecast.py
#!/usr/bin/env python3
import asyncio, os, platform, warnings, aiohttp, time, argparse
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
BUY_BPS, SELL_BPS = 10, -10
WEIGHTS = {"1w": 5, "1d": 4, "4h": 3, "1h": 2, "5m": 1}
MIN_ROWS, SEED = 200, 42
W_PRICE, W_RET_LGB, W_RET_EN = 0.5, 0.3, 0.2
# ==========================================================

GREEN = Fore.GREEN + Style.BRIGHT
RED   = Fore.RED   + Style.BRIGHT
YELL  = Fore.YELLOW+ Style.BRIGHT
CYAN  = Fore.CYAN  + Style.BRIGHT
WHITE = Style.BRIGHT
RESET = Style.RESET_ALL

def clear_console():
    os.system("cls" if platform.system().lower().startswith("win") else "clear")

# ---------- LOG/CSV PATHS ----------
def _script_base_name() -> str:
    return Path(__file__).resolve().with_suffix("").name

def _log_path(now_utc: datetime, symbol: str) -> Path:
    base = Path("logs") / _script_base_name() / symbol.upper()
    fname = now_utc.strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    return base / fname

def _csv_path() -> Path:
    return Path(_script_base_name()) / "summary.csv"

def _write_summary_log(text: str, now_utc: datetime, symbol: str) -> None:
    path = _log_path(now_utc, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _append_csv(rows: List[dict]) -> None:
    csv_path = _csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header, encoding="utf-8")

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

# --------- Fetch ----------
async def fetch_kline(session, interval_code: str, limit: int, symbol: str) -> pd.DataFrame:
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": CATEGORY, "symbol": symbol, "interval": interval_code, "limit": limit}
    async with session.get(url, params=params, timeout=30) as resp:
        data = await resp.json()
        rows = data.get("result", {}).get("list", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"])
        df["ts"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
        for c in ["open","high","low","close","volume","turnover"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna()[["ts","open","high","low","close","volume","turnover"]].sort_values("ts").reset_index(drop=True)

async def fetch_all_tf(symbol: str) -> Dict[str, pd.DataFrame]:
    async with aiohttp.ClientSession() as s:
        tasks = [fetch_kline(s, iv, LIMITS[tf], symbol) for tf, iv in TIMEFRAMES.items()]
        res = await asyncio.gather(*tasks, return_exceptions=True)
    return {tf: r for (tf,_), r in zip(TIMEFRAMES.items(), res)}

# --------- Features & models ----------
def make_features(df):
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
    return GREEN if bias == "BUY" else RED if bias == "SELL" else YELL

def paint_val(val_str: str, bias: str) -> str:
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

# --------- Core run ---------
async def run_once(symbol: str):
    start_time = time.perf_counter()

    raw = await fetch_all_tf(symbol)
    results = {}
    for tf, data in raw.items():
        if isinstance(data, Exception) or not isinstance(data, pd.DataFrame) or data.empty:
            continue
        code = TIMEFRAMES[tf]
        data = trim_to_last_closed(data, code)
        if len(data) < MIN_ROWS: continue
        feat = make_features(data)
        if feat.empty: continue

        last = float(data["close"].iloc[-1])
        pred, cv, acc, ens_std, preds = fit_predict_ensemble(feat, last)
        rsi_v, macd_hist = float(feat["rsi_14"].iloc[-1]), float(feat["macd_diff"].iloc[-1])

        dpct = (pred - last) / last * 100
        base_sig = decide_signal(dpct)
        sig = overlay_with_rsi_macd(base_sig, rsi_v, macd_hist, last)
        conf = max(5.0, min(95.0, (1.0 - cv)*100.0 - (ens_std/last)*100.0))

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
            "last": last, "pred": pred, "dpct": dpct,
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

    # ---------- SUMMARY ----------
    now_utc = datetime.now(timezone.utc); now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    clear_console()
    print(f"{WHITE}SUMMARY{RESET}")
    print(f"{WHITE}symbol:{RESET} {SYMBOL}  {WHITE}category:{RESET} {CATEGORY}  {WHITE}generated_at:{RESET} {CYAN}{now_str}{RESET}")
    print(f"{WHITE}thresholds_bps:{RESET} buy>={BUY_BPS}, sell<={SELL_BPS}")
    print(f"{WHITE}weights:{RESET} {WEIGHTS}")
    print(f"{WHITE}execution_time_seconds:{RESET} {duration:.2f}\n")

    # ===== OVERALL FIRST =====
    print(f"{WHITE}[OVERALL]{RESET}")
    tf_signals_colored = []
    for tf in ORDERED_TF:
        if tf in results:
            tf_sig = results[tf]["sig"]
            tf_signals_colored.append(f"{tf}:{bias_color(tf_sig)}{tf_sig}{RESET}")
    vote_bias = "BUY" if vote > 0 else "SELL" if vote < 0 else "FLAT"
    vote_colored = paint_val(str(vote), vote_bias)
    print(f"signals={{{', '.join(tf_signals_colored)}}}")
    print(f"weights={WEIGHTS}  vote={vote_colored}")
    print(f"Signal={bias_color(overall)}{overall}{RESET}\n")

    # prepare log header
    summary_lines = [
        "SUMMARY",
        f"symbol: {SYMBOL}  category: {CATEGORY}  generated_at: {now_str}",
        f"thresholds_bps: buy>={BUY_BPS}, sell<={SELL_BPS}",
        f"weights: {WEIGHTS}",
        f"execution_time_seconds: {duration:.2f}",
        ""
    ]
    csv_rows: List[dict] = []

    # ===== PER-TIMEFRAME BLOCKS =====
    for tf in ORDERED_TF:
        r = results.get(tf)
        if not r: 
            continue

        dpct_val = (r["pred"] - r["last"]) / r["last"] * 100
        print(f"{WHITE}[{tf}]{RESET}")
        print(f"last={r['last']:.2f}  pred={r['pred']:.2f}  "
              f"Δ%={color_num_delta(dpct_val)}  "
              f"MAPE={color_num_mape(r['cv']*100)}  "
              f"dir_acc={color_num_diracc(r['acc'])} "
              f"conf={color_num_conf(r['conf'])}")

        nums = r["nums"]
        print(
            "ind=["
            + ", ".join([
                f"rsi14:{color_num_rsi(r['rsi'])}",
                f"macd:{color_num_macd(r['macd'], r['last'])}",
                f"ema_diff%:{color_num_ema_diff(nums['ema_diff_pct'])}",
                f"bb_pos%:{color_num_bb_pos(nums['bb_pos_pct'])}",
                f"bb_width%:{color_num_bb_width(nums['bb_width_pct'])}",
                f"atr%:{color_num_atr_pct(nums['atr_pct'])}",
                f"stoch_k:{color_num_stoch(nums['stoch_k'])}",
                f"mfi:{color_num_mfi(nums['mfi'])}",
                f"obv_slope%:{color_num_obv_slope(nums['obv_slope_pct'])}",
            ])
            + "]"
        )

        ai_d = r["ai_deltas"]; ai = r["ai"]
        ai_tokens = [f"{name}:{color_num_delta(ai_d[name])}%" for name in ["A","B","C","ENS"]]
        agree_bias = "BUY" if ai["agree"] == 3 else "FLAT" if ai["agree"] == 2 else "SELL"
        agree_colored = paint_val(f"{ai['agree']}/3", agree_bias)
        overlay_str = f"{YELL}{ai['overlay']}{RESET}"
        print(f"AI=[{', '.join(ai_tokens)}, agree={agree_colored}, overlay={overlay_str}]")

        print(f"Signal={bias_color(r['sig'])}{r['sig']}{RESET}\n")

        # log line (plain)
        summary_lines.append(
            f"[{tf}] last={r['last']:.2f}  pred={r['pred']:.2f}  Δ%={dpct_val:.2f}  "
            f"MAPE={(r['cv']*100):.2f}%  dir_acc={(r['acc']):.2f}%  conf={r['conf']:.2f}%  "
            f"signal={r['sig']}"
        )

        # CSV
        csv_rows.append({
            "generated_at_utc": now_str, "exec_time_s": round(duration, 2),
            "timeframe": tf, "symbol": SYMBOL,
            "last": round(r["last"], 2), "pred": round(r["pred"], 2),
            "delta_pct": round(dpct_val, 2),
            "mape_pct": round(r['cv']*100.0, 2), "dir_acc_pct": round(r['acc'], 2),
            "conf_pct": round(r['conf'], 2), "signal": r["sig"],
            "vote": vote, "overall_decision": overall
        })

    # overall for log
    summary_lines.append("")
    summary_lines.append(f"overall_decision: {overall}  (vote={vote})")

    # persist
    _write_summary_log("\n".join(summary_lines) + "\n", now_utc, SYMBOL)
    _append_csv(csv_rows)

# --------- Scheduler / CLI ---------
async def scheduler_loop(loop: bool, every: int, symbol: str):
    await run_once(symbol)  # immediate first run
    if not loop:
        return
    while True:
        if every == 300:
            now = datetime.utcnow()
            wait = 300 - ((now.minute * 60 + now.second) % 300)
        else:
            wait = every
        await asyncio.sleep(wait)
        await run_once(symbol)

def parse_args():
    p = argparse.ArgumentParser(description="Bybit Multi-timeframe Forecast")
    p.add_argument("--symbol", default=SYMBOL, help="Symbol (e.g., BTCUSDT). Default: BTCUSDT")
    p.add_argument("--loop", action="store_true", help="Run continuously (default: single run)")
    p.add_argument("--every", type=int, default=300, help="Seconds between runs when --loop (default: 300)")
    p.add_argument("--category", default=CATEGORY, help="Bybit category: linear, inverse, spot (default: linear)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    SYMBOL = args.symbol.upper()
    CATEGORY = args.category
    asyncio.run(scheduler_loop(loop=args.loop, every=args.every, symbol=SYMBOL))
