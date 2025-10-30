#!/usr/bin/env python3
"""
watchlist_notify.py

Runs `cryptoforecast._core_forecast` for a small watchlist of symbols on a schedule
(hourly by default). When any pair's overall decision equals `STRONGBUY`, sends an
email via SMTP with the full summary in the body.

Configuration:
 - EMAIL_USER: SMTP username (Gmail address)
 - EMAIL_PASS: SMTP password (App Password recommended for Gmail)
 - EMAIL_TO:   recipient (defaults to EMAIL_USER)
 - WATCHLIST: optional comma-separated symbols to override the built-in list

Note: Gmail requires either an app password (if 2FA enabled) or allowing less-secure
apps for basic SMTP. Using an app password is recommended.
"""
import os
import asyncio
import time
from datetime import datetime, timezone
import smtplib
from email.message import EmailMessage
# OAuth2 support removed for simpler SMTP-only operation
import argparse
import traceback
import aiohttp
import warnings
import io
import contextlib
import re

import cryptoforecast as cf

# Suppress runtime/library warnings in console
warnings.filterwarnings("ignore")


DEFAULT_WATCHLIST = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
    "MATICUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "XRPUSDT",
]


def get_env(name: str, default=None):
    return os.environ.get(name, default)


def normalize_symbol(s: str) -> str:
    """Normalize a user-provided symbol: append USDT for short alphabetic tokens (e.g. 'OL' -> 'OLUSDT')."""
    if not s:
        return s
    s = s.strip().upper()
    if len(s) <= 5 and s.isalpha() and not s.endswith("USDT"):
        return s + "USDT"
    return s


def send_email(subject: str, body: str, recipient: str, smtp_user: str, smtp_pass: str,
               smtp_host: str = "smtp.gmail.com", smtp_port: int = 465):
    if not smtp_user or not smtp_pass:
        # Minimal reporting; avoid verbose traces
        print("[watchlist] SMTP credentials not configured. Set EMAIL_USER and EMAIL_PASS.")
        return False
    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as s:
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        # Keep output minimal
        print(f"[watchlist] Email sent to {recipient}")
        return True
    except smtplib.SMTPAuthenticationError:
        # Compact auth failure message
        print("[watchlist] SMTP authentication failed: check EMAIL_USER/EMAIL_PASS or use an App Password")
        return False
    except Exception as e:
        print(f"[watchlist] Failed to send email: {e}")
        return False


    



def parse_overall_from_summary(summary_lines):
    # find line starting with 'overall_decision:'
    for ln in summary_lines:
        if ln.startswith("overall_decision:"):
            # format: overall_decision: STRONGBUY  (vote=...)
            parts = ln.split(":", 1)
            if len(parts) < 2:
                continue
            rhs = parts[1].strip()
            # take the first token as overall
            overall = rhs.split()[0]
            return overall
    return None


async def check_once(symbols, smtp_user, smtp_pass, email_to, compact=False, concurrency: int = 5):
    """Run forecasts for `symbols` in parallel (bounded by `concurrency`).

    Returns a list of triggered alerts as (symbol, summary_lines).
    """
    triggered = []
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def worker(s):
        try:
            async with sem:
                # Retry logic: initial attempt + up to 3 retries on network-related errors
                max_attempts = 4
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        results, dpcts_tmp, duration, now_utc, summary_lines = await cf._core_forecast(s)
                        break
                    except Exception as e:
                        # Only retry on network/IO/timeout related exceptions
                        is_network = isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError, OSError))
                        if not is_network:
                            # Non-network error: report and skip this symbol
                            print(f"[watchlist] Error checking {s}: {e}")
                            return None
                        if attempt >= max_attempts:
                            print(f"[watchlist] Network failure for {s} after {attempt} attempts; skipping until next run")
                            return None
                        # Compact retry message and exponential backoff
                        backoff = 2 ** (attempt - 1)
                        print(f"[watchlist] Network error for {s} (attempt {attempt}/{max_attempts}), retrying in {backoff}s")
                        await asyncio.sleep(backoff)
                # end while
        except Exception as e:
            # Fallback compact reporting
            print(f"[watchlist] Error checking {s}: {e}")
            return None

        overall = parse_overall_from_summary(summary_lines)
        # Trigger on strong signals in either direction
        if overall in ("STRONGBUY", "STRONGSELL"):
            title = f"CRYPTOFORECAST ALERT: {overall} {s} @ {now_utc.isoformat()}"

            # Capture the human-friendly output from cryptoforecast's printing routines
            # (print_core_summary + each registered strategy) into a string and include
            # it in the email body. We strip ANSI escape sequences so the email body is
            # readable in plain-text clients.
            buf = io.StringIO()
            try:
                # First, call each strategy while capturing their return dicts (but not printing)
                strat_caps = {}
                for name, meta in getattr(cf, "STRATEGIES", {}).items():
                    fn = meta.get("fn")
                    try:
                        tmp_buf = io.StringIO()
                        with contextlib.redirect_stdout(tmp_buf):
                            try:
                                res = await fn(s, core=(results, dpcts_tmp, duration, now_utc, summary_lines), compact=compact)
                            except TypeError:
                                res = await fn(s)
                    except Exception:
                        res = None
                    if isinstance(res, dict):
                        strat_caps[name] = res

                # Aggregate strategy-level votes (excluding FirstOne) to compute overall
                strat_vote = 0
                for name, r in strat_caps.items():
                    if name == "FirstOne":
                        continue
                    sig = r.get("signal") or r.get("sig")
                    if sig:
                        strat_vote += cf.signal_to_multiplier(sig)
                n_strats = max(1, len(getattr(cf, "STRATEGIES", {})))
                max_strat_weight = 2 * n_strats
                strat_strong_th = int(max_strat_weight * 0.7)
                if strat_vote >= strat_strong_th:
                    overall_from_strats = "STRONGBUY"
                elif strat_vote > 0:
                    overall_from_strats = "BUY"
                elif strat_vote <= -strat_strong_th:
                    overall_from_strats = "STRONGSELL"
                elif strat_vote < 0:
                    overall_from_strats = "SELL"
                else:
                    overall_from_strats = "FLAT"

                # Now render core summary with the strategy-derived overall, and
                # append each strategy block (prefer printing via render when we
                # have captured data to avoid recomputation).
                with contextlib.redirect_stdout(buf):
                    cf.print_core_summary((results, dpcts_tmp, duration, now_utc, summary_lines), s, compact=compact, overall_override=overall_from_strats)
                    for name, meta in getattr(cf, "STRATEGIES", {}).items():
                        if name in strat_caps:
                            # Print using the captured decision via render_strategy_summary
                            cap = strat_caps[name]
                            try:
                                cf.render_strategy_summary(s, name, (results, dpcts_tmp, duration, now_utc, summary_lines), {"sig": cap.get("signal") or cap.get("sig"), "reason": cap.get("reason")}, compact=compact)
                            except Exception:
                                # fallback: call the function to print
                                fn = meta.get("fn")
                                try:
                                    await fn(s, core=(results, dpcts_tmp, duration, now_utc, summary_lines), compact=compact)
                                except TypeError:
                                    await fn(s)
                        else:
                            # No captured return — call the strategy to print its block
                            fn = meta.get("fn")
                            try:
                                await fn(s, core=(results, dpcts_tmp, duration, now_utc, summary_lines), compact=compact)
                            except TypeError:
                                await fn(s)
            except Exception:
                # On any failure, fall back to raw summary_lines
                buf = io.StringIO()
                buf.write("".join([l + "\n" for l in summary_lines]))

            raw_formatted = buf.getvalue()
            # remove ANSI escape sequences (colors) for email readability
            ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
            formatted = ansi_re.sub("", raw_formatted)

            # Prefer the Decision printed in the [OVERALL] block. Parse it from the
            # captured formatted output to ensure we only alert on overall STRONG signals.
            overall_from_overall_section = None
            try:
                idx = formatted.find("[OVERALL]")
                if idx != -1:
                    snippet = formatted[idx: idx + 1000]
                    m = re.search(r"Decision\s*:\s*([A-Z]+)", snippet)
                    if m:
                        overall_from_overall_section = m.group(1)
            except Exception:
                overall_from_overall_section = None

            # If we couldn't parse the OVERALL block, fall back to the summary_lines parse
            if overall_from_overall_section is None:
                overall_from_overall_section = overall

            # Only send mail when the OVERALL decision is STRONGBUY or STRONGSELL
            if overall_from_overall_section in ("STRONGBUY", "STRONGSELL"):
                body = title + "\n\n" + formatted
                ok = send_email(title, body, email_to, smtp_user, smtp_pass)
                if ok:
                    return (s, summary_lines)
                else:
                    print(f"[watchlist] Failed to send email for {s}")
            else:
                # Do not send an alert if overall decision is not strong — remain silent by design
                pass
        return None

    tasks = [asyncio.create_task(worker(s)) for s in symbols]
    results_list = await asyncio.gather(*tasks)
    triggered = [r for r in results_list if r]
    if triggered:
        print(f"[watchlist] Triggered alerts: {', '.join([t[0] for t in triggered])}")
    return triggered


async def hourly_loop(symbols, smtp_user, smtp_pass, email_to, run_once=False, compact=False, concurrency: int = 5):
    # Run immediately, then wait for next hour boundary
    await check_once(symbols, smtp_user, smtp_pass, email_to, compact=compact, concurrency=concurrency)
    if run_once:
        return
    while True:
        # sleep until the top of the next hour
        now = time.time()
        # next hour boundary
        next_hour = ((int(now) // 3600) + 1) * 3600
        wait = max(1, next_hour - now)
        print(f"[watchlist] Sleeping {wait:.0f}s until next hourly run")
        await asyncio.sleep(wait)
        await check_once(symbols, smtp_user, smtp_pass, email_to, compact=compact, concurrency=concurrency)


def build_watchlist_from_env():
    raw = get_env("WATCHLIST")
    if raw:
        return [normalize_symbol(x) for x in raw.split(",") if x.strip()]
    return DEFAULT_WATCHLIST


def main():
    p = argparse.ArgumentParser(description="Watchlist notifier for cryptoforecast")
    p.add_argument("--once", action="store_true", help="Run once and exit")
    p.add_argument("--compact", action="store_true", help="Ask cryptoforecast for compact outputs where relevant")
    p.add_argument("--test-email", action="store_true", help="Send a test email using configured SMTP creds and exit")
    p.add_argument("--email-to", default=None, help="Recipient email (overrides EMAIL_TO env var)")
    p.add_argument("--symbols", default=None, help="Comma-separated list of symbols to monitor (overrides WATCHLIST env var)")
    p.add_argument("--concurrency", type=int, default=None, help="Number of symbols to check in parallel (default: 5 or WATCHER_CONCURRENCY env)")
    args = p.parse_args()

    smtp_user = get_env("EMAIL_USER")
    smtp_pass = get_env("EMAIL_PASS")
    email_to = args.email_to or get_env("EMAIL_TO") or smtp_user

    if args.symbols:
        symbols = [normalize_symbol(x) for x in args.symbols.split(",") if x.strip()]
    else:
        symbols = build_watchlist_from_env()

    if not symbols:
        print("[watchlist] No symbols configured. Set WATCHLIST env or pass --symbols.")
        return

    print(f"[watchlist] Monitoring {len(symbols)} symbols: {', '.join(symbols)}")
    print("[watchlist] Email recipient:", email_to)

    # If user requested a test email, send one and exit
    if args.test_email:
        subj = f"CRYPTOFORECAST TEST EMAIL from {smtp_user or 'watchlist_notify'}"
        body_lines = [
            "This is a test email from watchlist_notify.py",
            f"Time: {datetime.now(timezone.utc).isoformat()}",
            "Configured watchlist:",
            ", ".join(symbols),
        ]
        body = "\n".join(body_lines)
        ok = send_email(subj, body, email_to, smtp_user, smtp_pass)

        if ok:
            print("[watchlist] Test email sent successfully. Exiting.")
        else:
            print("[watchlist] Test email failed. Check SMTP/OAuth configuration and logs.")
        return

    # determine concurrency: CLI arg > env var > default 5
    concurrency = args.concurrency or int(get_env("WATCHER_CONCURRENCY") or 5)
    try:
        asyncio.run(hourly_loop(symbols, smtp_user, smtp_pass, email_to, run_once=args.once, compact=args.compact, concurrency=concurrency))
    except KeyboardInterrupt:
        print("[watchlist] Stopped by user")


if __name__ == "__main__":
    main()
