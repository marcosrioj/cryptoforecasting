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
                results, dpcts_tmp, duration, now_utc, summary_lines = await cf._core_forecast(s)
        except Exception as e:
            # Compact error reporting only
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
                with contextlib.redirect_stdout(buf):
                    # Print the shared core summary (timeframes, overall)
                    cf.print_core_summary((results, dpcts_tmp, duration, now_utc, summary_lines), s, compact=compact)
                    # Run each registered strategy to allow them to append their blocks
                    for name, meta in getattr(cf, "STRATEGIES", {}).items():
                        fn = meta.get("fn")
                        try:
                            # most strategy functions accept (symbol, core=..., compact=...)
                            await fn(s, core=(results, dpcts_tmp, duration, now_utc, summary_lines), compact=compact)
                        except TypeError:
                            # fallback to older signature
                            await fn(s)
            except Exception as e:
                # If anything fails during rendering, fall back to the plain summary lines
                buf = io.StringIO()
                buf.write("".join([l + "\n" for l in summary_lines]))

            raw_formatted = buf.getvalue()
            # remove ANSI escape sequences (colors) for email readability
            ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
            formatted = ansi_re.sub("", raw_formatted)

            body = title + "\n\n" + formatted

            ok = send_email(title, body, email_to, smtp_user, smtp_pass)
            if ok:
                return (s, summary_lines)
            else:
                print(f"[watchlist] Failed to send email for {s}")
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
        return [x.strip().upper() for x in raw.split(",") if x.strip()]
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
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
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
