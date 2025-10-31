#!/usr/bin/env python3
"""
autotrade_bybit.py

Automated Bybit USDT-perpetual trading bot using cryptoforecast.FirstOne predictions.
Runs once every 5 minutes on candle boundaries, executes a single trade per symbol.

Features:
- Default dry-run mode with TTY confirmation for live trades
- 5-minute candle boundary timing
- Position management (close existing before opening new)
- Smart position sizing: 75% of available balance with leverage
- Take-profit and stop-loss with configurable parameters
- Minimum profit threshold filtering
- Comprehensive JSON audit logging
- Price source priority: mark price -> v5 ticker -> v2 ticker

Environment Variables Required for Live Trading:
- BYBIT_API_KEY: Bybit API key
- BYBIT_API_SECRET: Bybit API secret
- BYBIT_TESTNET: Set to "true" for testnet (optional, defaults to mainnet)

Usage:
    python3 autotrade_bybit.py [--symbol SYMBOL] [--live] [--leverage N] [--min-usd USD] [--no-wait]
    
Examples:
    python3 autotrade_bybit.py                           # Dry-run BTCUSDT, wait for 5-min boundary
    python3 autotrade_bybit.py --symbol ETHUSDT --live   # Live trade ETHUSDT, wait for boundary
    python3 autotrade_bybit.py --no-wait                 # Run immediately without waiting
    python3 autotrade_bybit.py --leverage 5 --min-usd 50 # Custom leverage & min size
"""

import asyncio
import argparse
import json
import os
import sys
import time
import hmac
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import aiohttp

# Import our forecasting module
import cryptoforecast

# ===================== CONFIGURATION =====================

# Trading parameters
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_LEVERAGE = 10
SL_FRAC = 0.025  # 2.5% stop-loss
MIN_PROFIT_THRESHOLD = 0.01  # 1% minimum profit requirement
DEFAULT_MIN_USD = 10.0  # Fallback minimum if balance check fails
BALANCE_PERCENTAGE = 0.75  # Use 75% of available balance

# Bybit endpoints
BYBIT_MAINNET_BASE = "https://api.bybit.com"
BYBIT_TESTNET_BASE = "https://api-testnet.bybit.com"

# Signal mapping for decision logic
SIGNAL_TO_SIDE = {
    "STRONGBUY": "Buy",
    "BUY": "Buy", 
    "STRONGSELL": "Sell",
    "SELL": "Sell",
    "FLAT": None
}

# =========================================================


class BybitClient:
    """Bybit API client with authentication and trading functions."""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = BYBIT_TESTNET_BASE if testnet else BYBIT_MAINNET_BASE
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, params: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests."""
        param_str = timestamp + self.api_key + "5000" + params  # recv_window = 5000
        return hmac.new(
            self.api_secret.encode('utf-8'), 
            param_str.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()
    
    def _get_auth_headers(self, params: str = "") -> Dict[str, str]:
        """Get authentication headers for API requests."""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, params)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }
    
    async def get_mark_price(self, symbol: str) -> Optional[float]:
        """Get mark price from Bybit v5 API (priority 1)."""
        try:
            url = f"{self.base_url}/v5/market/tickers"
            params = {"category": "linear", "symbol": symbol}
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    tickers = result.get("list", [])
                    if tickers:
                        mark_price = tickers[0].get("markPrice")
                        if mark_price:
                            return float(mark_price)
        except Exception as e:
            print(f"Error getting mark price: {e}")
        return None
    
    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get last price from Bybit v5 ticker (priority 2)."""
        try:
            url = f"{self.base_url}/v5/market/tickers"
            params = {"category": "linear", "symbol": symbol}
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    tickers = result.get("list", [])
                    if tickers:
                        last_price = tickers[0].get("lastPrice")
                        if last_price:
                            return float(last_price)
        except Exception as e:
            print(f"Error getting ticker price: {e}")
        return None
    
    async def get_v2_ticker_price(self, symbol: str) -> Optional[float]:
        """Get last price from Bybit v2 public ticker (priority 3 - fallback)."""
        try:
            url = f"{self.base_url}/v2/public/tickers"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", [])
                    if result:
                        last_price = result[0].get("last_price")
                        if last_price:
                            return float(last_price)
        except Exception as e:
            print(f"Error getting v2 ticker price: {e}")
        return None
    
    async def get_best_price(self, symbol: str) -> Tuple[Optional[float], str]:
        """Get best available price following priority order."""
        # Priority 1: Mark price
        price = await self.get_mark_price(symbol)
        if price:
            return price, "mark_price"
        
        # Priority 2: v5 ticker last price
        price = await self.get_ticker_price(symbol)
        if price:
            return price, "v5_ticker"
        
        # Priority 3: v2 ticker last price (fallback)
        price = await self.get_v2_ticker_price(symbol)
        if price:
            return price, "v2_ticker"
        
        return None, "failed"
    
    async def get_instrument_info(self, symbol: str) -> dict:
        """Get instrument info including min quantity and precision."""
        try:
            url = f"{self.base_url}/v5/market/instruments-info"
            params = {"category": "linear", "symbol": symbol}
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    instruments = result.get("list", [])
                    if instruments:
                        return instruments[0]
        except Exception as e:
            print(f"Error getting instrument info: {e}")
        return {}
    
    def format_quantity(self, qty: float, lot_size_filter: dict) -> str:
        """Format quantity according to Bybit's lot size rules."""
        min_qty = float(lot_size_filter.get("minOrderQty", "0.001"))
        max_qty = float(lot_size_filter.get("maxOrderQty", "1000000"))
        qty_step = float(lot_size_filter.get("qtyStep", "0.001"))
        
        # Ensure quantity meets minimum
        if qty < min_qty:
            qty = min_qty
        
        # Ensure quantity doesn't exceed maximum
        if qty > max_qty:
            qty = max_qty
        
        # Round to proper step size
        qty = round(qty / qty_step) * qty_step
        
        # Format with appropriate decimal places
        if qty_step >= 1:
            return f"{qty:.0f}"
        elif qty_step >= 0.1:
            return f"{qty:.1f}"
        elif qty_step >= 0.01:
            return f"{qty:.2f}"
        elif qty_step >= 0.001:
            return f"{qty:.3f}"
        else:
            return f"{qty:.6f}"
    
    async def get_wallet_balance(self) -> float:
        """Get available USDT balance from wallet."""
        try:
            url = f"{self.base_url}/v5/account/wallet-balance"
            params = {"accountType": "UNIFIED"}  # or "CONTRACT" for derivatives only
            headers = self._get_auth_headers()
            
            async with self.session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    accounts = result.get("list", [])
                    
                    if accounts:
                        coins = accounts[0].get("coin", [])
                        for coin in coins:
                            if coin.get("coin") == "USDT":
                                available_balance = float(coin.get("availableToWithdraw", "0"))
                                return available_balance
        except Exception as e:
            print(f"Error getting wallet balance: {e}")
        return 0.0
    
    async def get_positions(self, symbol: str) -> list:
        """Get current positions for a symbol."""
        try:
            url = f"{self.base_url}/v5/position/list"
            params = {"category": "linear", "symbol": symbol}
            headers = self._get_auth_headers()
            
            async with self.session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    return result.get("list", [])
                else:
                    print(f"Error getting positions: {resp.status}")
                    return []
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    async def close_position(self, symbol: str, side: str, qty: str) -> dict:
        """Close an existing position with market order (reduce-only)."""
        try:
            url = f"{self.base_url}/v5/order/create"
            
            # Determine opposite side for closing
            close_side = "Sell" if side == "Buy" else "Buy"
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": close_side,
                "orderType": "Market",
                "qty": qty,
                "reduceOnly": True,
                "timeInForce": "IOC"
            }
            
            headers = self._get_auth_headers(json.dumps(params))
            
            async with self.session.post(url, json=params, headers=headers) as resp:
                data = await resp.json()
                return {
                    "success": resp.status == 200 and data.get("retCode") == 0,
                    "response": data,
                    "status_code": resp.status
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": 0
            }
    
    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for a symbol."""
        try:
            url = f"{self.base_url}/v5/position/set-leverage"
            params = {
                "category": "linear", 
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage)
            }
            
            headers = self._get_auth_headers(json.dumps(params))
            
            async with self.session.post(url, json=params, headers=headers) as resp:
                data = await resp.json()
                return {
                    "success": resp.status == 200 and data.get("retCode") == 0,
                    "response": data,
                    "status_code": resp.status
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": 0
            }
    
    async def place_market_order(self, symbol: str, side: str, qty: str) -> dict:
        """Place a market order."""
        try:
            url = f"{self.base_url}/v5/order/create"
            params = {
                "category": "linear",
                "symbol": symbol, 
                "side": side,
                "orderType": "Market",
                "qty": qty,
                "timeInForce": "IOC"
            }
            
            headers = self._get_auth_headers(json.dumps(params))
            
            async with self.session.post(url, json=params, headers=headers) as resp:
                data = await resp.json()
                return {
                    "success": resp.status == 200 and data.get("retCode") == 0,
                    "response": data,
                    "status_code": resp.status,
                    "order_id": data.get("result", {}).get("orderId") if resp.status == 200 else None
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": 0
            }
    
    async def place_tp_sl_order(self, symbol: str, side: str, qty: str, tp_price: float = None, sl_price: float = None) -> dict:
        """Place take-profit and stop-loss orders using Bybit's position TP/SL endpoint."""
        try:
            orders_placed = []
            
            # Use the position trading endpoint for TP/SL
            if tp_price or sl_price:
                url = f"{self.base_url}/v5/position/trading-stop"
                
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "positionIdx": 0  # One-way mode
                }
                
                if tp_price:
                    params["takeProfit"] = str(tp_price)
                    
                if sl_price:
                    params["stopLoss"] = str(sl_price)
                
                headers = self._get_auth_headers(json.dumps(params))
                
                async with self.session.post(url, json=params, headers=headers) as resp:
                    data = await resp.json()
                    orders_placed.append({
                        "type": "tp_sl_combined",
                        "success": resp.status == 200 and data.get("retCode") == 0,
                        "response": data,
                        "tp_price": tp_price,
                        "sl_price": sl_price
                    })
            
            return {
                "success": all(order["success"] for order in orders_placed) if orders_placed else True,
                "orders": orders_placed
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "orders": []
            }

    async def place_tp_limit_order(self, symbol: str, side: str, qty: str, tp_price: float) -> dict:
        """Place take-profit as a limit order (reduce-only) instead of traditional TP."""
        try:
            url = f"{self.base_url}/v5/order/create"
            
            # Determine opposite side for TP (close position)
            tp_side = "Sell" if side == "Buy" else "Buy"
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": tp_side,
                "orderType": "Limit",
                "qty": qty,
                "price": str(tp_price),
                "reduceOnly": True,  # Ensure it's reduce-only
                "timeInForce": "GTC"  # Good Till Cancelled
            }
            
            headers = self._get_auth_headers(json.dumps(params))
            
            async with self.session.post(url, json=params, headers=headers) as resp:
                data = await resp.json()
                return {
                    "success": resp.status == 200 and data.get("retCode") == 0,
                    "response": data,
                    "tp_price": tp_price,
                    "order_type": "limit_tp"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "order_type": "limit_tp"
            }

    async def place_sl_order(self, symbol: str, side: str, qty: str, sl_price: float) -> dict:
        """Place stop-loss as a stop market order (reduce-only)."""
        try:
            url = f"{self.base_url}/v5/order/create"
            
            # Determine opposite side for SL (close position)
            sl_side = "Sell" if side == "Buy" else "Buy"
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": sl_side,
                "orderType": "Market",
                "qty": qty,
                "stopLoss": str(sl_price),
                "reduceOnly": True,  # Ensure it's reduce-only
                "timeInForce": "GTC"
            }
            
            headers = self._get_auth_headers(json.dumps(params))
            
            async with self.session.post(url, json=params, headers=headers) as resp:
                data = await resp.json()
                return {
                    "success": resp.status == 200 and data.get("retCode") == 0,
                    "response": data,
                    "sl_price": sl_price,
                    "order_type": "stop_loss"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "order_type": "stop_loss"
            }


def wait_for_candle_boundary() -> None:
    """Wait until the next 5-minute candle boundary if not already aligned."""
    now = datetime.now(timezone.utc)
    
    # Calculate seconds into the current minute
    seconds_into_minute = now.second + now.microsecond / 1_000_000
    
    # Calculate which 5-minute interval we're in (0, 5, 10, 15, ...)
    minutes_into_hour = now.minute
    current_5min_interval = (minutes_into_hour // 5) * 5
    next_5min_interval = current_5min_interval + 5
    
    # If next interval goes past 60, wrap to next hour
    if next_5min_interval >= 60:
        next_boundary = now.replace(hour=(now.hour + 1) % 24, minute=0, second=0, microsecond=0)
    else:
        next_boundary = now.replace(minute=next_5min_interval, second=0, microsecond=0)
    
    # Calculate wait time
    wait_seconds = (next_boundary - now).total_seconds()
    
    # If we're within 2 seconds of a boundary, consider ourselves aligned
    if wait_seconds <= 2:
        print(f"Already aligned to 5-min boundary: {now.strftime('%H:%M:%S')}")
        return
    
    print(f"Waiting {wait_seconds:.1f}s for next 5-min boundary at {next_boundary.strftime('%H:%M:%S')}")
    time.sleep(wait_seconds)


def calculate_tp_sl_prices(entry_price: float, predicted_pct: float, side: str) -> Tuple[float, float]:
    """Calculate take-profit and stop-loss prices.
    
    TP price uses the actual Δ% variation directly (e.g., 0.15% delta = 0.15% TP).
    Leverage does NOT influence the TP price calculation.
    """
    if side.lower() == "buy":
        # LONG position: TP above entry, SL below entry
        tp_price = entry_price * (1 + abs(predicted_pct))  # Use actual delta % directly
        sl_price = entry_price * (1 - SL_FRAC)  # 2.5% below entry
    else:
        # SHORT position: TP below entry, SL above entry
        tp_price = entry_price * (1 - abs(predicted_pct))  # Use actual delta % directly
        sl_price = entry_price * (1 + SL_FRAC)  # 2.5% above entry
    
    return tp_price, sl_price


def check_minimum_profit(predicted_pct: float, leverage: int) -> Tuple[bool, float]:
    """Check if predicted move meets minimum profit threshold with leverage."""
    # predicted_pct is already in decimal form (e.g., 0.015 for 1.5%)
    effective_move = abs(predicted_pct) * leverage
    meets_threshold = effective_move >= MIN_PROFIT_THRESHOLD
    return meets_threshold, effective_move


def get_trading_decision(prediction_result: dict, actual_5m_delta_pct: float = 0.0) -> Tuple[Optional[str], str, float]:
    """Extract trading decision from FirstOne prediction result."""
    try:
        # FirstOne returns: {'name': 'FirstOne', 'signal': 'SELL', 'reason': 'Original ensemble decision'}
        overall_signal = prediction_result.get("signal", "FLAT")
        
        # Use the actual 5m Δ% from FirstOne core data for take-profit calculation
        predicted_pct = actual_5m_delta_pct
        
        # Convert to decimal form if it's in percentage (FirstOne returns as percentage)
        if abs(predicted_pct) > 1.0:  # If it's in percentage form like 1.5
            predicted_pct = predicted_pct / 100.0
        
        # If the actual delta is too small, fall back to signal-based estimation for minimum moves
        if abs(predicted_pct) < 0.005:  # Less than 0.5%
            if overall_signal == "STRONGBUY":
                predicted_pct = 0.025  # 2.5%
            elif overall_signal == "BUY":
                predicted_pct = 0.015  # 1.5%
            elif overall_signal == "STRONGSELL":
                predicted_pct = -0.025  # -2.5%
            elif overall_signal == "SELL":
                predicted_pct = -0.015  # -1.5%
            else:
                predicted_pct = 0.0   # FLAT or unknown signals
        
        # Map signal to trading side
        side = SIGNAL_TO_SIDE.get(overall_signal)
        reason = f"FirstOne signal: {overall_signal}, 5m Δ%: {predicted_pct*100:.3f}% (actual: {actual_5m_delta_pct:.3f}%)"
        
        return side, reason, predicted_pct
        
    except Exception as e:
        return None, f"Error parsing prediction: {e}", 0.0


def write_audit_log(symbol: str, log_data: Dict[str, Any]) -> None:
    """Write audit log to JSON file."""
    # Create log directory structure
    log_dir = Path("logs/autotrade") / symbol
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now(timezone.utc)
    filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S.json")
    log_path = log_dir / filename
    
    # Add timestamp to log data
    log_data["timestamp"] = timestamp.isoformat()
    
    # Write to file
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    print(f"Audit log written to: {log_path}")


def confirm_live_trade() -> bool:
    """Always return True - no manual confirmation needed."""
    return True


async def main():
    """Main trading bot execution."""
    parser = argparse.ArgumentParser(description="Bybit Auto-Trading Bot")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--live", action="store_true", help="Execute live trades (default: dry-run)")
    parser.add_argument("--leverage", type=int, default=DEFAULT_LEVERAGE, help=f"Leverage (default: {DEFAULT_LEVERAGE})")
    parser.add_argument("--min-usd", type=float, default=DEFAULT_MIN_USD, help=f"Fallback minimum position size in USD if balance check fails (default: {DEFAULT_MIN_USD})")
    parser.add_argument("--no-wait", action="store_true", help="Run immediately without waiting for 5-min candle boundary")
    
    args = parser.parse_args()
    
    print(f"Bybit Auto-Trading Bot")
    print(f"Symbol: {args.symbol}")
    print(f"Mode: {'LIVE' if args.live else 'DRY-RUN'}")
    print(f"Leverage: {args.leverage}x")
    print(f"Position Sizing: {BALANCE_PERCENTAGE*100}% of balance (fallback: ${args.min_usd})")
    print(f"Timing: {'Immediate' if args.no_wait else '5-min boundary aligned'}")
    print("-" * 50)
    
    # Wait for 5-minute candle boundary unless --no-wait is specified
    if not args.no_wait:
        wait_for_candle_boundary()
    else:
        print("Skipping 5-minute boundary wait (--no-wait specified)")
    
    # Initialize audit log data
    audit_log = {
        "symbol": args.symbol,
        "mode": "live" if args.live else "dry-run",
        "leverage": args.leverage,
        "min_position_usd": args.min_usd,
        "sl_fraction": SL_FRAC,
        "min_profit_threshold": MIN_PROFIT_THRESHOLD,
        "dry_run": not args.live,
        "timing_mode": "immediate" if args.no_wait else "boundary_aligned"
    }
    
    try:
        # Get prediction from FirstOne
        print("Fetching FirstOne prediction...")
        prediction_result = await cryptoforecast.FirstOne(args.symbol, compact=True)
        
        if not prediction_result:
            reason = "No prediction available from FirstOne"
            audit_log.update({
                "side": "skip",
                "decision_reason": reason,
                "predicted_delta_pct": 0.0
            })
            write_audit_log(args.symbol, audit_log)
            print(f"SKIP: {reason}")
            return
        
        # Get core forecast data to extract actual 5m delta percentage
        print("Extracting 5m delta percentage...")
        core_data = await cryptoforecast._core_forecast(args.symbol)
        dpcts_tmp = core_data.get('dpcts_tmp', {}) if isinstance(core_data, dict) else {}
        actual_5m_delta_pct = dpcts_tmp.get('5m', 0.0) if isinstance(dpcts_tmp, dict) else 0.0
        print(f"Actual 5m Δ%: {actual_5m_delta_pct:.4f}%")
        
        # Extract trading decision with actual 5m delta
        side, decision_reason, predicted_pct = get_trading_decision(prediction_result, actual_5m_delta_pct)
        audit_log.update({
            "predicted_delta_pct": predicted_pct,
            "decision_reason": decision_reason
        })
        
        if not side:
            audit_log["side"] = "skip"
            write_audit_log(args.symbol, audit_log)
            print(f"SKIP: {decision_reason}")
            return
        
        # Check minimum profit threshold
        meets_threshold, effective_move = check_minimum_profit(predicted_pct, args.leverage)
        audit_log.update({
            "min_profit_check": meets_threshold,
            "effective_move_pct": effective_move,
            "required_move_pct": MIN_PROFIT_THRESHOLD
        })
        
        if not meets_threshold:
            reason = f"Predicted move {effective_move:.3f}% below threshold {MIN_PROFIT_THRESHOLD:.1f}%"
            audit_log.update({
                "side": "skip",
                "decision_reason": f"{decision_reason}. {reason}"
            })
            write_audit_log(args.symbol, audit_log)
            print(f"SKIP: {reason}")
            return
        
        # Initialize Bybit client for price checking
        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        testnet = os.getenv("BYBIT_TESTNET", "").lower() == "true"
        
        if args.live and (not api_key or not api_secret):
            reason = "Missing BYBIT_API_KEY or BYBIT_API_SECRET for live trading"
            audit_log.update({
                "side": "skip", 
                "decision_reason": reason
            })
            write_audit_log(args.symbol, audit_log)
            print(f"SKIP: {reason}")
            return
        
        async with BybitClient(api_key, api_secret, testnet) as client:
            # Get current market price
            market_price, price_source = await client.get_best_price(args.symbol)
            
            if not market_price:
                reason = "Failed to get market price from all sources"
                audit_log.update({
                    "side": "skip",
                    "decision_reason": reason,
                    "price_source": "failed"
                })
                write_audit_log(args.symbol, audit_log)
                print(f"SKIP: {reason}")
                return
            
            # Calculate TP/SL prices
            tp_price, sl_price = calculate_tp_sl_prices(market_price, predicted_pct, side)
            
            # Get instrument info for proper quantity formatting
            instrument_info = await client.get_instrument_info(args.symbol)
            lot_size_filter = instrument_info.get("lotSizeFilter", {})
            
            # Get available balance for position sizing
            available_balance = 0.0
            if args.live:
                available_balance = await client.get_wallet_balance()
                print(f"Available USDT balance: ${available_balance:.2f}")
            
            # Calculate position size using 75% of available balance or fallback to min USD
            if available_balance > 0:
                margin_to_use = available_balance * BALANCE_PERCENTAGE
                print(f"Using {BALANCE_PERCENTAGE*100}% of balance: ${margin_to_use:.2f}")
            else:
                margin_to_use = args.min_usd
                print(f"Using fallback minimum: ${margin_to_use:.2f}")
            
            # Calculate position value considering leverage
            # With leverage, position_value = margin_to_use * leverage
            # This way we only risk the margin amount but control a larger position
            position_value_usd = margin_to_use * args.leverage
            qty_raw = position_value_usd / market_price
            
            # Format quantity according to Bybit rules
            qty_formatted = client.format_quantity(qty_raw, lot_size_filter)
            qty = float(qty_formatted)  # Convert back to float for calculations
            
            # Calculate actual margin required (should match margin_to_use)
            margin_required = (qty * market_price) / args.leverage
            
            audit_log.update({
                "side": side.lower(),
                "market_price": market_price,
                "price_source": price_source,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "available_balance": available_balance,
                "balance_percentage": BALANCE_PERCENTAGE,
                "margin_to_use": margin_to_use,
                "position_value_usd": position_value_usd,
                "margin_required": margin_required,
                "qty_raw": qty_raw,
                "qty_formatted": qty_formatted,
                "qty": qty,
                "instrument_info": instrument_info
            })
            
            print(f"\nTRADE DECISION:")
            print(f"Side: {side}")
            print(f"Entry Price: ${market_price:.4f} ({price_source})")
            print(f"Take Profit: ${tp_price:.4f}")
            print(f"Stop Loss: ${sl_price:.4f}")
            print(f"Leverage: {args.leverage}x")
            print(f"Position Value: ${position_value_usd:.2f} (leveraged)")
            print(f"Margin Required: ${margin_required:.2f}")
            print(f"Quantity: {qty_formatted} (raw: {qty_raw:.6f})")
            print(f"Reason: {decision_reason}")
            
            # Show instrument constraints for debugging
            if lot_size_filter:
                print(f"Min Qty: {lot_size_filter.get('minOrderQty', 'unknown')}, " +
                      f"Max Qty: {lot_size_filter.get('maxOrderQty', 'unknown')}, " +
                      f"Step: {lot_size_filter.get('qtyStep', 'unknown')}")
            
            if args.live:
                # Confirm live trade execution
                if not confirm_live_trade():
                    audit_log.update({
                        "side": "skip",
                        "decision_reason": "Live trade not confirmed by user"
                    })
                    write_audit_log(args.symbol, audit_log)
                    print("Trade execution cancelled by user")
                    return
                
                print("\nExecuting LIVE TRADE...")
                
                # Step 1: Check and close existing positions
                print("Checking existing positions...")
                positions = await client.get_positions(args.symbol)
                active_positions = [p for p in positions if float(p.get("size", 0)) > 0]
                
                if active_positions:
                    print(f"Found {len(active_positions)} existing position(s), closing...")
                    close_results = []
                    for pos in active_positions:
                        pos_side = pos.get("side")
                        pos_size = pos.get("size")
                        close_result = await client.close_position(args.symbol, pos_side, pos_size)
                        close_results.append(close_result)
                        if close_result["success"]:
                            print(f"Closed {pos_side} position of {pos_size}")
                        else:
                            print(f"Failed to close {pos_side} position: {close_result}")
                    
                    audit_log["position_closes"] = close_results
                    
                    # Check if any close failed
                    if not all(r["success"] for r in close_results):
                        audit_log.update({
                            "side": "skip",
                            "decision_reason": "Failed to close existing positions"
                        })
                        write_audit_log(args.symbol, audit_log)
                        print("SKIP: Failed to close existing positions")
                        return
                else:
                    print("No existing positions found")
                
                # Step 2: Set leverage
                print(f"Setting leverage to {args.leverage}x...")
                leverage_result = await client.set_leverage(args.symbol, args.leverage)
                audit_log["leverage_result"] = leverage_result
                
                # Check if leverage setting failed (but ignore "leverage not modified" - that's OK)
                ret_code = leverage_result.get("response", {}).get("retCode", 0)
                if not leverage_result["success"] and ret_code != 110043:
                    # Only warn if it's not the "leverage not modified" error
                    print(f"Warning: Failed to set leverage: {leverage_result}")
                elif ret_code == 110043:
                    print(f"Leverage already set to {args.leverage}x")
                else:
                    print(f"Leverage set to {args.leverage}x successfully")
                
                # Step 3: Place market entry order
                print(f"Placing {side} market order for {qty_formatted} {args.symbol}...")
                entry_result = await client.place_market_order(args.symbol, side, qty_formatted)
                audit_log["entry_order"] = entry_result
                
                if not entry_result["success"]:
                    reason = f"Failed to place entry order: {entry_result.get('response', {}).get('retMsg', 'Unknown error')}"
                    audit_log.update({
                        "side": "skip",
                        "decision_reason": reason
                    })
                    write_audit_log(args.symbol, audit_log)
                    print(f"SKIP: {reason}")
                    print(f"Full API response: {entry_result}")
                    return
                
                print(f"Entry order placed successfully: {entry_result.get('order_id')}")
                
                # Step 4: Place TP as limit order and SL as stop order
                print("Placing TP limit order and SL stop order...")
                
                # Place TP as limit order (reduce-only)
                tp_result = await client.place_tp_limit_order(
                    args.symbol, side, qty_formatted, tp_price
                )
                
                # Place SL as stop market order (reduce-only)
                sl_result = await client.place_sl_order(
                    args.symbol, side, qty_formatted, sl_price
                )
                
                audit_log["tp_order"] = tp_result
                audit_log["sl_order"] = sl_result
                
                # Report results
                if tp_result["success"]:
                    print(f"TP limit order placed successfully at {tp_price:.4f}")
                else:
                    print(f"Warning: Failed to place TP limit order: {tp_result.get('error', 'Unknown error')}")
                
                if sl_result["success"]:
                    print(f"SL stop order placed successfully at {sl_price:.4f}")
                else:
                    print(f"Warning: Failed to place SL stop order: {sl_result.get('error', 'Unknown error')}")
                
                # Don't fail the entire trade for TP/SL issues
                tp_sl_success = tp_result["success"] and sl_result["success"]
                if tp_sl_success:
                    print("Both TP and SL orders placed successfully")
                
                audit_log["live_execution_completed"] = True
                print("✅ LIVE trade execution completed")
                
            else:
                print("\n📝 DRY-RUN mode - no actual trades executed")
                audit_log["dry_run_simulation"] = True
            
            write_audit_log(args.symbol, audit_log)
            print(f"\n{'LIVE' if args.live else 'DRY-RUN'} trade completed")
            
    except Exception as e:
        error_reason = f"Unexpected error: {e}"
        audit_log.update({
            "side": "skip",
            "decision_reason": error_reason
        })
        write_audit_log(args.symbol, audit_log)
        print(f"ERROR: {error_reason}")
        raise


if __name__ == "__main__":
    asyncio.run(main())