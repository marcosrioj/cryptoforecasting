#!/usr/bin/env python3
"""
autotrade_bybit.py

Automated Bybit USDT-perpetual trading bot using cryptoforecast.FirstOne predictions.
Runs once every 5 minutes on candle boundaries, executes a single trade per symbol.

Features:
- Default dry-run mode with TTY confirmation for live trades
- 5-minute candle boundary timing
- Position management (close existing before opening new)
- Take-profit and stop-loss with configurable parameters
- Minimum profit threshold filtering
- Comprehensive JSON audit logging
- Price source priority: mark price -> v5 ticker -> v2 ticker

Environment Variables Required for Live Trading:
- BYBIT_API_KEY: Bybit API key
- BYBIT_API_SECRET: Bybit API secret
- BYBIT_TESTNET: Set to "true" for testnet (optional, defaults to mainnet)

Usage:
    python3 autotrade_bybit.py [--symbol SYMBOL] [--live] [--leverage N] [--min-usd USD]
    
Examples:
    python3 autotrade_bybit.py                           # Dry-run BTCUSDT
    python3 autotrade_bybit.py --symbol ETHUSDT --live   # Live trade ETHUSDT
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
DEFAULT_MIN_USD = 10.0  # Minimum position size in USD

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
        """Place take-profit and stop-loss orders."""
        try:
            url = f"{self.base_url}/v5/order/create"
            
            # Determine opposite side for TP/SL
            close_side = "Sell" if side == "Buy" else "Buy"
            
            orders_placed = []
            
            # Place TP order if specified
            if tp_price:
                tp_params = {
                    "category": "linear",
                    "symbol": symbol,
                    "side": close_side,
                    "orderType": "Limit",
                    "qty": qty,
                    "price": str(tp_price),
                    "reduceOnly": True,
                    "timeInForce": "GTC"
                }
                
                headers = self._get_auth_headers(json.dumps(tp_params))
                async with self.session.post(url, json=tp_params, headers=headers) as resp:
                    tp_data = await resp.json()
                    orders_placed.append({
                        "type": "take_profit",
                        "success": resp.status == 200 and tp_data.get("retCode") == 0,
                        "response": tp_data,
                        "order_id": tp_data.get("result", {}).get("orderId") if resp.status == 200 else None
                    })
            
            # Place SL order if specified  
            if sl_price:
                sl_params = {
                    "category": "linear",
                    "symbol": symbol,
                    "side": close_side,
                    "orderType": "Market",
                    "qty": qty,
                    "stopLoss": str(sl_price),
                    "reduceOnly": True,
                    "timeInForce": "IOC"
                }
                
                headers = self._get_auth_headers(json.dumps(sl_params))
                async with self.session.post(url, json=sl_params, headers=headers) as resp:
                    sl_data = await resp.json()
                    orders_placed.append({
                        "type": "stop_loss", 
                        "success": resp.status == 200 and sl_data.get("retCode") == 0,
                        "response": sl_data,
                        "order_id": sl_data.get("result", {}).get("orderId") if resp.status == 200 else None
                    })
            
            return {
                "success": all(order["success"] for order in orders_placed),
                "orders": orders_placed
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "orders": []
            }


def wait_for_candle_boundary() -> None:
    """Wait until the next 5-minute candle boundary if not already aligned."""
    now = datetime.now(timezone.utc)
    # Calculate seconds past the hour
    seconds_past_hour = now.minute * 60 + now.second + now.microsecond / 1_000_000
    
    # Calculate seconds to next 5-minute boundary
    next_boundary_seconds = ((seconds_past_hour // 300) + 1) * 300
    
    # If we're within 1 second of a boundary, consider ourselves aligned
    if next_boundary_seconds - seconds_past_hour <= 1:
        print(f"Already aligned to 5-min boundary: {now.strftime('%H:%M:%S')}")
        return
    
    # Calculate wait time
    wait_seconds = next_boundary_seconds - seconds_past_hour
    if wait_seconds > 3600:  # If calculation goes past the hour
        wait_seconds -= 3600
    
    next_time = now + timedelta(seconds=wait_seconds)
    print(f"Waiting {wait_seconds:.1f}s for next 5-min boundary at {next_time.strftime('%H:%M:%S')}")
    time.sleep(wait_seconds)


def calculate_tp_sl_prices(entry_price: float, predicted_pct: float, side: str) -> Tuple[float, float]:
    """Calculate take-profit and stop-loss prices."""
    if side.lower() == "buy":
        # LONG position
        tp_price = entry_price * (1 + abs(predicted_pct))  # Use full predicted move
        sl_price = entry_price * (1 - SL_FRAC)  # 2.5% below entry
    else:
        # SHORT position  
        tp_price = entry_price * (1 - abs(predicted_pct))  # Use full predicted move
        sl_price = entry_price * (1 + SL_FRAC)  # 2.5% above entry
    
    return tp_price, sl_price


def check_minimum_profit(predicted_pct: float, leverage: int) -> Tuple[bool, float]:
    """Check if predicted move meets minimum profit threshold with leverage."""
    effective_move = abs(predicted_pct) * leverage
    meets_threshold = effective_move >= MIN_PROFIT_THRESHOLD
    return meets_threshold, effective_move


def get_trading_decision(prediction_result: dict) -> Tuple[Optional[str], str, float]:
    """Extract trading decision from FirstOne prediction result."""
    try:
        # FirstOne returns a dictionary with prediction data
        # The overall signal should be in the 'sig' field
        overall_signal = prediction_result.get("sig", "FLAT")
        
        # Get 5-minute prediction percentage from FirstOne data
        # FirstOne may return timeframe-specific data or overall data
        predicted_pct = 0.0
        
        # Try to extract prediction percentage from various possible locations
        if "predicted_pct" in prediction_result:
            predicted_pct = prediction_result["predicted_pct"]
        elif "delta_pct" in prediction_result:
            predicted_pct = prediction_result["delta_pct"]
        elif "5m" in prediction_result:
            tf_5m = prediction_result["5m"]
            if isinstance(tf_5m, dict):
                predicted_pct = tf_5m.get("predicted_pct", tf_5m.get("delta_pct", 0.0))
        
        # If we still don't have a prediction, try to infer from signal strength
        if predicted_pct == 0.0:
            if overall_signal in ["STRONGBUY", "STRONGSELL"]:
                # Assume strong signals represent at least 2% moves
                predicted_pct = 2.0 if overall_signal == "STRONGBUY" else -2.0
            elif overall_signal in ["BUY", "SELL"]:
                # Assume regular signals represent at least 1% moves  
                predicted_pct = 1.0 if overall_signal == "BUY" else -1.0
        
        # Convert to percentage if not already
        if abs(predicted_pct) > 1.0:
            predicted_pct = predicted_pct / 100.0
        
        # Map signal to trading side
        side = SIGNAL_TO_SIDE.get(overall_signal)
        reason = f"FirstOne signal: {overall_signal}, predicted move: {predicted_pct:.3f}%"
        
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
    """Ask for manual confirmation if running in TTY for live trades."""
    if not sys.stdin.isatty():
        return False  # Not in a TTY, don't execute live trades
    
    try:
        response = input("\nConfirm LIVE TRADE execution? (type 'YES' to confirm): ").strip()
        return response == "YES"
    except (EOFError, KeyboardInterrupt):
        print("\nAborted by user")
        return False


async def main():
    """Main trading bot execution."""
    parser = argparse.ArgumentParser(description="Bybit Auto-Trading Bot")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--live", action="store_true", help="Execute live trades (default: dry-run)")
    parser.add_argument("--leverage", type=int, default=DEFAULT_LEVERAGE, help=f"Leverage (default: {DEFAULT_LEVERAGE})")
    parser.add_argument("--min-usd", type=float, default=DEFAULT_MIN_USD, help=f"Minimum position size in USD (default: {DEFAULT_MIN_USD})")
    
    args = parser.parse_args()
    
    print(f"Bybit Auto-Trading Bot")
    print(f"Symbol: {args.symbol}")
    print(f"Mode: {'LIVE' if args.live else 'DRY-RUN'}")
    print(f"Leverage: {args.leverage}x")
    print(f"Min Position: ${args.min_usd}")
    print("-" * 50)
    
    # Wait for 5-minute candle boundary
    wait_for_candle_boundary()
    
    # Initialize audit log data
    audit_log = {
        "symbol": args.symbol,
        "mode": "live" if args.live else "dry-run",
        "leverage": args.leverage,
        "min_position_usd": args.min_usd,
        "sl_fraction": SL_FRAC,
        "min_profit_threshold": MIN_PROFIT_THRESHOLD,
        "dry_run": not args.live
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
        
        # Extract trading decision
        side, decision_reason, predicted_pct = get_trading_decision(prediction_result)
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
            
            # Calculate position size
            position_value_usd = max(args.min_usd, args.min_usd)  # Use minimum for now
            qty = position_value_usd / market_price
            
            audit_log.update({
                "side": side.lower(),
                "market_price": market_price,
                "price_source": price_source,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "qty": qty,
                "position_value_usd": position_value_usd
            })
            
            print(f"\nTRADE DECISION:")
            print(f"Side: {side}")
            print(f"Entry Price: ${market_price:.4f} ({price_source})")
            print(f"Take Profit: ${tp_price:.4f}")
            print(f"Stop Loss: ${sl_price:.4f}")
            print(f"Quantity: {qty:.6f}")
            print(f"Position Value: ${position_value_usd:.2f}")
            print(f"Reason: {decision_reason}")
            
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
                
                if not leverage_result["success"]:
                    # Warning but don't fail - might already be set
                    print(f"Warning: Failed to set leverage: {leverage_result}")
                
                # Step 3: Place market entry order
                print(f"Placing {side} market order for {qty:.6f} {args.symbol}...")
                entry_result = await client.place_market_order(args.symbol, side, f"{qty:.6f}")
                audit_log["entry_order"] = entry_result
                
                if not entry_result["success"]:
                    reason = f"Failed to place entry order: {entry_result}"
                    audit_log.update({
                        "side": "skip",
                        "decision_reason": reason
                    })
                    write_audit_log(args.symbol, audit_log)
                    print(f"SKIP: {reason}")
                    return
                
                print(f"Entry order placed successfully: {entry_result.get('order_id')}")
                
                # Step 4: Place TP/SL orders
                print("Placing TP/SL orders...")
                tp_sl_result = await client.place_tp_sl_order(
                    args.symbol, side, f"{qty:.6f}", tp_price, sl_price
                )
                audit_log["tp_sl_orders"] = tp_sl_result
                
                if not tp_sl_result["success"]:
                    print(f"Warning: Failed to place TP/SL orders: {tp_sl_result}")
                else:
                    print("TP/SL orders placed successfully")
                
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