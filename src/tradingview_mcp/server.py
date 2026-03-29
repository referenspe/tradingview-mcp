from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from mcp.server.fastmcp import FastMCP

# Import bollinger band screener modules
from tradingview_mcp.core.services.indicators import (
    compute_metrics, extract_extended_indicators, analyze_timeframe_context,
    compute_stock_score, compute_trade_setup, compute_trade_quality,
    compute_fibonacci_levels, analyze_fibonacci_position, detect_trend_for_fibonacci,
)
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.utils.validators import sanitize_timeframe, sanitize_exchange, EXCHANGE_SCREENER, ALLOWED_TIMEFRAMES, STOCK_EXCHANGES, is_stock_exchange, get_market_type
from tradingview_mcp.core.services.sentiment_service import analyze_sentiment
from tradingview_mcp.core.services.news_service import fetch_news_summary
from tradingview_mcp.core.services.yahoo_finance_service import get_price, get_prices_bulk, get_market_snapshot
from tradingview_mcp.core.services.backtest_service import run_backtest, compare_strategies as _compare_strategies

try:
    from tradingview_ta import TA_Handler, get_multiple_analysis
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False

try:
    from tradingview_screener import Query
    from tradingview_screener.column import Column
    TRADINGVIEW_SCREENER_AVAILABLE = True
except ImportError:
    TRADINGVIEW_SCREENER_AVAILABLE = False


class IndicatorMap(TypedDict, total=False):
	open: Optional[float]
	close: Optional[float]
	SMA20: Optional[float]
	BB_upper: Optional[float]
	BB_lower: Optional[float]
	EMA50: Optional[float]
	RSI: Optional[float]
	volume: Optional[float]


class Row(TypedDict):
	symbol: str
	changePercent: float
	indicators: IndicatorMap


class MultiRow(TypedDict):
	symbol: str
	changes: dict[str, Optional[float]]
	base_indicators: IndicatorMap


def _map_indicators(raw: Dict[str, Any]) -> IndicatorMap:
	return IndicatorMap(
		open=raw.get("open"),
		close=raw.get("close"),
		SMA20=raw.get("SMA20"),
		BB_upper=raw.get("BB.upper") if "BB.upper" in raw else raw.get("BB_upper"),
		BB_lower=raw.get("BB.lower") if "BB.lower" in raw else raw.get("BB_lower"),
		EMA50=raw.get("EMA50"),
		RSI=raw.get("RSI"),
		volume=raw.get("volume"),
	)


def _percent_change(o: Optional[float], c: Optional[float]) -> Optional[float]:
	try:
		if o in (None, 0) or c is None:
			return None
		return (c - o) / o * 100
	except Exception:
		return None


def _tf_to_tv_resolution(tf: Optional[str]) -> Optional[str]:
	if not tf:
		return None
	return {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1D": "1D", "1W": "1W", "1M": "1M"}.get(tf)


def _fetch_bollinger_analysis(exchange: str, timeframe: str = "4h", limit: int = 50, bbw_filter: float = None) -> List[Row]:
    """Fetch analysis using tradingview_ta with bollinger band logic from the original screener."""
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is missing; run `uv sync`.")
    
    # Load symbols from coinlist files
    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")
    
    # Limit symbols for performance
    symbols = symbols[:limit * 2]  # Get more to filter later
    
    # Get screener type based on exchange
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    
    try:
        analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=symbols)
    except Exception as e:
        raise RuntimeError(f"Analysis failed: {str(e)}")
    
    rows: List[Row] = []
    
    for key, value in analysis.items():
        try:
            if value is None:
                continue
                
            indicators = value.indicators
            metrics = compute_metrics(indicators)
            
            if not metrics or metrics.get('bbw') is None:
                continue
            
            # Apply BBW filter if specified
            if bbw_filter is not None and (metrics['bbw'] >= bbw_filter or metrics['bbw'] <= 0):
                continue
            
            # Check if we have required indicators
            if not (indicators.get("EMA50") and indicators.get("RSI")):
                continue
                
            rows.append(Row(
                symbol=key,
                changePercent=metrics['change'],
                indicators=IndicatorMap(
                    open=metrics.get('open'),
                    close=metrics.get('price'),
                    SMA20=indicators.get("SMA20"),
                    BB_upper=indicators.get("BB.upper"),
                    BB_lower=indicators.get("BB.lower"),
                    EMA50=indicators.get("EMA50"),
                    RSI=indicators.get("RSI"),
                    volume=indicators.get("volume"),
                )
            ))
                
        except (TypeError, ZeroDivisionError, KeyError):
            continue
    
    # Sort by change percentage in descending order (highest gainers first)
    rows.sort(key=lambda x: x["changePercent"], reverse=True)
    
    # Return the requested limit
    return rows[:limit]


def _fetch_trending_analysis(exchange: str, timeframe: str = "5m", filter_type: str = "", rating_filter: int = None, limit: int = 50) -> List[Row]:
    """Fetch trending coins analysis similar to the original app's trending endpoint."""
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is missing; run `uv sync`.")
    
    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")
    
    # Process symbols in batches due to TradingView API limits
    batch_size = 200  # Considering API limitations
    all_coins = []
    
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    
    # Process symbols in batches
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]
        
        try:
            analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=batch_symbols)
        except Exception as e:
            continue  # If this batch fails, move to the next one
            
        # Process coins in this batch
        for key, value in analysis.items():
            try:
                if value is None:
                    continue
                    
                indicators = value.indicators
                metrics = compute_metrics(indicators)
                
                if not metrics or metrics.get('bbw') is None:
                    continue
                
                # Apply rating filter if specified
                if filter_type == "rating" and rating_filter is not None:
                    if metrics['rating'] != rating_filter:
                        continue
                
                all_coins.append(Row(
                    symbol=key,
                    changePercent=metrics['change'],
                    indicators=IndicatorMap(
                        open=metrics.get('open'),
                        close=metrics.get('price'),
                        SMA20=indicators.get("SMA20"),
                        BB_upper=indicators.get("BB.upper"),
                        BB_lower=indicators.get("BB.lower"),
                        EMA50=indicators.get("EMA50"),
                        RSI=indicators.get("RSI"),
                        volume=indicators.get("volume"),
                    )
                ))
                
            except (TypeError, ZeroDivisionError, KeyError):
                continue
    
    # Sort all coins by change percentage
    all_coins.sort(key=lambda x: x["changePercent"], reverse=True)
    
    return all_coins[:limit]
def _fetch_multi_changes(exchange: str, timeframes: List[str] | None, base_timeframe: str = "4h", limit: int | None = None, cookies: Any | None = None) -> List[MultiRow]:
	try:
		from tradingview_screener import Query
		from tradingview_screener.column import Column
	except Exception as e:
		raise RuntimeError("tradingview-screener missing; run `uv sync`.") from e

	tfs = timeframes or ["15m", "1h", "4h", "1D"]
	suffix_map: dict[str, str] = {}
	for tf in tfs:
		s = _tf_to_tv_resolution(tf)
		if s:
			suffix_map[tf] = s
	if not suffix_map:
		suffix_map = {base_timeframe: _tf_to_tv_resolution(base_timeframe) or "240"}

	base_suffix = _tf_to_tv_resolution(base_timeframe) or next(iter(suffix_map.values()))
	cols: list[str] = []
	seen: set[str] = set()
	for tf, s in suffix_map.items():
		for c in (f"open|{s}", f"close|{s}"):
			if c not in seen:
				cols.append(c)
				seen.add(c)
	for c in (f"SMA20|{base_suffix}", f"BB.upper|{base_suffix}", f"BB.lower|{base_suffix}", f"volume|{base_suffix}"):
		if c not in seen:
			cols.append(c)
			seen.add(c)

	market = get_market_type(exchange) if exchange else "crypto"
	q = Query().set_markets(market).select(*cols)
	if exchange:
		q = q.where(Column("exchange") == exchange.upper())
	if limit:
		q = q.limit(int(limit))

	_total, df = q.get_scanner_data(cookies=cookies)
	if df is None or df.empty:
		return []

	out: List[MultiRow] = []
	for _, r in df.iterrows():
		symbol = r.get("ticker")
		changes: dict[str, Optional[float]] = {}
		for tf, s in suffix_map.items():
			o = r.get(f"open|{s}")
			c = r.get(f"close|{s}")
			changes[tf] = _percent_change(o, c)
		base_ind = IndicatorMap(
			open=r.get(f"open|{base_suffix}"),
			close=r.get(f"close|{base_suffix}"),
			SMA20=r.get(f"SMA20|{base_suffix}"),
			BB_upper=r.get(f"BB.upper|{base_suffix}"),
			BB_lower=r.get(f"BB.lower|{base_suffix}"),
			volume=r.get(f"volume|{base_suffix}"),
		)
		out.append(MultiRow(symbol=symbol, changes=changes, base_indicators=base_ind))
	return out


mcp = FastMCP(
	name="TradingView Multi-Market Screener",
	instructions=(
		"Multi-market screener backed by TradingView. "
		"Supports crypto exchanges (KuCoin, Binance, Bybit, etc.) and stock markets (EGX, BIST, NASDAQ, NYSE, Bursa Malaysia, HKEX). "
		"Tools: top_gainers, top_losers, bollinger_scan, coin_analysis, multi_agent_analysis, "
		"volume_breakout_scanner, egx_market_overview, egx_sector_scan, and more."
	),
)


@mcp.tool()
def top_gainers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """Return top gainers for an exchange and timeframe using bollinger band analysis.

    Args:
        exchange: Exchange name - crypto: KUCOIN, BINANCE, BYBIT; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    
    rows = _fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)
    # Convert Row objects to dicts properly
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"], 
        "indicators": dict(row["indicators"])
    } for row in rows]


@mcp.tool()
def top_losers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """Return top losers for an exchange and timeframe. Supports crypto (KUCOIN, BINANCE) and stocks (EGX, BIST, NASDAQ)."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    
    rows = _fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)
    # Reverse sort for losers (lowest change first)
    rows.sort(key=lambda x: x["changePercent"])
    
    # Convert to dict format
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"],
        "indicators": dict(row["indicators"])
    } for row in rows[:limit]]


@mcp.tool()
def bollinger_scan(exchange: str = "KUCOIN", timeframe: str = "4h", bbw_threshold: float = 0.04, limit: int = 50) -> list[dict]:
    """Scan for assets with low Bollinger Band Width (squeeze detection). Works with crypto and stocks (EGX, BIST, etc.).

    Args:
        exchange: Exchange - crypto: KUCOIN, BINANCE, BYBIT; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        bbw_threshold: Maximum BBW value to filter (default 0.04)
        limit: Number of rows to return (max 100)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "4h")
    limit = max(1, min(limit, 100))
    
    rows = _fetch_bollinger_analysis(exchange, timeframe=timeframe, bbw_filter=bbw_threshold, limit=limit)
    # Convert Row objects to dicts
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"],
        "indicators": dict(row["indicators"])
    } for row in rows]


@mcp.tool()
def rating_filter(exchange: str = "KUCOIN", timeframe: str = "5m", rating: int = 2, limit: int = 25) -> list[dict]:
    """Filter coins by Bollinger Band rating.
    
    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        rating: BB rating (-3 to +3): -3=Strong Sell, -2=Sell, -1=Weak Sell, 1=Weak Buy, 2=Buy, 3=Strong Buy
        limit: Number of rows to return (max 50)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "5m")
    rating = max(-3, min(3, rating))
    limit = max(1, min(limit, 50))
    
    rows = _fetch_trending_analysis(exchange, timeframe=timeframe, filter_type="rating", rating_filter=rating, limit=limit)
    # Convert Row objects to dicts
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"],
        "indicators": dict(row["indicators"])
    } for row in rows]

@mcp.tool()
def coin_analysis(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "15m"
) -> dict:
    """Get detailed analysis for a specific asset (coin or stock) on specified exchange and timeframe.

    Args:
        symbol: Symbol - crypto: "BTCUSDT", "ETHUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST)
        exchange: Exchange - crypto: KUCOIN, BINANCE; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)

    Returns:
        Detailed analysis with all indicators and metrics
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "15m")
        
        # Format symbol with exchange prefix
        if ":" not in symbol:
            full_symbol = f"{exchange.upper()}:{symbol.upper()}"
        else:
            full_symbol = symbol.upper()
        
        screener = EXCHANGE_SCREENER.get(exchange, "crypto")
        
        try:
            analysis = get_multiple_analysis(
                screener=screener,
                interval=timeframe,
                symbols=[full_symbol]
            )
            
            if full_symbol not in analysis or analysis[full_symbol] is None:
                return {
                    "error": f"No data found for {symbol} on {exchange}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                }
            
            data = analysis[full_symbol]
            indicators = data.indicators
            
            # Calculate all metrics
            metrics = compute_metrics(indicators)
            if not metrics:
                return {
                    "error": f"Could not compute metrics for {symbol}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                }
            
            # Price levels
            volume = indicators.get("volume", 0)
            high = indicators.get("high", 0)
            low = indicators.get("low", 0)
            open_price = indicators.get("open", 0)
            close_price = indicators.get("close", 0)

            # Extended indicators (RSI, OBV, SMA, EMA, ATR, MACD, Volume, BB, S/R, Structure)
            extended = extract_extended_indicators(indicators)

            # Timeframe-specific context and advice
            tf_context = analyze_timeframe_context(indicators, timeframe)

            # --- Trade Setup for Stock Exchanges ---
            trade_data = {}
            if is_stock_exchange(exchange):
                score_result = compute_stock_score(indicators)
                if score_result:
                    trade_data["stock_score"] = score_result["score"]
                    trade_data["grade"] = score_result["grade"]
                    trade_data["trend_state"] = score_result["trend_state"]

                    setup = compute_trade_setup(indicators)
                    if setup:
                        trade_data["trade_setup"] = {
                            "setup_types": setup["setup_types"],
                            "entry_points": setup["entry_points"],
                            "stop_loss": setup["stop_loss"],
                            "stop_distance_pct": setup["stop_distance_pct"],
                            "targets": setup["targets"],
                            "risk_reward": setup["risk_reward"],
                            "supports": setup["supports"],
                            "resistances": setup["resistances"],
                        }

                        quality = compute_trade_quality(indicators, score_result["score"], setup)
                        if quality:
                            trade_data["trade_quality_score"] = quality["trade_quality_score"]
                            trade_data["trade_quality"] = quality["quality"]
                            trade_data["trade_notes"] = quality["notes"]

            return {
                "symbol": full_symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": "real-time",
                "price_data": {
                    "current_price": metrics['price'],
                    "open": round(open_price, 6) if open_price else None,
                    "high": round(high, 6) if high else None,
                    "low": round(low, 6) if low else None,
                    "close": round(close_price, 6) if close_price else None,
                    "change_percent": metrics['change'],
                    "volume": volume
                },
                "timeframe_context": tf_context,
                "rsi": extended["rsi"],
                "macd": extended["macd"],
                "sma": extended["sma"],
                "ema": extended["ema"],
                "bollinger_bands": extended["bollinger_bands"],
                "atr": extended["atr"],
                "volume_analysis": extended["volume"],
                "obv": extended["obv"],
                "support_resistance": extended["support_resistance"],
                "stochastic": extended["stochastic"],
                "adx": extended["adx"],
                "market_structure": extended["market_structure"],
                **({
                    "vwap": extended["vwap"]
                } if "vwap" in extended else {}),
                "market_sentiment": {
                    "overall_rating": metrics['rating'],
                    "buy_sell_signal": metrics['signal'],
                    "volatility": "High" if metrics['bbw'] and metrics['bbw'] > 0.05 else "Medium" if metrics['bbw'] and metrics['bbw'] > 0.02 else "Low",
                    "momentum": "Bullish" if metrics['change'] > 0 else "Bearish"
                },
                **trade_data,
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
    except Exception as e:
        return {
            "error": f"Coin analysis failed: {str(e)}",
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe
        }

@mcp.tool()
def consecutive_candles_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    pattern_type: str = "bullish",
    candle_count: int = 3,
    min_growth: float = 2.0,
    limit: int = 20
) -> dict:
    """Scan for coins with consecutive growing/shrinking candles pattern.
    
    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h)
        pattern_type: "bullish" (growing candles) or "bearish" (shrinking candles)
        candle_count: Number of consecutive candles to check (2-5)
        min_growth: Minimum growth percentage for each candle
        limit: Maximum number of results to return
    
    Returns:
        List of coins with consecutive candle patterns
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "15m")
        candle_count = max(2, min(5, candle_count))
        min_growth = max(0.5, min(20.0, min_growth))
        limit = max(1, min(50, limit))
        
        # Get symbols for the exchange
        symbols = load_symbols(exchange)
        if not symbols:
            return {
                "error": f"No symbols found for exchange: {exchange}",
                "exchange": exchange,
                "timeframe": timeframe
            }
        
        # Limit symbols for performance (we need historical data)
        symbols = symbols[:min(limit * 3, 200)]
        
        # We need to get data from multiple timeframes to analyze candle progression
        # For now, we'll use current timeframe data and simulate pattern detection
        screener = EXCHANGE_SCREENER.get(exchange, "crypto")
        
        try:
            analysis = get_multiple_analysis(
                screener=screener,
                interval=timeframe,
                symbols=symbols
            )
            
            pattern_coins = []
            
            for symbol, data in analysis.items():
                if data is None:
                    continue
                    
                try:
                    indicators = data.indicators
                    
                    # Calculate current candle metrics
                    open_price = indicators.get("open")
                    close_price = indicators.get("close")
                    high_price = indicators.get("high") 
                    low_price = indicators.get("low")
                    volume = indicators.get("volume", 0)
                    
                    if not all([open_price, close_price, high_price, low_price]):
                        continue
                    
                    # Calculate current candle body size and change
                    current_change = ((close_price - open_price) / open_price) * 100
                    candle_body = abs(close_price - open_price)
                    candle_range = high_price - low_price
                    body_to_range_ratio = candle_body / candle_range if candle_range > 0 else 0
                    
                    # For consecutive pattern, we'll use available indicators to simulate
                    # In a real implementation, we'd need historical OHLC data
                    
                    # Use RSI and price momentum as proxy for consecutive pattern
                    rsi = indicators.get("RSI", 50)
                    sma20 = indicators.get("SMA20", close_price)
                    ema50 = indicators.get("EMA50", close_price)
                    
                    # Calculate momentum indicators
                    price_above_sma = close_price > sma20
                    price_above_ema = close_price > ema50
                    strong_momentum = abs(current_change) >= min_growth
                    
                    # Pattern detection logic
                    pattern_detected = False
                    pattern_strength = 0
                    
                    if pattern_type == "bullish":
                        # Bullish pattern: price rising, good momentum, strong candle body
                        conditions = [
                            current_change > min_growth,  # Current candle is bullish
                            body_to_range_ratio > 0.6,    # Strong candle body
                            price_above_sma,              # Above short MA
                            rsi > 45 and rsi < 80,        # RSI in momentum range
                            volume > 1000                 # Decent volume
                        ]
                        
                        pattern_strength = sum(conditions)
                        pattern_detected = pattern_strength >= 3
                        
                    elif pattern_type == "bearish":
                        # Bearish pattern: price falling, bearish momentum
                        conditions = [
                            current_change < -min_growth,  # Current candle is bearish
                            body_to_range_ratio > 0.6,     # Strong candle body
                            not price_above_sma,           # Below short MA
                            rsi < 55 and rsi > 20,         # RSI in bearish range
                            volume > 1000                  # Decent volume
                        ]
                        
                        pattern_strength = sum(conditions)
                        pattern_detected = pattern_strength >= 3
                    
                    if pattern_detected:
                        # Calculate additional metrics
                        metrics = compute_metrics(indicators)
                        
                        coin_data = {
                            "symbol": symbol,
                            "price": round(close_price, 6),
                            "current_change": round(current_change, 3),
                            "candle_body_ratio": round(body_to_range_ratio, 3),
                            "pattern_strength": pattern_strength,
                            "volume": volume,
                            "bollinger_rating": metrics.get('rating', 0) if metrics else 0,
                            "rsi": round(rsi, 2),
                            "price_levels": {
                                "open": round(open_price, 6),
                                "high": round(high_price, 6), 
                                "low": round(low_price, 6),
                                "close": round(close_price, 6)
                            },
                            "momentum_signals": {
                                "above_sma20": price_above_sma,
                                "above_ema50": price_above_ema,
                                "strong_volume": volume > 5000
                            }
                        }
                        
                        pattern_coins.append(coin_data)
                        
                except Exception as e:
                    continue
            
            # Sort by pattern strength and current change
            if pattern_type == "bullish":
                pattern_coins.sort(key=lambda x: (x['pattern_strength'], x['current_change']), reverse=True)
            else:
                pattern_coins.sort(key=lambda x: (x['pattern_strength'], -x['current_change']), reverse=True)
            
            return {
                "exchange": exchange,
                "timeframe": timeframe,
                "pattern_type": pattern_type,
                "candle_count": candle_count,
                "min_growth": min_growth,
                "total_found": len(pattern_coins),
                "data": pattern_coins[:limit]
            }
            
        except Exception as e:
            return {
                "error": f"Pattern analysis failed: {str(e)}",
                "exchange": exchange,
                "timeframe": timeframe
            }
            
    except Exception as e:
        return {
            "error": f"Consecutive candles scan failed: {str(e)}",
            "exchange": exchange,
            "timeframe": timeframe
        }

@mcp.tool()
def advanced_candle_pattern(
    exchange: str = "KUCOIN",
    base_timeframe: str = "15m",
    pattern_length: int = 3,
    min_size_increase: float = 10.0,
    limit: int = 15
) -> dict:
    """Advanced candle pattern analysis using multi-timeframe data.
    
    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        base_timeframe: Base timeframe for analysis (5m, 15m, 1h, 4h)
        pattern_length: Number of consecutive periods to analyze (2-4)
        min_size_increase: Minimum percentage increase in candle size
        limit: Maximum number of results to return
    
    Returns:
        Coins with progressive candle size increase patterns
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        base_timeframe = sanitize_timeframe(base_timeframe, "15m")
        pattern_length = max(2, min(4, pattern_length))
        min_size_increase = max(5.0, min(50.0, min_size_increase))
        limit = max(1, min(30, limit))
        
        # Get symbols
        symbols = load_symbols(exchange)
        if not symbols:
            return {
                "error": f"No symbols found for exchange: {exchange}",
                "exchange": exchange
            }
        
        # Limit for performance
        symbols = symbols[:min(limit * 2, 100)]
        
        # Use tradingview-screener for multi-timeframe data if available
        if TRADINGVIEW_SCREENER_AVAILABLE:
            try:
                # Get multiple timeframe data using screener
                results = _fetch_multi_timeframe_patterns(
                    exchange, symbols, base_timeframe, pattern_length, min_size_increase
                )
                
                return {
                    "exchange": exchange,
                    "base_timeframe": base_timeframe,
                    "pattern_length": pattern_length,
                    "min_size_increase": min_size_increase,
                    "method": "multi-timeframe",
                    "total_found": len(results),
                    "data": results[:limit]
                }
                
            except Exception as e:
                # Fallback to single timeframe analysis
                pass
        
        # Fallback: Use single timeframe with enhanced pattern detection
        screener = EXCHANGE_SCREENER.get(exchange, "crypto")
        
        analysis = get_multiple_analysis(
            screener=screener,
            interval=base_timeframe,
            symbols=symbols
        )
        
        pattern_results = []
        
        for symbol, data in analysis.items():
            if data is None:
                continue
                
            try:
                indicators = data.indicators
                
                # Enhanced pattern detection using available indicators
                pattern_score = _calculate_candle_pattern_score(
                    indicators, pattern_length, min_size_increase
                )
                
                if pattern_score['detected']:
                    metrics = compute_metrics(indicators)
                    
                    result = {
                        "symbol": symbol,
                        "pattern_score": pattern_score['score'],
                        "pattern_details": pattern_score['details'],
                        "current_price": pattern_score['price'],
                        "total_change": pattern_score['total_change'],
                        "volume": indicators.get("volume", 0),
                        "bollinger_rating": metrics.get('rating', 0) if metrics else 0,
                        "technical_strength": {
                            "rsi": round(indicators.get("RSI", 50), 2),
                            "momentum": "Strong" if abs(pattern_score['total_change']) > min_size_increase else "Moderate",
                            "volume_trend": "High" if indicators.get("volume", 0) > 10000 else "Low"
                        }
                    }
                    
                    pattern_results.append(result)
                    
            except Exception as e:
                continue
        
        # Sort by pattern score and total change
        pattern_results.sort(key=lambda x: (x['pattern_score'], abs(x['total_change'])), reverse=True)
        
        return {
            "exchange": exchange,
            "base_timeframe": base_timeframe,
            "pattern_length": pattern_length,
            "min_size_increase": min_size_increase,
            "method": "enhanced-single-timeframe",
            "total_found": len(pattern_results),
            "data": pattern_results[:limit]
        }
        
    except Exception as e:
        return {
            "error": f"Advanced pattern analysis failed: {str(e)}",
            "exchange": exchange,
            "base_timeframe": base_timeframe
        }

def _calculate_candle_pattern_score(indicators: dict, pattern_length: int, min_increase: float) -> dict:
    """Calculate candle pattern score based on available indicators."""
    try:
        open_price = indicators.get("open", 0)
        close_price = indicators.get("close", 0)
        high_price = indicators.get("high", 0)
        low_price = indicators.get("low", 0)
        volume = indicators.get("volume", 0)
        rsi = indicators.get("RSI", 50)
        
        if not all([open_price, close_price, high_price, low_price]):
            return {"detected": False, "score": 0}
        
        # Current candle analysis
        candle_body = abs(close_price - open_price)
        candle_range = high_price - low_price
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        # Price change
        price_change = ((close_price - open_price) / open_price) * 100
        
        # Pattern scoring
        score = 0
        details = []
        
        # Strong candle body
        if body_ratio > 0.7:
            score += 2
            details.append("Strong candle body")
        elif body_ratio > 0.5:
            score += 1
            details.append("Moderate candle body")
        
        # Significant price movement
        if abs(price_change) >= min_increase:
            score += 2
            details.append(f"Strong momentum ({price_change:.1f}%)")
        elif abs(price_change) >= min_increase / 2:
            score += 1
            details.append(f"Moderate momentum ({price_change:.1f}%)")
        
        # Volume confirmation
        if volume > 5000:
            score += 1
            details.append("Good volume")
        
        # RSI momentum
        if (price_change > 0 and 50 < rsi < 80) or (price_change < 0 and 20 < rsi < 50):
            score += 1
            details.append("RSI momentum aligned")
        
        # Trend consistency (using EMA vs price)
        ema50 = indicators.get("EMA50", close_price)
        if (price_change > 0 and close_price > ema50) or (price_change < 0 and close_price < ema50):
            score += 1
            details.append("Trend alignment")
        
        detected = score >= 3  # Minimum threshold
        
        return {
            "detected": detected,
            "score": score,
            "details": details,
            "price": round(close_price, 6),
            "total_change": round(price_change, 3),
            "body_ratio": round(body_ratio, 3),
            "volume": volume
        }
        
    except Exception as e:
        return {"detected": False, "score": 0, "error": str(e)}

def _fetch_multi_timeframe_patterns(exchange: str, symbols: List[str], base_tf: str, length: int, min_increase: float) -> List[dict]:
    """Fetch multi-timeframe pattern data using tradingview-screener."""
    try:
        from tradingview_screener import Query
        from tradingview_screener.column import Column
        
        # Map timeframe to TradingView format
        tf_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1D": "1D"}
        tv_interval = tf_map.get(base_tf, "15")
        
        # Create query for OHLC data
        cols = [
            f"open|{tv_interval}",
            f"close|{tv_interval}", 
            f"high|{tv_interval}",
            f"low|{tv_interval}",
            f"volume|{tv_interval}",
            "RSI"
        ]
        
        market = get_market_type(exchange)
        q = Query().set_markets(market).select(*cols)
        q = q.where(Column("exchange") == exchange.upper())
        q = q.limit(len(symbols))
        
        total, df = q.get_scanner_data()
        
        if df is None or df.empty:
            return []
        
        results = []
        
        for _, row in df.iterrows():
            symbol = row.get("ticker", "")
            
            try:
                open_val = row.get(f"open|{tv_interval}")
                close_val = row.get(f"close|{tv_interval}")
                high_val = row.get(f"high|{tv_interval}")
                low_val = row.get(f"low|{tv_interval}")
                volume_val = row.get(f"volume|{tv_interval}", 0)
                rsi_val = row.get("RSI", 50)
                
                if not all([open_val, close_val, high_val, low_val]):
                    continue
                
                # Calculate pattern metrics
                pattern_score = _calculate_candle_pattern_score({
                    "open": open_val,
                    "close": close_val,
                    "high": high_val,
                    "low": low_val,
                    "volume": volume_val,
                    "RSI": rsi_val
                }, length, min_increase)
                
                if pattern_score['detected']:
                    results.append({
                        "symbol": symbol,
                        "pattern_score": pattern_score['score'],
                        "price": pattern_score['price'],
                        "change": pattern_score['total_change'],
                        "body_ratio": pattern_score['body_ratio'],
                        "volume": volume_val,
                        "rsi": round(rsi_val, 2),
                        "details": pattern_score['details']
                    })
                    
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x['pattern_score'], reverse=True)
        
    except Exception as e:
        return []

@mcp.resource("exchanges://list")
def exchanges_list() -> str:
    """List available exchanges from coinlist directory."""
    try:
        import os
        # Get the directory where this module is located
        current_dir = os.path.dirname(__file__)
        coinlist_dir = os.path.join(current_dir, "coinlist")
        
        if os.path.exists(coinlist_dir):
            exchanges = []
            for filename in os.listdir(coinlist_dir):
                if filename.endswith('.txt'):
                    exchange_name = filename[:-4].upper()
                    exchanges.append(exchange_name)
            
            if exchanges:
                return f"Available exchanges: {', '.join(sorted(exchanges))}"
        
        # Fallback to static list
        return "Common exchanges: KUCOIN, BINANCE, BYBIT, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX, KRAKEN, BITSTAMP, BIST, EGX, NASDAQ"
    except Exception:
        return "Common exchanges: KUCOIN, BINANCE, BYBIT, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX, KRAKEN, BITSTAMP, BIST, EGX, NASDAQ"
def main() -> None:
	parser = argparse.ArgumentParser(description="TradingView Screener MCP server")
	parser.add_argument("transport", choices=["stdio", "streamable-http"], default="stdio", nargs="?", help="Transport (default stdio)")
	parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
	parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
	args = parser.parse_args()

	if os.environ.get("DEBUG_MCP"):
		import sys
		print(f"[DEBUG_MCP] pkg cwd={os.getcwd()} argv={sys.argv} file={__file__}", file=sys.stderr, flush=True)

	if args.transport == "stdio":
		mcp.run()
	else:
		try:
			mcp.settings.host = args.host
			mcp.settings.port = args.port
		except Exception:
			pass
		mcp.run(transport="streamable-http")


@mcp.tool()
def volume_breakout_scanner(exchange: str = "KUCOIN", timeframe: str = "15m", volume_multiplier: float = 2.0, price_change_min: float = 3.0, limit: int = 25) -> list[dict]:
	"""Detect coins with volume breakout + price breakout.
	
	Args:
		exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
		timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
		volume_multiplier: How many times the volume should be above normal level (default 2.0)
		price_change_min: Minimum price change percentage (default 3.0)
		limit: Number of rows to return (max 50)
	"""
	exchange = sanitize_exchange(exchange, "KUCOIN")
	timeframe = sanitize_timeframe(timeframe, "15m")
	volume_multiplier = max(1.5, min(10.0, volume_multiplier))
	price_change_min = max(1.0, min(20.0, price_change_min))
	limit = max(1, min(limit, 50))
	
	# Get symbols
	symbols = load_symbols(exchange)
	if not symbols:
		return []
	
	screener = EXCHANGE_SCREENER.get(exchange, "crypto")
	volume_breakouts = []
	
	# Process in batches
	batch_size = 100
	for i in range(0, min(len(symbols), 500), batch_size):  # Limit to 500 symbols for performance
		batch_symbols = symbols[i:i + batch_size]
		
		try:
			analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=batch_symbols)
		except Exception:
			continue
			
		for symbol, data in analysis.items():
			try:
				if not data or not hasattr(data, 'indicators'):
					continue
					
				indicators = data.indicators
				
				# Get required data
				volume = indicators.get('volume', 0)
				close = indicators.get('close', 0)
				open_price = indicators.get('open', 0)
				sma20_volume = indicators.get('volume.SMA20', 0)  # 20-period volume average
				
				if not all([volume, close, open_price]) or volume <= 0:
					continue
				
				# Calculate price change %
				price_change = ((close - open_price) / open_price) * 100 if open_price > 0 else 0
				
				# Volume ratio calculation
				# If SMA20 volume not available, use a simple heuristic
				if sma20_volume and sma20_volume > 0:
					volume_ratio = volume / sma20_volume
				else:
					# Estimate average volume as current volume / 2 (conservative)
					avg_volume_estimate = volume / 2
					volume_ratio = volume / avg_volume_estimate if avg_volume_estimate > 0 else 1
				
				# Check conditions
				if (abs(price_change) >= price_change_min and 
					volume_ratio >= volume_multiplier):
					
					# Get additional indicators
					rsi = indicators.get('RSI', 50)
					bb_upper = indicators.get('BB.upper', 0)
					bb_lower = indicators.get('BB.lower', 0)
					
					# Volume strength score
					volume_strength = min(10, volume_ratio)  # Cap at 10x
					
					volume_breakouts.append({
						"symbol": symbol,
						"changePercent": price_change,
						"volume_ratio": round(volume_ratio, 2),
						"volume_strength": round(volume_strength, 1),
						"current_volume": volume,
						"breakout_type": "bullish" if price_change > 0 else "bearish",
						"indicators": {
							"close": close,
							"RSI": rsi,
							"BB_upper": bb_upper,
							"BB_lower": bb_lower,
							"volume": volume
						}
					})
					
			except Exception:
				continue
	
	# Sort by volume strength first, then by price change
	volume_breakouts.sort(key=lambda x: (x["volume_strength"], abs(x["changePercent"])), reverse=True)
	
	return volume_breakouts[:limit]


@mcp.tool()
def volume_confirmation_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
	"""Detailed volume confirmation analysis for a specific coin.
	
	Args:
		symbol: Coin symbol (e.g., BTCUSDT)
		exchange: Exchange name
		timeframe: Time frame for analysis
	"""
	exchange = sanitize_exchange(exchange, "KUCOIN")
	timeframe = sanitize_timeframe(timeframe, "15m")

	# Only append USDT for crypto exchanges, not stock markets
	if not is_stock_exchange(exchange) and not symbol.upper().endswith('USDT'):
		symbol = symbol.upper() + 'USDT'
	else:
		symbol = symbol.upper()

	# Format with exchange prefix for stock markets
	if is_stock_exchange(exchange) and ":" not in symbol:
		full_symbol = f"{exchange.upper()}:{symbol}"
	else:
		full_symbol = symbol

	screener = EXCHANGE_SCREENER.get(exchange, "crypto")

	try:
		analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=[full_symbol])

		if not analysis or full_symbol not in analysis:
			return {"error": f"No data found for {full_symbol}"}
			
		data = analysis[full_symbol]
		if not data or not hasattr(data, 'indicators'):
			return {"error": f"No indicator data for {full_symbol}"}
			
		indicators = data.indicators
		
		# Get volume data
		volume = indicators.get('volume', 0)
		close = indicators.get('close', 0)
		open_price = indicators.get('open', 0)
		high = indicators.get('high', 0)
		low = indicators.get('low', 0)
		
		# Calculate price metrics
		price_change = ((close - open_price) / open_price) * 100 if open_price > 0 else 0
		candle_range = ((high - low) / low) * 100 if low > 0 else 0
		
		# Volume analysis
		sma20_volume = indicators.get('volume.SMA20', 0)
		volume_ratio = volume / sma20_volume if sma20_volume > 0 else 1
		
		# Technical indicators
		rsi = indicators.get('RSI', 50)
		bb_upper = indicators.get('BB.upper', 0)
		bb_lower = indicators.get('BB.lower', 0)
		bb_middle = (bb_upper + bb_lower) / 2 if bb_upper and bb_lower else close
		
		# Volume confirmation signals
		signals = []
		
		# Strong volume + price breakout
		if volume_ratio >= 2.0 and abs(price_change) >= 3.0:
			signals.append(f"🚀 STRONG BREAKOUT: {volume_ratio:.1f}x volume + {price_change:.1f}% price")
		
		# Volume divergence
		if volume_ratio >= 1.5 and abs(price_change) < 1.0:
			signals.append(f"⚠️ VOLUME DIVERGENCE: High volume ({volume_ratio:.1f}x) but low price movement")
		
		# Low volume on price move (weak signal)
		if abs(price_change) >= 2.0 and volume_ratio < 0.8:
			signals.append(f"❌ WEAK SIGNAL: Price moved but volume is low ({volume_ratio:.1f}x)")
		
		# Bollinger Band + Volume confirmation
		if close > bb_upper and volume_ratio >= 1.5:
			signals.append(f"💥 BB BREAKOUT CONFIRMED: Upper band breakout + volume confirmation")
		elif close < bb_lower and volume_ratio >= 1.5:
			signals.append(f"📉 BB SELL CONFIRMED: Lower band breakout + volume confirmation")
		
		# RSI + Volume analysis
		if rsi > 70 and volume_ratio >= 2.0:
			signals.append(f"🔥 OVERBOUGHT + VOLUME: RSI {rsi:.1f} + {volume_ratio:.1f}x volume")
		elif rsi < 30 and volume_ratio >= 2.0:
			signals.append(f"🛒 OVERSOLD + VOLUME: RSI {rsi:.1f} + {volume_ratio:.1f}x volume")
		
		# Overall assessment
		if volume_ratio >= 3.0:
			volume_strength = "VERY STRONG"
		elif volume_ratio >= 2.0:
			volume_strength = "STRONG"
		elif volume_ratio >= 1.5:
			volume_strength = "MEDIUM"
		elif volume_ratio >= 1.0:
			volume_strength = "NORMAL"
		else:
			volume_strength = "WEAK"
		
		return {
			"symbol": symbol,
			"price_data": {
				"close": close,
				"change_percent": round(price_change, 2),
				"candle_range_percent": round(candle_range, 2)
			},
			"volume_analysis": {
				"current_volume": volume,
				"volume_ratio": round(volume_ratio, 2),
				"volume_strength": volume_strength,
				"average_volume": sma20_volume
			},
			"technical_indicators": {
				"RSI": round(rsi, 1),
				"BB_position": "ABOVE" if close > bb_upper else "BELOW" if close < bb_lower else "WITHIN",
				"BB_upper": bb_upper,
				"BB_lower": bb_lower
			},
			"signals": signals,
			"overall_assessment": {
				"bullish_signals": len([s for s in signals if "🚀" in s or "💥" in s or "🛒" in s]),
				"bearish_signals": len([s for s in signals if "📉" in s or "❌" in s]),
				"warning_signals": len([s for s in signals if "⚠️" in s])
			}
		}
		
	except Exception as e:
		return {"error": f"Analysis failed: {str(e)}"}


@mcp.tool()
def smart_volume_scanner(exchange: str = "KUCOIN", min_volume_ratio: float = 2.0, min_price_change: float = 2.0, rsi_range: str = "any", limit: int = 20) -> list[dict]:
	"""Smart volume + technical analysis combination scanner.
	
	Args:
		exchange: Exchange name
		min_volume_ratio: Minimum volume multiplier (default 2.0)
		min_price_change: Minimum price change percentage (default 2.0)
		rsi_range: "oversold" (<30), "overbought" (>70), "neutral" (30-70), "any"
		limit: Number of results (max 30)
	"""
	exchange = sanitize_exchange(exchange, "KUCOIN")
	min_volume_ratio = max(1.2, min(10.0, min_volume_ratio))
	min_price_change = max(0.5, min(20.0, min_price_change))
	limit = max(1, min(limit, 30))
	
	# Get volume breakouts first
	volume_breakouts = volume_breakout_scanner(
		exchange=exchange, 
		volume_multiplier=min_volume_ratio,
		price_change_min=min_price_change,
		limit=limit * 2  # Get more to filter
	)
	
	if not volume_breakouts:
		return []
	
	# Apply RSI filter
	filtered_results = []
	for coin in volume_breakouts:
		rsi = coin["indicators"].get("RSI", 50)
		
		if rsi_range == "oversold" and rsi >= 30:
			continue
		elif rsi_range == "overbought" and rsi <= 70:
			continue
		elif rsi_range == "neutral" and (rsi <= 30 or rsi >= 70):
			continue
		# "any" passes all
		
		# Add trading recommendation
		recommendation = ""
		if coin["changePercent"] > 0 and coin["volume_ratio"] >= 2.0:
			if rsi < 70:
				recommendation = "🚀 STRONG BUY"
			else:
				recommendation = "⚠️ OVERBOUGHT - CAUTION"
		elif coin["changePercent"] < 0 and coin["volume_ratio"] >= 2.0:
			if rsi > 30:
				recommendation = "📉 STRONG SELL"
			else:
				recommendation = "🛒 OVERSOLD - OPPORTUNITY?"
		
		coin["trading_recommendation"] = recommendation
		filtered_results.append(coin)
	
	return filtered_results[:limit]


def _calculate_sentiment_score(indicators: dict, price_change: float) -> dict:
    """A basic heuristic for sentiment based on volume and price momentum."""
    rsi = indicators.get("RSI", 50.0)
    macd = indicators.get("MACD.macd", 0.0)
    macd_signal = indicators.get("MACD.signal", 0.0)
    
    score = 0
    signals = []
    
    if price_change > 0:
        score += 1
        signals.append("Positive price momentum")
    elif price_change < 0:
        score -= 1
        signals.append("Negative price momentum")
        
    if rsi > 60:
        score += 1
        signals.append("Bullish RSI (>60)")
    elif rsi < 40:
        score -= 1
        signals.append("Bearish RSI (<40)")
        
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            score += 1
            signals.append("MACD bullish crossover")
        elif macd < macd_signal:
            score -= 1
            signals.append("MACD bearish crossover")
        
    return {
        "score": score,
        "normalized": max(-3, min(3, score)),
        "signals": signals
    }

def _calculate_risk_score(indicators: dict, bbw: float) -> dict:
    """Risk assessment based on volatility and moving averages."""
    close = indicators.get("close", 0.0)
    sma20 = indicators.get("SMA20", close)
    ema200 = indicators.get("EMA200", close)
    
    score = 0
    warnings = []
    
    if bbw > 0.1:
        score -= 2
        warnings.append("High volatility (Wide BBW > 0.1)")
    elif bbw < 0.03:
        score += 1
        warnings.append("Low volatility (Squeeze)")
        
    if ema200 is not None and close < ema200:
        score -= 1
        warnings.append("Price below 200 EMA (Long-term bearish structure)")
    
    # Distance from SMA20 (reversion risk)
    if sma20 and sma20 > 0:
        dist = abs(close - sma20) / sma20
        if dist > 0.05:
            score -= 1
            direction = "above" if close > sma20 else "below"
            warnings.append(f"Extended from 20 SMA (5%+ {direction} mean)")
            
    return {
        "score": score,
        "warnings": warnings if warnings else ["Normal risk parameters"],
        "level": "High" if score < -1 else "Medium" if score == -1 else "Low"
    }


@mcp.tool()
def multi_agent_analysis(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "15m"
) -> dict:
    """Run a multi-agent debate (Technical, Sentiment, Risk) for a specific symbol.

    Args:
        symbol: Symbol - crypto: "BTCUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST)
        exchange: Exchange - crypto: KUCOIN, BINANCE; stocks: EGX, BIST, NASDAQ, NYSE
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W)
    
    Returns:
        A structured debate between 3 AI agents culminating in a final trading decision.
    """
    try:
        exchange_sanitized = sanitize_exchange(exchange, "KUCOIN")
        timeframe_sanitized = sanitize_timeframe(timeframe, "15m")
        
        if ":" not in symbol:
            full_symbol = f"{exchange_sanitized.upper()}:{symbol.upper()}"
        else:
            full_symbol = symbol.upper()
            
        screener = EXCHANGE_SCREENER.get(exchange_sanitized, "crypto")
        
        analysis = get_multiple_analysis(
            screener=screener,
            interval=timeframe_sanitized,
            symbols=[full_symbol]
        )
        
        if full_symbol not in analysis or analysis[full_symbol] is None:
             return {"error": f"No data found for {full_symbol}"}
             
        indicators = analysis[full_symbol].indicators
        metrics = compute_metrics(indicators)
        if not metrics:
             return {"error": f"Could not compute metrics for {full_symbol}"}
        
        price = metrics.get('price', 0.0)
        change = metrics.get('change', 0.0)
        bb_rating = metrics.get('rating', 0)
        bbw = metrics.get('bbw', 0.0)
        
        # --- AGENT 1: TECHNICAL ANALYST ---
        tech_analyst = {
            "role": "Technical Analyst",
            "stance": "Bullish" if bb_rating > 0 else "Bearish" if bb_rating < 0 else "Neutral",
            "score": bb_rating, # -3 to +3
            "key_observations": [
                f"Price is {price} ({change:+.2f}%)",
                f"Bollinger Rating: {bb_rating} ({metrics.get('signal', 'Neutral')})",
                f"RSI: {indicators.get('RSI', 50):.1f}"
            ]
        }
        
        # --- AGENT 2: SENTIMENT ANALYST ---
        sentiment_data = _calculate_sentiment_score(indicators, change)
        sentiment_analyst = {
            "role": "Sentiment & Momentum Analyst",
            "stance": "Bullish" if sentiment_data["normalized"] > 0 else "Bearish" if sentiment_data["normalized"] < 0 else "Neutral",
            "score": sentiment_data["normalized"],
            "key_observations": sentiment_data["signals"]
        }
        
        # --- AGENT 3: RISK MANAGER ---
        risk_data = _calculate_risk_score(indicators, bbw)
        risk_manager = {
            "role": "Risk Manager",
            "risk_level": risk_data["level"],
            "risk_score": risk_data["score"],
            "warnings": risk_data["warnings"]
        }
        
        # --- THE DEBATE & FINAL DECISION ---
        total_score = tech_analyst["score"] + sentiment_analyst["score"] + risk_manager["risk_score"]
        
        if total_score >= 3 and risk_manager["risk_level"] != "High":
            final_decision = "STRONG BUY"
            confidence = "High"
        elif total_score > 0:
            final_decision = "BUY"
            confidence = "Medium"
        elif total_score <= -3:
            final_decision = "STRONG SELL"
            confidence = "High"
        elif total_score < 0:
            final_decision = "SELL"
            confidence = "Medium"
        else:
            final_decision = "HOLD"
            confidence = "Low"
            
        return {
            "framework_name": "TradingAgents-MCP Pipeline",
            "target": full_symbol,
            "timeframe": timeframe_sanitized,
            "agents_debate": {
                "technical_analyst": tech_analyst,
                "sentiment_analyst": sentiment_analyst,
                "risk_manager": risk_manager
            },
            "consensus": {
                "decision": final_decision,
                "confidence": confidence,
                "net_score": total_score,
                "summary": f"Technical score: {tech_analyst['score']}, Sentiment score: {sentiment_analyst['score']}, Risk adjustment: {risk_manager['risk_score']}"
            }
        }
        
    except Exception as e:
        return {"error": f"Multi-agent analysis failed: {str(e)}"}


@mcp.tool()
def egx_market_overview(timeframe: str = "1D", limit: int = 10) -> dict:
    """Get a comprehensive overview of the Egyptian Exchange (EGX) market.

    Shows top gainers, top losers, and most active stocks on EGX.

    Args:
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D for stocks)
        limit: Number of stocks per category (max 20)
    """
    if not TRADINGVIEW_TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 20))

    symbols = load_symbols("egx")
    if not symbols:
        return {"error": "No EGX symbols found. Check coinlist/egx.txt"}

    screener = EXCHANGE_SCREENER.get("egx", "egypt")

    all_stocks = []
    batch_size = 200
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=batch)
        except Exception:
            continue

        for sym, data in analysis.items():
            if data is None:
                continue
            try:
                indicators = data.indicators
                metrics = compute_metrics(indicators)
                if not metrics:
                    continue

                volume = indicators.get("volume", 0)
                all_stocks.append({
                    "symbol": sym,
                    "price": metrics.get("price", 0),
                    "changePercent": metrics.get("change", 0),
                    "volume": volume,
                    "rsi": round(indicators.get("RSI", 0) or 0, 2),
                    "bbw": metrics.get("bbw", 0),
                    "rating": metrics.get("rating", 0),
                    "signal": metrics.get("signal", "N/A"),
                })
            except Exception:
                continue

    if not all_stocks:
        return {"error": "No data returned for EGX stocks", "timeframe": timeframe}

    # Sort for different views
    by_change = sorted(all_stocks, key=lambda x: x["changePercent"], reverse=True)
    by_volume = sorted(all_stocks, key=lambda x: x["volume"] or 0, reverse=True)

    return {
        "exchange": "EGX",
        "timeframe": timeframe,
        "total_analyzed": len(all_stocks),
        "top_gainers": by_change[:limit],
        "top_losers": by_change[-limit:][::-1],
        "most_active": by_volume[:limit],
        "market_stats": {
            "advancing": len([s for s in all_stocks if s["changePercent"] > 0]),
            "declining": len([s for s in all_stocks if s["changePercent"] < 0]),
            "unchanged": len([s for s in all_stocks if s["changePercent"] == 0]),
            "avg_change": round(sum(s["changePercent"] for s in all_stocks) / len(all_stocks), 2) if all_stocks else 0,
        }
    }


@mcp.tool()
def egx_sector_scan(
    sector: str = "",
    timeframe: str = "1D",
    limit: int = 20
) -> dict:
    """Scan EGX stocks by sector. Shows available sectors if none specified.

    Args:
        sector: Sector name - banks, basic_resources, healthcare_and_pharma,
                industrial_goods_and_services, real_estate, travel_and_leisure, utilities,
                it_media_and_communication, food_beverages_and_tobacco,
                energy_and_support_services, trade_and_distributors,
                shipping_and_transportation, education_services,
                non_bank_financial_services, contracting_and_construction,
                textiles_and_durables, building_materials, paper_and_packaging.
                Leave empty to list all sectors.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Max results per sector (max 50)
    """
    from tradingview_mcp.core.data.egx_sectors import get_all_sectors, get_symbols_by_sector, get_sector

    if not sector:
        sectors = get_all_sectors()
        return {
            "available_sectors": sectors,
            "usage": "Pass a sector name to scan. Example: sector='banks'"
        }

    if not TRADINGVIEW_TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 50))

    sector_key = sector.strip().lower().replace(" ", "_")
    symbols = get_symbols_by_sector(sector_key)

    if not symbols:
        return {
            "error": f"Unknown sector: {sector}",
            "available_sectors": get_all_sectors()
        }

    screener = EXCHANGE_SCREENER.get("egx", "egypt")

    try:
        analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=symbols)
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

    results = []
    for sym, data in analysis.items():
        if data is None:
            continue
        try:
            indicators = data.indicators
            metrics = compute_metrics(indicators)
            if not metrics:
                continue

            results.append({
                "symbol": sym,
                "sector": get_sector(sym),
                "price": metrics.get("price", 0),
                "changePercent": metrics.get("change", 0),
                "volume": indicators.get("volume", 0),
                "rsi": round(indicators.get("RSI", 0) or 0, 2),
                "bbw": metrics.get("bbw", 0),
                "rating": metrics.get("rating", 0),
                "signal": metrics.get("signal", "N/A"),
                "bb_upper": round(indicators.get("BB.upper", 0) or 0, 4),
                "bb_lower": round(indicators.get("BB.lower", 0) or 0, 4),
                "sma20": round(indicators.get("SMA20", 0) or 0, 4),
                "ema50": round(indicators.get("EMA50", 0) or 0, 4),
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["changePercent"], reverse=True)

    sector_change = [r["changePercent"] for r in results if r["changePercent"] is not None]
    avg_change = round(sum(sector_change) / len(sector_change), 2) if sector_change else 0

    return {
        "exchange": "EGX",
        "sector": sector_key,
        "timeframe": timeframe,
        "total_stocks": len(results),
        "sector_avg_change": avg_change,
        "sector_sentiment": "Bullish" if avg_change > 0.5 else "Bearish" if avg_change < -0.5 else "Neutral",
        "data": results[:limit],
    }


# ── Sector scanner helpers ───────────────────────────────────────────────────

def _compute_sector_momentum_score(
    avg_change: float,
    avg_rsi: float,
    breadth_pct: float,
    volume_flow_positive: bool,
    change_rank_pct: float,
) -> int:
    """Compute a 0-100 sector momentum score.

    Components:
        - Change rank among sectors (0-30 pts)
        - RSI in optimal zone 50-70 (0-25 pts)
        - Breadth percentage (0-25 pts)
        - Volume flow direction (0-20 pts)
    """
    # Change rank: 0-30 pts based on percentile rank among sectors
    change_pts = round(change_rank_pct * 30)

    # RSI zone: 50-70 is optimal (25 pts), degrades outside
    if 50 <= avg_rsi <= 70:
        rsi_pts = 25
    elif 40 <= avg_rsi < 50 or 70 < avg_rsi <= 80:
        rsi_pts = 15
    elif 30 <= avg_rsi < 40:
        rsi_pts = 10  # Oversold — potential recovery
    elif avg_rsi > 80:
        rsi_pts = 5   # Overbought — risk of pullback
    else:
        rsi_pts = 8   # Deep oversold

    # Breadth: 0-25 pts linearly
    breadth_pts = round(min(breadth_pct, 100) / 100 * 25)

    # Volume flow: binary 0 or 20
    volume_pts = 20 if volume_flow_positive else 0

    return max(0, min(100, change_pts + rsi_pts + breadth_pts + volume_pts))


def _generate_rotation_signals(ranked_sectors: list) -> List[str]:
    """Generate human-readable sector rotation signals from ranked heatmap."""
    signals = []
    for s in ranked_sectors:
        if s["status"] == "Hot":
            signals.append(
                f"Money rotating INTO {s['display_name']} "
                f"(Hot, {s['avg_change_pct']:+.2f}% avg, "
                f"{s['volume_flow']['signal'].lower()}, "
                f"weight {s['market_cap_weight']}%)"
            )
        elif s["status"] == "Cold":
            signals.append(
                f"Money rotating OUT OF {s['display_name']} "
                f"(Cold, {s['avg_change_pct']:+.2f}% avg, "
                f"{s['volume_flow']['signal'].lower()}, "
                f"weight {s['market_cap_weight']}%)"
            )
    return signals


@mcp.tool()
def egx_sector_scanner(
    timeframe: str = "1D",
    top_n_sectors: int = 5,
    top_n_stocks: int = 3,
    min_stock_score: int = 60,
) -> dict:
    """Sector rotation scanner for EGX — identifies hot/cold sectors and top picks.

    Scans all 18 EGX sectors, ranks them by momentum, and surfaces the best
    stocks in the top sectors with full trade setups.

    Args:
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
        top_n_sectors: Number of top sectors to show stock picks for (1-18, default 5)
        top_n_stocks: Number of top stocks per highlighted sector (1-10, default 3)
        min_stock_score: Minimum stock score for picks (0-100, default 60)

    Returns:
        Weighted market view, sector heatmap (all 18), top picks per sector,
        and rotation signals showing money flow direction.
    """
    from tradingview_mcp.core.data.egx_sectors import (
        EGX_SECTORS, EGX_SECTOR_META, SECTOR_DISPLAY_NAMES,
        get_sector, get_currency,
    )

    if not TRADINGVIEW_TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    timeframe = sanitize_timeframe(timeframe, "1D")
    top_n_sectors = max(1, min(18, top_n_sectors))
    top_n_stocks = max(1, min(10, top_n_stocks))
    min_stock_score = max(0, min(100, min_stock_score))

    screener = EXCHANGE_SCREENER.get("egx", "egypt")

    # ── Step A: Collect all sector symbols ──
    sector_symbol_map: Dict[str, List[str]] = {}
    all_symbols: List[str] = []
    symbol_to_sectors: Dict[str, List[str]] = {}

    for sector_key, sym_set in EGX_SECTORS.items():
        prefixed = [f"EGX:{s}" for s in sorted(sym_set)]
        sector_symbol_map[sector_key] = prefixed
        for s in prefixed:
            all_symbols.append(s)
            symbol_to_sectors.setdefault(s, []).append(sector_key)

    # Deduplicate for API calls (some symbols appear in multiple sectors)
    unique_symbols = list(dict.fromkeys(all_symbols))

    # ── Step B: Batch fetch TA data ──
    raw_data: Dict[str, Any] = {}
    batch_size = 200
    for i in range(0, len(unique_symbols), batch_size):
        batch = unique_symbols[i:i + batch_size]
        try:
            analysis = get_multiple_analysis(
                screener=screener, interval=timeframe, symbols=batch
            )
            for sym, data in analysis.items():
                if data is not None:
                    try:
                        ind = data.indicators
                        o = ind.get("open")
                        c = ind.get("close")
                        if o and c and o > 0:
                            change = ((c - o) / o) * 100
                            raw_data[sym] = {"indicators": ind, "change": change}
                    except Exception:
                        continue
        except Exception:
            continue

    if not raw_data:
        return {"error": "No data returned for EGX stocks", "timeframe": timeframe}

    # ── Step C: Cross-sectional percentile ranks ──
    all_changes = sorted([d["change"] for d in raw_data.values()])
    n_total = len(all_changes)

    def _percentile_rank(val):
        count_below = sum(1 for c in all_changes if c < val)
        return count_below / n_total if n_total > 0 else 0.5

    # ── Step D: Per-stock scoring ──
    stock_scores: Dict[str, Dict[str, Any]] = {}
    for sym, d in raw_data.items():
        try:
            pct_rank = _percentile_rank(d["change"])
            ccy = get_currency(sym)
            result = compute_stock_score(d["indicators"], change_pct_rank=pct_rank, currency=ccy)
            if result:
                stock_scores[sym] = {
                    "score_result": result,
                    "change": d["change"],
                    "indicators": d["indicators"],
                }
        except Exception:
            continue

    # ── Step E: Sector aggregation ──
    sector_agg: Dict[str, Dict[str, Any]] = {}

    for sector_key, symbols in sector_symbol_map.items():
        changes = []
        rsis = []
        scores = []
        advancing = 0
        declining = 0
        net_volume_flow = 0.0
        total_stocks = 0
        sector_stock_data = []

        for sym in symbols:
            if sym not in raw_data:
                continue
            total_stocks += 1
            d = raw_data[sym]
            ind = d["indicators"]
            chg = d["change"]
            changes.append(chg)

            if chg > 0:
                advancing += 1
            elif chg < 0:
                declining += 1

            rsi = ind.get("RSI")
            if rsi is not None:
                rsis.append(rsi)

            vol = ind.get("volume", 0) or 0
            vol_sma = ind.get("volume.SMA20", 0) or 0
            net_volume_flow += vol - vol_sma

            if sym in stock_scores:
                sc = stock_scores[sym]
                scores.append(sc["score_result"]["score"])
                sector_stock_data.append({
                    "symbol": sym,
                    "score_result": sc["score_result"],
                    "change": sc["change"],
                    "indicators": sc["indicators"],
                })

        if total_stocks == 0:
            sector_agg[sector_key] = {"status": "No Data", "total_stocks": 0}
            continue

        avg_change = round(sum(changes) / len(changes), 2) if changes else 0.0
        avg_rsi = round(sum(rsis) / len(rsis), 2) if rsis else 50.0
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
        breadth_pct = round(advancing / total_stocks * 100, 1)

        sector_agg[sector_key] = {
            "avg_change": avg_change,
            "avg_rsi": avg_rsi,
            "avg_score": avg_score,
            "advancing": advancing,
            "declining": declining,
            "total_stocks": total_stocks,
            "breadth_pct": breadth_pct,
            "net_volume_flow": net_volume_flow,
            "volume_flow_positive": net_volume_flow > 0,
            "stock_data": sector_stock_data,
        }

    # ── Step F: Sector ranking ──
    # Compute change rank for each sector
    valid_sectors = [k for k, v in sector_agg.items() if v.get("total_stocks", 0) > 0]
    sorted_by_change = sorted(valid_sectors, key=lambda k: sector_agg[k]["avg_change"])
    change_rank_map = {}
    for i, k in enumerate(sorted_by_change):
        change_rank_map[k] = i / len(sorted_by_change) if len(sorted_by_change) > 1 else 0.5

    # Compute momentum scores
    for sector_key in valid_sectors:
        agg = sector_agg[sector_key]
        momentum = _compute_sector_momentum_score(
            avg_change=agg["avg_change"],
            avg_rsi=agg["avg_rsi"],
            breadth_pct=agg["breadth_pct"],
            volume_flow_positive=agg["volume_flow_positive"],
            change_rank_pct=change_rank_map.get(sector_key, 0.5),
        )
        agg["momentum_score"] = momentum

        # Classify
        if momentum >= 65 and agg["volume_flow_positive"]:
            agg["status"] = "Hot"
        elif momentum >= 50 or (agg["avg_change"] > 0 and agg["breadth_pct"] > 50):
            agg["status"] = "Warming"
        elif momentum >= 35 and not agg["volume_flow_positive"]:
            agg["status"] = "Cooling"
        else:
            agg["status"] = "Cold"

    # Build heatmap sorted by momentum
    heatmap = []
    for sector_key in sorted(
        valid_sectors,
        key=lambda k: sector_agg[k].get("momentum_score", 0),
        reverse=True,
    ):
        agg = sector_agg[sector_key]
        meta = EGX_SECTOR_META.get(sector_key, {})
        heatmap.append({
            "sector": sector_key,
            "display_name": SECTOR_DISPLAY_NAMES.get(sector_key, sector_key),
            "market_cap_weight": meta.get("market_cap_weight", 0),
            "status": agg["status"],
            "momentum_score": agg.get("momentum_score", 0),
            "avg_change_pct": agg["avg_change"],
            "avg_rsi": agg["avg_rsi"],
            "avg_stock_score": agg["avg_score"],
            "breadth": {
                "advancing": agg["advancing"],
                "declining": agg["declining"],
                "breadth_pct": agg["breadth_pct"],
            },
            "volume_flow": {
                "net_flow": round(agg["net_volume_flow"]),
                "signal": "Inflow" if agg["volume_flow_positive"] else "Outflow",
            },
            "stocks_analyzed": agg["total_stocks"],
        })

    # Add sectors with no data at the end
    for sector_key in EGX_SECTORS:
        if sector_key not in valid_sectors:
            meta = EGX_SECTOR_META.get(sector_key, {})
            heatmap.append({
                "sector": sector_key,
                "display_name": SECTOR_DISPLAY_NAMES.get(sector_key, sector_key),
                "market_cap_weight": meta.get("market_cap_weight", 0),
                "status": "No Data",
                "momentum_score": 0,
                "avg_change_pct": 0,
                "avg_rsi": 0,
                "avg_stock_score": 0,
                "breadth": {"advancing": 0, "declining": 0, "breadth_pct": 0},
                "volume_flow": {"net_flow": 0, "signal": "N/A"},
                "stocks_analyzed": 0,
            })

    # ── Step G: Weighted market view ──
    weighted_change = 0.0
    weighted_rsi = 0.0
    weighted_momentum = 0.0
    total_weight = 0.0

    for sector_key in valid_sectors:
        agg = sector_agg[sector_key]
        weight = EGX_SECTOR_META.get(sector_key, {}).get("market_cap_weight", 0)
        weighted_change += agg["avg_change"] * weight
        weighted_rsi += agg["avg_rsi"] * weight
        weighted_momentum += agg.get("momentum_score", 0) * weight
        total_weight += weight

    if total_weight > 0:
        weighted_change = round(weighted_change / total_weight, 2)
        weighted_rsi = round(weighted_rsi / total_weight, 2)
        weighted_momentum = round(weighted_momentum / total_weight, 1)
    else:
        weighted_change = weighted_rsi = weighted_momentum = 0

    if weighted_change > 0.5:
        market_sentiment = "Bullish"
    elif weighted_change < -0.5:
        market_sentiment = "Bearish"
    else:
        market_sentiment = "Neutral"

    # ── Step H: Top picks per sector ──
    top_sector_keys = [h["sector"] for h in heatmap[:top_n_sectors]
                       if h["status"] != "No Data"]
    sector_top_picks: Dict[str, list] = {}

    for sector_key in top_sector_keys:
        agg = sector_agg[sector_key]
        candidates = agg.get("stock_data", [])
        # Filter by min score and sort
        qualified = [
            c for c in candidates
            if c["score_result"]["score"] >= min_stock_score
        ]
        qualified.sort(key=lambda x: x["score_result"]["score"], reverse=True)

        picks = []
        for c in qualified[:top_n_stocks]:
            result = c["score_result"]
            ind = c["indicators"]
            metrics = compute_metrics(ind)
            liq = result.get("liquidity", {})
            currency = get_currency(c["symbol"])
            entry = {
                "symbol": c["symbol"],
                "price": metrics["price"] if metrics else 0,
                "currency": currency,
                "stock_score": result["score"],
                "grade": result["grade"],
                "trend_state": result["trend_state"],
                "change_pct": result["change_pct"],
                "signals": result["signals"],
                "penalties": result.get("penalties", []),
                "liquidity": liq,
            }
            # Trade setup for stocks scoring 70+
            if result["score"] >= 70:
                setup = compute_trade_setup(ind)
                if setup:
                    quality = compute_trade_quality(ind, result["score"], setup)
                    entry["trade_setup"] = {
                        "setup_types": setup["setup_types"],
                        "entry_points": setup["entry_points"],
                        "stop_loss": setup["stop_loss"],
                        "stop_distance_pct": setup["stop_distance_pct"],
                        "targets": setup["targets"],
                        "risk_reward": setup["risk_reward"],
                        "supports": setup["supports"],
                        "resistances": setup["resistances"],
                    }
                    entry["trade_quality_score"] = quality["trade_quality_score"]
                    entry["trade_quality"] = quality["quality"]

            picks.append(entry)

        sector_top_picks[sector_key] = picks

    # ── Step I: Rotation signals ──
    rotation_signals = _generate_rotation_signals(heatmap)

    return {
        "exchange": "EGX",
        "timeframe": timeframe,
        "total_sectors": len(heatmap),
        "total_stocks_scanned": len(raw_data),
        "weighted_market_view": {
            "weighted_change_pct": weighted_change,
            "weighted_rsi": weighted_rsi,
            "weighted_momentum": weighted_momentum,
            "market_sentiment": market_sentiment,
        },
        "sector_heatmap": heatmap,
        "sector_top_picks": sector_top_picks,
        "rotation_signals": rotation_signals,
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


@mcp.tool()
def egx_index_analysis(
    index: str = "EGX30",
    timeframe: str = "1D",
    limit: int = 30
) -> dict:
    """Analyze an EGX index showing constituent performance with full indicators.

    Returns index-level statistics and per-stock breakdown.

    Args:
        index: Index name - EGX30 (blue chips), EGX70 (mid/small cap), EGX100 (broad),
               SHARIAH33 (Shariah-compliant), EGX35LV (low volatility), TAMAYUZ (small/micro-cap)
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
        limit: Number of stocks to show in detail (max 100)
    """
    from tradingview_mcp.core.data.egx_indices import EGX_INDICES, is_egx30_stock
    from tradingview_mcp.core.data.egx_sectors import get_sector

    if not TRADINGVIEW_TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    index_key = index.strip().upper()
    if index_key not in EGX_INDICES:
        return {
            "error": f"Unknown index: {index}",
            "available_indices": list(EGX_INDICES.keys()),
            "usage": "Use EGX30, EGX70, or EGX100",
        }

    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 100))

    index_info = EGX_INDICES[index_key]
    symbols = index_info["get_symbols"]()
    screener = EXCHANGE_SCREENER.get("egx", "egypt")

    all_stocks = []
    batch_size = 200
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=batch)
        except Exception:
            continue

        for sym, data in analysis.items():
            if data is None:
                continue
            try:
                indicators = data.indicators
                metrics = compute_metrics(indicators)
                if not metrics:
                    continue

                extended = extract_extended_indicators(indicators)
                volume = indicators.get("volume", 0)

                stock_data = {
                    "symbol": sym,
                    "sector": get_sector(sym),
                    "is_egx30": is_egx30_stock(sym),
                    "price": metrics.get("price", 0),
                    "changePercent": metrics.get("change", 0),
                    "volume": volume,
                    "rsi": extended["rsi"]["value"],
                    "rsi_signal": extended["rsi"]["signal"],
                    "sma20": extended["sma"]["sma20"],
                    "sma50": extended["sma"]["sma50"],
                    "sma200": extended["sma"]["sma200"],
                    "atr": extended["atr"]["value"],
                    "atr_volatility": extended["atr"]["volatility"],
                    "macd_crossover": extended["macd"]["crossover"],
                    "volume_signal": extended["volume"]["signal"],
                    "bbw": metrics.get("bbw", 0),
                    "bb_rating": metrics.get("rating", 0),
                    "bb_signal": metrics.get("signal", "N/A"),
                }
                all_stocks.append(stock_data)
            except Exception:
                continue

    if not all_stocks:
        return {"error": f"No data returned for {index_key} constituents", "timeframe": timeframe}

    # Index-level statistics
    changes = [s["changePercent"] for s in all_stocks]
    avg_change = sum(changes) / len(changes)
    advancing = len([c for c in changes if c > 0])
    declining = len([c for c in changes if c < 0])
    unchanged = len([c for c in changes if c == 0])

    # Sector breakdown
    sector_perf = {}
    for s in all_stocks:
        sec = s["sector"]
        if sec not in sector_perf:
            sector_perf[sec] = {"stocks": 0, "total_change": 0.0}
        sector_perf[sec]["stocks"] += 1
        sector_perf[sec]["total_change"] += s["changePercent"]

    sector_summary = []
    for sec, data in sorted(sector_perf.items(), key=lambda x: x[1]["total_change"] / x[1]["stocks"], reverse=True):
        sector_summary.append({
            "sector": sec,
            "stocks_count": data["stocks"],
            "avg_change": round(data["total_change"] / data["stocks"], 2),
        })

    # Sort stocks by change
    by_change = sorted(all_stocks, key=lambda x: x["changePercent"], reverse=True)

    return {
        "index": index_key,
        "index_name": index_info["name"],
        "description": index_info["description"],
        "timeframe": timeframe,
        "index_stats": {
            "total_constituents": index_info["constituents_count"],
            "analyzed": len(all_stocks),
            "avg_change": round(avg_change, 2),
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
            "breadth": round(advancing / len(all_stocks) * 100, 1) if all_stocks else 0,
            "sentiment": "Bullish" if avg_change > 0.5 else "Bearish" if avg_change < -0.5 else "Neutral",
        },
        "sector_breakdown": sector_summary,
        "top_gainers": by_change[:5],
        "top_losers": by_change[-5:][::-1],
        "all_stocks": by_change[:limit],
    }


@mcp.tool()
def multi_timeframe_analysis(
    symbol: str,
    exchange: str = "KUCOIN",
) -> dict:
    """Multi-timeframe alignment analysis (Weekly -> Daily -> 4H -> 1H -> 15m).

    Analyzes a symbol across all key timeframes to find confluence and optimal
    entry timing. Best trades happen when Weekly, Daily, and lower timeframes
    all align in the same direction.

    Args:
        symbol: Symbol - crypto: "BTCUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST)
        exchange: Exchange - crypto: KUCOIN, BINANCE; stocks: EGX, BIST, NASDAQ, NYSE

    Returns:
        Multi-timeframe breakdown with alignment score and trading recommendation.
    """
    if not TRADINGVIEW_TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    exchange = sanitize_exchange(exchange, "KUCOIN")

    if ":" not in symbol:
        full_symbol = f"{exchange.upper()}:{symbol.upper()}"
    else:
        full_symbol = symbol.upper()

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    # Analyze across timeframes: Weekly, Daily, 4H, 1H, 15m
    timeframes = ["1W", "1D", "4h", "1h", "15m"]
    tf_labels = {
        "1W": "Weekly (Trend Bias)",
        "1D": "Daily (Swing Setup)",
        "4h": "4-Hour (Refinement)",
        "1h": "1-Hour (Entry Timing)",
        "15m": "15-Min (Execution)",
    }

    tf_results = {}
    alignment_scores = []

    for tf in timeframes:
        try:
            analysis = get_multiple_analysis(
                screener=screener,
                interval=tf,
                symbols=[full_symbol]
            )

            if full_symbol not in analysis or analysis[full_symbol] is None:
                tf_results[tf] = {"error": f"No data for {tf}"}
                continue

            data = analysis[full_symbol]
            indicators = data.indicators
            metrics = compute_metrics(indicators)
            extended = extract_extended_indicators(indicators)
            tf_context = analyze_timeframe_context(indicators, tf)

            # Determine bias as numeric: +1 bullish, -1 bearish, 0 neutral
            bias_num = 0
            if tf_context["bias"] == "Bullish":
                bias_num = 1
            elif tf_context["bias"] == "Bearish":
                bias_num = -1
            alignment_scores.append(bias_num)

            tf_results[tf] = {
                "label": tf_labels.get(tf, tf),
                "bias": tf_context["bias"],
                "bias_reasons": tf_context["bias_reasons"],
                "key_indicators": tf_context["key_indicators_for_timeframe"],
                "advice": tf_context["advice"],
                "price": metrics.get("price") if metrics else None,
                "change_pct": metrics.get("change") if metrics else None,
                "rsi": extended["rsi"],
                "macd_crossover": extended["macd"]["crossover"],
                "ema_trend": {
                    "ema20": extended["ema"].get("ema20"),
                    "ema50": extended["ema"].get("ema50"),
                    "ema200": extended["ema"].get("ema200"),
                },
                "volume_signal": extended["volume"]["signal"],
                "market_structure": extended["market_structure"]["trend"],
                "trend_strength": extended["market_structure"]["trend_strength"],
                "momentum_aligned": extended["market_structure"]["momentum_aligned"],
            }

        except Exception as e:
            tf_results[tf] = {"error": str(e)}

    # --- Multi-Timeframe Alignment ---
    total_score = sum(alignment_scores)
    num_tf = len(alignment_scores)
    all_bullish = all(s > 0 for s in alignment_scores) if alignment_scores else False
    all_bearish = all(s < 0 for s in alignment_scores) if alignment_scores else False

    if all_bullish:
        alignment = "FULLY ALIGNED BULLISH"
        confidence = "Very High"
        action = "STRONG BUY - All timeframes bullish. Look for pullback entry on 1H/15m."
    elif all_bearish:
        alignment = "FULLY ALIGNED BEARISH"
        confidence = "Very High"
        action = "STRONG SELL - All timeframes bearish. Avoid longs."
    elif total_score >= 3:
        alignment = "MOSTLY BULLISH"
        confidence = "High"
        action = "BUY - Majority of timeframes bullish. Enter on 4H/1H pullback to support."
    elif total_score <= -3:
        alignment = "MOSTLY BEARISH"
        confidence = "High"
        action = "SELL - Majority of timeframes bearish. Avoid catching the falling knife."
    elif total_score > 0:
        alignment = "LEAN BULLISH"
        confidence = "Medium"
        action = "CAUTIOUS BUY - Some bullish signals but not fully aligned. Wait for better setup."
    elif total_score < 0:
        alignment = "LEAN BEARISH"
        confidence = "Medium"
        action = "CAUTIOUS SELL - Some bearish signals. Reduce position or wait."
    else:
        alignment = "MIXED/RANGING"
        confidence = "Low"
        action = "HOLD/NO TRADE - Timeframes conflict. Wait for alignment."

    # Identify which TF breaks the alignment
    divergent_tfs = []
    if num_tf >= 2:
        higher_tf_bias = alignment_scores[0] if alignment_scores else 0  # Weekly
        for i, score in enumerate(alignment_scores):
            if score != 0 and score != higher_tf_bias and higher_tf_bias != 0:
                divergent_tfs.append(timeframes[i])

    return {
        "symbol": full_symbol,
        "exchange": exchange,
        "analysis_type": "Multi-Timeframe Alignment",
        "timeframes": tf_results,
        "alignment": {
            "status": alignment,
            "confidence": confidence,
            "net_score": total_score,
            "scores_by_tf": dict(zip(timeframes, alignment_scores)),
            "divergent_timeframes": divergent_tfs,
        },
        "recommendation": {
            "action": action,
            "entry_timeframe": "1H or 4H pullback" if total_score > 0 else "Wait for alignment",
            "rules": [
                "Weekly sets BIAS (direction only, not entries)",
                "Daily finds SETUP (swing level, confluence)",
                "4H refines entry zone",
                "1H/15m triggers entry with tight stop",
                "Never trade against Weekly + Daily combined direction",
            ],
        },
    }


@mcp.tool()
def egx_stock_screener(
    timeframe: str = "1D",
    min_score: int = 55,
    index_filter: str = "",
    limit: int = 20
) -> dict:
    """Production stock ranking engine for EGX — finds strong stocks with actionable setups.

    Uses a 100-point hybrid model combining:
      A. Trend & Momentum (50 pts) — EMA structure, RSI, MACD, relative performance
      B. Confirmation (20 pts) — Volume, ADX trend strength
      C. Risk-Adjusted Quality (15 pts) — ATR volatility, drawdown stability
      D. Fundamental Overlay (15 pts) — TradingView recommendation proxy

    Stocks scoring 70+ automatically get trade setups (entry, stop, targets, S/R).

    Args:
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
        min_score: Minimum stock score to include (0-100, default 55)
        index_filter: Filter by index — EGX30, EGX70, EGX100, SHARIAH33,
                      EGX35LV, TAMAYUZ. Leave empty for all EGX stocks.
        limit: Number of results (max 50)

    Returns:
        Ranked stocks with score breakdown. Stocks >=70 include full trade plans.
    """
    from tradingview_mcp.core.data.egx_sectors import get_sector, get_currency

    if not TRADINGVIEW_TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    timeframe = sanitize_timeframe(timeframe, "1D")
    min_score = max(0, min(100, min_score))
    limit = max(1, min(50, limit))

    # Determine symbols to scan
    if index_filter:
        from tradingview_mcp.core.data.egx_indices import EGX_INDICES
        idx_key = index_filter.strip().upper()
        if idx_key in EGX_INDICES:
            symbols = EGX_INDICES[idx_key]["get_symbols"]()
            source_label = idx_key
        else:
            return {
                "error": f"Unknown index: {index_filter}",
                "available": list(EGX_INDICES.keys()),
            }
    else:
        symbols = load_symbols("egx")
        source_label = "All EGX"

    if not symbols:
        return {"error": "No EGX symbols found."}

    screener = EXCHANGE_SCREENER.get("egx", "egypt")

    # ── Pass 1: Fetch all data and compute change % for cross-sectional ranking ──
    raw_results = []
    batch_size = 200
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=batch)
        except Exception:
            continue

        for sym, data in analysis.items():
            if data is None:
                continue
            try:
                ind = data.indicators
                o = ind.get("open")
                c = ind.get("close")
                if not o or not c or o <= 0:
                    continue
                change = ((c - o) / o) * 100
                raw_results.append((sym, ind, change))
            except Exception:
                continue

    if not raw_results:
        return {"error": "No data returned for EGX stocks", "timeframe": timeframe}

    # ── Compute cross-sectional change percentile ranks ──
    changes = sorted([r[2] for r in raw_results])
    n = len(changes)

    def _percentile_rank(val):
        # Fraction of stocks this stock beats
        count_below = sum(1 for c in changes if c < val)
        return count_below / n if n > 0 else 0.5

    # ── Pass 2: Score every stock with cross-sectional context ──
    scored_stocks = []
    for sym, ind, change in raw_results:
        try:
            pct_rank = _percentile_rank(change)
            ccy = get_currency(sym)
            result = compute_stock_score(ind, change_pct_rank=pct_rank, currency=ccy)
            if not result or result["score"] < min_score:
                continue

            metrics = compute_metrics(ind)
            if not metrics:
                continue

            # Liquidity filter: skip very illiquid stocks
            vol_sma = ind.get("volume.SMA20")
            vol = ind.get("volume")
            liquidity_status = "Pass"
            if vol_sma and vol_sma < 10000:
                liquidity_status = "Fail — Very Low"
                if min_score >= 55:
                    continue  # Skip illiquid names for non-exploratory scans

            stock_entry = {
                "symbol": sym,
                "sector": get_sector(sym),
                "price": metrics["price"],
                "stock_score": result["score"],
                "grade": result["grade"],
                "trend_state": result["trend_state"],
                "change_pct": result["change_pct"],
                "score_breakdown": result["breakdown"],
                "signals": result["signals"],
                "penalties": result["penalties"],
                "liquidity_status": liquidity_status,
            }

            # ── Layer B+C: Trade setup for stocks scoring 70+ ──
            if result["score"] >= 70:
                setup = compute_trade_setup(ind)
                if setup:
                    quality = compute_trade_quality(ind, result["score"], setup)
                    stock_entry["trade_setup"] = {
                        "setup_types": setup["setup_types"],
                        "entry_points": setup["entry_points"],
                        "stop_loss": setup["stop_loss"],
                        "stop_distance_pct": setup["stop_distance_pct"],
                        "targets": setup["targets"],
                        "risk_reward": setup["risk_reward"],
                        "supports": setup["supports"],
                        "resistances": setup["resistances"],
                    }
                    stock_entry["trade_quality_score"] = quality["trade_quality_score"]
                    stock_entry["trade_quality"] = quality["quality"]
                    stock_entry["trade_notes"] = quality["notes"]
                    stock_entry["trade_quality_breakdown"] = quality["breakdown"]

            scored_stocks.append(stock_entry)
        except Exception:
            continue

    # ── Sort: stock score desc, then trade quality desc ──
    scored_stocks.sort(
        key=lambda x: (x["stock_score"], x.get("trade_quality_score", 0)),
        reverse=True,
    )

    # Grade distribution
    grades = {}
    for s in scored_stocks:
        g = s["grade"]
        grades[g] = grades.get(g, 0) + 1

    # Separate qualified trades from watchlist
    qualified = [s for s in scored_stocks if s["stock_score"] >= 70
                 and s.get("trade_quality_score", 0) >= 65]
    watchlist = [s for s in scored_stocks if s["stock_score"] < 70
                 or s.get("trade_quality_score", 0) < 65]

    return {
        "source": source_label,
        "timeframe": timeframe,
        "min_score": min_score,
        "total_scanned": len(raw_results),
        "total_passed": len(scored_stocks),
        "grade_distribution": grades,
        "qualified_trades": qualified[:limit],
        "qualified_count": len(qualified),
        "watchlist": watchlist[:max(5, limit - len(qualified))],
        "execution_rules": {
            "trade_threshold": "Stock Score >= 70 AND Trade Quality >= 65",
            "risk_reward_min": "R:R to Target 2 >= 2.0 preferred",
            "disclaimer": "For educational/informational purposes only. Not financial advice.",
        },
    }


@mcp.tool()
def egx_trade_plan(
    symbol: str,
    timeframe: str = "1D"
) -> dict:
    """Generate a full trade plan for a specific EGX stock.

    Produces: stock score, score breakdown, entry points (breakout + pullback),
    stop-loss, targets, risk/reward, nearest 3 supports, nearest 3 resistances,
    trade quality score, and actionable notes.

    Args:
        symbol: EGX stock symbol (e.g., "COMI", "TMGH", "FWRY")
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)

    Returns:
        Complete trade plan with score, setup, risk controls, and S/R levels.
    """
    from tradingview_mcp.core.data.egx_sectors import get_sector, get_currency

    if not TRADINGVIEW_TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    timeframe = sanitize_timeframe(timeframe, "1D")

    if ":" not in symbol:
        full_symbol = f"EGX:{symbol.upper()}"
    else:
        full_symbol = symbol.upper()

    screener = EXCHANGE_SCREENER.get("egx", "egypt")

    try:
        analysis = get_multiple_analysis(
            screener=screener, interval=timeframe, symbols=[full_symbol]
        )
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

    if full_symbol not in analysis or analysis[full_symbol] is None:
        return {"error": f"No data found for {full_symbol}"}

    ind = analysis[full_symbol].indicators
    metrics = compute_metrics(ind)
    if not metrics:
        return {"error": f"Could not compute metrics for {full_symbol}"}

    # Layer A: Stock Score
    ccy = get_currency(full_symbol)
    score_result = compute_stock_score(ind, currency=ccy)
    if not score_result:
        return {"error": f"Could not compute stock score for {full_symbol}"}

    # Layer B: Trade Setup
    setup = compute_trade_setup(ind)

    # Layer C: Trade Quality
    quality = None
    if setup:
        quality = compute_trade_quality(ind, score_result["score"], setup)

    # Extended indicators for context
    extended = extract_extended_indicators(ind)

    # Build output
    output = {
        "symbol": full_symbol,
        "sector": get_sector(full_symbol),
        "currency": ccy,
        "timeframe": timeframe,
        "price": metrics["price"],
        "change_pct": score_result["change_pct"],
        "stock_score": score_result["score"],
        "grade": score_result["grade"],
        "trend_state": score_result["trend_state"],
        "score_breakdown": score_result["breakdown"],
        "signals": score_result["signals"],
        "penalties": score_result["penalties"],
        "liquidity": score_result.get("liquidity", {}),
        "rsi": extended["rsi"],
        "macd": extended["macd"],
        "adx": extended["adx"],
        "volume": extended["volume"],
        "ema": extended["ema"],
        "bollinger_bands": extended["bollinger_bands"],
        "tv_recommendation": extended["tv_recommendation"],
    }

    if setup:
        output["trade_setup"] = {
            "setup_types": setup["setup_types"],
            "entry_points": setup["entry_points"],
            "stop_loss": setup["stop_loss"],
            "stop_distance_pct": setup["stop_distance_pct"],
            "targets": setup["targets"],
            "risk_reward": setup["risk_reward"],
            "supports": setup["supports"],
            "resistances": setup["resistances"],
        }

    if quality:
        output["trade_quality_score"] = quality["trade_quality_score"]
        output["trade_quality"] = quality["quality"]
        output["trade_quality_breakdown"] = quality["breakdown"]
        output["trade_notes"] = quality["notes"]

    # Execution recommendation
    ss = score_result["score"]
    tq = quality["trade_quality_score"] if quality else 0
    rr2 = setup["risk_reward"]["to_target_2"] if setup else 0

    if ss >= 70 and tq >= 65 and rr2 and rr2 >= 2.0:
        recommendation = "QUALIFIED — Strong stock with actionable setup"
    elif ss >= 70 and tq >= 50:
        recommendation = "CONDITIONAL — Good stock but setup needs improvement"
    elif ss >= 55:
        recommendation = "WATCHLIST — Monitor for better entry"
    else:
        recommendation = "AVOID — Does not meet momentum/quality criteria"

    output["recommendation"] = recommendation
    output["disclaimer"] = "For educational/informational purposes only. Not financial advice."

    return output


@mcp.tool()
def egx_fibonacci_retracement(
    symbol: str,
    lookback: str = "52W",
    timeframe: str = "1D"
) -> dict:
    """Fibonacci retracement analysis for EGX stocks — identifies key support/resistance
    levels at standard Fibonacci ratios (23.6%, 38.2%, 50%, 61.8%, 78.6%).

    Calculates retracement and extension levels from the period swing high/low,
    detects trend direction, and shows where the current price sits relative to
    key Fibonacci zones (golden pocket, 50% retracement, etc.).

    Args:
        symbol: EGX stock symbol (e.g., "COMI", "TMGH", "FWRY")
        lookback: Period for swing high/low detection — "1M", "3M", "6M", "52W", "ALL" (default 52W)
        timeframe: Analysis timeframe for current indicators — 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)

    Returns:
        Fibonacci retracement & extension levels, price position analysis, key zones, and context.
    """
    from tradingview_mcp.core.data.egx_sectors import get_sector, get_currency

    if not TRADINGVIEW_TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    timeframe = sanitize_timeframe(timeframe, "1D")
    lookback = lookback.strip().upper()
    valid_lookbacks = {"1M", "3M", "6M", "52W", "ALL"}
    if lookback not in valid_lookbacks:
        return {"error": f"Invalid lookback: {lookback}", "valid": sorted(valid_lookbacks)}

    if ":" not in symbol:
        full_symbol = f"EGX:{symbol.upper()}"
    else:
        full_symbol = symbol.upper()

    screener = EXCHANGE_SCREENER.get("egx", "egypt")

    # ── Step 1: Get period high/low via tradingview_screener ──────────────
    swing_high = None
    swing_low = None
    swing_source = None

    LOOKBACK_COLUMNS = {
        "1M":  ("High.1M", "Low.1M"),
        "3M":  ("High.3M", "Low.3M"),
        "6M":  ("High.6M", "Low.6M"),
        "52W": ("price_52_week_high", "price_52_week_low"),
        "ALL": ("High.All", "Low.All"),
    }

    if TRADINGVIEW_SCREENER_AVAILABLE:
        try:
            high_col, low_col = LOOKBACK_COLUMNS[lookback]
            q = (
                Query()
                .set_markets("egypt")
                .select("close", high_col, low_col)
                .set_tickers([full_symbol])
            )
            _, df = q.get_scanner_data()
            if not df.empty:
                row = df.iloc[0]
                h = row.get(high_col)
                l = row.get(low_col)
                if h is not None and l is not None and h > l:
                    swing_high = float(h)
                    swing_low = float(l)
                    swing_source = f"screener ({lookback} period high/low)"
        except Exception:
            pass  # Fall through to fallback

    # ── Step 2: Get current indicators via tradingview_ta ─────────────────
    try:
        analysis = get_multiple_analysis(
            screener=screener, interval=timeframe, symbols=[full_symbol]
        )
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

    if full_symbol not in analysis or analysis[full_symbol] is None:
        return {"error": f"No data found for {full_symbol}"}

    ind = analysis[full_symbol].indicators
    close = ind.get("close")
    if not close:
        return {"error": f"No price data for {full_symbol}"}

    # ── Fallback: Use Fibonacci pivot R3/S3 as swing proxies ──────────────
    if swing_high is None or swing_low is None:
        fib_r3 = ind.get("Pivot.M.Fibonacci.R3")
        fib_s3 = ind.get("Pivot.M.Fibonacci.S3")
        classic_r3 = ind.get("Pivot.M.Classic.R3")
        classic_s3 = ind.get("Pivot.M.Classic.S3")

        h_candidate = fib_r3 or classic_r3
        l_candidate = fib_s3 or classic_s3

        if h_candidate and l_candidate and h_candidate > l_candidate:
            swing_high = float(h_candidate)
            swing_low = float(l_candidate)
            swing_source = "pivot points (R3/S3 fallback)"
        else:
            return {
                "error": "Could not determine swing high/low for Fibonacci calculation",
                "hint": "Period high/low data not available for this symbol",
            }

    # Validate range
    swing_range_pct = ((swing_high - swing_low) / swing_low) * 100
    if swing_range_pct < 2:
        return {
            "error": f"Swing range too narrow ({swing_range_pct:.1f}%) for meaningful Fibonacci levels",
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
        }

    # ── Step 3: Compute Fibonacci ─────────────────────────────────────────
    ema50 = ind.get("EMA50")
    ema200 = ind.get("EMA200")
    trend, trend_reasoning = detect_trend_for_fibonacci(
        close, swing_high, swing_low, ema50, ema200
    )

    fib_levels = compute_fibonacci_levels(swing_high, swing_low, trend)
    position = analyze_fibonacci_position(close, fib_levels)

    # ── Step 4: Context indicators ────────────────────────────────────────
    rsi_val = ind.get("RSI")
    atr_val = ind.get("ATR")
    vol = ind.get("volume")
    vol_sma = ind.get("volume.SMA20")
    vol_ratio = round(vol / vol_sma, 2) if vol and vol_sma and vol_sma > 0 else None

    metrics = compute_metrics(ind)
    change_pct = round(((close - ind.get("open", close)) / ind.get("open", close)) * 100, 2) if ind.get("open") else None

    # ── Step 5: Build interpretation ──────────────────────────────────────
    interp_parts = []
    interp_parts.append(
        f"Price is at {position['retracement_depth_pct']}% retracement of the {trend}."
    )
    if position["key_zone"]:
        interp_parts.append(f"Currently in {position['key_zone']}.")
    if position["fib_supports"]:
        nearest_s = position["fib_supports"][0]
        interp_parts.append(f"Key Fib support at {nearest_s['price']} ({nearest_s['ratio']}).")
    if position["fib_resistances"]:
        nearest_r = position["fib_resistances"][0]
        interp_parts.append(f"Key Fib resistance at {nearest_r['price']} ({nearest_r['ratio']}).")

    return {
        "symbol": full_symbol,
        "sector": get_sector(full_symbol),
        "timeframe": timeframe,
        "lookback_period": lookback,
        "price": round(close, 2),
        "change_pct": change_pct,
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "swing_range_pct": round(swing_range_pct, 1),
        "swing_source": swing_source,
        "trend": trend,
        "trend_reasoning": trend_reasoning,
        "retracement_levels": fib_levels["retracement_levels"],
        "extension_levels": fib_levels["extension_levels"],
        "price_position": position,
        "context": {
            "rsi": round(rsi_val, 1) if rsi_val else None,
            "ema50": round(ema50, 2) if ema50 else None,
            "ema200": round(ema200, 2) if ema200 else None,
            "atr": round(atr_val, 2) if atr_val else None,
            "volume_ratio": vol_ratio,
        },
        "interpretation": " ".join(interp_parts),
        "disclaimer": "For educational/informational purposes only. Not financial advice.",
    }


def _safe_round(value, decimals: int = 4):
    """Module-level safe round for server.py usage."""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None

@mcp.tool()
def market_sentiment(symbol: str, category: str = "all", limit: int = 20) -> dict:
    """Real-time Reddit sentiment analysis for stocks and crypto.
    
    Args:
        symbol: Asset symbol ("AAPL", "BTC", "ETH", "TSLA")
        category: Subreddit group to search ("crypto", "stocks", "all")
        limit: Number of posts to analyze
    """
    return analyze_sentiment(symbol, category, limit)


@mcp.tool()
def financial_news(symbol: str = None, category: str = "stocks", limit: int = 10) -> dict:
    """Real-time financial news from RSS feeds (Reuters, CoinDesk, etc.)
    
    Args:
        symbol: Optional symbol filter ("AAPL", "BTC"). None = all news.
        category: Feed category ("crypto", "stocks", "all")
        limit: Max number of news items
    """
    return fetch_news_summary(symbol, category, limit)


@mcp.tool()
def combined_analysis(symbol: str, exchange: str = "NASDAQ", timeframe: str = "1D") -> dict:
    """POWER TOOL: TradingView technical analysis + Reddit sentiment + Financial news.
    Confluence analysis of all signals for a complete market picture.
    
    Args:
        symbol: Asset symbol ("AAPL", "BTCUSDT", "THYAO")
        exchange: Exchange (NASDAQ, NYSE, BINANCE, KUCOIN, BIST, EGX)
        timeframe: Analysis timeframe (5m, 15m, 1h, 4h, 1D, 1W)
    """
    # Technical analysis 
    tech = coin_analysis(symbol, exchange, timeframe)
    
    # Reddit sentiment
    cat = "crypto" if exchange.upper() in ["BINANCE", "KUCOIN", "BYBIT"] else "stocks"
    sentiment = analyze_sentiment(symbol, category=cat)
    
    # News
    news = fetch_news_summary(symbol, category=cat, limit=5)
    
    # Confluence
    tech_momentum = tech.get("market_sentiment", {}).get("momentum", "") if isinstance(tech, dict) else ""
    tech_bullish = tech_momentum == "Bullish"
    sent_bullish = sentiment.get("sentiment_score", 0) > 0.1
    signals_agree = tech_bullish == sent_bullish
    
    confidence = "HIGH" if signals_agree else "MIXED"
    tech_signal = tech.get("market_sentiment", {}).get("buy_sell_signal", "N/A") if isinstance(tech, dict) else "N/A"
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "technical": tech,
        "sentiment": sentiment,
        "news": {
            "count": news.get("count", 0),
            "latest": news.get("items", [])[:3],
        },
        "confluence": {
            "signals_agree": signals_agree,
            "confidence": confidence,
            "recommendation": (
                f"Technical {tech_signal} "
                f"{'confirmed by' if signals_agree else 'conflicts with'} "
                f"{sentiment.get('sentiment_label', 'Neutral')} Reddit sentiment "
                f"({sentiment.get('posts_analyzed', 0)} posts analyzed)"
            ),
        },
    }


@mcp.tool()
def backtest_strategy(
    symbol: str,
    strategy: str,
    period: str = "1y",
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> dict:
    """Backtest a trading strategy on historical daily data with institutional-grade metrics.

    Args:
        symbol:          Yahoo Finance symbol — stocks (AAPL, TSLA, NVDA), crypto (BTC-USD, ETH-USD),
                         ETFs (SPY, QQQ), indices (^GSPC, ^IXIC), Turkish (THYAO.IS)
        strategy:        Trading strategy to test:
                           'rsi'        — Buy oversold (RSI<30), Sell overbought (RSI>70)
                           'bollinger'  — Buy at lower Bollinger Band, Sell at middle band
                           'macd'       — Buy on MACD golden cross, Sell on death cross
                           'ema_cross'  — Buy EMA20>EMA50 crossover, Sell on reversal
                           'supertrend' — Buy on bullish Supertrend flip (🔥 trending 2025)
                           'donchian'   — Buy Donchian Channel breakout (Turtle Trader style)
        period:          Historical data period: '1mo', '3mo', '6mo', '1y', '2y'
        initial_capital: Starting capital in USD (default: $10,000)
        commission_pct:  Per-trade commission % (default: 0.1% — typical broker fee)
        slippage_pct:    Per-trade slippage % (default: 0.05%)

    Returns:
        Institutional-grade backtest report: win rate, total return, Sharpe ratio,
        Calmar ratio, max drawdown, profit factor, expectancy, best/worst trade,
        vs buy-and-hold benchmark. Includes transaction cost simulation.
    """
    return run_backtest(symbol, strategy, period, initial_capital, commission_pct, slippage_pct)


@mcp.tool()
def compare_strategies(
    symbol: str,
    period: str = "1y",
    initial_capital: float = 10000.0,
) -> dict:
    """Run all 4 strategies (RSI, Bollinger, MACD, EMA Cross) on the same symbol
    and return a ranked performance leaderboard.

    Args:
        symbol:          Yahoo Finance symbol (AAPL, BTC-USD, SPY…)
        period:          Historical data period: '1mo', '3mo', '6mo', '1y', '2y'
        initial_capital: Starting capital in USD (default: $10,000)

    Returns:
        Ranked leaderboard of all 4 strategies vs buy-and-hold benchmark.
    """
    return _compare_strategies(symbol, period, initial_capital)


@mcp.tool()
def yahoo_price(symbol: str) -> dict:
    """Real-time price quote from Yahoo Finance for any stock, crypto, ETF or index.

    Args:
        symbol: Yahoo Finance symbol. Examples:
                Stocks:  AAPL, TSLA, MSFT, NVDA, GOOGL
                Crypto:  BTC-USD, ETH-USD, SOL-USD
                ETFs:    SPY, QQQ, GLD
                Indices: ^GSPC (S&P500), ^DJI (Dow Jones), ^IXIC (NASDAQ)
                FX:      EURUSD=X, GBPUSD=X
                Turkish: THYAO.IS, SASA.IS
    """
    return get_price(symbol)


@mcp.tool()
def market_snapshot() -> dict:
    """Global market overview: major indices (S&P500, NASDAQ, Dow, VIX),
    top crypto (BTC, ETH, SOL, BNB), FX rates, and key ETFs.
    Powered by Yahoo Finance.
    """
    return get_market_snapshot()


if __name__ == "__main__":
	main()

