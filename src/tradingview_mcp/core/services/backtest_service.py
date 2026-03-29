"""
Backtesting Service for tradingview-mcp — v2 (Enhanced)

Runs trading strategy simulations on Yahoo Finance historical OHLCV data.
Pure Python — no pandas, no numpy, no external backtesting libraries.

Supported strategies (6 total):
  - rsi         : RSI oversold/overbought mean reversion
  - bollinger   : Bollinger Band mean reversion
  - macd        : MACD golden/death cross
  - ema_cross   : EMA 20/50 golden/death cross
  - supertrend  : Supertrend ATR-based trend following (🔥 most popular 2025)
  - donchian    : Donchian Channel breakout (🔥 institutional favorite)

Key metrics (institutional grade):
  - Win rate, profit factor, max drawdown
  - Sharpe ratio (risk-adjusted return)
  - Calmar ratio (return / max drawdown)
  - Expectancy (avg $ per trade)
  - Transaction costs (commission + slippage)

Usage:
    result = run_backtest("BTC-USD", "supertrend", "1y", commission_pct=0.1)
    compare = compare_strategies("AAPL", "2y")
"""
from __future__ import annotations

import json
import math
import statistics
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.indicators_calc import (
    calc_rsi,
    calc_bollinger,
    calc_macd,
    calc_ema,
    calc_supertrend,
    calc_donchian,
)

_UA = "tradingview-mcp/0.5.0 backtest-bot"
_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

_VALID_PERIODS   = {"1mo", "3mo", "6mo", "1y", "2y"}
_VALID_INTERVALS = {"1d"}

_STRATEGY_LABELS = {
    "rsi":        "RSI Oversold/Overbought",
    "bollinger":  "Bollinger Band Mean Reversion",
    "macd":       "MACD Crossover",
    "ema_cross":  "EMA 20/50 Golden/Death Cross",
    "supertrend": "Supertrend (ATR-based Trend Following)",
    "donchian":   "Donchian Channel Breakout",
}


# ─── Data Fetching ────────────────────────────────────────────────────────────

def _fetch_ohlcv(symbol: str, period: str, interval: str = "1d") -> list[dict]:
    """
    Fetch historical OHLCV from Yahoo Finance.
    Direct connection first; proxy fallback if needed.
    """
    url = f"{_YF_BASE}/{symbol}?interval={interval}&range={period}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})

    data = None
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass

    if data is None:
        try:
            from tradingview_mcp.core.services.proxy_manager import build_opener_with_proxy
            opener = build_opener_with_proxy(_UA)
            with opener.open(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Both direct and proxy connections failed: {e}")

    result     = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    q          = result["indicators"]["quote"][0]

    candles = []
    for i, ts in enumerate(timestamps):
        o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, l, c):
            continue
        candles.append({
            "date":   datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
            "open":   round(o, 4),
            "high":   round(h, 4),
            "low":    round(l, 4),
            "close":  round(c, 4),
            "volume": v or 0,
        })
    return candles


# ─── Strategy Engines ─────────────────────────────────────────────────────────

def _run_rsi(candles, oversold=30, overbought=70, period=14, **_):
    closes = [c["close"] for c in candles]
    rsi    = calc_rsi(closes, period)
    trades, position = [], None
    for i in range(1, len(candles)):
        if rsi[i] is None:
            continue
        price, date = candles[i]["close"], candles[i]["date"]
        if position is None and rsi[i] < oversold:
            position = {"entry_date": date, "entry_price": price, "strategy": "rsi"}
        elif position is not None and rsi[i] > overbought:
            trades.append({**position, "exit_date": date, "exit_price": price})
            position = None
    return trades


def _run_bollinger(candles, period=20, std_mult=2.0, **_):
    closes = [c["close"] for c in candles]
    bb     = calc_bollinger(closes, period, std_mult)
    trades, position = [], None
    for i in range(1, len(candles)):
        if bb["lower"][i] is None:
            continue
        price, date = candles[i]["close"], candles[i]["date"]
        if position is None and price < bb["lower"][i]:
            position = {"entry_date": date, "entry_price": price, "strategy": "bollinger"}
        elif position is not None and price > bb["middle"][i]:
            trades.append({**position, "exit_date": date, "exit_price": price})
            position = None
    return trades


def _run_macd(candles, fast=12, slow=26, signal=9, **_):
    closes = [c["close"] for c in candles]
    macd   = calc_macd(closes, fast, slow, signal)
    trades, position = [], None
    for i in range(1, len(candles)):
        m, s, mp, sp = macd["macd"][i], macd["signal"][i], macd["macd"][i-1], macd["signal"][i-1]
        if None in (m, s, mp, sp):
            continue
        price, date = candles[i]["close"], candles[i]["date"]
        if position is None and mp < sp and m >= s:
            position = {"entry_date": date, "entry_price": price, "strategy": "macd"}
        elif position is not None and mp > sp and m <= s:
            trades.append({**position, "exit_date": date, "exit_price": price})
            position = None
    return trades


def _run_ema_cross(candles, fast_period=20, slow_period=50, **_):
    closes   = [c["close"] for c in candles]
    ema_fast = calc_ema(closes, fast_period)
    ema_slow = calc_ema(closes, slow_period)
    trades, position = [], None
    for i in range(1, len(candles)):
        f, s, fp, sp = ema_fast[i], ema_slow[i], ema_fast[i-1], ema_slow[i-1]
        if None in (f, s, fp, sp):
            continue
        price, date = candles[i]["close"], candles[i]["date"]
        if position is None and fp < sp and f >= s:
            position = {"entry_date": date, "entry_price": price, "strategy": "ema_cross"}
        elif position is not None and fp > sp and f <= s:
            trades.append({**position, "exit_date": date, "exit_price": price})
            position = None
    return trades


def _run_supertrend(candles, atr_period=10, multiplier=3.0, **_):
    """
    Supertrend strategy — buy on bullish flip, sell on bearish flip.
    One of the most popular trend-following strategies in 2025.
    """
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    closes = [c["close"] for c in candles]
    st     = calc_supertrend(highs, lows, closes, atr_period, multiplier)

    trades, position = [], None
    for i in range(1, len(candles)):
        d, dp = st["direction"][i], st["direction"][i - 1]
        if d is None or dp is None:
            continue
        price, date = candles[i]["close"], candles[i]["date"]

        # Bullish flip: bearish → bullish
        if position is None and dp == -1 and d == 1:
            position = {"entry_date": date, "entry_price": price, "strategy": "supertrend"}

        # Bearish flip: bullish → bearish
        elif position is not None and dp == 1 and d == -1:
            trades.append({**position, "exit_date": date, "exit_price": price})
            position = None

    return trades


def _run_donchian(candles, period=20, **_):
    """
    Donchian Channel breakout — buy new highs, sell new lows.
    Classic breakout strategy used by the famous Turtle Traders.
    """
    highs  = [c["high"] for c in candles]
    lows   = [c["low"]  for c in candles]
    dc     = calc_donchian(highs, lows, period)
    trades, position = [], None

    for i in range(1, len(candles)):
        if dc["upper"][i] is None:
            continue
        price, date = candles[i]["close"], candles[i]["date"]
        prev_high   = highs[i - 1]
        prev_low    = lows[i - 1]

        # Breakout above upper band → BUY
        if position is None and prev_high > dc["upper"][i - 1] if dc["upper"][i - 1] else False:
            position = {"entry_date": date, "entry_price": price, "strategy": "donchian"}

        # Break below lower band → SELL
        elif position is not None and dc["lower"][i] is not None and price < dc["lower"][i]:
            trades.append({**position, "exit_date": date, "exit_price": price})
            position = None

    return trades


_STRATEGY_MAP = {
    "rsi":        _run_rsi,
    "bollinger":  _run_bollinger,
    "macd":       _run_macd,
    "ema_cross":  _run_ema_cross,
    "supertrend": _run_supertrend,
    "donchian":   _run_donchian,
}


# ─── Transaction Cost Application ─────────────────────────────────────────────

def _apply_costs(trades: list[dict], commission_pct: float, slippage_pct: float) -> list[dict]:
    """Apply realistic transaction costs: commission + slippage per round-trip."""
    total_cost_pct = (commission_pct + slippage_pct) * 2  # entry + exit
    result = []
    for t in trades:
        gross = (t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100
        net   = round(gross - total_cost_pct, 3)
        result.append({**t, "return_pct": net, "gross_return_pct": round(gross, 3),
                        "cost_pct": round(-total_cost_pct, 3)})
    return result


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _calc_metrics(trades: list[dict], initial_capital: float) -> dict:
    empty = {
        "total_trades": 0, "win_rate_pct": 0, "winning_trades": 0, "losing_trades": 0,
        "total_return_pct": 0, "final_capital": initial_capital,
        "avg_gain_pct": 0, "avg_loss_pct": 0, "max_drawdown_pct": 0,
        "profit_factor": 0, "sharpe_ratio": 0, "calmar_ratio": 0,
        "expectancy_pct": 0, "best_trade": None, "worst_trade": None,
    }
    if not trades:
        return empty

    winners = [t for t in trades if t["return_pct"] > 0]
    losers  = [t for t in trades if t["return_pct"] <= 0]

    # Compound capital
    capital = initial_capital
    peak    = capital
    max_dd  = 0.0
    returns = []
    for t in trades:
        r = t["return_pct"] / 100
        capital *= (1 + r)
        returns.append(r)
        peak   = max(peak, capital)
        dd     = (peak - capital) / peak * 100
        max_dd = max(max_dd, dd)

    total_return = (capital - initial_capital) / initial_capital * 100
    avg_gain  = sum(t["return_pct"] for t in winners) / len(winners) if winners else 0
    avg_loss  = sum(t["return_pct"] for t in losers)  / len(losers)  if losers  else 0
    gp = sum(t["return_pct"] for t in winners)
    gl = abs(sum(t["return_pct"] for t in losers))
    profit_factor = round(gp / gl, 2) if gl > 0 else float("inf")

    # Sharpe Ratio (annualized, assuming 252 trading days, risk-free = 4%)
    sharpe = 0.0
    if len(returns) > 1:
        mean_r = statistics.mean(returns)
        std_r  = statistics.stdev(returns)
        rf_daily = 0.04 / 252
        sharpe = round((mean_r - rf_daily) / std_r * math.sqrt(252), 2) if std_r > 0 else 0

    # Calmar Ratio = annualized return / max drawdown
    calmar = 0.0
    if max_dd > 0:
        calmar = round(total_return / max_dd, 2)

    # Expectancy = (WR × avg_gain) - (LR × avg_loss)
    wr = len(winners) / len(trades)
    lr = 1 - wr
    expectancy = round(wr * avg_gain + lr * avg_loss, 2)

    best  = max(trades, key=lambda t: t["return_pct"])
    worst = min(trades, key=lambda t: t["return_pct"])

    return {
        "total_trades":     len(trades),
        "winning_trades":   len(winners),
        "losing_trades":    len(losers),
        "win_rate_pct":     round(wr * 100, 1),
        "final_capital":    round(capital, 2),
        "total_return_pct": round(total_return, 2),
        "avg_gain_pct":     round(avg_gain, 2),
        "avg_loss_pct":     round(avg_loss, 2),
        "max_drawdown_pct": round(-max_dd, 2),
        "profit_factor":    profit_factor,
        "sharpe_ratio":     sharpe,
        "calmar_ratio":     calmar,
        "expectancy_pct":   expectancy,
        "best_trade":       {k: best[k]  for k in ("entry_date", "exit_date", "return_pct")},
        "worst_trade":      {k: worst[k] for k in ("entry_date", "exit_date", "return_pct")},
    }


def _buy_and_hold_return(candles: list[dict]) -> float:
    if len(candles) < 2:
        return 0.0
    return round((candles[-1]["close"] - candles[0]["close"]) / candles[0]["close"] * 100, 2)


# ─── Public API ───────────────────────────────────────────────────────────────

def run_backtest(
    symbol: str,
    strategy: str,
    period: str = "1y",
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> dict:
    """
    Run a backtest for the given symbol and strategy.

    Args:
        symbol:          Yahoo Finance symbol (AAPL, BTC-USD, ^GSPC, THYAO.IS…)
        strategy:        rsi | bollinger | macd | ema_cross | supertrend | donchian
        period:          1mo | 3mo | 6mo | 1y | 2y
        initial_capital: Starting capital in USD
        commission_pct:  Per-trade commission % (default 0.1% = typical broker)
        slippage_pct:    Per-trade slippage % (default 0.05%)

    Returns:
        Full institutional-grade performance report.
    """
    strategy = strategy.lower().strip()
    period   = period.lower().strip()

    if strategy not in _STRATEGY_MAP:
        return {"error": f"Unknown strategy '{strategy}'. Choose: {', '.join(_STRATEGY_MAP)}"}
    if period not in _VALID_PERIODS:
        return {"error": f"Invalid period '{period}'. Choose: {', '.join(_VALID_PERIODS)}"}

    try:
        candles = _fetch_ohlcv(symbol, period, "1d")
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 30:
        return {"error": f"Not enough data ({len(candles)} candles). Try a longer period."}

    raw_trades = _STRATEGY_MAP[strategy](candles)
    trades     = _apply_costs(raw_trades, commission_pct, slippage_pct)
    metrics    = _calc_metrics(trades, initial_capital)
    bnh        = _buy_and_hold_return(candles)

    return {
        "symbol":              symbol.upper(),
        "strategy":            strategy,
        "strategy_label":      _STRATEGY_LABELS[strategy],
        "period":              period,
        "timeframe":           "Daily (1d)",
        "candles_analyzed":    len(candles),
        "date_from":           candles[0]["date"],
        "date_to":             candles[-1]["date"],
        "initial_capital":     round(initial_capital, 2),
        "commission_pct":      commission_pct,
        "slippage_pct":        slippage_pct,
        **metrics,
        "buy_and_hold_return_pct": bnh,
        "vs_buy_and_hold_pct": round(metrics["total_return_pct"] - bnh, 2),
        "trade_log":           trades[-10:],
        "data_source":         "Yahoo Finance",
        "disclaimer":          "Past performance does not guarantee future results. For educational use only.",
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }


def compare_strategies(
    symbol: str,
    period: str = "1y",
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> dict:
    """
    Run all 6 strategies on the same symbol with a single OHLCV fetch.
    Returns a ranked leaderboard with Sharpe ratio, drawdown, and profit factor.
    """
    try:
        candles = _fetch_ohlcv(symbol, period, "1d")
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 30:
        return {"error": f"Not enough data ({len(candles)} candles)."}

    results = []
    for strat, fn in _STRATEGY_MAP.items():
        raw    = fn(candles)
        trades = _apply_costs(raw, commission_pct, slippage_pct)
        m      = _calc_metrics(trades, initial_capital)
        results.append({
            "strategy":         strat,
            "strategy_label":   _STRATEGY_LABELS[strat],
            "total_return_pct": m["total_return_pct"],
            "win_rate_pct":     m["win_rate_pct"],
            "total_trades":     m["total_trades"],
            "profit_factor":    m["profit_factor"],
            "sharpe_ratio":     m["sharpe_ratio"],
            "calmar_ratio":     m["calmar_ratio"],
            "max_drawdown_pct": m["max_drawdown_pct"],
            "expectancy_pct":   m["expectancy_pct"],
        })

    results.sort(key=lambda x: x["total_return_pct"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    bnh = _buy_and_hold_return(candles)

    return {
        "symbol":                  symbol.upper(),
        "period":                  period,
        "timeframe":               "Daily (1d)",
        "candles_analyzed":        len(candles),
        "date_from":               candles[0]["date"],
        "date_to":                 candles[-1]["date"],
        "initial_capital":         round(initial_capital, 2),
        "commission_pct":          commission_pct,
        "slippage_pct":            slippage_pct,
        "buy_and_hold_return_pct": bnh,
        "winner":                  results[0]["strategy"] if results else None,
        "ranking":                 results,
        "disclaimer":              "Past performance does not guarantee future results.",
        "timestamp":               datetime.now(timezone.utc).isoformat(),
    }
