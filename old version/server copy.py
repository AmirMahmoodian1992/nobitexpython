# server.py
# pip install fastapi uvicorn requests pandas

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import requests, datetime as dt
import pandas as pd
from typing import List, Dict, Any



app = FastAPI()

UDF_URL = "https://apiv2.nobitex.ir/market/udf/history"

def fetch_udf(symbol: str, resolution: str, start_ts: int, end_ts: int):
    params = {"symbol": symbol, "resolution": resolution, "from": start_ts, "to": end_ts}
    r = requests.get(UDF_URL, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    if j.get("s") != "ok":
        raise RuntimeError(f"UDF error: {j}")
    # convert to list of OHLC dicts with time as unix seconds
    out = []
    t = j.get("t", [])
    o = j.get("o", [])
    h = j.get("h", [])
    l = j.get("l", [])
    c = j.get("c", [])
    v = j.get("v", [])
    n = len(t)
    for i in range(n):
        out.append({
            "time": int(t[i]),           # unix seconds (preferred by lightweight-charts)
            "open": float(o[i]) if i < len(o) else float(c[i]),
            "high": float(h[i]) if i < len(h) else float(c[i]),
            "low":  float(l[i]) if i < len(l) else float(c[i]),
            "close":float(c[i]),
            "volume": float(v[i]) if i < len(v) else 0.0
        })
    return out

@app.get("/candles")
def candles(symbol: str = Query("BTCIRT"), resolution: str = Query("60"), days: int = Query(7)):
    now = int(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp())
    start = now - int(days) * 86400
    try:
        data = fetch_udf(symbol, resolution, start, now)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return JSONResponse(content={"candles": data})
def moving_average(values: List[float], window: int) -> List[float]:
    """Return list of moving averages aligned to the input indices.
       Indices with insufficient history contain None."""
    if window <= 0:
        raise ValueError("window must be > 0")
    n = len(values)
    out = [None] * n
    if n < window:
        return out
    # compute cumulative sum for O(n)
    csum = [0.0] * (n + 1)
    for i in range(n):
        csum[i+1] = csum[i] + values[i]
    for i in range(window - 1, n):
        s = csum[i+1] - csum[i+1-window]
        out[i] = s / window
    return out


def compute_max_drawdown(equity_series: List[float]) -> float:
    """Return max drawdown as positive fraction (e.g. 0.2 -> 20% drawdown)."""
    peak = -float('inf')
    max_dd = 0.0
    for v in equity_series:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


@app.get("/backtest")
def backtest(
    symbol: str = Query("BTCIRT"),
    resolution: str = Query("60"),
    days: int = Query(7),
    short: int = Query(9),
    long: int = Query(21),
    initial_capital: float = Query(1000.0),
    fee_pct: float = Query(0.1),            # percent, e.g. 0.1 => 0.1%
    position_size_pct: float = Query(1.0)   # fraction of capital to use per trade (1.0 => 100%)
) -> JSONResponse:
    """
    Simple SMA crossover backtest.
    Returns JSON {
      trades: [...],
      equity: [{time, equity}],
      stats: {net_profit, return_pct, trades, win_rate, gross_profit, gross_loss, max_drawdown}
    }
    """
    # fetch candles (same as /candles)
    now = int(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp())
    start = now - int(days) * 86400
    try:
        candles = fetch_udf(symbol, resolution, start, now)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not candles:
        return JSONResponse(content={"error": "no candles"}, status_code=400)

    closes = [c["close"] for c in candles]
    times = [c["time"] for c in candles]

    sma_short = moving_average(closes, short)
    sma_long = moving_average(closes, long)

    capital = float(initial_capital)
    cash = capital
    position = None  # None or dict with keys: entry_idx, entry_time, entry_price, units, entry_fee
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []

    fee_factor = fee_pct / 100.0

    n = len(candles)
    # iterate and detect crossovers; we will act on next candle open to avoid lookahead
    for i in range(1, n - 1):  # we need i+1 for the next candle open execution
        # append equity mark to curve at this candle's close (before potential next trade)
        if position is None:
            equity = cash
        else:
            # mark-to-market at close price
            equity = cash + position["units"] * closes[i]
        equity_curve.append({"time": times[i], "equity": equity})

        # both SMAs must be defined to evaluate crossover
        if sma_short[i-1] is None or sma_long[i-1] is None or sma_short[i] is None or sma_long[i] is None:
            continue

        # buy signal: short crosses above long
        if position is None and sma_short[i-1] <= sma_long[i-1] and sma_short[i] > sma_long[i]:
            # execute at next candle open (i+1)
            entry_price = candles[i+1]["open"]
            # compute units such that total_spent + entry_fee <= capital * position_size_pct
            available = capital * position_size_pct
            units = available / (entry_price * (1.0 + fee_factor))
            entry_fee = units * entry_price * fee_factor
            cost = units * entry_price
            # update cash: subtract cost + fee
            cash -= (cost + entry_fee)
            position = {
                "entry_idx": i+1,
                "entry_time": int(candles[i+1]["time"]),
                "entry_price": float(entry_price),
                "units": float(units),
                "entry_fee": float(entry_fee)
            }
            continue

        # sell signal: short crosses below long
        if position is not None and sma_short[i-1] >= sma_long[i-1] and sma_short[i] < sma_long[i]:
            # execute close at next candle open (i+1)
            exit_price = candles[i+1]["open"]
            units = position["units"]
            revenue = units * exit_price
            exit_fee = revenue * fee_factor
            cash += (revenue - exit_fee)
            # compute trade P&L
            entry_cost = position["units"] * position["entry_price"]
            total_fees = position["entry_fee"] + exit_fee
            pnl = revenue - entry_cost - total_fees
            trade = {
                "entry_time": position["entry_time"],
                "entry_price": position["entry_price"],
                "exit_time": int(candles[i+1]["time"]),
                "exit_price": float(exit_price),
                "units": units,
                "entry_fee": position["entry_fee"],
                "exit_fee": float(exit_fee),
                "pnl": float(pnl)
            }
            trades.append(trade)
            position = None
            continue

    # final equity mark for last candle
    if position is None:
        equity = cash
    else:
        equity = cash + position["units"] * closes[-1]
    equity_curve.append({"time": times[-1], "equity": equity})

    net_profit = equity - initial_capital
    return_pct = (net_profit / initial_capital) * 100.0 if initial_capital != 0 else 0.0

    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = sum(t["pnl"] for t in trades if t["pnl"] < 0)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0

    equity_values = [e["equity"] for e in equity_curve]
    max_dd = compute_max_drawdown(equity_values)

    stats = {
        "initial_capital": initial_capital,
        "final_equity": equity,
        "net_profit": net_profit,
        "return_pct": return_pct,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_drawdown_pct": max_dd * 100.0
    }

    return JSONResponse(content={"trades": trades, "equity": equity_curve, "stats": stats})


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Simple simulation endpoint; in real systems require API keys, auth, signing & robust error handling.
@app.post("/place_order")
async def place_order(req: Request):
    payload = await req.json()
    # payload example: { "action":"open"|"close", "side":"long"|"short", "price": 12345.0, "time": 169... }
    # Here just log and respond that order accepted.
    print("PLACE_ORDER received:", payload)
    # You could optionally simulate exchange order id and executed price/size
    return JSONResponse(content={"ok": True, "accepted": True, "payload": payload})



from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"]
    allow_methods=["*"],
    allow_headers=["*"],
)
