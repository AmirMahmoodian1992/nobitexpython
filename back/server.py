# back/server.py
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import datetime as dt
from data_fetch import fetch_udf
from backtest import run_walkforward_intraminute, run_conventional_hourly_backtest

app = FastAPI()
UDF_URL = "https://apiv2.nobitex.ir/market/udf/history"

# -----------------------
# Endpoints
# -----------------------
@app.get("/candles")
def candles(symbol: str = Query("BTCUSDT"), resolution: str = Query("60"), days: int = Query(7)):
    now = int(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp())
    start = now - int(days) * 86400
    try:
        data = fetch_udf(symbol, resolution, start, now)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return JSONResponse(content={"candles": data})

@app.get("/intraminute_backtest")
def intraminute_backtest(
    symbol: str = Query("BTCUSDT"),
    days: int = Query(7),
    fast: int = Query(9),
    slow: int = Query(21),
    rsi_len: int = Query(14),
    rsi_ob: float = Query(65.0),
    rsi_os: float = Query(35.0),
    initial_capital: float = Query(1000.0),
    fee_pct: float = Query(0.1),
    position_size_pct: float = Query(1.0)
) -> JSONResponse:
    try:
        now = int(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp())
        start = now - int(days) * 86400
        hourly = fetch_udf(symbol, "60", start, now)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"hourly fetch failed: {e}")
    if not hourly:
        raise HTTPException(status_code=400, detail="no hourly candles returned")

    now_dt = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=dt.timezone.utc)
    hour_start_ts = int(now_dt.timestamp())
    try:
        minute_candles = fetch_udf(symbol, "1", hour_start_ts, hour_start_ts + 3600)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"minute fetch failed: {e}")

    result = run_walkforward_intraminute(
        hourly_candles=hourly,
        minute_candles_all=minute_candles,
        lookback_hours=1,
        fast_period_hours=int(fast),
        slow_period_hours=int(slow),
        rsi_period_hours=int(rsi_len),
        rsi_overbought=float(rsi_ob),
        rsi_oversold=float(rsi_os),
        initial_capital=float(initial_capital),
        fee_pct=float(fee_pct),
        position_size_pct=float(position_size_pct)
    )

    return JSONResponse(content={"hourly": hourly, "minute_current_hour": minute_candles, **result})

@app.get("/walkforward")
def walkforward(
    symbol: str = Query("BTCUSDT"),
    hours: int = Query(12),
    fast: int = Query(9),
    slow: int = Query(21),
    rsi_len: int = Query(14),
    rsi_ob: float = Query(65.0),
    rsi_os: float = Query(35.0),
    initial_capital: float = Query(1000.0),
    fee_pct: float = Query(0.1),
    position_size_pct: float = Query(1.0),
    days_back_for_history: int = Query(2),
    method: str = Query("new"),
    start_ts: int | None = Query(None),
    end_ts: int | None = Query(None)
) -> JSONResponse:
    now_ts = int(dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).timestamp())
    start_ts_fetch = now_ts - int(days_back_for_history) * 86400
    try:
        hourly = fetch_udf(symbol, "60", start_ts_fetch, now_ts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"hourly fetch failed: {e}")
    if not hourly:
        raise HTTPException(status_code=400, detail="no hourly candles returned")

    latest_hour_start = hourly[-1]["time"] - (hourly[-1]["time"] % 3600)
    earliest_hour_start = latest_hour_start - (hours - 1) * 3600

    if method == "conventional":
        result = run_conventional_hourly_backtest(
            hourly_candles=hourly,
            fast_period=int(fast),
            slow_period=int(slow),
            rsi_period=int(rsi_len),
            rsi_overbought=float(rsi_ob),
            rsi_oversold=float(rsi_os),
            initial_capital=float(initial_capital),
            fee_pct=float(fee_pct),
            position_size_pct=float(position_size_pct),
            start_ts=start_ts,
            end_ts=end_ts
        )
        return JSONResponse(content={
            "method": "conventional",
            "hourly_window_start": earliest_hour_start,
            "hourly_window_end": latest_hour_start + 3600,
            "hourly": hourly,
            **result
        })

    # default/new intraminute method
    try:
        minute_all = fetch_udf(symbol, "1", earliest_hour_start, latest_hour_start + 3600)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"minute fetch failed: {e}")

    result = run_walkforward_intraminute(
        hourly_candles=hourly,
        minute_candles_all=minute_all,
        lookback_hours=hours,
        fast_period_hours=int(fast),
        slow_period_hours=int(slow),
        rsi_period_hours=int(rsi_len),
        rsi_overbought=float(rsi_ob),
        rsi_oversold=float(rsi_os),
        initial_capital=float(initial_capital),
        fee_pct=float(fee_pct),
        position_size_pct=float(position_size_pct)
    )

    return JSONResponse(content={
        "method": "new",
        "hourly_window_start": earliest_hour_start,
        "hourly_window_end": latest_hour_start + 3600,
        "hourly": hourly,
        **result
    })

@app.post("/place_order")
async def place_order(req: Request):
    payload = await req.json()
    print("PLACE_ORDER received:", payload)
    return JSONResponse(content={"ok": True, "accepted": True, "payload": payload})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)