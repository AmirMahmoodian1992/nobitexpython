# app/backtest.py
from typing import List, Dict, Any, Optional
from indicators import compute_ema_from_series, compute_wilder_avgs_from_series, compute_ema_series, compute_rsi_series
from helpers import compute_max_drawdown, pnl_for_close, detect_crossover_by_minute_candidates

# -----------------------
# Walk-forward intraminute simulator (unchanged)
# -----------------------
def run_walkforward_intraminute(
    hourly_candles: List[Dict[str, Any]],
    minute_candles_all: List[Dict[str, Any]],
    lookback_hours: int = 12,
    fast_period_hours: int = 9,
    slow_period_hours: int = 21,
    rsi_period_hours: int = 14,
    rsi_overbought: float = 65.0,
    rsi_oversold: float = 35.0,
    initial_capital: float = 1000.0,
    fee_pct: float = 0.1,
    position_size_pct: float = 1.0
) -> Dict[str, Any]:
    # (identical to your earlier implementation)
    minute_by_hour: Dict[int, List[Dict[str,Any]]] = {}
    for mb in minute_candles_all:
        t = int(mb["time"])
        hour_start = t - (t % 3600)
        minute_by_hour.setdefault(hour_start, []).append(mb)
    for k in minute_by_hour:
        minute_by_hour[k].sort(key=lambda x: int(x["time"]))

    n_hours = len(hourly_candles)
    if n_hours == 0:
        return {"error": "no hourly candles"}

    min_seed = max(fast_period_hours, slow_period_hours, rsi_period_hours)
    start_idx = max(min_seed, max(0, n_hours - lookback_hours))
    end_idx = n_hours

    trades: List[Dict[str,Any]] = []
    markers: List[Dict[str,Any]] = []
    equity_curve: List[Dict[str,Any]] = []

    fee_factor = fee_pct / 100.0
    capital = float(initial_capital)
    position = None

    hourly_closes = [float(h["close"]) for h in hourly_candles]

    for i in range(start_idx, end_idx):
        hour = hourly_candles[i]
        hour_start = int(hour["time"])
        prior_hourly_closes = hourly_closes[:i]
        ema_fast_seed = compute_ema_from_series(prior_hourly_closes, fast_period_hours)
        ema_slow_seed = compute_ema_from_series(prior_hourly_closes, slow_period_hours)
        avgUp_seed, avgDown_seed = compute_wilder_avgs_from_series(prior_hourly_closes, rsi_period_hours)
        last_hour_close = prior_hourly_closes[-1] if prior_hourly_closes else hourly_closes[max(0, i-1)]

        minute_bars = minute_by_hour.get(hour_start, [])
        minute_bars.sort(key=lambda x: int(x["time"]))

        candidate = detect_crossover_by_minute_candidates(
            ema_fast_seed=ema_fast_seed,
            ema_slow_seed=ema_slow_seed,
            avgUp_seed=avgUp_seed,
            avgDown_seed=avgDown_seed,
            last_hour_close=last_hour_close,
            minute_bars=minute_bars,
            fast_period_hours=fast_period_hours,
            slow_period_hours=slow_period_hours,
            rsi_period_hours=rsi_period_hours,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold
        )

        if minute_bars:
            for mb in minute_bars:
                p = float(mb["close"])
                t = int(mb["time"])

                if candidate is not None and t == int(candidate["time"]):
                    signal = candidate["signal"]
                    exec_price = candidate["price"]
                    exec_time = candidate["time"]
                    position_notional = capital * float(position_size_pct)
                    units = position_notional / exec_price if exec_price > 0 else 0.0
                    entry_fee = units * exec_price * fee_factor

                    if position is None:
                        position = {"side": signal, "entry_time": exec_time, "entry_price": exec_price, "units": units, "entry_fee": entry_fee}
                        markers.append({"time": exec_time, "position":"belowBar", "color":"green" if signal=="long" else "red", "shape":"circle", "text":"L" if signal=="long" else "S"})
                    elif position["side"] != signal:
                        pnl, exit_fee = pnl_for_close(position, exec_price, fee_factor)
                        trade = {
                            "side": "Long" if position["side"]=="long" else "Short",
                            "entry_time": position["entry_time"], "entry_price": position["entry_price"],
                            "exit_time": exec_time, "exit_price": exec_price,
                            "units": position["units"], "entry_fee": position["entry_fee"],
                            "exit_fee": exit_fee, "pnl": pnl
                        }
                        trades.append(trade)
                        markers.append({"time": exec_time, "position":"aboveBar", "color":"green" if position["side"]=="long" else "red", "shape":"circle", "text":"L" if position["side"]=="long" else "S"})
                        capital += pnl
                        position = {"side": signal, "entry_time": exec_time, "entry_price": exec_price, "units": units, "entry_fee": entry_fee}
                        markers.append({"time": exec_time, "position":"belowBar", "color":"green" if signal=="long" else "red", "shape":"circle", "text":"L" if signal=="long" else "S"})
                    else:
                        pass

                if position is None:
                    equity = capital
                else:
                    if position["side"] == "long":
                        unreal = position["units"] * (p - position["entry_price"]) - position.get("entry_fee", 0.0)
                    else:
                        unreal = (position["entry_price"] - p) * position["units"] - position.get("entry_fee", 0.0)
                    equity = capital + unreal
                equity_curve.append({"time": t, "equity": equity})
        else:
            p = float(hour["close"])
            t = int(hour["time"]) + 3599
            if position is None:
                equity = capital
            else:
                if position["side"] == "long":
                    unreal = position["units"] * (p - position["entry_price"]) - position.get("entry_fee", 0.0)
                else:
                    unreal = (position["entry_price"] - p) * position["units"] - position.get("entry_fee", 0.0)
                equity = capital + unreal
            equity_curve.append({"time": t, "equity": equity})

    net_profit = (equity_curve[-1]["equity"] - float(initial_capital)) if equity_curve else 0.0
    stats = {
        "initial_capital": float(initial_capital),
        "final_equity": equity_curve[-1]["equity"] if equity_curve else float(initial_capital),
        "net_profit": net_profit,
        "trades": len(trades),
        "max_drawdown_pct": compute_max_drawdown([e["equity"] for e in equity_curve]) * 100.0 if equity_curve else 0.0
    }
    return {"trades": trades, "markers": markers, "equity": equity_curve, "stats": stats}

# -----------------------
# Conventional hourly backtest (PineScript-like behaviour)
# -----------------------
def run_conventional_hourly_backtest(
    hourly_candles: List[Dict[str, Any]],
    fast_period: int = 9,
    slow_period: int = 21,
    rsi_period: int = 14,
    rsi_overbought: float = 65.0,
    rsi_oversold: float = 35.0,
    initial_capital: float = 1000.0,
    fee_pct: float = 0.1,
    position_size_pct: float = 1.0,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None
) -> Dict[str, Any]:
    """
    Emulates the Pine strategy provided:
      - uses crossover on EMA series
      - uses Wilder RSI threshold on the crossover bar (long requires rsi > rsi_oversold, short requires rsi < rsi_overbought)
      - executes the signal on the next bar's OPEN price
      - at execution close any existing position first, update capital, then open the new position with full equity (position_size_pct)
    """
    n = len(hourly_candles)
    closes = [float(h["close"]) for h in hourly_candles]
    times = [int(h["time"]) for h in hourly_candles]
    opens = [float(h["open"]) for h in hourly_candles]

    fast_ema = compute_ema_series(closes, int(fast_period))
    slow_ema = compute_ema_series(closes, int(slow_period))
    rsi = compute_rsi_series(closes, int(rsi_period))

    # compute crossover booleans at bar i (based on fast_ema & slow_ema at i)
    crossover_up = [False] * n
    crossover_down = [False] * n
    prev_diff = None
    for i in range(n):
        f = fast_ema[i] if i < len(fast_ema) else None
        s = slow_ema[i] if i < len(slow_ema) else None
        if f is None or s is None:
            prev_diff = None if f is None or s is None else (f - s)
            continue
        diff = f - s
        if prev_diff is not None:
            if prev_diff <= 0 and diff > 0:
                crossover_up[i] = True
            if prev_diff >= 0 and diff < 0:
                crossover_down[i] = True
        prev_diff = diff

    # evaluate longSignalNow / shortSignalNow on bar i: requires allowTrade on bar i
    long_now = [False] * n
    short_now = [False] * n
    for i in range(n):
        allowed = True
        if start_ts is not None and times[i] < start_ts:
            allowed = False
        if end_ts is not None and times[i] > end_ts:
            allowed = False
        # rsi may be None; require defined
        r = rsi[i] if i < len(rsi) else None
        if crossover_up[i] and (r is not None) and (r > rsi_oversold) and allowed:
            long_now[i] = True
        if crossover_down[i] and (r is not None) and (r < rsi_overbought) and allowed:
            short_now[i] = True

    # shift signals by 1: entry happens on next bar open
    long_signal = [False] * n
    short_signal = [False] * n
    for i in range(1, n):
        if long_now[i-1]:
            long_signal[i] = True
        if short_now[i-1]:
            short_signal[i] = True

    # run simulation executing at bar i open when long_signal[i] or short_signal[i] is True
    fee_factor = fee_pct / 100.0
    capital = float(initial_capital)
    position = None
    trades = []
    markers = []
    equity_curve = []

    for i in range(n):
        t = times[i]
        open_price = opens[i]
        close_price = closes[i]

        # execute at open if signal present
        if long_signal[i] or short_signal[i]:
            signal = "long" if long_signal[i] else "short"
            exec_price = float(open_price)
            exec_time = int(t)
            # close any existing position first (mimic strategy.close calls)
            if position is not None:
                pnl, exit_fee = pnl_for_close(position, exec_price, fee_factor)
                trade = {
                    "side": "Long" if position["side"]=="long" else "Short",
                    "entry_time": position["entry_time"], "entry_price": position["entry_price"],
                    "exit_time": exec_time, "exit_price": exec_price,
                    "units": position["units"], "entry_fee": position["entry_fee"],
                    "exit_fee": exit_fee, "pnl": pnl
                }
                trades.append(trade)
                markers.append({"time": exec_time, "position":"aboveBar", "color":"green" if position["side"]=="long" else "red", "shape":"circle", "text":"L" if position["side"]=="long" else "S"})
                capital += pnl
                position = None

            # now open new position using updated capital
            position_notional = capital * float(position_size_pct)
            units = position_notional / exec_price if exec_price > 0 else 0.0
            entry_fee = units * exec_price * fee_factor
            position = {"side": signal, "entry_time": exec_time, "entry_price": exec_price, "units": units, "entry_fee": entry_fee}
            markers.append({"time": exec_time, "position":"belowBar", "color":"green" if signal=="long" else "red", "shape":"circle", "text":"L" if signal=="long" else "S"})

        # mark-to-market at end of hour (use close price)
        t_equity_time = int(t) + 3599
        if position is None:
            equity = capital
        else:
            if position["side"] == "long":
                unreal = position["units"] * (close_price - position["entry_price"]) - position.get("entry_fee", 0.0)
            else:
                unreal = (position["entry_price"] - close_price) * position["units"] - position.get("entry_fee", 0.0)
            equity = capital + unreal
        equity_curve.append({"time": t_equity_time, "equity": equity})

    stats = {
        "initial_capital": float(initial_capital),
        "final_equity": equity_curve[-1]["equity"] if equity_curve else float(initial_capital),
        "net_profit": (equity_curve[-1]["equity"] - float(initial_capital)) if equity_curve else 0.0,
        "trades": len(trades),
        "max_drawdown_pct": compute_max_drawdown([e["equity"] for e in equity_curve]) * 100.0 if equity_curve else 0.0
    }

    # also return helper arrays for frontend
    return {
        "trades": trades,
        "markers": markers,
        "equity": equity_curve,
        "stats": stats,
        "fast_ema": fast_ema,
        "slow_ema": slow_ema,
        "rsi": rsi,
        "crosses": [{"index": i, "time": times[i], "up": crossover_up[i], "down": crossover_down[i]} for i in range(n)],
        "signals": [{"index": i, "time": times[i], "long_now": long_now[i], "short_now": short_now[i], "long_signal": long_signal[i], "short_signal": short_signal[i]} for i in range(n)]
    }