# app/helpers.py
from typing import List, Dict, Any, Optional, Tuple
import math

from indicators import compute_ema_series

# -----------------------
# Helpers: math & stats
# -----------------------
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

# -----------------------
# PnL helper
# -----------------------
def pnl_for_close(position: Dict[str, Any], exit_price: float, fee_factor: float) -> Tuple[float, float]:
    """
    Compute realized PnL for closing a position and exit fee.
    position holds 'side','units','entry_price','entry_fee'.
    Returns (pnl, exit_fee).
    """
    units = position["units"]
    entry_price = position["entry_price"]
    entry_fee = position.get("entry_fee", 0.0)
    exit_fee = abs(units * exit_price) * fee_factor
    if position["side"] == "long":
        pnl = units * (exit_price - entry_price) - (entry_fee + exit_fee)
    else:
        pnl = (entry_price * units) - (exit_price * units) - (entry_fee + exit_fee)
    return float(pnl), float(exit_fee)

# -----------------------
# Candidate detector (EMA-only signals; RSI removed from gating)
# (existing implementation kept unchanged)
# -----------------------
def detect_crossover_by_minute_candidates(
    ema_fast_seed: float,
    ema_slow_seed: float,
    avgUp_seed: float,
    avgDown_seed: float,
    last_hour_close: float,
    minute_bars: List[Dict[str, Any]],
    fast_period_hours: int,
    slow_period_hours: int,
    rsi_period_hours: int,
    rsi_overbought: float,
    rsi_oversold: float
) -> Optional[Dict[str, Any]]:
    k_fast_bar = 2.0 / (fast_period_hours + 1.0)
    k_slow_bar = 2.0 / (slow_period_hours + 1.0)
    prev_diff = ema_fast_seed - ema_slow_seed

    for mb in minute_bars:
        price = float(mb["close"])
        tstamp = int(mb["time"])

        ema_fast_cand = k_fast_bar * price + (1.0 - k_fast_bar) * ema_fast_seed
        ema_slow_cand = k_slow_bar * price + (1.0 - k_slow_bar) * ema_slow_seed
        cand_diff = ema_fast_cand - ema_slow_cand

        P = rsi_period_hours
        d = price - last_hour_close
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avgUp_cand = (avgUp_seed * (P - 1) + gain) / P if P > 0 else avgUp_seed
        avgDown_cand = (avgDown_seed * (P - 1) + loss) / P if P > 0 else avgDown_seed
        rsi_cand = 100.0 - 100.0 / (1.0 + (avgUp_cand / (avgDown_cand or 1e-12)))

        crossed_up = (prev_diff <= 0.0 and cand_diff > 0.0)
        crossed_down = (prev_diff >= 0.0 and cand_diff < 0.0)

        if crossed_up:
            return {
                "signal": "long",
                "time": tstamp,
                "price": price,
                "ema_fast_candidate": ema_fast_cand,
                "ema_slow_candidate": ema_slow_cand,
                "rsi_candidate": rsi_cand,
                "minute_bar": mb
            }
        if crossed_down:
            return {
                "signal": "short",
                "time": tstamp,
                "price": price,
                "ema_fast_candidate": ema_fast_cand,
                "ema_slow_candidate": ema_slow_cand,
                "rsi_candidate": rsi_cand,
                "minute_bar": mb
            }

    return None

# -----------------------
# hourly EMA cross detection (unchanged)
# -----------------------
def detect_hourly_ema_crosses(hourly_candles: List[Dict[str, Any]], fast_period: int, slow_period: int) -> List[Dict[str, Any]]:
    closes = [float(h["close"]) for h in hourly_candles]
    fast_ema_series = compute_ema_series(closes, fast_period)
    slow_ema_series = compute_ema_series(closes, slow_period)
    crosses = []
    prev_diff = None
    for i, h in enumerate(hourly_candles):
        f = fast_ema_series[i] if i < len(fast_ema_series) else None
        s = slow_ema_series[i] if i < len(slow_ema_series) else None
        if f is None or s is None:
            prev_diff = None if f is None or s is None else (f - s)
            continue
        diff = f - s
        if prev_diff is not None:
            if prev_diff <= 0 and diff > 0:
                crosses.append({"time": int(h["time"]), "type": "cross_up", "fast_ema": f, "slow_ema": s, "index": i})
            if prev_diff >= 0 and diff < 0:
                crosses.append({"time": int(h["time"]), "type": "cross_down", "fast_ema": f, "slow_ema": s, "index": i})
        prev_diff = diff
    return crosses