(async function() {
    // Helpers
    const log = (...a) => console.log(...a);
    const sleep = ms => new Promise(r => setTimeout(r, ms));

    // API for candles (matches your server)
    const apiCandles = "http://127.0.0.1:8000/candles?symbol=BTCIRT&resolution=60&days=30";

    // Chart setup
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {
        layout: { background: { color: '#071226' }, textColor: '#dfeeff' },
        timeScale: { timeVisible: true, secondsVisible: false }
    });
    const candleSeries = chart.addCandlestickSeries();
    const fastSeries = chart.addLineSeries({ color: '#58a6ff', lineWidth: 1.5 });
    const slowSeries = chart.addLineSeries({ color: '#ffb86b', lineWidth: 1.5 });

    // Fetch candles
    let candles = [];
    try {
        const r = await fetch(apiCandles);
        const j = await r.json();
        candles = (j.candles || []).map(c => ({ time: c.time, open: +c.open, high: +c.high, low: +c.low, close: +c.close }));
    } catch (e) {
        alert("Failed to fetch candles: " + e.message);
        return;
    }
    if (!candles.length) {
        alert("No candles");
        return;
    }

    // Indicator functions (EMA and RSI matching Pine Script)
    function computeEMA(values, period) {
        const out = [];
        const k = 2 / (period + 1);
        let ema = values[0];
        out.push(ema);
        for (let i = 1; i < values.length; i++) {
            ema = values[i] * k + ema * (1 - k);
            out.push(ema);
        }
        return out;
    }

    function computeRSI(prices, period = 14) {
        const out = new Array(prices.length).fill(null);
        if (prices.length < period + 1) return out;
        let avgGain = 0, avgLoss = 0;
        for (let i = 1; i <= period; i++) {
            const delta = prices[i] - prices[i - 1];
            if (delta > 0) avgGain += delta; else avgLoss -= delta;
        }
        avgGain /= period;
        avgLoss /= period;
        out[period] = 100 - 100 / (1 + avgGain / avgLoss);
        for (let i = period + 1; i < prices.length; i++) {
            const delta = prices[i] - prices[i - 1];
            const gain = delta > 0 ? delta : 0;
            const loss = -delta > 0 ? -delta : 0;
            avgGain = (avgGain * (period - 1) + gain) / period;
            avgLoss = (avgLoss * (period - 1) + loss) / period;
            out[i] = 100 - 100 / (1 + avgGain / avgLoss);
        }
        return out;
    }

    // State
    let running = false;
    let trades = [];
    let currentPos = null;
    let markers = [];

    // UI elements
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const fastIn = document.getElementById('fast');
    const slowIn = document.getElementById('slow');
    const rsiLenIn = document.getElementById('rsiLen');
    const rsiOBIn = document.getElementById('rsiOB');
    const rsiOSIn = document.getElementById('rsiOS');
    const initCapIn = document.getElementById('initCapital');
    const feeIn = document.getElementById('fee');
    const delayIn = document.getElementById('delay');
    const modeSel = document.getElementById('mode');
    const tradesTbody = document.querySelector('#tradesTable tbody');
    const summaryDiv = document.getElementById('summary');

    // Place order (for live mode)
    async function placeOrderToBackend(payload) {
        const url = 'http://127.0.0.1:8000/place_order';
        try {
            const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            if (!r.ok) throw new Error(await r.text());
            return await r.json();
        } catch (err) {
            console.error('Order error', err);
            throw err;
        }
    }

    // Run the backtest (matches Pine Script logic)
    async function runStrategy() {
        const fastPeriod = Number(fastIn.value) || 9;
        const slowPeriod = Number(slowIn.value) || 21;
        const rsiPeriod = Number(rsiLenIn.value) || 14;
        const rsiOverbought = Number(rsiOBIn.value) || 65;
        const rsiOversold = Number(rsiOSIn.value) || 35;
        const initialCapital = Number(initCapIn.value) || 1000;
        const feePct = Number(feeIn.value) || 0.1;
        const delayMs = Number(delayIn.value) || 0;
        const mode = modeSel.value;

        // Reset state
        trades = [];
        currentPos = null;
        markers = [];
        let currentCapital = initialCapital;
        candleSeries.setData([]);
        fastSeries.setData([]);
        slowSeries.setData([]);
        candleSeries.setMarkers([]);
        renderTrades();
        summaryDiv.innerHTML = '';

        // Compute indicators (shift EMA start to match Pine ta.ema which uses SMA seed, but close enough)
        const closes = candles.map(c => c.close);
        const fastEMA = computeEMA(closes, fastPeriod);
        const slowEMA = computeEMA(closes, slowPeriod);
        const rsi = computeRSI(closes, rsiPeriod);

        // Simulate bar by bar
        for (let i = 1; i < candles.length; i++) {  // Start from 1 for prev
            if (!running) break;

            // Update chart progressively
            candleSeries.setData(candles.slice(0, i + 1));
            const emaFastPoints = candles.slice(0, i + 1).map((c, idx) => ({ time: c.time, value: fastEMA[idx] }));
            const emaSlowPoints = candles.slice(0, i + 1).map((c, idx) => ({ time: c.time, value: slowEMA[idx] }));
            fastSeries.setData(emaFastPoints);
            slowSeries.setData(emaSlowPoints);
            candleSeries.setMarkers(markers.filter(m => m.time <= candles[i].time));
            if (i === candles.length - 1) chart.timeScale().fitContent();

            const fe = fastEMA[i];
            const se = slowEMA[i];
            const ri = rsi[i];
            const prevFe = fastEMA[i - 1];
            const prevSe = slowEMA[i - 1];
            if (ri === null) {
                await sleep(delayMs);
                continue;
            }

            // Detect crossover/crossunder as in Pine Script
            let signal = null;
            const crossover = prevFe <= prevSe && fe > se;
            const crossunder = prevFe >= prevSe && fe < se;
            if (crossover && ri > rsiOversold) signal = 'long';
            else if (crossunder && ri < rsiOverbought) signal = 'short';

            if (signal) {
                const price = candles[i].close;  // Execute on close as in your initial JS
                const time = candles[i].time;

                // If opposite position, close it
                if (currentPos && currentPos.side !== signal) {
                    const exitPrice = price;
                    const exitTime = time;
                    const units = currentPos.units;
                    const exitFee = (feePct / 100) * (units * exitPrice);
                    let pnl;
                    if (currentPos.side === 'long') {
                        pnl = units * (exitPrice - currentPos.entry_price) - currentPos.entry_fee - exitFee;
                    } else {
                        pnl = units * (currentPos.entry_price - exitPrice) - currentPos.entry_fee - exitFee;
                    }
                    trades.push({
                        side: currentPos.side,
                        entry_time: currentPos.entry_time,
                        entry_price: currentPos.entry_price,
                        exit_time: exitTime,
                        exit_price: exitPrice,
                        pnl
                    });
                    currentCapital += pnl;
                    markers.push({ time: exitTime, position: 'aboveBar', color: currentPos.side === 'long' ? 'green' : 'red', shape: 'circle', text: currentPos.side === 'long' ? 'L exit' : 'S exit' });
                    if (mode === 'live') {
                        try {
                            await placeOrderToBackend({ action: 'close', side: currentPos.side, price: exitPrice, time: exitTime });
                        } catch (e) {
                            console.error('Close order failed', e);
                        }
                    }
                    currentPos = null;
                }

                // Open new position
                if (!currentPos) {
                    const value = currentCapital;
                    const units = value / price;
                    const entryFee = (feePct / 100) * value;
                    currentPos = { side: signal, entry_time: time, entry_price: price, units, entry_fee: entryFee };
                    markers.push({ time: time, position: 'belowBar', color: signal === 'long' ? 'green' : 'red', shape: 'circle', text: signal === 'long' ? 'L entry' : 'S entry' });
                    if (mode === 'live') {
                        try {
                            await placeOrderToBackend({ action: 'open', side: signal, price, time });
                        } catch (e) {
                            console.error('Open order failed', e);
                        }
                    }
                    log(`${currentPos ? 'SWITCH to' : 'OPEN'} ${signal} at ${price} (${new Date(time * 1000).toLocaleString()})`);
                }
            }

            renderTrades();
            await sleep(delayMs);
        }

        // Close open position at end if any
        if (currentPos) {
            const exitPrice = candles[candles.length - 1].close;
            const exitTime = candles[candles.length - 1].time;
            const units = currentPos.units;
            const exitFee = (feePct / 100) * (units * exitPrice);
            let pnl;
            if (currentPos.side === 'long') {
                pnl = units * (exitPrice - currentPos.entry_price) - currentPos.entry_fee - exitFee;
            } else {
                pnl = units * (currentPos.entry_price - exitPrice) - currentPos.entry_fee - exitFee;
            }
            trades.push({
                side: currentPos.side,
                entry_time: currentPos.entry_time,
                entry_price: currentPos.entry_price,
                exit_time: exitTime,
                exit_price: exitPrice,
                pnl
            });
            currentCapital += pnl;
            markers.push({ time: exitTime, position: 'aboveBar', color: currentPos.side === 'long' ? 'green' : 'red', shape: 'circle', text: currentPos.side === 'long' ? 'L exit' : 'S exit' });
            if (mode === 'live') {
                try {
                    await placeOrderToBackend({ action: 'close', side: currentPos.side, price: exitPrice, time: exitTime });
                } catch (e) {
                    console.error('Close order failed', e);
                }
            }
        }

        // Render final
        candleSeries.setMarkers(markers);
        renderSummary(initialCapital, currentCapital);
        log('Backtest complete. Trades:', trades.length);
        running = false;
    }

    function renderTrades() {
        tradesTbody.innerHTML = '';
        if (!trades.length) {
            tradesTbody.innerHTML = '<tr><td colspan="5" class="muted">No trades yet</td></tr>';
            return;
        }
        let idx = 0;
        for (const t of trades) {
            idx++;
            const entryRow = document.createElement('tr');
            entryRow.innerHTML = `<td>${idx}</td><td>${t.side === 'long' ? '<span class="L">L</span>' : '<span class="S">S</span>'}</td><td>${new Date(t.entry_time * 1000).toLocaleString()}</td><td>${t.entry_price.toFixed(2)}</td><td class="${t.pnl >= 0 ? 'positive' : 'negative'}">${(t.pnl >= 0 ? '+' : '') + t.pnl.toFixed(2)}</td>`;
            tradesTbody.appendChild(entryRow);
            const exitRow = document.createElement('tr');
            exitRow.innerHTML = `<td></td><td></td><td>${new Date(t.exit_time * 1000).toLocaleString()}</td><td>${t.exit_price.toFixed(2)}</td><td></td>`;
            tradesTbody.appendChild(exitRow);
        }
    }

    function renderSummary(initial, current) {
        const totalPnl = current - initial;
        const numTrades = trades.length;
        const wins = trades.filter(t => t.pnl > 0).length;
        const winRate = numTrades ? (wins / numTrades * 100).toFixed(2) : 0;
        const avgPnl = numTrades ? (trades.reduce((sum, t) => sum + t.pnl, 0) / numTrades).toFixed(2) : 0;
        summaryDiv.innerHTML = `
            <div class="muted">Summary</div>
            <div>Total P&L: <span class="${totalPnl >= 0 ? 'positive' : 'negative'}">${(totalPnl >= 0 ? '+' : '') + totalPnl.toFixed(2)}</span></div>
            <div>Final Capital: ${current.toFixed(2)}</div>
            <div>Trades: ${numTrades}</div>
            <div>Win Rate: ${winRate}%</div>
            <div>Avg P&L: ${avgPnl}</div>
        `;
    }

    // Controls
    startBtn.addEventListener('click', () => {
        if (running) return;
        running = true;
        runStrategy();
    });
    stopBtn.addEventListener('click', () => running = false);
})();