// ui.js
import { $ } from './utils.js';

export function getUIOptions() {
  return {
    mode: $('#mode').value,
    days: Number($('#days').value) || 2,
    short: Number($('#short').value) || 9,
    long: Number($('#long').value) || 21,
    initial_capital: Number($('#initCapital').value) || 1000,
    fee_pct: Number($('#fee').value) || 0.1,
    startISO: $('#startISO').value.trim(),
    endISO: $('#endISO').value.trim()
  };
}

export function setSeedWarn(text) { $('#seedWarn').innerHTML = text; }
export function setStats(html) { $('#statsBox').innerHTML = html; }
export const runBtn = $('#runBtn');
export const stopBtn = $('#stopBtn');
