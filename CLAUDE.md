# MNQ Backtesting Engine

## What this project is
A modular, strategy-agnostic backtesting engine for futures day trading.
Built to test whether a trading strategy has a statistically real edge.
Target instrument: MNQ (Micro E-mini Nasdaq-100), 5-minute bars, 2016-02-18 to 2026-02-18.

## How to run
1. Run `prepare_data.py` once to convert the Databento zip → `MNQ_5min.csv`
2. Run `run.py` to execute the backtest
3. Results are saved to `results/` as an Excel file

## Project structure (5-layer architecture)
```
backtest/
  data/loader.py          — loads + validates MNQ_5min.csv
  indicators/
    volatility.py         — ATR (Wilder 14-period)
    regime.py             — ADX + regime labels (trending/ranging × high/low vol)
  strategies/
    base.py               — Signal dataclass + BaseStrategy ABC
    orb.py                — Opening Range Breakout strategy
  execution/fills.py      — bar-by-bar trade simulation
  metrics/report.py       — 5-tier metrics engine + Excel export
  engine.py               — master orchestrator
prepare_data.py           — one-time Databento zip → CSV converter
run.py                    — entry point
```

## Key design decisions
- **No lookahead bias**: entry always at next bar's open, never signal bar's close
- **Pessimistic fill**: stop is checked before target within the same bar
- **Second trade filter**: only allowed if first trade of the day won
- **EOD exit**: all positions closed at 16:00
- **Bar interval auto-detected**: works for 1-min, 5-min, or any timeframe

## MNQ instrument spec (fills.py)
- Tick size: 0.25, Tick value: $0.50
- Commission: $0.35/side, Slippage: 1 tick on entry/stop/EOD exits

## ORB strategy params (run.py)
- orb_minutes: 30 (first 30 min of session defines the range)
- rr: 1.5 (risk-reward ratio)
- session_open: 09:30, session_close: 11:30
- direction: "both" (takes long and short signals)
- retest: False

## Data file
`MNQ_5min.csv` is excluded from git (too large). Source zip:
`GLBX-20260218-5GBT9PUQ8Q.zip` from Databento.
Run `prepare_data.py` with the zip on each machine to regenerate it.
The script handles UTC → Eastern time conversion automatically.

## Python
Use `python` or `py` to run scripts. Project uses Python 3.14.

## User context
- First-year EE student learning Python
- Prefers plain-English explanations before writing any new code
- Wants to understand design decisions, not just see the output
