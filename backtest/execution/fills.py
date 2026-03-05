"""
=============================================================================
BACKTESTING ENGINE — OVERALL SYSTEM GOAL
=============================================================================
This system takes historical 1-minute price data for a futures instrument,
runs a trading strategy against it, simulates realistic trade execution,
and produces a full performance report. The goal is to answer one question
honestly: does this strategy make money across different market conditions,
or does it only work sometimes?

The engine is organized into five layers that never import from each other
except through the master engine (backtest/engine.py):

    DATA LAYER        backtest/data/loader.py
    INDICATORS LAYER  backtest/indicators/volatility.py, regime.py
    STRATEGY LAYER    backtest/strategies/base.py, orb.py
    EXECUTION LAYER   backtest/execution/fills.py              <-- YOU ARE HERE
    METRICS LAYER     backtest/metrics/report.py

=============================================================================
THIS FILE — backtest/execution/fills.py
=============================================================================
Simulates realistic trade execution from a list of Signal objects. Enforces
all trade rules: next-bar-open entry, pessimistic stop-before-target check,
the second-trade-per-day filter, and EOD forced exits at 16:00.

Produces a complete trade log as a list of TradeRecord objects (one per
executed trade) and a pandas DataFrame version of the same log, ready for
the metrics layer.

HOW IT CONNECTS:
  - Receives Signal objects from backtest/strategies/base.py (via engine.py).
  - Receives the clean 1-minute DataFrame from backtest/data/loader.py
    (via engine.py).
  - Receives the daily indicators table from backtest/indicators/regime.py
    (via engine.py) to enrich each trade record with ATR, ADX, and regime.
  - Passes the completed trade log DataFrame to backtest/metrics/report.py
    (via engine.py).

KEY DESIGN RULES ENFORCED HERE:
  1. Entry on NEXT BAR OPEN after signal bar — never the signal bar's close.
  2. Stop checked BEFORE target within the same bar (pessimistic assumption).
  3. Second trade per day only allowed if the first trade of that day won.
  4. EOD forced exit at 16:00 — no positions held overnight.
  5. Slippage applied on entry, on stop exits, and on EOD exits.
     Target exits are limit orders — filled at exact target price, no slippage.

INSTRUMENT SPECIFICATIONS (hardcoded per spec):
  MES — Micro E-mini S&P 500   tick=$0.25, tick_value=$1.25,  commission=$0.35
  MGC — Micro Gold             tick=$0.10, tick_value=$1.00,  commission=$0.50
  MCL — Micro Crude Oil        tick=$0.01, tick_value=$1.00,  commission=$0.50
  MNQ — Micro E-mini Nasdaq-100 tick=$0.25, tick_value=$0.50, commission=$0.35
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtest.strategies.base import Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instrument specifications
# ---------------------------------------------------------------------------

# Each instrument's tick is the minimum allowed price increment.
# Tick value is the dollar profit/loss for a one-tick move on ONE contract.
# Commission is per contract per SIDE (so multiply by 2 for a round trip).
# Slippage_ticks is how many ticks of adverse fill we assume on market orders.
#
# WARNING: These specs are for MICRO contracts (MES, MGC, MCL), not the
# full-size contracts (ES, GC, CL). A full-size ES contract has a tick value
# of $12.50 — ten times larger than MES. Using the wrong spec will produce
# dollar P&L values that are off by a factor of ten.
INSTRUMENT_SPECS: dict[str, dict[str, float]] = {
    "MES": {"tick": 0.25,  "tick_value": 1.25, "commission": 0.35, "slippage_ticks": 1},
    "MGC": {"tick": 0.10,  "tick_value": 1.00, "commission": 0.50, "slippage_ticks": 1},
    "MCL": {"tick": 0.01,  "tick_value": 1.00, "commission": 0.50, "slippage_ticks": 2},

    # MNQ — Micro E-mini Nasdaq-100
    # Tick size  : 0.25 index points (same as full-size NQ)
    # Tick value : $0.50 per tick per contract.
    #              Full-size NQ is $5.00/tick; MNQ is exactly 1/10th,
    #              so 1 full Nasdaq-100 index point = $2.00 on MNQ.
    # Commission : $0.35/side — standard micro futures rate.
    # Slippage   : 1 tick. MNQ is liquid during the regular session
    #              (09:30–16:00 ET). Pre/post-market is much thinner;
    #              if you ever trade those sessions, increase to 2–3 ticks.
    #
    # WARNING: A 100-point Nasdaq move = $200 per MNQ contract.
    # Confusing MNQ with NQ (which would be $2,000) is a factor-of-10
    # P&L error. Always verify tick_value before running a live strategy.
    "MNQ": {"tick": 0.25,  "tick_value": 0.50, "commission": 0.35, "slippage_ticks": 1},
}

# ---------------------------------------------------------------------------
# Exit reason labels — used in the trade log and metrics filter
# ---------------------------------------------------------------------------

EXIT_STOP:   str = "stop"    # hit stop loss
EXIT_TARGET: str = "target"  # hit profit target
EXIT_EOD:    str = "eod"     # forced out at end of session

# ---------------------------------------------------------------------------
# Session boundary
# ---------------------------------------------------------------------------

# All positions are force-closed at or after this time.
# We match the session_end used in the indicators layer so that "EOD" in the
# trade log aligns with the last ATR snapshot of the day.
SESSION_FORCED_EXIT_TIME: str = "16:00"

# Maximum trades allowed per day. The second-trade filter enforces that the
# second trade is only taken if the first won. A third trade is never taken.
MAX_TRADES_PER_DAY: int = 2


# ---------------------------------------------------------------------------
# TradeRecord dataclass — one instance per executed trade
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """
    Immutable record of one fully simulated trade, from signal to exit.

    Every field that the metrics layer needs is stored here. The dataclass
    is intentionally flat (no nested objects) so it can be converted to a
    pandas DataFrame row with a single dict() call.

    Fields derived from the signal
    --------------------------------
    stop_price, target_price : the levels set by the strategy.

    Fields added by the execution layer
    -------------------------------------
    entry_price, exit_price : include slippage adjustments.
    gross_pnl               : dollar P&L before commission.
    commission              : total round-trip commission in dollars.
    net_pnl                 : gross_pnl minus commission.
    r_multiple              : net_pnl expressed as multiples of initial dollar risk.
    won                     : True if net_pnl > 0.
    exit_reason             : "stop", "target", or "eod".

    Fields enriched from daily indicators
    ----------------------------------------
    daily_atr, adx_value, regime : from the daily indicators table.

    Strategy-specific fields
    -------------------------
    orb_high, orb_low : populated by the ORB strategy via Signal.metadata.
                        None for non-ORB strategies.
    """
    trade_id:     int
    date:         str          # ISO date string "YYYY-MM-DD"
    entry_time:   str          # ISO datetime string
    exit_time:    str          # ISO datetime string
    direction:    str          # "long" or "short"
    entry_price:  float
    exit_price:   float
    stop_price:   float
    target_price: float
    contracts:    int
    gross_pnl:    float
    commission:   float
    net_pnl:      float
    exit_reason:  str
    r_multiple:   float
    won:          bool
    day_of_week:  str          # "Monday", "Tuesday", etc.
    hour_of_entry: int         # 0-23
    daily_atr:    float | None
    adx_value:    float | None
    regime:       str | None
    orb_high:     float | None
    orb_low:      float | None

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary for DataFrame construction."""
        return {
            "trade_id":      self.trade_id,
            "date":          self.date,
            "entry_time":    self.entry_time,
            "exit_time":     self.exit_time,
            "direction":     self.direction,
            "entry_price":   self.entry_price,
            "exit_price":    self.exit_price,
            "stop_price":    self.stop_price,
            "target_price":  self.target_price,
            "contracts":     self.contracts,
            "gross_pnl":     self.gross_pnl,
            "commission":    self.commission,
            "net_pnl":       self.net_pnl,
            "exit_reason":   self.exit_reason,
            "r_multiple":    self.r_multiple,
            "won":           self.won,
            "day_of_week":   self.day_of_week,
            "hour_of_entry": self.hour_of_entry,
            "daily_atr":     self.daily_atr,
            "adx_value":     self.adx_value,
            "regime":        self.regime,
            "orb_high":      self.orb_high,
            "orb_low":       self.orb_low,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_trades(
    data: pd.DataFrame,
    signals: list[Signal],
    daily_indicators: pd.DataFrame,
    instrument: str,
    contracts: int,
) -> pd.DataFrame:
    """
    Simulate all signals through the execution rules and return a trade log.

    Parameters
    ----------
    data : pd.DataFrame
        Clean 1-minute OHLCV DataFrame (DatetimeIndex). From loader.py.

    signals : list[Signal]
        All signals generated by the strategy. Must be sorted by bar_index
        ascending. The engine (engine.py) is responsible for this ordering.

    daily_indicators : pd.DataFrame
        One row per trading day with columns: atr, adx, regime.
        From regime.py. Used to enrich each trade record.

    instrument : str
        One of "MES", "MGC", "MCL". Selects the correct INSTRUMENT_SPECS row.

    contracts : int
        Number of contracts to trade on each signal.

    Returns
    -------
    pd.DataFrame
        One row per executed trade. All columns match the TradeRecord fields.
        Returns an empty DataFrame (with the correct columns) if no trades
        were executed.

    Raises
    ------
    ValueError
        If instrument is not in INSTRUMENT_SPECS.
    """
    if instrument not in INSTRUMENT_SPECS:
        raise ValueError(
            f"Unknown instrument '{instrument}'. "
            f"Valid options: {list(INSTRUMENT_SPECS.keys())}"
        )
    if contracts < 1:
        raise ValueError(f"contracts must be at least 1, got {contracts}.")

    spec = INSTRUMENT_SPECS[instrument]
    eod_time = pd.Timestamp(f"1970-01-01 {SESSION_FORCED_EXIT_TIME}").time()

    # Sort signals chronologically (defensive — strategy should already do this).
    signals_sorted = sorted(signals, key=lambda s: s.bar_index)

    # Group signals by trading date so we can enforce the per-day trade limit.
    # We use the DATE of the signal bar (bar_index), not the entry bar,
    # because signals are generated and counted on the day they fire.
    signals_by_date: dict[str, list[Signal]] = {}
    for sig in signals_sorted:
        if sig.bar_index >= len(data):
            logger.warning(
                f"Signal with bar_index={sig.bar_index} is out of range "
                f"(data has {len(data)} rows). Skipping."
            )
            continue
        sig_date = str(data.index[sig.bar_index].date())
        signals_by_date.setdefault(sig_date, []).append(sig)

    records: list[TradeRecord] = []
    trade_id = 1

    for date_str in sorted(signals_by_date.keys()):
        day_signals = signals_by_date[date_str]
        day_trade_count = 0
        first_trade_won: bool | None = None

        for signal in day_signals:
            # ------------------------------------------------------------------
            # Second-trade filter
            # ------------------------------------------------------------------
            # Rule: a second trade on the same day is only permitted if the
            # first trade of that day was a winner.
            #
            # Why this rule exists: if the market has already stopped you out
            # once today, conditions are working against your strategy. Taking
            # another trade multiplies your exposure to a bad day. This rule
            # caps the daily loss at one full stop-out rather than two.
            if day_trade_count >= MAX_TRADES_PER_DAY:
                break
            if day_trade_count == 1 and first_trade_won is False:
                break

            # ------------------------------------------------------------------
            # Identify the entry bar (bar_index + 1)
            # ------------------------------------------------------------------
            entry_bar_idx = signal.bar_index + 1

            if entry_bar_idx >= len(data):
                # Signal fired on the very last bar of the dataset — no bar
                # exists to enter on. Skip.
                logger.debug(f"Signal on last bar of dataset, skipping.")
                continue

            entry_bar_time = data.index[entry_bar_idx]

            # Entry must be on the same calendar date as the signal.
            # If the signal fired on the last bar of the trading day, bar + 1
            # is the next morning's first bar — we must NOT enter that trade.
            if str(entry_bar_time.date()) != date_str:
                logger.debug(
                    f"Signal on {date_str} has no same-day entry bar. Skipping."
                )
                continue

            # ------------------------------------------------------------------
            # Simulate the trade
            # ------------------------------------------------------------------
            record = _simulate_single_trade(
                data=data,
                signal=signal,
                entry_bar_idx=entry_bar_idx,
                daily_indicators=daily_indicators,
                spec=spec,
                contracts=contracts,
                trade_id=trade_id,
                date_str=date_str,
                eod_time=eod_time,
            )

            if record is None:
                # Should not happen; _simulate_single_trade always returns a
                # record or raises. Guard against unexpected None returns.
                logger.warning(f"Trade simulation returned None for signal at "
                               f"bar {signal.bar_index}. Skipping.")
                continue

            records.append(record)
            trade_id += 1
            day_trade_count += 1

            if day_trade_count == 1:
                first_trade_won = record.won

    logger.info(
        f"Execution complete. {len(records)} trades simulated across "
        f"{len(signals_by_date)} signal days."
    )

    if not records:
        # Return an empty DataFrame with the correct column names so
        # downstream code (metrics layer) can always assume the schema exists.
        return pd.DataFrame(columns=list(TradeRecord.__dataclass_fields__.keys()))

    return pd.DataFrame([r.to_dict() for r in records])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _simulate_single_trade(
    data: pd.DataFrame,
    signal: Signal,
    entry_bar_idx: int,
    daily_indicators: pd.DataFrame,
    spec: dict[str, float],
    contracts: int,
    trade_id: int,
    date_str: str,
    eod_time: object,   # datetime.time
) -> TradeRecord:
    """
    Simulate one trade from entry to exit, bar by bar.

    Execution sequence for each bar after entry:
        1. Check if this bar's time is at or after SESSION_FORCED_EXIT_TIME.
           If yes → EOD exit at this bar's open (with adverse slippage).
        2. Check if stop is hit (pessimistic: stop before target).
        3. Check if target is hit.
        4. If neither: move to the next bar.

    Parameters
    ----------
    data : pd.DataFrame
        Full 1-minute OHLCV DataFrame.
    signal : Signal
        The signal being executed.
    entry_bar_idx : int
        iloc index of the entry bar (signal.bar_index + 1).
    daily_indicators : pd.DataFrame
        Daily ATR/ADX/regime table (DatetimeIndex at midnight).
    spec : dict
        Instrument spec (tick, tick_value, commission, slippage_ticks).
    contracts : int
        Number of contracts.
    trade_id : int
        Unique ID for this trade (assigned sequentially by simulate_trades).
    date_str : str
        ISO date string of the signal date.
    eod_time : datetime.time
        The time object representing the forced-exit boundary.

    Returns
    -------
    TradeRecord
        Fully populated trade record.
    """
    tick            = spec["tick"]
    tick_value      = spec["tick_value"]
    commission_side = spec["commission"]
    slippage        = spec["slippage_ticks"] * tick

    # --- Entry price ---
    # We enter at the OPEN of the entry bar plus adverse slippage.
    # Long entry: we're buying, so we get filled ABOVE the open (worse for us).
    # Short entry: we're selling, so we get filled BELOW the open (worse for us).
    raw_entry_open = data.iloc[entry_bar_idx]["open"]
    if signal.direction == "long":
        entry_price = raw_entry_open + slippage
    else:
        entry_price = raw_entry_open - slippage

    entry_timestamp = data.index[entry_bar_idx]

    exit_price:  float | None = None
    exit_time:   pd.Timestamp | None = None
    exit_reason: str | None = None

    # --- Bar-by-bar simulation loop ---
    # We start AT the entry bar (bar_index + 1) because even on the entry
    # bar itself, the high/low can cross our stop or target after the open.
    # Example: we enter at the bar's open (4802), stop at 4796. If the bar's
    # low is 4793, we're stopped out on the entry bar itself.
    for i in range(entry_bar_idx, len(data)):
        bar       = data.iloc[i]
        bar_time  = data.index[i]
        bar_lo    = bar["low"]
        bar_hi    = bar["high"]

        # --- Rule 4: EOD forced exit ---
        # If this bar starts at or after the session close, or if we've
        # crossed into the next calendar day, exit immediately at this bar's
        # open with adverse slippage.
        #
        # We check the calendar date too (bar_time.date() != date_str) to
        # handle the case where data includes overnight bars: a bar at 18:00
        # on the same calendar day should NOT trigger an EOD exit for a 16:00
        # session close if the overnight session is filtered out. The date
        # check is a belt-and-suspenders guard.
        is_eod = (bar_time.time() >= eod_time) or (str(bar_time.date()) != date_str)

        if is_eod:
            # Exit at bar open with adverse slippage (market order).
            raw_eod_open = bar["open"]
            if signal.direction == "long":
                exit_price = raw_eod_open - slippage   # selling → worse = lower
            else:
                exit_price = raw_eod_open + slippage   # buying back → worse = higher
            exit_time   = bar_time
            exit_reason = EXIT_EOD
            break

        # --- Rule 2: Check stop BEFORE target (pessimistic assumption) ---
        # Within a single 1-minute bar we cannot know if price reached the
        # high or the low first. We always assume the worst: stop is hit
        # before target. This is the honest, conservative approach.
        #
        # Long trade:  stop is below entry, target is above. Stop hit if
        #              bar's low dips to or below stop_price.
        # Short trade: stop is above entry, target is below. Stop hit if
        #              bar's high rises to or above stop_price.
        if signal.direction == "long":
            stop_hit   = bar_lo <= signal.stop_price
            target_hit = bar_hi >= signal.target_price
        else:
            stop_hit   = bar_hi >= signal.stop_price
            target_hit = bar_lo <= signal.target_price

        if stop_hit:
            # Stopped out. Actual fill includes adverse slippage beyond the
            # stop level (price moved past the stop before filling).
            if signal.direction == "long":
                exit_price = signal.stop_price - slippage
            else:
                exit_price = signal.stop_price + slippage
            exit_time   = bar_time
            exit_reason = EXIT_STOP
            break

        # --- Rule 3: Check target ---
        # Target is a resting limit order. Limit orders fill at the limit
        # price or better — no adverse slippage. We assume exact fill.
        if target_hit:
            exit_price  = signal.target_price
            exit_time   = bar_time
            exit_reason = EXIT_TARGET
            break

    # Safety net: if we somehow exit the loop without filling (data ends
    # mid-trade), force-exit at the last bar's close.
    if exit_price is None:
        last_bar   = data.iloc[-1]
        exit_price = last_bar["close"]
        exit_time  = data.index[-1]
        exit_reason = EXIT_EOD
        logger.warning(
            f"Trade {trade_id} ran to end of dataset without an exit. "
            f"Force-closed at last bar close {exit_price}."
        )

    # --- P&L calculation ---
    gross_pnl, commission, net_pnl, r_multiple = _compute_pnl(
        direction=signal.direction,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_price=signal.stop_price,
        tick=tick,
        tick_value=tick_value,
        commission_side=commission_side,
        contracts=contracts,
    )

    # --- Enrich with daily indicators ---
    # Look up by the normalized date timestamp (midnight on trade date).
    date_ts = pd.Timestamp(date_str)
    if date_ts in daily_indicators.index:
        day_row   = daily_indicators.loc[date_ts]
        daily_atr = float(day_row["atr"])
        adx_value = float(day_row["adx"])
        regime    = str(day_row["regime"])
    else:
        daily_atr = None
        adx_value = None
        regime    = None

    return TradeRecord(
        trade_id     = trade_id,
        date         = date_str,
        entry_time   = str(entry_timestamp),
        exit_time    = str(exit_time),
        direction    = signal.direction,
        entry_price  = round(entry_price, 4),
        exit_price   = round(exit_price,  4),
        stop_price   = round(signal.stop_price,   4),
        target_price = round(signal.target_price, 4),
        contracts    = contracts,
        gross_pnl    = round(gross_pnl,   2),
        commission   = round(commission,  2),
        net_pnl      = round(net_pnl,     2),
        exit_reason  = exit_reason,
        r_multiple   = round(r_multiple,  4),
        won          = net_pnl > 0,
        day_of_week  = entry_timestamp.strftime("%A"),
        hour_of_entry= entry_timestamp.hour,
        daily_atr    = daily_atr,
        adx_value    = adx_value,
        regime       = regime,
        orb_high     = signal.metadata.get("orb_high"),
        orb_low      = signal.metadata.get("orb_low"),
    )


def _compute_pnl(
    direction:       str,
    entry_price:     float,
    exit_price:      float,
    stop_price:      float,
    tick:            float,
    tick_value:      float,
    commission_side: float,
    contracts:       int,
) -> tuple[float, float, float, float]:
    """
    Calculate gross P&L, commission, net P&L, and R-multiple for one trade.

    How futures P&L works
    ----------------------
    Futures P&L is not simply (exit - entry) × contracts × price, because
    the dollar value of a one-point move depends on the instrument's tick size
    and tick value.

    For MES: tick = 0.25 points, tick_value = $1.25 per tick per contract.
    A 1-point move = (1 / 0.25) ticks = 4 ticks = 4 × $1.25 = $5.00 per contract.

    General formula:
        gross_pnl = (price_diff_in_points / tick) × tick_value × contracts

    R-multiple
    ----------
    R is the dollar risk you accepted when entering the trade:
        R = |entry_price - stop_price| / tick × tick_value × contracts

    R-multiple is the trade's gross P&L expressed as a multiple of R:
        r_multiple = gross_pnl / R

    A trade that hit its target at 1.5R returns r_multiple ≈ +1.5.
    A trade stopped out returns r_multiple ≈ -1.0 (minus commission drag).

    WARNING: We use gross_pnl (before commission) for R-multiple to match
    the convention used in trading literature. Net R-multiple would be
    slightly lower due to commission drag. The metrics layer reports both
    the raw r_multiple and net_pnl separately so nothing is hidden.

    Commission
    ----------
    commission_side is charged per contract per SIDE (entry and exit).
    Round-trip cost = commission_side × contracts × 2.

    Returns
    -------
    (gross_pnl, commission, net_pnl, r_multiple) : tuple of floats
    """
    # Price difference in points (positive = profit for correct direction).
    if direction == "long":
        price_diff = exit_price - entry_price
    else:
        price_diff = entry_price - exit_price

    # Convert points to dollars.
    gross_pnl = (price_diff / tick) * tick_value * contracts

    # Round-trip commission (entry + exit, both sides).
    commission = commission_side * contracts * 2

    net_pnl = gross_pnl - commission

    # Dollar risk at entry (before any slippage on exit — R is defined at entry).
    if direction == "long":
        dollar_risk = (entry_price - stop_price) / tick * tick_value * contracts
    else:
        dollar_risk = (stop_price - entry_price) / tick * tick_value * contracts

    # Guard against zero or negative dollar_risk (would mean stop is above/at
    # entry for a long, which the Signal validator should have caught, but
    # slippage on entry could in theory create this edge case).
    if dollar_risk <= 0:
        logger.warning(
            f"dollar_risk={dollar_risk:.4f} is non-positive. "
            f"entry={entry_price}, stop={stop_price}, direction={direction}. "
            f"R-multiple set to 0."
        )
        r_multiple = 0.0
    else:
        r_multiple = gross_pnl / dollar_risk

    return gross_pnl, commission, net_pnl, r_multiple
