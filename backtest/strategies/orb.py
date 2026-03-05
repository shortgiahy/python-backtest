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
    STRATEGY LAYER    backtest/strategies/base.py
                      backtest/strategies/orb.py              <-- YOU ARE HERE
    EXECUTION LAYER   backtest/execution/fills.py
    METRICS LAYER     backtest/metrics/report.py

=============================================================================
THIS FILE — backtest/strategies/orb.py
=============================================================================
Implements the Opening Range Breakout (ORB) strategy as the first concrete
strategy in the system. It inherits from BaseStrategy and implements the
generate_signals() interface.

STRATEGY LOGIC (full detail):
  1. For each trading day, identify the Opening Range: all bars from
     session_open through session_open + orb_minutes (exclusive).
  2. ORB High = max(High) during the opening range.
     ORB Low  = min(Low)  during the opening range.
  3. Scan each bar from the end of the opening range up to session_close.
     ─ If retest=False (default):
       • Long signal  → bar closes ABOVE ORB High for the first time.
       • Short signal → bar closes BELOW ORB Low  for the first time.
     ─ If retest=True:
       • Long signal  → after a bar closes above ORB High, look for a
                        subsequent bar whose Low ≤ ORB High (retest of
                        the level) AND Close ≥ ORB High (still above it,
                        confirming the level as support).
       • Short signal → mirror for the downside.
  4. At most one long signal and one short signal per day.
  5. For each signal:
       stop_price   = ORB Low  (long) / ORB High (short)
       target_price = ORB High + (ORB Range × RR) for long
                      ORB Low  − (ORB Range × RR) for short
  6. entry is enforced by the execution layer on the NEXT BAR OPEN.

HOW IT CONNECTS:
  - Inherits from backtest/strategies/base.py (BaseStrategy).
  - generate_signals() returns Signal objects consumed by fills.py.
  - Signal.metadata carries orb_high and orb_low, which appear in the
    trade log and the Excel report's Trade Log sheet.
  - The engine instantiates this class and passes it to BacktestEngine.
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backtest.strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ORBStrategy
# ---------------------------------------------------------------------------

class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) strategy.

    Trades the first breakout above or below the N-minute opening range.
    Stops are placed at the opposite extreme of the opening range; targets
    are set at a fixed reward:risk multiple from the ORB level.

    Parameters (all configurable via the params dict at construction)
    ------------------------------------------------------------------
    orb_minutes : int
        Duration of the opening range in minutes. 30 is the most widely
        tested value. Shorter ranges (15) produce more signals but more
        false breakouts. Longer ranges (60) produce fewer, higher-quality
        breakouts but miss the best early moves.

    rr : float
        Reward-to-risk ratio. A value of 1.5 means the target is 1.5×
        the ORB range beyond the breakout level. Higher RR = fewer wins
        but larger wins when they occur.

    session_open : str ("HH:MM")
        The time at which the opening range begins. For US equity index
        futures during the regular session, this is "09:30".

    session_close : str ("HH:MM")
        No new signals are generated after this time. Trades entered just
        before this time are subject to EOD exit at 16:00 (enforced by
        the execution layer).

    direction : str
        "both"       — take long and short breakouts.
        "long_only"  — ignore short breakouts.
        "short_only" — ignore long breakouts.

    retest : bool
        False (default) — enter immediately on the breakout bar's close.
        True — wait for price to return and test the broken ORB level
               before entering. Produces fewer but potentially
               higher-quality signals.

    Example
    -------
    >>> from backtest.strategies.orb import ORBStrategy
    >>> strat = ORBStrategy(params={"orb_minutes": 15, "rr": 2.0, "direction": "long_only"})
    >>> print(strat.name)
    ORB_15min_RR2.0_long_only
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "orb_minutes":   30,
        "rr":            1.5,
        "session_open":  "09:30",
        "session_close": "11:30",
        "direction":     "both",
        "retest":        False,
    }

    # ------------------------------------------------------------------
    # Required implementations from BaseStrategy
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Descriptive name including the key parameters.

        Example outputs:
            "ORB_30min_RR1.5_both"
            "ORB_15min_RR2.0_long_only_retest"
        """
        p = self.params
        retest_suffix = "_retest" if p["retest"] else ""
        return f"ORB_{p['orb_minutes']}min_RR{p['rr']}_{p['direction']}{retest_suffix}"

    def validate_params(self) -> None:
        """
        Validate every parameter immediately at construction time.

        Raises ValueError with a descriptive message for any invalid value
        so errors surface before the backtest runs, not mid-run.
        """
        p = self.params

        if not isinstance(p["orb_minutes"], int) or not (1 <= p["orb_minutes"] <= 120):
            raise ValueError(
                f"orb_minutes must be an integer between 1 and 120, "
                f"got {p['orb_minutes']!r}."
            )
        if not isinstance(p["rr"], (int, float)) or p["rr"] <= 0:
            raise ValueError(
                f"rr (reward:risk ratio) must be a positive number, "
                f"got {p['rr']!r}."
            )
        if p["direction"] not in ("both", "long_only", "short_only"):
            raise ValueError(
                f"direction must be 'both', 'long_only', or 'short_only', "
                f"got {p['direction']!r}."
            )
        if not isinstance(p["retest"], bool):
            raise ValueError(
                f"retest must be True or False, got {p['retest']!r}."
            )

        # Validate time strings and their logical ordering.
        try:
            open_ts  = pd.Timestamp(f"2000-01-01 {p['session_open']}")
            close_ts = pd.Timestamp(f"2000-01-01 {p['session_close']}")
        except Exception as exc:
            raise ValueError(
                f"session_open and session_close must be in HH:MM format. "
                f"Error: {exc}"
            ) from exc

        orb_end_ts = open_ts + pd.Timedelta(minutes=p["orb_minutes"])
        if close_ts <= orb_end_ts:
            raise ValueError(
                f"session_close ({p['session_close']}) must be strictly later "
                f"than session_open ({p['session_open']}) + orb_minutes "
                f"({p['orb_minutes']}min = {orb_end_ts.time()}). "
                f"Otherwise there are no bars to trade after the ORB."
            )

    def generate_signals(
        self,
        data: pd.DataFrame,
        daily_indicators: pd.DataFrame,
    ) -> list[Signal]:
        """
        Scan the full price history and return every ORB signal.

        Parameters
        ----------
        data : pd.DataFrame
            Clean 1-minute OHLCV DataFrame from the data layer.
            DatetimeIndex. Columns: open, high, low, close, volume.

        daily_indicators : pd.DataFrame
            Daily ATR/ADX/regime table from the indicators layer.
            Not used for signal filtering in this base ORB implementation,
            but accepted as required by the BaseStrategy interface.
            A more sophisticated version could filter signals to trending
            regimes only (e.g., only trade when ADX > 25).

        Returns
        -------
        list[Signal]
            One Signal per detected ORB breakout (or retest). In
            chronological order. At most 2 signals per trading day
            (one long, one short). Returns an empty list if no
            breakouts are detected across the entire date range.
        """
        # Pre-compute time boundaries once rather than inside the day loop.
        # Using .time() objects for comparison is more reliable than string
        # comparison and avoids timezone issues.
        session_open_time  = pd.Timestamp(
            f"2000-01-01 {self.params['session_open']}"
        ).time()
        session_close_time = pd.Timestamp(
            f"2000-01-01 {self.params['session_close']}"
        ).time()
        orb_end_offset = pd.Timedelta(minutes=self.params["orb_minutes"])

        # Unique trading dates in chronological order.
        # Using np.unique() on an array of date objects is faster than
        # groupby() on large DataFrames.
        all_dates = np.unique(data.index.date)

        all_signals: list[Signal] = []
        days_with_signals = 0

        for date in all_dates:
            # np.where returns a tuple; [0] extracts the 1D integer array
            # of positions (iloc indices) where condition is True.
            date_mask   = data.index.date == date
            day_ilocs   = np.where(date_mask)[0]

            if len(day_ilocs) == 0:
                continue

            day_signals = self._process_one_day(
                data=data,
                day_ilocs=day_ilocs,
                date=date,
                session_open_time=session_open_time,
                session_close_time=session_close_time,
                orb_end_offset=orb_end_offset,
            )

            if day_signals:
                all_signals.extend(day_signals)
                days_with_signals += 1

        logger.info(
            f"{self.name}: {len(all_signals)} signals generated across "
            f"{days_with_signals} of {len(all_dates)} trading days."
        )
        return all_signals

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_one_day(
        self,
        data: pd.DataFrame,
        day_ilocs: np.ndarray,
        date: object,   # datetime.date
        session_open_time: object,   # datetime.time
        session_close_time: object,  # datetime.time
        orb_end_offset: pd.Timedelta,
    ) -> list[Signal]:
        """
        Run the full ORB logic for one trading day.

        Steps:
            1. Identify the bars that fall within the opening range period.
            2. Validate that enough bars exist to establish a meaningful ORB.
            3. Compute ORB High and ORB Low.
            4. Scan post-ORB bars (up to session_close) for breakout signals.

        Parameters
        ----------
        data : pd.DataFrame
            The full dataset (all days).
        day_ilocs : np.ndarray
            Integer positions (iloc) of all bars on this trading day.
        date : datetime.date
            The trading date.
        session_open_time : datetime.time
            When the opening range starts.
        session_close_time : datetime.time
            When the signal window ends.
        orb_end_offset : pd.Timedelta
            Duration of the opening range.

        Returns
        -------
        list[Signal]
            0, 1, or 2 signals for this day.
        """
        day_index = data.index[day_ilocs]   # DatetimeIndex for this day's bars

        # ---- Build the ORB period time boundary ----
        # Combine the date with the session_open time to get a full Timestamp,
        # then add the ORB duration.
        session_open_ts = pd.Timestamp.combine(date, session_open_time)
        orb_end_ts      = session_open_ts + orb_end_offset
        orb_end_time    = orb_end_ts.time()

        # ---- Identify ORB bars (session_open <= bar_time < orb_end) ----
        # Bars before session_open (pre-market) are excluded.
        # The bar AT orb_end is the first tradeable bar, not part of the ORB.
        orb_mask  = (
            (day_index.time >= session_open_time) &
            (day_index.time <  orb_end_time)
        )
        orb_ilocs = day_ilocs[orb_mask]

        # Detect the actual bar interval from the data so this check works
        # correctly on any timeframe (1-min, 5-min, 15-min, etc.).
        #
        # Why this matters: with orb_minutes=30 and 5-minute bars, we only
        # get 6 bars during the ORB period. The old hardcoded formula
        # (orb_minutes // 2 = 15) would silently skip every trading day
        # because 6 < 15. By measuring the actual bar spacing we compute
        # the correct expected bar count for any timeframe.
        #
        # We detect bar interval from the first two ORB bars. If only one
        # bar exists we cannot measure, so we fall back to 1-minute and
        # let the count check below decide whether to skip.
        if len(orb_ilocs) >= 2:
            bar_interval_minutes = max(1, int(
                (data.index[orb_ilocs[1]] - data.index[orb_ilocs[0]])
                .total_seconds() / 60
            ))
        else:
            bar_interval_minutes = 1  # safe fallback

        expected_orb_bars     = max(1, self.params["orb_minutes"] // bar_interval_minutes)
        min_required_orb_bars = max(1, expected_orb_bars // 2)

        if len(orb_ilocs) < min_required_orb_bars:
            logger.debug(
                f"  {date}: only {len(orb_ilocs)} ORB bars "
                f"(need ≥{min_required_orb_bars}). Skipping."
            )
            return []

        orb_bars = data.iloc[orb_ilocs]
        orb_high = float(orb_bars["high"].max())
        orb_low  = float(orb_bars["low"].min())

        # A zero-range ORB (completely flat market for 30 minutes) cannot
        # generate meaningful breakout levels. Skip.
        if orb_high <= orb_low:
            logger.debug(f"  {date}: ORB high == ORB low ({orb_high}). Skipping.")
            return []

        # ---- Identify post-ORB signal window ----
        # Bars from orb_end (inclusive) up to but NOT including session_close.
        # A signal fired at session_close would have no same-day entry bar.
        post_mask  = (
            (day_index.time >= orb_end_time) &
            (day_index.time <  session_close_time)
        )
        post_ilocs = day_ilocs[post_mask]

        if len(post_ilocs) == 0:
            logger.debug(f"  {date}: no bars in post-ORB signal window.")
            return []

        # ---- Scan for breakout signals ----
        return self._scan_for_signals(data, post_ilocs, orb_high, orb_low)

    def _scan_for_signals(
        self,
        data: pd.DataFrame,
        post_ilocs: np.ndarray,
        orb_high: float,
        orb_low: float,
    ) -> list[Signal]:
        """
        Iterate through post-ORB bars and generate long and/or short signals.

        State machine per day:
            long_breakout_seen  — a bar has already closed above ORB High
            short_breakout_seen — a bar has already closed below ORB Low
            long_fired          — a long Signal has been generated for this day
            short_fired         — a short Signal has been generated for this day

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset.
        post_ilocs : np.ndarray
            iloc positions of the post-ORB signal window for this day.
        orb_high : float
            The high extreme of the opening range.
        orb_low : float
            The low extreme of the opening range.

        Returns
        -------
        list[Signal]
            At most one long Signal and one short Signal.
        """
        direction = self.params["direction"]
        retest    = self.params["retest"]
        rr        = self.params["rr"]
        orb_range = orb_high - orb_low

        # Pre-compute stop and target prices.
        # These are based on ORB LEVELS, not on actual fill price.
        # The realized R:R will differ slightly from `rr` once entry slippage
        # is applied by the execution layer — this is intentional and realistic.
        long_stop    = orb_low
        long_target  = orb_high + orb_range * rr

        short_stop   = orb_high
        short_target = orb_low  - orb_range * rr

        # WARNING: long_target > orb_high and short_target < orb_low
        # are guaranteed by construction (rr > 0, orb_range > 0).
        # If rr were zero or negative, validate_params() would have rejected it.

        # State per day
        long_breakout_seen  = False
        short_breakout_seen = False
        long_fired          = False
        short_fired         = False

        signals: list[Signal] = []

        for iloc_idx in post_ilocs:
            bar        = data.iloc[iloc_idx]
            bar_close  = float(bar["close"])
            bar_high   = float(bar["high"])
            bar_low    = float(bar["low"])

            meta = {"orb_high": orb_high, "orb_low": orb_low}

            # ---- Long signal logic ----
            if direction in ("both", "long_only") and not long_fired:
                if not retest:
                    # Simple breakout: bar closes above ORB High for the first time.
                    if bar_close > orb_high:
                        signals.append(Signal(
                            bar_index    = int(iloc_idx),
                            direction    = "long",
                            stop_price   = long_stop,
                            target_price = long_target,
                            metadata     = meta,
                        ))
                        long_fired = True

                else:
                    # Retest mode: two phases.
                    # Phase 1 — wait for a bar to close above ORB High.
                    if not long_breakout_seen:
                        if bar_close > orb_high:
                            long_breakout_seen = True
                            logger.debug(
                                f"  Long breakout detected at iloc {iloc_idx}. "
                                f"Waiting for retest of {orb_high}."
                            )

                    # Phase 2 — after the breakout, look for a retest bar:
                    # bar's Low touches ORB High (price came back down to the
                    # level), AND bar's Close is still ≥ ORB High (confirming
                    # the level as support — price didn't fall THROUGH it).
                    else:
                        is_retest = (bar_low <= orb_high) and (bar_close >= orb_high)
                        if is_retest:
                            signals.append(Signal(
                                bar_index    = int(iloc_idx),
                                direction    = "long",
                                stop_price   = long_stop,
                                target_price = long_target,
                                metadata     = meta,
                            ))
                            long_fired = True

            # ---- Short signal logic (mirror of long) ----
            if direction in ("both", "short_only") and not short_fired:
                if not retest:
                    if bar_close < orb_low:
                        signals.append(Signal(
                            bar_index    = int(iloc_idx),
                            direction    = "short",
                            stop_price   = short_stop,
                            target_price = short_target,
                            metadata     = meta,
                        ))
                        short_fired = True

                else:
                    if not short_breakout_seen:
                        if bar_close < orb_low:
                            short_breakout_seen = True
                            logger.debug(
                                f"  Short breakout detected at iloc {iloc_idx}. "
                                f"Waiting for retest of {orb_low}."
                            )
                    else:
                        # Retest: bar's High touches ORB Low from below (came
                        # back up to the level), AND Close is still ≤ ORB Low
                        # (confirming the level as resistance).
                        is_retest = (bar_high >= orb_low) and (bar_close <= orb_low)
                        if is_retest:
                            signals.append(Signal(
                                bar_index    = int(iloc_idx),
                                direction    = "short",
                                stop_price   = short_stop,
                                target_price = short_target,
                                metadata     = meta,
                            ))
                            short_fired = True

            # Early exit: if both directions have fired, no need to scan more bars.
            if long_fired and short_fired:
                break
            # If only one direction is enabled and it has fired, we're done.
            if direction == "long_only"  and long_fired:
                break
            if direction == "short_only" and short_fired:
                break

        return signals
