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
    INDICATORS LAYER  backtest/indicators/volatility.py   <-- YOU ARE HERE
                      backtest/indicators/regime.py
    STRATEGY LAYER    backtest/strategies/base.py, orb.py
    EXECUTION LAYER   backtest/execution/fills.py
    METRICS LAYER     backtest/metrics/report.py

=============================================================================
THIS FILE — backtest/indicators/volatility.py
=============================================================================
Computes Average True Range (ATR) from the 1-minute OHLCV DataFrame and
aggregates it to a single value per trading day. Also computes a 20-day
rolling average of daily ATR and produces a boolean flag indicating whether
current volatility is above or below that rolling average.

The daily output is a DataFrame with one row per trading date:

    date         | atr   | atr_20d_avg | atr_above_avg
    -------------|-------|-------------|---------------
    2022-01-03   | 8.25  | 7.91        | True
    2022-01-04   | 7.60  | 7.93        | False

HOW IT CONNECTS:
  - Receives the clean DataFrame from backtest/data/loader.py (via engine.py).
  - Its output DataFrame is consumed by backtest/indicators/regime.py, which
    joins it with ADX data to produce the final four-label regime per day.
  - The regime label ultimately appears in every trade record in the trade log.

FORMULA REFERENCE:
    True Range (TR):
        TR = max(High - Low,  |High - Close_prev|,  |Low - Close_prev|)

    ATR (Wilder's smoothing, 14-period):
        ATR_t = ATR_{t-1} * (13/14) + TR_t * (1/14)
        Equivalently: ewm(alpha = 1/14, adjust=False)

    Daily ATR:
        The ATR value at the last bar of the regular trading session each day.

    Volatility regime flag:
        atr_above_avg = (daily_ATR > rolling_20day_mean(daily_ATR))
=============================================================================
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wilder's original ATR uses a 14-period smoothing window.
# This value is widely used in technical analysis literature.
# Changing it significantly (e.g., to 5) produces a much noisier ATR;
# changing it to 50 produces a much slower-reacting ATR that misses
# short-term volatility shifts.
ATR_PERIOD: int = 14

# The rolling average period for the volatility regime flag.
# 20 trading days ≈ 1 calendar month. This means the flag compares today's
# ATR to the typical ATR over the past month of trading.
ATR_ROLLING_AVG_PERIOD: int = 20

# The time at which we "snapshot" the ATR for the day.
# We use the last bar ON OR BEFORE this time. For US equity index futures
# (MES, ES), the regular session ends at 16:00 Eastern.
# WARNING: If you are testing an instrument with a different session close
# (e.g., crude oil futures which close at 14:30 CME local), change this.
SESSION_END_TIME: str = "16:00"


# ---------------------------------------------------------------------------
# Named tuple for the public return value
# ---------------------------------------------------------------------------

class VolatilityResult(NamedTuple):
    """
    The output of compute_daily_atr().

    Attributes
    ----------
    daily : pd.DataFrame
        One row per trading day. Index is a DatetimeIndex (midnight timestamps,
        one per trading day). Columns:
            atr           — daily ATR value (in price points)
            atr_20d_avg   — 20-day rolling mean of daily ATR
            atr_above_avg — bool, True if today's ATR > 20-day average

    minute_atr : pd.Series
        The raw per-bar ATR series on the 1-minute data. This is retained
        so that regime.py can also compute ADX at the bar level before
        aggregating, ensuring both indicators use exactly the same bars.
    """
    daily: pd.DataFrame
    minute_atr: pd.Series


# ---------------------------------------------------------------------------
# Public API — one entry point
# ---------------------------------------------------------------------------

def compute_daily_atr(
    data: pd.DataFrame,
    atr_period: int = ATR_PERIOD,
    rolling_avg_period: int = ATR_ROLLING_AVG_PERIOD,
    session_end: str = SESSION_END_TIME,
) -> VolatilityResult:
    """
    Compute ATR on 1-minute bars and aggregate to a daily volatility table.

    Parameters
    ----------
    data : pd.DataFrame
        Clean 1-minute OHLCV DataFrame from backtest/data/loader.py.
        Must have a DatetimeIndex and columns: open, high, low, close, volume.

    atr_period : int
        Smoothing window for Wilder's ATR. Default 14.

    rolling_avg_period : int
        Number of trading days in the rolling average used for the regime
        flag. Default 20 (≈ 1 calendar month).

    session_end : str
        Time string "HH:MM" for the regular session close. Only bars at or
        before this time are included in the daily snapshot. Default "16:00".

    Returns
    -------
    VolatilityResult
        Named tuple with .daily (per-day DataFrame) and .minute_atr (per-bar
        Series) — see VolatilityResult docstring for column details.

    Example
    -------
    >>> from backtest.data.loader import load_csv
    >>> from backtest.indicators.volatility import compute_daily_atr
    >>> result_load = load_csv("MES_1min.csv")
    >>> vol = compute_daily_atr(result_load.data)
    >>> print(vol.daily.head())
    """
    _validate_input(data)

    # Step 1: Compute True Range for every 1-minute bar
    tr: pd.Series = _compute_true_range(data)

    # Step 2: Apply Wilder's exponential smoothing to get per-bar ATR
    minute_atr: pd.Series = _apply_wilders_smoothing(tr, period=atr_period)

    # Step 3: Take the session-end snapshot of ATR for each trading day
    daily_atr: pd.Series = _aggregate_to_daily(minute_atr, data.index, session_end)

    # Step 4: Compute the rolling average and the regime flag
    atr_20d_avg: pd.Series = daily_atr.rolling(
        window=rolling_avg_period,
        min_periods=1,          # explained in the warning below
    ).mean()
    atr_20d_avg.name = "atr_20d_avg"

    # WARNING — WARM-UP PERIOD: When min_periods=1, the rolling average for
    # the first (rolling_avg_period - 1) days is based on fewer than
    # rolling_avg_period observations. This means the regime flag is less
    # reliable during that period. The engine should either skip the first
    # rolling_avg_period days of trading or treat early-regime labels with
    # caution.
    #
    # We use min_periods=1 rather than min_periods=rolling_avg_period because
    # the alternative would produce NaN for the first 19 rows, forcing regime.py
    # and the strategy to handle NaN regime labels — a source of subtle bugs.
    # A "slightly unreliable but always present" value is safer than NaN.

    atr_above_avg: pd.Series = (daily_atr > atr_20d_avg)
    atr_above_avg.name = "atr_above_avg"

    daily = pd.DataFrame({
        "atr": daily_atr,
        "atr_20d_avg": atr_20d_avg,
        "atr_above_avg": atr_above_avg,
    })

    logger.info(
        f"ATR computed over {len(daily)} trading days. "
        f"Mean daily ATR: {daily['atr'].mean():.4f}. "
        f"High-vol days: {atr_above_avg.sum()} "
        f"({atr_above_avg.mean():.1%})."
    )

    return VolatilityResult(daily=daily, minute_atr=minute_atr)


# ---------------------------------------------------------------------------
# Private helpers — called only by compute_daily_atr()
# ---------------------------------------------------------------------------

def _validate_input(data: pd.DataFrame) -> None:
    """
    Check that the DataFrame has the columns and index type we require.

    Raises ValueError immediately so errors are caught before any computation
    rather than surfacing as a confusing downstream AttributeError.
    """
    required = {"high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"volatility.py requires columns {required}. "
            f"Missing: {missing}. Got: {list(data.columns)}"
        )
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            "volatility.py requires a DatetimeIndex. "
            "Make sure you loaded the data with backtest/data/loader.py."
        )


def _compute_true_range(data: pd.DataFrame) -> pd.Series:
    """
    Compute the True Range for every bar in the 1-minute DataFrame.

    True Range is the largest of three measurements:
        1. High - Low          (the bar's own internal spread)
        2. |High - Close_prev| (overnight gap up + further intraday rise)
        3. |Low  - Close_prev| (overnight gap down + further intraday fall)

    For the very first bar of the dataset, Close_prev is NaN (there is no
    prior bar), so TR for that bar is also NaN. This is correct — we cannot
    know the true range of a bar without knowing where price was before it.
    The Wilder smoothing in the next step handles the NaN naturally via
    min_periods.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns "high", "low", "close" with a DatetimeIndex.

    Returns
    -------
    pd.Series
        True Range values, same index as data. Named "true_range".
    """
    high = data["high"]
    low  = data["low"]

    # shift(1) moves every value one row forward, so prev_close[i] = close[i-1].
    # For i=0 (the first bar), prev_close is NaN — this is intentional.
    prev_close = data["close"].shift(1)

    # Build a three-column DataFrame — one column per term of the TR formula.
    # Then take the row-wise maximum (axis=1 means "across columns").
    #
    # We use pd.concat instead of pd.DataFrame({...}) because it preserves
    # the index alignment precisely, even if the inputs have different lengths
    # (which they won't here, but it's a good habit).
    tr_components = pd.concat(
        [
            high - low,                        # term 1: bar range
            (high - prev_close).abs(),          # term 2: gap-adjusted high
            (low  - prev_close).abs(),          # term 3: gap-adjusted low
        ],
        axis=1,
    )

    tr = tr_components.max(axis=1)
    tr.name = "true_range"
    return tr


def _apply_wilders_smoothing(tr: pd.Series, period: int) -> pd.Series:
    """
    Apply Wilder's exponential smoothing to the True Range series to produce
    ATR.

    Wilder's smoothing formula:
        ATR_t = ATR_{t-1} * ((period - 1) / period) + TR_t * (1 / period)

    This is equivalent to an Exponential Weighted Moving Average (EWM) with:
        alpha     = 1 / period
        adjust    = False   (use the recursive formula above, not the sum formula)
        min_periods = period (no output until we have a full window of TR values)

    Why min_periods = period:
        With fewer than `period` bars, Wilder's formula would produce an ATR
        based on, say, 3 observations instead of 14. That number is
        meaningless and could cause the regime flag to fire incorrectly very
        early in the dataset. We wait until we have a full window.

    Parameters
    ----------
    tr : pd.Series
        True Range series from _compute_true_range().
    period : int
        Smoothing window (default: ATR_PERIOD = 14).

    Returns
    -------
    pd.Series
        ATR values, same index as tr. Named "atr".
        First (period - 1) values will be NaN.
    """
    alpha = 1.0 / period

    atr = tr.ewm(
        alpha=alpha,
        adjust=False,           # use recursive (Wilder) form, not sum form
        min_periods=period,     # produce NaN until we have `period` TR values
    ).mean()

    atr.name = "atr"
    return atr


def _aggregate_to_daily(
    minute_atr: pd.Series,
    original_index: pd.DatetimeIndex,
    session_end: str,
) -> pd.Series:
    """
    Take the ATR value at the last bar of each trading day's regular session.

    Steps:
        1. Filter bars to those at or before session_end (e.g., 16:00)
        2. Resample to daily frequency, taking the last non-NaN value per day
        3. Drop days with no valid ATR (e.g., the very first day if the ATR
           warm-up period runs into the next day)

    Parameters
    ----------
    minute_atr : pd.Series
        Per-bar ATR from _apply_wilders_smoothing().
    original_index : pd.DatetimeIndex
        The index of the original DataFrame (used to build the time filter).
    session_end : str
        Time string "HH:MM", e.g. "16:00".

    Returns
    -------
    pd.Series
        One ATR value per trading day. Index is DatetimeIndex at midnight.
        Named "atr".

    WARNING — SESSION END FILTERING: We filter BEFORE resampling. If we
    resampled first and then tried to filter, resample("D") would group all
    23.5 hours of overnight trading into the same calendar day, and the "last"
    value might be from 23:59 the same calendar night — not the 16:00 session
    close we care about. Always filter first, then resample.
    """
    # Parse the session end string to a time object for comparison.
    # pd.Timestamp is used here purely for its .time() method.
    end_time = pd.Timestamp(f"1970-01-01 {session_end}").time()

    # Keep only bars that fall within the regular session.
    # DatetimeIndex.time returns an array of datetime.time objects,
    # one per row, which we can compare element-wise.
    session_mask = original_index.time <= end_time
    session_atr = minute_atr[session_mask]

    # Resample to daily, taking the last valid ATR value each day.
    # "D" = calendar day. .last() returns the last non-NaN value in the group.
    # .dropna() removes any days where every bar had NaN ATR (warm-up days
    # at the very start of the dataset where ATR was still accumulating).
    daily_atr = session_atr.resample("D").last().dropna()
    daily_atr.name = "atr"

    return daily_atr
