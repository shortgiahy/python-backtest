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
    INDICATORS LAYER  backtest/indicators/volatility.py
                      backtest/indicators/regime.py        <-- YOU ARE HERE
    STRATEGY LAYER    backtest/strategies/base.py, orb.py
    EXECUTION LAYER   backtest/execution/fills.py
    METRICS LAYER     backtest/metrics/report.py

=============================================================================
THIS FILE — backtest/indicators/regime.py
=============================================================================
Computes the ADX (Average Directional Index) from the same 1-minute OHLCV
data, aggregates it to a daily value, then combines it with the daily ATR
output from volatility.py to produce one of four regime labels per trading day.

This file's output is the final "daily indicators" table that flows into
every layer below it. The regime label on each row becomes a permanent field
in the trade log, enabling the metrics layer to slice performance by regime.

OUTPUT — one row per trading day:
    date (index) | atr | atr_20d_avg | atr_above_avg | adx | regime
    -------------|-----|-------------|---------------|-----|------------------
    2022-01-03   |8.25 |    7.91     |     True      |31.4 | trending_high_vol
    2022-01-04   |7.60 |    7.93     |     False     |18.2 | ranging_low_vol

HOW IT CONNECTS:
  - Receives the clean 1-minute DataFrame from backtest/data/loader.py
    (via engine.py).
  - Receives the VolatilityResult from backtest/indicators/volatility.py
    (via engine.py). Using the minute-level ATR computed there avoids
    recomputing TR and ensures both indicators share identical smoothing.
  - Its output DataFrame (the full daily indicators table) is passed to:
      * backtest/strategies/orb.py  → generate_signals()
      * backtest/execution/fills.py → trade log enrichment
      * backtest/metrics/report.py  → regime breakdown sheet

FORMULA REFERENCE:
    up_move   = High_today  − High_yesterday
    down_move = Low_yesterday − Low_today

    +DM = up_move   if (up_move > down_move  AND up_move > 0)   else 0
    -DM = down_move if (down_move > up_move  AND down_move > 0) else 0

    +DI  = 100 × Wilder_smooth(+DM, 14) / ATR_smoothed
    -DI  = 100 × Wilder_smooth(-DM, 14) / ATR_smoothed

    DX   = 100 × |+DI − −DI| / (+DI + −DI)
    ADX  = Wilder_smooth(DX, 14)

    regime label:
        ADX > 25  AND  atr_above_avg  →  "trending_high_vol"
        ADX > 25  AND  NOT above_avg  →  "trending_low_vol"
        ADX ≤ 25  AND  atr_above_avg  →  "ranging_high_vol"
        ADX ≤ 25  AND  NOT above_avg  →  "ranging_low_vol"
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

# Wilder's ADX uses the same 14-period smoothing as ATR.
# +DI/-DI are smoothed over this window; DX is then smoothed again.
# This means ADX has a warm-up period of roughly 2 × ADX_PERIOD bars.
ADX_PERIOD: int = 14

# The conventional threshold separating trending from ranging markets.
# Wilder himself recommended 25 as the boundary. Values between 20 and 30
# are commonly used; 25 is the most widely cited and tested.
ADX_TRENDING_THRESHOLD: float = 25.0

# Session end time. Must match the value used in volatility.py so that
# daily aggregation is consistent between ATR and ADX.
# WARNING: Both files use this constant independently (to avoid a cross-file
# import within the indicators layer). If you change one, change both.
SESSION_END_TIME: str = "16:00"

# The four regime label strings used throughout the system.
# They appear in the trade log, the metrics report, and the regime sheet.
# Do NOT change these strings after running any backtests — the metrics
# layer filters on them by exact string match.
REGIME_TRENDING_HIGH_VOL: str = "trending_high_vol"
REGIME_TRENDING_LOW_VOL:  str = "trending_low_vol"
REGIME_RANGING_HIGH_VOL:  str = "ranging_high_vol"
REGIME_RANGING_LOW_VOL:   str = "ranging_low_vol"


# ---------------------------------------------------------------------------
# Named tuple for this file's public return value
# ---------------------------------------------------------------------------

class RegimeResult(NamedTuple):
    """
    The combined daily indicators table produced by compute_regime().

    Attributes
    ----------
    daily : pd.DataFrame
        One row per trading day. DatetimeIndex (midnight timestamps).
        Columns:
            atr           — daily ATR (from volatility.py, passed through)
            atr_20d_avg   — 20-day rolling mean of daily ATR
            atr_above_avg — bool: True = high-vol regime
            adx           — daily ADX value
            regime        — one of the four REGIME_* label strings above
    """
    daily: pd.DataFrame


# ---------------------------------------------------------------------------
# Public API — one entry point
# ---------------------------------------------------------------------------

def compute_regime(
    data: pd.DataFrame,
    minute_atr: pd.Series,
    volatility_daily: pd.DataFrame,
    adx_period: int = ADX_PERIOD,
    adx_threshold: float = ADX_TRENDING_THRESHOLD,
    session_end: str = SESSION_END_TIME,
) -> RegimeResult:
    """
    Compute daily ADX and combine with daily ATR data to produce regime labels.

    Parameters
    ----------
    data : pd.DataFrame
        Clean 1-minute OHLCV DataFrame (DatetimeIndex, columns: high, low, …).

    minute_atr : pd.Series
        Per-bar Wilder-smoothed ATR from backtest/indicators/volatility.py.
        Using this directly avoids recomputing True Range and guarantees that
        the ATR used to normalise +DI/−DI is identical to the one used for
        the volatility regime flag.

    volatility_daily : pd.DataFrame
        The .daily DataFrame from VolatilityResult (one row per trading day).
        Must contain columns: atr, atr_20d_avg, atr_above_avg.

    adx_period : int
        Wilder smoothing window for +DM, −DM, and DX. Default 14.

    adx_threshold : float
        ADX value above which a market is considered trending. Default 25.

    session_end : str
        Time string "HH:MM" for session close. Used to filter bars before
        daily aggregation — must match the value used in volatility.py.

    Returns
    -------
    RegimeResult
        Named tuple with .daily containing the full daily indicators table
        (ATR columns + adx + regime).

    Example
    -------
    >>> from backtest.data.loader import load_csv
    >>> from backtest.indicators.volatility import compute_daily_atr
    >>> from backtest.indicators.regime import compute_regime
    >>>
    >>> data = load_csv("MES_1min.csv").data
    >>> vol  = compute_daily_atr(data)
    >>> reg  = compute_regime(data, vol.minute_atr, vol.daily)
    >>> print(reg.daily["regime"].value_counts())
    """
    _validate_input(data, minute_atr, volatility_daily)

    # Step 1: Compute raw +DM and −DM for every 1-minute bar
    plus_dm, minus_dm = _compute_directional_movement(data)

    # Step 2: Smooth +DM and −DM with Wilder's exponential smoothing
    alpha = 1.0 / adx_period
    smoothed_plus_dm  = _wilder_smooth(plus_dm,  alpha, adx_period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, alpha, adx_period)

    # Step 3: Compute +DI and −DI by normalising smoothed DM against ATR
    plus_di, minus_di = _compute_directional_indicators(
        smoothed_plus_dm, smoothed_minus_dm, minute_atr
    )

    # Step 4: Compute DX (raw directional movement index) then smooth to ADX
    dx  = _compute_dx(plus_di, minus_di)
    adx = _wilder_smooth(dx, alpha, adx_period)
    adx.name = "adx_minute"

    # Step 5: Aggregate minute-level ADX to one value per trading day
    daily_adx = _aggregate_to_daily(adx, data.index, session_end)

    # Step 6: Join daily ADX with the volatility daily table on date
    # Both DataFrames have DatetimeIndex at midnight; inner join keeps only
    # days present in both (drops any day where either indicator is NaN).
    combined = volatility_daily.join(daily_adx, how="inner")

    if combined.empty:
        raise ValueError(
            "No rows remain after joining ATR and ADX daily tables. "
            "Check that data covers enough bars for both warm-up periods "
            f"(at least {adx_period * 2} 1-minute bars per day for ADX)."
        )

    # Step 7: Assign regime label from the 2×2 combination of ADX and ATR flag
    combined["regime"] = _assign_regime_labels(
        combined["adx"], combined["atr_above_avg"], adx_threshold
    )

    # Log the regime distribution so the user can sanity-check it.
    counts = combined["regime"].value_counts()
    logger.info(
        f"Regime labels assigned across {len(combined)} trading days:\n"
        + "\n".join(f"  {label}: {n} days" for label, n in counts.items())
    )

    return RegimeResult(daily=combined)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_input(
    data: pd.DataFrame,
    minute_atr: pd.Series,
    volatility_daily: pd.DataFrame,
) -> None:
    """
    Check inputs for the required columns and index types.
    Raises ValueError immediately on any mismatch.
    """
    required_data_cols = {"high", "low"}
    missing = required_data_cols - set(data.columns)
    if missing:
        raise ValueError(
            f"regime.py requires data columns {required_data_cols}. "
            f"Missing: {missing}."
        )
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have a DatetimeIndex.")
    if not isinstance(minute_atr.index, pd.DatetimeIndex):
        raise ValueError("minute_atr must have a DatetimeIndex.")

    required_vol_cols = {"atr", "atr_20d_avg", "atr_above_avg"}
    missing_vol = required_vol_cols - set(volatility_daily.columns)
    if missing_vol:
        raise ValueError(
            f"volatility_daily is missing columns: {missing_vol}. "
            f"Make sure you pass the .daily attribute of VolatilityResult."
        )


def _compute_directional_movement(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute raw Plus Directional Movement (+DM) and Minus Directional
    Movement (−DM) for every bar.

    Rules (Wilder's original definition):
        up_move   = High_today  − High_yesterday   (positive = price pushed higher)
        down_move = Low_yesterday − Low_today       (positive = price pushed lower)

        +DM = up_move   when (up_move > down_move) AND (up_move > 0)
        −DM = down_move when (down_move > up_move) AND (down_move > 0)
        Both = 0 when neither condition holds (e.g., an inside bar where
               today's high and low are both within yesterday's range).

    Why both conditions must be checked:
        Condition 1 (up_move > down_move) ensures we assign the move to the
        dominant direction. If both up_move and down_move are large, we
        credit only the larger one, never both.
        Condition 2 (> 0) ensures we never count a lower high as positive
        upward directional movement.

    Returns
    -------
    (plus_dm, minus_dm) : tuple of pd.Series
        Both named and indexed identically to data.
        First row of each is NaN (no prior bar to diff against).
    """
    # diff() computes current - previous for each column.
    # For high: positive = today's high is above yesterday's = upward pressure.
    # For low:  negative = today's low is below yesterday's = downward pressure.
    #           We negate it so down_move is positive when price pushed lower.
    up_move   =  data["high"].diff()   # up_move[0] = NaN (no prior bar)
    down_move = -data["low"].diff()    # negated: positive = lower low

    # .where(condition, other=0.0) keeps the value where condition is True,
    # replaces with 0.0 everywhere else.
    # This is more readable and faster than writing a for-loop with if/else.
    plus_dm  = up_move.where(
        (up_move > down_move) & (up_move > 0),
        other=0.0,
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        other=0.0,
    )

    plus_dm.name  = "plus_dm"
    minus_dm.name = "minus_dm"
    return plus_dm, minus_dm


def _wilder_smooth(series: pd.Series, alpha: float, period: int) -> pd.Series:
    """
    Apply Wilder's exponential smoothing (identical to the ATR smoothing in
    volatility.py).

    Parameters
    ----------
    series : pd.Series
        The raw series to smooth (+DM, −DM, or DX).
    alpha : float
        Smoothing factor = 1 / period. For period=14: alpha = 0.0714…
    period : int
        Used only for min_periods — we require this many non-NaN values
        before producing the first output value.

    Returns
    -------
    pd.Series
        Smoothed series, same index as input. First (period − 1) values
        will be NaN.
    """
    return series.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def _compute_directional_indicators(
    smoothed_plus_dm: pd.Series,
    smoothed_minus_dm: pd.Series,
    minute_atr: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Convert smoothed directional movement into directional indicators (+DI/−DI)
    by normalising against the ATR.

    +DI = 100 × smoothed_+DM / ATR
    −DI = 100 × smoothed_−DM / ATR

    Why divide by ATR?
        Without normalisation, a high-priced instrument (e.g., NQ at 15,000)
        would always have larger raw DM values than a low-priced one (MES at
        4,800), making cross-instrument comparison meaningless. Dividing by
        ATR (the typical bar size) converts directional movement into a
        percentage of typical range — a scale-free number.

    Division by zero:
        If ATR = 0 (no price movement whatsoever), +DI and −DI would be
        infinite. We replace ATR = 0 with NaN before dividing, which
        propagates NaN to +DI/−DI for those bars. This is correct — a bar
        with zero range carries no directional information.

    Returns
    -------
    (plus_di, minus_di) : tuple of pd.Series
    """
    # Replace zero ATR with NaN to avoid ZeroDivisionError / inf.
    # NOTE: minute_atr may have NaN during its own warm-up period;
    # those propagate naturally — we do not need to handle them separately.
    safe_atr = minute_atr.replace(0.0, float("nan"))

    plus_di  = 100.0 * smoothed_plus_dm  / safe_atr
    minus_di = 100.0 * smoothed_minus_dm / safe_atr

    plus_di.name  = "plus_di"
    minus_di.name = "minus_di"
    return plus_di, minus_di


def _compute_dx(plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    """
    Compute DX (Directional Movement Index), the pre-smoothed precursor
    to ADX.

    DX = 100 × |+DI − −DI| / (+DI + −DI)

    Interpretation:
        DX = 0   → +DI and −DI are equal → no net directional bias
        DX = 100 → one DI is zero, the other is large → pure trend

    Division by zero:
        If +DI + −DI = 0, both smoothed DMs are zero — a completely flat
        market with no directional movement. We replace the denominator with
        NaN in that case, producing DX = NaN, which then propagates to ADX.
        This is correct: we cannot assign a trend strength to a market with
        no directional data.

    Returns
    -------
    pd.Series named "dx".
    """
    di_sum  = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()

    safe_di_sum = di_sum.replace(0.0, float("nan"))

    dx = 100.0 * di_diff / safe_di_sum
    dx.name = "dx"
    return dx


def _aggregate_to_daily(
    minute_adx: pd.Series,
    original_index: pd.DatetimeIndex,
    session_end: str,
) -> pd.Series:
    """
    Take the last ADX value of each trading day's regular session.

    This mirrors the aggregation logic in volatility._aggregate_to_daily()
    exactly: filter to session bars first, then resample, then drop NaN days.

    WARNING — DOUBLE WARM-UP: ADX has a longer warm-up than ATR. +DM and
    −DM are first smoothed over `adx_period` bars to produce +DI/−DI. Then
    DX is smoothed over another `adx_period` bars to produce ADX. The total
    warm-up is approximately 2 × adx_period bars before ADX becomes valid.
    With adx_period = 14, the first ~28 1-minute bars of each day may have
    NaN ADX. The resample().last() call naturally handles this — it picks the
    last non-NaN value for the day, which is always past the warm-up period
    on any day with at least 28 bars of 1-minute data.

    Returns
    -------
    pd.Series named "adx", DatetimeIndex at midnight, one entry per trading day.
    """
    end_time = pd.Timestamp(f"1970-01-01 {session_end}").time()
    session_mask = original_index.time <= end_time

    session_adx = minute_adx[session_mask]

    daily_adx = session_adx.resample("D").last().dropna()
    daily_adx.name = "adx"
    return daily_adx


def _assign_regime_labels(
    daily_adx: pd.Series,
    atr_above_avg: pd.Series,
    adx_threshold: float,
) -> pd.Series:
    """
    Map the 2×2 combination of (trending/ranging) × (high-vol/low-vol) to
    one of the four regime label strings.

    Parameters
    ----------
    daily_adx : pd.Series
        Daily ADX values (one per trading day).
    atr_above_avg : pd.Series
        Boolean flag from volatility.py (True = high vol).
    adx_threshold : float
        ADX value above which the market is considered trending (default 25).

    Returns
    -------
    pd.Series of string labels, same index as inputs.

    WARNING — INDEX ALIGNMENT: Both inputs must share the same DatetimeIndex.
    If they differ (e.g., because ATR has more warm-up days dropped than ADX
    or vice versa), pandas will silently produce NaN labels for misaligned
    dates. The inner join in compute_regime() guarantees alignment before
    this function is called — do not call this function directly with
    misaligned inputs.
    """
    is_trending = daily_adx > adx_threshold    # True = trending regime

    # numpy-style conditional assignment using pd.Series.where() chaining:
    # Start with the "default" label (ranging_low_vol).
    # Overwrite with more specific labels where conditions apply.
    # We build all four labels explicitly so the logic is easy to audit.

    labels = pd.Series(REGIME_RANGING_LOW_VOL, index=daily_adx.index)

    labels = labels.where(~(~is_trending & atr_above_avg), REGIME_RANGING_HIGH_VOL)
    labels = labels.where(~( is_trending & ~atr_above_avg), REGIME_TRENDING_LOW_VOL)
    labels = labels.where(~( is_trending & atr_above_avg),  REGIME_TRENDING_HIGH_VOL)

    # Equivalently written as np.select (clearer but requires numpy import):
    #   conditions = [
    #       is_trending & atr_above_avg,
    #       is_trending & ~atr_above_avg,
    #       ~is_trending & atr_above_avg,
    #   ]
    #   choices = [TRENDING_HIGH_VOL, TRENDING_LOW_VOL, RANGING_HIGH_VOL]
    #   default = RANGING_LOW_VOL
    # Both approaches produce identical results. The pd.Series.where() version
    # is used here to avoid an extra import.

    labels.name = "regime"
    return labels
