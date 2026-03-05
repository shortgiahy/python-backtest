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

    DATA LAYER        backtest/data/loader.py              <-- YOU ARE HERE
    INDICATORS LAYER  backtest/indicators/volatility.py, regime.py
    STRATEGY LAYER    backtest/strategies/base.py, orb.py
    EXECUTION LAYER   backtest/execution/fills.py
    METRICS LAYER     backtest/metrics/report.py

=============================================================================
THIS FILE — backtest/data/loader.py
=============================================================================
Loads a 1-minute OHLCV CSV file, validates column presence and data types,
detects and flags rows that violate physical price constraints or data
integrity rules, optionally filters to a date range, and returns a clean
pandas DataFrame ready for the indicator and strategy layers.

HOW IT CONNECTS:
  - backtest/engine.py calls load_csv() as the first step of every run.
  - The returned DataFrame is passed directly to:
      * backtest/indicators/volatility.py  (needs OHLC columns + DatetimeIndex)
      * backtest/indicators/regime.py      (same)
      * backtest/strategies/orb.py         (same)
  - The ValidationReport returned alongside the DataFrame can be inspected
    by the user or logged by the engine for audit purposes.

EXPECTED CSV FORMAT:
    timestamp,open,high,low,close,volume
    2022-01-03 09:30:00,4800.25,4802.50,4799.00,4801.75,1523

    Column names are case-insensitive; they are normalized to lowercase on load.
    The timestamp column must be parseable by pandas.to_datetime().
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# We use Python's standard logging module rather than print() so that the
# calling code (engine.py) can control verbosity — e.g., suppress loader
# messages during a parameter sweep of 500 runs.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — every magic number must be named and explained
# ---------------------------------------------------------------------------

# The exact column names we require after normalizing to lowercase.
REQUIRED_COLUMNS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]

# Price columns that must be strictly positive finite numbers.
PRICE_COLUMNS: list[str] = ["open", "high", "low", "close"]

# Maximum allowed single-bar price change as a fraction of the bar's close.
# A bar where the high is 20% above the close is almost certainly a bad tick.
# Real 1-minute bars in liquid futures rarely move more than 2–3%.
# We use 15% as a conservative threshold that catches obvious corruptions
# without flagging legitimate (if rare) large moves.
#
# WARNING: In March 2020 and April 2020, some instruments moved 5–10% in a
# single day, but rarely 15% in a single MINUTE. If you are testing data from
# an extreme crisis, you may need to widen this threshold. Inspect the flagged
# rows before discarding them.
MAX_SINGLE_BAR_RANGE_FRACTION: float = 0.15

# Reason strings stored in the validation report.
# Using named constants prevents typos that would make filtering the report
# by reason silently return empty results.
REASON_MISSING_VALUES     = "missing_values"
REASON_OHLC_VIOLATION     = "ohlc_violation"        # e.g. High < Low
REASON_NONPOSITIVE_PRICE  = "nonpositive_price"
REASON_NEGATIVE_VOLUME    = "negative_volume"
REASON_ZERO_VOLUME        = "zero_volume"            # flagged but NOT dropped
REASON_DUPLICATE_TIMESTAMP = "duplicate_timestamp"
REASON_EXTREME_RANGE      = "extreme_range"          # likely bad tick


# ---------------------------------------------------------------------------
# ValidationReport — a record of every flagged row
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Holds all rows that were flagged during data validation, grouped by
    the reason they were flagged.

    Not all flagged rows are dropped. Some (like zero-volume bars) are
    only warnings — the row is kept in the clean DataFrame but recorded here
    so the user can inspect it.

    Attributes
    ----------
    flagged : pd.DataFrame
        All flagged rows from the original file, with an extra "reason" column
        explaining why each row was flagged. A single row may appear more than
        once if it violated multiple rules.
    dropped_count : int
        Number of rows removed from the final clean DataFrame.
    warning_count : int
        Number of rows kept but flagged as suspicious.
    original_row_count : int
        Total rows in the file before any cleaning.
    """
    flagged: pd.DataFrame
    dropped_count: int
    warning_count: int
    original_row_count: int

    @property
    def clean_row_count(self) -> int:
        """Rows remaining after cleaning."""
        return self.original_row_count - self.dropped_count

    def summary(self) -> str:
        """Return a human-readable one-paragraph summary."""
        return (
            f"Loaded {self.original_row_count:,} rows. "
            f"Dropped {self.dropped_count:,} bad rows. "
            f"Flagged {self.warning_count:,} rows as warnings (kept). "
            f"{self.clean_row_count:,} rows in clean output."
        )


# ---------------------------------------------------------------------------
# Named tuple for the return value of load_csv
# ---------------------------------------------------------------------------

class LoadResult(NamedTuple):
    """
    The two objects returned by load_csv().

    Using a NamedTuple rather than a plain tuple means callers can write:
        result = load_csv(...)
        df = result.data
        report = result.report
    instead of the fragile:
        df, report = load_csv(...)   # easy to accidentally swap the order
    """
    data: pd.DataFrame
    report: ValidationReport


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(
    file_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    drop_zero_volume: bool = False,
) -> LoadResult:
    """
    Load a 1-minute OHLCV CSV file, validate it, and return a clean DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file. Accepts both absolute and relative paths.

    start_date : str or None
        Optional ISO date string, e.g. "2022-01-01". If provided, only rows
        on or after this date are kept. Applied AFTER cleaning so that
        validation statistics reflect the full file.

    end_date : str or None
        Optional ISO date string, e.g. "2024-12-31". If provided, only rows
        on or before this date are kept.

    drop_zero_volume : bool
        If True, bars with volume == 0 are dropped rather than just flagged.
        Default False because zero-volume bars sometimes occur legitimately
        at the open/close of a session and you may want to see them.

    Returns
    -------
    LoadResult
        A named tuple with two fields:
          .data   — clean pd.DataFrame with DatetimeIndex, columns:
                    open, high, low, close, volume
          .report — ValidationReport describing every flagged row

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    ValueError
        If required columns are missing, or if the timestamp column cannot
        be parsed as datetime, or if the file is empty after cleaning.

    Example
    -------
    >>> result = load_csv("MES_1min.csv", start_date="2022-01-01")
    >>> print(result.report.summary())
    >>> df = result.data
    """
    file_path = Path(file_path)
    _assert_file_exists(file_path)

    # -----------------------------------------------------------------------
    # Step 1: Read raw CSV
    # -----------------------------------------------------------------------
    # low_memory=False tells pandas to read the entire column before deciding
    # its data type. The default (True) reads in chunks and can assign a column
    # "object" type (strings) if early rows look like strings — causing silent
    # failures downstream when you try to do math on what should be a float.
    raw: pd.DataFrame = pd.read_csv(file_path, low_memory=False)
    original_row_count = len(raw)
    logger.info(f"Read {original_row_count:,} rows from {file_path.name}")

    # -----------------------------------------------------------------------
    # Step 2: Normalize column names to lowercase, strip whitespace
    # -----------------------------------------------------------------------
    # Vendors often use "Timestamp", "OPEN", "High " (with trailing space).
    # We normalize once here so everything downstream can assume lowercase.
    raw.columns = [col.strip().lower() for col in raw.columns]

    _assert_required_columns(raw.columns.tolist(), file_path)

    # -----------------------------------------------------------------------
    # Step 3: Parse and set the timestamp index
    # -----------------------------------------------------------------------
    raw["timestamp"] = _parse_timestamps(raw["timestamp"], file_path)
    raw = raw.set_index("timestamp")
    raw.index.name = "timestamp"

    # -----------------------------------------------------------------------
    # Step 4: Coerce price and volume columns to numeric
    # -----------------------------------------------------------------------
    # pd.to_numeric with errors="coerce" turns anything that cannot be
    # converted to a number into NaN (Not a Number) rather than raising.
    # We then detect those NaN rows in the validation step.
    for col in PRICE_COLUMNS + ["volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # -----------------------------------------------------------------------
    # Step 5: Validate and collect flagged rows
    # -----------------------------------------------------------------------
    all_flagged_pieces: list[pd.DataFrame] = []
    rows_to_drop: pd.Index = pd.Index([])       # accumulates timestamps to drop
    warning_timestamps: set = set()             # kept but warned

    # --- Rule 1: Missing values in any required column ---
    missing_mask = raw[PRICE_COLUMNS + ["volume"]].isna().any(axis=1)
    if missing_mask.any():
        flagged = _tag(raw[missing_mask], REASON_MISSING_VALUES)
        all_flagged_pieces.append(flagged)
        rows_to_drop = rows_to_drop.union(raw.index[missing_mask])
        logger.warning(f"  {missing_mask.sum()} rows dropped: {REASON_MISSING_VALUES}")

    # Work on the subset that still has no NaN for the remaining checks.
    # Otherwise comparisons like high < low would throw on NaN rows.
    valid = raw[~missing_mask].copy()

    # --- Rule 2: OHLC physical consistency ---
    # High must be the highest value. Low must be the lowest.
    # Open and Close must fall within [Low, High].
    # Any violation is a data corruption, not a market event.
    ohlc_bad = (
        (valid["high"] < valid["low"])
        | (valid["open"] > valid["high"])
        | (valid["open"] < valid["low"])
        | (valid["close"] > valid["high"])
        | (valid["close"] < valid["low"])
    )
    if ohlc_bad.any():
        flagged = _tag(valid[ohlc_bad], REASON_OHLC_VIOLATION)
        all_flagged_pieces.append(flagged)
        rows_to_drop = rows_to_drop.union(valid.index[ohlc_bad])
        logger.warning(f"  {ohlc_bad.sum()} rows dropped: {REASON_OHLC_VIOLATION}")

    # --- Rule 3: All prices must be strictly positive ---
    # Non-positive prices are always a data error for the instruments this
    # engine targets (equity index futures, gold, crude oil).
    #
    # WARNING: Crude oil (CL/MCL) briefly traded at negative prices in
    # April 2020. If you are testing MCL data from that period and the price
    # data is correct, you may need to disable this check for that date range.
    nonpositive_price = (valid[PRICE_COLUMNS] <= 0).any(axis=1)
    if nonpositive_price.any():
        flagged = _tag(valid[nonpositive_price], REASON_NONPOSITIVE_PRICE)
        all_flagged_pieces.append(flagged)
        rows_to_drop = rows_to_drop.union(valid.index[nonpositive_price])
        logger.warning(
            f"  {nonpositive_price.sum()} rows dropped: {REASON_NONPOSITIVE_PRICE}"
        )

    # --- Rule 4: Volume must be non-negative ---
    negative_volume = valid["volume"] < 0
    if negative_volume.any():
        flagged = _tag(valid[negative_volume], REASON_NEGATIVE_VOLUME)
        all_flagged_pieces.append(flagged)
        rows_to_drop = rows_to_drop.union(valid.index[negative_volume])
        logger.warning(
            f"  {negative_volume.sum()} rows dropped: {REASON_NEGATIVE_VOLUME}"
        )

    # --- Rule 5: Zero volume — warning only, drop is optional ---
    # A bar with volume = 0 means no contracts changed hands.
    # This can be legitimate (very first bar of pre-market) or a vendor error.
    # We flag it but keep it unless drop_zero_volume=True.
    zero_volume = valid["volume"] == 0
    if zero_volume.any():
        flagged = _tag(valid[zero_volume], REASON_ZERO_VOLUME)
        all_flagged_pieces.append(flagged)
        if drop_zero_volume:
            rows_to_drop = rows_to_drop.union(valid.index[zero_volume])
            logger.warning(
                f"  {zero_volume.sum()} rows dropped: {REASON_ZERO_VOLUME}"
            )
        else:
            warning_timestamps.update(valid.index[zero_volume].tolist())
            logger.warning(
                f"  {zero_volume.sum()} rows flagged (kept): {REASON_ZERO_VOLUME}"
            )

    # --- Rule 6: Duplicate timestamps ---
    # If two rows share the same timestamp, something went wrong in the data
    # pipeline (double-exported, rollover seam, etc.). We keep the FIRST
    # occurrence and drop all subsequent duplicates.
    #
    # We check BEFORE dropping other bad rows so duplicates among bad rows
    # are also caught.
    dup_mask = raw.index.duplicated(keep="first")
    if dup_mask.any():
        flagged = _tag(raw[dup_mask], REASON_DUPLICATE_TIMESTAMP)
        all_flagged_pieces.append(flagged)
        rows_to_drop = rows_to_drop.union(raw.index[dup_mask])
        logger.warning(
            f"  {dup_mask.sum()} rows dropped: {REASON_DUPLICATE_TIMESTAMP}"
        )

    # --- Rule 7: Extreme single-bar range (likely bad tick) ---
    # If a bar's (high - low) / close exceeds our threshold, the bar is
    # probably a bad tick — a momentary data glitch that never reflected
    # a real trade price.
    #
    # Example: MES normally trades in a 2-point range per minute. A bar
    # showing a 500-point high-to-low range is almost certainly corrupt.
    #
    # We check on the "still valid" subset to avoid divide-by-zero on
    # nonpositive prices already queued for dropping.
    still_valid = valid[~valid.index.isin(rows_to_drop)]
    bar_range_fraction = (
        (still_valid["high"] - still_valid["low"]) / still_valid["close"]
    )
    extreme_range = bar_range_fraction > MAX_SINGLE_BAR_RANGE_FRACTION
    if extreme_range.any():
        flagged = _tag(still_valid[extreme_range], REASON_EXTREME_RANGE)
        all_flagged_pieces.append(flagged)
        rows_to_drop = rows_to_drop.union(still_valid.index[extreme_range])
        logger.warning(
            f"  {extreme_range.sum()} rows dropped: {REASON_EXTREME_RANGE} "
            f"(range > {MAX_SINGLE_BAR_RANGE_FRACTION:.0%} of close)"
        )

    # -----------------------------------------------------------------------
    # Step 6: Build clean DataFrame
    # -----------------------------------------------------------------------
    clean = raw[~raw.index.isin(rows_to_drop)].copy()
    clean = clean.sort_index()          # ensure chronological order

    # -----------------------------------------------------------------------
    # Step 7: Apply date range filter (AFTER validation stats are recorded)
    # -----------------------------------------------------------------------
    # We filter after validation so that the ValidationReport reflects the
    # full file's quality, not just the requested date window.
    if start_date is not None:
        clean = clean[clean.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        clean = clean[clean.index <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]

    if clean.empty:
        raise ValueError(
            f"No data remaining after cleaning and date filtering. "
            f"Check your start_date/end_date and the validation report."
        )

    # -----------------------------------------------------------------------
    # Step 8: Build ValidationReport
    # -----------------------------------------------------------------------
    dropped_count = len(rows_to_drop.unique())
    warning_count = len(warning_timestamps - set(rows_to_drop.tolist()))

    if all_flagged_pieces:
        flagged_df = pd.concat(all_flagged_pieces)
    else:
        # Build an empty DataFrame with the expected columns so downstream
        # code that inspects report.flagged never has to check for None.
        flagged_df = pd.DataFrame(columns=list(raw.columns) + ["reason"])

    report = ValidationReport(
        flagged=flagged_df,
        dropped_count=dropped_count,
        warning_count=warning_count,
        original_row_count=original_row_count,
    )

    logger.info(report.summary())

    return LoadResult(data=clean, report=report)


def log_gap_report(
    data: pd.DataFrame,
    session_start_hour: int = 9,
    bar_interval_minutes: int | None = None,
) -> None:
    """
    Scan the clean DataFrame for missing bars during market hours and log
    a summary. Does not modify the DataFrame.

    This is called separately from load_csv() because gaps are not necessarily
    errors — they can be legitimate (early-close days, data vendor outages).
    The user decides what to do with the information.

    Parameters
    ----------
    data : pd.DataFrame
        The clean DataFrame returned by load_csv().data.
    session_start_hour : int
        Hour (24h) at which the regular session begins. Gaps before this hour
        are ignored (pre-market is often sparse). Default 9 (for 09:30 open).
    bar_interval_minutes : int or None
        The expected spacing between consecutive bars in minutes.
        If None (default), the interval is auto-detected from the median
        gap between the first 100 consecutive session bars. This makes the
        function work correctly on 1-min, 5-min, 15-min, or any other data.

    What counts as a gap
    --------------------
    A gap is any pair of consecutive rows where the time difference is
    strictly greater than bar_interval_minutes AND both rows fall during
    market hours. Overnight breaks (16:00 → 09:30) are intentionally
    excluded by the session_start_hour filter.
    """
    session_data = data[data.index.hour >= session_start_hour]
    if session_data.empty:
        logger.info("Gap report: no data in session hours.")
        return

    time_diffs = session_data.index.to_series().diff().dropna()

    # Auto-detect bar interval if not provided.
    # We use the median of the first 100 observed gaps (rather than the mean)
    # so that a few large overnight gaps don't skew the estimate.
    if bar_interval_minutes is None:
        sample = time_diffs.head(100)
        detected_seconds = sample.median().total_seconds()
        bar_interval_minutes = max(1, int(round(detected_seconds / 60)))
        logger.info(f"Gap report: auto-detected bar interval = {bar_interval_minutes} min.")

    expected_gap = pd.Timedelta(minutes=bar_interval_minutes)

    # A "gap" is any consecutive pair whose spacing exceeds the expected
    # bar interval by at least one full bar interval.
    # We use strictly greater-than to ignore floating-point timestamp noise
    # (e.g., 00:05:00.001 should not be flagged on 5-minute data).
    gaps = time_diffs[time_diffs > expected_gap]

    if gaps.empty:
        logger.info(
            f"Gap report: no intra-session gaps detected "
            f"(bar interval = {bar_interval_minutes} min)."
        )
    else:
        logger.warning(
            f"Gap report: {len(gaps)} intra-session gaps detected "
            f"(bar interval = {bar_interval_minutes} min)."
        )
        for ts, gap_len in gaps.items():
            logger.warning(f"  Gap of {gap_len} ending at {ts}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assert_file_exists(file_path: Path) -> None:
    """Raise FileNotFoundError with a helpful message if the file is missing."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path.resolve()}\n"
            f"Check that the path is correct and the file has not been moved."
        )


def _assert_required_columns(columns: list[str], file_path: Path) -> None:
    """
    Raise ValueError listing every missing column if any required column
    is absent.

    We list ALL missing columns at once rather than raising on the first one,
    so the user can fix everything in a single edit of the CSV.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing:
        raise ValueError(
            f"File '{file_path.name}' is missing required columns: {missing}\n"
            f"Found columns: {columns}\n"
            f"Expected columns: {REQUIRED_COLUMNS}"
        )


def _parse_timestamps(series: pd.Series, file_path: Path) -> pd.Series:
    """
    Parse the timestamp column to pandas Timestamps.

    pandas.to_datetime() is very flexible — it handles ISO 8601 strings,
    Unix epoch integers, and many regional date formats automatically.

    We use errors="raise" (the default) so that a completely unparseable
    timestamp column fails immediately with a clear error, rather than
    silently producing NaT (Not a Time) values that corrupt all time-based
    filtering downstream.
    """
    try:
        return pd.to_datetime(series)
    except Exception as exc:
        raise ValueError(
            f"Could not parse 'timestamp' column in '{file_path.name}' as "
            f"datetime. First few values: {series.head().tolist()}\n"
            f"Original error: {exc}"
        ) from exc


def _tag(subset: pd.DataFrame, reason: str) -> pd.DataFrame:
    """
    Return a copy of subset with a 'reason' column added.

    Used to build the ValidationReport. Each call produces one "slab" of
    flagged rows; they are concatenated at the end of load_csv().
    """
    tagged = subset.copy()
    tagged["reason"] = reason
    return tagged
