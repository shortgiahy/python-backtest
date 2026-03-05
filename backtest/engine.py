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
    EXECUTION LAYER   backtest/execution/fills.py
    METRICS LAYER     backtest/metrics/report.py

=============================================================================
THIS FILE — backtest/engine.py                             <-- YOU ARE HERE
=============================================================================
The master orchestrator. BacktestEngine is the single object a user
instantiates and calls. It connects all five layers in the correct order,
ensuring no layer ever knows about or imports from another directly.

THREE ENTRY POINTS:
  run()              — full backtest on the configured date range.
  walk_forward()     — chronological train / validate / test split analysis.
  parameter_sweep()  — sensitivity table across a range of one parameter.

DESIGN RULE:
  This file is the ONLY place in the codebase where all five layers are
  imported together. Every other file imports only from its own layer or
  from the standard library. This strict discipline is what makes the layers
  independently swappable.
=============================================================================
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --- Layer imports (only done here, in the master engine) ---
from backtest.data.loader import load_csv
from backtest.execution.fills import INSTRUMENT_SPECS, simulate_trades
from backtest.indicators.regime import compute_regime
from backtest.indicators.volatility import compute_daily_atr
from backtest.metrics.report import MetricsEngine
from backtest.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default starting account balance used for drawdown and equity curve metrics.
DEFAULT_INITIAL_CAPITAL: float = 10_000.0

# Minimum fraction of the total date range that a walk-forward split must
# cover to be considered meaningful. A split with fewer than this fraction
# of total trading days is likely too small to produce reliable statistics.
MIN_SPLIT_FRACTION: float = 0.05

# Columns extracted from compute_all() for compact comparison tables.
# These are the metrics shown in the walk-forward and parameter sweep outputs.
SUMMARY_METRIC_KEYS: list[str] = [
    "n_trades",
    "win_rate",
    "ev_per_trade",
    "profit_factor",
    "total_net_pnl",
    "sharpe_ratio",
    "max_drawdown_pct",
    "calmar_ratio",
    "t_statistic",
    "p_value",
    "commission_drag_pct",
]


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Master orchestrator for the backtesting system.

    Instantiate once, call run() to execute the full backtest. Optionally
    call walk_forward() or parameter_sweep() after run() (they reuse the
    already-loaded data to save time).

    Parameters
    ----------
    data_path : str or Path
        Path to the 1-minute OHLCV CSV file.

    instrument : str
        One of "MES", "MGC", "MCL". Selects commission, tick size,
        slippage from the INSTRUMENT_SPECS table in execution/fills.py.

    strategy : BaseStrategy
        Any object that inherits from BaseStrategy. The engine calls
        strategy.generate_signals() and uses strategy.name for output
        file naming.

    contracts : int
        Number of contracts to trade on each signal. Must be ≥ 1.

    start : str or None
        ISO date string "YYYY-MM-DD". Only data on or after this date
        is used. If None, the full file is used from the beginning.

    end : str or None
        ISO date string "YYYY-MM-DD". Only data on or before this date
        is used. If None, the full file is used to the end.

    output_dir : str or Path
        Directory where Excel workbooks are saved. Created automatically
        if it does not exist.

    initial_capital : float
        Starting account balance in dollars. Used for equity curve,
        drawdown percentages, and the Calmar ratio.

    Example
    -------
    >>> from backtest.engine import BacktestEngine
    >>> from backtest.strategies.orb import ORBStrategy
    >>>
    >>> engine = BacktestEngine(
    ...     data_path   = "MES_1min.csv",
    ...     instrument  = "MES",
    ...     strategy    = ORBStrategy(params={"orb_minutes": 30, "rr": 1.5}),
    ...     contracts   = 1,
    ...     start       = "2022-01-01",
    ...     end         = "2024-12-31",
    ...     output_dir  = "results/",
    ... )
    >>> results = engine.run()
    """

    def __init__(
        self,
        data_path: str | Path,
        instrument: str,
        strategy: BaseStrategy,
        contracts: int = 1,
        start: str | None = None,
        end: str | None = None,
        output_dir: str | Path = "results/",
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    ) -> None:
        # --- Validate inputs immediately so errors surface at construction ---
        if instrument not in INSTRUMENT_SPECS:
            raise ValueError(
                f"Unknown instrument '{instrument}'. "
                f"Valid options: {list(INSTRUMENT_SPECS.keys())}"
            )
        if not isinstance(strategy, BaseStrategy):
            raise TypeError(
                f"strategy must be an instance of BaseStrategy, "
                f"got {type(strategy).__name__}."
            )
        if contracts < 1:
            raise ValueError(f"contracts must be ≥ 1, got {contracts}.")
        if initial_capital <= 0:
            raise ValueError(
                f"initial_capital must be positive, got {initial_capital}."
            )

        self.data_path       = Path(data_path)
        self.instrument      = instrument
        self.strategy        = strategy
        self.contracts       = contracts
        self.start           = start
        self.end             = end
        self.output_dir      = Path(output_dir)
        self.initial_capital = initial_capital

        # Data and indicators are loaded lazily on first use.
        # This lets the user construct BacktestEngine and inspect its
        # attributes before committing to loading a potentially large file.
        self._data:             pd.DataFrame | None = None
        self._last_metrics_engine: MetricsEngine | None = None

    # ------------------------------------------------------------------
    # Primary public interface
    # ------------------------------------------------------------------

    def run(self, n_permutations: int = 10_000) -> MetricsEngine:
        """
        Execute the full backtest pipeline and return the MetricsEngine.

        Steps:
            1. Load and validate the CSV (data layer).
            2. Compute ATR and volatility regime (indicators layer).
            3. Compute ADX and combine into regime labels (indicators layer).
            4. Generate trade signals (strategy layer).
            5. Simulate fills with execution rules (execution layer).
            6. Compute all performance statistics (metrics layer).
            7. Print a summary to the terminal.
            8. Save a full Excel workbook to output_dir.

        Parameters
        ----------
        n_permutations : int
            Number of shuffles for the Tier 5 permutation test. Set to 0
            to skip the permutation test (useful during rapid iteration).

        Returns
        -------
        MetricsEngine
            The fully computed metrics engine. Access metrics programmatically
            via .compute_all(), or print them via .summary_text().
        """
        _setup_logging()
        t_start = time.perf_counter()

        logger.info(
            f"{'='*60}\n"
            f"  BacktestEngine starting\n"
            f"  Instrument : {self.instrument}\n"
            f"  Strategy   : {self.strategy.name}\n"
            f"  Contracts  : {self.contracts}\n"
            f"  Period     : {self.start or 'start'} → {self.end or 'end'}\n"
            f"{'='*60}"
        )

        data = self._ensure_data_loaded()
        metrics_engine = self._run_pipeline(data, self.strategy)

        # Cache for walk_forward / parameter_sweep to reference if needed.
        self._last_metrics_engine = metrics_engine

        # Print summary to terminal.
        print(metrics_engine.summary_text())

        # Save Excel workbook.
        filename = self._make_filename(suffix="backtest")
        excel_path = metrics_engine.export_excel(self.output_dir / filename)
        print(f"  Excel saved → {excel_path}\n")

        elapsed = time.perf_counter() - t_start
        logger.info(f"Run complete in {elapsed:.1f}s.")

        return metrics_engine

    def walk_forward(
        self,
        train_pct: float = 0.60,
        val_pct:   float = 0.20,
        n_permutations: int = 1_000,
    ) -> dict[str, MetricsEngine]:
        """
        Run the full pipeline on three chronological splits and compare.

        Why chronological splits matter for financial data
        --------------------------------------------------
        Financial data has time structure: what was true in 2021 (low
        volatility bull market) may be completely untrue in 2022 (high
        volatility bear market). A random split could accidentally train
        on 2023 data and test on 2021 data, giving the strategy access
        to future market structure during development — another form of
        lookahead bias.

        The only valid split is chronological:
            |-- Train --|-- Validate --|-- Test --|
             (60%)          (20%)         (20%)
                 ↑               ↑             ↑
            develop here    check here    final answer

        If performance degrades gracefully across splits (e.g. 62% → 58%
        → 54% win rate), the strategy likely captures a real pattern.
        If it collapses (62% → 35%), the edge was regime-specific or
        overfit to the training period.

        Parameters
        ----------
        train_pct : float
            Fraction of total trading days allocated to the training split.
        val_pct : float
            Fraction allocated to the validation split. The remaining
            fraction (1 − train_pct − val_pct) is the test split.
        n_permutations : int
            Permutation test iterations per split. Default lower than run()
            because this method runs the pipeline three times.

        Returns
        -------
        dict mapping split name → MetricsEngine
            Keys: "train", "validate", "test"

        Side effects
        ------------
        Saves a comparison Excel to output_dir.
        """
        _setup_logging()

        if not (0 < train_pct < 1) or not (0 < val_pct < 1):
            raise ValueError("train_pct and val_pct must be between 0 and 1.")
        if train_pct + val_pct >= 1.0:
            raise ValueError(
                f"train_pct ({train_pct}) + val_pct ({val_pct}) must be < 1.0 "
                f"to leave room for a test split."
            )

        data = self._ensure_data_loaded()
        all_dates = np.unique(data.index.date)
        n_days = len(all_dates)

        train_end = int(n_days * train_pct)
        val_end   = int(n_days * (train_pct + val_pct))

        splits = {
            "train":    (all_dates[0],          all_dates[train_end - 1]),
            "validate": (all_dates[train_end],   all_dates[val_end - 1]),
            "test":     (all_dates[val_end],     all_dates[-1]),
        }

        logger.info("Walk-forward split boundaries:")
        for split_name, (s, e) in splits.items():
            logger.info(f"  {split_name:10s}: {s} → {e}")

        results: dict[str, MetricsEngine] = {}
        summary_rows: list[dict] = []

        for split_name, (split_start, split_end) in splits.items():
            logger.info(f"\nRunning split: {split_name.upper()}")

            # Filter data to this split's date range.
            mask        = (
                (data.index.date >= split_start) &
                (data.index.date <= split_end)
            )
            split_data  = data[mask]

            if split_data.empty:
                logger.warning(f"  Split '{split_name}' has no data. Skipping.")
                continue

            me = self._run_pipeline(
                split_data, self.strategy, n_permutations=n_permutations
            )
            results[split_name] = me

            m = me.compute_all(n_permutations=n_permutations)
            row = {
                "split":      split_name,
                "date_start": str(split_start),
                "date_end":   str(split_end),
                "n_days":     (split_end - split_start).days,
            }
            row.update({k: m.get(k, None) for k in SUMMARY_METRIC_KEYS})
            summary_rows.append(row)

        # Export comparison Excel.
        comparison_df = pd.DataFrame(summary_rows).set_index("split")
        filename = self._make_filename(suffix="walk_forward")
        out_path = self.output_dir / filename
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(str(out_path), engine="xlsxwriter") as writer:
            comparison_df.to_excel(writer, sheet_name="Walk-Forward Comparison")
            for split_name, me in results.items():
                # Write each split's trade log to its own sheet.
                me.trade_log.to_excel(
                    writer,
                    sheet_name=f"{split_name}_trades"[:31],  # Excel 31-char limit
                    index=False,
                )
            _format_comparison_sheet(
                writer.book,
                writer.sheets["Walk-Forward Comparison"],
                comparison_df,
            )

        print(f"\nWalk-forward comparison saved → {out_path}")
        print(comparison_df.to_string())

        return results

    def parameter_sweep(
        self,
        param_name: str,
        values: list[Any],
        n_permutations: int = 0,
    ) -> pd.DataFrame:
        """
        Run the backtest with one parameter varied across a list of values.

        Why parameter sensitivity matters
        ----------------------------------
        A robust strategy should make money across a range of parameter
        values, not just at one specific setting. If profits collapse when
        you change a parameter by 10%, the strategy was tuned too precisely
        to historical data — it "memorized" the past rather than discovering
        a durable pattern.

        The healthy pattern is a broad PLATEAU: the strategy works well
        across many values, with gradual degradation toward the edges.

        The dangerous pattern is a sharp CLIFF: the strategy only works at
        or near one exact value. In live trading, where fills, spreads,
        and market conditions are never identical to backtested conditions,
        you'll inevitably operate "off-peak." A cliff-edge strategy fails.

        Parameters
        ----------
        param_name : str
            The name of the parameter to vary. Must be a key in the
            strategy's DEFAULT_PARAMS or currently configured params.
        values : list
            The list of values to test. Each value replaces the current
            value of param_name; all other parameters remain unchanged.
        n_permutations : int
            Permutation test iterations per run. Default 0 (skip) because
            a sweep may run dozens of times.

        Returns
        -------
        pd.DataFrame
            One row per value. Index = param_name values. Columns =
            SUMMARY_METRIC_KEYS. Sorted by the input value order.

        Side effects
        ------------
        Saves a sensitivity Excel to output_dir.

        Example
        -------
        >>> df = engine.parameter_sweep("rr", [1.0, 1.2, 1.5, 1.8, 2.0, 2.5])
        >>> print(df[["win_rate", "ev_per_trade", "total_net_pnl"]])
        """
        _setup_logging()

        if param_name not in {**self.strategy.DEFAULT_PARAMS, **self.strategy.params}:
            raise ValueError(
                f"'{param_name}' is not a recognised parameter for "
                f"{self.strategy.__class__.__name__}. "
                f"Known parameters: {list(self.strategy.params.keys())}"
            )
        if len(values) == 0:
            raise ValueError("values list must not be empty.")

        data = self._ensure_data_loaded()
        rows: list[dict] = []

        for value in values:
            logger.info(f"Sweep: {param_name}={value!r}")

            # Build a new strategy with the overridden parameter.
            # All other parameters are inherited from self.strategy.params.
            new_params = {**self.strategy.params, param_name: value}
            try:
                new_strategy = self.strategy.__class__(params=new_params)
            except ValueError as exc:
                logger.warning(
                    f"  Skipping {param_name}={value!r}: "
                    f"validate_params() rejected it. ({exc})"
                )
                continue

            me = self._run_pipeline(
                data, new_strategy, n_permutations=n_permutations
            )
            m  = me.compute_all(n_permutations=n_permutations)

            row = {param_name: value}
            row.update({k: m.get(k) for k in SUMMARY_METRIC_KEYS})
            rows.append(row)

        sweep_df = pd.DataFrame(rows).set_index(param_name)

        # Export sensitivity Excel.
        filename = self._make_filename(suffix=f"sweep_{param_name}")
        out_path = self.output_dir / filename
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(str(out_path), engine="xlsxwriter") as writer:
            sweep_df.to_excel(writer, sheet_name="Parameter Sensitivity")

            wb = writer.book
            ws = writer.sheets["Parameter Sensitivity"]

            # Add a line chart of net P&L vs. the swept parameter.
            n = len(sweep_df)
            chart = wb.add_chart({"type": "line"})
            chart.add_series({
                "name":       "Net P&L",
                "categories": ["Parameter Sensitivity", 1, 0, n, 0],
                "values":     ["Parameter Sensitivity", 1,
                               list(sweep_df.columns).index("total_net_pnl") + 1,
                               n,
                               list(sweep_df.columns).index("total_net_pnl") + 1],
                "line":       {"color": "#2E75B6"},
            })
            chart.set_title({"name": f"Net P&L vs. {param_name}"})
            chart.set_x_axis({"name": param_name})
            chart.set_y_axis({"name": "Net P&L ($)"})
            chart.set_size({"width": 600, "height": 360})
            ws.insert_chart(f"A{n + 4}", chart)

        print(f"\nParameter sweep saved → {out_path}")
        print(sweep_df[["win_rate", "ev_per_trade", "total_net_pnl",
                         "sharpe_ratio"]].to_string())

        return sweep_df

    # ------------------------------------------------------------------
    # Internal pipeline — used by all three public methods
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        n_permutations: int = 10_000,
    ) -> MetricsEngine:
        """
        Core five-layer pipeline. All three public methods call this.

        Keeping the pipeline in one private method means any fix or
        improvement here automatically benefits run(), walk_forward(),
        and parameter_sweep() — no duplication.

        Parameters
        ----------
        data : pd.DataFrame
            Clean, validated 1-minute OHLCV data (any date range).
        strategy : BaseStrategy
            The strategy to run (may differ from self.strategy during sweeps).
        n_permutations : int
            Passed to MetricsEngine.compute_all().

        Returns
        -------
        MetricsEngine
            Fully computed. Call .summary_text() or .export_excel() on it.
        """
        if data.empty:
            raise ValueError(
                "Cannot run pipeline: data is empty. "
                "Check your start/end dates and the CSV file."
            )

        # ---- Layer 2a: Volatility (ATR) ----
        logger.info("Computing ATR...")
        vol_result = compute_daily_atr(data)

        # ---- Layer 2b: Regime (ADX + volatility flag → 4 regime labels) ----
        logger.info("Computing ADX and regime labels...")
        reg_result = compute_regime(
            data=data,
            minute_atr=vol_result.minute_atr,
            volatility_daily=vol_result.daily,
        )
        daily_indicators = reg_result.daily

        # ---- Layer 3: Strategy signals ----
        logger.info(f"Generating signals ({strategy.name})...")
        signals = strategy.generate_signals(data, daily_indicators)
        logger.info(f"  {len(signals)} signals generated.")

        if len(signals) == 0:
            logger.warning(
                "No signals generated. Check that your data covers the "
                "session_open/session_close window and that the strategy "
                "parameters are appropriate for the instrument."
            )

        # ---- Layer 4: Execution ----
        logger.info("Simulating trade execution...")
        trade_log = simulate_trades(
            data=data,
            signals=signals,
            daily_indicators=daily_indicators,
            instrument=self.instrument,
            contracts=self.contracts,
        )
        logger.info(f"  {len(trade_log)} trades executed.")

        # ---- Layer 5: Metrics ----
        logger.info("Computing performance metrics...")
        metrics_engine = MetricsEngine(
            trade_log=trade_log,
            initial_capital=self.initial_capital,
        )
        # Trigger computation now (rather than lazily) so any errors
        # surface immediately with a meaningful stack trace.
        metrics_engine.compute_all(n_permutations=n_permutations)

        return metrics_engine

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _ensure_data_loaded(self) -> pd.DataFrame:
        """
        Load the CSV on first call; return the cached DataFrame on subsequent
        calls.

        This is the "lazy loading" pattern. We do NOT load data in __init__
        because:
          1. Constructing an engine and inspecting its attributes should be
             fast and side-effect-free.
          2. Parameter sweep creates no new data loads (it reuses the cache).
          3. If the file path is wrong, the error appears at run() time with
             a clear call stack, not buried in __init__.
        """
        if self._data is not None:
            return self._data

        logger.info(f"Loading data from {self.data_path}...")
        load_result = load_csv(
            self.data_path,
            start_date=self.start,
            end_date=self.end,
        )
        logger.info(load_result.report.summary())

        if load_result.report.dropped_count > 0:
            logger.warning(
                f"  {load_result.report.dropped_count} rows were dropped "
                f"during validation. Inspect load_result.report.flagged "
                f"for details."
            )

        self._data = load_result.data
        return self._data

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _make_filename(self, suffix: str) -> str:
        """
        Build a descriptive, timestamped filename for Excel output.

        Format: {strategy_name}_{suffix}_{YYYYMMDD_HHMMSS}.xlsx

        The timestamp prevents overwriting previous runs, making it safe to
        call run() multiple times with different parameters without losing
        prior results.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitise the strategy name: replace characters that are invalid in
        # filenames on Windows and macOS with underscores.
        safe_name = (
            self.strategy.name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )
        return f"{safe_name}_{suffix}_{ts}.xlsx"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _setup_logging(level: int = logging.INFO) -> None:
    """
    Configure basic logging if no handlers are already registered.

    Using basicConfig() with force=False means this is a no-op if the
    calling application has already set up its own logging. This is the
    correct pattern for a library — we don't override the user's config,
    but we do provide sensible defaults for users who haven't configured
    logging at all.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def _format_comparison_sheet(wb, ws, df: pd.DataFrame) -> None:
    """
    Apply bold headers and number formatting to the walk-forward comparison
    sheet to make it easier to read at a glance.

    Highlights the "test" split row in a light yellow so it stands out as
    the out-of-sample result — the number that matters most.
    """
    bold_fmt   = wb.add_format({"bold": True})
    pct_fmt    = wb.add_format({"num_format": "0.00%"})
    dollar_fmt = wb.add_format({"num_format": '$#,##0.00'})
    highlight  = wb.add_format({"bg_color": "#FFF2CC", "bold": True})

    ws.set_column("A:A", 12)
    ws.set_column("B:Z", 16)

    # Highlight the "test" split row (the true out-of-sample result).
    test_row_idx = list(df.index).index("test") + 1 if "test" in df.index else None
    if test_row_idx is not None:
        for col in range(len(df.columns) + 1):
            ws.write(test_row_idx, col, ws, highlight)

    # Bold the header row.
    for col_idx, col_name in enumerate(["split"] + list(df.columns)):
        ws.write(0, col_idx, col_name, bold_fmt)
