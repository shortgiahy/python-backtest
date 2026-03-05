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
    METRICS LAYER     backtest/metrics/report.py              <-- YOU ARE HERE

=============================================================================
THIS FILE — backtest/metrics/report.py
=============================================================================
Accepts the trade log DataFrame from the execution layer and computes the
full five-tier performance analysis. Exports results to an Excel workbook
with eight sheets including two chart sheets.

This file answers the central question of the system: does this strategy
make money consistently, or does it only appear to work due to luck,
overfitting, or regime dependency?

HOW IT CONNECTS:
  - Receives trade_log (pd.DataFrame) from backtest/execution/fills.py
    via backtest/engine.py.
  - Returns a MetricsEngine object which engine.run() both uses internally
    (for the terminal summary printout) and returns to the caller for
    programmatic access.
  - The Excel workbook is written to output_dir specified in BacktestEngine.

DEPENDENCIES:
  - numpy    — permutation test, array operations
  - pandas   — all tabular operations
  - scipy    — t-test p-value
  - xlsxwriter — Excel chart rendering (install: pip install xlsxwriter)
    If xlsxwriter is not installed, export_excel() raises ImportError with
    installation instructions rather than crashing with an opaque message.
=============================================================================
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of consecutive trades in the rolling win-rate and EV window.
# 30 is large enough to smooth noise but small enough to detect regime shifts
# over a typical trading year (~250 trading days, ~1 trade/day = 250 trades).
ROLLING_WINDOW_TRADES: int = 30

# Used to annualize the Sharpe ratio. There are 252 trading days per year
# on average in US equity markets. Futures trade ~24 hours but volume
# concentrates during regular sessions, so 252 is the accepted convention.
TRADING_DAYS_PER_YEAR: float = 252.0

# Number of permutations for the permutation test.
# 10,000 is sufficient for a p-value resolution of 0.01 (1%).
# Raising this to 100,000 increases runtime ~10× for marginal precision gain.
N_PERMUTATIONS_DEFAULT: int = 10_000

# Random seed for the permutation test.
# Fixing the seed ensures the same backtest produces the same p-value every
# run. Change this only if you have reason to believe the seed is pathological
# for your specific data (extremely rare).
PERMUTATION_RANDOM_SEED: int = 42

# Columns expected in the trade log. Used for validation in __init__.
REQUIRED_TRADE_LOG_COLUMNS: list[str] = [
    "trade_id", "date", "direction", "entry_price", "exit_price",
    "stop_price", "gross_pnl", "commission", "net_pnl",
    "exit_reason", "won", "day_of_week", "hour_of_entry",
    "daily_atr", "adx_value", "regime",
]

# Day-of-week order for display (Monday first).
DOW_ORDER: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
]


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------

class MetricsEngine:
    """
    Computes and stores all performance metrics for a completed backtest.

    Usage
    -----
    >>> engine = MetricsEngine(trade_log_df, initial_capital=10_000.0)
    >>> metrics = engine.compute_all()
    >>> print(engine.summary_text())
    >>> engine.export_excel("results/my_backtest.xlsx")

    Attributes
    ----------
    trade_log : pd.DataFrame
        The trade log passed in at construction, with derived columns added
        (year, equity).
    initial_capital : float
        Starting account balance in dollars.
    """

    def __init__(
        self,
        trade_log: pd.DataFrame,
        initial_capital: float = 10_000.0,
    ) -> None:
        """
        Parameters
        ----------
        trade_log : pd.DataFrame
            Output of backtest/execution/fills.py simulate_trades(). Must
            contain at minimum the columns in REQUIRED_TRADE_LOG_COLUMNS.
        initial_capital : float
            The dollar balance at the start of the backtest. Used to compute
            drawdown percentages, Calmar ratio, and the equity curve baseline.
        """
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}.")

        self.initial_capital = initial_capital
        self._raw_metrics: dict | None = None   # cached after first compute_all()

        self.trade_log = self._prepare_log(trade_log)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compute_all(self, n_permutations: int = N_PERMUTATIONS_DEFAULT) -> dict:
        """
        Run all five tiers of analysis and return every metric in one dict.

        The dict contains:
          - Scalar metrics (floats, ints, strings): all Tier 1/3/4/5 values.
          - DataFrame metrics: by_year_df, by_regime_df, by_dow_df,
            by_hour_df — one row per group, standard stat columns.
          - Series metrics: equity_curve, rolling_win_rate, rolling_ev —
            indexed by trade number (integer).

        Results are cached after the first call. Subsequent calls return the
        cached dict immediately (the permutation test is expensive).

        Parameters
        ----------
        n_permutations : int
            Number of shuffles for the Tier 5 permutation test.
        """
        if self._raw_metrics is not None:
            return self._raw_metrics

        if self.trade_log.empty:
            logger.warning("Trade log is empty. No metrics to compute.")
            self._raw_metrics = {}
            return self._raw_metrics

        m: dict = {}
        m.update(self._tier1_edge())
        m.update(self._tier2_stability())
        m.update(self._tier3_drawdown())
        m.update(self._tier4_honesty())
        m.update(self._tier5_overfit(n_permutations))

        self._raw_metrics = m
        return m

    def summary_text(self) -> str:
        """
        Return a formatted multi-line string suitable for printing to the
        terminal at the end of engine.run().

        Covers the most important Tier 1 and Tier 3 metrics only — a full
        breakdown is in the Excel workbook.
        """
        m = self.compute_all()
        if not m:
            return "No trades were executed. Cannot compute metrics."

        sig = "YES (p<0.05)" if m.get("p_value", 1.0) < 0.05 else "NO"

        return (
            f"\n{'='*60}\n"
            f"  BACKTEST RESULTS\n"
            f"{'='*60}\n"
            f"  Trades:          {m['n_trades']:>8,}\n"
            f"  Win Rate:        {m['win_rate']:>8.1%}\n"
            f"  Avg Win:         {m['avg_win']:>8.2f}\n"
            f"  Avg Loss:        {m['avg_loss']:>8.2f}\n"
            f"  Realized R:R:    {m['realized_rr']:>8.2f}\n"
            f"  EV / Trade:      {m['ev_per_trade']:>8.2f}\n"
            f"  Profit Factor:   {m['profit_factor']:>8.2f}\n"
            f"  Net P&L:         {m['total_net_pnl']:>8.2f}\n"
            f"  Statistically significant: {sig}\n"
            f"{'─'*60}\n"
            f"  Max Drawdown $:  {m['max_drawdown_dollars']:>8.2f}\n"
            f"  Max Drawdown %:  {m['max_drawdown_pct']:>8.1%}\n"
            f"  Max Consec. L:   {m['max_consecutive_losses']:>8}\n"
            f"  Sharpe (ann.):   {m['sharpe_ratio']:>8.2f}\n"
            f"  Calmar Ratio:    {m['calmar_ratio']:>8.2f}\n"
            f"{'─'*60}\n"
            f"  Commission Drag: {m['commission_drag_pct']:>8.1f}% of gross\n"
            f"  Permutation p:   {m['permutation_p_value']:>8.3f} "
            f"({m['pct_random_worse']:.1%} of random worse)\n"
            f"{'='*60}\n"
        )

    def export_excel(self, output_path: str | Path) -> Path:
        """
        Write all metrics and charts to an Excel workbook.

        Sheets created:
            1. Summary        — key scalar metrics in a readable table
            2. Trade Log      — full trade log, every row and column
            3. By Year        — per-year performance breakdown
            4. By Regime      — performance split by the four regime labels
            5. By Day of Week — Monday through Friday breakdown
            6. By Hour        — performance by entry hour
            7. Equity Curve   — running balance line chart
            8. Rolling Metrics— rolling 30-trade win rate and EV chart

        Parameters
        ----------
        output_path : str or Path
            Full path to the output .xlsx file. The directory is created if
            it does not exist.

        Returns
        -------
        Path
            The resolved absolute path to the written file.
        """
        try:
            import xlsxwriter  # noqa: F401 — just checking it's installed
        except ImportError:
            raise ImportError(
                "xlsxwriter is required for Excel export. "
                "Install it with: pip install xlsxwriter"
            )

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        m = self.compute_all()
        tl = self.trade_log

        with pd.ExcelWriter(str(output_path), engine="xlsxwriter") as writer:
            wb = writer.book

            self._write_summary_sheet(writer, wb, m)
            self._write_trade_log_sheet(writer, tl)
            self._write_group_sheet(writer, m["by_year_df"],    "By Year")
            self._write_group_sheet(writer, m["by_regime_df"],  "By Regime")
            self._write_group_sheet(writer, m["by_dow_df"],     "By Day of Week")
            self._write_group_sheet(writer, m["by_hour_df"],    "By Hour")
            self._write_equity_curve_sheet(writer, wb, m["equity_curve"])
            self._write_rolling_metrics_sheet(writer, wb, m)

        logger.info(f"Excel workbook written to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Tier 1 — does the edge exist?
    # ------------------------------------------------------------------

    def _tier1_edge(self) -> dict:
        """
        Core edge metrics: win rate, average win/loss, EV, profit factor,
        and statistical significance via the one-sample t-test.

        The t-test null hypothesis: mean(net_pnl) == 0 (no edge).
        We reject the null (claim edge exists) when p < 0.05.

        WARNING: The t-test assumes trade P&Ls are roughly normally
        distributed and independent. Neither assumption is perfectly true for
        day trading (P&Ls are bounded by stop/target, and sequential trades
        on the same instrument have some correlation). Treat the p-value as
        a guide, not a guarantee. The permutation test in Tier 5 is a
        non-parametric alternative that does not require these assumptions.
        """
        tl  = self.trade_log
        net = tl["net_pnl"]
        gross = tl["gross_pnl"]
        won = tl["won"]

        n_trades = len(tl)
        n_wins   = int(won.sum())
        n_losses = n_trades - n_wins

        win_rate = n_wins / n_trades if n_trades > 0 else 0.0

        win_pnls  = net[won]
        loss_pnls = net[~won]

        avg_win  = float(win_pnls.mean())  if len(win_pnls)  > 0 else 0.0
        avg_loss = float(loss_pnls.mean()) if len(loss_pnls) > 0 else 0.0

        # Realized R:R: how many dollars of average win per dollar of average loss.
        # avg_loss is negative, so we take abs().
        realized_rr = (
            abs(avg_win / avg_loss) if avg_loss != 0.0 else float("inf")
        )

        # EV per trade: weighted average outcome.
        ev_per_trade = win_rate * avg_win + (1.0 - win_rate) * avg_loss

        # Profit factor: ratio of total gross profits to total gross losses.
        total_gross_wins   = float(gross[won].sum())
        total_gross_losses = float(abs(gross[~won].sum()))
        profit_factor = (
            total_gross_wins / total_gross_losses
            if total_gross_losses > 0
            else float("inf")
        )

        # One-sample t-test against zero.
        # We need at least 2 observations; with 1 or 0, the test is undefined.
        if n_trades >= 2:
            t_stat, p_value = stats.ttest_1samp(net.dropna(), 0.0)
        else:
            t_stat, p_value = 0.0, 1.0

        return {
            "n_trades":        n_trades,
            "n_wins":          n_wins,
            "n_losses":        n_losses,
            "win_rate":        win_rate,
            "avg_win":         avg_win,
            "avg_loss":        avg_loss,
            "realized_rr":     realized_rr,
            "ev_per_trade":    ev_per_trade,
            "profit_factor":   profit_factor,
            "total_net_pnl":   float(net.sum()),
            "total_gross_pnl": float(gross.sum()),
            "total_commission": float(tl["commission"].sum()),
            "t_statistic":     float(t_stat),
            "p_value":         float(p_value),
        }

    # ------------------------------------------------------------------
    # Tier 2 — is the edge stable?
    # ------------------------------------------------------------------

    def _tier2_stability(self) -> dict:
        """
        Breakdown tables and rolling series that reveal whether the edge is
        consistent across time, market regimes, days, and hours.

        Returns DataFrames (stored in the metrics dict under *_df keys) that
        are written directly to Excel sheets by export_excel().
        """
        tl = self.trade_log

        # Rolling metrics — indexed by trade number (0, 1, 2, …)
        rolling_wr = tl["won"].rolling(ROLLING_WINDOW_TRADES).mean()
        rolling_ev = tl["net_pnl"].rolling(ROLLING_WINDOW_TRADES).mean()

        return {
            "rolling_win_rate": rolling_wr,
            "rolling_ev":       rolling_ev,
            "by_year_df":   _group_stats(tl, "year"),
            "by_regime_df": _group_stats(
                tl[tl["regime"].notna()], "regime"
            ),
            "by_dow_df":    _group_stats(
                tl, "day_of_week",
                # Sort Monday–Friday, not alphabetically.
                category_order=DOW_ORDER,
            ),
            "by_hour_df":   _group_stats(tl, "hour_of_entry"),
        }

    # ------------------------------------------------------------------
    # Tier 3 — does it survive drawdown limits?
    # ------------------------------------------------------------------

    def _tier3_drawdown(self) -> dict:
        """
        Drawdown metrics and risk-adjusted return ratios.

        Equity curve is built as:
            equity[0] = initial_capital + net_pnl[0]
            equity[n] = equity[n-1] + net_pnl[n]

        Max drawdown: the largest fall from any prior peak in the equity curve,
        measured both in dollars and as a percentage of the peak.

        Sharpe ratio: annualized using daily P&L (trades grouped by date).
        Uses daily granularity because two trades on the same day share market
        exposure — treating them independently would overstate the sample size.

        Calmar ratio: annualized return divided by absolute max drawdown %.
        Requires at least two distinct trade dates to compute elapsed years.
        """
        tl = self.trade_log
        equity = self.initial_capital + tl["net_pnl"].cumsum().reset_index(drop=True)

        max_dd_dollars, max_dd_pct = _compute_max_drawdown(equity)
        max_dd_duration = _longest_streak(equity < equity.cummax())
        max_consec_losses = _longest_streak(~tl["won"])

        # --- Sharpe ratio (annualized, on daily P&L) ---
        daily_pnl = tl.groupby("date")["net_pnl"].sum()
        if len(daily_pnl) > 1 and daily_pnl.std() > 0:
            sharpe = (daily_pnl.mean() / daily_pnl.std()) * math.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        else:
            sharpe = 0.0

        # --- Calmar ratio ---
        dates = pd.to_datetime(tl["date"])
        years_elapsed = (dates.max() - dates.min()).days / 365.25
        total_net_pnl = float(tl["net_pnl"].sum())

        if years_elapsed > 0 and self.initial_capital > 0:
            annualized_return_pct = (total_net_pnl / self.initial_capital) / years_elapsed
        else:
            annualized_return_pct = 0.0

        if max_dd_pct < 0:
            calmar = annualized_return_pct / abs(max_dd_pct)
        else:
            calmar = float("inf")   # no drawdown at all

        return {
            "equity_curve":             equity,
            "max_drawdown_dollars":     max_dd_dollars,
            "max_drawdown_pct":         max_dd_pct,
            "max_drawdown_duration_trades": max_dd_duration,
            "max_consecutive_losses":   max_consec_losses,
            "sharpe_ratio":             sharpe,
            "annualized_return_pct":    annualized_return_pct,
            "calmar_ratio":             calmar,
        }

    # ------------------------------------------------------------------
    # Tier 4 — is the backtest honest?
    # ------------------------------------------------------------------

    def _tier4_honesty(self) -> dict:
        """
        Metrics that reveal structural issues in the backtest:
        how much commission cost matters, and how often the strategy exits
        for each reason.

        Commission drag > 30% of gross profit is a serious warning sign.
        It means the edge is almost entirely consumed by transaction costs,
        and any slight real-world underperformance (wider spreads, partial
        fills, data delays) will turn the strategy negative.
        """
        tl = self.trade_log
        total_gross = float(tl["gross_pnl"].sum())
        total_commission = float(tl["commission"].sum())

        commission_drag_pct = (
            (total_commission / total_gross * 100.0) if total_gross > 0 else float("inf")
        )

        exit_reason_counts = tl["exit_reason"].value_counts().to_dict()

        return {
            "commission_drag_pct":    commission_drag_pct,
            "exit_reason_breakdown":  exit_reason_counts,
        }

    # ------------------------------------------------------------------
    # Tier 5 — is it overfit?
    # ------------------------------------------------------------------

    def _tier5_overfit(self, n_permutations: int) -> dict:
        """
        Permutation test: shuffle the ORDER of trade P&Ls and compare
        the resulting max drawdown to the actual max drawdown.

        Why order matters even though total P&L doesn't:
            The sum of a sequence is the same regardless of order, but the
            maximum drawdown — the worst peak-to-trough loss — depends
            critically on whether large losses cluster together or are spread
            across the sequence. If your strategy's losses happened to be
            spread out (good luck with sequencing), the max drawdown looks
            better than it "deserves." The permutation test reveals this.

        Interpretation of pct_random_worse:
            0.90 → 90% of random orderings had WORSE max drawdown than yours.
                   Your trade ordering was favorable. Not necessarily a red flag,
                   but don't assume this luck will repeat.
            0.50 → Your max drawdown is about average for this set of outcomes.
                   Normal.
            0.10 → Only 10% of random orderings had worse drawdown. Your losses
                   clustered unusually badly. Your realized risk was worse than
                   a random sequence of the same outcomes. Investigate whether
                   losses are regime-correlated.

        permutation_p_value:
            Fraction of permutations with BETTER (lower absolute) max drawdown
            than actual. Low = your drawdown is worse than random → losses
            clustered. High = your drawdown is better than random → lucky ordering.
        """
        net_pnl_array = self.trade_log["net_pnl"].values.astype(float)
        n = len(net_pnl_array)

        actual_equity = self.initial_capital + np.cumsum(net_pnl_array)
        actual_max_dd, _ = _compute_max_drawdown(pd.Series(actual_equity))

        rng = np.random.default_rng(PERMUTATION_RANDOM_SEED)

        n_random_better = 0    # permutations with BETTER (less negative) max DD
        n_random_worse  = 0    # permutations with WORSE (more negative) max DD

        for _ in range(n_permutations):
            perm = rng.permutation(net_pnl_array)
            perm_equity = self.initial_capital + np.cumsum(perm)
            perm_max_dd, _ = _compute_max_drawdown(pd.Series(perm_equity))

            # max_dd is negative (a loss). More negative = worse.
            if perm_max_dd < actual_max_dd:    # perm is worse
                n_random_worse += 1
            elif perm_max_dd > actual_max_dd:  # perm is better
                n_random_better += 1

        pct_random_worse  = n_random_worse  / n_permutations
        pct_random_better = n_random_better / n_permutations

        # p_value here = fraction of permutations that produced a BETTER
        # max drawdown than actual. Low p_value = your drawdown was unusually
        # bad (losses clustered). High p_value = your drawdown was unusually
        # good (losses were spread out, lucky ordering).
        permutation_p_value = pct_random_better

        logger.info(
            f"Permutation test ({n_permutations:,} shuffles): "
            f"{pct_random_worse:.1%} of random orderings had WORSE max drawdown. "
            f"Permutation p-value: {permutation_p_value:.3f}"
        )

        return {
            "permutation_p_value": permutation_p_value,
            "pct_random_worse":    pct_random_worse,
            "pct_random_better":   pct_random_better,
            "n_permutations":      n_permutations,
        }

    # ------------------------------------------------------------------
    # Excel sheet writers
    # ------------------------------------------------------------------

    def _write_summary_sheet(self, writer, wb, m: dict) -> None:
        """Write the Summary sheet: all key scalar metrics in two columns."""
        rows = [
            ("TIER 1 — EDGE",                              ""),
            ("Total Trades",                               m.get("n_trades", "N/A")),
            ("Wins / Losses",
             f"{m.get('n_wins','?')} / {m.get('n_losses','?')}"),
            ("Win Rate",                  f"{m.get('win_rate', 0):.1%}"),
            ("Avg Win ($)",               f"{m.get('avg_win', 0):.2f}"),
            ("Avg Loss ($)",              f"{m.get('avg_loss', 0):.2f}"),
            ("Realized R:R",              f"{m.get('realized_rr', 0):.2f}"),
            ("EV / Trade ($)",            f"{m.get('ev_per_trade', 0):.2f}"),
            ("Profit Factor",             f"{m.get('profit_factor', 0):.2f}"),
            ("Total Net P&L ($)",         f"{m.get('total_net_pnl', 0):.2f}"),
            ("Total Commission ($)",      f"{m.get('total_commission', 0):.2f}"),
            ("t-Statistic",               f"{m.get('t_statistic', 0):.3f}"),
            ("p-Value",                   f"{m.get('p_value', 1):.4f}"),
            ("Statistically Significant",
             "YES (p<0.05)" if m.get("p_value", 1) < 0.05 else "NO"),
            ("",                                           ""),
            ("TIER 3 — DRAWDOWN",                          ""),
            ("Max Drawdown ($)",          f"{m.get('max_drawdown_dollars', 0):.2f}"),
            ("Max Drawdown (%)",          f"{m.get('max_drawdown_pct', 0):.1%}"),
            ("Max DD Duration (trades)",  m.get("max_drawdown_duration_trades", 0)),
            ("Max Consecutive Losses",    m.get("max_consecutive_losses", 0)),
            ("Sharpe Ratio (annualized)", f"{m.get('sharpe_ratio', 0):.2f}"),
            ("Calmar Ratio",              f"{m.get('calmar_ratio', 0):.2f}"),
            ("",                                           ""),
            ("TIER 4 — HONESTY",                           ""),
            ("Commission Drag",           f"{m.get('commission_drag_pct', 0):.1f}% of gross"),
            ("Exit: Stop",                m.get("exit_reason_breakdown", {}).get("stop",   0)),
            ("Exit: Target",              m.get("exit_reason_breakdown", {}).get("target", 0)),
            ("Exit: EOD",                 m.get("exit_reason_breakdown", {}).get("eod",    0)),
            ("",                                           ""),
            ("TIER 5 — OVERFIT",                           ""),
            ("Permutation p-value",       f"{m.get('permutation_p_value', 0):.3f}"),
            ("% Random Sequences Worse",  f"{m.get('pct_random_worse', 0):.1%}"),
            ("Permutations Run",          f"{m.get('n_permutations', 0):,}"),
        ]

        summary_df = pd.DataFrame(rows, columns=["Metric", "Value"])
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        ws  = writer.sheets["Summary"]
        fmt = wb.add_format({"bold": True, "bg_color": "#D9E1F2"})
        ws.set_column("A:A", 35)
        ws.set_column("B:B", 22)

        # Bold + shade the tier header rows
        for row_idx, (label, _) in enumerate(rows, start=1):
            if label.startswith("TIER"):
                ws.write(row_idx, 0, label, fmt)

    def _write_trade_log_sheet(self, writer, tl: pd.DataFrame) -> None:
        """Write every trade as a row with column autofit and freeze-pane."""
        sheet_name = "Trade Log"
        display = tl.drop(columns=["year", "equity"], errors="ignore")
        display.to_excel(writer, sheet_name=sheet_name, index=False)

        ws = writer.sheets[sheet_name]
        ws.freeze_panes(1, 0)   # freeze the header row

        # Auto-width columns based on content length.
        for col_idx, col_name in enumerate(display.columns):
            col_width = max(len(str(col_name)), 10)
            ws.set_column(col_idx, col_idx, col_width)

    def _write_group_sheet(self, writer, df: pd.DataFrame, sheet_name: str) -> None:
        """Write a breakdown DataFrame to its own sheet."""
        df.to_excel(writer, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        ws.set_column("A:A", 20)
        ws.set_column("B:Z", 14)

    def _write_equity_curve_sheet(
        self, writer, wb, equity: pd.Series
    ) -> None:
        """
        Write the equity curve values and insert a line chart.

        The chart shows running account balance vs. trade number. A healthy
        equity curve trends upward with drawdowns that recover. A flat or
        downward-sloping curve means the strategy is losing money.
        """
        sheet_name = "Equity Curve"
        df = pd.DataFrame({
            "Trade #": range(1, len(equity) + 1),
            "Equity ($)": equity.values,
        })
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        ws = writer.sheets[sheet_name]
        chart = wb.add_chart({"type": "line"})

        n_rows = len(df)
        chart.add_series({
            "name":       "Account Balance",
            "categories": [sheet_name, 1, 0, n_rows, 0],   # Trade # column
            "values":     [sheet_name, 1, 1, n_rows, 1],   # Equity column
            "line":       {"color": "#2E75B6", "width": 1.5},
        })
        chart.set_title({"name": "Equity Curve"})
        chart.set_x_axis({"name": "Trade Number"})
        chart.set_y_axis({"name": "Account Balance ($)"})
        chart.set_legend({"none": True})
        chart.set_size({"width": 720, "height": 400})

        ws.insert_chart("D2", chart)

    def _write_rolling_metrics_sheet(self, writer, wb, m: dict) -> None:
        """
        Write rolling 30-trade win rate and EV and insert a dual-series chart.

        A stable rolling win rate (flat line near the overall win rate) is
        the strongest visual evidence that the edge is real and consistent.
        A declining rolling win rate at the end of the backtest period is a
        red flag — it may mean the strategy has stopped working.
        """
        sheet_name = "Rolling Metrics"
        rolling_wr = m["rolling_win_rate"].reset_index(drop=True)
        rolling_ev = m["rolling_ev"].reset_index(drop=True)

        df = pd.DataFrame({
            "Trade #":            range(1, len(rolling_wr) + 1),
            f"Win Rate (last {ROLLING_WINDOW_TRADES})": rolling_wr.values,
            f"EV/Trade (last {ROLLING_WINDOW_TRADES})": rolling_ev.values,
        }).dropna()

        df.to_excel(writer, sheet_name=sheet_name, index=False)

        ws = writer.sheets[sheet_name]
        n = len(df)

        # Win rate chart (left y-axis)
        chart = wb.add_chart({"type": "line"})
        chart.add_series({
            "name":       f"Rolling Win Rate ({ROLLING_WINDOW_TRADES} trades)",
            "categories": [sheet_name, 1, 0, n, 0],
            "values":     [sheet_name, 1, 1, n, 1],
            "line":       {"color": "#70AD47", "width": 1.5},
        })
        chart.add_series({
            "name":       f"Rolling EV/Trade ({ROLLING_WINDOW_TRADES} trades)",
            "categories": [sheet_name, 1, 0, n, 0],
            "values":     [sheet_name, 1, 2, n, 2],
            "line":       {"color": "#ED7D31", "width": 1.5},
            "y2_axis":    True,    # second y-axis on the right
        })
        chart.set_title({"name": f"Rolling {ROLLING_WINDOW_TRADES}-Trade Metrics"})
        chart.set_x_axis({"name": "Trade Number"})
        chart.set_y_axis({"name": "Win Rate"})
        chart.set_y2_axis({"name": "EV / Trade ($)"})
        chart.set_size({"width": 720, "height": 400})

        ws.insert_chart("E2", chart)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_log(self, trade_log: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date column, add derived columns (year, equity), and validate
        the schema. Returns a working copy — never modifies the original.
        """
        if trade_log.empty:
            return trade_log.copy()

        missing = [c for c in REQUIRED_TRADE_LOG_COLUMNS if c not in trade_log.columns]
        if missing:
            raise ValueError(
                f"MetricsEngine: trade log is missing columns: {missing}."
            )

        tl = trade_log.copy()
        tl["date"] = pd.to_datetime(tl["date"])
        tl["year"] = tl["date"].dt.year

        # equity is the running account balance, starting from initial_capital.
        # reset_index(drop=True) ensures the index is 0, 1, 2, … (integer),
        # which is required for the Excel chart row references.
        tl["equity"] = self.initial_capital + tl["net_pnl"].cumsum().reset_index(drop=True)

        return tl


# ---------------------------------------------------------------------------
# Module-level helper functions (not class methods — usable independently)
# ---------------------------------------------------------------------------

def _group_stats(
    tl: pd.DataFrame,
    group_col: str,
    category_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute standard performance statistics for each value in group_col.

    Parameters
    ----------
    tl : pd.DataFrame
        Trade log (output of MetricsEngine._prepare_log).
    group_col : str
        Column name to group by (e.g., "year", "regime", "day_of_week").
    category_order : list[str] or None
        If provided, the output rows are sorted in this order. Used for
        day-of-week grouping so Monday appears before Friday.

    Returns
    -------
    pd.DataFrame
        One row per group. Columns:
            n_trades, n_wins, win_rate, avg_win, avg_loss,
            ev_per_trade, profit_factor, total_net_pnl
    """
    results = []

    for group_val, sub in tl.groupby(group_col, sort=True):
        net  = sub["net_pnl"]
        gross = sub["gross_pnl"]
        won  = sub["won"]

        n      = len(sub)
        n_wins = int(won.sum())
        wr     = n_wins / n if n > 0 else 0.0

        win_pnls  = net[won]
        loss_pnls = net[~won]
        avg_win   = float(win_pnls.mean())  if len(win_pnls)  > 0 else 0.0
        avg_loss  = float(loss_pnls.mean()) if len(loss_pnls) > 0 else 0.0
        ev        = wr * avg_win + (1.0 - wr) * avg_loss

        gw = float(gross[won].sum())
        gl = float(abs(gross[~won].sum()))
        pf = gw / gl if gl > 0 else float("inf")

        results.append({
            group_col:       group_val,
            "n_trades":      n,
            "n_wins":        n_wins,
            "win_rate":      round(wr,  4),
            "avg_win":       round(avg_win,  2),
            "avg_loss":      round(avg_loss, 2),
            "ev_per_trade":  round(ev,  2),
            "profit_factor": round(pf,  3),
            "total_net_pnl": round(float(net.sum()), 2),
        })

    df = pd.DataFrame(results).set_index(group_col)

    if category_order:
        # Reindex to the given order, keeping only rows that actually exist.
        df = df.reindex([c for c in category_order if c in df.index])

    return df


def _compute_max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """
    Compute the maximum drawdown in dollars and as a fraction of the peak.

    The drawdown at each point is how far the equity has fallen from its
    highest prior value (the "high-water mark" or "running peak").

    Returns
    -------
    (max_dd_dollars, max_dd_fraction) : both are zero or negative.
        max_dd_dollars  — e.g., -1500.0 means the worst drop was $1,500
        max_dd_fraction — e.g., -0.15   means the worst drop was 15% of peak

    If equity never falls below its prior peak (every trade is profitable),
    both values are 0.0.
    """
    equity = equity.reset_index(drop=True)
    peak   = equity.cummax()

    drawdown_dollars = equity - peak            # always <= 0
    max_dd_dollars   = float(drawdown_dollars.min())

    # Percentage drawdown relative to the peak at each point.
    # Guard against peak == 0 (should not happen if initial_capital > 0).
    drawdown_pct = drawdown_dollars / peak.replace(0, float("nan"))
    max_dd_pct   = float(drawdown_pct.min())

    return max_dd_dollars, max_dd_pct


def _longest_streak(bool_series: pd.Series) -> int:
    """
    Find the length of the longest consecutive run of True values in a
    boolean Series.

    Used for two metrics:
      - Max drawdown duration: bool_series = (equity < running_peak)
      - Max consecutive losses: bool_series = (~won)

    This is a pure Python loop, which is acceptable here because it runs
    only once per backtest over at most a few thousand values.
    """
    max_run = 0
    current = 0
    for val in bool_series:
        if val:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run
