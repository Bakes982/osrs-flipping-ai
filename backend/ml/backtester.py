"""
Backtesting Engine for OSRS Flipping AI
Replays historical price data, makes predictions at each timestamp,
compares predicted vs actual outcomes, and generates performance reports.
Compares: ML model vs simple margin scanning vs VWAP-only baseline.
"""

import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from backend.database import (
    get_db, PriceSnapshot, FlipHistory,
    get_price_history, get_item_flips,
)
from backend.ml.feature_engine import FeatureEngine, HORIZONS, HORIZON_SECONDS
from backend.ml.forecaster import (
    MultiHorizonForecaster, DIRECTION_LABELS, DIRECTION_NAMES,
    FLAT_THRESHOLD_PCT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GE Tax constants
# ---------------------------------------------------------------------------

GE_TAX_RATE = 0.02
GE_TAX_CAP = 5_000_000


def calculate_tax(sell_price: int) -> int:
    """Calculate GE tax on a sell price."""
    return min(int(sell_price * GE_TAX_RATE), GE_TAX_CAP)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """A single prediction and its outcome."""
    timestamp: datetime
    item_id: int
    horizon: str
    strategy: str  # "ml", "margin", "vwap"

    predicted_buy: int = 0
    predicted_sell: int = 0
    predicted_direction: str = "flat"
    confidence: float = 0.0

    actual_buy: int = 0
    actual_sell: int = 0
    actual_direction: str = "flat"

    # Computed after outcome
    direction_correct: bool = False
    price_error: float = 0.0
    price_error_pct: float = 0.0
    simulated_profit: int = 0  # GP profit if we'd acted on this prediction


@dataclass
class StrategyMetrics:
    """Aggregated metrics for a single strategy + horizon."""
    strategy: str
    horizon: str

    total_predictions: int = 0
    direction_correct: int = 0
    direction_accuracy: float = 0.0

    mae: float = 0.0       # Mean Absolute Error (GP)
    mape: float = 0.0      # Mean Absolute Percentage Error
    rmse: float = 0.0      # Root Mean Squared Error

    profitable_predictions: int = 0
    profitable_accuracy: float = 0.0

    cumulative_profit: int = 0       # GP
    max_drawdown: int = 0            # worst peak-to-trough (GP)
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class BacktestReport:
    """Full backtest report across all strategies and horizons."""
    item_id: int
    start_time: datetime
    end_time: datetime
    total_snapshots: int

    strategies: Dict[str, Dict[str, StrategyMetrics]] = field(default_factory=dict)
    # Key: strategy_name -> {horizon -> StrategyMetrics}

    predictions: List[PredictionResult] = field(default_factory=list)

    elapsed_seconds: float = 0.0


class Backtester:
    """
    Backtesting engine that replays historical price data and
    evaluates ML, margin-scan, and VWAP-only prediction strategies.
    """

    def __init__(self, model_dir: str = "models"):
        self.feature_engine = FeatureEngine()
        self.forecaster = MultiHorizonForecaster(model_dir=model_dir)
        self._models_loaded = False

    def load_models(self) -> int:
        """Load trained ML models for backtesting."""
        count = self.forecaster.load_models()
        self._models_loaded = count > 0
        return count

    # ------------------------------------------------------------------
    # Main Backtest Interface
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        item_id: int,
        hours: int = 48,
        step_interval: int = 60,
        horizons: Optional[List[str]] = None,
    ) -> BacktestReport:
        """
        Run a full backtest for a single item.

        Parameters
        ----------
        item_id : int
            OSRS item ID.
        hours : int
            Hours of historical data to replay.
        step_interval : int
            Seconds between prediction steps (default: 60s).
        horizons : list of str, optional
            Which horizons to test. Defaults to all.

        Returns
        -------
        BacktestReport
            Full report with per-strategy, per-horizon metrics.
        """
        start_time = time.time()

        if horizons is None:
            horizons = HORIZONS

        # Load data
        db = get_db()
        try:
            snapshots = get_price_history(db, item_id, hours=hours)
            flips = get_item_flips(db, item_id, days=max(hours // 24, 7))
        finally:
            db.close()

        if len(snapshots) < 100:
            logger.warning(
                f"Item {item_id}: Only {len(snapshots)} snapshots, "
                f"need at least 100 for backtest"
            )
            return BacktestReport(
                item_id=item_id,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                total_snapshots=len(snapshots),
            )

        # Load ML models if not already loaded
        if not self._models_loaded:
            self.load_models()

        report = BacktestReport(
            item_id=item_id,
            start_time=snapshots[0].timestamp,
            end_time=snapshots[-1].timestamp,
            total_snapshots=len(snapshots),
        )

        # Build a time-indexed structure for fast lookups
        price_index = self._build_price_index(snapshots)

        # Determine step points: skip the first chunk to have enough history
        min_history = 60  # need at least 60 snapshots for feature computation
        step_delta = timedelta(seconds=step_interval)

        current_time = snapshots[min_history].timestamp
        end_time = snapshots[-1].timestamp

        all_results: List[PredictionResult] = []

        step_count = 0
        while current_time < end_time:
            # Get history up to current_time
            history = [s for s in snapshots if s.timestamp <= current_time]
            if len(history) < 30:
                current_time += step_delta
                continue

            current_price = history[-1].instant_buy
            current_sell = history[-1].instant_sell
            if not current_price or not current_sell or current_price <= 0:
                current_time += step_delta
                continue

            # Compute features at this point in time
            try:
                features = self.feature_engine.compute_features(
                    item_id, history, flips,
                )
            except Exception:
                current_time += step_delta
                continue

            # For each horizon, make predictions with each strategy
            for horizon in horizons:
                horizon_secs = HORIZON_SECONDS[horizon]
                target_time = current_time + timedelta(seconds=horizon_secs)

                # Find actual price at target time
                actual = self._find_price_at_time(price_index, target_time)
                if actual is None:
                    continue

                actual_buy, actual_sell = actual

                # Determine actual direction
                actual_change_pct = (
                    (actual_buy - current_price) / current_price * 100
                )
                if actual_change_pct > FLAT_THRESHOLD_PCT:
                    actual_dir = "up"
                elif actual_change_pct < -FLAT_THRESHOLD_PCT:
                    actual_dir = "down"
                else:
                    actual_dir = "flat"

                # Strategy 1: ML model
                ml_pred = self._ml_strategy(
                    item_id, horizon, features, current_price, current_sell,
                )
                ml_result = self._evaluate_prediction(
                    current_time, item_id, horizon, "ml",
                    ml_pred, actual_buy, actual_sell, actual_dir, current_price,
                )
                all_results.append(ml_result)

                # Strategy 2: Simple margin scanning
                margin_pred = self._margin_strategy(
                    current_price, current_sell, horizon,
                )
                margin_result = self._evaluate_prediction(
                    current_time, item_id, horizon, "margin",
                    margin_pred, actual_buy, actual_sell, actual_dir, current_price,
                )
                all_results.append(margin_result)

                # Strategy 3: VWAP-only baseline
                vwap_pred = self._vwap_strategy(
                    features, current_price, current_sell, horizon,
                )
                vwap_result = self._evaluate_prediction(
                    current_time, item_id, horizon, "vwap",
                    vwap_pred, actual_buy, actual_sell, actual_dir, current_price,
                )
                all_results.append(vwap_result)

            step_count += 1
            current_time += step_delta

        logger.info(f"Backtest: {step_count} steps, {len(all_results)} predictions")

        # Aggregate results
        report.predictions = all_results
        report.strategies = self._aggregate_metrics(all_results, horizons)
        report.elapsed_seconds = time.time() - start_time

        return report

    # ------------------------------------------------------------------
    # Prediction Strategies
    # ------------------------------------------------------------------

    def _ml_strategy(
        self,
        item_id: int,
        horizon: str,
        features: Dict[str, float],
        current_buy: int,
        current_sell: int,
    ) -> Dict:
        """ML model prediction (falls back to statistical if no model)."""
        predictions = self.forecaster.predict(item_id, features)
        pred = predictions.get(horizon, {})
        return {
            "buy": pred.get("buy", current_buy),
            "sell": pred.get("sell", current_sell),
            "direction": pred.get("direction", "flat"),
            "confidence": pred.get("confidence", 0.5),
        }

    def _margin_strategy(
        self,
        current_buy: int,
        current_sell: int,
        horizon: str,
    ) -> Dict:
        """
        Simple margin scanning: assume prices stay roughly the same.
        Buy at insta-sell, sell at insta-buy (basic flip logic).
        Direction: always flat (margin scan doesn't predict direction).
        """
        return {
            "buy": current_buy,
            "sell": current_sell,
            "direction": "flat",
            "confidence": 0.3,
        }

    def _vwap_strategy(
        self,
        features: Dict[str, float],
        current_buy: int,
        current_sell: int,
        horizon: str,
    ) -> Dict:
        """
        VWAP-only baseline: predict price reverts to VWAP.
        Uses vwap_deviation feature to estimate future price.
        """
        vwap_dev = features.get("vwap_deviation", 0.0)
        spread_pct = features.get("spread_pct", 1.0)

        # Predict price reverts halfway to VWAP
        reversion_factor = 0.5
        predicted_price = current_buy * (1.0 - vwap_dev * reversion_factor)

        # Direction based on VWAP deviation
        if vwap_dev < -0.005:
            direction = "up"   # below VWAP, expect reversion up
        elif vwap_dev > 0.005:
            direction = "down"  # above VWAP, expect reversion down
        else:
            direction = "flat"

        half_spread = abs(spread_pct) / 200.0 * predicted_price
        predicted_buy = int(predicted_price + half_spread)
        predicted_sell = int(predicted_price - half_spread)

        return {
            "buy": max(1, predicted_buy),
            "sell": max(1, predicted_sell),
            "direction": direction,
            "confidence": 0.4,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_prediction(
        self,
        timestamp: datetime,
        item_id: int,
        horizon: str,
        strategy: str,
        prediction: Dict,
        actual_buy: int,
        actual_sell: int,
        actual_direction: str,
        current_price: int,
    ) -> PredictionResult:
        """Compare a prediction against actual outcome."""
        result = PredictionResult(
            timestamp=timestamp,
            item_id=item_id,
            horizon=horizon,
            strategy=strategy,
            predicted_buy=prediction.get("buy", 0),
            predicted_sell=prediction.get("sell", 0),
            predicted_direction=prediction.get("direction", "flat"),
            confidence=prediction.get("confidence", 0.0),
            actual_buy=actual_buy,
            actual_sell=actual_sell,
            actual_direction=actual_direction,
        )

        # Direction accuracy
        result.direction_correct = (
            result.predicted_direction == result.actual_direction
        )

        # Price error (buy side)
        if result.predicted_buy > 0 and actual_buy > 0:
            result.price_error = abs(result.predicted_buy - actual_buy)
            result.price_error_pct = result.price_error / actual_buy * 100
        else:
            result.price_error = 0.0
            result.price_error_pct = 0.0

        # Simulated profit: if we bought at current_price and sold at actual_buy
        # based on the predicted direction
        if result.predicted_direction == "up" and current_price > 0:
            # We'd buy now and sell later
            sell_price = actual_buy
            tax = calculate_tax(sell_price)
            result.simulated_profit = sell_price - current_price - tax
        elif result.predicted_direction == "down" and current_price > 0:
            # We'd avoid buying (or short if possible) -> no profit/loss
            result.simulated_profit = 0
        else:
            # Flat -> do a standard margin flip
            if actual_buy > 0 and actual_sell > 0:
                spread = actual_buy - actual_sell
                tax = calculate_tax(actual_buy)
                result.simulated_profit = spread - tax
            else:
                result.simulated_profit = 0

        return result

    # ------------------------------------------------------------------
    # Metrics Aggregation
    # ------------------------------------------------------------------

    def _aggregate_metrics(
        self,
        results: List[PredictionResult],
        horizons: List[str],
    ) -> Dict[str, Dict[str, StrategyMetrics]]:
        """Aggregate prediction results into per-strategy, per-horizon metrics."""
        strategies = ["ml", "margin", "vwap"]
        output: Dict[str, Dict[str, StrategyMetrics]] = {}

        for strategy in strategies:
            output[strategy] = {}
            for horizon in horizons:
                filtered = [
                    r for r in results
                    if r.strategy == strategy and r.horizon == horizon
                ]

                metrics = StrategyMetrics(strategy=strategy, horizon=horizon)
                metrics.total_predictions = len(filtered)

                if not filtered:
                    output[strategy][horizon] = metrics
                    continue

                # Direction accuracy
                metrics.direction_correct = sum(
                    1 for r in filtered if r.direction_correct
                )
                metrics.direction_accuracy = (
                    metrics.direction_correct / metrics.total_predictions
                )

                # Price errors
                errors = [r.price_error for r in filtered if r.price_error >= 0]
                pct_errors = [r.price_error_pct for r in filtered if r.price_error_pct >= 0]

                metrics.mae = statistics.mean(errors) if errors else 0.0
                metrics.mape = statistics.mean(pct_errors) if pct_errors else 0.0
                metrics.rmse = (
                    (sum(e ** 2 for e in errors) / len(errors)) ** 0.5
                    if errors else 0.0
                )

                # Profit metrics
                profits = [r.simulated_profit for r in filtered]
                profitable = [p for p in profits if p > 0]
                metrics.profitable_predictions = len(profitable)
                metrics.profitable_accuracy = (
                    len(profitable) / len(profits) if profits else 0.0
                )
                metrics.cumulative_profit = sum(profits)

                trades_with_action = [p for p in profits if p != 0]
                metrics.win_rate = (
                    len(profitable) / len(trades_with_action)
                    if trades_with_action else 0.0
                )
                metrics.avg_profit_per_trade = (
                    statistics.mean(trades_with_action)
                    if trades_with_action else 0.0
                )

                # Max drawdown
                metrics.max_drawdown = self._max_drawdown(profits)

                # Sharpe ratio (using trade-level returns)
                if len(trades_with_action) >= 2:
                    mean_ret = statistics.mean(trades_with_action)
                    std_ret = statistics.stdev(trades_with_action)
                    metrics.sharpe_ratio = (
                        mean_ret / std_ret if std_ret > 0 else 0.0
                    )

                output[strategy][horizon] = metrics

        return output

    def _max_drawdown(self, profits: List[int]) -> int:
        """Calculate maximum drawdown from a series of profits."""
        if not profits:
            return 0

        cumulative = 0
        peak = 0
        max_dd = 0

        for p in profits:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return max_dd

    # ------------------------------------------------------------------
    # Price Lookup Helpers
    # ------------------------------------------------------------------

    def _build_price_index(
        self, snapshots: List[PriceSnapshot],
    ) -> Dict[int, Tuple[int, int]]:
        """
        Build a lookup from unix timestamp (rounded to 10s) to (buy, sell).
        """
        index: Dict[int, Tuple[int, int]] = {}
        for s in snapshots:
            ts = int(s.timestamp.timestamp())
            ts_rounded = (ts // 10) * 10
            buy = s.instant_buy or 0
            sell = s.instant_sell or 0
            if buy > 0 and sell > 0:
                index[ts_rounded] = (buy, sell)
        return index

    def _find_price_at_time(
        self,
        index: Dict[int, Tuple[int, int]],
        target_time: datetime,
    ) -> Optional[Tuple[int, int]]:
        """Find price closest to target_time within +/- 30s."""
        target_ts = int(target_time.timestamp())
        target_rounded = (target_ts // 10) * 10

        for offset in [0, 10, -10, 20, -20, 30, -30]:
            result = index.get(target_rounded + offset)
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def generate_report(self, report: BacktestReport) -> str:
        """Generate a human-readable text report from backtest results."""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("BACKTEST REPORT")
        lines.append("=" * 70)
        lines.append(f"Item ID:        {report.item_id}")
        lines.append(f"Period:         {report.start_time} -> {report.end_time}")
        lines.append(f"Snapshots:      {report.total_snapshots}")
        lines.append(f"Predictions:    {len(report.predictions)}")
        lines.append(f"Elapsed:        {report.elapsed_seconds:.1f}s")
        lines.append("")

        for strategy_name in ["ml", "margin", "vwap"]:
            horizons = report.strategies.get(strategy_name, {})
            if not horizons:
                continue

            lines.append(f"--- Strategy: {strategy_name.upper()} ---")
            lines.append(
                f"{'Horizon':<8} {'DirAcc':>7} {'MAE':>10} {'MAPE':>7} "
                f"{'WinRate':>8} {'CumP/L':>12} {'MaxDD':>10} {'Sharpe':>7}"
            )
            lines.append("-" * 70)

            total_profit = 0
            for horizon in HORIZONS:
                m = horizons.get(horizon)
                if not m or m.total_predictions == 0:
                    continue

                total_profit += m.cumulative_profit

                lines.append(
                    f"{horizon:<8} "
                    f"{m.direction_accuracy:>6.1%} "
                    f"{m.mae:>10.0f} "
                    f"{m.mape:>6.2f}% "
                    f"{m.win_rate:>7.1%} "
                    f"{m.cumulative_profit:>11,} "
                    f"{m.max_drawdown:>10,} "
                    f"{m.sharpe_ratio:>7.2f}"
                )

            lines.append(f"{'TOTAL':<8} {'':>7} {'':>10} {'':>7} "
                         f"{'':>8} {total_profit:>11,}")
            lines.append("")

        # Comparison summary
        lines.append("=" * 70)
        lines.append("STRATEGY COMPARISON (cumulative P/L)")
        lines.append("=" * 70)

        for horizon in HORIZONS:
            profits = {}
            for strat in ["ml", "margin", "vwap"]:
                m = report.strategies.get(strat, {}).get(horizon)
                if m and m.total_predictions > 0:
                    profits[strat] = m.cumulative_profit

            if not profits:
                continue

            best = max(profits, key=profits.get)  # type: ignore[arg-type]
            lines.append(
                f"  {horizon:<6}: "
                + " | ".join(
                    f"{s}: {p:>10,} GP{'  <-- BEST' if s == best else ''}"
                    for s, p in profits.items()
                )
            )

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch Backtesting
    # ------------------------------------------------------------------

    def run_batch_backtest(
        self,
        item_ids: List[int],
        hours: int = 48,
        step_interval: int = 60,
    ) -> Dict[int, BacktestReport]:
        """Run backtests for multiple items."""
        reports: Dict[int, BacktestReport] = {}

        for item_id in item_ids:
            logger.info(f"Backtesting item {item_id}...")
            try:
                reports[item_id] = self.run_backtest(
                    item_id, hours=hours, step_interval=step_interval,
                )
            except Exception as e:
                logger.error(f"Backtest failed for item {item_id}: {e}")

        return reports

    def generate_summary_report(
        self, reports: Dict[int, BacktestReport],
    ) -> str:
        """Generate a summary report across multiple items."""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("MULTI-ITEM BACKTEST SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Items tested: {len(reports)}")
        lines.append("")

        # Aggregate across all items
        strategy_totals: Dict[str, Dict[str, int]] = {}
        for strat in ["ml", "margin", "vwap"]:
            strategy_totals[strat] = {}
            for horizon in HORIZONS:
                total = 0
                for report in reports.values():
                    m = report.strategies.get(strat, {}).get(horizon)
                    if m:
                        total += m.cumulative_profit
                strategy_totals[strat][horizon] = total

        lines.append(f"{'Strategy':<10} " + " ".join(f"{h:>10}" for h in HORIZONS))
        lines.append("-" * 70)

        for strat in ["ml", "margin", "vwap"]:
            vals = " ".join(
                f"{strategy_totals[strat].get(h, 0):>10,}"
                for h in HORIZONS
            )
            lines.append(f"{strat.upper():<10} {vals}")

        lines.append("")

        # Per-item breakdown
        for item_id, report in reports.items():
            lines.append(f"\nItem {item_id}: {report.total_snapshots} snapshots")
            ml_total = sum(
                m.cumulative_profit
                for m in report.strategies.get("ml", {}).values()
                if m.total_predictions > 0
            )
            margin_total = sum(
                m.cumulative_profit
                for m in report.strategies.get("margin", {}).values()
                if m.total_predictions > 0
            )
            vwap_total = sum(
                m.cumulative_profit
                for m in report.strategies.get("vwap", {}).values()
                if m.total_predictions > 0
            )
            lines.append(
                f"  ML: {ml_total:>12,} GP | "
                f"Margin: {margin_total:>12,} GP | "
                f"VWAP: {vwap_total:>12,} GP"
            )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
