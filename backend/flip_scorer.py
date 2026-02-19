"""
Flip Scorer – composite 0-100 scoring system for flip quality.

Designed to achieve 80%+ win rate by aggressively filtering out
bad flips using multiple independent signals. Each signal has a
veto threshold that can reject an item entirely.

Score breakdown (100 points max):
  - Spread quality    (25 pts)  – is the margin in the 0.5-2% sweet spot?
  - Volume & liquidity (25 pts) – can you actually fill orders?
  - Freshness         (12 pts)  – how recent is the price data?
  - Trend alignment   (10 pts)  – is the trend helping or hurting?
  - Historical winrate (10 pts) – how well has this item flipped before?
  - Stability         (08 pts)  – is the price stable or whipsawing?
  - ML prediction     (10 pts)  – AI confidence signal

Post-score multiplier for price range (10M-50M sweet spot).
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

from backend.database import (
    get_db, PriceSnapshot, FlipHistory,
    get_price_history, get_item_flips,
)
from backend.smart_pricer import SmartPricer, PriceRecommendation, Trend, GE_TAX_RATE, GE_TAX_CAP

logger = logging.getLogger(__name__)


@dataclass
class FlipScore:
    """Full scored assessment of a flip opportunity."""
    item_id: int
    item_name: str = ""

    # Component scores (0-100 scale each, then weighted)
    spread_score: float = 0.0
    volume_score: float = 0.0
    freshness_score: float = 0.0
    trend_score: float = 0.0
    history_score: float = 0.0
    stability_score: float = 0.0
    ml_score: float = 0.0  # ML model confidence signal

    # Composite
    total_score: float = 0.0

    # Vetoes (any True = do not suggest)
    vetoed: bool = False
    veto_reasons: list = field(default_factory=list)

    # From SmartPricer
    recommended_buy: Optional[int] = None
    recommended_sell: Optional[int] = None
    expected_profit: Optional[int] = None
    expected_profit_pct: Optional[float] = None
    tax: Optional[int] = None
    trend: str = "NEUTRAL"
    confidence: float = 0.0
    volume_5m: int = 0

    # Context
    instant_buy: Optional[int] = None
    instant_sell: Optional[int] = None
    spread: Optional[int] = None
    spread_pct: Optional[float] = None

    # ML prediction context
    ml_direction: Optional[str] = None
    ml_confidence: Optional[float] = None
    ml_method: Optional[str] = None

    # Historical context
    win_rate: Optional[float] = None
    total_flips: int = 0
    avg_profit: Optional[float] = None

    reason: str = ""


class FlipScorer:
    """
    Composite scoring engine that combines SmartPricer output with
    additional checks to produce a 0-100 flip quality score.
    """

    # Weights for each component (must sum to 1.0)
    # ML component gets weight when models are trained; otherwise
    # its weight is redistributed to other components automatically
    WEIGHTS = {
        "spread": 0.25,     # margin quality is #1 predictor of profitable flips
        "volume": 0.25,     # liquidity is critical — can't profit if stuck
        "freshness": 0.12,
        "trend": 0.10,
        "history": 0.10,
        "stability": 0.08,
        "ml": 0.10,         # ML prediction signal
    }

    # Minimum score to suggest a flip (out of 100)
    MIN_SUGGEST_SCORE = 45

    def __init__(self):
        self.pricer = SmartPricer()
        self._ml_predictor = None

    def _get_ml_predictor(self):
        """Lazily load the ML predictor for scoring enrichment."""
        if self._ml_predictor is None:
            try:
                from backend.ml.predictor import Predictor
                self._ml_predictor = Predictor()
                self._ml_predictor.load_models()
            except Exception:
                pass
        return self._ml_predictor

    def score_item(
        self,
        item_id: int,
        item_name: str = "",
        snapshots: Optional[List[PriceSnapshot]] = None,
        flips: Optional[List[FlipHistory]] = None,
    ) -> FlipScore:
        """Score a single item. Returns FlipScore with vetoes and component scores."""
        fs = FlipScore(item_id=item_id, item_name=item_name)

        # Only open a DB connection when we actually need data
        need_db = snapshots is None or flips is None
        db = get_db() if need_db else None
        try:
            if snapshots is None:
                snapshots = get_price_history(db, item_id, hours=4)
            if flips is None:
                flips = get_item_flips(db, item_id, days=30)

            if not snapshots:
                fs.vetoed = True
                fs.veto_reasons.append("No price data")
                return fs

            # Run SmartPricer
            rec = self.pricer.price_item(item_id, snapshots=snapshots)
            fs.recommended_buy = rec.recommended_buy
            fs.recommended_sell = rec.recommended_sell
            fs.expected_profit = rec.expected_profit
            fs.expected_profit_pct = rec.expected_profit_pct
            fs.tax = rec.tax
            fs.trend = rec.trend.value
            fs.confidence = rec.confidence
            fs.volume_5m = rec.volume_5m
            fs.instant_buy = rec.instant_buy
            fs.instant_sell = rec.instant_sell

            if not rec.instant_buy or not rec.instant_sell:
                fs.vetoed = True
                fs.veto_reasons.append("Missing buy or sell price")
                return fs

            fs.spread = rec.instant_buy - rec.instant_sell
            fs.spread_pct = (fs.spread / rec.instant_sell * 100) if rec.instant_sell > 0 else 0

            # ---- HARD VETOES ----
            self._check_vetoes(fs, rec, snapshots)
            if fs.vetoed:
                return fs

            # ---- COMPONENT SCORING ----
            fs.spread_score = self._score_spread(fs, rec)
            fs.volume_score = self._score_volume(fs, rec, snapshots)
            fs.freshness_score = self._score_freshness(rec, snapshots)
            fs.trend_score = self._score_trend(rec)
            fs.history_score = self._score_history(fs, flips)
            fs.stability_score = self._score_stability(snapshots)
            fs.ml_score = self._score_ml(fs, item_id, snapshots, flips)

            # Composite weighted score
            # If ML models aren't trained, redistribute ML weight to others
            if fs.ml_score < 0:
                # ML unavailable — use original weights without ML
                fs.ml_score = 0
                ml_weight = 0
                non_ml_total = sum(
                    v for k, v in self.WEIGHTS.items() if k != "ml"
                )
                fs.total_score = (
                    fs.spread_score * (self.WEIGHTS["spread"] / non_ml_total)
                    + fs.volume_score * (self.WEIGHTS["volume"] / non_ml_total)
                    + fs.freshness_score * (self.WEIGHTS["freshness"] / non_ml_total)
                    + fs.trend_score * (self.WEIGHTS["trend"] / non_ml_total)
                    + fs.history_score * (self.WEIGHTS["history"] / non_ml_total)
                    + fs.stability_score * (self.WEIGHTS["stability"] / non_ml_total)
                )
            else:
                fs.total_score = (
                    fs.spread_score * self.WEIGHTS["spread"]
                    + fs.volume_score * self.WEIGHTS["volume"]
                    + fs.freshness_score * self.WEIGHTS["freshness"]
                    + fs.trend_score * self.WEIGHTS["trend"]
                    + fs.history_score * self.WEIGHTS["history"]
                    + fs.stability_score * self.WEIGHTS["stability"]
                    + fs.ml_score * self.WEIGHTS["ml"]
                )

            # Apply SmartPricer confidence as a multiplier
            fs.total_score *= max(0.3, rec.confidence)

            # Price-range multiplier — data from 1,522 Copilot trades:
            #   10M-50M:  sweet spot (93% WR, 144K avg/flip, 150K GP/hr)
            #   50M+:     strong (83% WR, 322K avg/flip)
            #   1M-10M:   solid
            #   <10K:     good for bulk (46K avg/flip via volume)
            #   10K-100K: neutral
            #   100K-1M:  WORST bracket (-7.9M net loss!) — overcompeted
            mid_price = (rec.instant_buy + rec.instant_sell) // 2 if rec.instant_buy and rec.instant_sell else 0
            if mid_price > 0:
                if 10_000_000 <= mid_price <= 50_000_000:
                    fs.total_score *= 1.15   # best bracket
                elif mid_price > 50_000_000:
                    fs.total_score *= 1.10   # strong
                elif 1_000_000 <= mid_price < 10_000_000:
                    fs.total_score *= 1.05   # solid
                elif mid_price < 10_000:
                    fs.total_score *= 1.05   # bulk flipping works
                elif 100_000 <= mid_price < 1_000_000:
                    fs.total_score *= 0.85   # historically net loser
                # 10K-100K stays at 1.0 (neutral)

            # Cap at 100
            fs.total_score = min(100, fs.total_score)

            # Generate human-readable reason
            fs.reason = self._build_reason(fs, rec)

        finally:
            if db is not None:
                db.close()

        return fs

    # ------------------------------------------------------------------
    # Hard Vetoes – if any trigger, the item is rejected entirely
    # ------------------------------------------------------------------

    def _check_vetoes(
        self, fs: FlipScore, rec: PriceRecommendation, snapshots: List[PriceSnapshot]
    ):
        """Apply hard veto rules that reject obviously bad flips."""

        # Veto 1: Negative or zero profit after tax
        if rec.expected_profit is not None and rec.expected_profit <= 0:
            fs.vetoed = True
            fs.veto_reasons.append(f"Unprofitable after tax ({rec.expected_profit} GP)")

        # Veto 2: Extremely stale data (>45 min old with low volume)
        latest = snapshots[-1]
        now = int(time.time())
        if latest.buy_time:
            buy_age = (now - latest.buy_time) / 60
            if buy_age > 45 and rec.volume_5m < 5:
                fs.vetoed = True
                fs.veto_reasons.append(f"Stale buy price ({buy_age:.0f}m old, vol={rec.volume_5m})")
        if latest.sell_time:
            sell_age = (now - latest.sell_time) / 60
            if sell_age > 45 and rec.volume_5m < 5:
                fs.vetoed = True
                fs.veto_reasons.append(f"Stale sell price ({sell_age:.0f}m old)")

        # Veto 3: Zero volume trap
        if rec.volume_5m == 0:
            fs.vetoed = True
            fs.veto_reasons.append("Zero volume in last 5 minutes - likely illiquid trap")

        # Veto 4: Wide spread (>8%) — data shows >10% margins are net money losers
        if fs.spread_pct and fs.spread_pct > 8:
            fs.vetoed = True
            fs.veto_reasons.append(f"Spread too wide ({fs.spread_pct:.1f}%) - margins >8% lose money historically")

        # Veto 5: Spread is negative or item is inverted
        if fs.spread is not None and fs.spread <= 0:
            fs.vetoed = True
            fs.veto_reasons.append("Inverted spread (buy <= sell)")

        # Veto 6: Strong downtrend with shrinking volume
        if rec.trend in (Trend.STRONG_DOWN,) and rec.volume_5m < 10:
            recent_vols = [
                (s.buy_volume or 0) + (s.sell_volume or 0)
                for s in snapshots[-6:]  # last minute
            ]
            if recent_vols and max(recent_vols) < 3:
                fs.vetoed = True
                fs.veto_reasons.append("Crashing with no volume - avoid")

        # Veto 7: Volume velocity trap - volume died recently
        # Even if total_volume was historically high, if recent volume is 0
        # the item is now illiquid and you'll be stuck holding the bag
        if len(snapshots) >= 6:
            recent_3 = snapshots[-3:]
            older = snapshots[:-3]
            recent_vol = sum((s.buy_volume or 0) + (s.sell_volume or 0) for s in recent_3)
            if older:
                older_vol = sum((s.buy_volume or 0) + (s.sell_volume or 0) for s in older)
                avg_older = (older_vol / len(older)) * 3  # scale to same window
                if recent_vol == 0 and avg_older > 5:
                    fs.vetoed = True
                    fs.veto_reasons.append("Volume velocity trap - trades died in last interval")
                elif avg_older > 0:
                    velocity = recent_vol / max(avg_older, 1)
                    if velocity < 0.2 and rec.volume_5m < 5:
                        fs.vetoed = True
                        fs.veto_reasons.append(f"Volume crashed to {velocity:.0%} of normal - liquidity freeze")

        # Veto 8: Waterfall crash detection
        if self.pricer.detect_waterfall(snapshots):
            fs.vetoed = True
            fs.veto_reasons.append("Waterfall crash detected - price dropping >5% in 15 minutes")

    # ------------------------------------------------------------------
    # Component Scoring Functions (each returns 0-100)
    # ------------------------------------------------------------------

    def _score_spread(self, fs: FlipScore, rec: PriceRecommendation) -> float:
        """Score spread quality.

        Data from 1,522 Copilot trades shows the sweet spot is 0.5-2%:
          0.5-1%  → 36.7M profit, 89% WR
          1-2%    → 46.6M profit, 94% WR  (BEST)
          2-5%    → diminishing returns
          5-10%   → poor
          10%+    → net LOSER (-11.3M)
        """
        pct = fs.spread_pct or 0

        if pct <= 0:
            return 0

        # Data-driven scoring — sweet spot 0.5-2% regardless of volume
        if rec.volume_5m > 30:
            # High volume: tight spreads are king
            if 0.5 <= pct <= 2:
                return 100   # sweet spot
            elif pct < 0.5:
                return 35    # might not cover tax
            elif pct <= 3:
                return 80
            elif pct <= 5:
                return 55
            else:
                return 20    # wide spread on liquid item = suspicious
        elif rec.volume_5m > 10:
            if 0.5 <= pct <= 2:
                return 95
            elif 2 < pct <= 3:
                return 75
            elif pct < 0.5:
                return 25
            elif pct <= 5:
                return 50
            else:
                return 15
        else:
            # Low volume: wider spreads are normal but still risky
            if 1 <= pct <= 3:
                return 75
            elif 0.5 <= pct < 1:
                return 55
            elif 3 < pct <= 5:
                return 45
            elif pct < 0.5:
                return 15
            else:
                return 10    # >5% on low vol = trap

    def _score_volume(
        self, fs: FlipScore, rec: PriceRecommendation, snapshots: List[PriceSnapshot]
    ) -> float:
        """Score volume/liquidity. Higher = easier to fill orders."""
        vol = rec.volume_5m

        if vol >= 100:
            return 100
        elif vol >= 50:
            return 90
        elif vol >= 20:
            return 75
        elif vol >= 10:
            return 60
        elif vol >= 5:
            return 40
        elif vol >= 2:
            return 20
        elif vol >= 1:
            return 10
        else:
            return 0

    def _score_freshness(
        self, rec: PriceRecommendation, snapshots: List[PriceSnapshot]
    ) -> float:
        """Score how fresh the price data is."""
        latest = snapshots[-1]
        now = int(time.time())

        ages = []
        if latest.buy_time:
            ages.append((now - latest.buy_time) / 60)
        if latest.sell_time:
            ages.append((now - latest.sell_time) / 60)

        if not ages:
            return 20  # No timestamp data

        max_age = max(ages)

        if max_age < 2:
            return 100
        elif max_age < 5:
            return 90
        elif max_age < 10:
            return 75
        elif max_age < 15:
            return 60
        elif max_age < 30:
            return 40
        elif max_age < 60:
            return 20
        else:
            return 5

    def _score_trend(self, rec: PriceRecommendation) -> float:
        """Score trend alignment. Neutral and mild trends are best for flipping."""
        # For margin flipping, NEUTRAL is ideal (stable spreads)
        # Mild trends are OK (can exploit direction)
        # Strong trends are risky (spread can collapse)
        scores = {
            Trend.NEUTRAL: 90,
            Trend.UP: 75,
            Trend.DOWN: 60,      # Downtrend: risky but can still profit if careful
            Trend.STRONG_UP: 50,  # Strong trends make spreads volatile
            Trend.STRONG_DOWN: 25,  # Very risky for flipping
        }
        base = scores.get(rec.trend, 50)

        # Bonus: Bollinger position near middle is ideal for flipping
        if rec.bb_position is not None:
            if 0.3 <= rec.bb_position <= 0.7:
                base = min(100, base + 10)  # Stable zone
            elif rec.bb_position < 0.1 or rec.bb_position > 0.9:
                base = max(0, base - 15)  # Extreme = risky

        return base

    def _score_history(self, fs: FlipScore, flips: List[FlipHistory]) -> float:
        """Score based on historical flip success for this item."""
        if not flips:
            return 50  # No history = neutral

        wins = sum(1 for f in flips if f.net_profit and f.net_profit > 0)
        total = len(flips)
        win_rate = wins / total if total > 0 else 0

        fs.win_rate = round(win_rate, 3)
        fs.total_flips = total
        profits = [f.net_profit for f in flips if f.net_profit is not None]
        fs.avg_profit = round(sum(profits) / len(profits), 0) if profits else None

        if total < 3:
            return 50  # Not enough data to judge

        # Scale linearly: 50% win rate = 0, 100% = 100
        if win_rate >= 0.9:
            return 100
        elif win_rate >= 0.8:
            return 85
        elif win_rate >= 0.7:
            return 70
        elif win_rate >= 0.6:
            return 55
        elif win_rate >= 0.5:
            return 40
        else:
            return 15  # Historically losing item

    def _score_stability(self, snapshots: List[PriceSnapshot]) -> float:
        """Score price stability. Stable prices = predictable flips."""
        # Use coefficient of variation on recent buy prices
        recent = snapshots[-30:]  # Last 5 minutes of 10s snapshots
        prices = [s.instant_buy for s in recent if s.instant_buy and s.instant_buy > 0]

        if len(prices) < 3:
            return 50  # Not enough data

        import statistics
        mean = statistics.mean(prices)
        if mean == 0:
            return 50

        stdev = statistics.stdev(prices) if len(prices) > 1 else 0
        cv = stdev / mean  # Coefficient of variation

        # Lower CV = more stable
        if cv < 0.005:
            return 100  # Very stable
        elif cv < 0.01:
            return 85
        elif cv < 0.02:
            return 70
        elif cv < 0.05:
            return 50
        elif cv < 0.1:
            return 30
        else:
            return 10  # Very volatile

    def _score_ml(
        self,
        fs: FlipScore,
        item_id: int,
        snapshots: List[PriceSnapshot],
        flips: List[FlipHistory],
    ) -> float:
        """Score based on ML model predictions.

        Returns -1 if ML is unavailable (weight will be redistributed).
        Returns 0-100 based on ML confidence and direction alignment.
        """
        predictor = self._get_ml_predictor()
        if predictor is None:
            return -1  # sentinel: ML unavailable

        try:
            result = predictor.predict_item(
                item_id, snapshots=snapshots, flips=flips, save_to_db=False,
            )
            meta = result.get("_meta", {})
            method = meta.get("method", "none")

            if method == "none":
                return -1

            # Use the 5m and 30m horizons as primary signals for flipping
            pred_5m = result.get("5m", {})
            pred_30m = result.get("30m", {})

            direction_5m = pred_5m.get("direction", "flat")
            confidence_5m = pred_5m.get("confidence", 0.5)
            direction_30m = pred_30m.get("direction", "flat")
            confidence_30m = pred_30m.get("confidence", 0.5)

            # Store on FlipScore for frontend display
            fs.ml_direction = direction_5m
            fs.ml_confidence = confidence_5m
            fs.ml_method = method

            # Scoring logic:
            # For flipping, we want: short-term stable or up, medium-term stable
            # If ML says price will crash, that's bad for a flip
            score = 50.0  # neutral baseline

            # 5m signal (short-term — most important for flip timing)
            if direction_5m == "up":
                score += 20 * confidence_5m
            elif direction_5m == "flat":
                score += 10 * confidence_5m  # flat is fine for flipping
            elif direction_5m == "down":
                score -= 25 * confidence_5m  # price dropping = risky

            # 30m signal (medium-term — helps avoid getting stuck)
            if direction_30m == "up":
                score += 15 * confidence_30m
            elif direction_30m == "flat":
                score += 5 * confidence_30m
            elif direction_30m == "down":
                score -= 15 * confidence_30m

            # Bonus for ML vs statistical: ML models are more trustworthy
            if method == "ml":
                score += 5

            return max(0, min(100, score))

        except Exception:
            return -1  # ML failed, redistribute weight

    # ------------------------------------------------------------------
    # Reason builder
    # ------------------------------------------------------------------

    def _build_reason(self, fs: FlipScore, rec: PriceRecommendation) -> str:
        parts = []

        # Overall rating
        if fs.total_score >= 70:
            parts.append("STRONG FLIP")
        elif fs.total_score >= 55:
            parts.append("GOOD FLIP")
        elif fs.total_score >= 45:
            parts.append("MARGINAL")
        else:
            parts.append("WEAK")

        # Trend
        parts.append(f"Trend: {rec.trend.value}")

        # Volume
        if rec.volume_5m >= 30:
            parts.append("High liquidity")
        elif rec.volume_5m >= 10:
            parts.append("Decent liquidity")
        else:
            parts.append("Low liquidity")

        # Historical
        if fs.win_rate is not None and fs.total_flips >= 3:
            parts.append(f"History: {fs.win_rate*100:.0f}% win ({fs.total_flips} flips)")

        # Profit
        if fs.expected_profit:
            parts.append(f"+{fs.expected_profit:,} GP ({fs.expected_profit_pct:.1f}%)")

        # ML signal
        if fs.ml_direction and fs.ml_confidence:
            ml_label = f"AI: {fs.ml_direction} ({fs.ml_confidence:.0%})"
            if fs.ml_method == "ml":
                ml_label += " [ML]"
            parts.append(ml_label)

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Convenience: score multiple items from the scan pipeline
# ---------------------------------------------------------------------------

def score_opportunities(
    items: List[dict],
    min_score: float = 45,
    limit: int = 50,
) -> List[FlipScore]:
    """
    Take raw opportunity dicts (from ai_strategist or similar) and
    score them through the FlipScorer. Returns sorted by total_score desc.

    Builds synthetic PriceSnapshot from the Wiki API data so items can
    be scored even when they have no DB history yet.
    """
    scorer = FlipScorer()
    scored = []

    for item in items:
        try:
            item_id = item.get("item_id") or item.get("id", 0)

            # Build a synthetic snapshot from the live scan data so that
            # score_item doesn't veto items that simply lack DB history.
            # NOTE: scan dict uses "instant_buy" = buyer's bid (low),
            # but PriceSnapshot uses "instant_buy" = insta-buy price (high).
            # We must swap them.
            now = datetime.utcnow()
            ts = int(time.time())
            snap = PriceSnapshot(
                item_id=item_id,
                timestamp=now,
                instant_buy=item.get("instant_sell") or item.get("sell_at"),
                instant_sell=item.get("instant_buy") or item.get("buy_at"),
                buy_time=ts,
                sell_time=ts,
                avg_buy=item.get("sell_at"),
                avg_sell=item.get("buy_at"),
                buy_volume=item.get("high_volume", 0),
                sell_volume=item.get("low_volume", 0),
            )

            fs = scorer.score_item(
                item_id=item_id,
                item_name=item.get("name", ""),
                snapshots=[snap],
                flips=[],  # no flip history for scan-sourced items
            )
            if fs.vetoed:
                logger.debug(
                    "Vetoed %s: %s", fs.item_name, ", ".join(fs.veto_reasons)
                )
                continue

            if fs.total_score >= min_score:
                scored.append(fs)
        except Exception as e:
            logger.warning("Failed to score item %s: %s", item.get("name"), e)

    scored.sort(key=lambda x: x.total_score, reverse=True)
    return scored[:limit]
