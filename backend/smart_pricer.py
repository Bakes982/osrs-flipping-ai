"""
Smart Margin Pricing Engine
Trend-aware buy/sell price calculation with undercut/overcut logic.
Clamps against recent price data to avoid dumb suggestions.
"""

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Tuple

from backend.database import (
    get_db, PriceSnapshot, FlipHistory, get_price_history, get_item_flips,
)

# GE Tax constants
GE_TAX_RATE = 0.02
GE_TAX_CAP = 5_000_000


class Trend(Enum):
    STRONG_UP = "STRONG_UP"
    UP = "UP"
    NEUTRAL = "NEUTRAL"
    DOWN = "DOWN"
    STRONG_DOWN = "STRONG_DOWN"


@dataclass
class PriceRecommendation:
    """Output of the smart pricer for a single item."""
    item_id: int
    timestamp: datetime

    # Raw market data
    instant_buy: Optional[int] = None   # insta-buy (what buyers pay now)
    instant_sell: Optional[int] = None  # insta-sell (what sellers get now)

    # VWAP at different windows
    vwap_1m: Optional[float] = None
    vwap_5m: Optional[float] = None
    vwap_30m: Optional[float] = None
    vwap_2h: Optional[float] = None

    # Trend analysis
    trend: Trend = Trend.NEUTRAL
    momentum: float = 0.0  # rate of price change (GP/min)

    # Bollinger bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_position: Optional[float] = None  # 0.0 = at lower, 1.0 = at upper

    # Clamped prices (before undercut/overcut)
    clamped_buy: Optional[int] = None
    clamped_sell: Optional[int] = None

    # Final recommended prices (after undercut/overcut)
    recommended_buy: Optional[int] = None
    recommended_sell: Optional[int] = None

    # Expected outcome
    expected_profit: Optional[int] = None
    expected_profit_pct: Optional[float] = None
    tax: Optional[int] = None

    # Confidence & flags
    confidence: float = 0.0  # 0-1
    stale_data: bool = False
    anomalous_spread: bool = False
    volume_5m: int = 0

    reason: str = ""


class SmartPricer:
    """
    Trend-aware pricing engine that clamps buy/sell prices against
    recent market data and applies intelligent undercut/overcut.
    """

    def __init__(self):
        # Average daily volume cache (item_id -> vol)
        self._avg_volume_cache: Dict[int, float] = {}

    def calculate_vwap(
        self,
        snapshots: List[PriceSnapshot],
        minutes: int,
        use_buy: bool = True,
    ) -> Optional[float]:
        """
        Volume-Weighted Average Price over a time window.
        If no volume data, falls back to simple mean.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        relevant = [s for s in snapshots if s.timestamp >= cutoff]
        if not relevant:
            return None

        prices = []
        volumes = []
        for s in relevant:
            price = s.instant_buy if use_buy else s.instant_sell
            vol = s.buy_volume if use_buy else s.sell_volume
            if price and price > 0:
                prices.append(price)
                volumes.append(max(vol or 0, 1))  # min 1 to avoid div-by-0

        if not prices:
            return None

        total_vol = sum(volumes)
        if total_vol == 0:
            return statistics.mean(prices)

        return sum(p * v for p, v in zip(prices, volumes)) / total_vol

    def calculate_sma(
        self,
        snapshots: List[PriceSnapshot],
        minutes: int,
        use_buy: bool = True,
    ) -> Optional[float]:
        """Simple Moving Average over a time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        relevant = [s for s in snapshots if s.timestamp >= cutoff]
        prices = []
        for s in relevant:
            price = s.instant_buy if use_buy else s.instant_sell
            if price and price > 0:
                prices.append(price)
        return statistics.mean(prices) if prices else None

    def detect_trend(self, snapshots: List[PriceSnapshot]) -> Tuple[Trend, float]:
        """
        Detect trend by comparing short vs medium vs long SMAs.
        Returns (trend, momentum_gp_per_min).
        """
        sma_5m = self.calculate_sma(snapshots, 5)
        sma_30m = self.calculate_sma(snapshots, 30)
        sma_2h = self.calculate_sma(snapshots, 120)

        if not sma_5m or not sma_30m:
            return Trend.NEUTRAL, 0.0

        # Momentum: GP change per minute over last 5 minutes
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        recent = [s for s in snapshots if s.timestamp >= cutoff and s.instant_buy]
        if len(recent) >= 2:
            first_price = recent[0].instant_buy
            last_price = recent[-1].instant_buy
            elapsed_min = max((recent[-1].timestamp - recent[0].timestamp).total_seconds() / 60, 0.1)
            momentum = (last_price - first_price) / elapsed_min
        else:
            momentum = 0.0

        # Classify trend
        if sma_2h:
            if sma_5m > sma_30m > sma_2h and momentum > 0:
                return Trend.STRONG_UP, momentum
            elif sma_5m < sma_30m < sma_2h and momentum < 0:
                return Trend.STRONG_DOWN, momentum

        if sma_5m > sma_30m:
            return Trend.UP, momentum
        elif sma_5m < sma_30m:
            return Trend.DOWN, momentum
        else:
            return Trend.NEUTRAL, momentum

    def calculate_bollinger_bands(
        self,
        snapshots: List[PriceSnapshot],
        period_minutes: int = 120,
        num_std: float = 2.0,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Bollinger Bands: middle = SMA, upper/lower = SMA +/- N*stdev.
        Returns (upper, middle, lower).
        """
        cutoff = datetime.utcnow() - timedelta(minutes=period_minutes)
        prices = [
            s.instant_buy
            for s in snapshots
            if s.timestamp >= cutoff and s.instant_buy
        ]
        if len(prices) < 5:
            return None, None, None

        middle = statistics.mean(prices)
        stdev = statistics.stdev(prices)
        upper = middle + num_std * stdev
        lower = middle - num_std * stdev
        return upper, middle, lower

    def validate_against_5m(
        self,
        instant_buy: Optional[int],
        instant_sell: Optional[int],
        snapshots: List[PriceSnapshot],
    ) -> Tuple[Optional[int], Optional[int], bool]:
        """
        Ghost Margin Fix: validate /latest prices against 5m averages.
        If instant price deviates >5% from 5m avg, use the 5m avg instead.
        Returns (validated_buy, validated_sell, was_ghost).
        """
        vwap_5m_buy = self.calculate_vwap(snapshots, 5, use_buy=True)
        vwap_5m_sell = self.calculate_vwap(snapshots, 5, use_buy=False)
        was_ghost = False

        validated_buy = instant_buy
        validated_sell = instant_sell

        if vwap_5m_buy and instant_buy:
            # If instant buy is >5% higher than 5m avg, it's a fake spike
            if instant_buy > vwap_5m_buy * 1.05:
                validated_buy = int(vwap_5m_buy)
                was_ghost = True
            # If instant buy is >5% lower than 5m avg, it's a fake dip
            if instant_buy < vwap_5m_buy * 0.95:
                validated_buy = int(vwap_5m_buy)
                was_ghost = True

        if vwap_5m_sell and instant_sell:
            if instant_sell > vwap_5m_sell * 1.05:
                validated_sell = int(vwap_5m_sell)
                was_ghost = True
            if instant_sell < vwap_5m_sell * 0.95:
                validated_sell = int(vwap_5m_sell)
                was_ghost = True

        return validated_buy, validated_sell, was_ghost

    def calculate_undercut(
        self,
        price: int,
        volume_5m: int,
        is_buy: bool,
    ) -> int:
        """
        Calculate optimal queue-jumping offset based on volume and price.
        For buys: INCREASE offer price to jump ahead in the buy queue.
        For sells: DECREASE list price to sell faster.

        Key insight: low-volume items have barely any queue, so large
        offsets just burn profit.  Only high-volume items need serious
        queue-jumping.
        """
        if price < 10_000:
            # Cheap items: 1 GP offset
            return 1

        # Base percentage by volume tier (these are per-side, not round-trip)
        if volume_5m >= 100:
            pct = 0.0008   # 0.08% – extremely liquid
        elif volume_5m >= 50:
            pct = 0.001    # 0.10%
        elif volume_5m >= 20:
            pct = 0.0015   # 0.15%
        elif volume_5m >= 10:
            pct = 0.002    # 0.20%
        elif volume_5m >= 5:
            pct = 0.0015   # 0.15% – moderate, small queue
        elif volume_5m >= 2:
            pct = 0.001    # 0.10% – low volume, barely a queue
        else:
            pct = 0.0005   # 0.05% – practically no queue

        amount = max(1, int(price * pct))

        # Hard cap: never offset more than 20K on items under 50M
        if price < 50_000_000:
            amount = min(amount, 20_000)

        # For very expensive items (50M+), cap at 0.15% or 50K
        if price >= 50_000_000:
            amount = min(amount, max(1, int(price * 0.0015)), 50_000)

        return amount

    def clamp_buy_price(
        self,
        instant_sell: int,
        trend: Trend,
        vwap_5m: Optional[float],
        vwap_30m: Optional[float],
        momentum: float,
        spread: int,
    ) -> int:
        """
        Clamp buy price against trend.
        Buy price starts at insta_sell and gets adjusted.
        """
        base = instant_sell

        if trend in (Trend.STRONG_UP, Trend.UP):
            # Uptrend: price might keep going up, be willing to pay more
            # but cap at VWAP_5m to avoid chasing
            if vwap_5m:
                base = max(instant_sell, int(vwap_5m))
        elif trend in (Trend.STRONG_DOWN, Trend.DOWN):
            # Downtrend: wait for price to stabilize
            # Use lower of insta_sell and VWAP_30m minus a cushion
            cushion = int(abs(momentum) * 5)  # 5 minutes of momentum
            if vwap_30m:
                base = min(instant_sell, int(vwap_30m) - cushion)
            else:
                base = instant_sell - cushion
        # NEUTRAL: use insta_sell as-is

        return max(1, base)

    def clamp_sell_price(
        self,
        instant_buy: int,
        trend: Trend,
        vwap_5m: Optional[float],
        vwap_30m: Optional[float],
        momentum: float,
        spread: int,
        volume_5m: int = 0,
    ) -> int:
        """
        Clamp sell price against trend.
        Sell price starts at insta_buy and gets adjusted.
        For low-volume items, use VWAP as a sanity check to avoid
        relying on a single stale high-price transaction.
        """
        base = instant_buy

        # For low-volume items, the last instant_buy might be an outlier.
        # Use the lower of instant_buy and VWAP-5m to be conservative.
        if volume_5m < 5 and vwap_5m and vwap_5m < instant_buy:
            base = int(vwap_5m)

        if trend in (Trend.STRONG_UP, Trend.UP):
            # Uptrend: ride the wave, sell higher
            if vwap_30m:
                momentum_bonus = int(abs(momentum) * 10)  # 10 min projection
                base = max(instant_buy, int(vwap_30m) + momentum_bonus)
        elif trend in (Trend.STRONG_DOWN, Trend.DOWN):
            # Downtrend: sell fast before further drops
            if vwap_5m:
                base = min(instant_buy, int(vwap_5m))
        # NEUTRAL: use insta_buy as-is

        return max(1, base)

    def detect_waterfall(self, snapshots: List[PriceSnapshot]) -> bool:
        """
        Detect a 'waterfall' crash: price dropping >2% every 5 minutes
        for 3+ consecutive intervals. If detected, DO NOT BUY.
        """
        if len(snapshots) < 10:
            return False

        # Group snapshots into 5-minute buckets and get avg price per bucket
        from collections import defaultdict
        buckets = defaultdict(list)
        for s in snapshots:
            if s.instant_buy and s.instant_buy > 0:
                bucket = s.timestamp.replace(
                    minute=(s.timestamp.minute // 5) * 5,
                    second=0, microsecond=0,
                )
                buckets[bucket].append(s.instant_buy)

        if len(buckets) < 4:
            return False

        # Get average price per bucket, sorted by time
        sorted_buckets = sorted(buckets.items())
        avg_prices = [statistics.mean(prices) for _, prices in sorted_buckets]

        # Check last 3 intervals for consecutive drops
        if len(avg_prices) < 4:
            return False

        recent = avg_prices[-4:]  # Last 4 buckets = 3 intervals
        pct_changes = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                pct_changes.append((recent[i] - recent[i - 1]) / recent[i - 1])

        # Waterfall = all 3 intervals negative AND total drop > 5%
        if len(pct_changes) >= 3:
            if all(c < 0 for c in pct_changes) and sum(pct_changes) < -0.05:
                return True

        return False

    def check_volume_liveness(self, snapshots: List[PriceSnapshot]) -> str:
        """
        Check if volume has recently died. An item with historical volume
        but 0 recent trades is a trap.
        Returns: 'HEALTHY', 'DECLINING', or 'DEAD_VOLUME_TRAP'
        """
        if len(snapshots) < 6:
            return "HEALTHY"

        # Recent volume (last 3 snapshots = ~30 seconds)
        recent = snapshots[-3:]
        recent_vol = sum((s.buy_volume or 0) + (s.sell_volume or 0) for s in recent)

        # Historical volume (older snapshots)
        older = snapshots[:-3]
        if not older:
            return "HEALTHY"
        older_vol = sum((s.buy_volume or 0) + (s.sell_volume or 0) for s in older)
        avg_older = older_vol / len(older) * 3  # Scale to same window size

        if recent_vol == 0 and avg_older > 5:
            return "DEAD_VOLUME_TRAP"
        elif avg_older > 0 and recent_vol / max(avg_older, 1) < 0.2:
            return "DECLINING"

        return "HEALTHY"

    def check_sanity(
        self,
        instant_buy: Optional[int],
        instant_sell: Optional[int],
        buy_time: Optional[int],
        sell_time: Optional[int],
        volume_5m: int,
        historical_spread_pct: Optional[float],
    ) -> Tuple[bool, bool, float]:
        """
        Sanity checks to avoid dumb suggestions.
        Returns (stale_data, anomalous_spread, confidence).
        """
        now = int(time.time())
        stale = False
        anomalous = False
        confidence = 1.0

        # Stale data check
        if buy_time:
            age_min = (now - buy_time) / 60
            if age_min > 15 and volume_5m < 3:
                stale = True
                confidence *= 0.5
            elif age_min > 30:
                stale = True
                confidence *= 0.3

        if sell_time:
            age_min = (now - sell_time) / 60
            if age_min > 15 and volume_5m < 3:
                stale = True
                confidence *= 0.5

        # Spread sanity check
        if instant_buy and instant_sell and instant_sell > 0:
            current_spread_pct = (instant_buy - instant_sell) / instant_sell * 100
            if historical_spread_pct is not None:
                if current_spread_pct > historical_spread_pct * 3:
                    anomalous = True
                    confidence *= 0.4
            elif current_spread_pct > 5:
                anomalous = True
                confidence *= 0.6

        # Volume confidence
        if volume_5m >= 20:
            confidence *= 1.0
        elif volume_5m >= 5:
            confidence *= 0.85
        elif volume_5m >= 1:
            confidence *= 0.6
        else:
            confidence *= 0.3

        return stale, anomalous, min(1.0, confidence)

    def price_item(
        self,
        item_id: int,
        snapshots: Optional[List[PriceSnapshot]] = None,
    ) -> PriceRecommendation:
        """
        Generate a smart price recommendation for an item.
        Uses recent snapshots from the database.
        """
        rec = PriceRecommendation(item_id=item_id, timestamp=datetime.utcnow())

        # Get snapshots from DB if not provided
        if snapshots is None:
            db = get_db()
            try:
                snapshots = get_price_history(db, item_id, hours=4)
            finally:
                db.close()

        if not snapshots:
            rec.reason = "No price data available"
            return rec

        # Latest snapshot for current prices
        latest = snapshots[-1]
        rec.instant_buy = latest.instant_buy
        rec.instant_sell = latest.instant_sell

        if not rec.instant_buy or not rec.instant_sell:
            rec.reason = "Missing buy or sell price"
            return rec

        # Volume from latest snapshot
        rec.volume_5m = (latest.buy_volume or 0) + (latest.sell_volume or 0)

        # Ghost Margin Fix: validate /latest against 5m averages
        # If instant price deviates >5% from 5m VWAP, it's likely a one-off
        # fat-finger trade, not a real price. Use 5m avg instead.
        validated_buy, validated_sell, was_ghost = self.validate_against_5m(
            rec.instant_buy, rec.instant_sell, snapshots,
        )
        if was_ghost:
            rec.anomalous_spread = True
            rec.confidence *= 0.6  # Reduce confidence for ghost margins
            rec.instant_buy = validated_buy
            rec.instant_sell = validated_sell

        # Calculate VWAPs
        rec.vwap_1m = self.calculate_vwap(snapshots, 1)
        rec.vwap_5m = self.calculate_vwap(snapshots, 5)
        rec.vwap_30m = self.calculate_vwap(snapshots, 30)
        rec.vwap_2h = self.calculate_vwap(snapshots, 120)

        # Detect trend
        rec.trend, rec.momentum = self.detect_trend(snapshots)

        # Bollinger bands
        rec.bb_upper, rec.bb_middle, rec.bb_lower = self.calculate_bollinger_bands(snapshots)
        if rec.bb_upper and rec.bb_lower and rec.bb_upper > rec.bb_lower:
            rec.bb_position = (rec.instant_buy - rec.bb_lower) / (rec.bb_upper - rec.bb_lower)
            rec.bb_position = max(0.0, min(1.0, rec.bb_position))

        # Waterfall detection: if price is crashing, don't buy
        if self.detect_waterfall(snapshots):
            rec.confidence *= 0.2
            rec.reason = "WATERFALL DETECTED - price crashing, avoid buying"

        # Volume liveness check
        vol_status = self.check_volume_liveness(snapshots)
        if vol_status == "DEAD_VOLUME_TRAP":
            rec.confidence *= 0.3
            if not rec.reason:
                rec.reason = "DEAD VOLUME - recent trades dried up, likely trap"
        elif vol_status == "DECLINING":
            rec.confidence *= 0.7

        # Compute historical spread from flip history
        historical_spread_pct = None
        try:
            db2 = get_db()
            try:
                flips = get_item_flips(db2, item_id, days=14)
                if len(flips) >= 3:
                    margins = [
                        f.margin_pct for f in flips
                        if f.margin_pct is not None and f.margin_pct > 0
                    ]
                    if margins:
                        historical_spread_pct = statistics.median(margins)
            finally:
                db2.close()
        except Exception:
            pass

        # Sanity checks
        spread = rec.instant_buy - rec.instant_sell
        rec.stale_data, rec.anomalous_spread, rec.confidence = self.check_sanity(
            rec.instant_buy,
            rec.instant_sell,
            latest.buy_time,
            latest.sell_time,
            rec.volume_5m,
            historical_spread_pct=historical_spread_pct,
        )

        # Clamp prices against trend
        rec.clamped_buy = self.clamp_buy_price(
            rec.instant_sell, rec.trend, rec.vwap_5m, rec.vwap_30m,
            rec.momentum, spread,
        )
        rec.clamped_sell = self.clamp_sell_price(
            rec.instant_buy, rec.trend, rec.vwap_5m, rec.vwap_30m,
            rec.momentum, spread, volume_5m=rec.volume_5m,
        )

        # Apply queue-jumping offsets
        # Buy: INCREASE offer to jump ahead of other buyers
        # Sell: DECREASE price to sell faster than other sellers
        buy_offset = self.calculate_undercut(rec.clamped_buy, rec.volume_5m, is_buy=True)
        sell_offset = self.calculate_undercut(rec.clamped_sell, rec.volume_5m, is_buy=False)

        rec.recommended_buy = rec.clamped_buy + buy_offset   # Offer MORE to buy first
        rec.recommended_sell = rec.clamped_sell - sell_offset  # List LOWER to sell first

        # Ensure buy < sell
        if rec.recommended_buy >= rec.recommended_sell:
            rec.recommended_buy = rec.instant_sell
            rec.recommended_sell = rec.instant_buy

        # Calculate expected profit
        gross = rec.recommended_sell - rec.recommended_buy
        tax = int(min(rec.recommended_sell * GE_TAX_RATE, GE_TAX_CAP))
        rec.tax = tax
        rec.expected_profit = gross - tax
        rec.expected_profit_pct = (
            (rec.expected_profit / rec.recommended_buy * 100)
            if rec.recommended_buy > 0
            else 0.0
        )

        # Generate reason
        rec.reason = self._generate_reason(rec)

        return rec

    def _generate_reason(self, rec: PriceRecommendation) -> str:
        """Generate a human-readable reason for the recommendation."""
        parts = []

        if rec.stale_data:
            parts.append("STALE DATA - prices may be outdated")
        if rec.anomalous_spread:
            parts.append("ANOMALOUS SPREAD - unusually wide, may be a trap")

        trend_msg = {
            Trend.STRONG_UP: "Strong uptrend - riding momentum",
            Trend.UP: "Uptrend - slightly aggressive buy",
            Trend.NEUTRAL: "Neutral - standard margin flip",
            Trend.DOWN: "Downtrend - conservative buy, fast sell",
            Trend.STRONG_DOWN: "Strong downtrend - very conservative",
        }
        parts.append(trend_msg.get(rec.trend, ""))

        if rec.bb_position is not None:
            if rec.bb_position < 0.2:
                parts.append("Near lower Bollinger - potential oversold")
            elif rec.bb_position > 0.8:
                parts.append("Near upper Bollinger - potential overbought")

        if rec.expected_profit and rec.expected_profit > 0:
            parts.append(f"Expected: +{rec.expected_profit:,} GP ({rec.expected_profit_pct:.1f}%)")
        elif rec.expected_profit:
            parts.append(f"UNPROFITABLE after tax: {rec.expected_profit:,} GP")

        return " | ".join(parts)
