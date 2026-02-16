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

    def calculate_undercut(
        self,
        price: int,
        volume_5m: int,
        is_buy: bool,
    ) -> int:
        """
        Calculate optimal undercut/overcut amount based on volume.
        For buys: undercut means offering LESS (to buy cheaper).
        For sells: overcut means listing LOWER (to sell faster).
        """
        if volume_5m > 50:
            # High volume: tiny undercut, 1 GP
            amount = 1
        elif volume_5m > 20:
            # Medium-high: 0.05% of price
            amount = max(1, int(price * 0.0005))
        elif volume_5m > 5:
            # Medium: 0.1-0.5% of price
            amount = max(1, int(price * 0.002))
        elif volume_5m > 1:
            # Low: 0.5-1%
            amount = max(1, int(price * 0.005))
        else:
            # Very low / no trades: 1-2%
            amount = max(1, int(price * 0.015))

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
    ) -> int:
        """
        Clamp sell price against trend.
        Sell price starts at insta_buy and gets adjusted.
        """
        base = instant_buy

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

        # Sanity checks
        spread = rec.instant_buy - rec.instant_sell
        rec.stale_data, rec.anomalous_spread, rec.confidence = self.check_sanity(
            rec.instant_buy,
            rec.instant_sell,
            latest.buy_time,
            latest.sell_time,
            rec.volume_5m,
            historical_spread_pct=None,  # TODO: compute from flip_history
        )

        # Clamp prices against trend
        rec.clamped_buy = self.clamp_buy_price(
            rec.instant_sell, rec.trend, rec.vwap_5m, rec.vwap_30m,
            rec.momentum, spread,
        )
        rec.clamped_sell = self.clamp_sell_price(
            rec.instant_buy, rec.trend, rec.vwap_5m, rec.vwap_30m,
            rec.momentum, spread,
        )

        # Apply undercut/overcut
        buy_undercut = self.calculate_undercut(rec.clamped_buy, rec.volume_5m, is_buy=True)
        sell_overcut = self.calculate_undercut(rec.clamped_sell, rec.volume_5m, is_buy=False)

        rec.recommended_buy = rec.clamped_buy - buy_undercut
        rec.recommended_sell = rec.clamped_sell + sell_overcut

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
