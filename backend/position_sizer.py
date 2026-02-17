"""
Position Sizing & Risk Management for OSRS Flipping AI

Implements Kelly Criterion for optimal bet sizing, plus hard risk
limits (max exposure, stop-loss, portfolio correlation).

Usage:
    sizer = PositionSizer(bankroll=100_000_000)
    advice = sizer.size_position(flip_score, item_id)
    # advice.quantity, advice.max_investment, advice.stop_loss_price, etc.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.database import (
    get_db,
    get_item_flips,
    get_latest_price,
    get_setting,
    FlipHistory,
)

logger = logging.getLogger(__name__)


@dataclass
class PositionAdvice:
    """Recommended position size and risk parameters for a flip."""
    item_id: int
    item_name: str = ""

    # Kelly sizing
    kelly_fraction: float = 0.0      # Raw Kelly fraction (can be > 1)
    half_kelly: float = 0.0          # Conservative half-Kelly
    recommended_fraction: float = 0.0  # After applying all caps

    # Concrete recommendations
    max_investment: int = 0          # Max GP to invest in this flip
    quantity: int = 0                # Recommended quantity to buy
    buy_price: int = 0

    # Risk management
    stop_loss_price: int = 0         # Exit if price drops below this
    stop_loss_pct: float = 0.0       # Stop loss as % below buy price
    take_profit_price: int = 0       # Consider selling above this
    max_hold_minutes: int = 0        # Time-based stop

    # Exposure checks
    portfolio_exposure_pct: float = 0.0  # % of bankroll on this flip
    item_concentration_pct: float = 0.0  # % of bankroll on this item total
    within_limits: bool = True
    limit_warnings: list = field(default_factory=list)

    # Reasoning
    reason: str = ""


class PositionSizer:
    """
    Computes optimal position sizes using Kelly Criterion
    with conservative adjustments for the OSRS Grand Exchange.

    Kelly formula:  f* = (p * b - q) / b
    where:
        p = probability of winning (from historical win rate)
        q = 1 - p (probability of losing)
        b = win/loss ratio (avg win / avg loss)

    We use half-Kelly for safety, then cap at hard limits.
    """

    # Hard limits
    MAX_SINGLE_POSITION_PCT = 0.15   # Never put more than 15% of bankroll on one flip
    MAX_ITEM_EXPOSURE_PCT = 0.25     # Never have more than 25% of bankroll in one item
    MAX_TOTAL_EXPOSURE_PCT = 0.80    # Keep 20% of bankroll as cash reserve
    MIN_WIN_RATE_FOR_SIZING = 0.45   # Need at least 45% win rate to suggest sizing
    DEFAULT_STOP_LOSS_PCT = 0.03     # 3% stop loss for most items
    HIGH_VOL_STOP_LOSS_PCT = 0.05    # 5% for high-value volatile items
    LOW_VOL_STOP_LOSS_PCT = 0.02     # 2% for low-value stable items

    def __init__(self, bankroll: Optional[int] = None):
        self._bankroll = bankroll

    @property
    def bankroll(self) -> int:
        """Get current bankroll from settings or use provided value."""
        if self._bankroll is not None:
            return self._bankroll
        db = get_db()
        try:
            return get_setting(db, "bankroll", default=50_000_000)
        finally:
            db.close()

    def size_position(
        self,
        item_id: int,
        buy_price: int,
        sell_price: int,
        score: float = 50.0,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        volume_5m: int = 10,
        item_name: str = "",
        buy_limit: int = 10000,
        existing_exposure: int = 0,
    ) -> PositionAdvice:
        """
        Compute recommended position size for a flip.

        Parameters
        ----------
        item_id : int
            OSRS item ID.
        buy_price : int
            Price per unit to buy at.
        sell_price : int
            Expected sell price per unit.
        score : float
            FlipScorer composite score (0-100).
        win_rate : float, optional
            Historical win rate (0-1). Computed from DB if None.
        avg_win : float, optional
            Average profit on winning flips. Computed from DB if None.
        avg_loss : float, optional
            Average loss on losing flips (positive number). Computed from DB if None.
        volume_5m : int
            5-minute trading volume.
        item_name : str
            Item name for display.
        buy_limit : int
            GE buy limit for this item.
        existing_exposure : int
            GP already invested in this item.
        """
        advice = PositionAdvice(item_id=item_id, item_name=item_name, buy_price=buy_price)
        bankroll = self.bankroll

        if buy_price <= 0 or bankroll <= 0:
            advice.reason = "Invalid price or bankroll"
            advice.within_limits = False
            return advice

        # Fetch historical data if not provided
        if win_rate is None or avg_win is None or avg_loss is None:
            win_rate, avg_win, avg_loss = self._compute_historical_stats(item_id)

        # --- Kelly Criterion ---
        kelly = self._kelly_fraction(win_rate, avg_win, avg_loss, buy_price, sell_price)
        advice.kelly_fraction = kelly
        advice.half_kelly = kelly / 2

        # Start with half-Kelly (industry standard conservative approach)
        fraction = kelly / 2

        # Scale by score (higher score = more confidence in sizing)
        score_multiplier = max(0.3, min(1.0, score / 80))
        fraction *= score_multiplier

        # Cap at hard limits
        fraction = min(fraction, self.MAX_SINGLE_POSITION_PCT)
        fraction = max(fraction, 0.0)

        # Check total exposure
        if existing_exposure > 0:
            current_item_pct = existing_exposure / bankroll
            remaining_item_room = self.MAX_ITEM_EXPOSURE_PCT - current_item_pct
            if remaining_item_room <= 0:
                advice.within_limits = False
                advice.limit_warnings.append(
                    f"Already {current_item_pct:.0%} exposed to this item (max {self.MAX_ITEM_EXPOSURE_PCT:.0%})"
                )
                fraction = 0
            else:
                fraction = min(fraction, remaining_item_room)

        advice.recommended_fraction = fraction

        # --- Convert to GP and quantity ---
        max_investment = int(bankroll * fraction)
        max_quantity = max_investment // buy_price if buy_price > 0 else 0

        # Respect GE buy limit
        max_quantity = min(max_quantity, buy_limit)

        # Respect volume (don't try to buy more than 2x the 5-min volume)
        if volume_5m > 0:
            volume_cap = volume_5m * 2
            if max_quantity > volume_cap:
                max_quantity = volume_cap
                advice.limit_warnings.append(
                    f"Quantity capped at {volume_cap} (2x 5-min volume of {volume_5m})"
                )

        advice.max_investment = max_quantity * buy_price
        advice.quantity = max_quantity
        advice.portfolio_exposure_pct = (advice.max_investment / bankroll * 100) if bankroll > 0 else 0
        advice.item_concentration_pct = ((existing_exposure + advice.max_investment) / bankroll * 100) if bankroll > 0 else 0

        # --- Stop-loss & Take-profit ---
        stop_pct = self._compute_stop_loss_pct(buy_price, volume_5m, score)
        advice.stop_loss_pct = stop_pct
        advice.stop_loss_price = int(buy_price * (1 - stop_pct))

        # Take profit at expected sell price (already tax-adjusted from SmartPricer)
        advice.take_profit_price = sell_price

        # Time-based stop: lower volume = shorter hold time
        if volume_5m >= 50:
            advice.max_hold_minutes = 240   # 4 hours for high liquidity
        elif volume_5m >= 20:
            advice.max_hold_minutes = 120   # 2 hours
        elif volume_5m >= 5:
            advice.max_hold_minutes = 60    # 1 hour
        else:
            advice.max_hold_minutes = 30    # 30 min for low liquidity

        # --- Build reason ---
        advice.reason = self._build_reason(advice, win_rate, kelly, score)

        return advice

    def _kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        buy_price: int,
        sell_price: int,
    ) -> float:
        """
        Compute Kelly fraction.

        f* = (p * b - q) / b

        where p = win probability, q = 1-p, b = win/loss ratio
        """
        p = max(0.01, min(0.99, win_rate))
        q = 1 - p

        # Use historical win/loss ratio if available, else estimate from prices
        if avg_win > 0 and avg_loss > 0:
            b = avg_win / avg_loss
        elif buy_price > 0 and sell_price > buy_price:
            # Estimate: win = spread, loss = stop loss (3%)
            expected_win = sell_price - buy_price
            expected_loss = buy_price * self.DEFAULT_STOP_LOSS_PCT
            b = expected_win / max(expected_loss, 1)
        else:
            b = 1.0  # Even odds

        kelly = (p * b - q) / b if b > 0 else 0
        return max(0, kelly)

    def _compute_historical_stats(self, item_id: int):
        """Fetch win rate, avg win, avg loss from flip history."""
        db = get_db()
        try:
            flips = get_item_flips(db, item_id, days=30)
            if len(flips) < 3:
                return (0.55, 0, 0)  # Default moderate assumption

            wins = [f for f in flips if f.net_profit and f.net_profit > 0]
            losses = [f for f in flips if f.net_profit and f.net_profit <= 0]

            win_rate = len(wins) / len(flips) if flips else 0.5
            avg_win = sum(f.net_profit for f in wins) / len(wins) if wins else 0
            avg_loss = abs(sum(f.net_profit for f in losses) / len(losses)) if losses else 0

            return (win_rate, avg_win, avg_loss)
        finally:
            db.close()

    def _compute_stop_loss_pct(self, buy_price: int, volume_5m: int, score: float) -> float:
        """Determine stop-loss percentage based on item characteristics."""
        if buy_price >= 10_000_000:
            # High-value items: wider stop to avoid noise
            base = self.HIGH_VOL_STOP_LOSS_PCT
        elif buy_price <= 100_000:
            # Low-value items: tighter stop
            base = self.LOW_VOL_STOP_LOSS_PCT
        else:
            base = self.DEFAULT_STOP_LOSS_PCT

        # Widen stop slightly for low-volume items (more volatile)
        if volume_5m < 5:
            base *= 1.5
        elif volume_5m > 50:
            base *= 0.8

        # Tighten stop for low-score flips (less confident)
        if score < 50:
            base *= 0.7

        return round(base, 4)

    def _build_reason(self, advice: PositionAdvice, win_rate: float, kelly: float, score: float) -> str:
        parts = []

        if kelly <= 0:
            parts.append("Kelly says SKIP (negative edge)")
        elif kelly < 0.05:
            parts.append("Small edge - minimal position")
        elif kelly < 0.15:
            parts.append("Moderate edge - standard position")
        else:
            parts.append("Strong edge - full position")

        parts.append(f"WR: {win_rate:.0%}")
        parts.append(f"Kelly: {kelly:.1%} -> Half: {advice.half_kelly:.1%}")

        if advice.quantity > 0:
            parts.append(f"Buy {advice.quantity:,} @ {advice.buy_price:,}")
            parts.append(f"Risk: {advice.portfolio_exposure_pct:.1f}% of bank")
        else:
            parts.append("No position recommended")

        if advice.limit_warnings:
            parts.append(f"Warnings: {'; '.join(advice.limit_warnings)}")

        return " | ".join(parts)
