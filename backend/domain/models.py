"""
backend.domain.models — Canonical Pydantic / dataclass models.

These are the single source of truth for data structures flowing through
the entire platform.  Layers that produce or consume these models must not
invent their own parallel types.

Import pattern::

    from backend.domain.models import ItemMetrics, UserContext, FlipOutcome
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.domain.enums import (
    FlipStatus, RiskLevel, RiskProfile, TrendDirection, VolumeRating,
)


# ---------------------------------------------------------------------------
# User context (resolved once per API request)
# ---------------------------------------------------------------------------

@dataclass
class UserContext:
    """
    Lightweight context object built from the authenticated request.
    Passed down through service calls so every layer can personalise output
    without touching the request object directly.
    """
    user_id: Optional[str] = None           # Discord user ID (str) or None
    username: Optional[str] = None
    risk_profile: RiskProfile = RiskProfile.BALANCED

    # Calibration multipliers (loaded from DB, default = 1.0 if no history)
    profit_multiplier: float = 1.0          # median(realized / expected) profit
    hold_multiplier:   float = 1.0          # median(realized / estimated) hold time

    # Affinity boosts keyed by item_id or category tag
    item_affinity:     Dict[int, float] = field(default_factory=dict)
    category_affinity: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def anonymous(cls) -> "UserContext":
        """Return a default balanced context for unauthenticated requests."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "UserContext":
        profile = RiskProfile(d.get("risk_profile", "balanced"))
        return cls(
            user_id=d.get("user_id"),
            username=d.get("username"),
            risk_profile=profile,
            profit_multiplier=float(d.get("profit_multiplier", 1.0)),
            hold_multiplier=float(d.get("hold_multiplier", 1.0)),
            item_affinity=d.get("item_affinity", {}),
            category_affinity=d.get("category_affinity", {}),
        )


# ---------------------------------------------------------------------------
# Canonical per-item metrics (output of analytics.scoring)
# ---------------------------------------------------------------------------

@dataclass
class ItemMetrics:
    """
    Full analytics output for a single OSRS item at a point in time.
    Returned by ``backend.analytics.scoring.score_item()``.

    All monetary values are in GP (integer).
    Score fields are in range [0, 100].
    """
    # Identity
    item_id:   int
    item_name: str = ""

    # Prices
    buy:       int = 0
    sell:      int = 0

    # Profit after GE tax
    margin_after_tax: int = 0
    roi_pct:          float = 0.0

    # Volatility
    volatility_1h:  float = 0.0
    volatility_24h: float = 0.0

    # Volume / liquidity
    volume_score:     float = 0.0
    liquidity_score:  float = 0.0
    fill_probability: float = 0.0          # [0, 1]
    est_fill_time_minutes: float = 0.0

    # Trend / decay
    trend_score: float = 0.0
    trend:       TrendDirection = TrendDirection.NEUTRAL
    decay_score: float = 0.0               # spread compression speed

    # Risk / confidence
    risk_level:     RiskLevel = RiskLevel.MEDIUM
    risk_score:     float = 5.0            # 0 = safest, 10 = riskiest
    confidence_pct: float = 50.0           # [0, 100]

    # Profit projections
    expected_profit:          int   = 0
    risk_adjusted_gp_per_hour: float = 0.0

    # Final score (after applying risk-profile overrides + personalization)
    final_score: float = 0.0

    # Veto flag
    vetoed:       bool = False
    veto_reason:  str  = ""

    # Component sub-scores (for debugging / UI drill-down)
    score_spread:  float = 0.0
    score_volume:  float = 0.0
    score_trend:   float = 0.0
    score_history: float = 0.0
    score_stability: float = 0.0
    score_freshness: float = 0.0
    score_ml:      float = 0.0

    # Raw computed extras
    volume_rating:          VolumeRating = VolumeRating.LOW
    estimated_hold_minutes: float = 0.0
    recommended_buy:        int   = 0
    recommended_sell:       int   = 0
    net_profit:             int   = 0      # alias for margin_after_tax
    total_score:            float = 0.0    # alias for final_score
    gp_per_hour:            float = 0.0   # un-adjusted
    ma_signal:              float = 0.0
    volume_delta:           float = 0.0
    spread_compression:     float = 0.0

    # Personalisation adjustments applied
    personalization_applied: bool = False
    affinity_boost:          float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict for JSON responses / caching."""
        import dataclasses
        d = dataclasses.asdict(self)
        # Convert enum values to their string representations
        d["trend"]      = self.trend.value
        d["risk_level"] = self.risk_level.value
        d["volume_rating"] = self.volume_rating.value
        return d


# ---------------------------------------------------------------------------
# User record (stored in MongoDB `users` collection)
# ---------------------------------------------------------------------------

class UserRecord(BaseModel):
    """MongoDB document for a registered user."""
    user_id:   str                            # Discord ID (primary key _id)
    username:  str
    avatar:    Optional[str] = None

    risk_profile: RiskProfile = RiskProfile.BALANCED

    # Alert preferences (overrides global defaults)
    alert_margin_threshold: Optional[int]   = None
    alert_volume_spike_x:   Optional[float] = None   # e.g. 3.0 = 3× average
    watchlist:              List[int]        = Field(default_factory=list)

    # Personalisation calibration (updated periodically by calibration.py)
    profit_multiplier: float = 1.0
    hold_multiplier:   float = 1.0

    # Affinity maps (updated by affinity.py)
    item_affinity:     Dict[str, float] = Field(default_factory=dict)  # str(item_id) → boost
    category_affinity: Dict[str, float] = Field(default_factory=dict)

    subscription_tier: str = "free"   # "free" | "pro" | "enterprise"

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Flip outcome (stored in `flip_outcomes` collection)
# ---------------------------------------------------------------------------

class FlipOutcome(BaseModel):
    """
    Records everything about a completed (or in-progress) flip so we can
    train personalisation models.
    """
    user_id:   str
    item_id:   int
    item_name: str = ""

    ts_open:  datetime
    ts_close: Optional[datetime] = None

    qty: int = 1

    buy_target:      int = 0
    buy_filled_avg:  int = 0
    sell_target:     int = 0
    sell_filled_avg: int = 0

    expected_profit_at_open: int   = 0
    realized_profit:         int   = 0

    est_hold_minutes_at_open:  float = 0.0
    realized_hold_minutes:     Optional[float] = None

    status: FlipStatus = FlipStatus.OPEN

    risk_profile_used: RiskProfile = RiskProfile.BALANCED

    # Snapshot of every analytics metric at decision time
    feature_snapshot_at_open: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Slim price snapshot (domain layer alias)
# ---------------------------------------------------------------------------

@dataclass
class PricePoint:
    """Minimal price record for analytics computations."""
    item_id:    int
    ts:         datetime
    buy:        int     # instant buy (low) price
    sell:       int     # instant sell (high) price
    buy_vol:    int = 0
    sell_vol:   int = 0
    source:     str = "wiki"   # "wiki_latest" | "wiki_5m" | "wiki_1h"


# ---------------------------------------------------------------------------
# Portfolio slot allocation
# ---------------------------------------------------------------------------

@dataclass
class SlotAllocation:
    """One GE slot in an optimised portfolio plan."""
    slot:       int
    item_id:    int
    item_name:  str

    buy_price:  int
    sell_price: int
    quantity:   int

    investment:          int
    expected_net_profit: int
    stop_loss_price:     int

    estimated_hold_minutes: float
    gp_per_hour:            float
    risk_adjusted_gph:      float

    risk_score:   float
    risk_level:   RiskLevel
    confidence:   float
    kelly_fraction: float
    score:        float

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        d = dataclasses.asdict(self)
        d["risk_level"] = self.risk_level.value
        return d


@dataclass
class PortfolioPlan:
    """Full portfolio allocation plan returned by the optimizer."""
    capital:      int
    ge_slots:     int
    risk_profile: RiskProfile
    allocations:  List[SlotAllocation] = field(default_factory=list)

    total_invested:        int   = 0
    total_expected_profit: int   = 0
    total_expected_gph:    float = 0.0
    remaining_capital:     int   = 0

    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capital":               self.capital,
            "ge_slots":              self.ge_slots,
            "risk_profile":          self.risk_profile.value,
            "allocations":           [a.to_dict() for a in self.allocations],
            "total_invested":        self.total_invested,
            "total_expected_profit": self.total_expected_profit,
            "total_expected_gph":    self.total_expected_gph,
            "remaining_capital":     self.remaining_capital,
            "generated_at":          self.generated_at.isoformat(),
        }
