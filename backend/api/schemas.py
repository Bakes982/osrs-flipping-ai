"""
OSRS Flipping AI — API request/response schemas (Pydantic).

All FastAPI endpoints that return structured data must use these models.
This gives us:
  • Automatic OpenAPI documentation
  • Runtime validation / coercion
  • A stable contract between backend and frontend

Phases covered: 4 (Dashboard API) and 7 (RuneLite endpoint).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, RootModel


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class FlipMetricsResponse(BaseModel):
    """Full flip metrics for a single item (Phase 2 + 4)."""

    item_id: int
    item_name: str

    # Spread
    spread: int = 0
    spread_pct: float = 0.0

    # Pricing
    recommended_buy: int = 0
    recommended_sell: int = 0

    # Profit / ROI
    gross_profit: int = 0
    tax: int = 0
    net_profit: int = 0
    roi_pct: float = 0.0

    # Phase-2 fields
    gp_per_hour: float = 0.0
    estimated_hold_time: int = 0          # minutes
    fill_probability: float = 0.0
    spread_compression: float = 0.0
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    volume_delta: float = 0.0
    ma_signal: float = 0.0

    # Trend
    trend: str = "NEUTRAL"
    momentum: float = 0.0
    bb_position: Optional[float] = None
    vwap_1m: Optional[float] = None
    vwap_5m: Optional[float] = None
    vwap_30m: Optional[float] = None
    vwap_2h: Optional[float] = None

    # Historical
    win_rate: Optional[float] = None
    total_flips: int = 0
    avg_profit: Optional[float] = None

    # Scores
    score_spread: float = 0.0
    score_volume: float = 0.0
    score_freshness: float = 0.0
    score_trend: float = 0.0
    score_history: float = 0.0
    score_stability: float = 0.0
    score_ml: float = -1.0
    total_score: float = 0.0

    # Risk
    confidence: float = 1.0
    risk_score: float = 5.0
    risk_tier: str = "MEDIUM"          # LOW | MEDIUM | HIGH | VERY_HIGH
    stale_data: bool = False
    anomalous_spread: bool = False

    # Vetoes
    vetoed: bool = False
    veto_reasons: List[str] = Field(default_factory=list)

    reason: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": 4151,
                "item_name": "Abyssal whip",
                "net_profit": 78_000,
                "roi_pct": 0.8,
                "gp_per_hour": 4_680_000,
                "total_score": 72.4,
            }
        }


# ---------------------------------------------------------------------------
# Opportunity list (GET /flips/top, GET /flips/filtered)
# ---------------------------------------------------------------------------

class FlipSummary(BaseModel):
    """Lightweight opportunity record returned by list endpoints."""

    item_id: int
    item_name: str
    name: Optional[str] = None
    buy: int                    # recommended buy price
    sell: int                   # recommended sell price
    margin: int                 # net_profit
    margin_after_tax: int = 0
    roi: float                  # roi_pct
    roi_pct: float = 0.0
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    liquidity_score: float = 0.0
    fill_probability: float = 0.0
    est_fill_time_minutes: float = 0.0
    trend_score: float = 0.0
    decay_penalty: float = 0.0
    risk_level: str = "MEDIUM"
    confidence_pct: float = 0.0
    qty_suggested: int = 0
    expected_profit_personal: int = 0
    risk_adjusted_gph_personal: float = 0.0
    final_score: float = 0.0
    score: float                # total_score
    risk: float                 # risk_score
    confidence: float
    volume_rating: str          # HIGH | MEDIUM | LOW
    estimated_hold_time: int    # minutes
    gp_per_hour: float
    trend: str
    vetoed: bool = False


class FlipsTopResponse(BaseModel):
    """Response for GET /flips/top and GET /flips/top5."""
    count: int
    generated_at: datetime
    flips: List[FlipSummary]
    cache_ts: Optional[datetime] = None
    cache_age_seconds: Optional[int] = None
    profile_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Filter request (GET /flips/filtered)
# ---------------------------------------------------------------------------

class FlipFilterRequest(BaseModel):
    """Query-parameter equivalent — used for POST-based filtering."""
    min_roi: float = 0.0
    max_risk: float = 10.0
    min_volume_rating: str = "LOW"   # LOW | MEDIUM | HIGH
    min_price: int = 0
    max_price: int = 0               # 0 = no limit
    sort_by: str = "score"           # score | roi | gp_per_hour
    limit: int = 20


# ---------------------------------------------------------------------------
# Portfolio (Phase 5)
# ---------------------------------------------------------------------------

class SlotAllocationResponse(BaseModel):
    slot: int
    item_id: int
    item_name: str
    buy_price: int
    sell_price: int
    quantity: int
    investment: int
    expected_net_profit: int
    expected_roi_pct: float
    estimated_hold_minutes: int
    gp_per_hour: float
    risk_score: float
    confidence: float
    stop_loss_price: int
    kelly_fraction: float
    score: float
    reason: str = ""


class PortfolioAllocationResponse(BaseModel):
    capital: int
    ge_slots: int
    allocated_capital: int
    reserved_capital: int
    slots_used: int
    total_expected_profit: int
    total_expected_gp_per_hour: float
    portfolio_roi_pct: float
    warnings: List[str]
    slots: List[SlotAllocationResponse]


class OptimizePortfolioRequest(BaseModel):
    capital: int
    ge_slots: int = 8
    min_score: float = 45.0
    risk_tolerance: str = "MEDIUM"


# ---------------------------------------------------------------------------
# RuneLite endpoint (Phase 7)
# ---------------------------------------------------------------------------

class RuneLiteFlip(BaseModel):
    """Minimal payload optimised for RuneLite plugin bandwidth."""
    id: int             = Field(alias="item_id")
    n: str              = Field(alias="item_name")
    b: int              = Field(alias="recommended_buy")
    s: int              = Field(alias="recommended_sell")
    p: int              = Field(alias="net_profit")
    r: float            = Field(alias="roi_pct")
    sc: float           = Field(alias="total_score")
    c: float            = Field(alias="confidence_pct")
    rl: str             = Field(alias="risk_level")

    class Config:
        populate_by_name = True


class RuneLiteTop5Response(BaseModel):
    """Response for GET /flips/top5 — optimised for plugin latency."""
    ts: int             # Unix timestamp
    flips: List[RuneLiteFlip]
    cache_ts: Optional[datetime] = None
    cache_age_seconds: Optional[int] = None
    profile_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Health check (Phase 8)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    db: str = "ok"
    background_tasks: int = 0
    uptime_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SettingsResponse(BaseModel):
    """Merged settings (stored + defaults)."""
    settings: Dict[str, Any]


class UpdateSettingsRequest(RootModel[Dict[str, Any]]):
    """Partial settings update — only provided keys are updated."""
    pass

    class Config:
        # Allow direct dict-like access
        json_schema_extra = {
            "example": {"position_auto_archive_days": 7, "risk_tolerance": "MEDIUM"}
        }
