"""
backend.domain.enums — All enumerations used across the platform.

Keep this module import-clean (stdlib only).
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Risk profiles (user-selectable)
# ---------------------------------------------------------------------------

class RiskProfile(str, Enum):
    """
    User-level risk preference.  Affects scoring weights, portfolio
    position caps, alert sensitivity, and minimum confidence thresholds.
    """
    CONSERVATIVE = "conservative"
    BALANCED     = "balanced"
    AGGRESSIVE   = "aggressive"

    @property
    def score_weight_overrides(self) -> dict:
        """
        Returns per-component weight multipliers applied on top of defaults.
        1.0 = unchanged.  Values >1 increase a component's influence.
        """
        _OVERRIDES = {
            "conservative": {
                "stability":         1.4,
                "risk":             -1.3,   # negative = penalty amplifier
                "fill_probability":  1.2,
                "volume":            1.2,
                "trend":             0.8,
                "margin":            0.7,
            },
            "balanced": {
                # All weights unchanged
            },
            "aggressive": {
                "margin":            1.4,
                "trend":             1.3,
                "risk":             -0.7,   # smaller risk penalty
                "stability":         0.8,
                "fill_probability":  0.9,
            },
        }
        return _OVERRIDES.get(self.value, {})

    @property
    def min_confidence(self) -> float:
        """Minimum confidence_pct to show item in default feed."""
        return {"conservative": 70.0, "balanced": 50.0, "aggressive": 30.0}[self.value]

    @property
    def max_risk_score(self) -> float:
        """Maximum risk_score (0-10) before item is filtered out."""
        return {"conservative": 4.0, "balanced": 7.0, "aggressive": 10.0}[self.value]

    @property
    def position_cap_pct(self) -> float:
        """Max fraction of capital in a single item."""
        return {"conservative": 0.15, "balanced": 0.25, "aggressive": 0.40}[self.value]

    @property
    def alert_margin_threshold_gp(self) -> int:
        """Default margin alert threshold in GP."""
        return {"conservative": 30_000, "balanced": 50_000, "aggressive": 100_000}[self.value]


# ---------------------------------------------------------------------------
# Risk levels (computed per item)
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW       = "LOW"
    MEDIUM    = "MEDIUM"
    HIGH      = "HIGH"
    VERY_HIGH = "VERY_HIGH"


# ---------------------------------------------------------------------------
# Trend direction
# ---------------------------------------------------------------------------

class TrendDirection(str, Enum):
    STRONG_UP   = "STRONG_UP"
    UP          = "UP"
    NEUTRAL     = "NEUTRAL"
    DOWN        = "DOWN"
    STRONG_DOWN = "STRONG_DOWN"


# ---------------------------------------------------------------------------
# Flip outcome status
# ---------------------------------------------------------------------------

class FlipStatus(str, Enum):
    OPEN      = "open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PARTIAL   = "partial"


# ---------------------------------------------------------------------------
# Volume rating (display only)
# ---------------------------------------------------------------------------

class VolumeRating(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


# ---------------------------------------------------------------------------
# Strategy mode (PR9) — governs core/spice slot allocation
# ---------------------------------------------------------------------------

class StrategyMode(str, Enum):
    """
    Controls how GE slots are allocated across risk buckets.

    steady:       All slots from core bucket (high-certainty flips).
    steady_spice: 7 core slots + 1 spice slot (default personal mode).
    spice_only:   All slots from spice bucket (high-upside, lower certainty).
    """
    STEADY       = "steady"
    STEADY_SPICE = "steady_spice"
    SPICE_ONLY   = "spice_only"
