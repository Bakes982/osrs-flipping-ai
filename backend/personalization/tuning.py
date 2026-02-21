"""
backend.personalization.tuning — Optional guarded weight adjustment.

When a user has enough flip history (>= TUNING_MIN_FLIPS) this module
can nudge the per-component scoring weights toward whatever empirically
works best for that user, subject to strict guardrails:

    1. Weights can deviate at most ±MAX_DEVIATION from the balanced default.
    2. Learning rate is bounded by MAX_LEARNING_RATE per refresh cycle.
    3. Tuning is disabled unless the user explicitly opts in.

This is intentionally conservative: the baseline balanced model should
already be strong; tuning is a small refinement, not a replacement.

NOTE: Chunk 2+ will wire this into the scoring pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict

from backend.analytics.scoring import _DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)

TUNING_MIN_FLIPS   = 50
MAX_DEVIATION      = 0.30   # weights can shift ±30 % from default
MAX_LEARNING_RATE  = 0.05   # max change per refresh cycle (5 %)


def compute_weight_adjustments(
    db,
    user_id: str,
) -> Dict[str, float]:
    """
    Compute per-component weight multipliers for a user.

    Returns dict mapping component name → multiplier (1.0 = unchanged).
    Returns all-1.0 dict if insufficient data or tuning is disabled.

    Full implementation deferred to Chunk 2.
    """
    from backend.personalization.outcomes import get_user_outcomes

    outcomes = get_user_outcomes(db, user_id, status="completed", limit=2000)
    if len(outcomes) < TUNING_MIN_FLIPS:
        logger.debug(
            "tuning: user %s has %d flips (need %d) — skipping",
            user_id, len(outcomes), TUNING_MIN_FLIPS,
        )
        return {k: 1.0 for k in _DEFAULT_WEIGHTS}

    # TODO (Chunk 2): implement gradient-based or correlation-based weight tuning.
    # Placeholder: return identity multipliers.
    return {k: 1.0 for k in _DEFAULT_WEIGHTS}


def save_weight_adjustments(db, user_id: str, adjustments: Dict[str, float]) -> None:
    """Persist weight adjustments to the user document."""
    db.db["users"].update_one(
        {"_id": user_id},
        {"$set": {"weight_adjustments": adjustments}},
        upsert=False,
    )
