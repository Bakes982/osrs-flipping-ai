"""
backend.personalization.calibration — Per-user calibration multipliers.

Computes:
    profit_multiplier_user = median(realized_profit / expected_profit)
    hold_multiplier_user   = median(realized_hold   / est_hold)

These are stored on the user record and loaded into UserContext at
request time so every scoring call is automatically personalised.

Calibration should run periodically (e.g., daily) via a background task,
NOT on every API request.
"""

from __future__ import annotations

import logging
import statistics
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Minimum number of completed flips required before calibration is applied.
MIN_FLIPS_FOR_CALIBRATION = 10

# Clamp multipliers to prevent outliers from distorting the model.
MULTIPLIER_MIN = 0.25
MULTIPLIER_MAX = 4.0


def compute_calibration(
    db,
    user_id: str,
) -> Dict[str, float]:
    """
    Compute profit and hold calibration multipliers for a user.

    Reads from the ``flip_outcomes`` collection (completed flips only).

    Returns dict with keys:
        profit_multiplier: float (default 1.0 if insufficient data)
        hold_multiplier:   float (default 1.0 if insufficient data)
        sample_size:       int
    """
    from backend.personalization.outcomes import get_user_outcomes

    outcomes = get_user_outcomes(db, user_id, status="completed", limit=1000)
    n = len(outcomes)

    if n < MIN_FLIPS_FOR_CALIBRATION:
        logger.debug(
            "calibration: user %s has only %d flips (need %d)",
            user_id, n, MIN_FLIPS_FOR_CALIBRATION,
        )
        return {"profit_multiplier": 1.0, "hold_multiplier": 1.0, "sample_size": n}

    profit_ratios = []
    hold_ratios   = []

    for o in outcomes:
        expected = o.get("expected_profit_at_open", 0)
        realized = o.get("realized_profit", 0)
        est_hold = o.get("est_hold_minutes_at_open")
        real_hold = o.get("realized_hold_minutes")

        if expected and expected != 0:
            ratio = realized / expected
            if MULTIPLIER_MIN <= ratio <= MULTIPLIER_MAX:
                profit_ratios.append(ratio)

        if est_hold and est_hold > 0 and real_hold and real_hold > 0:
            ratio = real_hold / est_hold
            if MULTIPLIER_MIN <= ratio <= MULTIPLIER_MAX:
                hold_ratios.append(ratio)

    profit_mult = statistics.median(profit_ratios) if profit_ratios else 1.0
    hold_mult   = statistics.median(hold_ratios)   if hold_ratios   else 1.0

    profit_mult = _clamp(profit_mult)
    hold_mult   = _clamp(hold_mult)

    logger.info(
        "calibration: user=%s n=%d profit_mult=%.3f hold_mult=%.3f",
        user_id, n, profit_mult, hold_mult,
    )
    return {
        "profit_multiplier": profit_mult,
        "hold_multiplier":   hold_mult,
        "sample_size":       n,
    }


def save_calibration(db, user_id: str, calibration: Dict[str, float]) -> None:
    """Persist calibration multipliers to the user document."""
    db.db["users"].update_one(
        {"_id": user_id},
        {"$set": {
            "profit_multiplier": calibration["profit_multiplier"],
            "hold_multiplier":   calibration["hold_multiplier"],
            "calibration_sample_size": calibration.get("sample_size", 0),
        }},
        upsert=False,
    )


def refresh_all_users(db) -> int:
    """
    Background task: recompute calibration for every user who has enough data.
    Returns count of users updated.
    """
    users = list(db.db["users"].find({}, {"_id": 1}))
    updated = 0
    for u in users:
        uid = str(u["_id"])
        try:
            cal = compute_calibration(db, uid)
            if cal["sample_size"] >= MIN_FLIPS_FOR_CALIBRATION:
                save_calibration(db, uid, cal)
                updated += 1
        except Exception as exc:
            logger.warning("calibration: failed for user %s: %s", uid, exc)
    logger.info("calibration: refreshed %d / %d users", updated, len(users))
    return updated


def _clamp(v: float) -> float:
    return max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, v))
