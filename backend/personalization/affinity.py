"""
backend.personalization.affinity — Per-item and per-category affinity boosts.

Affinity measures how well a user personally performs on a given item or
item category compared to the model's expectation.

Boost formula (per item)::

    score_boost = clamp(
        (median_realized_gph / model_gph - 1.0) * ITEM_BOOST_SCALE,
        MIN_BOOST, MAX_BOOST
    )

Boosts are stored on the user document and refreshed by a background task.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)

# Minimum completed flips for a reliable per-item boost
MIN_FLIPS_PER_ITEM = 5

# Boost clamping (score points added/subtracted)
ITEM_BOOST_SCALE = 15.0
MIN_BOOST = -10.0
MAX_BOOST  = 10.0


def compute_item_affinity(
    db,
    user_id: str,
) -> Dict[int, float]:
    """
    Compute per-item score boosts based on how user historically performs
    on each item versus model expectations.

    Returns dict mapping item_id (int) → boost (float in [MIN_BOOST, MAX_BOOST]).
    """
    from backend.personalization.outcomes import get_user_outcomes

    outcomes = get_user_outcomes(db, user_id, status="completed", limit=2000)

    # Group by item_id
    by_item: Dict[int, List[dict]] = defaultdict(list)
    for o in outcomes:
        by_item[o["item_id"]].append(o)

    affinity: Dict[int, float] = {}
    for item_id, flips in by_item.items():
        if len(flips) < MIN_FLIPS_PER_ITEM:
            continue
        try:
            boost = _item_boost(flips)
            if boost != 0.0:
                affinity[item_id] = boost
        except Exception as exc:
            logger.debug("affinity: item %d error: %s", item_id, exc)

    return affinity


def _item_boost(flips: List[dict]) -> float:
    """Compute boost for one item from its flip history."""
    ratios = []
    for f in flips:
        realized = f.get("realized_profit", 0)
        expected = f.get("expected_profit_at_open", 0)
        if not expected:
            continue
        ratios.append(realized / expected)

    if len(ratios) < MIN_FLIPS_PER_ITEM:
        return 0.0

    median_ratio = statistics.median(ratios)
    # 1.0 = exactly as expected → 0 boost
    boost = (median_ratio - 1.0) * ITEM_BOOST_SCALE
    return max(MIN_BOOST, min(MAX_BOOST, boost))


def save_affinity(db, user_id: str, item_affinity: Dict[int, float]) -> None:
    """Persist affinity map to the user document."""
    # MongoDB requires string keys for nested dicts
    str_affinity = {str(k): v for k, v in item_affinity.items()}
    db.db["users"].update_one(
        {"_id": user_id},
        {"$set": {"item_affinity": str_affinity}},
        upsert=False,
    )


def refresh_all_users(db) -> int:
    """Recompute affinity for all users.  Returns count updated."""
    users = list(db.db["users"].find({}, {"_id": 1}))
    updated = 0
    for u in users:
        uid = str(u["_id"])
        try:
            aff = compute_item_affinity(db, uid)
            if aff:
                save_affinity(db, uid, aff)
                updated += 1
        except Exception as exc:
            logger.warning("affinity: failed for user %s: %s", uid, exc)
    return updated
