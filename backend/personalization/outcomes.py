"""
backend.personalization.outcomes — Flip outcome storage and retrieval.

Records a feature snapshot at decision time alongside the eventual realized
outcome so that calibration and affinity modules have data to learn from.

Collection: ``flip_outcomes``
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# MongoDB collection name
COLLECTION = "flip_outcomes"


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def open_flip(
    db,
    user_id: str,
    item_id: int,
    item_name: str,
    buy_target: int,
    sell_target: int,
    qty: int,
    expected_profit: int,
    est_hold_minutes: float,
    risk_profile: str,
    feature_snapshot: Dict[str, Any],
) -> str:
    """
    Insert a new open flip outcome record.

    Returns the inserted document's string ID.
    """
    doc = {
        "user_id":   user_id,
        "item_id":   item_id,
        "item_name": item_name,
        "ts_open":   datetime.utcnow(),
        "ts_close":  None,
        "qty":       qty,
        "buy_target":       buy_target,
        "buy_filled_avg":   0,
        "sell_target":      sell_target,
        "sell_filled_avg":  0,
        "expected_profit_at_open":    expected_profit,
        "realized_profit":            0,
        "est_hold_minutes_at_open":   est_hold_minutes,
        "realized_hold_minutes":      None,
        "status":             "open",
        "risk_profile_used":  risk_profile,
        "feature_snapshot_at_open": feature_snapshot,
    }
    result = db.db[COLLECTION].insert_one(doc)
    return str(result.inserted_id)


def close_flip(
    db,
    outcome_id: str,
    buy_filled_avg: int,
    sell_filled_avg: int,
    realized_profit: int,
    realized_hold_minutes: float,
    status: str = "completed",
) -> bool:
    """
    Mark an open flip as completed/cancelled/partial and record realized outcome.

    Returns True if a document was updated.
    """
    from bson import ObjectId
    result = db.db[COLLECTION].update_one(
        {"_id": ObjectId(outcome_id)},
        {"$set": {
            "ts_close":             datetime.utcnow(),
            "buy_filled_avg":       buy_filled_avg,
            "sell_filled_avg":      sell_filled_avg,
            "realized_profit":      realized_profit,
            "realized_hold_minutes": realized_hold_minutes,
            "status":               status,
        }},
    )
    return result.modified_count > 0


def get_user_outcomes(
    db,
    user_id: str,
    status: Optional[str] = "completed",
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Return recent flip outcomes for a user.

    Args:
        status: Filter by status ("open" / "completed" / "cancelled" / None = all).
        limit:  Maximum records to return (most-recent first).
    """
    query: Dict[str, Any] = {"user_id": user_id}
    if status is not None:
        query["status"] = status
    cursor = (
        db.db[COLLECTION]
        .find(query, {"_id": 0})
        .sort("ts_open", -1)
        .limit(limit)
    )
    return list(cursor)


def get_outcomes_for_item(
    db,
    user_id: str,
    item_id: int,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Return completed outcomes for a specific item."""
    cursor = (
        db.db[COLLECTION]
        .find(
            {"user_id": user_id, "item_id": item_id, "status": "completed"},
            {"_id": 0},
        )
        .sort("ts_close", -1)
        .limit(limit)
    )
    return list(cursor)


# ---------------------------------------------------------------------------
# Index definitions (called from database.init_db)
# ---------------------------------------------------------------------------

def ensure_indexes(db) -> None:
    col = db.db[COLLECTION]
    col.create_index([("user_id", 1), ("ts_open", -1)])
    col.create_index([("user_id", 1), ("item_id", 1), ("status", 1)])
    col.create_index([("user_id", 1), ("status", 1)])
    logger.debug("flip_outcomes indexes ensured")
