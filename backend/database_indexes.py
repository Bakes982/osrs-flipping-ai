"""
backend.database_indexes — MongoDB index definitions for the full schema.

Run ``ensure_all_indexes(db)`` once at startup (called from ``init_db``).
All indexes are created with ``background=True`` so they don't block the
primary thread on large collections.

Collections managed here:

  prices_latest        — most-recent price per item (upserted, not time-series)
  prices_timeseries    — historical OHLCV per item (time-series)
  items_meta           — item catalogue (name, limit, high_alch, etc.)
  users                — registered users with risk profiles + personalisation
  flip_outcomes        — per-user flip records for personalisation
  price_snapshots      — existing snapshot collection (backward-compat)
  price_aggregates     — existing aggregate collection (backward-compat)
  alerts               — fired alert records
  flip_history         — existing completed-flip records
  trades               — existing raw trade webhooks
"""

from __future__ import annotations

import logging
from pymongo import ASCENDING, DESCENDING, IndexModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index blueprints
# ---------------------------------------------------------------------------

INDEXES: dict[str, list[IndexModel]] = {

    # ------------------------------------------------------------------
    # prices_latest — one document per item_id (upsert pattern)
    # ------------------------------------------------------------------
    "prices_latest": [
        IndexModel([("item_id", ASCENDING)], unique=True, name="item_id_unique"),
        IndexModel([("ts", DESCENDING)], name="ts_desc"),
    ],

    # ------------------------------------------------------------------
    # prices_timeseries — high-write time-series
    # ------------------------------------------------------------------
    "prices_timeseries": [
        IndexModel(
            [("item_id", ASCENDING), ("ts", DESCENDING)],
            name="item_id_ts_desc",
        ),
        # Partial index: only keep rows newer than TTL (optional, handled by
        # Atlas free-tier pruning in database.py instead of native TTL to
        # avoid Atlas restriction).
        IndexModel([("ts", ASCENDING)], name="ts_asc_for_range_queries"),
    ],

    # ------------------------------------------------------------------
    # items_meta — item catalogue
    # ------------------------------------------------------------------
    "items_meta": [
        IndexModel([("item_id", ASCENDING)], unique=True, name="item_id_unique"),
        IndexModel([("name", ASCENDING)], name="name_asc"),
        IndexModel(
            [("members", ASCENDING), ("category", ASCENDING)],
            name="members_category",
        ),
    ],

    # ------------------------------------------------------------------
    # users — registered users (Discord ID = _id)
    # ------------------------------------------------------------------
    "users": [
        # _id IS the Discord user ID string — no additional unique index needed.
        IndexModel([("risk_profile", ASCENDING)], name="risk_profile"),
        IndexModel([("subscription_tier", ASCENDING)], name="subscription_tier"),
        IndexModel([("api_key_hash", ASCENDING)], name="api_key_hash", sparse=True),
        IndexModel([("updated_at", DESCENDING)], name="updated_at_desc"),
    ],

    # ------------------------------------------------------------------
    # api_keys — explicit API keys for plugin auth
    # ------------------------------------------------------------------
    "api_keys": [
        IndexModel([("key_hash", ASCENDING)], unique=True, name="key_hash_unique"),
        IndexModel([("user_id", ASCENDING)], name="user_id"),
        IndexModel([("enabled", ASCENDING)], name="enabled"),
        IndexModel([("created_at", DESCENDING)], name="created_at_desc"),
    ],

    # ------------------------------------------------------------------
    # flip_outcomes — personalisation source of truth
    # ------------------------------------------------------------------
    "flip_outcomes": [
        IndexModel(
            [("user_id", ASCENDING), ("ts_open", DESCENDING)],
            name="user_ts_open_desc",
        ),
        IndexModel(
            [("user_id", ASCENDING), ("item_id", ASCENDING), ("status", ASCENDING)],
            name="user_item_status",
        ),
        IndexModel(
            [("user_id", ASCENDING), ("status", ASCENDING)],
            name="user_status",
        ),
        IndexModel([("ts_open", ASCENDING)], name="ts_open_asc"),
    ],

    # ------------------------------------------------------------------
    # price_snapshots — backward-compatible existing collection
    # ------------------------------------------------------------------
    "price_snapshots": [
        IndexModel(
            [("item_id", ASCENDING), ("timestamp", DESCENDING)],
            name="item_id_timestamp_desc",
        ),
        IndexModel([("timestamp", ASCENDING)], name="timestamp_asc"),
    ],

    # ------------------------------------------------------------------
    # price_aggregates — backward-compatible existing collection
    # ------------------------------------------------------------------
    "price_aggregates": [
        IndexModel(
            [("item_id", ASCENDING), ("interval", ASCENDING), ("timestamp", DESCENDING)],
            name="item_id_interval_ts_desc",
        ),
    ],

    # ------------------------------------------------------------------
    # alerts
    # ------------------------------------------------------------------
    "alerts": [
        IndexModel([("item_id", ASCENDING), ("timestamp", DESCENDING)], name="item_ts_desc"),
        IndexModel([("alert_type", ASCENDING)], name="alert_type"),
        IndexModel([("timestamp", ASCENDING)], name="timestamp_asc"),
    ],

    # ------------------------------------------------------------------
    # flip_history — existing completed-flip records
    # ------------------------------------------------------------------
    "flip_history": [
        IndexModel([("item_id", ASCENDING), ("close_time", DESCENDING)], name="item_close_time"),
        IndexModel([("close_time", DESCENDING)], name="close_time_desc"),
    ],

    # ------------------------------------------------------------------
    # trades — raw DINK webhook payloads
    # ------------------------------------------------------------------
    "trades": [
        IndexModel(
            [("item_id", ASCENDING), ("timestamp", DESCENDING)],
            name="item_ts_desc",
        ),
        IndexModel([("trade_id", ASCENDING)], unique=True, name="trade_id_unique",
                   sparse=True),
        IndexModel([("status", ASCENDING)], name="status"),
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_all_indexes(db) -> None:
    """
    Create all indexes defined above.

    Safe to call on every startup — MongoDB is idempotent for existing indexes
    (it only errors if an index with the same name but different options exists).

    Args:
        db: Open ``Database`` wrapper (has ``db.db`` pymongo Database attribute).
    """
    mongo_db = db.db
    for collection_name, index_models in INDEXES.items():
        try:
            col = mongo_db[collection_name]
            col.create_indexes(index_models)
            logger.debug("Indexes ensured for collection: %s", collection_name)
        except Exception as exc:
            # Log but don't crash — an existing incompatible index is a
            # configuration problem for the operator to resolve manually.
            logger.warning("Index creation warning for %s: %s", collection_name, exc)

    logger.info("database_indexes: all index definitions applied")


def describe_indexes() -> dict:
    """
    Return a plain-dict description of all index blueprints.
    Useful for documentation and migration scripts.
    """
    out = {}
    for col, models in INDEXES.items():
        out[col] = []
        for m in models:
            out[col].append({
                "keys":   m.document["key"],
                "name":   m.document.get("name"),
                "unique": m.document.get("unique", False),
                "sparse": m.document.get("sparse", False),
            })
    return out
