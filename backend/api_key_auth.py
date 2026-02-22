"""
API-key authentication helpers for plugin-facing endpoints.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

from backend.database import get_db


def hash_api_key(raw_api_key: str) -> str:
    return hashlib.sha256(raw_api_key.encode("utf-8")).hexdigest()


def resolve_api_key_owner(raw_api_key: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Validate an API key and return (key_hash, user_id?) if found.

    Resolution order:
    1) api_keys collection (key_hash, enabled!=False)
    2) users collection (api_key_hash field)
    """
    if not raw_api_key:
        return None
    key_hash = hash_api_key(raw_api_key)

    db = get_db()
    try:
        mongo = db.db if hasattr(db, "db") else db._db
        key_doc = mongo["api_keys"].find_one(
            {"key_hash": key_hash, "enabled": {"$ne": False}},
            {"user_id": 1},
        )
        if key_doc:
            user_id = key_doc.get("user_id")
            return key_hash, (str(user_id) if user_id is not None else None)

        user_doc = mongo["users"].find_one({"api_key_hash": key_hash}, {"_id": 1})
        if user_doc:
            return key_hash, str(user_doc.get("_id"))
        return None
    finally:
        db.close()

