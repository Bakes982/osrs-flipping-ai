"""Dumps candidate endpoints served from Redis cache."""

import json

from fastapi import APIRouter, HTTPException

from backend.cache import get_redis

router = APIRouter(prefix="/api/dumps", tags=["dumps"])


@router.get("")
async def list_dumps():
    redis = get_redis()
    raw = redis.get("dumps:top")

    if not raw:
        return {"generated_at": None, "count": 0, "items": []}

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    try:
        return json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invalid cached dumps payload: {exc}")
