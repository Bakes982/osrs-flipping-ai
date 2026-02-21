"""
OSRS Flipping AI — Feature engineering interface.

Thin adapter over ``backend.ml.feature_engine.FeatureEngine`` that exposes
a clean, typed API for the rest of the system.  All callers should import
from this module rather than from the ``ml`` sub-package directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_features(
    item_id: int,
    snapshots: Optional[List[Any]] = None,
    flip_history: Optional[List[Any]] = None,
) -> Dict[str, float]:
    """Compute the full ML feature vector for ``item_id``.

    Parameters
    ----------
    item_id:
        OSRS item ID.
    snapshots:
        Optional pre-loaded price snapshots.  If ``None``, they are fetched
        from the database.
    flip_history:
        Optional pre-loaded flip history.  If ``None``, it is fetched from
        the database.

    Returns
    -------
    dict
        Feature name → float value, or an empty dict if computation fails.
    """
    try:
        from backend.ml.feature_engine import FeatureEngine

        engine = FeatureEngine()

        db = None
        if snapshots is None or flip_history is None:
            from backend.database import get_db, get_price_history, get_item_flips
            db = get_db()
            try:
                if snapshots is None:
                    snapshots = get_price_history(db, item_id, hours=4)
                if flip_history is None:
                    flip_history = get_item_flips(db, item_id, days=30)
            finally:
                db.close()

        if not snapshots:
            return {}

        return engine.compute_features(item_id, snapshots, flip_history or [])

    except Exception as exc:
        logger.warning("Feature computation failed for item %d: %s", item_id, exc)
        return {}


def compute_features_batch(
    item_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compute features for multiple items in one database session.

    Returns a dict of item_id → feature_vector.  Items that fail are
    silently omitted from the result.
    """
    from backend.database import get_db, get_price_history, get_item_flips

    results: Dict[int, Dict[str, float]] = {}
    db = get_db()
    try:
        from backend.ml.feature_engine import FeatureEngine
        engine = FeatureEngine()

        for item_id in item_ids:
            try:
                snaps = get_price_history(db, item_id, hours=4)
                flips = get_item_flips(db, item_id, days=30)
                if snaps:
                    results[item_id] = engine.compute_features(item_id, snaps, flips)
            except Exception as exc:
                logger.warning("Batch feature computation failed for item %d: %s", item_id, exc)
    finally:
        db.close()

    return results
