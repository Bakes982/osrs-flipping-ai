from __future__ import annotations

from unittest.mock import MagicMock

from backend.ml.predictor import Predictor


def test_predictor_load_models_only_once_when_no_models(monkeypatch):
    predictor = Predictor()

    calls = {"load": 0}

    def _fake_load_models():
        calls["load"] += 1
        return 0

    monkeypatch.setattr(predictor.forecaster, "load_models", _fake_load_models)
    monkeypatch.setattr(
        predictor,
        "_get_features",
        lambda *_args, **_kwargs: {"current_price": 1_000_000, "spread_pct": 1.0},
    )
    monkeypatch.setattr(
        predictor.forecaster,
        "predict",
        lambda *_args, **_kwargs: {
            "1m": {"buy": 1, "sell": 1, "direction": "flat", "confidence": 0.5},
            "5m": {"buy": 1, "sell": 1, "direction": "flat", "confidence": 0.5},
            "30m": {"buy": 1, "sell": 1, "direction": "flat", "confidence": 0.5},
            "2h": {"buy": 1, "sell": 1, "direction": "flat", "confidence": 0.5},
            "8h": {"buy": 1, "sell": 1, "direction": "flat", "confidence": 0.5},
            "24h": {"buy": 1, "sell": 1, "direction": "flat", "confidence": 0.5},
        },
    )

    predictor.predict_item(item_id=4151, snapshots=[MagicMock()], flips=[], save_to_db=False)
    predictor.predict_item(item_id=4151, snapshots=[MagicMock()], flips=[], save_to_db=False)

    assert calls["load"] == 1

