from __future__ import annotations

from backend.logic.guidance_engine import compute_guidance_state


def test_buying_exit_on_low_margin():
    trade = {"state": "BUYING", "buy_target": 101_000, "sell_target": 100_500}
    market = {"low": 100_000, "high": 100_200}
    assert compute_guidance_state(trade, market) == "EXIT"


def test_buying_adjust_when_fill_gap_large():
    trade = {"state": "BUYING", "buy_target": 105_000, "sell_target": 110_000}
    market = {"low": 100_000, "high": 101_000}
    assert compute_guidance_state(trade, market) == "ADJUST"


def test_selling_watch_on_thin_spread():
    trade = {"state": "SELLING", "buy_price": 100_000, "sell_target": 101_000}
    market = {"high": 100_800, "low": 100_600}
    assert compute_guidance_state(trade, market) == "WATCH"


def test_holding_exit_on_stop_loss():
    trade = {"state": "HOLDING", "buy_price": 100_000, "stop_loss_pct": 2.0}
    market = {"low": 97_500, "high": 98_000}
    assert compute_guidance_state(trade, market) == "EXIT"

