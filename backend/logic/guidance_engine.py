from __future__ import annotations

from backend import config


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compute_guidance_state(trade: dict, market: dict) -> str:
    """
    Compute position guidance state for a trade.

    Args:
        trade: strategy_trades document
        market: {"high": int, "low": int}
    """
    state = str(trade.get("state") or "").upper()
    high = _to_float(market.get("high"))
    low = _to_float(market.get("low"))
    buy_target = _to_float(trade.get("buy_target"))
    sell_target = _to_float(trade.get("sell_target"))
    buy_price = _to_float(trade.get("buy_price"))

    if low <= 0 and high <= 0:
        return "HEALTHY"

    if state == "BUYING":
        if low <= 0:
            return "HEALTHY"
        margin = sell_target - low
        fill_gap_pct = ((buy_target - low) / low) * 100

        if margin < config.MIN_MARGIN_GP:
            return "EXIT"
        if fill_gap_pct <= 1.0:
            return "HEALTHY"
        if fill_gap_pct > config.BUY_ADJUST_PCT:
            return "ADJUST"
        return "WATCH"

    if state == "SELLING":
        if high <= 0:
            return "HEALTHY"
        spread = high - buy_price
        undercut_pct = ((sell_target - high) / high) * 100

        if spread <= 0:
            return "EXIT"
        if undercut_pct > config.SELL_UNDERCUT_PCT:
            return "ADJUST"
        if spread < config.MIN_MARGIN_GP:
            return "WATCH"
        return "HEALTHY"

    if state == "HOLDING":
        if buy_price <= 0 or low <= 0:
            return "HEALTHY"
        drawdown_pct = ((buy_price - low) / buy_price) * 100
        stop = _to_float(trade.get("stop_loss_pct"), 2.0)

        if drawdown_pct >= stop:
            return "EXIT"
        if drawdown_pct >= stop * config.WATCH_DRAW_DOWN_RATIO:
            return "WATCH"
        return "HEALTHY"

    return "HEALTHY"

