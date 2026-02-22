"""
backend.analytics.trade_plan — Single source of truth for trade plan sizing.

build_trade_plan() is deterministic and free of I/O.  It converts raw flip
metrics into a human-readable plan that the dashboard, RuneLite plugin, and
dump alerts can render directly.
"""

from __future__ import annotations

import math
from typing import Optional


def build_trade_plan(
    *,
    buy_price: int,
    sell_price: int,
    item_limit: Optional[int],
    liquidity_score: Optional[float],   # 0..100 (or None → treated as 50)
    risk_profile_position_cap_pct: float,  # 0..1
    capital_gp: int,
    ge_tax_rate: float,
    ge_tax_cap: Optional[int],
    ge_tax_free_below: Optional[int],
) -> dict:
    """Compute a deterministic trade plan from flip metrics.

    Parameters
    ----------
    buy_price:
        Recommended buy price (GP).
    sell_price:
        Recommended sell price (GP).
    item_limit:
        GE 4-hour buy limit for the item (None or 0 means uncapped).
    liquidity_score:
        Liquidity score on a 0–100 scale; None defaults to 50.
    risk_profile_position_cap_pct:
        Maximum fraction of ``capital_gp`` to invest in one item (0–1).
    capital_gp:
        Total available capital in GP.
    ge_tax_rate:
        Fractional GE tax rate (e.g. 0.02 for 2 %).
    ge_tax_cap:
        Maximum GE tax per item sold (None = no cap).
    ge_tax_free_below:
        Items with sell_price strictly below this threshold are tax-free.
        None means all items are taxed.

    Returns
    -------
    dict with keys:
        buy_price, sell_price, qty_to_buy, profit_per_item,
        total_profit, max_invest_gp
    """
    # 1) Maximum investment for this position
    max_invest_gp = math.floor(capital_gp * risk_profile_position_cap_pct)

    # 2) Raw quantity cap from capital
    qty_cap = math.floor(max_invest_gp / max(buy_price, 1))

    # 3) Respect the GE 4-hour item buy limit
    if item_limit is not None and item_limit > 0:
        qty_cap = min(qty_cap, item_limit)

    # 4) Liquidity throttle — scale back when the market is thin
    liq_norm = max(0.0, min((liquidity_score if liquidity_score is not None else 50.0) / 100.0, 1.0))
    qty_to_buy = math.floor(qty_cap * (0.5 + 0.5 * liq_norm))

    # 5) Profit per item after tax
    if ge_tax_free_below is not None and sell_price < ge_tax_free_below:
        tax = 0
    else:
        tax = math.floor(sell_price * ge_tax_rate)
    if ge_tax_cap is not None:
        tax = min(tax, ge_tax_cap)
    profit_per_item = sell_price - buy_price - tax

    # 6) Guard: no trade if negative profit or zero quantity
    if profit_per_item <= 0 or qty_to_buy <= 0:
        qty_to_buy = 0
        total_profit = 0
        profit_per_item = max(profit_per_item, 0)
    else:
        total_profit = profit_per_item * qty_to_buy

    return {
        "buy_price": int(buy_price),
        "sell_price": int(sell_price),
        "qty_to_buy": int(qty_to_buy),
        "profit_per_item": int(profit_per_item),
        "total_profit": int(total_profit),
        "max_invest_gp": int(max_invest_gp),
    }
