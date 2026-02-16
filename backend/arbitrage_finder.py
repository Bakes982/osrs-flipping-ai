"""
Arbitrage Finder – Set unpacking & decanting for risk-free profits.

Most flipping scripts only look at buy-low/sell-high on a single item ID.
This module finds price discrepancies between:
1. Item Sets vs their individual pieces (buy set -> unpack -> sell pieces)
2. Potion doses (buy 3-dose -> decant to 4-dose at Bob Barter)

These are near-zero-risk trades because execution is instant.
"""

import logging
import time
from typing import Dict, List, Optional

import httpx

from backend.smart_pricer import GE_TAX_RATE, GE_TAX_CAP

logger = logging.getLogger(__name__)

WIKI_BASE = "https://prices.runescape.wiki/api/v1/osrs"
USER_AGENT = "OSRS-AI-Flipper v2.0 - Discord: bakes982"
HEADERS = {"User-Agent": USER_AGENT}


def _ge_tax(sell_price: int) -> int:
    """Calculate GE tax: 2% capped at 5M per item."""
    if sell_price < 100:
        return 0
    return min(int(sell_price * GE_TAX_RATE), GE_TAX_CAP)


# ---------------------------------------------------------------------------
# Set definitions: set_item_id -> list of piece item IDs
# These are sets you can buy as a single item and unpack via the GE interface
# ---------------------------------------------------------------------------
ITEM_SETS = {
    # Barrows sets
    "Dharok's armour set": {"id": 11846, "pieces": [4716, 4718, 4720, 4722]},
    "Ahrim's armour set": {"id": 11848, "pieces": [4708, 4710, 4712, 4714]},
    "Karil's armour set": {"id": 11850, "pieces": [4732, 4734, 4736, 4738]},
    "Guthan's armour set": {"id": 11852, "pieces": [4724, 4726, 4728, 4730]},
    "Torag's armour set": {"id": 11854, "pieces": [4745, 4747, 4749, 4751]},
    "Verac's armour set": {"id": 11856, "pieces": [4753, 4755, 4757, 4759]},

    # High-tier armor sets
    "Inquisitor's armour set": {"id": 24488, "pieces": [24419, 24420, 24421]},
    "Justiciar armour set": {"id": 22438, "pieces": [22326, 22327, 22328]},
    "Ancestral robes set": {"id": 21015, "pieces": [21018, 21021, 21024]},
    "Masori armour set (f)": {"id": 27241, "pieces": [27226, 27229, 27232]},

    # Dragon sets
    "Dragon armour set (lg)": {"id": 11834, "pieces": [1149, 1187, 3140, 4087]},
    "Dragon armour set (sk)": {"id": 11836, "pieces": [1149, 1187, 3140, 4585]},

    # Rune/Adamant sets
    "Rune armour set (lg)": {"id": 11838, "pieces": [1163, 1127, 1201, 1079]},
    "Rune armour set (sk)": {"id": 11840, "pieces": [1163, 1127, 1201, 1093]},
    "Adamant armour set (lg)": {"id": 11830, "pieces": [1161, 1123, 1199, 1073]},
    "Adamant armour set (sk)": {"id": 11832, "pieces": [1161, 1123, 1199, 1091]},

    # Gilded sets
    "Gilded armour set (lg)": {"id": 11858, "pieces": [3481, 3483, 3486, 3488]},
    "Gilded armour set (sk)": {"id": 11860, "pieces": [3481, 3483, 3486, 3485]},
}

# ---------------------------------------------------------------------------
# Potion decanting: buy cheaper dose, decant to more valuable dose
# Bob Barter (Herbs) at GE can decant for free
# ---------------------------------------------------------------------------
POTION_DECANTS = {
    # Super restore: 3-dose -> 4-dose (4 * 3-dose = 3 * 4-dose)
    "Super restore": {
        "source_id": 3028,  # Super restore (3)
        "target_id": 3024,  # Super restore (4)
        "source_doses": 3,
        "target_doses": 4,
    },
    "Prayer potion": {
        "source_id": 141,   # Prayer potion (3)
        "target_id": 139,   # Prayer potion (4)
        "source_doses": 3,
        "target_doses": 4,
    },
    "Saradomin brew": {
        "source_id": 6689,  # Saradomin brew (3)
        "target_id": 6685,  # Saradomin brew (4)
        "source_doses": 3,
        "target_doses": 4,
    },
    "Ranging potion": {
        "source_id": 171,   # Ranging potion (3)
        "target_id": 169,   # Ranging potion (4)
        "source_doses": 3,
        "target_doses": 4,
    },
    "Stamina potion": {
        "source_id": 12629, # Stamina potion (3)
        "target_id": 12625, # Stamina potion (4)
        "source_doses": 3,
        "target_doses": 4,
    },
}


class ArbitrageFinder:
    """
    Finds risk-free profits by comparing:
    1. Sets vs individual pieces (buy set -> unpack -> sell pieces)
    2. Potion doses via decanting (buy 3-dose -> decant to 4-dose)
    """

    def __init__(self):
        self._latest_prices: Dict = {}
        self._last_fetch: float = 0.0

    async def _fetch_prices(self) -> Dict:
        """Fetch latest prices, cache for 30 seconds."""
        now = time.time()
        if now - self._last_fetch < 30 and self._latest_prices:
            return self._latest_prices

        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=15.0) as client:
                resp = await client.get(f"{WIKI_BASE}/latest")
                resp.raise_for_status()
                self._latest_prices = resp.json().get("data", {})
                self._last_fetch = now
        except Exception as e:
            logger.error("ArbitrageFinder: failed to fetch prices: %s", e)

        return self._latest_prices

    def _fetch_prices_sync(self) -> Dict:
        """Synchronous price fetch for non-async contexts."""
        now = time.time()
        if now - self._last_fetch < 30 and self._latest_prices:
            return self._latest_prices

        try:
            with httpx.Client(headers=HEADERS, timeout=15.0) as client:
                resp = client.get(f"{WIKI_BASE}/latest")
                resp.raise_for_status()
                self._latest_prices = resp.json().get("data", {})
                self._last_fetch = now
        except Exception as e:
            logger.error("ArbitrageFinder: failed to fetch prices: %s", e)

        return self._latest_prices

    def find_set_arbitrage(self, prices: Optional[Dict] = None) -> List[Dict]:
        """
        Find profitable set unpacking opportunities.

        Strategy: Buy Set (insta-buy) -> Unpack at GE NPC -> Sell Pieces (insta-sell)
        Risk: Near zero. Instant execution.
        """
        if prices is None:
            prices = self._fetch_prices_sync()

        opportunities = []

        for name, data in ITEM_SETS.items():
            set_id = str(data["id"])
            piece_ids = [str(pid) for pid in data["pieces"]]

            # We INSTA-BUY the set (pay the 'high' price)
            set_data = prices.get(set_id, {})
            set_buy_price = set_data.get("high")
            if not set_buy_price:
                continue

            # We INSTA-SELL the pieces (get the 'low' price per piece)
            pieces_sell_total = 0
            piece_details = []
            valid = True

            for pid in piece_ids:
                p_data = prices.get(pid, {})
                piece_sell = p_data.get("low")
                if not piece_sell:
                    valid = False
                    break

                tax = _ge_tax(piece_sell)
                net = piece_sell - tax
                pieces_sell_total += net
                piece_details.append({
                    "id": int(pid),
                    "sell_price": piece_sell,
                    "tax": tax,
                    "net": net,
                })

            if not valid:
                continue

            # Also account for tax on buying the set (buyer doesn't pay tax,
            # but seller does - we're buying so no tax on purchase)
            profit = pieces_sell_total - set_buy_price
            roi = (profit / set_buy_price * 100) if set_buy_price > 0 else 0

            # Check freshness - are prices recent?
            set_high_time = set_data.get("highTime", 0)
            now = int(time.time())
            age_mins = (now - set_high_time) / 60 if set_high_time else 999

            if profit > 50_000:  # Only report if > 50k profit per set
                opportunities.append({
                    "type": "SET_UNPACK",
                    "name": name,
                    "action": "BUY SET -> UNPACK -> SELL PIECES",
                    "set_buy_price": set_buy_price,
                    "pieces_sell_total": pieces_sell_total,
                    "profit": int(profit),
                    "roi_pct": round(roi, 2),
                    "buy_limit": 8,  # Standard armor set limit
                    "max_profit_per_4h": int(profit * 8),
                    "pieces": piece_details,
                    "price_age_mins": round(age_mins, 1),
                    "risk": "VERY LOW",
                    "confidence": "HIGH" if age_mins < 15 else "MEDIUM" if age_mins < 60 else "LOW",
                })

            # Also check REVERSE: buy pieces -> pack into set -> sell set
            # Sometimes pieces are cheaper than the set
            pieces_buy_total = 0
            valid_reverse = True
            for pid in piece_ids:
                p_data = prices.get(pid, {})
                piece_buy = p_data.get("high")  # insta-buy the piece
                if not piece_buy:
                    valid_reverse = False
                    break
                pieces_buy_total += piece_buy

            if valid_reverse:
                set_sell = set_data.get("low")  # insta-sell the set
                if set_sell:
                    set_tax = _ge_tax(set_sell)
                    reverse_profit = (set_sell - set_tax) - pieces_buy_total

                    if reverse_profit > 50_000:
                        opportunities.append({
                            "type": "SET_PACK",
                            "name": name,
                            "action": "BUY PIECES -> PACK SET -> SELL SET",
                            "pieces_buy_total": pieces_buy_total,
                            "set_sell_price": set_sell,
                            "profit": int(reverse_profit),
                            "roi_pct": round(reverse_profit / pieces_buy_total * 100, 2) if pieces_buy_total > 0 else 0,
                            "buy_limit": 8,
                            "max_profit_per_4h": int(reverse_profit * 8),
                            "risk": "VERY LOW",
                            "confidence": "HIGH" if age_mins < 15 else "MEDIUM",
                        })

        # Sort by profit
        opportunities.sort(key=lambda x: x["profit"], reverse=True)
        return opportunities

    def find_decant_arbitrage(self, prices: Optional[Dict] = None) -> List[Dict]:
        """
        Find profitable potion decanting opportunities.

        Strategy: Buy 3-dose -> Decant at Bob Barter (free) -> Sell 4-dose
        Conversion: 4 x 3-dose potions = 3 x 4-dose potions (12 doses each way)
        """
        if prices is None:
            prices = self._fetch_prices_sync()

        opportunities = []

        for name, data in POTION_DECANTS.items():
            src_id = str(data["source_id"])
            tgt_id = str(data["target_id"])
            src_doses = data["source_doses"]
            tgt_doses = data["target_doses"]

            src_data = prices.get(src_id, {})
            tgt_data = prices.get(tgt_id, {})

            # Buy source (insta-buy = 'high'), sell target (insta-sell = 'low')
            src_buy = src_data.get("high")
            tgt_sell = tgt_data.get("low")

            if not src_buy or not tgt_sell:
                continue

            # Conversion ratio: to get N target doses, need N * (target_doses/source_doses) source potions
            # For 3->4: buy 4 x 3-dose = 12 doses, get 3 x 4-dose = 12 doses
            lcm_doses = (src_doses * tgt_doses)  # Least common multiple for simple cases
            src_count = tgt_doses  # How many source potions to buy
            tgt_count = src_doses  # How many target potions you get

            cost = src_buy * src_count
            revenue = sum(tgt_sell - _ge_tax(tgt_sell) for _ in range(tgt_count))
            profit = revenue - cost

            if profit > 5_000:  # Potions have smaller margins
                roi = (profit / cost * 100) if cost > 0 else 0
                opportunities.append({
                    "type": "DECANT",
                    "name": f"{name} ({src_doses}-dose -> {tgt_doses}-dose)",
                    "action": f"BUY {src_count}x {src_doses}-dose -> DECANT -> SELL {tgt_count}x {tgt_doses}-dose",
                    "source_buy_each": src_buy,
                    "target_sell_each": tgt_sell,
                    "cost": cost,
                    "revenue": int(revenue),
                    "profit": int(profit),
                    "roi_pct": round(roi, 2),
                    "risk": "ZERO",
                    "confidence": "HIGH",
                })

        opportunities.sort(key=lambda x: x["profit"], reverse=True)
        return opportunities

    def find_all_arbitrage(self, prices: Optional[Dict] = None) -> List[Dict]:
        """Find all arbitrage opportunities (sets + decanting)."""
        if prices is None:
            prices = self._fetch_prices_sync()

        results = []
        results.extend(self.find_set_arbitrage(prices))
        results.extend(self.find_decant_arbitrage(prices))
        results.sort(key=lambda x: x["profit"], reverse=True)
        return results
