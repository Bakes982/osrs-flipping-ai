#!/usr/bin/env python3
"""
OSRS Quantitative Market Analyzer
Advanced statistical analysis for GE flipping - Z-Scores, RSI, Volume Velocity
"""

import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics

# User-Agent for Wiki API
USER_AGENT = 'OSRS-AI-Flipper v2.0 - Discord: bakes982 - Contact: mike.baker982@hotmail.com'

# GE Tax: 2% capped at 5M per item
GE_TAX_RATE = 0.02
GE_TAX_CAP = 5_000_000

# Slippage factor for high-volume items
SLIPPAGE_FACTOR = 0.002  # 0.2%


class QuantAnalyzer:
    """Quantitative analysis engine for OSRS GE data"""

    def __init__(self):
        self.price_cache = {}
        self.timeseries_cache = {}
        self.mapping_cache = None
        self.last_fetch = None

    def fetch_latest_prices(self) -> Dict:
        """Fetch all current prices from Wiki API"""
        try:
            response = requests.get(
                "https://prices.runescape.wiki/api/v1/osrs/latest",
                headers={'User-Agent': USER_AGENT},
                timeout=15
            )
            data = response.json().get('data', {})
            self.price_cache = data
            self.last_fetch = datetime.now()
            return data
        except Exception as e:
            print(f"Error fetching latest prices: {e}")
            return self.price_cache

    def fetch_5m_prices(self) -> Dict:
        """
        Fetch 5-minute averaged prices - MORE ACCURATE than /latest
        This is what Flipping Copilot uses for price suggestions
        Returns avgHighPrice, avgLowPrice, highPriceVolume, lowPriceVolume
        """
        try:
            response = requests.get(
                "https://prices.runescape.wiki/api/v1/osrs/5m",
                headers={'User-Agent': USER_AGENT},
                timeout=15
            )
            data = response.json().get('data', {})
            return data
        except Exception as e:
            print(f"Error fetching 5m prices: {e}")
            return {}

    def fetch_1h_prices(self) -> Dict:
        """
        Fetch 1-hour averaged prices for trend analysis
        Returns avgHighPrice, avgLowPrice, highPriceVolume, lowPriceVolume
        """
        try:
            response = requests.get(
                "https://prices.runescape.wiki/api/v1/osrs/1h",
                headers={'User-Agent': USER_AGENT},
                timeout=15
            )
            data = response.json().get('data', {})
            return data
        except Exception as e:
            print(f"Error fetching 1h prices: {e}")
            return {}

    def get_accurate_prices(self, item_id: str) -> Dict:
        """
        Get the most accurate prices by combining instant and averaged data
        Prioritizes 5m averaged prices when available, falls back to instant
        """
        instant = self.price_cache.get(item_id, {})

        # Fetch 5m if not cached recently
        if not hasattr(self, '_5m_cache') or not self._5m_cache:
            self._5m_cache = self.fetch_5m_prices()

        avg_5m = self._5m_cache.get(item_id, {})

        # Use 5-minute averages when available (more reliable)
        high = avg_5m.get('avgHighPrice') or instant.get('high')
        low = avg_5m.get('avgLowPrice') or instant.get('low')

        # Volume from 5m data
        high_volume = avg_5m.get('highPriceVolume', 0)
        low_volume = avg_5m.get('lowPriceVolume', 0)
        total_volume = high_volume + low_volume

        # Freshness from instant data
        high_time = instant.get('highTime', 0)
        low_time = instant.get('lowTime', 0)

        import time
        now = int(time.time())
        high_age_mins = (now - high_time) // 60 if high_time else 999
        low_age_mins = (now - low_time) // 60 if low_time else 999

        # Confidence based on freshness and volume
        if high_age_mins < 5 and low_age_mins < 5 and total_volume > 10:
            confidence = "HIGH"
        elif high_age_mins < 30 and low_age_mins < 30 and total_volume > 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            'high': high,
            'low': low,
            'instant_high': instant.get('high'),
            'instant_low': instant.get('low'),
            'avg_high': avg_5m.get('avgHighPrice'),
            'avg_low': avg_5m.get('avgLowPrice'),
            'high_volume': high_volume,
            'low_volume': low_volume,
            'total_volume': total_volume,
            'high_age_mins': high_age_mins,
            'low_age_mins': low_age_mins,
            'confidence': confidence
        }

    def fetch_timeseries(self, timestep: str = '5m') -> Dict:
        """
        Fetch time-series data from Wiki API
        timestep: '5m' for 5-minute intervals, '1h' for hourly
        """
        try:
            response = requests.get(
                f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep={timestep}",
                headers={'User-Agent': USER_AGENT},
                timeout=15
            )
            data = response.json().get('data', {})
            self.timeseries_cache[timestep] = {
                'data': data,
                'fetched': datetime.now()
            }
            return data
        except Exception as e:
            print(f"Error fetching timeseries: {e}")
            return {}

    def fetch_item_mapping(self) -> Dict:
        """Fetch item ID to name mapping"""
        if self.mapping_cache:
            return self.mapping_cache
        try:
            response = requests.get(
                "https://prices.runescape.wiki/api/v1/osrs/mapping",
                headers={'User-Agent': USER_AGENT},
                timeout=15
            )
            items = response.json()
            self.mapping_cache = {str(item['id']): item for item in items}
            return self.mapping_cache
        except Exception as e:
            print(f"Error fetching mapping: {e}")
            return {}

    def calculate_tax(self, sell_price: int, quantity: int = 1) -> int:
        """Calculate GE tax (2% capped at 5M per item)"""
        tax_per_item = min(sell_price * GE_TAX_RATE, GE_TAX_CAP)
        return int(tax_per_item * quantity)

    def calculate_true_margin(self, high: int, low: int, quantity: int = 1) -> Dict:
        """
        Calculate the TRUE margin after tax and slippage
        Returns dict with gross margin, tax, slippage, and net margin
        """
        if not high or not low:
            return {'gross': 0, 'tax': 0, 'slippage': 0, 'net': 0, 'profitable': False}

        gross_margin = (high - low) * quantity
        tax = self.calculate_tax(high, quantity)
        slippage = int(high * SLIPPAGE_FACTOR * quantity)  # Expected slippage
        net_margin = gross_margin - tax - slippage

        return {
            'gross': gross_margin,
            'tax': tax,
            'slippage': slippage,
            'net': net_margin,
            'profitable': net_margin > 0,
            'margin_pct': (net_margin / (low * quantity) * 100) if low else 0
        }

    def calculate_z_score(self, current_price: int, prices_24h: List[int]) -> Optional[float]:
        """
        Calculate Z-Score: How many standard deviations from the 24h mean
        Z > 2.0 = Overbought (likely to crash)
        Z < -2.0 = Oversold (potential buy opportunity)
        """
        if not prices_24h or len(prices_24h) < 2:
            return None

        try:
            mean = statistics.mean(prices_24h)
            stdev = statistics.stdev(prices_24h)
            if stdev == 0:
                return 0
            return (current_price - mean) / stdev
        except:
            return None

    def calculate_rsi(self, prices: List[int], period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index (RSI)
        RSI > 70 = Overbought
        RSI < 30 = Oversold
        """
        if not prices or len(prices) < period + 1:
            return None

        try:
            gains = []
            losses = []

            for i in range(1, len(prices)):
                diff = prices[i] - prices[i-1]
                if diff > 0:
                    gains.append(diff)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(diff))

            # Use last 'period' values
            recent_gains = gains[-period:]
            recent_losses = losses[-period:]

            avg_gain = sum(recent_gains) / period
            avg_loss = sum(recent_losses) / period

            if avg_loss == 0:
                return 100  # No losses = maximum RSI

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return None

    def calculate_volume_velocity(self, current_volume: int, avg_24h_volume: int) -> Dict:
        """
        Calculate volume velocity compared to 24h average
        Returns multiplier and interpretation
        """
        if not avg_24h_volume or avg_24h_volume == 0:
            return {'velocity': 0, 'interpretation': 'Unknown', 'contested': False}

        velocity = (current_volume / avg_24h_volume) * 24  # Normalize to hourly rate

        if velocity > 3.0:
            interpretation = 'EXTREME - Heavy competition or panic'
            contested = True
        elif velocity > 2.0:
            interpretation = 'HIGH - Active trading, margins contested'
            contested = True
        elif velocity > 1.0:
            interpretation = 'NORMAL - Healthy activity'
            contested = False
        elif velocity > 0.5:
            interpretation = 'LOW - Illiquid, margins may be traps'
            contested = False
        else:
            interpretation = 'DEAD - Very illiquid, avoid'
            contested = False

        return {
            'velocity': velocity,
            'interpretation': interpretation,
            'contested': contested,
            'pct_of_avg': velocity * 100
        }

    def calculate_sma(self, prices: List[int], period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if not prices or len(prices) < period:
            return None
        return statistics.mean(prices[-period:])

    def analyze_trend(self, current_price: int, prices: List[int]) -> Dict:
        """
        Analyze price trend using multiple timeframes
        """
        sma_1h = self.calculate_sma(prices[-12:], 12) if len(prices) >= 12 else None  # 12 x 5min = 1hr
        sma_4h = self.calculate_sma(prices[-48:], 48) if len(prices) >= 48 else None  # 48 x 5min = 4hr
        sma_24h = self.calculate_sma(prices, len(prices)) if prices else None

        trend = 'NEUTRAL'
        if sma_1h and sma_4h:
            if current_price > sma_1h > sma_4h:
                trend = 'BULLISH'
            elif current_price < sma_1h < sma_4h:
                trend = 'BEARISH'
            elif current_price > sma_1h:
                trend = 'SHORT_BULLISH'
            elif current_price < sma_1h:
                trend = 'SHORT_BEARISH'

        return {
            'trend': trend,
            'sma_1h': sma_1h,
            'sma_4h': sma_4h,
            'sma_24h': sma_24h,
            'above_1h_sma': current_price > sma_1h if sma_1h else None,
            'above_4h_sma': current_price > sma_4h if sma_4h else None
        }

    def full_analysis(self, item_id: str, current_high: int = None, current_low: int = None) -> Dict:
        """
        Perform full quantitative analysis on an item
        """
        # Fetch fresh data if needed
        if not self.price_cache or not self.last_fetch or (datetime.now() - self.last_fetch).seconds > 60:
            self.fetch_latest_prices()

        # Get current prices
        item_data = self.price_cache.get(str(item_id), {})
        high = current_high or item_data.get('high')
        low = current_low or item_data.get('low')
        high_time = item_data.get('highTime')
        low_time = item_data.get('lowTime')

        # Get timeseries for historical analysis
        timeseries = self.fetch_timeseries('5m')
        item_history = timeseries.get(str(item_id), [])

        # Extract price history
        high_prices = [p.get('avgHighPrice') for p in item_history if p.get('avgHighPrice')]
        low_prices = [p.get('avgLowPrice') for p in item_history if p.get('avgLowPrice')]
        volumes = [p.get('highPriceVolume', 0) + p.get('lowPriceVolume', 0) for p in item_history]

        # Calculate metrics
        z_score_high = self.calculate_z_score(high, high_prices) if high else None
        z_score_low = self.calculate_z_score(low, low_prices) if low else None
        rsi = self.calculate_rsi(high_prices)

        avg_volume = statistics.mean(volumes) if volumes else 0
        current_volume = volumes[-1] if volumes else 0
        volume_data = self.calculate_volume_velocity(current_volume, avg_volume)

        margin_data = self.calculate_true_margin(high, low)
        trend_data = self.analyze_trend(high or 0, high_prices)

        # Risk assessment (1-10 scale)
        risk_score = self._calculate_risk_score(z_score_high, volume_data, rsi, margin_data)

        # Strategic verdict
        verdict = self._determine_verdict(z_score_high, z_score_low, rsi, volume_data, margin_data, trend_data)

        return {
            'item_id': item_id,
            'timestamp': datetime.now().isoformat(),
            'prices': {
                'high': high,
                'low': low,
                'high_time': high_time,
                'low_time': low_time
            },
            'margin': margin_data,
            'z_score': {
                'high': round(z_score_high, 2) if z_score_high else None,
                'low': round(z_score_low, 2) if z_score_low else None,
                'interpretation': self._interpret_z_score(z_score_high)
            },
            'rsi': {
                'value': round(rsi, 1) if rsi else None,
                'interpretation': self._interpret_rsi(rsi)
            },
            'volume': volume_data,
            'trend': trend_data,
            'risk_score': risk_score,
            'verdict': verdict,
            'data_points': len(high_prices)
        }

    def _interpret_z_score(self, z: Optional[float]) -> str:
        """Interpret Z-Score for trading"""
        if z is None:
            return 'Insufficient data'
        if z > 2.0:
            return 'OVERBOUGHT - Price spike, likely to crash'
        elif z > 1.5:
            return 'HIGH - Elevated price, risky entry'
        elif z < -2.0:
            return 'OVERSOLD - Price dump, potential buy'
        elif z < -1.5:
            return 'LOW - Below average, possible opportunity'
        else:
            return 'NORMAL - Price within typical range'

    def _interpret_rsi(self, rsi: Optional[float]) -> str:
        """Interpret RSI for trading"""
        if rsi is None:
            return 'Insufficient data'
        if rsi > 70:
            return 'OVERBOUGHT - Momentum exhausted'
        elif rsi > 60:
            return 'STRONG - Bullish momentum'
        elif rsi < 30:
            return 'OVERSOLD - Potential reversal up'
        elif rsi < 40:
            return 'WEAK - Bearish momentum'
        else:
            return 'NEUTRAL - No strong momentum'

    def _calculate_risk_score(self, z_score, volume_data, rsi, margin_data) -> int:
        """Calculate overall risk score 1-10 (10 = highest risk)"""
        risk = 5  # Base risk

        # Z-Score risk
        if z_score:
            if abs(z_score) > 2.5:
                risk += 3
            elif abs(z_score) > 2.0:
                risk += 2
            elif abs(z_score) > 1.5:
                risk += 1

        # Volume risk
        if volume_data['velocity'] > 3.0:
            risk += 2  # Extreme competition
        elif volume_data['velocity'] < 0.5:
            risk += 2  # Illiquid trap

        # RSI risk
        if rsi:
            if rsi > 80 or rsi < 20:
                risk += 2
            elif rsi > 70 or rsi < 30:
                risk += 1

        # Margin risk
        if not margin_data['profitable']:
            risk += 3
        elif margin_data['margin_pct'] < 1:
            risk += 1

        return min(10, max(1, risk))

    def _determine_verdict(self, z_high, z_low, rsi, volume, margin, trend) -> Dict:
        """Determine trading verdict based on all metrics"""
        if not margin['profitable']:
            return {
                'action': 'AVOID',
                'reason': 'Fake margin - Tax eliminates profit',
                'confidence': 'HIGH'
            }

        if z_high and z_high > 2.0:
            return {
                'action': 'AVOID',
                'reason': 'Price spike (Z>2.0) - Will likely crash',
                'confidence': 'HIGH'
            }

        if volume['velocity'] < 0.5:
            return {
                'action': 'CAUTION',
                'reason': 'Illiquid item - Margin may be a trap',
                'confidence': 'MEDIUM'
            }

        if volume['contested']:
            return {
                'action': 'QUICK_FLIP',
                'reason': 'High competition - Move fast or skip',
                'confidence': 'MEDIUM'
            }

        if z_low and z_low < -1.5 and margin['profitable']:
            return {
                'action': 'AGGRESSIVE_BUY',
                'reason': 'Oversold + profitable margin',
                'confidence': 'HIGH'
            }

        if trend['trend'] == 'BULLISH' and margin['profitable']:
            return {
                'action': 'TREND_FLIP',
                'reason': 'Bullish trend with margin',
                'confidence': 'MEDIUM'
            }

        if margin['profitable'] and margin['margin_pct'] > 2:
            return {
                'action': 'STANDARD_FLIP',
                'reason': 'Healthy margin, normal conditions',
                'confidence': 'MEDIUM'
            }

        return {
            'action': 'PATIENT_LIMIT',
            'reason': 'Marginal opportunity - Use limit orders',
            'confidence': 'LOW'
        }


# Item categories for market indexing
ITEM_CATEGORIES = {
    'high_pvm_gear': [
        22325,  # Scythe of vitur
        22323,  # Ghrazi rapier
        27275,  # Tumeken's shadow
        26374,  # Osmumten's fang
        28313,  # Magus ring
        28316,  # Venator ring
        28310,  # Bellator ring
        11802,  # Armadyl godsword
        11804,  # Bandos godsword
        11806,  # Saradomin godsword
        11808,  # Zamorak godsword
        12821,  # Spectral spirit shield
        12825,  # Arcane spirit shield
        12817,  # Elysian spirit shield
        13652,  # Dragon claws
        21034,  # Inquisitor's mace
    ],
    'mid_tier_gear': [
        11834,  # Bandos chestplate
        11836,  # Bandos tassets
        11832,  # Bandos boots
        11828,  # Armadyl helmet
        11830,  # Armadyl chestplate
        11832,  # Armadyl chainskirt
        4151,   # Abyssal whip
        12924,  # Toxic blowpipe
        11785,  # Armadyl crossbow
        22978,  # Dragonfire ward
    ],
    'supplies_consumables': [
        12695,  # Super combat potion(4)
        3024,   # Super restore(4)
        2434,   # Prayer potion(4)
        385,    # Shark
        391,    # Manta ray
        13441,  # Anglerfish
        21510,  # Cooked karambwan
    ],
    'runes_bulk': [
        554,    # Fire rune
        555,    # Water rune
        556,    # Air rune
        557,    # Earth rune
        558,    # Mind rune
        559,    # Body rune
        560,    # Death rune
        561,    # Nature rune
        562,    # Chaos rune
        563,    # Law rune
        564,    # Cosmic rune
        565,    # Blood rune
        566,    # Soul rune
    ],
    'skilling_resources': [
        1515,   # Yew logs
        1513,   # Magic logs
        2362,   # Runite bar
        2360,   # Adamantite bar
        444,    # Gold ore
        454,    # Coal
    ]
}

# GE Buy limits (4-hour limits)
BUY_LIMITS = {
    # High-value uniques (very low limits)
    22325: 8,    # Scythe
    22323: 8,    # Rapier
    27275: 8,    # Tumeken's
    26374: 8,    # Fang
    28313: 8,    # Magus
    28316: 8,    # Venator
    28310: 8,    # Bellator
    12817: 8,    # Elysian
    13652: 8,    # Dragon claws

    # Mid-tier gear
    11834: 8,    # Bandos cp
    11836: 8,    # Bandos tassets
    4151: 70,    # Whip
    12924: 8,    # Blowpipe

    # Supplies (high limits)
    12695: 2000,  # Super combat
    3024: 2000,   # Super restore
    2434: 2000,   # Prayer pot
    385: 13000,   # Shark
    391: 10000,   # Manta ray
    13441: 13000, # Anglerfish

    # Runes (very high)
    560: 25000,   # Death rune
    565: 25000,   # Blood rune
    566: 25000,   # Soul rune
}


def get_buy_limit(item_id: int) -> int:
    """Get buy limit for an item, default to 10000 if unknown"""
    return BUY_LIMITS.get(item_id, 10000)


def get_item_category(item_id: int) -> Optional[str]:
    """Get the category an item belongs to"""
    for category, items in ITEM_CATEGORIES.items():
        if item_id in items:
            return category
    return None


# Singleton instance
_analyzer = None

def get_analyzer() -> QuantAnalyzer:
    """Get singleton analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = QuantAnalyzer()
    return _analyzer


if __name__ == '__main__':
    # Test the analyzer
    analyzer = get_analyzer()

    print("Testing Quantitative Analyzer...")
    print("=" * 60)

    # Test with Dragon Claws (13652)
    analysis = analyzer.full_analysis('13652')

    print(f"\nDragon Claws Analysis:")
    print(f"  High: {analysis['prices']['high']:,} GP")
    print(f"  Low: {analysis['prices']['low']:,} GP")
    print(f"  Net Margin: {analysis['margin']['net']:,} GP ({analysis['margin']['margin_pct']:.1f}%)")
    print(f"  Z-Score: {analysis['z_score']['high']} - {analysis['z_score']['interpretation']}")
    print(f"  RSI: {analysis['rsi']['value']} - {analysis['rsi']['interpretation']}")
    print(f"  Volume: {analysis['volume']['interpretation']}")
    print(f"  Trend: {analysis['trend']['trend']}")
    print(f"  Risk Score: {analysis['risk_score']}/10")
    print(f"  Verdict: {analysis['verdict']['action']} - {analysis['verdict']['reason']}")
