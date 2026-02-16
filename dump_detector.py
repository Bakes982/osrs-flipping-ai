#!/usr/bin/env python3
"""
OSRS Dump Detector - Detects sudden price crashes with recovery prediction
AI-powered dump alerts for catching discounted items
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json
import os

# Import chart generator
try:
    from chart_generator import get_chart_generator
    CHARTS_ENABLED = True
except ImportError:
    CHARTS_ENABLED = False


@dataclass
class DumpAlert:
    """Represents a detected price dump"""
    item_id: int
    item_name: str
    pre_dump_price: int
    current_price: int
    drop_amount: int
    drop_pct: float
    predicted_recovery: int
    recovery_profit: int
    confidence: str  # HIGH, MEDIUM, LOW
    risk_level: str  # LOW, MEDIUM, HIGH
    timestamp: datetime
    chart_path: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'item_name': self.item_name,
            'pre_dump_price': self.pre_dump_price,
            'current_price': self.current_price,
            'drop_amount': self.drop_amount,
            'drop_pct': self.drop_pct,
            'predicted_recovery': self.predicted_recovery,
            'recovery_profit': self.recovery_profit,
            'confidence': self.confidence,
            'risk_level': self.risk_level,
            'timestamp': self.timestamp.isoformat(),
            'chart_path': self.chart_path
        }


class DumpDetector:
    """
    Detects sudden price crashes and predicts recovery potential

    Algorithm:
    1. Compare current price to recent average (1h, 4h, 24h)
    2. If drop > threshold, flag as potential dump
    3. Analyze volume to confirm panic selling
    4. Calculate recovery probability based on historical patterns
    5. Predict recovery price (mean reversion target)
    """

    def __init__(self):
        self.api_url = "https://prices.runescape.wiki/api/v1/osrs"
        self.headers = {
            'User-Agent': 'OSRS-AI-Flipper v1.0 - Contact: mike.baker982@hotmail.com'
        }

        # Dump detection thresholds
        self.min_drop_pct = 3.0  # Minimum 3% drop to consider
        self.min_price = 1_000_000  # Only items worth 1M+
        self.max_price = 2_000_000_000  # Cap at 2B

        # Alert history to avoid duplicates
        self.recent_alerts: Dict[int, datetime] = {}
        self.alert_cooldown = timedelta(hours=1)

        # Cache
        self._price_cache = {}
        self._mapping_cache = None
        self._cache_time = None

    def fetch_latest_prices(self) -> Dict:
        """Fetch current prices for all items"""
        try:
            response = requests.get(
                f"{self.api_url}/latest",
                headers=self.headers,
                timeout=10
            )
            return response.json().get('data', {})
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return {}

    def fetch_item_mapping(self) -> Dict:
        """Fetch item ID to name mapping"""
        if self._mapping_cache:
            return self._mapping_cache

        try:
            response = requests.get(
                f"{self.api_url}/mapping",
                headers=self.headers,
                timeout=10
            )
            items = response.json()
            self._mapping_cache = {str(item['id']): item for item in items}
            return self._mapping_cache
        except Exception as e:
            print(f"Error fetching mapping: {e}")
            return {}

    def fetch_price_history(self, item_id: int, hours: int = 24) -> List[Dict]:
        """Fetch historical prices for an item"""
        try:
            response = requests.get(
                f"{self.api_url}/timeseries?timestep=5m&id={item_id}",
                headers=self.headers,
                timeout=10
            )
            data = response.json().get('data', [])

            # Filter to time range
            cutoff = datetime.now() - timedelta(hours=hours)
            cutoff_ts = cutoff.timestamp()

            return [d for d in data if d.get('timestamp', 0) >= cutoff_ts]
        except Exception as e:
            print(f"Error fetching history for {item_id}: {e}")
            return []

    def calculate_historical_averages(self, history: List[Dict]) -> Dict:
        """Calculate price averages over different time periods"""
        if not history:
            return {}

        now = datetime.now().timestamp()

        # Get high prices (insta-buy)
        prices_1h = []
        prices_4h = []
        prices_24h = []
        volumes_1h = []

        for entry in history:
            ts = entry.get('timestamp', 0)
            high = entry.get('avgHighPrice')
            vol = entry.get('highPriceVolume', 0)

            if high:
                age_hours = (now - ts) / 3600

                if age_hours <= 1:
                    prices_1h.append(high)
                    volumes_1h.append(vol)
                if age_hours <= 4:
                    prices_4h.append(high)
                if age_hours <= 24:
                    prices_24h.append(high)

        return {
            'avg_1h': np.mean(prices_1h) if prices_1h else None,
            'avg_4h': np.mean(prices_4h) if prices_4h else None,
            'avg_24h': np.mean(prices_24h) if prices_24h else None,
            'std_24h': np.std(prices_24h) if len(prices_24h) > 5 else None,
            'min_24h': min(prices_24h) if prices_24h else None,
            'max_24h': max(prices_24h) if prices_24h else None,
            'avg_volume_1h': np.mean(volumes_1h) if volumes_1h else None,
            'price_count': len(prices_24h)
        }

    def detect_dump(self, item_id: int, current_price: int, history: List[Dict]) -> Optional[DumpAlert]:
        """
        Detect if current price represents a dump
        Returns DumpAlert if dump detected, None otherwise
        """
        averages = self.calculate_historical_averages(history)

        if not averages.get('avg_4h'):
            return None

        # Use 4h average as reference (more stable than 1h)
        reference_price = averages['avg_4h']

        # Calculate drop
        drop_amount = reference_price - current_price
        drop_pct = (drop_amount / reference_price) * 100

        # Check if this qualifies as a dump
        if drop_pct < self.min_drop_pct:
            return None

        # Check cooldown
        if item_id in self.recent_alerts:
            if datetime.now() - self.recent_alerts[item_id] < self.alert_cooldown:
                return None

        # Calculate recovery prediction
        # Use mean reversion: predict price will return to recent average
        # But discount based on overall trend

        avg_24h = averages.get('avg_24h', reference_price)
        std_24h = averages.get('std_24h', reference_price * 0.02)

        # If current price is more than 2 std devs below mean, strong recovery expected
        z_score = (current_price - avg_24h) / std_24h if std_24h else 0

        # Conservative recovery prediction
        # Don't expect full recovery, aim for 50-80% of the drop
        if z_score < -2:
            recovery_pct = 0.8  # Strong mean reversion expected
            confidence = "HIGH"
        elif z_score < -1.5:
            recovery_pct = 0.7
            confidence = "HIGH"
        elif z_score < -1:
            recovery_pct = 0.6
            confidence = "MEDIUM"
        else:
            recovery_pct = 0.5
            confidence = "LOW"

        predicted_recovery = int(current_price + (drop_amount * recovery_pct))

        # Calculate potential profit (after tax)
        gross_profit = predicted_recovery - current_price
        tax = min(predicted_recovery * 0.02, 5_000_000)
        recovery_profit = int(gross_profit - tax)

        # Skip if profit too low
        if recovery_profit < 50_000:  # Min 50k profit
            return None

        # Determine risk level
        if drop_pct > 10:
            risk_level = "HIGH"  # Massive dumps might not recover
        elif drop_pct > 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Get item name
        mapping = self.fetch_item_mapping()
        item_info = mapping.get(str(item_id), {})
        item_name = item_info.get('name', f'Item {item_id}')

        # Update cooldown
        self.recent_alerts[item_id] = datetime.now()

        return DumpAlert(
            item_id=item_id,
            item_name=item_name,
            pre_dump_price=int(reference_price),
            current_price=current_price,
            drop_amount=int(drop_amount),
            drop_pct=drop_pct,
            predicted_recovery=predicted_recovery,
            recovery_profit=recovery_profit,
            confidence=confidence,
            risk_level=risk_level,
            timestamp=datetime.now()
        )

    def scan_for_dumps(
        self,
        min_profit: int = 100_000,
        risk_filter: str = "ALL"  # ALL, LOW, MEDIUM, HIGH
    ) -> List[DumpAlert]:
        """
        Scan all items for dump opportunities

        Args:
            min_profit: Minimum recovery profit to alert
            risk_filter: Filter by risk level
        """
        print("Scanning for price dumps...")

        alerts = []
        prices = self.fetch_latest_prices()
        mapping = self.fetch_item_mapping()

        # Filter items by price range
        candidates = []
        for item_id_str, price_data in prices.items():
            try:
                item_id = int(item_id_str)
                high = price_data.get('high')
                low = price_data.get('low')

                if not high or not low:
                    continue

                # Price range filter
                if low < self.min_price or high > self.max_price:
                    continue

                # Get item name to filter junk
                item_info = mapping.get(item_id_str, {})
                item_name = item_info.get('name', '')

                if not item_name or item_name.startswith('Item ') or '(noted)' in item_name.lower():
                    continue

                candidates.append({
                    'id': item_id,
                    'name': item_name,
                    'high': high,
                    'low': low
                })
            except:
                continue

        print(f"Analyzing {len(candidates)} items for dumps...")

        # Check top candidates (by price to catch high-value dumps)
        candidates.sort(key=lambda x: x['high'], reverse=True)

        for i, candidate in enumerate(candidates[:500]):  # Check top 500 items
            try:
                history = self.fetch_price_history(candidate['id'], hours=24)

                if len(history) < 10:  # Need enough data
                    continue

                alert = self.detect_dump(
                    candidate['id'],
                    candidate['low'],  # Use insta-sell as entry point
                    history
                )

                if alert:
                    # Apply filters
                    if alert.recovery_profit < min_profit:
                        continue

                    if risk_filter != "ALL" and alert.risk_level != risk_filter:
                        continue

                    # Generate chart if available
                    if CHARTS_ENABLED:
                        try:
                            chart_gen = get_chart_generator()
                            alert.chart_path = chart_gen.create_dump_alert_chart(
                                item_name=alert.item_name,
                                item_id=alert.item_id,
                                dump_price=alert.current_price,
                                pre_dump_price=alert.pre_dump_price,
                                predicted_recovery=alert.predicted_recovery,
                                drop_pct=alert.drop_pct,
                                hours=6
                            )
                        except Exception as e:
                            print(f"Chart generation error: {e}")

                    alerts.append(alert)
                    print(f"  DUMP DETECTED: {alert.item_name} -{alert.drop_pct:.1f}% (Profit: {alert.recovery_profit:,} GP)")

            except Exception as e:
                continue

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Checked {i + 1}/500 items...")

        # Sort by profit potential
        alerts.sort(key=lambda x: x.recovery_profit, reverse=True)

        print(f"Found {len(alerts)} dump alerts")
        return alerts

    def get_dump_summary(self, alerts: List[DumpAlert]) -> str:
        """Generate a summary of dump alerts for Discord/display"""
        if not alerts:
            return "No dump alerts at this time."

        lines = [
            "# DUMP ALERTS",
            f"*{len(alerts)} opportunities detected*",
            ""
        ]

        for i, alert in enumerate(alerts[:10], 1):
            confidence_emoji = {"HIGH": "[+++]", "MEDIUM": "[++]", "LOW": "[+]"}[alert.confidence]
            risk_emoji = {"LOW": "[Safe]", "MEDIUM": "[Mod]", "HIGH": "[Risk]"}[alert.risk_level]

            lines.append(f"**{i}. {alert.item_name}**")
            lines.append(f"   Drop: -{alert.drop_pct:.1f}% ({alert.drop_amount:,} GP)")
            lines.append(f"   Buy Now: {alert.current_price:,} GP")
            lines.append(f"   Target: {alert.predicted_recovery:,} GP")
            lines.append(f"   Profit: {alert.recovery_profit:,} GP")
            lines.append(f"   {confidence_emoji} {risk_emoji}")
            lines.append("")

        return "\n".join(lines)


class DumpAlertNotifier:
    """Send dump alerts to Discord"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_dump_alert(self, alert: DumpAlert):
        """Send a single dump alert to Discord"""
        try:
            # Confidence and risk colors
            colors = {
                'HIGH': 0x00ff00,  # Green for high confidence
                'MEDIUM': 0xffaa00,  # Orange
                'LOW': 0xff4444  # Red for low confidence
            }

            embed = {
                "title": f"DUMP ALERT: {alert.item_name}",
                "description": f"Price dropped **-{alert.drop_pct:.1f}%** from recent average!",
                "color": colors.get(alert.confidence, 0xffaa00),
                "fields": [
                    {"name": "Pre-Dump Price", "value": f"{alert.pre_dump_price:,} GP", "inline": True},
                    {"name": "Current Price", "value": f"{alert.current_price:,} GP", "inline": True},
                    {"name": "Drop", "value": f"-{alert.drop_amount:,} GP (-{alert.drop_pct:.1f}%)", "inline": True},
                    {"name": "Predicted Recovery", "value": f"{alert.predicted_recovery:,} GP", "inline": True},
                    {"name": "Potential Profit", "value": f"{alert.recovery_profit:,} GP", "inline": True},
                    {"name": "Confidence", "value": alert.confidence, "inline": True},
                    {"name": "Risk Level", "value": alert.risk_level, "inline": True},
                ],
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": "OSRS AI Flipper - Dump Detection"}
            }

            payload = {"embeds": [embed]}

            # Attach chart image if available
            if alert.chart_path and os.path.exists(alert.chart_path):
                with open(alert.chart_path, 'rb') as f:
                    files = {'file': (os.path.basename(alert.chart_path), f, 'image/png')}
                    # For file uploads, need multipart/form-data
                    embed["image"] = {"url": f"attachment://{os.path.basename(alert.chart_path)}"}

                    response = requests.post(
                        self.webhook_url,
                        data={'payload_json': json.dumps(payload)},
                        files=files,
                        timeout=10
                    )
            else:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10
                )

            return response.status_code in [200, 204]

        except Exception as e:
            print(f"Error sending dump alert: {e}")
            return False

    def send_dump_summary(self, alerts: List[DumpAlert]):
        """Send a summary of all dump alerts"""
        if not alerts:
            return

        try:
            embed = {
                "title": f"DUMP ALERT SUMMARY - {len(alerts)} Opportunities!",
                "description": "Items with significant price drops detected",
                "color": 0xff4444,
                "fields": [],
                "timestamp": datetime.now().isoformat(),
                "footer": {"text": "OSRS AI Flipper - Dump Detection"}
            }

            for alert in alerts[:5]:
                embed["fields"].append({
                    "name": f"{alert.item_name} (-{alert.drop_pct:.1f}%)",
                    "value": f"Buy: {alert.current_price:,} GP\nTarget: {alert.predicted_recovery:,} GP\nProfit: {alert.recovery_profit:,} GP\nRisk: {alert.risk_level}",
                    "inline": False
                })

            response = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=10
            )

            return response.status_code in [200, 204]

        except Exception as e:
            print(f"Error sending dump summary: {e}")
            return False


# Singleton instance
_dump_detector = None

def get_dump_detector() -> DumpDetector:
    global _dump_detector
    if _dump_detector is None:
        _dump_detector = DumpDetector()
    return _dump_detector


if __name__ == '__main__':
    print("OSRS Dump Detector")
    print("=" * 60)

    detector = get_dump_detector()

    # Scan for dumps
    alerts = detector.scan_for_dumps(
        min_profit=100_000,
        risk_filter="ALL"
    )

    # Print summary
    print("\n" + detector.get_dump_summary(alerts))

    # Test Discord notification (if webhook configured)
    webhook_url = os.environ.get('DISCORD_WEBHOOK')
    if webhook_url and alerts:
        notifier = DumpAlertNotifier(webhook_url)
        notifier.send_dump_summary(alerts)
        print("\nSent dump summary to Discord!")
