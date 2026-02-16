#!/usr/bin/env python3
"""
OSRS Live Price Monitor - Automated updates via OSRS Wiki API
Runs continuously, updates prices, and generates new opportunities
"""

import time
import requests
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import pandas as pd
import threading
from pathlib import Path

from flip_data_analyzer import FlipDataAnalyzer
from flip_predictor import FlipPredictor
from investment_finder import InvestmentFinder
from user_config import UserConfig

# Import new visual/dump detection modules
try:
    from chart_generator import get_chart_generator
    CHARTS_ENABLED = True
except ImportError:
    CHARTS_ENABLED = False
    print("Note: chart_generator.py not found - visual charts disabled")

try:
    from dump_detector import get_dump_detector, DumpAlert
    DUMP_DETECTION_ENABLED = True
except ImportError:
    DUMP_DETECTION_ENABLED = False
    print("Note: dump_detector.py not found - dump alerts disabled")

class LivePriceMonitor:
    """
    Continuously monitors OSRS prices and generates opportunities
    """
    
    def __init__(self, csv_path: str, config_file: str = "user_config.json"):
        print("🔴 LIVE PRICE MONITOR")
        print("="*80)
        print()
        
        # Initialize components
        print("Loading components...")
        self.analyzer = FlipDataAnalyzer(csv_path)
        self.predictor = FlipPredictor()
        self.predictor.train(self.analyzer)
        self.investment_finder = InvestmentFinder()
        self.config = UserConfig(config_file)
        
        # OSRS Wiki API setup
        self.api_base = "https://prices.runescape.wiki/api/v1/osrs"
        self.headers = {
            'User-Agent': 'OSRS-AI-Flipper v1.0 - Discord: bakes982 - Contact: mike.baker982@hotmail.com'
        }
        
        # Cache
        self.item_mapping = None
        self.latest_prices = {}
        self.price_history = {}
        
        # Settings
        self.update_interval = 300  # 5 minutes (don't spam API)
        self.running = False
        
        print("✅ Live monitor ready!")

        # Check Discord webhook status
        discord_config = self.config.get_discord_config()
        if discord_config.get('enabled') and discord_config.get('url'):
            print("📱 Discord notifications: ENABLED")
        else:
            print("📱 Discord notifications: DISABLED (set webhook in config)")
        print()

    def send_discord_notification(self, title: str, description: str, color: int = 0x00ff00, fields: List[Dict] = None):
        """Send a notification to Discord webhook"""
        discord_config = self.config.get_discord_config()

        if not discord_config.get('enabled') or not discord_config.get('url'):
            return False

        webhook_url = discord_config['url']

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "OSRS AI Flipper"}
        }

        if fields:
            embed["fields"] = fields

        payload = {"embeds": [embed]}

        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            return response.status_code == 204
        except Exception as e:
            print(f"Discord notification failed: {e}")
            return False

    def notify_opportunities(self, opportunities: pd.DataFrame):
        """Send top opportunities to Discord"""
        discord_config = self.config.get_discord_config()

        if not discord_config.get('enabled') or not discord_config.get('notify_opportunities'):
            return

        min_score = discord_config.get('min_score_to_notify', 50)
        top_opps = opportunities[opportunities['opportunity_score'] >= min_score].head(5)

        if len(top_opps) == 0:
            return

        fields = []
        for _, row in top_opps.iterrows():
            tax_str = f" (Tax: {row['tax']:,})" if 'tax' in row else ""
            fields.append({
                "name": f"{row['item_name']} ({row['opportunity_score']:.0f}/100)",
                "value": f"**Net Profit: {row['predicted_profit']:,} GP**{tax_str}\nBuy: {row['current_low']:,} | Sell: {row['current_high']:,}\nMargin: {row['spread_pct']:.2f}% (after tax) | Risk: {row['risk_level']}",
                "inline": False
            })

        self.send_discord_notification(
            title=f"🏆 {len(top_opps)} Flip Opportunities Found!",
            description=f"Top opportunities scoring {min_score}+",
            color=0x00ff00,  # Green
            fields=fields
        )

    def notify_price_spikes(self, spikes: List[Dict]):
        """Send price spike alerts to Discord (separate webhook if configured)"""
        discord_config = self.config.get_discord_config()

        if not discord_config.get('enabled') or not discord_config.get('notify_price_spikes'):
            return

        if not spikes:
            return

        # Use separate webhook for price alerts if configured
        price_alerts_webhook = discord_config.get('price_alerts_webhook')
        webhook_url = price_alerts_webhook if price_alerts_webhook else discord_config.get('url')

        if not webhook_url:
            return

        # Split into UP and DOWN alerts
        up_spikes = [s for s in spikes if s['direction'] == 'UP']
        down_spikes = [s for s in spikes if s['direction'] == 'DOWN']

        fields = []

        if down_spikes:
            down_text = "\n".join([
                f"📉 **{s['item_name']}**: {s['change_pct']:+.1f}%\n"
                f"   {s['historical_price']:,.0f} → {s['current_price']:,} GP | "
                f"You flipped {s.get('flip_count', 0)}x for {s.get('total_profit', 0):,.0f} GP total"
                for s in down_spikes[:5]
            ])
            fields.append({
                "name": f"🔻 PRICE CRASHES ({len(down_spikes)} items >1M GP)",
                "value": down_text[:1024],
                "inline": False
            })

        if up_spikes:
            up_text = "\n".join([
                f"📈 **{s['item_name']}**: {s['change_pct']:+.1f}%\n"
                f"   {s['historical_price']:,.0f} → {s['current_price']:,} GP | "
                f"You flipped {s.get('flip_count', 0)}x for {s.get('total_profit', 0):,.0f} GP total"
                for s in up_spikes[:5]
            ])
            fields.append({
                "name": f"🔺 PRICE SPIKES ({len(up_spikes)} items >1M GP)",
                "value": up_text[:1024],
                "inline": False
            })

        embed = {
            "title": f"⚡ {len(spikes)} Price Alerts on Items You've Profited From!",
            "description": f"**Filtered:** Items >1M GP that you've made profit on\n📉 {len(down_spikes)} crashes | 📈 {len(up_spikes)} spikes",
            "color": 0xff4444 if len(down_spikes) > len(up_spikes) else 0x00ff00,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "OSRS AI Flipper - Smart Price Monitor"}
        }

        if fields:
            embed["fields"] = fields

        payload = {"embeds": [embed]}

        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            if response.status_code == 204:
                print("📱 Price alerts sent to Discord")
        except Exception as e:
            print(f"Discord price alert failed: {e}")

    def send_visual_opportunity(self, item_name: str, item_id: int, buy_price: int,
                                 sell_price: int, profit: int, margin_pct: float,
                                 risk_level: str, verdict: str):
        """Send an opportunity with a visual chart to Discord"""
        if not CHARTS_ENABLED:
            return False

        discord_config = self.config.get_discord_config()
        if not discord_config.get('enabled') or not discord_config.get('url'):
            return False

        try:
            chart_gen = get_chart_generator()

            # Generate chart
            chart_path = chart_gen.create_suggestion_chart(
                item_name=item_name,
                item_id=item_id,
                buy_price=buy_price,
                sell_price=sell_price,
                current_price=(buy_price + sell_price) // 2,
                risk_level=risk_level,
                verdict=verdict,
                profit=profit,
                margin_pct=margin_pct,
                hours=24
            )

            # Send to Discord with image
            embed = {
                "title": f"FLIP OPPORTUNITY: {item_name}",
                "description": f"**{verdict}** - {risk_level} Risk",
                "color": 0x00ff00 if risk_level == "LOW" else 0xffaa00 if risk_level == "MEDIUM" else 0xff4444,
                "fields": [
                    {"name": "Buy At", "value": f"{buy_price:,} GP", "inline": True},
                    {"name": "Sell At", "value": f"{sell_price:,} GP", "inline": True},
                    {"name": "Net Profit", "value": f"{profit:,} GP ({margin_pct:.1f}%)", "inline": True},
                ],
                "image": {"url": f"attachment://chart.png"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "OSRS AI Flipper - Visual Suggestions"}
            }

            import os
            with open(chart_path, 'rb') as f:
                files = {'file': ('chart.png', f, 'image/png')}
                payload = {'payload_json': json.dumps({"embeds": [embed]})}

                response = requests.post(
                    discord_config['url'],
                    data=payload,
                    files=files,
                    timeout=15
                )

            return response.status_code in [200, 204]

        except Exception as e:
            print(f"Visual opportunity notification failed: {e}")
            return False

    def notify_dump_alerts(self, alerts: list):
        """Send dump alerts to Discord with optional charts"""
        discord_config = self.config.get_discord_config()

        if not discord_config.get('enabled') or not discord_config.get('url'):
            return

        if not alerts:
            return

        webhook_url = discord_config.get('dump_alerts_webhook', discord_config.get('url'))

        for alert in alerts[:3]:  # Send top 3 dump alerts
            try:
                embed = {
                    "title": f"DUMP ALERT: {alert.item_name}",
                    "description": f"Price crashed **-{alert.drop_pct:.1f}%** - Recovery opportunity!",
                    "color": 0xff4444,
                    "fields": [
                        {"name": "Pre-Dump Price", "value": f"{alert.pre_dump_price:,} GP", "inline": True},
                        {"name": "Current Price", "value": f"{alert.current_price:,} GP", "inline": True},
                        {"name": "Drop", "value": f"-{alert.drop_amount:,} GP", "inline": True},
                        {"name": "Predicted Recovery", "value": f"{alert.predicted_recovery:,} GP", "inline": True},
                        {"name": "Potential Profit", "value": f"{alert.recovery_profit:,} GP", "inline": True},
                        {"name": "Confidence", "value": alert.confidence, "inline": True},
                        {"name": "Risk", "value": alert.risk_level, "inline": True},
                    ],
                    "timestamp": alert.timestamp.isoformat(),
                    "footer": {"text": "OSRS AI Flipper - Dump Detection"}
                }

                # If chart available, send with image
                if alert.chart_path and os.path.exists(alert.chart_path):
                    embed["image"] = {"url": "attachment://dump_chart.png"}

                    with open(alert.chart_path, 'rb') as f:
                        files = {'file': ('dump_chart.png', f, 'image/png')}
                        payload = {'payload_json': json.dumps({"embeds": [embed]})}

                        response = requests.post(
                            webhook_url,
                            data=payload,
                            files=files,
                            timeout=15
                        )
                else:
                    response = requests.post(
                        webhook_url,
                        json={"embeds": [embed]},
                        timeout=10
                    )

                if response.status_code in [200, 204]:
                    print(f"   Sent dump alert for {alert.item_name}")

            except Exception as e:
                print(f"Error sending dump alert: {e}")

    def scan_for_dumps(self, min_profit: int = 200_000) -> list:
        """Scan for price dumps using the dump detector"""
        if not DUMP_DETECTION_ENABLED:
            return []

        try:
            detector = get_dump_detector()
            alerts = detector.scan_for_dumps(min_profit=min_profit, risk_filter="ALL")
            return alerts
        except Exception as e:
            print(f"Error scanning for dumps: {e}")
            return []

    def fetch_item_mapping(self):
        """Fetch item name -> ID mapping"""
        try:
            response = requests.get(
                f"{self.api_base}/mapping",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            items = response.json()
            self.item_mapping = {
                item['name']: item['id'] 
                for item in items
            }
            self.id_to_name = {
                item['id']: item['name']
                for item in items
            }
            
            print(f"✅ Loaded {len(self.item_mapping)} items from OSRS Wiki")
            return True
            
        except Exception as e:
            print(f"❌ Error fetching item mapping: {e}")
            return False
    
    def fetch_latest_prices(self):
        """Fetch current prices for all items"""
        try:
            response = requests.get(
                f"{self.api_base}/latest",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            self.latest_prices = data.get('data', {})
            
            print(f"✅ Updated prices at {datetime.now().strftime('%H:%M:%S')}")
            return True
            
        except Exception as e:
            print(f"❌ Error fetching prices: {e}")
            return False
    
    def get_item_current_price(self, item_name: str) -> Dict:
        """Get current price for specific item"""
        if not self.item_mapping:
            return None
        
        item_id = self.item_mapping.get(item_name)
        if not item_id:
            return None
        
        price_data = self.latest_prices.get(str(item_id))
        if not price_data:
            return None
        
        high = price_data.get('high')
        low = price_data.get('low')

        # Skip items with None prices
        if high is None or low is None:
            return None

        return {
            'item_name': item_name,
            'item_id': item_id,
            'high_price': high,
            'low_price': low,
            'high_time': price_data.get('highTime'),
            'low_time': price_data.get('lowTime'),
            'spread': high - low,
            'spread_pct': ((high - low) / high) * 100 if high > 0 else 0
        }
    
    def fetch_price_history(self, item_id: int, timestep: str = '5m'):
        """Fetch historical prices for an item"""
        try:
            response = requests.get(
                f"{self.api_base}/timeseries",
                params={'timestep': timestep, 'id': item_id},
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', [])
            
        except Exception as e:
            print(f"❌ Error fetching history for item {item_id}: {e}")
            return []
    
    def calculate_live_features(self, item_name: str) -> Dict:
        """Calculate features from live prices + historical data"""
        # Get current price
        current = self.get_item_current_price(item_name)
        if not current:
            return None
        
        # Get historical performance from your data
        hist_features = self.analyzer.calculate_features_for_item(item_name)
        if not hist_features:
            # New item never flipped before
            hist_features = {
                'avg_profit': 0,
                'avg_time_hours': 12,
                'avg_gp_hr': 0,
                'success_rate': 0.5,
                'flip_count': 0,
                'avg_price': current['high_price'],
                'profit_consistency': 0.5,
                'time_consistency': 0.5
            }
        
        # Combine live + historical
        features = {
            **hist_features,
            'current_high': current['high_price'],
            'current_low': current['low_price'],
            'current_spread': current['spread'],
            'current_spread_pct': current['spread_pct'],
            'last_update': datetime.now()
        }
        
        return features
    
    def calculate_real_profit(self, buy_price: int, sell_price: int, quantity: int = 1) -> dict:
        """
        Calculate REAL profit after 2% GE tax (capped at 5M per item)
        This is the only source of truth for profit calculations!
        """
        gross_margin = sell_price - buy_price
        # GE tax is 2% of sell price, capped at 5M per item
        tax_per_item = min(sell_price * 0.02, 5_000_000)
        total_tax = tax_per_item * quantity
        net_profit = (gross_margin * quantity) - total_tax
        margin_pct = (net_profit / (buy_price * quantity) * 100) if buy_price > 0 else 0

        return {
            'gross_margin': gross_margin * quantity,
            'tax': total_tax,
            'net_profit': net_profit,
            'margin_pct': margin_pct,
            'profitable': net_profit > 0
        }

    def find_live_opportunities(self, min_price: int = 1_000_000, max_price: int = 500_000_000) -> pd.DataFrame:
        """
        Find opportunities by scanning ALL items - not just your history!
        Uses REAL profit after 2% GE tax.
        """
        print("🔍 Scanning ALL items for opportunities...")

        opportunities = []

        # Scan ALL items from the API, not just history
        for item_id_str, price_data in self.latest_prices.items():
            try:
                item_id = int(item_id_str)
            except:
                continue

            # Get prices
            sell_price = price_data.get('high')  # Insta-buy = your sell target
            buy_price = price_data.get('low')    # Insta-sell = your buy target

            # Skip items without both prices
            if not sell_price or not buy_price:
                continue

            # Filter by price range
            if buy_price < min_price or sell_price > max_price:
                continue

            # Calculate REAL profit after tax
            real_profit = self.calculate_real_profit(buy_price, sell_price)

            # CRITICAL: Skip if not profitable after tax!
            if not real_profit['profitable']:
                continue

            # Skip if margin too small (< 0.5% after tax)
            if real_profit['margin_pct'] < 0.5:
                continue

            # Get item name
            item_name = self.id_to_name.get(item_id, f'Item {item_id}')

            # Skip placeholder names
            if item_name.startswith('Item ') or '(noted)' in item_name.lower():
                continue

            # Skip if blocked
            if self.config.is_blocked(item_name):
                continue

            # Calculate risk based on price staleness
            high_time = price_data.get('highTime', 0)
            low_time = price_data.get('lowTime', 0)
            now = int(time.time())

            risk_score = 3  # Base risk
            stale_hours = 0
            if high_time and (now - high_time) > 3600:
                stale_hours = max(stale_hours, (now - high_time) // 3600)
            if low_time and (now - low_time) > 3600:
                stale_hours = max(stale_hours, (now - low_time) // 3600)
            if stale_hours > 0:
                risk_score += min(stale_hours, 3)

            # High margin items are riskier (may be illiquid)
            if real_profit['margin_pct'] > 10:
                risk_score += 2
            elif real_profit['margin_pct'] > 5:
                risk_score += 1

            # Determine risk level
            if risk_score <= 4:
                risk_level = 'LOW'
            elif risk_score <= 6:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'

            # Calculate opportunity score (profit-weighted)
            # Higher profit = higher score, but cap risk penalty
            base_score = min(100, real_profit['net_profit'] / 10000)  # 1M profit = 100 score
            risk_penalty = risk_score * 5
            opportunity_score = max(0, base_score - risk_penalty)

            # Get slot assignments
            available_slots = self.config.get_available_slots_for_item(sell_price)

            opportunities.append({
                'item_name': item_name,
                'item_id': item_id,
                'opportunity_score': opportunity_score,
                'predicted_profit': int(real_profit['net_profit']),
                'risk_level': risk_level,
                'risk_score': risk_score,
                'confidence': 'HIGH' if risk_score <= 4 else 'MEDIUM' if risk_score <= 6 else 'LOW',
                'current_high': sell_price,
                'current_low': buy_price,
                'current_spread': sell_price - buy_price,
                'spread_pct': real_profit['margin_pct'],
                'tax': int(real_profit['tax']),
                'available_slots': available_slots,
                'offer_time': self.config.get_offer_time(item_name),
                'last_update': datetime.now()
            })

        print(f"Found {len(opportunities)} profitable items (after 2% tax)")

        if not opportunities:
            return pd.DataFrame()

        df = pd.DataFrame(opportunities)
        # Sort by net profit (absolute GP)
        df = df.sort_values('predicted_profit', ascending=False)

        return df
    
    def detect_price_spikes(self) -> List[Dict]:
        """Detect sudden price movements (investment signals) - SMART FILTERING"""
        spikes = []

        # Get your historical stats once
        hist_stats = self.analyzer.get_item_statistics()

        # Settings for smart filtering
        MIN_PRICE = 1000000  # Only items worth >1M GP
        MIN_CHANGE_PCT = 10  # 10% price change threshold
        ONLY_PROFITABLE_ITEMS = True  # Only items you've made profit on

        # Compare current prices to historical averages
        for item_id_str in self.latest_prices.keys():
            try:
                item_id = int(item_id_str)
            except:
                continue

            if not self.id_to_name.get(item_id):
                continue

            item_name_str = self.id_to_name[item_id]

            # Get current price
            current = self.get_item_current_price(item_name_str)
            if not current:
                continue

            # FILTER 1: Minimum price threshold
            if current['high_price'] < MIN_PRICE:
                continue

            # Check if item is in your history
            if item_name_str not in hist_stats.index:
                continue

            hist_price = hist_stats.loc[item_name_str, 'Avg Price']

            # Skip if prices are None or zero
            if not current['high_price'] or not hist_price or hist_price == 0:
                continue

            # FILTER 2: Only items you've profited from
            if ONLY_PROFITABLE_ITEMS:
                total_profit = hist_stats.loc[item_name_str, 'Total Profit'] if 'Total Profit' in hist_stats.columns else 0
                if total_profit <= 0:
                    continue

            # Calculate change
            price_change = ((current['high_price'] - hist_price) / hist_price) * 100

            # Detect significant changes
            if abs(price_change) > MIN_CHANGE_PCT:
                # Get additional context
                flip_count = hist_stats.loc[item_name_str, 'Flip Count'] if 'Flip Count' in hist_stats.columns else 0
                total_profit = hist_stats.loc[item_name_str, 'Total Profit'] if 'Total Profit' in hist_stats.columns else 0

                spikes.append({
                    'item_name': item_name_str,
                    'current_price': current['high_price'],
                    'historical_price': hist_price,
                    'change_pct': price_change,
                    'direction': 'UP' if price_change > 0 else 'DOWN',
                    'flip_count': flip_count,
                    'total_profit': total_profit,
                    'timestamp': datetime.now()
                })

        # Sort by absolute change percentage (biggest moves first)
        spikes.sort(key=lambda x: abs(x['change_pct']), reverse=True)

        return spikes
    
    def run_continuous_monitor(self, interval_seconds: int = 300):
        """
        Run continuous monitoring loop
        
        Args:
            interval_seconds: Update interval (default 5 minutes)
        """
        self.running = True
        self.update_interval = interval_seconds
        
        print("="*80)
        print("🔴 STARTING CONTINUOUS MONITORING")
        print("="*80)
        print(f"Update interval: {interval_seconds} seconds ({interval_seconds/60:.1f} minutes)")
        print("Press Ctrl+C to stop")
        print("="*80)
        print()
        
        # Initial fetch
        if not self.fetch_item_mapping():
            print("❌ Failed to load item mapping. Exiting.")
            return
        
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                print(f"\n{'='*80}")
                print(f"UPDATE #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}\n")
                
                # Fetch latest prices
                if not self.fetch_latest_prices():
                    print("⚠️  Price fetch failed, retrying next cycle...")
                    time.sleep(self.update_interval)
                    continue
                
                # Find opportunities
                opportunities = self.find_live_opportunities()
                
                if len(opportunities) > 0:
                    print(f"\n🏆 TOP 5 LIVE OPPORTUNITIES\n")
                    print(f"{'─'*80}\n")
                    
                    for i, (idx, row) in enumerate(opportunities.head(5).iterrows(), 1):
                        tax_str = f" (Tax: {row['tax']:,})" if 'tax' in row else ""
                        print(f"{i}. {row['item_name']}")
                        print(f"   Score: {row['opportunity_score']:.0f}/100 | "
                              f"Net Profit: {row['predicted_profit']:,} GP{tax_str} | "
                              f"Risk: {row['risk_level']}")
                        print(f"   Buy: {row['current_low']:,} | "
                              f"Sell: {row['current_high']:,} | "
                              f"Margin: {row['spread_pct']:.2f}% (after 2% tax)")
                        print(f"   Slots: {', '.join(map(str, row['available_slots']))} | "
                              f"Time: {row['offer_time']}min")
                        print()

                    # Send Discord notification
                    self.notify_opportunities(opportunities)
                else:
                    print("❌ No opportunities found this cycle")

                # Check for price spikes
                spikes = self.detect_price_spikes()
                if spikes:
                    print(f"\n⚡ PRICE ALERTS ({len(spikes)} detected)\n")
                    print(f"{'─'*80}\n")

                    for spike in spikes[:5]:
                        direction_emoji = "📈" if spike['direction'] == 'UP' else "📉"
                        print(f"{direction_emoji} {spike['item_name']}: "
                              f"{spike['change_pct']:+.1f}% "
                              f"({spike['historical_price']:,} → {spike['current_price']:,})")

                    # Send Discord notification for price spikes
                    self.notify_price_spikes(spikes)

                # Check for dumps (every other cycle to save API calls)
                if DUMP_DETECTION_ENABLED and iteration % 2 == 0:
                    print(f"\n📉 SCANNING FOR DUMPS...\n")
                    dump_alerts = self.scan_for_dumps(min_profit=200_000)
                    if dump_alerts:
                        print(f"🚨 {len(dump_alerts)} DUMP ALERTS FOUND!\n")
                        for alert in dump_alerts[:5]:
                            print(f"   {alert.item_name}: -{alert.drop_pct:.1f}% | "
                                  f"Buy: {alert.current_price:,} | "
                                  f"Profit: {alert.recovery_profit:,} GP | "
                                  f"{alert.confidence}")
                        # Send to Discord
                        self.notify_dump_alerts(dump_alerts)

                # Send visual chart for top opportunity (every 3rd cycle)
                if CHARTS_ENABLED and iteration % 3 == 0 and len(opportunities) > 0:
                    top_opp = opportunities.iloc[0]
                    print(f"\n📊 Generating visual chart for {top_opp['item_name']}...")
                    self.send_visual_opportunity(
                        item_name=top_opp['item_name'],
                        item_id=top_opp['item_id'],
                        buy_price=top_opp['current_low'],
                        sell_price=top_opp['current_high'],
                        profit=top_opp['predicted_profit'],
                        margin_pct=top_opp['spread_pct'],
                        risk_level=top_opp['risk_level'],
                        verdict="STANDARD_FLIP" if top_opp['spread_pct'] > 2 else "QUICK_FLIP"
                    )

                # Save opportunities to file
                if len(opportunities) > 0:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"live_opportunities_{timestamp}.json"
                    
                    opportunities.to_json(filename, orient='records', indent=2)
                    print(f"\n💾 Saved to {filename}")
                
                # Wait for next cycle
                print(f"\n⏰ Next update in {interval_seconds} seconds...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping monitor...")
            self.running = False
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")
            self.running = False
    
    def run_scheduled_updates(self, times: List[str]):
        """
        Run updates at specific times each day
        
        Args:
            times: List of times in HH:MM format (e.g., ['08:00', '12:00', '18:00'])
        """
        print("="*80)
        print("📅 SCHEDULED UPDATE MODE")
        print("="*80)
        print(f"Update times: {', '.join(times)}")
        print("="*80)
        print()
        
        # Initial setup
        if not self.fetch_item_mapping():
            print("❌ Failed to load item mapping. Exiting.")
            return
        
        self.running = True
        
        try:
            while self.running:
                now = datetime.now()
                current_time = now.strftime('%H:%M')
                
                # Check if it's time for an update
                if current_time in times:
                    print(f"\n⏰ SCHEDULED UPDATE - {now.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Run update
                    self.fetch_latest_prices()
                    opportunities = self.find_live_opportunities()
                    
                    # Display top opportunities
                    if len(opportunities) > 0:
                        print(f"\n🏆 TOP 10 OPPORTUNITIES\n")
                        for i, (idx, row) in enumerate(opportunities.head(10).iterrows(), 1):
                            print(f"{i:2d}. {row['item_name']:40s} | "
                                  f"{row['opportunity_score']:>3.0f}/100 | "
                                  f"{row['predicted_profit']:>8,} GP")
                        
                        # Save
                        timestamp = now.strftime('%Y%m%d_%H%M%S')
                        filename = f"scheduled_opportunities_{timestamp}.json"
                        opportunities.to_json(filename, orient='records', indent=2)
                        print(f"\n💾 Saved to {filename}")
                    
                    # Sleep for 61 seconds to avoid duplicate triggers
                    time.sleep(61)
                else:
                    # Sleep for 30 seconds and check again
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping scheduled updates...")
            self.running = False


def main():
    """Main entry point"""
    import sys
    
    csv_path = r'C:\Users\Mikeb\OneDrive\Desktop\flips.csv'
    
    monitor = LivePriceMonitor(csv_path)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--continuous':
            # Continuous monitoring mode
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
            monitor.run_continuous_monitor(interval_seconds=interval)
        
        elif sys.argv[1] == '--scheduled':
            # Scheduled mode
            times = sys.argv[2:] if len(sys.argv) > 2 else ['08:00', '12:00', '18:00']
            monitor.run_scheduled_updates(times)
    else:
        print("LIVE PRICE MONITOR")
        print("="*80)
        print()
        print("Usage:")
        print("  Continuous monitoring (every 5 minutes):")
        print("    python3 live_price_monitor.py --continuous")
        print()
        print("  Continuous with custom interval:")
        print("    python3 live_price_monitor.py --continuous 600  # 10 minutes")
        print()
        print("  Scheduled updates:")
        print("    python3 live_price_monitor.py --scheduled 08:00 12:00 18:00")
        print()


if __name__ == "__main__":
    main()
