#!/usr/bin/env python3
"""
Complete Automated OSRS Flip Bot
Monitors live prices, detects opportunities, sends alerts
"""

import sys
import argparse
from datetime import datetime
import time

from flip_data_analyzer import FlipDataAnalyzer
from flip_predictor import FlipPredictor
from user_config import UserConfig
from automated_monitor import AutomatedFlipMonitor, NotificationSystem, LivePriceMonitor

class FlipBot:
    """Main automated flip bot"""
    
    def __init__(
        self,
        csv_path: str,
        config_file: str = "user_config.json",
        discord_webhook: str = None
    ):
        print("="*80)
        print("🤖 OSRS FLIP BOT - FULL AUTOMATION")
        print("="*80)
        print()
        
        # Load components
        print("📊 Loading your flip history...")
        self.analyzer = FlipDataAnalyzer(csv_path)
        
        print("⚙️  Loading configuration...")
        self.config = UserConfig(config_file)
        
        print("🧠 Training AI model...")
        self.predictor = FlipPredictor()
        self.predictor.train(self.analyzer)
        
        print("📡 Initializing live price monitor...")
        self.monitor = AutomatedFlipMonitor(
            self.analyzer,
            self.predictor,
            self.config,
            update_interval=300  # 5 minutes
        )
        
        print("🔔 Setting up notifications...")
        self.notifier = NotificationSystem(discord_webhook)
        
        print()
        print("="*80)
        print("✅ BOT READY!")
        print("="*80)
        print()
    
    def run_automated(self):
        """Run fully automated monitoring"""
        print("🚀 Starting automated monitoring...")
        print()
        print("The bot will:")
        print("  • Fetch live prices every 5 minutes")
        print("  • Check your watchlist (top 50 items)")
        print("  • Alert when opportunities appear")
        print("  • Track price spikes")
        print("  • Respect your blocklist and slot configuration")
        print()
        print("Press Ctrl+C to stop")
        print()
        
        # Initialize
        self.monitor.initialize()
        
        # Start monitoring
        self.monitor.monitor_loop()
    
    def run_single_check(self):
        """Run one-time check"""
        print("🔍 Running single check...")
        print()
        
        # Initialize
        self.monitor.initialize()
        
        # Check once
        opportunities = self.monitor.check_opportunities()
        
        if opportunities:
            print(f"✅ Found {len(opportunities)} opportunities:")
            print()
            
            for i, opp in enumerate(opportunities, 1):
                print(f"#{i} - {opp['item_name']}")
                print(f"  💰 Profit: {opp['profit']:,} GP ({opp['margin_pct']:.1f}%)")
                print(f"  💵 Buy: {opp['buy_price']:,} | Sell: {opp['sell_price']:,}")
                print(f"  🎰 Available Slots: {', '.join(map(str, opp['available_slots']))}")
                print(f"  ⏰ Offer Time: {opp['offer_time']} minutes")
                print()
        else:
            print("❌ No opportunities found above your profit threshold")
    
    def setup_wizard(self):
        """Interactive setup wizard"""
        print("="*80)
        print("🧙 SETUP WIZARD")
        print("="*80)
        print()
        
        print("Let's configure your bot for optimal performance!")
        print()
        
        # Step 1: Update interval
        print("STEP 1: Update Frequency")
        print("─"*80)
        print("How often should the bot check for opportunities?")
        print("  1. Every 3 minutes (Aggressive)")
        print("  2. Every 5 minutes (Balanced) - Recommended")
        print("  3. Every 10 minutes (Conservative)")
        print("  4. Every 15 minutes (Passive)")
        
        choice = input("\nChoose [1-4, default: 2]: ") or "2"
        intervals = {"1": 180, "2": 300, "3": 600, "4": 900}
        self.monitor.update_interval = intervals.get(choice, 300)
        
        print(f"✅ Set to {self.monitor.update_interval/60:.0f} minutes")
        print()
        
        # Step 2: Profit threshold
        print("STEP 2: Minimum Profit")
        print("─"*80)
        print("Minimum profit to alert you?")
        
        min_profit = input("Enter amount in GP [default: 100000]: ") or "100000"
        self.config.config['min_profit_threshold'] = int(min_profit)
        self.config.save_config()
        
        print(f"✅ Set to {int(min_profit):,} GP")
        print()
        
        # Step 3: GE Slots
        print("STEP 3: GE Slot Configuration")
        print("─"*80)
        print("Load a preset slot configuration?")
        print("  1. High-Value Only (all slots 10M+)")
        print("  2. Balanced (mix of small/medium/large)")
        print("  3. Skip (use current config)")
        
        choice = input("\nChoose [1-3, default: 2]: ") or "2"
        
        if choice == "1":
            self.config.load_preset('high_value_only')
        elif choice == "2":
            self.config.load_preset('balanced')
        
        print("✅ Slot configuration set")
        print()
        
        # Step 4: Discord notifications
        print("STEP 4: Discord Notifications (Optional)")
        print("─"*80)
        print("Want Discord notifications?")
        
        webhook = input("Enter Discord webhook URL (or press Enter to skip): ")
        if webhook:
            self.notifier.discord_webhook = webhook
            print("✅ Discord notifications enabled")
        else:
            print("⏭️  Skipped")
        
        print()
        
        # Step 5: Blocklist
        print("STEP 5: Blocklist")
        print("─"*80)
        print(f"Current blocklist: {len(self.config.get_blocklist())} items")
        
        import_fc = input("Import from FlippingCopilot? (y/n): ").lower()
        if import_fc == 'y':
            path = input("Enter path to FlippingCopilot profile JSON: ")
            self.config.import_blocklist_from_copilot(path)
        
        print()
        
        print("="*80)
        print("✅ SETUP COMPLETE!")
        print("="*80)
        print()
        print("Your bot is ready to run!")
        print()
        print("Start with:")
        print("  python3 flip_bot.py --start")
        print()
    
    def show_status(self):
        """Show current bot status"""
        print("="*80)
        print("📊 BOT STATUS")
        print("="*80)
        print()
        
        print("Configuration:")
        print(f"  Update Interval: {self.monitor.update_interval/60:.0f} minutes")
        print(f"  Min Profit: {self.config.config.get('min_profit_threshold', 100000):,} GP")
        print(f"  Watchlist: {len(self.monitor.watchlist)} items")
        print(f"  Blocklist: {len(self.config.get_blocklist())} items")
        print(f"  Discord: {'✅ Enabled' if self.notifier.discord_webhook else '❌ Disabled'}")
        print()
        
        print(self.config.get_slot_summary())
        print()


def main():
    parser = argparse.ArgumentParser(description='OSRS Flip Bot - Full Automation')
    
    parser.add_argument(
        '--start',
        action='store_true',
        help='Start automated monitoring'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Run single check (no continuous monitoring)'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run setup wizard'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show bot status'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default='/mnt/user-data/uploads/flips.csv',
        help='Path to flips CSV file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='user_config.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--discord',
        type=str,
        help='Discord webhook URL for notifications'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Update interval in seconds (default: 300 = 5 minutes)'
    )
    
    args = parser.parse_args()
    
    # Create bot
    bot = FlipBot(
        csv_path=args.csv,
        config_file=args.config,
        discord_webhook=args.discord
    )
    
    if args.interval != 300:
        bot.monitor.update_interval = args.interval
    
    # Execute command
    if args.setup:
        bot.setup_wizard()
    
    elif args.start:
        bot.run_automated()
    
    elif args.check:
        bot.run_single_check()
    
    elif args.status:
        bot.show_status()
    
    else:
        # Default: show help
        print("OSRS Flip Bot - Usage:")
        print()
        print("  --setup     Run setup wizard")
        print("  --start     Start automated monitoring")
        print("  --check     Run single check")
        print("  --status    Show bot status")
        print()
        print("Example:")
        print("  python3 flip_bot.py --setup")
        print("  python3 flip_bot.py --start")
        print("  python3 flip_bot.py --start --discord YOUR_WEBHOOK_URL")
        print()


if __name__ == "__main__":
    main()
