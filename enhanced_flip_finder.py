#!/usr/bin/env python3
"""
Enhanced OSRS AI Flip Finder - With investments, blocklist, offer times, and slot allocation
"""

from flip_data_analyzer import FlipDataAnalyzer
from flip_predictor import FlipPredictor
from investment_finder import InvestmentFinder
from user_config import UserConfig
import pandas as pd
from typing import List, Dict

class EnhancedFlipFinder:
    """Enhanced flip finder with all advanced features"""
    
    def __init__(self, csv_path: str, config_file: str = "user_config.json"):
        print("🤖 ENHANCED OSRS AI FLIP FINDER")
        print("="*80)
        print()
        
        # Load components
        print("Loading your flip history...")
        self.analyzer = FlipDataAnalyzer(csv_path)
        
        print()
        print("Loading user configuration...")
        self.config = UserConfig(config_file)
        
        print()
        print("Training AI model...")
        self.predictor = FlipPredictor()
        self.predictor.train(self.analyzer)
        
        print()
        print("Initializing investment finder...")
        self.investment_finder = InvestmentFinder()
        
        print()
        print("="*80)
        print("✅ Ready with enhanced features!")
        print("="*80)
        print()
    
    def find_opportunities_with_slots(
        self,
        min_score: int = 20,
        min_time_hours: float = 4,
        max_time_hours: float = 168,
        min_price: int = 10000000,
        max_risk: str = "HIGH",
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Find opportunities and match them to appropriate GE slots
        """
        # Get base opportunities
        opportunities = self.predictor.find_opportunities(
            self.analyzer,
            min_score=min_score,
            min_time_hours=min_time_hours,
            max_time_hours=max_time_hours,
            min_price=min_price,
            max_risk=max_risk
        )
        
        if len(opportunities) == 0:
            return opportunities
        
        # Filter by blocklist
        opportunities = opportunities[
            ~opportunities['item_name'].isin(self.config.get_blocklist())
        ]
        
        # Add slot assignments
        opportunities['available_slots'] = opportunities['avg_price'].apply(
            lambda x: self.config.get_available_slots_for_item(int(x))
        )
        
        # Add offer time recommendations
        opportunities['recommended_offer_time'] = opportunities['item_name'].apply(
            lambda x: self.config.get_offer_time(x)
        )
        
        return opportunities.head(top_n)
    
    def display_opportunities(self, opportunities: pd.DataFrame):
        """Display opportunities with enhanced information"""
        if len(opportunities) == 0:
            print("❌ No opportunities found matching your criteria.")
            return
        
        print(f"🏆 TOP {len(opportunities)} OPPORTUNITIES")
        print("="*80)
        print()
        
        for i, (idx, row) in enumerate(opportunities.iterrows(), 1):
            print(f"#{i} - {row['item_name']}")
            print(f"{'─'*80}")
            print(f"  📊 Opportunity Score:     {row['opportunity_score']:.0f}/100")
            print(f"  💰 Predicted Profit:      {row['predicted_profit']:,} GP")
            print(f"  ⚠️  Risk Level:            {row['risk_level']}")
            print(f"  ✅ Confidence:            {row['confidence']:.1f}%")
            print()
            print(f"  💵 Avg Price:             {row['avg_price']:,} GP")
            print(f"  ⏱️  Avg Flip Time:         {row['avg_time_hours']:.1f} hours")
            print(f"  ⏰ Offer Time:            {row['recommended_offer_time']} minutes")
            print(f"  🎰 Available GE Slots:    {', '.join(map(str, row['available_slots']))}")
            print()
            print(f"  📈 Historical Performance:")
            print(f"     Flips: {row['flip_count']} | Avg Profit: {row['historical_avg_profit']:,} GP")
            print(f"     GP/hr: {row['historical_gp_hr']:,} | Success Rate: {row['success_rate']:.1f}%")
            print()
    
    def find_investment_opportunities(self, news_text: str = None):
        """Find investment opportunities from news or market trends"""
        print("="*80)
        print("🔍 INVESTMENT OPPORTUNITY FINDER")
        print("="*80)
        print()
        
        # Analyze news if provided
        news_opps = []
        if news_text:
            print("Analyzing provided news...")
            analysis = self.investment_finder.analyze_manual_news(news_text)
            news_opps = analysis['opportunities']
            
            print(f"Found {analysis['total_signals']} signals in news:")
            print(f"  ✅ Buffs: {analysis['buff_signals']}")
            print(f"  ⚠️  Nerfs: {analysis['nerf_signals']}")
            print(f"  📰 New Content: {analysis['content_signals']}")
            print()
        
        # Get market trends
        print("Analyzing market trends from your flip history...")
        watchlist = self.investment_finder.get_investment_watchlist(
            self.analyzer,
            news_opportunities=news_opps,
            min_confidence='MEDIUM'
        )
        
        if len(watchlist) == 0:
            print("❌ No investment opportunities detected")
            return
        
        # Filter by blocklist
        watchlist = watchlist[
            ~watchlist['item_name'].isin(self.config.get_blocklist())
        ]
        
        print(f"📊 INVESTMENT WATCHLIST ({len(watchlist)} opportunities)")
        print("="*80)
        print()
        
        for i, (idx, row) in enumerate(watchlist.head(15).iterrows(), 1):
            print(f"#{i} - {row['item_name']}")
            print(f"  Type: {row['opportunity_type']}")
            print(f"  Action: {row['action']}")
            print(f"  Confidence: {row['confidence']}")
            print(f"  Reasoning: {row['reasoning']}")
            print(f"  Source: {row['source']}")
            print()
    
    def manage_blocklist(self):
        """Interactive blocklist management"""
        while True:
            print("="*80)
            print("📋 BLOCKLIST MANAGEMENT")
            print("="*80)
            print()
            print(f"Current blocklist: {len(self.config.get_blocklist())} items")
            print()
            print("Options:")
            print("  1. View blocklist")
            print("  2. Add items to blocklist")
            print("  3. Remove items from blocklist")
            print("  4. Clear entire blocklist")
            print("  5. Import from FlippingCopilot")
            print("  0. Back to main menu")
            print()
            
            choice = input("Choose option: ")
            
            if choice == "1":
                blocklist = self.config.get_blocklist()
                if not blocklist:
                    print("Blocklist is empty")
                else:
                    print(f"\nBlocked items ({len(blocklist)}):")
                    for i, item in enumerate(blocklist, 1):
                        print(f"  {i:3d}. {item}")
                input("\nPress Enter to continue...")
            
            elif choice == "2":
                items_str = input("Enter item names (comma-separated): ")
                items = [item.strip() for item in items_str.split(',')]
                self.config.add_to_blocklist(items)
            
            elif choice == "3":
                items_str = input("Enter item names to remove (comma-separated): ")
                items = [item.strip() for item in items_str.split(',')]
                self.config.remove_from_blocklist(items)
            
            elif choice == "4":
                confirm = input("Are you sure you want to clear the entire blocklist? (yes/no): ")
                if confirm.lower() == 'yes':
                    self.config.clear_blocklist()
            
            elif choice == "5":
                path = input("Enter path to FlippingCopilot profile JSON: ")
                self.config.import_blocklist_from_copilot(path)
            
            elif choice == "0":
                break
    
    def manage_slot_configuration(self):
        """Interactive slot configuration"""
        while True:
            print("="*80)
            print("🎰 GE SLOT CONFIGURATION")
            print("="*80)
            print()
            print(self.config.get_slot_summary())
            print()
            print("Options:")
            print("  1. Configure a slot")
            print("  2. Set investment slot")
            print("  3. Load preset (high-value only, balanced, etc.)")
            print("  0. Back to main menu")
            print()
            
            choice = input("Choose option: ")
            
            if choice == "1":
                slot = int(input("Slot number (1-8): "))
                min_price = int(input("Minimum price (GP): "))
                max_price = int(input("Maximum price (GP): "))
                purpose = input("Purpose/label: ")
                self.config.configure_slot(slot, min_price, max_price, purpose)
            
            elif choice == "2":
                slot = int(input("Which slot for investments (1-8): "))
                self.config.set_investment_slot(slot)
            
            elif choice == "3":
                print("\nAvailable presets:")
                print("  - conservative")
                print("  - balanced")
                print("  - aggressive")
                print("  - high_value_only")
                preset = input("Choose preset: ")
                self.config.load_preset(preset)
            
            elif choice == "0":
                break
    
    def manage_offer_times(self):
        """Interactive offer time management"""
        while True:
            print("="*80)
            print("⏰ OFFER TIME SETTINGS")
            print("="*80)
            print()
            print(f"Default offer time: {self.config.config['offer_time_settings']['default']} minutes")
            custom_times = self.config.config['offer_time_settings'].get('per_item', {})
            print(f"Custom overrides: {len(custom_times)} items")
            print()
            print("Options:")
            print("  1. Set default offer time")
            print("  2. Set custom time for specific item")
            print("  3. View custom times")
            print("  4. Remove custom time")
            print("  0. Back to main menu")
            print()
            
            choice = input("Choose option: ")
            
            if choice == "1":
                minutes = int(input("Default offer time (minutes): "))
                self.config.set_default_offer_time(minutes)
            
            elif choice == "2":
                item = input("Item name: ")
                minutes = int(input(f"Offer time for {item} (minutes): "))
                self.config.set_offer_time_for_item(item, minutes)
            
            elif choice == "3":
                if not custom_times:
                    print("No custom times set")
                else:
                    for item, minutes in custom_times.items():
                        print(f"  {item}: {minutes} minutes")
                input("\nPress Enter to continue...")
            
            elif choice == "4":
                item = input("Item name to remove: ")
                self.config.remove_offer_time_override(item)
            
            elif choice == "0":
                break
    
    def interactive_menu(self):
        """Main interactive menu"""
        while True:
            print("\n")
            print("="*80)
            print("🤖 ENHANCED FLIP FINDER - MAIN MENU")
            print("="*80)
            print()
            print("1. Find Regular Flip Opportunities")
            print("2. Find Investment Opportunities")
            print("3. Analyze News for Investments")
            print("4. Manage Blocklist")
            print("5. Configure GE Slots")
            print("6. Configure Offer Times")
            print("7. View Current Configuration")
            print("0. Exit")
            print()
            
            choice = input("Choose option: ")
            
            if choice == "1":
                self.quick_flip_search()
            
            elif choice == "2":
                self.find_investment_opportunities()
                input("\nPress Enter to continue...")
            
            elif choice == "3":
                print("\nPaste OSRS news text (press Enter twice when done):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                news_text = "\n".join(lines)
                self.find_investment_opportunities(news_text)
                input("\nPress Enter to continue...")
            
            elif choice == "4":
                self.manage_blocklist()
            
            elif choice == "5":
                self.manage_slot_configuration()
            
            elif choice == "6":
                self.manage_offer_times()
            
            elif choice == "7":
                self.config.print_current_config()
                input("\nPress Enter to continue...")
            
            elif choice == "0":
                print("\n👋 Goodbye!")
                break
    
    def quick_flip_search(self):
        """Quick search with user settings applied"""
        print()
        print("⚡ QUICK FLIP SEARCH")
        print("="*80)
        print()
        
        opportunities = self.find_opportunities_with_slots(
            min_score=30,
            min_time_hours=4,
            max_time_hours=48,
            min_price=10000000,
            max_risk=self.config.config.get('risk_tolerance', 'MEDIUM'),
            top_n=15
        )
        
        self.display_opportunities(opportunities)
        
        # Summary
        if len(opportunities) > 0:
            print("="*80)
            print("📊 SUMMARY")
            print("="*80)
            total_predicted = opportunities['predicted_profit'].sum()
            avg_time = opportunities['avg_time_hours'].mean()
            avg_score = opportunities['opportunity_score'].mean()
            
            print(f"Total Predicted Profit: {total_predicted:,} GP")
            print(f"Average Flip Time:      {avg_time:.1f} hours")
            print(f"Average Score:          {avg_score:.1f}/100")
            print()


def main():
    """Main entry point"""
    import sys
    
    csv_path = '/mnt/user-data/uploads/flips.csv'
    
    finder = EnhancedFlipFinder(csv_path)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--menu':
        finder.interactive_menu()
    else:
        # Default: quick search
        finder.quick_flip_search()
        print("\n💡 TIP: Run with '--menu' for full interactive mode")
        print("   python3 enhanced_flip_finder.py --menu")


if __name__ == "__main__":
    main()
