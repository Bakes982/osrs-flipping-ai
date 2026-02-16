#!/usr/bin/env python3
"""
OSRS AI Flip Finder - Interactive tool for finding long-term flip opportunities
"""

from flip_data_analyzer import FlipDataAnalyzer
from flip_predictor import FlipPredictor
import pandas as pd

class FlipFinder:
    """Interactive flip opportunity finder"""
    
    def __init__(self, csv_path: str):
        print("🤖 OSRS AI Flip Finder")
        print("="*80)
        print()
        
        # Load and analyze data
        print("Loading your flip history...")
        self.analyzer = FlipDataAnalyzer(csv_path)
        
        # Train model
        print()
        print("Training AI model...")
        self.predictor = FlipPredictor()
        self.predictor.train(self.analyzer)
        
        print()
        print("="*80)
        print("✅ Ready to find opportunities!")
        print("="*80)
        print()
    
    def find_opportunities(
        self,
        min_score: int = 20,
        min_time_hours: float = 4,
        max_time_hours: float = 168,
        min_price: int = 10000000,
        max_risk: str = "HIGH",
        top_n: int = 10
    ):
        """Find and display top opportunities"""
        
        opportunities = self.predictor.find_opportunities(
            self.analyzer,
            min_score=min_score,
            min_time_hours=min_time_hours,
            max_time_hours=max_time_hours,
            min_price=min_price,
            max_risk=max_risk
        )
        
        if len(opportunities) == 0:
            print("❌ No opportunities found matching your criteria.")
            print("   Try lowering the min_score or expanding the price/time range.")
            return
        
        print(f"🏆 TOP {min(top_n, len(opportunities))} OPPORTUNITIES")
        print("="*80)
        print()
        
        for i, (idx, row) in enumerate(opportunities.head(top_n).iterrows(), 1):
            print(f"#{i} - {row['item_name']}")
            print(f"{'─'*80}")
            print(f"  📊 Opportunity Score:     {row['opportunity_score']:.0f}/100")
            print(f"  💰 Predicted Profit:      {row['predicted_profit']:,} GP")
            print(f"  ⚠️  Risk Level:            {row['risk_level']}")
            print(f"  ✅ Confidence:            {row['confidence']:.1f}%")
            print()
            print(f"  💵 Avg Price:             {row['avg_price']:,} GP")
            print(f"  ⏱️  Avg Flip Time:         {row['avg_time_hours']:.1f} hours")
            print()
            print(f"  📈 Historical Performance:")
            print(f"     Flips: {row['flip_count']} | Avg Profit: {row['historical_avg_profit']:,} GP | GP/hr: {row['historical_gp_hr']:,}")
            print(f"     Success Rate: {row['success_rate']:.1f}%")
            print()
        
        # Summary statistics
        print("="*80)
        print("📊 SUMMARY")
        print("="*80)
        total_predicted = opportunities.head(top_n)['predicted_profit'].sum()
        avg_time = opportunities.head(top_n)['avg_time_hours'].mean()
        avg_score = opportunities.head(top_n)['opportunity_score'].mean()
        
        print(f"Total Predicted Profit (top {min(top_n, len(opportunities))}): {total_predicted:,} GP")
        print(f"Average Flip Time:                     {avg_time:.1f} hours")
        print(f"Average Opportunity Score:             {avg_score:.1f}/100")
        print()
    
    def interactive_mode(self):
        """Run interactive mode with user input"""
        print("🎮 INTERACTIVE MODE")
        print("="*80)
        print()
        
        while True:
            try:
                print("\nSettings:")
                print("─"*80)
                
                # Get user inputs
                min_score = int(input("Min Opportunity Score (0-100) [default: 20]: ") or "20")
                min_time = float(input("Min Flip Time (hours) [default: 4]: ") or "4")
                max_time = float(input("Max Flip Time (hours) [default: 168]: ") or "168")
                min_price = int(input("Min Item Price (GP) [default: 10000000]: ") or "10000000")
                
                print("\nRisk Level:")
                print("  1. LOW only")
                print("  2. LOW and MEDIUM")
                print("  3. ALL (LOW, MEDIUM, HIGH)")
                risk_choice = input("Choose [1/2/3, default: 3]: ") or "3"
                
                risk_map = {
                    "1": "LOW",
                    "2": "MEDIUM",
                    "3": "HIGH"
                }
                max_risk = risk_map.get(risk_choice, "HIGH")
                
                top_n = int(input("How many opportunities to show? [default: 10]: ") or "10")
                
                print()
                print("="*80)
                print("🔍 Searching...")
                print("="*80)
                print()
                
                # Find opportunities
                self.find_opportunities(
                    min_score=min_score,
                    min_time_hours=min_time,
                    max_time_hours=max_time,
                    min_price=min_price,
                    max_risk=max_risk,
                    top_n=top_n
                )
                
                # Ask to continue
                continue_search = input("\n\nSearch again? (y/n): ").lower()
                if continue_search != 'y':
                    break
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")
    
    def quick_search(self):
        """Quick search with sensible defaults"""
        print("⚡ QUICK SEARCH - BEST LONG-TERM FLIPS")
        print("="*80)
        print()
        print("Settings: 4-48 hours, >10M GP, Min Score 30, All risk levels")
        print()
        
        self.find_opportunities(
            min_score=30,
            min_time_hours=4,
            max_time_hours=48,
            min_price=10000000,
            max_risk="HIGH",
            top_n=15
        )


def main():
    """Main entry point"""
    import sys
    
    csv_path = '/mnt/user-data/uploads/flips.csv'
    
    finder = FlipFinder(csv_path)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick mode
        finder.quick_search()
    elif len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # Interactive mode
        finder.interactive_mode()
    else:
        # Default: run quick search
        finder.quick_search()
        
        print("\n\n💡 TIP: Run with '--interactive' for custom search settings")
        print("   python3 flip_finder.py --interactive")


if __name__ == "__main__":
    main()
