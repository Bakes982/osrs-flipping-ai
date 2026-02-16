#!/usr/bin/env python3
"""
OSRS Investment Finder - Analyzes news, updates, and market trends for investment opportunities
"""

import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd

class InvestmentFinder:
    """
    Analyzes OSRS news and market data to find investment opportunities
    based on upcoming content, nerfs/buffs, and market trends
    """
    
    def __init__(self):
        # Keywords that signal investment opportunities
        self.buff_keywords = [
            'buff', 'buffed', 'buffing', 'increase', 'increased', 'stronger',
            'better', 'improved', 'enhancement', 'boost', 'boosted'
        ]
        
        self.nerf_keywords = [
            'nerf', 'nerfed', 'nerfing', 'decrease', 'decreased', 'weaker',
            'worse', 'reduced', 'reduction', 'downgrade'
        ]
        
        self.content_keywords = [
            'raid', 'boss', 'quest', 'update', 'new content', 'upcoming',
            'release', 'beta', 'leak', 'announcement', 'poll', 'wilderness'
        ]
        
        # Item categories that typically spike with content
        self.investment_categories = {
            'raid_prep': {
                'keywords': ['raid', 'pvm', 'boss'],
                'items': [
                    'Twisted bow', 'Scythe of vitur', 'Shadow of tumeken',
                    'Dragon hunter lance', 'Dragon hunter crossbow',
                    'Avernic defender', 'Lightbearer', 'Ring of suffering',
                    'Anguish', 'Tormented bracelet', 'Ancestral pieces',
                    'Super combat potion', 'Ranging potion', 'Prayer potion',
                    'Stamina potion', 'Anglerfish'
                ]
            },
            'wilderness': {
                'keywords': ['wilderness', 'pvp', 'pking'],
                'items': [
                    'Black d\'hide', 'Amulet of fury', 'Glory', 'Dragon claws',
                    'AGS', 'Granite maul', 'Looting bag', 'Ring of wealth',
                    'Blighted supplies', 'Revenant ether'
                ]
            },
            'skilling': {
                'keywords': ['skill', 'training', 'xp', 'method'],
                'items': [
                    'Dragon pickaxe', 'Dragon axe', 'Crystal pickaxe',
                    'Dragon harpoon', 'Runecrafting supplies', 'Essence'
                ]
            },
            'quest_rewards': {
                'keywords': ['quest', 'grandmaster', 'master'],
                'items': [
                    'Quest requirements', 'Skill requirements'
                ]
            }
        }
    
    def analyze_news_text(self, news_text: str, news_date: datetime = None) -> List[Dict]:
        """
        Analyze a news post/update for investment signals
        
        Args:
            news_text: Text content of news post
            news_date: Date of the news
        
        Returns: List of investment opportunities detected
        """
        opportunities = []
        news_text_lower = news_text.lower()
        
        # Check for buffs
        for keyword in self.buff_keywords:
            if keyword in news_text_lower:
                # Try to extract item names near the keyword
                context = self._extract_context(news_text, keyword, radius=50)
                
                opportunities.append({
                    'signal_type': 'BUFF',
                    'keyword': keyword,
                    'context': context,
                    'date': news_date or datetime.now(),
                    'confidence': 'MEDIUM',
                    'action': 'BUY',
                    'reasoning': f'Item being buffed - likely to increase in demand'
                })
        
        # Check for nerfs
        for keyword in self.nerf_keywords:
            if keyword in news_text_lower:
                context = self._extract_context(news_text, keyword, radius=50)
                
                opportunities.append({
                    'signal_type': 'NERF',
                    'keyword': keyword,
                    'context': context,
                    'date': news_date or datetime.now(),
                    'confidence': 'MEDIUM',
                    'action': 'SELL',
                    'reasoning': f'Item being nerfed - likely to decrease in demand'
                })
        
        # Check for new content
        for keyword in self.content_keywords:
            if keyword in news_text_lower:
                # Determine content type and suggest related items
                for category, data in self.investment_categories.items():
                    if any(kw in news_text_lower for kw in data['keywords']):
                        opportunities.append({
                            'signal_type': 'NEW_CONTENT',
                            'keyword': keyword,
                            'category': category,
                            'suggested_items': data['items'],
                            'date': news_date or datetime.now(),
                            'confidence': 'HIGH',
                            'action': 'BUY',
                            'reasoning': f'New {category.replace("_", " ")} content - demand spike expected'
                        })
        
        return opportunities
    
    def _extract_context(self, text: str, keyword: str, radius: int = 50) -> str:
        """Extract text around a keyword"""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        idx = text_lower.find(keyword_lower)
        if idx == -1:
            return ""
        
        start = max(0, idx - radius)
        end = min(len(text), idx + len(keyword) + radius)
        
        return text[start:end]
    
    def detect_market_trends(self, analyzer, days: int = 7) -> List[Dict]:
        """
        Detect unusual market activity that might signal investments
        
        Args:
            analyzer: FlipDataAnalyzer instance
            days: Number of days to analyze
        
        Returns: List of trending items
        """
        # Get recent flips
        recent_date = datetime.now() - timedelta(days=days)
        recent_flips = analyzer.finished_df[
            analyzer.finished_df['Last sell time'] >= recent_date
        ]
        
        if len(recent_flips) == 0:
            return []
        
        # Analyze each item
        trending = []
        
        for item_name in recent_flips['Item'].unique():
            item_data = recent_flips[recent_flips['Item'] == item_name]
            
            # Calculate trend metrics
            price_trend = self._calculate_price_trend(item_data)
            volume_trend = len(item_data)
            avg_profit = item_data['Profit'].mean()
            
            # Detect upward price movement
            if price_trend > 0.05:  # 5% increase
                trending.append({
                    'item_name': item_name,
                    'trend_type': 'PRICE_INCREASE',
                    'price_change_pct': price_trend * 100,
                    'recent_volume': volume_trend,
                    'avg_profit': avg_profit,
                    'confidence': 'HIGH' if price_trend > 0.10 else 'MEDIUM',
                    'action': 'CONSIDER_BUY',
                    'reasoning': f'Price increased {price_trend*100:.1f}% in last {days} days'
                })
            
            # Detect increased trading volume
            if volume_trend >= 5:  # 5+ flips in period
                trending.append({
                    'item_name': item_name,
                    'trend_type': 'VOLUME_SPIKE',
                    'recent_volume': volume_trend,
                    'avg_profit': avg_profit,
                    'confidence': 'MEDIUM',
                    'action': 'MONITOR',
                    'reasoning': f'High trading activity ({volume_trend} flips in {days} days)'
                })
        
        return trending
    
    def _calculate_price_trend(self, item_data: pd.DataFrame) -> float:
        """Calculate price trend (% change)"""
        if len(item_data) < 2:
            return 0.0
        
        # Sort by time
        sorted_data = item_data.sort_values('Last sell time')
        
        # Compare first half vs second half
        midpoint = len(sorted_data) // 2
        early_avg = sorted_data.iloc[:midpoint]['Avg. sell price'].mean()
        recent_avg = sorted_data.iloc[midpoint:]['Avg. sell price'].mean()
        
        if early_avg == 0:
            return 0.0
        
        return (recent_avg - early_avg) / early_avg
    
    def get_investment_watchlist(
        self,
        analyzer,
        news_opportunities: List[Dict] = None,
        min_confidence: str = 'MEDIUM'
    ) -> pd.DataFrame:
        """
        Generate a comprehensive investment watchlist
        
        Args:
            analyzer: FlipDataAnalyzer instance
            news_opportunities: Opportunities from news analysis
            min_confidence: Minimum confidence level (LOW/MEDIUM/HIGH)
        
        Returns: DataFrame of investment opportunities
        """
        confidence_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        min_conf_level = confidence_levels[min_confidence]
        
        all_opportunities = []
        
        # Add market trends
        trends = self.detect_market_trends(analyzer, days=7)
        for trend in trends:
            if confidence_levels[trend['confidence']] >= min_conf_level:
                all_opportunities.append({
                    'item_name': trend['item_name'],
                    'opportunity_type': trend['trend_type'],
                    'confidence': trend['confidence'],
                    'action': trend['action'],
                    'reasoning': trend['reasoning'],
                    'source': 'Market Analysis',
                    'priority': confidence_levels[trend['confidence']]
                })
        
        # Add news-based opportunities
        if news_opportunities:
            for opp in news_opportunities:
                if confidence_levels[opp['confidence']] >= min_conf_level:
                    # If suggested items list
                    if 'suggested_items' in opp:
                        for item in opp['suggested_items']:
                            all_opportunities.append({
                                'item_name': item,
                                'opportunity_type': opp['signal_type'],
                                'confidence': opp['confidence'],
                                'action': opp['action'],
                                'reasoning': opp['reasoning'],
                                'source': 'News Analysis',
                                'priority': confidence_levels[opp['confidence']]
                            })
                    else:
                        all_opportunities.append({
                            'item_name': 'See context',
                            'opportunity_type': opp['signal_type'],
                            'confidence': opp['confidence'],
                            'action': opp['action'],
                            'reasoning': opp['reasoning'],
                            'source': 'News Analysis',
                            'context': opp.get('context', ''),
                            'priority': confidence_levels[opp['confidence']]
                        })
        
        # Convert to DataFrame
        if not all_opportunities:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_opportunities)
        df = df.sort_values('priority', ascending=False)
        
        return df
    
    def analyze_manual_news(self, news_text: str) -> Dict:
        """
        Analyze user-provided news text
        
        Args:
            news_text: Text to analyze (e.g., pasted from OSRS news)
        
        Returns: Analysis results
        """
        opportunities = self.analyze_news_text(news_text)
        
        result = {
            'total_signals': len(opportunities),
            'buff_signals': len([o for o in opportunities if o['signal_type'] == 'BUFF']),
            'nerf_signals': len([o for o in opportunities if o['signal_type'] == 'NERF']),
            'content_signals': len([o for o in opportunities if o['signal_type'] == 'NEW_CONTENT']),
            'opportunities': opportunities
        }
        
        return result


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("INVESTMENT FINDER TEST")
    print("="*80)
    print()
    
    finder = InvestmentFinder()
    
    # Test 1: Analyze example news
    print("TEST 1: Analyzing example news text...")
    print()
    
    example_news = """
    Jagex announces Raids 4 coming in March 2026!
    
    The new raid will feature challenging boss mechanics requiring
    high-level PvM gear. Players should prepare their best equipment
    including Twisted bow, Shadow, and other endgame items.
    
    Additionally, we're buffing the Dragon hunter lance to make it
    more viable for dragon-based content. The accuracy bonus will
    increase by 15%.
    
    Due to balancing concerns, we're reducing the effectiveness of
    the Toxic blowpipe in certain situations.
    """
    
    analysis = finder.analyze_manual_news(example_news)
    
    print(f"Found {analysis['total_signals']} signals:")
    print(f"  - Buffs: {analysis['buff_signals']}")
    print(f"  - Nerfs: {analysis['nerf_signals']}")
    print(f"  - New Content: {analysis['content_signals']}")
    print()
    
    print("Detailed Opportunities:")
    for i, opp in enumerate(analysis['opportunities'], 1):
        print(f"\n{i}. {opp['signal_type']} - {opp['action']}")
        print(f"   Confidence: {opp['confidence']}")
        print(f"   Reasoning: {opp['reasoning']}")
        if 'suggested_items' in opp:
            print(f"   Suggested items: {', '.join(opp['suggested_items'][:5])}...")
    
    print()
    print("="*80)
    print("✅ Investment Finder test complete!")
