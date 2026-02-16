#!/usr/bin/env python3
"""
OSRS Flip Analyzer - Analyzes your historical flip data to build AI model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

class FlipDataAnalyzer:
    """Analyzes your historical flip data"""
    
    def __init__(self, csv_path: str):
        """Initialize with your flips.csv file"""
        print(f"Loading flip data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.process_data()
        print(f"✅ Loaded {len(self.df)} flips")
        
    def process_data(self):
        """Clean and process the flip data"""
        # Convert timestamps
        self.df['First buy time'] = pd.to_datetime(self.df['First buy time'], errors='coerce')
        self.df['Last sell time'] = pd.to_datetime(self.df['Last sell time'], errors='coerce')
        
        # Calculate metrics
        self.df['Time taken (hours)'] = (
            self.df['Last sell time'] - self.df['First buy time']
        ).dt.total_seconds() / 3600
        
        self.df['GP/hr'] = self.df['Profit'] / self.df['Time taken (hours)'].replace(0, np.nan)
        self.df['ROI %'] = (self.df['Profit'] / (self.df['Bought'] * self.df['Avg. buy price'])) * 100
        self.df['Profit per item'] = self.df['Profit'] / self.df['Bought']
        
        # Filter to finished flips only
        self.finished_df = self.df[self.df['Status'] == 'FINISHED'].copy()
        
        # Remove extreme outliers (< 1 minute)
        self.finished_df = self.finished_df[self.finished_df['Time taken (hours)'] >= 0.0167]
        
    def get_item_statistics(self) -> pd.DataFrame:
        """Get aggregated statistics per item"""
        stats = self.finished_df.groupby('Item').agg({
            'Profit': ['sum', 'mean', 'count', 'std'],
            'GP/hr': ['mean', 'median', 'std'],
            'Time taken (hours)': ['mean', 'median'],
            'ROI %': 'mean',
            'Bought': 'sum',
            'Avg. buy price': 'mean',
            'Profit per item': 'mean'
        }).round(2)
        
        stats.columns = [
            'Total Profit', 'Avg Profit', 'Flip Count', 'Profit StdDev',
            'Avg GP/hr', 'Median GP/hr', 'GP/hr StdDev',
            'Avg Time (hrs)', 'Median Time (hrs)',
            'Avg ROI %', 'Total Qty', 'Avg Price', 'Profit/Item'
        ]
        
        # Calculate success rate (% of flips that were profitable)
        success_rate = self.finished_df.groupby('Item')['Profit'].apply(
            lambda x: (x > 0).sum() / len(x) * 100
        )
        stats['Success Rate %'] = success_rate
        
        return stats
    
    def get_high_value_items(self, min_price: int = 10000000) -> pd.DataFrame:
        """Get items above price threshold"""
        stats = self.get_item_statistics()
        high_value = stats[stats['Avg Price'] >= min_price].copy()
        return high_value.sort_values('Avg GP/hr', ascending=False)
    
    def get_long_term_flips(self, min_hours: float = 4, max_hours: float = 168) -> pd.DataFrame:
        """Get items that typically take 4+ hours to flip"""
        stats = self.get_item_statistics()
        long_term = stats[
            (stats['Avg Time (hrs)'] >= min_hours) & 
            (stats['Avg Time (hrs)'] <= max_hours)
        ].copy()
        return long_term.sort_values('Avg GP/hr', ascending=False)
    
    def get_best_performers(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N items by GP/hr"""
        stats = self.get_item_statistics()
        return stats.nlargest(top_n, 'Avg GP/hr')
    
    def get_worst_performers(self, bottom_n: int = 20) -> pd.DataFrame:
        """Get worst N items"""
        stats = self.get_item_statistics()
        return stats.nsmallest(bottom_n, 'Total Profit')
    
    def calculate_features_for_item(self, item_name: str) -> Dict:
        """Calculate ML features for a specific item"""
        item_flips = self.finished_df[self.finished_df['Item'] == item_name]
        
        if len(item_flips) == 0:
            return None
        
        features = {
            'item_name': item_name,
            
            # Historical performance
            'avg_profit': item_flips['Profit'].mean(),
            'median_profit': item_flips['Profit'].median(),
            'total_profit': item_flips['Profit'].sum(),
            'profit_std': item_flips['Profit'].std(),
            
            # Time metrics
            'avg_time_hours': item_flips['Time taken (hours)'].mean(),
            'median_time_hours': item_flips['Time taken (hours)'].median(),
            'time_std': item_flips['Time taken (hours)'].std(),
            
            # Efficiency
            'avg_gp_hr': item_flips['GP/hr'].mean(),
            'median_gp_hr': item_flips['GP/hr'].median(),
            'peak_gp_hr': item_flips['GP/hr'].max(),
            
            # Success metrics
            'flip_count': len(item_flips),
            'success_rate': (item_flips['Profit'] > 0).mean(),
            'avg_roi': item_flips['ROI %'].mean(),
            
            # Price info
            'avg_price': item_flips['Avg. buy price'].mean(),
            'avg_quantity': item_flips['Bought'].mean(),
            
            # Risk metrics
            'profit_consistency': 1 - (item_flips['Profit'].std() / (abs(item_flips['Profit'].mean()) + 1)),
            'time_consistency': 1 - (item_flips['Time taken (hours)'].std() / (item_flips['Time taken (hours)'].mean() + 1)),
        }
        
        return features
    
    def get_all_item_features(self) -> pd.DataFrame:
        """Calculate features for all items"""
        all_features = []
        
        for item_name in self.finished_df['Item'].unique():
            features = self.calculate_features_for_item(item_name)
            if features:
                all_features.append(features)
        
        return pd.DataFrame(all_features)
    
    def get_daily_summary(self) -> pd.DataFrame:
        """Get daily profit summary"""
        self.finished_df['Date'] = self.finished_df['Last sell time'].dt.date
        
        daily = self.finished_df.groupby('Date').agg({
            'Profit': ['sum', 'count'],
            'GP/hr': 'mean'
        }).round(2)
        
        daily.columns = ['Total Profit', 'Flips', 'Avg GP/hr']
        return daily.sort_values('Total Profit', ascending=False)
    
    def analyze_item_categories(self) -> Dict:
        """Categorize items by performance"""
        stats = self.get_item_statistics()
        
        categories = {
            'elite': stats[
                (stats['Avg GP/hr'] > 1000000) & 
                (stats['Avg Price'] > 10000000)
            ],
            'high_value_good': stats[
                (stats['Avg GP/hr'] > 500000) & 
                (stats['Avg Price'] > 10000000)
            ],
            'long_term_profitable': stats[
                (stats['Avg Time (hrs)'] > 4) &
                (stats['Avg GP/hr'] > 500000)
            ],
            'mass_flip_garbage': stats[
                (stats['Flip Count'] >= 5) &
                (stats['Avg GP/hr'] < 500000)
            ],
            'losers': stats[stats['Total Profit'] < 0],
        }
        
        return categories


# Test the analyzer
if __name__ == "__main__":
    print("="*80)
    print("FLIP DATA ANALYZER TEST")
    print("="*80)
    print()
    
    # Load data
    analyzer = FlipDataAnalyzer('/mnt/user-data/uploads/flips.csv')
    
    print()
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total flips: {len(analyzer.finished_df)}")
    print(f"Unique items: {analyzer.finished_df['Item'].nunique()}")
    print(f"Total profit: {analyzer.finished_df['Profit'].sum():,} GP")
    print(f"Average GP/hr: {analyzer.finished_df['GP/hr'].mean():,.0f}")
    print()
    
    # Top performers
    print("="*80)
    print("TOP 10 ITEMS BY GP/HR")
    print("="*80)
    top_10 = analyzer.get_best_performers(10)
    for i, (item, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {item:40s} | {row['Avg GP/hr']:>10,.0f} GP/hr | {row['Avg Price']:>12,.0f} GP")
    print()
    
    # High-value items
    print("="*80)
    print("HIGH-VALUE ITEMS (>10M)")
    print("="*80)
    high_value = analyzer.get_high_value_items()
    print(f"Found {len(high_value)} items >10M GP")
    for i, (item, row) in enumerate(high_value.head(10).iterrows(), 1):
        print(f"{i:2d}. {item:40s} | {row['Avg GP/hr']:>10,.0f} GP/hr | {row['Flip Count']:>2.0f} flips")
    print()
    
    # Long-term flips
    print("="*80)
    print("LONG-TERM PROFITABLE FLIPS (4-168 hours)")
    print("="*80)
    long_term = analyzer.get_long_term_flips()
    print(f"Found {len(long_term)} items that take 4+ hours")
    for i, (item, row) in enumerate(long_term.head(10).iterrows(), 1):
        print(f"{i:2d}. {item:40s} | {row['Avg Time (hrs)']:>6.1f}h | {row['Avg Profit']:>10,.0f} GP")
    print()
    
    # Categories
    print("="*80)
    print("ITEM CATEGORIES")
    print("="*80)
    categories = analyzer.analyze_item_categories()
    for cat_name, cat_df in categories.items():
        print(f"{cat_name:25s}: {len(cat_df):3d} items")
    print()
    
    print("✅ Analysis complete!")
