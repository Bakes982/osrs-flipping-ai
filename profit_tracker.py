#!/usr/bin/env python3
"""
OSRS Profit Tracker - Analyzes your DINK trades to calculate real P/L
Matches buys with sells, tracks performance, and provides AI insights
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json


def load_dink_trades(filepath: str = "dink_trades.csv") -> pd.DataFrame:
    """Load and parse DINK trades CSV"""
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading trades: {e}")
        return pd.DataFrame()


def match_trades(df: pd.DataFrame) -> List[Dict]:
    """
    Match BUY trades with SELL trades to calculate actual profit.
    Uses FIFO (First In, First Out) matching.
    """
    completed_flips = []

    # Group by player and item
    buy_queue = {}  # {(player, item): [list of buy records]}

    # Sort by timestamp
    df = df.sort_values('timestamp')

    for _, row in df.iterrows():
        player = row.get('player', 'Unknown')
        item = row.get('item_name', 'Unknown')
        item_id = row.get('item_id', 0)
        trade_type = row.get('type', '')
        status = row.get('status', '')

        key = (player, item)

        # Skip cancelled/unknown trades
        if status in ['CANCELLED_BUY', 'CANCELLED_SELL'] or trade_type == 'UNKNOWN':
            continue

        if trade_type == 'BUY' and status == 'BOUGHT':
            # Add to buy queue
            if key not in buy_queue:
                buy_queue[key] = []

            buy_queue[key].append({
                'timestamp': row['timestamp'],
                'quantity': row['quantity'],
                'price': row['price'],
                'item_id': item_id,
                'remaining': row['quantity']  # Track remaining quantity
            })

        elif trade_type == 'SELL' and status == 'SOLD':
            # Try to match with buys (FIFO)
            if key in buy_queue and len(buy_queue[key]) > 0:
                sell_qty = row['quantity']
                sell_price = row['price']
                sell_time = row['timestamp']
                seller_tax = row.get('seller_tax', 0)

                while sell_qty > 0 and len(buy_queue[key]) > 0:
                    buy = buy_queue[key][0]

                    # How much can we match?
                    match_qty = min(sell_qty, buy['remaining'])

                    # Calculate profit for this match
                    buy_total = buy['price'] * match_qty
                    sell_total = sell_price * match_qty

                    # Calculate tax (2% of sell, capped at 5M per item)
                    if seller_tax and match_qty == row['quantity']:
                        # Use actual tax from DINK
                        tax = seller_tax
                    else:
                        # Calculate proportional tax
                        tax_per_item = min(sell_price * 0.02, 5_000_000)
                        tax = tax_per_item * match_qty

                    gross_profit = sell_total - buy_total
                    net_profit = gross_profit - tax

                    # Calculate flip time
                    flip_time = (sell_time - buy['timestamp']).total_seconds() / 3600  # hours

                    completed_flips.append({
                        'player': player,
                        'item_name': item,
                        'item_id': item_id,
                        'quantity': match_qty,
                        'buy_price': buy['price'],
                        'sell_price': sell_price,
                        'buy_total': buy_total,
                        'sell_total': sell_total,
                        'tax': tax,
                        'gross_profit': gross_profit,
                        'net_profit': net_profit,
                        'profit_per_item': net_profit / match_qty if match_qty > 0 else 0,
                        'margin_pct': (net_profit / buy_total * 100) if buy_total > 0 else 0,
                        'buy_time': buy['timestamp'],
                        'sell_time': sell_time,
                        'flip_time_hours': flip_time,
                        'gp_per_hour': net_profit / flip_time if flip_time > 0 else 0
                    })

                    # Update remaining quantities
                    sell_qty -= match_qty
                    buy['remaining'] -= match_qty

                    # Remove exhausted buys
                    if buy['remaining'] <= 0:
                        buy_queue[key].pop(0)

    return completed_flips


def get_active_positions(df: pd.DataFrame) -> List[Dict]:
    """Get items that were bought but not yet sold"""
    buy_queue = {}

    df = df.sort_values('timestamp')

    for _, row in df.iterrows():
        player = row.get('player', 'Unknown')
        item = row.get('item_name', 'Unknown')
        trade_type = row.get('type', '')
        status = row.get('status', '')

        key = (player, item)

        if status in ['CANCELLED_BUY', 'CANCELLED_SELL'] or trade_type == 'UNKNOWN':
            continue

        if trade_type == 'BUY' and status == 'BOUGHT':
            if key not in buy_queue:
                buy_queue[key] = []
            buy_queue[key].append({
                'timestamp': row['timestamp'],
                'quantity': row['quantity'],
                'price': row['price'],
                'item_id': row.get('item_id', 0),
                'remaining': row['quantity']
            })

        elif trade_type == 'SELL' and status == 'SOLD':
            if key in buy_queue:
                sell_qty = row['quantity']
                while sell_qty > 0 and len(buy_queue[key]) > 0:
                    buy = buy_queue[key][0]
                    match_qty = min(sell_qty, buy['remaining'])
                    sell_qty -= match_qty
                    buy['remaining'] -= match_qty
                    if buy['remaining'] <= 0:
                        buy_queue[key].pop(0)

    # Collect remaining positions
    active = []
    for (player, item), buys in buy_queue.items():
        for buy in buys:
            if buy['remaining'] > 0:
                active.append({
                    'player': player,
                    'item_name': item,
                    'item_id': buy['item_id'],
                    'quantity': buy['remaining'],
                    'buy_price': buy['price'],
                    'total_invested': buy['price'] * buy['remaining'],
                    'buy_time': buy['timestamp'],
                    'hold_time_hours': (datetime.now() - buy['timestamp']).total_seconds() / 3600
                })

    return active


def calculate_session_stats(flips: List[Dict], hours: int = 24) -> Dict:
    """Calculate stats for a time period"""
    if not flips:
        return {
            'total_flips': 0,
            'total_profit': 0,
            'total_volume': 0,
            'avg_profit': 0,
            'avg_margin': 0,
            'avg_flip_time': 0,
            'gp_per_hour': 0,
            'win_rate': 0,
            'best_flip': None,
            'worst_flip': None
        }

    # Filter by time
    cutoff = datetime.now() - timedelta(hours=hours)
    recent = [f for f in flips if f['sell_time'] >= cutoff]

    if not recent:
        recent = flips  # Use all if none in timeframe

    total_profit = sum(f['net_profit'] for f in recent)
    total_volume = sum(f['buy_total'] for f in recent)
    profitable = [f for f in recent if f['net_profit'] > 0]

    # Find best and worst
    best = max(recent, key=lambda x: x['net_profit']) if recent else None
    worst = min(recent, key=lambda x: x['net_profit']) if recent else None

    # Calculate total time
    if recent:
        first_buy = min(f['buy_time'] for f in recent)
        last_sell = max(f['sell_time'] for f in recent)
        total_hours = (last_sell - first_buy).total_seconds() / 3600
    else:
        total_hours = 0

    return {
        'total_flips': len(recent),
        'total_profit': total_profit,
        'total_volume': total_volume,
        'avg_profit': total_profit / len(recent) if recent else 0,
        'avg_margin': sum(f['margin_pct'] for f in recent) / len(recent) if recent else 0,
        'avg_flip_time': sum(f['flip_time_hours'] for f in recent) / len(recent) if recent else 0,
        'gp_per_hour': total_profit / total_hours if total_hours > 0 else 0,
        'win_rate': len(profitable) / len(recent) * 100 if recent else 0,
        'best_flip': best,
        'worst_flip': worst
    }


def get_item_performance(flips: List[Dict]) -> pd.DataFrame:
    """Analyze performance by item"""
    if not flips:
        return pd.DataFrame()

    item_stats = {}

    for flip in flips:
        item = flip['item_name']
        if item not in item_stats:
            item_stats[item] = {
                'item_name': item,
                'item_id': flip['item_id'],
                'flip_count': 0,
                'total_profit': 0,
                'total_volume': 0,
                'total_time': 0,
                'wins': 0,
                'losses': 0
            }

        item_stats[item]['flip_count'] += 1
        item_stats[item]['total_profit'] += flip['net_profit']
        item_stats[item]['total_volume'] += flip['buy_total']
        item_stats[item]['total_time'] += flip['flip_time_hours']

        if flip['net_profit'] > 0:
            item_stats[item]['wins'] += 1
        else:
            item_stats[item]['losses'] += 1

    # Calculate averages
    for item in item_stats.values():
        item['avg_profit'] = item['total_profit'] / item['flip_count']
        item['avg_margin'] = (item['total_profit'] / item['total_volume'] * 100) if item['total_volume'] > 0 else 0
        item['avg_time'] = item['total_time'] / item['flip_count']
        item['gp_per_hour'] = item['total_profit'] / item['total_time'] if item['total_time'] > 0 else 0
        item['win_rate'] = item['wins'] / item['flip_count'] * 100

    df = pd.DataFrame(list(item_stats.values()))
    df = df.sort_values('total_profit', ascending=False)

    return df


def generate_ai_insights(flips: List[Dict], active: List[Dict]) -> List[Dict]:
    """Generate AI-powered insights based on trading patterns"""
    insights = []

    if not flips:
        insights.append({
            'type': 'INFO',
            'title': 'No completed flips yet',
            'message': 'Complete some flips to get AI insights!'
        })
        return insights

    # Get item performance
    item_perf = get_item_performance(flips)

    # Insight 1: Best performing items
    if not item_perf.empty:
        best_items = item_perf.head(3)
        for _, item in best_items.iterrows():
            if item['total_profit'] > 100000:  # At least 100k profit
                insights.append({
                    'type': 'SUCCESS',
                    'title': f"Strong performer: {item['item_name']}",
                    'message': f"{item['flip_count']} flips, {item['total_profit']:,.0f} GP profit, {item['win_rate']:.0f}% win rate. Consider flipping more of this!"
                })

    # Insight 2: Items to avoid
    if not item_perf.empty:
        losers = item_perf[item_perf['total_profit'] < 0]
        for _, item in losers.iterrows():
            insights.append({
                'type': 'WARNING',
                'title': f"Losing item: {item['item_name']}",
                'message': f"Lost {abs(item['total_profit']):,.0f} GP over {item['flip_count']} flips. Consider avoiding this item."
            })

    # Insight 3: Session stats
    stats = calculate_session_stats(flips, hours=24)
    if stats['total_flips'] > 0:
        insights.append({
            'type': 'INFO',
            'title': 'Today\'s Performance',
            'message': f"{stats['total_flips']} flips, {stats['total_profit']:,.0f} GP profit, {stats['gp_per_hour']:,.0f} GP/hr, {stats['win_rate']:.0f}% win rate"
        })

    # Insight 4: Flip time analysis
    if flips:
        avg_time = sum(f['flip_time_hours'] for f in flips) / len(flips)
        fast_flips = [f for f in flips if f['flip_time_hours'] < 1]
        slow_flips = [f for f in flips if f['flip_time_hours'] > 4]

        if fast_flips:
            fast_profit = sum(f['net_profit'] for f in fast_flips)
            insights.append({
                'type': 'INFO',
                'title': 'Quick Flip Analysis',
                'message': f"{len(fast_flips)} flips under 1 hour = {fast_profit:,.0f} GP. {'Quick flipping is working well!' if fast_profit > 0 else 'Quick flips are losing money - try patience.'}"
            })

        if slow_flips:
            slow_profit = sum(f['net_profit'] for f in slow_flips)
            insights.append({
                'type': 'INFO',
                'title': 'Patient Flip Analysis',
                'message': f"{len(slow_flips)} flips over 4 hours = {slow_profit:,.0f} GP. {'Patient flipping pays off!' if slow_profit > 0 else 'Long holds are risky for you.'}"
            })

    # Insight 5: Active position warnings
    for pos in active:
        if pos['hold_time_hours'] > 6:
            insights.append({
                'type': 'WARNING',
                'title': f"Stale position: {pos['item_name']}",
                'message': f"Held for {pos['hold_time_hours']:.1f} hours. {pos['quantity']}x @ {pos['buy_price']:,} GP. Consider cutting losses or adjusting price."
            })

    # Insight 6: Margin analysis
    if flips:
        high_margin = [f for f in flips if f['margin_pct'] > 3]
        low_margin = [f for f in flips if f['margin_pct'] < 1]

        if high_margin:
            hm_profit = sum(f['net_profit'] for f in high_margin)
            insights.append({
                'type': 'SUCCESS' if hm_profit > 0 else 'WARNING',
                'title': 'High Margin Flips (>3%)',
                'message': f"{len(high_margin)} trades = {hm_profit:,.0f} GP. {'High margins are profitable!' if hm_profit > 0 else 'High margin items may be illiquid traps.'}"
            })

    # Insight 7: Tax impact
    total_tax = sum(f['tax'] for f in flips)
    total_gross = sum(f['gross_profit'] for f in flips)
    if total_gross > 0:
        tax_pct = (total_tax / total_gross) * 100
        insights.append({
            'type': 'INFO',
            'title': 'Tax Impact',
            'message': f"Paid {total_tax:,.0f} GP in GE tax ({tax_pct:.1f}% of gross profit). High-value items have lower tax impact due to 5M cap."
        })

    return insights


def get_recommendations_from_history(flips: List[Dict], active: List[Dict]) -> List[Dict]:
    """Generate trading recommendations based on your history"""
    recommendations = []

    if not flips:
        return recommendations

    item_perf = get_item_performance(flips)

    # Recommend items you're good at
    if not item_perf.empty:
        top_items = item_perf[
            (item_perf['total_profit'] > 0) &
            (item_perf['flip_count'] >= 1) &  # Lower threshold for new users
            (item_perf['win_rate'] >= 50)
        ].head(5)

        for _, item in top_items.iterrows():
            # Check if we already have active position
            has_active = any(a['item_name'] == item['item_name'] for a in active)

            recommendations.append({
                'item_name': item['item_name'],
                'item_id': item['item_id'],
                'reason': 'PROVEN_WINNER',
                'flip_count': item['flip_count'],
                'total_profit': item['total_profit'],
                'avg_profit': item['avg_profit'],
                'win_rate': item['win_rate'],
                'avg_time': item['avg_time'],
                'has_active': has_active,
                'confidence': 'HIGH' if item['flip_count'] >= 3 else 'MEDIUM' if item['flip_count'] >= 2 else 'LOW'
            })

    return recommendations


def print_summary(filepath: str = "dink_trades.csv"):
    """Print a comprehensive trading summary"""
    df = load_dink_trades(filepath)

    if df.empty:
        print("No trades found!")
        return

    print("=" * 80)
    print("OSRS FLIP PROFIT TRACKER")
    print("=" * 80)

    # Match trades
    flips = match_trades(df)
    active = get_active_positions(df)

    print(f"\nTotal trades in log: {len(df)}")
    print(f"Completed flips: {len(flips)}")
    print(f"Active positions: {len(active)}")

    # Session stats
    print("\n" + "-" * 40)
    print("SESSION STATS (Last 24h)")
    print("-" * 40)

    stats = calculate_session_stats(flips, hours=24)
    print(f"Flips: {stats['total_flips']}")
    print(f"Total Profit: {stats['total_profit']:,.0f} GP")
    print(f"Total Volume: {stats['total_volume']:,.0f} GP")
    print(f"Avg Profit/Flip: {stats['avg_profit']:,.0f} GP")
    print(f"Avg Margin: {stats['avg_margin']:.2f}%")
    print(f"GP/Hour: {stats['gp_per_hour']:,.0f}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")

    if stats['best_flip']:
        print(f"\nBest Flip: {stats['best_flip']['item_name']} = {stats['best_flip']['net_profit']:,.0f} GP")
    if stats['worst_flip']:
        print(f"Worst Flip: {stats['worst_flip']['item_name']} = {stats['worst_flip']['net_profit']:,.0f} GP")

    # Item performance
    print("\n" + "-" * 40)
    print("TOP ITEMS BY PROFIT")
    print("-" * 40)

    item_perf = get_item_performance(flips)
    if not item_perf.empty:
        for _, item in item_perf.head(10).iterrows():
            print(f"{item['item_name']:40} | {item['total_profit']:>12,.0f} GP | {item['flip_count']} flips | {item['win_rate']:.0f}% win")

    # Active positions
    if active:
        print("\n" + "-" * 40)
        print("ACTIVE POSITIONS")
        print("-" * 40)

        for pos in active:
            print(f"{pos['item_name']:40} | {pos['quantity']}x @ {pos['buy_price']:,} | {pos['hold_time_hours']:.1f}h held")

    # AI Insights
    print("\n" + "-" * 40)
    print("AI INSIGHTS")
    print("-" * 40)

    insights = generate_ai_insights(flips, active)
    for insight in insights:
        icon = {'SUCCESS': '[OK]', 'WARNING': '[!]', 'INFO': '[i]'}.get(insight['type'], '-')
        print(f"\n{icon} {insight['title']}")
        print(f"   {insight['message']}")


if __name__ == "__main__":
    print_summary()
