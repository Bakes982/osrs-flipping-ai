#!/usr/bin/env python3
"""
OSRS AI Strategist - Advanced market analysis using quantitative data
Designed to work with Claude for deep market reasoning
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from quant_analyzer import get_analyzer, ITEM_CATEGORIES, get_buy_limit, get_item_category

# The advanced system prompt for Claude-based analysis
QUANT_STRATEGIST_PROMPT = """**System Role: OSRS Quantitative Flipping Strategist**

**Objective:** Analyze provided GE data using statistical finance principles (Mean Reversion and Momentum) to identify high-probability flips.

**Data Interpretation Rules:**
1. **Tax Neutralization:** Always verify if `(High_Price * 0.98) - Low_Price > 0`. If not, the margin is an illusion (2% GE tax).
2. **The Z-Score Filter:**
   - Z < -1.5: "Oversold/Dip" - Strong Buy if volume is stable.
   - Z > 1.5: "Overbought/Spike" - Do not enter; price will likely revert to the mean.
3. **Volume Liquidity Check:** If `Current_Volume < (24h_Avg_Volume / 24)`, flag the item as "Illiquid." High margins on illiquid items are traps.
4. **Trend Analysis:** Compare `Current_Price` to `1h_SMA`. If `Current_Price > 1h_SMA`, the trend is bullish.
5. **Buy Limit Consideration:** A 1M profit margin on an item with limit of 8 per 4 hours is LESS valuable than 100k margin on limit of 10,000.

**Risk Factors to Consider:**
- Price spike (Z > 2.0) = HIGH RISK - likely to crash
- Illiquid (Volume Velocity < 0.5) = TRAP RISK
- RSI > 70 = Overbought, momentum exhausted
- RSI < 30 = Oversold, potential reversal

**Output Format:**
- **Item Name & Real Margin (Tax-adjusted)**
- **Risk Score (1-10):** Based on Z-score, Volume Velocity, and RSI
- **Strategic Verdict:** (e.g., "Aggressive Flip," "Patient Limit Order," "Quick Flip," or "Avoid - Fake Margin")
- **Reasoning:** 1-2 sentence explanation of the recommendation
"""


def generate_market_report(item_ids: List[int] = None) -> Dict:
    """
    Generate a comprehensive market report for Claude analysis
    """
    analyzer = get_analyzer()

    # Default to high-value items if none specified
    if not item_ids:
        item_ids = ITEM_CATEGORIES.get('high_pvm_gear', [])[:10]

    report = {
        'generated_at': datetime.now().isoformat(),
        'market_summary': {},
        'items': [],
        'alerts': [],
        'category_trends': {}
    }

    all_analyses = []

    for item_id in item_ids:
        try:
            analysis = analyzer.full_analysis(str(item_id))
            if analysis['prices']['high'] and analysis['prices']['low']:
                all_analyses.append(analysis)

                # Get item metadata
                mapping = analyzer.fetch_item_mapping()
                item_info = mapping.get(str(item_id), {})
                item_name = item_info.get('name', f'Item {item_id}')

                item_report = {
                    'id': item_id,
                    'name': item_name,
                    'category': get_item_category(item_id),
                    'buy_limit': get_buy_limit(item_id),
                    'prices': analysis['prices'],
                    'margin': analysis['margin'],
                    'z_score': analysis['z_score'],
                    'rsi': analysis['rsi'],
                    'volume': analysis['volume'],
                    'trend': analysis['trend'],
                    'risk_score': analysis['risk_score'],
                    'verdict': analysis['verdict']
                }

                report['items'].append(item_report)

                # Generate alerts for notable conditions
                if analysis['z_score']['high'] and analysis['z_score']['high'] < -2.0:
                    report['alerts'].append({
                        'type': 'OVERSOLD',
                        'item': item_name,
                        'message': f"{item_name} is significantly oversold (Z={analysis['z_score']['high']:.1f})",
                        'action': 'Consider buying'
                    })
                elif analysis['z_score']['high'] and analysis['z_score']['high'] > 2.0:
                    report['alerts'].append({
                        'type': 'OVERBOUGHT',
                        'item': item_name,
                        'message': f"{item_name} is overbought (Z={analysis['z_score']['high']:.1f})",
                        'action': 'Avoid or sell'
                    })

                if analysis['margin']['profitable'] and analysis['margin']['margin_pct'] > 3:
                    report['alerts'].append({
                        'type': 'HIGH_MARGIN',
                        'item': item_name,
                        'message': f"{item_name} has {analysis['margin']['margin_pct']:.1f}% margin",
                        'action': 'Verify volume before entering'
                    })

        except Exception as e:
            print(f"Error analyzing item {item_id}: {e}")

    # Market summary
    if all_analyses:
        profitable_count = sum(1 for a in all_analyses if a['margin']['profitable'])
        avg_risk = sum(a['risk_score'] for a in all_analyses) / len(all_analyses)

        bullish_count = sum(1 for a in all_analyses if a['trend']['trend'] == 'BULLISH')
        bearish_count = sum(1 for a in all_analyses if a['trend']['trend'] == 'BEARISH')

        report['market_summary'] = {
            'items_analyzed': len(all_analyses),
            'profitable_items': profitable_count,
            'average_risk_score': round(avg_risk, 1),
            'market_sentiment': 'BULLISH' if bullish_count > bearish_count else 'BEARISH' if bearish_count > bullish_count else 'NEUTRAL',
            'bullish_items': bullish_count,
            'bearish_items': bearish_count
        }

    return report


def format_report_for_claude(report: Dict) -> str:
    """
    Format the market report as a prompt for Claude analysis
    """
    lines = [
        "# OSRS GE Market Analysis Report",
        f"Generated: {report['generated_at']}",
        "",
        "## Market Summary",
        f"- Items Analyzed: {report['market_summary'].get('items_analyzed', 0)}",
        f"- Profitable Items: {report['market_summary'].get('profitable_items', 0)}",
        f"- Average Risk Score: {report['market_summary'].get('average_risk_score', 'N/A')}/10",
        f"- Market Sentiment: {report['market_summary'].get('market_sentiment', 'UNKNOWN')}",
        ""
    ]

    if report['alerts']:
        lines.append("## Alerts")
        for alert in report['alerts']:
            lines.append(f"- **{alert['type']}**: {alert['message']} ({alert['action']})")
        lines.append("")

    lines.append("## Item Analysis")
    for item in report['items']:
        lines.append(f"\n### {item['name']}")
        lines.append(f"- Category: {item['category'] or 'Unknown'}")
        lines.append(f"- Buy Limit: {item['buy_limit']:,}/4hr")
        lines.append(f"- High: {item['prices']['high']:,} GP | Low: {item['prices']['low']:,} GP")
        lines.append(f"- Net Margin: {item['margin']['net']:,} GP ({item['margin']['margin_pct']:.1f}%)")
        lines.append(f"- Tax: {item['margin']['tax']:,} GP")
        lines.append(f"- Z-Score: {item['z_score']['high']} ({item['z_score']['interpretation']})")
        lines.append(f"- RSI: {item['rsi']['value']} ({item['rsi']['interpretation']})")
        lines.append(f"- Volume: {item['volume']['interpretation']}")
        lines.append(f"- Trend: {item['trend']['trend']}")
        lines.append(f"- Risk Score: {item['risk_score']}/10")
        lines.append(f"- **Verdict: {item['verdict']['action']}** - {item['verdict']['reason']}")

    return "\n".join(lines)


def get_flip_recommendations(max_risk: int = 8, min_margin_pct: float = 0.5) -> List[Dict]:
    """
    Get filtered flip recommendations based on risk and margin criteria
    Now scans ALL items, not just predefined categories
    """
    recommendations = scan_all_items_for_flips(
        min_price=100_000,  # 100k+ items (lowered for more results)
        max_price=500_000_000,  # Cap at 500M
        min_margin_pct=min_margin_pct,
        max_risk=max_risk,
        limit=100  # Return top 100
    )
    return recommendations


def scan_all_items_for_flips(
    min_price: int = 100_000,
    max_price: int = 500_000_000,
    min_margin_pct: float = 1.0,
    max_risk: int = 7,
    limit: int = 50
) -> List[Dict]:
    """
    Scan ALL tradeable items for flip opportunities
    Uses 5-minute averaged prices (like Flipping Copilot) for accuracy
    """
    analyzer = get_analyzer()

    # Fetch both instant and 5-minute averaged prices
    print("Fetching GE prices (instant + 5m averaged)...")
    instant_prices = analyzer.fetch_latest_prices()
    avg_5m_prices = analyzer.fetch_5m_prices()
    mapping = analyzer.fetch_item_mapping()

    candidates = []
    import time
    now = int(time.time())

    print(f"Scanning {len(instant_prices)} items for opportunities...")

    for item_id, instant_data in instant_prices.items():
        # Get 5-minute averaged prices (more accurate, like Flipping Copilot)
        avg_data = avg_5m_prices.get(item_id, {})

        # Use 5m averaged prices for DECISION MAKING (removes ghost margins)
        # Use instant prices only for EXECUTION (setting specific buy/sell)
        avg_high = avg_data.get('avgHighPrice')
        avg_low = avg_data.get('avgLowPrice')
        inst_high = instant_data.get('high')
        inst_low = instant_data.get('low')

        # Primary: use 5m average for margin calculation (ghost margin fix)
        high = avg_high or inst_high
        low = avg_low or inst_low

        # Ghost margin validation: if instant price deviates >5% from 5m avg,
        # it's a one-off fat-finger trade, not a real price
        if avg_high and inst_high:
            if inst_high > avg_high * 1.05 or inst_high < avg_high * 0.95:
                high = avg_high  # Use 5m avg, ignore the ghost
        if avg_low and inst_low:
            if inst_low > avg_low * 1.05 or inst_low < avg_low * 0.95:
                low = avg_low  # Use 5m avg, ignore the ghost

        # Skip items without both prices
        if not high or not low:
            continue

        # Filter by price range
        if low < min_price or high > max_price:
            continue

        # Quick margin check (before expensive analysis)
        gross_margin = high - low
        tax = min(high * 0.02, 5_000_000)
        net_margin = gross_margin - tax

        if net_margin <= 0:
            continue

        margin_pct = (net_margin / low) * 100
        if margin_pct < min_margin_pct:
            continue

        # Get item name
        item_info = mapping.get(str(item_id), {})
        item_name = item_info.get('name', f'Item {item_id}')

        # Skip noted/placeholder items
        if '(noted)' in item_name.lower() or item_name.startswith('Item '):
            continue

        # Get volume data from 5m endpoint
        high_volume = avg_data.get('highPriceVolume', 0)
        low_volume = avg_data.get('lowPriceVolume', 0)
        total_volume = high_volume + low_volume

        # Get freshness from instant prices
        high_time = instant_data.get('highTime', 0)
        low_time = instant_data.get('lowTime', 0)
        high_age_mins = (now - high_time) // 60 if high_time else 999
        low_age_mins = (now - low_time) // 60 if low_time else 999

        candidates.append({
            'id': item_id,
            'name': item_name,
            'high': high,
            'low': low,
            'instant_high': instant_data.get('high'),
            'instant_low': instant_data.get('low'),
            'avg_high': avg_data.get('avgHighPrice'),
            'avg_low': avg_data.get('avgLowPrice'),
            'net_margin': net_margin,
            'margin_pct': margin_pct,
            'tax': tax,
            'high_volume': high_volume,
            'low_volume': low_volume,
            'total_volume': total_volume,
            'high_age_mins': high_age_mins,
            'low_age_mins': low_age_mins
        })

    # Sort by net profit (absolute GP) for better balance
    candidates.sort(key=lambda x: x['net_margin'], reverse=True)

    print(f"Found {len(candidates)} items with positive margins")

    # Take top candidates and do full analysis
    recommendations = []
    for candidate in candidates[:limit * 3]:
        try:
            # Risk assessment based on volume, freshness, and margin
            risk = 3  # Base risk

            # Volume-based risk (low volume = higher risk)
            if candidate['total_volume'] == 0:
                risk += 4  # No trades in last 5m = dead market, DO NOT BUY
            elif candidate['total_volume'] < 5:
                risk += 2
            elif candidate['total_volume'] < 20:
                risk += 1

            # Volume velocity check: project hourly rate from 5m data
            # If volume is way below normal for this item, it's dying
            hourly_rate = candidate['total_volume'] * 12  # projected hourly
            # Items with 0 projected hourly volume are traps
            if hourly_rate == 0:
                risk += 3

            # Freshness-based risk
            max_age = max(candidate['high_age_mins'], candidate['low_age_mins'])
            if max_age > 60:  # Over 1 hour old
                risk += 2
            elif max_age > 30:
                risk += 1

            # High margin often means illiquid
            if candidate['margin_pct'] > 10:
                risk += 2
            elif candidate['margin_pct'] > 5:
                risk += 1

            # Volume bonus (high volume = lower risk)
            if candidate['total_volume'] > 50:
                risk = max(risk - 1, 1)

            if risk > max_risk:
                continue

            # Volume velocity trap: if hourly rate is way below what's normal,
            # the item has "died" and you will be stuck holding the bag
            if candidate['total_volume'] > 0:
                # Check if high_volume and low_volume are severely imbalanced
                # (one side has volume but other doesn't = one-sided market)
                hv = candidate.get('high_volume', 0)
                lv = candidate.get('low_volume', 0)
                if (hv > 0 and lv == 0) or (lv > 0 and hv == 0):
                    risk += 1  # One-sided volume = harder to fill both sides

            # Determine confidence level
            if candidate['total_volume'] > 20 and max_age < 15:
                confidence = "HIGH"
            elif candidate['total_volume'] > 5 and max_age < 30:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # Determine verdict
            if candidate['margin_pct'] > 10:
                verdict = 'CAUTION'
                reason = 'Very high margin - may be illiquid'
            elif candidate['margin_pct'] > 5:
                verdict = 'PATIENT_LIMIT'
                reason = 'Good margin - use limit orders'
            elif candidate['margin_pct'] > 2:
                verdict = 'STANDARD_FLIP'
                reason = 'Healthy margin for flipping'
            else:
                verdict = 'QUICK_FLIP'
                reason = 'Tight margin - move fast'

            recommendations.append({
                'name': candidate['name'],
                'item_id': candidate['id'],
                'buy_at': candidate['low'],
                'sell_at': candidate['high'],
                'instant_buy': candidate['instant_low'],
                'instant_sell': candidate['instant_high'],
                'expected_profit': int(candidate['net_margin']),
                'margin_pct': candidate['margin_pct'],
                'risk_score': risk,
                'verdict': verdict,
                'reason': reason,
                'buy_limit': get_buy_limit(int(candidate['id'])),
                'volume_5m': candidate['total_volume'],
                'high_volume': candidate['high_volume'],
                'low_volume': candidate['low_volume'],
                'price_age_mins': max(candidate['high_age_mins'], candidate['low_age_mins']),
                'confidence': confidence
            })

            if len(recommendations) >= limit:
                break

        except Exception as e:
            print(f"Error analyzing {candidate['name']}: {e}")
            continue

    print(f"Returning {len(recommendations)} recommendations")
    return recommendations


def analyze_single_item(item_id: int) -> str:
    """
    Generate a detailed analysis for a single item
    """
    analyzer = get_analyzer()
    analysis = analyzer.full_analysis(str(item_id))

    mapping = analyzer.fetch_item_mapping()
    item_info = mapping.get(str(item_id), {})
    item_name = item_info.get('name', f'Item {item_id}')

    lines = [
        f"# Detailed Analysis: {item_name}",
        f"Item ID: {item_id}",
        f"Category: {get_item_category(item_id) or 'Unknown'}",
        f"Buy Limit: {get_buy_limit(item_id):,}/4hr",
        "",
        "## Current Prices",
        f"- Insta-Buy (sell target): {analysis['prices']['high']:,} GP",
        f"- Insta-Sell (buy target): {analysis['prices']['low']:,} GP",
        "",
        "## Margin Analysis",
        f"- Gross Margin: {analysis['margin']['gross']:,} GP",
        f"- GE Tax (2%): {analysis['margin']['tax']:,} GP",
        f"- Slippage Est: {analysis['margin']['slippage']:,} GP",
        f"- **Net Margin: {analysis['margin']['net']:,} GP ({analysis['margin']['margin_pct']:.1f}%)**",
        f"- Profitable: {'Yes' if analysis['margin']['profitable'] else 'NO - Tax eliminates profit!'}",
        "",
        "## Statistical Analysis",
        f"- Z-Score: {analysis['z_score']['high']}",
        f"  - Interpretation: {analysis['z_score']['interpretation']}",
        f"- RSI: {analysis['rsi']['value']}",
        f"  - Interpretation: {analysis['rsi']['interpretation']}",
        "",
        "## Volume Analysis",
        f"- Volume Velocity: {analysis['volume']['velocity']:.1f}x average",
        f"- Interpretation: {analysis['volume']['interpretation']}",
        f"- Contested: {'Yes - expect competition' if analysis['volume']['contested'] else 'No'}",
        "",
        "## Trend Analysis",
        f"- Current Trend: {analysis['trend']['trend']}",
        f"- Above 1h SMA: {analysis['trend']['above_1h_sma']}",
        f"- Above 4h SMA: {analysis['trend']['above_4h_sma']}",
        "",
        "## Risk Assessment",
        f"- **Risk Score: {analysis['risk_score']}/10**",
        "",
        "## AI Verdict",
        f"- **Action: {analysis['verdict']['action']}**",
        f"- Reason: {analysis['verdict']['reason']}",
        f"- Confidence: {analysis['verdict']['confidence']}",
        "",
        "## Trading Strategy",
    ]

    # Add specific strategy based on verdict
    verdict = analysis['verdict']['action']
    if verdict == 'AGGRESSIVE_BUY':
        lines.extend([
            "1. Place buy offer at or slightly above insta-sell price",
            "2. Once filled, immediately list at insta-buy price",
            "3. Monitor for 5-10 minutes, undercut if needed"
        ])
    elif verdict == 'QUICK_FLIP':
        lines.extend([
            "1. Only enter if you can babysit the offer",
            "2. Be prepared to undercut aggressively",
            "3. Take profit quickly, don't hold for max margin"
        ])
    elif verdict == 'PATIENT_LIMIT':
        lines.extend([
            "1. Place limit order below insta-sell price",
            "2. Wait for fill (may take hours)",
            "3. List slightly below insta-buy for faster sell"
        ])
    elif verdict == 'AVOID':
        lines.extend([
            "**DO NOT TRADE THIS ITEM**",
            f"Reason: {analysis['verdict']['reason']}"
        ])
    else:
        lines.extend([
            "1. Standard flip approach",
            "2. Buy at insta-sell, sell at insta-buy",
            "3. Monitor market conditions"
        ])

    return "\n".join(lines)


if __name__ == '__main__':
    print("Testing AI Strategist...")
    print("=" * 60)

    # Test single item analysis
    print("\n" + analyze_single_item(13652))  # Dragon claws

    print("\n" + "=" * 60)
    print("\nTop Recommendations:")
    recs = get_flip_recommendations(max_risk=6, min_margin_pct=1.0)
    for rec in recs[:5]:
        print(f"  {rec['name']}: {rec['margin_pct']:.1f}% margin, Risk {rec['risk_score']}/10 - {rec['verdict']}")
