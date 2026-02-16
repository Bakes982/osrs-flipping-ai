#!/usr/bin/env python3
"""
OSRS Flip Dashboard - Professional Web-based GUI
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
import json
import os
from typing import Dict, List, Optional

# Page config
st.set_page_config(
    page_title="OSRS AI Flipper",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #151c2c 100%);
        border-right: 1px solid #2d3748;
    }

    /* Card styling */
    .opportunity-card {
        background: linear-gradient(135deg, #1a1f35 0%, #252d45 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .portfolio-card {
        background: linear-gradient(135deg, #1a1f35 0%, #252d45 100%);
        border-radius: 12px;
        padding: 15px 20px;
        margin: 8px 0;
        border: 1px solid #2d3748;
    }

    /* Header bar */
    .header-bar {
        background: linear-gradient(90deg, #1a1f35 0%, #252d45 100%);
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
        border: 1px solid #3d4a6a;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 8px;
    }

    .badge-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }

    .badge-sell {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }

    .badge-score {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
    }

    .badge-low-risk {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }

    .badge-medium-risk {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }

    .badge-high-risk {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }

    /* Price boxes */
    .price-box {
        background: #1a1f35;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #2d3748;
    }

    .price-box-buy {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
    }

    .price-box-sell {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
    }

    /* Metric styling */
    .metric-label {
        color: #9ca3af;
        font-size: 11px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 18px;
        font-weight: bold;
    }

    .metric-value-green {
        color: #10b981;
    }

    .metric-value-red {
        color: #ef4444;
    }

    .metric-value-yellow {
        color: #f59e0b;
    }

    /* Button styles */
    .action-btn {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        color: white;
        cursor: pointer;
        font-weight: 500;
        margin: 4px;
    }

    .action-btn-secondary {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }

    .action-btn-chart {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
    }

    /* Portfolio overview boxes */
    .portfolio-overview {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 20px;
    }

    .overview-box {
        background: linear-gradient(135deg, #1a1f35 0%, #252d45 100%);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3748;
    }

    /* Table styling */
    .portfolio-table {
        width: 100%;
        border-collapse: collapse;
    }

    .portfolio-table th {
        background: #1a1f35;
        color: #9ca3af;
        padding: 12px;
        text-align: left;
        font-weight: 500;
        border-bottom: 1px solid #2d3748;
    }

    .portfolio-table td {
        padding: 12px;
        border-bottom: 1px solid #1a1f35;
        color: #ffffff;
    }

    /* Stats row */
    .stats-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #2d3748;
        font-size: 13px;
    }

    .stats-label {
        color: #9ca3af;
    }

    .stats-value {
        color: #ffffff;
        font-weight: 500;
    }

    /* Section title */
    .section-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: 600;
        margin: 20px 0 15px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1f35;
    }

    ::-webkit-scrollbar-thumb {
        background: #3d4a6a;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #4d5a7a;
    }
</style>
""", unsafe_allow_html=True)

# ===== Data Loading Functions =====

def load_config():
    config_path = "user_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    with open("user_config.json", 'w') as f:
        json.dump(config, f, indent=2)

def load_portfolio():
    """Load portfolio from JSON file"""
    portfolio_path = r'C:\Users\Mikeb\OneDrive\Desktop\Flipping AI\portfolio.json'
    if os.path.exists(portfolio_path):
        with open(portfolio_path, 'r') as f:
            return json.load(f)
    return {"investments": [], "realized_pl": 0}

def save_portfolio(portfolio):
    """Save portfolio to JSON file"""
    portfolio_path = r'C:\Users\Mikeb\OneDrive\Desktop\Flipping AI\portfolio.json'
    with open(portfolio_path, 'w') as f:
        json.dump(portfolio, f, indent=2)

@st.cache_data(ttl=300)
def load_flip_data():
    csv_path = r'C:\Users\Mikeb\OneDrive\Desktop\flips.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@st.cache_data(ttl=30)
def load_dink_trades():
    dink_path = r'C:\Users\Mikeb\OneDrive\Desktop\Flipping AI\dink_trades.csv'
    if os.path.exists(dink_path):
        try:
            return pd.read_csv(dink_path)
        except pd.errors.ParserError:
            return pd.read_csv(dink_path, on_bad_lines='skip')
    return pd.DataFrame()

def get_dink_status():
    try:
        response = requests.get("http://localhost:5000/status", timeout=2)
        return response.json()
    except:
        return None

def get_active_trades():
    try:
        response = requests.get("http://localhost:5000/trades", timeout=2)
        return response.json()
    except:
        return {"active": [], "completed": []}

@st.cache_data(ttl=60)
def fetch_live_prices():
    """Fetch instant prices from Wiki API"""
    try:
        response = requests.get(
            "https://prices.runescape.wiki/api/v1/osrs/latest",
            headers={'User-Agent': 'OSRS-AI-Flipper v1.0 - Discord: bakes982'},
            timeout=10
        )
        return response.json().get('data', {})
    except:
        return {}

@st.cache_data(ttl=60)
def fetch_5m_prices():
    """
    Fetch 5-minute averaged prices - MORE ACCURATE
    This matches what Flipping Copilot uses for suggestions
    """
    try:
        response = requests.get(
            "https://prices.runescape.wiki/api/v1/osrs/5m",
            headers={'User-Agent': 'OSRS-AI-Flipper v1.0 - Discord: bakes982'},
            timeout=10
        )
        return response.json().get('data', {})
    except:
        return {}

def get_accurate_price(item_id: str, instant_prices: Dict, avg_prices: Dict) -> Dict:
    """
    Get accurate price by combining instant and 5m averaged data
    Prioritizes 5m averages (more reliable), falls back to instant
    """
    instant = instant_prices.get(item_id, {})
    avg = avg_prices.get(item_id, {})

    high = avg.get('avgHighPrice') or instant.get('high')
    low = avg.get('avgLowPrice') or instant.get('low')

    return {
        'high': high,
        'low': low,
        'instant_high': instant.get('high'),
        'instant_low': instant.get('low'),
        'avg_high': avg.get('avgHighPrice'),
        'avg_low': avg.get('avgLowPrice'),
        'high_volume': avg.get('highPriceVolume', 0),
        'low_volume': avg.get('lowPriceVolume', 0),
        'total_volume': avg.get('highPriceVolume', 0) + avg.get('lowPriceVolume', 0)
    }

@st.cache_data(ttl=300)
def fetch_item_mapping():
    try:
        response = requests.get(
            "https://prices.runescape.wiki/api/v1/osrs/mapping",
            headers={'User-Agent': 'OSRS-AI-Flipper v1.0 - Discord: bakes982'},
            timeout=10
        )
        items = response.json()
        return {str(item['id']): item for item in items}
    except:
        return {}

@st.cache_data(ttl=3600)
def fetch_timeseries(item_id: int, timestep: str = "1h"):
    try:
        response = requests.get(
            f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep={timestep}&id={item_id}",
            headers={'User-Agent': 'OSRS-AI-Flipper v1.0 - Discord: bakes982'},
            timeout=10
        )
        return response.json().get('data', [])
    except:
        return []

def send_discord_test(webhook_url):
    embed = {
        "title": "Test Notification",
        "description": "Your Discord webhook is working!",
        "color": 0x00ff00,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "OSRS AI Flipper Dashboard"}
    }
    try:
        response = requests.post(webhook_url, json={"embeds": [embed]}, headers={'Content-Type': 'application/json'}, timeout=10)
        return response.status_code == 204
    except:
        return False

# ===== Analysis Functions =====

def calculate_opportunity_score(item_data: Dict, prices: Dict, mapping: Dict) -> Dict:
    """Calculate comprehensive opportunity score for an item"""
    item_id = str(item_data.get('id', ''))
    price_data = prices.get(item_id, {})
    item_info = mapping.get(item_id, {})

    high = price_data.get('high', 0) or 0
    low = price_data.get('low', 0) or 0

    if not high or not low:
        return None

    # Calculate margins
    gross_margin = high - low
    tax = min(high * 0.02, 5_000_000)
    net_profit = gross_margin - tax

    if net_profit <= 0:
        return None

    margin_pct = (net_profit / low) * 100 if low > 0 else 0
    roi = (net_profit / low) * 100 if low > 0 else 0

    # Get historical data for averages
    timeseries = fetch_timeseries(int(item_id), "6h")

    avg_30d = high  # Default
    avg_180d = high
    price_range_low = low
    price_range_high = high
    volatility = 0
    deviation = 0

    if timeseries and len(timeseries) > 5:
        prices_list = [p.get('avgHighPrice', 0) or p.get('avgLowPrice', 0) for p in timeseries if p.get('avgHighPrice') or p.get('avgLowPrice')]
        if prices_list:
            avg_30d = sum(prices_list[-120:]) / min(len(prices_list), 120) if len(prices_list) >= 1 else high
            avg_180d = sum(prices_list) / len(prices_list) if prices_list else high
            price_range_low = min(prices_list)
            price_range_high = max(prices_list)

            # Volatility
            if len(prices_list) >= 2:
                mean_price = sum(prices_list) / len(prices_list)
                variance = sum((p - mean_price) ** 2 for p in prices_list) / len(prices_list)
                volatility = (variance ** 0.5 / mean_price) * 100 if mean_price > 0 else 0

            # Deviation from 30d average
            deviation = ((high - avg_30d) / avg_30d) * 100 if avg_30d > 0 else 0

    # Calculate scores
    margin_score = min(margin_pct * 10, 30)  # Up to 30 points for margin
    volume_score = 100  # Placeholder - would need trade volume data

    # Risk assessment
    if margin_pct > 10:
        risk_level = "HIGH"
        risk_penalty = 30
    elif margin_pct > 5:
        risk_level = "MEDIUM"
        risk_penalty = 15
    else:
        risk_level = "LOW"
        risk_penalty = 0

    # Staleness check
    high_time = price_data.get('highTime', 0)
    low_time = price_data.get('lowTime', 0)
    now = int(datetime.now().timestamp())
    stale_hours = 0
    if high_time and (now - high_time) > 3600:
        stale_hours = max(stale_hours, (now - high_time) // 3600)
    if low_time and (now - low_time) > 3600:
        stale_hours = max(stale_hours, (now - low_time) // 3600)

    staleness_penalty = min(stale_hours * 5, 20)

    # Final score
    base_score = margin_score + (volume_score * 0.3)
    final_score = max(0, min(100, base_score - risk_penalty - staleness_penalty))

    # Trend detection
    if len(timeseries) >= 2:
        recent = timeseries[-1].get('avgHighPrice', 0) or timeseries[-1].get('avgLowPrice', 0)
        older = timeseries[-min(5, len(timeseries))].get('avgHighPrice', 0) or timeseries[-min(5, len(timeseries))].get('avgLowPrice', 0)
        if recent and older:
            trend = "bullish" if recent > older else "bearish" if recent < older else "neutral"
        else:
            trend = "neutral"
    else:
        trend = "neutral"

    # Hold time estimate
    if margin_pct > 10:
        hold_time = "2-8 weeks"
    elif margin_pct > 5:
        hold_time = "1-2 weeks"
    elif margin_pct > 2:
        hold_time = "1-7 days"
    else:
        hold_time = "< 1 day"

    # Confidence
    confidence = 100 if stale_hours < 2 else 80 if stale_hours < 6 else 60

    return {
        'name': item_info.get('name', f'Item {item_id}'),
        'id': item_id,
        'trend': trend,
        'score': int(final_score),
        'risk_level': risk_level,
        'current_price': high,
        'avg_30d': int(avg_30d),
        'avg_180d': int(avg_180d),
        'change_30d': round(((high - avg_30d) / avg_30d) * 100, 1) if avg_30d > 0 else 0,
        'change_90d': round(((high - avg_180d) / avg_180d) * 100, 1) if avg_180d > 0 else 0,
        'profit': int(net_profit),
        'roi': round(roi, 1),
        'margin': round(margin_pct, 1),
        'buy_at': low,
        'sell_at': high,
        'price_range': f"{price_range_low:,} - {price_range_high:,}",
        'volatility': round(volatility, 1),
        'hold_time': hold_time,
        'deviation': round(abs(deviation), 1),
        'confidence': confidence,
        'volume_score': volume_score,
        'buy_limit': item_info.get('limit', 1)
    }

# ===== Main App =====

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://oldschool.runescape.wiki/images/Grand_Exchange_logo.png?74a09", width=180)
        st.markdown("### OSRS AI Flipper")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Dashboard", "Opportunities", "Portfolio", "Performance", "Price Alerts", "Favorites", "Settings"],
            index=0
        )

        st.markdown("---")

        # Quick stats
        dink_df = load_dink_trades()
        if not dink_df.empty and 'type' in dink_df.columns:
            buys = len(dink_df[dink_df['type'] == 'BUY'])
            sells = len(dink_df[dink_df['type'] == 'SELL'])
            st.metric("Trades Today", f"{buys + sells}")

        # DINK Status
        status = get_dink_status()
        if status:
            st.success("DINK: Online")
        else:
            st.error("DINK: Offline")

    # Main content
    if page == "Dashboard":
        show_dashboard()
    elif page == "Opportunities":
        show_opportunities()
    elif page == "Portfolio":
        show_portfolio_page()
    elif page == "Performance":
        show_performance()
    elif page == "Price Alerts":
        show_price_alerts()
    elif page == "Favorites":
        show_favorites()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Main dashboard overview"""
    st.markdown("## Dashboard")

    # Header bar with last updated
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"*Last updated: {datetime.now().strftime('%I:%M:%S %p')}*")
    with col2:
        if st.button("Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # Load data
    portfolio = load_portfolio()
    prices = fetch_live_prices()
    mapping = fetch_item_mapping()

    # Calculate portfolio value
    total_invested = 0
    current_value = 0

    for inv in portfolio.get('investments', []):
        qty = inv.get('quantity', 0)
        buy_price = inv.get('buy_price', 0)
        item_id = str(inv.get('item_id', ''))

        invested = qty * buy_price
        total_invested += invested

        current_price = prices.get(item_id, {}).get('high', buy_price) or buy_price
        current_value += qty * current_price

    total_pl = current_value - total_invested
    realized_pl = portfolio.get('realized_pl', 0)
    unrealized_pl = total_pl

    # Overview cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Invested", f"{int(total_invested):,}gp")

    with col2:
        st.metric("Current Value", f"{int(current_value):,}gp")

    with col3:
        pct = (total_pl / total_invested * 100) if total_invested > 0 else 0
        st.metric("Total P/L", f"{'+' if total_pl >= 0 else ''}{int(total_pl):,}gp", delta=f"{pct:+.1f}%")

    with col4:
        st.metric("Realized / Unrealized", f"{int(realized_pl):+,}gp / {int(unrealized_pl):+,}gp")

    st.markdown("---")

    # Quick opportunities preview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Top Opportunities")

        try:
            from ai_strategist import get_flip_recommendations
            recommendations = get_flip_recommendations(max_risk=7, min_margin_pct=1.0)[:5]

            for idx, rec in enumerate(recommendations):
                risk_score = rec['risk_score']
                risk_label = "LOW" if risk_score <= 4 else "MED" if risk_score <= 6 else "HIGH"

                with st.container():
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.markdown(f"**{rec['name']}**")
                    with c2:
                        if risk_score <= 4:
                            st.success(risk_label)
                        elif risk_score <= 6:
                            st.warning(risk_label)
                        else:
                            st.error(risk_label)
                    with c3:
                        st.metric("Profit", f"+{rec['expected_profit']:,}", delta=f"{rec['margin_pct']:.1f}%")
        except ImportError:
            st.info("Enable ai_strategist.py for opportunity recommendations")

    with col2:
        st.markdown("### Portfolio Summary")
        investments = portfolio.get('investments', [])
        st.metric("Active Investments", len(investments))
        st.metric("Total Value", f"{int(current_value):,} GP")

def show_opportunities():
    """AI-Detected Flip Opportunities with professional card layout"""

    # Header
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"*Last updated: {datetime.now().strftime('%I:%M:%S %p')}*")
    with col2:
        st.caption("Using 5-minute averaged prices (like Flipping Copilot)")
    with col3:
        if st.button("Refresh Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # Analysis Settings Panel
    st.markdown("### Analysis Settings")

    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])

    with col1:
        min_profit = st.selectbox("Min Profit", [0, 50000, 100000, 250000, 500000, 1000000],
                                   format_func=lambda x: f"{x:,} GP" if x > 0 else "Any")

    with col2:
        risk_level = st.slider("Max Risk", 1, 10, 7)
        st.caption("Lower = Safer" if risk_level <= 4 else "Balanced" if risk_level <= 6 else "Higher Risk OK")

    with col3:
        min_volume = st.selectbox("Min Volume (5m)", [0, 1, 5, 10, 20, 50],
                                   format_func=lambda x: f"{x}+ trades" if x > 0 else "Any")

    with col4:
        sort_by = st.selectbox("Sort By", ["Profit", "ROI", "Volume", "Confidence"])

    with col5:
        confidence_filter = st.selectbox("Confidence", ["All", "HIGH", "MEDIUM"])

    st.markdown("---")

    # Get recommendations
    try:
        from ai_strategist import get_flip_recommendations
        recommendations = get_flip_recommendations(max_risk=risk_level, min_margin_pct=0.5)
    except ImportError as e:
        st.error(f"Required modules not available: {e}")
        return

    # Apply filters
    filtered = []
    for r in recommendations:
        # Risk filter
        if r.get('risk_score', 10) > risk_level:
            continue
        # Profit filter
        if min_profit > 0 and r.get('expected_profit', 0) < min_profit:
            continue
        # Volume filter
        if min_volume > 0 and r.get('volume_5m', 0) < min_volume:
            continue
        # Confidence filter
        if confidence_filter != "All" and r.get('confidence', 'LOW') != confidence_filter:
            continue
        filtered.append(r)

    recommendations = filtered

    # Sort
    if sort_by == "Profit":
        recommendations.sort(key=lambda x: x.get('expected_profit', 0), reverse=True)
    elif sort_by == "ROI":
        recommendations.sort(key=lambda x: x.get('margin_pct', 0), reverse=True)
    elif sort_by == "Volume":
        recommendations.sort(key=lambda x: x.get('volume_5m', 0), reverse=True)
    elif sort_by == "Confidence":
        conf_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        recommendations.sort(key=lambda x: conf_order.get(x.get('confidence', 'LOW'), 0), reverse=True)

    # Count display
    st.markdown(f"### AI-Detected Flip Opportunities ({len(recommendations)})")

    if not recommendations:
        st.info("No opportunities found. Adjust filters or check back later.")
        return

    # Display opportunities in 3-column grid using native Streamlit components
    prices = fetch_live_prices()
    mapping = fetch_item_mapping()

    cols = st.columns(3)

    for i, rec in enumerate(recommendations[:30]):
        col = cols[i % 3]

        with col:
            item_id = str(rec.get('item_id', ''))
            price_data = prices.get(item_id, {})

            high = price_data.get('high', rec.get('sell_at', 0)) or rec.get('sell_at', 0)
            low = price_data.get('low', rec.get('buy_at', 0)) or rec.get('buy_at', 0)

            # Risk level
            risk_score = rec.get('risk_score', 5)
            if risk_score <= 4:
                risk_label = "LOW RISK"
                risk_color = "green"
            elif risk_score <= 6:
                risk_label = "MEDIUM RISK"
                risk_color = "orange"
            else:
                risk_label = "HIGH RISK"
                risk_color = "red"

            score = min(100, risk_score * 15)

            # Card container
            with st.container():
                # Header with confidence badge
                confidence = rec.get('confidence', 'MEDIUM')
                volume_5m = rec.get('volume_5m', 0)
                price_age = rec.get('price_age_mins', 0)

                st.markdown(f"**{rec['name']}**")

                # Badges row - risk, confidence, volume
                badge_cols = st.columns(4)
                with badge_cols[0]:
                    if risk_color == "green":
                        st.success(risk_label)
                    elif risk_color == "orange":
                        st.warning(risk_label)
                    else:
                        st.error(risk_label)
                with badge_cols[1]:
                    if confidence == "HIGH":
                        st.success(f"HIGH CONF")
                    elif confidence == "MEDIUM":
                        st.warning(f"MED CONF")
                    else:
                        st.error(f"LOW CONF")
                with badge_cols[2]:
                    st.info(f"Vol: {volume_5m}")
                with badge_cols[3]:
                    if price_age < 10:
                        st.success(f"{price_age}m ago")
                    elif price_age < 30:
                        st.warning(f"{price_age}m ago")
                    else:
                        st.error(f"{price_age}m ago")

                # Price metrics - show instant vs averaged if different
                price_cols = st.columns(2)
                with price_cols[0]:
                    buy_price = rec['buy_at']
                    instant_buy = rec.get('instant_buy', buy_price)
                    if instant_buy and instant_buy != buy_price:
                        st.metric("Buy (5m avg)", f"{buy_price:,}", delta=f"Inst: {instant_buy:,}")
                    else:
                        st.metric("Buy At", f"{buy_price:,}")
                with price_cols[1]:
                    sell_price = rec['sell_at']
                    instant_sell = rec.get('instant_sell', sell_price)
                    if instant_sell and instant_sell != sell_price:
                        st.metric("Sell (5m avg)", f"{sell_price:,}", delta=f"Inst: {instant_sell:,}")
                    else:
                        st.metric("Sell At", f"{sell_price:,}")

                # Profit metrics
                profit_cols = st.columns(3)
                with profit_cols[0]:
                    st.metric("Profit", f"+{rec['expected_profit']:,} GP")
                with profit_cols[1]:
                    st.metric("ROI", f"{rec['margin_pct']:.1f}%")
                with profit_cols[2]:
                    buy_limit = rec.get('buy_limit', 'N/A')
                    st.metric("Limit", f"{buy_limit:,}" if isinstance(buy_limit, int) else buy_limit)

                # Strategy info
                hold_time = '< 1 day' if rec['margin_pct'] < 2 else '1-7 days' if rec['margin_pct'] < 5 else '1-2 weeks'
                st.caption(f"Hold: {hold_time} | {rec['verdict']} | {rec.get('reason', '')}")

                # Action buttons
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    st.button("Ask AI", key=f"ai_{i}", type="secondary", width="stretch")
                with bcol2:
                    st.button("View Chart", key=f"chart_{i}", type="primary", width="stretch")

                st.divider()

def show_portfolio_page():
    """Portfolio tracking page"""

    # Header with tabs
    tab1, tab2, tab3 = st.columns(3)
    with tab1:
        st.markdown("### Portfolio")
        st.caption("Track buys, sales, and performance")

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Trade History", type="secondary"):
            st.session_state['show_history'] = True
    with col2:
        if st.button("Refresh", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        if st.button("Add Investment", type="primary"):
            st.session_state['show_add_investment'] = True

    st.markdown("---")

    # Load portfolio data
    portfolio = load_portfolio()
    prices = fetch_live_prices()
    mapping = fetch_item_mapping()

    investments = portfolio.get('investments', [])

    # Calculate totals
    total_invested = 0
    current_value = 0

    for inv in investments:
        qty = inv.get('quantity', 0)
        buy_price = inv.get('buy_price', 0)
        item_id = str(inv.get('item_id', ''))

        invested = qty * buy_price
        total_invested += invested

        current_price = prices.get(item_id, {}).get('high', buy_price) or buy_price
        current_value += qty * current_price

    total_pl = current_value - total_invested
    realized_pl = portfolio.get('realized_pl', 0)
    unrealized_pl = total_pl

    # Portfolio Overview
    ov_col1, ov_col2 = st.columns([3, 1])
    with ov_col1:
        st.markdown("**Portfolio Overview**")
    with ov_col2:
        st.caption(f"{len(investments)} items")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Invested", f"{int(total_invested):,}gp")

    with col2:
        st.metric("Current Value", f"{int(current_value):,}gp")

    with col3:
        pct = (total_pl / total_invested * 100) if total_invested > 0 else 0
        st.metric("Total P/L", f"{'+' if total_pl >= 0 else ''}{int(total_pl):,}gp", delta=f"{pct:+.1f}%")

    with col4:
        st.metric("Realized / Unrealized", f"{int(realized_pl):+,}gp / {int(unrealized_pl):+,}gp")

    st.markdown("---")

    # Add Investment form
    if st.session_state.get('show_add_investment', False):
        st.markdown("### Add New Investment")

        with st.form("add_investment_form"):
            col1, col2 = st.columns(2)
            with col1:
                item_search = st.text_input("Item Name or ID")
            with col2:
                quantity = st.number_input("Quantity", min_value=1, value=1)

            col3, col4 = st.columns(2)
            with col3:
                buy_price = st.number_input("Buy Price (GP each)", min_value=1, value=1000)
            with col4:
                buy_date = st.date_input("Buy Date", value=datetime.now())

            submitted = st.form_submit_button("Add Investment")

            if submitted:
                # Find item ID
                item_id = None
                for iid, info in mapping.items():
                    if item_search.lower() in info.get('name', '').lower() or item_search == iid:
                        item_id = iid
                        item_name = info.get('name', f'Item {iid}')
                        break

                if item_id:
                    new_inv = {
                        'item_id': item_id,
                        'item_name': item_name,
                        'quantity': quantity,
                        'remaining': quantity,
                        'buy_price': buy_price,
                        'buy_date': buy_date.isoformat(),
                        'lots': [{'qty': quantity, 'price': buy_price}]
                    }
                    portfolio['investments'].append(new_inv)
                    save_portfolio(portfolio)
                    st.success(f"Added {quantity}x {item_name} @ {buy_price:,} GP each")
                    st.session_state['show_add_investment'] = False
                    st.rerun()
                else:
                    st.error("Item not found. Try using the item ID.")

        st.markdown("---")

    # Investment table
    if investments:
        st.markdown("### Holdings")

        for i, inv in enumerate(investments):
            item_id = str(inv.get('item_id', ''))
            item_name = inv.get('item_name', f'Item {item_id}')
            qty = inv.get('quantity', 0)
            remaining = inv.get('remaining', qty)
            buy_price = inv.get('buy_price', 0)
            buy_date = inv.get('buy_date', 'Unknown')
            lots = inv.get('lots', [])

            current_price = prices.get(item_id, {}).get('high', buy_price) or buy_price
            pl = (current_price - buy_price) * remaining

            col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 2])

            with col1:
                st.markdown(f"**{item_name}**")
                st.caption(f"Bought {buy_date[:10] if isinstance(buy_date, str) else buy_date}")
                lots_str = " | ".join([f"{l['qty']:,} @ {l['price']:,}gp" for l in lots[:3]])
                st.caption(f"Lots: {lots_str}")

            with col2:
                st.metric("Qty", f"{qty:,}")

            with col3:
                st.metric("Remaining", f"{remaining:,}")

            with col4:
                st.metric("Buy", f"{buy_price:,}gp")

            with col5:
                st.metric("Current", f"{current_price:,}gp")

            with col6:
                pl_color = "normal" if pl >= 0 else "inverse"
                st.metric("P/L", f"{'+' if pl >= 0 else ''}{int(pl):,}gp", delta_color=pl_color)

            # Action buttons for this item
            bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
            with bcol1:
                if st.button("Ask AI", key=f"port_ai_{i}", type="secondary"):
                    pass
            with bcol2:
                if st.button("Record Flip", key=f"port_flip_{i}", type="primary"):
                    st.session_state[f'record_flip_{i}'] = True
            with bcol3:
                if st.button("Set Alert", key=f"port_alert_{i}"):
                    pass
            with bcol4:
                if st.button("Notes", key=f"port_notes_{i}"):
                    pass
            with bcol5:
                if st.button("Remove", key=f"port_remove_{i}"):
                    portfolio['investments'].pop(i)
                    save_portfolio(portfolio)
                    st.rerun()

            # Record flip form
            if st.session_state.get(f'record_flip_{i}', False):
                with st.form(f"record_flip_form_{i}"):
                    st.markdown(f"**Record Sale for {item_name}**")
                    sell_qty = st.number_input("Quantity Sold", min_value=1, max_value=remaining, value=1, key=f"sell_qty_{i}")
                    sell_price = st.number_input("Sell Price (GP each)", min_value=1, value=current_price, key=f"sell_price_{i}")

                    if st.form_submit_button("Record Sale"):
                        profit = (sell_price - buy_price) * sell_qty
                        tax = min(sell_price * 0.02, 5_000_000) * sell_qty
                        net_profit = profit - tax

                        inv['remaining'] = remaining - sell_qty
                        if inv['remaining'] <= 0:
                            portfolio['investments'].pop(i)

                        portfolio['realized_pl'] = portfolio.get('realized_pl', 0) + net_profit
                        save_portfolio(portfolio)

                        st.success(f"Recorded sale: {sell_qty}x {item_name} for {int(net_profit):,} GP profit")
                        st.session_state[f'record_flip_{i}'] = False
                        st.rerun()

            st.markdown("---")
    else:
        st.info("No investments yet. Click 'Add Investment' to start tracking!")

def show_performance():
    """Performance analytics page"""
    st.markdown("## Performance")

    # Import profit tracker
    try:
        from profit_tracker import (
            load_dink_trades, match_trades, calculate_session_stats,
            get_item_performance, generate_ai_insights
        )

        dink_df = load_dink_trades()
        if not dink_df.empty:
            flips = match_trades(dink_df)
            stats = calculate_session_stats(flips, hours=24)

            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Profit", f"{stats['total_profit']:,.0f} GP")
            with col2:
                st.metric("Completed Flips", f"{stats['total_flips']}")
            with col3:
                st.metric("Win Rate", f"{stats['win_rate']:.0f}%")
            with col4:
                st.metric("GP/Hour", f"{stats['gp_per_hour']:,.0f}")
            with col5:
                st.metric("Avg Profit", f"{stats['avg_profit']:,.0f} GP")

            st.markdown("---")

            # Item performance table
            item_perf = get_item_performance(flips)
            if not item_perf.empty:
                st.markdown("### Item Performance")
                st.dataframe(item_perf, width="stretch", hide_index=True)
        else:
            st.info("No trade data available yet.")

    except ImportError:
        st.info("Performance tracking requires profit_tracker.py")

def show_price_alerts():
    """Price alerts page"""
    st.markdown("## Price Alerts")
    st.info("Price alerts feature coming soon! Set target prices and get notified via Discord.")

def show_favorites():
    """Favorites page"""
    st.markdown("## Favorites")
    st.info("Favorites feature coming soon! Star items to track them easily.")

def show_settings():
    """Settings page"""
    st.markdown("## Settings")

    config = load_config()

    # Discord Settings
    st.subheader("Discord Notifications")

    discord_config = config.get('discord_webhook', {})

    webhook_url = st.text_input("Webhook URL", value=discord_config.get('url', ''), type="password")
    enabled = st.checkbox("Enable notifications", value=discord_config.get('enabled', False))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Settings", type="primary"):
            config['discord_webhook'] = {
                'enabled': enabled,
                'url': webhook_url
            }
            save_config(config)
            st.success("Settings saved!")

    with col2:
        if st.button("Test Webhook"):
            if webhook_url:
                if send_discord_test(webhook_url):
                    st.success("Test sent!")
                else:
                    st.error("Failed to send")
            else:
                st.warning("Enter a webhook URL first")

if __name__ == "__main__":
    main()
