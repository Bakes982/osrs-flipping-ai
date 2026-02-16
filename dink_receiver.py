#!/usr/bin/env python3
"""
DINK Webhook Receiver - Captures real-time GE trades from RuneLite DINK plugin
Run this as a local server to receive trade notifications
Now with Quantitative Analysis (Z-Score, RSI, Volume Velocity)
"""

from flask import Flask, request, jsonify
import json
import pandas as pd
from datetime import datetime
import requests
import os

# Import quantitative analyzer
try:
    from quant_analyzer import get_analyzer, get_buy_limit, get_item_category
    QUANT_ENABLED = True
    print("Quantitative Analyzer loaded successfully")
except ImportError:
    QUANT_ENABLED = False
    print("WARNING: quant_analyzer.py not found - running without quant features")

app = Flask(__name__)

# Your Discord webhook for trade notifications
TRADE_WEBHOOK = "https://discord.com/api/webhooks/1468694732673908888/XFK8PE3oTHgDIjQ6YvFks2pAdHZ4QiR4PXVWCzaxZ2w0aKvCaO7vFLjz_Wi1zCttICny"

# Store active trades
active_trades = {}
completed_trades = []

# OSRS Wiki API for live prices
def get_live_price(item_id):
    """Get current GE prices from Wiki API"""
    try:
        response = requests.get(
            "https://prices.runescape.wiki/api/v1/osrs/latest",
            headers={'User-Agent': 'OSRS-AI-Flipper v1.0 - Discord: bakes982'},
            timeout=10
        )
        data = response.json().get('data', {})
        item_data = data.get(str(item_id), {})
        return {
            'high': item_data.get('high'),
            'low': item_data.get('low')
        }
    except:
        return {'high': None, 'low': None}

def send_discord_notification(title, description, color=0x00ff00, fields=None):
    """Send notification to Discord"""
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "OSRS AI Flipper - DINK Integration"}
    }
    if fields:
        embed["fields"] = fields

    try:
        requests.post(
            TRADE_WEBHOOK,
            json={"embeds": [embed]},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
    except Exception as e:
        print(f"Discord notification failed: {e}")

@app.route('/dink', methods=['POST'])
def receive_dink_webhook():
    """
    Receive DINK webhook notifications
    DINK can send JSON or multipart/form-data (when images enabled)
    """
    try:
        print(f"\n{'='*60}")
        print(f"DINK WEBHOOK RECEIVED: {datetime.now()}")
        print(f"{'='*60}")
        print(f"Content-Type: {request.content_type}")

        # Handle different content types
        data = None

        # Check for multipart/form-data (DINK with images)
        if request.content_type and 'multipart/form-data' in request.content_type:
            print("Received multipart/form-data")
            print(f"Form fields: {list(request.form.keys())}")
            print(f"Files: {list(request.files.keys())}")

            # DINK sends payload in 'payload_json' field
            if 'payload_json' in request.form:
                data = json.loads(request.form['payload_json'])
            elif 'payload' in request.form:
                data = json.loads(request.form['payload'])
            else:
                # Try to get any JSON field
                for key in request.form:
                    try:
                        data = json.loads(request.form[key])
                        print(f"Found JSON in field: {key}")
                        break
                    except:
                        print(f"Field {key}: {request.form[key][:200] if len(request.form[key]) > 200 else request.form[key]}")

        # Check for JSON
        elif request.is_json:
            data = request.json

        # Fallback - try to parse raw data
        else:
            print(f"Raw data: {request.data[:500] if len(request.data) > 500 else request.data}")
            try:
                data = json.loads(request.data)
            except:
                data = {'raw': request.data.decode('utf-8', errors='ignore')}

        if data:
            print(f"\nParsed data:")
            print(json.dumps(data, indent=2, default=str))

        # Log everything for debugging
        log_webhook(data if data else {'error': 'no data parsed'})

        if not data:
            print("WARNING: No data could be parsed from request")
            return jsonify({"status": "received but no data parsed"}), 200

        # DINK webhook structure for GE trades
        # The exact format depends on DINK version, but typically includes:
        # - type: "GRAND_EXCHANGE"
        # - playerName: account name
        # - extra: contains trade details

        webhook_type = data.get('type', '')

        if webhook_type == 'GRAND_EXCHANGE' or 'grandExchange' in str(data).lower():
            handle_ge_trade(data)
        elif webhook_type == 'LOOT' or 'loot' in str(data).lower():
            handle_loot(data)
        else:
            print(f"Unknown webhook type: {webhook_type}")
            # Log it anyway for debugging
            log_webhook(data)

        return jsonify({"status": "received"}), 200

    except Exception as e:
        print(f"Error processing webhook: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def handle_ge_trade(data):
    """Process Grand Exchange trade notification"""
    try:
        # Extract trade data - DINK actual format
        player_name = data.get('playerName', 'Unknown')
        extra = data.get('extra', {})

        # DINK actual format:
        # extra.item.id, extra.item.name, extra.item.quantity, extra.item.priceEach
        # extra.status: BOUGHT, SOLD, BUYING, SELLING, CANCELLED
        # extra.slot, extra.marketPrice, extra.sellerTax

        item_data = extra.get('item', {})
        item_name = item_data.get('name', 'Unknown')
        item_id = item_data.get('id', 0)
        quantity = item_data.get('quantity', 0)
        price = item_data.get('priceEach', 0)
        total_value = quantity * price
        status = extra.get('status', 'UNKNOWN')
        slot = extra.get('slot', 0)
        market_price = extra.get('marketPrice', 0)
        seller_tax = extra.get('sellerTax', 0)

        # Determine trade type from status
        if status in ['BOUGHT', 'BUYING']:
            trade_type = 'BUY'
        elif status in ['SOLD', 'SELLING']:
            trade_type = 'SELL'
        else:
            trade_type = 'UNKNOWN'

        print(f"\n📊 GE TRADE DETECTED:")
        print(f"   Player: {player_name}")
        print(f"   Item: {item_name} (ID: {item_id})")
        print(f"   Type: {trade_type}")
        print(f"   Status: {status}")
        print(f"   Qty: {quantity:,} @ {price:,} = {total_value:,} GP")

        # Get current market price for comparison
        live_prices = get_live_price(item_id)

        # Store trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'player': player_name,
            'item_name': item_name,
            'item_id': item_id,
            'type': trade_type,
            'status': status,
            'quantity': quantity,
            'price': price,
            'total_value': total_value,
            'slot': slot,
            'market_price': market_price,
            'seller_tax': seller_tax,
            'market_high': live_prices['high'],
            'market_low': live_prices['low']
        }

        # Check if this is a BUY completion
        if status == 'BOUGHT':
            # Store as active position
            key = f"{player_name}_{item_name}"
            active_trades[key] = trade_record

            # Calculate potential profit with REALISTIC margins
            insta_buy = live_prices['high']  # What buyers pay NOW (your sell target)
            insta_sell = live_prices['low']   # What sellers get NOW (ideal buy price)

            # Run quantitative analysis if available
            quant_data = None
            if QUANT_ENABLED:
                try:
                    analyzer = get_analyzer()
                    quant_data = analyzer.full_analysis(str(item_id), insta_buy, insta_sell)
                    buy_limit = get_buy_limit(item_id)
                    category = get_item_category(item_id)
                except Exception as e:
                    print(f"Quant analysis error: {e}")
                    quant_data = None

            if insta_buy and insta_sell:
                # Use quant margin data if available, otherwise calculate
                if quant_data:
                    net_profit = quant_data['margin']['net'] * quantity
                    total_tax = quant_data['margin']['tax'] * quantity
                    margin_pct = quant_data['margin']['margin_pct']
                else:
                    tax_per_item = min(insta_buy * 0.02, 5000000)
                    total_tax = tax_per_item * quantity
                    gross_profit = (insta_buy - price) * quantity
                    net_profit = gross_profit - total_tax
                    margin_pct = (net_profit / (price * quantity) * 100) if price else 0

                # Check if you overpaid (bought above insta-sell)
                overpaid = price > insta_sell
                overpaid_amount = (price - insta_sell) * quantity if overpaid else 0

                # Build notification with quant data
                fields = [
                    {"name": "Item", "value": f"{item_name}", "inline": True},
                    {"name": "Quantity", "value": f"{quantity:,}", "inline": True},
                    {"name": "Your Buy Price", "value": f"{price:,} GP", "inline": True},
                    {"name": "Insta-Sell (ideal buy)", "value": f"{insta_sell:,} GP", "inline": True},
                    {"name": "Insta-Buy (sell target)", "value": f"{insta_buy:,} GP", "inline": True},
                    {"name": "Expected Profit", "value": f"{net_profit:,.0f} GP ({margin_pct:.1f}%)", "inline": True},
                ]

                # Add quant analysis fields if available
                if quant_data:
                    z_score = quant_data['z_score']['high']
                    rsi = quant_data['rsi']['value']
                    risk = quant_data['risk_score']
                    verdict = quant_data['verdict']

                    quant_summary = []
                    if z_score is not None:
                        z_emoji = "🔴" if z_score > 1.5 else "🟢" if z_score < -1.5 else "🟡"
                        quant_summary.append(f"Z: {z_score:.1f} {z_emoji}")
                    if rsi is not None:
                        rsi_emoji = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
                        quant_summary.append(f"RSI: {rsi:.0f} {rsi_emoji}")

                    fields.append({"name": "Quant Analysis", "value": " | ".join(quant_summary), "inline": True})
                    fields.append({"name": "Risk Score", "value": f"{risk}/10", "inline": True})
                    fields.append({"name": "AI Verdict", "value": f"**{verdict['action']}**\n{verdict['reason']}", "inline": False})

                # Determine notification color based on analysis
                if quant_data and quant_data['verdict']['action'] == 'AVOID':
                    title = f"🚫 BUY COMPLETE (RISKY): {item_name}"
                    color = 0xff0000  # Red
                    description = f"**{player_name}** bought {quantity:,}x {item_name}\n\n**⚠️ QUANT WARNING:** {quant_data['verdict']['reason']}"
                elif overpaid:
                    title = f"⚠️ BUY COMPLETE (OVERPAID): {item_name}"
                    color = 0xffaa00  # Orange warning
                    description = f"**{player_name}** bought {quantity:,}x {item_name}\n\n**WARNING:** You paid {overpaid_amount:,} GP MORE than insta-sell!\nIdeal buy: {insta_sell:,} | Your buy: {price:,}"
                elif quant_data and quant_data['risk_score'] >= 7:
                    title = f"⚠️ BUY COMPLETE (HIGH RISK): {item_name}"
                    color = 0xffaa00
                    description = f"**{player_name}** bought {quantity:,}x {item_name}\n\nRisk Score: {quant_data['risk_score']}/10"
                else:
                    title = f"✅ BUY COMPLETE: {item_name}"
                    color = 0x00ff00
                    description = f"**{player_name}** bought {quantity:,}x {item_name}"

                send_discord_notification(
                    title=title,
                    description=description,
                    color=color,
                    fields=fields
                )

                # Console output with quant data
                print(f"\n💡 AI SUGGESTION: List for {insta_buy:,} GP")
                print(f"   Estimated profit: {net_profit:,.0f} GP (after 2% tax)")
                if quant_data:
                    print(f"   Z-Score: {quant_data['z_score']['high']} - {quant_data['z_score']['interpretation']}")
                    print(f"   RSI: {quant_data['rsi']['value']} - {quant_data['rsi']['interpretation']}")
                    print(f"   Volume: {quant_data['volume']['interpretation']}")
                    print(f"   Risk: {quant_data['risk_score']}/10")
                    print(f"   Verdict: {quant_data['verdict']['action']} - {quant_data['verdict']['reason']}")

        # Check if this is a SELL completion
        elif status == 'SOLD':
            # Check if we have a matching buy
            key = f"{player_name}_{item_name}"
            if key in active_trades:
                buy_trade = active_trades[key]
                buy_price = buy_trade['price']
                profit_per_item = price - buy_price
                total_profit = profit_per_item * quantity
                # Use actual tax from DINK if available, otherwise calculate 2% capped at 5M
                tax = seller_tax if seller_tax else min(price * 0.02, 5000000) * quantity
                profit_after_tax = total_profit - tax

                fields = [
                    {"name": "Item", "value": f"{item_name}", "inline": True},
                    {"name": "Quantity", "value": f"{quantity:,}", "inline": True},
                    {"name": "Buy Price", "value": f"{buy_price:,} GP", "inline": True},
                    {"name": "Sell Price", "value": f"{price:,} GP", "inline": True},
                    {"name": "Profit", "value": f"{profit_after_tax:,.0f} GP", "inline": True},
                    {"name": "GP/Item", "value": f"{profit_per_item:,.0f} GP", "inline": True},
                ]

                emoji = "🎉" if profit_after_tax > 0 else "😢"

                send_discord_notification(
                    title=f"{emoji} FLIP COMPLETE: {item_name}",
                    description=f"**{player_name}** sold {quantity:,}x {item_name}",
                    color=0x00ff00 if profit_after_tax > 0 else 0xff0000,
                    fields=fields
                )

                # Log completed trade
                completed_trades.append({
                    **trade_record,
                    'buy_price': buy_price,
                    'profit': profit_after_tax
                })

                # Remove from active
                del active_trades[key]

            else:
                # Sell without matching buy
                send_discord_notification(
                    title=f"💰 SELL COMPLETE: {item_name}",
                    description=f"**{player_name}** sold {quantity:,}x {item_name} for {total_value:,} GP",
                    color=0xffaa00
                )

        # Check for undercuts on active sell offers
        if status == 'SELLING':
            # Active sell offer - check if undercut
            if market_price and price > market_price:
                send_discord_notification(
                    title=f"⚠️ UNDERCUT ALERT: {item_name}",
                    description=f"Your offer: {price:,} GP\nMarket price: {market_price:,} GP\n**You've been undercut by {price - market_price:,} GP!**",
                    color=0xff4444
                )

        # Save to log file
        log_trade(trade_record)

    except Exception as e:
        print(f"Error handling GE trade: {e}")
        import traceback
        traceback.print_exc()

def handle_loot(data):
    """Process loot drop notification (for PVM tracking)"""
    print(f"Loot drop received: {data}")

def log_webhook(data):
    """Log unknown webhooks for debugging"""
    log_file = "dink_webhooks_log.json"
    try:
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)

        logs.append({
            'timestamp': datetime.now().isoformat(),
            'data': data
        })

        with open(log_file, 'w') as f:
            json.dump(logs[-100:], f, indent=2)  # Keep last 100

    except Exception as e:
        print(f"Error logging webhook: {e}")

def log_trade(trade):
    """Log trade to CSV file with consistent columns"""
    trade_file = "dink_trades.csv"

    # Define consistent columns
    columns = [
        'timestamp', 'player', 'item_name', 'item_id', 'type', 'status',
        'quantity', 'price', 'total_value', 'slot', 'market_price', 'seller_tax',
        'market_high', 'market_low'
    ]

    try:
        # Ensure all columns exist with defaults
        row = {col: trade.get(col, '') for col in columns}
        df = pd.DataFrame([row])

        if os.path.exists(trade_file):
            df.to_csv(trade_file, mode='a', header=False, index=False, columns=columns)
        else:
            df.to_csv(trade_file, index=False, columns=columns)
        print(f"Trade logged to {trade_file}")
    except Exception as e:
        print(f"Error logging trade: {e}")

@app.route('/status', methods=['GET'])
def status():
    """Check server status"""
    return jsonify({
        "status": "running",
        "active_trades": len(active_trades),
        "completed_trades": len(completed_trades),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/trades', methods=['GET'])
def get_trades():
    """Get active and completed trades"""
    return jsonify({
        "active": list(active_trades.values()),
        "completed": completed_trades[-50:]  # Last 50
    })

if __name__ == '__main__':
    print("="*60)
    print("DINK WEBHOOK RECEIVER")
    print("="*60)
    print()
    print("This server receives webhooks from RuneLite DINK plugin")
    print()
    print("SETUP INSTRUCTIONS:")
    print("1. Install 'DINK' plugin from RuneLite Plugin Hub")
    print("2. In DINK settings, enable 'Grand Exchange' notifications")
    print("3. Set webhook URL to: http://YOUR_IP:5000/dink")
    print("   (Use ngrok for public URL if needed)")
    print()
    print("Starting server on http://0.0.0.0:5000")
    print("="*60)

    app.run(host='0.0.0.0', port=5000, debug=True)
