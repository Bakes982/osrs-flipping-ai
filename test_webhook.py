#!/usr/bin/env python3
"""Quick test to verify Discord webhook is working"""

import requests
from datetime import datetime, timezone

WEBHOOK_URL = "https://discord.com/api/webhooks/1468587251507396692/NFjpc03UHJmrTVuKLW9_f0qzMKQdElZGQMEtPQltK2FLgNO6UaGKnilnXrKbufE7uBR5"

def test_webhook():
    embed = {
        "title": "Test Notification",
        "description": "If you see this, your Discord webhook is working!",
        "color": 0x00ff00,  # Green
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "OSRS AI Flipper"},
        "fields": [
            {
                "name": "Example Flip: Dragon claws",
                "value": "Profit: 298,042 GP\nBuy: 51,763,000 | Sell: 52,061,042\nSpread: 0.57% | Risk: MEDIUM",
                "inline": False
            },
            {
                "name": "Example Flip: Abyssal bludgeon",
                "value": "Profit: 416,421 GP\nBuy: 20,369,568 | Sell: 20,785,989\nSpread: 2.04% | Risk: LOW",
                "inline": False
            }
        ]
    }

    payload = {"embeds": [embed]}

    try:
        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        if response.status_code == 204:
            print("SUCCESS! Check your Discord channel!")
        else:
            print(f"Failed with status code: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_webhook()
