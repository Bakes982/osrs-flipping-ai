#!/usr/bin/env python3
"""
User Configuration System - Manage blocklist, offer times, and per-slot capital allocation
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class UserConfig:
    """Manages user preferences and settings"""
    
    def __init__(self, config_file: str = "user_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'blocklist': [],
                'discord_webhook': {
                    'enabled': False,
                    'url': '',
                    'notify_opportunities': True,
                    'notify_price_spikes': True,
                    'min_score_to_notify': 50  # Only notify for opportunities scoring 50+
                },
                'offer_time_settings': {
                    'default': 5,  # minutes
                    'per_item': {}
                },
                'ge_slot_settings': [
                    {'slot': 1, 'min_price': 0, 'max_price': 1000000, 'purpose': 'Small flips'},
                    {'slot': 2, 'min_price': 1000000, 'max_price': 10000000, 'purpose': 'Medium flips'},
                    {'slot': 3, 'min_price': 10000000, 'max_price': 50000000, 'purpose': 'High-value flips'},
                    {'slot': 4, 'min_price': 10000000, 'max_price': 100000000, 'purpose': 'High-value flips'},
                    {'slot': 5, 'min_price': 0, 'max_price': 100000000, 'purpose': 'Investments'},
                    {'slot': 6, 'min_price': 0, 'max_price': 100000000, 'purpose': 'Flexible'},
                    {'slot': 7, 'min_price': 0, 'max_price': 100000000, 'purpose': 'Flexible'},
                    {'slot': 8, 'min_price': 0, 'max_price': 100000000, 'purpose': 'Flexible'},
                ],
                'risk_tolerance': 'MEDIUM',
                'min_profit_threshold': 100000,
                'last_updated': datetime.now().isoformat()
            }
    
    def save_config(self):
        """Save configuration to file"""
        self.config['last_updated'] = datetime.now().isoformat()
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✅ Configuration saved to {self.config_file}")
    
    # BLOCKLIST MANAGEMENT
    
    def add_to_blocklist(self, item_names: List[str]):
        """Add items to blocklist"""
        for item in item_names:
            if item not in self.config['blocklist']:
                self.config['blocklist'].append(item)
        self.save_config()
        print(f"✅ Added {len(item_names)} items to blocklist")
    
    def remove_from_blocklist(self, item_names: List[str]):
        """Remove items from blocklist"""
        removed = 0
        for item in item_names:
            if item in self.config['blocklist']:
                self.config['blocklist'].remove(item)
                removed += 1
        self.save_config()
        print(f"✅ Removed {removed} items from blocklist")
    
    def is_blocked(self, item_name: str) -> bool:
        """Check if item is blocked"""
        return item_name in self.config['blocklist']
    
    def get_blocklist(self) -> List[str]:
        """Get all blocked items"""
        return self.config['blocklist'].copy()
    
    def clear_blocklist(self):
        """Clear entire blocklist"""
        self.config['blocklist'] = []
        self.save_config()
        print("✅ Blocklist cleared")
    
    def import_blocklist_from_copilot(self, copilot_profile_path: str):
        """Import blocklist from FlippingCopilot profile JSON"""
        try:
            with open(copilot_profile_path, 'r') as f:
                copilot_data = json.load(f)
            
            blocked_ids = copilot_data.get('blockedItemIds', [])
            
            # You'd need item ID mapping here
            # For now, just store the IDs
            self.config['blocklist_ids'] = blocked_ids
            self.save_config()
            print(f"✅ Imported {len(blocked_ids)} blocked item IDs from FlippingCopilot")
            
        except Exception as e:
            print(f"❌ Error importing from FlippingCopilot: {e}")
    
    # OFFER TIME SETTINGS
    
    def set_default_offer_time(self, minutes: int):
        """Set default offer time for all items"""
        self.config['offer_time_settings']['default'] = minutes
        self.save_config()
        print(f"✅ Default offer time set to {minutes} minutes")
    
    def set_offer_time_for_item(self, item_name: str, minutes: int):
        """Set custom offer time for specific item"""
        self.config['offer_time_settings']['per_item'][item_name] = minutes
        self.save_config()
        print(f"✅ Offer time for '{item_name}' set to {minutes} minutes")
    
    def get_offer_time(self, item_name: str) -> int:
        """Get offer time for item (returns custom or default)"""
        per_item = self.config['offer_time_settings'].get('per_item', {})
        return per_item.get(item_name, self.config['offer_time_settings']['default'])
    
    def remove_offer_time_override(self, item_name: str):
        """Remove custom offer time for item (revert to default)"""
        per_item = self.config['offer_time_settings'].get('per_item', {})
        if item_name in per_item:
            del per_item[item_name]
            self.save_config()
            print(f"✅ Removed custom offer time for '{item_name}'")
    
    # GE SLOT ALLOCATION
    
    def configure_slot(
        self,
        slot_number: int,
        min_price: int,
        max_price: int,
        purpose: str = ""
    ):
        """Configure a specific GE slot"""
        if not (1 <= slot_number <= 8):
            print("❌ Slot number must be between 1 and 8")
            return
        
        # Find or create slot config
        slot_config = None
        for slot in self.config['ge_slot_settings']:
            if slot['slot'] == slot_number:
                slot_config = slot
                break
        
        if slot_config is None:
            slot_config = {'slot': slot_number}
            self.config['ge_slot_settings'].append(slot_config)
        
        # Update settings
        slot_config['min_price'] = min_price
        slot_config['max_price'] = max_price
        slot_config['purpose'] = purpose
        
        self.save_config()
        print(f"✅ Slot {slot_number} configured: {min_price:,} - {max_price:,} GP ({purpose})")
    
    def get_slot_config(self, slot_number: int) -> Optional[Dict]:
        """Get configuration for a specific slot"""
        for slot in self.config['ge_slot_settings']:
            if slot['slot'] == slot_number:
                return slot
        return None
    
    def get_available_slots_for_item(self, item_price: int) -> List[int]:
        """Get list of slot numbers that can accommodate this item price"""
        available_slots = []
        
        for slot in self.config['ge_slot_settings']:
            if slot['min_price'] <= item_price <= slot['max_price']:
                available_slots.append(slot['slot'])
        
        return available_slots
    
    def get_slot_summary(self) -> str:
        """Get human-readable summary of slot configuration"""
        lines = ["GE Slot Configuration:"]
        lines.append("─" * 80)
        
        for slot in sorted(self.config['ge_slot_settings'], key=lambda x: x['slot']):
            lines.append(
                f"Slot {slot['slot']}: {slot['min_price']:>12,} - {slot['max_price']:>12,} GP | {slot['purpose']}"
            )
        
        return "\n".join(lines)
    
    def set_investment_slot(self, slot_number: int):
        """Mark a slot specifically for investments"""
        self.configure_slot(
            slot_number=slot_number,
            min_price=0,
            max_price=1000000000,
            purpose="INVESTMENTS"
        )

    # DISCORD WEBHOOK SETTINGS

    def set_discord_webhook(self, webhook_url: str):
        """Set Discord webhook URL and enable notifications"""
        if 'discord_webhook' not in self.config:
            self.config['discord_webhook'] = {
                'enabled': False,
                'url': '',
                'notify_opportunities': True,
                'notify_price_spikes': True,
                'min_score_to_notify': 50
            }

        self.config['discord_webhook']['url'] = webhook_url
        self.config['discord_webhook']['enabled'] = True
        self.save_config()
        print(f"Discord webhook configured and enabled!")

    def disable_discord_webhook(self):
        """Disable Discord notifications"""
        if 'discord_webhook' in self.config:
            self.config['discord_webhook']['enabled'] = False
            self.save_config()
            print("Discord notifications disabled")

    def enable_discord_webhook(self):
        """Enable Discord notifications (must have URL set)"""
        if 'discord_webhook' not in self.config or not self.config['discord_webhook'].get('url'):
            print("Error: Set webhook URL first with set_discord_webhook(url)")
            return
        self.config['discord_webhook']['enabled'] = True
        self.save_config()
        print("Discord notifications enabled")

    def set_discord_min_score(self, min_score: int):
        """Set minimum opportunity score to trigger Discord notification"""
        if 'discord_webhook' not in self.config:
            self.config['discord_webhook'] = {'enabled': False, 'url': '', 'min_score_to_notify': 50}
        self.config['discord_webhook']['min_score_to_notify'] = min_score
        self.save_config()
        print(f"Discord will notify for opportunities scoring {min_score}+")

    def get_discord_config(self) -> dict:
        """Get Discord webhook configuration"""
        return self.config.get('discord_webhook', {
            'enabled': False,
            'url': '',
            'notify_opportunities': True,
            'notify_price_spikes': True,
            'min_score_to_notify': 50
        })

    # PRESET CONFIGURATIONS
    
    def load_preset(self, preset_name: str):
        """Load a preset configuration"""
        presets = {
            'conservative': {
                'risk_tolerance': 'LOW',
                'min_profit_threshold': 200000,
                'offer_time_settings': {'default': 10}
            },
            'balanced': {
                'risk_tolerance': 'MEDIUM',
                'min_profit_threshold': 100000,
                'offer_time_settings': {'default': 5}
            },
            'aggressive': {
                'risk_tolerance': 'HIGH',
                'min_profit_threshold': 50000,
                'offer_time_settings': {'default': 3}
            },
            'high_value_only': {
                'risk_tolerance': 'MEDIUM',
                'min_profit_threshold': 200000,
                'ge_slot_settings': [
                    {'slot': i, 'min_price': 10000000, 'max_price': 1000000000, 'purpose': 'High-value only'}
                    for i in range(1, 9)
                ]
            }
        }
        
        if preset_name not in presets:
            print(f"❌ Unknown preset: {preset_name}")
            print(f"Available presets: {', '.join(presets.keys())}")
            return
        
        # Merge preset into config
        preset_config = presets[preset_name]
        for key, value in preset_config.items():
            self.config[key] = value
        
        self.save_config()
        print(f"✅ Loaded '{preset_name}' preset")
    
    # DISPLAY METHODS
    
    def print_current_config(self):
        """Print current configuration"""
        print("="*80)
        print("CURRENT CONFIGURATION")
        print("="*80)
        print()
        
        print(f"Risk Tolerance: {self.config.get('risk_tolerance', 'MEDIUM')}")
        print(f"Min Profit Threshold: {self.config.get('min_profit_threshold', 100000):,} GP")
        print(f"Default Offer Time: {self.config['offer_time_settings']['default']} minutes")
        print()

        # Discord webhook status
        discord = self.get_discord_config()
        if discord.get('enabled') and discord.get('url'):
            print(f"Discord Webhook: ENABLED (min score: {discord.get('min_score_to_notify', 50)})")
        else:
            print("Discord Webhook: DISABLED")
        print()
        
        print(f"Blocklist: {len(self.config['blocklist'])} items")
        if len(self.config['blocklist']) > 0:
            print(f"  Sample: {', '.join(self.config['blocklist'][:5])}")
            if len(self.config['blocklist']) > 5:
                print(f"  ... and {len(self.config['blocklist']) - 5} more")
        print()
        
        print(self.get_slot_summary())
        print()
        
        custom_times = self.config['offer_time_settings'].get('per_item', {})
        if custom_times:
            print(f"Custom Offer Times: {len(custom_times)} items")
            for item, minutes in list(custom_times.items())[:5]:
                print(f"  {item}: {minutes} minutes")
            if len(custom_times) > 5:
                print(f"  ... and {len(custom_times) - 5} more")
        print()


# Test the configuration system
if __name__ == "__main__":
    print("="*80)
    print("USER CONFIGURATION SYSTEM TEST")
    print("="*80)
    print()
    
    config = UserConfig("test_config.json")
    
    # Test blocklist
    print("TEST 1: Blocklist Management")
    print("─"*80)
    config.add_to_blocklist(['Coal', 'Iron ore', 'Bronze arrow'])
    print(f"Blocklist: {config.get_blocklist()}")
    print(f"Is 'Coal' blocked? {config.is_blocked('Coal')}")
    print()
    
    # Test offer times
    print("TEST 2: Offer Time Settings")
    print("─"*80)
    config.set_default_offer_time(5)
    config.set_offer_time_for_item('Dragon claws', 15)
    config.set_offer_time_for_item('Twisted bow', 30)
    print(f"Default offer time: {config.get_offer_time('Random item')} min")
    print(f"Dragon claws offer time: {config.get_offer_time('Dragon claws')} min")
    print(f"Twisted bow offer time: {config.get_offer_time('Twisted bow')} min")
    print()
    
    # Test slot configuration
    print("TEST 3: GE Slot Configuration")
    print("─"*80)
    config.configure_slot(1, 0, 1000000, "Small flips")
    config.configure_slot(2, 1000000, 10000000, "Medium flips")
    config.configure_slot(3, 10000000, 100000000, "High-value flips")
    config.set_investment_slot(5)
    
    print(config.get_slot_summary())
    print()
    
    print(f"Slots available for 5M item: {config.get_available_slots_for_item(5000000)}")
    print(f"Slots available for 50M item: {config.get_available_slots_for_item(50000000)}")
    print()
    
    # Print full config
    config.print_current_config()
    
    print("="*80)
    print("✅ Configuration system test complete!")
    
    # Clean up test file
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
        print("🗑️  Test config file cleaned up")
