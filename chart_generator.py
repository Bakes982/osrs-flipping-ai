#!/usr/bin/env python3
"""
OSRS Chart Generator - Visual price charts for flip suggestions
Creates professional charts for Discord and dashboard display
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import io
import base64
import os

# Set dark theme for OSRS aesthetic
plt.style.use('dark_background')

# OSRS-inspired color palette
COLORS = {
    'background': '#1a1a2e',
    'panel': '#16213e',
    'green': '#00ff00',
    'red': '#ff4444',
    'gold': '#ffd700',
    'blue': '#4fc3f7',
    'orange': '#ff9800',
    'purple': '#9c27b0',
    'text': '#ffffff',
    'text_dim': '#888888',
    'grid': '#333333',
    'buy_zone': '#00ff0033',
    'sell_zone': '#ff444433',
}


class OSRSChartGenerator:
    """Generate professional trading charts for OSRS flipping"""

    def __init__(self):
        self.output_dir = "charts"
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_price_history(self, item_id: int, hours: int = 24) -> Dict:
        """Fetch price history from OSRS Wiki API"""
        import requests

        try:
            # Get 5-minute timeseries data
            response = requests.get(
                f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=5m&id={item_id}",
                headers={'User-Agent': 'OSRS-AI-Flipper v1.0 - Contact: mike.baker982@hotmail.com'},
                timeout=10
            )
            data = response.json().get('data', [])

            # Filter to requested time range
            cutoff = datetime.now() - timedelta(hours=hours)
            cutoff_ts = cutoff.timestamp()

            filtered = [d for d in data if d.get('timestamp', 0) >= cutoff_ts]

            return {
                'timestamps': [datetime.fromtimestamp(d['timestamp']) for d in filtered],
                'high': [d.get('avgHighPrice') for d in filtered],
                'low': [d.get('avgLowPrice') for d in filtered],
                'high_volume': [d.get('highPriceVolume', 0) for d in filtered],
                'low_volume': [d.get('lowPriceVolume', 0) for d in filtered],
            }
        except Exception as e:
            print(f"Error fetching price history: {e}")
            return {'timestamps': [], 'high': [], 'low': [], 'high_volume': [], 'low_volume': []}

    def create_suggestion_chart(
        self,
        item_name: str,
        item_id: int,
        buy_price: int,
        sell_price: int,
        current_price: int,
        risk_level: str = "MEDIUM",
        verdict: str = "STANDARD_FLIP",
        profit: int = 0,
        margin_pct: float = 0,
        hours: int = 24,
        show_zones: bool = True
    ) -> str:
        """
        Create a visual suggestion chart with buy/sell zones
        Returns path to saved image
        """
        # Fetch price history
        history = self.fetch_price_history(item_id, hours)

        if not history['timestamps']:
            return self._create_no_data_chart(item_name)

        # Create figure with subplots
        fig, (ax_price, ax_volume) = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={'height_ratios': [3, 1]},
            facecolor=COLORS['background']
        )

        # Price chart
        ax_price.set_facecolor(COLORS['panel'])

        # Plot high/low prices
        timestamps = history['timestamps']
        high_prices = [p for p in history['high'] if p]
        low_prices = [p for p in history['low'] if p]

        # Filter None values and align timestamps
        valid_high = [(t, p) for t, p in zip(timestamps, history['high']) if p]
        valid_low = [(t, p) for t, p in zip(timestamps, history['low']) if p]

        if valid_high:
            high_times, high_vals = zip(*valid_high)
            ax_price.plot(high_times, high_vals, color=COLORS['green'], linewidth=2, label='Insta-Buy (High)')

        if valid_low:
            low_times, low_vals = zip(*valid_low)
            ax_price.plot(low_times, low_vals, color=COLORS['red'], linewidth=2, label='Insta-Sell (Low)')

        # Add buy/sell zones
        if show_zones and valid_high and valid_low:
            # Buy zone (around buy_price)
            buy_zone_top = buy_price * 1.01
            buy_zone_bottom = buy_price * 0.99
            ax_price.axhspan(buy_zone_bottom, buy_zone_top, alpha=0.3, color=COLORS['green'], label='Buy Zone')
            ax_price.axhline(y=buy_price, color=COLORS['green'], linestyle='--', linewidth=2, alpha=0.8)

            # Sell zone (around sell_price)
            sell_zone_top = sell_price * 1.01
            sell_zone_bottom = sell_price * 0.99
            ax_price.axhspan(sell_zone_bottom, sell_zone_top, alpha=0.3, color=COLORS['gold'], label='Sell Zone')
            ax_price.axhline(y=sell_price, color=COLORS['gold'], linestyle='--', linewidth=2, alpha=0.8)

            # Current price marker
            ax_price.axhline(y=current_price, color=COLORS['blue'], linestyle=':', linewidth=1.5, alpha=0.8, label='Current')

        # Styling
        ax_price.set_title(
            f"{item_name} - {verdict}",
            fontsize=16, fontweight='bold', color=COLORS['text'], pad=20
        )
        ax_price.set_ylabel('Price (GP)', fontsize=12, color=COLORS['text'])
        ax_price.tick_params(colors=COLORS['text'])
        ax_price.grid(True, alpha=0.3, color=COLORS['grid'])
        ax_price.legend(loc='upper left', facecolor=COLORS['panel'], edgecolor=COLORS['grid'])

        # Format y-axis with GP abbreviations
        ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self._format_gp(x)))

        # Add info box
        info_text = (
            f"Buy: {self._format_gp(buy_price)}\n"
            f"Sell: {self._format_gp(sell_price)}\n"
            f"Profit: {self._format_gp(profit)} ({margin_pct:.1f}%)\n"
            f"Risk: {risk_level}"
        )

        # Risk-based color for info box
        risk_colors = {'LOW': COLORS['green'], 'MEDIUM': COLORS['orange'], 'HIGH': COLORS['red']}
        box_color = risk_colors.get(risk_level, COLORS['orange'])

        props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['panel'], edgecolor=box_color, alpha=0.9)
        ax_price.text(0.98, 0.98, info_text, transform=ax_price.transAxes, fontsize=11,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props, color=COLORS['text'], family='monospace')

        # Volume chart
        ax_volume.set_facecolor(COLORS['panel'])

        if valid_high:
            high_vols = [history['high_volume'][i] for i, p in enumerate(history['high']) if p]
            ax_volume.bar(high_times, high_vols, width=0.002, color=COLORS['green'], alpha=0.7, label='Buy Volume')

        if valid_low:
            low_vols = [history['low_volume'][i] for i, p in enumerate(history['low']) if p]
            ax_volume.bar(low_times, low_vols, width=0.002, color=COLORS['red'], alpha=0.7, label='Sell Volume')

        ax_volume.set_ylabel('Volume', fontsize=12, color=COLORS['text'])
        ax_volume.set_xlabel('Time', fontsize=12, color=COLORS['text'])
        ax_volume.tick_params(colors=COLORS['text'])
        ax_volume.grid(True, alpha=0.3, color=COLORS['grid'])
        ax_volume.legend(loc='upper left', facecolor=COLORS['panel'], edgecolor=COLORS['grid'])

        # Format x-axis
        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_volume.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, hours // 6)))

        plt.tight_layout()

        # Save chart
        filename = f"{self.output_dir}/suggestion_{item_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, facecolor=COLORS['background'], edgecolor='none', bbox_inches='tight')
        plt.close()

        return filename

    def create_dump_alert_chart(
        self,
        item_name: str,
        item_id: int,
        dump_price: int,
        pre_dump_price: int,
        predicted_recovery: int,
        drop_pct: float,
        hours: int = 6
    ) -> str:
        """
        Create a dump alert chart showing the price crash and predicted recovery
        """
        history = self.fetch_price_history(item_id, hours)

        if not history['timestamps']:
            return self._create_no_data_chart(item_name)

        fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
        ax.set_facecolor(COLORS['panel'])

        # Plot prices
        timestamps = history['timestamps']
        valid_high = [(t, p) for t, p in zip(timestamps, history['high']) if p]
        valid_low = [(t, p) for t, p in zip(timestamps, history['low']) if p]

        if valid_high:
            high_times, high_vals = zip(*valid_high)
            ax.plot(high_times, high_vals, color=COLORS['green'], linewidth=2, label='Insta-Buy')

        if valid_low:
            low_times, low_vals = zip(*valid_low)
            ax.plot(low_times, low_vals, color=COLORS['red'], linewidth=2, label='Insta-Sell')

        # Mark the dump
        ax.axhline(y=pre_dump_price, color=COLORS['blue'], linestyle='--', linewidth=1.5, alpha=0.7, label='Pre-Dump Price')
        ax.axhline(y=dump_price, color=COLORS['red'], linestyle='-', linewidth=2, alpha=0.9, label='Dump Price')
        ax.axhline(y=predicted_recovery, color=COLORS['gold'], linestyle='--', linewidth=2, alpha=0.9, label='Predicted Recovery')

        # Fill dump zone
        ax.axhspan(dump_price, pre_dump_price, alpha=0.2, color=COLORS['red'])

        # Recovery zone
        ax.axhspan(dump_price, predicted_recovery, alpha=0.2, color=COLORS['green'])

        # Add dump arrow annotation
        if valid_low:
            last_time = low_times[-1]
            ax.annotate(
                f'DUMP!\n-{drop_pct:.1f}%',
                xy=(last_time, dump_price),
                xytext=(last_time, pre_dump_price),
                fontsize=14, fontweight='bold', color=COLORS['red'],
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=3),
                ha='center'
            )

        # Title with alert styling
        ax.set_title(
            f"DUMP ALERT: {item_name}",
            fontsize=18, fontweight='bold', color=COLORS['red'], pad=20
        )

        # Info box
        potential_profit = predicted_recovery - dump_price
        tax = min(predicted_recovery * 0.02, 5_000_000)
        net_profit = potential_profit - tax

        info_text = (
            f"Pre-Dump: {self._format_gp(pre_dump_price)}\n"
            f"Current: {self._format_gp(dump_price)}\n"
            f"Drop: -{drop_pct:.1f}%\n"
            f"Predicted Recovery: {self._format_gp(predicted_recovery)}\n"
            f"Potential Profit: {self._format_gp(net_profit)}"
        )

        props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['panel'], edgecolor=COLORS['gold'], alpha=0.9)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=props, color=COLORS['text'], family='monospace')

        ax.set_ylabel('Price (GP)', fontsize=12, color=COLORS['text'])
        ax.set_xlabel('Time', fontsize=12, color=COLORS['text'])
        ax.tick_params(colors=COLORS['text'])
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        ax.legend(loc='upper right', facecolor=COLORS['panel'], edgecolor=COLORS['grid'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self._format_gp(x)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.tight_layout()

        filename = f"{self.output_dir}/dump_{item_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, facecolor=COLORS['background'], edgecolor='none', bbox_inches='tight')
        plt.close()

        return filename

    def create_flip_visualization(
        self,
        item_name: str,
        item_id: int,
        buy_price: int,
        buy_time: datetime,
        sell_price: int,
        sell_time: datetime,
        profit: int,
        hours_before: int = 2,
        hours_after: int = 2
    ) -> str:
        """
        Create a visualization of a completed flip showing entry/exit points
        """
        # Calculate time range
        start_time = buy_time - timedelta(hours=hours_before)
        end_time = sell_time + timedelta(hours=hours_after)
        total_hours = int((end_time - start_time).total_seconds() / 3600) + 1

        history = self.fetch_price_history(item_id, total_hours)

        if not history['timestamps']:
            return self._create_no_data_chart(item_name)

        # Filter to our time range
        filtered_times = []
        filtered_high = []
        filtered_low = []

        for i, t in enumerate(history['timestamps']):
            if start_time <= t <= end_time:
                filtered_times.append(t)
                filtered_high.append(history['high'][i])
                filtered_low.append(history['low'][i])

        fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
        ax.set_facecolor(COLORS['panel'])

        # Plot prices
        valid_high = [(t, p) for t, p in zip(filtered_times, filtered_high) if p]
        valid_low = [(t, p) for t, p in zip(filtered_times, filtered_low) if p]

        if valid_high:
            high_times, high_vals = zip(*valid_high)
            ax.plot(high_times, high_vals, color=COLORS['green'], linewidth=2, alpha=0.7)

        if valid_low:
            low_times, low_vals = zip(*valid_low)
            ax.plot(low_times, low_vals, color=COLORS['red'], linewidth=2, alpha=0.7)

        # Mark buy point
        ax.scatter([buy_time], [buy_price], color=COLORS['green'], s=200, zorder=5, marker='^', edgecolors='white', linewidths=2)
        ax.annotate(
            f'BUY\n{self._format_gp(buy_price)}',
            xy=(buy_time, buy_price),
            xytext=(0, -40), textcoords='offset points',
            fontsize=11, fontweight='bold', color=COLORS['green'],
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['panel'], edgecolor=COLORS['green'])
        )

        # Mark sell point
        ax.scatter([sell_time], [sell_price], color=COLORS['gold'], s=200, zorder=5, marker='v', edgecolors='white', linewidths=2)
        ax.annotate(
            f'SELL\n{self._format_gp(sell_price)}',
            xy=(sell_time, sell_price),
            xytext=(0, 40), textcoords='offset points',
            fontsize=11, fontweight='bold', color=COLORS['gold'],
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['panel'], edgecolor=COLORS['gold'])
        )

        # Draw connection line
        ax.plot([buy_time, sell_time], [buy_price, sell_price],
               color=COLORS['blue'], linestyle='--', linewidth=2, alpha=0.5)

        # Fill profit/loss area
        profit_color = COLORS['green'] if profit > 0 else COLORS['red']
        ax.fill_between([buy_time, sell_time], [buy_price, buy_price], [sell_price, sell_price],
                       alpha=0.2, color=profit_color)

        # Title
        profit_str = f"+{self._format_gp(profit)}" if profit > 0 else self._format_gp(profit)
        title_color = COLORS['green'] if profit > 0 else COLORS['red']
        ax.set_title(
            f"Flip: {item_name} | Profit: {profit_str}",
            fontsize=16, fontweight='bold', color=title_color, pad=20
        )

        # Info box
        duration = sell_time - buy_time
        duration_str = f"{duration.total_seconds() / 3600:.1f}h" if duration.total_seconds() > 3600 else f"{duration.total_seconds() / 60:.0f}m"
        margin_pct = ((sell_price - buy_price) / buy_price) * 100

        info_text = (
            f"Buy: {self._format_gp(buy_price)}\n"
            f"Sell: {self._format_gp(sell_price)}\n"
            f"Profit: {profit_str}\n"
            f"Margin: {margin_pct:.2f}%\n"
            f"Duration: {duration_str}"
        )

        props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['panel'], edgecolor=profit_color, alpha=0.9)
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=props, color=COLORS['text'], family='monospace')

        ax.set_ylabel('Price (GP)', fontsize=12, color=COLORS['text'])
        ax.set_xlabel('Time', fontsize=12, color=COLORS['text'])
        ax.tick_params(colors=COLORS['text'])
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self._format_gp(x)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.tight_layout()

        filename = f"{self.output_dir}/flip_{item_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, facecolor=COLORS['background'], edgecolor='none', bbox_inches='tight')
        plt.close()

        return filename

    def create_profit_graph(
        self,
        daily_profits: List[Tuple[datetime, int]],
        cumulative: bool = True
    ) -> str:
        """Create a profit over time graph"""
        if not daily_profits:
            return self._create_no_data_chart("Profit History")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor=COLORS['background'])

        dates, profits = zip(*daily_profits)

        # Cumulative profit chart
        ax1 = axes[0]
        ax1.set_facecolor(COLORS['panel'])

        cumulative_profits = np.cumsum(profits)
        ax1.fill_between(dates, 0, cumulative_profits, alpha=0.3, color=COLORS['green'])
        ax1.plot(dates, cumulative_profits, color=COLORS['green'], linewidth=2)

        ax1.set_title('Cumulative Profit', fontsize=14, fontweight='bold', color=COLORS['text'])
        ax1.set_ylabel('Total Profit (GP)', fontsize=12, color=COLORS['text'])
        ax1.tick_params(colors=COLORS['text'])
        ax1.grid(True, alpha=0.3, color=COLORS['grid'])
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self._format_gp(x)))

        # Daily profit chart
        ax2 = axes[1]
        ax2.set_facecolor(COLORS['panel'])

        colors = [COLORS['green'] if p >= 0 else COLORS['red'] for p in profits]
        ax2.bar(dates, profits, color=colors, alpha=0.8, width=0.8)

        ax2.set_title('Daily Profit/Loss', fontsize=14, fontweight='bold', color=COLORS['text'])
        ax2.set_ylabel('Daily P/L (GP)', fontsize=12, color=COLORS['text'])
        ax2.set_xlabel('Date', fontsize=12, color=COLORS['text'])
        ax2.tick_params(colors=COLORS['text'])
        ax2.grid(True, alpha=0.3, color=COLORS['grid'])
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self._format_gp(x)))
        ax2.axhline(y=0, color=COLORS['text_dim'], linestyle='-', linewidth=1)

        plt.tight_layout()

        filename = f"{self.output_dir}/profit_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, facecolor=COLORS['background'], edgecolor='none', bbox_inches='tight')
        plt.close()

        return filename

    def _create_no_data_chart(self, title: str) -> str:
        """Create a placeholder chart when no data is available"""
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['background'])
        ax.set_facecolor(COLORS['panel'])
        ax.text(0.5, 0.5, 'No price data available', ha='center', va='center',
               fontsize=20, color=COLORS['text_dim'], transform=ax.transAxes)
        ax.set_title(title, fontsize=16, fontweight='bold', color=COLORS['text'])
        ax.set_xticks([])
        ax.set_yticks([])

        filename = f"{self.output_dir}/no_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, facecolor=COLORS['background'], edgecolor='none', bbox_inches='tight')
        plt.close()

        return filename

    def _format_gp(self, value: float) -> str:
        """Format GP values with K/M/B suffixes"""
        if value >= 1_000_000_000:
            return f"{value/1_000_000_000:.1f}B"
        elif value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value/1_000:.1f}K"
        else:
            return f"{value:.0f}"

    def get_chart_as_base64(self, filepath: str) -> str:
        """Convert chart image to base64 for embedding"""
        with open(filepath, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')


# Singleton instance
_chart_generator = None

def get_chart_generator() -> OSRSChartGenerator:
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = OSRSChartGenerator()
    return _chart_generator


if __name__ == '__main__':
    # Test chart generation
    print("Testing OSRS Chart Generator...")

    gen = get_chart_generator()

    # Test suggestion chart
    chart_path = gen.create_suggestion_chart(
        item_name="Dragon claws",
        item_id=13652,
        buy_price=47_000_000,
        sell_price=48_000_000,
        current_price=47_500_000,
        risk_level="MEDIUM",
        verdict="STANDARD_FLIP",
        profit=40_000,
        margin_pct=0.85,
        hours=24
    )
    print(f"Suggestion chart saved: {chart_path}")

    # Test dump chart
    dump_path = gen.create_dump_alert_chart(
        item_name="Twisted bow",
        item_id=20997,
        dump_price=1_400_000_000,
        pre_dump_price=1_500_000_000,
        predicted_recovery=1_480_000_000,
        drop_pct=6.7,
        hours=6
    )
    print(f"Dump alert chart saved: {dump_path}")

    print("\nCharts generated successfully!")
