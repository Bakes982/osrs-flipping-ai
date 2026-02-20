"""
Feature Engine for OSRS Flipping AI
Computes feature vectors for ML models. Each item gets a feature vector
recomputed every 60 seconds across multiple time horizons.
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from backend.database import (
    PriceSnapshot, FlipHistory, get_db, get_price_history, get_item_flips,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONS = ["1m", "5m", "30m", "2h", "8h", "24h"]
HORIZON_SECONDS = {
    "1m": 60,
    "5m": 300,
    "30m": 1800,
    "2h": 7200,
    "8h": 28800,
    "24h": 86400,
}

# SMA windows in minutes
SMA_WINDOWS = {
    "5m": 5,
    "30m": 30,
    "2h": 120,
    "24h": 1440,
}

# RSI period count (scaled per horizon)
RSI_PERIODS = 14

# Bollinger Band parameters
BB_PERIOD_MINUTES = 120
BB_NUM_STD = 2.0

# MACD parameters (in periods, scaled per horizon)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Stochastic parameters
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

# ATR period
ATR_PERIOD = 14


class FeatureEngine:
    """
    Computes a full feature vector for a given item using its price
    snapshots and flip history. Returns a flat dict of feature_name -> float.
    """

    def compute_features(
        self,
        item_id: int,
        snapshots: Optional[List[PriceSnapshot]] = None,
        flips: Optional[List[FlipHistory]] = None,
    ) -> Dict[str, float]:
        """
        Compute the complete feature vector for an item.

        Parameters
        ----------
        item_id : int
            OSRS item ID.
        snapshots : list of PriceSnapshot, optional
            Price history. If None, fetched from DB (last 48h).
        flips : list of FlipHistory, optional
            Flip history. If None, fetched from DB (last 30 days).

        Returns
        -------
        dict
            feature_name -> float value. Missing data defaults to 0.0.
        """
        if snapshots is None or flips is None:
            db = get_db()
            try:
                if snapshots is None:
                    snapshots = get_price_history(db, item_id, hours=48)
                if flips is None:
                    flips = get_item_flips(db, item_id, days=30)
            finally:
                db.close()

        features: Dict[str, float] = {}

        # Price features
        features.update(self._price_features(snapshots))

        # Volume features
        features.update(self._volume_features(snapshots))

        # Technical indicators
        features.update(self._technical_features(snapshots))

        # Temporal features
        features.update(self._temporal_features())

        # Historical flip features
        features.update(self._historical_features(flips))

        # Microstructure features (order flow, spread dynamics)
        features.update(self._microstructure_features(snapshots))

        # Volatility regime features (regime detection, support/resistance)
        features.update(self._regime_features(snapshots))

        return features

    # ------------------------------------------------------------------
    # Price Features
    # ------------------------------------------------------------------

    def _price_features(self, snapshots: List[PriceSnapshot]) -> Dict[str, float]:
        features: Dict[str, float] = {}

        current_price = self._latest_buy_price(snapshots)
        features["current_price"] = current_price

        if current_price == 0.0:
            # Can't compute relative features without a valid price
            for key in [
                "price_vs_sma_5m", "price_vs_sma_30m", "price_vs_sma_2h",
                "price_vs_sma_24h", "bollinger_position", "momentum_1x",
                "momentum_2x", "momentum_4x", "vwap_deviation", "spread_pct",
                "price_vs_24h_high", "price_vs_24h_low",
            ]:
                features[key] = 0.0
            return features

        # SMA deviations
        for label, minutes in SMA_WINDOWS.items():
            sma = self._sma(snapshots, minutes)
            key = f"price_vs_sma_{label}"
            if sma and sma > 0:
                features[key] = (current_price - sma) / sma
            else:
                features[key] = 0.0

        # Bollinger position
        features["bollinger_position"] = self._bollinger_position(snapshots, current_price)

        # Momentum at 1x, 2x, 4x the base lookback (5 minutes)
        base_lookback_min = 5
        features["momentum_1x"] = self._rate_of_change(snapshots, base_lookback_min)
        features["momentum_2x"] = self._rate_of_change(snapshots, base_lookback_min * 2)
        features["momentum_4x"] = self._rate_of_change(snapshots, base_lookback_min * 4)

        # VWAP deviation
        vwap = self._vwap(snapshots, minutes=30)
        if vwap and vwap > 0:
            features["vwap_deviation"] = (current_price - vwap) / vwap
        else:
            features["vwap_deviation"] = 0.0

        # Spread percentage
        latest_sell = self._latest_sell_price(snapshots)
        if latest_sell and latest_sell > 0:
            features["spread_pct"] = (current_price - latest_sell) / latest_sell * 100.0
        else:
            features["spread_pct"] = 0.0

        # Price vs 24h high/low
        high_24h, low_24h = self._high_low(snapshots, minutes=1440)
        features["price_vs_24h_high"] = current_price / high_24h if high_24h > 0 else 0.0
        features["price_vs_24h_low"] = current_price / low_24h if low_24h > 0 else 0.0

        return features

    # ------------------------------------------------------------------
    # Volume Features
    # ------------------------------------------------------------------

    def _volume_features(self, snapshots: List[PriceSnapshot]) -> Dict[str, float]:
        features: Dict[str, float] = {}

        # Current 5-minute volume
        current_5m_vol = self._volume_in_window(snapshots, minutes=5)

        # Average 24h volume per 5-minute bucket
        total_24h_vol = self._volume_in_window(snapshots, minutes=1440)
        buckets_24h = 1440 / 5  # 288 five-minute buckets in 24h
        avg_5m_vol = total_24h_vol / buckets_24h if buckets_24h > 0 else 0.0

        features["volume_ratio"] = (
            current_5m_vol / avg_5m_vol if avg_5m_vol > 0 else 0.0
        )

        # Volume trend: slope of volume over last 30 minutes (6 buckets of 5m)
        features["volume_trend"] = self._volume_slope(snapshots, window_minutes=30, bucket_minutes=5)

        # Buy/sell ratio
        buy_vol, sell_vol = self._buy_sell_volumes(snapshots, minutes=30)
        total = buy_vol + sell_vol
        features["buy_sell_ratio"] = buy_vol / total if total > 0 else 0.5

        # Volume-price divergence
        # Positive = volume up + price down (accumulation)
        # Negative = volume down + price up (distribution)
        vol_change = self._volume_slope(snapshots, window_minutes=30, bucket_minutes=5)
        price_change = self._rate_of_change(snapshots, minutes=30)
        if vol_change > 0 and price_change < 0:
            features["volume_price_divergence"] = abs(vol_change)  # accumulation
        elif vol_change < 0 and price_change > 0:
            features["volume_price_divergence"] = -abs(vol_change)  # distribution
        else:
            features["volume_price_divergence"] = 0.0

        return features

    # ------------------------------------------------------------------
    # Technical Indicators
    # ------------------------------------------------------------------

    def _technical_features(self, snapshots: List[PriceSnapshot]) -> Dict[str, float]:
        features: Dict[str, float] = {}

        prices = self._extract_prices(snapshots)

        # RSI (14 periods)
        features["rsi_14"] = self._rsi(prices, RSI_PERIODS)

        # Z-Score (24h)
        prices_24h = self._extract_prices_in_window(snapshots, minutes=1440)
        if len(prices_24h) >= 2:
            mean_24h = statistics.mean(prices_24h)
            stdev_24h = statistics.stdev(prices_24h)
            current = prices[-1] if prices else 0.0
            features["z_score"] = (
                (current - mean_24h) / stdev_24h if stdev_24h > 0 else 0.0
            )
        else:
            features["z_score"] = 0.0

        # MACD
        macd_line, signal_line, histogram = self._macd(prices)
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = histogram

        # Stochastic
        k_val, d_val = self._stochastic(prices, STOCH_K_PERIOD, STOCH_D_PERIOD)
        features["stochastic_k"] = k_val
        features["stochastic_d"] = d_val

        # ATR (Average True Range)
        features["atr"] = self._atr(snapshots, ATR_PERIOD)

        return features

    # ------------------------------------------------------------------
    # Temporal Features
    # ------------------------------------------------------------------

    def _temporal_features(self) -> Dict[str, float]:
        """Time-based cyclical features."""
        now = datetime.utcnow()
        hour = now.hour + now.minute / 60.0
        dow = now.weekday()  # Monday=0, Sunday=6

        features: Dict[str, float] = {}
        features["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
        features["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
        features["dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
        features["dow_cos"] = math.cos(2 * math.pi * dow / 7.0)

        # Minutes until peak trading window (18:00-22:00 GMT)
        peak_start = 18 * 60  # 1080 minutes
        peak_end = 22 * 60    # 1320 minutes
        current_minutes = now.hour * 60 + now.minute

        if peak_start <= current_minutes <= peak_end:
            features["minutes_to_peak"] = 0.0
        elif current_minutes < peak_start:
            features["minutes_to_peak"] = float(peak_start - current_minutes)
        else:
            # After peak, minutes until tomorrow's peak
            features["minutes_to_peak"] = float(1440 - current_minutes + peak_start)

        features["is_weekend"] = 1.0 if dow >= 5 else 0.0

        return features

    # ------------------------------------------------------------------
    # Historical Features (from FlipHistory)
    # ------------------------------------------------------------------

    def _historical_features(self, flips: List[FlipHistory]) -> Dict[str, float]:
        features: Dict[str, float] = {}

        if not flips:
            features["win_rate"] = 0.0
            features["avg_flip_duration"] = 0.0
            features["avg_profit_per_flip"] = 0.0
            features["consistency_score"] = 0.0
            return features

        profits = [f.net_profit for f in flips]
        durations = [f.duration_seconds for f in flips]

        profitable = sum(1 for p in profits if p > 0)
        features["win_rate"] = profitable / len(profits) if profits else 0.0
        features["avg_flip_duration"] = statistics.mean(durations) if durations else 0.0
        features["avg_profit_per_flip"] = statistics.mean(profits) if profits else 0.0

        # Consistency: stdev / mean (lower = more consistent). Use abs(mean) to handle
        # edge cases where mean is near zero or negative.
        if len(profits) >= 2:
            mean_p = statistics.mean(profits)
            stdev_p = statistics.stdev(profits)
            features["consistency_score"] = (
                stdev_p / abs(mean_p) if abs(mean_p) > 0 else 0.0
            )
        else:
            features["consistency_score"] = 0.0

        return features

    # ------------------------------------------------------------------
    # Microstructure Features (order flow, spread dynamics)
    # ------------------------------------------------------------------

    def _microstructure_features(self, snapshots: List[PriceSnapshot]) -> Dict[str, float]:
        features: Dict[str, float] = {}

        # --- Order Flow Imbalance (OFI) ---
        # Measures buying vs selling pressure from volume asymmetry
        buy_5m, sell_5m = self._buy_sell_volumes(snapshots, minutes=5)
        buy_15m, sell_15m = self._buy_sell_volumes(snapshots, minutes=15)
        total_5m = buy_5m + sell_5m
        total_15m = buy_15m + sell_15m
        features["ofi_5m"] = (buy_5m - sell_5m) / total_5m if total_5m > 0 else 0.0
        features["ofi_15m"] = (buy_15m - sell_15m) / total_15m if total_15m > 0 else 0.0

        # --- Spread Volatility ---
        # High spread volatility = uncertain market, harder to flip
        cutoff_30m = datetime.utcnow() - timedelta(minutes=30)
        spreads = []
        for s in snapshots:
            if s.timestamp >= cutoff_30m and s.instant_buy and s.instant_sell:
                if s.instant_sell > 0:
                    spreads.append((s.instant_buy - s.instant_sell) / s.instant_sell)
        if len(spreads) >= 3:
            features["spread_volatility"] = statistics.stdev(spreads)
            features["spread_mean"] = statistics.mean(spreads)
        else:
            features["spread_volatility"] = 0.0
            features["spread_mean"] = 0.0

        # --- Price Efficiency Ratio ---
        # Directional movement / total movement. 1.0 = trending, 0.0 = choppy
        prices_30m = self._extract_prices_in_window(snapshots, minutes=30)
        if len(prices_30m) >= 5:
            directional = abs(prices_30m[-1] - prices_30m[0])
            total_movement = sum(
                abs(prices_30m[i] - prices_30m[i - 1])
                for i in range(1, len(prices_30m))
            )
            features["efficiency_ratio"] = (
                directional / total_movement if total_movement > 0 else 0.0
            )
        else:
            features["efficiency_ratio"] = 0.0

        # --- Momentum Acceleration ---
        # Rate of change of momentum (is the trend accelerating or decelerating?)
        mom_1x = self._rate_of_change(snapshots, 5)
        mom_2x = self._rate_of_change(snapshots, 10)
        features["momentum_acceleration"] = mom_1x - mom_2x

        # --- Volume-Weighted Momentum ---
        # Weight price changes by volume so high-volume moves count more
        cutoff_15m = datetime.utcnow() - timedelta(minutes=15)
        vw_changes = []
        vw_total_vol = 0.0
        prev_price = None
        for s in snapshots:
            if s.timestamp >= cutoff_15m and s.instant_buy and s.instant_buy > 0:
                vol = (s.buy_volume or 0) + (s.sell_volume or 0)
                if prev_price and prev_price > 0 and vol > 0:
                    change = (s.instant_buy - prev_price) / prev_price
                    vw_changes.append(change * vol)
                    vw_total_vol += vol
                prev_price = s.instant_buy
        features["volume_weighted_momentum"] = (
            sum(vw_changes) / vw_total_vol if vw_total_vol > 0 else 0.0
        )

        # --- Relative Spread Position ---
        # Current spread relative to recent spread range
        if len(spreads) >= 5:
            max_spread = max(spreads)
            min_spread = min(spreads)
            current_spread = spreads[-1] if spreads else 0.0
            spread_range = max_spread - min_spread
            features["relative_spread_pos"] = (
                (current_spread - min_spread) / spread_range
                if spread_range > 0 else 0.5
            )
        else:
            features["relative_spread_pos"] = 0.5

        return features

    # ------------------------------------------------------------------
    # Volatility Regime Features
    # ------------------------------------------------------------------

    def _regime_features(self, snapshots: List[PriceSnapshot]) -> Dict[str, float]:
        features: Dict[str, float] = {}

        # --- Volatility Regime Detection ---
        # Compare short-term vol to long-term vol to detect regime changes
        prices_1h = self._extract_prices_in_window(snapshots, minutes=60)
        prices_4h = self._extract_prices_in_window(snapshots, minutes=240)

        if len(prices_1h) >= 5:
            returns_1h = [
                (prices_1h[i] - prices_1h[i - 1]) / prices_1h[i - 1]
                for i in range(1, len(prices_1h))
                if prices_1h[i - 1] > 0
            ]
            vol_1h = statistics.stdev(returns_1h) if len(returns_1h) >= 2 else 0.0
        else:
            vol_1h = 0.0

        if len(prices_4h) >= 10:
            returns_4h = [
                (prices_4h[i] - prices_4h[i - 1]) / prices_4h[i - 1]
                for i in range(1, len(prices_4h))
                if prices_4h[i - 1] > 0
            ]
            vol_4h = statistics.stdev(returns_4h) if len(returns_4h) >= 2 else 0.0
        else:
            vol_4h = 0.0

        # Vol ratio > 1 means short-term volatility is elevated (regime shift)
        features["vol_regime_ratio"] = vol_1h / vol_4h if vol_4h > 0 else 1.0
        features["realized_vol_1h"] = vol_1h
        features["realized_vol_4h"] = vol_4h

        # --- Support/Resistance Proximity ---
        # How close is current price to recent highs/lows (key decision levels)
        prices_2h = self._extract_prices_in_window(snapshots, minutes=120)
        current = self._latest_buy_price(snapshots)

        if len(prices_2h) >= 10 and current > 0:
            high_2h = max(prices_2h)
            low_2h = min(prices_2h)
            price_range = high_2h - low_2h

            if price_range > 0:
                # 0 = at low (support), 1 = at high (resistance)
                features["support_resistance_pos"] = (current - low_2h) / price_range
                # Distance to resistance as % (small = near resistance)
                features["dist_to_resistance_pct"] = (high_2h - current) / current * 100
                # Distance to support as % (small = near support)
                features["dist_to_support_pct"] = (current - low_2h) / current * 100
            else:
                features["support_resistance_pos"] = 0.5
                features["dist_to_resistance_pct"] = 0.0
                features["dist_to_support_pct"] = 0.0
        else:
            features["support_resistance_pos"] = 0.5
            features["dist_to_resistance_pct"] = 0.0
            features["dist_to_support_pct"] = 0.0

        # --- Mean Reversion Strength ---
        # How strongly prices revert to mean (measured by autocorrelation of returns)
        if len(prices_1h) >= 10:
            returns = [
                (prices_1h[i] - prices_1h[i - 1]) / prices_1h[i - 1]
                for i in range(1, len(prices_1h))
                if prices_1h[i - 1] > 0
            ]
            if len(returns) >= 6:
                # Lag-1 autocorrelation: negative = mean-reverting, positive = trending
                n = len(returns)
                mean_r = statistics.mean(returns)
                var = sum((r - mean_r) ** 2 for r in returns)
                if var > 0:
                    cov = sum(
                        (returns[i] - mean_r) * (returns[i - 1] - mean_r)
                        for i in range(1, n)
                    )
                    features["return_autocorr"] = cov / var
                else:
                    features["return_autocorr"] = 0.0
            else:
                features["return_autocorr"] = 0.0
        else:
            features["return_autocorr"] = 0.0

        # --- Price Concentration ---
        # What fraction of recent time was price near current level (±1%)
        if len(prices_1h) >= 5 and current > 0:
            near_count = sum(
                1 for p in prices_1h
                if abs(p - current) / current < 0.01
            )
            features["price_concentration"] = near_count / len(prices_1h)
        else:
            features["price_concentration"] = 0.0

        return features

    # ------------------------------------------------------------------
    # Utility / Helper Methods
    # ------------------------------------------------------------------

    def _latest_buy_price(self, snapshots: List[PriceSnapshot]) -> float:
        """Return the most recent instant_buy price, or 0.0."""
        for s in reversed(snapshots):
            if s.instant_buy and s.instant_buy > 0:
                return float(s.instant_buy)
        return 0.0

    def _latest_sell_price(self, snapshots: List[PriceSnapshot]) -> float:
        """Return the most recent instant_sell price, or 0.0."""
        for s in reversed(snapshots):
            if s.instant_sell and s.instant_sell > 0:
                return float(s.instant_sell)
        return 0.0

    def _sma(self, snapshots: List[PriceSnapshot], minutes: int) -> Optional[float]:
        """Simple Moving Average over a time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        prices = [
            s.instant_buy
            for s in snapshots
            if s.timestamp >= cutoff and s.instant_buy and s.instant_buy > 0
        ]
        return statistics.mean(prices) if prices else None

    def _vwap(self, snapshots: List[PriceSnapshot], minutes: int) -> Optional[float]:
        """Volume-Weighted Average Price over a time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        relevant = [s for s in snapshots if s.timestamp >= cutoff]
        if not relevant:
            return None

        total_pv = 0.0
        total_v = 0.0
        for s in relevant:
            price = s.instant_buy
            vol = (s.buy_volume or 0) + (s.sell_volume or 0)
            if price and price > 0:
                vol = max(vol, 1)
                total_pv += price * vol
                total_v += vol

        return total_pv / total_v if total_v > 0 else None

    def _bollinger_position(
        self, snapshots: List[PriceSnapshot], current_price: float,
    ) -> float:
        """Bollinger Band position: 0.0 (lower) to 1.0 (upper)."""
        cutoff = datetime.utcnow() - timedelta(minutes=BB_PERIOD_MINUTES)
        prices = [
            s.instant_buy
            for s in snapshots
            if s.timestamp >= cutoff and s.instant_buy and s.instant_buy > 0
        ]
        if len(prices) < 5:
            return 0.5  # neutral default

        middle = statistics.mean(prices)
        stdev = statistics.stdev(prices)
        if stdev == 0:
            return 0.5

        upper = middle + BB_NUM_STD * stdev
        lower = middle - BB_NUM_STD * stdev
        band_width = upper - lower
        if band_width <= 0:
            return 0.5

        position = (current_price - lower) / band_width
        return max(0.0, min(1.0, position))

    def _rate_of_change(self, snapshots: List[PriceSnapshot], minutes: int) -> float:
        """Rate of change: (current - past) / past. Returns 0.0 on missing data."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        current = self._latest_buy_price(snapshots)
        if current == 0.0:
            return 0.0

        # Find the price closest to cutoff
        past_price = 0.0
        for s in snapshots:
            if s.timestamp >= cutoff and s.instant_buy and s.instant_buy > 0:
                past_price = float(s.instant_buy)
                break

        if past_price == 0.0:
            return 0.0

        return (current - past_price) / past_price

    def _high_low(
        self, snapshots: List[PriceSnapshot], minutes: int,
    ) -> Tuple[float, float]:
        """24h high and low prices."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        prices = [
            s.instant_buy
            for s in snapshots
            if s.timestamp >= cutoff and s.instant_buy and s.instant_buy > 0
        ]
        if not prices:
            return 0.0, 0.0
        return float(max(prices)), float(min(prices))

    def _volume_in_window(self, snapshots: List[PriceSnapshot], minutes: int) -> float:
        """Total volume (buy + sell) in a time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        total = 0.0
        for s in snapshots:
            if s.timestamp >= cutoff:
                total += (s.buy_volume or 0) + (s.sell_volume or 0)
        return total

    def _buy_sell_volumes(
        self, snapshots: List[PriceSnapshot], minutes: int,
    ) -> Tuple[float, float]:
        """Separate buy and sell volumes in a time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        buy_vol = 0.0
        sell_vol = 0.0
        for s in snapshots:
            if s.timestamp >= cutoff:
                buy_vol += s.buy_volume or 0
                sell_vol += s.sell_volume or 0
        return buy_vol, sell_vol

    def _volume_slope(
        self,
        snapshots: List[PriceSnapshot],
        window_minutes: int = 30,
        bucket_minutes: int = 5,
    ) -> float:
        """
        Linear slope of volume over time buckets.
        Positive = increasing volume, negative = decreasing.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        relevant = [s for s in snapshots if s.timestamp >= cutoff]
        if len(relevant) < 2:
            return 0.0

        # Group into buckets
        num_buckets = max(1, window_minutes // bucket_minutes)
        bucket_vols: List[float] = [0.0] * num_buckets

        bucket_start = cutoff
        for i in range(num_buckets):
            bucket_end = bucket_start + timedelta(minutes=bucket_minutes)
            for s in relevant:
                if bucket_start <= s.timestamp < bucket_end:
                    bucket_vols[i] += (s.buy_volume or 0) + (s.sell_volume or 0)
            bucket_start = bucket_end

        # Simple linear regression slope
        n = len(bucket_vols)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = statistics.mean(bucket_vols)
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(bucket_vols))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator > 0 else 0.0

    def _extract_prices(self, snapshots: List[PriceSnapshot]) -> List[float]:
        """Extract all valid instant_buy prices in chronological order."""
        return [
            float(s.instant_buy)
            for s in snapshots
            if s.instant_buy and s.instant_buy > 0
        ]

    def _extract_prices_in_window(
        self, snapshots: List[PriceSnapshot], minutes: int,
    ) -> List[float]:
        """Extract prices within a time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            float(s.instant_buy)
            for s in snapshots
            if s.timestamp >= cutoff and s.instant_buy and s.instant_buy > 0
        ]

    def _rsi(self, prices: List[float], periods: int = 14) -> float:
        """
        Relative Strength Index.
        Returns 0-100 scale. 50.0 if insufficient data.
        """
        if len(prices) < periods + 1:
            return 50.0

        # Use the most recent (periods + 1) prices
        recent = prices[-(periods + 1):]
        gains: List[float] = []
        losses: List[float] = []

        for i in range(1, len(recent)):
            change = recent[i] - recent[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))

        avg_gain = statistics.mean(gains) if gains else 0.0
        avg_loss = statistics.mean(losses) if losses else 0.0

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _ema(self, values: List[float], period: int) -> List[float]:
        """Exponential Moving Average."""
        if not values:
            return []
        if len(values) < period:
            return [statistics.mean(values)] * len(values)

        multiplier = 2.0 / (period + 1)
        ema_vals = [statistics.mean(values[:period])]

        for i in range(period, len(values)):
            ema_val = (values[i] - ema_vals[-1]) * multiplier + ema_vals[-1]
            ema_vals.append(ema_val)

        return ema_vals

    def _macd(
        self, prices: List[float],
    ) -> Tuple[float, float, float]:
        """
        MACD indicator.
        Returns (macd_line, signal_line, histogram).
        """
        if len(prices) < MACD_SLOW + MACD_SIGNAL:
            return 0.0, 0.0, 0.0

        fast_ema = self._ema(prices, MACD_FAST)
        slow_ema = self._ema(prices, MACD_SLOW)

        # Align lengths: slow_ema starts later
        offset = len(fast_ema) - len(slow_ema)
        macd_line_series = [
            fast_ema[offset + i] - slow_ema[i] for i in range(len(slow_ema))
        ]

        if len(macd_line_series) < MACD_SIGNAL:
            return 0.0, 0.0, 0.0

        signal_series = self._ema(macd_line_series, MACD_SIGNAL)

        macd_val = macd_line_series[-1] if macd_line_series else 0.0
        signal_val = signal_series[-1] if signal_series else 0.0
        histogram = macd_val - signal_val

        return macd_val, signal_val, histogram

    def _stochastic(
        self, prices: List[float], k_period: int, d_period: int,
    ) -> Tuple[float, float]:
        """
        Stochastic oscillator %K and %D.
        Returns values on 0-100 scale. Defaults to 50.0 on insufficient data.
        """
        if len(prices) < k_period:
            return 50.0, 50.0

        # Calculate %K values for the last d_period windows
        k_values: List[float] = []
        for i in range(max(0, len(prices) - d_period - k_period + 1), len(prices) - k_period + 1):
            window = prices[i : i + k_period]
            high = max(window)
            low = min(window)
            current = window[-1]
            if high == low:
                k_values.append(50.0)
            else:
                k_values.append((current - low) / (high - low) * 100.0)

        k_val = k_values[-1] if k_values else 50.0
        d_val = statistics.mean(k_values[-d_period:]) if len(k_values) >= d_period else k_val

        return k_val, d_val

    def _atr(self, snapshots: List[PriceSnapshot], period: int = 14) -> float:
        """
        Average True Range using snapshot high/low approximation.
        Since we don't have OHLC candles, we approximate TR using
        instant_buy as high and instant_sell as low.
        """
        if len(snapshots) < period + 1:
            return 0.0

        recent = snapshots[-(period + 1):]
        true_ranges: List[float] = []

        for i in range(1, len(recent)):
            high = recent[i].instant_buy or 0
            low = recent[i].instant_sell or 0
            prev_close = recent[i - 1].instant_buy or 0

            if high == 0 or low == 0 or prev_close == 0:
                continue

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(float(tr))

        return statistics.mean(true_ranges) if true_ranges else 0.0

    # ------------------------------------------------------------------
    # Feature Names (for model training / column alignment)
    # ------------------------------------------------------------------

    @staticmethod
    def feature_names() -> List[str]:
        """Return the ordered list of all feature names."""
        return [
            # Price features
            "current_price",
            "price_vs_sma_5m",
            "price_vs_sma_30m",
            "price_vs_sma_2h",
            "price_vs_sma_24h",
            "bollinger_position",
            "momentum_1x",
            "momentum_2x",
            "momentum_4x",
            "vwap_deviation",
            "spread_pct",
            "price_vs_24h_high",
            "price_vs_24h_low",
            # Volume features
            "volume_ratio",
            "volume_trend",
            "buy_sell_ratio",
            "volume_price_divergence",
            # Technical indicators
            "rsi_14",
            "z_score",
            "macd",
            "macd_signal",
            "macd_histogram",
            "stochastic_k",
            "stochastic_d",
            "atr",
            # Temporal features
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "minutes_to_peak",
            "is_weekend",
            # Historical features
            "win_rate",
            "avg_flip_duration",
            "avg_profit_per_flip",
            "consistency_score",
            # Microstructure features
            "ofi_5m",
            "ofi_15m",
            "spread_volatility",
            "spread_mean",
            "efficiency_ratio",
            "momentum_acceleration",
            "volume_weighted_momentum",
            "relative_spread_pos",
            # Regime features
            "vol_regime_ratio",
            "realized_vol_1h",
            "realized_vol_4h",
            "support_resistance_pos",
            "dist_to_resistance_pct",
            "dist_to_support_pct",
            "return_autocorr",
            "price_concentration",
        ]
