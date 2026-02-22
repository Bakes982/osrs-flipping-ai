"""
OSRS Flipping AI — System-wide constants.

Every magic number lives here. If you find a literal in the codebase that is
not a local variable, it belongs here instead.
"""

# ---------------------------------------------------------------------------
# Grand Exchange Tax
# ---------------------------------------------------------------------------

GE_TAX_RATE: float = 0.02        # 2% on each sale
GE_TAX_CAP: int = 5_000_000      # Maximum tax per single item sold

# Minimum item value below which the GE does NOT charge tax (confirmed Wiki).
GE_TAX_FREE_BELOW: int = 100

# ---------------------------------------------------------------------------
# Price brackets — validated against 1,522 Copilot trades
# ---------------------------------------------------------------------------

# Sweet-spot bracket: 93% win-rate, 144k GP avg/flip
PRICE_BRACKET_OPTIMAL_LOW: int = 10_000_000
PRICE_BRACKET_OPTIMAL_HIGH: int = 50_000_000

# Score multipliers per bracket
PRICE_MULTIPLIER_OPTIMAL: float = 1.15    # 10M – 50M
PRICE_MULTIPLIER_HIGH: float = 1.10       # > 50M
PRICE_MULTIPLIER_SOLID: float = 1.05      # 1M – 10M
PRICE_MULTIPLIER_BULK: float = 1.05       # < 10K  (bulk flipping still works)
PRICE_MULTIPLIER_WEAK: float = 0.90       # 100K – 1M (historically worst bracket)
PRICE_MULTIPLIER_NEUTRAL: float = 1.0     # 10K – 100K

# ---------------------------------------------------------------------------
# Spread / margin thresholds
# ---------------------------------------------------------------------------

# Hard veto: spreads wider than this are illiquid traps
SPREAD_MAX_PCT: float = 12.0

# Realized margin sweet spots (after SmartPricer + tax)
MARGIN_OPTIMAL_LOW: float = 1.0    # 1–2% → 94% WR in empirical data
MARGIN_OPTIMAL_HIGH: float = 2.0
MARGIN_DECENT_LOW: float = 0.5     # 0.5–1% → still great (89% WR)
MARGIN_THIN_LOW: float = 0.3       # Below this: tax risk; above this: workable
MARGIN_WIDE_HIGH: float = 3.0      # 2–3%: diminishing returns
MARGIN_VERY_WIDE: float = 5.0      # > 5%: likely illiquid trap

# ---------------------------------------------------------------------------
# Volume thresholds (5-minute window)
# ---------------------------------------------------------------------------

# High-value items trade infrequently — scale thresholds down with price.
VOL_HIGH_PRICE_THRESHOLD: int = 50_000_000   # Items >= 50M
VOL_MID_PRICE_THRESHOLD: int = 10_000_000    # Items >= 10M

# Per-bracket "excellent" volume
VOL_EXCELLENT_HIGH_PRICE: int = 10   # >= 50M item: 10 trades/5m = 100%
VOL_EXCELLENT_MID_PRICE: int = 30    # >= 10M item
VOL_EXCELLENT_LOW_PRICE: int = 100   # < 10M item

# ---------------------------------------------------------------------------
# Stale data limits
# ---------------------------------------------------------------------------

# High-value items trade infrequently, allow longer staleness before veto.
STALE_LIMIT_MINUTES_HIGH_VALUE: int = 120   # items >= 10M: up to 2 hours
STALE_LIMIT_MINUTES_DEFAULT: int = 45       # everything else: 45 minutes
STALE_VOLUME_THRESHOLD_HIGH: int = 2        # min volume to allow stale hi-val
STALE_VOLUME_THRESHOLD_DEFAULT: int = 5

# ---------------------------------------------------------------------------
# Waterfall crash detection
# ---------------------------------------------------------------------------

WATERFALL_BUCKET_MINUTES: int = 5
WATERFALL_MIN_BUCKETS: int = 4           # need 4 buckets (3 intervals) to detect
WATERFALL_MIN_CONSECUTIVE_DROPS: int = 3
WATERFALL_TOTAL_DROP_THRESHOLD: float = -0.05  # total drop > 5%

# ---------------------------------------------------------------------------
# Volume velocity detection
# ---------------------------------------------------------------------------

DEAD_VOLUME_LOOKBACK: int = 3        # snapshots to treat as "recent"
DEAD_VOLUME_HISTORICAL_FLOOR: int = 5
DECLINING_VOLUME_RATIO: float = 0.2  # recent < 20% of historical = declining

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------

SCORE_WEIGHT_SPREAD: float = 0.25
SCORE_WEIGHT_VOLUME: float = 0.25
SCORE_WEIGHT_FRESHNESS: float = 0.12
SCORE_WEIGHT_TREND: float = 0.10
SCORE_WEIGHT_HISTORY: float = 0.10
SCORE_WEIGHT_STABILITY: float = 0.08
SCORE_WEIGHT_ML: float = 0.10

# Minimum score to surface an opportunity (out of 100)
MIN_SUGGEST_SCORE: float = 45.0

# Stability: coefficient-of-variation breakpoints
STABILITY_CV_EXCELLENT: float = 0.005
STABILITY_CV_GOOD: float = 0.01
STABILITY_CV_FAIR: float = 0.02
STABILITY_CV_POOR: float = 0.05
STABILITY_CV_BAD: float = 0.10

# ---------------------------------------------------------------------------
# Position sizing (Kelly Criterion)
# ---------------------------------------------------------------------------

MAX_SINGLE_POSITION_PCT: float = 0.15   # Never > 15% of bankroll on one flip
MAX_ITEM_EXPOSURE_PCT: float = 0.25     # Never > 25% on one item total
MAX_TOTAL_EXPOSURE_PCT: float = 0.80    # Keep 20% as cash reserve
MIN_WIN_RATE_FOR_SIZING: float = 0.45

# Stop-loss defaults
STOP_LOSS_DEFAULT_PCT: float = 0.03    # 3% for most items
STOP_LOSS_HIGH_VALUE_PCT: float = 0.05 # 5% for high-value / volatile
STOP_LOSS_LOW_VALUE_PCT: float = 0.02  # 2% for low-value stable

# ---------------------------------------------------------------------------
# Wiki API
# ---------------------------------------------------------------------------

WIKI_BASE_URL: str = "https://prices.runescape.wiki/api/v1/osrs"
WIKI_USER_AGENT: str = "OSRS-AI-Flipper v2.0 - Discord: bakes982"

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

# How many top-by-volume items to persist to MongoDB each cycle
SNAPSHOT_TOP_N_ITEMS: int = 50

# How often to write snapshots to MongoDB (seconds)
SNAPSHOT_STORE_INTERVAL: int = 300  # 5 minutes

# How many seconds between /latest fetches
PRICE_FETCH_INTERVAL: int = 10

# How many seconds between /5m fetches
VOLUME_FETCH_INTERVAL: int = 60

# ---------------------------------------------------------------------------
# Background task intervals
# ---------------------------------------------------------------------------

FEATURE_COMPUTE_INTERVAL: int = 60        # seconds
ML_SCORE_INTERVAL: int = 60
ALERT_CHECK_INTERVAL: int = 30
POSITION_CHECK_INTERVAL: int = 30
MODEL_RETRAIN_INTERVAL: int = 6 * 3600    # 6 hours
DATA_PRUNE_INTERVAL: int = 24 * 3600      # daily
AUTO_ARCHIVE_INTERVAL: int = 6 * 3600     # 6 hours
