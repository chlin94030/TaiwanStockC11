"""Frozen core strategy configuration for Alpha Radar V16.6 Dynamic Intraday Freeze.

Architecture contract
---------------------
1. Selection score ranks *which stock* is attractive for a horizon.
2. Intraday score measures *current session strength* only.
3. Entry timing decides *whether the current price is attractive enough to buy*.
4. Market regime controls *position size*, not stock quality/ranking.

Keep changes to strategy thresholds in this file.  The lower-level feature engine
(return_first_model.py) is intentionally frozen through V16.6 to avoid accidental
model drift while live validation is running.
"""
from __future__ import annotations

ARCHITECTURE_VERSION = "v16.6.0-dynamic-intraday"
HORIZONS = ("short", "mid", "long")

# Daily stock-selection layer.  These are the high-level evidence weights.
SELECTION_WEIGHTS = {
    "short": {"technical": 0.52, "empirical": 0.20, "fundamental": 0.08, "flow": 0.20},
    "mid": {"technical": 0.42, "empirical": 0.22, "fundamental": 0.22, "flow": 0.14},
    "long": {"technical": 0.34, "empirical": 0.22, "fundamental": 0.34, "flow": 0.10},
}

# Historical-analog evidence is useful only in proportion to its own quality.
# The raw empirical score is therefore shrunk toward neutral (50) when the
# analog evidence confidence is weak. This prevents a small/noisy analog set
# from dominating the daily selector without deleting the evidence entirely.
EMPIRICAL_RELIABILITY_POLICY = {
    "neutral_score": 50.0,
    "confidence_floor_multiplier": 0.45,
    "minimum_confidence_for_action": 35.0,
    "minimum_effective_samples_for_action": 26,
}

# Intraday-strength layer.  No chase/overheat penalty belongs here; price
# extension is an execution issue handled by ENTRY_POLICY below.
INTRADAY_FACTOR_WEIGHTS = {
    "relative_market": 0.30,
    "vwap": 0.20,
    "volume": 0.18,
    "day_position": 0.12,
    "open_move": 0.08,
    "bid_pressure": 0.05,
    "turnover": 0.07,
}
INTRADAY_FACTOR_RANGES = {
    "relative_market": (-2.0, 4.0),
    "vwap": (-1.5, 2.0),
    "volume": (0.55, 2.50),
    "day_position": (0.18, 0.88),
    "open_move": (-2.5, 4.0),
    "bid_pressure": (0.35, 0.65),
    "turnover_twd": (30_000_000.0, 2_500_000_000.0),
}
INTRADAY_SPREAD_FREE_PCT = 0.60
INTRADAY_SPREAD_PENALTY_PER_PCT = 6.0

# Intraday signal should not dominate immediately after the open.  The weights
# mature as more of the session is observed.  Values are maximum blend weights
# with the complete daily model.  Key = minutes elapsed since 09:00.
INTRADAY_BLEND_SCHEDULE = {
    "short": ((0, 0.22), (20, 0.28), (60, 0.35), (150, 0.41), (225, 0.45)),
    "mid": ((0, 0.06), (20, 0.08), (60, 0.10), (150, 0.13), (225, 0.15)),
    "long": ((0, 0.02), (20, 0.025), (60, 0.03), (150, 0.04), (225, 0.05)),
}

# Entry/execution layer.  These rules NEVER change the selection score.
ENTRY_POLICY = {
    "base_score": 78.0,
    "extension": {
        "below_breakout": -0.60,
        "ideal_max": 0.45,
        "acceptable_max": 0.90,
        "warm_max": 1.50,
        "hot_max": 2.00,
    },
    "ma5_extension": {"warm": 1.20, "hot": 2.00},
    "one_day_move": {"watch": 3.0, "hot": 6.0, "very_hot": 8.0, "near_limit": 9.5},
    "market_surge": 2.0,
    "relative_strength_bonus": 2.0,
    "relative_weakness": -1.0,
    "volume": {"healthy_low": 1.10, "healthy_high": 2.50, "extreme": 3.50},
    "day_position": {"weak": 0.28, "strong": 0.72},
    "action": {"good": 82.0, "small": 70.0, "wait": 58.0, "watch": 42.0},
    "risk": {"medium": 15.0, "high": 34.0, "extreme": 55.0},
}

# Regime is a risk-budget layer only.  1.0 means 100% of the user's normal
# planned position, not 100% of portfolio capital.
REGIME_POSITION_MULTIPLIER = {
    "BULL": 1.00,
    "NEUTRAL": 0.70,
    "BEAR": 0.35,
    "UNKNOWN": 0.50,
}
ENTRY_POSITION_MULTIPLIER = {
    "good": 1.00,      # entry score >= 82
    "small": 0.55,     # >= 70
    "wait": 0.25,      # >= 58
    "avoid": 0.00,     # < 58
}
CHASE_POSITION_CAP = {
    "低": 1.00,
    "中": 0.55,
    "高": 0.25,
    "極高": 0.00,
    "待確認": 0.25,
}

# Data/scan defaults remain complete.  Faster operation comes from caching,
# batching and feature reuse, not shrinking the candidate universe.
SCAN_DEFAULTS = {
    "reference_size": 600,
    "candidate_size": 1000,
    "research_pool_per_horizon": 10,
    "history_period": "5y",
    "min_price": 10.0,
    "min_avg_turnover": 10_000_000.0,
}

TRADE_PLAN = {
    "short": {"trigger_mult": 1.005, "zone_low_mult": 0.98, "zone_high_mult": 1.01, "chase_mult": 1.03, "invalidation_atr": 2.0, "use_low20_floor": True, "entry_mode": "breakout_confirmed"},
    "mid": {"trigger_mult": 1.01, "zone_low_mult": 0.96, "zone_high_mult": 1.015, "chase_mult": 1.04, "invalidation_atr": 2.5, "use_low20_floor": False, "entry_mode": "zone_confirmed"},
    "long": {"trigger_mult": 1.02, "zone_low_mult": 0.94, "zone_high_mult": 1.02, "chase_mult": 1.05, "invalidation_atr": 3.0, "use_low20_floor": False, "entry_mode": "zone_confirmed"},
}

INTRADAY_STATE = {
    "relative_weak": -1.5,
    "vwap_weak": -1.2,
    "day_position_weak": 0.35,
    "strong_score": 70.0,
    "strong_relative": 0.8,
    "strong_volume": 1.05,
    "stable_score": 60.0,
    "stable_vwap": -0.2,
    "volume_watch": 1.20,
    "volume_watch_score": 50.0,
    "reason_relative": 1.0,
    "reason_volume": 1.25,
    "reason_vwap_up": 0.4,
    "reason_vwap_down": -0.5,
}

# V16.6 decision layer.  This does not feed back into the frozen selection score;
# it only ranks *actionable* opportunities after a stock has already passed the
# selection layer.
BUY_PRIORITY_POLICY = {
    "weights": {
        "selection": 0.30,
        "entry": 0.30,
        "expected_return": 0.20,
        "positive_rate": 0.10,
        "reward_risk": 0.10,
    },
    "expected_return_range": (-0.02, 0.20),
    "positive_rate_range": (0.45, 0.75),
    "reward_risk_range": (0.50, 3.00),
    "min_selection": 62.0,
    "min_entry": 70.0,
    "strong_entry": 82.0,
    "min_expected_return": 0.02,
    "min_positive_rate": 0.52,
    "min_reward_risk": 1.10,
    "min_forecast_confidence": 35.0,
    "min_sample_n": 26,
    # Positive expected return is discounted when historical analog confidence
    # is weak; negative estimates are never softened.
    "expected_confidence_floor": 0.55,
    # Reward/risk uses the worse of structural stop distance and historical
    # p10 downside, avoiding an unrealistically attractive ratio.
    "tail_risk_floor": 0.002,
}

NEXT_SESSION_POLICY = {
    "weights": {
        "selection": 0.55,
        "entry": 0.18,
        "expected_return": 0.15,
        "positive_rate": 0.07,
        "reward_risk": 0.05,
    },
    "strong_watch": 75.0,
    "normal_watch": 65.0,
}

# Dynamic intraday scan. The completed-day model remains frozen during the
# session. Every minute we fetch quotes for the liquid universe, use a cheap
# discovery pass to find new leaders, then run the heavier execution layer only
# on a bounded detail pool.
LIVE_SCAN = {
    "per_horizon": 80,          # completed-day core from each horizon
    "daily_core_max": 220,      # union cap after de-duplication
    "full_market_quote_max": 1800,
    "discovery_top": 140,       # strongest live movers from the liquid universe
    "detail_max": 300,          # max names entering entry/forecast execution pass
    "refresh_seconds": 60,
    "cache_ttl_seconds": 52,
    "quote_batch_size": 45,
    "gate_summary_top": 6,
}

# A stock that has moved above the prior-session pullback zone may still be an
# executable continuation entry, but only when the live tape confirms strength
# and the stock is not extended. This path is intentionally stricter than a
# normal pullback entry so "continuation" never becomes a synonym for chasing.
CONTINUATION_ENTRY_POLICY = {
    "min_selection": 65.0,
    "min_intraday_score": 68.0,
    "min_entry": 68.0,
    "min_relative_market_pct_pt": 0.80,
    "volume_ratio_low": 1.00,
    "volume_ratio_high": 2.80,
    "min_day_position": 0.55,
    "max_change_pct": 6.0,
    "max_extension_atr": 1.20,
    "min_extension_atr": 0.00,
    "min_vwap_gap_pct": -0.20,
    "max_vwap_gap_pct": 2.50,
    "max_spread_pct": 0.80,
    "min_expected_return": 0.025,
    "min_reward_risk": 1.20,
    "min_positive_rate": 0.52,
    "max_position_percent": 35,
}


# Reproducible 2026 walk-forward validation defaults. These values do not alter
# production ranking; they define the external validation protocol.
VALIDATION_POLICY = {
    "year": 2026,
    "stocks": 100,
    "points_per_horizon": 12,
    "top_n": 5,
    "history_period": "8y",
    "seed": 1337,
    "calibration_fraction": 0.67,
    "technical_share_grid": (0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
    "min_history_rows": 380,
}
