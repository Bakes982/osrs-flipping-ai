"""
Canonical scoring engine using the chunked math spec.

Public API:
    calculate_flip_metrics(item_data: dict) -> dict
    apply_ml_score(metrics: dict, ml_score: float) -> dict
"""

from __future__ import annotations

import math
import os
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.core.constants import GE_TAX_CAP, GE_TAX_RATE
from backend.core.utils import clamp, safe_div


def _cfg_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default


def _cfg_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_attr(snapshot: Any, *names: str, default: Any = None) -> Any:
    if isinstance(snapshot, dict):
        for n in names:
            if n in snapshot and snapshot[n] is not None:
                return snapshot[n]
        return default
    for n in names:
        v = getattr(snapshot, n, None)
        if v is not None:
            return v
    return default


def _to_dt(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    if isinstance(v, (int, float)):
        try:
            return datetime.utcfromtimestamp(v)
        except Exception:
            return None
    return None


def _snapshot_points(snapshots: Sequence[Any]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for s in snapshots:
        high = _get_attr(s, "instant_buy", "high", "sell", default=0)
        low = _get_attr(s, "instant_sell", "low", "buy", default=0)
        ts_raw = _get_attr(s, "timestamp", "ts", "time", default=None)
        ts = _to_dt(ts_raw)
        if not ts:
            continue
        try:
            high_i = int(high or 0)
            low_i = int(low or 0)
        except Exception:
            continue
        if high_i <= 0 or low_i <= 0:
            continue
        if high_i < low_i:
            high_i, low_i = low_i, high_i
        buy_vol = int(_get_attr(s, "buy_volume", "buy_vol", "highPriceVolume", default=0) or 0)
        sell_vol = int(_get_attr(s, "sell_volume", "sell_vol", "lowPriceVolume", default=0) or 0)
        points.append(
            {
                "ts": ts,
                "high": high_i,
                "low": low_i,
                "mid": (high_i + low_i) / 2.0,
                "spread": high_i - low_i,
                "vol": buy_vol + sell_vol,
            }
        )
    points.sort(key=lambda x: x["ts"])
    return points


def _window(points: Sequence[Dict[str, Any]], minutes: int) -> List[Dict[str, Any]]:
    if not points:
        return []
    cutoff = points[-1]["ts"] - timedelta(minutes=minutes)
    return [p for p in points if p["ts"] >= cutoff]


def _log_return_vol(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    rets: List[float] = []
    for i in range(1, len(values)):
        p0 = values[i - 1]
        p1 = values[i]
        if p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))
    if len(rets) < 2:
        return 0.0
    return statistics.stdev(rets) * math.sqrt(len(rets))


def _ema(values: Sequence[float], span: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    acc = float(values[0])
    for v in values[1:]:
        acc = alpha * float(v) + (1.0 - alpha) * acc
    return acc


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _tax(sell: int) -> int:
    raw = int(sell * GE_TAX_RATE)
    if _cfg_bool("GE_TAX_CAP_ENABLED", False):
        return min(raw, GE_TAX_CAP)
    return raw


def _spread_15m_ago(points: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not points:
        return None
    target = points[-1]["ts"] - timedelta(minutes=15)
    candidate = None
    for p in points:
        if p["ts"] <= target:
            candidate = p
        else:
            break
    if candidate is None:
        return None
    return float(candidate["spread"])


def _avg_update_seconds(points_1h: Sequence[Dict[str, Any]]) -> float:
    if len(points_1h) < 2:
        return 300.0
    deltas = []
    for i in range(1, len(points_1h)):
        dt = (points_1h[i]["ts"] - points_1h[i - 1]["ts"]).total_seconds()
        if dt > 0:
            deltas.append(dt)
    if not deltas:
        return 300.0
    return sum(deltas) / len(deltas)


def _normalise_gph(value: float) -> float:
    ref = _cfg_float("GPH_REF", 10_000_000.0)
    return clamp(safe_div(value, max(ref, 1.0)), 0.0, 1.0)


_PROFILE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "conservative": {
        "w_margin": 0.18, "w_roi": 0.10, "w_volume": 0.18, "w_trend": 0.08,
        "w_fill": 0.20, "w_conf": 0.20, "w_gph": 0.10,
        "p_vol": 0.35, "p_decay": 0.30, "p_risk": 0.25,
    },
    "balanced": {
        "w_margin": 0.24, "w_roi": 0.12, "w_volume": 0.16, "w_trend": 0.12,
        "w_fill": 0.16, "w_conf": 0.12, "w_gph": 0.18,
        "p_vol": 0.26, "p_decay": 0.22, "p_risk": 0.18,
    },
    "aggressive": {
        "w_margin": 0.30, "w_roi": 0.14, "w_volume": 0.12, "w_trend": 0.18,
        "w_fill": 0.10, "w_conf": 0.06, "w_gph": 0.22,
        "p_vol": 0.14, "p_decay": 0.12, "p_risk": 0.10,
    },
}


def calculate_flip_metrics(item_data: dict) -> dict:
    item_id = int(item_data.get("item_id", 0) or 0)
    item_name = str(item_data.get("item_name", f"Item {item_id}"))

    high_now = int(item_data.get("instant_buy") or item_data.get("sell") or 0)
    low_now = int(item_data.get("instant_sell") or item_data.get("buy") or 0)
    if high_now > 0 and low_now > 0 and high_now < low_now:
        high_now, low_now = low_now, high_now
    if high_now <= 0 or low_now <= 0:
        out = _empty(item_id, item_name)
        out["vetoed"] = True
        out["veto_reasons"] = ["Missing or invalid buy/sell prices"]
        return out

    spread_now = high_now - low_now
    spread_pct_now = safe_div(spread_now, max(low_now, 1))

    snapshots = item_data.get("snapshots") or []
    points = _snapshot_points(snapshots)
    if not points:
        now = datetime.utcnow()
        points = [{"ts": now, "high": high_now, "low": low_now, "mid": (high_now + low_now) / 2.0, "spread": spread_now, "vol": 0}]

    points_1h = _window(points, 60)
    points_24h = _window(points, 24 * 60)

    buy_now = low_now
    sell_now = high_now
    tax = _tax(sell_now)
    margin_after_tax = sell_now - buy_now - tax
    roi_pct = safe_div(margin_after_tax, max(buy_now, 1)) * 100.0

    mids_1h = [p["mid"] for p in points_1h]
    mids_24h = [p["mid"] for p in points_24h] or mids_1h
    volatility_1h = _log_return_vol(mids_1h)
    volatility_24h = _log_return_vol(mids_24h)

    vol_ref_mode = str(item_data.get("vol_ref_mode") or os.getenv("VOL_REF_MODE", "fixed")).lower()
    if vol_ref_mode == "daily_median":
        vol_ref = float(item_data.get("vol_ref_daily_median") or _cfg_float("VOL_REF_DAILY_MEDIAN", 0.02))
    else:
        vol_ref = float(item_data.get("vol_ref") or _cfg_float("VOL_REF_FIXED", 0.02))
    vol_norm = clamp(safe_div(volatility_1h, max(vol_ref, 1e-6)), 0.0, 1.0)

    ema_fast = _ema(mids_1h, span=6)
    ema_slow = _ema(mids_1h, span=18)
    trend_raw = safe_div((ema_fast - ema_slow), max(ema_slow, 1.0))
    trend_ref = _cfg_float("TREND_REF", 0.01)
    trend_score_01 = clamp((safe_div(trend_raw, max(trend_ref, 1e-9)) + 1.0) / 2.0, 0.0, 1.0)
    trend = "NEUTRAL"
    if trend_raw > trend_ref:
        trend = "STRONG_UP"
    elif trend_raw > 0:
        trend = "UP"
    elif trend_raw < -trend_ref:
        trend = "STRONG_DOWN"
    elif trend_raw < 0:
        trend = "DOWN"

    spread_15m = _spread_15m_ago(points)
    if spread_15m is None:
        decay_raw = 0.0
    else:
        decay_raw = safe_div(spread_now - spread_15m, max(spread_15m, 1.0))
    decay_ref = _cfg_float("DECAY_REF", 0.25)
    decay_penalty = clamp(safe_div(-decay_raw, max(decay_ref, 1e-9)), 0.0, 1.0)

    spreads_1h = [p["spread"] for p in points_1h if p["spread"] >= 0]
    spread_mean = statistics.mean(spreads_1h) if spreads_1h else 0.0
    spread_cv = 0.0
    if len(spreads_1h) >= 2 and spread_mean > 0:
        spread_cv = statistics.stdev(spreads_1h) / spread_mean
    cv_ref = _cfg_float("SPREAD_CV_REF", 0.5)
    spread_stability = clamp(1.0 - safe_div(spread_cv, max(cv_ref, 1e-9)), 0.0, 1.0)

    freq_ref = _cfg_float("LIQ_FREQ_REF_SECONDS", 60.0)
    avg_update_seconds = _avg_update_seconds(points_1h)
    freq_score = clamp(safe_div(freq_ref, max(avg_update_seconds, 1.0)), 0.0, 1.0)
    spread_pct_ref = _cfg_float("LIQ_SPREAD_PCT_REF", 0.05)
    spread_quality = clamp(1.0 - safe_div(spread_pct_now, max(spread_pct_ref, 1e-9)), 0.0, 1.0)
    liquidity_score = clamp(0.4 * freq_score + 0.3 * spread_quality + 0.3 * spread_stability, 0.0, 1.0)

    a0 = _cfg_float("FILL_A0", -0.2)
    a1 = _cfg_float("FILL_A1", 2.0)
    a2 = _cfg_float("FILL_A2", 1.2)
    a3 = _cfg_float("FILL_A3", 1.0)
    a4 = _cfg_float("FILL_A4", 1.2)
    fill_probability = clamp(
        _sigmoid(a0 + a1 * liquidity_score + a2 * spread_stability - a3 * vol_norm - a4 * decay_penalty),
        0.0,
        1.0,
    )

    t_min = _cfg_float("FILL_T_MIN", 2.0)
    t_max = _cfg_float("FILL_T_MAX", 120.0)
    est_fill_time = clamp(t_min + (t_max - t_min) * (1.0 - fill_probability), t_min, t_max)

    confidence = clamp(
        0.35 * fill_probability
        + 0.25 * spread_stability
        + 0.20 * (1.0 - vol_norm)
        + 0.20 * (1.0 - decay_penalty),
        0.0,
        1.0,
    )
    confidence_pct = round(confidence * 100.0)

    risk_score_raw = clamp(
        0.45 * vol_norm + 0.35 * decay_penalty + 0.20 * (1.0 - fill_probability),
        0.0,
        1.0,
    )
    risk_score = risk_score_raw * 10.0
    if risk_score_raw < 0.33:
        risk_level = "LOW"
    elif risk_score_raw <= 0.66:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    user_capital = int(item_data.get("user_capital") or 10_000_000)
    item_limit = int(item_data.get("item_limit") or item_data.get("buy_limit") or 100_000)
    qty_cap = int(user_capital // max(buy_now, 1))
    qty_suggested = min(max(qty_cap, 0), max(item_limit, 1))
    qty_suggested = int(math.floor(qty_suggested * (0.5 + 0.5 * liquidity_score)))
    qty_suggested = max(qty_suggested, 1 if margin_after_tax > 0 else 0)

    expected_profit = margin_after_tax * qty_suggested
    hours = max(est_fill_time, 1.0) / 60.0
    raw_gph = safe_div(expected_profit, hours)
    risk_adjusted_gph = raw_gph * confidence * (1.0 - 0.5 * risk_score_raw)

    margin_ref = float(item_data.get("margin_ref") or _cfg_float("MARGIN_REF", 50_000.0))
    roi_ref = float(item_data.get("roi_ref") or _cfg_float("ROI_REF", 5.0))
    margin_norm = clamp(safe_div(margin_after_tax, max(margin_ref, 1.0)), 0.0, 1.0)
    roi_norm = clamp(safe_div(roi_pct, max(roi_ref, 1e-9)), 0.0, 1.0)
    gph_norm = _normalise_gph(risk_adjusted_gph)

    profile = str(item_data.get("risk_profile") or "balanced").lower()
    w = _PROFILE_WEIGHTS.get(profile, _PROFILE_WEIGHTS["balanced"])
    base = (
        w["w_margin"] * margin_norm
        + w["w_roi"] * roi_norm
        + w["w_volume"] * liquidity_score
        + w["w_trend"] * trend_score_01
        + w["w_fill"] * fill_probability
        + w["w_conf"] * confidence
        + w["w_gph"] * gph_norm
    )
    penalty = w["p_vol"] * vol_norm + w["p_decay"] * decay_penalty + w["p_risk"] * risk_score_raw
    final_score_01 = clamp(base - penalty, 0.0, 1.0)
    final_score = round(final_score_01 * 100.0)

    veto_reasons: List[str] = []
    vetoed = False
    if margin_after_tax <= 0:
        final_score = min(final_score, 5)
        vetoed = True
        veto_reasons.append("Non-positive margin after tax")
    if fill_probability < 0.2 and profile in {"conservative", "balanced"}:
        final_score = min(final_score, 30)
    if spread_pct_now > float(item_data.get("spread_max_pct") or 0.20):
        vetoed = True
        veto_reasons.append("Spread too wide")

    now_ts = int(datetime.utcnow().timestamp())
    latest_ts = int(points[-1]["ts"].timestamp()) if points else now_ts
    stale_minutes = safe_div(now_ts - latest_ts, 60.0)
    stale_data = stale_minutes > _cfg_float("STALE_MINUTES", 45.0)
    anomalous_spread = spread_cv > 1.5

    # -- Dump risk (PR11) ---------------------------------------------------
    _dump_input = {
        "volatility_1h":    volatility_1h,
        "volatility_24h":   volatility_24h,
        "spread_compression": decay_penalty,
        "fill_probability": fill_probability,
    }
    dump_score, dump_signal = _compute_dump_risk(_dump_input, points)
    dump_risk_score = round(dump_score, 2)
    if dump_signal == "high":
        final_score = max(0.0, final_score - 30.0)
        confidence  = confidence * 0.6
        if "DUMP_HIGH" not in veto_reasons:
            veto_reasons.append("DUMP_HIGH")
    elif dump_signal == "watch":
        if "DUMP_WATCH" not in veto_reasons:
            veto_reasons.append("DUMP_WATCH")

    return {
        "item_id": item_id,
        "item_name": item_name,
        "spread": int(spread_now),
        "spread_pct": round(spread_pct_now * 100.0, 4),
        "recommended_buy": int(buy_now),
        "recommended_sell": int(sell_now),
        "gross_profit": int(sell_now - buy_now),
        "tax": int(tax),
        "net_profit": int(margin_after_tax),
        "roi_pct": round(roi_pct, 4),
        "gp_per_hour": round(raw_gph, 2),
        "estimated_hold_time": int(round(est_fill_time)),
        "fill_probability": round(fill_probability, 4),
        "spread_compression": round(decay_penalty, 6),
        "volatility_1h": round(volatility_1h, 8),
        "volatility_24h": round(volatility_24h, 8),
        "volume_delta": round((points_1h[-1]["vol"] - points_1h[0]["vol"]) / max(points_1h[0]["vol"], 1) if len(points_1h) > 1 else 0.0, 4),
        "ma_signal": round(trend_raw, 6),
        "trend": trend,
        "momentum": round(mids_1h[-1] - mids_1h[0], 3) if len(mids_1h) > 1 else 0.0,
        "bb_position": None,
        "vwap_1m": None,
        "vwap_5m": None,
        "vwap_30m": None,
        "vwap_2h": None,
        "win_rate": None,
        "total_flips": len(item_data.get("flip_history") or []),
        "avg_profit": None,
        "score_spread": round(margin_norm * 100.0, 2),
        "score_volume": round(liquidity_score * 100.0, 2),
        "score_freshness": round(freq_score * 100.0, 2),
        "score_trend": round(trend_score_01 * 100.0, 2),
        "score_history": 50.0,
        "score_stability": round(spread_stability * 100.0, 2),
        "score_ml": -1.0,
        "total_score": float(final_score),
        "confidence": round(confidence, 4),
        "confidence_pct": int(confidence_pct),
        "risk_score": round(risk_score, 4),
        "risk_score_raw": round(risk_score_raw, 6),
        "risk_level": risk_level,
        "stale_data": bool(stale_data),
        "anomalous_spread": bool(anomalous_spread),
        "vetoed": bool(vetoed),
        "veto_reasons": veto_reasons,
        "reason": f"{risk_level} risk | fill={fill_probability:.2f} | conf={confidence_pct}%",
        "liquidity_score": round(liquidity_score, 4),
        "trend_score": round(trend_score_01, 4),
        "decay_penalty": round(decay_penalty, 4),
        "est_fill_time_minutes": round(est_fill_time, 2),
        "qty_suggested": int(qty_suggested),
        "expected_profit": int(expected_profit),
        "risk_adjusted_gp_per_hour": round(risk_adjusted_gph, 2),
        "margin_after_tax": int(margin_after_tax),
        "final_score": float(final_score),
        # PR11 — dump risk
        "dump_risk_score": dump_risk_score,
        "dump_signal": dump_signal,
    }


def apply_ml_score(metrics: dict, ml_score: float) -> dict:
    metrics["score_ml"] = float(ml_score)
    # Keep ML as a bounded additive bias.
    bias = clamp((float(ml_score) - 50.0) * 0.1, -5.0, 5.0)
    metrics["total_score"] = round(clamp((metrics.get("total_score", 0.0) or 0.0) + bias, 0.0, 100.0), 2)
    metrics["final_score"] = metrics["total_score"]
    return metrics


def _empty(item_id: int, item_name: str) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "item_name": item_name,
        "spread": 0,
        "spread_pct": 0.0,
        "recommended_buy": 0,
        "recommended_sell": 0,
        "gross_profit": 0,
        "tax": 0,
        "net_profit": 0,
        "roi_pct": 0.0,
        "gp_per_hour": 0.0,
        "estimated_hold_time": 0,
        "fill_probability": 0.0,
        "spread_compression": 0.0,
        "volatility_1h": 0.0,
        "volatility_24h": 0.0,
        "volume_delta": 0.0,
        "ma_signal": 0.0,
        "trend": "NEUTRAL",
        "momentum": 0.0,
        "bb_position": None,
        "vwap_1m": None,
        "vwap_5m": None,
        "vwap_30m": None,
        "vwap_2h": None,
        "win_rate": None,
        "total_flips": 0,
        "avg_profit": None,
        "score_spread": 0.0,
        "score_volume": 0.0,
        "score_freshness": 0.0,
        "score_trend": 0.0,
        "score_history": 0.0,
        "score_stability": 0.0,
        "score_ml": -1.0,
        "total_score": 0.0,
        "confidence": 0.0,
        "confidence_pct": 0,
        "risk_score": 10.0,
        "risk_score_raw": 1.0,
        "risk_level": "HIGH",
        "stale_data": False,
        "anomalous_spread": False,
        "vetoed": False,
        "veto_reasons": [],
        "reason": "",
        "liquidity_score": 0.0,
        "trend_score": 0.0,
        "decay_penalty": 0.0,
        "est_fill_time_minutes": 0.0,
        "qty_suggested": 0,
        "expected_profit": 0,
        "risk_adjusted_gp_per_hour": 0.0,
        "margin_after_tax": 0,
        "final_score": 0.0,
        # PR11 — dump risk
        "dump_risk_score": 0.0,
        "dump_signal": "none",
    }


# ---------------------------------------------------------------------------
# PR11 — Dump Risk Score
# ---------------------------------------------------------------------------

def _compute_dump_risk(result: dict, snapshots: list) -> tuple[float, str]:
    """Compute a dump risk score 0..100 for the item.

    Signals used (weighted sum → clamp to 0-100):
      35% short_return   — negative % price change over ~10-minute window
      25% vol_spike      — volatility_1h / max(volatility_24h, 0.001)
      25% spread_comp    — spread compression / spread widening spike proxy
      15% fill_drop      — drop in fill_probability vs the current cycle baseline

    Returns
    -------
    (dump_risk_score, dump_signal)
        dump_risk_score : float  0–100
        dump_signal     : str    "none" | "watch" | "high"
    """
    from backend import config as _cfg
    _TINY = 1e-6

    # 1) Short-return: % change in mid price over last N snapshots
    #    A sharp negative return → high dump signal
    window = _cfg.DUMP_SHORT_RETURN_WINDOW_MIN   # minutes
    # Each snapshot is spaced ~1 min apart in the data model
    n_snaps = max(1, window)
    if len(snapshots) >= n_snaps + 1:
        def _mid(p) -> float:
            # p is either a normalized points dict (has "mid") or a snapshot object/dict
            if isinstance(p, dict):
                return p.get("mid") or (
                    ((p.get("instant_buy") or p.get("high") or 0)
                     + (p.get("instant_sell") or p.get("low") or 0)) / 2.0
                ) or _TINY
            return ((getattr(p, "instant_buy", 0) or 0)
                    + (getattr(p, "instant_sell", 0) or 0)) / 2.0 or _TINY

        old_mid = _mid(snapshots[-(n_snaps + 1)])
        new_mid = _mid(snapshots[-1])
        short_return = (new_mid - old_mid) / max(old_mid, _TINY)
        # Negative return → dump; clamp to [-1, 0]
        neg_return = max(-short_return, 0.0)   # 0 = no drop, 1 = 100% drop
        norm_neg_return = min(neg_return * 10, 1.0)  # scale so -10% → 1.0
    else:
        norm_neg_return = 0.0

    # 2) Volatility spike
    vol_1h  = result.get("volatility_1h",  0.0)
    vol_24h = result.get("volatility_24h", 0.0)
    vol_ratio = vol_1h / max(vol_24h, _TINY)
    norm_vol_spike = min((vol_ratio - 1.0) / 4.0, 1.0)   # 5× spike → 1.0
    norm_vol_spike = max(0.0, norm_vol_spike)

    # 3) Spread compression (a sudden spread compression may signal a dump
    #    as market makers pull bids; we look for abnormal spread_compression)
    spread_comp = result.get("spread_compression", 0.0)
    # Negative spread_compression means spread widening → suspicious
    norm_compression = min(max(-spread_comp, 0.0) / 0.05, 1.0)  # 5%/min → 1.0

    # 4) Fill-probability drop (compare to a small baseline of 0.5)
    #    If fill_probability is very low relative to expected, it signals dump
    fill_prob = result.get("fill_probability", 0.5)
    fill_drop = max(0.5 - fill_prob, 0.0) / 0.5   # 0 = normal, 1 = fill_prob=0
    norm_fill_drop = min(fill_drop, 1.0)

    # Weighted sum
    raw = (
        0.35 * norm_neg_return
        + 0.25 * norm_vol_spike
        + 0.25 * norm_compression
        + 0.15 * norm_fill_drop
    )
    dump_score = min(max(raw, 0.0), 1.0) * 100.0

    watch_threshold = _cfg.DUMP_WATCH_THRESHOLD
    high_threshold  = _cfg.DUMP_HIGH_THRESHOLD

    if dump_score >= high_threshold:
        signal = "high"
    elif dump_score >= watch_threshold:
        signal = "watch"
    else:
        signal = "none"

    return dump_score, signal


def _apply_dump_penalties(result: dict, dump_signal: str) -> None:
    """Add DUMP badges/reasons and penalise score when dump signal is elevated."""
    if dump_signal == "none":
        return

    veto_reasons = result.setdefault("veto_reasons", [])

    if dump_signal in ("watch", "high"):
        if "DUMP_WATCH" not in veto_reasons:
            veto_reasons.append("DUMP_WATCH")
        result["reason"] = "Dump risk elevated (short-term drop / vol spike)"

    if dump_signal == "high":
        if "DUMP_HIGH" not in veto_reasons:
            veto_reasons.append("DUMP_HIGH")
        # Penalise total_score by 30 points for high-dump items
        result["total_score"] = max(0.0, result.get("total_score", 0.0) - 30.0)
        # Penalise confidence
        result["confidence"]  = result.get("confidence", 0.0) * 0.6
