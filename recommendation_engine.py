"""Actionable recommendation layer for Alpha Radar V16.6 Dynamic Intraday Freeze.

This module is intentionally downstream from stock selection:
- selection score = which stocks are attractive;
- intraday score = what is strong now;
- entry timing = whether the current price is acceptable;
- buy priority = among acceptable entries, which offers the best estimated
  reward/risk and remaining return potential.

Nothing here mutates the underlying short/mid/long ranking score.
"""
from __future__ import annotations

import math
from typing import Iterable
from collections import Counter

import numpy as np

from strategy_config import BUY_PRIORITY_POLICY, NEXT_SESSION_POLICY, CONTINUATION_ENTRY_POLICY


def _f(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _scale(v: float | None, lo: float, hi: float, neutral: float = 50.0) -> float:
    if v is None or not math.isfinite(float(v)) or hi <= lo:
        return neutral
    return float(np.clip((float(v) - lo) / (hi - lo), 0.0, 1.0) * 100.0)


def _forecast_strategy(stock: dict, horizon: str) -> tuple[dict, dict]:
    block = (stock.get("horizons", {}) or {}).get(horizon, {}) or {}
    forecast = block.get("forecast") or {}
    return block, (forecast.get("strategy") or {})


def expected_return_profile(stock: dict, horizon: str, price: float | None = None) -> dict:
    """Translate the historical-analog forecast to a current/entry price.

    The model's median forward return is anchored to the completed-session
    reference close.  During the session we hold that target price fixed and
    recompute the remaining upside from the live price.  This prevents a stock
    that already surged intraday from appearing to retain the same upside.
    """
    block, strat = _forecast_strategy(stock, horizon)
    forecast = block.get("forecast") or {}
    ref = _f(stock.get("price"))
    current = _f(price, ref)
    median = _f(strat.get("median"))
    mean = _f(strat.get("mean"))
    p10 = _f(strat.get("p10"))
    positive = _f(strat.get("smoothed_positive_rate"), _f(strat.get("historical_positive_rate")))
    n = int(_f(forecast.get("local_effective_n"), 0) or 0)
    confidence = float(np.clip(_f(forecast.get("confidence_score"), 0.0) or 0.0, 0.0, 100.0))
    confidence_floor = float(BUY_PRIORITY_POLICY.get("expected_confidence_floor", 0.55))
    confidence_multiplier = confidence_floor + (1.0 - confidence_floor) * (confidence / 100.0)
    if ref is None or ref <= 0 or current is None or current <= 0 or median is None:
        return {
            "available": False,
            "median_return": median,
            "positive_rate": positive,
            "sample_n": n,
            "forecast_confidence": confidence,
            "confidence_multiplier": confidence_multiplier,
        }
    target = ref * (1.0 + median)
    expected = target / current - 1.0
    # Weak historical evidence may overstate upside.  Positive upside is shrunk
    # toward zero; negative expected return is never softened.
    conservative_expected = expected * confidence_multiplier if expected > 0 else expected
    p10_target = ref * (1.0 + p10) if p10 is not None else None
    p10_current = (p10_target / current - 1.0) if p10_target is not None else None
    return {
        "available": bool(forecast.get("estimate_available", True)),
        "reference_price": ref,
        "current_price": current,
        "median_return": median,
        "mean_return": mean,
        "p10_return": p10,
        "target_price": target,
        "remaining_expected_return": expected,
        "conservative_expected_return": conservative_expected,
        "remaining_p10_return": p10_current,
        "positive_rate": positive,
        "sample_n": n,
        "forecast_confidence": confidence,
        "confidence_multiplier": confidence_multiplier,
    }


def suggested_buy_zone(stock: dict, horizon: str, current_price: float | None = None) -> dict:
    """Return a structural buy zone without chasing the current print."""
    block = (stock.get("horizons", {}) or {}).get(horizon, {}) or {}
    plan = block.get("plan") or {}
    if not plan:
        return {"available": False}
    ref = _f(plan.get("reference_close"), _f(stock.get("price")))
    if ref is None or ref <= 0:
        return {"available": False}
    atr = max(_f(plan.get("atr14"), ref * 0.03) or ref * 0.03, ref * 0.008)
    zone_low = _f(plan.get("zone_low"), ref * 0.98) or ref * 0.98
    zone_high = _f(plan.get("zone_high"), ref * 1.01) or ref * 1.01
    trigger = _f(plan.get("trigger"), ref) or ref
    chase = _f(plan.get("chase_limit"), ref * 1.03) or ref * 1.03
    stop = _f(plan.get("invalidation"), ref - 2.0 * atr) or ref - 2.0 * atr
    current = _f(current_price, ref) or ref

    if horizon == "short":
        # For breakout trades, the useful buying area is around the trigger, not
        # wherever the live price happens to be after a surge.
        low = max(zone_low, trigger - 0.20 * atr)
        high = min(chase, max(low, trigger + 0.45 * atr))
        if high < low:
            low, high = min(zone_low, zone_high), max(zone_low, zone_high)
        needs_trigger = current < trigger * 0.997
    elif horizon == "mid":
        low = max(zone_low, ref - 0.55 * atr)
        high = min(chase, zone_high, ref + 0.45 * atr)
        if high < low:
            low, high = min(zone_low, zone_high), max(zone_low, zone_high)
        needs_trigger = False
    else:
        low = max(zone_low, ref - 0.80 * atr)
        high = min(chase, zone_high, ref + 0.65 * atr)
        if high < low:
            low, high = min(zone_low, zone_high), max(zone_low, zone_high)
        needs_trigger = False

    midpoint = (low + high) / 2.0
    return {
        "available": True,
        "low": round(float(low), 2),
        "high": round(float(high), 2),
        "mid": round(float(midpoint), 2),
        "trigger": round(float(trigger), 2),
        "chase_limit": round(float(chase), 2),
        "stop": round(float(stop), 2),
        "atr14": round(float(atr), 2),
        "needs_trigger": bool(needs_trigger),
        "current_in_zone": bool(low <= current <= high),
        "current_above_zone": bool(current > high),
        "current_below_zone": bool(current < low),
    }


def _components(stock: dict, horizon: str, *, live: bool) -> dict:
    block = (stock.get("horizons", {}) or {}).get(horizon, {}) or {}
    live_info = stock.get("intraday") or {}
    current = _f(live_info.get("price"), _f(stock.get("price"))) if live else _f(stock.get("price"))
    selection = _f(stock.get("live_ranking_score")) if live else _f(block.get("ranking_score"))
    if selection is None:
        selection = _f(block.get("ranking_score"), 0.0) or 0.0
    timing = (live_info.get("entry_timing") or {}) if live else (block.get("entry_timing") or {})
    position = (live_info.get("position_guidance") or {}) if live else (block.get("position_guidance") or {})
    entry = _f(timing.get("score"), 0.0) or 0.0
    profile = expected_return_profile(stock, horizon, current)
    zone = suggested_buy_zone(stock, horizon, current)
    raw_expected = _f(profile.get("remaining_expected_return"))
    expected = _f(profile.get("conservative_expected_return"), raw_expected)
    positive = _f(profile.get("positive_rate"))
    confidence = _f(profile.get("forecast_confidence"), 0.0) or 0.0
    sample_n = int(_f(profile.get("sample_n"), 0) or 0)
    stop = _f(zone.get("stop"))
    structural_downside = None
    tail_downside = max(0.0, -(_f(profile.get("remaining_p10_return"), 0.0) or 0.0))
    downside = None
    rr = None
    if current is not None and current > 0 and stop is not None:
        structural_downside = max(0.0, (current - stop) / current)
        downside = max(
            structural_downside,
            tail_downside,
            float(BUY_PRIORITY_POLICY.get("tail_risk_floor", 0.002)),
        )
        if expected is not None:
            rr = max(0.0, expected) / downside
    return {
        "block": block,
        "live_info": live_info,
        "price": current,
        "selection": selection,
        "timing": timing,
        "position": position,
        "entry": entry,
        "profile": profile,
        "zone": zone,
        "raw_expected": raw_expected,
        "expected": expected,
        "positive": positive,
        "confidence": confidence,
        "sample_n": sample_n,
        "structural_downside": structural_downside,
        "tail_downside": tail_downside,
        "downside": downside,
        "reward_risk": rr,
    }


def _gate_failure(code: str, label: str, actual=None, threshold=None) -> dict:
    return {"code": code, "label": label, "actual": actual, "threshold": threshold}


def _continuation_status(c: dict) -> tuple[bool, list[dict]]:
    """Strict live continuation gate for prices already above the pullback zone."""
    cfg = CONTINUATION_ENTRY_POLICY
    live = c.get("live_info") or {}
    timing = c.get("timing") or {}
    failures: list[dict] = []
    checks = [
        ("continuation_selection", c.get("selection"), cfg["min_selection"], lambda a, b: a >= b, "標的分數不足"),
        ("continuation_intraday", _f(live.get("intraday_score")), cfg["min_intraday_score"], lambda a, b: a >= b, "盤中強度不足"),
        ("continuation_entry", c.get("entry"), cfg["min_entry"], lambda a, b: a >= b, "續攻進場分數不足"),
        ("continuation_relative", _f(live.get("relative_market_pct_pt")), cfg["min_relative_market_pct_pt"], lambda a, b: a >= b, "相對大盤強度不足"),
        ("continuation_day_position", _f(live.get("day_position")), cfg["min_day_position"], lambda a, b: a >= b, "價格未守在日內強勢區"),
        ("continuation_change", _f(live.get("change_rate_pct")), cfg["max_change_pct"], lambda a, b: a <= b, "單日漲幅已偏高"),
        ("continuation_extension", _f(timing.get("extension_atr")), cfg["max_extension_atr"], lambda a, b: a <= b, "距突破區過遠"),
    ]
    for code, actual, threshold, fn, label in checks:
        if actual is None or not fn(float(actual), float(threshold)):
            failures.append(_gate_failure(code, label, actual, threshold))
    ext = _f(timing.get("extension_atr"))
    if ext is not None and ext < float(cfg.get("min_extension_atr", 0.0)):
        failures.append(_gate_failure("continuation_extension_low", "尚未形成有效續攻", ext, cfg.get("min_extension_atr")))
    vr = _f(live.get("volume_ratio"))
    if vr is None or not (float(cfg["volume_ratio_low"]) <= vr <= float(cfg["volume_ratio_high"])):
        failures.append(_gate_failure("continuation_volume", "續攻量能不在健康區間", vr, f"{cfg['volume_ratio_low']}–{cfg['volume_ratio_high']}"))
    vwap = _f(live.get("vwap_gap_pct"))
    # MIS may not expose VWAP; missing VWAP is neutral rather than an automatic fail.
    if vwap is not None and not (float(cfg["min_vwap_gap_pct"]) <= vwap <= float(cfg["max_vwap_gap_pct"])):
        failures.append(_gate_failure("continuation_vwap", "價格偏離盤中均價過多", vwap, f"{cfg['min_vwap_gap_pct']}–{cfg['max_vwap_gap_pct']}"))
    spread = _f(live.get("spread_pct"))
    if spread is not None and spread > float(cfg["max_spread_pct"]):
        failures.append(_gate_failure("continuation_spread", "買賣價差過大", spread, cfg["max_spread_pct"]))
    expected = c.get("expected")
    if expected is None or expected < float(cfg["min_expected_return"]):
        failures.append(_gate_failure("continuation_expected", "續攻後剩餘報酬不足", expected, cfg["min_expected_return"]))
    rr = c.get("reward_risk")
    if rr is None or rr < float(cfg["min_reward_risk"]):
        failures.append(_gate_failure("continuation_reward_risk", "續攻報酬/風險不足", rr, cfg["min_reward_risk"]))
    positive = c.get("positive")
    if positive is not None and positive < float(cfg.get("min_positive_rate", 0.52)):
        failures.append(_gate_failure("continuation_positive_rate", "續攻歷史偏正向比例不足", positive, cfg.get("min_positive_rate", 0.52)))
    return len(failures) == 0, failures


def evaluate_buy_opportunity(stock: dict, horizon: str = "short", *, live: bool = True) -> dict:
    """Score an actionable buy after selection and timing have been evaluated.

    V16.6 supports two explicit execution paths:
    * PULLBACK: price is in the structural buy zone;
    * CONTINUATION: price is above that zone but the live tape is strong, not
      extended, and still offers sufficient conservative reward/risk.
    """
    c = _components(stock, horizon, live=live)
    policy = BUY_PRIORITY_POLICY
    er_lo, er_hi = policy["expected_return_range"]
    pr_lo, pr_hi = policy["positive_rate_range"]
    rr_lo, rr_hi = policy["reward_risk_range"]
    subs = {
        "selection": float(np.clip(c["selection"], 0, 100)),
        "entry": float(np.clip(c["entry"], 0, 100)),
        "expected_return": _scale(c["expected"], er_lo, er_hi),
        "positive_rate": _scale(c["positive"], pr_lo, pr_hi),
        "reward_risk": _scale(c["reward_risk"], rr_lo, rr_hi),
    }
    weights = policy["weights"]
    score = sum(subs[k] * float(weights[k]) for k in weights)
    timing = c["timing"]
    action_text = str(timing.get("action") or "")
    chase = str(timing.get("chase_risk") or "待確認")
    position_pct = int(_f(c["position"].get("percent"), 0) or 0)
    zone = c["zone"]

    failures: list[dict] = []
    if "不追" in action_text or "暫不" in action_text:
        failures.append(_gate_failure("hard_no", action_text or "目前不宜追價"))
    if chase in {"高", "極高"}:
        failures.append(_gate_failure("chase_risk", f"追價風險{chase}"))
    if position_pct <= 0:
        failures.append(_gate_failure("position", "目前風險預算為 0%", position_pct, ">0"))
    if c["entry"] < 58.0:
        failures.append(_gate_failure("entry_floor", "進場結構尚未成熟", c["entry"], 58.0))

    selection_ok = c["selection"] >= float(policy["min_selection"])
    entry_ok = c["entry"] >= float(policy["min_entry"])
    expected_ok = c["expected"] is not None and c["expected"] >= float(policy["min_expected_return"])
    positive_ok = c["positive"] is None or c["positive"] >= float(policy["min_positive_rate"])
    rr_ok = c["reward_risk"] is not None and c["reward_risk"] >= float(policy["min_reward_risk"])
    evidence_ok = (
        c["confidence"] >= float(policy.get("min_forecast_confidence", 0.0))
        and c["sample_n"] >= int(policy.get("min_sample_n", 0))
    )
    if not selection_ok:
        failures.append(_gate_failure("selection", "標的分數未達門檻", c["selection"], policy["min_selection"]))
    if not entry_ok:
        failures.append(_gate_failure("entry", "進場分數未達門檻", c["entry"], policy["min_entry"]))
    if not expected_ok:
        failures.append(_gate_failure("expected_return", "保守剩餘報酬不足", c["expected"], policy["min_expected_return"]))
    if not positive_ok:
        failures.append(_gate_failure("positive_rate", "歷史偏正向比例不足", c["positive"], policy["min_positive_rate"]))
    if not rr_ok:
        failures.append(_gate_failure("reward_risk", "報酬/風險不足", c["reward_risk"], policy["min_reward_risk"]))
    if not evidence_ok:
        failures.append(_gate_failure("evidence", "歷史案例信心不足或樣本不足", f"{c['confidence']:.0f}/100, n={c['sample_n']}", f">={policy.get('min_forecast_confidence',0):.0f}, n>={int(policy.get('min_sample_n',0))}"))

    hard_no = any(x["code"] in {"hard_no", "chase_risk", "position", "entry_floor"} for x in failures)
    base_ok = selection_ok and entry_ok and expected_ok and positive_ok and rr_ok and evidence_ok and not hard_no
    in_zone = bool(zone.get("current_in_zone")) if zone.get("available") else False
    above_zone = bool(zone.get("current_above_zone")) if zone.get("available") else False
    continuation_ok = False
    continuation_failures: list[dict] = []
    if live and above_zone and not hard_no and evidence_ok:
        continuation_ok, continuation_failures = _continuation_status(c)

    entry_path = "NONE"
    if hard_no:
        state = "WATCH"
        action = failures[0]["label"] if failures else "目前不宜進場"
    elif evidence_ok and in_zone and base_ok and c["entry"] >= float(policy["strong_entry"]):
        state = "BUY"
        entry_path = "PULLBACK"
        action = "拉回買點佳，可分批進場"
    elif evidence_ok and in_zone and base_ok:
        state = "SMALL"
        entry_path = "PULLBACK"
        action = "拉回承接，可小量試單"
    elif continuation_ok:
        state = "CONTINUATION"
        entry_path = "CONTINUATION"
        action = "續攻確認，可小量試單"
        position_pct = min(position_pct, int(CONTINUATION_ENTRY_POLICY.get("max_position_percent", 35)))
    elif above_zone:
        state = "WAIT_PULLBACK"
        entry_path = "WAIT_PULLBACK"
        # Continuation-specific failures explain why a price above the original
        # zone is not automatically rejected with a generic pullback message.
        failures.extend(x for x in continuation_failures if x["code"] not in {f["code"] for f in failures})
        action = "高於原買入區，續攻條件未完整；等拉回或再確認"
    elif zone.get("needs_trigger"):
        state = "WAIT_BREAKOUT"
        entry_path = "WAIT_BREAKOUT"
        failures.append(_gate_failure("price_trigger", "尚未突破確認價", c["price"], zone.get("trigger")))
        action = "先等突破確認，不提前追價"
    else:
        state = "WATCH"
        action = failures[0]["label"] if failures else "先觀察，尚未形成高報酬風險比買點"

    # Remove duplicate failure codes while preserving priority/order.
    dedup: list[dict] = []
    seen: set[str] = set()
    for f in failures:
        code = str(f.get("code") or "")
        if not code or code in seen:
            continue
        seen.add(code)
        dedup.append(f)
    failures = dedup

    if hard_no:
        score = min(score, 49.0)
    elif state == "WATCH":
        score = min(score, 64.0)
    elif state in {"WAIT_PULLBACK", "WAIT_BREAKOUT"}:
        score = min(score, 69.0)
    elif state == "CONTINUATION":
        score = min(score, 82.0)  # continuation is deliberately smaller-size

    eligible = state in {"BUY", "SMALL", "CONTINUATION"}
    return {
        "score": round(float(np.clip(score, 0, 100)), 1),
        "state": state,
        "entry_path": entry_path,
        "action": action,
        "selection_score": round(float(c["selection"]), 1),
        "entry_score": round(float(c["entry"]), 1),
        "intraday_score": _f((c.get("live_info") or {}).get("intraday_score")),
        "price": c["price"],
        "expected_return": c["expected"],
        "raw_expected_return": c["raw_expected"],
        "forecast_confidence": round(float(c["confidence"]), 1),
        "positive_rate": c["positive"],
        "reward_risk": c["reward_risk"],
        "risk_downside": c["downside"],
        "structural_downside": c["structural_downside"],
        "tail_downside": c["tail_downside"],
        "downside_to_stop": c["structural_downside"],
        "target_price": c["profile"].get("target_price"),
        "sample_n": c["sample_n"],
        "buy_zone": c["zone"],
        "chase_risk": chase,
        "position_percent": position_pct,
        "timing_action": action_text,
        "eligible_now": eligible,
        "failed_gates": failures,
        "failed_gate_codes": [x["code"] for x in failures],
        "primary_blocker": failures[0]["label"] if failures and not eligible else "",
        "components": {k: round(v, 1) for k, v in subs.items()},
    }


def rank_intraday_buys(stocks: Iterable[dict], horizon: str = "short", n: int = 5) -> list[dict]:
    rows = []
    for stock in stocks or []:
        if not isinstance(stock, dict):
            continue
        item = dict(stock)
        item["buy_opportunity"] = evaluate_buy_opportunity(item, horizon, live=True)
        rows.append(item)
    order = {"BUY": 0, "CONTINUATION": 1, "SMALL": 2, "WAIT_PULLBACK": 3, "WAIT_BREAKOUT": 4, "WATCH": 5}
    rows.sort(
        key=lambda x: (
            order.get((x.get("buy_opportunity") or {}).get("state"), 9),
            -(float((x.get("buy_opportunity") or {}).get("score") or 0.0)),
        )
    )
    return rows[: max(1, int(n))]


def summarize_buy_gate_failures(stocks: Iterable[dict], top_n: int = 6) -> dict:
    """Aggregate why the current detail pool did or did not produce buys."""
    total = 0
    eligible = 0
    all_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    states: Counter[str] = Counter()
    for stock in stocks or []:
        if not isinstance(stock, dict):
            continue
        decision = stock.get("buy_opportunity") or {}
        if not decision:
            continue
        total += 1
        states[str(decision.get("state") or "UNKNOWN")] += 1
        if decision.get("eligible_now"):
            eligible += 1
            continue
        failures = decision.get("failed_gates") or []
        for f in failures:
            code = str(f.get("code") or "other")
            labels[code] = str(f.get("label") or code)
            all_counts[code] += 1
        if failures:
            code = str(failures[0].get("code") or "other")
            labels[code] = str(failures[0].get("label") or code)
            primary_counts[code] += 1
    top = []
    for code, count in primary_counts.most_common(max(1, int(top_n))):
        top.append({"code": code, "label": labels.get(code, code), "count": int(count)})
    return {
        "total": total,
        "eligible": eligible,
        "ineligible": max(0, total - eligible),
        "primary": top,
        "all_counts": {k: int(v) for k, v in all_counts.items()},
        "states": {k: int(v) for k, v in states.items()},
    }

def evaluate_next_session(stock: dict, horizon: str = "short") -> dict:
    c = _components(stock, horizon, live=False)
    policy = NEXT_SESSION_POLICY
    er = _scale(c["expected"], *BUY_PRIORITY_POLICY["expected_return_range"])
    pr = _scale(c["positive"], *BUY_PRIORITY_POLICY["positive_rate_range"])
    rr = _scale(c["reward_risk"], *BUY_PRIORITY_POLICY["reward_risk_range"])
    parts = {
        "selection": float(np.clip(c["selection"], 0, 100)),
        "entry": float(np.clip(c["entry"], 0, 100)),
        "expected_return": er,
        "positive_rate": pr,
        "reward_risk": rr,
    }
    score = sum(parts[k] * float(policy["weights"][k]) for k in policy["weights"])
    timing_action = str(c["timing"].get("action") or "")
    zone = c["zone"]
    evidence_ok = c["confidence"] >= float(BUY_PRIORITY_POLICY.get("min_forecast_confidence", 0.0)) and c["sample_n"] >= int(BUY_PRIORITY_POLICY.get("min_sample_n", 0))
    if not evidence_ok:
        stance = "歷史案例信心不足，列觀察不追"
    elif "不追" in timing_action or c["entry"] < 42:
        stance = "強勢觀察，開盤不追"
    elif zone.get("needs_trigger"):
        stance = "等突破確認後再進場"
    elif c["entry"] >= 70 and (c["expected"] is None or c["expected"] > 0):
        stance = "列入下一交易日優先觀察"
    else:
        stance = "等待回到較佳買入區"
    return {
        "score": round(float(np.clip(score, 0, 100)), 1),
        "stance": stance,
        "selection_score": round(float(c["selection"]), 1),
        "entry_score": round(float(c["entry"]), 1),
        "expected_return": c["expected"],
        "raw_expected_return": c["raw_expected"],
        "forecast_confidence": round(float(c["confidence"]), 1),
        "sample_n": c["sample_n"],
        "positive_rate": c["positive"],
        "reward_risk": c["reward_risk"],
        "target_price": c["profile"].get("target_price"),
        "buy_zone": zone,
        "timing_action": timing_action,
        "components": {k: round(v, 1) for k, v in parts.items()},
    }


def rank_next_session(snap: dict | None, horizon: str = "short", n: int = 5) -> list[dict]:
    if not snap:
        return []
    rows = []
    for stock in snap.get("stocks", []) or []:
        if not isinstance(stock, dict):
            continue
        item = dict(stock)
        item["next_session"] = evaluate_next_session(item, horizon)
        rows.append(item)
    rows.sort(key=lambda x: float((x.get("next_session") or {}).get("score") or 0.0), reverse=True)
    return rows[: max(1, int(n))]


def evaluate_next_session_live(stock: dict, horizon: str = "short") -> dict:
    """Next-session planning from a post-close quote overlay when official EOD is pending."""
    c = _components(stock, horizon, live=True)
    policy = NEXT_SESSION_POLICY
    er = _scale(c["expected"], *BUY_PRIORITY_POLICY["expected_return_range"])
    pr = _scale(c["positive"], *BUY_PRIORITY_POLICY["positive_rate_range"])
    rr = _scale(c["reward_risk"], *BUY_PRIORITY_POLICY["reward_risk_range"])
    parts = {
        "selection": float(np.clip(c["selection"], 0, 100)),
        "entry": float(np.clip(c["entry"], 0, 100)),
        "expected_return": er,
        "positive_rate": pr,
        "reward_risk": rr,
    }
    score = sum(parts[k] * float(policy["weights"][k]) for k in policy["weights"])
    timing_action = str(c["timing"].get("action") or "")
    zone = c["zone"]
    evidence_ok = c["confidence"] >= float(BUY_PRIORITY_POLICY.get("min_forecast_confidence", 0.0)) and c["sample_n"] >= int(BUY_PRIORITY_POLICY.get("min_sample_n", 0))
    if not evidence_ok:
        stance = "歷史案例信心不足，明日列觀察不追"
    elif "不追" in timing_action or c["entry"] < 42:
        stance = "收盤強勢但隔日不追，等拉回"
    elif zone.get("needs_trigger"):
        stance = "明日等突破確認後再進場"
    elif c["entry"] >= 70 and (c["expected"] is None or c["expected"] > 0):
        stance = "列入明日盤前優先觀察"
    else:
        stance = "明日等待回到較佳買入區"
    return {
        "score": round(float(np.clip(score, 0, 100)), 1),
        "stance": stance,
        "selection_score": round(float(c["selection"]), 1),
        "entry_score": round(float(c["entry"]), 1),
        "expected_return": c["expected"],
        "raw_expected_return": c["raw_expected"],
        "forecast_confidence": round(float(c["confidence"]), 1),
        "sample_n": c["sample_n"],
        "positive_rate": c["positive"],
        "reward_risk": c["reward_risk"],
        "target_price": c["profile"].get("target_price"),
        "buy_zone": zone,
        "timing_action": timing_action,
        "components": {k: round(v, 1) for k, v in parts.items()},
        "provisional": True,
    }


def rank_next_session_live(stocks: Iterable[dict], horizon: str = "short", n: int = 5) -> list[dict]:
    rows = []
    for stock in stocks or []:
        if not isinstance(stock, dict):
            continue
        item = dict(stock)
        item["next_session"] = evaluate_next_session_live(item, horizon)
        rows.append(item)
    rows.sort(key=lambda x: float((x.get("next_session") or {}).get("score") or 0.0), reverse=True)
    return rows[: max(1, int(n))]
