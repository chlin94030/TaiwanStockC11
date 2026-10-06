"""Offline regression tests for Alpha Radar V16.6 Dynamic Intraday Freeze.

Run with: python test_architecture.py
No network calls are made.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

import intraday_engine as ie
import radar_service as rs
import recommendation_engine as re
from trading_calendar import market_phase, session_reference_policy, PREOPEN, INTRADAY, POSTCLOSE, OFFDAY
from policy_engine import evaluate_entry_timing, generate_trade_plan, position_guidance
from return_first_model import estimate_all_horizons, estimate_horizon_return
from strategy_config import (
    ARCHITECTURE_VERSION,
    INTRADAY_BLEND_SCHEDULE,
    SELECTION_WEIGHTS,
    VALIDATION_POLICY,
    LIVE_SCAN,
)

TZ = dt.timezone(dt.timedelta(hours=8))


def synthetic_ohlcv(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    drift = 0.00045
    ret = rng.normal(drift, 0.018, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = close * (1.0 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.018, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.018, n))
    vol = rng.integers(500_000, 8_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)


def assert_between(x, lo, hi, label):
    assert x is not None and lo <= float(x) <= hi, f"{label}: {x} not in [{lo}, {hi}]"


def test_config_weights():
    assert int(LIVE_SCAN["refresh_seconds"]) == 60
    assert int(LIVE_SCAN["detail_max"]) >= int(LIVE_SCAN["daily_core_max"])
    for h, weights in SELECTION_WEIGHTS.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-9, (h, weights)
    for h, schedule in INTRADAY_BLEND_SCHEDULE.items():
        vals = [x[1] for x in schedule]
        assert vals == sorted(vals), (h, vals)
        assert 0 <= vals[0] <= vals[-1] <= 0.5


def test_entry_separation():
    plan = {
        "reference_close": 100.0,
        "breakout_level": 100.0,
        "atr14": 3.0,
        "ma5": 100.0,
        "invalidation": 94.0,
    }
    normal = evaluate_entry_timing(
        101.0, plan, "short", change_pct=1.2, market_change_pct=0.4,
        volume_ratio=1.4, relative_market_pct_pt=0.8, day_position=0.72,
    )
    hot = evaluate_entry_timing(
        109.8, plan, "short", change_pct=9.8, market_change_pct=2.4,
        volume_ratio=2.0, relative_market_pct_pt=7.4, day_position=0.98,
    )
    assert normal["score"] > hot["score"], (normal, hot)
    assert hot["score"] <= 28.0, hot
    assert "不追" in hot["action"], hot
    assert hot["overnight_risk"] is True


def test_regime_only_changes_position():
    timing = {"score": 86.0, "chase_risk": "低", "action": "買點佳，可分批"}
    bull = position_guidance("BULL", "short", timing)
    neutral = position_guidance("NEUTRAL", "short", timing)
    bear = position_guidance("BEAR", "short", timing)
    assert bull["percent"] > neutral["percent"] > bear["percent"] > 0, (bull, neutral, bear)
    hot = position_guidance("BULL", "short", {"score": 25.0, "chase_risk": "極高", "action": "漲停/近漲停：強勢但不追"})
    assert hot["percent"] == 0, hot


def test_time_matured_intraday_weight():
    w_open = ie.intraday_blend_weight("short", dt.datetime(2026, 10, 5, 9, 5, tzinfo=TZ))
    w_mid = ie.intraday_blend_weight("short", dt.datetime(2026, 10, 5, 11, 0, tzinfo=TZ))
    w_late = ie.intraday_blend_weight("short", dt.datetime(2026, 10, 5, 12, 55, tzinfo=TZ))
    assert w_open < w_mid < w_late <= 0.45, (w_open, w_mid, w_late)


def _live_row(stock_id, close, open_, high, low, avg, change, volume_ratio, amount=500_000_000):
    return {
        "stock_id": stock_id,
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "average_price": avg,
        "change_rate": change,
        "volume_ratio": volume_ratio,
        "total_volume": 2_000_000,
        "total_amount": amount,
        "buy_price": close - 0.1,
        "sell_price": close + 0.1,
        "buy_volume": 600,
        "sell_volume": 400,
        "date": "2026-10-05",
    }


def test_limit_up_can_rank_high_but_not_be_buyable():
    plan_a = {"reference_close": 100.0, "breakout_level": 100.0, "atr14": 3.0, "ma5": 100.0, "invalidation": 94.0}
    plan_b = {"reference_close": 100.0, "breakout_level": 100.0, "atr14": 3.0, "ma5": 100.0, "invalidation": 94.0}
    base_stocks = [
        {"ticker": "1111.TW", "name": "A", "avg_volume_20d": 2_000_000, "horizons": {"short": {"ranking_score": 92.0, "plan": plan_a}}},
        {"ticker": "2222.TW", "name": "B", "avg_volume_20d": 2_000_000, "horizons": {"short": {"ranking_score": 84.0, "plan": plan_b}}},
    ]
    rows = [
        _live_row("001", 22000, 21600, 22100, 21500, 21800, 2.4, 1.4, 30_000_000_000),
        _live_row("1111", 109.8, 103.0, 109.8, 102.0, 106.0, 9.8, 2.1),
        _live_row("2222", 103.2, 101.0, 104.0, 100.5, 102.0, 3.2, 1.5),
    ]
    df = pd.DataFrame(rows)
    now = dt.datetime(2026, 10, 5, 12, 30, tzinfo=TZ)
    snap_bull = {"market": {"regime": "BULL"}, "stocks": base_stocks}
    snap_bear = {"market": {"regime": "BEAR"}, "stocks": base_stocks}
    bull = ie.rerank_snapshot(snap_bull, "short", df, top_n=2, now=now)
    bear = ie.rerank_snapshot(snap_bear, "short", df, top_n=2, now=now)
    assert bull and bear
    assert bull[0]["ticker"] == "1111.TW", bull
    # Market regime must not change ranking/score, only position budget.
    assert [x["ticker"] for x in bull] == [x["ticker"] for x in bear]
    assert [x["live_ranking_score"] for x in bull] == [x["live_ranking_score"] for x in bear]
    a_bull = bull[0]["intraday"]
    a_bear = bear[0]["intraday"]
    assert a_bull["intraday_score"] >= 60.0, a_bull
    assert "overheat_penalty" not in a_bull
    assert a_bull["entry_timing"]["score"] <= 28.0
    assert "不追" in a_bull["entry_timing"]["action"]
    assert a_bull["position_guidance"]["percent"] == 0
    assert a_bear["position_guidance"]["percent"] == 0


def test_dynamic_full_market_pool_adds_new_live_leader():
    plan = {"reference_close":100.0,"breakout_level":100.0,"trigger":100.5,"zone_low":98.0,"zone_high":101.0,"chase_limit":103.0,"atr14":3.0,"ma5":100.0,"invalidation":94.0}
    stocks=[]
    universe=[]
    for i in range(260):
        ticker=f"{6000+i}.TW"
        score=95-i*0.15
        stocks.append(_decision_stock(ticker, score, 0.10, 0.64, plan, 100.5, 0.5, 80.0, confidence=70))
        universe.append({"ticker":ticker,"name":ticker,"avg_volume_20d":1_000_000,"avg_turnover_20d":100_000_000,"pre_score":50-i*0.1})
    # Make an otherwise lower-ranked stock a clear intraday leader.
    leader="6250.TW"
    rows=[{"stock_id":"001","close":22000,"open":21950,"high":22100,"low":21900,"average_price":22000,"change_rate":0.5,"total_volume":1,"date":"2026-10-06"}]
    for i in range(260):
        code=str(6000+i)
        hot=(f"{code}.TW"==leader)
        rows.append({"stock_id":code,"close":105.0 if hot else 100.5,"open":100.0,"high":105.2 if hot else 101.0,"low":99.8,"average_price":101.5 if hot else 100.3,"change_rate":5.0 if hot else 0.5,"total_volume":2_000_000 if hot else 600_000,"total_amount":210_000_000 if hot else 60_000_000,"buy_price":104.9 if hot else 100.4,"sell_price":105.1 if hot else 100.6,"buy_volume":600,"sell_volume":400,"date":"2026-10-06"})
    snap={"stocks":stocks,"intraday_universe":universe,"market":{"regime":"BULL"}}
    df=pd.DataFrame(rows)
    now=dt.datetime(2026,10,6,11,30,tzinfo=TZ)
    core=ie.candidate_tickers(snap, per_horizon=80, max_total=220)
    assert leader not in core, "fixture must place leader outside daily core"
    dyn=ie.dynamic_candidate_pool(snap, df, now=now)
    assert leader in dyn["detail_tickers"], dyn["stats"]
    assert dyn["stats"]["detail_n"] <= int(LIVE_SCAN["detail_max"])
    assert dyn["stats"]["quoted_n"] >= 250


def test_continuation_entry_is_explicit_but_not_chase():
    plan={"reference_close":100.0,"breakout_level":100.0,"trigger":100.5,"zone_low":98.0,"zone_high":101.0,"chase_limit":106.0,"atr14":3.0,"ma5":100.0,"invalidation":94.0}
    stock=_decision_stock("CONT.TW",88,0.15,0.68,plan,103.0,3.0,76.0,chase="低",p10=-0.04,confidence=80)
    stock["intraday"].update({"intraday_score":82.0,"relative_market_pct_pt":2.2,"volume_ratio":1.6,"day_position":0.78,"vwap_gap_pct":1.0,"spread_pct":0.15})
    stock["intraday"]["entry_timing"].update({"extension_atr":1.0,"score":76.0,"chase_risk":"低","action":"等拉回/等確認"})
    stock["intraday"]["position_guidance"]={"percent":70}
    out=re.evaluate_buy_opportunity(stock,"short",live=True)
    assert out["eligible_now"] is True, out
    assert out["state"] == "CONTINUATION", out
    assert out["entry_path"] == "CONTINUATION", out
    assert out["position_percent"] <= 35, out

    hot=_decision_stock("HOT.TW",92,0.15,0.70,plan,107.0,7.0,72.0,chase="高",p10=-0.04,confidence=80)
    hot["intraday"].update({"intraday_score":90.0,"relative_market_pct_pt":4.0,"volume_ratio":3.2,"day_position":0.95,"vwap_gap_pct":3.0,"spread_pct":0.15})
    hot["intraday"]["entry_timing"].update({"extension_atr":2.3,"score":72.0,"chase_risk":"高","action":"強勢但不追，等拉回"})
    hot["intraday"]["position_guidance"]={"percent":0}
    out2=re.evaluate_buy_opportunity(hot,"short",live=True)
    assert out2["eligible_now"] is False, out2
    assert "chase_risk" in out2["failed_gate_codes"] or "hard_no" in out2["failed_gate_codes"], out2


def test_failed_gate_diagnostics_are_specific():
    plan={"reference_close":100.0,"breakout_level":100.0,"trigger":100.5,"zone_low":98.0,"zone_high":101.0,"chase_limit":104.0,"atr14":3.0,"ma5":100.0,"invalidation":94.0}
    weak=_decision_stock("WEAK.TW",84,0.01,0.48,plan,100.5,0.5,74.0,chase="低",p10=-0.08,confidence=80)
    weak["intraday"].update({"intraday_score":70.0,"relative_market_pct_pt":0.5,"volume_ratio":1.2,"day_position":0.6})
    out=re.evaluate_buy_opportunity(weak,"short",live=True)
    assert out["eligible_now"] is False, out
    assert "expected_return" in out["failed_gate_codes"], out
    assert "positive_rate" in out["failed_gate_codes"], out
    ranked=re.rank_intraday_buys([weak],"short",n=1)
    summary=re.summarize_buy_gate_failures(ranked)
    assert summary["total"] == 1 and summary["eligible"] == 0, summary
    assert summary["primary"], summary


def test_shared_feature_engine_matches_legacy_calls():
    df = synthetic_ohlcv(900, seed=11)
    benchmark = synthetic_ohlcv(900, seed=22)
    settings = rs.RunSettings(candidate_size=1000)
    t0 = time.perf_counter()
    shared = estimate_all_horizons(df, settings, twii_ret_20d=0.01, benchmark_df=benchmark)
    shared_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    separate = {h: estimate_horizon_return(df, h, settings, twii_ret_20d=0.01, benchmark_df=benchmark) for h in ("short", "mid", "long")}
    separate_sec = time.perf_counter() - t1
    for h in ("short", "mid", "long"):
        assert shared[h].get("technical_factor_score") == separate[h].get("technical_factor_score"), h
        assert shared[h].get("estimate_available") == separate[h].get("estimate_available"), h
    return shared_sec, separate_sec


def test_offline_single_stock_pipeline():
    df = synthetic_ohlcv(900, seed=33)
    settings = rs.RunSettings(candidate_size=1000)
    stock = rs._evaluate_one("9999.TW", "測試股", "測試產業", df, settings, 0.01, {})
    assert stock["ticker"] == "9999.TW"
    for h in ("short", "mid", "long"):
        block = stock["horizons"][h]
        assert_between(block["ranking_score"], 0, 100, f"ranking {h}")
        assert "entry_timing" in block
        assert "position_guidance" in block
        assert "forecast" in block




def test_utf8_ui_strings_are_clean():
    root = Path(__file__).resolve().parent
    for name in ("app.py", "policy_engine.py", "intraday_engine.py", "strategy_config.py", "recommendation_engine.py", "trading_calendar.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "\ufffd" not in text, f"UTF-8 replacement character found in {name}"
    app_text = (root / "app.py").read_text(encoding="utf-8")
    for phrase in ("標的分數", "進場分數", "追價風險", "部位上限"):
        assert phrase in app_text, phrase


def test_session_phases():
    assert market_phase(dt.datetime(2026, 10, 6, 8, 30, tzinfo=TZ)) == PREOPEN
    assert market_phase(dt.datetime(2026, 10, 6, 10, 30, tzinfo=TZ)) == INTRADAY
    assert market_phase(dt.datetime(2026, 10, 6, 14, 30, tzinfo=TZ)) == POSTCLOSE
    assert market_phase(dt.datetime(2026, 10, 4, 10, 30, tzinfo=TZ)) == OFFDAY
    assert session_reference_policy(dt.datetime(2026, 10, 6, 10, 30, tzinfo=TZ))["official_target_date"] is None
    assert session_reference_policy(dt.datetime(2026, 10, 6, 14, 30, tzinfo=TZ))["official_target_date"] == "2026-10-06"


def _decision_stock(ticker: str, selection: float, median: float, positive: float, plan: dict, live_price: float, live_change: float, entry_score: float, chase: str = "低", p10: float = -0.04, confidence: float = 75.0):
    return {
        "ticker": ticker,
        "name": ticker,
        "price": 100.0,
        "avg_volume_20d": 2_000_000,
        "horizons": {
            "short": {
                "ranking_score": selection,
                "plan": plan,
                "forecast": {
                    "estimate_available": True,
                    "local_effective_n": 50,
                    "confidence_score": confidence,
                    "empirical_quality_score": 72.0,
                    "strategy": {
                        "median": median,
                        "mean": median,
                        "p10": p10,
                        "smoothed_positive_rate": positive,
                    },
                },
                "entry_timing": {"score": entry_score, "chase_risk": chase, "action": "買點佳，可分批"},
                "position_guidance": {"percent": 70},
            }
        },
        "live_ranking_score": selection,
        "intraday": {
            "price": live_price,
            "change_rate_pct": live_change,
            "entry_timing": {"score": entry_score, "chase_risk": chase, "action": "買點佳，可分批" if chase == "低" else "漲停/近漲停：強勢但不追"},
            "position_guidance": {"percent": 70 if chase == "低" else 0},
        },
    }


def test_actionable_buy_priority_not_raw_momentum():
    plan = {
        "reference_close": 100.0,
        "breakout_level": 100.0,
        "trigger": 100.5,
        "zone_low": 98.0,
        "zone_high": 101.0,
        "chase_limit": 103.0,
        "atr14": 3.0,
        "ma5": 100.0,
        "invalidation": 94.0,
    }
    # A is the strongest raw idea but has already run to near-limit-up and must
    # not outrank a still-actionable B in the BUY-NOW list.
    a = _decision_stock("A.TW", 95.0, 0.10, 0.70, plan, 109.8, 9.8, 25.0, chase="極高")
    b = _decision_stock("B.TW", 86.0, 0.12, 0.66, plan, 101.0, 1.0, 88.0, chase="低")
    ranked = re.rank_intraday_buys([a, b], "short", n=2)
    assert ranked[0]["ticker"] == "B.TW", ranked
    assert ranked[0]["buy_opportunity"]["eligible_now"] is True, ranked[0]
    assert ranked[1]["buy_opportunity"]["eligible_now"] is False, ranked[1]
    assert ranked[1]["buy_opportunity"]["score"] <= 49.0, ranked[1]


def test_empirical_confidence_shrinkage():
    base = {
        "estimate_available": True,
        "empirical_quality_score": 82.0,
        "strategy": {"median": 0.08, "mean": 0.07, "p10": -0.05, "smoothed_positive_rate": 0.64},
    }
    hi = dict(base, confidence_score=100.0)
    lo = dict(base, confidence_score=0.0)
    hi_score = rs.empirical_evidence_score(hi)
    lo_score = rs.empirical_evidence_score(lo)
    assert hi_score > lo_score > 50.0, (hi_score, lo_score)
    assert abs(hi_score - 82.0) < 0.2, hi_score


def test_tail_risk_controls_reward_risk():
    plan = {
        "reference_close": 100.0, "breakout_level": 100.0, "trigger": 100.5,
        "zone_low": 98.0, "zone_high": 101.0, "chase_limit": 103.0,
        "atr14": 3.0, "ma5": 100.0, "invalidation": 96.0,
    }
    safe = _decision_stock("SAFE.TW", 88, 0.12, 0.66, plan, 101.0, 1.0, 88.0, p10=-0.03, confidence=80)
    risky = _decision_stock("RISK.TW", 88, 0.12, 0.66, plan, 101.0, 1.0, 88.0, p10=-0.18, confidence=80)
    a = re.evaluate_buy_opportunity(safe, "short", live=True)
    b = re.evaluate_buy_opportunity(risky, "short", live=True)
    assert a["reward_risk"] > b["reward_risk"], (a, b)
    assert b["tail_downside"] > b["structural_downside"], b


def test_low_evidence_cannot_be_promoted_to_buy():
    plan = {
        "reference_close": 100.0, "breakout_level": 100.0, "trigger": 100.5,
        "zone_low": 98.0, "zone_high": 101.0, "chase_limit": 103.0,
        "atr14": 3.0, "ma5": 100.0, "invalidation": 94.0,
    }
    weak = _decision_stock("LOWCONF.TW", 94, 0.15, 0.72, plan, 101.0, 1.0, 92.0, confidence=20)
    out = re.evaluate_buy_opportunity(weak, "short", live=True)
    assert out["eligible_now"] is False, out
    assert out["state"] == "WATCH", out
    assert "信心不足" in out["action"], out


def test_validation_protocol_has_at_least_ten_points():
    assert int(VALIDATION_POLICY["points_per_horizon"]) >= 10
    import validate_2026_walkforward as v
    dates = pd.bdate_range("2021-01-04", "2026-12-31")
    bench = pd.DataFrame({"Close": np.linspace(100, 180, len(dates))}, index=dates)
    cutoffs = v.shared_2026_cutoffs(bench, 2026, int(VALIDATION_POLICY["points_per_horizon"]), 120)
    assert len(cutoffs) >= 10, cutoffs
    assert all(x.year == 2026 for x in cutoffs)


def test_validation_holdout_can_reject_overfit_tuning():
    import validate_2026_walkforward as v
    rows = []
    dates = [f"2026-01-{i:02d}" for i in range(2, 14)]
    for di, date in enumerate(dates):
        for j in range(20):
            technical = 95.0 - j * 2.0
            empirical = 55.0 + j * 2.0
            # Calibration rewards technical-heavy ranking; untouched holdout
            # reverses the relation so an overfit candidate must be rejected.
            rank_signal = (20 - j) if di < 8 else j
            realized = (rank_signal - 10) * 0.01
            rows.append({
                "ticker": f"S{j:02d}", "date": date, "horizon": "short",
                "technical_score": technical, "empirical_score": empirical,
                "realized_net_return": realized, "realized_alpha": realized,
                "mae": min(0.0, realized - 0.02), "mfe": max(0.0, realized + 0.03),
            })
    obs = pd.DataFrame(rows)
    result = v.tune_horizon(obs, "short", dates, top_n=5)
    assert result["best_calibration_share"] != result["production_share"], result
    assert result["accepted"] is False, result
    assert abs(result["selected_share"] - result["production_share"]) < 1e-4, result


def test_offline_run_scan_regime_does_not_rank():
    """Exercise run_scan and verify current-day partial bars are excluded intraday."""
    base = synthetic_ohlcv(420, seed=101)
    completed_date = str(base.index[-1].date())
    partial_date = (base.index[-1] + pd.offsets.BDay(1)).date().isoformat()
    partial_ts = pd.Timestamp(partial_date)
    tickers = [f"{3000+i}.TW" for i in range(12)]
    frames = {}
    for i, ticker in enumerate(tickers):
        df = base.copy()
        scale = 0.86 + i * 0.025
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] * scale
        df["Volume"] = df["Volume"] * (0.8 + i * 0.04)
        df.loc[df.index[-40:], ["Open", "High", "Low", "Close"]] *= np.linspace(1.0, 1.0 + i * 0.006, 40)[:, None]
        # Deliberately inject an unfinished current-day bar that must be ignored.
        last = df.iloc[-1].copy()
        last["Close"] *= 1.10
        last["High"] = max(last["High"], last["Close"])
        df.loc[partial_ts] = last
        frames[ticker] = df.sort_index()
    benchmark = synthetic_ohlcv(420, seed=202)
    idx_last = benchmark.iloc[-1].copy()
    idx_last["Close"] *= 1.05
    benchmark.loc[partial_ts] = idx_last
    frames["^TWII"] = benchmark.sort_index()

    class FakeStore:
        def __init__(self, _path):
            self.frames = frames
        def batch_fetch_and_update(self, requested, period="5y", progress=None):
            if progress:
                progress(len(requested), len(requested), "fake price cache")
            return {"downloaded_tickers": len(requested), "requested": len(requested), "mode": "incremental", "covered_ratio": 1.0, "last_full_refresh": completed_date, "errors": []}
        def overlay_official_eod(self, target_date=None, progress=None):
            if progress:
                progress(1, 1, "fake official close")
            return {
                "twse_date": completed_date, "tpex_date": completed_date, "index_date": completed_date,
                "twse_target_response": True, "twse_target_has_data": True,
                "tpex_target_response": True, "tpex_target_has_data": True,
                "errors": [],
            }
        def iter_prices(self, requested, limit=None, end_date=None):
            items = requested[:limit] if limit else requested
            for t in items:
                if t in self.frames:
                    df = self.frames[t].copy()
                    if end_date:
                        df = df[df.index <= pd.Timestamp(end_date)]
                    yield t, df
        def get_prices(self, ticker, limit=None, end_date=None):
            df = self.frames.get(ticker, pd.DataFrame()).copy()
            if end_date and not df.empty:
                df = df[df.index <= pd.Timestamp(end_date)]
            if limit and not df.empty:
                df = df.tail(int(limit))
            return df

    universe = pd.DataFrame({
        "ticker": tickers,
        "name": [f"測試{i}" for i in range(len(tickers))],
        "industry": ["測試產業"] * len(tickers),
    })

    originals = {
        "DailyPriceStore": rs.DailyPriceStore,
        "fetch_twse_universe": rs.fetch_twse_universe,
        "_market_context": rs._market_context,
        "_enrich_research": rs._enrich_research,
        "_enrich_tickers": rs._enrich_tickers,
    }
    try:
        rs.DailyPriceStore = FakeStore
        rs.fetch_twse_universe = lambda: universe.copy()
        rs._enrich_research = lambda stocks, data_dir, settings, progress=None: None
        rs._enrich_tickers = lambda stocks, data_dir, tickers, progress=None, progress_value=0.88: None

        def run(regime):
            rs._market_context = lambda store, end_date=None: {
                "benchmark": "^TWII", "regime": regime, "ret20": 0.01,
                "day_change_pct": 0.5, "price": 22000.0, "ma20": 21500.0,
                "ma60": 21000.0, "price_date": completed_date,
            }
            settings = rs.RunSettings(candidate_size=50, research_pool_per_horizon=5)
            now = dt.datetime.combine(pd.Timestamp(partial_date).date(), dt.time(10, 30), tzinfo=TZ)
            return rs.run_scan(Path("/tmp/alpha-v164-test"), settings, now=now)

        bull = run("BULL")
        bear = run("BEAR")
        assert bull["price_date"] == completed_date, (bull["price_date"], completed_date)
        assert bull["session"]["analysis_mode"] == "INTRADAY_BASELINE", bull["session"]
        assert bull["session"]["twse_complete_date"] == completed_date
        for h in ("short", "mid", "long"):
            bull_rank = [(x["ticker"], x["horizons"][h]["ranking_score"]) for x in rs.select_market_best(bull, h, n=5)]
            bear_rank = [(x["ticker"], x["horizons"][h]["ranking_score"]) for x in rs.select_market_best(bear, h, n=5)]
            assert bull_rank == bear_rank, (h, bull_rank, bear_rank)
        b = rs.select_market_best(bull, "short", n=1)[0]["horizons"]["short"]["position_guidance"]["percent"]
        r = rs.select_market_best(bear, "short", n=1)[0]["horizons"]["short"]["position_guidance"]["percent"]
        assert b >= r, (b, r)
        assert bull["model_version"] == ARCHITECTURE_VERSION
        assert len(bull.get("intraday_universe") or []) == len(tickers)
    finally:
        for name, value in originals.items():
            setattr(rs, name, value)




def test_postclose_pending_to_final_transition():
    base = synthetic_ohlcv(420, seed=303)
    prev_date = str(base.index[-1].date())
    today_ts = base.index[-1] + pd.offsets.BDay(1)
    today = today_ts.date().isoformat()
    tickers = [f"{5000+i}.TW" for i in range(8)]
    frames = {}
    for i, ticker in enumerate(tickers):
        df = base.copy()
        last = df.iloc[-1].copy()
        last[["Open", "High", "Low", "Close"]] = last[["Open", "High", "Low", "Close"]] * (1.0 + 0.01 * i)
        last["Volume"] *= 1.0 + 0.05 * i
        df.loc[today_ts] = last
        frames[ticker] = df.sort_index()
    benchmark = synthetic_ohlcv(420, seed=404)
    idx_last = benchmark.iloc[-1].copy()
    idx_last["Close"] *= 1.01
    benchmark.loc[today_ts] = idx_last
    frames["^TWII"] = benchmark.sort_index()
    state = {"official_date": prev_date, "targets": []}

    class FakeStore:
        def __init__(self, _path): self.frames = frames
        def batch_fetch_and_update(self, requested, period="5y", progress=None):
            return {"downloaded_tickers": len(requested), "requested": len(requested), "mode": "incremental", "covered_ratio": 1.0, "last_full_refresh": prev_date, "errors": []}
        def overlay_official_eod(self, target_date=None, progress=None):
            state["targets"].append(target_date)
            d = state["official_date"]
            return {"twse_date": d, "tpex_date": d, "index_date": d, "errors": []}
        def iter_prices(self, requested, limit=None, end_date=None):
            items = requested[:limit] if limit else requested
            for t in items:
                if t in self.frames:
                    df = self.frames[t].copy()
                    if end_date: df = df[df.index <= pd.Timestamp(end_date)]
                    yield t, df
        def get_prices(self, ticker, limit=None, end_date=None):
            df = self.frames.get(ticker, pd.DataFrame()).copy()
            if end_date and not df.empty: df = df[df.index <= pd.Timestamp(end_date)]
            if limit and not df.empty: df = df.tail(int(limit))
            return df

    universe = pd.DataFrame({"ticker": tickers, "name": [f"P{i}" for i in range(len(tickers))], "industry": ["測試"] * len(tickers)})
    originals = {"DailyPriceStore": rs.DailyPriceStore, "fetch_twse_universe": rs.fetch_twse_universe, "_enrich_research": rs._enrich_research, "_enrich_tickers": rs._enrich_tickers}
    try:
        rs.DailyPriceStore = FakeStore
        rs.fetch_twse_universe = lambda: universe.copy()
        rs._enrich_research = lambda stocks, data_dir, settings, progress=None: None
        rs._enrich_tickers = lambda stocks, data_dir, tickers, progress=None, progress_value=0.88: None
        settings = rs.RunSettings(candidate_size=50, research_pool_per_horizon=5)
        now = dt.datetime.combine(today_ts.date(), dt.time(14, 30), tzinfo=TZ)
        pending = rs.run_scan(Path("/tmp/alpha-v164-postclose"), settings, now=now)
        assert state["targets"][-1] == today
        assert pending["session"]["analysis_mode"] == "POSTCLOSE_PENDING", pending["session"]
        assert pending["price_date"] == prev_date
        state["official_date"] = today
        final = rs.run_scan(Path("/tmp/alpha-v164-postclose"), settings, now=now)
        assert state["targets"][-1] == today
        assert final["session"]["analysis_mode"] == "POSTCLOSE_FINAL", final["session"]
        assert final["price_date"] == today
        assert final["session"]["official_today_complete"] is True
    finally:
        for name, value in originals.items(): setattr(rs, name, value)

def test_cross_horizon_allocation_helpers():
    stocks = []
    for i in range(8):
        # The first names deliberately rank highly in all horizons so overlap is
        # guaranteed and the max-appearance constraint is exercised.
        stocks.append({
            "ticker": f"{1000+i}.TW",
            "name": f"S{i}",
            "horizons": {
                "short": {"ranking_score": 100 - i},
                "mid": {"ranking_score": 100 - i * 0.9},
                "long": {"ranking_score": 100 - i * 0.8},
            },
        })
    snap = {"stocks": stocks}
    out = rs.select_cross_horizon_shortlists(snap, n=5, max_appearances=2)
    assert set(out) == {"short", "mid", "long"}, out
    assert all(len(out[h]) == 5 for h in out), out
    counts = {}
    for picks in out.values():
        for stock in picks:
            code = stock["ticker"]
            counts[code] = counts.get(code, 0) + 1
    assert max(counts.values()) <= 2, counts
    metrics = rs.shortlist_overlap_metrics(out)
    assert metrics["total_slots"] == 15, metrics
    assert metrics["unique_tickers"] == len(counts), metrics
    assert metrics["max_appearances"] <= 2, metrics


def test_app_service_api_contract():
    """Catch app.py calling a radar_service API that was omitted from packaging."""
    import ast
    root = Path(__file__).resolve().parent
    tree = ast.parse((root / "app.py").read_text(encoding="utf-8"))
    attrs = sorted({
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "service"
    })
    missing = [name for name in attrs if not hasattr(rs, name)]
    assert not missing, f"app.py references missing radar_service APIs: {missing}"


def test_app_all_session_contract():
    root = Path(__file__).resolve().parent
    text = (root / "app.py").read_text(encoding="utf-8")
    for phrase in ("盤中買點", "明日關注", "完整日線前 5 名"):
        assert phrase in text, phrase
    cal_text = (root / "trading_calendar.py").read_text(encoding="utf-8")
    assert "更新盤後資料與下一交易日關注清單" in cal_text
    assert "live_overlay_" not in text, "short/mid/long pages must not silently mix intraday overlay"
    assert "service.run_scan(DATA_DIR, settings, progress=update, now=scan_now)" in text
    assert "intraday_buy_fragment" in text
    assert "_render_next_session" in text

def main():
    tests = [
        test_config_weights,
        test_entry_separation,
        test_regime_only_changes_position,
        test_time_matured_intraday_weight,
        test_session_phases,
        test_limit_up_can_rank_high_but_not_be_buyable,
        test_actionable_buy_priority_not_raw_momentum,
        test_empirical_confidence_shrinkage,
        test_tail_risk_controls_reward_risk,
        test_low_evidence_cannot_be_promoted_to_buy,
        test_dynamic_full_market_pool_adds_new_live_leader,
        test_continuation_entry_is_explicit_but_not_chase,
        test_failed_gate_diagnostics_are_specific,
        test_validation_protocol_has_at_least_ten_points,
        test_validation_holdout_can_reject_overfit_tuning,
        test_offline_single_stock_pipeline,
        test_utf8_ui_strings_are_clean,
        test_cross_horizon_allocation_helpers,
        test_app_service_api_contract,
        test_app_all_session_contract,
        test_offline_run_scan_regime_does_not_rank,
        test_postclose_pending_to_final_transition,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    shared_sec, separate_sec = test_shared_feature_engine_matches_legacy_calls()
    print("PASS test_shared_feature_engine_matches_legacy_calls")
    print(f"BENCH shared_features={shared_sec:.4f}s separate_3x={separate_sec:.4f}s speedup={separate_sec/max(shared_sec,1e-9):.2f}x")
    print(f"ALL TESTS PASSED | {ARCHITECTURE_VERSION}")


if __name__ == "__main__":
    main()
