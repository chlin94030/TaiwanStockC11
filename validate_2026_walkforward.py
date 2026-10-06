"""Leakage-controlled 2026 walk-forward validation for Alpha Radar V16.6.

Purpose
-------
Replay the *price/benchmark selection core* at at least 10 historical 2026
cutoffs using only data available at each cutoff, then measure realized
10/40/120-session returns.  Present-day fundamentals and institutional flows are
intentionally excluded because a point-in-time archive for those fields is not
part of the app yet; backfilling today's values would create look-ahead bias.

The script also performs a deliberately small calibration of the technical vs.
historical-analog share.  Parameters are selected on the first portion of the
cutoffs and accepted only if they also improve an untouched holdout without
materially worsening downside.  It never silently edits production config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from market_data import DailyPriceStore, fetch_twse_universe
from radar_service import RunSettings, empirical_evidence_score
from return_first_model import estimate_all_horizons
from strategy_config import SELECTION_WEIGHTS, VALIDATION_POLICY, ARCHITECTURE_VERSION

HORIZONS = {"short": 10, "mid": 40, "long": 120}


def _position_at_or_before(df: pd.DataFrame, date: pd.Timestamp) -> int | None:
    if df is None or df.empty:
        return None
    pos = int(df.index.searchsorted(pd.Timestamp(date), side="right") - 1)
    return pos if 0 <= pos < len(df) else None


def trading_cost(settings: RunSettings) -> float:
    return float(settings.commission * 2 + settings.sell_tax + settings.slippage * 2)


def future_path_stats(df: pd.DataFrame, cutoff: pd.Timestamp, days: int, total_cost: float) -> dict | None:
    pos = _position_at_or_before(df, cutoff)
    if pos is None or pos + days >= len(df):
        return None
    entry = float(df["Close"].iloc[pos])
    path = pd.to_numeric(df["Close"].iloc[pos : pos + days + 1], errors="coerce").dropna()
    if not np.isfinite(entry) or entry <= 0 or len(path) < days + 1:
        return None
    gross_path = path.to_numpy(dtype=float) / entry - 1.0
    realized = float(gross_path[-1] - total_cost)
    # MAE/MFE are marked from the entry price.  Costs are applied only to the
    # realized exit return, not to intermediate marks.
    return {
        "realized_net_return": realized,
        "mae": float(np.nanmin(gross_path)),
        "mfe": float(np.nanmax(gross_path)),
    }


def benchmark_future(benchmark: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> float | None:
    pos = _position_at_or_before(benchmark, cutoff)
    if pos is None or pos + days >= len(benchmark):
        return None
    a = float(benchmark["Close"].iloc[pos])
    b = float(benchmark["Close"].iloc[pos + days])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return b / a - 1.0


def choose_real_sample(store: DailyPriceStore, universe: pd.DataFrame, target: int, period: str, seed: int) -> list[str]:
    tickers = universe["ticker"].astype(str).drop_duplicates().tolist()
    rng = random.Random(int(seed))
    rng.shuffle(tickers)
    selected: list[str] = []
    cursor = 0
    while cursor < len(tickers) and len(selected) < target:
        batch = tickers[cursor : cursor + 80]
        cursor += len(batch)
        store.batch_fetch_and_update(batch, period=period)
        for ticker in batch:
            df = store.get_prices(ticker)
            if len(df) < int(VALIDATION_POLICY["min_history_rows"]) + 130:
                continue
            recent = df.tail(120)
            if float(pd.to_numeric(recent["Volume"], errors="coerce").fillna(0).mean()) <= 0:
                continue
            selected.append(ticker)
            if len(selected) >= target:
                break
    return selected


def shared_2026_cutoffs(benchmark: pd.DataFrame, year: int, points: int, longest_days: int = 120) -> list[pd.Timestamp]:
    """Choose evenly spread dates in the requested year with 120-session future.

    Using common cutoffs means short/mid/long are judged on exactly the same
    historical market states.  The newest eligible cutoff is naturally earlier
    for long-horizon validation; this is intentional and avoids incomplete
    forward outcomes.
    """
    if benchmark is None or benchmark.empty:
        return []
    idx = pd.DatetimeIndex(benchmark.index)
    eligible: list[pd.Timestamp] = []
    min_hist = int(VALIDATION_POLICY["min_history_rows"])
    for pos, d in enumerate(idx):
        d = pd.Timestamp(d)
        if d.year != int(year):
            continue
        if pos < min_hist or pos + int(longest_days) >= len(idx):
            continue
        eligible.append(d)
    if len(eligible) < points:
        return eligible
    ids = np.linspace(0, len(eligible) - 1, int(points), dtype=int)
    # np.linspace can duplicate only when points > eligible, already handled.
    return [eligible[int(i)] for i in ids]


def production_core_share(horizon: str) -> float:
    w = SELECTION_WEIGHTS[horizon]
    core = float(w["technical"] + w["empirical"])
    return float(w["technical"] / core) if core > 0 else 0.5


def score_from_share(technical: float, empirical: float, technical_share: float) -> float:
    s = float(np.clip(technical_share, 0.0, 1.0))
    return float(technical * s + empirical * (1.0 - s))


def _safe_spearman(g: pd.DataFrame, score_col: str) -> float | None:
    if len(g) < 8 or g[score_col].nunique() < 3 or g["realized_net_return"].nunique() < 3:
        return None
    v = g[score_col].corr(g["realized_net_return"], method="spearman")
    return None if pd.isna(v) else float(v)


def evaluate_config(obs: pd.DataFrame, horizon: str, technical_share: float, top_n: int, dates: Iterable[str] | None = None) -> dict:
    g = obs[obs["horizon"] == horizon].copy()
    if dates is not None:
        keep = set(str(x) for x in dates)
        g = g[g["date"].astype(str).isin(keep)].copy()
    if g.empty:
        return {"dates": 0, "top_picks": 0, "objective": None}
    score_col = "candidate_score"
    g[score_col] = g.apply(
        lambda r: score_from_share(float(r["technical_score"]), float(r["empirical_score"]), technical_share), axis=1
    )
    tops = []
    date_rows = []
    ics = []
    for date, d in g.groupby("date", sort=True):
        d = d.dropna(subset=[score_col, "realized_net_return"]).sort_values(score_col, ascending=False)
        if d.empty:
            continue
        td = d.head(int(top_n)).copy()
        if td.empty:
            continue
        tops.append(td)
        ic = _safe_spearman(d, score_col)
        if ic is not None:
            ics.append(ic)
        date_rows.append({
            "date": str(date),
            "top_mean": float(td["realized_net_return"].mean()),
            "top_median": float(td["realized_net_return"].median()),
            "all_median": float(d["realized_net_return"].median()),
            "alpha_mean": float(td["realized_alpha"].dropna().mean()) if td["realized_alpha"].notna().any() else np.nan,
            "positive_rate": float((td["realized_net_return"] > 0).mean()),
            "beat_market_rate": float((td["realized_alpha"].dropna() > 0).mean()) if td["realized_alpha"].notna().any() else np.nan,
            "mae_mean": float(td["mae"].mean()),
        })
    if not tops:
        return {"dates": 0, "top_picks": 0, "objective": None}
    top = pd.concat(tops, ignore_index=True)
    dd = pd.DataFrame(date_rows)
    base_positive = float((g["realized_net_return"] > 0).mean())
    top_positive = float((top["realized_net_return"] > 0).mean())
    median_lift = float((dd["top_median"] - dd["all_median"]).mean())
    alpha = float(top["realized_alpha"].dropna().mean()) if top["realized_alpha"].notna().any() else 0.0
    p10 = float(top["realized_net_return"].quantile(0.10))
    pos_lift = top_positive - base_positive
    mean_ic = float(np.mean(ics)) if ics else 0.0
    # Robust objective: reward alpha/lift; small reward for hit-rate and rank IC;
    # explicitly penalize a negative p10 tail.  All return terms are decimals.
    objective = alpha + 0.75 * median_lift + 0.10 * pos_lift + 0.02 * mean_ic + 0.20 * min(0.0, p10)
    return {
        "dates": int(dd["date"].nunique()),
        "top_picks": int(len(top)),
        "technical_share": round(float(technical_share), 4),
        "objective": round(float(objective), 6),
        "mean_rank_ic": round(mean_ic, 4),
        "top_mean_return": round(float(top["realized_net_return"].mean()), 6),
        "top_median_return": round(float(top["realized_net_return"].median()), 6),
        "top_positive_rate": round(top_positive, 4),
        "positive_rate_lift": round(pos_lift, 4),
        "same_date_median_lift": round(median_lift, 6),
        "top_mean_alpha": round(alpha, 6),
        "beat_market_rate": round(float((top["realized_alpha"].dropna() > 0).mean()), 4) if top["realized_alpha"].notna().any() else None,
        "top_p10_return": round(p10, 6),
        "mean_mae": round(float(top["mae"].mean()), 6),
        "mean_mfe": round(float(top["mfe"].mean()), 6),
        "worst_date_top_mean": round(float(dd["top_mean"].min()), 6),
        "best_date_top_mean": round(float(dd["top_mean"].max()), 6),
        "date_metrics": date_rows,
    }


def _alpha_win_dates(candidate: dict, baseline: dict) -> int:
    a = {x["date"]: x for x in candidate.get("date_metrics", [])}
    b = {x["date"]: x for x in baseline.get("date_metrics", [])}
    wins = 0
    for d in set(a) & set(b):
        av = a[d].get("alpha_mean")
        bv = b[d].get("alpha_mean")
        if np.isfinite(av) and np.isfinite(bv) and av > bv:
            wins += 1
    return wins


def tune_horizon(obs: pd.DataFrame, horizon: str, cutoff_dates: list[str], top_n: int) -> dict:
    frac = float(VALIDATION_POLICY["calibration_fraction"])
    split = max(2, min(len(cutoff_dates) - 2, int(round(len(cutoff_dates) * frac))))
    calibration_dates = cutoff_dates[:split]
    holdout_dates = cutoff_dates[split:]
    base_share = production_core_share(horizon)
    grid = sorted(set([base_share, *[float(x) for x in VALIDATION_POLICY["technical_share_grid"]]]))
    calib_results = [evaluate_config(obs, horizon, share, top_n, calibration_dates) for share in grid]
    usable = [x for x in calib_results if x.get("objective") is not None]
    best_calib = max(usable, key=lambda x: x["objective"]) if usable else evaluate_config(obs, horizon, base_share, top_n, calibration_dates)
    tuned_share = float(best_calib.get("technical_share", base_share))
    base_holdout = evaluate_config(obs, horizon, base_share, top_n, holdout_dates)
    tuned_holdout = evaluate_config(obs, horizon, tuned_share, top_n, holdout_dates)

    base_obj = base_holdout.get("objective")
    tuned_obj = tuned_holdout.get("objective")
    holdout_dates_n = int(tuned_holdout.get("dates") or 0)
    wins = _alpha_win_dates(tuned_holdout, base_holdout)
    accepted = False
    reasons = []
    if tuned_share == base_share:
        reasons.append("calibration retained production share")
    elif base_obj is None or tuned_obj is None or holdout_dates_n < 2:
        reasons.append("insufficient untouched holdout")
    else:
        objective_improved = tuned_obj >= base_obj + 0.002
        downside_ok = float(tuned_holdout.get("top_p10_return", -9)) >= float(base_holdout.get("top_p10_return", -9)) - 0.01
        lift_ok = float(tuned_holdout.get("same_date_median_lift", -9)) >= float(base_holdout.get("same_date_median_lift", -9)) - 0.002
        wins_ok = wins >= max(1, int(np.ceil(holdout_dates_n / 2)))
        accepted = bool(objective_improved and downside_ok and lift_ok and wins_ok)
        reasons.extend([
            f"objective_improved={objective_improved}",
            f"downside_ok={downside_ok}",
            f"median_lift_ok={lift_ok}",
            f"alpha_date_wins={wins}/{holdout_dates_n}",
        ])
    selected_share = tuned_share if accepted else base_share
    selected_all = evaluate_config(obs, horizon, selected_share, top_n, cutoff_dates)
    return {
        "horizon": horizon,
        "calibration_dates": calibration_dates,
        "holdout_dates": holdout_dates,
        "production_share": round(base_share, 4),
        "calibration_grid": calib_results,
        "best_calibration_share": round(tuned_share, 4),
        "baseline_holdout": base_holdout,
        "candidate_holdout": tuned_holdout,
        "accepted": accepted,
        "acceptance_reasons": reasons,
        "selected_share": round(selected_share, 4),
        "selected_all_dates": selected_all,
    }


def full_weights_from_core_share(horizon: str, share: float) -> dict:
    w = dict(SELECTION_WEIGHTS[horizon])
    core = float(w["technical"] + w["empirical"])
    w["technical"] = core * float(share)
    w["empirical"] = core * (1.0 - float(share))
    return {k: round(float(v), 6) for k, v in w.items()}


def run(out_dir: Path, stocks: int, period: str, year: int, points: int, seed: int, top_n: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = DailyPriceStore(out_dir / "validation_prices.sqlite")
    settings = RunSettings(history_period=period, model_family="price_only")
    universe = fetch_twse_universe()
    if len(universe) < stocks:
        raise RuntimeError(f"股票名單僅取得 {len(universe)} 檔，無法做 {stocks} 檔真實標的驗證")

    store.batch_fetch_and_update(["^TWII"], period=period)
    benchmark = store.get_prices("^TWII")
    if len(benchmark) < int(VALIDATION_POLICY["min_history_rows"]) + 130:
        raise RuntimeError("TAIEX 歷史資料不足，驗證中止")
    sample = choose_real_sample(store, universe, stocks, period, seed)
    if len(sample) < stocks:
        raise RuntimeError(f"僅取得 {len(sample)} 檔具足夠歷史資料的真實標的，未達 {stocks} 檔")

    cutoffs = shared_2026_cutoffs(benchmark, year, points, max(HORIZONS.values()))
    if len(cutoffs) < 10:
        raise RuntimeError(f"{year} 年具完整 120 日後驗結果的共同時間點僅 {len(cutoffs)} 個，少於 10 個")
    cutoff_dates = [str(x.date()) for x in cutoffs]

    meta = universe.set_index("ticker").to_dict("index")
    pd.DataFrame([{"ticker": t, **meta.get(t, {})} for t in sample]).to_csv(out_dir / "sample_stocks.csv", index=False, encoding="utf-8-sig")

    total_cost = trading_cost(settings)
    records: list[dict] = []
    for ci, cutoff in enumerate(cutoffs, start=1):
        bench_hist = benchmark[benchmark.index <= cutoff].copy()
        if len(bench_hist) < int(VALIDATION_POLICY["min_history_rows"]):
            continue
        b20 = float(bench_hist["Close"].iloc[-1] / bench_hist["Close"].iloc[-21] - 1.0) if len(bench_hist) >= 21 else 0.0
        print(f"[{ci}/{len(cutoffs)}] cutoff {cutoff.date()} ...", flush=True)
        for ticker in sample:
            df = store.get_prices(ticker)
            hist = df[df.index <= cutoff].copy()
            if len(hist) < int(VALIDATION_POLICY["min_history_rows"]):
                continue
            forecasts = estimate_all_horizons(hist, settings, twii_ret_20d=b20, benchmark_df=bench_hist)
            for horizon, days in HORIZONS.items():
                forecast = forecasts[horizon]
                if not forecast.get("estimate_available"):
                    continue
                technical = forecast.get("technical_factor_score")
                empirical = empirical_evidence_score(forecast)
                path = future_path_stats(df, cutoff, days, total_cost)
                bret = benchmark_future(benchmark, cutoff, days)
                if technical is None or empirical is None or path is None:
                    continue
                strat = forecast.get("strategy") or {}
                records.append({
                    "ticker": ticker,
                    "date": str(cutoff.date()),
                    "horizon": horizon,
                    "days": days,
                    "technical_score": float(technical),
                    "empirical_score": float(empirical),
                    "empirical_raw": forecast.get("empirical_quality_score"),
                    "confidence_score": forecast.get("confidence_score"),
                    "analog_n": forecast.get("local_effective_n"),
                    "predicted_median": strat.get("median"),
                    "predicted_positive_rate": strat.get("smoothed_positive_rate"),
                    "predicted_p10": strat.get("p10"),
                    **path,
                    "benchmark_return": bret,
                    "realized_alpha": None if bret is None else path["realized_net_return"] - bret,
                })

    obs = pd.DataFrame(records)
    if obs.empty:
        raise RuntimeError("沒有產生可用 walk-forward 觀察值")
    obs.to_csv(out_dir / "walkforward_observations.csv", index=False, encoding="utf-8-sig")

    tuning = {h: tune_horizon(obs, h, cutoff_dates, top_n) for h in HORIZONS}
    selected_weights = {h: full_weights_from_core_share(h, float(tuning[h]["selected_share"])) for h in HORIZONS}
    all_baseline = {h: evaluate_config(obs, h, production_core_share(h), top_n, cutoff_dates) for h in HORIZONS}
    all_selected = {h: tuning[h]["selected_all_dates"] for h in HORIZONS}

    report = {
        "status": "COMPLETED",
        "architecture_version": ARCHITECTURE_VERSION,
        "validation_year": year,
        "real_stocks": int(obs["ticker"].nunique()),
        "shared_cutoff_dates": cutoff_dates,
        "cutoff_count": len(cutoff_dates),
        "top_n": top_n,
        "holding_sessions": HORIZONS,
        "lookahead_control": "Each stock and TAIEX history is truncated at the cutoff before any forward return is read.",
        "scope": "Price/benchmark selection core only; no present-day fundamental/flow values are backfilled into history.",
        "survivorship_note": "Universe is sampled from currently listed TWSE/TPEx stocks, so delisted names are not represented.",
        "baseline_metrics": all_baseline,
        "tuning": tuning,
        "selected_weights_if_applied": selected_weights,
        "selected_metrics": all_selected,
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "selected_weights.json").write_text(json.dumps(selected_weights, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report, out_dir / "validation_report.md")
    return report


def _pct(x) -> str:
    return "—" if x is None else f"{float(x)*100:.2f}%"


def write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        f"# Alpha Radar {report.get('architecture_version')} — {report.get('validation_year')} Walk-forward",
        "",
        f"- Real stocks: **{report.get('real_stocks')}**",
        f"- Shared historical cutoffs: **{report.get('cutoff_count')}**",
        f"- Top picks per cutoff/horizon: **{report.get('top_n')}**",
        "- Holding periods: short=10, mid=40, long=120 trading sessions",
        "- Point-in-time rule: future prices are read only after ranking is completed.",
        "- Fundamental/flow history is not backfilled from today's values.",
        "",
        "| Horizon | Baseline top median | Selected top median | Baseline alpha | Selected alpha | Selected positive | P10 | Mean MAE | Tech share | Tuning accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for h in HORIZONS:
        b = report["baseline_metrics"][h]
        t = report["tuning"][h]
        s = report["selected_metrics"][h]
        lines.append(
            f"| {h} | {_pct(b.get('top_median_return'))} | {_pct(s.get('top_median_return'))} | "
            f"{_pct(b.get('top_mean_alpha'))} | {_pct(s.get('top_mean_alpha'))} | {_pct(s.get('top_positive_rate'))} | "
            f"{_pct(s.get('top_p10_return'))} | {_pct(s.get('mean_mae'))} | {float(t.get('selected_share',0)):.3f} | {t.get('accepted')} |"
        )
    lines += ["", "## Cutoff dates", "", ", ".join(report.get("shared_cutoff_dates", [])), "", "## Weight decision", ""]
    for h in HORIZONS:
        t = report["tuning"][h]
        lines.append(f"- **{h}**: production share={t['production_share']}, calibration best={t['best_calibration_share']}, selected={t['selected_share']}, accepted={t['accepted']}; " + "; ".join(t.get("acceptance_reasons", [])))
    lines += ["", "## Important limitation", "", report.get("scope", ""), "", report.get("survivorship_note", "")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/validation_2026_v165")
    ap.add_argument("--stocks", type=int, default=int(VALIDATION_POLICY["stocks"]))
    ap.add_argument("--period", default=str(VALIDATION_POLICY["history_period"]))
    ap.add_argument("--year", type=int, default=int(VALIDATION_POLICY["year"]))
    ap.add_argument("--points", type=int, default=int(VALIDATION_POLICY["points_per_horizon"]))
    ap.add_argument("--seed", type=int, default=int(VALIDATION_POLICY["seed"]))
    ap.add_argument("--top-n", type=int, default=int(VALIDATION_POLICY["top_n"]))
    args = ap.parse_args()
    out = Path(args.out)
    try:
        report = run(out, args.stocks, args.period, args.year, args.points, args.seed, args.top_n)
        print(json.dumps({
            "status": report["status"],
            "version": report["architecture_version"],
            "cutoffs": report["cutoff_count"],
            "stocks": report["real_stocks"],
            "selected_weights": report["selected_weights_if_applied"],
        }, ensure_ascii=False, indent=2))
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "NOT_RUN",
            "reason": f"{type(exc).__name__}: {exc}",
            "note": "No synthetic data was substituted. Real 2026 validation requires reachable historical market data or a populated validation_prices.sqlite cache.",
        }
        (out / "validation_not_run.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
