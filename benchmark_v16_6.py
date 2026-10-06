"""Offline performance probe for the V16.6 one-minute dynamic intraday layer.

No network is used. The benchmark measures only local quote scoring / candidate
pool construction so provider latency is not confused with model CPU time.
"""
from __future__ import annotations
import datetime as dt
import statistics
import time
import pandas as pd
import intraday_engine as ie
from strategy_config import LIVE_SCAN

TZ = dt.timezone(dt.timedelta(hours=8))


def build_fixture(n: int = 1500):
    stocks=[]; universe=[]
    rows=[{"stock_id":"001","close":22000,"open":21900,"high":22100,"low":21850,"average_price":21980,"change_rate":0.8,"total_volume":1,"date":"2026-10-06"}]
    for i in range(n):
        code=str(1000+i); ticker=code+'.TW'
        horizons={h:{"ranking_score":80-(i%100)*0.2} for h in ('short','mid','long')}
        stocks.append({"ticker":ticker,"horizons":horizons})
        universe.append({"ticker":ticker,"name":ticker,"avg_volume_20d":1_500_000,"avg_turnover_20d":120_000_000,"pre_score":40+(i%60)*0.5})
        chg=((i%80)-20)/10
        rows.append({"stock_id":code,"close":100*(1+chg/100),"open":100,"high":101+max(chg,0),"low":99,"average_price":100.2,"change_rate":chg,"total_volume":500000+(i%40)*50000,"total_amount":80_000_000+(i%50)*3_000_000,"buy_price":99.9,"sell_price":100.1,"buy_volume":600,"sell_volume":400,"date":"2026-10-06"})
    return {"stocks":stocks,"intraday_universe":universe}, pd.DataFrame(rows)


def main():
    snap, df = build_fixture()
    now=dt.datetime(2026,10,6,12,0,tzinfo=TZ)
    samples=[]; result=None
    for _ in range(15):
        t0=time.perf_counter(); result=ie.dynamic_candidate_pool(snap,df,now=now); samples.append(time.perf_counter()-t0)
    print(f"quotes={result['stats']['quoted_n']} detail={result['stats']['detail_n']} refresh={LIVE_SCAN['refresh_seconds']}s")
    print(f"dynamic_pool_median={statistics.median(samples):.4f}s p95={sorted(samples)[int(len(samples)*0.95)-1]:.4f}s max={max(samples):.4f}s")

if __name__ == '__main__':
    main()
