"""Smoke the V16.6 decision render paths with synthetic data and no network."""
from __future__ import annotations
import datetime as dt
import importlib
import sys
import types
import pandas as pd

class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *args): return False

st = types.ModuleType("streamlit")
st.session_state = {}
st.secrets = {}
st.sidebar = _Ctx()
def _decorator(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs: return args[0]
    return lambda fn: fn
for name in ("cache_data","fragment"): setattr(st, name, _decorator)
for name in ("markdown","caption","info","warning","error","success","dataframe","plotly_chart"):
    setattr(st, name, lambda *a, **k: None)
st.expander = lambda *a, **k: _Ctx()
st.toggle = lambda *a, value=False, **k: value
sys.modules["streamlit"] = st
app = importlib.import_module("app")

TZ = dt.timezone(dt.timedelta(hours=8))
now = dt.datetime(2026,10,6,10,30,tzinfo=TZ)
app._taipei_timestamp = lambda: now

plan={"reference_close":100.0,"breakout_level":100.0,"trigger":100.5,"zone_low":98.0,"zone_high":101.0,"chase_limit":103.0,"atr14":3.0,"ma5":100.0,"ma20":99.0,"invalidation":94.0}
def stock(code, score, median):
    return {"ticker":code,"name":code,"industry":"測試","price":100.0,"price_date":"2026-10-05","avg_volume_20d":2_000_000,"research":{},"horizons":{"short":{"ranking_score":score,"plan":plan,"entry_timing":{"score":86.0,"chase_risk":"低","action":"買點佳，可分批","reasons":[]},"position_guidance":{"percent":70,"label":"分批"},"forecast":{"estimate_available":True,"local_effective_n":45,"strategy":{"median":median,"mean":median,"p10":-0.04,"smoothed_positive_rate":0.66}}}}}
snap={"snapshot_id":"smoke","price_date":"2026-10-05","market":{"regime":"BULL"},"session":{"analysis_mode":"INTRADAY_BASELINE","twse_complete_date":"2026-10-05"},"stocks":[stock("1111.TW",90,0.12),stock("2222.TW",84,0.09)],"intraday_universe":[{"ticker":"1111.TW","name":"1111.TW","avg_volume_20d":2_000_000,"pre_score":60},{"ticker":"2222.TW","name":"2222.TW","avg_volume_20d":2_000_000,"pre_score":55}]}
df=pd.DataFrame([
    {"stock_id":"001","close":22000,"open":21900,"high":22100,"low":21800,"average_price":21950,"change_rate":0.7,"volume_ratio":1.2,"total_volume":1,"total_amount":1,"buy_price":21999,"sell_price":22001,"buy_volume":1,"sell_volume":1,"date":"2026-10-06"},
    {"stock_id":"1111","close":101.0,"open":100.2,"high":101.5,"low":99.8,"average_price":100.7,"change_rate":1.0,"volume_ratio":1.4,"total_volume":2000000,"total_amount":202000000,"buy_price":100.9,"sell_price":101.1,"buy_volume":600,"sell_volume":400,"date":"2026-10-06"},
    {"stock_id":"2222","close":103.5,"open":100.5,"high":104.0,"low":100.2,"average_price":102.0,"change_rate":3.5,"volume_ratio":1.7,"total_volume":2500000,"total_amount":258750000,"buy_price":103.4,"sell_price":103.6,"buy_volume":550,"sell_volume":450,"date":"2026-10-06"},
])
status=types.SimpleNamespace(available=True, reason="", source="synthetic", fetched_at=now.isoformat(), quote_date="2026-10-06")
app._realtime_batch=lambda snap: (["1111.TW","2222.TW"],df,status)
app._intraday_buy_render_once(snap,"short",top_n=2)
app._render_next_session(snap,"short",n=2)
print("Alpha Radar V16.6 decision views smoke: PASS")
