"""Import-smoke app.py without requiring Streamlit to be installed."""
from __future__ import annotations
import importlib
import sys
import types

st = types.ModuleType("streamlit")

def _decorator(*args, **kwargs):
    def wrap(fn):
        return fn
    # Support @st.cache_data without parentheses as well.
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return wrap

st.cache_data = _decorator
st.fragment = _decorator
st.secrets = {}
sys.modules["streamlit"] = st

app = importlib.import_module("app")
assert hasattr(app, "main")
assert hasattr(app, "_intraday_buy_render_once")
assert hasattr(app, "_render_next_session")
print("Alpha Radar V16.6 app import smoke: PASS")
