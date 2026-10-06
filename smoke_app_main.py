"""Run app.main() through the no-data branch with a lightweight Streamlit stub."""
from __future__ import annotations
import importlib
import sys
import types

class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *args): return False

class _Progress:
    def progress(self, *args, **kwargs): return self
    def empty(self): return None

st = types.ModuleType("streamlit")
st.session_state = {}
st.secrets = {}
st.sidebar = _Ctx()

def _decorator(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    def wrap(fn): return fn
    return wrap

st.cache_data = _decorator
st.fragment = _decorator
st.set_page_config = lambda *a, **k: None
st.markdown = lambda *a, **k: None
st.caption = lambda *a, **k: None
st.info = lambda *a, **k: None
st.warning = lambda *a, **k: None
st.error = lambda *a, **k: None
st.success = lambda *a, **k: None
st.button = lambda *a, **k: False
st.progress = lambda *a, **k: _Progress()
st.selectbox = lambda label, options, index=0, **k: list(options)[index]
st.radio = lambda label, options, index=0, **k: list(options)[index]
st.checkbox = lambda *a, **k: False
st.number_input = lambda *a, value=0.0, **k: value
st.text_input = lambda *a, value="", **k: value
st.form = lambda *a, **k: _Ctx()
st.form_submit_button = lambda *a, **k: False
st.columns = lambda spec, **k: [_Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
st.expander = lambda *a, **k: _Ctx()
st.toggle = lambda *a, value=False, **k: value
st.plotly_chart = lambda *a, **k: None
st.select_slider = lambda label, options, value=None, **k: value
sys.modules["streamlit"] = st

app = importlib.import_module("app")
app.DATA_DIR = app.ROOT / "_smoke_data_no_snapshot"
app.main()
print("Alpha Radar V16.6 app main smoke: PASS")
