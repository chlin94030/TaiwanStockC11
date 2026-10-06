# V16.6 Validation Status

V16.6 主要改動是盤中 discovery / execution layer；短／中／長 completed-day selection core 與 V16.5 的 leakage-controlled 2026 walk-forward protocol 不變。

本版已完成：
- 60 秒全市場 quote refresh 架構測試
- dynamic daily-core + live-discovery candidate pool 測試
- pullback / continuation 兩路徑測試
- failed-gate diagnostics 測試
- anti-chase regression
- completed-day / intraday separation
- app/service API contract
- compile + smoke + FakeStore integration

真實 2026 walk-forward 仍須在可連外的 GitHub Actions / Streamlit Cloud 執行 `validate_2026_walkforward.py`；本地 synthetic regression 不冒充真實投資績效。
