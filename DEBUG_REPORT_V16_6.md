# Alpha Radar V16.6 — Debug / Regression Report

版本：`v16.6.0-dynamic-intraday`

## 本次修正的根因

V16.5 的盤中候選來自短／中／長各 Top 80 合併去重，因此實際可能只剩約 138 檔。這不是 bug，但會形成「昨晚候選池再排序」的視野限制；同時，價格高於昨晚買入區時，盤中決策層過度容易落入 `WAIT_PULLBACK`，而 UI 又沒有把真正未通過的 gate 明確顯示出來。

## V16.6 修正

1. **60 秒更新**：盤中 fragment 改為每 60 秒；quote cache TTL 52 秒。
2. **全市場輕量 discovery**：snapshot 新增 `intraday_universe`，盤中先抓所有流動性合格股票的即時報價，預設最多 1800 檔。
3. **動態精算池**：daily core（短中長各 Top 80，合併最多 220）＋盤中新轉強 discovery，合併後最多 300 檔進入完整執行判斷。
4. **不冒充完整模型**：若盤中新異動股票未進入 completed-day heavy model，只列 discovery-only，不直接升格成 BUY。
5. **Continuation Entry**：價格高於原買入區時，新增嚴格續攻路徑；必須同時通過相對強弱、量能、ATR 延伸、當日漲幅、剩餘報酬、R/R、案例信心等門檻。
6. **Failed gates**：每檔保留 `failed_gates` / `failed_gate_codes` / `primary_blocker`，UI 同時顯示精算池主要淘汰原因。
7. **追高保護不取消**：近漲停、chase risk 高、position=0、嚴重延伸仍不能因 Continuation 路徑被放行。
8. **選股核心不改**：短／中／長線 Selection 權重、基本面／法人／歷史案例架構保持不變。

## 已執行測試

- `python -m compileall -q .` → PASS
- `python test_architecture.py` → 全部 PASS
- `python smoke_app_import.py` → PASS
- `python smoke_app_main.py` → PASS
- `python smoke_decision_views.py` → PASS
- App ↔ radar_service API contract → PASS
- 盤中未完成日 K 隔離 → PASS
- 盤後 PENDING → FINAL → PASS
- Bull/Bear 只改部位、不改排名 → PASS
- 漲停仍可維持強標的，但不可追 → PASS
- full-market dynamic pool 可把原本不在 daily core 的盤中強股納入 detail pool → PASS
- Continuation Entry 可在不追高條件下通過 → PASS
- 過熱／追價標的即使盤中很強仍拒絕 → PASS
- failed-gate diagnostics → PASS
- UTF-8 / 中文 replacement char → PASS

## 效能測試

離線 synthetic 1500 檔 quote：

- full-market discovery + dynamic pool median：約 `0.55 s`
- 本次測試 detail pool：約 `220` 檔
- 設定刷新週期：`60 s`

這只代表本機 CPU 計算時間，不包含 MIS / FinMind 網路 latency。

三週期歷史 feature reuse benchmark：本次約 `1.32x` 比三次分開重算快。

## 尚未能在本容器證明的項目

- Streamlit server 真實啟動（容器未安裝 streamlit）。
- TWSE/TPEx MIS / FinMind 1500 檔即時批次在 Streamlit Cloud 的實際網路耗時與 rate limit。
- 真實盤中成交後報酬，仍需後續實盤／walk-forward 記錄，不可由 synthetic regression 宣稱。

因此 V16.6 可視為**程式架構與離線整合已通過的 production candidate**，但不是對未來報酬的保證。
