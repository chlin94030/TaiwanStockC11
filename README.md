# Alpha Radar V16.6 — 1 分鐘動態盤中版

V16.6 的目標不是再改短／中／長線選股核心，而是把盤中執行層改成真正的「全市場發現 → 動態精算 → 可執行買點」。完整日線模型仍使用正式收盤資料，盤中每 60 秒只刷新即時價量，不把未完成日 K 寫回多年模型。

## 核心流程

### 盤後／盤前
1. 下載／增量更新完整日線快取。
2. 依流動性、歷史特徵、相似案例、基本面與法人資料計算最多 1000 檔完整模型。
3. 產生短線／中線／長線與下一交易日關注清單。
4. 同時保存 `intraday_universe`：所有通過最低價格與成交額條件的流動性股票，只含盤中掃描所需的輕量 metadata，不保存重型 DataFrame。

### 盤中每 60 秒
1. 對 `intraday_universe` 抓一次即時報價（預設最多 1800 檔）。
2. 用相對大盤、量能速度、日內位置、開盤後強弱、買賣壓力、成交金額等做輕量 discovery ranking。
3. 與昨晚短／中／長各 Top 80 的 daily core 合併。
4. 去重後最多 300 檔進入精算層。
5. 精算 Entry Score、ATR、剩餘預期報酬、Reward/Risk、歷史案例 confidence、買入區與部位。
6. 最後只把真正通過門檻者列為可買；若 0 檔，不硬湊推薦。

## 兩條合法進場路徑

### PULLBACK — 拉回承接
價格位於結構買入區，Selection / Entry / Expected Return / Positive Rate / Reward-Risk / Evidence 全部通過時，可顯示「拉回買點佳，可分批進場」或「拉回承接，可小量試單」。

### CONTINUATION — 強勢續攻
若價格已高於昨晚買入區，不再一律判成「等拉回」。必須同時符合更嚴格條件，例如：
- 標的分數與盤中強度足夠；
- 相對大盤明顯強；
- 量比在健康區間；
- 日內位置偏強；
- 距突破區不超過約 1.2 ATR；
- 單日漲幅未進入追高區；
- 保守剩餘報酬、Reward/Risk、歷史案例信心仍足夠。

通過時顯示「續攻確認，可小量試單」，且部位上限比一般拉回買點更低。近漲停、追價風險高、過度延伸的標的仍會被拒絕。

## 盤中診斷

每一檔不符合買進條件時，現在會保留 `failed_gates`，例如：
- 進場分數未達門檻
- 保守剩餘報酬不足
- 報酬／風險不足
- 歷史案例信心或樣本不足
- 續攻量能不在健康區間
- 相對大盤強度不足
- 距突破區過遠
- 追價風險高

頁面上方會統計精算池的主要淘汰原因，不再讓所有 WAIT 狀態都顯示同一句「等拉回」。

## 更新頻率

- Streamlit fragment：`60` 秒。
- 即時 quote cache TTL：`52` 秒。
- 每分鐘刷新即時層；多年日線／基本面／歷史案例不重算。
- 外部行情 provider 若無法一次完成全市場批次，會自動退回 daily core，不讓盤中頁整體失效。

## 主要檔案

- `app.py`：UI、60 秒 fragment、全市場／精算池統計。
- `radar_service.py`：完整日線模型與 `intraday_universe`。
- `intraday_engine.py`：全市場 quote、輕量 discovery、動態候選池、盤中 rerank。
- `recommendation_engine.py`：PULLBACK / CONTINUATION、Reward/Risk、failed gates。
- `strategy_config.py`：所有盤中與進場門檻集中管理。
- `return_first_model.py`：歷史報酬／相似案例核心，本版不改底層特徵架構。
- `test_architecture.py`：離線 regression tests。
- `benchmark_v16_6.py`：1500 檔盤中 discovery CPU benchmark。

## 部署

建議完整覆蓋目前 repository 中同名檔案後重新部署，不要只換 `app.py`。V16.6 改動了 `app.py`、`radar_service.py`、`intraday_engine.py`、`recommendation_engine.py`、`strategy_config.py` 等協作介面。

第一次部署後請手動按一次完整更新，讓新的 snapshot 建立 `intraday_universe`。舊 snapshot 即使沒有此欄位也有 fallback，但完整功能以重建 V16.6 snapshot 為準。

## 測試限制

本容器沒有安裝 Streamlit，也沒有用實際台股即時網路行情做 end-to-end provider 測試；因此 App 渲染採 Streamlit stub smoke test，資料來源以 synthetic/FakeStore integration 測試。真正 MIS／FinMind 每分鐘批次速度仍會受 Streamlit Cloud 與外部行情服務回應時間影響。
