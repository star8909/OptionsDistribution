# CLAUDE.md

이 파일은 Claude Code (claude.ai/code) 가 이 저장소에서 작업할 때 참고하는 가이드.

## 프로젝트 목적

**옵션 (Options)** 시장을 **확률분포로 다루는** 분석/베팅 플레이그라운드.
strike price = market-implied probability of expiration ITM. 시장 implied prob ≠ 진짜 확률 영역 찾아 +EV 베팅.

자매 프로젝트 (모두 같은 방법론):
- `c:\Projects\ProbabilityDistribution` — 암호화폐
- `c:\Projects\EquityDistribution` — 한투 해외주식
- `c:\Projects\FuturesDistribution` — 한투 해외선물옵션
- `c:\Projects\polymarket-analyze` — Polymarket 예측시장
- `c:\Projects\OptionsDistribution` — **옵션** (이 프로젝트)

## 핵심 가설

**옵션 가격 = 행사가에서 만기 도달할 시장 implied probability.**
Polymarket과 거의 같은 구조 — 차이점: 옵션은 vanilla (Yes/No 아님), 변동성/시간 차원 추가.

작동하는 이유 — 옵션 시장에 비합리성 박혀있음:

1. **Vol Risk Premium** — IV (implied vol) > RV (realized vol) 거의 항상. SPY ATM 옵션 매도 시 평균 +EV (학술 정설)
2. **Vol Smile/Skew** — OTM put이 OTM call보다 비싸 (crash 공포). 단, calibration 분석으로 reasonable한지 측정
3. **Term structure** — 단기 vs 장기 IV 차이 (contango/backwardation)
4. **0DTE 효과** — 만기 당일 옵션은 strike별 가격 = breakeven probability. calibration 가능
5. **Earnings/event 직전 IV crush** — 발표 직후 IV 급락 (예측 가능)

## 데이터 출처

### 1차 (무료, 즉시 가능)
- **Yahoo Finance** (`yfinance.Ticker.options`) — SPY/QQQ/AAPL 옵션 chain (실시간 + 단기 history)
- **CBOE** — 무료 historical options data (다운로드 가능)
- **FRED** — VIX, VIX9D, VIX3M term structure

### 2차 (한투 retail)
- **KOSPI200 옵션** — 한투 API 거래 가능
- **VKOSPI** — 한국 implied vol index

### 3차 (가상화폐 옵션)
- **Deribit API** — BTC/ETH 옵션 무료 API (자매 ProbabilityDistribution과 시너지)

### 4차 (대용량 onchain)
- **Lyra (Optimism)** — onchain crypto options
- **Opyn (Ethereum)** — onchain options

## 폴더 구조

```
OptionsDistribution/
├── src/                       # 코어 (data_loader, config)
├── iters/                     # iter*.py 분석 스크립트
├── tools/                     # fetch_chain.py, fetch_history.py
├── results/                   # iter*.json (commit, 재현성)
├── logs/                      # iter*.log
├── dashboard/                 # 챔피언 대시보드
├── archive/                   # 옛 파일
├── data/                      # 캐시 (gitignored)
│   └── cache/                 # parquet 옵션 chain
├── CLAUDE.md / README.md / requirements.txt
```

## 분석 전략 (iter 진행 순서)

### Phase 1 — Strike Calibration (가장 강력)
0DTE / 단기 (5d 미만) 옵션의 strike price → expiration ITM 확률 곡선 측정.

```
SPY $480 strike (현재가 $485, 만기 1d):
- 시장 implied prob = call price / max payout (간단 근사)
- 실제 ITM 마감 비율 측정
- 비싸면 매도 +EV
```

### Phase 2 — IV vs RV Gap (Vol Risk Premium)
ATM 옵션 implied vol vs 다음 N일 실제 변동성 비교.
- IV - RV 평균 +2~3% (학술 정설)
- ATM straddle short → +EV but tail risk

### Phase 3 — Term Structure Carry
- Front-month vs back-month IV
- VIX curve contango harvest
- 학술: 평균 -0.5~1%/month carry

### Phase 4 — Skew Anomaly
- 25-delta put vs call price
- Put 과대평가 시 → 매도 +EV
- 단, 6시그마 위험 (LTCM 교훈)

### Phase 5 — Event-Driven (Earnings/FOMC)
- 발표 직전 IV vs actual move calibration
- 보통 IV implied move > realized → straddle short +EV
- 단, big surprise 위험

## 비용 모델 (옵션 vs 다른 자산)

| 항목 | 미국 옵션 (한투) | KOSPI200 옵션 |
|------|---------------|--------------|
| Commission | ~$0.65/계약 | 0.025% + 1pt |
| Spread | 1~2 ticks (ATM 작음, OTM 큼) | 1~2 ticks |
| Slippage | depth depends | 한국은 큼 |
| 합산 baseline | ~$1.5/계약 | ~10pt/계약 |

**규칙**: 백테스트 비용 보수적 — 옵션은 small premium 베팅 시 cost ratio 높음.

## 봇/전략 계층 (계획)

```
src/strategies.py
    BaseStrategy
    ├── StrikeCalibrationStrategy   # 시장 vs 진짜 ITM 확률
    ├── VolRiskPremiumStrategy      # IV vs RV gap (ATM straddle short)
    ├── TermStructureStrategy       # VIX contango harvest
    ├── SkewArbStrategy             # OTM put 과대평가
    └── EventDrivenStrategy         # earnings/FOMC IV crush
```

## Critical conventions

- **항상 `PYTHONIOENCODING=utf-8`** — Windows 한글
- **Tail risk 항상 인지** — 옵션 매도 = 무한 손실 가능성. 항상 long protection 또는 명확한 정지 손실
- **Open interest 필터** — OI < 100 인 옵션은 거래 X (slippage 폭증)
- **Bid/ask spread > 5%면 skip** — 유동성 부족 시 진짜 수익 못 봄
- **만기 1주일 미만 옵션** — 0DTE 분석 강력하지만 gamma risk 폭발
- **Walk-forward 만 신뢰** — 단일 기간 calibration은 false signal

## 실거래 안전 규칙 (한투 옵션 적용 시)

1. **Paper trading 4주 이상** — 백테스트 vs 실거래 갭
2. **Max position 5% per trade** — 한 옵션에 자본 5% 이하
3. **Long protection 의무** — 매도만 하면 tail risk 무한
4. **Greeks 모니터** — delta/gamma/theta/vega 항상 추적
5. **만기 직전 자동 청산** — 0DTE는 만기 1시간 전 강제 청산
6. **자동화 필수** — 옵션은 수동으로 못 다룸 (가격 변화 빠름)

## Iteration history

iter 번호 진행:
- iter01: SPY strike calibration baseline (0DTE)
- iter02: IV vs RV gap (ATM straddle 분석)
- iter03: VIX term structure carry
- iter04: Skew anomaly (25-delta put/call)
- iter05+: Event-driven, KOSPI200, Deribit
