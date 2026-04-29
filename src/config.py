"""프로젝트 디렉토리 + 옵션 universe."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in (DATA_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# 분석 대상 underlying (yfinance ticker)
INDEX_UNDERLYINGS = ["SPY", "QQQ", "IWM", "DIA"]
HIGH_LIQ_STOCKS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL"]
VIX_FAMILY = ["^VIX", "^VIX9D", "^VIX3M"]


# 비용 모델 (보수적)
COMMISSION_PER_CONTRACT = 0.65   # USD per contract (한투 미국 옵션 기준)
SLIPPAGE_BPS = 50                 # 0.5% slippage on premium
MIN_OPEN_INTEREST = 100           # OI 미만 마켓 무시
MAX_SPREAD_PCT = 5                # bid-ask spread > 5%면 skip
