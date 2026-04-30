"""yfinance 옵션 chain 데이터 로더 (parquet 캐시).

yfinance.Ticker.options → expirations 리스트.
yfinance.Ticker.option_chain(date) → calls + puts DataFrame.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR

warnings.filterwarnings("ignore", category=FutureWarning)


def fetch_chain(ticker_symbol: str, expiration: str | None = None) -> dict:
    """옵션 chain 가져오기. expiration이 None이면 가장 가까운 만기.

    반환: {
        'underlying_price': float,
        'expiration': 'YYYY-MM-DD',
        'calls': DataFrame (strike, lastPrice, bid, ask, volume, openInterest, impliedVolatility),
        'puts':  DataFrame,
    }
    """
    import yfinance as yf
    t = yf.Ticker(ticker_symbol)
    expirations = t.options
    if not expirations:
        return {}
    if expiration is None:
        expiration = expirations[0]
    if expiration not in expirations:
        return {}
    chain = t.option_chain(expiration)

    # 현재가
    try:
        info = t.fast_info
        underlying = float(info.last_price) if hasattr(info, 'last_price') else float(t.info.get('regularMarketPrice', 0))
    except Exception:
        underlying = 0.0

    return {
        'underlying_price': underlying,
        'expiration': expiration,
        'calls': chain.calls,
        'puts': chain.puts,
    }


def fetch_history(ticker_symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """현물 가격 시계열 (cache 있음)."""
    safe = ticker_symbol.replace("^", "").replace("=", "_")
    cache_path = CACHE_DIR / f"history_{safe}_{interval}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    import yfinance as yf
    df = yf.download(ticker_symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"adj close": "adj_close"})
    df.to_parquet(cache_path)
    return df


def realized_vol(prices: pd.Series, window_days: int) -> float:
    """직전 N일 realized volatility (annualized %)."""
    if len(prices) < window_days + 1:
        return float('nan')
    rets = prices.pct_change().dropna().tail(window_days)
    return float(rets.std() * (252 ** 0.5))


def deduplicate_events(signal: pd.Series, cooldown_days: int = 0) -> pd.Series:
    """연속 패닉 날들을 독립 이벤트로 압축.

    이벤트 정의: signal이 False→True로 전환되는 첫 날만 선택.
    cooldown_days > 0이면 이벤트 후 N일간 추가 이벤트 억제 (독립성 보장).

    Args:
        signal: bool Series (True=패닉/신호 발생)
        cooldown_days: 이벤트 후 억제 기간 (0=연속 이벤트만 제거)

    Returns:
        bool Series — 독립 이벤트 시작일만 True

    사용 예:
        signal = (vix > 35) & (dd < -0.15)
        events = deduplicate_events(signal, cooldown_days=180)
        df_events = df[events].dropna(subset=["spy_180d"])

    주의:
        N_raw=18이 N_independent=2로 줄면 통계 신뢰도 없을 수 있음.
        N>=10 이상이어야 Sharpe 계산 신뢰 가능.
    """
    import pandas as pd
    signal = signal.fillna(False).astype(bool)
    result = pd.Series(False, index=signal.index)
    last_event_pos = -cooldown_days - 1

    for i, (dt, val) in enumerate(signal.items()):
        if not val:
            continue
        prev_val = signal.iloc[i - 1] if i > 0 else False
        in_cooldown = (cooldown_days > 0) and (i - last_event_pos <= cooldown_days)
        if not prev_val and not in_cooldown:
            result.loc[dt] = True
            last_event_pos = i

    return result


def event_sharpe(returns_180d: "pd.Series", annualize_factor: float = 2.0) -> float:
    """N개 이벤트의 180d return으로 Sharpe 계산.

    annualize_factor=sqrt(2) 가정: 180d 이벤트가 연간 2번.
    N<10이면 nan 반환 (신뢰 불가).
    """
    import numpy as np
    r = returns_180d.dropna()
    if len(r) < 10:
        return float('nan')
    return float(r.mean() / r.std() * (annualize_factor ** 0.5)) if r.std() > 0 else 0.0
