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
