"""iter03: VXX/UVXY short 백테스트 (진짜 거래 가능 vol product).

iter02 발견: VIX contango 92.2% (학술 정설)
iter03: VXX/UVXY ETF 진짜 가격 시계열로 carry harvest 시뮬.

가설:
- VXX long-only는 평균 -50%/year (잘 알려짐)
- VXX short = 평균 +50%/year but tail risk
- DD stop으로 catastrophic 막음
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_history


def metrics(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return {"CAGR": 0, "Sharpe": 0, "MDD": 0}
    eq = (1 + pnl).cumprod()
    n_years = max((pnl.index[-1] - pnl.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1)
    sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)) if pnl.std(ddof=1) > 0 else 0
    cm = eq.cummax()
    return {"CAGR": cagr, "Sharpe": sharpe, "MDD": float((eq / cm - 1).min())}


def main():
    print("[iter03] VXX/UVXY/VIXY short carry harvest")

    # ETF 시도
    for ticker in ["VXX", "VIXY", "UVXY", "SVXY"]:
        df = fetch_history(ticker, period="10y", interval="1d")
        if df.empty:
            print(f"  ❌ {ticker} 데이터 없음")
            continue
        close = df["adj_close"] if "adj_close" in df.columns else df["close"]
        rets = close.pct_change().dropna()
        n = len(rets)
        if n < 100:
            continue

        # Long-only
        m_long = metrics(rets)
        # Short-only (-1x)
        m_short = metrics(-rets)

        print(f"\n=== {ticker} ({n} days, period {rets.index[0].date()} ~ {rets.index[-1].date()}) ===")
        print(f"  Long: Sharpe={m_long['Sharpe']:.2f} CAGR={m_long['CAGR']*100:+.1f}% MDD={m_long['MDD']*100:.1f}%")
        print(f"  Short (1x): Sharpe={m_short['Sharpe']:.2f} CAGR={m_short['CAGR']*100:+.1f}% MDD={m_short['MDD']*100:.1f}%")

    # VXX short + DD stop 시도
    print(f"\n=== VXX short + DD stop 변형 ===")
    df = fetch_history("VXX", period="10y", interval="1d")
    if df.empty:
        print(f"  ❌ VXX 없음")
        return
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    rets = -close.pct_change().dropna()  # short

    for dd_stop, lock_days in [(-0.10, 30), (-0.10, 63), (-0.15, 63), (-0.20, 90), (-0.25, 90)]:
        # 누적 252d DD 모니터, threshold 도달 시 lock_days 동안 0 비중
        pnl = pd.Series(0.0, index=rets.index)
        locked_until = -1
        for i in range(len(rets)):
            ts = rets.index[i]
            if i > locked_until:
                pnl.iloc[i] = rets.iloc[i]
                # DD 체크 (직전 252일)
                lookback = pnl.iloc[max(0, i-252):i+1]
                if len(lookback) > 30:
                    eq = (1 + lookback).cumprod()
                    cm = eq.cummax()
                    dd = float((eq.iloc[-1] / cm.iloc[-1] - 1)) if cm.iloc[-1] > 0 else 0
                    if dd < dd_stop:
                        locked_until = i + lock_days
        m = metrics(pnl)
        color = "🚀" if m['Sharpe'] > 1 and m['MDD'] > -0.30 else \
                "✅" if m['Sharpe'] > 0.5 else \
                "⚠️" if m['Sharpe'] > 0 else "❌"
        print(f"  {color} VXX short DD{dd_stop*100:.0f}% lock{lock_days}d: Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']*100:+.1f}% MDD={m['MDD']*100:.1f}%")

    # XIV/SVXY 같은 inverse ETF
    print(f"\n=== SVXY (-0.5x VXX) long ===")
    df_svxy = fetch_history("SVXY", period="10y", interval="1d")
    if not df_svxy.empty:
        close_svxy = df_svxy["adj_close"] if "adj_close" in df_svxy.columns else df_svxy["close"]
        rets_svxy = close_svxy.pct_change().dropna()
        m_svxy = metrics(rets_svxy)
        print(f"  Long SVXY: Sharpe={m_svxy['Sharpe']:.2f} CAGR={m_svxy['CAGR']*100:+.1f}% MDD={m_svxy['MDD']*100:.1f}%")

    out_path = RESULTS_DIR / "iter03_vxx_short.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
