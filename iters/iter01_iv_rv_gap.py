"""iter01: IV vs RV gap baseline (Vol Risk Premium).

가설: ATM 옵션 implied volatility > realized volatility.
즉 시장이 변동성을 과대평가 → ATM straddle 매도 평균 +EV.

분석:
1. SPY/QQQ/AAPL 등 underlying의 historical RV (21d, 63d)
2. VIX/VXN을 IV proxy로 사용 (실제 ATM IV 대용)
3. (IV - RV) 평균 + 분포 → +EV 영역 발견

학술 정설: SPY VIX vs SPX 21d RV → 평균 +2~5%/년 spread.
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


def realized_vol_series(prices: pd.Series, window: int) -> pd.Series:
    rets = prices.pct_change()
    return rets.rolling(window).std() * np.sqrt(252)


def main():
    print("[iter01] IV vs RV gap baseline (Vol Risk Premium)")

    # SPY price + VIX (IV proxy)
    spy = fetch_history("SPY", period="10y", interval="1d")
    vix = fetch_history("^VIX", period="10y", interval="1d")

    if spy.empty or vix.empty:
        print("  ❌ 데이터 없음")
        return

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    print(f"  SPY: {len(spy_close)} days / VIX: {len(vix_close)} days")

    # SPY 21d, 63d RV (annualized %)
    rv_21 = realized_vol_series(spy_close, 21) * 100
    rv_63 = realized_vol_series(spy_close, 63) * 100

    df = pd.concat([
        spy_close.rename("spy"),
        vix_close.rename("vix"),
        rv_21.rename("rv_21d"),
        rv_63.rename("rv_63d"),
    ], axis=1).dropna()

    df["iv_minus_rv21"] = df["vix"] - df["rv_21d"]
    df["iv_minus_rv63"] = df["vix"] - df["rv_63d"]

    print(f"\n=== IV (VIX) vs SPY RV gap (10년) ===")
    print(f"  N: {len(df)}")
    print(f"  VIX 평균: {df['vix'].mean():.2f}%")
    print(f"  RV 21d 평균: {df['rv_21d'].mean():.2f}%")
    print(f"  IV - RV21 평균: {df['iv_minus_rv21'].mean():+.2f}%")
    print(f"  IV - RV63 평균: {df['iv_minus_rv63'].mean():+.2f}%")
    print(f"  IV > RV21 일자 비율: {(df['iv_minus_rv21'] > 0).sum() / len(df) * 100:.1f}%")
    print(f"  IV > RV21 by 5%+ 일자: {(df['iv_minus_rv21'] > 5).sum() / len(df) * 100:.1f}%")

    # IV / RV 비율
    df["iv_over_rv"] = df["vix"] / df["rv_21d"]
    print(f"\n  IV/RV 비율 평균: {df['iv_over_rv'].mean():.2f}x")
    print(f"  IV/RV > 1.5x 일자: {(df['iv_over_rv'] > 1.5).sum() / len(df) * 100:.1f}%")
    print(f"  IV/RV < 0.8x 일자 (RV이 IV 능가, vol spike): {(df['iv_over_rv'] < 0.8).sum() / len(df) * 100:.1f}%")

    # ATM straddle short 단순 백테스트
    # 매일 IV 매도 + N일 후 RV로 매수 → 차이만큼 수익 (근사)
    print(f"\n=== Vol Risk Premium 시뮬레이션 ===")
    for hold_days in [21, 42, 63]:
        # IV (VIX) 받고 N일 후 realized로 마감 가정
        # 단순: pnl = (vix - rv_future) / 100 (bps)
        rv_future = realized_vol_series(spy_close, hold_days).shift(-hold_days) * 100
        pnl_pct = (df["vix"] - rv_future).reindex(df.index)
        pnl_pct = pnl_pct.dropna()
        # 평균 vol risk premium 수익
        sharpe_approx = (pnl_pct.mean() / pnl_pct.std() * np.sqrt(252 / hold_days)) if pnl_pct.std() > 0 else 0
        print(f"  hold {hold_days}d: avg premium {pnl_pct.mean():+.2f}%, std {pnl_pct.std():.2f}%, approx Sharpe {sharpe_approx:.2f}")

    # 분포 분석
    print(f"\n=== IV-RV21 분포 (벨 형태) ===")
    bins = [-15, -10, -5, 0, 5, 10, 15, 30]
    df["iv_rv_bin"] = pd.cut(df["iv_minus_rv21"], bins=bins)
    grouped = df.groupby("iv_rv_bin", observed=True).size()
    for bin_, cnt in grouped.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct)
        print(f"  {str(bin_):20s} N={cnt:>5} ({pct:>5.1f}%) {bar}")

    out = {
        "n_days": len(df),
        "vix_mean": float(df["vix"].mean()),
        "rv_21d_mean": float(df["rv_21d"].mean()),
        "iv_rv21_mean": float(df["iv_minus_rv21"].mean()),
        "iv_over_rv_mean": float(df["iv_over_rv"].mean()),
        "iv_gt_rv_pct": float((df["iv_minus_rv21"] > 0).sum() / len(df) * 100),
    }
    out_path = RESULTS_DIR / "iter01_iv_rv_gap.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
