"""iter10: KOSPI200 IV vs RV gap (한투 거래 가능).

가설: KOSPI200 옵션도 미국과 비슷한 vol risk premium.
- VKOSPI vs KOSPI200 RV gap 측정
- 한투에서 직접 거래 가능 (코스피200 미니 옵션)

데이터: yfinance ^KS200, ^KS11 (KOSPI), ^KOSDAQ
VKOSPI는 yfinance에 없음 → KS200 RV만 분석.
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


def realized_vol(prices, window):
    rets = prices.pct_change()
    return rets.rolling(window).std() * np.sqrt(252)


def main():
    print("[iter10] KOSPI200 vol analysis")

    ks200 = fetch_history("^KS200", period="20y", interval="1d")
    if ks200.empty:
        ks200 = fetch_history("^KS11", period="20y", interval="1d")  # KOSPI fallback
    if ks200.empty:
        print("  ❌ KOSPI 데이터 없음")
        return

    close = ks200["adj_close"] if "adj_close" in ks200.columns else ks200["close"]
    print(f"  KS200/KS11: {len(close)} days")

    rets = close.pct_change().dropna()
    rv_21 = realized_vol(close, 21) * 100
    rv_63 = realized_vol(close, 63) * 100

    df = pd.DataFrame({
        "close": close,
        "rv_21d": rv_21,
        "rv_63d": rv_63,
        "ret_1d": rets,
    }).dropna()

    print(f"\n=== KOSPI 통계 (20년) ===")
    print(f"  N: {len(df)}")
    print(f"  RV 21d 평균: {df['rv_21d'].mean():.2f}%")
    print(f"  RV 63d 평균: {df['rv_63d'].mean():.2f}%")
    print(f"  RV 21d std: {df['rv_21d'].std():.2f}%")
    print(f"  RV 21d max: {df['rv_21d'].max():.2f}%")

    # KS RV vs SPY RV 비교 (VIX → SPY와 비슷한 패턴)
    spy = fetch_history("SPY", period="20y", interval="1d")
    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    spy_rv21 = realized_vol(spy_close, 21) * 100
    print(f"\n=== KOSPI vs SPY RV 비교 ===")
    print(f"  KOSPI RV 21d 평균: {df['rv_21d'].mean():.2f}%")
    print(f"  SPY RV 21d 평균: {spy_rv21.mean():.2f}%")
    # 한국이 변동성 더 높음

    # RV regime 분석 (이전 high RV 후 future return)
    df["future_60d"] = df["close"].pct_change(60).shift(-60)
    df["future_180d"] = df["close"].pct_change(180).shift(-180)

    print(f"\n=== KOSPI RV regime → future return ===")
    bins = [0, 12, 18, 25, 35, 100]
    df["rv_bin"] = pd.cut(df["rv_21d"], bins=bins)
    for bin_ in df["rv_bin"].cat.categories:
        sub = df[df["rv_bin"] == bin_].dropna(subset=["future_180d"])
        if len(sub) < 30:
            continue
        avg60 = sub["future_60d"].mean() * 100 if not sub["future_60d"].isna().all() else 0
        avg180 = sub["future_180d"].mean() * 100
        win180 = (sub["future_180d"] > 0).sum() / len(sub) * 100
        marker = "🚀 LONG" if avg180 > 8 else ""
        print(f"  RV {str(bin_)}: N={len(sub)}, 60d {avg60:+.2f}%, 180d {avg180:+.2f}%, win {win180:.1f}%  {marker}")

    # Strong panic
    panic = df[df["rv_21d"] > 30].dropna(subset=["future_180d"])
    if len(panic) >= 20:
        avg = panic["future_180d"].mean() * 100
        win = (panic["future_180d"] > 0).sum() / len(panic) * 100
        print(f"\n  KS RV > 30 (panic) {len(panic)}일 → KS 180d {avg:+.1f}% (win {win:.1f}%)")

    out = {"n_days": len(df), "kospi_rv_21d_mean": float(df['rv_21d'].mean())}
    out_path = RESULTS_DIR / "iter10_kospi200.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
