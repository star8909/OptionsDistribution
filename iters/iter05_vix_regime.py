"""iter05: VIX regime → SPY future return calibration.

가설: 높은 VIX 후 SPY 평균 회귀 (mean reversion 정설).
VIX > 30 후 6개월 SPY 평균 +10~15% (학술).

분석:
1. VIX level별 (10, 15, 20, 25, 30, 40+) bucket
2. 다음 30/60/90/180일 SPY return
3. Calibration 곡선 (VIX → SPY future drift)
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


def main():
    print("[iter05] VIX regime → SPY future return calibration")

    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    if spy.empty or vix.empty:
        print("  ❌ 데이터 없음")
        return

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]

    df = pd.concat([
        spy_close.rename("spy"),
        vix_close.rename("vix"),
    ], axis=1).dropna()
    print(f"  N days: {len(df)}")

    # Future returns
    for horizon in [30, 60, 90, 180]:
        df[f"spy_{horizon}d"] = df["spy"].pct_change(horizon).shift(-horizon)

    # VIX bucket별 future returns
    bins = [0, 12, 16, 20, 25, 30, 40, 100]
    df["vix_bin"] = pd.cut(df["vix"], bins=bins, include_lowest=True)

    print(f"\n=== VIX level → SPY future return ===")
    print(f"  {'VIX bin':15s} {'N':>5}  {'30d':>8} {'60d':>8} {'90d':>8} {'180d':>8}")
    print(f"  {'-'*60}")

    for bin_ in df["vix_bin"].cat.categories:
        sub = df[df["vix_bin"] == bin_]
        n = len(sub)
        if n < 30:
            continue
        ret_30 = sub["spy_30d"].mean() * 100 if not sub["spy_30d"].isna().all() else 0
        ret_60 = sub["spy_60d"].mean() * 100 if not sub["spy_60d"].isna().all() else 0
        ret_90 = sub["spy_90d"].mean() * 100 if not sub["spy_90d"].isna().all() else 0
        ret_180 = sub["spy_180d"].mean() * 100 if not sub["spy_180d"].isna().all() else 0
        marker = ""
        if ret_180 > 8:
            marker = "🚀 LONG SPY"
        elif ret_180 < 0:
            marker = "⚠️ SHORT SPY"
        print(f"  {str(bin_):15s} {n:>5} {ret_30:>+7.2f}% {ret_60:>+7.2f}% {ret_90:>+7.2f}% {ret_180:>+7.2f}%  {marker}")

    # Win rate (양수 future return)
    print(f"\n=== VIX level → SPY 180d win rate ===")
    print(f"  {'VIX bin':15s} {'N':>5}  Win%  Avg")
    for bin_ in df["vix_bin"].cat.categories:
        sub = df[df["vix_bin"] == bin_].dropna(subset=["spy_180d"])
        if len(sub) < 30:
            continue
        win = (sub["spy_180d"] > 0).sum() / len(sub) * 100
        avg = sub["spy_180d"].mean() * 100
        print(f"  {str(bin_):15s} {len(sub):>5}  {win:>5.1f}%  {avg:>+5.1f}%")

    # Strategy: VIX > 30 시 SPY long, VIX < 12 시 cash
    print(f"\n=== Simple strategy: VIX > 30 시 SPY long ===")
    panic_signals = df[df["vix"] > 30].dropna(subset=["spy_180d"])
    if len(panic_signals) >= 50:
        avg_ret = panic_signals["spy_180d"].mean() * 100
        win = (panic_signals["spy_180d"] > 0).sum() / len(panic_signals) * 100
        print(f"  VIX > 30 신호 {len(panic_signals)}일 → 180d SPY {avg_ret:+.1f}% (win {win:.1f}%)")
        if avg_ret > 8:
            print(f"  🏆 강력! VIX > 30 시 SPY 180d 평균 +{avg_ret:.0f}%")

    # 매우 calm (VIX < 12) 시 — over-confident
    calm = df[df["vix"] < 12].dropna(subset=["spy_180d"])
    if len(calm) >= 50:
        avg_ret = calm["spy_180d"].mean() * 100
        print(f"  VIX < 12 신호 {len(calm)}일 → 180d SPY {avg_ret:+.1f}%")

    out = {"n_days": len(df)}
    out_path = RESULTS_DIR / "iter05_vix_regime.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
