"""iter08: VIX9D > VIX (단기 panic) signal → SPY future return.

가설: VIX9D / VIX > 1.10 = 단기 vol > 한달 vol = 즉시 panic.
이 신호 후 SPY 단기 평균회귀 (mean reversion).

iter05 (VIX > 30) + iter07 (DD × VIX) 보다 더 frequent + faster signal.
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
    print("[iter08] VIX9D > VIX 단기 panic signal")

    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    vix9d = fetch_history("^VIX9D", period="20y", interval="1d")
    if spy.empty or vix.empty or vix9d.empty:
        print("  ❌ 데이터 없음")
        return

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    vix9d_close = vix9d["adj_close"] if "adj_close" in vix9d.columns else vix9d["close"]

    df = pd.concat([
        spy_close.rename("spy"),
        vix_close.rename("vix"),
        vix9d_close.rename("vix9d"),
    ], axis=1).dropna()

    df["ratio"] = df["vix9d"] / df["vix"]
    print(f"  N: {len(df)}, ratio 평균 {df['ratio'].mean():.3f}")

    for h in [5, 10, 21, 42, 63]:
        df[f"spy_{h}d"] = df["spy"].pct_change(h).shift(-h)

    # Ratio bucket
    bins = [0, 0.85, 0.95, 1.0, 1.05, 1.10, 1.20, 2.0]
    df["ratio_bin"] = pd.cut(df["ratio"], bins=bins)

    print(f"\n=== VIX9D / VIX ratio → SPY future return ===")
    print(f"  {'Ratio':15s} {'N':>5}  {'5d':>7} {'10d':>7} {'21d':>7} {'42d':>7} {'63d':>7} W21d")
    for bin_ in df["ratio_bin"].cat.categories:
        sub = df[df["ratio_bin"] == bin_].dropna(subset=["spy_21d"])
        if len(sub) < 30:
            continue
        r5 = sub["spy_5d"].mean()*100 if not sub["spy_5d"].isna().all() else 0
        r10 = sub["spy_10d"].mean()*100 if not sub["spy_10d"].isna().all() else 0
        r21 = sub["spy_21d"].mean()*100 if not sub["spy_21d"].isna().all() else 0
        r42 = sub["spy_42d"].mean()*100 if not sub["spy_42d"].isna().all() else 0
        r63 = sub["spy_63d"].mean()*100 if not sub["spy_63d"].isna().all() else 0
        win21 = (sub["spy_21d"] > 0).sum() / len(sub) * 100
        marker = "🚀" if r21 > 3 and win21 > 70 else ""
        print(f"  {str(bin_):15s} {len(sub):>5} {r5:>+6.2f}% {r10:>+6.2f}% {r21:>+6.2f}% {r42:>+6.2f}% {r63:>+6.2f}% {win21:>4.0f}% {marker}")

    # Strong panic signal
    strong = df[df["ratio"] > 1.10].dropna(subset=["spy_21d"])
    if len(strong) >= 20:
        avg = strong["spy_21d"].mean() * 100
        win = (strong["spy_21d"] > 0).sum() / len(strong) * 100
        print(f"\n  Ratio > 1.10 (단기 panic) {len(strong)}일 → SPY 21d {avg:+.2f}%, win {win:.1f}%")
        if avg > 2:
            print(f"  🏆 mean reversion 확인!")

    very = df[df["ratio"] > 1.20].dropna(subset=["spy_21d"])
    if len(very) >= 10:
        avg = very["spy_21d"].mean() * 100
        win = (very["spy_21d"] > 0).sum() / len(very) * 100
        print(f"  Ratio > 1.20 (severe panic) {len(very)}일 → SPY 21d {avg:+.2f}%, win {win:.1f}%")

    out = {"n_days": len(df)}
    out_path = RESULTS_DIR / "iter08_vix9d_panic.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
