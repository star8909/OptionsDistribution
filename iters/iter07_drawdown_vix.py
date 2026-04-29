"""iter07: SPY drawdown × VIX regime joint signal.

iter05 발견: VIX > 30 시 SPY 180d +24%
iter07 가설: SPY 자체 drawdown + VIX 결합 시 더 강한 신호.

조건:
- SPY 252일 high 대비 -10%, -20%, -30% drawdown
- VIX level 결합
- 다음 90/180/365일 SPY return

학술: 1y high에서 -20%+ 떨어진 후 VIX > 30 = "panic bottom" — 평균 +30%+ 회귀.
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
    print("[iter07] SPY drawdown × VIX regime joint signal")
    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    if spy.empty or vix.empty:
        print("  ❌ 데이터 없음")
        return

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]

    df = pd.concat([spy_close.rename("spy"), vix_close.rename("vix")], axis=1).dropna()
    df["spy_252d_high"] = df["spy"].rolling(252, min_periods=50).max()
    df["dd_from_high"] = (df["spy"] / df["spy_252d_high"] - 1) * 100  # %

    for h in [60, 90, 180, 365]:
        df[f"spy_{h}d"] = df["spy"].pct_change(h).shift(-h)

    print(f"\n=== Drawdown × VIX joint table (180d future return) ===")
    print(f"  {'DD':>10} {'VIX':>10}    N    Mean    Win%    Max  Min")
    print(f"  {'-'*70}")

    for dd_low, dd_high in [(-100, -30), (-30, -20), (-20, -10), (-10, -5), (-5, 0)]:
        for v_low, v_high in [(0, 16), (16, 25), (25, 35), (35, 100)]:
            mask = (df["dd_from_high"] >= dd_low) & (df["dd_from_high"] < dd_high) & \
                   (df["vix"] >= v_low) & (df["vix"] < v_high)
            sub = df[mask].dropna(subset=["spy_180d"])
            if len(sub) < 20:
                continue
            mean_ret = sub["spy_180d"].mean() * 100
            win = (sub["spy_180d"] > 0).sum() / len(sub) * 100
            max_r = sub["spy_180d"].max() * 100
            min_r = sub["spy_180d"].min() * 100
            marker = "🚀 STRONG" if mean_ret > 15 and win > 80 else ""
            print(f"  {dd_low:>4}~{dd_high:<4} {v_low:>3}~{v_high:<3}  {len(sub):>4}  {mean_ret:>+6.1f}%  {win:>4.0f}%  {max_r:>+5.0f}%  {min_r:>+5.0f}% {marker}")

    # Best signal
    print(f"\n=== Strong panic bottom: DD < -20% AND VIX > 30 ===")
    panic = df[(df["dd_from_high"] < -20) & (df["vix"] > 30)].dropna(subset=["spy_180d"])
    if len(panic) >= 20:
        avg = panic["spy_180d"].mean() * 100
        win = (panic["spy_180d"] > 0).sum() / len(panic) * 100
        print(f"  N={len(panic)}, 180d 평균 {avg:+.1f}%, win {win:.1f}%")
        if avg > 15:
            print(f"  🏆 매우 강력!")

    print(f"\n=== Mild panic: DD < -10% AND VIX > 25 ===")
    mild = df[(df["dd_from_high"] < -10) & (df["vix"] > 25)].dropna(subset=["spy_180d"])
    if len(mild) >= 20:
        avg = mild["spy_180d"].mean() * 100
        win = (mild["spy_180d"] > 0).sum() / len(mild) * 100
        print(f"  N={len(mild)}, 180d 평균 {avg:+.1f}%, win {win:.1f}%")

    out = {"n_days": len(df)}
    out_path = RESULTS_DIR / "iter07_drawdown_vix.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
