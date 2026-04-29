"""iter11: KOSPI DD × RV joint signal (iter07 미국 패턴 한국 적용).

iter07 미국 (DD<-20% × VIX>30): +33% / win 100%
iter11 한국: DD<-20% × KS RV>30 시도.
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
    print("[iter11] KOSPI DD × RV joint signal")

    ks = fetch_history("^KS200", period="20y", interval="1d")
    if ks.empty:
        ks = fetch_history("^KS11", period="20y", interval="1d")
    close = ks["adj_close"] if "adj_close" in ks.columns else ks["close"]
    rv_21 = realized_vol(close, 21) * 100

    df = pd.DataFrame({
        "ks": close,
        "rv_21d": rv_21,
    }).dropna()
    df["ks_252d_high"] = df["ks"].rolling(252, min_periods=50).max()
    df["dd_from_high"] = (df["ks"] / df["ks_252d_high"] - 1) * 100

    for h in [60, 90, 180, 365]:
        df[f"future_{h}d"] = df["ks"].pct_change(h).shift(-h)

    print(f"\n=== KOSPI DD × RV joint table (180d future) ===")
    print(f"  {'DD':>10} {'RV':>8}    N    Mean   Win%   Max  Min")
    print(f"  {'-'*65}")
    for dd_low, dd_high in [(-100, -30), (-30, -20), (-20, -10), (-10, -5), (-5, 0)]:
        for rv_low, rv_high in [(0, 18), (18, 25), (25, 35), (35, 100)]:
            mask = (df["dd_from_high"] >= dd_low) & (df["dd_from_high"] < dd_high) & \
                   (df["rv_21d"] >= rv_low) & (df["rv_21d"] < rv_high)
            sub = df[mask].dropna(subset=["future_180d"])
            if len(sub) < 20:
                continue
            mean_ret = sub["future_180d"].mean() * 100
            win = (sub["future_180d"] > 0).sum() / len(sub) * 100
            max_r = sub["future_180d"].max() * 100
            min_r = sub["future_180d"].min() * 100
            marker = "🚀 STRONG" if mean_ret > 15 and win > 80 else ""
            print(f"  {dd_low:>4}~{dd_high:<4} {rv_low:>3}~{rv_high:<3} {len(sub):>4} {mean_ret:>+5.1f}% {win:>4.0f}%  {max_r:>+4.0f}%  {min_r:>+4.0f}% {marker}")

    panic = df[(df["dd_from_high"] < -20) & (df["rv_21d"] > 30)].dropna(subset=["future_180d"])
    if len(panic) >= 20:
        avg = panic["future_180d"].mean() * 100
        win = (panic["future_180d"] > 0).sum() / len(panic) * 100
        print(f"\n  KOSPI panic bottom (DD<-20% AND RV>30) {len(panic)}일 → 180d {avg:+.1f}%, win {win:.1f}%")

    out = {"n_panic": int(len(panic)) if len(panic) >= 20 else 0}
    out_path = RESULTS_DIR / "iter11_kospi_dd_joint.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
