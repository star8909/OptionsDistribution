"""iter06: Put-Call Ratio (PCC) → SPY future return.

가설: PCC 극단 = contrarian signal.
- PCC 매우 높음 (1.5+) = put 매수 폭증 = 공포 극단 → SPY long +EV
- PCC 매우 낮음 (0.5-) = call 매수 폭증 = 탐욕 극단 → SPY short or hedge

CBOE PCC index (^PCC) 시계열로 분석.
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
    print("[iter06] Put-Call Ratio → SPY future return")
    pcc = fetch_history("^PCC", period="20y", interval="1d")  # CBOE Put-Call Ratio
    if pcc.empty:
        # PCC 없으면 ^CPC (CBOE Put-Call Total) 시도
        pcc = fetch_history("^CPC", period="20y", interval="1d")
    if pcc.empty:
        print("  ❌ PCC 데이터 없음")
        return

    spy = fetch_history("SPY", period="20y", interval="1d")
    pcc_close = pcc["adj_close"] if "adj_close" in pcc.columns else pcc["close"]
    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]

    df = pd.concat([pcc_close.rename("pcc"), spy_close.rename("spy")], axis=1).dropna()
    print(f"  PCC: {len(df)} days, mean {df['pcc'].mean():.2f}")
    for h in [10, 30, 60, 90]:
        df[f"spy_{h}d"] = df["spy"].pct_change(h).shift(-h)

    # PCC bucket vs future return
    bins = [0, 0.6, 0.8, 1.0, 1.2, 1.5, 5.0]
    df["pcc_bin"] = pd.cut(df["pcc"], bins=bins)

    print(f"\n=== PCC level → SPY future return ===")
    print(f"  {'PCC bin':15s} {'N':>5}  {'10d':>8} {'30d':>8} {'60d':>8} {'90d':>8} Win90d")
    for bin_ in df["pcc_bin"].cat.categories:
        sub = df[df["pcc_bin"] == bin_]
        if len(sub) < 30:
            continue
        r10 = sub["spy_10d"].mean()*100 if not sub["spy_10d"].isna().all() else 0
        r30 = sub["spy_30d"].mean()*100 if not sub["spy_30d"].isna().all() else 0
        r60 = sub["spy_60d"].mean()*100 if not sub["spy_60d"].isna().all() else 0
        r90 = sub["spy_90d"].mean()*100 if not sub["spy_90d"].isna().all() else 0
        win90 = (sub["spy_90d"] > 0).sum() / sub["spy_90d"].dropna().shape[0] * 100 if sub["spy_90d"].dropna().shape[0] else 0
        marker = "🚀 contrarian LONG" if bin_.left >= 1.2 and r90 > 5 else ""
        print(f"  {str(bin_):15s} {len(sub):>5} {r10:>+7.2f}% {r30:>+7.2f}% {r60:>+7.2f}% {r90:>+7.2f}% {win90:>5.1f}%  {marker}")

    # PCC > 1.5 신호
    extreme = df[df["pcc"] > 1.5].dropna(subset=["spy_90d"])
    if len(extreme) >= 20:
        avg = extreme["spy_90d"].mean() * 100
        win = (extreme["spy_90d"] > 0).sum() / len(extreme) * 100
        print(f"\n  PCC > 1.5 (panic) {len(extreme)}일 → SPY 90d {avg:+.1f}% (win {win:.1f}%)")

    out = {"n_days": len(df), "pcc_mean": float(df["pcc"].mean())}
    out_path = RESULTS_DIR / "iter06_pcc_signal.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
