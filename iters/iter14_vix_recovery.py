"""iter14: VIX spike 후 회복 속도 분석.

가설: VIX > 30/40/50 발화 후 며칠 만에 정상 회귀하는지.
- 회복 N일 동안 SPY 평균 +%
- VIX peak vs SPY bottom 시차
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
    print("[iter14] VIX spike → SPY 회복 속도 분석")
    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]

    df = pd.concat([spy_close.rename("spy"), vix_close.rename("vix")], axis=1).dropna()

    # VIX > 30 발화 시점 (이전 5일 평균 < 25 → 진입)
    df["vix_5d_avg"] = df["vix"].rolling(5).mean()
    df["spike"] = (df["vix"] > 30) & (df["vix_5d_avg"].shift(1) < 25)
    spike_dates = df.index[df["spike"]]
    print(f"  VIX spike 발화 (이전 평온): {len(spike_dates)}회")

    # 각 spike 후 N일 SPY return
    print(f"\n=== VIX spike 후 SPY return 추이 ===")
    horizons = [5, 10, 21, 42, 63, 126, 180]
    print(f"  {'Spike date':12s}", end="")
    for h in horizons:
        print(f" {h}d", end="")
    print()
    spike_returns = {h: [] for h in horizons}
    for sd in spike_dates:
        try:
            pos = df.index.get_loc(sd)
            for h in horizons:
                if pos + h < len(df):
                    ret = (df["spy"].iloc[pos + h] / df["spy"].iloc[pos]) - 1
                    spike_returns[h].append(ret)
        except Exception:
            pass

    print(f"\n  각 horizon 평균 return:")
    print(f"  {'Horizon':10s} {'N':>4} {'Mean':>8} {'Win%':>6} {'Median':>8}")
    for h in horizons:
        rets = spike_returns[h]
        if rets:
            mean = np.mean(rets) * 100
            win = sum(1 for r in rets if r > 0) / len(rets) * 100
            median = np.median(rets) * 100
            marker = "🚀" if mean > 5 and win > 70 else ""
            print(f"  {h}d {len(rets):>5} {mean:>+7.2f}% {win:>5.1f}% {median:>+7.2f}% {marker}")

    # VIX 정상 회귀 시간
    df["vix_below_20"] = df["vix"] < 20
    print(f"\n=== VIX > 30 spike 후 < 20 회귀까지 평균 일수 ===")
    recovery_days = []
    for sd in spike_dates:
        try:
            pos = df.index.get_loc(sd)
            for d in range(1, 365):
                if pos + d >= len(df):
                    break
                if df["vix_below_20"].iloc[pos + d]:
                    recovery_days.append(d)
                    break
            else:
                recovery_days.append(365)
        except Exception:
            pass
    if recovery_days:
        print(f"  N: {len(recovery_days)}, 평균 {np.mean(recovery_days):.0f}일, median {np.median(recovery_days):.0f}일")
        print(f"  분포: 30일 이내 {sum(1 for d in recovery_days if d<=30)/len(recovery_days)*100:.0f}%, 90일 이내 {sum(1 for d in recovery_days if d<=90)/len(recovery_days)*100:.0f}%")

    out = {"n_spikes": int(len(spike_dates))}
    out_path = RESULTS_DIR / "iter14_vix_recovery.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
