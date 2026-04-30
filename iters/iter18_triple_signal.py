"""iter18: VIX × VVIX × DD 삼중 신호 (ultra panic).

iter12 US+KR Both Panic + iter17 VIX×VVIX → 다음 단계.
가설: VIX>30 AND VVIX>120 AND SPY DD<-15% → 가장 정교한 panic bottom.
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
    print("[iter18] VIX × VVIX × DD triple panic")
    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    vvix = fetch_history("^VVIX", period="20y", interval="1d")
    if vvix.empty:
        print("  ❌ ^VVIX 없음")
        return

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    vvix_close = vvix["adj_close"] if "adj_close" in vvix.columns else vvix["close"]

    df = pd.concat([
        spy_close.rename("spy"),
        vix_close.rename("vix"),
        vvix_close.rename("vvix"),
    ], axis=1).dropna()
    df["spy_252h"] = df["spy"].rolling(252, min_periods=50).max()
    df["dd"] = df["spy"] / df["spy_252h"] - 1
    print(f"  데이터: {len(df)} days")

    for h in [5, 21, 63, 180]:
        df[f"future_{h}d"] = df["spy"].pct_change(h).shift(-h)

    print(f"\n=== Triple panic (180d) ===")
    combos = [
        (25, 100, -0.10),
        (25, 110, -0.10),
        (30, 110, -0.10),
        (30, 120, -0.10),
        (30, 120, -0.15),
        (30, 130, -0.15),
        (35, 130, -0.10),
        (35, 130, -0.15),
        (35, 130, -0.20),
        (35, 140, -0.15),
        (40, 130, -0.15),
        (40, 150, -0.15),
        (40, 150, -0.20),
    ]
    for v_t, vv_t, dd_t in combos:
        sub = df[(df["vix"] >= v_t) & (df["vvix"] >= vv_t) & (df["dd"] <= dd_t)].dropna(subset=["future_180d"])
        if len(sub) < 5:
            continue
        m = sub["future_180d"].mean() * 100
        w = (sub["future_180d"] > 0).sum() / len(sub) * 100
        s = sub["future_180d"].mean() / sub["future_180d"].std() * np.sqrt(2) if sub["future_180d"].std() > 0 else 0
        marker = "🚀" if w >= 95 and m > 20 else "✅" if w > 85 else ""
        print(f"  VIX>{v_t} VVIX>{vv_t} DD<{int(dd_t*100)}%: N={len(sub):>3} 180d {m:>+7.2f}% Win {w:>5.1f}% Sharpe {s:>5.2f} {marker}")

    print(f"\n=== Triple panic (21d) ===")
    for v_t, vv_t, dd_t in combos:
        sub = df[(df["vix"] >= v_t) & (df["vvix"] >= vv_t) & (df["dd"] <= dd_t)].dropna(subset=["future_21d"])
        if len(sub) < 5:
            continue
        m = sub["future_21d"].mean() * 100
        w = (sub["future_21d"] > 0).sum() / len(sub) * 100
        marker = "🚀" if w >= 95 and m > 5 else ""
        print(f"  VIX>{v_t} VVIX>{vv_t} DD<{int(dd_t*100)}%: N={len(sub):>3} 21d {m:>+6.2f}% Win {w:>5.1f}% {marker}")

    out_path = RESULTS_DIR / "iter18_triple_signal.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
