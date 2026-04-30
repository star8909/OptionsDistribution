"""iter20: ULTRA panic = ALL 4 신호 동시 (VIX × VVIX × VIX9D/VIX × DD).

iter18 (VIX × VVIX × DD): Sharpe 8.86
iter19 (VIX9D/VIX × DD): Sharpe 8.50
iter20: 4개 동시 발화 = 가장 강한 confirmation.

가설: N 적어도 (5-10일) 100% Win, 큰 알파.
또는 너무 빡빡해서 신호 발화 적을 때 신뢰 어려움 점검.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_history, deduplicate_events, event_sharpe


def main():
    print("[iter20] ULTRA panic = VIX × VVIX × VIX9D/VIX × DD")
    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    vvix = fetch_history("^VVIX", period="20y", interval="1d")
    vix9d = fetch_history("^VIX9D", period="20y", interval="1d")

    if vvix.empty or vix9d.empty:
        print("  ❌ ^VVIX 또는 ^VIX9D 없음")
        return

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    vvix_close = vvix["adj_close"] if "adj_close" in vvix.columns else vvix["close"]
    vix9d_close = vix9d["adj_close"] if "adj_close" in vix9d.columns else vix9d["close"]

    df = pd.concat([
        spy_close.rename("spy"),
        vix_close.rename("vix"),
        vvix_close.rename("vvix"),
        vix9d_close.rename("vix9d"),
    ], axis=1).dropna()
    df["term"] = df["vix9d"] / df["vix"]
    df["spy_252h"] = df["spy"].rolling(252, min_periods=50).max()
    df["dd"] = df["spy"] / df["spy_252h"] - 1
    print(f"  데이터: {len(df)} days")

    for h in [5, 21, 63, 180]:
        df[f"future_{h}d"] = df["spy"].pct_change(h).shift(-h)

    combos = [
        # (vix_t, vvix_t, term_t, dd_t)
        (25, 100, 1.00, -0.05),
        (25, 100, 1.00, -0.10),
        (25, 110, 1.05, -0.10),
        (30, 110, 1.00, -0.10),
        (30, 110, 1.05, -0.10),
        (30, 120, 1.05, -0.15),
        (30, 120, 1.10, -0.15),
        (30, 130, 1.10, -0.15),
        (35, 130, 1.10, -0.15),
        (35, 130, 1.15, -0.15),
        (35, 140, 1.10, -0.20),
        (40, 130, 1.10, -0.15),
        (40, 150, 1.15, -0.20),
    ]

    print(f"\n=== ULTRA panic (180d, 독립 이벤트 기준) ===")
    print(f"  {'Signal':40s} {'RawN':>5} {'IndN':>5} {'Mean':>8} {'Win%':>6} {'Sharpe':>7}")
    for v_t, vv_t, term_t, dd_t in combos:
        signal = (df["vix"] >= v_t) & (df["vvix"] >= vv_t) & (df["term"] >= term_t) & (df["dd"] <= dd_t)
        dedup = deduplicate_events(signal, cooldown_days=180)
        raw_n = len(df[signal].dropna(subset=["future_180d"]))
        sub = df[dedup].dropna(subset=["future_180d"])
        if len(sub) < 1:
            continue
        m = sub["future_180d"].mean() * 100
        w = (sub["future_180d"] > 0).sum() / len(sub) * 100
        sh = event_sharpe(sub["future_180d"])
        s = f"{sh:.2f}" if not np.isnan(sh) else "N/A"
        marker = "🚀" if w >= 95 and m > 20 and not np.isnan(sh) else "✅" if w > 85 and not np.isnan(sh) else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  V>{v_t} VV>{vv_t} t>{term_t} DD<{int(dd_t*100)}%:  {raw_n:>5} {len(sub):>5} {m:>+7.2f}% {w:>5.1f}% {s:>7}{warning}  {marker}")

    print(f"\n=== ULTRA panic (21d, 독립 이벤트 기준) ===")
    print(f"  {'Signal':40s} {'RawN':>5} {'IndN':>5} {'Mean':>8} {'Win%':>6}")
    for v_t, vv_t, term_t, dd_t in combos:
        signal = (df["vix"] >= v_t) & (df["vvix"] >= vv_t) & (df["term"] >= term_t) & (df["dd"] <= dd_t)
        dedup = deduplicate_events(signal, cooldown_days=21)
        raw_n = len(df[signal].dropna(subset=["future_21d"]))
        sub = df[dedup].dropna(subset=["future_21d"])
        if len(sub) < 1:
            continue
        m = sub["future_21d"].mean() * 100
        w = (sub["future_21d"] > 0).sum() / len(sub) * 100
        marker = "🚀" if w >= 95 and m > 5 and len(sub) >= 10 else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  V>{v_t} VV>{vv_t} t>{term_t} DD<{int(dd_t*100)}%:  {raw_n:>5} {len(sub):>5} {m:>+6.2f}% {w:>5.1f}%  {marker}{warning}")

    out_path = RESULTS_DIR / "iter20_ultra_panic.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
