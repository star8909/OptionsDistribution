"""iter19: VIX9D > VIX (term inversion) + DD = 단기 term-structure panic.

VIX9D > VIX = 단기 panic이 장기보다 더 심함 (term inversion).
Coronavirus, Lehman 같은 시장 conv 시점 발생.

가설: 이런 inversion + DD<-10% 동시 → 진짜 단기 bottom.
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
    print("[iter19] VIX9D > VIX term inversion + DD")
    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    vix9d = fetch_history("^VIX9D", period="20y", interval="1d")
    if vix9d.empty:
        print("  ❌ ^VIX9D 없음")
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
    df["spy_252h"] = df["spy"].rolling(252, min_periods=50).max()
    df["dd"] = df["spy"] / df["spy_252h"] - 1
    print(f"  데이터: {len(df)} days, ratio range {df['ratio'].min():.2f}~{df['ratio'].max():.2f}")

    for h in [5, 21, 63, 180]:
        df[f"future_{h}d"] = df["spy"].pct_change(h).shift(-h)

    print(f"\n=== Term inversion (ratio > 1.X) 단독 (180d, 독립 이벤트 기준) ===")
    print(f"  {'Signal':15s} {'RawN':>6} {'IndepN':>7} {'Mean':>8} {'Win%':>6} {'Sharpe':>7}")
    for r_t in [1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40]:
        signal = df["ratio"] >= r_t
        dedup = deduplicate_events(signal, cooldown_days=180)
        raw_sub = df[signal].dropna(subset=["future_180d"])
        sub = df[dedup].dropna(subset=["future_180d"])
        if len(sub) < 3:
            continue
        m = sub["future_180d"].mean() * 100
        w = (sub["future_180d"] > 0).sum() / len(sub) * 100
        sh = event_sharpe(sub["future_180d"])
        s = f"{sh:.2f}" if not np.isnan(sh) else "N/A"
        marker = "🚀" if w > 80 and m > 10 and not np.isnan(sh) else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  ratio>{r_t:.2f}      {len(raw_sub):>6} {len(sub):>7} {m:>+7.2f}% {w:>5.1f}% {s:>7}{warning}  {marker}")

    combos = [
        (1.05, -0.10), (1.10, -0.10), (1.10, -0.15),
        (1.15, -0.10), (1.15, -0.15), (1.20, -0.10),
        (1.20, -0.15), (1.20, -0.20), (1.30, -0.15),
    ]

    print(f"\n=== Term inversion + DD (180d, 독립 이벤트 기준) ===")
    print(f"  {'Signal':25s} {'RawN':>5} {'IndN':>5} {'Mean':>8} {'Win%':>6} {'Sharpe':>7}")
    for r_t, dd_t in combos:
        signal = (df["ratio"] >= r_t) & (df["dd"] <= dd_t)
        dedup = deduplicate_events(signal, cooldown_days=180)
        raw_n = len(df[signal].dropna(subset=["future_180d"]))
        sub = df[dedup].dropna(subset=["future_180d"])
        if len(sub) < 2:
            continue
        m = sub["future_180d"].mean() * 100
        w = (sub["future_180d"] > 0).sum() / len(sub) * 100
        sh = event_sharpe(sub["future_180d"])
        s = f"{sh:.2f}" if not np.isnan(sh) else "N/A"
        marker = "🚀" if w >= 95 and m > 15 and not np.isnan(sh) else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  ratio>{r_t} DD<{int(dd_t*100)}%:   {raw_n:>5} {len(sub):>5} {m:>+7.2f}% {w:>5.1f}% {s:>7}{warning}  {marker}")

    print(f"\n=== Term inversion + DD (21d, 독립 이벤트 기준) ===")
    print(f"  {'Signal':25s} {'RawN':>5} {'IndN':>5} {'Mean':>8} {'Win%':>6}")
    for r_t, dd_t in combos:
        signal = (df["ratio"] >= r_t) & (df["dd"] <= dd_t)
        dedup = deduplicate_events(signal, cooldown_days=21)
        raw_n = len(df[signal].dropna(subset=["future_21d"]))
        sub = df[dedup].dropna(subset=["future_21d"])
        if len(sub) < 2:
            continue
        m = sub["future_21d"].mean() * 100
        w = (sub["future_21d"] > 0).sum() / len(sub) * 100
        marker = "🚀" if w >= 95 and m > 5 and len(sub) >= 10 else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  ratio>{r_t} DD<{int(dd_t*100)}%:   {raw_n:>5} {len(sub):>5} {m:>+6.2f}% {w:>5.1f}%  {marker}{warning}")

    out_path = RESULTS_DIR / "iter19_term_inversion.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
