"""iter15: 단순 retail strategy 종합 비교.

iter05/07/11/12/13 외에 추가:
- VIX > X 시 SPY long 다양한 X
- VIX < X 시 SPY short (VIX bottom = top of bull market)
- VIX9D/VIX 비율 신호
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
    print("[iter15] VIX 다양한 threshold 종합 비교")
    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]

    df = pd.concat([spy_close.rename("spy"), vix_close.rename("vix")], axis=1).dropna()
    for h in [21, 63, 180, 365]:
        df[f"future_{h}d"] = df["spy"].pct_change(h).shift(-h)

    print(f"\n=== VIX Threshold 종합 (180d future, 독립 이벤트 기준) ===")
    print(f"  {'Threshold':15s} {'RawN':>6} {'IndepN':>7} {'Mean':>8} {'Win%':>6} {'Sharpe':>7}")
    for thr in [15, 18, 20, 22, 25, 27, 30, 35, 40, 50]:
        signal = df["vix"] >= thr
        dedup = deduplicate_events(signal, cooldown_days=180)
        raw_sub = df[signal].dropna(subset=["future_180d"])
        sub = df[dedup].dropna(subset=["future_180d"])
        if len(sub) < 5:
            continue
        m = sub["future_180d"].mean() * 100
        w = (sub["future_180d"] > 0).sum() / len(sub) * 100
        sh = event_sharpe(sub["future_180d"])
        s = f"{sh:.2f}" if not np.isnan(sh) else "N/A"
        marker = "🚀" if w > 80 and m > 10 and not np.isnan(sh) else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  VIX > {thr:<5}  {len(raw_sub):>6} {len(sub):>7} {m:>+7.2f}% {w:>5.1f}% {s:>7}{warning}  {marker}")

    print(f"\n=== VIX 낮은 threshold 후 SPY (over-confidence, 독립 이벤트 기준) ===")
    print(f"  {'Threshold':15s} {'RawN':>6} {'IndepN':>7} {'180d mean':>10} {'Win%':>6}")
    for thr in [10, 12, 14, 16]:
        signal = df["vix"] < thr
        dedup = deduplicate_events(signal, cooldown_days=180)
        raw_sub = df[signal].dropna(subset=["future_180d"])
        sub = df[dedup].dropna(subset=["future_180d"])
        if len(sub) < 5:
            continue
        m = sub["future_180d"].mean() * 100
        w = (sub["future_180d"] > 0).sum() / len(sub) * 100
        marker = "⚠️ contrarian" if m < 5 else ""
        warning = " ⚠️ N부족" if len(sub) < 10 else ""
        print(f"  VIX < {thr:<5}  {len(raw_sub):>6} {len(sub):>7} {m:>+9.2f}% {w:>5.1f}%  {marker}{warning}")

    out_path = RESULTS_DIR / "iter15_simple_strategies.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
