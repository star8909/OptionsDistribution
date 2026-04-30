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
from src.data_loader import fetch_history, deduplicate_events, event_sharpe


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

    print(f"\n=== Triple panic (180d, 독립 이벤트 기준) ===")
    print(f"  {'Signal':35s} {'RawN':>5} {'IndN':>5} {'Mean':>8} {'Win%':>6} {'Sharpe':>7}")
    for v_t, vv_t, dd_t in combos:
        signal = (df["vix"] >= v_t) & (df["vvix"] >= vv_t) & (df["dd"] <= dd_t)
        dedup = deduplicate_events(signal, cooldown_days=180)
        raw_n = len(df[signal].dropna(subset=["future_180d"]))
        sub = df[dedup].dropna(subset=["future_180d"])
        if len(sub) < 2:
            continue
        m = sub["future_180d"].mean() * 100
        w = (sub["future_180d"] > 0).sum() / len(sub) * 100
        sh = event_sharpe(sub["future_180d"])
        s = f"{sh:.2f}" if not np.isnan(sh) else "N/A"
        marker = "🚀" if w >= 95 and m > 20 and not np.isnan(sh) else "✅" if w > 85 and not np.isnan(sh) else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  VIX>{v_t} VVIX>{vv_t} DD<{int(dd_t*100)}%:  {raw_n:>5} {len(sub):>5} {m:>+7.2f}% {w:>5.1f}% {s:>7}{warning}  {marker}")

    print(f"\n=== Triple panic (21d, 독립 이벤트 기준) ===")
    print(f"  {'Signal':35s} {'RawN':>5} {'IndN':>5} {'Mean':>8} {'Win%':>6}")
    for v_t, vv_t, dd_t in combos:
        signal = (df["vix"] >= v_t) & (df["vvix"] >= vv_t) & (df["dd"] <= dd_t)
        dedup = deduplicate_events(signal, cooldown_days=21)
        raw_n = len(df[signal].dropna(subset=["future_21d"]))
        sub = df[dedup].dropna(subset=["future_21d"])
        if len(sub) < 2:
            continue
        m = sub["future_21d"].mean() * 100
        w = (sub["future_21d"] > 0).sum() / len(sub) * 100
        marker = "🚀" if w >= 95 and m > 5 and len(sub) >= 10 else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  VIX>{v_t} VVIX>{vv_t} DD<{int(dd_t*100)}%:  {raw_n:>5} {len(sub):>5} {m:>+6.2f}% {w:>5.1f}%  {marker}{warning}")

    out_path = RESULTS_DIR / "iter18_triple_signal.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
