"""iter17: VIX × VVIX 결합 신호 (double panic).

iter05 VIX > 30: SPY 180d +23.7% Win 84.2% (152일)
iter16 VVIX > 130: SPY 180d +18.6% Win 72.5% (109일) / 21d +5.75% Win 86%
iter17: VIX > 30 AND VVIX > 130 동시 발화 → 진짜 double panic.

가설: 두 신호 동시 = 더 강한 확신, 더 높은 알파 또는 더 높은 win.
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
    print("[iter17] VIX × VVIX joint panic")
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
    print(f"  데이터: {len(df)} days, VIX {df['vix'].min():.1f}~{df['vix'].max():.1f}, VVIX {df['vvix'].min():.0f}~{df['vvix'].max():.0f}")

    for h in [5, 21, 63, 180]:
        df[f"future_{h}d"] = df["spy"].pct_change(h).shift(-h)

    combos = [
        (20, 90), (25, 100), (25, 110), (30, 110), (30, 120), (30, 130),
        (35, 130), (35, 140), (40, 130), (40, 150),
    ]

    print(f"\n=== VIX×VVIX 결합 (180d future, 독립 이벤트 기준) ===")
    print(f"  {'Signal':25s} {'RawN':>6} {'IndepN':>7} {'Mean':>8} {'Win%':>6} {'Sharpe':>7}")
    for v_t, vv_t in combos:
        signal = (df["vix"] >= v_t) & (df["vvix"] >= vv_t)
        dedup = deduplicate_events(signal, cooldown_days=180)
        raw_sub = df[signal].dropna(subset=["future_180d"])
        sub = df[dedup].dropna(subset=["future_180d"])
        if len(sub) < 3:
            continue
        m = sub["future_180d"].mean() * 100
        w = (sub["future_180d"] > 0).sum() / len(sub) * 100
        sh = event_sharpe(sub["future_180d"])
        s = f"{sh:.2f}" if not np.isnan(sh) else "N/A"
        marker = "🚀" if w >= 90 and m > 15 and not np.isnan(sh) else "✅" if w > 80 and not np.isnan(sh) else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  VIX>{v_t} AND VVIX>{vv_t}    {len(raw_sub):>6} {len(sub):>7} {m:>+7.2f}% {w:>5.1f}% {s:>7}{warning}  {marker}")

    print(f"\n=== VIX×VVIX 결합 (21d future, 독립 이벤트 기준) ===")
    print(f"  {'Signal':25s} {'RawN':>6} {'IndepN':>7} {'Mean':>8} {'Win%':>6}")
    for v_t, vv_t in combos:
        signal = (df["vix"] >= v_t) & (df["vvix"] >= vv_t)
        dedup = deduplicate_events(signal, cooldown_days=21)
        raw_sub = df[signal].dropna(subset=["future_21d"])
        sub = df[dedup].dropna(subset=["future_21d"])
        if len(sub) < 3:
            continue
        m = sub["future_21d"].mean() * 100
        w = (sub["future_21d"] > 0).sum() / len(sub) * 100
        marker = "🚀" if w >= 90 and m > 5 and len(sub) >= 10 else "✅" if w > 85 and len(sub) >= 10 else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  VIX>{v_t} AND VVIX>{vv_t}    {len(raw_sub):>6} {len(sub):>7} {m:>+6.2f}% {w:>5.1f}%  {marker}{warning}")

    out_path = RESULTS_DIR / "iter17_vix_vvix_joint.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
