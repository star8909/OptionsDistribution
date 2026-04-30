"""iter16: VVIX (vol of vol) → SPY contrarian signal.

VVIX = VIX의 변동성. 시장 panic 강도 측정.
가설:
- VVIX > 130 (extreme fear of fear) → SPY 단기 contrarian rally
- VVIX < 80 (calm calm) → SPY continuation

VVIX = ^VVIX yfinance ticker.
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
    print("[iter16] VVIX (vol of vol) signal")
    spy = fetch_history("SPY", period="20y", interval="1d")
    vvix = fetch_history("^VVIX", period="20y", interval="1d")
    if vvix.empty:
        print("  ❌ ^VVIX 데이터 없음 → ^VIX 대체 시도")
        vvix = fetch_history("^VIX", period="20y", interval="1d")
        if vvix.empty:
            return

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vvix_close = vvix["adj_close"] if "adj_close" in vvix.columns else vvix["close"]

    df = pd.concat([spy_close.rename("spy"), vvix_close.rename("vvix")], axis=1).dropna()
    print(f"  VVIX 데이터: {len(df)} days, range {df['vvix'].min():.1f} ~ {df['vvix'].max():.1f}")

    for h in [5, 21, 63, 180]:
        df[f"future_{h}d"] = df["spy"].pct_change(h).shift(-h)

    print(f"\n=== VVIX threshold 분석 (180d future, 독립 이벤트 기준) ===")
    print(f"  {'Threshold':12s} {'RawN':>6} {'IndepN':>7} {'Mean':>8} {'Win%':>6} {'Sharpe':>7}")
    for thr in [80, 90, 100, 110, 120, 130, 140, 150]:
        signal = df["vvix"] >= thr
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
        print(f"  VVIX > {thr:>3}    {len(raw_sub):>6} {len(sub):>7} {m:>+7.2f}% {w:>5.1f}% {s:>7}{warning}  {marker}")

    print(f"\n=== VVIX 단기 (21d future, 독립 이벤트 기준) ===")
    print(f"  {'Threshold':12s} {'RawN':>6} {'IndepN':>7} {'Mean':>8} {'Win%':>6}")
    for thr in [100, 110, 120, 130, 140, 150]:
        signal = df["vvix"] >= thr
        dedup = deduplicate_events(signal, cooldown_days=21)
        raw_sub = df[signal].dropna(subset=["future_21d"])
        sub = df[dedup].dropna(subset=["future_21d"])
        if len(sub) < 5:
            continue
        m = sub["future_21d"].mean() * 100
        w = (sub["future_21d"] > 0).sum() / len(sub) * 100
        marker = "🚀" if m > 5 and len(sub) >= 10 else ""
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  VVIX > {thr:>3}    {len(raw_sub):>6} {len(sub):>7} {m:>+7.2f}% {w:>5.1f}%  {marker}{warning}")

    print(f"\n=== VVIX 낮은 calm (continuation, 독립 이벤트 기준) ===")
    print(f"  {'Threshold':12s} {'RawN':>6} {'IndepN':>7} {'180d mean':>10} {'Win%':>6}")
    for thr in [70, 75, 80, 85]:
        signal = df["vvix"] < thr
        dedup = deduplicate_events(signal, cooldown_days=180)
        raw_sub = df[signal].dropna(subset=["future_180d"])
        sub = df[dedup].dropna(subset=["future_180d"])
        if len(sub) < 5:
            continue
        m = sub["future_180d"].mean() * 100
        w = (sub["future_180d"] > 0).sum() / len(sub) * 100
        warning = " ⚠️" if len(sub) < 10 else ""
        print(f"  VVIX < {thr:>3}    {len(raw_sub):>6} {len(sub):>7} {m:>+9.2f}% {w:>5.1f}%  {warning}")

    out_path = RESULTS_DIR / "iter16_vvix_signal.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
