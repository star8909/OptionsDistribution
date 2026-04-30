"""iter13: KOSPI 1d return → 단기 mean reversion.

가설: KOSPI 1d -3% 이상 떨어진 후 다음 5d 평균 회귀.
한투에서 KOSPI200 옵션 매매 가능 → 단기 신호.
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
    print("[iter13] KOSPI 1d return → 단기 mean reversion")
    ks = fetch_history("^KS200", period="20y", interval="1d")
    if ks.empty:
        ks = fetch_history("^KS11", period="20y", interval="1d")
    close = ks["adj_close"] if "adj_close" in ks.columns else ks["close"]

    df = pd.DataFrame({"ks": close})
    df["ret_1d"] = df["ks"].pct_change()
    for h in [3, 5, 10, 21]:
        df[f"future_{h}d"] = df["ks"].pct_change(h).shift(-h)

    print(f"\n=== KOSPI 1d return bucket → future return ===")
    bins = [-1, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05, 1]
    df["ret_bin"] = pd.cut(df["ret_1d"], bins=bins)
    print(f"  {'1d Ret':15s} {'N':>5}  {'3d':>7} {'5d':>7} {'10d':>7} {'21d':>7} W21d")
    for bin_ in df["ret_bin"].cat.categories:
        sub = df[df["ret_bin"] == bin_].dropna(subset=["future_21d"])
        if len(sub) < 30:
            continue
        r3 = sub["future_3d"].mean() * 100 if not sub["future_3d"].isna().all() else 0
        r5 = sub["future_5d"].mean() * 100 if not sub["future_5d"].isna().all() else 0
        r10 = sub["future_10d"].mean() * 100 if not sub["future_10d"].isna().all() else 0
        r21 = sub["future_21d"].mean() * 100
        win21 = (sub["future_21d"] > 0).sum() / len(sub) * 100
        marker = "🚀" if r21 > 3 and win21 > 60 else ""
        print(f"  {str(bin_):15s} {len(sub):>5} {r3:>+6.2f}% {r5:>+6.2f}% {r10:>+6.2f}% {r21:>+6.2f}% {win21:>4.0f}% {marker}")

    # Strong drop (독립 이벤트로 중복 제거, cooldown=5d for short-horizon signals)
    drop3_raw_signal = df["ret_1d"] < -0.03
    drop3_dedup = deduplicate_events(drop3_raw_signal, cooldown_days=5)
    drop3 = df[drop3_dedup].dropna(subset=["future_5d"])
    raw_n_drop3 = len(df[drop3_raw_signal].dropna(subset=["future_5d"]))
    warning = " ⚠️ N 부족 (신뢰 불가)" if len(drop3) < 10 else ""
    print(f"\n  KOSPI 1d -3%+ drop: Raw N={raw_n_drop3} → 독립 이벤트 N={len(drop3)}{warning}")
    if len(drop3) >= 5:
        avg5 = drop3["future_5d"].mean() * 100
        win5 = (drop3["future_5d"] > 0).sum() / len(drop3) * 100
        avg21 = drop3["future_21d"].dropna().mean() * 100 if not drop3["future_21d"].isna().all() else 0
        win21 = (drop3["future_21d"] > 0).sum() / max(len(drop3.dropna(subset=["future_21d"])), 1) * 100
        print(f"    5d {avg5:+.2f}% (win {win5:.1f}%)")
        print(f"    21d {avg21:+.2f}% (win {win21:.1f}%)")

    drop5_raw_signal = df["ret_1d"] < -0.05
    drop5_dedup = deduplicate_events(drop5_raw_signal, cooldown_days=5)
    drop5 = df[drop5_dedup].dropna(subset=["future_5d"])
    raw_n_drop5 = len(df[drop5_raw_signal].dropna(subset=["future_5d"]))
    warning5 = " ⚠️ N 부족 (신뢰 불가)" if len(drop5) < 10 else ""
    print(f"\n  KOSPI 1d -5%+ drop: Raw N={raw_n_drop5} → 독립 이벤트 N={len(drop5)}{warning5}")
    if len(drop5) >= 5:
        avg5 = drop5["future_5d"].mean() * 100
        win5 = (drop5["future_5d"] > 0).sum() / len(drop5) * 100
        print(f"    5d {avg5:+.2f}% (win {win5:.1f}%)")

    out_path = RESULTS_DIR / "iter13_kospi_intraday.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
