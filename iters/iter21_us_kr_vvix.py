"""iter21: US (VIX × VVIX) + KR (KOSPI panic) 결합 (cross-market triple).

iter12 US+KR Both Panic: SPY+KOSPI 50/50 +57% Win 100% N=17
iter17 VIX×VVIX: SPY +47% Win 100% Sharpe 7.17

iter21: US VIX>30 + VVIX>120 + KOSPI panic 동시 → cross-market 검증.
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


def realized_vol(prices, window):
    rets = prices.pct_change()
    return rets.rolling(window).std() * np.sqrt(252)


def main():
    print("[iter21] US (VIX×VVIX) + KR (KOSPI panic) cross-market")
    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    vvix = fetch_history("^VVIX", period="20y", interval="1d")
    ks = fetch_history("^KS200", period="20y", interval="1d")
    if ks.empty:
        ks = fetch_history("^KS11", period="20y", interval="1d")

    if any([d.empty for d in [spy, vix, vvix, ks]]):
        print("  ❌ 데이터 부재")
        return

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    vvix_close = vvix["adj_close"] if "adj_close" in vvix.columns else vvix["close"]
    ks_close = ks["adj_close"] if "adj_close" in ks.columns else ks["close"]

    # US
    df_us = pd.concat([spy_close.rename("spy"), vix_close.rename("vix"), vvix_close.rename("vvix")], axis=1).dropna()
    df_us["spy_252h"] = df_us["spy"].rolling(252, min_periods=50).max()
    df_us["dd"] = df_us["spy"] / df_us["spy_252h"] - 1
    df_us["us_panic"] = (df_us["vix"] >= 30) & (df_us["vvix"] >= 120) & (df_us["dd"] <= -0.10)
    df_us["us_panic_strong"] = (df_us["vix"] >= 35) & (df_us["vvix"] >= 130) & (df_us["dd"] <= -0.15)
    df_us["spy_180d"] = df_us["spy"].pct_change(180).shift(-180)
    df_us["spy_21d"] = df_us["spy"].pct_change(21).shift(-21)

    # KR
    df_kr = pd.DataFrame({"ks": ks_close})
    df_kr["rv_21"] = realized_vol(df_kr["ks"], 21) * 100
    df_kr["ks_252h"] = df_kr["ks"].rolling(252, min_periods=50).max()
    df_kr["ks_dd"] = df_kr["ks"] / df_kr["ks_252h"] - 1
    df_kr["kr_panic"] = (df_kr["ks_dd"] < -0.20) & (df_kr["rv_21"] > 30)
    df_kr["ks_180d"] = df_kr["ks"].pct_change(180).shift(-180)
    df_kr["ks_21d"] = df_kr["ks"].pct_change(21).shift(-21)

    # Join
    df = df_us[["us_panic", "us_panic_strong", "spy_180d", "spy_21d"]].join(
        df_kr[["kr_panic", "ks_180d", "ks_21d"]], how='inner')
    print(f"  Joined: {len(df)} days")

    both_signal = df["us_panic"] & df["kr_panic"]
    strong_signal = df["us_panic_strong"] & df["kr_panic"]

    # Raw N (이전 방식 — 연속 날 전부 포함, 통계 부풀림)
    both_raw = df[both_signal].dropna(subset=["spy_180d", "ks_180d"])
    both_s_raw = df[strong_signal].dropna(subset=["spy_180d", "ks_180d"])

    # 독립 이벤트 (연속 패닉 = 1 이벤트, cooldown 180일)
    both_dedup = deduplicate_events(both_signal, cooldown_days=180)
    strong_dedup = deduplicate_events(strong_signal, cooldown_days=180)
    both_events = df[both_dedup].dropna(subset=["spy_180d", "ks_180d"])
    strong_events = df[strong_dedup].dropna(subset=["spy_180d", "ks_180d"])

    def print_result(label, events, raw_n):
        n = len(events)
        warning = " ⚠️ N 부족 (신뢰 불가)" if n < 10 else ""
        print(f"\n=== {label} ===")
        print(f"  Raw N={raw_n}  →  독립 이벤트 N={n}{warning}")
        if n == 0:
            return {}
        port = (events["spy_180d"] + events["ks_180d"]) / 2
        sh = event_sharpe(port)
        sh_str = f"{sh:.2f}" if not np.isnan(sh) else "N/A (N<10)"
        print(f"  SPY 180d: {events['spy_180d'].mean()*100:+.1f}%  Win {(events['spy_180d']>0).mean()*100:.0f}%")
        print(f"  KS  180d: {events['ks_180d'].mean()*100:+.1f}%  Win {(events['ks_180d']>0).mean()*100:.0f}%")
        print(f"  50/50: {port.mean()*100:+.1f}%  Win {(port>0).mean()*100:.0f}%  Sharpe={sh_str}")
        for dt, row in events.iterrows():
            p = (row["spy_180d"] + row["ks_180d"]) / 2
            print(f"    {dt.date()}: 50/50={p*100:+.1f}%")
        return {"n_independent": n, "raw_n": raw_n, "mean_50_50_pct": float(port.mean()*100), "sharpe": float(sh) if not np.isnan(sh) else None}

    r_both = print_result("Both panic (US+KR)", both_events, len(both_raw))
    r_strong = print_result("Both panic STRONG", strong_events, len(both_s_raw))

    out = {
        "n_both_raw": int(len(both_raw)),
        "n_both_independent": int(len(both_events)),
        "n_strong_raw": int(len(both_s_raw)),
        "n_strong_independent": int(len(strong_events)),
        "both": r_both,
        "strong": r_strong,
        "note": "독립 이벤트 기준 (cooldown 180일). N<10이면 Sharpe 신뢰 불가.",
    }
    out_path = RESULTS_DIR / "iter21_us_kr_vvix.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
