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
from src.data_loader import fetch_history


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

    # Combined signals
    print(f"\n=== Both panic (US+KR) ===")
    both = df[df["us_panic"] & df["kr_panic"]].dropna(subset=["spy_180d", "ks_180d"])
    print(f"  N={len(both)}")
    if len(both) >= 5:
        port = (both["spy_180d"] + both["ks_180d"]) / 2
        print(f"  SPY 180d: {both['spy_180d'].mean()*100:+.2f}% Win {(both['spy_180d']>0).sum()/len(both)*100:.1f}%")
        print(f"  KS  180d: {both['ks_180d'].mean()*100:+.2f}% Win {(both['ks_180d']>0).sum()/len(both)*100:.1f}%")
        print(f"  50/50: {port.mean()*100:+.2f}% Win {(port>0).sum()/len(both)*100:.1f}%")

    print(f"\n=== Both panic STRONG (US strong + KR) ===")
    both_s = df[df["us_panic_strong"] & df["kr_panic"]].dropna(subset=["spy_180d", "ks_180d"])
    print(f"  N={len(both_s)}")
    if len(both_s) >= 3:
        port_s = (both_s["spy_180d"] + both_s["ks_180d"]) / 2
        print(f"  SPY 180d: {both_s['spy_180d'].mean()*100:+.2f}% Win {(both_s['spy_180d']>0).sum()/len(both_s)*100:.1f}%")
        print(f"  KS  180d: {both_s['ks_180d'].mean()*100:+.2f}% Win {(both_s['ks_180d']>0).sum()/len(both_s)*100:.1f}%")
        print(f"  50/50: {port_s.mean()*100:+.2f}% Win {(port_s>0).sum()/len(both_s)*100:.1f}%")
        s = port_s.mean() / port_s.std() * np.sqrt(2) if port_s.std() > 0 else 0
        print(f"  Sharpe: {s:.2f}")

    # 21d 단기
    print(f"\n=== Both panic (21d 단기) ===")
    if len(both) >= 5:
        sub = both.dropna(subset=["spy_21d", "ks_21d"])
        if len(sub) > 0:
            port_21 = (sub["spy_21d"] + sub["ks_21d"]) / 2
            print(f"  Both 21d 50/50: {port_21.mean()*100:+.2f}% Win {(port_21>0).sum()/len(sub)*100:.1f}%")

    out_path = RESULTS_DIR / "iter21_us_kr_vvix.json"
    out_path.write_text(json.dumps({"n_both": int(len(both)), "n_both_strong": int(len(both_s))}, indent=2), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
