"""iter12: 미국 + 한국 panic bottom 결합 strategy.

iter07 미국 (DD<-20% × VIX>30): +33% / win 100% / N=40
iter11 한국 (DD<-20% × RV>30): +31.7% / win 91% / N=194

가설: 두 시장 신호 결합 시 더 안정 (분산효과).
50% SPY + 50% KOSPI 시 신호 발화 시점 동시/별도 분석.
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
    print("[iter12] 미국 + 한국 panic bottom 결합")

    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    ks = fetch_history("^KS200", period="20y", interval="1d")
    if ks.empty:
        ks = fetch_history("^KS11", period="20y", interval="1d")

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    ks_close = ks["adj_close"] if "adj_close" in ks.columns else ks["close"]

    # 미국 panic
    df_us = pd.concat([spy_close.rename("spy"), vix_close.rename("vix")], axis=1).dropna()
    df_us["spy_252h"] = df_us["spy"].rolling(252, min_periods=50).max()
    df_us["spy_dd"] = df_us["spy"] / df_us["spy_252h"] - 1
    df_us["us_panic"] = (df_us["spy_dd"] < -0.20) & (df_us["vix"] > 30)
    df_us["spy_180d"] = df_us["spy"].pct_change(180).shift(-180)

    # 한국 panic
    df_kr = pd.DataFrame({"ks": ks_close})
    df_kr["rv_21"] = realized_vol(df_kr["ks"], 21) * 100
    df_kr["ks_252h"] = df_kr["ks"].rolling(252, min_periods=50).max()
    df_kr["ks_dd"] = df_kr["ks"] / df_kr["ks_252h"] - 1
    df_kr["kr_panic"] = (df_kr["ks_dd"] < -0.20) & (df_kr["rv_21"] > 30)
    df_kr["ks_180d"] = df_kr["ks"].pct_change(180).shift(-180)

    # Join
    df = df_us[["us_panic", "spy_180d"]].join(df_kr[["kr_panic", "ks_180d"]], how='inner')
    print(f"  Joined: {len(df)} days")

    # Both panic
    both = df[df["us_panic"] & df["kr_panic"]].dropna(subset=["spy_180d", "ks_180d"])
    only_us = df[df["us_panic"] & ~df["kr_panic"]].dropna(subset=["spy_180d"])
    only_kr = df[~df["us_panic"] & df["kr_panic"]].dropna(subset=["ks_180d"])
    either = df[df["us_panic"] | df["kr_panic"]]

    print(f"\n=== Panic 신호 비교 ===")
    print(f"  US panic only: {len(only_us)}일")
    print(f"  KR panic only: {len(only_kr)}일")
    print(f"  Both panic: {len(both)}일")
    print(f"  Either panic: {len(either)}일")

    if len(both) >= 10:
        spy_avg = both["spy_180d"].mean() * 100
        ks_avg = both["ks_180d"].mean() * 100
        spy_win = (both["spy_180d"] > 0).sum() / len(both) * 100
        ks_win = (both["ks_180d"] > 0).sum() / len(both) * 100
        print(f"\n=== Both panic 신호 (강한 확신) ===")
        print(f"  SPY 180d: {spy_avg:+.1f}% (win {spy_win:.1f}%)")
        print(f"  KS 180d: {ks_avg:+.1f}% (win {ks_win:.1f}%)")
        # 50/50 portfolio
        portfolio_50 = (both["spy_180d"] + both["ks_180d"]) / 2
        port_avg = portfolio_50.mean() * 100
        port_win = (portfolio_50 > 0).sum() / len(both) * 100
        print(f"  50/50 portfolio: {port_avg:+.1f}% (win {port_win:.1f}%)")

    if len(only_kr) >= 20:
        avg = only_kr["ks_180d"].mean() * 100
        win = (only_kr["ks_180d"] > 0).sum() / len(only_kr) * 100
        print(f"\n  KR panic only ({len(only_kr)}일): KS 180d {avg:+.1f}% (win {win:.1f}%)")

    if len(only_us) >= 10:
        avg = only_us["spy_180d"].mean() * 100
        win = (only_us["spy_180d"] > 0).sum() / len(only_us) * 100
        print(f"  US panic only ({len(only_us)}일): SPY 180d {avg:+.1f}% (win {win:.1f}%)")

    out = {
        "n_both_panic": int(len(both)),
        "n_us_only": int(len(only_us)),
        "n_kr_only": int(len(only_kr)),
    }
    out_path = RESULTS_DIR / "iter12_combined_panic.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
