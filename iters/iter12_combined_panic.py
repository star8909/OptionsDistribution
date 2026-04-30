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
from src.data_loader import fetch_history, deduplicate_events, event_sharpe


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

    # Raw counts (before deduplication)
    both_signal_raw = df["us_panic"] & df["kr_panic"]
    only_us_signal_raw = df["us_panic"] & ~df["kr_panic"]
    only_kr_signal_raw = ~df["us_panic"] & df["kr_panic"]

    raw_n_both = len(df[both_signal_raw].dropna(subset=["spy_180d", "ks_180d"]))
    raw_n_us = len(df[only_us_signal_raw].dropna(subset=["spy_180d"]))
    raw_n_kr = len(df[only_kr_signal_raw].dropna(subset=["ks_180d"]))

    # Deduplicate each signal independently (cooldown 180d)
    both_dedup = deduplicate_events(both_signal_raw, cooldown_days=180)
    only_us_dedup = deduplicate_events(only_us_signal_raw, cooldown_days=180)
    only_kr_dedup = deduplicate_events(only_kr_signal_raw, cooldown_days=180)

    both = df[both_dedup].dropna(subset=["spy_180d", "ks_180d"])
    only_us = df[only_us_dedup].dropna(subset=["spy_180d"])
    only_kr = df[only_kr_dedup].dropna(subset=["ks_180d"])

    print(f"\n=== Panic 신호 비교 (독립 이벤트 기준, cooldown 180일) ===")
    print(f"  US panic only: Raw N={raw_n_us} → 독립 N={len(only_us)}")
    print(f"  KR panic only: Raw N={raw_n_kr} → 독립 N={len(only_kr)}")
    print(f"  Both panic:    Raw N={raw_n_both} → 독립 N={len(both)}")

    if len(both) >= 5:
        spy_avg = both["spy_180d"].mean() * 100
        ks_avg = both["ks_180d"].mean() * 100
        spy_win = (both["spy_180d"] > 0).sum() / len(both) * 100
        ks_win = (both["ks_180d"] > 0).sum() / len(both) * 100
        warning = " ⚠️ N 부족 (신뢰 불가)" if len(both) < 10 else ""
        print(f"\n=== Both panic 신호 (강한 확신) N={len(both)}{warning} ===")
        print(f"  SPY 180d: {spy_avg:+.1f}% (win {spy_win:.1f}%)")
        print(f"  KS 180d: {ks_avg:+.1f}% (win {ks_win:.1f}%)")
        # 50/50 portfolio
        portfolio_50 = (both["spy_180d"] + both["ks_180d"]) / 2
        port_avg = portfolio_50.mean() * 100
        port_win = (portfolio_50 > 0).sum() / len(both) * 100
        sh = event_sharpe(portfolio_50)
        sh_str = f"{sh:.2f}" if not np.isnan(sh) else "N/A (N<10)"
        print(f"  50/50 portfolio: {port_avg:+.1f}% (win {port_win:.1f}%, Sharpe={sh_str})")

    if len(only_kr) >= 5:
        avg = only_kr["ks_180d"].mean() * 100
        win = (only_kr["ks_180d"] > 0).sum() / len(only_kr) * 100
        warning = " ⚠️ N 부족 (신뢰 불가)" if len(only_kr) < 10 else ""
        sh = event_sharpe(only_kr["ks_180d"])
        sh_str = f"{sh:.2f}" if not np.isnan(sh) else "N/A (N<10)"
        print(f"\n  KR panic only (독립 N={len(only_kr)}{warning}): KS 180d {avg:+.1f}% (win {win:.1f}%, Sharpe={sh_str})")

    if len(only_us) >= 5:
        avg = only_us["spy_180d"].mean() * 100
        win = (only_us["spy_180d"] > 0).sum() / len(only_us) * 100
        warning = " ⚠️ N 부족 (신뢰 불가)" if len(only_us) < 10 else ""
        sh = event_sharpe(only_us["spy_180d"])
        sh_str = f"{sh:.2f}" if not np.isnan(sh) else "N/A (N<10)"
        print(f"  US panic only (독립 N={len(only_us)}{warning}): SPY 180d {avg:+.1f}% (win {win:.1f}%, Sharpe={sh_str})")

    out = {
        "n_both_panic_raw": raw_n_both,
        "n_both_panic_independent": int(len(both)),
        "n_us_only_raw": raw_n_us,
        "n_us_only_independent": int(len(only_us)),
        "n_kr_only_raw": raw_n_kr,
        "n_kr_only_independent": int(len(only_kr)),
        "note": "독립 이벤트 기준 (cooldown 180일). N<10이면 Sharpe 신뢰 불가.",
    }
    out_path = RESULTS_DIR / "iter12_combined_panic.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
