"""Options iter21 검증: overlapping observation window 문제.

핵심 의심:
- 패닉이 며칠 연속으로 발생 시 (예: 코로나 2020년 3월 10~27일),
  모든 날을 별개 관측치로 취급 → 180d forward return 거의 동일 (1일 차이)
  → N이 부풀려지고, std가 0에 가까워져 Sharpe 폭등

진짜 N: 독립 이벤트 수 (연속 패닉 = 1개 이벤트)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.data_loader import fetch_history


def realized_vol(prices, window):
    rets = prices.pct_change()
    return rets.rolling(window).std() * np.sqrt(252)


def main():
    print("[debug] iter21 overlapping window 검증")

    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    vvix = fetch_history("^VVIX", period="20y", interval="1d")
    ks = fetch_history("^KS11", period="20y", interval="1d")

    if any(d.empty for d in [spy, vix, vvix, ks]):
        print("  ❌ 데이터 부재")
        return

    spy_c = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_c = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    vvix_c = vvix["adj_close"] if "adj_close" in vvix.columns else vvix["close"]
    ks_c = ks["adj_close"] if "adj_close" in ks.columns else ks["close"]

    df_us = pd.concat([spy_c.rename("spy"), vix_c.rename("vix"), vvix_c.rename("vvix")], axis=1).dropna()
    df_us["spy_252h"] = df_us["spy"].rolling(252, min_periods=50).max()
    df_us["dd"] = df_us["spy"] / df_us["spy_252h"] - 1
    df_us["us_panic"] = (df_us["vix"] >= 30) & (df_us["vvix"] >= 120) & (df_us["dd"] <= -0.10)
    df_us["us_panic_strong"] = (df_us["vix"] >= 35) & (df_us["vvix"] >= 130) & (df_us["dd"] <= -0.15)
    df_us["spy_180d"] = df_us["spy"].pct_change(180).shift(-180)

    df_kr = pd.DataFrame({"ks": ks_c})
    df_kr["rv_21"] = realized_vol(df_kr["ks"], 21) * 100
    df_kr["ks_252h"] = df_kr["ks"].rolling(252, min_periods=50).max()
    df_kr["ks_dd"] = df_kr["ks"] / df_kr["ks_252h"] - 1
    df_kr["kr_panic"] = (df_kr["ks_dd"] < -0.20) & (df_kr["rv_21"] > 30)
    df_kr["ks_180d"] = df_kr["ks"].pct_change(180).shift(-180)

    df = df_us[["us_panic", "us_panic_strong", "spy_180d"]].join(
        df_kr[["kr_panic", "ks_180d"]], how="inner")

    both = df[df["us_panic"] & df["kr_panic"]].dropna(subset=["spy_180d", "ks_180d"])
    both_s = df[df["us_panic_strong"] & df["kr_panic"]].dropna(subset=["spy_180d", "ks_180d"])

    print(f"\n=== 원래 iter21 Both panic ===")
    print(f"  Raw N = {len(both)} (모든 패닉 날 포함)")
    if len(both) >= 3:
        port = (both["spy_180d"] + both["ks_180d"]) / 2
        sharpe_raw = port.mean() / port.std() * np.sqrt(2) if port.std() > 0 else 0
        print(f"  50/50 180d: mean={port.mean()*100:+.1f}%, std={port.std()*100:.1f}%, Win={( port>0).sum()}/{len(port)}")
        print(f"  Sharpe(raw N): {sharpe_raw:.2f}  ← 부풀려진 버전")

    # 연속 패닉 날을 하나의 이벤트로 압축 (독립 이벤트 수 측정)
    print(f"\n=== 독립 이벤트 식별 (연속 패닉 = 1 이벤트) ===")
    both_s_signal = df["us_panic"] & df["kr_panic"]
    # 이벤트 시작 = 전날은 False, 오늘은 True
    event_start = both_s_signal & (~both_s_signal.shift(1).fillna(False))
    events = df[event_start].dropna(subset=["spy_180d", "ks_180d"])
    print(f"  독립 이벤트 N = {len(events)}")

    if len(events) > 0:
        print(f"\n  이벤트 날짜 목록:")
        for d, row in events.iterrows():
            port_ret = (row["spy_180d"] + row["ks_180d"]) / 2
            print(f"    {d.date()}: SPY 180d={row['spy_180d']*100:+.1f}%  KS 180d={row['ks_180d']*100:+.1f}%  50/50={port_ret*100:+.1f}%")

    if len(events) >= 3:
        port_e = (events["spy_180d"] + events["ks_180d"]) / 2
        sharpe_e = port_e.mean() / port_e.std() * np.sqrt(2) if port_e.std() > 0 else 0
        print(f"\n  독립 이벤트 기준:")
        print(f"    50/50 mean = {port_e.mean()*100:+.1f}%  std = {port_e.std()*100:.1f}%  Win = {(port_e>0).sum()}/{len(port_e)}")
        print(f"    Sharpe(독립 이벤트): {sharpe_e:.2f}")

    # Strong panic 버전
    print(f"\n=== Both panic STRONG (독립 이벤트) ===")
    strong_signal = df["us_panic_strong"] & df["kr_panic"]
    strong_start = strong_signal & (~strong_signal.shift(1).fillna(False))
    events_s = df[strong_start].dropna(subset=["spy_180d", "ks_180d"])
    print(f"  원래 N = {len(both_s)}  →  독립 이벤트 N = {len(events_s)}")
    if len(events_s) > 0:
        for d, row in events_s.iterrows():
            port_ret = (row["spy_180d"] + row["ks_180d"]) / 2
            print(f"    {d.date()}: 50/50={port_ret*100:+.1f}%")
    if len(events_s) >= 2:
        port_se = (events_s["spy_180d"] + events_s["ks_180d"]) / 2
        sharpe_se = port_se.mean() / port_se.std() * np.sqrt(2) if port_se.std() > 0 else 0
        print(f"  Sharpe(독립이벤트): {sharpe_se:.2f}")
        print(f"  → 통계적 신뢰도: N={len(events_s)} (최소 10 필요 / 이건 {'충분' if len(events_s) >= 10 else '부족 ❌'})")

    # 크라이시스별 분석 (2008, 2020, 2022 구분)
    print(f"\n=== 크라이시스 기간 분포 ===")
    if len(both) > 0:
        years = both.index.year.value_counts().sort_index()
        for yr, cnt in years.items():
            print(f"  {yr}: {cnt}일")

    # 무조건 SPY 180d 수익률과 비교 (base rate)
    print(f"\n=== Base Rate 비교 ===")
    spy_all = df["spy_180d"].dropna()
    print(f"  SPY 무조건 180d: mean={spy_all.mean()*100:+.1f}%  Win={( spy_all>0).sum()}/{len(spy_all)} ({(spy_all>0).mean()*100:.0f}%)")
    if len(both) >= 3:
        spy_cond = both["spy_180d"]
        print(f"  SPY 패닉 후 180d: mean={spy_cond.mean()*100:+.1f}%  Win={( spy_cond>0).sum()}/{len(spy_cond)} ({(spy_cond>0).mean()*100:.0f}%)")
        diff = spy_cond.mean() - spy_all.mean()
        print(f"  조건부 - 무조건 = {diff*100:+.1f}%p  ({'신호 있음' if abs(diff) > 0.05 else '신호 미약'})")


if __name__ == "__main__":
    main()
