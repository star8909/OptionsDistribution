"""iter04: Put/Call skew anomaly (현재 SPY chain).

가설: OTM put이 OTM call보다 비쌈 (crash 공포 = vol smile/skew).
25-delta put IV vs 25-delta call IV gap → "skew premium"
- 평균 put IV > call IV by 2~5% (학술 정설)
- skew 극단 시 → put 매도 +EV (crash 공포 과대)

분석:
1. SPY 현재 options chain
2. ATM IV 기준점
3. -10% OTM put vs +10% OTM call IV 비교
4. Skew percentile (현재 vs 역사)

yfinance는 현재 chain만 → snapshot 분석.
역사적 skew는 CBOE SKEW index (^SKEW) 사용.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_history, fetch_chain


def main():
    print("[iter04] Put/Call skew anomaly")

    # 1. CBOE SKEW index 역사 (^SKEW)
    skew = fetch_history("^SKEW", period="10y", interval="1d")
    if not skew.empty:
        skew_close = skew["adj_close"] if "adj_close" in skew.columns else skew["close"]
        print(f"\n=== CBOE SKEW Index 10년 ===")
        print(f"  N: {len(skew_close)}")
        print(f"  평균: {skew_close.mean():.2f}")
        print(f"  std: {skew_close.std():.2f}")
        print(f"  min: {skew_close.min():.2f} (skew 약함, crash 공포 ↓)")
        print(f"  max: {skew_close.max():.2f} (skew 강함, crash 공포 ↑)")
        print(f"  현재값: {skew_close.iloc[-1]:.2f}")
        print(f"  현재 percentile: {(skew_close <= skew_close.iloc[-1]).sum() / len(skew_close) * 100:.1f}%")

        # SKEW > 140 → 강한 crash 공포 / SKEW < 110 → 안정
        high = (skew_close > 140).sum() / len(skew_close) * 100
        low = (skew_close < 110).sum() / len(skew_close) * 100
        print(f"  SKEW > 140 (panic) 비율: {high:.1f}%")
        print(f"  SKEW < 110 (calm) 비율: {low:.1f}%")

    # 2. SKEW vs SPY future return
    if not skew.empty:
        spy = fetch_history("SPY", period="10y", interval="1d")
        if not spy.empty:
            spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
            df = pd.concat([skew_close.rename("skew"), spy_close.rename("spy")], axis=1).dropna()
            df["spy_30d_ret"] = df["spy"].pct_change(30).shift(-30)

            print(f"\n=== SKEW level vs SPY 30d future return ===")
            for low, high in [(100, 120), (120, 130), (130, 140), (140, 150), (150, 200)]:
                mask = (df["skew"] >= low) & (df["skew"] < high)
                sub = df[mask]
                if len(sub) >= 20:
                    avg_ret = sub["spy_30d_ret"].mean() * 100
                    pos_ratio = (sub["spy_30d_ret"] > 0).sum() / len(sub) * 100
                    print(f"  SKEW {low}-{high}: N={len(sub):>4} | SPY 30d return 평균 {avg_ret:+.2f}% | 양수 비율 {pos_ratio:.1f}%")

    # 3. 현재 SPY options chain — skew snapshot
    print(f"\n=== SPY 현재 chain skew snapshot ===")
    try:
        chain = fetch_chain("SPY", expiration=None)
        if chain:
            spy_price = chain.get('underlying_price', 0)
            print(f"  SPY 현재가: ${spy_price:.2f}")
            print(f"  Expiration: {chain['expiration']}")

            calls = chain['calls']
            puts = chain['puts']

            if not calls.empty and not puts.empty and 'impliedVolatility' in calls.columns:
                # ATM 찾기
                atm_call = calls.iloc[(calls['strike'] - spy_price).abs().argsort()[:1]]
                atm_put = puts.iloc[(puts['strike'] - spy_price).abs().argsort()[:1]]
                if len(atm_call) and len(atm_put):
                    print(f"  ATM Call IV: {float(atm_call['impliedVolatility'].iloc[0])*100:.1f}%")
                    print(f"  ATM Put IV: {float(atm_put['impliedVolatility'].iloc[0])*100:.1f}%")

                # 10% OTM
                otm_put = puts[puts['strike'] < spy_price * 0.95]
                otm_call = calls[calls['strike'] > spy_price * 1.05]
                if not otm_put.empty and not otm_call.empty:
                    iv_otm_put = otm_put['impliedVolatility'].mean() * 100
                    iv_otm_call = otm_call['impliedVolatility'].mean() * 100
                    skew_curr = iv_otm_put - iv_otm_call
                    print(f"  OTM Put avg IV (-5%~): {iv_otm_put:.1f}%")
                    print(f"  OTM Call avg IV (+5%~): {iv_otm_call:.1f}%")
                    print(f"  Skew (Put - Call): {skew_curr:+.1f}%")
    except Exception as e:
        print(f"  ❌ 현재 chain fetch 실패: {e}")

    out = {
        "skew_index_avg": float(skew_close.mean()) if not skew.empty else None,
        "skew_index_current": float(skew_close.iloc[-1]) if not skew.empty else None,
    }
    out_path = RESULTS_DIR / "iter04_skew.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
