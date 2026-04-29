"""iter02: VIX term structure carry harvest.

가설: VIX (1m) < VIX3M (3m) → contango. 평균 -0.5~1%/month carry.
즉 VXX (1m vol future ETF) short → 평균 +EV (학술 정설).

분석:
1. VIX vs VIX3M spread (10년)
2. Contango vs backwardation 일자 비율
3. VIX3M / VIX 비율 분포
4. UVXY/VIXY long-only는 평균 -50%/year (well-known)
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


def main():
    print("[iter02] VIX term structure carry harvest")

    vix = fetch_history("^VIX", period="10y", interval="1d")
    vix3m = fetch_history("^VIX3M", period="10y", interval="1d")
    vix9d = fetch_history("^VIX9D", period="10y", interval="1d")

    if vix.empty:
        print("  ❌ VIX 데이터 없음")
        return

    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    print(f"  VIX: {len(vix_close)} days")

    if not vix3m.empty:
        vix3m_close = vix3m["adj_close"] if "adj_close" in vix3m.columns else vix3m["close"]
        print(f"  VIX3M: {len(vix3m_close)} days")
    else:
        vix3m_close = pd.Series()

    if not vix9d.empty:
        vix9d_close = vix9d["adj_close"] if "adj_close" in vix9d.columns else vix9d["close"]
        print(f"  VIX9D: {len(vix9d_close)} days")
    else:
        vix9d_close = pd.Series()

    # Term structure 분석
    if not vix3m_close.empty:
        df = pd.concat([
            vix_close.rename("vix_1m"),
            vix3m_close.rename("vix_3m"),
        ], axis=1).dropna()

        df["spread_3m_1m"] = df["vix_3m"] - df["vix_1m"]
        df["ratio_3m_1m"] = df["vix_3m"] / df["vix_1m"]

        print(f"\n=== VIX term structure (1m vs 3m, {len(df)} days) ===")
        print(f"  VIX 1m 평균: {df['vix_1m'].mean():.2f}")
        print(f"  VIX 3m 평균: {df['vix_3m'].mean():.2f}")
        print(f"  Spread (3m - 1m) 평균: {df['spread_3m_1m'].mean():+.2f}")
        print(f"  Ratio (3m / 1m) 평균: {df['ratio_3m_1m'].mean():.3f}")

        contango_pct = (df['spread_3m_1m'] > 0).sum() / len(df) * 100
        print(f"  Contango 일자 (3m > 1m): {contango_pct:.1f}%")
        print(f"  Backwardation 일자 (3m < 1m): {100 - contango_pct:.1f}%")

        # Strong contango/backwardation
        strong_c = (df['ratio_3m_1m'] > 1.10).sum() / len(df) * 100
        strong_b = (df['ratio_3m_1m'] < 0.90).sum() / len(df) * 100
        print(f"  Strong contango (ratio > 1.10): {strong_c:.1f}%")
        print(f"  Strong backwardation (ratio < 0.90, vol spike): {strong_b:.1f}%")

        # Carry 시뮬: contango 일자에 short, backwardation 일자에 long
        # 매일 next day return = (vix_1m_t+1 - vix_1m_t) / vix_1m_t
        df["vix_ret_1d"] = df["vix_1m"].pct_change().shift(-1)
        # Strategy: contango → short VIX (수익 = -ret)
        df["pnl_carry"] = df["vix_ret_1d"] * (-1) * (df['ratio_3m_1m'] > 1.0).astype(int)
        carry_pnl = df["pnl_carry"].dropna()

        if len(carry_pnl) > 100:
            sharpe = carry_pnl.mean() / carry_pnl.std() * np.sqrt(252) if carry_pnl.std() > 0 else 0
            cum = (1 + carry_pnl).cumprod()
            n_y = max((carry_pnl.index[-1] - carry_pnl.index[0]).days / 365.25, 1e-9)
            cagr = cum.iloc[-1] ** (1 / n_y) - 1
            cm = cum.cummax()
            mdd = (cum / cm - 1).min()
            print(f"\n=== Contango harvest (VIX short on contango days) ===")
            print(f"  N days: {len(carry_pnl)}")
            print(f"  Mean daily PnL: {carry_pnl.mean()*100:+.3f}%")
            print(f"  Std daily PnL: {carry_pnl.std()*100:.3f}%")
            print(f"  Sharpe: {sharpe:.2f}")
            print(f"  CAGR: {cagr*100:+.1f}%")
            print(f"  MDD: {mdd*100:.1f}%")

        # Skew 분석 (1d return distribution)
        rets = df["vix_1m"].pct_change().dropna()
        print(f"\n=== VIX 1d return 분포 ===")
        print(f"  Mean: {rets.mean()*100:+.3f}%")
        print(f"  Std: {rets.std()*100:.3f}%")
        print(f"  Skew: {rets.skew():.2f} (양수 큰 → 우측 fat tail = vol spike up)")
        print(f"  Kurtosis: {rets.kurt():.2f}")

    # VIX9D vs VIX (단기 vol spike 신호)
    if not vix9d_close.empty:
        df2 = pd.concat([
            vix_close.rename("vix_1m"),
            vix9d_close.rename("vix_9d"),
        ], axis=1).dropna()

        df2["ratio_9d_1m"] = df2["vix_9d"] / df2["vix_1m"]
        print(f"\n=== VIX9D / VIX 1m 비율 (단기 vs 한달) ===")
        print(f"  Ratio 평균: {df2['ratio_9d_1m'].mean():.3f}")
        print(f"  Ratio > 1.10 (단기 panic): {(df2['ratio_9d_1m'] > 1.10).sum() / len(df2) * 100:.1f}%")
        print(f"  Ratio > 1.20 (severe panic): {(df2['ratio_9d_1m'] > 1.20).sum() / len(df2) * 100:.1f}%")

    out = {
        "n_days_3m": len(df) if not vix3m_close.empty else 0,
        "contango_pct": float(contango_pct) if not vix3m_close.empty else None,
        "spread_mean": float(df['spread_3m_1m'].mean()) if not vix3m_close.empty else None,
    }
    if not vix3m_close.empty and len(carry_pnl) > 100:
        out["carry_sharpe"] = float(sharpe)
        out["carry_cagr_pct"] = float(cagr * 100)
        out["carry_mdd_pct"] = float(mdd * 100)

    out_path = RESULTS_DIR / "iter02_term_structure.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
