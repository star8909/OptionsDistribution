"""다양한 옵션 전략 — panic event 외에 다른 alpha source.

20가지 전략:
1-10. IV-RV / term structure / skew / FOMC / VIX mean rev / VIX9D / DD recovery /
      turn-of-month / KOSPI panic / KOSPI DD
11-20. VIX>35 extreme / VIX<12 calm-bull / 5d crash rev / 21d slow drop /
      VVIX>130 / VIX9D strong inv / VIX z>2 / SPY 50d trend / TLT inv /
      multi-trigger AND
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_history, deduplicate_events, event_sharpe


def realized_vol(prices, window):
    return prices.pct_change().rolling(window).std() * np.sqrt(252)


def evaluate_signal(df, signal, ret_col, cooldown=30, min_n=10):
    raw_n = int(signal.sum())
    dedup = deduplicate_events(signal, cooldown_days=cooldown)
    events = df[dedup].dropna(subset=[ret_col])
    n = len(events)
    if n < min_n:
        return {"raw_n": raw_n, "n": n, "mean_pct": None, "win_pct": None, "sharpe": None}
    rets = events[ret_col]
    sh = event_sharpe(rets)
    return {
        "raw_n": raw_n, "n": n,
        "mean_pct": float(rets.mean() * 100),
        "win_pct": float((rets > 0).mean() * 100),
        "sharpe": float(sh) if not np.isnan(sh) else None,
    }


def strat_iv_rv_gap(spy, vix):
    """VIX > realized vol 21d → straddle short proxy (mean reversion)."""
    rv = realized_vol(spy, 21) * 100
    df = pd.concat([spy.rename("spy"), vix.rename("vix"), rv.rename("rv")], axis=1).dropna()
    df["fwd_21d"] = df["spy"].pct_change(21).shift(-21)
    sig = (df["vix"] > df["rv"] * 1.3) & (df["rv"] < 15)  # IV > RV*1.3 in low vol regime
    return df, sig, "fwd_21d"


def strat_vix_term_inv(vix, vix3m, spy):
    """VIX > VIX3M (term structure inversion) — historical SPY recovery signal."""
    df = pd.concat([vix.rename("vix"), vix3m.rename("vix3m"), spy.rename("spy")], axis=1).dropna()
    df["fwd_180d"] = df["spy"].pct_change(180).shift(-180)
    sig = df["vix"] > df["vix3m"]
    return df, sig, "fwd_180d"


def strat_skew(skew, spy):
    """CBOE SKEW < 120 (낮은 crash fear) → SPY drift 양수."""
    df = pd.concat([skew.rename("skew"), spy.rename("spy")], axis=1).dropna()
    df["fwd_30d"] = df["spy"].pct_change(30).shift(-30)
    sig = df["skew"] < 120
    return df, sig, "fwd_30d"


def strat_fomc_proxy(spy):
    """월 첫째 주 FOMC 효과 proxy — 매월 첫 수요일 다음 21일 long."""
    df = pd.DataFrame({"spy": spy})
    df["fwd_21d"] = df["spy"].pct_change(21).shift(-21)
    # 매월 첫 수요일
    sig = pd.Series(False, index=df.index)
    for ym in df.index.to_period('M').unique():
        month_idx = df.index[df.index.to_period('M') == ym]
        wed_days = month_idx[month_idx.weekday == 2]
        if len(wed_days) > 0:
            sig.loc[wed_days[0]] = True
    return df, sig, "fwd_21d"


def strat_vix_mean_rev(vix, spy):
    """VIX > 25 (낮은 임계, 빈번) → SPY 21d long."""
    df = pd.concat([vix.rename("vix"), spy.rename("spy")], axis=1).dropna()
    df["fwd_21d"] = df["spy"].pct_change(21).shift(-21)
    sig = df["vix"] >= 25
    return df, sig, "fwd_21d"


def strat_vix9d_inv(vix, vix9d, spy):
    """VIX9D > VIX (단기 term inversion) → SPY 21d mean reversion."""
    df = pd.concat([vix.rename("vix"), vix9d.rename("vix9d"), spy.rename("spy")], axis=1).dropna()
    df["fwd_21d"] = df["spy"].pct_change(21).shift(-21)
    sig = df["vix9d"] > df["vix"] * 1.05
    return df, sig, "fwd_21d"


def strat_dd_recovery(spy):
    """SPY DD < -15% → 180d long (recovery)."""
    df = pd.DataFrame({"spy": spy})
    df["high_252"] = df["spy"].rolling(252, min_periods=50).max()
    df["dd"] = df["spy"] / df["high_252"] - 1
    df["fwd_180d"] = df["spy"].pct_change(180).shift(-180)
    sig = df["dd"] <= -0.15
    return df, sig, "fwd_180d"


def strat_month_end(spy):
    """월말 (마지막 5거래일) long, 월초 flat — turn-of-month effect."""
    df = pd.DataFrame({"spy": spy})
    df["fwd_5d"] = df["spy"].pct_change(5).shift(-5)
    # 월말 5거래일 식별
    sig = pd.Series(False, index=df.index)
    for ym in df.index.to_period('M').unique():
        month_idx = df.index[df.index.to_period('M') == ym]
        if len(month_idx) >= 5:
            for d in month_idx[-5:]:
                sig.loc[d] = True
    return df, sig, "fwd_5d"


def strat_kospi_panic(ks):
    """KOSPI RV>30 → 180d long (한국 panic 매수)."""
    df = pd.DataFrame({"ks": ks})
    df["rv_21"] = realized_vol(df["ks"], 21) * 100
    df["fwd_180d"] = df["ks"].pct_change(180).shift(-180)
    sig = df["rv_21"] >= 30
    return df, sig, "fwd_180d"


def strat_kospi_dd(ks):
    """KOSPI DD<-20% → 180d long."""
    df = pd.DataFrame({"ks": ks})
    df["high"] = df["ks"].rolling(252, min_periods=50).max()
    df["dd"] = df["ks"] / df["high"] - 1
    df["fwd_180d"] = df["ks"].pct_change(180).shift(-180)
    sig = df["dd"] <= -0.20
    return df, sig, "fwd_180d"


def strat_vix_extreme(vix, spy):
    """VIX > 35 (극단 패닉) → 180d long."""
    df = pd.concat([vix.rename("vix"), spy.rename("spy")], axis=1).dropna()
    df["fwd_180d"] = df["spy"].pct_change(180).shift(-180)
    sig = df["vix"] >= 35
    return df, sig, "fwd_180d"


def strat_vix_calm_bull(vix, spy):
    """VIX < 12 (극저변동성) → 21d long (calm-bull continuation)."""
    df = pd.concat([vix.rename("vix"), spy.rename("spy")], axis=1).dropna()
    df["fwd_21d"] = df["spy"].pct_change(21).shift(-21)
    sig = df["vix"] < 12
    return df, sig, "fwd_21d"


def strat_5d_crash_rev(spy):
    """SPY 5d -5% 이상 급락 → 30d 반등 long."""
    df = pd.DataFrame({"spy": spy})
    df["ret_5d"] = df["spy"].pct_change(5)
    df["fwd_30d"] = df["spy"].pct_change(30).shift(-30)
    sig = df["ret_5d"] <= -0.05
    return df, sig, "fwd_30d"


def strat_slow_drop(spy):
    """SPY 21d -10% 이상 (느린 하락) → 90d long."""
    df = pd.DataFrame({"spy": spy})
    df["ret_21d"] = df["spy"].pct_change(21)
    df["fwd_90d"] = df["spy"].pct_change(90).shift(-90)
    sig = df["ret_21d"] <= -0.10
    return df, sig, "fwd_90d"


def strat_vvix_extreme(vvix, spy):
    """VVIX > 130 (vol of vol 패닉) → 90d long."""
    if vvix is None:
        return None, None, None
    df = pd.concat([vvix.rename("vvix"), spy.rename("spy")], axis=1).dropna()
    df["fwd_90d"] = df["spy"].pct_change(90).shift(-90)
    sig = df["vvix"] >= 130
    return df, sig, "fwd_90d"


def strat_vix9d_strong_inv(vix, vix9d, spy):
    """VIX9D > VIX × 1.10 (강한 단기 inversion) → 90d long."""
    if vix9d is None:
        return None, None, None
    df = pd.concat([vix.rename("vix"), vix9d.rename("vix9d"), spy.rename("spy")], axis=1).dropna()
    df["fwd_90d"] = df["spy"].pct_change(90).shift(-90)
    sig = df["vix9d"] > df["vix"] * 1.10
    return df, sig, "fwd_90d"


def strat_vix_zscore_high(vix, spy, lookback=252):
    """VIX z-score >= 2 (1년 기준 극단) → 60d long."""
    df = pd.concat([vix.rename("vix"), spy.rename("spy")], axis=1).dropna()
    z = (df["vix"] - df["vix"].rolling(lookback).mean()) / df["vix"].rolling(lookback).std()
    df["z"] = z
    df["fwd_60d"] = df["spy"].pct_change(60).shift(-60)
    sig = df["z"] >= 2.0
    return df, sig, "fwd_60d"


def strat_spy_50d_trend(spy, vix):
    """SPY 50d momentum 양수 AND VIX < 20 (calm trend) → 21d long."""
    df = pd.concat([spy.rename("spy"), vix.rename("vix")], axis=1).dropna()
    df["mom_50"] = df["spy"].pct_change(50)
    df["fwd_21d"] = df["spy"].pct_change(21).shift(-21)
    sig = (df["mom_50"] > 0) & (df["vix"] < 20)
    return df, sig, "fwd_21d"


def strat_tlt_inverse(spy, tlt):
    """TLT 21d -3% 하락 (yield 상승) → SPY 60d long (risk-on)."""
    if tlt is None:
        return None, None, None
    df = pd.concat([spy.rename("spy"), tlt.rename("tlt")], axis=1).dropna()
    df["tlt_21d"] = df["tlt"].pct_change(21)
    df["fwd_60d"] = df["spy"].pct_change(60).shift(-60)
    sig = df["tlt_21d"] <= -0.03
    return df, sig, "fwd_60d"


def strat_multi_trigger_and(vix, vvix, spy):
    """VIX>25 AND VVIX>100 AND SPY DD<-5% (3중 panic 합의) → 180d long."""
    if vvix is None:
        return None, None, None
    df = pd.concat([vix.rename("vix"), vvix.rename("vvix"), spy.rename("spy")], axis=1).dropna()
    df["high_252"] = df["spy"].rolling(252, min_periods=50).max()
    df["dd"] = df["spy"] / df["high_252"] - 1
    df["fwd_180d"] = df["spy"].pct_change(180).shift(-180)
    sig = (df["vix"] >= 25) & (df["vvix"] >= 100) & (df["dd"] <= -0.05)
    return df, sig, "fwd_180d"


# ====== Tier-S 알파메일 (학술/프로 검증된 옵션 시그널) ======

def strat_vxx_rolldown_proxy(vix, vix3m, spy):
    """VIX < 17 AND VIX3M > VIX (steep contango) → SPY 21d long. VXX rolldown harvest proxy."""
    if vix3m is None:
        return None, None, None
    df = pd.concat([vix.rename("vix"), vix3m.rename("vix3m"), spy.rename("spy")], axis=1).dropna()
    df["fwd_21d"] = df["spy"].pct_change(21).shift(-21)
    sig = (df["vix"] < 17) & (df["vix3m"] > df["vix"] * 1.05)
    return df, sig, "fwd_21d"


def strat_term_curve_extreme(vix, vix3m, spy):
    """VIX/VIX3M < 0.85 (극단 contango) → SPY 90d long."""
    if vix3m is None:
        return None, None, None
    df = pd.concat([vix.rename("vix"), vix3m.rename("vix3m"), spy.rename("spy")], axis=1).dropna()
    df["ratio"] = df["vix"] / df["vix3m"]
    df["fwd_90d"] = df["spy"].pct_change(90).shift(-90)
    sig = df["ratio"] < 0.85
    return df, sig, "fwd_90d"


def strat_vvix_vix_ratio(vix, vvix, spy):
    """VVIX/VIX > 7 (vol of vol 극단) → SPY 60d long."""
    if vvix is None:
        return None, None, None
    df = pd.concat([vix.rename("vix"), vvix.rename("vvix"), spy.rename("spy")], axis=1).dropna()
    df["ratio"] = df["vvix"] / df["vix"]
    df["fwd_60d"] = df["spy"].pct_change(60).shift(-60)
    sig = df["ratio"] > 7
    return df, sig, "fwd_60d"


def strat_panic_put_write(vix, spy):
    """VIX>30 AND SPY 5d crash<-5% → 60d long. Put-write upside proxy."""
    df = pd.concat([vix.rename("vix"), spy.rename("spy")], axis=1).dropna()
    df["ret_5d"] = df["spy"].pct_change(5)
    df["fwd_60d"] = df["spy"].pct_change(60).shift(-60)
    sig = (df["vix"] >= 30) & (df["ret_5d"] <= -0.05)
    return df, sig, "fwd_60d"


def strat_treasury_panic_flight(spy, tlt):
    """TLT 21d > +7% (안전자산 패닉 유입) → SPY 90d long (반전)."""
    if tlt is None:
        return None, None, None
    df = pd.concat([spy.rename("spy"), tlt.rename("tlt")], axis=1).dropna()
    df["tlt_21d"] = df["tlt"].pct_change(21)
    df["fwd_90d"] = df["spy"].pct_change(90).shift(-90)
    sig = df["tlt_21d"] >= 0.07
    return df, sig, "fwd_90d"


# ====== Tier-S+ 추가 (calm bull/MOVE/yield curve/VIX9D crash/extreme combo) ======

def strat_vix3m_calm_bull(vix3m, spy):
    """VIX3M < 14 (장기 저변동성) → SPY 90d long. Calm-bull continuation."""
    if vix3m is None:
        return None, None, None
    df = pd.concat([vix3m.rename("vix3m"), spy.rename("spy")], axis=1).dropna()
    df["fwd_90d"] = df["spy"].pct_change(90).shift(-90)
    sig = df["vix3m"] < 14
    return df, sig, "fwd_90d"


def strat_tlt_vol_spike(spy, tlt):
    """TLT 21d 변동성 z-score > 2 (MOVE proxy) → SPY 60d long."""
    if tlt is None:
        return None, None, None
    df = pd.concat([spy.rename("spy"), tlt.rename("tlt")], axis=1).dropna()
    tlt_vol = df["tlt"].pct_change().rolling(21).std() * np.sqrt(252) * 100
    z = (tlt_vol - tlt_vol.rolling(252).mean()) / tlt_vol.rolling(252).std()
    df["tlt_z"] = z
    df["fwd_60d"] = df["spy"].pct_change(60).shift(-60)
    sig = df["tlt_z"] >= 2.0
    return df, sig, "fwd_60d"


def strat_yield_curve_proxy(spy, tlt):
    """TLT/IEF proxy 사용 어려움 — TLT 50d가 200d 위 (장기금리 하락 추세) → SPY 30d long."""
    if tlt is None:
        return None, None, None
    df = pd.concat([spy.rename("spy"), tlt.rename("tlt")], axis=1).dropna()
    ma50 = df["tlt"].rolling(50).mean()
    ma200 = df["tlt"].rolling(200).mean()
    df["fwd_30d"] = df["spy"].pct_change(30).shift(-30)
    sig = (ma50 > ma200) & (df["tlt"] > ma50)
    return df, sig, "fwd_30d"


def strat_vix9d_crash_combined(vix, vix9d, spy):
    """VIX9D > 30 AND VIX < VIX9D × 0.85 (단기 급등 + 곧 정상화) → SPY 60d long."""
    if vix9d is None:
        return None, None, None
    df = pd.concat([vix.rename("vix"), vix9d.rename("vix9d"), spy.rename("spy")], axis=1).dropna()
    df["fwd_60d"] = df["spy"].pct_change(60).shift(-60)
    sig = (df["vix9d"] >= 30) & (df["vix"] < df["vix9d"] * 0.85)
    return df, sig, "fwd_60d"


def strat_extreme_combo_panic(vix, vvix, spy, tlt):
    """VIX>30 AND VVIX>120 AND TLT>+5% (3중 panic + flight) → SPY 180d long."""
    if vvix is None or tlt is None:
        return None, None, None
    df = pd.concat([vix.rename("vix"), vvix.rename("vvix"),
                     spy.rename("spy"), tlt.rename("tlt")], axis=1).dropna()
    df["tlt_21d"] = df["tlt"].pct_change(21)
    df["fwd_180d"] = df["spy"].pct_change(180).shift(-180)
    sig = (df["vix"] >= 30) & (df["vvix"] >= 120) & (df["tlt_21d"] >= 0.05)
    return df, sig, "fwd_180d"


# ====== Tier-S++ 추가 ======

def strat_vix_butterfly(vix, vix9d, vix3m, spy):
    """VIX9D < VIX < VIX3M (정상 contango) AND VIX < 18 → SPY 60d long.
    Calm trend continuation 신호."""
    if vix9d is None or vix3m is None:
        return None, None, None
    df = pd.concat([vix.rename("vix"), vix9d.rename("vix9d"),
                     vix3m.rename("vix3m"), spy.rename("spy")], axis=1).dropna()
    df["fwd_60d"] = df["spy"].pct_change(60).shift(-60)
    sig = (df["vix9d"] < df["vix"]) & (df["vix"] < df["vix3m"]) & (df["vix"] < 18)
    return df, sig, "fwd_60d"


def strat_vix_zscore_low(vix, spy, lookback=252):
    """VIX z-score < -1 (1년 기준 저변동성 구간) → SPY 30d long. 'calm bull continuation'."""
    df = pd.concat([vix.rename("vix"), spy.rename("spy")], axis=1).dropna()
    z = (df["vix"] - df["vix"].rolling(lookback).mean()) / df["vix"].rolling(lookback).std()
    df["z"] = z
    df["fwd_30d"] = df["spy"].pct_change(30).shift(-30)
    sig = df["z"] < -1.0
    return df, sig, "fwd_30d"


def strat_spy_above_200ma_vix(spy, vix):
    """SPY 200d MA 위 + VIX < 25 → 90d long. Trend + low fear filter."""
    df = pd.concat([spy.rename("spy"), vix.rename("vix")], axis=1).dropna()
    df["ma200"] = df["spy"].rolling(200).mean()
    df["fwd_90d"] = df["spy"].pct_change(90).shift(-90)
    sig = (df["spy"] > df["ma200"]) & (df["vix"] < 25)
    return df, sig, "fwd_90d"


def strat_vvix_zscore_high(vvix, spy):
    """VVIX z-score > 2 → SPY 60d long. Vol-of-vol panic recovery."""
    if vvix is None:
        return None, None, None
    df = pd.concat([vvix.rename("vvix"), spy.rename("spy")], axis=1).dropna()
    z = (df["vvix"] - df["vvix"].rolling(252).mean()) / df["vvix"].rolling(252).std()
    df["z"] = z
    df["fwd_60d"] = df["spy"].pct_change(60).shift(-60)
    sig = df["z"] >= 2.0
    return df, sig, "fwd_60d"


def strat_panic_kospi_combined(vix, ks, spy):
    """VIX>25 AND KOSPI DD<-10% → 180d long SPY (글로벌 패닉 = 매수)."""
    if ks is None:
        return None, None, None
    df = pd.concat([vix.rename("vix"), ks.rename("ks"), spy.rename("spy")], axis=1).dropna()
    df["ks_high"] = df["ks"].rolling(252, min_periods=50).max()
    df["ks_dd"] = df["ks"] / df["ks_high"] - 1
    df["fwd_180d"] = df["spy"].pct_change(180).shift(-180)
    sig = (df["vix"] >= 25) & (df["ks_dd"] <= -0.10)
    return df, sig, "fwd_180d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    rd = args.round

    spy = fetch_history("SPY", period="20y")["adj_close"]
    vix = fetch_history("^VIX", period="20y")["adj_close"]
    vix9d = fetch_history("^VIX9D", period="20y")
    vix9d_c = vix9d["adj_close"] if not vix9d.empty and "adj_close" in vix9d.columns else None
    vix3m = fetch_history("^VIX3M", period="20y")
    vix3m_c = vix3m["adj_close"] if not vix3m.empty and "adj_close" in vix3m.columns else None
    skew = fetch_history("^SKEW", period="20y")
    skew_c = skew["adj_close"] if not skew.empty and "adj_close" in skew.columns else None
    ks = fetch_history("^KS200", period="20y")
    if ks.empty:
        ks = fetch_history("^KS11", period="20y")
    ks_c = ks["adj_close"] if not ks.empty and "adj_close" in ks.columns else None
    vvix = fetch_history("^VVIX", period="20y")
    vvix_c = vvix["adj_close"] if not vvix.empty and "adj_close" in vvix.columns else None
    tlt = fetch_history("TLT", period="20y")
    tlt_c = tlt["adj_close"] if not tlt.empty and "adj_close" in tlt.columns else None

    strategies = [
        # 1-10 기존
        ("iv_rv_gap", lambda: strat_iv_rv_gap(spy, vix)),
        ("vix_term_inv", lambda: strat_vix_term_inv(vix, vix3m_c, spy) if vix3m_c is not None else (None, None, None)),
        ("skew_low", lambda: strat_skew(skew_c, spy) if skew_c is not None else (None, None, None)),
        ("fomc_drift", lambda: strat_fomc_proxy(spy)),
        ("vix_25_meanrev_21d", lambda: strat_vix_mean_rev(vix, spy)),
        ("vix9d_inv_21d", lambda: strat_vix9d_inv(vix, vix9d_c, spy) if vix9d_c is not None else (None, None, None)),
        ("dd_recovery_180d", lambda: strat_dd_recovery(spy)),
        ("turn_of_month", lambda: strat_month_end(spy)),
        ("kospi_panic_180d", lambda: strat_kospi_panic(ks_c) if ks_c is not None else (None, None, None)),
        ("kospi_dd_180d", lambda: strat_kospi_dd(ks_c) if ks_c is not None else (None, None, None)),
        # 11-20 새 전략
        ("vix_extreme_35_180d", lambda: strat_vix_extreme(vix, spy)),
        ("vix_calm_under12_21d", lambda: strat_vix_calm_bull(vix, spy)),
        ("crash_5d_rev_30d", lambda: strat_5d_crash_rev(spy)),
        ("slow_drop_21d_rev_90d", lambda: strat_slow_drop(spy)),
        ("vvix_extreme_130_90d", lambda: strat_vvix_extreme(vvix_c, spy)),
        ("vix9d_strong_inv_90d", lambda: strat_vix9d_strong_inv(vix, vix9d_c, spy)),
        ("vix_z2_60d", lambda: strat_vix_zscore_high(vix, spy)),
        ("spy_50d_trend_calm", lambda: strat_spy_50d_trend(spy, vix)),
        ("tlt_drop_spy_60d", lambda: strat_tlt_inverse(spy, tlt_c)),
        ("multi_trigger_panic_180d", lambda: strat_multi_trigger_and(vix, vvix_c, spy)),
        # Tier-S 알파메일 (학술/프로 검증)
        ("vxx_rolldown_contango_21d", lambda: strat_vxx_rolldown_proxy(vix, vix3m_c, spy)),
        ("term_curve_extreme_90d", lambda: strat_term_curve_extreme(vix, vix3m_c, spy)),
        ("vvix_vix_ratio_60d", lambda: strat_vvix_vix_ratio(vix, vvix_c, spy)),
        ("panic_putwrite_60d", lambda: strat_panic_put_write(vix, spy)),
        ("treasury_flight_rev_90d", lambda: strat_treasury_panic_flight(spy, tlt_c)),
        # Tier-S+ 추가
        ("vix3m_calm_bull_90d", lambda: strat_vix3m_calm_bull(vix3m_c, spy)),
        ("tlt_vol_spike_60d", lambda: strat_tlt_vol_spike(spy, tlt_c)),
        ("yield_curve_proxy_30d", lambda: strat_yield_curve_proxy(spy, tlt_c)),
        ("vix9d_crash_combined_60d", lambda: strat_vix9d_crash_combined(vix, vix9d_c, spy)),
        ("extreme_combo_panic_180d", lambda: strat_extreme_combo_panic(vix, vvix_c, spy, tlt_c)),
        # Tier-S++ 추가
        ("vix_butterfly_calm_60d", lambda: strat_vix_butterfly(vix, vix9d_c, vix3m_c, spy)),
        ("vix_zscore_low_30d", lambda: strat_vix_zscore_low(vix, spy)),
        ("spy_200ma_vix_below_25_90d", lambda: strat_spy_above_200ma_vix(spy, vix)),
        ("vvix_zscore_high_60d", lambda: strat_vvix_zscore_high(vvix_c, spy)),
        ("panic_kospi_global_180d", lambda: strat_panic_kospi_combined(vix, ks_c, spy)),
    ]
    name, strat_fn = strategies[(rd - 1) % len(strategies)]
    print(f"[diverse round {rd}] {name}")

    df, sig, ret_col = strat_fn()
    if df is None:
        print("  data missing — skip")
        result = {"round": rd, "strategy": name, "skipped": True}
    else:
        cooldown = 30 if "180d" in (ret_col or "") else 7
        r = evaluate_signal(df, sig, ret_col, cooldown=cooldown, min_n=10)
        sh_str = f"{r['sharpe']:+.2f}" if r['sharpe'] is not None else "N<10"
        mean_str = f"{r['mean_pct']:+.2f}" if r['mean_pct'] is not None else "—"
        win_str = f"{r['win_pct']:.0f}" if r['win_pct'] is not None else "—"
        print(f"  raw N={r['raw_n']} indep N={r['n']} mean={mean_str}% win={win_str}% Sh={sh_str}")
        result = {"round": rd, "strategy": name, "horizon": ret_col, **r}

    out_path = RESULTS_DIR / f"iter_diverse_strategies_r{rd}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"  → {out_path.name}")


if __name__ == "__main__":
    main()
