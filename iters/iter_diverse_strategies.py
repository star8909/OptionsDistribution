"""다양한 옵션 전략 — panic event 외에 다른 alpha source.

10가지 전략:
1. IV-RV gap (vol risk premium)
2. VIX term structure (1m vs 3m contango)
3. SKEW anomaly
4. FOMC drift
5. SPY momentum + VIX overlay
6. VIX mean reversion
7. VIX9D > VIX (term inversion)
8. Earnings IV crush proxy (월말 효과)
9. KOSPI200 specific signal
10. SPY 장기 mean reversion (DD-based)
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

    strategies = [
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
