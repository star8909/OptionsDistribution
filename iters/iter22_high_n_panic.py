"""iter22: High-N panic signals (통계 신뢰성 확보).

iter21 N=19 통계 신뢰 불가. 목표: N≥30 + Sharpe≥2 + Win≥75%.

전략:
- 임계값 완화 (VIX>25 등) + 짧은 cooldown (60-90d)
- 단일 신호 baseline 재검증 (VIX 단독, KOSPI 단독)
- 21d/63d return window 추가 (180d만이 아닌)
- robustness sweep — 어떤 임계값이 N과 Sharpe trade-off 최적
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import fetch_history, deduplicate_events, event_sharpe


def realized_vol(prices, window):
    rets = prices.pct_change()
    return rets.rolling(window).std() * np.sqrt(252)


def evaluate_signal(df, signal_col, return_cols, cooldown_days, label):
    """Evaluate a panic signal: N, mean return, Sharpe, win rate."""
    sig = df[signal_col]
    raw_n = int(sig.sum())
    dedup = deduplicate_events(sig, cooldown_days=cooldown_days)
    events = df[dedup].dropna(subset=return_cols)
    n = len(events)
    if n == 0:
        return None
    result = {
        "label": label,
        "raw_n": raw_n,
        "n": n,
        "cooldown_days": cooldown_days,
    }
    for col in return_cols:
        rets = events[col].dropna()
        if len(rets) == 0:
            continue
        sh = event_sharpe(rets)
        result[f"{col}_mean"] = float(rets.mean() * 100)
        result[f"{col}_win"] = float((rets > 0).mean() * 100)
        result[f"{col}_sharpe"] = float(sh) if not np.isnan(sh) else None
    return result


def main():
    print("[iter22] High-N panic signals — 통계 신뢰성 우선")

    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    vvix = fetch_history("^VVIX", period="20y", interval="1d")
    ks = fetch_history("^KS200", period="20y", interval="1d")
    if ks.empty:
        ks = fetch_history("^KS11", period="20y", interval="1d")

    if any([d.empty for d in [spy, vix]]):
        print("  ❌ 데이터 부재")
        return

    spy_c = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_c = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    vvix_c = vvix["adj_close"] if "adj_close" in vvix.columns else vvix["close"]
    ks_c = ks["adj_close"] if "adj_close" in ks.columns else ks["close"]

    # US dataframe
    us = pd.concat([spy_c.rename("spy"), vix_c.rename("vix"), vvix_c.rename("vvix")], axis=1).dropna(subset=["spy", "vix"])
    us["spy_252h"] = us["spy"].rolling(252, min_periods=50).max()
    us["dd"] = us["spy"] / us["spy_252h"] - 1
    us["spy_21d"] = us["spy"].pct_change(21).shift(-21)
    us["spy_63d"] = us["spy"].pct_change(63).shift(-63)
    us["spy_180d"] = us["spy"].pct_change(180).shift(-180)

    # 다양한 panic 임계값 sweep
    us_signals = [
        # 단일 VIX baseline
        ("VIX>25", us["vix"] >= 25),
        ("VIX>30", us["vix"] >= 30),
        ("VIX>35", us["vix"] >= 35),
        # VIX + DD 결합
        ("VIX>25 & DD<-10%", (us["vix"] >= 25) & (us["dd"] <= -0.10)),
        ("VIX>30 & DD<-10%", (us["vix"] >= 30) & (us["dd"] <= -0.10)),
        ("VIX>30 & DD<-15%", (us["vix"] >= 30) & (us["dd"] <= -0.15)),
        # VVIX 추가
        ("VIX>25 & VVIX>110", (us["vix"] >= 25) & (us["vvix"] >= 110)),
        ("VIX>30 & VVIX>120", (us["vix"] >= 30) & (us["vvix"] >= 120)),
    ]

    return_cols = ["spy_21d", "spy_63d", "spy_180d"]
    cooldowns = [30, 60, 90, 180]

    print(f"\n=== US signals — N vs Sharpe sweep ===")
    print(f"{'signal':<32} {'cd':>4} {'N':>4} {'180d_mean%':>10} {'180d_win%':>10} {'180d_Sh':>8}")
    print("-" * 90)
    us_results = []
    for label, signal in us_signals:
        sig_col = f"sig_{label}"
        us[sig_col] = signal
        for cd in cooldowns:
            r = evaluate_signal(us, sig_col, return_cols, cd, label)
            if r is None:
                continue
            sh180 = r.get("spy_180d_sharpe")
            sh_str = f"{sh180:+.2f}" if sh180 is not None else "N<10"
            mean180 = r.get("spy_180d_mean", 0)
            win180 = r.get("spy_180d_win", 0)
            marker = " ✅" if r["n"] >= 30 and (sh180 or 0) >= 2 else ""
            print(f"{label:<32} {cd:>4} {r['n']:>4} {mean180:>+10.1f} {win180:>10.0f} {sh_str:>8}{marker}")
            us_results.append(r)

    # KR analysis
    if not ks.empty:
        kr = pd.DataFrame({"ks": ks_c})
        kr["rv_21"] = realized_vol(kr["ks"], 21) * 100
        kr["ks_252h"] = kr["ks"].rolling(252, min_periods=50).max()
        kr["ks_dd"] = kr["ks"] / kr["ks_252h"] - 1
        kr["ks_21d"] = kr["ks"].pct_change(21).shift(-21)
        kr["ks_63d"] = kr["ks"].pct_change(63).shift(-63)
        kr["ks_180d"] = kr["ks"].pct_change(180).shift(-180)

        kr_signals = [
            ("KR RV>25", kr["rv_21"] >= 25),
            ("KR RV>30", kr["rv_21"] >= 30),
            ("KR RV>35", kr["rv_21"] >= 35),
            ("KR DD<-15%", kr["ks_dd"] <= -0.15),
            ("KR DD<-20%", kr["ks_dd"] <= -0.20),
            ("KR DD<-15% & RV>25", (kr["ks_dd"] <= -0.15) & (kr["rv_21"] >= 25)),
            ("KR DD<-20% & RV>30", (kr["ks_dd"] <= -0.20) & (kr["rv_21"] >= 30)),
        ]

        kr_return_cols = ["ks_21d", "ks_63d", "ks_180d"]
        print(f"\n=== KR signals — N vs Sharpe sweep ===")
        print(f"{'signal':<32} {'cd':>4} {'N':>4} {'180d_mean%':>10} {'180d_win%':>10} {'180d_Sh':>8}")
        print("-" * 90)
        kr_results = []
        for label, signal in kr_signals:
            sig_col = f"sig_{label}"
            kr[sig_col] = signal
            for cd in cooldowns:
                r = evaluate_signal(kr, sig_col, kr_return_cols, cd, label)
                if r is None:
                    continue
                sh180 = r.get("ks_180d_sharpe")
                sh_str = f"{sh180:+.2f}" if sh180 is not None else "N<10"
                mean180 = r.get("ks_180d_mean", 0)
                win180 = r.get("ks_180d_win", 0)
                marker = " ✅" if r["n"] >= 30 and (sh180 or 0) >= 2 else ""
                print(f"{label:<32} {cd:>4} {r['n']:>4} {mean180:>+10.1f} {win180:>10.0f} {sh_str:>8}{marker}")
                kr_results.append(r)
    else:
        kr_results = []

    # 챔피언 후보 추출
    candidates = []
    for r in us_results:
        sh = r.get("spy_180d_sharpe") or 0
        if r["n"] >= 30 and sh >= 1.5:
            candidates.append({
                "market": "US",
                "signal": r["label"],
                "cooldown": r["cooldown_days"],
                "n": r["n"],
                "180d_mean_pct": r.get("spy_180d_mean"),
                "180d_win_pct": r.get("spy_180d_win"),
                "180d_sharpe": sh,
            })
    for r in kr_results:
        sh = r.get("ks_180d_sharpe") or 0
        if r["n"] >= 30 and sh >= 1.5:
            candidates.append({
                "market": "KR",
                "signal": r["label"],
                "cooldown": r["cooldown_days"],
                "n": r["n"],
                "180d_mean_pct": r.get("ks_180d_mean"),
                "180d_win_pct": r.get("ks_180d_win"),
                "180d_sharpe": sh,
            })

    candidates.sort(key=lambda c: (c["180d_sharpe"], c["n"]), reverse=True)

    print(f"\n{'='*90}")
    print(f"🏆 챔피언 후보 (N≥30 AND Sharpe≥1.5):")
    print(f"{'='*90}")
    if not candidates:
        print("  ❌ 조건 만족 없음 — N과 Sharpe trade-off 한계")
    else:
        for i, c in enumerate(candidates[:8], 1):
            print(f"  #{i} [{c['market']}] {c['signal']} cd={c['cooldown']}d N={c['n']} "
                  f"180d {c['180d_mean_pct']:+.1f}% Win{c['180d_win_pct']:.0f}% Sh={c['180d_sharpe']:+.2f}")

    out = {
        "iter": "iter22_high_n_panic",
        "us_results": us_results,
        "kr_results": kr_results,
        "champions": candidates[:8],
    }
    out_path = RESULTS_DIR / "iter22_high_n_panic.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
