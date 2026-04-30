"""iter23: Short-horizon panic signals (21d/63d) — N≥30 확보 가능.

iter22: 180d horizon은 cooldown≥180d 필요 → N=20 한계.
21d horizon + cooldown=21d → 거의 독립 + N 다수 확보 가능.

목표: N≥30 + Sharpe≥1.5 in 21d/63d window.
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


def evaluate(df, signal_col, ret_col, cooldown):
    sig = df[signal_col]
    raw_n = int(sig.sum())
    dedup = deduplicate_events(sig, cooldown_days=cooldown)
    events = df[dedup].dropna(subset=[ret_col])
    n = len(events)
    if n == 0:
        return None
    rets = events[ret_col]
    sh = event_sharpe(rets)
    return {
        "raw_n": raw_n, "n": n, "cd": cooldown,
        "mean_pct": float(rets.mean() * 100),
        "win_pct": float((rets > 0).mean() * 100),
        "sharpe": float(sh) if not np.isnan(sh) else None,
        "median_pct": float(rets.median() * 100),
    }


def main():
    print("[iter23] Short-horizon panic signals (21d/63d) — N≥30 가능")

    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    vvix = fetch_history("^VVIX", period="20y", interval="1d")

    spy_c = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_c = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    vvix_c = vvix["adj_close"] if "adj_close" in vvix.columns else vvix["close"]

    df = pd.concat([spy_c.rename("spy"), vix_c.rename("vix"), vvix_c.rename("vvix")], axis=1).dropna(subset=["spy", "vix"])
    df["spy_252h"] = df["spy"].rolling(252, min_periods=50).max()
    df["dd"] = df["spy"] / df["spy_252h"] - 1
    df["spy_5d"] = df["spy"].pct_change(5).shift(-5)
    df["spy_21d"] = df["spy"].pct_change(21).shift(-21)
    df["spy_63d"] = df["spy"].pct_change(63).shift(-63)

    signals = [
        ("VIX>20", df["vix"] >= 20),
        ("VIX>25", df["vix"] >= 25),
        ("VIX>30", df["vix"] >= 30),
        ("VIX>22 & DD<-5%", (df["vix"] >= 22) & (df["dd"] <= -0.05)),
        ("VIX>25 & DD<-5%", (df["vix"] >= 25) & (df["dd"] <= -0.05)),
        ("VIX>25 & DD<-7%", (df["vix"] >= 25) & (df["dd"] <= -0.07)),
        ("VIX>22 & VVIX>105", (df["vix"] >= 22) & (df["vvix"] >= 105)),
        ("VIX>25 & VVIX>110", (df["vix"] >= 25) & (df["vvix"] >= 110)),
        # spike: VIX 5d 변화량 큰 경우
        ("VIX 5d Δ>+30%", df["vix"].pct_change(5) >= 0.30),
        ("VIX 5d Δ>+50%", df["vix"].pct_change(5) >= 0.50),
    ]

    horizons = [
        ("spy_5d", 7),     # 5d horizon, cd 7d (거의 독립)
        ("spy_21d", 21),   # 21d horizon, cd 21d
        ("spy_63d", 63),   # 63d horizon, cd 63d
    ]

    results = []
    print(f"\n{'signal':<28} {'horizon':>8} {'cd':>4} {'N':>4} {'mean%':>8} {'win%':>6} {'Sh':>7}")
    print("-" * 80)
    for label, signal in signals:
        sig_col = f"sig_{label}"
        df[sig_col] = signal
        for ret_col, cd in horizons:
            r = evaluate(df, sig_col, ret_col, cd)
            if r is None or r["n"] < 5:
                continue
            sh = r["sharpe"]
            sh_str = f"{sh:+.2f}" if sh is not None else "N<10"
            marker = ""
            if r["n"] >= 30 and (sh or 0) >= 1.5:
                marker = " ✅"
            elif r["n"] >= 20 and (sh or 0) >= 2.0:
                marker = " 🥈"
            print(f"{label:<28} {ret_col:>8} {cd:>4} {r['n']:>4} {r['mean_pct']:>+8.2f} {r['win_pct']:>6.0f} {sh_str:>7}{marker}")
            results.append({
                "signal": label, "horizon": ret_col, **r,
            })

    # 챔피언
    candidates = [r for r in results
                  if r["n"] >= 30 and (r["sharpe"] or 0) >= 1.5]
    candidates.sort(key=lambda c: c["sharpe"], reverse=True)

    print(f"\n{'='*80}")
    print(f"🏆 챔피언 후보 (N≥30 AND Sharpe≥1.5):")
    if not candidates:
        # 차선 (N≥20 AND Sh≥2)
        candidates = [r for r in results
                      if r["n"] >= 20 and (r["sharpe"] or 0) >= 2.0]
        candidates.sort(key=lambda c: c["sharpe"], reverse=True)
        print("  (차선 기준 N≥20 AND Sharpe≥2.0)")
    if not candidates:
        candidates = [r for r in results if r["sharpe"] is not None]
        candidates.sort(key=lambda c: (c["n"] >= 20, c["sharpe"] or 0), reverse=True)
        print("  (차선x2 기준 — 모든 통계 유효 결과)")

    for i, c in enumerate(candidates[:10], 1):
        print(f"  #{i} {c['signal']:<28} h={c['horizon']:<8} N={c['n']:>3} "
              f"{c['mean_pct']:+.2f}% Win{c['win_pct']:.0f}% Sh={c['sharpe']:+.2f}")

    out = {
        "iter": "iter23_short_horizon",
        "results": results,
        "champions": candidates[:10],
    }
    out_path = RESULTS_DIR / "iter23_short_horizon.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
