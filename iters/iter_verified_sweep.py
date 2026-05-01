"""verified sweep — 50 configs (Options panic 신호 cooldown 적용)."""
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


CONFIGS = []
# VIX threshold × cooldown × horizon (50 combinations)
for vix_thr in [18, 20, 22, 25, 28, 30, 35]:
    for cd in [21, 30, 60, 90]:
        for h in [21, 63, 180]:
            CONFIGS.append({
                "name": f"VIX>{vix_thr}_cd{cd}_h{h}",
                "vix_thr": vix_thr, "cooldown": cd, "horizon": h,
            })
# 부족분 보충 with VVIX/DD configs
extras = [
    {"name": "VIX>22_VVIX>105_cd30_h180", "vix_thr": 22, "vvix_thr": 105, "cooldown": 30, "horizon": 180},
    {"name": "VIX>25_VVIX>110_cd30_h180", "vix_thr": 25, "vvix_thr": 110, "cooldown": 30, "horizon": 180},
    {"name": "VIX>30_VVIX>120_cd30_h180", "vix_thr": 30, "vvix_thr": 120, "cooldown": 30, "horizon": 180},
    {"name": "VIX>25_DD-5_cd30_h180", "vix_thr": 25, "dd_thr": -0.05, "cooldown": 30, "horizon": 180},
    {"name": "VIX>25_DD-10_cd30_h180", "vix_thr": 25, "dd_thr": -0.10, "cooldown": 30, "horizon": 180},
    {"name": "VIX>30_DD-15_cd30_h180", "vix_thr": 30, "dd_thr": -0.15, "cooldown": 30, "horizon": 180},
    {"name": "VIX>22_VVIX>105_cd60_h180", "vix_thr": 22, "vvix_thr": 105, "cooldown": 60, "horizon": 180},
    {"name": "VIX>25_VVIX>110_cd60_h180", "vix_thr": 25, "vvix_thr": 110, "cooldown": 60, "horizon": 180},
    {"name": "VIX>20_DD-5_cd30_h180", "vix_thr": 20, "dd_thr": -0.05, "cooldown": 30, "horizon": 180},
    {"name": "VIX>20_DD-10_cd30_h180", "vix_thr": 20, "dd_thr": -0.10, "cooldown": 30, "horizon": 180},
    {"name": "VIX>20_VVIX>100_cd30_h180", "vix_thr": 20, "vvix_thr": 100, "cooldown": 30, "horizon": 180},
    {"name": "VIX>30_VVIX>130_cd30_h180", "vix_thr": 30, "vvix_thr": 130, "cooldown": 30, "horizon": 180},
    {"name": "VIX>22_DD-3_cd30_h180", "vix_thr": 22, "dd_thr": -0.03, "cooldown": 30, "horizon": 180},
    {"name": "VIX>25_DD-15_cd30_h180", "vix_thr": 25, "dd_thr": -0.15, "cooldown": 30, "horizon": 180},
    {"name": "VIX>30_DD-10_cd30_h180", "vix_thr": 30, "dd_thr": -0.10, "cooldown": 30, "horizon": 180},
    {"name": "VIX>25_VVIX>105_DD-5_cd30", "vix_thr": 25, "vvix_thr": 105, "dd_thr": -0.05, "cooldown": 30, "horizon": 180},
    {"name": "VIX>30_VVIX>110_DD-10_cd30", "vix_thr": 30, "vvix_thr": 110, "dd_thr": -0.10, "cooldown": 30, "horizon": 180},
    {"name": "VIX>22_VVIX>110_cd30_h63", "vix_thr": 22, "vvix_thr": 110, "cooldown": 30, "horizon": 63},
    {"name": "VIX>25_VVIX>110_cd30_h63", "vix_thr": 25, "vvix_thr": 110, "cooldown": 30, "horizon": 63},
    {"name": "VIX>30_cd180_h180", "vix_thr": 30, "cooldown": 180, "horizon": 180},
    {"name": "VIX>20_cd180_h180", "vix_thr": 20, "cooldown": 180, "horizon": 180},
    {"name": "VIX>25_cd14_h63", "vix_thr": 25, "cooldown": 14, "horizon": 63},
]
# 7*4*3 = 84 -> trim to first 50
CONFIGS = CONFIGS[:50]
# Replace last 22 with extras
CONFIGS = CONFIGS[:28] + extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    rd = args.round

    cfg = CONFIGS[(rd - 1) % len(CONFIGS)]
    print(f"[round {rd}] {cfg['name']}")

    spy = fetch_history("SPY", period="20y")["adj_close"]
    vix = fetch_history("^VIX", period="20y")["adj_close"]
    vvix = fetch_history("^VVIX", period="20y")
    vvix_c = vvix["adj_close"] if not vvix.empty and "adj_close" in vvix.columns else None

    df = pd.concat([spy.rename("spy"), vix.rename("vix")], axis=1).dropna(subset=["spy", "vix"])
    if vvix_c is not None:
        df["vvix"] = vvix_c
    df["spy_252h"] = df["spy"].rolling(252, min_periods=50).max()
    df["dd"] = df["spy"] / df["spy_252h"] - 1
    df["fwd"] = df["spy"].pct_change(cfg["horizon"]).shift(-cfg["horizon"])

    sig = df["vix"] >= cfg["vix_thr"]
    if "vvix_thr" in cfg and "vvix" in df.columns:
        sig &= df["vvix"] >= cfg["vvix_thr"]
    if "dd_thr" in cfg:
        sig &= df["dd"] <= cfg["dd_thr"]

    raw_n = int(sig.sum())
    dedup = deduplicate_events(sig, cooldown_days=cfg["cooldown"])
    events = df[dedup].dropna(subset=["fwd"])
    n = len(events)

    rets = events["fwd"] if n > 0 else pd.Series([], dtype=float)
    if n >= 10:
        sh = event_sharpe(rets)
        # MDD: simulate sequential bets at this signal (assume hold horizon → return on capital)
        # Rolling equity curve
        ordered_rets = rets.sort_index()
        eq = (1 + ordered_rets).cumprod()
        peak = eq.cummax()
        mdd = float((eq / peak - 1).min())
    else:
        sh = None
        mdd = None

    result = {
        "round": rd, "config": cfg["name"], "params": cfg,
        "raw_n": raw_n,
        "independent_n": n,
        "mean_pct": float(rets.mean() * 100) if n > 0 else None,
        "median_pct": float(rets.median() * 100) if n > 0 else None,
        "win_pct": float((rets > 0).mean() * 100) if n > 0 else None,
        "sharpe": float(sh) if sh is not None and not np.isnan(sh) else None,
        "mdd_pct": float(mdd * 100) if mdd is not None else None,
    }
    sh_str = f"{sh:+.2f}" if sh is not None and not np.isnan(sh) else "N<10"
    mdd_str = f"{mdd*100:.1f}%" if mdd is not None else "—"
    mean_str = f"{rets.mean()*100:+.2f}%" if n > 0 else "—"
    win_str = f"{(rets>0).mean()*100:.0f}%" if n > 0 else "—"
    print(f"  raw={raw_n} indep={n} mean={mean_str} win={win_str} Sh={sh_str} MDD={mdd_str}")

    out_path = RESULTS_DIR / f"iter_verified_sweep_r{rd}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
