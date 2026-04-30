"""verified sweep — Options panic 신호 cooldown 적용 (독립 N).

iter22 패턴 따라 cooldown으로 독립 이벤트 확보.
각 round마다 다른 임계값/horizon/cooldown.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    rd = args.round

    configs = [
        {"name": "VIX>20_cd30_180d",  "vix_thr": 20, "cooldown": 30,  "horizon": 180},
        {"name": "VIX>25_cd30_180d",  "vix_thr": 25, "cooldown": 30,  "horizon": 180},
        {"name": "VIX>25_cd60_180d",  "vix_thr": 25, "cooldown": 60,  "horizon": 180},
        {"name": "VIX>30_cd30_180d",  "vix_thr": 30, "cooldown": 30,  "horizon": 180},
        {"name": "VIX>25_cd30_63d",   "vix_thr": 25, "cooldown": 30,  "horizon": 63},
        {"name": "VIX>25_cd30_21d",   "vix_thr": 25, "cooldown": 30,  "horizon": 21},
        {"name": "VIX>22_VVIX>105_cd30", "vix_thr": 22, "vvix_thr": 105, "cooldown": 30, "horizon": 180},
        {"name": "VIX>25_VVIX>110_cd30", "vix_thr": 25, "vvix_thr": 110, "cooldown": 30, "horizon": 180},
        {"name": "VIX>25_DD-5_cd30",  "vix_thr": 25, "dd_thr": -0.05, "cooldown": 30, "horizon": 180},
        {"name": "VIX>25_DD-10_cd30", "vix_thr": 25, "dd_thr": -0.10, "cooldown": 30, "horizon": 180},
    ]
    cfg = configs[(rd - 1) % len(configs)]
    print(f"[round {rd}] {cfg['name']}")

    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    vvix = fetch_history("^VVIX", period="20y", interval="1d")

    spy_c = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_c = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]
    vvix_c = vvix["adj_close"] if "adj_close" in vvix.columns else vvix["close"]

    df = pd.concat([spy_c.rename("spy"), vix_c.rename("vix"), vvix_c.rename("vvix")], axis=1).dropna(subset=["spy", "vix"])
    df["spy_252h"] = df["spy"].rolling(252, min_periods=50).max()
    df["dd"] = df["spy"] / df["spy_252h"] - 1
    df["fwd"] = df["spy"].pct_change(cfg["horizon"]).shift(-cfg["horizon"])

    sig = df["vix"] >= cfg["vix_thr"]
    if "vvix_thr" in cfg:
        sig &= df["vvix"] >= cfg["vvix_thr"]
    if "dd_thr" in cfg:
        sig &= df["dd"] <= cfg["dd_thr"]

    raw_n = int(sig.sum())
    dedup = deduplicate_events(sig, cooldown_days=cfg["cooldown"])
    events = df[dedup].dropna(subset=["fwd"])
    n = len(events)
    if n == 0:
        print("  no events")
        return

    rets = events["fwd"]
    sh = event_sharpe(rets)

    result = {
        "round": rd, "config": cfg["name"], "params": cfg,
        "raw_n": raw_n,
        "independent_n": n,
        "mean_pct": float(rets.mean() * 100),
        "median_pct": float(rets.median() * 100),
        "win_pct": float((rets > 0).mean() * 100),
        "sharpe": float(sh) if not np.isnan(sh) else None,
    }
    sh_str = f"{sh:+.2f}" if not np.isnan(sh) else "N<10"
    print(f"  raw N={raw_n} → independent N={n} mean={result['mean_pct']:+.1f}% "
          f"win={result['win_pct']:.0f}% Sh={sh_str}")

    out_path = RESULTS_DIR / f"iter_verified_sweep_r{rd}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"  → {out_path.name}")


if __name__ == "__main__":
    main()
