"""iter09: FOMC 일자 효과 (vol crush + drift).

가설:
- FOMC 회의 직전 IV 상승 (불확실성)
- 회의 직후 IV crush (불확실성 해소)
- 평균 SPY drift 양수 (학술 정설 — Bernanke effect)

데이터: FOMC 일자 직접 정의 (8회/년)
분석:
- FOMC 직전 1주 SPY return
- FOMC 당일 SPY return
- FOMC 직후 1주 SPY return
- VIX FOMC day 변화
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


# FOMC 회의 일자 (2014~2025) — 일부 (8회/년 평균)
FOMC_DATES = [
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]


def main():
    print("[iter09] FOMC 일자 효과 분석")

    spy = fetch_history("SPY", period="20y", interval="1d")
    vix = fetch_history("^VIX", period="20y", interval="1d")
    if spy.empty:
        print("  ❌ 데이터 없음")
        return

    spy_close = spy["adj_close"] if "adj_close" in spy.columns else spy["close"]
    vix_close = vix["adj_close"] if "adj_close" in vix.columns else vix["close"]

    df = pd.concat([spy_close.rename("spy"), vix_close.rename("vix")], axis=1).dropna()
    df["spy_ret_1d"] = df["spy"].pct_change()
    df["vix_change_1d"] = df["vix"].diff()

    # FOMC 일자 표시
    fomc_set = set(pd.to_datetime(FOMC_DATES))
    df["is_fomc"] = df.index.isin(fomc_set)
    fomc_count = df["is_fomc"].sum()
    print(f"  FOMC 일자 매칭: {fomc_count} / {len(FOMC_DATES)}")

    # FOMC day return
    fomc_day = df[df["is_fomc"]]
    print(f"\n=== FOMC day SPY return ===")
    print(f"  N: {len(fomc_day)}")
    print(f"  평균 SPY 1d: {fomc_day['spy_ret_1d'].mean()*100:+.3f}%")
    print(f"  std: {fomc_day['spy_ret_1d'].std()*100:.3f}%")
    print(f"  양수 비율: {(fomc_day['spy_ret_1d'] > 0).sum() / len(fomc_day) * 100:.1f}%")
    print(f"  평균 VIX change: {fomc_day['vix_change_1d'].mean():.3f}")

    # 비교: 비-FOMC day
    non_fomc = df[~df["is_fomc"]]
    print(f"\n=== 비교: 비-FOMC day ===")
    print(f"  평균 SPY 1d: {non_fomc['spy_ret_1d'].mean()*100:+.3f}%")
    print(f"  std: {non_fomc['spy_ret_1d'].std()*100:.3f}%")

    # FOMC 후 5d return
    fomc_idx = df.index[df["is_fomc"]]
    after_5d_returns = []
    after_21d_returns = []
    for fd in fomc_idx:
        try:
            pos = df.index.get_loc(fd)
            if pos + 5 < len(df):
                ret_5d = (df["spy"].iloc[pos+5] / df["spy"].iloc[pos]) - 1
                after_5d_returns.append(ret_5d)
            if pos + 21 < len(df):
                ret_21d = (df["spy"].iloc[pos+21] / df["spy"].iloc[pos]) - 1
                after_21d_returns.append(ret_21d)
        except Exception:
            pass

    print(f"\n=== FOMC 후 SPY return ===")
    if after_5d_returns:
        avg5 = np.mean(after_5d_returns) * 100
        win5 = sum(1 for r in after_5d_returns if r > 0) / len(after_5d_returns) * 100
        print(f"  +5d: 평균 {avg5:+.2f}% (win {win5:.1f}%) [N={len(after_5d_returns)}]")
    if after_21d_returns:
        avg21 = np.mean(after_21d_returns) * 100
        win21 = sum(1 for r in after_21d_returns if r > 0) / len(after_21d_returns) * 100
        print(f"  +21d: 평균 {avg21:+.2f}% (win {win21:.1f}%) [N={len(after_21d_returns)}]")

    # FOMC 직전 (5d before)
    before_5d_returns = []
    for fd in fomc_idx:
        try:
            pos = df.index.get_loc(fd)
            if pos - 5 > 0:
                ret_5d = (df["spy"].iloc[pos] / df["spy"].iloc[pos-5]) - 1
                before_5d_returns.append(ret_5d)
        except Exception:
            pass

    if before_5d_returns:
        avg = np.mean(before_5d_returns) * 100
        win = sum(1 for r in before_5d_returns if r > 0) / len(before_5d_returns) * 100
        print(f"\n=== FOMC 5d 직전 ===")
        print(f"  평균 {avg:+.2f}% (win {win:.1f}%) [N={len(before_5d_returns)}]")

    out = {
        "fomc_count": int(fomc_count),
        "fomc_day_avg_ret": float(fomc_day['spy_ret_1d'].mean()) if len(fomc_day) > 0 else None,
        "after_5d_avg": float(np.mean(after_5d_returns)) if after_5d_returns else None,
        "after_21d_avg": float(np.mean(after_21d_returns)) if after_21d_returns else None,
    }
    out_path = RESULTS_DIR / "iter09_fomc.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
