#!/usr/bin/env python3
"""
Run a 2019 TLS parameter sweep (1000 configs) against 1Y and 5Y returns.

Windows:
  - 1Y: (2019-12-31, 2020-12-31]
  - 5Y: (2019-12-31, 2024-12-31]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from edgar import build_regex_list, compute_parametric_score, load_terms_from_file


def parse_tickers(path: str) -> List[str]:
    content = Path(path).read_text(encoding="utf-8").replace(",", " ")
    return sorted({t.strip().upper() for t in content.split() if t.strip()})


def recompute_forward_return(
    df_daily: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Compute cumulative return over (start_date, end_date] for each ticker.
    """
    out = []
    for ticker, g in df_daily.groupby("ticker", sort=False):
        w = g[(g["date"] > start_date) & (g["date"] <= end_date)]
        if len(w) == 0:
            out.append((ticker, np.nan))
            continue
        ret = float(np.prod(1.0 + w["ret"].values) - 1.0)
        out.append((ticker, ret))
    return pd.DataFrame(out, columns=["ticker", "ret"])


def build_returns_panel(
    daily_returns_path: str,
    eval_start: str,
    eval_1y_end: str,
    eval_5y_end: str,
) -> pd.DataFrame:
    df = pd.read_csv(daily_returns_path, usecols=["Ticker", "DlyCalDt", "DlyRet"])
    df = df.rename(columns={"Ticker": "ticker", "DlyCalDt": "date", "DlyRet": "ret"})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df = df.dropna(subset=["ticker", "date", "ret"]).copy()

    start = pd.Timestamp(eval_start)
    end_1y = pd.Timestamp(eval_1y_end)
    end_5y = pd.Timestamp(eval_5y_end)

    ret_1y = recompute_forward_return(df, start, end_1y).rename(columns={"ret": "ret_1y"})
    ret_5y = recompute_forward_return(df, start, end_5y).rename(columns={"ret": "ret_5y"})
    panel = ret_1y.merge(ret_5y, on="ticker", how="outer")
    return panel


def load_manifest_candidates(manifest_path: str, tickers: List[str]) -> pd.DataFrame:
    m = pd.read_csv(manifest_path)
    m["ticker"] = m["ticker"].astype(str).str.strip().str.upper()
    m["year"] = pd.to_numeric(m["year"], errors="coerce")
    m = m[m["ticker"].isin(set(tickers))].copy()
    m = m[m["year"] == 2019].copy()
    m = m[m["status"].isin(["cached", "already_cached"])].copy()
    m["filing_date"] = pd.to_datetime(m["filing_date"], errors="coerce")
    # keep most recent 2019 filing per ticker
    m = m.sort_values(["ticker", "filing_date"], ascending=[True, False]).drop_duplicates("ticker")
    m = m.dropna(subset=["cache_path"])
    return m[["ticker", "cik", "accession", "doc", "filing_date", "cache_path"]].copy()


def run_ols(y: pd.Series, x: pd.Series) -> Dict[str, float]:
    if len(y) < 10:
        return {"N": len(y), "beta": np.nan, "t_stat": np.nan, "p_val": np.nan, "r2": np.nan, "adj_r2": np.nan}
    x_std = x.std(ddof=0)
    if x_std == 0 or np.isnan(x_std):
        xz = pd.Series(np.zeros(len(x)), index=x.index)
    else:
        xz = (x - x.mean()) / x_std
    X = sm.add_constant(xz)
    model = sm.OLS(y, X).fit()
    return {
        "N": int(len(y)),
        "beta": float(model.params.iloc[1]) if len(model.params) > 1 else np.nan,
        "t_stat": float(model.tvalues.iloc[1]) if len(model.tvalues) > 1 else np.nan,
        "p_val": float(model.pvalues.iloc[1]) if len(model.pvalues) > 1 else np.nan,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }


def score_distribution(s: pd.Series) -> Dict[str, float]:
    s = s.dropna()
    if len(s) == 0:
        return {
            "mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan,
            "skew": np.nan, "kurtosis": np.nan, "p01": np.nan, "p05": np.nan,
            "p95": np.nan, "p99": np.nan, "zero_share": np.nan, "p99_minus_p1": np.nan
        }
    q = s.quantile([0.01, 0.05, 0.95, 0.99])
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
        "p01": float(q.loc[0.01]),
        "p05": float(q.loc[0.05]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "zero_share": float((s == 0).mean()),
        "p99_minus_p1": float(q.loc[0.99] - q.loc[0.01]),
    }


def _score_one_ticker(
    html: str,
    substantive_pats,
    boilerplate_pats,
    section_weight: float,
    proximity_weight: float,
    proximity_window: int,
    score_mode: str,
) -> Dict[str, float]:
    return compute_parametric_score(
        html=html,
        substantive_pats=substantive_pats,
        boilerplate_pats=boilerplate_pats,
        section_weight=section_weight,
        proximity_weight=proximity_weight,
        proximity_window_chars=proximity_window,
        score_mode=score_mode,
    )


def main():
    ap = argparse.ArgumentParser(description="2019 TLS sweep regression runner")
    ap.add_argument("--tickers-file", type=str, default="tickers.txt")
    ap.add_argument("--cache-dir", type=str, default="cached_edgar")
    ap.add_argument("--cache-manifest", type=str, default=None)
    ap.add_argument("--daily-returns", type=str, default="returns/daily_ret_5y.csv")
    ap.add_argument("--eval-date-start", type=str, default="2019-12-31")
    ap.add_argument("--eval-1y-end", type=str, default="2020-12-31")
    ap.add_argument("--eval-5y-end", type=str, default="2024-12-31")
    ap.add_argument("--window-grid", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default="sweep_outputs_2019")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--substantive-file", type=str, default="substantive_terms.txt")
    ap.add_argument("--boilerplate-file", type=str, default="boilerplate_terms.txt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = parse_tickers(args.tickers_file)
    manifest_path = args.cache_manifest or str(Path(args.cache_dir) / "cache_manifest.csv")
    candidates = load_manifest_candidates(manifest_path, tickers)
    returns_panel = build_returns_panel(
        daily_returns_path=args.daily_returns,
        eval_start=args.eval_date_start,
        eval_1y_end=args.eval_1y_end,
        eval_5y_end=args.eval_5y_end,
    )

    returns_panel["ticker"] = returns_panel["ticker"].astype(str).str.upper()
    returns_panel = returns_panel.set_index("ticker")

    dropped = []
    eligible_rows = []
    candidate_ticker_set = set(candidates["ticker"])
    for t in tickers:
        if t not in candidate_ticker_set:
            dropped.append({"ticker": t, "reason": "no_cached_2019_filing"})
            continue
        if t not in returns_panel.index:
            dropped.append({"ticker": t, "reason": "missing_returns"})
            continue
        r1 = returns_panel.at[t, "ret_1y"]
        r5 = returns_panel.at[t, "ret_5y"]
        if pd.isna(r1):
            dropped.append({"ticker": t, "reason": "missing_1y_return"})
            continue
        if pd.isna(r5):
            dropped.append({"ticker": t, "reason": "missing_5y_return"})
            continue
        eligible_rows.append(t)

    dropped_df = pd.DataFrame(dropped).sort_values(["reason", "ticker"])
    dropped_df.to_csv(out_dir / "dropped_tickers.csv", index=False)

    candidates = candidates[candidates["ticker"].isin(set(eligible_rows))].copy()
    if len(candidates) == 0:
        raise RuntimeError("No eligible tickers after manifest + returns intersection.")

    # preload html into memory once
    html_map: Dict[str, str] = {}
    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Loading cached HTML", unit="filing"):
        p = Path(str(row["cache_path"]))
        if not p.exists():
            continue
        html_map[row["ticker"]] = p.read_text(encoding="utf-8", errors="ignore")

    candidate_tickers = sorted(set(html_map.keys()))
    if len(candidate_tickers) == 0:
        raise RuntimeError("No readable cached HTML files found for eligible tickers.")

    returns_ready = returns_panel.loc[candidate_tickers, ["ret_1y", "ret_5y"]].copy().dropna()
    candidate_tickers = sorted(returns_ready.index.tolist())

    substantive_terms = load_terms_from_file(args.substantive_file)
    boilerplate_terms = load_terms_from_file(args.boilerplate_file)
    substantive_pats = build_regex_list(substantive_terms)
    boilerplate_pats = build_regex_list(boilerplate_terms)

    section_grid = list(range(1, 11))
    proximity_grid = list(range(1, 11))
    window_grid = np.linspace(10, 500, args.window_grid).round().astype(int).tolist()
    window_grid = sorted(set(int(x) for x in window_grid))

    configs: List[Tuple[int, int, int]] = []
    for sw in section_grid:
        for pw in proximity_grid:
            for cw in window_grid:
                configs.append((sw, pw, cw))

    metrics_records = []
    dist_records = []
    modes = ["substantive_only", "net_tls"]

    for sw, pw, cw in tqdm(configs, total=len(configs), desc="Configs", unit="config"):
        config_id = f"sw{sw}_pw{pw}_cw{cw}"
        for mode in tqdm(modes, total=len(modes), desc=f"{config_id} modes", leave=False, unit="mode"):
            if args.n_jobs > 1:
                with ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
                    scored = list(
                        ex.map(
                            lambda t: _score_one_ticker(
                                html_map[t],
                                substantive_pats,
                                boilerplate_pats,
                                float(sw),
                                float(pw),
                                int(cw),
                                mode,
                            ),
                            candidate_tickers,
                        )
                    )
            else:
                scored = [
                    _score_one_ticker(
                        html=html_map[t],
                        substantive_pats=substantive_pats,
                        boilerplate_pats=boilerplate_pats,
                        section_weight=float(sw),
                        proximity_weight=float(pw),
                        proximity_window=int(cw),
                        score_mode=mode,
                    )
                    for t in candidate_tickers
                ]

            scores_df = pd.DataFrame(scored)
            scores_df.insert(0, "ticker", candidate_tickers)
            merged = scores_df.merge(returns_ready.reset_index(), on="ticker", how="inner")

            dist = score_distribution(merged["beta_score"])
            dist_records.append(
                {
                    "config_id": config_id,
                    "score_mode": mode,
                    "year": 2019,
                    "section_weight": sw,
                    "proximity_weight": pw,
                    "proximity_window": cw,
                    **dist,
                }
            )

            reg_1y = run_ols(merged["ret_1y"], merged["beta_score"])
            reg_5y = run_ols(merged["ret_5y"], merged["beta_score"])
            metrics_records.append(
                {
                    "config_id": config_id,
                    "score_mode": mode,
                    "section_weight": sw,
                    "proximity_weight": pw,
                    "proximity_window": cw,
                    "window_type": "1Y",
                    "window_start_year": 2019,
                    "window_end_year": 2020,
                    **reg_1y,
                }
            )
            metrics_records.append(
                {
                    "config_id": config_id,
                    "score_mode": mode,
                    "section_weight": sw,
                    "proximity_weight": pw,
                    "proximity_window": cw,
                    "window_type": "5Y",
                    "window_start_year": 2019,
                    "window_end_year": 2024,
                    **reg_5y,
                }
            )

    metrics_df = pd.DataFrame(metrics_records)
    dist_df = pd.DataFrame(dist_records)
    metrics_df.to_csv(out_dir / "metrics_by_config_window.csv", index=False)
    dist_df.to_csv(out_dir / "distribution_stats_by_config_year.csv", index=False)

    # leaderboard
    metric_pivot = metrics_df.pivot_table(
        index=["config_id", "score_mode", "section_weight", "proximity_weight", "proximity_window"],
        columns="window_type",
        values=["beta", "t_stat", "r2"],
        aggfunc="first",
    )
    metric_pivot.columns = [f"{a}_{b}" for a, b in metric_pivot.columns]
    metric_pivot = metric_pivot.reset_index()
    lb = metric_pivot.merge(
        dist_df[["config_id", "score_mode", "skew", "kurtosis", "p99_minus_p1", "std", "iqr"]],
        on=["config_id", "score_mode"],
        how="left",
    )

    lb["score_predictive_raw"] = lb[["t_stat_1Y", "t_stat_5Y"]].abs().mean(axis=1)
    lb["score_stability_raw"] = np.where(
        np.sign(lb["beta_1Y"]).fillna(0) == np.sign(lb["beta_5Y"]).fillna(0), 1.0, 0.0
    )
    lb["tail_penalty_raw"] = (
        lb["skew"].abs().fillna(0)
        + lb["kurtosis"].abs().fillna(0)
        + lb["p99_minus_p1"].fillna(0) / (lb["iqr"].abs().replace(0, np.nan)).fillna(1.0)
    )

    def minmax(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - lo) / (hi - lo)

    lb["score_predictive"] = minmax(lb["score_predictive_raw"])
    lb["score_stability"] = minmax(lb["score_stability_raw"])
    lb["score_distribution"] = 1.0 - minmax(lb["tail_penalty_raw"])
    lb["final_score"] = (
        0.60 * lb["score_predictive"]
        + 0.25 * lb["score_stability"]
        + 0.15 * lb["score_distribution"]
    )
    lb = lb.sort_values("final_score", ascending=False)
    lb.to_csv(out_dir / "leaderboard.csv", index=False)

    top10 = lb.head(10).copy()
    report_lines = [
        "# 2019 TLS Sweep Decision Report",
        "",
        f"- Universe tickers provided: {len(tickers)}",
        f"- Eligible tickers scored: {len(candidate_tickers)}",
        f"- Configs evaluated: {len(configs)}",
        f"- Modes evaluated per config: {len(modes)}",
        "",
        "## Top 10 Configurations",
        "",
        top10[
            [
                "config_id",
                "score_mode",
                "section_weight",
                "proximity_weight",
                "proximity_window",
                "t_stat_1Y",
                "t_stat_5Y",
                "beta_1Y",
                "beta_5Y",
                "final_score",
            ]
        ].to_markdown(index=False),
        "",
        "## Recommended Production Setting",
        "",
        f"- config_id: `{top10.iloc[0]['config_id']}`",
        f"- score_mode: `{top10.iloc[0]['score_mode']}`",
        f"- section_weight: `{top10.iloc[0]['section_weight']}`",
        f"- proximity_weight: `{top10.iloc[0]['proximity_weight']}`",
        f"- proximity_window: `{top10.iloc[0]['proximity_window']}`",
    ]
    (out_dir / "decision_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "tickers_file": args.tickers_file,
        "cache_dir": args.cache_dir,
        "cache_manifest": manifest_path,
        "daily_returns": args.daily_returns,
        "eval_date_start": args.eval_date_start,
        "eval_1y_end": args.eval_1y_end,
        "eval_5y_end": args.eval_5y_end,
        "section_grid": section_grid,
        "proximity_grid": proximity_grid,
        "window_grid": window_grid,
        "config_count": len(configs),
        "modes": modes,
        "input_ticker_count": len(tickers),
        "eligible_ticker_count": len(candidate_tickers),
        "dropped_ticker_count": int(len(dropped_df)),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[COMPLETE] Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
