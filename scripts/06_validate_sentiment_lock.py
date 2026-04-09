#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from thesis_pipeline.sentiment import compute_forward_1y, evaluate_sentiment_file


SHORTLIST: List[Dict[str, str]] = [
    {
        "label": "core_c15_f256",
        "family": "stable_core",
        "config_id": "finbert_core_c15_h1_filter_off_repeat_on_j088_f256",
        "score_name": "transition_sentiment_median",
        "transform": "winsor_5_95",
        "missing_score_policy": "drop",
    },
    {
        "label": "core_c15_f192",
        "family": "stable_core",
        "config_id": "finbert_core_c15_h1_filter_off_repeat_on_j088_f192",
        "score_name": "transition_sentiment_median",
        "transform": "winsor_5_95",
        "missing_score_policy": "drop",
    },
    {
        "label": "core_c20_f256",
        "family": "stable_core",
        "config_id": "finbert_core_c20_h1_filter_off_repeat_on_j088_f256",
        "score_name": "transition_sentiment_median",
        "transform": "winsor_5_95",
        "missing_score_policy": "drop",
    },
    {
        "label": "core_c20_f192",
        "family": "stable_core",
        "config_id": "finbert_core_c20_h1_filter_off_repeat_on_j088_f192",
        "score_name": "transition_sentiment_median",
        "transform": "winsor_5_95",
        "missing_score_policy": "drop",
    },
    {
        "label": "core_c30_f256",
        "family": "stable_core",
        "config_id": "finbert_core_c30_h1_filter_off_repeat_on_j088_f256",
        "score_name": "transition_sentiment_median",
        "transform": "winsor_5_95",
        "missing_score_policy": "drop",
    },
    {
        "label": "core_c30_f192",
        "family": "stable_core",
        "config_id": "finbert_core_c30_h1_filter_off_repeat_on_j088_f192",
        "score_name": "transition_sentiment_median",
        "transform": "winsor_5_95",
        "missing_score_policy": "drop",
    },
    {
        "label": "full_h2_raw_c15_f256",
        "family": "aggressive_full_raw",
        "config_id": "finbert_full_c15_h2_filter_on_repeat_on_j08_f256",
        "score_name": "transition_sentiment_median",
        "transform": "raw",
        "missing_score_policy": "drop",
    },
    {
        "label": "full_h2_raw_c30_f256",
        "family": "aggressive_full_raw",
        "config_id": "finbert_full_c30_h2_filter_on_repeat_on_j08_f256",
        "score_name": "transition_sentiment_median",
        "transform": "raw",
        "missing_score_policy": "drop",
    },
    {
        "label": "full_h2_w95_c15_f256",
        "family": "aggressive_full_w95",
        "config_id": "finbert_full_c15_h2_filter_on_repeat_off_j088_f256",
        "score_name": "transition_sentiment_median",
        "transform": "winsor_5_95",
        "missing_score_policy": "drop",
    },
    {
        "label": "full_h2_w95_c30_f256",
        "family": "aggressive_full_w95",
        "config_id": "finbert_full_c30_h2_filter_on_repeat_off_j088_f256",
        "score_name": "transition_sentiment_median",
        "transform": "winsor_5_95",
        "missing_score_policy": "drop",
    },
]


def exact_row(sent_df: pd.DataFrame, fwd: pd.DataFrame, *, score_name: str, transform: str, missing_score_policy: str) -> Dict[str, float]:
    res = evaluate_sentiment_file(sent_df, fwd, missing_score_policy=missing_score_policy)
    if res.empty:
        return {
            "N": np.nan,
            "beta": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "adj_r2": np.nan,
        }
    row = res[
        (res["score_name"] == score_name)
        & (res["transform"] == transform)
        & (res["missing_score_policy"] == missing_score_policy)
    ]
    if row.empty:
        return {
            "N": np.nan,
            "beta": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "adj_r2": np.nan,
        }
    r = row.iloc[0]
    return {
        "N": float(r["N"]),
        "beta": float(r["beta"]),
        "t_stat": float(r["t_stat"]),
        "p_value": float(r["p_value"]),
        "adj_r2": float(r["adj_r2"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Holdout and expanding-window CV for shortlisted cached R1000 sentiment models")
    ap.add_argument(
        "--search-dir",
        type=Path,
        default=Path("data/processed/search/sentiment/r1000"),
    )
    ap.add_argument(
        "--returns-file",
        type=Path,
        default=Path("data/raw/returns/daily_ret_10y_full_r1000.csv"),
    )
    ap.add_argument("--train-years", type=str, default="2015-2019")
    ap.add_argument("--test-years", type=str, default="2020-2024")
    ap.add_argument("--min-months-forward", type=int, default=9)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/search/sentiment/r1000/holdout_cv_analysis"),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_years = list(range(2015, 2025))
    fwd_all = compute_forward_1y(args.returns_file, all_years, min_months=int(args.min_months_forward))
    if fwd_all.empty:
        raise RuntimeError("No forward returns found.")

    train_years = list(range(2015, 2020))
    test_years = list(range(2020, 2025))

    shortlist_df = pd.DataFrame(SHORTLIST)
    shortlist_df.to_csv(args.out_dir / "shortlist_models.csv", index=False)

    holdout_rows: List[Dict[str, object]] = []
    cv_rows: List[Dict[str, object]] = []

    for spec in SHORTLIST:
        sent_path = args.search_dir / "runs" / spec["config_id"] / "sentiment.csv"
        sent_df = pd.read_csv(sent_path)

        train_res = exact_row(
            sent_df,
            fwd_all[fwd_all["year"].isin(train_years)].copy(),
            score_name=spec["score_name"],
            transform=spec["transform"],
            missing_score_policy=spec["missing_score_policy"],
        )
        test_res = exact_row(
            sent_df,
            fwd_all[fwd_all["year"].isin(test_years)].copy(),
            score_name=spec["score_name"],
            transform=spec["transform"],
            missing_score_policy=spec["missing_score_policy"],
        )
        holdout_rows.append({
            **spec,
            "split": "train_2015_2019",
            **train_res,
        })
        holdout_rows.append({
            **spec,
            "split": "test_2020_2024",
            **test_res,
        })

        for val_year in range(2019, 2025):
            fold_train_years = list(range(2015, val_year))
            fold_val_years = [val_year]
            tr = exact_row(
                sent_df,
                fwd_all[fwd_all["year"].isin(fold_train_years)].copy(),
                score_name=spec["score_name"],
                transform=spec["transform"],
                missing_score_policy=spec["missing_score_policy"],
            )
            va = exact_row(
                sent_df,
                fwd_all[fwd_all["year"].isin(fold_val_years)].copy(),
                score_name=spec["score_name"],
                transform=spec["transform"],
                missing_score_policy=spec["missing_score_policy"],
            )
            cv_rows.append({
                **spec,
                "validation_year": val_year,
                "train_years": f"2015-{val_year-1}",
                "train_N": tr["N"],
                "train_beta": tr["beta"],
                "train_t_stat": tr["t_stat"],
                "train_p_value": tr["p_value"],
                "val_N": va["N"],
                "val_beta": va["beta"],
                "val_t_stat": va["t_stat"],
                "val_p_value": va["p_value"],
            })

    holdout_df = pd.DataFrame(holdout_rows)
    cv_df = pd.DataFrame(cv_rows)

    holdout_df.to_csv(args.out_dir / "holdout_results.csv", index=False)
    cv_df.to_csv(args.out_dir / "expanding_window_cv_results.csv", index=False)

    test_df = holdout_df[holdout_df["split"] == "test_2020_2024"].copy()
    cv_summary = (
        cv_df.groupby(["label", "family", "config_id", "score_name", "transform", "missing_score_policy"], as_index=False)
        .agg(
            cv_val_years=("validation_year", "count"),
            cv_negative_folds=("val_beta", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
            cv_negative_sig_10=("val_p_value", lambda s: 0),
            cv_negative_sig_5=("val_p_value", lambda s: 0),
            cv_mean_val_beta=("val_beta", "mean"),
            cv_median_val_beta=("val_beta", "median"),
            cv_mean_val_p=("val_p_value", "mean"),
            cv_median_val_p=("val_p_value", "median"),
            cv_min_val_beta=("val_beta", "min"),
            cv_max_val_beta=("val_beta", "max"),
            cv_mean_val_N=("val_N", "mean"),
        )
    )

    # fill significance counts cleanly
    sig5 = (
        cv_df.assign(_neg_sig_5=(pd.to_numeric(cv_df["val_beta"], errors="coerce") < 0) & (pd.to_numeric(cv_df["val_p_value"], errors="coerce") < 0.05))
        .groupby("label")["_neg_sig_5"].sum().astype(int)
    )
    sig10 = (
        cv_df.assign(_neg_sig_10=(pd.to_numeric(cv_df["val_beta"], errors="coerce") < 0) & (pd.to_numeric(cv_df["val_p_value"], errors="coerce") < 0.10))
        .groupby("label")["_neg_sig_10"].sum().astype(int)
    )
    cv_summary["cv_negative_sig_5"] = cv_summary["label"].map(sig5).fillna(0).astype(int)
    cv_summary["cv_negative_sig_10"] = cv_summary["label"].map(sig10).fillna(0).astype(int)

    summary = test_df.merge(
        cv_summary,
        on=["label", "family", "config_id", "score_name", "transform", "missing_score_policy"],
        how="left",
    )
    summary = summary.rename(columns={
        "N": "test_N",
        "beta": "test_beta",
        "t_stat": "test_t_stat",
        "p_value": "test_p_value",
        "adj_r2": "test_adj_r2",
    })
    summary = summary.sort_values(["test_p_value", "cv_negative_sig_5", "cv_median_val_p"], ascending=[True, False, True])
    summary.to_csv(args.out_dir / "candidate_stability_summary.csv", index=False)

    lines = []
    lines.append("# R1000 Sentiment Holdout and Expanding-Window CV\n")
    lines.append("## Setup\n")
    lines.append("- Selection shortlist: 10 exact model variants\n")
    lines.append("- Holdout train: 2015-2019\n")
    lines.append("- Holdout test: 2020-2024\n")
    lines.append("- Expanding-window validation years: 2019-2024\n")
    lines.append("\n## Ranked Summary\n")
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['label']} ({r['family']}): test beta={r['test_beta']:.4f}, test p={r['test_p_value']:.4g}, "
            f"CV neg folds={int(r['cv_negative_folds'])}/{int(r['cv_val_years'])}, "
            f"CV neg sig @5%={int(r['cv_negative_sig_5'])}, median CV p={r['cv_median_val_p']:.4g}\n"
        )
    (args.out_dir / "holdout_cv_summary.md").write_text("".join(lines), encoding="utf-8")

    print(f"[OUT] {args.out_dir / 'shortlist_models.csv'}")
    print(f"[OUT] {args.out_dir / 'holdout_results.csv'}")
    print(f"[OUT] {args.out_dir / 'expanding_window_cv_results.csv'}")
    print(f"[OUT] {args.out_dir / 'candidate_stability_summary.csv'}")
    print(f"[OUT] {args.out_dir / 'holdout_cv_summary.md'}")


if __name__ == "__main__":
    main()
