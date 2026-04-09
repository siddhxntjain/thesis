#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "search" / "tls" / "shared"

SW_GRID = [1.0, 3.25, 5.5, 7.75, 10.0]
PW_GRID = [1.0, 3.25, 5.5, 7.75, 10.0]
CW_GRID = [10, 132, 255, 378, 500]


@dataclass(frozen=True)
class UniverseConfig:
    name: str
    feature_cache_dir: Path
    tickers_file: Path
    tls_results_file: Path


UNIVERSES = [
    UniverseConfig(
        name="t100",
        feature_cache_dir=ROOT / "data" / "processed" / "feature_cache" / "r1000" / "t100",
        tickers_file=ROOT / "data" / "raw" / "universe" / "transition_100_tickers.txt",
        tls_results_file=ROOT / "data" / "processed" / "search" / "tls" / "t100" / "all_results.csv",
    ),
    UniverseConfig(
        name="r1000",
        feature_cache_dir=ROOT / "data" / "processed" / "feature_cache" / "r1000",
        tickers_file=ROOT / "data" / "raw" / "universe" / "tickers.txt",
        tls_results_file=ROOT / "data" / "processed" / "search" / "tls" / "r1000" / "all_results.csv",
    ),
]

CURRENT_SHARED_SPEC = (1.0, 1.0, 10)
MILD_THEORY_SPEC = (3.25, 1.0, 10)
SCALE_FREE_OBJECTIVES = [
    "avg_nonzero_share",
    "avg_sign_entropy",
    "avg_abs_z_gt_0_5_share",
    "avg_abs_z_gt_1_share",
    "avg_deciles_filled",
]
NONZERO_FRONTIER_GAP = 0.01

OBJECTIVE_META = {
    "avg_year_std": {
        "label": "Raw standard deviation",
        "scale_sensitive": True,
        "description": "Average within-year TLS standard deviation. Useful as a baseline, but mechanically increases when weights are scaled up.",
    },
    "avg_year_iqr": {
        "label": "Raw interquartile range",
        "scale_sensitive": True,
        "description": "Average within-year IQR of TLS. Also scale-sensitive, but less driven by outliers than raw variance.",
    },
    "avg_year_p90_p10": {
        "label": "Raw P90-P10 range",
        "scale_sensitive": True,
        "description": "Average within-year difference between the 90th and 10th percentile. Scale-sensitive and dominated by large weights.",
    },
    "avg_nonzero_share": {
        "label": "Nonzero share",
        "scale_sensitive": False,
        "description": "Average within-year share of firms with nonzero net TLS. Directly measures how often the signal avoids collapsing at zero.",
    },
    "avg_sign_entropy": {
        "label": "Sign entropy",
        "scale_sensitive": False,
        "description": "Average within-year entropy of the negative / zero / positive split, normalized to [0,1]. Higher means the sign distribution is less clustered.",
    },
    "avg_abs_z_gt_0_5_share": {
        "label": "Share with |z| > 0.5",
        "scale_sensitive": False,
        "description": "Average within-year share of firms more than 0.5 within-year standard deviations away from the mean. Captures shape-based dispersion after standardization.",
    },
    "avg_abs_z_gt_1_share": {
        "label": "Share with |z| > 1.0",
        "scale_sensitive": False,
        "description": "Average within-year share of firms more than 1 within-year standard deviation away from the mean.",
    },
    "avg_deciles_filled": {
        "label": "Deciles filled",
        "scale_sensitive": False,
        "description": "Average number of populated decile buckets under qcut. Useful for portfolio-sort feasibility when ties are present.",
    },
    "blend_scale_free": {
        "label": "Scale-free blend",
        "scale_sensitive": False,
        "description": "Average rank across nonzero share, sign entropy, |z|>0.5 share, |z|>1 share, and deciles filled.",
    },
}


def parse_tickers(path: Path) -> list[str]:
    return sorted({x.strip().upper() for x in path.read_text().replace(",", " ").split() if x.strip()})


def load_feature_rows(universe: UniverseConfig) -> pd.DataFrame:
    tickers = set(parse_tickers(universe.tickers_file))
    keep_base = ["ticker", "tokens", "sub_total", "sub_in_section", "bp_total", "bp_in_section"]
    rows: list[pd.DataFrame] = []
    for path in sorted(universe.feature_cache_dir.glob("feature_cache_*.csv")):
        suffix = path.stem.split("_")[-1]
        if not suffix.isdigit():
            continue
        year = int(suffix)
        df = pd.read_csv(path)
        if df.empty or "ticker" not in df.columns:
            continue
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df[df["ticker"].isin(tickers)].copy()
        if df.empty:
            continue
        cols = keep_base + [c for c in df.columns if c.startswith("sub_near_") or c.startswith("sub_both_")]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].copy()
        df["signal_year"] = year
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No feature rows found for {universe.name}.")
    return pd.concat(rows, ignore_index=True).sort_values(["signal_year", "ticker"]).reset_index(drop=True)


def score_tls(feature_df: pd.DataFrame, sw: float, pw: float, cw: int) -> pd.Series:
    near_col = f"sub_near_{cw}"
    both_col = f"sub_both_{cw}"
    tok = pd.to_numeric(feature_df["tokens"], errors="coerce").clip(lower=1)
    sub_raw = (
        pd.to_numeric(feature_df["sub_total"], errors="coerce")
        + sw * pd.to_numeric(feature_df["sub_in_section"], errors="coerce")
        + pw * pd.to_numeric(feature_df[near_col], errors="coerce")
        + (sw * pw) * pd.to_numeric(feature_df[both_col], errors="coerce")
    )
    bp_raw = pd.to_numeric(feature_df["bp_total"], errors="coerce") + sw * pd.to_numeric(
        feature_df["bp_in_section"], errors="coerce"
    )
    return (sub_raw - bp_raw) * (10000.0 / tok)


def normalized_entropy(probs: Iterable[float]) -> float:
    vals = np.asarray(list(probs), dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    raw = float(-(vals * np.log(vals)).sum())
    return raw / math.log(3.0)


def deciles_filled(values: pd.Series) -> int:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return 0
    try:
        return int(pd.qcut(s, 10, duplicates="drop").nunique())
    except ValueError:
        return int(s.nunique() > 0)


def measure_year(values: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return {
            "year_std": np.nan,
            "year_iqr": np.nan,
            "year_p90_p10": np.nan,
            "year_nonzero_share": np.nan,
            "year_sign_entropy": np.nan,
            "year_abs_z_gt_0_5_share": np.nan,
            "year_abs_z_gt_1_share": np.nan,
            "year_deciles_filled": np.nan,
        }

    std = float(s.std(ddof=0))
    centered = s - float(s.mean())
    if std > 0:
        z = centered / std
    else:
        z = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    neg_share = float((s < 0).mean())
    zero_share = float((s == 0).mean())
    pos_share = float((s > 0).mean())

    return {
        "year_std": std,
        "year_iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        "year_p90_p10": float(s.quantile(0.90) - s.quantile(0.10)),
        "year_nonzero_share": float((s != 0).mean()),
        "year_sign_entropy": normalized_entropy([neg_share, zero_share, pos_share]),
        "year_abs_z_gt_0_5_share": float((z.abs() > 0.5).mean()),
        "year_abs_z_gt_1_share": float((z.abs() > 1.0).mean()),
        "year_deciles_filled": float(deciles_filled(s)),
    }


def separation_grid(universe: UniverseConfig) -> pd.DataFrame:
    feature_df = load_feature_rows(universe)
    rows: list[dict[str, float | str | int]] = []
    for sw, pw, cw in product(SW_GRID, PW_GRID, CW_GRID):
        tls = score_tls(feature_df, sw, pw, cw)
        year_df = feature_df[["signal_year"]].copy()
        year_df["tls"] = tls
        yearly = (
            year_df.groupby("signal_year")["tls"]
            .apply(lambda s: pd.Series(measure_year(s)))
            .reset_index()
            .pivot(index="signal_year", columns="level_1", values="tls")
            .reset_index()
        )
        row = {
            "section_weight": sw,
            "proximity_weight": pw,
            "character_window": cw,
            "avg_year_std": float(yearly["year_std"].mean()),
            "avg_year_iqr": float(yearly["year_iqr"].mean()),
            "avg_year_p90_p10": float(yearly["year_p90_p10"].mean()),
            "avg_nonzero_share": float(yearly["year_nonzero_share"].mean()),
            "avg_sign_entropy": float(yearly["year_sign_entropy"].mean()),
            "avg_abs_z_gt_0_5_share": float(yearly["year_abs_z_gt_0_5_share"].mean()),
            "avg_abs_z_gt_1_share": float(yearly["year_abs_z_gt_1_share"].mean()),
            "avg_deciles_filled": float(yearly["year_deciles_filled"].mean()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    rank_cols = [
        "avg_year_std",
        "avg_year_iqr",
        "avg_year_p90_p10",
        "avg_nonzero_share",
        "avg_sign_entropy",
        "avg_abs_z_gt_0_5_share",
        "avg_abs_z_gt_1_share",
        "avg_deciles_filled",
    ]
    for col in rank_cols:
        out[f"rank_{col}"] = out[col].rank(ascending=False, method="min")
    out["blend_scale_free"] = out[
        [
            "rank_avg_nonzero_share",
            "rank_avg_sign_entropy",
            "rank_avg_abs_z_gt_0_5_share",
            "rank_avg_abs_z_gt_1_share",
            "rank_avg_deciles_filled",
        ]
    ].mean(axis=1)
    return out


def add_regression_results(universe: UniverseConfig, metric_df: pd.DataFrame) -> pd.DataFrame:
    reg = pd.read_csv(universe.tls_results_file)
    reg = reg[reg["control_spec"] == "beta_mkt+log_assets"].copy()
    keep = reg[
        [
            "section_weight",
            "proximity_weight",
            "character_window",
            "horizon",
            "beta_z_tls_score",
            "t_z_tls_score",
            "p_z_tls_score",
            "adj_r2",
            "N",
        ]
    ].copy()
    wide = keep.pivot_table(
        index=["section_weight", "proximity_weight", "character_window"],
        columns="horizon",
        values=["beta_z_tls_score", "t_z_tls_score", "p_z_tls_score", "adj_r2", "N"],
        aggfunc="first",
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    return metric_df.merge(wide, on=["section_weight", "proximity_weight", "character_window"], how="left")


def winner_rows(universe: str, metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for objective, meta in OBJECTIVE_META.items():
        ascending = objective == "blend_scale_free"
        winner = metric_df.sort_values(objective, ascending=ascending).iloc[0]
        row = winner[
            [
                "section_weight",
                "proximity_weight",
                "character_window",
                objective,
                "beta_z_tls_score_1y",
                "t_z_tls_score_1y",
                "p_z_tls_score_1y",
                "adj_r2_1y",
                "beta_z_tls_score_2y",
                "t_z_tls_score_2y",
                "p_z_tls_score_2y",
                "adj_r2_2y",
            ]
        ].to_dict()
        row["objective"] = objective
        row["objective_label"] = meta["label"]
        row["scale_sensitive"] = meta["scale_sensitive"]
        row["universe"] = universe
        rows.append(row)
    return pd.DataFrame(rows)


def common_winners(metric_by_universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    left = metric_by_universe["t100"].copy()
    right = metric_by_universe["r1000"].copy()
    key = ["section_weight", "proximity_weight", "character_window"]
    merged = left.merge(right, on=key, suffixes=("_t100", "_r1000"))
    for objective, meta in OBJECTIVE_META.items():
        if objective == "blend_scale_free":
            score = (
                merged["blend_scale_free_t100"].rank(ascending=True, method="min")
                + merged["blend_scale_free_r1000"].rank(ascending=True, method="min")
            ) / 2.0
            merged_local = merged.assign(common_score=score)
            winner = merged_local.sort_values("common_score", ascending=True).iloc[0]
        else:
            rank_t = merged[f"{objective}_t100"].rank(ascending=False, method="min")
            rank_r = merged[f"{objective}_r1000"].rank(ascending=False, method="min")
            merged_local = merged.assign(common_score=(rank_t + rank_r) / 2.0)
            winner = merged_local.sort_values("common_score", ascending=True).iloc[0]

        rows.append(
            {
                "objective": objective,
                "objective_label": meta["label"],
                "scale_sensitive": meta["scale_sensitive"],
                "section_weight": winner["section_weight"],
                "proximity_weight": winner["proximity_weight"],
                "character_window": winner["character_window"],
                "common_score": winner["common_score"],
                "objective_t100": winner.get(f"{objective}_t100", np.nan),
                "objective_r1000": winner.get(f"{objective}_r1000", np.nan),
                "p_1y_t100": winner["p_z_tls_score_1y_t100"],
                "t_1y_t100": winner["t_z_tls_score_1y_t100"],
                "p_2y_t100": winner["p_z_tls_score_2y_t100"],
                "t_2y_t100": winner["t_z_tls_score_2y_t100"],
                "p_1y_r1000": winner["p_z_tls_score_1y_r1000"],
                "t_1y_r1000": winner["t_z_tls_score_1y_r1000"],
                "p_2y_r1000": winner["p_z_tls_score_2y_r1000"],
                "t_2y_r1000": winner["t_z_tls_score_2y_r1000"],
            }
        )
    return pd.DataFrame(rows)


def selected_specs_summary(metric_by_universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec_name, spec in [("current_shared", CURRENT_SHARED_SPEC), ("mild_theory", MILD_THEORY_SPEC)]:
        sw, pw, cw = spec
        for universe, df in metric_by_universe.items():
            row = df[
                (df["section_weight"] == sw)
                & (df["proximity_weight"] == pw)
                & (df["character_window"] == cw)
            ].iloc[0]
            rows.append(
                {
                    "spec_name": spec_name,
                    "universe": universe,
                    "section_weight": sw,
                    "proximity_weight": pw,
                    "character_window": cw,
                    "avg_nonzero_share": row["avg_nonzero_share"],
                    "avg_sign_entropy": row["avg_sign_entropy"],
                    "avg_abs_z_gt_0_5_share": row["avg_abs_z_gt_0_5_share"],
                    "avg_abs_z_gt_1_share": row["avg_abs_z_gt_1_share"],
                    "avg_deciles_filled": row["avg_deciles_filled"],
                    "p_1y": row["p_z_tls_score_1y"],
                    "t_1y": row["t_z_tls_score_1y"],
                    "p_2y": row["p_z_tls_score_2y"],
                    "t_2y": row["t_z_tls_score_2y"],
                }
            )
    return pd.DataFrame(rows)


def shared_candidate_ranking(metric_by_universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = ["section_weight", "proximity_weight", "character_window"]
    merged = metric_by_universe["t100"].merge(
        metric_by_universe["r1000"], on=key, suffixes=("_t100", "_r1000")
    )

    for universe in ["t100", "r1000"]:
        rank_cols = []
        for objective in SCALE_FREE_OBJECTIVES:
            rank_col = f"rank_{objective}_{universe}"
            merged[rank_col] = merged[f"{objective}_{universe}"].rank(ascending=False, method="min")
            rank_cols.append(rank_col)
        merged[f"avg_scale_free_rank_{universe}"] = merged[rank_cols].mean(axis=1)

    merged["combined_avg_scale_free_rank"] = merged[
        ["avg_scale_free_rank_t100", "avg_scale_free_rank_r1000"]
    ].mean(axis=1)
    max_nonzero_t100 = float(merged["avg_nonzero_share_t100"].max())
    max_nonzero_r1000 = float(merged["avg_nonzero_share_r1000"].max())
    merged["gap_nonzero_t100"] = max_nonzero_t100 - pd.to_numeric(merged["avg_nonzero_share_t100"], errors="coerce")
    merged["gap_nonzero_r1000"] = max_nonzero_r1000 - pd.to_numeric(merged["avg_nonzero_share_r1000"], errors="coerce")
    merged["max_gap_nonzero"] = merged[["gap_nonzero_t100", "gap_nonzero_r1000"]].max(axis=1)
    merged = merged.sort_values(
        ["combined_avg_scale_free_rank", "section_weight", "proximity_weight", "character_window"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    merged["top_tier"] = (
        (merged["gap_nonzero_t100"] <= NONZERO_FRONTIER_GAP)
        & (merged["gap_nonzero_r1000"] <= NONZERO_FRONTIER_GAP)
    )
    merged["pw_is_unit"] = (merged["proximity_weight"] == 1.0).astype(int)
    merged["sw_is_nonunit"] = (merged["section_weight"] > 1.0).astype(int)
    merged["parsimony_sort_sw"] = np.where(merged["section_weight"] > 1.0, merged["section_weight"], np.inf)
    return merged


def select_parsimonious_top_tier(shared_ranked: pd.DataFrame) -> pd.DataFrame:
    top = shared_ranked[shared_ranked["top_tier"]].copy()
    if top.empty:
        raise RuntimeError("No top-tier shared TLS candidates available under the nonzero-share frontier rule.")
    top = top.sort_values(
        [
            "sw_is_nonunit",
            "parsimony_sort_sw",
            "pw_is_unit",
            "character_window",
            "max_gap_nonzero",
            "combined_avg_scale_free_rank",
        ],
        ascending=[False, True, False, True, True, True],
    )
    selected = top.iloc[0].copy()
    return pd.DataFrame(
        [
            {
                "section_weight": selected["section_weight"],
                "proximity_weight": selected["proximity_weight"],
                "character_window": selected["character_window"],
                "combined_avg_scale_free_rank": selected["combined_avg_scale_free_rank"],
                "avg_scale_free_rank_t100": selected["avg_scale_free_rank_t100"],
                "avg_scale_free_rank_r1000": selected["avg_scale_free_rank_r1000"],
                "gap_nonzero_t100": selected["gap_nonzero_t100"],
                "gap_nonzero_r1000": selected["gap_nonzero_r1000"],
                "max_gap_nonzero": selected["max_gap_nonzero"],
                "selection_rule": "within_1pp_nonzero_share_frontier_then_mildest_nonunit_section_weight_with_unit_proximity_and_shortest_window",
                "nonzero_frontier_gap": NONZERO_FRONTIER_GAP,
            }
        ]
    )


def write_summary(
    winner_by_universe: pd.DataFrame,
    common_winner_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    shared_ranked: pd.DataFrame,
    selected_shared_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# TLS Objective Comparison")
    lines.append("")
    lines.append("This note compares several non-return TLS hyperparameter objectives against the main pooled TLS regressions.")
    lines.append("")
    lines.append("## Read This First")
    lines.append("")
    lines.append("- `avg_year_std`, `avg_year_iqr`, and `avg_year_p90_p10` are **scale-sensitive**. They mostly reward larger weights and wider windows.")
    lines.append("- The more defensible separation objectives are the scale-free ones: `avg_nonzero_share`, `avg_sign_entropy`, `avg_abs_z_gt_0_5_share`, `avg_abs_z_gt_1_share`, `avg_deciles_filled`, and the `blend_scale_free` composite.")
    lines.append("- `blend_scale_free` is the average rank across nonzero share, sign entropy, |z|>0.5 share, |z|>1 share, and deciles filled.")
    lines.append(f"- The final shared spec is selected from the set of specifications that lie within {NONZERO_FRONTIER_GAP:.0%} of the maximum nonzero-share separation in both universes, then broken by parsimony.")
    lines.append("")
    lines.append("## Current Specs")
    lines.append("")
    lines.append("- `current_shared`: `(sw=1, pw=1, cw=10)`")
    lines.append("- `mild_theory`: `(sw=3.25, pw=1, cw=10)`")
    lines.append("")
    lines.append("## Common-Winner Specs by Objective")
    lines.append("")
    top = common_winner_df[
        [
            "objective_label",
            "scale_sensitive",
            "section_weight",
            "proximity_weight",
            "character_window",
            "p_1y_t100",
            "p_1y_r1000",
            "p_2y_t100",
            "p_2y_r1000",
        ]
    ].copy()
    lines.append(top.to_markdown(index=False))
    lines.append("")
    lines.append("## Selected Spec Comparison")
    lines.append("")
    lines.append(selected_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Shared Candidate Ranking")
    lines.append("")
    lines.append(
        shared_ranked[
            [
                "section_weight",
                "proximity_weight",
                "character_window",
                "combined_avg_scale_free_rank",
                "gap_nonzero_t100",
                "gap_nonzero_r1000",
                "avg_scale_free_rank_t100",
                "avg_scale_free_rank_r1000",
                "top_tier",
            ]
        ].head(20).to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Selected Shared TLS Spec")
    lines.append("")
    lines.append(selected_shared_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Universe-Specific Winners")
    lines.append("")
    lines.append(winner_by_universe.to_markdown(index=False))
    (out_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare non-return TLS objective functions across universes")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--t100-results-file", type=Path, default=UNIVERSES[0].tls_results_file)
    ap.add_argument("--r1000-results-file", type=Path, default=UNIVERSES[1].tls_results_file)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    universes = [
        UniverseConfig(
            name="t100",
            feature_cache_dir=UNIVERSES[0].feature_cache_dir,
            tickers_file=UNIVERSES[0].tickers_file,
            tls_results_file=args.t100_results_file,
        ),
        UniverseConfig(
            name="r1000",
            feature_cache_dir=UNIVERSES[1].feature_cache_dir,
            tickers_file=UNIVERSES[1].tickers_file,
            tls_results_file=args.r1000_results_file,
        ),
    ]

    metric_by_universe: dict[str, pd.DataFrame] = {}
    winner_frames = []
    for universe in universes:
        metrics = separation_grid(universe)
        metrics = add_regression_results(universe, metrics)
        metrics["universe"] = universe.name
        metrics.to_csv(out_dir / f"{universe.name}_objective_grid.csv", index=False)
        metric_by_universe[universe.name] = metrics
        winner_frames.append(winner_rows(universe.name, metrics))

    winner_by_universe = pd.concat(winner_frames, ignore_index=True)
    common_winner_df = common_winners(metric_by_universe)
    selected_df = selected_specs_summary(metric_by_universe)
    shared_ranked = shared_candidate_ranking(metric_by_universe)
    selected_shared_df = select_parsimonious_top_tier(shared_ranked)

    winner_by_universe.to_csv(out_dir / "objective_winners_by_universe.csv", index=False)
    common_winner_df.to_csv(out_dir / "objective_winners_common.csv", index=False)
    selected_df.to_csv(out_dir / "selected_spec_comparison.csv", index=False)
    shared_ranked.to_csv(out_dir / "shared_candidate_ranking.csv", index=False)
    selected_shared_df.to_csv(out_dir / "selected_shared_spec.csv", index=False)
    write_summary(
        winner_by_universe,
        common_winner_df,
        selected_df,
        shared_ranked,
        selected_shared_df,
        out_dir,
    )
    print(f"[OUT] {out_dir / 'objective_winners_by_universe.csv'}")
    print(f"[OUT] {out_dir / 'objective_winners_common.csv'}")
    print(f"[OUT] {out_dir / 'selected_spec_comparison.csv'}")
    print(f"[OUT] {out_dir / 'shared_candidate_ranking.csv'}")
    print(f"[OUT] {out_dir / 'selected_shared_spec.csv'}")
    print(f"[OUT] {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
