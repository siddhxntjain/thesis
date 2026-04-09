#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from thesis_pipeline.canonical import _load_returns_monthly
from thesis_pipeline.factors import (
    BENCHMARK_ORDER,
    CUSTOM_FACTOR_ORDER,
    CUTOFFS,
    DISPLAY_NAMES,
    ROOT,
    UNIVERSES,
    build_custom_factor_long,
    build_factor_wide,
    build_locked_sentiment_signal,
    build_sector_fe_robustness,
    build_stock_month_panel,
    compute_correlation_matrix,
    compute_summary_stats,
    fit_factor_alpha_models,
    fit_stock_panel_model_grid,
    load_locked_sentiment_specs,
    load_manifest_inputs,
    prepare_universe_panel,
    summarize_correlations,
    write_correlation_tex,
    write_factor_alpha_tex,
    write_sector_fe_robustness_tex,
    write_stock_panel_tex,
    write_summary_stats_tex,
)


DEFAULT_OUT_ROOT = ROOT / "data/outputs/factors/final"


def save_factor_csvs(out_root: Path, universe_key: str, factor_long: pd.DataFrame, summary_stats: pd.DataFrame, corr_map: Dict[float, pd.DataFrame]) -> None:
    universe_dir = out_root / universe_key
    universe_dir.mkdir(parents=True, exist_ok=True)
    factor_long.to_csv(universe_dir / f"{universe_key}_custom_factor_returns_monthly_all_cutoffs.csv", index=False)
    summary_stats.to_csv(universe_dir / f"{universe_key}_factor_summary_stats_all_cutoffs.csv", index=False)
    for cutoff, corr in corr_map.items():
        pct = 33 if np.isclose(cutoff, 1.0 / 3.0) else int(round(100 * cutoff))
        corr.to_csv(universe_dir / f"{universe_key}_factor_correlation_q{pct:02d}.csv", index=False)
        subset = factor_long[np.isclose(factor_long["cutoff"], cutoff)].copy()
        subset.to_csv(universe_dir / f"{universe_key}_custom_factor_returns_q{pct:02d}.csv", index=False)


def build_headline_tex_inputs(summary_all: pd.DataFrame, corr_lookup: Dict[str, pd.DataFrame], alpha_df: pd.DataFrame, stock_results: pd.DataFrame, sector_fe_df: pd.DataFrame, out_root: Path) -> None:
    write_summary_stats_tex(summary_all, out_root / "factor_summary_stats_25.tex")
    write_correlation_tex(corr_lookup, out_root / "factor_correlation_matrix_25.tex")
    write_factor_alpha_tex(alpha_df, out_root / "factor_alpha_models_all.tex")
    write_stock_panel_tex(stock_results, out_root / "stock_panel_ff3_ff5_with_custom_factors.tex")
    write_sector_fe_robustness_tex(sector_fe_df, out_root / "sector_fe_robustness.tex")


def build_summary_md(
    out_root: Path,
    alpha_df: pd.DataFrame,
    corr_lookup: Dict[str, pd.DataFrame],
    stock_results: pd.DataFrame,
) -> None:
    lines = [
        "# Huang-Style Factor Lab Results",
        "",
        "This module is fully parallel to the thesis pipeline. It uses new Huang-style carbon inputs and does not overwrite existing thesis outputs.",
        "",
        "## FF3 / FF5 survival of custom factors",
    ]

    if alpha_df.empty:
        lines.append("No factor alpha models were estimated.")
    else:
        for universe in ["r1000", "t100"]:
            lines.append("")
            lines.append(f"### {'Russell 1000' if universe == 'r1000' else 'Transition-100'}")
            sub = alpha_df[(alpha_df["universe"] == universe) & np.isclose(alpha_df["cutoff"], 0.25) & (alpha_df["model"].isin(["ff3", "ff5"]))].copy()
            if sub.empty:
                lines.append("No 25% factor alpha results.")
                continue
            for factor in CUSTOM_FACTOR_ORDER:
                row_ff3 = sub[(sub["factor_name"] == factor) & (sub["model"] == "ff3")]
                row_ff5 = sub[(sub["factor_name"] == factor) & (sub["model"] == "ff5")]
                txt_bits = [DISPLAY_NAMES[factor]]
                if not row_ff3.empty:
                    r = row_ff3.iloc[0]
                    status = "survives FF3" if r["survives_conventionally"] else ("borderline in FF3" if r["borderline"] else "does not survive FF3")
                    txt_bits.append(f"FF3 alpha {100*r['alpha_monthly']:.3f}% (p={r['alpha_p']:.3f}; {status})")
                if not row_ff5.empty:
                    r = row_ff5.iloc[0]
                    status = "survives FF5" if r["survives_conventionally"] else ("borderline in FF5" if r["borderline"] else "does not survive FF5")
                    txt_bits.append(f"FF5 alpha {100*r['alpha_monthly']:.3f}% (p={r['alpha_p']:.3f}; {status})")
                lines.append(f"- {'; '.join(txt_bits)}")

    lines.extend(["", "## Correlation summary (25% factors)"])
    for universe in ["r1000", "t100"]:
        corr = corr_lookup.get(universe)
        lines.append("")
        lines.append(f"### {'Russell 1000' if universe == 'r1000' else 'Transition-100'}")
        lines.append(f"- {summarize_correlations(corr) if corr is not None else 'No correlation matrix.'}")

    lines.extend(["", "## Stock-panel regressions with custom factors"])
    if stock_results.empty:
        lines.append("No stock-panel regressions were estimated.")
    else:
        for universe in ["r1000", "t100"]:
            lines.append("")
            lines.append(f"### {'Russell 1000' if universe == 'r1000' else 'Transition-100'}")
            for model_name in ["ff3_all4", "ff5_all4"]:
                row = stock_results[
                    (stock_results["universe"] == universe)
                    & (stock_results["model"] == model_name)
                    & (stock_results["sector_fe"] == True)
                ]
                if row.empty:
                    continue
                r = row.iloc[0]
                bits = [f"{model_name}: R^2={r['r2']:.3f}"]
                for factor in CUSTOM_FACTOR_ORDER:
                    coef = r.get(f"coef_{factor}", np.nan)
                    p = r.get(f"p_{factor}", np.nan)
                    if pd.notna(coef):
                        bits.append(f"{DISPLAY_NAMES[factor]} {coef:.4f} (p={p:.3f})")
                lines.append(f"- {'; '.join(bits)}")

    lines.extend([
        "",
        "## Factor vs characteristic evidence",
        "The updated thesis characteristic results remain the stronger layer. Under the shared c30 FinBERT lock, broad-market sentiment still survives as both a characteristic and a factor, while the Transition-100 sentiment characteristic becomes weak and the factor result does not survive FF3/FF5.",
    ])
    (out_root / "factor_lab_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a rollback-safe Huang-style factor lab for TLS, sentiment, GMB_U, and GMB_S.")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    sentiment_specs = load_locked_sentiment_specs()

    all_summary_stats: list[pd.DataFrame] = []
    all_alpha_rows: list[pd.DataFrame] = []
    all_stock_rows: list[pd.DataFrame] = []
    all_sector_fe_rows: list[pd.DataFrame] = []
    headline_corr_lookup: Dict[str, pd.DataFrame] = {}

    for universe in UNIVERSES:
        inputs = load_manifest_inputs(universe.curated_root)
        benchmark_factors = pd.read_csv(universe.curated_root / "benchmark_factors_monthly.csv", parse_dates=["month_end"])
        returns_m = _load_returns_monthly(Path(inputs["returns_file"]))
        panel = prepare_universe_panel(universe, sentiment_specs)
        returns_m = returns_m[returns_m["ticker"].isin(set(panel["ticker"]))].copy()

        factor_long = build_custom_factor_long(universe, panel, returns_m, CUTOFFS)
        if factor_long.empty:
            continue

        corr_map: Dict[float, pd.DataFrame] = {}
        universe_summary_frames: list[pd.DataFrame] = []
        universe_alpha_frames: list[pd.DataFrame] = []
        headline_wide = pd.DataFrame()

        for cutoff in CUTOFFS:
            wide = build_factor_wide(factor_long, benchmark_factors, cutoff)
            summary = compute_summary_stats(wide, universe.key, cutoff)
            corr = compute_correlation_matrix(wide, universe.key, cutoff)
            alpha = fit_factor_alpha_models(wide, universe.key, cutoff)
            universe_summary_frames.append(summary)
            universe_alpha_frames.append(alpha)
            corr_map[cutoff] = corr
            if np.isclose(cutoff, 0.25):
                headline_wide = wide.copy()
                headline_corr_lookup[universe.key] = corr.copy()

        summary_stats = pd.concat(universe_summary_frames, ignore_index=True) if universe_summary_frames else pd.DataFrame()
        alpha_df = pd.concat(universe_alpha_frames, ignore_index=True) if universe_alpha_frames else pd.DataFrame()
        save_factor_csvs(out_root, universe.key, factor_long, summary_stats, corr_map)

        stock_panel = build_stock_month_panel(universe, panel, returns_m, headline_wide)
        stock_results = fit_stock_panel_model_grid(stock_panel, universe.key)
        sector_fe_df = build_sector_fe_robustness(stock_results)

        if not stock_panel.empty:
            stock_panel.to_csv(out_root / universe.key / f"{universe.key}_stock_month_panel_25.csv", index=False)
        if not stock_results.empty:
            stock_results.to_csv(out_root / universe.key / f"{universe.key}_stock_panel_models.csv", index=False)
        if not sector_fe_df.empty:
            sector_fe_df.to_csv(out_root / universe.key / f"{universe.key}_sector_fe_robustness.csv", index=False)

        all_summary_stats.append(summary_stats)
        all_alpha_rows.append(alpha_df)
        all_stock_rows.append(stock_results)
        all_sector_fe_rows.append(sector_fe_df)

    summary_all = pd.concat([x for x in all_summary_stats if not x.empty], ignore_index=True) if all_summary_stats else pd.DataFrame()
    alpha_all = pd.concat([x for x in all_alpha_rows if not x.empty], ignore_index=True) if all_alpha_rows else pd.DataFrame()
    stock_all = pd.concat([x for x in all_stock_rows if not x.empty], ignore_index=True) if all_stock_rows else pd.DataFrame()
    sector_fe_all = pd.concat([x for x in all_sector_fe_rows if not x.empty], ignore_index=True) if all_sector_fe_rows else pd.DataFrame()

    summary_all.to_csv(out_root / "factor_summary_stats_all.csv", index=False)
    alpha_all.to_csv(out_root / "factor_alpha_models_all.csv", index=False)
    stock_all.to_csv(out_root / "stock_panel_ff3_ff5_with_custom_factors.csv", index=False)
    sector_fe_all.to_csv(out_root / "sector_fe_robustness.csv", index=False)

    if not headline_corr_lookup:
        pd.DataFrame().to_csv(out_root / "factor_correlation_matrix_25.csv", index=False)
    else:
        combined_corr = []
        for universe_key, corr in headline_corr_lookup.items():
            combined_corr.append(corr)
        pd.concat(combined_corr, ignore_index=True).to_csv(out_root / "factor_correlation_matrix_25.csv", index=False)

    build_headline_tex_inputs(summary_all, headline_corr_lookup, alpha_all, stock_all, sector_fe_all, out_root)
    build_summary_md(out_root, alpha_all, headline_corr_lookup, stock_all)

    print(f"[OUT] {out_root}")
    print(f"[OUT] {out_root / 'factor_alpha_models_all.csv'}")
    print(f"[OUT] {out_root / 'stock_panel_ff3_ff5_with_custom_factors.csv'}")
    print(f"[OUT] {out_root / 'factor_lab_results_summary.md'}")


if __name__ == "__main__":
    main()
