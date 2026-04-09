#!/usr/bin/env python3
"""
Build canonical curated tables for the thesis panel data.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
from pathlib import Path

from thesis_pipeline.canonical import BuildConfig, build_all_canonical_tables, parse_years


def main() -> None:
    ap = argparse.ArgumentParser(description="Build canonical thesis panel tables")
    ap.add_argument(
        "--universe-file",
        type=Path,
        default=Path("data/raw/universe/transition_100_tickers.txt"),
    )
    ap.add_argument(
        "--years",
        type=str,
        default="2015,2016,2017,2018,2019,2020,2021,2022,2023,2024",
    )
    ap.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("data/processed/feature_cache/t100"),
    )
    ap.add_argument(
        "--sentiment-file",
        type=Path,
        default=Path("data/processed/search/sentiment/t100/runs/finbert_core_c30_h1_filter_off_repeat_on_j088_f256/sentiment.csv"),
    )
    ap.add_argument(
        "--sentiment-primary-col",
        type=str,
        default="transition_sentiment_median",
    )
    ap.add_argument(
        "--returns-file",
        type=Path,
        default=Path("data/raw/returns/daily_ret_10y.csv"),
    )
    ap.add_argument(
        "--factors-ff3-file",
        type=Path,
        default=Path("data/raw/factors/ff_factors_clean.csv"),
    )
    ap.add_argument(
        "--factors-ff5-file",
        type=Path,
        default=Path("data/raw/factors/ff5_factors_clean.csv"),
    )
    ap.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("data/raw/metadata/ticker_addtl_data.csv"),
    )
    ap.add_argument(
        "--assets-file",
        type=Path,
        default=Path("data/raw/assets/asset_data.csv"),
    )
    ap.add_argument(
        "--co2-file",
        type=Path,
        default=Path("data/raw/emissions/co2data.csv"),
    )
    ap.add_argument(
        "--curated-root",
        type=Path,
        default=Path("data/curated/t100/final_within_year_shared"),
    )
    ap.add_argument("--tls-sw", type=float, default=1.0)
    ap.add_argument("--tls-pw", type=float, default=10.0)
    ap.add_argument("--tls-cw", type=int, default=255)
    ap.add_argument("--factor-quantile", type=float, default=0.25)
    args = ap.parse_args()

    cfg = BuildConfig(
        universe_file=args.universe_file,
        years=parse_years(args.years),
        feature_cache_dir=args.feature_cache_dir,
        sentiment_file=args.sentiment_file,
        sentiment_primary_col=args.sentiment_primary_col,
        returns_file=args.returns_file,
        factors_ff3_file=args.factors_ff3_file,
        factors_ff5_file=args.factors_ff5_file,
        metadata_file=args.metadata_file,
        assets_file=args.assets_file,
        co2_file=args.co2_file,
        curated_root=args.curated_root,
        tls_sw=args.tls_sw,
        tls_pw=args.tls_pw,
        tls_cw=args.tls_cw,
        factor_quantile=args.factor_quantile,
    )

    manifest = build_all_canonical_tables(cfg)

    print("[OK] Canonical build complete")
    print(f"[OUT] {args.curated_root / 'ticker_year_panel.csv'}")
    print(f"[OUT] {args.curated_root / 'benchmark_factors_monthly.csv'}")
    print(f"[OUT] {args.curated_root / 'custom_factor_returns_monthly.csv'}")
    print(f"[OUT] {args.curated_root / 'schema.md'}")
    print(f"[OUT] {args.curated_root / 'build_manifest.json'}")
    print(json.dumps(manifest.get("validations", {}), indent=2))


if __name__ == "__main__":
    main()
