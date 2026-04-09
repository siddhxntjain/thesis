#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from thesis_pipeline.regressions import (
    HORIZON_META,
    build_market_beta_map,
    build_regression_frame,
    control_specs,
    fit_formula,
    load_panel,
    parse_years,
    parse_tickers,
    regular_formula,
)

DEFAULTS = {
    "years": "2015-2024",
    "sw_grid": "1,3.25,5.5,7.75,10",
    "pw_grid": "1,3.25,5.5,7.75,10",
    "cw_grid": "10,132,255,378,500",
    "horizons": "1y,2y",
    "target_control_spec": "",
    "min_n_pooled": 80,
    "r1000": {
        "tickers_file": Path("data/raw/universe/tickers.txt"),
        "curated_root": Path("data/curated/r1000/final_within_year"),
        "feature_cache_dir": Path("data/processed/feature_cache/r1000"),
        "returns_file": Path("data/raw/returns/daily_ret_10y_full_r1000.csv"),
        "benchmark_factors_file": Path("data/curated/r1000/final_within_year/benchmark_factors_monthly.csv"),
        "out_dir": Path("data/processed/search/tls/r1000"),
    },
    "t100": {
        "tickers_file": Path("data/raw/universe/transition_100_tickers.txt"),
        "curated_root": Path("data/curated/t100/final_within_year_shared"),
        "feature_cache_dir": Path("data/processed/feature_cache/t100"),
        "returns_file": Path("data/raw/returns/daily_ret_10y.csv"),
        "benchmark_factors_file": Path("data/curated/t100/final_within_year_shared/benchmark_factors_monthly.csv"),
        "out_dir": Path("data/processed/search/tls/t100"),
    },
}


@dataclass(frozen=True)
class TLSConfig:
    section_weight: float
    proximity_weight: float
    character_window: int

    @property
    def config_id(self) -> str:
        return f"sw{self.section_weight:g}_pw{self.proximity_weight:g}_cw{self.character_window}"


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_num_grid(raw: str, cast):
    vals = sorted({cast(x.strip()) for x in str(raw).split(",") if x.strip()})
    if not vals:
        raise ValueError("Grid cannot be empty.")
    return vals


def build_grid(sw_grid: Sequence[float], pw_grid: Sequence[float], cw_grid: Sequence[int]) -> List[TLSConfig]:
    return [
        TLSConfig(section_weight=float(sw), proximity_weight=float(pw), character_window=int(cw))
        for sw in sw_grid
        for pw in pw_grid
        for cw in cw_grid
    ]


def load_feature_rows(feature_dir: Path, years: Sequence[int], tickers: Sequence[str]) -> pd.DataFrame:
    keep_base = ["ticker", "tokens", "sub_total", "sub_in_section", "bp_total", "bp_in_section"]
    rows: List[pd.DataFrame] = []
    ticker_set = set(tickers)
    for year in years:
        path = feature_dir / f"feature_cache_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df[df["ticker"].isin(ticker_set)].copy()
        if df.empty:
            continue
        cols = keep_base + [c for c in df.columns if c.startswith("sub_near_") or c.startswith("sub_both_")]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].copy()
        df["signal_year"] = int(year)
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=keep_base + ["signal_year"])
    return pd.concat(rows, ignore_index=True).sort_values(["signal_year", "ticker"]).reset_index(drop=True)


def score_from_config(feature_df: pd.DataFrame, cfg: TLSConfig) -> pd.Series:
    near_col = f"sub_near_{cfg.character_window}"
    both_col = f"sub_both_{cfg.character_window}"
    if near_col not in feature_df.columns or both_col not in feature_df.columns:
        raise KeyError(f"Missing feature columns for cw={cfg.character_window}: {near_col}, {both_col}")
    tok = pd.to_numeric(feature_df["tokens"], errors="coerce").clip(lower=1)
    sub_raw = (
        pd.to_numeric(feature_df["sub_total"], errors="coerce")
        + cfg.section_weight * pd.to_numeric(feature_df["sub_in_section"], errors="coerce")
        + cfg.proximity_weight * pd.to_numeric(feature_df[near_col], errors="coerce")
        + (cfg.section_weight * cfg.proximity_weight) * pd.to_numeric(feature_df[both_col], errors="coerce")
    )
    bp_raw = pd.to_numeric(feature_df["bp_total"], errors="coerce") + cfg.section_weight * pd.to_numeric(feature_df["bp_in_section"], errors="coerce")
    return (sub_raw - bp_raw) * (10000.0 / tok)


def evaluate_cfg(cfg: TLSConfig, feature_df: pd.DataFrame, base_panel: pd.DataFrame, horizons: Sequence[str], target_control_spec: str | None, min_n_pooled: int) -> pd.DataFrame:
    scores = feature_df[["ticker", "signal_year"]].copy()
    scores["tls_score"] = score_from_config(feature_df, cfg)
    panel = base_panel.drop(columns=["tls_score", "delta_tls"], errors="ignore").merge(scores, on=["ticker", "signal_year"], how="left")
    panel = panel.sort_values(["ticker", "signal_year"]).reset_index(drop=True)
    panel["delta_tls"] = panel.groupby("ticker")["tls_score"].diff()
    reg_df = build_regression_frame(panel)

    out_rows: List[Dict[str, object]] = []
    for horizon in horizons:
        meta = HORIZON_META[horizon]
        base = reg_df[reg_df[meta["complete_col"]].fillna(False)].copy()
        if base.empty:
            continue
        for control_name, controls in control_specs():
            if target_control_spec and control_name != target_control_spec:
                continue
            formula, focus = regular_formula(meta["ret_col"], "tls_level", controls, pooled=True)
            fit_row, _ = fit_formula(base, formula, min_n_pooled, focus)
            if fit_row is None:
                continue
            row = fit_row.iloc[0].to_dict()
            row.update(
                {
                    "config_id": cfg.config_id,
                    "section_weight": cfg.section_weight,
                    "proximity_weight": cfg.proximity_weight,
                    "character_window": cfg.character_window,
                    "horizon": horizon,
                    "control_spec": control_name,
                }
            )
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the final within-year TLS locking search")
    ap.add_argument("--universe", choices=["r1000", "t100"], default="r1000")
    ap.add_argument("--tickers-file", type=Path, default=None)
    ap.add_argument("--curated-root", type=Path, default=None)
    ap.add_argument("--feature-cache-dir", type=Path, default=None)
    ap.add_argument("--returns-file", type=Path, default=None)
    ap.add_argument("--benchmark-factors-file", type=Path, default=None)
    ap.add_argument("--years", type=str, default=DEFAULTS["years"])
    ap.add_argument("--sw-grid", type=str, default=DEFAULTS["sw_grid"])
    ap.add_argument("--pw-grid", type=str, default=DEFAULTS["pw_grid"])
    ap.add_argument("--cw-grid", type=str, default=DEFAULTS["cw_grid"])
    ap.add_argument("--horizons", type=str, default=DEFAULTS["horizons"])
    ap.add_argument("--target-control-spec", type=str, default=DEFAULTS["target_control_spec"])
    ap.add_argument("--min-n-pooled", type=int, default=DEFAULTS["min_n_pooled"])
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    universe_defaults = DEFAULTS[args.universe]
    tickers_file = args.tickers_file or universe_defaults["tickers_file"]
    curated_root = args.curated_root or universe_defaults["curated_root"]
    feature_cache_dir = args.feature_cache_dir or universe_defaults["feature_cache_dir"]
    returns_file = args.returns_file or universe_defaults["returns_file"]
    benchmark_factors_file = args.benchmark_factors_file or universe_defaults["benchmark_factors_file"]
    out_dir = args.out_dir or universe_defaults["out_dir"]

    tickers = parse_tickers(tickers_file)
    years = parse_years(args.years)
    horizons = parse_csv_list(args.horizons)
    sw_grid = parse_num_grid(args.sw_grid, float)
    pw_grid = parse_num_grid(args.pw_grid, float)
    cw_grid = parse_num_grid(args.cw_grid, int)
    target_control_spec = args.target_control_spec.strip() or None

    base_panel = load_panel(curated_root, tickers, years)
    beta_map = build_market_beta_map(base_panel, returns_file, benchmark_factors_file)
    base_panel = base_panel.merge(beta_map[["ticker", "signal_year", "beta_mkt"]], on=["ticker", "signal_year"], how="left")
    feature_df = load_feature_rows(feature_cache_dir, years, tickers)
    if feature_df.empty:
        raise RuntimeError("No feature rows available for the requested universe/years.")

    cfgs = build_grid(sw_grid, pw_grid, cw_grid)
    frames = [evaluate_cfg(cfg, feature_df, base_panel, horizons, target_control_spec, int(args.min_n_pooled)) for cfg in cfgs]
    res = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if res.empty:
        raise RuntimeError("No evaluable TLS configurations.")

    res["focus_p"] = pd.to_numeric(res["p_z_tls_score"], errors="coerce")
    res["focus_beta"] = pd.to_numeric(res["beta_z_tls_score"], errors="coerce")
    res["focus_t"] = pd.to_numeric(res["t_z_tls_score"], errors="coerce")
    res = res.sort_values(["horizon", "control_spec", "focus_p", "adj_r2"], ascending=[True, True, True, False], na_position="last").reset_index(drop=True)
    best = res.groupby(["horizon", "control_spec"], as_index=False).first()

    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "all_results.csv", index=False)
    best.to_csv(out_dir / "best_by_horizon_control.csv", index=False)
    print(f"[OUT] {out_dir / 'all_results.csv'} ({len(res)} rows)")
    print(f"[OUT] {out_dir / 'best_by_horizon_control.csv'} ({len(best)} rows)")


if __name__ == "__main__":
    main()
