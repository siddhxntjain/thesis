#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from thesis_pipeline.canonical import BuildConfig, _load_returns_monthly, build_all_canonical_tables
from thesis_pipeline.regressions import (
    HORIZON_META,
    build_control_grid_summary,
    build_market_beta_map,
    best_specs_table,
    control_specs,
    fit_formula,
    load_panel,
    make_sector_flags,
    run_interaction_models,
    run_regular_models,
    standardize,
)

BEST_SENTIMENT_CFG = {
    "scorer": "finbert",
    "substantive_file": "assets/terms/substantive_terms.txt",
    "min_sentence_chars": 30,
    "min_transition_term_hits": 1,
    "drop_cross_year_boilerplate": True,
    "repeat_jaccard_threshold": 0.88,
    "finbert_max_length": 256,
}


def parse_tickers(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return sorted({x.strip().upper() for x in raw.replace(",", " ").split() if x.strip()})


def parse_years(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def ensure_best_sentiment_file(
    *,
    tickers_file: Path,
    years: Sequence[int],
    cache_manifest: Path,
    text_cache_dir: Path,
    out_file: Path,
    sentences_out: Path,
    dropped_out: Path,
    force: bool,
) -> Path:
    if out_file.exists() and not force:
        return out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "transition_sentiment_item1.py",
        "--tickers-file",
        str(tickers_file),
        "--cache-manifest",
        str(cache_manifest),
        "--text-cache-dir",
        str(text_cache_dir),
        "--substantive-file",
        BEST_SENTIMENT_CFG["substantive_file"],
        "--years",
        f"{min(years)}-{max(years)}",
        "--scorer",
        BEST_SENTIMENT_CFG["scorer"],
        "--finbert-max-length",
        str(BEST_SENTIMENT_CFG["finbert_max_length"]),
        "--min-sentence-chars",
        str(BEST_SENTIMENT_CFG["min_sentence_chars"]),
        "--min-transition-term-hits",
        str(BEST_SENTIMENT_CFG["min_transition_term_hits"]),
        "--repeat-jaccard-threshold",
        str(BEST_SENTIMENT_CFG["repeat_jaccard_threshold"]),
        "--out-file",
        str(out_file),
        "--sentences-out-file",
        str(sentences_out),
        "--dropped-file",
        str(dropped_out),
    ]
    if BEST_SENTIMENT_CFG["drop_cross_year_boilerplate"]:
        cmd.append("--drop-cross-year-boilerplate")
    subprocess.run(cmd, check=True)
    return out_file


def load_sentiment_panel(sentiment_file: Path, tickers: Sequence[str], years: Sequence[int]) -> pd.DataFrame:
    d = pd.read_csv(sentiment_file)
    d["ticker"] = d["ticker"].astype(str).str.upper().str.strip()
    d["signal_year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    d = d[d["ticker"].isin(set(tickers)) & d["signal_year"].isin(set(years))].copy()
    if "filing_date" in d.columns:
        d["filing_date"] = pd.to_datetime(d["filing_date"], errors="coerce")
        d = d.sort_values(["ticker", "signal_year", "filing_date"], ascending=[True, True, False])
    d = d.drop_duplicates(["ticker", "signal_year"], keep="first")
    posh = pd.to_numeric(d.get("transition_pos_hits_total"), errors="coerce")
    negh = pd.to_numeric(d.get("transition_neg_hits_total"), errors="coerce")
    d["hit_balance"] = posh - negh
    d["hit_ratio"] = posh / (negh + 1.0)
    keep = [
        "ticker",
        "signal_year",
        "transition_stance_index",
        "transition_sentiment_mean",
        "transition_sentiment_median",
        "transition_pos_share",
        "transition_neg_share",
        "transition_pos_hits_total",
        "transition_neg_hits_total",
        "n_transition_sentences",
        "n_transition_filtered_repeat",
        "hit_balance",
        "hit_ratio",
    ]
    existing = [c for c in keep if c in d.columns]
    return d[existing].copy()


def winsorize_by_group(v: pd.Series, q_low: float, q_high: float) -> pd.Series:
    s = pd.to_numeric(v, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=v.index, dtype=float)
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    return s.clip(lo, hi)


def add_processed_sentiment_signal(
    d: pd.DataFrame,
    *,
    score_col: str,
    missing_score_policy: str,
    score_transform: str,
) -> pd.DataFrame:
    out = d.copy()
    if score_col not in out.columns:
        raise KeyError(f"Sentiment score column not found: {score_col}")
    out["sentiment_signal_raw"] = pd.to_numeric(out[score_col], errors="coerce")
    if missing_score_policy == "zero":
        out["sentiment_signal_raw"] = out["sentiment_signal_raw"].fillna(0.0)
    elif missing_score_policy != "drop":
        raise ValueError(f"Unsupported missing score policy: {missing_score_policy}")

    if score_transform == "raw":
        out["sentiment_signal"] = out["sentiment_signal_raw"]
    elif score_transform == "winsor_1_99":
        out["sentiment_signal"] = out.groupby("signal_year", group_keys=False)["sentiment_signal_raw"].apply(
            lambda s: winsorize_by_group(s, 0.01, 0.99)
        )
    elif score_transform == "winsor_5_95":
        out["sentiment_signal"] = out.groupby("signal_year", group_keys=False)["sentiment_signal_raw"].apply(
            lambda s: winsorize_by_group(s, 0.05, 0.95)
        )
    else:
        raise ValueError(f"Unsupported sentiment transform: {score_transform}")
    return out


def ensure_common_panel_fields(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["beta_mkt"] = pd.to_numeric(out.get("beta_mkt"), errors="coerce")
    co2 = pd.to_numeric(out.get("co2_unscaled"), errors="coerce")
    assets = pd.to_numeric(out.get("year_assets"), errors="coerce")
    out["co2_assets"] = np.where((co2 >= 0) & (assets > 0), co2 / assets, np.nan)
    return out


def build_r1000_canonical(
    *,
    universe_file: Path,
    years: Sequence[int],
    feature_cache_dir: Path,
    sentiment_file: Path,
    sentiment_primary_col: str,
    returns_file: Path,
    ff3_file: Path,
    ff5_file: Path,
    metadata_file: Path,
    assets_file: Path,
    co2_file: Path,
    curated_root: Path,
    tls_sw: float,
    tls_pw: float,
    tls_cw: int,
) -> Dict[str, object]:
    cfg = BuildConfig(
        universe_file=universe_file,
        years=years,
        feature_cache_dir=feature_cache_dir,
        sentiment_file=sentiment_file,
        sentiment_primary_col=sentiment_primary_col,
        returns_file=returns_file,
        factors_ff3_file=ff3_file,
        factors_ff5_file=ff5_file,
        metadata_file=metadata_file,
        assets_file=assets_file,
        co2_file=co2_file,
        curated_root=curated_root,
        tls_sw=float(tls_sw),
        tls_pw=float(tls_pw),
        tls_cw=int(tls_cw),
        factor_quantile=0.25,
    )
    return build_all_canonical_tables(cfg)


def run_r1000_tls_only(
    *,
    curated_root: Path,
    tickers_file: Path,
    years: Sequence[int],
    returns_file: Path,
    out_dir: Path,
    min_n_yearly: int,
    min_n_pooled: int,
) -> Dict[str, pd.DataFrame]:
    tickers = parse_tickers(tickers_file)
    panel = load_panel(curated_root, tickers, years)
    factors_file = curated_root / "benchmark_factors_monthly.csv"
    beta_map = build_market_beta_map(panel, returns_file, factors_file)
    panel = panel.merge(beta_map[["ticker", "signal_year", "beta_mkt"]], on=["ticker", "signal_year"], how="left")
    yearly, pooled = run_regular_models(panel, min_n_yearly=min_n_yearly, min_n_pooled=min_n_pooled)
    iy, ip = run_interaction_models(panel, min_n_yearly=min_n_yearly, min_n_pooled=min_n_pooled)

    horizon_yearly = yearly[yearly["family"] == "tls_level"].copy() if not yearly.empty else pd.DataFrame()
    horizon_pooled = pooled[pooled["family"] == "tls_level"].copy() if not pooled.empty else pd.DataFrame()
    control_grid = build_control_grid_summary(horizon_pooled, ip)
    best_specs = best_specs_table(horizon_pooled)

    out_dir.mkdir(parents=True, exist_ok=True)
    horizon_yearly.to_csv(out_dir / "r1000_tls_horizon_regressions_yearly.csv", index=False)
    horizon_pooled.to_csv(out_dir / "r1000_tls_horizon_regressions_pooled.csv", index=False)
    iy.to_csv(out_dir / "r1000_tls_sector_interactions_yearly.csv", index=False)
    ip.to_csv(out_dir / "r1000_tls_sector_interactions_pooled.csv", index=False)
    control_grid.to_csv(out_dir / "r1000_tls_control_grid_summary.csv", index=False)
    best_specs.to_csv(out_dir / "r1000_tls_best_specs.csv", index=False)

    lines = [
        "# R1000 TLS Thesis Pipeline Summary",
        "",
        "## Pooled best specs",
        best_specs.to_markdown(index=False) if not best_specs.empty else "_No pooled TLS results._",
        "",
        "## Control-grid top rows",
        control_grid.head(20).to_markdown(index=False) if not control_grid.empty else "_No control-grid rows._",
    ]
    (out_dir / "r1000_tls_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "panel": panel,
        "yearly": horizon_yearly,
        "pooled": horizon_pooled,
        "interaction_yearly": iy,
        "interaction_pooled": ip,
        "control_grid": control_grid,
        "best_specs": best_specs,
    }


def pooled_signal_beta(panel: pd.DataFrame, signal_col: str, horizon_col: str = "ret_1y") -> Tuple[float, float, int]:
    d = panel[["signal_year", horizon_col, signal_col]].copy()
    d[signal_col] = pd.to_numeric(d[signal_col], errors="coerce")
    d[horizon_col] = pd.to_numeric(d[horizon_col], errors="coerce")
    d = d.dropna(subset=[signal_col, horizon_col]).copy()
    if len(d) < 30:
        return np.nan, np.nan, int(len(d))

    def _zscore(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        sd = float(x.std(ddof=0))
        if sd <= 0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (x - float(x.mean())) / sd

    d[f"z_{signal_col}"] = d.groupby("signal_year", group_keys=False)[signal_col].transform(_zscore)
    d = d.dropna(subset=[f"z_{signal_col}"]).copy()
    if len(d) < 30:
        return np.nan, np.nan, int(len(d))
    fit = smf.ols(f"{horizon_col} ~ z_{signal_col} + C(signal_year)", data=d).fit()
    return float(fit.params.get(f"z_{signal_col}", np.nan)), float(fit.pvalues.get(f"z_{signal_col}", np.nan)), int(fit.nobs)


def build_oriented_monthly_factor(
    *,
    signal_panel: pd.DataFrame,
    returns_file: Path,
    benchmark_factors: pd.DataFrame,
    signal_col: str,
    quantile: float,
    direction: str,
    fillna_zero: bool,
    factor_name: str,
) -> pd.DataFrame:
    d = signal_panel[["ticker", "signal_year", signal_col]].copy()
    d[signal_col] = pd.to_numeric(d[signal_col], errors="coerce")
    if fillna_zero:
        d[signal_col] = d[signal_col].fillna(0.0)
    d = d.dropna(subset=[signal_col]).copy()
    returns_m = _load_returns_monthly(returns_file, tickers=set(d["ticker"]))
    rows = []
    for y, g in d.groupby("signal_year"):
        if len(g) < 20:
            continue
        q_hi = float(g[signal_col].quantile(1.0 - quantile))
        q_lo = float(g[signal_col].quantile(quantile))
        high = set(g[g[signal_col] >= q_hi]["ticker"])
        low = set(g[g[signal_col] <= q_lo]["ticker"])
        if not high or not low:
            continue
        if direction == "high_minus_low":
            long_t, short_t = high, low
        elif direction == "low_minus_high":
            long_t, short_t = low, high
        else:
            raise ValueError(direction)
        start = pd.Timestamp(f"{int(y)}-12-31")
        end = pd.Timestamp(f"{int(y)+1}-12-31")
        w = returns_m[(returns_m["date"] > start) & (returns_m["date"] <= end)].copy()
        if w.empty:
            continue
        long_m = w[w["ticker"].isin(long_t)].groupby("month_end", as_index=False)["ret"].mean().rename(columns={"ret": "long_ret"})
        short_m = w[w["ticker"].isin(short_t)].groupby("month_end", as_index=False)["ret"].mean().rename(columns={"ret": "short_ret"})
        m = long_m.merge(short_m, on="month_end", how="inner")
        if m.empty:
            continue
        m["ls_ret"] = m["long_ret"] - m["short_ret"]
        m["signal_year"] = int(y)
        m["factor_name"] = factor_name
        m["direction"] = direction
        m["n_long"] = int(len(long_t))
        m["n_short"] = int(len(short_t))
        m["universe_n"] = int(len(g))
        rows.append(m)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["month_end", "long_ret", "short_ret", "ls_ret", "signal_year", "factor_name", "direction", "n_long", "n_short", "universe_n"])
    if out.empty:
        return out
    out = out.merge(benchmark_factors[["month_end", "rf"]], on="month_end", how="left")
    out["ls_excess"] = out["ls_ret"] - out["rf"]
    return out.drop(columns=["rf"]).sort_values(["month_end", "factor_name"]).reset_index(drop=True)


def build_universe_basket_monthly(returns_file: Path, tickers: Sequence[str], benchmark_factors: pd.DataFrame) -> pd.DataFrame:
    r = _load_returns_monthly(returns_file, tickers=set(tickers))
    b = r.groupby("month_end", as_index=False)["ret"].mean().rename(columns={"ret": "basket_ret"})
    b = b.merge(benchmark_factors[["month_end", "rf"]], on="month_end", how="left")
    b["basket_excess"] = b["basket_ret"] - b["rf"]
    return b.drop(columns=["rf"]).sort_values("month_end").reset_index(drop=True)


def run_ts_model(df: pd.DataFrame, y_col: str, x_cols: Sequence[str], model_name: str) -> Dict[str, object] | None:
    use = df[[y_col, *x_cols]].dropna().copy()
    if len(use) < 24:
        return None
    fit = sm.OLS(use[y_col], sm.add_constant(use[list(x_cols)])).fit()
    row: Dict[str, object] = {
        "model": model_name,
        "N": int(fit.nobs),
        "alpha_monthly": float(fit.params.get("const", np.nan)),
        "alpha_t": float(fit.tvalues.get("const", np.nan)),
        "alpha_p": float(fit.pvalues.get("const", np.nan)),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "alpha_annualized_approx": float((1.0 + float(fit.params.get("const", 0.0))) ** 12 - 1.0),
    }
    for c in x_cols:
        row[f"beta_{c}"] = float(fit.params.get(c, np.nan))
        row[f"t_{c}"] = float(fit.tvalues.get(c, np.nan))
        row[f"p_{c}"] = float(fit.pvalues.get(c, np.nan))
    return row


def run_unified_factor_models(
    *,
    universe_name: str,
    tickers: Sequence[str],
    panel: pd.DataFrame,
    sentiment_panel: pd.DataFrame,
    returns_file: Path,
    benchmark_factors: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, pd.DataFrame]:
    base = panel.copy()
    base = base.merge(sentiment_panel[["ticker", "signal_year", "sentiment_signal"]], on=["ticker", "signal_year"], how="left")
    base["sentiment_signal"] = pd.to_numeric(base["sentiment_signal"], errors="coerce")

    direction_rows = []
    directions: Dict[str, str] = {}
    for signal_col, factor_name, fill_zero in [
        ("tls_score", "tls_factor", False),
        ("sentiment_signal", "sentiment_factor", False),
        ("co2_scaled", "co2_factor", False),
    ]:
        beta, pval, nobs = pooled_signal_beta(base, signal_col, horizon_col="ret_1y")
        direction = "high_minus_low" if (np.isnan(beta) or beta >= 0) else "low_minus_high"
        directions[factor_name] = direction
        direction_rows.append({
            "universe": universe_name,
            "signal_col": signal_col,
            "factor_name": factor_name,
            "beta_ret_1y": beta,
            "p_ret_1y": pval,
            "N": nobs,
            "direction": direction,
            "fillna_zero": fill_zero,
        })

    factor_frames = []
    for signal_col, factor_name, fill_zero in [
        ("tls_score", "tls_factor", False),
        ("sentiment_signal", "sentiment_factor", False),
        ("co2_scaled", "co2_factor", False),
    ]:
        fac = build_oriented_monthly_factor(
            signal_panel=base,
            returns_file=returns_file,
            benchmark_factors=benchmark_factors,
            signal_col=signal_col,
            quantile=0.25,
            direction=directions[factor_name],
            fillna_zero=fill_zero,
            factor_name=factor_name,
        )
        fac["universe"] = universe_name
        factor_frames.append(fac)
    factor_monthly = pd.concat(factor_frames, ignore_index=True) if factor_frames else pd.DataFrame()
    direction_df = pd.DataFrame(direction_rows)

    wide = benchmark_factors[["month_end", "mkt_rf", "smb", "hml", "rf"]].copy()
    if not factor_monthly.empty:
        pivot = factor_monthly.pivot_table(index="month_end", columns="factor_name", values="ls_excess", aggfunc="first").reset_index()
        wide = wide.merge(pivot, on="month_end", how="left")
    basket = build_universe_basket_monthly(returns_file, tickers, benchmark_factors)
    basket_wide = basket.merge(wide, on="month_end", how="inner")

    basket_rows = []
    row = run_ts_model(basket_wide, "basket_excess", ["mkt_rf", "smb", "hml"], "basket_ff3")
    if row is not None:
        row["universe"] = universe_name
        basket_rows.append(row)
    overlap_cols = ["mkt_rf", "smb", "hml", "tls_factor", "sentiment_factor", "co2_factor"]
    basket_overlap = basket_wide.dropna(subset=overlap_cols).copy()
    row = run_ts_model(basket_overlap, "basket_excess", ["mkt_rf", "smb", "hml"], "basket_ff3_overlap")
    if row is not None:
        row["universe"] = universe_name
        basket_rows.append(row)
    row = run_ts_model(basket_wide, "basket_excess", ["mkt_rf", "smb", "hml", "tls_factor", "sentiment_factor", "co2_factor"], "basket_ff6")
    if row is not None:
        row["universe"] = universe_name
        basket_rows.append(row)
    basket_models = pd.DataFrame(basket_rows)

    target_rows = []
    for target, others in {
        "tls_factor": ["mkt_rf", "smb", "hml"],
        "sentiment_factor": ["mkt_rf", "smb", "hml"],
        "co2_factor": ["mkt_rf", "smb", "hml"],
    }.items():
        if target not in wide.columns:
            continue
        temp = wide.rename(columns={target: "target_excess"}).copy()
        row = run_ts_model(temp, "target_excess", ["mkt_rf", "smb", "hml"], f"{target}_ff3")
        if row is not None:
            row["universe"] = universe_name
            row["target_factor"] = target
            target_rows.append(row)
        other_custom = [x for x in ["tls_factor", "sentiment_factor", "co2_factor"] if x != target]
        row = run_ts_model(temp, "target_excess", ["mkt_rf", "smb", "hml", *other_custom], f"{target}_ff6_ex_target")
        if row is not None:
            row["universe"] = universe_name
            row["target_factor"] = target
            target_rows.append(row)
    target_models = pd.DataFrame(target_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    factor_monthly.to_csv(out_dir / f"{universe_name}_monthly_custom_factors.csv", index=False)
    basket.to_csv(out_dir / f"{universe_name}_basket_monthly.csv", index=False)
    basket_models.to_csv(out_dir / f"{universe_name}_basket_factor_models.csv", index=False)
    target_models.to_csv(out_dir / f"{universe_name}_target_factor_models.csv", index=False)
    direction_df.to_csv(out_dir / f"{universe_name}_factor_directions.csv", index=False)

    return {
        "factor_monthly": factor_monthly,
        "basket": basket,
        "basket_models": basket_models,
        "target_models": target_models,
        "directions": direction_df,
    }


def run_cross_sectional_horserace(universe_name: str, panel: pd.DataFrame, sentiment_panel: pd.DataFrame) -> pd.DataFrame:
    d = ensure_common_panel_fields(panel)
    d = d.merge(sentiment_panel[["ticker", "signal_year", "sentiment_signal"]], on=["ticker", "signal_year"], how="left")
    d["sentiment_signal"] = pd.to_numeric(d["sentiment_signal"], errors="coerce")
    d = d[d["has_complete_1y"].fillna(False)].copy()

    def _zscore(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        sd = float(x.std(ddof=0))
        if sd <= 0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (x - float(x.mean())) / sd

    for c in ["tls_score", "sentiment_signal", "co2_assets", "beta_mkt"]:
        d[f"z_{c}"] = d.groupby("signal_year", group_keys=False)[c].transform(_zscore)
    formula = "ret_1y ~ z_tls_score + z_sentiment_signal + z_beta_mkt + z_co2_assets + C(sector) + C(signal_year)"
    fit_row, _ = fit_formula(
        d,
        formula,
        min_n=100 if universe_name == "r1000" else 40,
        focus_terms=["z_tls_score", "z_sentiment_signal", "z_beta_mkt", "z_co2_assets"],
    )
    if fit_row is None:
        return pd.DataFrame()
    fit_row["universe"] = universe_name
    fit_row["formula_type"] = "combined_cross_section_1y"
    return fit_row


def signal_interaction_formula(signal_term: str, controls: Sequence[str], pooled: bool) -> Tuple[str, List[str]]:
    terms = [
        signal_term,
        f"{signal_term}:sector_energy",
        f"{signal_term}:sector_industrials",
        f"{signal_term}:sector_utilities_telco",
    ]
    focus = list(terms)
    terms.extend(f"z_{c}" for c in controls)
    terms.append("C(sector)")
    if pooled:
        terms.append("C(signal_year)")
    return f"ret_1y ~ " + " + ".join(terms), focus


def run_sentiment_sector_interactions(
    universe_name: str,
    panel: pd.DataFrame,
    sentiment_panel: pd.DataFrame,
    *,
    min_n_yearly: int,
    min_n_pooled: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = ensure_common_panel_fields(panel)
    d = d.merge(sentiment_panel[["ticker", "signal_year", "sentiment_signal"]], on=["ticker", "signal_year"], how="left")
    d["sentiment_signal"] = pd.to_numeric(d["sentiment_signal"], errors="coerce")
    d = make_sector_flags(d)

    pooled_rows: List[pd.DataFrame] = []
    yearly_rows: List[pd.DataFrame] = []
    for horizon in ["1y", "2y"]:
        meta = HORIZON_META[horizon]
        base = d[d[meta["complete_col"]].fillna(False)].copy()
        if base.empty:
            continue
        for col in ["sentiment_signal", "beta_mkt", "co2_assets"]:
            if col not in base.columns:
                base[col] = np.nan
        base = standardize(base, ["sentiment_signal", "beta_mkt", "co2_assets"])

        for control_name, controls in control_specs():
            formula_pooled, focus = signal_interaction_formula("z_sentiment_signal", controls, pooled=True)
            formula_pooled = formula_pooled.replace("ret_1y", meta["ret_col"])
            fit_row, _ = fit_formula(base, formula_pooled, min_n_pooled, focus)
            if fit_row is not None:
                fit_row["universe"] = universe_name
                fit_row["signal"] = "sentiment"
                fit_row["horizon"] = horizon
                fit_row["control_spec"] = control_name
                fit_row["scope"] = "pooled"
                pooled_rows.append(fit_row)
            for year, g in base.groupby("signal_year"):
                formula_yearly, focus = signal_interaction_formula("z_sentiment_signal", controls, pooled=False)
                formula_yearly = formula_yearly.replace("ret_1y", meta["ret_col"])
                fit_row, _ = fit_formula(g, formula_yearly, min_n_yearly, focus)
                if fit_row is not None:
                    fit_row["universe"] = universe_name
                    fit_row["signal"] = "sentiment"
                    fit_row["horizon"] = horizon
                    fit_row["control_spec"] = control_name
                    fit_row["scope"] = "yearly"
                    fit_row["signal_year"] = int(year)
                    yearly_rows.append(fit_row)
    pooled = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
    yearly = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    return yearly, pooled


def leave_year_out_sets(years: Sequence[int], max_k: int = 3) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    years = sorted({int(y) for y in years})
    for k in range(1, max_k + 1):
        out.extend(list(combinations(years, k)))
    return out


def run_leave_year_out_characteristic_robustness(
    universe_name: str,
    panel: pd.DataFrame,
    sentiment_panel: pd.DataFrame,
    *,
    min_n: int,
    max_k: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_panel = ensure_common_panel_fields(panel)
    base_panel = base_panel.merge(
        sentiment_panel[["ticker", "signal_year", "sentiment_signal"]],
        on=["ticker", "signal_year"],
        how="left",
    )
    base_panel["sentiment_signal"] = pd.to_numeric(base_panel["sentiment_signal"], errors="coerce")
    years = sorted(int(y) for y in base_panel["signal_year"].dropna().unique())
    omit_sets = [tuple()] + leave_year_out_sets(years, max_k=max_k)

    model_defs = [
        {
            "model_name": "tls_1y_controlled",
            "complete_col": "has_complete_1y",
            "outcome": "ret_1y",
            "raw_cols": ["tls_score", "beta_mkt", "co2_assets"],
            "focus_terms": ["z_tls_score"],
            "formula": "ret_1y ~ z_tls_score + z_beta_mkt + z_co2_assets + C(sector) + C(signal_year)",
        },
        {
            "model_name": "tls_2y_controlled",
            "complete_col": "has_complete_2y",
            "outcome": "ret_2y",
            "raw_cols": ["tls_score", "beta_mkt", "co2_assets"],
            "focus_terms": ["z_tls_score"],
            "formula": "ret_2y ~ z_tls_score + z_beta_mkt + z_co2_assets + C(sector) + C(signal_year)",
        },
        {
            "model_name": "sentiment_1y_screen",
            "complete_col": "has_complete_1y",
            "outcome": "ret_1y",
            "raw_cols": ["sentiment_signal"],
            "focus_terms": ["z_sentiment_signal"],
            "formula": "ret_1y ~ z_sentiment_signal + C(signal_year)",
        },
        {
            "model_name": "combined_1y",
            "complete_col": "has_complete_1y",
            "outcome": "ret_1y",
            "raw_cols": ["tls_score", "sentiment_signal", "beta_mkt", "co2_assets"],
            "focus_terms": ["z_tls_score", "z_sentiment_signal"],
            "formula": "ret_1y ~ z_tls_score + z_sentiment_signal + z_beta_mkt + z_co2_assets + C(sector) + C(signal_year)",
        },
    ]

    detailed_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for model_def in model_defs:
        model_name = model_def["model_name"]
        base = base_panel[base_panel[model_def["complete_col"]].fillna(False)].copy()
        full_betas: Dict[str, float] = {}
        run_rows: List[Dict[str, object]] = []
        for omitted in omit_sets:
            d = base.copy()
            if omitted:
                d = d[~d["signal_year"].isin(set(omitted))].copy()
            d = standardize(d, model_def["raw_cols"])
            fit_row, _ = fit_formula(d, model_def["formula"], min_n, model_def["focus_terms"])
            if fit_row is None:
                continue
            row = fit_row.iloc[0].to_dict()
            row["universe"] = universe_name
            row["model_name"] = model_name
            row["omit_k"] = int(len(omitted))
            row["omitted_years"] = ",".join(str(x) for x in omitted)
            row["n_years_retained"] = int(len(set(d["signal_year"].dropna().astype(int))))
            run_rows.append(row)
            if not omitted:
                for focus in model_def["focus_terms"]:
                    full_betas[focus] = float(row.get(f"beta_{focus}", np.nan))
        detailed_rows.extend(run_rows)

        for focus in model_def["focus_terms"]:
            full_beta = full_betas.get(focus, np.nan)
            beta_col = f"beta_{focus}"
            p_col = f"p_{focus}"
            for k in range(1, max_k + 1):
                g = [r for r in run_rows if int(r["omit_k"]) == k]
                if not g:
                    continue
                betas = np.array([float(r.get(beta_col, np.nan)) for r in g], dtype=float)
                pvals = np.array([float(r.get(p_col, np.nan)) for r in g], dtype=float)
                valid_beta = np.isfinite(betas)
                valid_p = np.isfinite(pvals)
                same_sign = np.sign(betas[valid_beta]) == np.sign(full_beta) if np.isfinite(full_beta) else np.zeros(valid_beta.sum(), dtype=bool)
                summary_rows.append({
                    "universe": universe_name,
                    "model_name": model_name,
                    "focus_term": focus,
                    "omit_k": k,
                    "n_runs": int(len(g)),
                    "full_sample_beta": float(full_beta) if np.isfinite(full_beta) else np.nan,
                    "share_same_sign_as_full": float(same_sign.mean()) if same_sign.size else np.nan,
                    "share_p_lt_0_10": float((pvals[valid_p] < 0.10).mean()) if valid_p.any() else np.nan,
                    "share_p_lt_0_05": float((pvals[valid_p] < 0.05).mean()) if valid_p.any() else np.nan,
                    "beta_min": float(np.nanmin(betas)) if valid_beta.any() else np.nan,
                    "beta_median": float(np.nanmedian(betas)) if valid_beta.any() else np.nan,
                    "beta_max": float(np.nanmax(betas)) if valid_beta.any() else np.nan,
                })

    detailed = pd.DataFrame(detailed_rows)
    summary = pd.DataFrame(summary_rows)
    return detailed, summary
def main() -> None:
    ap = argparse.ArgumentParser(description="Run the canonical thesis characteristic analysis")
    ap.add_argument("--r1000-tickers-file", type=Path, default=Path("data/raw/universe/tickers.txt"))
    ap.add_argument("--t100-tickers-file", type=Path, default=Path("data/raw/universe/transition_100_tickers.txt"))
    ap.add_argument("--r1000-years", type=str, default="2015,2016,2017,2018,2019,2020,2021,2022,2023,2024")
    ap.add_argument("--t100-years", type=str, default="2015,2016,2017,2018,2019,2020,2021,2022,2023,2024")
    ap.add_argument("--cache-manifest", type=Path, default=Path("data/cache/edgar_html/cache_manifest.csv"))
    ap.add_argument("--text-cache-dir", type=Path, default=Path("data/cache/edgar_text"))
    ap.add_argument("--r1000-feature-cache-dir", type=Path, default=Path("data/processed/feature_cache/r1000"))
    ap.add_argument("--returns-file", type=Path, default=Path("data/raw/returns/daily_ret_10y_full_r1000.csv"))
    ap.add_argument("--ff3-file", type=Path, default=Path("data/raw/factors/ff_factors_clean.csv"))
    ap.add_argument("--ff5-file", type=Path, default=Path("data/raw/factors/ff5_factors_clean.csv"))
    ap.add_argument("--metadata-file", type=Path, default=Path("data/raw/metadata/ticker_addtl_data.csv"))
    ap.add_argument("--assets-file", type=Path, default=Path("data/raw/assets/asset_data.csv"))
    ap.add_argument("--co2-file", type=Path, default=Path("data/raw/emissions/co2data.csv"))
    ap.add_argument("--r1000-curated-root", type=Path, default=Path("data/curated/r1000/final_within_year"))
    ap.add_argument("--t100-curated-root", type=Path, default=Path("data/curated/t100/final_within_year_shared"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/outputs/final/within_year"))
    ap.add_argument("--r1000-sentiment-file", type=Path, default=None)
    ap.add_argument("--t100-sentiment-file", type=Path, default=None)
    ap.add_argument("--r1000-sentiment-primary-col", type=str, default="transition_sentiment_median")
    ap.add_argument("--t100-sentiment-primary-col", type=str, default="transition_sentiment_median")
    ap.add_argument("--r1000-tls-sw", type=float, default=3.25)
    ap.add_argument("--r1000-tls-pw", type=float, default=1.0)
    ap.add_argument("--r1000-tls-cw", type=int, default=10)
    ap.add_argument("--r1000-sentiment-score-col", type=str, default="transition_sentiment_median")
    ap.add_argument("--t100-sentiment-score-col", type=str, default="transition_sentiment_median")
    ap.add_argument("--r1000-sentiment-missing-policy", choices=["drop", "zero"], default="drop")
    ap.add_argument("--t100-sentiment-missing-policy", choices=["drop", "zero"], default="drop")
    ap.add_argument("--r1000-sentiment-transform", choices=["raw", "winsor_1_99", "winsor_5_95"], default="winsor_5_95")
    ap.add_argument("--t100-sentiment-transform", choices=["raw", "winsor_1_99", "winsor_5_95"], default="winsor_5_95")
    ap.add_argument("--force-sentiment-refresh", action="store_true")
    ap.add_argument("--min-n-yearly-r1000", type=int, default=100)
    ap.add_argument("--min-n-pooled-r1000", type=int, default=500)
    args = ap.parse_args()

    r1000_years = parse_years(args.r1000_years)
    t100_years = parse_years(args.t100_years)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.r1000_sentiment_file is not None:
        r1000_sent = args.r1000_sentiment_file
    else:
        r1000_sent = ensure_best_sentiment_file(
            tickers_file=args.r1000_tickers_file,
            years=r1000_years,
            cache_manifest=args.cache_manifest,
            text_cache_dir=args.text_cache_dir,
            out_file=Path("data/processed/search/sentiment/r1000/r1000_finbert_best_current_2015_2024.csv"),
            sentences_out=Path("data/processed/search/sentiment/r1000/r1000_finbert_best_current_2015_2024_sentences.csv"),
            dropped_out=Path("data/processed/search/sentiment/r1000/r1000_finbert_best_current_2015_2024_dropped.csv"),
            force=args.force_sentiment_refresh,
        )
    if args.t100_sentiment_file is not None:
        t100_sent = args.t100_sentiment_file
    else:
        t100_sent = ensure_best_sentiment_file(
            tickers_file=args.t100_tickers_file,
            years=t100_years,
            cache_manifest=args.cache_manifest,
            text_cache_dir=args.text_cache_dir,
            out_file=Path("data/processed/search/sentiment/t100/transition100_finbert_best_current_2015_2024.csv"),
            sentences_out=Path("data/processed/search/sentiment/t100/transition100_finbert_best_current_2015_2024_sentences.csv"),
            dropped_out=Path("data/processed/search/sentiment/t100/transition100_finbert_best_current_2015_2024_dropped.csv"),
            force=args.force_sentiment_refresh,
        )

    manifest = build_r1000_canonical(
        universe_file=args.r1000_tickers_file,
        years=r1000_years,
        feature_cache_dir=args.r1000_feature_cache_dir,
        sentiment_file=r1000_sent,
        sentiment_primary_col=args.r1000_sentiment_primary_col,
        returns_file=args.returns_file,
        ff3_file=args.ff3_file,
        ff5_file=args.ff5_file,
        metadata_file=args.metadata_file,
        assets_file=args.assets_file,
        co2_file=args.co2_file,
        curated_root=args.r1000_curated_root,
        tls_sw=args.r1000_tls_sw,
        tls_pw=args.r1000_tls_pw,
        tls_cw=args.r1000_tls_cw,
    )

    r1000_tls = run_r1000_tls_only(
        curated_root=args.r1000_curated_root,
        tickers_file=args.r1000_tickers_file,
        years=r1000_years,
        returns_file=args.returns_file,
        out_dir=out_dir / "r1000_tls",
        min_n_yearly=args.min_n_yearly_r1000,
        min_n_pooled=args.min_n_pooled_r1000,
    )

    bench_r1000 = pd.read_csv(args.r1000_curated_root / "benchmark_factors_monthly.csv")
    bench_r1000["month_end"] = pd.to_datetime(bench_r1000["month_end"], errors="coerce")
    bench_t100 = pd.read_csv(args.t100_curated_root / "benchmark_factors_monthly.csv")
    bench_t100["month_end"] = pd.to_datetime(bench_t100["month_end"], errors="coerce")

    r1000_panel = pd.read_csv(args.r1000_curated_root / "ticker_year_panel.csv")
    r1000_panel["ticker"] = r1000_panel["ticker"].astype(str).str.upper().str.strip()
    r1000_panel = r1000_panel[r1000_panel["signal_year"].isin(set(r1000_years))].copy()
    r1000_panel = r1000_panel.merge(build_market_beta_map(r1000_panel, args.returns_file, args.r1000_curated_root / "benchmark_factors_monthly.csv")[["ticker", "signal_year", "beta_mkt"]], on=["ticker", "signal_year"], how="left")
    r1000_panel = ensure_common_panel_fields(r1000_panel)

    t100_panel = pd.read_csv(args.t100_curated_root / "ticker_year_panel.csv")
    t100_panel["ticker"] = t100_panel["ticker"].astype(str).str.upper().str.strip()
    t100_panel = t100_panel[t100_panel["signal_year"].isin(set(t100_years))].copy()
    t100_panel = t100_panel.merge(build_market_beta_map(t100_panel, args.returns_file, args.t100_curated_root / "benchmark_factors_monthly.csv")[["ticker", "signal_year", "beta_mkt"]], on=["ticker", "signal_year"], how="left")
    t100_panel = ensure_common_panel_fields(t100_panel)

    r1000_sent_panel = load_sentiment_panel(r1000_sent, parse_tickers(args.r1000_tickers_file), r1000_years)
    r1000_sent_panel = add_processed_sentiment_signal(
        r1000_sent_panel,
        score_col=args.r1000_sentiment_score_col,
        missing_score_policy=args.r1000_sentiment_missing_policy,
        score_transform=args.r1000_sentiment_transform,
    )
    t100_sent_panel = load_sentiment_panel(t100_sent, parse_tickers(args.t100_tickers_file), t100_years)
    t100_sent_panel = add_processed_sentiment_signal(
        t100_sent_panel,
        score_col=args.t100_sentiment_score_col,
        missing_score_policy=args.t100_sentiment_missing_policy,
        score_transform=args.t100_sentiment_transform,
    )

    r1000_factors = run_unified_factor_models(
        universe_name="r1000",
        tickers=parse_tickers(args.r1000_tickers_file),
        panel=r1000_panel,
        sentiment_panel=r1000_sent_panel,
        returns_file=args.returns_file,
        benchmark_factors=bench_r1000,
        out_dir=out_dir / "unified_factor_models",
    )
    t100_factors = run_unified_factor_models(
        universe_name="t100",
        tickers=parse_tickers(args.t100_tickers_file),
        panel=t100_panel,
        sentiment_panel=t100_sent_panel,
        returns_file=args.returns_file,
        benchmark_factors=bench_t100,
        out_dir=out_dir / "unified_factor_models",
    )

    horserace = pd.concat([
        run_cross_sectional_horserace("r1000", r1000_panel, r1000_sent_panel),
        run_cross_sectional_horserace("t100", t100_panel, t100_sent_panel),
    ], ignore_index=True)
    horserace.to_csv(out_dir / "cross_sectional_combined_signal_models.csv", index=False)

    r1000_sent_inter_yearly, r1000_sent_inter_pooled = run_sentiment_sector_interactions(
        "r1000",
        r1000_panel,
        r1000_sent_panel,
        min_n_yearly=args.min_n_yearly_r1000,
        min_n_pooled=args.min_n_pooled_r1000,
    )
    t100_sent_inter_yearly, t100_sent_inter_pooled = run_sentiment_sector_interactions(
        "t100",
        t100_panel,
        t100_sent_panel,
        min_n_yearly=40,
        min_n_pooled=100,
    )
    sentiment_inter_yearly = pd.concat([r1000_sent_inter_yearly, t100_sent_inter_yearly], ignore_index=True)
    sentiment_inter_pooled = pd.concat([r1000_sent_inter_pooled, t100_sent_inter_pooled], ignore_index=True)
    sentiment_inter_yearly.to_csv(out_dir / "sentiment_sector_interactions_yearly.csv", index=False)
    sentiment_inter_pooled.to_csv(out_dir / "sentiment_sector_interactions_pooled.csv", index=False)

    r1000_leave_detailed, r1000_leave_summary = run_leave_year_out_characteristic_robustness(
        "r1000",
        r1000_panel,
        r1000_sent_panel,
        min_n=args.min_n_pooled_r1000,
    )
    t100_leave_detailed, t100_leave_summary = run_leave_year_out_characteristic_robustness(
        "t100",
        t100_panel,
        t100_sent_panel,
        min_n=100,
    )
    leave_detailed = pd.concat([r1000_leave_detailed, t100_leave_detailed], ignore_index=True)
    leave_summary = pd.concat([r1000_leave_summary, t100_leave_summary], ignore_index=True)
    leave_detailed.to_csv(out_dir / "leave_year_out_robustness_detailed.csv", index=False)
    leave_summary.to_csv(out_dir / "leave_year_out_robustness_summary.csv", index=False)

    basket_models = pd.concat([r1000_factors["basket_models"], t100_factors["basket_models"]], ignore_index=True)
    target_models = pd.concat([r1000_factors["target_models"], t100_factors["target_models"]], ignore_index=True)
    direction_summary = pd.concat([r1000_factors["directions"], t100_factors["directions"]], ignore_index=True)
    basket_models.to_csv(out_dir / "unified_basket_factor_models_all.csv", index=False)
    target_models.to_csv(out_dir / "unified_target_factor_models_all.csv", index=False)
    direction_summary.to_csv(out_dir / "unified_factor_direction_summary.csv", index=False)

    report_lines = [
        "# R1000 TLS + Unified Multifactor Summary",
        "",
        "## R1000 canonical rebuild",
        f"- Curated root: `{args.r1000_curated_root}`",
        f"- Panel hash: `{manifest['output_hashes']['ticker_year_panel_sha256']}`",
        f"- Panel rows: {manifest['stats'].get('panel_rows')}",
        f"- Unique tickers: {manifest['stats'].get('panel_unique_tickers')}",
        "",
        "## Standalone R1000 TLS pooled best specs",
        r1000_tls["best_specs"].to_markdown(index=False) if not r1000_tls["best_specs"].empty else "_No pooled R1000 TLS results._",
        "",
        "## Cross-sectional combined TLS + sentiment + beta + CO2/assets",
        horserace.to_markdown(index=False) if not horserace.empty else "_No combined cross-sectional models._",
        "",
        "## Sentiment sector interactions (1Y pooled)",
        sentiment_inter_pooled.to_markdown(index=False) if not sentiment_inter_pooled.empty else "_No sentiment interaction models._",
        "",
        "## Leave-1/2/3-years-out characteristic robustness summary",
        leave_summary.to_markdown(index=False) if not leave_summary.empty else "_No leave-year-out robustness results._",
        "",
        "## Basket FF3 vs FF6 models",
        basket_models.to_markdown(index=False) if not basket_models.empty else "_No basket factor models._",
        "",
        "## Target-factor FF3 vs pooled-custom models",
        target_models.to_markdown(index=False) if not target_models.empty else "_No target factor models._",
        "",
        "## Signal directions used for monthly factor construction",
        direction_summary.to_markdown(index=False) if not direction_summary.empty else "_No factor directions._",
    ]
    report_path = out_dir / "r1000_tls_and_unified_factor_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OUT] {report_path}")
    print(f"[OUT] {out_dir / 'r1000_tls/r1000_tls_horizon_regressions_pooled.csv'}")
    print(f"[OUT] {out_dir / 'sentiment_sector_interactions_pooled.csv'}")
    print(f"[OUT] {out_dir / 'leave_year_out_robustness_summary.csv'}")
    print(f"[OUT] {out_dir / 'unified_basket_factor_models_all.csv'}")
    print(f"[OUT] {out_dir / 'unified_target_factor_models_all.csv'}")


if __name__ == "__main__":
    main()
