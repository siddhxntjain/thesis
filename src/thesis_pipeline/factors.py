#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from thesis_pipeline.canonical import _load_returns_monthly


ROOT = Path(__file__).resolve().parents[2]
SELECTED_SPECS = ROOT / "data/outputs/final/within_year/selected_optimized_specs.csv"
LEGACY_RESULT_SUMMARY = ROOT / "rerun_verification_summary.md"

CUTOFFS: List[float] = [0.10, 0.25, 1.0 / 3.0]
BENCHMARK_ORDER: List[str] = ["mkt_rf", "smb", "hml", "rmw", "cma"]
CUSTOM_FACTOR_ORDER: List[str] = ["tls_factor", "sentiment_factor", "gmb_u", "gmb_s"]
DISPLAY_NAMES: Dict[str, str] = {
    "mkt_rf": "Mkt-RF",
    "smb": "SMB",
    "hml": "HML",
    "rmw": "RMW",
    "cma": "CMA",
    "tls_factor": "TLS",
    "sentiment_factor": "Sentiment",
    "gmb_u": r"GMB$_U$",
    "gmb_s": r"GMB$_S$",
}
EXPECTED_DIRECTIONS: Dict[str, float] = {
    "tls_factor": 1.0,
    "sentiment_factor": 1.0,
    "gmb_u": 1.0,
    "gmb_s": 1.0,
}
SORT_DIRECTIONS: Dict[str, str] = {
    "tls_factor": "high_minus_low",
    "sentiment_factor": "low_minus_high",
    "gmb_u": "low_minus_high",
    "gmb_s": "low_minus_high",
}


@dataclass(frozen=True)
class UniverseSpec:
    key: str
    label: str
    curated_root: Path


UNIVERSES: List[UniverseSpec] = [
    UniverseSpec("r1000", "Russell 1000", ROOT / "data/curated/r1000/final_within_year"),
    UniverseSpec("t100", "Transition-100", ROOT / "data/curated/t100/final_within_year_shared"),
]


def stars(p: float | None) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def format_pct(x: float | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100.0 * float(x):.{digits}f}"


def format_num(x: float | None, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def format_p(x: float | None) -> str:
    if x is None or pd.isna(x):
        return ""
    if float(x) < 0.001:
        return "<0.001"
    return f"{float(x):.3f}"


def display_cutoff(cutoff: float) -> str:
    if np.isclose(cutoff, 1.0 / 3.0):
        return "33%"
    return f"{int(round(100 * cutoff))}%"


def load_manifest_inputs(curated_root: Path) -> Dict[str, object]:
    manifest_path = curated_root / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest["inputs"]


def load_locked_sentiment_specs() -> Dict[str, Dict[str, object]]:
    df = pd.read_csv(SELECTED_SPECS)
    out: Dict[str, Dict[str, object]] = {}
    for universe in ["r1000", "t100"]:
        row = df[(df["universe"] == universe) & (df["signal"] == "sentiment")]
        if row.empty:
            raise KeyError(f"No locked sentiment spec found for {universe}")
        out[universe] = row.iloc[0].to_dict()
    return out


def map_sentiment_score_column(score_name: str) -> str:
    mapping = {
        "transition_sentiment_median": "sentiment_median",
        "transition_sentiment_mean": "sentiment_mean",
        "transition_stance_index": "year_sentiment",
        "transition_sentiment_median_zero": "year_sentiment",
        "transition_sentiment_mean_zero": "year_sentiment",
        "transition_stance_index_zero": "year_sentiment",
    }
    if score_name not in mapping:
        raise KeyError(f"Unsupported sentiment score name: {score_name}")
    return mapping[score_name]


def winsorize_by_group(series: pd.Series, q_low: float, q_high: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    lo = float(s.quantile(q_low))
    hi = float(s.quantile(q_high))
    return s.clip(lower=lo, upper=hi)


def build_locked_sentiment_signal(panel: pd.DataFrame, spec_row: Dict[str, object]) -> pd.Series:
    source_col = map_sentiment_score_column(str(spec_row["score_name"]))
    raw = pd.to_numeric(panel[source_col], errors="coerce")

    policy = str(spec_row["missing_score_policy"])
    if policy == "zero":
        raw = raw.fillna(0.0)
    elif policy != "drop":
        raise ValueError(f"Unsupported missing policy: {policy}")

    transform = str(spec_row["transform"])
    if transform == "raw":
        out = raw
    elif transform == "winsor_1_99":
        out = panel.assign(_raw=raw).groupby("signal_year", group_keys=False)["_raw"].apply(
            lambda s: winsorize_by_group(s, 0.01, 0.99)
        )
    elif transform == "winsor_5_95":
        out = panel.assign(_raw=raw).groupby("signal_year", group_keys=False)["_raw"].apply(
            lambda s: winsorize_by_group(s, 0.05, 0.95)
        )
    else:
        raise ValueError(f"Unsupported sentiment transform: {transform}")
    return pd.to_numeric(out, errors="coerce")


def prepare_universe_panel(universe: UniverseSpec, sentiment_specs: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    panel = pd.read_csv(universe.curated_root / "ticker_year_panel.csv")
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
    panel["signal_year"] = pd.to_numeric(panel["signal_year"], errors="coerce").astype(int)
    panel["has_complete_1y"] = panel["has_complete_1y"].fillna(False).astype(bool)
    panel["sector"] = panel["sector"].fillna("Unknown").astype(str)
    panel["tls_factor_input"] = pd.to_numeric(panel["tls_score"], errors="coerce")
    panel["sentiment_factor_input"] = build_locked_sentiment_signal(panel, sentiment_specs[universe.key])
    co2 = pd.to_numeric(panel["co2_unscaled"], errors="coerce")
    assets = pd.to_numeric(panel["year_assets"], errors="coerce")
    panel["gmb_u_input"] = np.where(co2 > 0, np.log(co2), np.nan)
    panel["gmb_s_input"] = np.where((co2 >= 0) & (assets > 0), co2 / assets, np.nan)
    return panel


def build_factor_membership(
    signal_panel: pd.DataFrame,
    signal_col: str,
    cutoff: float,
    direction: str,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for year, group in signal_panel.groupby("signal_year"):
        d = group.loc[group["has_complete_1y"], ["ticker", "signal_year", signal_col]].copy()
        d = d.dropna(subset=[signal_col]).sort_values(signal_col).reset_index(drop=True)
        if d.empty:
            continue
        universe_n = len(d)
        bucket_n = max(1, int(np.floor(universe_n * cutoff)))
        if bucket_n * 2 > universe_n:
            continue
        if direction == "high_minus_low":
            short_tickers = set(d.head(bucket_n)["ticker"])
            long_tickers = set(d.tail(bucket_n)["ticker"])
        elif direction == "low_minus_high":
            long_tickers = set(d.head(bucket_n)["ticker"])
            short_tickers = set(d.tail(bucket_n)["ticker"])
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        if not long_tickers or not short_tickers:
            continue
        out = d[["ticker", "signal_year"]].copy()
        out["side"] = "MID"
        out.loc[out["ticker"].isin(long_tickers), "side"] = "LONG"
        out.loc[out["ticker"].isin(short_tickers), "side"] = "SHORT"
        out = out[out["side"] != "MID"].copy()
        out["universe_n"] = universe_n
        out["n_long"] = len(long_tickers)
        out["n_short"] = len(short_tickers)
        out["cutoff"] = cutoff
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["ticker", "signal_year", "side", "universe_n", "n_long", "n_short", "cutoff"])
    return pd.concat(rows, ignore_index=True)


def build_monthly_factor_series(
    membership: pd.DataFrame,
    returns_m: pd.DataFrame,
    factor_name: str,
    universe_key: str,
    cutoff: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for year, group in membership.groupby("signal_year"):
        start = pd.Timestamp(f"{int(year)}-12-31")
        end = pd.Timestamp(f"{int(year) + 1}-12-31")
        hold = returns_m[(returns_m["date"] > start) & (returns_m["date"] <= end)].copy()
        if hold.empty:
            continue
        long_tickers = set(group.loc[group["side"] == "LONG", "ticker"])
        short_tickers = set(group.loc[group["side"] == "SHORT", "ticker"])
        if not long_tickers or not short_tickers:
            continue
        long_m = (
            hold[hold["ticker"].isin(long_tickers)]
            .groupby("month_end", as_index=False)["ret"]
            .mean()
            .rename(columns={"ret": "long_ret"})
        )
        short_m = (
            hold[hold["ticker"].isin(short_tickers)]
            .groupby("month_end", as_index=False)["ret"]
            .mean()
            .rename(columns={"ret": "short_ret"})
        )
        m = long_m.merge(short_m, on="month_end", how="inner")
        if m.empty:
            continue
        m["factor_ret"] = m["long_ret"] - m["short_ret"]
        m["universe"] = universe_key
        m["factor_name"] = factor_name
        m["signal_year"] = int(year)
        m["cutoff"] = cutoff
        m["n_long"] = int(group["n_long"].iloc[0])
        m["n_short"] = int(group["n_short"].iloc[0])
        m["universe_n"] = int(group["universe_n"].iloc[0])
        rows.extend(m.to_dict("records"))
    if not rows:
        return pd.DataFrame(
            columns=[
                "month_end",
                "long_ret",
                "short_ret",
                "factor_ret",
                "universe",
                "factor_name",
                "signal_year",
                "cutoff",
                "n_long",
                "n_short",
                "universe_n",
            ]
        )
    out = pd.DataFrame(rows).sort_values(["factor_name", "month_end"]).reset_index(drop=True)
    return out


def build_custom_factor_long(
    universe: UniverseSpec,
    panel: pd.DataFrame,
    returns_m: pd.DataFrame,
    cutoffs: Sequence[float],
) -> pd.DataFrame:
    signal_map = {
        "tls_factor": "tls_factor_input",
        "sentiment_factor": "sentiment_factor_input",
        "gmb_u": "gmb_u_input",
        "gmb_s": "gmb_s_input",
    }
    frames: List[pd.DataFrame] = []
    for cutoff in cutoffs:
        for factor_name, signal_col in signal_map.items():
            membership = build_factor_membership(
                signal_panel=panel,
                signal_col=signal_col,
                cutoff=cutoff,
                direction=SORT_DIRECTIONS[factor_name],
            )
            if membership.empty:
                continue
            monthly = build_monthly_factor_series(
                membership=membership,
                returns_m=returns_m,
                factor_name=factor_name,
                universe_key=universe.key,
                cutoff=cutoff,
            )
            frames.append(monthly)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_factor_wide(
    factor_long: pd.DataFrame,
    benchmark_factors: pd.DataFrame,
    cutoff: float,
) -> pd.DataFrame:
    subset = factor_long[np.isclose(factor_long["cutoff"], cutoff)].copy()
    if subset.empty:
        return pd.DataFrame()
    pivot = subset.pivot_table(index="month_end", columns="factor_name", values="factor_ret", aggfunc="first").reset_index()
    cols = ["month_end", "rf", *BENCHMARK_ORDER]
    wide = benchmark_factors[cols].copy().merge(pivot, on="month_end", how="inner")
    wide = wide.sort_values("month_end").reset_index(drop=True)
    return wide


def compute_summary_stats(wide: pd.DataFrame, universe_key: str, cutoff: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    order = [*BENCHMARK_ORDER, *CUSTOM_FACTOR_ORDER]
    if wide.empty:
        return pd.DataFrame()
    for name in order:
        if name not in wide.columns:
            continue
        s = pd.to_numeric(wide[name], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "universe": universe_key,
                "cutoff": cutoff,
                "factor": name,
                "mean": float(s.mean()),
                "sd": float(s.std(ddof=1)),
                "min": float(s.min()),
                "max": float(s.max()),
                "mean_pct": float(100.0 * s.mean()),
                "sd_pct": float(100.0 * s.std(ddof=1)),
                "min_pct": float(100.0 * s.min()),
                "max_pct": float(100.0 * s.max()),
                "n_months": int(s.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def compute_correlation_matrix(wide: pd.DataFrame, universe_key: str, cutoff: float) -> pd.DataFrame:
    order = [*BENCHMARK_ORDER, *CUSTOM_FACTOR_ORDER]
    cols = [c for c in order if c in wide.columns]
    use = wide[cols].dropna()
    corr = use.corr()
    corr.insert(0, "row_factor", corr.index)
    corr.insert(0, "universe", universe_key)
    corr.insert(1, "cutoff", cutoff)
    return corr.reset_index(drop=True)


def fit_factor_alpha_models(wide: pd.DataFrame, universe_key: str, cutoff: float) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for factor_name in CUSTOM_FACTOR_ORDER:
        if factor_name not in wide.columns:
            continue
        model_specs = [
            ("capm", ["mkt_rf"]),
            ("ff3", ["mkt_rf", "smb", "hml"]),
            ("ff5", ["mkt_rf", "smb", "hml", "rmw", "cma"]),
            ("ff5_plus_other_custom", ["mkt_rf", "smb", "hml", "rmw", "cma", *[x for x in CUSTOM_FACTOR_ORDER if x != factor_name]]),
        ]
        for model_name, x_cols in model_specs:
            needed = [factor_name, *x_cols]
            use = wide[needed].dropna().copy()
            if len(use) < 24:
                continue
            fit = sm.OLS(use[factor_name], sm.add_constant(use[x_cols])).fit()
            alpha = float(fit.params.get("const", np.nan))
            p_value = float(fit.pvalues.get("const", np.nan))
            expected = EXPECTED_DIRECTIONS[factor_name]
            survives = bool(np.isfinite(alpha) and np.isfinite(p_value) and alpha * expected > 0 and p_value < 0.05)
            borderline = bool(np.isfinite(alpha) and np.isfinite(p_value) and alpha * expected > 0 and 0.05 <= p_value < 0.10)
            rows.append(
                {
                    "universe": universe_key,
                    "cutoff": cutoff,
                    "factor_name": factor_name,
                    "model": model_name,
                    "N": int(fit.nobs),
                    "alpha_monthly": alpha,
                    "alpha_t": float(fit.tvalues.get("const", np.nan)),
                    "alpha_p": p_value,
                    "alpha_annualized_approx": float((1.0 + alpha) ** 12 - 1.0),
                    "r2": float(fit.rsquared),
                    "adj_r2": float(fit.rsquared_adj),
                    "survives_conventionally": survives,
                    "borderline": borderline,
                }
            )
    return pd.DataFrame(rows)


def build_stock_month_panel(
    universe: UniverseSpec,
    panel: pd.DataFrame,
    returns_m: pd.DataFrame,
    headline_wide: pd.DataFrame,
) -> pd.DataFrame:
    if headline_wide.empty:
        return pd.DataFrame()
    tickers = set(panel["ticker"].astype(str).str.upper().str.strip())
    sectors = (
        panel[["ticker", "sector"]]
        .dropna(subset=["ticker"])
        .sort_values(["ticker", "sector"])
        .drop_duplicates("ticker", keep="first")
        .copy()
    )
    hold_months = set(pd.to_datetime(headline_wide["month_end"]))
    stock = returns_m[returns_m["ticker"].isin(tickers) & returns_m["month_end"].isin(hold_months)].copy()
    stock = stock.merge(sectors, on="ticker", how="left")
    merged = stock.merge(headline_wide, on="month_end", how="inner", suffixes=("", "_bench"))
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["excess_ret"] = pd.to_numeric(merged["ret"], errors="coerce") - pd.to_numeric(merged["rf"], errors="coerce")
    merged["universe"] = universe.key
    merged["month_cluster"] = merged["month_end"].astype(str)
    return merged


def fit_stock_panel_model(
    df: pd.DataFrame,
    universe_key: str,
    model_name: str,
    x_cols: Sequence[str],
    sector_fe: bool,
) -> Dict[str, object] | None:
    needed = ["excess_ret", *x_cols]
    if sector_fe:
        needed.append("sector")
    use = df[["excess_ret", "sector", "month_cluster", *x_cols]].dropna(subset=needed).copy()
    if use.empty or use["month_cluster"].nunique() < 24:
        return None
    rhs = " + ".join(x_cols)
    if sector_fe:
        rhs = rhs + " + C(sector)"
    formula = f"excess_ret ~ {rhs}"
    fit = smf.ols(formula, data=use).fit(
        cov_type="cluster",
        cov_kwds={"groups": use["month_cluster"]},
    )
    row: Dict[str, object] = {
        "universe": universe_key,
        "model": model_name,
        "sector_fe": bool(sector_fe),
        "N": int(fit.nobs),
        "n_months": int(use["month_cluster"].nunique()),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "coef_const": float(fit.params.get("Intercept", np.nan)),
        "t_const": float(fit.tvalues.get("Intercept", np.nan)),
        "p_const": float(fit.pvalues.get("Intercept", np.nan)),
    }
    for col in x_cols:
        row[f"coef_{col}"] = float(fit.params.get(col, np.nan))
        row[f"t_{col}"] = float(fit.tvalues.get(col, np.nan))
        row[f"p_{col}"] = float(fit.pvalues.get(col, np.nan))
    return row


def fit_stock_panel_model_grid(df: pd.DataFrame, universe_key: str) -> pd.DataFrame:
    ff3 = ["mkt_rf", "smb", "hml"]
    ff5 = [*ff3, "rmw", "cma"]
    models = [
        ("ff3_baseline", ff3),
        ("ff5_baseline", ff5),
        ("ff3_gmb_u", [*ff3, "gmb_u"]),
        ("ff5_gmb_u", [*ff5, "gmb_u"]),
        ("ff3_gmb_s", [*ff3, "gmb_s"]),
        ("ff5_gmb_s", [*ff5, "gmb_s"]),
        ("ff3_tls", [*ff3, "tls_factor"]),
        ("ff5_tls", [*ff5, "tls_factor"]),
        ("ff3_sentiment", [*ff3, "sentiment_factor"]),
        ("ff5_sentiment", [*ff5, "sentiment_factor"]),
        ("ff3_all4", [*ff3, *CUSTOM_FACTOR_ORDER]),
        ("ff5_all4", [*ff5, *CUSTOM_FACTOR_ORDER]),
    ]
    rows: List[Dict[str, object]] = []
    for model_name, x_cols in models:
        sector_options = [True]
        if model_name in {"ff3_all4", "ff5_all4"}:
            sector_options = [True, False]
        for sector_fe in sector_options:
            row = fit_stock_panel_model(df=df, universe_key=universe_key, model_name=model_name, x_cols=x_cols, sector_fe=sector_fe)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def build_sector_fe_robustness(stock_results: pd.DataFrame) -> pd.DataFrame:
    if stock_results.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for universe in stock_results["universe"].dropna().unique():
        for model_name in ["ff3_all4", "ff5_all4"]:
            with_fe = stock_results[
                (stock_results["universe"] == universe)
                & (stock_results["model"] == model_name)
                & (stock_results["sector_fe"] == True)
            ]
            without_fe = stock_results[
                (stock_results["universe"] == universe)
                & (stock_results["model"] == model_name)
                & (stock_results["sector_fe"] == False)
            ]
            if with_fe.empty or without_fe.empty:
                continue
            a = with_fe.iloc[0]
            b = without_fe.iloc[0]
            for term in [*BENCHMARK_ORDER, *CUSTOM_FACTOR_ORDER]:
                rows.append(
                    {
                        "universe": universe,
                        "model": model_name,
                        "term": term,
                        "coef_with_sector_fe": a.get(f"coef_{term}", np.nan),
                        "p_with_sector_fe": a.get(f"p_{term}", np.nan),
                        "coef_without_sector_fe": b.get(f"coef_{term}", np.nan),
                        "p_without_sector_fe": b.get(f"p_{term}", np.nan),
                        "delta_coef": a.get(f"coef_{term}", np.nan) - b.get(f"coef_{term}", np.nan),
                        "r2_with_sector_fe": a.get("r2", np.nan),
                        "r2_without_sector_fe": b.get("r2", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def _latex_table_header(caption: str, label: str, col_spec: str) -> List[str]:
    return [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]


def _latex_table_footer(note: str | None = None) -> List[str]:
    out = [r"\bottomrule", r"\end{tabular}"]
    if note:
        out.extend(
            [
                r"\vspace{0.4em}",
                r"\begin{minipage}{0.92\textwidth}",
                rf"\footnotesize {note}",
                r"\end{minipage}",
            ]
        )
    out.append(r"\end{table}")
    return out


def write_summary_stats_tex(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    for universe in ["r1000", "t100"]:
        sub = summary_df[(summary_df["universe"] == universe) & np.isclose(summary_df["cutoff"], 0.25)].copy()
        if sub.empty:
            continue
        caption = f"{'Russell 1000' if universe == 'r1000' else 'Transition-100'} factor summary statistics (25\\% tails)"
        label = f"tab:{universe}-factor-summary-stats-25"
        lines.extend(_latex_table_header(caption, label, "lrrrr"))
        lines.append(r"Factor & Mean & SD & Min & Max \\")
        lines.append(r"\midrule")
        for factor in [*BENCHMARK_ORDER, *CUSTOM_FACTOR_ORDER]:
            row = sub[sub["factor"] == factor]
            if row.empty:
                continue
            r0 = row.iloc[0]
            lines.append(
                f"{DISPLAY_NAMES[factor]} & {r0['mean_pct']:.2f} & {r0['sd_pct']:.2f} & {r0['min_pct']:.2f} & {r0['max_pct']:.2f} \\\\"
            )
        lines.extend(
            _latex_table_footer(
                "All entries are monthly returns in percent. Custom factors are annual-rebalanced equal-weight long-short portfolios built from 25\\% tails."
            )
        )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_correlation_tex(corr_matrices: Dict[str, pd.DataFrame], out_path: Path) -> None:
    lines: List[str] = []
    for universe in ["r1000", "t100"]:
        corr = corr_matrices.get(universe)
        if corr is None or corr.empty:
            continue
        caption = f"{'Russell 1000' if universe == 'r1000' else 'Transition-100'} factor correlation matrix (25\\% tails)"
        label = f"tab:{universe}-factor-corr-25"
        cols = [*BENCHMARK_ORDER, *CUSTOM_FACTOR_ORDER]
        lines.extend(_latex_table_header(caption, label, "l" + "r" * len(cols)))
        header = " & ".join(["", *[DISPLAY_NAMES[c] for c in cols]]) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        corr2 = corr.set_index("row_factor")
        for row_name in cols:
            values = [row_name]
            for col_name in cols:
                v = corr2.loc[row_name, col_name]
                values.append(f"{float(v):.2f}")
            lines.append(" & ".join([DISPLAY_NAMES.get(values[0], values[0]), *values[1:]]) + r" \\")
        lines.extend(_latex_table_footer("Correlations are computed on the overlapping months for which all Fama-French and custom-factor returns are observed."))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_factor_alpha_tex(alpha_df: pd.DataFrame, out_path: Path) -> None:
    lines = _latex_table_header(
        "Custom-factor alpha tests under CAPM, FF3, FF5, and FF5 plus the other custom factors",
        "tab:factor-alpha-models-all",
        "lllrccc",
    )
    lines.append(r"Universe & Cutoff & Factor & Model & Monthly $\alpha$ & $p$-value & $R^2$ \\")
    lines.append(r"\midrule")
    if alpha_df.empty:
        lines.append(r"\multicolumn{7}{c}{No factor alpha results.} \\")
    else:
        for _, row in alpha_df.sort_values(["universe", "cutoff", "factor_name", "model"]).iterrows():
            alpha_txt = f"{100.0 * float(row['alpha_monthly']):.3f}{stars(row['alpha_p'])}"
            lines.append(
                f"{tex_escape('Russell 1000' if row['universe']=='r1000' else 'Transition-100')} & "
                f"{display_cutoff(float(row['cutoff']))} & "
                f"{DISPLAY_NAMES[row['factor_name']]} & "
                f"{tex_escape(str(row['model']))} & "
                f"{alpha_txt} & {format_p(row['alpha_p'])} & {float(row['r2']):.3f} \\\\"
            )
    lines.extend(_latex_table_footer("Monthly alpha is reported in percent. Stars denote *** $p<0.01$, ** $p<0.05$, and * $p<0.10$."))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _stock_table_block(results: pd.DataFrame, universe: str, family: str) -> List[str]:
    family_models = {
        "ff3": ["ff3_baseline", "ff3_gmb_u", "ff3_gmb_s", "ff3_tls", "ff3_sentiment", "ff3_all4"],
        "ff5": ["ff5_baseline", "ff5_gmb_u", "ff5_gmb_s", "ff5_tls", "ff5_sentiment", "ff5_all4"],
    }[family]
    sub = results[(results["universe"] == universe) & (results["model"].isin(family_models)) & (results["sector_fe"] == True)].copy()
    if sub.empty:
        return []
    sub = sub.set_index("model").loc[family_models].reset_index()
    title = f"{'Russell 1000' if universe == 'r1000' else 'Transition-100'} stock-month excess-return regressions: {family.upper()} family"
    label = f"tab:{universe}-stock-panel-{family}"
    lines = _latex_table_header(title, label, "l" + "c" * len(family_models))
    lines.append(" & ".join(["", *[f"({i})" for i in range(1, len(family_models) + 1)]]) + r" \\")
    lines.append(r"\midrule")
    coef_terms = {
        "gmb_u": r"GMB$_U$",
        "gmb_s": r"GMB$_S$",
        "tls_factor": "TLS",
        "sentiment_factor": "Sentiment",
        "mkt_rf": "Mkt-RF",
        "smb": "SMB",
        "hml": "HML",
        "rmw": "RMW",
        "cma": "CMA",
        "const": "constant",
    }
    ordered_terms = ["gmb_u", "gmb_s", "tls_factor", "sentiment_factor", "mkt_rf", "smb", "hml"]
    if family == "ff5":
        ordered_terms.extend(["rmw", "cma"])
    ordered_terms.append("const")
    for term in ordered_terms:
        coef_key = f"coef_{term}"
        p_key = f"p_{term}"
        coef_cells = [coef_terms[term]]
        p_cells = [""]
        for _, row in sub.iterrows():
            coef = row.get(coef_key, np.nan)
            pval = row.get(p_key, np.nan)
            if pd.isna(coef):
                coef_cells.append("")
                p_cells.append("")
            else:
                coef_cells.append(f"{float(coef):.4f}{stars(pval)}")
                p_cells.append(f"({format_p(pval)})")
        lines.append(" & ".join(coef_cells) + r" \\")
        lines.append(" & ".join(p_cells) + r" \\")
    lines.append(r"\midrule")
    meta_rows = [
        ("Sector FE", ["Yes"] * len(sub)),
        ("Observations", [str(int(x)) for x in sub["N"]]),
        ("Months", [str(int(x)) for x in sub["n_months"]]),
        (r"$R^2$", [f"{float(x):.3f}" for x in sub["r2"]]),
        ("Adj. $R^2$", [f"{float(x):.3f}" for x in sub["adj_r2"]]),
    ]
    for label_txt, vals in meta_rows:
        lines.append(" & ".join([label_txt, *vals]) + r" \\")
    lines.extend(_latex_table_footer("Dependent variable is monthly stock excess return. Standard errors are clustered by month."))
    lines.append("")
    return lines


def write_stock_panel_tex(results: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    for universe in ["r1000", "t100"]:
        lines.extend(_stock_table_block(results, universe, "ff3"))
        lines.extend(_stock_table_block(results, universe, "ff5"))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_sector_fe_robustness_tex(robustness_df: pd.DataFrame, out_path: Path) -> None:
    lines = _latex_table_header(
        "Sector fixed-effect robustness for the headline all-custom-factor stock-panel models",
        "tab:sector-fe-robustness-custom-factors",
        "lllrrrr",
    )
    lines.append(r"Universe & Model & Term & With FE & Without FE & $\Delta$ coef. & $\Delta R^2$ \\")
    lines.append(r"\midrule")
    if robustness_df.empty:
        lines.append(r"\multicolumn{7}{c}{No sector fixed-effect robustness results.} \\")
    else:
        for _, row in robustness_df.sort_values(["universe", "model", "term"]).iterrows():
            delta_r2 = float(row["r2_with_sector_fe"]) - float(row["r2_without_sector_fe"])
            lines.append(
                f"{tex_escape('Russell 1000' if row['universe']=='r1000' else 'Transition-100')} & "
                f"{tex_escape(str(row['model']))} & {DISPLAY_NAMES.get(row['term'], row['term'])} & "
                f"{float(row['coef_with_sector_fe']):.4f} & {float(row['coef_without_sector_fe']):.4f} & "
                f"{float(row['delta_coef']):.4f} & {delta_r2:.3f} \\\\"
            )
    lines.extend(_latex_table_footer("Coefficients are from the headline FF3+all four custom factors and FF5+all four custom factors stock-panel regressions, estimated once with sector fixed effects and once without."))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def summarize_correlations(corr_df: pd.DataFrame) -> str:
    if corr_df.empty:
        return "No correlation results."
    corr = corr_df.set_index("row_factor")
    pairs = []
    for factor in CUSTOM_FACTOR_ORDER:
        for bench in BENCHMARK_ORDER:
            val = float(corr.loc[factor, bench])
            pairs.append((factor, bench, val))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top = pairs[:4]
    parts = []
    for factor, bench, val in top:
        parts.append(f"{DISPLAY_NAMES[factor]} vs {DISPLAY_NAMES[bench]}: {val:.2f}")
    return "; ".join(parts)


def read_characteristic_context() -> str:
    if not LEGACY_RESULT_SUMMARY.exists():
        return "Existing characteristic-summary file not found."
    text = LEGACY_RESULT_SUMMARY.read_text(encoding="utf-8")
    return text

