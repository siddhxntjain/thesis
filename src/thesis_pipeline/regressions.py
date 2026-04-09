from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from thesis_pipeline.canonical import _load_returns_monthly

KEY_SECTORS = ["Energy", "Industrials", "Utilities & Telecommunications"]
CONTROL_COLS = ["beta_mkt", "co2_assets"]
HORIZON_META = {
    "3m": {"ret_col": "ret_3m", "complete_col": "has_complete_3m"},
    "6m": {"ret_col": "ret_6m", "complete_col": "has_complete_6m"},
    "1y": {"ret_col": "ret_1y", "complete_col": "has_complete_1y"},
    "2y": {"ret_col": "ret_2y", "complete_col": "has_complete_2y"},
}


def canonicalize_sector_name(sector: str | float | None) -> str | float | None:
    if not isinstance(sector, str):
        return sector
    x = sector.strip()
    xl = x.lower()
    if xl in {
        "utilities & telecommunications",
        "utilities and telecommunications",
        "utilities/telecommunications",
        "utilities & telecom",
        "utilities and telecom",
        "telecommunications & utilities",
    }:
        return "Utilities & Telecommunications"
    return x


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


def load_panel(curated_root: Path, tickers: Sequence[str], years: Sequence[int]) -> pd.DataFrame:
    panel = pd.read_csv(curated_root / "ticker_year_panel.csv")
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
    panel = panel[panel["ticker"].isin(set(tickers)) & panel["signal_year"].isin(set(int(y) for y in years))].copy()
    panel["sector"] = panel["sector"].astype(str).map(canonicalize_sector_name)
    co2 = pd.to_numeric(panel.get("co2_unscaled"), errors="coerce")
    assets = pd.to_numeric(panel.get("year_assets"), errors="coerce")
    panel["co2_assets"] = np.where((co2 >= 0) & (assets > 0), co2 / assets, np.nan)
    return panel


def load_factors_monthly(path: Path) -> pd.DataFrame:
    fac = pd.read_csv(path)
    fac["month_end"] = pd.to_datetime(fac["month_end"], errors="coerce")
    for col in ["rf", "mkt_rf", "rm"]:
        if col in fac.columns:
            fac[col] = pd.to_numeric(fac[col], errors="coerce")
    keep = [c for c in ["month_end", "rf", "mkt_rf", "rm"] if c in fac.columns]
    return fac[keep].dropna(subset=["month_end"]).copy()


def build_market_beta_map(panel: pd.DataFrame, returns_file: Path, factors_file: Path, min_months: int = 9) -> pd.DataFrame:
    tickers = sorted(panel["ticker"].astype(str).str.upper().unique())
    returns_m = _load_returns_monthly(returns_file, set(tickers))
    fac = load_factors_monthly(factors_file)
    d = returns_m.merge(fac, on="month_end", how="left")
    d["ret_excess"] = d["ret"] - d["rf"] if "rf" in d.columns else d["ret"]

    rows = []
    for ticker, year in panel[["ticker", "signal_year"]].drop_duplicates().itertuples(index=False):
        start = pd.Timestamp(f"{int(year)}-12-31")
        pre_start = start - pd.offsets.MonthEnd(12)
        g = d[(d["ticker"] == ticker) & (d["month_end"] > pre_start) & (d["month_end"] <= start)].copy()
        g = g.dropna(subset=["ret_excess", "mkt_rf"])
        beta = np.nan
        if len(g) >= min_months:
            x = g["mkt_rf"].to_numpy(dtype=float)
            y = g["ret_excess"].to_numpy(dtype=float)
            vx = float(np.var(x))
            if vx > 1e-12:
                beta = float(np.cov(y, x, ddof=0)[0, 1] / vx)
        rows.append({"ticker": ticker, "signal_year": int(year), "beta_mkt": beta, "beta_months": int(len(g))})
    return pd.DataFrame(rows)


def control_specs() -> List[Tuple[str, List[str]]]:
    out = [("none", [])]
    for r in range(1, len(CONTROL_COLS) + 1):
        for cols in combinations(CONTROL_COLS, r):
            out.append(("+".join(cols), list(cols)))
    return out


def standardize(df: pd.DataFrame, cols: Iterable[str], group_col: str = "signal_year") -> pd.DataFrame:
    out = df.copy()

    def _zscore(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        sd = float(x.std(ddof=0))
        if math.isclose(sd, 0.0) or np.isnan(sd):
            return pd.Series(np.nan, index=s.index, dtype=float)
        return (x - float(x.mean())) / sd

    if group_col not in out.columns:
        raise KeyError(f"Missing grouping column for standardization: {group_col}")
    for col in cols:
        out[f"z_{col}"] = out.groupby(group_col, group_keys=False)[col].transform(_zscore)
    return out


def fit_formula(df: pd.DataFrame, formula: str, min_n: int, focus_terms: Sequence[str]) -> Tuple[pd.DataFrame, object] | Tuple[None, None]:
    try:
        model = smf.ols(formula, data=df, missing="drop").fit(cov_type="HC1")
    except Exception:
        return None, None
    if int(model.nobs) < min_n:
        return None, None
    row: Dict[str, object] = {
        "N": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "formula": formula,
    }
    for term in focus_terms:
        row[f"beta_{term}"] = float(model.params.get(term, np.nan))
        row[f"t_{term}"] = float(model.tvalues.get(term, np.nan))
        row[f"p_{term}"] = float(model.pvalues.get(term, np.nan))
        if term in model.params.index:
            ci = model.conf_int().loc[term]
            row[f"ci_low_{term}"] = float(ci.iloc[0])
            row[f"ci_high_{term}"] = float(ci.iloc[1])
        else:
            row[f"ci_low_{term}"] = np.nan
            row[f"ci_high_{term}"] = np.nan
    return pd.DataFrame([row]), model


def make_sector_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sector_energy"] = (out["sector"] == "Energy").astype(int)
    out["sector_industrials"] = (out["sector"] == "Industrials").astype(int)
    out["sector_utilities_telco"] = (out["sector"] == "Utilities & Telecommunications").astype(int)
    return out


def regular_formula(outcome: str, family: str, controls: Sequence[str], pooled: bool) -> Tuple[str, List[str]]:
    terms: List[str] = []
    focus: List[str] = []
    if family == "tls_level":
        terms.append("z_tls_score")
        focus.append("z_tls_score")
    elif family == "delta_tls":
        terms.append("z_delta_tls")
        focus.append("z_delta_tls")
    elif family == "joint_tls_delta":
        terms.extend(["z_tls_score", "z_delta_tls"])
        focus.extend(["z_tls_score", "z_delta_tls"])
    else:
        raise ValueError(f"Unknown family: {family}")
    terms.extend(f"z_{c}" for c in controls)
    terms.append("C(sector)")
    if pooled:
        terms.append("C(signal_year)")
    return f"{outcome} ~ " + " + ".join(terms), focus


def interaction_formula(outcome: str, controls: Sequence[str], pooled: bool) -> Tuple[str, List[str]]:
    terms = [
        "z_tls_score",
        "z_tls_score:sector_energy",
        "z_tls_score:sector_industrials",
        "z_tls_score:sector_utilities_telco",
    ]
    focus = list(terms)
    terms.extend(f"z_{c}" for c in controls)
    terms.append("C(sector)")
    if pooled:
        terms.append("C(signal_year)")
    return f"{outcome} ~ " + " + ".join(terms), focus


def build_regression_frame(panel: pd.DataFrame) -> pd.DataFrame:
    df = make_sector_flags(panel.copy())
    for col in ["tls_score", "delta_tls", *CONTROL_COLS]:
        if col not in df.columns:
            df[col] = np.nan
    return standardize(df, ["tls_score", "delta_tls", *CONTROL_COLS])


def run_regular_models(panel: pd.DataFrame, min_n_yearly: int, min_n_pooled: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    yearly_rows: List[pd.DataFrame] = []
    pooled_rows: List[pd.DataFrame] = []
    reg_df = build_regression_frame(panel)
    for horizon, meta in HORIZON_META.items():
        outcome = meta["ret_col"]
        complete = meta["complete_col"]
        base = reg_df[reg_df[complete].fillna(False)].copy()
        if base.empty:
            continue
        for family in ["tls_level", "delta_tls", "joint_tls_delta"]:
            for control_name, controls in control_specs():
                formula_pooled, focus = regular_formula(outcome, family, controls, pooled=True)
                fit_row, _ = fit_formula(base, formula_pooled, min_n_pooled, focus)
                if fit_row is not None:
                    fit_row["scope"] = "pooled"
                    fit_row["horizon"] = horizon
                    fit_row["family"] = family
                    fit_row["control_spec"] = control_name
                    pooled_rows.append(fit_row)
                for year, g in base.groupby("signal_year"):
                    formula_yearly, focus = regular_formula(outcome, family, controls, pooled=False)
                    fit_row, _ = fit_formula(g, formula_yearly, min_n_yearly, focus)
                    if fit_row is not None:
                        fit_row["scope"] = "yearly"
                        fit_row["signal_year"] = int(year)
                        fit_row["horizon"] = horizon
                        fit_row["family"] = family
                        fit_row["control_spec"] = control_name
                        yearly_rows.append(fit_row)
    yearly = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    pooled = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
    return yearly, pooled


def run_interaction_models(panel: pd.DataFrame, min_n_yearly: int, min_n_pooled: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    yearly_rows: List[pd.DataFrame] = []
    pooled_rows: List[pd.DataFrame] = []
    reg_df = build_regression_frame(panel)
    for horizon, meta in HORIZON_META.items():
        outcome = meta["ret_col"]
        complete = meta["complete_col"]
        base = reg_df[reg_df[complete].fillna(False)].copy()
        if base.empty:
            continue
        for control_name, controls in control_specs():
            formula_pooled, focus = interaction_formula(outcome, controls, pooled=True)
            fit_row, _ = fit_formula(base, formula_pooled, min_n_pooled, focus)
            if fit_row is not None:
                fit_row["scope"] = "pooled"
                fit_row["horizon"] = horizon
                fit_row["control_spec"] = control_name
                pooled_rows.append(fit_row)
            for year, g in base.groupby("signal_year"):
                formula_yearly, focus = interaction_formula(outcome, controls, pooled=False)
                fit_row, _ = fit_formula(g, formula_yearly, min_n_yearly, focus)
                if fit_row is not None:
                    fit_row["scope"] = "yearly"
                    fit_row["signal_year"] = int(year)
                    fit_row["horizon"] = horizon
                    fit_row["control_spec"] = control_name
                    yearly_rows.append(fit_row)
    yearly = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    pooled = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
    return yearly, pooled


def build_control_grid_summary(horizon_pooled: pd.DataFrame, interaction_pooled: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not horizon_pooled.empty:
        for _, row in horizon_pooled.iterrows():
            focal = "z_delta_tls" if row["family"] == "delta_tls" else "z_tls_score"
            rows.append(
                {
                    "model_group": row["family"],
                    "horizon": row["horizon"],
                    "control_spec": row["control_spec"],
                    "focus_term": focal,
                    "beta": row.get(f"beta_{focal}", np.nan),
                    "t_stat": row.get(f"t_{focal}", np.nan),
                    "p_value": row.get(f"p_{focal}", np.nan),
                    "adj_r2": row.get("adj_r2", np.nan),
                    "N": row.get("N", np.nan),
                }
            )
    if not interaction_pooled.empty:
        for _, row in interaction_pooled.iterrows():
            rows.append(
                {
                    "model_group": "sector_interaction",
                    "horizon": row["horizon"],
                    "control_spec": row["control_spec"],
                    "focus_term": "z_tls_score",
                    "beta": row.get("beta_z_tls_score", np.nan),
                    "t_stat": row.get("t_z_tls_score", np.nan),
                    "p_value": row.get("p_z_tls_score", np.nan),
                    "adj_r2": row.get("adj_r2", np.nan),
                    "N": row.get("N", np.nan),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["p_value", "model_group", "horizon", "control_spec"], na_position="last").reset_index(drop=True)


def best_specs_table(horizon_pooled: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if horizon_pooled.empty:
        return pd.DataFrame()
    for (family, horizon), g in horizon_pooled.groupby(["family", "horizon"]):
        focal = "z_delta_tls" if family == "delta_tls" else "z_tls_score"
        g = g.copy()
        g["focus_p"] = pd.to_numeric(g[f"p_{focal}"], errors="coerce")
        g = g.sort_values(["focus_p", "adj_r2"], ascending=[True, False], na_position="last")
        top = g.iloc[0]
        rows.append(
            {
                "family": family,
                "horizon": horizon,
                "control_spec": top["control_spec"],
                "beta": top.get(f"beta_{focal}", np.nan),
                "t_stat": top.get(f"t_{focal}", np.nan),
                "p_value": top.get(f"p_{focal}", np.nan),
                "adj_r2": top.get("adj_r2", np.nan),
                "N": top.get("N", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "horizon"]).reset_index(drop=True)
