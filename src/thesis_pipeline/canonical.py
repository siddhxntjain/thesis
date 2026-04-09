#!/usr/bin/env python3
"""
Canonical data builders for thesis factor research.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


HORIZON_SPECS: List[Tuple[str, int]] = [
    ("3m", 3),
    ("6m", 6),
    ("1y", 12),
    ("2y", 24),
    ("5y", 60),
]


@dataclass(frozen=True)
class BuildConfig:
    universe_file: Path
    years: Sequence[int]
    feature_cache_dir: Path
    sentiment_file: Path
    sentiment_primary_col: str
    returns_file: Path
    factors_ff3_file: Path
    factors_ff5_file: Path
    metadata_file: Path
    assets_file: Path
    co2_file: Path
    curated_root: Path
    tls_sw: float = 1.0
    tls_pw: float = 10.0
    tls_cw: int = 255
    factor_quantile: float = 0.25


def parse_tickers(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    out = sorted({x.strip().upper() for x in raw.replace(",", " ").split() if x.strip()})
    return out


def parse_years(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    x = str(v).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean from: {v}")


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _df_hash(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return _sha256_bytes(payload)


def _sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["ticker", "signal_year"]).reset_index(drop=True)


def _sort_factors(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["month_end"]).reset_index(drop=True)


def _sort_custom(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["factor_id", "month_end"]).reset_index(drop=True)


def _coalesce_mode(s: pd.Series) -> float | str | None:
    t = s.dropna()
    if t.empty:
        return None
    m = t.mode()
    if m.empty:
        return None
    out = sorted(m.astype(str))[0]
    # Attempt numeric cast when applicable.
    try:
        if "." in out:
            return float(out)
        return int(out)
    except Exception:
        return out


def _load_returns_monthly(path: Path, tickers: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Ticker", "PERMNO", "HdrCUSIP", "DlyCalDt", "DlyRet"])
    df["ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)].copy()
    df["permno"] = pd.to_numeric(df["PERMNO"], errors="coerce")
    df["cusip_ret"] = df["HdrCUSIP"].astype(str).str.strip()
    df["cusip_ret"] = df["cusip_ret"].where(~df["cusip_ret"].isin(["", "nan", "NaN"]), np.nan)
    df["date"] = pd.to_datetime(df["DlyCalDt"], errors="coerce")
    df["ret"] = pd.to_numeric(df["DlyRet"], errors="coerce")
    df = df.dropna(subset=["ticker", "date", "ret"]).copy()
    # Input file is monthly despite "daily" name. Keep month-end normalized key.
    df["month_end"] = df["date"] + pd.offsets.MonthEnd(0)
    df["year"] = df["month_end"].dt.year
    return df[["ticker", "permno", "cusip_ret", "date", "month_end", "year", "ret"]].sort_values(
        ["ticker", "month_end"]
    )


def _ticker_id_maps(returns_m: pd.DataFrame, assets: pd.DataFrame, feature_cik: pd.DataFrame) -> pd.DataFrame:
    ret_map = (
        returns_m.groupby("ticker", as_index=False)
        .agg(
            permno=("permno", _coalesce_mode),
            cusip_ret=("cusip_ret", _coalesce_mode),
        )
        .rename(columns={"cusip_ret": "cusip_ret_mode"})
    )
    asset_map = (
        assets.groupby("ticker", as_index=False)
        .agg(cusip_asset_mode=("cusip", _coalesce_mode))
    )
    cik_map = (
        feature_cik.groupby("ticker", as_index=False)
        .agg(cik=("cik", _coalesce_mode))
    )
    out = ret_map.merge(asset_map, on="ticker", how="outer").merge(cik_map, on="ticker", how="outer")
    return out


def _compute_tls_for_year(df: pd.DataFrame, sw: float, pw: float, cw: int) -> pd.DataFrame:
    near_col = f"sub_near_{cw}"
    both_col = f"sub_both_{cw}"
    need = [
        "ticker",
        "tokens",
        "sub_total",
        "sub_in_section",
        "bp_total",
        "bp_in_section",
        near_col,
        both_col,
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing feature columns for TLS score: {miss}")

    x = df.copy()
    x["ticker"] = x["ticker"].astype(str).str.upper().str.strip()
    tok = pd.to_numeric(x["tokens"], errors="coerce").clip(lower=1)
    sub_raw = (
        pd.to_numeric(x["sub_total"], errors="coerce")
        + sw * pd.to_numeric(x["sub_in_section"], errors="coerce")
        + pw * pd.to_numeric(x[near_col], errors="coerce")
        + (sw * pw) * pd.to_numeric(x[both_col], errors="coerce")
    )
    bp_raw = pd.to_numeric(x["bp_total"], errors="coerce") + sw * pd.to_numeric(x["bp_in_section"], errors="coerce")
    tls = (sub_raw - bp_raw) * (10000.0 / tok)
    out = pd.DataFrame({"ticker": x["ticker"], "tls_score": tls, "cik": x.get("cik", np.nan)})
    out = out.dropna(subset=["ticker", "tls_score"]).drop_duplicates("ticker", keep="first")
    return out


def _build_tls_year_table(
    tickers: Sequence[str],
    years: Sequence[int],
    feature_cache_dir: Path,
    sw: float,
    pw: float,
    cw: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    cik_rows: List[pd.DataFrame] = []
    ticker_set = set(tickers)
    for y in years:
        p = feature_cache_dir / f"feature_cache_{y}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["ticker"] = d["ticker"].astype(str).str.upper().str.strip()
        d = d[d["ticker"].isin(ticker_set)].copy()
        if d.empty:
            continue
        t = _compute_tls_for_year(d, sw=sw, pw=pw, cw=cw)
        t["signal_year"] = int(y)
        rows.append(t[["ticker", "signal_year", "tls_score", "cik"]].copy())
        cik_rows.append(t[["ticker", "cik"]].copy())
    tls = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ticker", "signal_year", "tls_score", "cik"])
    cik_map = pd.concat(cik_rows, ignore_index=True) if cik_rows else pd.DataFrame(columns=["ticker", "cik"])
    return tls, cik_map


def _load_sentiment(
    sentiment_file: Path,
    tickers: set[str],
    years: set[int],
    primary_col: str,
) -> pd.DataFrame:
    s = pd.read_csv(sentiment_file)
    needed = [
        "ticker",
        "year",
        "transition_stance_index",
        "transition_sentiment_mean",
        "transition_sentiment_median",
        "transition_pos_share",
        "transition_neg_share",
    ]
    miss = [c for c in needed if c not in s.columns]
    if miss:
        raise KeyError(f"Sentiment file missing columns: {miss}")
    if primary_col not in s.columns:
        raise KeyError(f"Sentiment primary column not found in sentiment file: {primary_col}")
    s["ticker"] = s["ticker"].astype(str).str.upper().str.strip()
    s["signal_year"] = pd.to_numeric(s["year"], errors="coerce").astype("Int64")
    s = s[s["ticker"].isin(tickers) & s["signal_year"].isin(years)].copy()
    s = s.sort_values(["ticker", "signal_year"]).drop_duplicates(["ticker", "signal_year"], keep="last")
    s["year_sentiment"] = pd.to_numeric(s[primary_col], errors="coerce")
    out = s.rename(
        columns={
            "transition_sentiment_mean": "sentiment_mean",
            "transition_sentiment_median": "sentiment_median",
            "transition_pos_share": "sentiment_pos_share",
            "transition_neg_share": "sentiment_neg_share",
        }
    )
    return out[
        [
            "ticker",
            "signal_year",
            "year_sentiment",
            "sentiment_mean",
            "sentiment_median",
            "sentiment_pos_share",
            "sentiment_neg_share",
        ]
    ].copy()


def _load_assets(assets_file: Path, tickers: set[str], years: set[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(assets_file, usecols=["tic", "datadate", "at", "cusip", "indfmt"])
    a["ticker"] = a["tic"].astype(str).str.upper().str.strip()
    a["signal_year"] = pd.to_datetime(a["datadate"], errors="coerce").dt.year
    a["year_assets"] = pd.to_numeric(a["at"], errors="coerce")
    a["cusip"] = a["cusip"].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    a = a[a["ticker"].isin(tickers) & a["signal_year"].isin(years)].copy()
    if a.empty:
        return (
            pd.DataFrame(columns=["ticker", "signal_year", "year_assets", "cusip"]),
            pd.DataFrame(columns=["ticker", "cusip"]),
        )
    a["is_indl"] = (a["indfmt"] == "INDL").astype(int)
    # latest datadate, then INDL, then largest assets
    a = a.sort_values(
        ["ticker", "signal_year", "datadate", "is_indl", "year_assets"],
        ascending=[True, True, False, False, False],
    )
    by_year = a.drop_duplicates(["ticker", "signal_year"], keep="first")
    cusip_mode = a.groupby("ticker", as_index=False).agg(cusip=("cusip", _coalesce_mode))
    return by_year[["ticker", "signal_year", "year_assets", "cusip"]], cusip_mode


def _load_co2(
    co2_file: Path,
    tickers: set[str],
    years: set[int],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    c = pd.read_csv(co2_file, usecols=["year", "isin", "ticker", "fieldid", "fieldname", "value"])
    c["ticker"] = c["ticker"].astype(str).str.upper().str.strip()
    c["signal_year"] = pd.to_numeric(c["year"], errors="coerce").astype("Int64")
    c = c[c["ticker"].isin(tickers) & c["signal_year"].isin(years)].copy()
    c = c[
        (pd.to_numeric(c["fieldid"], errors="coerce") == 90)
        | (c["fieldname"].astype(str) == "AnalyticEstimatesCO2EquivalentsEmissionTotal")
    ].copy()
    c["co2_unscaled"] = pd.to_numeric(c["value"], errors="coerce")
    c = c.dropna(subset=["co2_unscaled"]).copy()
    c["is_us_isin"] = c["isin"].astype(str).str.startswith("US").astype(int)
    raw_rows = int(len(c))
    raw_groups = int(c.groupby(["ticker", "signal_year"]).ngroups)
    dup_groups = int((c.groupby(["ticker", "signal_year"]).size() > 1).sum())
    c = c.sort_values(
        ["ticker", "signal_year", "is_us_isin", "co2_unscaled"],
        ascending=[True, True, False, False],
    )
    c1 = c.drop_duplicates(["ticker", "signal_year"], keep="first").copy()
    c1["co2_log"] = np.log1p(c1["co2_unscaled"])
    c1["co2_scaled"] = np.nan
    for y, g in c1.groupby("signal_year"):
        v = g["co2_unscaled"].astype(float)
        if v.notna().sum() == 0:
            continue
        p1 = float(v.quantile(0.01))
        p99 = float(v.quantile(0.99))
        clipped = v.clip(lower=p1, upper=p99)
        sd = float(clipped.std(ddof=0))
        if sd == 0 or np.isnan(sd):
            z = pd.Series(np.zeros(len(clipped)), index=clipped.index)
        else:
            z = (clipped - float(clipped.mean())) / sd
        c1.loc[g.index, "co2_scaled"] = z.values
    stats = {
        "co2_raw_rows": raw_rows,
        "co2_raw_groups": raw_groups,
        "co2_duplicate_groups": dup_groups,
        "co2_dedup_rows": int(len(c1)),
        "co2_us_isin_selected_rows": int(c1["is_us_isin"].sum()),
    }
    return c1[["ticker", "signal_year", "co2_unscaled", "co2_log", "co2_scaled"]].copy(), stats


def _build_ret_windows(
    returns_m: pd.DataFrame,
    tickers: Sequence[str],
    years: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ticker_set = set(tickers)
    r = returns_m[returns_m["ticker"].isin(ticker_set)].copy()
    for y in years:
        start = pd.Timestamp(f"{y}-12-31")
        m = pd.DataFrame({"ticker": sorted(ticker_set)})
        m["signal_year"] = int(y)
        for label, n_months in HORIZON_SPECS:
            end = start + pd.offsets.MonthEnd(n_months)
            sub = r[(r["date"] > start) & (r["date"] <= end)].copy()
            agg = (
                sub.groupby("ticker", as_index=False)
                .agg(
                    **{
                        f"ret_{label}": ("ret", lambda x: float(np.prod(1.0 + x.values) - 1.0)),
                        f"n_months_{label}": ("month_end", "nunique"),
                    }
                )
                if not sub.empty
                else pd.DataFrame(columns=["ticker", f"ret_{label}", f"n_months_{label}"])
            )
            m = m.merge(agg, on="ticker", how="left")
            m[f"ret_{label}_start"] = start
            m[f"ret_{label}_end"] = end
            m[f"n_months_{label}"] = pd.to_numeric(m[f"n_months_{label}"], errors="coerce")
            m[f"has_complete_{label}"] = m[f"n_months_{label}"].fillna(0).astype(int) >= n_months
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_metadata(path: Path, tickers: set[str]) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=["filename", "sector", "industry"])
    d["ticker"] = d["filename"].astype(str).str.upper().str.strip()
    d = d[d["ticker"].isin(tickers)].copy()
    d = d.sort_values("ticker").drop_duplicates("ticker", keep="first")
    return d[["ticker", "sector", "industry"]]


def build_ticker_year_panel(cfg: BuildConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    tickers = parse_tickers(cfg.universe_file)
    years = list(cfg.years)
    ticker_set = set(tickers)
    year_set = set(years)

    tls, cik_rows = _build_tls_year_table(
        tickers=tickers,
        years=years,
        feature_cache_dir=cfg.feature_cache_dir,
        sw=cfg.tls_sw,
        pw=cfg.tls_pw,
        cw=cfg.tls_cw,
    )
    sentiment = _load_sentiment(
        cfg.sentiment_file,
        tickers=ticker_set,
        years=year_set,
        primary_col=cfg.sentiment_primary_col,
    )
    assets_year, assets_cusip_mode = _load_assets(cfg.assets_file, tickers=ticker_set, years=year_set)
    co2, co2_stats = _load_co2(cfg.co2_file, tickers=ticker_set, years=year_set)
    meta = _load_metadata(cfg.metadata_file, tickers=ticker_set)
    returns_m = _load_returns_monthly(cfg.returns_file, tickers=ticker_set)
    retw = _build_ret_windows(returns_m, tickers=tickers, years=years)

    ids = _ticker_id_maps(returns_m=returns_m, assets=assets_cusip_mode, feature_cik=cik_rows)

    base = pd.MultiIndex.from_product([tickers, years], names=["ticker", "signal_year"]).to_frame(index=False)
    panel = (
        base.merge(tls[["ticker", "signal_year", "tls_score"]], on=["ticker", "signal_year"], how="left")
        .merge(sentiment, on=["ticker", "signal_year"], how="left")
        .merge(assets_year[["ticker", "signal_year", "year_assets", "cusip"]], on=["ticker", "signal_year"], how="left")
        .merge(co2, on=["ticker", "signal_year"], how="left")
        .merge(retw, on=["ticker", "signal_year"], how="left")
        .merge(ids, on="ticker", how="left")
        .merge(meta, on="ticker", how="left")
    )

    panel["signal_anchor_date"] = pd.to_datetime(panel["signal_year"].astype(int).astype(str) + "-12-31")
    panel["permno"] = pd.to_numeric(panel["permno"], errors="coerce").astype("Int64")
    panel["cik"] = pd.to_numeric(panel["cik"], errors="coerce").astype("Int64")
    panel["cusip"] = panel["cusip"].combine_first(panel["cusip_asset_mode"]).combine_first(panel["cusip_ret_mode"])
    panel = panel.drop(columns=[c for c in ["cusip_asset_mode", "cusip_ret_mode"] if c in panel.columns])
    panel = panel.sort_values(["ticker", "signal_year"]).reset_index(drop=True)
    panel["delta_tls"] = panel.groupby("ticker")["tls_score"].diff()

    panel = panel[
        [
            "ticker",
            "permno",
            "cusip",
            "cik",
            "sector",
            "industry",
            "signal_year",
            "signal_anchor_date",
            "tls_score",
            "delta_tls",
            "year_sentiment",
            "sentiment_mean",
            "sentiment_median",
            "sentiment_pos_share",
            "sentiment_neg_share",
            "co2_unscaled",
            "co2_log",
            "co2_scaled",
            "year_assets",
            "ret_3m",
            "ret_3m_start",
            "ret_3m_end",
            "ret_6m",
            "ret_6m_start",
            "ret_6m_end",
            "ret_1y",
            "ret_1y_start",
            "ret_1y_end",
            "ret_2y",
            "ret_2y_start",
            "ret_2y_end",
            "ret_5y",
            "ret_5y_start",
            "ret_5y_end",
            "n_months_3m",
            "n_months_6m",
            "n_months_1y",
            "n_months_2y",
            "n_months_5y",
            "has_complete_3m",
            "has_complete_6m",
            "has_complete_1y",
            "has_complete_2y",
            "has_complete_5y",
        ]
    ].copy()
    panel["n_months_3m"] = pd.to_numeric(panel["n_months_3m"], errors="coerce").fillna(0).astype(int)
    panel["n_months_6m"] = pd.to_numeric(panel["n_months_6m"], errors="coerce").fillna(0).astype(int)
    panel["n_months_1y"] = pd.to_numeric(panel["n_months_1y"], errors="coerce").fillna(0).astype(int)
    panel["n_months_2y"] = pd.to_numeric(panel["n_months_2y"], errors="coerce").fillna(0).astype(int)
    panel["n_months_5y"] = pd.to_numeric(panel["n_months_5y"], errors="coerce").fillna(0).astype(int)
    panel["has_complete_3m"] = panel["has_complete_3m"].fillna(False).astype(bool)
    panel["has_complete_6m"] = panel["has_complete_6m"].fillna(False).astype(bool)
    panel["has_complete_1y"] = panel["has_complete_1y"].fillna(False).astype(bool)
    panel["has_complete_2y"] = panel["has_complete_2y"].fillna(False).astype(bool)
    panel["has_complete_5y"] = panel["has_complete_5y"].fillna(False).astype(bool)
    panel = _sort_panel(panel)

    stats: Dict[str, object] = dict(co2_stats)
    stats["panel_rows"] = int(len(panel))
    stats["panel_unique_tickers"] = int(panel["ticker"].nunique())
    stats["panel_year_min"] = int(panel["signal_year"].min()) if not panel.empty else None
    stats["panel_year_max"] = int(panel["signal_year"].max()) if not panel.empty else None
    return panel, stats


def _parse_ff5_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["month_end", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) < 7:
                continue
            key = parts[0]
            if key.isdigit() and len(key) == 6:
                try:
                    vals = [float(parts[i]) for i in range(1, 7)]
                except Exception:
                    continue
                rows.append((key, *vals))
    if not rows:
        return pd.DataFrame(columns=["month_end", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    d = pd.DataFrame(rows, columns=["yyyymm", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    d["month_end"] = pd.to_datetime(d["yyyymm"] + "01", format="%Y%m%d", errors="coerce") + pd.offsets.MonthEnd(0)
    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        d[c] = d[c] / 100.0
    return d[["month_end", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]].dropna(subset=["month_end"])


def build_benchmark_factors_monthly(cfg: BuildConfig) -> pd.DataFrame:
    ff3 = pd.read_csv(cfg.factors_ff3_file)
    need = {"date", "MKT_RF", "SMB", "HML", "RF"}
    miss = need - set(ff3.columns)
    if miss:
        raise KeyError(f"FF3 clean factors missing columns: {sorted(miss)}")
    ff3 = ff3.copy()
    ff3["month_end"] = pd.to_datetime(ff3["date"], errors="coerce") + pd.offsets.MonthEnd(0)
    ff3 = ff3.dropna(subset=["month_end"]).copy()
    ff3 = ff3.rename(columns={"MKT_RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RF": "rf"})
    ff3 = ff3[["month_end", "rf", "mkt_rf", "smb", "hml"]].copy()
    for c in ["rf", "mkt_rf", "smb", "hml"]:
        ff3[c] = pd.to_numeric(ff3[c], errors="coerce")

    ff5 = _parse_ff5_raw(cfg.factors_ff5_file)
    out = ff3.merge(ff5[["month_end", "rmw", "cma"]], on="month_end", how="left")
    # Backfill from FF5 if FF3 values missing.
    if not ff5.empty:
        full = ff5.merge(out, on="month_end", how="right", suffixes=("_ff5", ""))
        for c in ["rf", "mkt_rf", "smb", "hml"]:
            ff5_col = f"{c}_ff5"
            if ff5_col in full.columns:
                full[c] = full[c].combine_first(full[ff5_col])
        out = full[["month_end", "rf", "mkt_rf", "smb", "hml", "rmw", "cma"]].copy()
    out["rm"] = out["rf"] + out["mkt_rf"]
    out = _sort_factors(out.drop_duplicates("month_end", keep="first"))
    return out


def build_custom_factor_returns_monthly(
    panel: pd.DataFrame,
    factors_monthly: pd.DataFrame,
    returns_file: Path,
    signals: Sequence[str],
    quantile: float,
) -> pd.DataFrame:
    r = _load_returns_monthly(returns_file, tickers=set(panel["ticker"]))
    rows: List[pd.DataFrame] = []
    for sig in signals:
        if sig not in panel.columns:
            continue
        factor_id = f"{sig}_q{int(quantile * 100):02d}_ew_annual_ls"
        for y, g in panel.groupby("signal_year"):
            d = g[["ticker", sig]].dropna().copy()
            if len(d) < 20:
                continue
            q_hi = float(d[sig].quantile(1.0 - quantile))
            q_lo = float(d[sig].quantile(quantile))
            long_t = set(d[d[sig] >= q_hi]["ticker"])
            short_t = set(d[d[sig] <= q_lo]["ticker"])
            if not long_t or not short_t:
                continue
            start = pd.Timestamp(f"{int(y)}-12-31")
            end = pd.Timestamp(f"{int(y)+1}-12-31")
            w = r[(r["date"] > start) & (r["date"] <= end)].copy()
            if w.empty:
                continue
            long_m = (
                w[w["ticker"].isin(long_t)].groupby("month_end", as_index=False)["ret"].mean().rename(columns={"ret": "long_ret"})
            )
            short_m = (
                w[w["ticker"].isin(short_t)].groupby("month_end", as_index=False)["ret"].mean().rename(columns={"ret": "short_ret"})
            )
            m = long_m.merge(short_m, on="month_end", how="inner")
            if m.empty:
                continue
            m["ls_ret"] = m["long_ret"] - m["short_ret"]
            m["factor_id"] = factor_id
            m["signal_name"] = sig
            m["construction_rule"] = "Q25_EW_Annual_LS"
            m["n_long"] = int(len(long_t))
            m["n_short"] = int(len(short_t))
            m["universe_n"] = int(len(d))
            m["signal_year"] = int(y)
            rows.append(m)
    out = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "factor_id",
                "signal_name",
                "construction_rule",
                "month_end",
                "long_ret",
                "short_ret",
                "ls_ret",
                "ls_excess",
                "n_long",
                "n_short",
                "universe_n",
                "signal_year",
            ]
        )
    )
    if out.empty:
        return out
    f = factors_monthly[["month_end", "rf"]].copy()
    out = out.merge(f, on="month_end", how="left")
    out["ls_excess"] = out["ls_ret"] - out["rf"]
    out = out.drop(columns=["rf"])
    out = out[
        [
            "factor_id",
            "signal_name",
            "construction_rule",
            "month_end",
            "long_ret",
            "short_ret",
            "ls_ret",
            "ls_excess",
            "n_long",
            "n_short",
            "universe_n",
            "signal_year",
        ]
    ].copy()
    return _sort_custom(out)


def validate_canonical_tables(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    custom: pd.DataFrame,
    sentiment_source: pd.DataFrame | None = None,
    tls_legacy: pd.DataFrame | None = None,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["panel_unique_key_ok"] = bool(
        panel[["ticker", "signal_year"]].drop_duplicates().shape[0] == panel.shape[0]
    )
    out["factors_unique_key_ok"] = bool(
        factors[["month_end"]].drop_duplicates().shape[0] == factors.shape[0]
    )
    out["custom_unique_key_ok"] = bool(
        custom[["factor_id", "month_end"]].drop_duplicates().shape[0] == custom.shape[0]
    )
    out["coverage_by_year"] = {
        col: panel.groupby("signal_year")[col].apply(lambda s: float(s.notna().mean())).to_dict()
        for col in [
            "tls_score",
            "delta_tls",
            "year_sentiment",
            "co2_unscaled",
            "co2_scaled",
            "year_assets",
            "ret_3m",
            "ret_6m",
            "ret_1y",
            "ret_2y",
            "ret_5y",
        ]
        if col in panel.columns
    }
    out["null_horizon_rows_3m"] = int(panel["ret_3m"].isna().sum()) if "ret_3m" in panel.columns else None
    out["null_horizon_rows_6m"] = int(panel["ret_6m"].isna().sum()) if "ret_6m" in panel.columns else None
    out["null_horizon_rows_1y"] = int(panel["ret_1y"].isna().sum()) if "ret_1y" in panel.columns else None
    out["null_horizon_rows_2y"] = int(panel["ret_2y"].isna().sum()) if "ret_2y" in panel.columns else None
    out["null_horizon_rows_5y"] = int(panel["ret_5y"].isna().sum()) if "ret_5y" in panel.columns else None

    if sentiment_source is not None and not sentiment_source.empty:
        chk = panel.merge(
            sentiment_source[["ticker", "signal_year", "year_sentiment"]],
            on=["ticker", "signal_year"],
            how="inner",
            suffixes=("", "_src"),
        )
        if not chk.empty:
            diff = (chk["year_sentiment"] - chk["year_sentiment_src"]).abs()
            out["sentiment_match_n"] = int(len(chk))
            out["sentiment_match_max_abs_diff"] = float(diff.max())
            out["sentiment_match_exact"] = bool((diff.fillna(0) < 1e-12).all())

    if tls_legacy is not None and not tls_legacy.empty:
        chk = panel.merge(
            tls_legacy[["ticker", "signal_year", "tls_score"]].rename(columns={"tls_score": "tls_score_legacy"}),
            on=["ticker", "signal_year"],
            how="inner",
        )
        if not chk.empty:
            diff = (chk["tls_score"] - chk["tls_score_legacy"]).abs()
            out["tls_match_n"] = int(len(chk))
            out["tls_match_max_abs_diff"] = float(diff.max())
            out["tls_match_exact"] = bool((diff.fillna(0) < 1e-10).all())
    return out


def write_schema_md(path: Path) -> None:
    txt = """# Canonical Schema

## `ticker_year_panel.csv`
One row per `ticker, signal_year`.

Columns:
- IDs: `ticker, permno, cusip, cik, sector, industry`
- Timing: `signal_year, signal_anchor_date`
- Signals: `tls_score, delta_tls, year_sentiment, sentiment_mean, sentiment_median, sentiment_pos_share, sentiment_neg_share, co2_unscaled, co2_log, co2_scaled, year_assets`
- Outcomes: `ret_3m, ret_3m_start, ret_3m_end, ret_6m, ret_6m_start, ret_6m_end, ret_1y, ret_1y_start, ret_1y_end, ret_2y, ret_2y_start, ret_2y_end, ret_5y, ret_5y_start, ret_5y_end`
- QC: `n_months_3m, n_months_6m, n_months_1y, n_months_2y, n_months_5y, has_complete_3m, has_complete_6m, has_complete_1y, has_complete_2y, has_complete_5y`

## `benchmark_factors_monthly.csv`
One row per month end with FF benchmark series.

Columns:
- `month_end, rf, mkt_rf, smb, hml, rmw, cma, rm`
- `rm = rf + mkt_rf`

## `custom_factor_returns_monthly.csv`
One row per `factor_id, month_end`.

Columns:
- `factor_id, signal_name, construction_rule, month_end`
- `long_ret, short_ret, ls_ret, ls_excess`
- `n_long, n_short, universe_n, signal_year`
"""
    path.write_text(txt, encoding="utf-8")


def build_all_canonical_tables(cfg: BuildConfig) -> Dict[str, object]:
    cfg.curated_root.mkdir(parents=True, exist_ok=True)

    panel, panel_stats = build_ticker_year_panel(cfg)
    factors = build_benchmark_factors_monthly(cfg)
    custom = build_custom_factor_returns_monthly(
        panel=panel,
        factors_monthly=factors,
        returns_file=cfg.returns_file,
        signals=["tls_score", "year_sentiment", "co2_unscaled", "co2_scaled", "co2_log"],
        quantile=cfg.factor_quantile,
    )

    panel_out = cfg.curated_root / "ticker_year_panel.csv"
    factors_out = cfg.curated_root / "benchmark_factors_monthly.csv"
    custom_out = cfg.curated_root / "custom_factor_returns_monthly.csv"
    schema_out = cfg.curated_root / "schema.md"
    manifest_out = cfg.curated_root / "build_manifest.json"

    panel.to_csv(panel_out, index=False)
    factors.to_csv(factors_out, index=False)
    custom.to_csv(custom_out, index=False)
    write_schema_md(schema_out)

    sentiment_src = _load_sentiment(
        cfg.sentiment_file,
        tickers=set(parse_tickers(cfg.universe_file)),
        years=set(cfg.years),
        primary_col=cfg.sentiment_primary_col,
    )
    tls_legacy = pd.DataFrame()
    legacy_tls_path = Path("data/outputs/transition_100/transition_100_panel.csv")
    if legacy_tls_path.exists():
        x = pd.read_csv(legacy_tls_path)
        if {"ticker", "tls_year", "tls_score"}.issubset(x.columns):
            tls_legacy = x.rename(columns={"tls_year": "signal_year"})[["ticker", "signal_year", "tls_score"]].copy()
            tls_legacy["ticker"] = tls_legacy["ticker"].astype(str).str.upper().str.strip()

    validations = validate_canonical_tables(
        panel=panel,
        factors=factors,
        custom=custom,
        sentiment_source=sentiment_src,
        tls_legacy=tls_legacy,
    )

    # Reproducibility check: rebuild dataframes in-memory and compare hashes.
    panel2, _ = build_ticker_year_panel(cfg)
    factors2 = build_benchmark_factors_monthly(cfg)
    custom2 = build_custom_factor_returns_monthly(
        panel=panel2,
        factors_monthly=factors2,
        returns_file=cfg.returns_file,
        signals=["tls_score", "year_sentiment", "co2_unscaled", "co2_scaled", "co2_log"],
        quantile=cfg.factor_quantile,
    )
    reproducible = (
        _df_hash(panel) == _df_hash(panel2)
        and _df_hash(factors) == _df_hash(factors2)
        and _df_hash(custom) == _df_hash(custom2)
    )

    manifest: Dict[str, object] = {
        "inputs": {
            "universe_file": str(cfg.universe_file),
            "years": list(cfg.years),
            "feature_cache_dir": str(cfg.feature_cache_dir),
            "sentiment_file": str(cfg.sentiment_file),
            "returns_file": str(cfg.returns_file),
            "factors_ff3_file": str(cfg.factors_ff3_file),
            "factors_ff5_file": str(cfg.factors_ff5_file),
            "metadata_file": str(cfg.metadata_file),
            "assets_file": str(cfg.assets_file),
            "co2_file": str(cfg.co2_file),
            "tls_formula": {
                "sw": cfg.tls_sw,
                "pw": cfg.tls_pw,
                "cw": cfg.tls_cw,
                "formula": "(sub_total + sw*sub_in_section + pw*sub_near_cw + sw*pw*sub_both_cw - bp_total - sw*bp_in_section) * 10000/tokens",
            },
            "factor_quantile": cfg.factor_quantile,
        },
        "output_files": {
            "ticker_year_panel": str(panel_out),
            "benchmark_factors_monthly": str(factors_out),
            "custom_factor_returns_monthly": str(custom_out),
            "schema_md": str(schema_out),
        },
        "output_hashes": {
            "ticker_year_panel_sha256": _sha256_file(panel_out),
            "benchmark_factors_monthly_sha256": _sha256_file(factors_out),
            "custom_factor_returns_monthly_sha256": _sha256_file(custom_out),
        },
        "stats": panel_stats,
        "validations": validations,
        "reproducible_rebuild": reproducible,
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
