#!/usr/bin/env python3
"""
Build reusable filing feature caches for TLS searches.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
from bisect import bisect_right
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesis_pipeline.tls import (
    NUMERIC_CONTEXT,
    build_regex_list,
    html_to_text,
    load_terms_from_file,
    section_weights,
)


DEFAULTS = {
    "years": "2015-2024",
    "cache_dir": Path("data/cache/edgar_html"),
    "text_cache_dir": Path("data/cache/edgar_text"),
    "cache_manifest": Path("data/cache/edgar_html/cache_manifest.csv"),
    "windows": "10,132,255,378,500",
    "r1000": {
        "tickers_file": Path("data/raw/universe/tickers.txt"),
        "out_dir": Path("data/processed/feature_cache/r1000"),
    },
    "t100": {
        "tickers_file": Path("data/raw/universe/transition_100_tickers.txt"),
        "out_dir": Path("data/processed/feature_cache/t100"),
    },
    "substantive_file": Path("assets/terms/substantive_terms.txt"),
    "boilerplate_file": Path("assets/terms/boilerplate_terms.txt"),
}


def parse_tickers(path: str) -> List[str]:
    content = Path(path).read_text(encoding="utf-8").replace(",", " ")
    return sorted({t.strip().upper() for t in content.split() if t.strip()})


def parse_windows(window_str: str) -> List[int]:
    vals = sorted({int(x.strip()) for x in window_str.split(",") if x.strip()})
    if not vals:
        raise ValueError("At least one window value is required.")
    if any(v <= 0 for v in vals):
        raise ValueError("All window values must be positive integers.")
    return vals


def parse_years(year_str: str) -> List[int]:
    years: List[int] = []
    for chunk in str(year_str).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            years.extend(range(start, end + step, step))
        else:
            years.append(int(token))
    unique = sorted(set(years))
    if not unique:
        raise ValueError("At least one year is required.")
    return unique


def load_manifest_candidates(manifest_path: str, tickers: List[str], year: int) -> pd.DataFrame:
    m = pd.read_csv(manifest_path)
    m["ticker"] = m["ticker"].astype(str).str.strip().str.upper()
    m["year"] = pd.to_numeric(m["year"], errors="coerce")
    m = m[m["ticker"].isin(set(tickers))].copy()
    m = m[m["year"] == year].copy()
    m = m[m["status"].isin(["cached", "already_cached"])].copy()
    m["filing_date"] = pd.to_datetime(m["filing_date"], errors="coerce")
    m = m.sort_values(["ticker", "filing_date"], ascending=[True, False]).drop_duplicates("ticker")
    return m[["ticker", "cik", "accession", "doc", "filing_date", "cache_path"]].copy()


def _idx_in_section(idx: int, spans: List[Tuple[int, int]]) -> bool:
    for lo, hi in spans:
        if lo <= idx <= hi:
            return True
    return False


def _build_numeric_prefix(text: str) -> np.ndarray:
    n = len(text)
    mask = np.zeros(n, dtype=np.uint8)
    for m in NUMERIC_CONTEXT.finditer(text):
        lo, hi = m.start(), m.end()
        if lo < hi:
            mask[lo:hi] = 1
    prefix = np.zeros(n + 1, dtype=np.int32)
    prefix[1:] = np.cumsum(mask, dtype=np.int32)
    return prefix


def _has_numeric(prefix: np.ndarray, idx: int, window: int) -> bool:
    n = len(prefix) - 1
    lo = max(0, idx - window)
    hi = min(n, idx + window)
    return (prefix[hi] - prefix[lo]) > 0


def build_feature_row(
    text: str,
    substantive_pats,
    boilerplate_pats,
    windows: List[int],
) -> Dict[str, float]:
    n_tokens = max(1, len(text.split()))
    spans = [span for span, _w in section_weights(text, section_weight=1.0)]
    prefix = _build_numeric_prefix(text)

    row: Dict[str, float] = {
        "tokens": float(n_tokens),
        "sub_total": 0.0,
        "sub_in_section": 0.0,
        "bp_total": 0.0,
        "bp_in_section": 0.0,
    }
    for w in windows:
        row[f"sub_near_{w}"] = 0.0
        row[f"sub_both_{w}"] = 0.0

    for pat in substantive_pats:
        for m in pat.finditer(text):
            idx = m.start()
            in_sec = _idx_in_section(idx, spans)
            row["sub_total"] += 1.0
            if in_sec:
                row["sub_in_section"] += 1.0
            for w in windows:
                near = _has_numeric(prefix, idx, w)
                if near:
                    row[f"sub_near_{w}"] += 1.0
                    if in_sec:
                        row[f"sub_both_{w}"] += 1.0

    for pat in boilerplate_pats:
        for m in pat.finditer(text):
            idx = m.start()
            row["bp_total"] += 1.0
            if _idx_in_section(idx, spans):
                row["bp_in_section"] += 1.0

    return row


def build_feature_cache_for_year(
    tickers_file: str,
    cache_dir: str,
    text_cache_dir: str,
    cache_manifest: str,
    year: int,
    windows: List[int],
    substantive_file: str,
    boilerplate_file: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tickers = parse_tickers(tickers_file)
    candidates = load_manifest_candidates(cache_manifest, tickers, year)

    substantive_pats = build_regex_list(load_terms_from_file(substantive_file))
    boilerplate_pats = build_regex_list(load_terms_from_file(boilerplate_file))

    cache_dir_path = Path(cache_dir)
    text_cache_dir_path = Path(text_cache_dir)

    out_rows = []
    dropped_rows = []

    for _, r in tqdm(candidates.iterrows(), total=len(candidates), desc="Building feature cache", unit="filing"):
        ticker = r["ticker"]
        raw_cache_path = str(r["cache_path"])
        normalized = raw_cache_path.replace("\\", "/")
        rel_hint = None
        if normalized.startswith("cached_edgar/"):
            rel_hint = Path(normalized.split("cached_edgar/", 1)[1])
        elif "/cached_edgar/" in normalized:
            rel_hint = Path(normalized.split("/cached_edgar/", 1)[1])

        if rel_hint is not None:
            html_path = cache_dir_path / rel_hint
        else:
            html_path = Path(raw_cache_path)

        if not html_path.exists():
            dropped_rows.append({"ticker": ticker, "reason": "missing_html_cache"})
            continue

        if rel_hint is not None:
            rel = rel_hint
        else:
            try:
                rel = html_path.relative_to(cache_dir_path)
            except ValueError:
                rel = Path(html_path.name)
        text_path = text_cache_dir_path / rel.with_suffix(".txt")
        if not text_path.exists():
            try:
                html = html_path.read_text(encoding="utf-8", errors="ignore")
                text = html_to_text(html)
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(text, encoding="utf-8")
            except Exception:
                dropped_rows.append({"ticker": ticker, "reason": "missing_text_cache"})
                continue

        text = text_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            dropped_rows.append({"ticker": ticker, "reason": "empty_text"})
            continue

        feat = build_feature_row(
            text=text,
            substantive_pats=substantive_pats,
            boilerplate_pats=boilerplate_pats,
            windows=windows,
        )
        out_rows.append(
            {
                "ticker": ticker,
                "cik": r["cik"],
                "filing_date": r["filing_date"],
                "accession": r["accession"],
                "doc": r["doc"],
                **feat,
            }
        )

    out_df = pd.DataFrame(out_rows)
    if not out_df.empty and "ticker" in out_df.columns:
        out_df = out_df.sort_values("ticker")
    else:
        out_df = pd.DataFrame(columns=[
            "ticker", "cik", "filing_date", "accession", "doc", "tokens",
            "sub_total", "sub_in_section", "bp_total", "bp_in_section",
            *[f"sub_near_{w}" for w in windows],
            *[f"sub_both_{w}" for w in windows],
        ])
    drop_df = pd.DataFrame(dropped_rows).sort_values(["reason", "ticker"]) if dropped_rows else pd.DataFrame(columns=["ticker", "reason"])
    return out_df, drop_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build TLS feature caches for one or more years")
    ap.add_argument("--universe", choices=["r1000", "t100"], default="r1000")
    ap.add_argument("--tickers-file", type=str, default=None)
    ap.add_argument("--cache-dir", type=str, default=str(DEFAULTS["cache_dir"]))
    ap.add_argument("--text-cache-dir", type=str, default=str(DEFAULTS["text_cache_dir"]))
    ap.add_argument("--cache-manifest", type=str, default=str(DEFAULTS["cache_manifest"]))
    ap.add_argument("--years", type=str, default=DEFAULTS["years"])
    ap.add_argument("--windows", type=str, default=DEFAULTS["windows"],
                    help="Comma-separated proximity windows (chars)")
    ap.add_argument("--substantive-file", type=str, default=str(DEFAULTS["substantive_file"]))
    ap.add_argument("--boilerplate-file", type=str, default=str(DEFAULTS["boilerplate_file"]))
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    windows = parse_windows(args.windows)
    years = parse_years(args.years)
    tickers_file = args.tickers_file or str(DEFAULTS[args.universe]["tickers_file"])
    out_dir = Path(args.out_dir or DEFAULTS[args.universe]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        out_df, drop_df = build_feature_cache_for_year(
            tickers_file=tickers_file,
            cache_dir=args.cache_dir,
            text_cache_dir=args.text_cache_dir,
            cache_manifest=args.cache_manifest,
            year=year,
            windows=windows,
            substantive_file=args.substantive_file,
            boilerplate_file=args.boilerplate_file,
        )
        out_file = out_dir / f"feature_cache_{year}.csv"
        dropped_file = out_dir / f"feature_cache_{year}_dropped.csv"
        out_df.to_csv(out_file, index=False)
        drop_df.to_csv(dropped_file, index=False)
        print(f"[COMPLETE] {args.universe} {year}: {out_file} ({len(out_df)} rows)")
        print(f"[COMPLETE] {args.universe} {year}: {dropped_file} ({len(drop_df)} rows)")

    print(f"[INFO] Windows cached: {windows}")


if __name__ == "__main__":
    main()
