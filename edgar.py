#!/usr/bin/env python3
"""
EDGAR 10-K NLP scorer
- Downloads recent 10-Ks for given tickers from SEC EDGAR
- Scores filings for substantive energy-transition language vs boilerplate ESG language
- Computes 18 weighting variations (6 methodologies × 3 metrics)
- Normalizes all counts per 10k words for comparability
- Z-scores are computed in edgar_cleaning.py after data cleaning
"""

import argparse, json, os, re, time, unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm

# ----------------------------- CONFIG ---------------------------------

# IMPORTANT: Use a clear, identifying User-Agent per SEC guidance.
UA = "Mozilla/5.0 (Sid Jain; sidjain@princeton.edu)"
BASE = "https://www.sec.gov"

# === SECTION HEADERS (for weighting) ==================================
# We lightly upweight matches found in these canonical 10-K sections,
# since they tend to contain material operational detail.
SECTION_HEADS = [
    r"Item\s+1\.\s*Business",
    r"Item\s+1A\.\s*Risk\s+Factors",
    r"Item\s+7\.\s*Management(?:’|'|’)s\s+Discussion.*?Analysis",
    r"Item\s+7A\.\s*Quantitative.*?Market\s+Risk"
]

# === NUMERIC CONTEXT REGEX ============================================
# Detects nearby "number + unit/money" phrases to upweight substantive matches.
# Example matches: "500 MW", "$1.2B", "300 million capex", "2 GWh"
# Why it's important:
#   - Quantified mentions (capacity, dollars) are stronger signals of concrete action
#     than vague language, so we boost hits near these phrases.
NUMERIC_CONTEXT = re.compile(
    r"(\$?\b\d[\d,\.]*\b\s*(?:MW|GW|GWh|MWh|MM|B|billion|million|capex|capital|dollars?))",
    re.I
)

# --------------------------- HELPERS ----------------------------------

def http_get(url, params=None, binary=False, sleep=0.8):
    """
    Thin GET wrapper with SEC-compliant headers and gentle pacing.
    - Retries up to 3 times.
    - Sleep (~0.8s) after a successful 200 to keep request rate modest.
    """
    headers = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    for _ in range(3):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            time.sleep(sleep)
            return r.content if binary else r.text
        time.sleep(2)
    raise RuntimeError(f"GET failed: {url} (status {r.status_code})")

def load_ticker_map() -> pd.DataFrame:
    """Load the SEC 'company_tickers.json' list mapping tickers to CIKs."""
    data = json.loads(Path("edgar_cache/company_tickers.json").read_text(encoding="utf-8"))
    recs = []
    for obj in data.values():
        recs.append({"ticker": obj["ticker"].upper(), "cik": int(obj["cik_str"]), "title": obj["title"]})
    return pd.DataFrame(recs)

def get_recent_filings(cik: int) -> pd.DataFrame:
    """
    Retrieve the 'recent' filings index for a company by CIK (official JSON).
    Returns a DataFrame with arrays: form, filingDate, accessionNumber, primaryDocument, etc.
    """
    cik_str = f"{cik:010d}"
    txt = http_get(f"https://data.sec.gov/submissions/CIK{cik_str}.json")
    j = json.loads(txt)
    fr = j["filings"]["recent"]
    df = pd.DataFrame(fr)
    df["cik"] = cik
    return df

def pick_10k_rows(df: pd.DataFrame, max_filings: int, target_year: Optional[int] = None) -> pd.DataFrame:
    """
    Filter to 10-K / 10-K/A forms and return up to 'max_filings' entries.
    If 'target_year' is provided, filings are restricted to that filing year.
    """
    m = df["form"].isin(["10-K", "10-K/A"])
    out = df[m].copy()
    if target_year is not None:
        filing_dates = pd.to_datetime(out["filingDate"], errors="coerce")
        out = out[filing_dates.dt.year == target_year]
    out = out.head(max_filings).copy()
    return out

def normalize_text(s: str) -> str:
    """
    Normalize HTML text:
    - NFKC Unicode normalization (compatibility composed)
    - Collapse all whitespace to single spaces
    """
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def download_primary_doc(cik: int, accession: str, primary_doc: str) -> str:
    """
    Build the EDGAR archive URL for the primary document and download it.
    Example URL pattern:
      https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}
    """
    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{primary_doc}"
    html = http_get(url)
    return html

def cache_primary_doc(cache_dir: str, cik: int, accession: str, primary_doc: str, html: str) -> Path:
    """
    Persist filing HTML into cache directory:
      {cache_dir}/{cik}/{accession_no_dashes}/{primary_doc}
    """
    acc_nodash = accession.replace("-", "")
    cache_path = Path(cache_dir) / str(cik) / acc_nodash / Path(primary_doc).name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    return cache_path

def load_cached_primary_doc(cache_dir: str, cik: int, accession: str, primary_doc: str) -> Optional[str]:
    """Load filing HTML from cache if present, else return None."""
    acc_nodash = accession.replace("-", "")
    cache_path = Path(cache_dir) / str(cik) / acc_nodash / Path(primary_doc).name
    if not cache_path.exists():
        return None
    return cache_path.read_text(encoding="utf-8", errors="ignore")

def html_to_text(html: str) -> str:
    """
    Strip markup with BeautifulSoup, remove scripts/styles/noscript, and return normalized text.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    return normalize_text(text)

def section_weights(text: str, section_weight: float = 1.5) -> List[Tuple[Tuple[int,int], float]]:
    """
    Locate key section headers and assign a weighting window for each ENTIRE section.
    Each section runs from its header until the next 'Item' header (or end of document).
    We apply a 1.5x weight to matches within these sections.
    """
    # Find all Item headers (not just the key ones we weight)
    all_item_pattern = re.compile(r"\bItem\s+\d+[A-Z]?\.", re.I)
    all_items = [(m.start(), m.group()) for m in all_item_pattern.finditer(text)]

    # Sort by position
    all_items.sort(key=lambda x: x[0])

    weights = []
    for pat in SECTION_HEADS:
        for m in re.finditer(pat, text, re.I):
            start = m.start()

            # Find the next Item header after this one
            next_item_pos = None
            for item_pos, item_text in all_items:
                if item_pos > start:
                    next_item_pos = item_pos
                    break

            # End is either next Item or end of document
            end = next_item_pos if next_item_pos is not None else len(text)

            weights.append(((start, end), section_weight))

    return weights

def build_regex_list(terms: List[str]) -> List[re.Pattern]:
    """
    Convert a list of human-readable terms into compiled regex patterns.
    - Escapes each term to treat it literally (so '+' or '(' in terms won't act as regex operators).
    - Adds word boundaries where appropriate (edge cases like 'EV-charging' or multi-word phrases still match).
    """
    patterns = []
    for term in terms:
        t = term.strip()
        if not t:
            continue
        # Escape to make a literal pattern; then allow whitespace/hyphen variation for spaces/hyphens.
        # e.g., "EV charging" should match "EV charging" or "EV-charging".
        escaped = re.escape(t)
        escaped = escaped.replace(r"\ ", r"(?:\s|-)")  # spaces can be space or hyphen
        # Anchor at word boundaries on both sides when term is alnum-like
        pat = rf"\b{escaped}\b"
        patterns.append(re.compile(pat, re.I))
    return patterns

def count_matches(text: str,
                  patterns: List[re.Pattern],
                  section_spans: List[Tuple[Tuple[int,int], float]],
                  proximity_multiplier: float=1.0,
                  proximity_window_chars: int = 120) -> float:
    """
    Count weighted matches for a list of compiled regex 'patterns' in 'text'.
    - 'section_spans' is a list of ((start_idx, end_idx), weight) to upweight matches in key sections.
    - 'proximity_multiplier': weight multiplier when match is near numeric context (1.0 = no boost)
    Returns a float total (weights summed).
    """
    total = 0.0
    for pat in patterns:
        for m in pat.finditer(text):
            w = 1.0
            idx = m.start()

            # 1) Section weighting: if match index falls in any boosted section window, multiply weight
            for (lo, hi), ww in section_spans:
                if lo <= idx <= hi:
                    w *= ww
                    break

            # 2) Numeric proximity: search within a configurable ±window char radius
            if proximity_multiplier > 1.0:
                lo = max(0, idx - proximity_window_chars)
                hi = min(len(text), idx + proximity_window_chars)
                if NUMERIC_CONTEXT.search(text[lo:hi]):
                    w *= proximity_multiplier

            total += w
    return total

def compute_parametric_score(
    html: str,
    substantive_pats: List[re.Pattern],
    boilerplate_pats: List[re.Pattern],
    section_weight: float,
    proximity_weight: float,
    proximity_window_chars: int,
    score_mode: str,
) -> Dict[str, float]:
    """
    Compute a single parametric score used for sweep runs.
    score_mode:
      - substantive_only: score = substantive score
      - net_tls: score = substantive - boilerplate
    """
    text = html_to_text(html)
    length_tokens = max(1, len(text.split()))
    scale = 10000.0 / length_tokens

    spans = section_weights(text, section_weight=section_weight)
    substantive_raw = count_matches(
        text,
        substantive_pats,
        spans,
        proximity_multiplier=proximity_weight,
        proximity_window_chars=proximity_window_chars,
    )
    boilerplate_raw = count_matches(
        text,
        boilerplate_pats,
        spans,
        proximity_multiplier=1.0,
        proximity_window_chars=proximity_window_chars,
    )

    substantive = substantive_raw * scale
    boilerplate = boilerplate_raw * scale
    if score_mode == "substantive_only":
        beta_score = substantive
    else:
        beta_score = substantive - boilerplate

    return {
        "tokens": length_tokens,
        "substantive_score": substantive,
        "boilerplate_score": boilerplate,
        "beta_score": beta_score,
    }

def analyze_filing(html: str,
                   substantive_pats: List[re.Pattern],
                   boilerplate_pats: List[re.Pattern],
                   full15_only: bool = False) -> Dict[str, float]:
    """
    Convert HTML -> plain text, compute normalized counts per 10k words with multiple weighting variations.

    Variations tested:
      - Base (no weights)
      - Section-weighted only
      - Proximity-weighted with multipliers: 1.5x, 2x
      - Section + Proximity combinations

    This allows testing different NLP methodologies without re-downloading filings.
    """
    text = html_to_text(html)
    length_tokens = max(1, len(text.split()))
    scale = 10000.0 / length_tokens

    # Get section spans for weighting (empty list = no section weighting)
    spans = section_weights(text)
    no_spans = []  # For unweighted baseline

    results = {"tokens": length_tokens}

    if full15_only:
        sub_sp = count_matches(text, substantive_pats, spans, proximity_multiplier=1.5)
        bp_sp = count_matches(text, boilerplate_pats, spans, proximity_multiplier=1.0)
        results["substantive_full15"] = sub_sp * scale
        results["boilerplate_full15"] = bp_sp * scale
        results["tls_full15"] = (sub_sp - bp_sp) * scale
        return results

    # 1. Base (no weights at all)
    sub_base = count_matches(text, substantive_pats, no_spans, proximity_multiplier=1.0)
    bp_base = count_matches(text, boilerplate_pats, no_spans, proximity_multiplier=1.0)
    results["substantive_base"] = sub_base * scale
    results["boilerplate_base"] = bp_base * scale
    results["tls_base"] = (sub_base - bp_base) * scale

    # 2. Section-weighted only (no proximity)
    sub_section = count_matches(text, substantive_pats, spans, proximity_multiplier=1.0)
    bp_section = count_matches(text, boilerplate_pats, spans, proximity_multiplier=1.0)
    results["substantive_section"] = sub_section * scale
    results["boilerplate_section"] = bp_section * scale
    results["tls_section"] = (sub_section - bp_section) * scale

    # 3. Proximity variations (no section weighting)
    for pm in [1.5, 2.0]:
        suffix = f"prox{int(pm*10)}"  # prox15, prox20
        sub_p = count_matches(text, substantive_pats, no_spans, proximity_multiplier=pm)
        bp_p = count_matches(text, boilerplate_pats, no_spans, proximity_multiplier=1.0)
        results[f"substantive_{suffix}"] = sub_p * scale
        results[f"boilerplate_{suffix}"] = bp_p * scale
        results[f"tls_{suffix}"] = (sub_p - bp_p) * scale

    # 4. Section + Proximity combinations
    for pm in [1.5, 2.0]:
        suffix = f"full{int(pm*10)}"  # full15, full20
        sub_sp = count_matches(text, substantive_pats, spans, proximity_multiplier=pm)
        bp_sp = count_matches(text, boilerplate_pats, spans, proximity_multiplier=1.0)
        results[f"substantive_{suffix}"] = sub_sp * scale
        results[f"boilerplate_{suffix}"] = bp_sp * scale
        results[f"tls_{suffix}"] = (sub_sp - bp_sp) * scale

    return results

def load_terms_from_file(path: str) -> List[str]:
    """Load term list from a text file (one term per line)."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def filter_top_market_cap_tickers(
    tickers: List[str],
    market_cap_file: str,
    top_n: int = 1000
) -> List[str]:
    """
    Keep only the top-N tickers by market cap from ticker metadata.
    Expects `ticker_addtl_data.csv`-style schema:
      - ticker column: `filename`
      - market cap column: `value`
    """
    if top_n <= 0:
        return [t.strip().upper() for t in tickers if t.strip()]

    df = pd.read_csv(market_cap_file)
    if "filename" not in df.columns or "value" not in df.columns:
        raise ValueError(
            f"{market_cap_file} must contain columns `filename` (ticker) and `value` (market cap)"
        )

    df = df.copy()
    df["ticker"] = df["filename"].astype(str).str.strip().str.upper()
    df["market_cap"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["ticker", "market_cap"])

    top = (
        df.sort_values("market_cap", ascending=False)
          .drop_duplicates(subset=["ticker"], keep="first")
          .head(top_n)
    )
    top_set = set(top["ticker"])

    incoming = [t.strip().upper() for t in tickers if t.strip()]
    filtered = [t for t in incoming if t in top_set]
    print(f"[INFO] Market-cap filter: kept {len(filtered)} / {len(incoming)} tickers (top {top_n})")
    return filtered

def run(tickers: List[str], max_filings: int,
        score_metric: str,
        output_file: str,
        substantive_file: str=None,
        boilerplate_file: str=None,
        target_year: Optional[int] = None,
        full15_only: bool = False,
        cache_dir: Optional[str] = None,
        cache_only: bool = False,
        section_weight: float = 1.5,
        proximity_weight: float = 1.5,
        proximity_window: int = 120,
        score_mode: str = "net_tls") -> pd.DataFrame:
    """
    Orchestrates the workflow for a list of tickers:
    - Map tickers -> CIK
    - Pull recent filings index
    - Download 10-K primary docs
    - Analyze, normalize, and compute chosen score and z-score
    - Continuously saves results and skips already-processed tickers
    """
    # 1) Load lexicons from external files
    substantive_terms = load_terms_from_file(substantive_file)
    boilerplate_terms = load_terms_from_file(boilerplate_file)

    # Compile terms into regex patterns once
    substantive_pats = build_regex_list(substantive_terms)
    boilerplate_pats = build_regex_list(boilerplate_terms)

    # 2) Load existing results to skip already-processed tickers
    existing_tickers = set()
    if Path(output_file).exists():
        existing_df = pd.read_csv(output_file)
        existing_tickers = set(existing_df["ticker"].unique())
        tqdm.write(f"[INFO] Loaded {len(existing_tickers)} already-processed tickers, will skip them")

    # 3) Map tickers -> CIKs
    tickmap = load_ticker_map()
    tickmap["ticker"] = tickmap["ticker"].str.upper()
    want = pd.DataFrame({"ticker": [t.strip().upper() for t in tickers]})
    dfm = want.merge(tickmap, on="ticker", how="left")

    # Filter out already-processed tickers
    dfm = dfm[~dfm["ticker"].isin(existing_tickers)]
    tqdm.write(f"[INFO] {len(dfm)} tickers remaining to process")

    rows = []
    for _, r in tqdm(dfm.iterrows(), total=len(dfm), desc="Processing tickers", unit="ticker"):
        tkr, cik = r["ticker"], r["cik"]
        if pd.isna(cik):
            tqdm.write(f"[WARN] No CIK for {tkr} (skipping)")
            continue
        try:
            # 4) Pull the recent filings JSON and filter to 10-K rows
            filings = get_recent_filings(int(cik))
            krows = pick_10k_rows(filings, max_filings, target_year=target_year)
            if len(krows) == 0:
                tqdm.write(f"[WARN] {tkr}: no 10-K filings found for year {target_year}")
                continue

            # 5) Download and analyze each 10-K primary document
            ticker_rows = []
            for _, fr in krows.iterrows():
                html = None
                if cache_dir:
                    html = load_cached_primary_doc(
                        cache_dir,
                        int(cik),
                        fr["accessionNumber"],
                        fr["primaryDocument"]
                    )
                if html is None:
                    if cache_only:
                        tqdm.write(
                            f"[WARN] {tkr}: cache miss for {fr['accessionNumber']} / {fr['primaryDocument']} (cache_only enabled)"
                        )
                        continue
                    html = download_primary_doc(int(cik), fr["accessionNumber"], fr["primaryDocument"])
                    if cache_dir:
                        cache_primary_doc(
                            cache_dir,
                            int(cik),
                            fr["accessionNumber"],
                            fr["primaryDocument"],
                            html
                        )
                if full15_only:
                    param_scores = compute_parametric_score(
                        html=html,
                        substantive_pats=substantive_pats,
                        boilerplate_pats=boilerplate_pats,
                        section_weight=section_weight,
                        proximity_weight=proximity_weight,
                        proximity_window_chars=proximity_window,
                        score_mode=score_mode,
                    )
                    metrics = {
                        "tokens": param_scores["tokens"],
                        "substantive_full15": param_scores["substantive_score"],
                        "boilerplate_full15": param_scores["boilerplate_score"],
                        "tls_full15": param_scores["beta_score"],
                    }
                else:
                    metrics = analyze_filing(
                        html,
                        substantive_pats,
                        boilerplate_pats,
                        full15_only=full15_only
                    )
                ticker_rows.append({
                    "ticker": tkr,
                    "cik": int(cik),
                    "filing_date": fr["filingDate"],
                    "accession": fr["accessionNumber"],
                    "doc": fr["primaryDocument"],
                    **metrics
                })

            rows.extend(ticker_rows)
            tqdm.write(f"[DONE] {tkr}: {len(krows)} filings analyzed")

            # 6) Append to CSV file immediately after each ticker
            if ticker_rows:
                temp_df = pd.DataFrame(ticker_rows)
                # Write header only if file doesn't exist
                write_header = not Path(output_file).exists()
                temp_df.to_csv(output_file, mode='a', header=write_header, index=False)

        except Exception as e:
            tqdm.write(f"[ERR] {tkr}: {e}")

    # 7) Load all results for final z-score computation
    out = pd.read_csv(output_file) if Path(output_file).exists() else pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # Note: z-scores are now computed in edgar_cleaning.py after data cleaning
    return out

def cache_filings_for_years(
    tickers: List[str],
    years: List[int],
    max_filings_per_year: int,
    cache_dir: str,
    manifest_path: Optional[str] = None
):
    """
    Download and cache filing HTML for each ticker-year request.
    Writes a manifest CSV with cache status for auditing/reproducibility.
    """
    tickmap = load_ticker_map()
    tickmap["ticker"] = tickmap["ticker"].str.upper()
    want = pd.DataFrame({"ticker": [t.strip().upper() for t in tickers]})
    dfm = want.merge(tickmap, on="ticker", how="left")

    records = []
    total_targets = len(dfm) * len(years)
    print(f"[CACHE] Tickers: {len(dfm)}, years: {len(years)}, target ticker-year pairs: {total_targets}")

    for _, r in tqdm(dfm.iterrows(), total=len(dfm), desc="Caching tickers", unit="ticker"):
        tkr, cik = r["ticker"], r["cik"]
        if pd.isna(cik):
            for y in years:
                records.append({
                    "ticker": tkr,
                    "cik": None,
                    "year": y,
                    "accession": None,
                    "doc": None,
                    "filing_date": None,
                    "status": "missing_cik",
                    "cache_path": None
                })
            tqdm.write(f"[WARN] No CIK for {tkr} (skipping all years)")
            continue

        try:
            filings = get_recent_filings(int(cik))
        except Exception as e:
            for y in years:
                records.append({
                    "ticker": tkr,
                    "cik": int(cik),
                    "year": y,
                    "accession": None,
                    "doc": None,
                    "filing_date": None,
                    "status": f"filings_error: {e}",
                    "cache_path": None
                })
            tqdm.write(f"[ERR] {tkr}: could not fetch filings index ({e})")
            continue

        for year in years:
            krows = pick_10k_rows(filings, max_filings_per_year, target_year=year)
            if len(krows) == 0:
                records.append({
                    "ticker": tkr,
                    "cik": int(cik),
                    "year": year,
                    "accession": None,
                    "doc": None,
                    "filing_date": None,
                    "status": "no_filing_for_year",
                    "cache_path": None
                })
                continue

            for _, fr in krows.iterrows():
                accession = fr["accessionNumber"]
                primary_doc = fr["primaryDocument"]
                filing_date = fr["filingDate"]
                cached = load_cached_primary_doc(cache_dir, int(cik), accession, primary_doc)
                if cached is not None:
                    acc_nodash = accession.replace("-", "")
                    cache_path = Path(cache_dir) / str(int(cik)) / acc_nodash / Path(primary_doc).name
                    records.append({
                        "ticker": tkr,
                        "cik": int(cik),
                        "year": year,
                        "accession": accession,
                        "doc": primary_doc,
                        "filing_date": filing_date,
                        "status": "already_cached",
                        "cache_path": str(cache_path)
                    })
                    continue

                try:
                    html = download_primary_doc(int(cik), accession, primary_doc)
                    cache_path = cache_primary_doc(cache_dir, int(cik), accession, primary_doc, html)
                    records.append({
                        "ticker": tkr,
                        "cik": int(cik),
                        "year": year,
                        "accession": accession,
                        "doc": primary_doc,
                        "filing_date": filing_date,
                        "status": "cached",
                        "cache_path": str(cache_path)
                    })
                except Exception as e:
                    records.append({
                        "ticker": tkr,
                        "cik": int(cik),
                        "year": year,
                        "accession": accession,
                        "doc": primary_doc,
                        "filing_date": filing_date,
                        "status": f"download_error: {e}",
                        "cache_path": None
                    })

    manifest_df = pd.DataFrame(records)
    manifest_file = manifest_path or str(Path(cache_dir) / "cache_manifest.csv")
    Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_file, index=False)
    print(f"[CACHE] Manifest written: {manifest_file} ({len(manifest_df)} rows)")

def run_year_specific_exports(
    tickers: List[str],
    years: List[int],
    max_filings_per_year: int,
    out_prefix: str,
    substantive_file: str,
    boilerplate_file: str,
    cache_dir: Optional[str] = None,
    cache_only: bool = False,
    section_weight: float = 1.5,
    proximity_weight: float = 1.5,
    proximity_window: int = 120,
    score_mode: str = "net_tls",
):
    """
    Run NLP scoring for each year and write one CSV per year.
    In this mode, only FULL15 methodology columns are written.
    """
    cols = [
        "ticker", "cik", "filing_date", "accession", "doc", "tokens",
        "substantive_full15", "boilerplate_full15", "tls_full15",
    ]

    for year in years:
        output_file = f"{out_prefix}_{year}.csv"
        print(f"\n[RUN] Year {year} -> {output_file}")
        res = run(
            tickers=tickers,
            max_filings=max_filings_per_year,
            score_metric="tls",
            output_file=output_file,
            substantive_file=substantive_file,
            boilerplate_file=boilerplate_file,
            target_year=year,
            full15_only=True,
            cache_dir=cache_dir,
            cache_only=cache_only,
            section_weight=section_weight,
            proximity_weight=proximity_weight,
            proximity_window=proximity_window,
            score_mode=score_mode,
        )

        if len(res) > 0:
            missing = [c for c in cols if c not in res.columns]
            if missing:
                raise RuntimeError(f"Missing expected FULL15 columns in {output_file}: {missing}")
            Path(output_file).write_text(res[cols].to_csv(index=False), encoding="utf-8")
            print(f"[COMPLETE] {output_file}: {len(res)} rows, FULL15-only")
        else:
            print(f"[COMPLETE] {output_file}: no rows")

# --------------------------- CLI ENTRYPOINT ----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EDGAR 10-K transition-language scorer")
    ap.add_argument("--tickers-file", type=str, default="tickers.txt",
                    help="Path to a file containing tickers (space or comma-separated)")
    ap.add_argument("--max-filings", type=int, default=1,
                    help="How many 10-Ks per ticker (most recent first)")
    ap.add_argument("--out", type=str, default="tls_scores.csv",
                    help="Output CSV path")
    ap.add_argument("--years", type=str, default=None,
                    help="Comma-separated filing years to export separately (e.g., 2019,2020,2024)")
    ap.add_argument("--out-prefix", type=str, default="tls_scores",
                    help="Prefix for per-year output files when --years is used")
    ap.add_argument("--max-filings-per-year", type=int, default=1,
                    help="How many 10-Ks per ticker to keep within each target year")
    ap.add_argument("--cache-years", type=str, default=None,
                    help="Comma-separated years to download/cache filing HTML only (e.g., 2019,2020,2024)")
    ap.add_argument("--cache-dir", type=str, default="cached_edgar",
                    help="Directory for cached EDGAR filing HTML")
    ap.add_argument("--cache-manifest", type=str, default=None,
                    help="Optional manifest CSV path for cache mode")
    ap.add_argument("--cache-only", action="store_true",
                    help="Do not download missing filings; only read from cache")
    ap.add_argument("--top-market-cap", type=int, default=1000,
                    help="Keep only top N tickers by market cap (<=0 disables filter)")
    ap.add_argument("--market-cap-file", type=str, default="ticker_addtl_data.csv",
                    help="CSV path with market cap data (expects `filename` and `value` columns)")
    ap.add_argument("--section-weight", type=float, default=1.5,
                    help="Section weighting multiplier for parametric score runs")
    ap.add_argument("--proximity-weight", type=float, default=1.5,
                    help="Numeric proximity multiplier for substantive term matches")
    ap.add_argument("--proximity-window", type=int, default=120,
                    help="Character window radius for numeric proximity checks")
    ap.add_argument("--score-mode", choices=["substantive_only", "net_tls"], default="net_tls",
                    help="Parametric score mode: substantive_only or substantive - boilerplate")
    ap.add_argument("--score-metric", choices=["tls","substantive"], default="tls",
                    help="Scoring basis: 'tls' (substantive - boilerplate) or 'substantive' only")
    ap.add_argument("--substantive-file", type=str, default="substantive_terms.txt",
                    help="Path to a text file of substantive terms (one per line, default: substantive_terms.txt)")
    ap.add_argument("--boilerplate-file", type=str, default="boilerplate_terms.txt",
                    help="Path to a text file of boilerplate terms (one per line, default: boilerplate_terms.txt)")
    args = ap.parse_args()

    # Always load tickers from tickers file (default: tickers.txt)
    content = Path(args.tickers_file).read_text().replace(',', ' ')
    tickers = [t.strip() for t in content.split() if t.strip()]
    tickers = filter_top_market_cap_tickers(
        tickers=tickers,
        market_cap_file=args.market_cap_file,
        top_n=args.top_market_cap
    )
    if not tickers:
        print("No tickers left after market-cap filter.")
        raise SystemExit(0)

    if args.cache_years:
        years = [int(y.strip()) for y in args.cache_years.split(",") if y.strip()]
        cache_filings_for_years(
            tickers=tickers,
            years=years,
            max_filings_per_year=args.max_filings_per_year,
            cache_dir=args.cache_dir,
            manifest_path=args.cache_manifest
        )
        raise SystemExit(0)

    if args.years:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
        run_year_specific_exports(
            tickers=tickers,
            years=years,
            max_filings_per_year=args.max_filings_per_year,
            out_prefix=args.out_prefix,
            substantive_file=args.substantive_file,
            boilerplate_file=args.boilerplate_file,
            cache_dir=args.cache_dir,
            cache_only=args.cache_only,
            section_weight=args.section_weight,
            proximity_weight=args.proximity_weight,
            proximity_window=args.proximity_window,
            score_mode=args.score_mode,
        )
        raise SystemExit(0)

    res = run(
        tickers,
        args.max_filings,
        score_metric=args.score_metric,
        output_file=args.out,
        substantive_file=args.substantive_file,
        boilerplate_file=args.boilerplate_file,
        cache_dir=args.cache_dir,
        cache_only=args.cache_only,
        section_weight=args.section_weight,
        proximity_weight=args.proximity_weight,
        proximity_window=args.proximity_window,
        score_mode=args.score_mode
    )

    # Final save (z-scores will be computed in edgar_cleaning.py)
    if len(res) > 0:
        cols = [
            "ticker","cik","filing_date","accession","doc","tokens",
            "substantive_base","boilerplate_base","tls_base",
            "substantive_section","boilerplate_section","tls_section",
            "substantive_prox15","boilerplate_prox15","tls_prox15",
            "substantive_prox20","boilerplate_prox20","tls_prox20",
            "substantive_full15","boilerplate_full15","tls_full15",
            "substantive_full20","boilerplate_full20","tls_full20",
        ]
        Path(args.out).write_text(res[cols].to_csv(index=False), encoding="utf-8")
        print(f"\n[COMPLETE] {args.out}: {len(res)} rows, {len(cols)} columns")
        print(f"6 methodologies × 3 metrics = 18 raw scores (z-scores added in edgar_cleaning.py)")
    else:
        print("No results.")
