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
from typing import List, Dict, Tuple
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

def pick_10k_rows(df: pd.DataFrame, max_filings: int) -> pd.DataFrame:
    """
    Filter to 10-K / 10-K/A forms and return the most recent 'max_filings' entries.
    """
    m = df["form"].isin(["10-K", "10-K/A"])
    out = df[m].head(max_filings).copy()
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

def html_to_text(html: str) -> str:
    """
    Strip markup with BeautifulSoup, remove scripts/styles/noscript, and return normalized text.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    return normalize_text(text)

def section_weights(text: str) -> List[Tuple[Tuple[int,int], float]]:
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

            weights.append(((start, end), 1.5))

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
                  proximity_multiplier: float=1.0) -> float:
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

            # 2) Numeric proximity: if multiplier > 1.0, search ±120 chars for a number+unit/money phrase
            if proximity_multiplier > 1.0:
                lo = max(0, idx - 120)  # ~±8–10 words in char space
                hi = min(len(text), idx + 120)
                if NUMERIC_CONTEXT.search(text[lo:hi]):
                    w *= proximity_multiplier

            total += w
    return total

def analyze_filing(html: str,
                   substantive_pats: List[re.Pattern],
                   boilerplate_pats: List[re.Pattern]) -> Dict[str, float]:
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

    # Proximity multipliers to test
    PROX_MULTS = [1.0, 1.5, 2.0]  # 1.0 = no boost, 1.5 = moderate, 2.0 = strong

    results = {"tokens": length_tokens}

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

def run(tickers: List[str], max_filings: int,
        score_metric: str,
        output_file: str,
        substantive_file: str=None,
        boilerplate_file: str=None) -> pd.DataFrame:
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
            krows = pick_10k_rows(filings, max_filings)

            # 5) Download and analyze each 10-K primary document
            ticker_rows = []
            for _, fr in krows.iterrows():
                html = download_primary_doc(int(cik), fr["accessionNumber"], fr["primaryDocument"])
                metrics = analyze_filing(html, substantive_pats, boilerplate_pats)
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

# --------------------------- CLI ENTRYPOINT ----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EDGAR 10-K transition-language scorer")
    ap.add_argument("--tickers", type=str, default=None,
                    help="Comma-separated list, e.g., TSLA,NEE,GE")
    ap.add_argument("--tickers-file", type=str, default=None,
                    help="Path to a file containing tickers (space or comma-separated)")
    ap.add_argument("--max-filings", type=int, default=1,
                    help="How many 10-Ks per ticker (most recent first)")
    ap.add_argument("--out", type=str, default="tls_scores.csv",
                    help="Output CSV path")
    ap.add_argument("--score-metric", choices=["tls","substantive"], default="tls",
                    help="Scoring basis: 'tls' (substantive - boilerplate) or 'substantive' only")
    ap.add_argument("--substantive-file", type=str, default="substantive_terms.txt",
                    help="Path to a text file of substantive terms (one per line, default: substantive_terms.txt)")
    ap.add_argument("--boilerplate-file", type=str, default="boilerplate_terms.txt",
                    help="Path to a text file of boilerplate terms (one per line, default: boilerplate_terms.txt)")
    args = ap.parse_args()

    # Get tickers from either --tickers or --tickers-file
    if args.tickers_file:
        content = Path(args.tickers_file).read_text().replace(',', ' ')
        tickers = [t.strip() for t in content.split() if t.strip()]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        ap.error("Must provide either --tickers or --tickers-file")
    res = run(tickers,
              args.max_filings,
              score_metric=args.score_metric,
              output_file=args.out,
              substantive_file=args.substantive_file,
              boilerplate_file=args.boilerplate_file)

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
