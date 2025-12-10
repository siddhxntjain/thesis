#!/usr/bin/env python3
"""
EDGAR 10-K NLP scorer
- Downloads recent 10-Ks for given tickers (official EDGAR JSON endpoints)
- Scores each filing for "substantive" energy-transition language vs "boilerplate" ESG language
- Normalizes counts per 10k words
- Computes z-scores on a chosen metric:
    --score-metric tls          -> tls_raw = (substantive - boilerplate) per 10k words
    --score-metric substantive  -> substantive_per10k only
- Saves a CSV with all components plus the chosen score and z-score

Notes:
- This script analyzes HTML text without scraping tricks; uses SEC endpoints with a compliant User-Agent.
- If you want to add/remove terms, edit the SUBSTANTIVE_TERMS / BOILERPLATE_TERMS blocks
  (or pass external text files via --substantive-file / --boilerplate-file).
"""

import argparse, json, os, re, time, unicodedata
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# ----------------------------- CONFIG ---------------------------------

# IMPORTANT: Use a clear, identifying User-Agent per SEC guidance.
UA = "Mozilla/5.0 (Sid Jain; sidjain@princeton.edu)"
BASE = "https://www.sec.gov"

# Local cache for the SEC ticker map JSON (avoids re-downloading every run)
DATA_DIR = Path("edgar_cache")
DATA_DIR.mkdir(exist_ok=True)

# === LEXICONS (easy to edit) ==========================================
# You can edit these multi-line strings directly (one term per line), or
# provide external files with --substantive-file / --boilerplate-file.

SUBSTANTIVE_TERMS = """
renewable
solar
wind
offshore wind
wind turbine
wind farm
hydrogen
electrolyzer
direct air capture
DAC
carbon capture
CCS
battery
energy storage
EV charging
EV-charging
heat pump
interconnection
grid modernization
transmission
power purchase agreement
PPA
offtake
capacity
MW
GW
GWh
LCOE
IRA
45Q
48C
45V
45X
tax credit
capex
capital expenditure
retrofit
scope 1
scope 2
scope 3
""".strip().splitlines()

BOILERPLATE_TERMS = """
ESG
CSR
sustainability values
materiality assessment
GRI
SASB
TCFD
purpose
responsible business
diversity and inclusion
D&I
""".strip().splitlines()

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

def zscore(x: pd.Series) -> pd.Series:
    """Standardize a vector to mean 0, std 1 (adds small epsilon to avoid divide-by-zero)."""
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

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
    """
    Load the SEC 'company_tickers.json' list mapping tickers to CIKs.
    Caches to DATA_DIR to avoid repeated downloads.
    """
    path = DATA_DIR / "company_tickers.json"
    if not path.exists():
        txt = http_get(f"{BASE}/files/company_tickers.json")
        path.write_text(txt, encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
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
    Locate key section headers and assign a weighting window after each header.
    We apply a 1.5x weight to matches within ~20k characters after each header.
    """
    weights = []
    for pat in SECTION_HEADS:
        for m in re.finditer(pat, text, re.I):
            start = max(0, m.start())
            end = min(len(text), m.start() + 20000)  # 20k chars window
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
                  proximity: bool=False) -> float:
    """
    Count weighted matches for a list of compiled regex 'patterns' in 'text'.
    - 'section_spans' is a list of ((start_idx, end_idx), weight) to upweight matches in key sections.
    - If 'proximity' is True, boost matches with nearby numeric context (capacity/$).
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

            # 2) Numeric proximity: if enabled, search ±120 chars for a number+unit/money phrase
            if proximity:
                lo = max(0, idx - 120)  # ~±8–10 words in char space
                hi = min(len(text), idx + 120)
                if NUMERIC_CONTEXT.search(text[lo:hi]):
                    w *= 1.4

            total += w
    return total

def analyze_filing(html: str,
                   substantive_pats: List[re.Pattern],
                   boilerplate_pats: List[re.Pattern]) -> Dict[str, float]:
    """
    Convert HTML -> plain text, compute normalized counts per 10k words for:
      - substantive_per10k
      - boilerplate_per10k
      - tls_raw = (substantive - boilerplate) per 10k
    """
    text = html_to_text(html)
    length_tokens = max(1, len(text.split()))
    spans = section_weights(text)

    sub = count_matches(text, substantive_pats, spans, proximity=True)
    pr  = count_matches(text, boilerplate_pats, spans, proximity=False)

    # Normalize per 10k words so filings of different lengths are comparable
    scale = 10000.0 / length_tokens
    return {
        "tokens": length_tokens,
        "substantive_per10k": sub * scale,
        "boilerplate_per10k": pr * scale,
        "tls_raw": (sub - pr) * scale
    }

def load_terms_from_file(path: str) -> List[str]:
    """
    Optional helper: load a term list from a text file (one term per line).
    Blank lines are ignored.
    """
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                lines.append(t)
    return lines

def run(tickers: List[str], max_filings: int,
        score_metric: str,
        substantive_file: str=None,
        boilerplate_file: str=None) -> pd.DataFrame:
    """
    Orchestrates the workflow for a list of tickers:
    - Map tickers -> CIK
    - Pull recent filings index
    - Download 10-K primary docs
    - Analyze, normalize, and compute chosen score and z-score
    """
    # 1) Build lexicons (either from defaults above, or override from files)
    substantive_terms = SUBSTANTIVE_TERMS if not substantive_file else load_terms_from_file(substantive_file)
    boilerplate_terms = BOILERPLATE_TERMS if not boilerplate_file else load_terms_from_file(boilerplate_file)

    # Compile terms into regex patterns once (faster than compiling every match)
    substantive_pats = build_regex_list(substantive_terms)
    boilerplate_pats = build_regex_list(boilerplate_terms)

    # 2) Map tickers -> CIKs
    tickmap = load_ticker_map()
    tickmap["ticker"] = tickmap["ticker"].str.upper()
    want = pd.DataFrame({"ticker": [t.strip().upper() for t in tickers]})
    dfm = want.merge(tickmap, on="ticker", how="left")

    rows = []
    for _, r in dfm.iterrows():
        tkr, cik = r["ticker"], r["cik"]
        if pd.isna(cik):
            print(f"[WARN] No CIK for {tkr} (skipping)")
            continue
        try:
            # 3) Pull the recent filings JSON and filter to 10-K rows
            filings = get_recent_filings(int(cik))
            krows = pick_10k_rows(filings, max_filings)

            # 4) Download and analyze each 10-K primary document
            for _, fr in krows.iterrows():
                html = download_primary_doc(int(cik), fr["accessionNumber"], fr["primaryDocument"])
                metrics = analyze_filing(html, substantive_pats, boilerplate_pats)
                rows.append({
                    "ticker": tkr,
                    "cik": int(cik),
                    "filing_date": fr["filingDate"],
                    "accession": fr["accessionNumber"],
                    "doc": fr["primaryDocument"],
                    **metrics
                })
        except Exception as e:
            print(f"[ERR] {tkr}: {e}")

    out = pd.DataFrame(rows)
    if not len(out):
        return out

    # 5) Choose the scoring metric (toggle):
    #    - tls:          use tls_raw (substantive - boilerplate)
    #    - substantive:  use substantive_per10k only
    if score_metric == "tls":
        out["score_raw"] = out["tls_raw"]
    elif score_metric == "substantive":
        out["score_raw"] = out["substantive_per10k"]
    else:
        raise ValueError("score_metric must be 'tls' or 'substantive'")

    # 6) Compute z-scores on the chosen score (for comparability across firms)
    out["score_z"] = zscore(out["score_raw"])

    # Keep legacy tls_z for backward compatibility (when using tls)
    out["tls_z"] = zscore(out["tls_raw"])

    return out

# --------------------------- CLI ENTRYPOINT ----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EDGAR 10-K transition-language scorer")
    ap.add_argument("--tickers", type=str, required=True,
                    help="Comma-separated list, e.g., TSLA,NEE,GE")
    ap.add_argument("--max-filings", type=int, default=1,
                    help="How many 10-Ks per ticker (most recent first)")
    ap.add_argument("--out", type=str, default="tls_scores.csv",
                    help="Output CSV path")
    ap.add_argument("--score-metric", choices=["tls","substantive"], default="tls",
                    help="Scoring basis: 'tls' (substantive - boilerplate) or 'substantive' only")
    ap.add_argument("--substantive-file", type=str, default=None,
                    help="Optional path to a text file of substantive terms (one per line)")
    ap.add_argument("--boilerplate-file", type=str, default=None,
                    help="Optional path to a text file of boilerplate terms (one per line)")
    args = ap.parse_args()

    # Split tickers and run
    tickers = [t for t in args.tickers.split(",") if t.strip()]
    res = run(tickers,
              args.max_filings,
              score_metric=args.score_metric,
              substantive_file=args.substantive_file,
              boilerplate_file=args.boilerplate_file)

    # Save or report empty
    if len(res):
        # Always save all component columns plus chosen score + z
        cols = [
            "ticker","cik","filing_date","accession","doc","tokens",
            "substantive_per10k","boilerplate_per10k","tls_raw","tls_z",
            "score_raw","score_z"
        ]
        # Ensure we only write available columns (robust to schema tweaks)
        cols = [c for c in cols if c in res.columns]
        Path(args.out).write_text(res[cols].to_csv(index=False), encoding="utf-8")
        print(f"Wrote {args.out} with {len(res)} rows. Scored on: {args.score_metric}")
    else:
        print("No results. Check tickers or network.")
